## @package batch_lr_loss
# Module caffe2.python.layers.batch_lr_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


class BatchLRLoss(ModelLayer):

    def __init__(
        self,
        model,
        input_record,
        name='batch_lr_loss',
        average_loss=True,
        jsd_weight=0.0,
        pos_label_target=1.0,
        neg_label_target=0.0,
        homotopy_weighting=False,
        log_D_trick=False,
        unjoined_lr_loss=False,
        uncertainty_penalty=1.0,
        uncertainty_before_sigmoid=True,
        sampling_size=100,
        **kwargs
    ):
        super(BatchLRLoss, self).__init__(model, name, input_record, **kwargs)

        self.average_loss = average_loss

        assert (schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('logit', schema.Scalar())
            ),
            input_record
        ))

        self.jsd_fuse = False
        assert jsd_weight >= 0 and jsd_weight <= 1
        if jsd_weight > 0 or homotopy_weighting:
            assert 'prediction' in input_record
            self.init_weight(jsd_weight, homotopy_weighting)
            self.jsd_fuse = True
        self.homotopy_weighting = homotopy_weighting

        assert pos_label_target <= 1 and pos_label_target >= 0
        assert neg_label_target <= 1 and neg_label_target >= 0
        assert pos_label_target >= neg_label_target
        self.pos_label_target = pos_label_target
        self.neg_label_target = neg_label_target

        assert not (log_D_trick and unjoined_lr_loss)
        self.log_D_trick = log_D_trick
        self.unjoined_lr_loss = unjoined_lr_loss
        assert uncertainty_penalty >= 0
        self.uncertainty_penalty = uncertainty_penalty
        self.uncertainty_before_sigmoid = uncertainty_before_sigmoid

        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output')
        )

        # the sampling size to do MC integration for the CTR expectation
        self.T = sampling_size
        self.noise = self.create_param(
            param_name='noise',
            shape=[1, self.T],
            initializer=('GaussianFill', {'mean': 0.0, 'std': 1.0}),
            optimizer=self.model.NoOptim
        )

        self.vec1 = self.create_param(
            param_name='v1',
            shape=[1, self.T],
            initializer=('ConstantFill', {'value': 1.0}),
            optimizer=self.model.NoOptim
        )

    def init_weight(self, jsd_weight, homotopy_weighting):
        if homotopy_weighting:
            self.mutex = self.create_param(
                param_name=('%s_mutex' % self.name),
                shape=None,
                initializer=('CreateMutex', ),
                optimizer=self.model.NoOptim,
            )
            self.counter = self.create_param(
                param_name=('%s_counter' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 0,
                        'dtype': core.DataType.INT64
                    }
                ),
                optimizer=self.model.NoOptim,
            )
            self.xent_weight = self.create_param(
                param_name=('%s_xent_weight' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 1.,
                        'dtype': core.DataType.FLOAT
                    }
                ),
                optimizer=self.model.NoOptim,
            )
            self.jsd_weight = self.create_param(
                param_name=('%s_jsd_weight' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 0.,
                        'dtype': core.DataType.FLOAT
                    }
                ),
                optimizer=self.model.NoOptim,
            )
        else:
            self.jsd_weight = self.model.add_global_constant(
                '%s_jsd_weight' % self.name, jsd_weight
            )
            self.xent_weight = self.model.add_global_constant(
                '%s_xent_weight' % self.name, 1. - jsd_weight
            )

    def _get_xent_for_uncertainty_before_sigmoid(self, net, label):
        stop_grad_noise = net.StopGradient(
            [self.noise], [net.NextScopedBlob("stop_grad_noise")]
        )
        stop_grad_vec1 = net.StopGradient(
            [self.vec1], [net.NextScopedBlob("stop_grad_vec1")]
        )

        # each row is for one sample, each column is for each random sampling
        # label expanding
        label = net.StopGradient([label], [net.NextScopedBlob("stop_grad_label")])
        label = net.ExpandDims(label, net.NextScopedBlob('expanded_label'), dims=[1])
        mat_label = net.MatMul([label, stop_grad_vec1])

        # logit expanding
        logit = self.input_record.logit()
        logit = net.ExpandDims(logit, net.NextScopedBlob('expanded_logit'), dims=[1])
        mat_logit = net.MatMul([logit, stop_grad_vec1], net.NextScopedBlob("mat_logit"))

        # noise expanding
        log_variance_blob = self.input_record.log_variance()
        log_variance_blob = net.ExpandDims(
            log_variance_blob, net.NextScopedBlob('expanded_log_variance'),
            dims=[1]
        )
        variance = net.Exp(log_variance_blob, net.NextScopedBlob("variance"))
        std = net.Sqrt(variance, net.NextScopedBlob("std"))
        mat_noise = net.MatMul([std, stop_grad_noise], net.NextScopedBlob("mat_noise"))

        mat_x = net.Add([mat_logit, mat_noise], net.NextScopedBlob("before_sigmoid"))
        mat_ctr = net.Sigmoid(mat_x, net.NextScopedBlob("mat_ctr"))
        mat_m1 = net.ConstantFill(mat_label, net.NextScopedBlob("matrix1"), value=-1.0)

        # likelihood = label* p + (1-p)*(1-label)
        part1 = net.Mul([mat_label, mat_ctr], net.NextScopedBlob("p1"))
        part2_1 = net.Add([mat_label, mat_m1])
        part2_2 = net.Add([mat_ctr, mat_m1])
        part2 = net.Mul([part2_1, part2_2], net.NextScopedBlob('p2'))
        likelihood = net.Add([part1, part2])

        # get the likelihood expectation for each sample
        expected_likelihood = net.ReduceBackMean(likelihood, net.NextScopedBlob("exp_l"))
        log_likelihood = net.Log(expected_likelihood, net.NextScopedBlob("logloss"))
        xent = net.Negative(log_likelihood, net.NextScopedBlob('uncertainty_xent'))
        return xent

    def update_weight(self, net):
        net.AtomicIter([self.mutex, self.counter], [self.counter])
        # iter = 0: lr = 1;
        # iter = 1e6; lr = 0.5^0.1  = 0.93
        # iter = 1e9; lr = 1e-3^0.1 = 0.50
        net.LearningRate([self.counter], [self.xent_weight], base_lr=1.0,
                         policy='inv', gamma=1e-6, power=0.1,)
        net.Sub(
            [self.model.global_constants['ONE'], self.xent_weight],
            [self.jsd_weight]
        )
        return self.xent_weight, self.jsd_weight

    def add_ops(self, net):
        # numerically stable log-softmax with crossentropy
        label = self.input_record.label()
        # mandatory cast to float32
        # self.input_record.label.field_type().base is np.float32 but
        # label type is actually int
        label = net.Cast(
            label,
            net.NextScopedBlob('label_float32'),
            to=core.DataType.FLOAT)
        label = net.ExpandDims(label, net.NextScopedBlob('expanded_label'),
                                dims=[1])
        if self.pos_label_target != 1.0 or self.neg_label_target != 0.0:
            label = net.StumpFunc(
                label,
                net.NextScopedBlob('smoothed_label'),
                threshold=0.5,
                low_value=self.neg_label_target,
                high_value=self.pos_label_target,
            )

        if (
            'log_variance' in self.input_record.fields
            and self.uncertainty_before_sigmoid
        ):
            xent = self._get_xent_for_uncertainty_before_sigmoid(net, label)
        else:
            xent = net.SigmoidCrossEntropyWithLogits(
                [self.input_record.logit(), label],
                net.NextScopedBlob('cross_entropy'),
                log_D_trick=self.log_D_trick,
                unjoined_lr_loss=self.unjoined_lr_loss
            )

        # fuse with JSD
        if self.jsd_fuse:
            jsd = net.BernoulliJSD(
                [self.input_record.prediction(), label],
                net.NextScopedBlob('jsd'),
            )
            if self.homotopy_weighting:
                self.update_weight(net)
            loss = net.WeightedSum(
                [xent, self.xent_weight, jsd, self.jsd_weight],
                net.NextScopedBlob('loss'),
            )
        else:
            loss = xent

        if 'log_variance' in self.input_record.fields and not self.uncertainty_before_sigmoid:
            # mean (0.5 * exp(-s) * loss + 0.5 * penalty * s)
            log_variance_blob = self.input_record.log_variance()

            log_variance_blob = net.ExpandDims(
                log_variance_blob, net.NextScopedBlob('expanded_log_variance'),
                dims=[1]
            )

            neg_log_variance_blob = net.Negative(
                [log_variance_blob],
                net.NextScopedBlob('neg_log_variance')
            )

            # enforce less than 88 to avoid OverflowError
            neg_log_variance_blob = net.Clip(
                [neg_log_variance_blob],
                net.NextScopedBlob('clipped_neg_log_variance'),
                max=88.0
            )

            exp_neg_log_variance_blob = net.Exp(
                [neg_log_variance_blob],
                net.NextScopedBlob('exp_neg_log_variance')
            )

            exp_neg_log_variance_loss_blob = net.Mul(
                [exp_neg_log_variance_blob, loss],
                net.NextScopedBlob('exp_neg_log_variance_loss')
            )

            penalized_uncertainty = net.Scale(
                log_variance_blob, net.NextScopedBlob("penalized_unceratinty"),
                scale=float(self.uncertainty_penalty)
            )

            loss_2x = net.Add(
                [exp_neg_log_variance_loss_blob, penalized_uncertainty],
                net.NextScopedBlob('loss')
            )
            loss = net.Scale(loss_2x, net.NextScopedBlob("loss"), scale=0.5)

        if 'weight' in self.input_record.fields:
            weight_blob = self.input_record.weight()
            if self.input_record.weight.field_type().base != np.float32:
                weight_blob = net.Cast(
                    weight_blob,
                    weight_blob + '_float32',
                    to=core.DataType.FLOAT
                )
            weight_blob = net.StopGradient(
                [weight_blob],
                [net.NextScopedBlob('weight_stop_gradient')],
            )
            loss = net.Mul(
                [loss, weight_blob],
                net.NextScopedBlob('weighted_cross_entropy'),
            )

        if self.average_loss:
            net.AveragedLoss(loss, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(loss, self.output_schema.field_blobs())
