#ifndef CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_
#define CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_

#include <string>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AffineChannelOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AffineChannelOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                        : RunOnDeviceWithOrderNHWC();
  }

  bool RunOnDeviceWithOrderNCHW() {
    const auto& X = Input(0);
    const auto& scale = Input(1);
    const auto& bias = Input(2);
    auto* Y = Output(0);
    if (is_learnable_) {
      CAFFE_ENFORCE_NE(
          Y,
          &X,
          "In-place affine_channel_op is not supported when "
          "is_learnable = true.");
    }
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const int HxW = X.size() / (N * C);
    Y->ResizeLike(X);
    math::AffineChannel<T, Context, StorageOrder::NCHW>(
        N,
        C,
        HxW,
        X.template data<T>(),
        scale.template data<T>(),
        bias.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

  bool RunOnDeviceWithOrderNHWC() {
    const auto& X = Input(0);
    const auto& scale = Input(1);
    const auto& bias = Input(2);
    auto* Y = Output(0);
    if (is_learnable_) {
      CAFFE_ENFORCE_NE(
          Y,
          &X,
          "In-place affine_channel_op is not supported when "
          "is_learnable = true.");
    }
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    const int HxW = X.size() / (N * C);
    Y->ResizeLike(X);
    math::AffineChannel<T, Context, StorageOrder::NHWC>(
        N,
        C,
        HxW,
        X.template data<T>(),
        scale.template data<T>(),
        bias.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

 private:
  const StorageOrder order_;
  const bool is_learnable_;
};

template <typename T, class Context>
class AffineChannelGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  AffineChannelGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                        : RunOnDeviceWithOrderNHWC();
  }

  bool RunOnDeviceWithOrderNCHW();

  bool RunOnDeviceWithOrderNHWC();

 private:
  const StorageOrder order_;
  const bool is_learnable_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_AFFINE_CHANNEL_OP_H_
