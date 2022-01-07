import argparse
import csv
import functools
import gc
import io
import itertools
import logging
import math
import numpy as np
import os
import re
import sys
import time
import torch
from torch import nn
from torch.jit import fuser, optimized_execution
from os.path import abspath
from scipy.stats import ttest_ind
import importlib
import glob
import time
import collections
import csv

def get_unique_suffix():
    return f"{time.time()}_{os.getpid()}"

def get_benchmark_cls(model_name):
    if ("Benchmark(dims=[" in model_name):
        # just evaluate the model name + args
        # it should create a model with the right dim
        return eval(model_name)
    try:
        module = importlib.import_module(f'.models.{model_name}', package="torchbenchmark")
        Model = getattr(module, 'Model', None)
        if Model is None:
             raise RuntimeError(f"{module} does not define attribute Model, skip it")
        if not hasattr(Model, 'name'):
            Model.name = model_name
        return Model
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Could not find dependent module {e.name} for Model {model_name}, skip it")

# from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-5'])

import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam

log = logging.getLogger(__name__)

# Models that are known to crash or otherwise not work with lazy tensor are
# disabled, but should be removed from these lists once fixed
SKIP = {
    # out of memory test
    #"fastNLP_Bert",
    #"vision_maskrcnn",
    #"speech_trasformer",
    #"nvidia_deeprecommender",
    #"pytorch_struct",
    #"dlrm",
    #"LearningToPaint",
    #"vision_maskrcnn",
    #"drq",
    #"moco",
    # slow tests
    #"maml",
}
SKIP_TRAIN_ONLY = {
    # out of memory test
    "squeezenet1_1",
    "mobilenet_v2_quantized_qat",
    "hf_Reformer",
    "hf_GPT2",
    "hf_BigBird",
    "pyhpc_equation_of_state",
    "pyhpc_isoneutral_mixing",
    "densenet121",
    "resnet50_quantized_qat",
    "Background_Matting",
    "hf_Bart",
    "hf_Longformer",
    # slow tests
    "timm_efficientnet",
    "Super_SloMo",
    "BERT_pytorch",
    "demucs",
    "opacus_cifar10",
    # others
    "hf_DistilBert",
}

current_name = ""
current_device = ""

@functools.lru_cache(maxsize=None)
def output_csv(name, headers):
    output = csv.writer(
        io.TextIOWrapper(
            open(name, "wb", buffering=0),
            "utf-8",
            write_through=True,
        ),
        delimiter = ",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL
    )
    output.writerow(headers)
    return output

class HardSwishBenchmark:
    def __init__(self, dims):
        self.name = "HardSwishBenchmark(dims=[" + ','.join([str(d) for d in dims]) + '])'
        self.dims = dims

    def __call__(self, device, jit):
        return HardSwish(self.dims, device, jit)

class HardSwish(nn.Module):
    def __init__(self, dims, device='cuda', jit=False):
        super(HardSwish, self).__init__()
        self.name = "HardSwish[" + ','.join([str(d) for d in dims]) + ']'
        self.example_inputs = (
            torch.randn(*dims, device=device, dtype=torch.float32),
        )

    def get_module(self):
        return self, self.example_inputs

    def name(self):
        return self.name

    def forward(self, x):
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0

class DivAddMulBenchmark:
    """This wrapper helps interface with the same iterator as torchbench models
    """
    def __init__(self, dims):
        self.name = "DivAddMulBenchmark(dims=[" + ','.join([str(d) for d in dims]) + '])'
        self.dims = dims

    def __call__(self, device, jit):
        return DivAddMul(self.dims, device, jit)

class DivAddMul(nn.Module):
    def __init__(self, dims, device='cuda', jit=False):
        super(DivAddMul, self).__init__()
        self.attention_head_size = dims[1]
        self.name = "DivAddMul[" + ','.join([str(d) for d in dims]) + ']'
        self.example_inputs = (
            torch.randn(*dims, device=device, dtype=torch.float32),
            torch.randn(*dims, device=device, dtype=torch.float32),
        )

    def get_module(self):
        return self, self.example_inputs

    def name(self):
        return self.name

    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = out2 * 5.0
        return out3
toy_models = [
    HardSwishBenchmark,
    DivAddMulBenchmark,
]
toy_dims = [
    [1, 1, 1, 1],
    [32, 16, 128, 128],
    [128, 16, 128, 128],
    [256, 16, 128, 128],
]
for dims in toy_dims:
    # The toy benchmarks don't support training..
    # and it's too late to add it inside the generator func below...

    SKIP_TRAIN_ONLY.add("DivAddMulBenchmark(dims=[" + ','.join([str(d) for d in dims]) + '])')
    SKIP_TRAIN_ONLY.add("HardSwishBenchmark(dims=[" + ','.join([str(d) for d in dims]) + '])')
def iter_toy_model_names():
    for dims in toy_dims:
        for model in toy_models:
            yield model(dims=dims).name

def pick_grad(args, name):
    if args.test == 'train':
        return torch.enable_grad()

    if name in ("maml",):
        return torch.enable_grad()
    else:
        return torch.no_grad()

def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."

def iter_torchbench_model_names():
    from torchbenchmark import _list_model_paths
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        yield model_name

def iter_models(args, dirpath):
    for name in itertools.chain(iter_toy_model_names(), iter_torchbench_model_names()):
        if (
            (len(args.filter) and (not re.search("|".join(args.filter), name, re.I)))
            or (len(args.exclude) and re.search("|".join(args.exclude), name, re.I))
            or name in SKIP
            or (name in SKIP_TRAIN_ONLY and args.test == "train")
        ):
            save_error(name, args.test, "in SKIP or SKIP_TRAIN_ONLY", dirpath)
            continue
        yield name

def call_model_with(model, inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return model(*inputs)
    elif isinstance(inputs, dict):
        return model(**inputs)
    elif isistance(inputs, torch.Tensor):
        return model(inputs)
    raise RuntimeError("invalid example inputs ", inputs)

class CudaSync:
    def __init__(self, sync_every_iter=False):
        self.sync_every_iter = sync_every_iter

    def iter_sync(self, results):
        if self.sync_every_iter:
            torch.cuda.synchronize()

    def final_sync(self, results):
        torch.cuda.synchronize()

class NoOpSync:
    def __init__(self, sync_every_iter=False):
        pass

    def iter_sync(self, results):
        pass

    def final_sync(self, results):
        pass

class LazySync:
    def __init__(self, sync_every_iter=False, skip_final_sync=False):
        self.sync_every_iter = sync_every_iter
        self.skip_final_sync = skip_final_sync

    def iter_sync(self, results):
        ltm.mark_step()
        if self.sync_every_iter:
            ltm.wait_device_ops()
            if current_device == 'cuda':
                torch.cuda.synchronize()

    def final_sync(self, results):
        ltm.mark_step()
        if self.skip_final_sync:
            return
        ltm.wait_device_ops()
        if current_device == 'cuda':
            torch.cuda.synchronize()

class ToDeviceSync:
    def __init__(self, device, sync_every_iter=False):
        self.sync_every_iter = sync_every_iter
        self.device = device

    def iter_sync(self, results):
        if self.sync_every_iter:
            to_device(results[-1], self.device)
            if current_device == 'cuda':
                torch.cuda.synchronize()

    def final_sync(self, results):
        if len(results):
            if self.sync_every_iter:
                to_device(results[-1], self.device)
            else:
                to_device(results, self.device)

        if current_device == 'cuda':
            torch.cuda.synchronize()

def dump_lazy_metrics(reset=False):
    met = {name: int(metrics.counter_value(name)) for name in metrics.counter_names() if int(metrics.counter_value(name) > 0)}
    if reset:
        metrics.reset_metrics()
    return met

def timed(args, benchmark, sync, times=1):
    results = []
    sync.final_sync(results)
    torch.manual_seed(1337)
    if args.test == 'eval':
        model, example_inputs = benchmark.get_module()

    if current_device == 'lazy':
        torch.cuda.set_sync_debug_mode(2)
    else:
        torch.cuda.set_sync_debug_mode(0)

    # keep the lazy tensor results alive until the final sync
    t0 = time.perf_counter()
    for i in range(times):
        if args.test == 'eval':
            results.append(call_model_with(model, example_inputs))
        elif args.test == 'train':
            benchmark.train(niter=1)

        # for the last i, let final_sync take care of it
        if i < times - 1:
            # may be just an async 'mark_step' for lazy, or no-op for cuda
            sync.iter_sync(results)

    torch.cuda.set_sync_debug_mode(0)

    # should be a hard sync for lazy and cuda
    # unless strictly measuring lazy trace overhead, then no-op
    sync.final_sync(results)
    t1 = time.perf_counter()
    rc = results[-1] if args.test == 'eval' else None
    return rc, t1 - t0

def to_device(tensors, device):
    """Handles moving tensor or tensors (in various containers) to a new device.
        Used for various purposes (either correctness checking, or even as an impromptu
        means of synchronization.) Note: this method doesn't apply a cuda sync, do that outside.
    """

    try:
        import transformers.modeling_outputs
        if isinstance(tensors, transformers.modeling_outputs.MaskedLMOutput) \
        or isinstance(tensors, transformers.modeling_outputs.Seq2SeqLMOutput):
            # huggingface transformers return classes as model output with many attributes
            # we don't want to sync (such as hidden states of every layer) - just sync the logits
            tensors = tensors.logits
    except ImportError:
        pass

    try:
        import torchbenchmark.models.soft_actor_critic.nets
        if isinstance(tensors, torchbenchmark.models.soft_actor_critic.nets.SquashedNormal):
            # a SquashedNormal is a py class that holds a loc and scale torch tensor,
            # so convert it to a tuple for compatibility with downstream check_results
            tensors = (tensors.loc, tensors.scale)
    except ImportError:
        pass

    if isinstance(tensors, tuple) or isinstance(tensors, list):
        return tuple(to_device(i, device) for i in tensors)
    elif isinstance(tensors, dict):
        return {k: to_device(tensors[k], device) for k in tensors}
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    raise RuntimeError("invalid example tensors ", tensors)

def lazy_overhead_experiment(args, results, benchmark, lazy_benchmark):
    timings = np.zeros((args.repeat, 2), np.float64)
    ref_sync = CudaSync if current_device == 'cuda' else NoOpSync
    warmup0 = time.perf_counter()
    for rep in range(args.warmup):
        # interleave the runs to handle frequency scaling and load changes
        timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        timed(args, lazy_benchmark, sync=LazySync(sync_every_iter=True))
    warmup_time = time.perf_counter() - warmup0
    bench0 = time.perf_counter()
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        _, timings[rep, 1] = timed(args, lazy_benchmark, sync=LazySync(skip_final_sync=True))
        ltm.wait_device_ops()
        if current_device == 'cuda':
            torch.cuda.synchronize()
    bench_time = time.perf_counter() - bench0
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    overhead = median[1] / median[0]
    results.append(overhead)
    output_csv(
        os.path.join(args.output_dir, f"lazy-overheads_{args.test}_{get_unique_suffix()}.csv"),
        ("dev", "name", "test", "overhead", "pvalue"),
    ).writerow([current_device, current_name, args.test,  f"{overhead:.4f}", f"{pvalue:.4e}"])
    print(f"{short_name(name, limit=30):<30} {current_device:<4} {args.test:<5} {'trace overheads':<20} overhead: {overhead:.3f} pvalue: {pvalue:.2e}")
    if args.verbose:
        print(f"CIDEBUGOUTPUT,lazy_overhead_experiment,{current_name},{args.test},{current_device},{overhead:.4f},{pvalue:.4e},{args.warmup},{args.repeat},{warmup_time:.2f},{bench_time:.2f}")
    return (overhead, pvalue)

def lazy_compute_experiment(args, experiment, results, benchmark, lazy_benchmark, sync_every_iter=False, to_dev_sync=None):
    timings = np.zeros((args.repeat, 2), np.float64)
    if to_dev_sync is not None:
        ref_sync = ToDeviceSync(to_dev_sync, sync_every_iter=sync_every_iter)
        lazy_sync = ToDeviceSync(to_dev_sync, sync_every_iter=sync_every_iter)
    else:
        ref_sync = CudaSync(sync_every_iter=sync_every_iter) if current_device == 'cuda' else NoOpSync()
        lazy_sync = LazySync(sync_every_iter=sync_every_iter)

    # interleave the runs to handle frequency scaling and load changes
    warmup0 = time.perf_counter()
    for rep in range(args.warmup):
        # warmup
        timed(args, benchmark, sync=ref_sync)
        timed(args, lazy_benchmark, sync=lazy_sync)
    warmup_time = time.perf_counter() - warmup0

    # fresh metrics for each timed run
    dump_lazy_metrics(reset=True)
    bench0 = time.perf_counter()
    for rep in range(args.repeat):
        # measure
        _, timings[rep, 0] = timed(args, benchmark, times=args.inner_loop_repeat, sync=ref_sync)
        _, timings[rep, 1] = timed(args, lazy_benchmark, times=args.inner_loop_repeat, sync=lazy_sync)
    bench_time = time.perf_counter() - bench0
    lazy_metrics = dump_lazy_metrics(reset=True)
    if 'CachedCompile' not in lazy_metrics or lazy_metrics['CachedCompile'] != args.repeat * args.inner_loop_repeat:
        print("WARNING: lazy cached compile count indicates fallbacks, or something else")
    fallbacks = {k: v for (k, v) in lazy_metrics.items() if 'aten::' in k}
    if len(fallbacks):
        print("WARNING: lazy-eager fallbacks detected for ["+ ",".join(fallbacks.keys()) + ']')
    if args.dump_lazy_counters:
        print(lazy_metrics)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    results.append(speedup)
    output_csv(
        os.path.join(args.output_dir, f"lazy-compute_{args.test}_{get_unique_suffix()}.csv"),
        ("name", "dev", "experiment", "test", "speedup", "pvalue"),
    ).writerow([current_name, current_device, experiment, args.test, f"{speedup:.4f}", f"{pvalue:.2e}"])
    print(f"{short_name(current_name, limit=30):<30} {current_device:<4} {args.test:<5} {experiment:<20} speedup:  {speedup:.3f} pvalue: {pvalue:.2e}")
    if args.verbose:
        print(f"CIDEBUGOUTPUT,lazy_compute_experiment,{current_name},{current_device},{experiment},{args.test},{speedup:.4f},{pvalue:.2e},{args.warmup},{args.repeat},{warmup_time:.2f},{bench_time:.2f}")
    return (speedup, pvalue)


def check_results_impl(correct_result, lazy_result):
    # recursive helper for dealing with nested data structures
    if type(correct_result) is tuple:
        for c, l in zip(correct_result, lazy_result):
            return check_results_impl(c, l)

    if type(correct_result) is dict:
        print(correct_result.keys())
        for k in correct_result:
            assert k in lazy_result
            return check_results_impl(correct_result[k], lazy_result[k])

    assert type(correct_result) is torch.Tensor, f"Expect torch.Tensor but got {type(correct_result)}."
    return torch.allclose(correct_result, lazy_result)

def check_results(correct_result, lazy_result, device):
    # to_device has recursive logic and special handling for
    # extracting relevant tensors from huggingface data structures
    correct_result = to_device(correct_result, device)
    lazy_result = to_device(lazy_result, device)

    return check_results_impl(correct_result, lazy_result)


def check_fuser(args):
    if args.fuser == 'noopt':
        return
    if args.fuser is None:
        args.fuser = 'fuser1' if args.device == 'cpu' else 'fuser2'
    if args.device == 'cpu':
        assert args.fuser in ['fuser0', 'fuser1']
        if args.fuser == 'fuser1':
            assert torch._C._llvm_enabled(), "Can't use fuser1 (nnc) for CPU without building torch with llvm."
    if args.device == 'cuda':
        assert args.fuser in ['fuser0', 'fuser1', 'fuser2']

def run_tracing_execute_noops(test, lazy_benchmark):
    ltm.set_noop_execution_mode(True)
    if test == 'eval':
        model, example_inputs = lazy_benchmark.get_module()
    # doesn't actualyl collect a profile, but runs just the lazy trace
    # so you can use a profiler on top of the program.
    # note: depends on making the backend do a 'no-op' for executecomputation
    results = []
    for i in range(300):
        if test == 'eval':
            results.append(call_model_with(model, example_inputs))
        elif test == 'train':
            lazy_benchmark.train(niter=1)
        # we still do a mark step, to preserve the ratio of how often we split the graph
        # and run through the process of 'compile and execute' (even though these are now noops)
        ltm.mark_step()
    ltm.set_noop_execution_mode(False)

def merge_with_prefix(prefix, tmp_dir, out_dir, headers):
    results = []
    rfnames = glob.glob(os.path.join(tmp_dir, prefix + "*"))
    for rfname in rfnames:
        results.extend(open(rfname).readlines()[1:]) #skip header

    # the header shouldn't require quotations and the results should already be properly
    # quoted via output_csv
    with open(os.path.join(out_dir, prefix + "acc.csv"), "a+") as acc_csv:
        acc_csv.write(",".join(headers) + "\n")
        for l in results:
            acc_csv.write(l)

def merge_reformat(tmp_dir, out_dir, table):
    out_dir = args.output_dir
    # depending on the type of an experiment, fields can be in a different order
    # `get_field` deals with all three types including `error`
    def get_field(row, name, file_type):
        headers = {
            "error": ("name", "test", "error"),
            "lazy-compute" : ("name", "dev", "experiment", "test", "speedup", "pvalue"),
            "lazy-overheads" : ("dev", "name", "test", "overhead", "pvalue") 
        }

        header = headers[file_type]
        r = row[header.index(name)] if name in header else "N/A"
        return r

    csv_files = glob.glob(os.path.join(tmp_dir, "*.csv"))
    for csvf in csv_files:

        with open(csvf, "r") as csvfile:
            prefix = os.path.basename(csvf).split("_")[0]
            csvreader = csv.reader(csvfile, delimiter = ",", quotechar='"')
            # This skips the first row of the CSV file.
            next(csvreader)

            for r in csvreader:
                key = (get_field(r, "name", prefix), get_field(r, "test", prefix))
                entry = table[key]

            if prefix == "error":
                entry["error"] = get_field(r, "error", prefix)
            elif prefix == "lazy-overheads":
                entry["overhead"] = get_field(r, "overhead", prefix)
            else:
                entry[get_field(r, "experiment", prefix)] = get_field(r, "speedup", prefix)
            

    amortized_header = f"amortized {args.inner_loop_repeat}x"
    headers = ("name", "test", amortized_header, "unamortized", "overhead", "error", "rc")

    cw = output_csv(
        os.path.join(out_dir, f"{args.test}_reformat.csv"),
        headers
    )

    for k, v in table.items():
        cw.writerow((k[0], k[1], v.get(amortized_header, 'N/A'), v.get('unamortized', 'N/A'), v.get('overhead', 'N/A'), v.get('error', 'N/A'), v.get('rc')))

def save_error(name, test, error, dir):
    output_csv(
        os.path.join(dir, f"error_{get_unique_suffix()}.csv"),
        ("name", "test", "error"),
    ).writerow([name, test, error])


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--device", "-d", default='cuda', help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=4, help="number of warmup runs")
    parser.add_argument("--timeout", type=int, default=60 * 10, help="time allocated to each model")
    parser.add_argument("--repeat", "-n", type=int, default=4, help="number of timing runs (samples)")
    parser.add_argument("--inner_loop_repeat", type=int, default=10, help="repeat the computation this many times per sample")
    parser.add_argument("--fuser", type=str, choices=['noopt', 'fuser0', 'fuser1', 'fuser2'], help="0=legacy, 1=nnc, 2=nvfuser")
    parser.add_argument("--test", type=str, choices=['eval', 'train'], default='eval')
    parser.add_argument("--verbose", action='store_false')
    parser.add_argument("--torchbench_dir", type=str, help="path to torchbenchmark repo")
    parser.add_argument("--output_dir", type=str, default=".", help="path to write output files")
    parser.add_argument("--dump_lazy_counters", action='store_true', help="dump lazy counter values after each timing run")
    parser.add_argument("--run_tracing_execute_noops", action='store_true', help="Run the tracing portion only, with noop backend, useful for running under a profiler.")
    parser.add_argument("--run_in_subprocess", "-s", type=str, help="which model run in subprocess.This will ignore filter and exclude")
    args = parser.parse_args()
    results = []

    check_fuser(args)

    torchbench_dir = abspath(args.torchbench_dir) if args.torchbench_dir else abspath("../../benchmark")
    assert os.path.exists(os.path.join(torchbench_dir, "torchbenchmark")), "set --torchbench_dir to installed torchbench repo"
    sys.path.append(torchbench_dir)

    copy_argv = [] + sys.argv
    if args.run_in_subprocess:
        try:
            from fastNLP.core import logger
            logger.setLevel(logging.WARNING)
            name = args.run_in_subprocess
            benchmark_cls = get_benchmark_cls(args.run_in_subprocess)
            bench_name = benchmark_cls.name if hasattr(benchmark_cls, 'name') else benchmark_cls.name()
            for device in [args.device]:

                # no try since we should've already filtered out models we can't create
                torch.manual_seed(1337)
                benchmark = benchmark_cls(device=device, jit=False)
                torch.manual_seed(1337)
                lazy_benchmark = benchmark_cls(device='lazy', jit=False)
                # TODO: might be redundant
                gc.collect()

                current_name = name
                current_device = device

                if device == 'cuda':
                    assert 'LTC_TS_CUDA' in os.environ and bool(os.environ['LTC_TS_CUDA'])

                if args.run_tracing_execute_noops:
                    print(f"Profiling {name}")
                    run_tracing_execute_noops(args.test, lazy_benchmark)
                    # when profiling, we really don't want to do anything else
                    exit(0)

                with pick_grad(args, name):
                    with fuser(args.fuser) if args.fuser != 'noopt' else optimized_execution(False):
                        if args.fuser == 'noopt':
                            # TODO(whc) cleaner way to configure the fusers; seems i have to set both optimized_execution(False)
                            # _and_ disable fusers to get no-optimization
                            torch._C._jit_override_can_fuse_on_cpu(False)
                            torch._C._jit_override_can_fuse_on_gpu(False)
                            torch._C._jit_set_texpr_fuser_enabled(False)
                            torch._C._jit_set_nvfuser_enabled(False)
                        if args.fuser == 'fuser2':
                            # special case to disable nvfuser horizontal fusion as it is currently broken
                            # TODO(whc) remove this once it's fixed
                            torch._C._jit_set_nvfuser_horizontal_mode(False)
                        try:
                            if args.test == 'eval':
                                # Correctness Check
                                torch.manual_seed(1337)
                                model, example_inputs = benchmark.get_module()
                                model.eval()
                                correct_result = call_model_with(model, example_inputs)
                                torch.manual_seed(1337)
                                lazy_model, lazy_inputs = lazy_benchmark.get_module()
                                lazy_model.eval()
                                lazy_result = call_model_with(lazy_model, lazy_inputs)
                                if not check_results(correct_result, lazy_result, device):
                                    print(f"INCORRECT: {name}")
                                    save_error(name, "eval", "Incorrect results.", args.output_dir)
                                    continue
                        except Exception as e:
                            print(f"ERROR: {name}: {e}")
                            save_error(name, "eval", e, args.output_dir)
                            continue

                        lazy_overhead_experiment(args, results, benchmark, lazy_benchmark)
                        lazy_compute_experiment(args, f"amortized {args.inner_loop_repeat}x", results, benchmark, lazy_benchmark)
                        lazy_compute_experiment(args, "unamortized", results, benchmark, lazy_benchmark, sync_every_iter=True)

        except Exception as e:
            print(f"ERROR: {name}: {e}")
            save_error(name, "eval", e, args.output_dir)
            exit(1)
        exit(0)

    import psutil
    import subprocess
    import tempfile
    dirpath = tempfile.mkdtemp()
    table = collections.defaultdict(dict)
    for model_name in iter_models(args, dirpath):
        # if `--run_in_subprocess` is specified, it will override any filters and excludes
        # pass the rest of arguments intact such as device, test, repeat, etc
        # note, the latest output_dir will override the original one and this is exactly what we want
        # for child processes
        launch_command = f"python {' '.join(copy_argv)} --run_in_subprocess '{model_name}' --output_dir={dirpath}"
        env = os.environ
        env["LTC_TS_CUDA"] = "1"
        rc = 0
        try:
            if args.verbose:
                cp = subprocess.run("nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader", capture_output=True, text=True, shell=True)
                print(f"CIDEBUGOUTPUT,BEFORE subprocess.run,{model_name},{cp.stdout}")
            proc = subprocess.Popen(launch_command,
                        env=env,
                        shell=True,
                        stderr=subprocess.STDOUT)
            
            outs, errs = proc.communicate(timeout=args.timeout)
            rc = proc.poll()
        except subprocess.TimeoutExpired:
            print(f"{model_name} timed out after {args.timeout // 60} minutes! Include it in SKIP or SKIP_TRAIN_ONLY")
            save_error(model_name, args.test, "Timed out.", dirpath)
            # to visualize highlight timeouts, they will also have 
            # "timed out" in the error column
            rc = 17
            process = psutil.Process(proc.pid)
            for p in process.children(recursive=True):
                p.kill()
            process.kill()
        if args.verbose:
            cp = subprocess.run("nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader", capture_output=True, text=True, shell=True)
            print(f"CIDEBUGOUTPUT,AFTER subprocess.run,{model_name},{args.test},{cp.stdout}")

        entry = table[(model_name, args.test)]
        entry["rc"] = rc
    merge_with_prefix("lazy-overheads_", dirpath, args.output_dir, ("dev", "name", "test", "overhead", "pvalue"))
    merge_with_prefix("lazy-compute_", dirpath, args.output_dir, ("name", "dev", "experiment", "test", "speedup", "pvalue"))
    merge_with_prefix("error_", dirpath, args.output_dir, ("name", "test", "error"))
    merge_reformat(dirpath, args, table)
