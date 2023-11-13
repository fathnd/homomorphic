# flake8: noqa: B950
from ._internal import register_artifact, register_log

register_log("dynamo", "torch._dynamo")
register_log("aot", "torch._functorch.aot_autograd")
register_log("inductor", "torch._inductor")
register_log("dynamic", "torch.fx.experimental.symbolic_shapes")
register_log("torch", "torch")
register_log("distributed", "torch.distributed")
register_log("onnx", "torch.onnx")

register_artifact(
    "guards",
    "This prints the guards for every compiled Dynamo frame. It does not tell you where the guards come from.",
    visible=True,
)
register_artifact("verbose_guards", "", off_by_default=True)
register_artifact(
    "bytecode",
    "Prints the original and modified bytecode from Dynamo. Mostly useful if you're debugging our bytecode generation in Dynamo.",
    off_by_default=True,
)
register_artifact(
    "graph",
    "Prints the dynamo traced graph (prior to AOTDispatch) in a table. If you prefer python code use `graph_code` instead. ",
)
register_artifact("graph_code", "Like `graph`, but gives you the Python code instead.")
register_artifact(
    "graph_sizes", "Prints the sizes of all FX nodes in the dynamo graph."
)
register_artifact(
    "trace_source",
    "As we execute bytecode, prints the file name / line number we are processing and the actual source code. Useful with `bytecode`",
)
register_artifact(
    "trace_call",
    "Like trace_source, but it will give you the per-expression blow-by-blow if your Python is recent enough.",
)
register_artifact(
    "aot_graphs",
    "Prints the FX forward and backward graph generated by AOTDispatch, after partitioning. Useful to understand what's being given to Inductor",
    visible=True,
)
register_artifact(
    "aot_joint_graph",
    "Print FX joint graph from AOTAutograd, prior to partitioning. Useful for debugging partitioning",
)
register_artifact(
    "post_grad_graphs",
    "Prints the FX graph generated by post grad passes. Useful to understand what's being given to Inductor after post grad passes",
)
register_artifact(
    "compiled_autograd",
    "Prints various logs in compiled_autograd, including but not limited to the graphs. Useful for debugging compiled_autograd.",
    visible=True,
)
register_artifact(
    "ddp_graphs",
    "Only relevant for compiling DDP. DDP splits into multiple graphs to trigger comms early. This will print each individual graph here.",
)
register_artifact(
    "recompiles",
    "Prints the reason why we recompiled a graph. Very, very useful.",
    visible=True,
)
register_artifact(
    "graph_breaks",
    "Prints whenever Dynamo decides that it needs to graph break (i.e. create a new graph). Useful for debugging why torch.compile has poor performance",
    visible=True,
)
register_artifact(
    "not_implemented",
    "Prints log messages whenever we return NotImplemented in a multi-dispatch, letting you trace through each object we attempted to dispatch to",
)
register_artifact(
    "output_code",
    "Prints the code that Inductor generates (either Triton or C++)",
    off_by_default=True,
    visible=True,
)
register_artifact(
    "schedule",
    "Inductor scheduler information. Useful if working on Inductor fusion algo",
    off_by_default=True,
)
register_artifact("perf_hints", "", off_by_default=True)
register_artifact("onnx_diagnostics", "", off_by_default=True)
register_artifact(
    "fusion",
    "Detailed Inductor fusion decisions. More detailed than 'schedule'",
    off_by_default=True,
)
register_artifact(
    "overlap",
    "Detailed Inductor compute/comm overlap decisions",
    off_by_default=True,
)

register_artifact("custom_format_test_artifact", "Testing only", log_format="")
