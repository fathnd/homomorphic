import enum
from typing import NamedTuple

from torch.fx.graph import Node

from typing import Dict, Any, List

class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'
    NODE_INPUT = 'node_input'

NSSubgraph = NamedTuple(
    'NSSubgraph',
    [('start_node', Node), ('end_node', Node), ('base_op_node', Node)]
)

# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys
# {
#   # one of NSSingleResultValuesType
#   'type': 'weight',
#   # the values of type specified above
#   'values': [torch.tensor(...), ...],
#   # name of the node directly before the logger
#   'prev_node_name': 'linear1',
#   # type of the underlying function or module
#   'prev_node_target_type': torch.nn.functional.linear  # or torch.nn.Linear, etc
#   # name of the node responsible for adding this logger
#   # Note: this may differ from prev_node_name if we are logging inputs
#   'ref_node_name': 'linear1',
#   # index of this node within the arg of the input/output node
#   # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
#   'index_within_arg': 0,
# }
NSSingleResultType = Dict[str, Any]

# {
#   'layer_name_1': {  # subgraph name
#     'node_output': {  # results type (node_output, node_input, weight)
#       'model_name_a':  # model name
#          [NSSingleResultType, ...],  # results, ordered by index_within_arg
#       'model_name_b':
#          [NSSingleResultType, ...],
#     },
#   },
# }
#
NSResultsType = Dict[str, Dict[str, Dict[str, List[NSSingleResultType]]]]
