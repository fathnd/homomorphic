#include <torch/library.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(sparse, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::_semi_structured_sparsify_both_ways(Tensor input, str algorithm = '', str backend = 'cutlass') -> (Tensor, Tensor, Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::_semi_structured_apply(Tensor input, Tensor threads_masks) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::_semi_structured_apply_dense_output(Tensor input, Tensor threads_masks) -> Tensor"));
}
