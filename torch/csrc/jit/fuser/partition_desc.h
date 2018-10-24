#pragma once

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/fuser/tensor_desc.h"

#include <memory>
#include <cstdint>
#include <vector>

namespace torch { namespace jit { namespace fuser {

// Descriptor for chunk-ing an input tensor into subtensors
// OR concat-ing an output tensor from subtensors
struct PartitionDesc {
  
  PartitionDesc()
  : nSubTensors_{1}
  , dim_{0} 
  { }

  PartitionDesc(
    const TensorDesc& _desc
  , size_t _nSubTensors
  , size_t _dim)
  : nSubTensors_{_nSubTensors}
  , dim_{_dim} {
    JIT_ASSERT(nSubTensors_ > 1);
    std::vector<bool> cont = _desc.contiguity;
    if(dim_ > 0) {
      // when we narrow the concatenated output/chunked input
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim_ - 1] = false;
    }
    subTensorDesc_.reset(new TensorDesc(_desc.scalar_type, cont));
  }

  bool isNoop() const { return (nSubTensors_ == 1);}
  size_t nSubTensors() const { return nSubTensors_; }
  size_t dim() const { return dim_; }
  std::shared_ptr<TensorDesc> subTensorDesc() { return subTensorDesc_; }
  const std::shared_ptr<TensorDesc> subTensorDesc() const { return subTensorDesc_; }

private: 
  size_t nSubTensors_; // == 1 for tensors that should not be operated on via chunk/cat
  size_t dim_; // dimension along which the chunk/concat occurs
  std::shared_ptr<TensorDesc> subTensorDesc_; // descriptor for the subtensor, if it exists
};

} // namespace fuser
} // namespace jit 
} // namespace torch
