#pragma once

#include "torch/csrc/autograd/variable_version.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <list>
#include <memory>

namespace torch {
namespace autograd {
class Variable;
struct Function;
} // namespace autograd
namespace jit { namespace tracer {
struct ValueTracingStateElem;
using ValueTracingState = std::list<ValueTracingStateElem>;
}} // namespace jit::tracer
} // namespace torch

namespace torch { namespace autograd {

extern const char* ERR_BACKWARD_TWICE;

/// A snapshot of a variable at a certain version. A `SavedVariable` stores
/// enough information to reconstruct a variable from a certain point in time.
class SavedVariable {
 public:
  SavedVariable() = default;
  SavedVariable(const Variable& variable, bool is_output);
  SavedVariable(SavedVariable&&) = default;
  SavedVariable& operator=(SavedVariable&&) = default;

  // Must be defined externally to avoid it being inlined by the compiler,
  // which would require it to see the definition of ValueTracingState.
  ~SavedVariable();

  /// Reconstructs the saved variable. Pass `saved_for` as the gradient
  /// function if constructing the `SavedVariable` with it would have caused a
  /// circular reference.
  Variable unpack(std::shared_ptr<Function> saved_for = nullptr) const;

  void reset_data() {
    return data_.reset();
  }

 private:
  bool was_default_constructed_ = true;
  at::Tensor data_;

  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  std::shared_ptr<Function> grad_fn_;
  bool has_grad_fn_ = false;
  std::weak_ptr<Function> grad_accumulator_;

  VariableVersion version_counter_;
  uint32_t saved_version_ = 0;

  bool requires_grad_ = false;
  int output_nr_ = -1;

  std::unique_ptr<jit::tracer::ValueTracingState> tracing_state_;
};
}} // namespace torch::autograd
