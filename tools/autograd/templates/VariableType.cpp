#include "VariableType.h"

// ${generated_comment}

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/tensor.h"

#include <initializer_list>
#include <iostream>
#include <functional>

using namespace at;

namespace torch { namespace autograd {

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context)
  , baseType(baseType) {
}

ScalarType VariableType::scalarType() {
  return baseType->scalarType();
}
Backend VariableType::backend() {
  return baseType->backend();
}
bool VariableType::isCuda() { return baseType->isCuda(); }
bool VariableType::isSparse() { return baseType->isSparse(); }
bool VariableType::isDistributed() { return baseType->isDistributed(); }

std::unique_ptr<Storage> VariableType::storage() {
  return baseType->storage();
}
std::unique_ptr<Storage> VariableType::storage(size_t size) {
  return baseType->storage(size);
}
std::unique_ptr<Storage> VariableType::storageFromBlob(void * data, int64_t size) {
  return baseType->storageFromBlob(data, size);
}
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) {
  return baseType->unsafeTensorFromTH(th_pointer, retain);
}
std::unique_ptr<Generator> VariableType::generator() {
  return baseType->generator();
}

const char * VariableType::toString() const {
  return VariableType::typeString();
}
size_t VariableType::elementSizeInBytes() const {
  return baseType->elementSizeInBytes();
}
TypeID VariableType::ID() const {
  throw std::runtime_error("VariableType::ID() not implemented");
}

const char * VariableType::typeString() {
  return "VariableType";
}

Tensor & VariableType::checked_unpack(const Tensor & t, const char * name, int pos) const
{
 if(!t.defined()) {
   runtime_error("Expected a Tensor of type %s but found an undefined Tensor for argument #%d '%s'",
     toString(),pos,name);
 }
 if (&t.type() == this) {
   return static_cast<VariableImpl*>(t.pImpl)->data;
 }
 runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
   toString(),t.type().toString(),pos,name);
}

Variable VariableType::as_variable(Tensor tensor) const {
  return make_variable(std::move(tensor));
}

Variable VariableType::as_variable(const Scalar & scalar) const {
  auto tensor = scalar.toTensor();
  if (&tensor.type() != baseType) {
    tensor = tensor.toType(*baseType);
  }
  return make_variable(std::move(tensor));
}

void check_inplace(const VariableImpl& pImpl) {
  auto& version_counter = *pImpl.version_counter;
  if (pImpl.requires_grad && !pImpl.grad_fn) {
    at::runtime_error(
      "a leaf Variable that requires grad has been used in an in-place operation.");
  }
  if (version_counter.var_refcnt() > 1) {
    at::runtime_error(
      "in-place operations can be only used on variables that don't share "
      "storage with any other variables, but detected that there are %d objects "
      "sharing it", version_counter.var_refcnt());
  }
}

void wrap_output(VariableImpl& pImpl, FunctionFlags flags, std::shared_ptr<Function> grad_fn) {
  // Hooks up the grad_fn and sets the flags of the function output. This only
  // supports a single differentiable output.
  pImpl.requires_grad = flags.is_executable;
  pImpl.is_volatile = flags.is_volatile;
  if (!flags.is_volatile) {
    pImpl.output_nr = grad_fn->num_inputs++;
    grad_fn->set_flags(std::move(flags));
    pImpl.grad_fn = std::move(grad_fn);
  }
}

void VariableType::copy(const Tensor & src, Tensor & dst) {
  auto& src_ = checked_unpack(src, "src", 0);
  auto& dst_ = checked_unpack(dst, "dst", 1);
  auto& pImpl = static_cast<VariableImpl&>(*dst.get());
  check_inplace(pImpl);
  auto flags = Function::flags({ src });
  baseType->copy(src_, dst_);
  (*pImpl.version_counter)++;
  wrap_output(pImpl, std::move(flags), std::make_shared<Identity>());
}

Tensor & VariableType::m_resize_(Tensor & self, IntList size) {
  auto& self_ = checked_unpack(self, "self", 0);
  auto& pImpl = static_cast<VariableImpl&>(*self.get());
  check_inplace(pImpl);
  if (pImpl.grad_fn) {
    at::runtime_error("cannot resize non-leaf variables");
  }
  if (pImpl.requires_grad) {
    at::runtime_error("cannot resize variables which require grad");
  }
  baseType->m_resize_(self_, size);
  return self;
}

${type_derived_method_definitions}

}} // namespace torch::autograd
