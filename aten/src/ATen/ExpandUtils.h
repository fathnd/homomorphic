#pragma once

#include "ATen/Tensor.h"
#include "ATen/Error.h"

#include <functional>
#include <sstream>
#include <tuple>

namespace at {

AT_API std::vector<int64_t> infer_size(IntList a, IntList b);
AT_API std::tuple<std::vector<int64_t>, std::vector<int64_t> > inferExpandGeometry(
    IntList tensor_sizes, IntList tensor_strides, IntList sizes);

// avoid copy-construction of Tensor by using a reference_wrapper.
inline void check_defined(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      AT_ERROR(api_name, "(...) called with an undefined Tensor");
    }
  }
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(tensor.sizes(), /*implicit=*/true)); // see [expand implicit]
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand, const char *api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace(tensor, to_expand);
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  return std::make_tuple(
      to_expand1.expand(tensor.sizes(), /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(tensor.sizes(), /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2,
                                                 const char *api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace(tensor, to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = infer_size(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size(expanded_size12, to_expand3.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size, /*implicit=*/true),
      to_expand3.expand(expanded_size, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3,
                                                          const char *api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(sizes, /*implicit=*/true)); // see [expand implicit]
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes, const char *api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sizes();
      first = false;
    } else {
      sizes = infer_size(sizes, to_expand[i].sizes());
    }
  }

  std::vector<Tensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = to_expand[i].expand(sizes, /*implicit=*/true); // see [expand implicit]
    }
  }
  return result;
}

// Sums `tensor` repeatedly to produce a tensor of shape `shape`.
// Precondition: is_expandable_to(shape, tensor.sizes()) must be true
static inline Tensor reduce_to(Tensor tensor, IntList shape) {
  if (shape.size() == 0) {
    return tensor.sum();
  }
  Tensor result = tensor;
  while (result.dim() > (int64_t)shape.size()) {
    result = result.sum(0, false);
  }
  for (int64_t i = 0; i < result.dim(); ++i) {
    if (shape[i] == 1 && result.sizes()[i] > 1) {
      result = result.sum(i, true);
    }
  }
  return result;
}

// True if `shape` can be broadcasted to `desired`
static inline bool is_expandable_to(IntList shape, IntList desired) {
  int ndim = shape.size();
  int target_dim = desired.size();
  if (ndim > target_dim) {
    return false;
  }
  for (int i = 0; i < ndim; i++) {
    int64_t size = shape[ndim - i - 1];
    int64_t target = desired[target_dim - i - 1];
    if (size != target && size != 1) {
      return false;
    }
  }
  return true;
}

}
