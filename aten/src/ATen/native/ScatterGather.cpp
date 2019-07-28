#include <tuple>
#include <algorithm>
#include <ATen/ATen.h>
#include <c10/core/WrapDimMinimal.h>

namespace {

inline void expand_size(int64_t dim, int64_t &size1, int64_t &size2) {
  TORCH_CHECK(size1 == 1 || size2 == 1, "Size mismatch at dim=", dim, ", get: ", size1, " and ", size2);
  size1 = size2 = size1 + size2 - 1;
}

std::tuple<std::vector<int64_t>, at::Tensor, at::Tensor>
inline expand_scatter(const at::Tensor &self, int64_t dim, at::Tensor index, const at::Tensor &src) {
  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  std::vector<int64_t> src_sizes = src.sizes().vec();
  if (src_sizes.size() == 0) {  // when src is a scalar tensor
    src_sizes = std::vector<int64_t>(self_sizes.size());
    std::fill(src_sizes.begin(), src_sizes.end(), 1);
  }
  TORCH_CHECK(self_sizes.size() == src_sizes.size(), "torch.scatter requires src and dest to have the same number of dimensions");
  TORCH_CHECK(index_sizes.size() <= src_sizes.size(), "torch.scatter requires src to have more dimensions than index");
  dim = c10::maybe_wrap_dim(dim, index_sizes.size());
  for (int64_t i = 0; i < self_sizes.size(); i++) {
    if (i == dim) {
      if (src_sizes[i] != index_sizes[i]) {
        expand_size(i, index_sizes[i], src_sizes[i]);
      }
    } else if (i < index_sizes.size()) {
      if (self_sizes[i] == index_sizes[i] && self_sizes[i] == src_sizes[i]) {
        continue;
      } else if (self_sizes[i] == index_sizes[i] && src_sizes[i] == 1) {
        src_sizes[i] = self_sizes[i];
      } else if (self_sizes[i] == src_sizes[i] && index_sizes[i] == 1) {
        index_sizes[i] = self_sizes[i];
      } else if (index_sizes[i] == src_sizes[i] && self_sizes[i] == 1) {
        self_sizes[i] = index_sizes[i];
      } else {
        AT_ERROR("Size mismatch at dim=", dim, ", get: ", self_sizes[i], ", ", index_sizes[i], " and ", src_sizes[i]);
      }
    } else {
      if (src_sizes[i] != self_sizes[i]) {
        expand_size(i, src_sizes[i], self_sizes[i]);
      }
      index.unsqueeze_(-1);
    }
  }
  return std::make_tuple(self_sizes, index.expand(index_sizes), src.expand(src_sizes));
}

std::tuple<at::Tensor, at::Tensor, std::vector<int64_t>>
inline expand_gather(const at::Tensor &self, int64_t dim, at::Tensor index) {
  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  TORCH_CHECK(self_sizes.size() >= index_sizes.size(), "torch.gather requires input to have more dimensions than index");
  dim = c10::maybe_wrap_dim(dim, index_sizes.size());
  std::vector<int64_t> result_sizes(self_sizes.size());
  for(int64_t i = 0; i < self_sizes.size(); i++) {
    if (i == dim) {
      result_sizes[i] = index_sizes[i];
    } else if (i < index_sizes.size()) {
      if (self_sizes[i] == index_sizes[i]) {
        result_sizes[i] = index_sizes[i];
      } else {
        expand_size(i, index_sizes[i], self_sizes[i]);
        result_sizes[i] = self_sizes[i];
      }
    } else {
      result_sizes[i] = self_sizes[i];
      index.unsqueeze_(-1);
    }
  }
  return std::make_tuple(self.expand(self_sizes), index.expand(index_sizes), result_sizes);
}

std::tuple<std::vector<int64_t>, at::Tensor>
inline expand_scatter_scalar(const at::Tensor &self, int64_t dim, at::Tensor index) {
  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  TORCH_CHECK(self_sizes.size() >= index_sizes.size(), "scatter and scatter_add requires input to have more dimensions than index");
  dim = c10::maybe_wrap_dim(dim, index_sizes.size());
  for(int64_t i = 0; i < self_sizes.size(); i++) {
    if (i == dim) {
      continue;
    } else if (i < index_sizes.size()) {
      if (self_sizes[i] != index_sizes[i]) {
        expand_size(i, index_sizes[i], self_sizes[i]);
      }
    } else {
      index.unsqueeze_(-1);
    }
  }
  return std::make_tuple(self_sizes, index.expand(index_sizes));
}

}  // namespace

namespace at { namespace native {

Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_gather_out(result, self, dim, index);
  }
  Tensor expanded_self, expanded_index;
  std::vector<int64_t> result_sizes;
  std::tie(expanded_self, expanded_index, result_sizes) = expand_gather(self, dim, index);
  TORCH_CHECK(result_sizes == result.sizes(), "broadcasting change the shape of out");
  return at::_gather_out(result, expanded_self, dim, expanded_index, sparse_grad);
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_gather(self, dim, index);
  }
  Tensor expanded_self, expanded_index;
  std::tie(expanded_self, expanded_index, std::ignore) = expand_gather(self, dim, index);
  return at::_gather(expanded_self, dim, expanded_index, sparse_grad);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_(self, dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand_scatter(self, dim, index, source);
  TORCH_CHECK(self_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_(self, dim, expanded_index, expanded_source);
}

Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_(self, dim, index, value);
  }
  Tensor expanded_index;
  std::vector<int64_t> result_sizes;
  std::tie(result_sizes, expanded_index) = expand_scatter_scalar(self, dim, index);
  TORCH_CHECK(result_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_(self, dim, expanded_index, value);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_(dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand_scatter(self, dim, index, source);
  Tensor ret = self.clone().expand(self_sizes);
  return at::_scatter_(ret, dim, expanded_index, expanded_source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_(dim, index, value);
  }
  Tensor expanded_index;
  std::vector<int64_t> result_sizes;
  std::tie(result_sizes, expanded_index) = expand_scatter_scalar(self, dim, index);
  Tensor ret = self.clone().expand(result_sizes);
  return at::_scatter_(ret, dim, expanded_index, value);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self;
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return at::_scatter_add_(self, dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand_scatter(self, dim, index, source);
  TORCH_CHECK(self_sizes == self.sizes(), "broadcasting change the shape of self");
  return at::_scatter_add_(self, dim, expanded_index, expanded_source);
}

Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return self.scatter_add_(dim, index, at::full({}, value, self.options()));
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  if (index.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0 || index.dim() == 0) {
    return self.clone().scatter_add_(dim, index, source);
  }
  Tensor expanded_source, expanded_index;
  std::vector<int64_t> self_sizes;
  std::tie(self_sizes, expanded_index, expanded_source) = expand_scatter(self, dim, index, source);
  Tensor ret = self.clone().expand(self_sizes);
  return at::_scatter_add_(ret, dim, expanded_index, expanded_source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  return at::scatter_add(self, dim, index, at::full({}, value, self.options()));
}

Tensor _gather_sparse_backward(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& grad){
// special case scalar input and/or index
    if (self.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(at::empty({0,grad.numel()}, index.options()), grad, self.sizes());
    if (grad.ndimension() == 0) return at::_sparse_coo_tensor_unsafe(index.view({1,1}), grad, self.sizes());
    Tensor sparse_ind = at::empty({self.ndimension(), grad.numel()}, self.options().dtype(at::kLong));
    int64_t n_above = grad.numel();
    int64_t n_below = 1;
    if (dim < 0) dim += self.ndimension();
    for (int i=0; i<self.ndimension(); i++) {
        n_above /= grad.size(i);
        if (i == dim) {
            sparse_ind[i] = index.reshape(-1);
        } else {
            sparse_ind[i] = at::arange(grad.size(i),self.options().dtype(at::kLong)).unsqueeze(1).expand({grad.size(i), n_above}).reshape(-1).repeat(n_below);
        }
        n_below *= grad.size(i);
    }
    return at::_sparse_coo_tensor_unsafe(sparse_ind, grad.reshape(-1), self.sizes());
}

}} // at::native
