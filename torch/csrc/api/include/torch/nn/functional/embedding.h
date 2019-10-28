#pragma once

#include <torch/nn/options/embedding.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor one_hot(const Tensor& tensor, int64_t num_classes = -1) {
  return torch::one_hot(tensor, num_classes);
}

inline Tensor embedding(const Tensor& input, Tensor weight, EmbeddingOptions options) {
  if (options.padding_idx() != c10::nullopt) {
    if (*options.padding_idx() > 0) {
      TORCH_CHECK(*options.padding_idx() < weight.size(0), "Padding_idx must be within num_embeddings");
    }
    else if (*options.padding_idx() < 0) {
      TORCH_CHECK(*options.padding_idx() >= -weight.size(0), "Padding_idx must be within num_embedding");
      options.padding_idx(weight.size(0) + *options.padding_idx());
    }
  } else {
    options.padding_idx(-1);
  }

  if (options.max_norm() != c10::nullopt) {
    torch::NoGradGuard no_grad;
    torch::embedding_renorm_(weight, input.contiguous(), *options.max_norm(), options.norm_type());
  }
  return torch::embedding(weight, input.contiguous(), *options.padding_idx(), options.scale_grad_by_freq(), options.sparse());
}

inline Tensor embedding_bag(
    const Tensor& input,
    Tensor weight,
    const EmbeddingBagOptions& options = {}) {

  Tensor input_ = input;
  Tensor offsets_ = options.offsets();
  Tensor per_sample_weights_ = options.per_sample_weights();

  TORCH_CHECK(!per_sample_weights_.defined() || input.sizes() == per_sample_weights_.sizes(),
    "embedding_bag: If per_sample_weights (", per_sample_weights_.sizes(), ") is not null, then it must have the same shape as the input (", input.sizes(), ")");
  if (input.dim() == 2) {
    TORCH_CHECK(!offsets_.defined(),
      "If input is 2D, then offsets has to be null, as input is treated is a mini-batch of fixed length sequences. However, found offsets of type Tensor");
    offsets_ = torch::arange(0, input.numel(), input.size(1),
                                 torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    input_ = input_.reshape(-1);
    if (per_sample_weights_.defined()) {
      per_sample_weights_ = per_sample_weights_.reshape(-1);
    }
  } else if (input.dim() == 1) {
    TORCH_CHECK(offsets_.defined(), "offsets has to be a 1D Tensor but got null");
    TORCH_CHECK(offsets_.dim() == 1, "offsets has to be a 1D Tensor");
    TORCH_CHECK(offsets_[0].item<int64_t>() == 0, "offsets[0] has to be 0, i.e., the first sequence in the mini-batch has to start from position 0. However, got ",
     offsets_[0].item<int64_t>());
    TORCH_CHECK(offsets_[-1].item<int64_t>() <= input.size(0), "offsets[-1] can not be greater than input's length({)",
              input.size(0), "}), but got offsets[-1] of {", offsets_[-1].item<int64_t>(), "}");
  } else {
    TORCH_CHECK(false, "input has to be 1D or 2D Tensor,but got Tensor of dimension ", input.dim());
  }

  int mode_enum;
  if (c10::get_if<enumtype::kSum>(&options.mode())) {
    mode_enum = 0;
  } else if (c10::get_if<enumtype::kMean>(&options.mode())) {
    mode_enum = 1;
  } else if (c10::get_if<enumtype::kMax>(&options.mode())) {
    mode_enum = 2;
    TORCH_CHECK(!options.scale_grad_by_freq(), "max mode does not support scaling the gradient by the frequency");
    TORCH_CHECK(!options.sparse(), "max mode does not support sparse weights");
  } else {
    TORCH_CHECK(false, "mode has to be one of sum, mean or max");
  }

  if (options.max_norm() != c10::nullopt) {
    torch::NoGradGuard no_grad;
    torch::embedding_renorm_(weight, input_, *options.max_norm(), options.norm_type());
  }

  TORCH_CHECK(
    !per_sample_weights_.defined() || c10::get_if<enumtype::kMean>(&options.mode()),
    "embedding_bag: per_sample_weights was not null. ",
    "per_sample_weights is only supported for mode='kSum' (got mode='",
    c10::visit(torch::enumtype::enum_name{}, options.mode()), "').Please open a feature request on GitHub.");

  return std::get<0>(
    torch::embedding_bag(
      weight,
      input_,
      offsets_,
      options.scale_grad_by_freq(),
      mode_enum,
      options.sparse(),
      per_sample_weights_));
}
} // namespace functional
} // namespace nn
} // namespace torch
