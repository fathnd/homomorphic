#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>

#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace torch {
namespace optim {

struct TORCH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  LBFGSOptions(double lr);
  TORCH_ARG(double, lr) = 1;
  TORCH_ARG(int64_t, max_iter) = 20;
  TORCH_ARG(c10::optional<int64_t>, max_eval) = c10::nullopt;
  TORCH_ARG(double, tolerance_grad) = 1e-7;
  TORCH_ARG(double, tolerance_change) = 1e-9;
  TORCH_ARG(size_t, history_size) = 100;
  TORCH_ARG(c10::optional<std::string>, line_search_fn) = c10::nullopt;
public:
  //void serialize(torch::serialize::InputArchive& archive) override;
  //void serialize(torch::serialize::OutputArchive& archive) const override;
  //TORCH_API friend bool operator==(const LBFGSOptions& lhs, const LBFGSOptions& rhs);
  ~LBFGSOptions() = default;
};

struct TORCH_API LBFGSParamState : public OptimizerCloneableParamState<LBFGSParamState> {
  TORCH_ARG(torch::Tensor, d) = {};
  TORCH_ARG(double, t);
  TORCH_ARG(std::deque<Tensor>, old_dirs);
  TORCH_ARG(std::deque<Tensor>, old_stps);
  TORCH_ARG(std::deque<Tensor>, ro);
  TORCH_ARG(torch::Tensor, H_diag) = {};
  TORCH_ARG(torch::Tensor, prev_flat_grad) = {};
  TORCH_ARG(torch::Tensor, prev_loss) = {};
  TORCH_ARG(std::vector<torch::Tensor>, al);
  TORCH_ARG(int64_t, func_evals) = 0;
  TORCH_ARG(int64_t, n_iter) = 0;

public:
//   void serialize(torch::serialize::InputArchive& archive) override;
//   void serialize(torch::serialize::OutputArchive& archive) const override;
//   TORCH_API friend bool operator==(const LBFGSParamState& lhs, const LBFGSParamState& rhs);
  ~LBFGSParamState() = default;
};

class TORCH_API LBFGS : public Optimizer {
 public:
   explicit LBFGS(std::vector<OptimizerParamGroup> param_groups,
       LBFGSOptions defaults) : Optimizer(std::move(param_groups), std::make_unique<LBFGSOptions>(defaults)) {
      TORCH_CHECK(param_groups_.size() == 1, "LBFGS doesn't support per-parameter options (parameter groups)");
      _params = param_groups_[0].params();
      _numel_cache = c10::nullopt;
   }

  Tensor step(LossClosure closure) override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  std::vector<Tensor> _params;
  c10::optional<int64_t> _numel_cache;
  int64_t _numel();
  Tensor _gather_flat_grad();
  void add_grad(const torch::Tensor& step_size, const Tensor& update);
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    //something
  }
};
} // namespace optim
} // namespace torch
