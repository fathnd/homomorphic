
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

class LocalSyncInserter final : private OptOutDispatch {
 public:
  static void InsertSyncs(Expr* expr) {
    LocalSyncInserter sync_inserter;
    sync_inserter.handle(expr);
  }

  void handle(Expr* expr) final {
    if (ir_utils::isTVOp(expr)) {
      // For this SyncInserter
      (!initial_sync_) ? hasOutputSmemExpr(expr, initial_)
                       : hasInputSmemExpr(expr, final_);

      // For parent SyncInserter
      hasOutputSmemExpr(expr, all_smem_outputs_);
      hasInputSmemExpr(expr, all_smem_inputs_);
    } else {
      OptOutDispatch::handle(expr);
    }
  }

  const std::unordered_set<const TensorView*>& initial() const {
    return initial_;
  }

  const std::unordered_set<const TensorView*>& final() const {
    return final_;
  }

  const std::unordered_set<const TensorView*>& all_smem_inputs() const {
    return all_smem_inputs_;
  }

  const std::unordered_set<const TensorView*>& all_smem_outputs() const {
    return all_smem_outputs_;
  }

 private:
  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

  void handle(kir::ForLoop* fl) final {
    // Track if last op in body is sync in nested for-loop
    bool is_last_op_sync_ = false;
    for (auto expr : fl->body().exprs()) {
      is_last_op_sync_ = false;
      if (expr->getExprType().value() == ExprType::Sync) {
        initial_sync_ = true;
        final_.clear();
      } else if (expr->getExprType().value() == ExprType::ForLoop) {
        // Recursively handle nested for-loop
        LocalSyncInserter child_sync_inserter;
        child_sync_inserter.handle(expr);
        const auto& child_inputs = child_sync_inserter.all_smem_inputs();
        const auto& child_outputs = child_sync_inserter.all_smem_outputs();

        // Default - Track all smem inputs / outputs
        all_smem_inputs_.insert(child_inputs.begin(), child_inputs.end());
        all_smem_outputs_.insert(child_outputs.begin(), child_outputs.end());

        if (!initial_sync_) {
          // Parent - None
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child Outputs to Parent Initial
            initial_.insert(child_outputs.begin(), child_outputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Parent first sync
            // Inherit Child Initial / Clear Parent Final
            initial_sync_ = true;
            is_last_op_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.clear();
          } else {
            // Child - 1+
            // Parent first sync
            // Inherit Child Initial + Final
            initial_sync_ = true;
            initial_.insert(
                child_sync_inserter.initial().begin(),
                child_sync_inserter.initial().end());
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        } else {
          // Parent - 1+
          if (!child_sync_inserter.initial_sync_) {
            // Child - None
            // Append All Child to Parent Last
            final_.insert(child_inputs.begin(), child_inputs.end());
          } else if (child_sync_inserter.has_war_hazard_sync_) {
            // Child - WAR race
            // Clear Parent Last / Discard Child Initial
            is_last_op_sync_ = true;
            final_.clear();
          } else {
            // Child - 1+
            // Inherit Child Final / Discard Child Initial
            final_.insert(
                child_sync_inserter.final().begin(),
                child_sync_inserter.final().end());
          }
        }
      } else {
        handle(expr);
      }
    }

    // This level of the nested for-loop may not exist in the kernel.
    // However, subsequent levels can exist, so we handle the body of the
    // for-loop first.
    if (!fl->iter_domain()->isThread() && !fl->iter_domain()->isBroadcast()) {
      // Determine if any smem TV is written to at beginning of the for-loop
      // and whether that smem TV is read from at the end of the for-loop
      // Insert new SyncThreads at end of for-loop to prevent WAR race condition
      if (detect_intersection(initial_, final_) &&
          fl->body().exprs().back()->getExprType().value() != ExprType::Sync &&
          !is_last_op_sync_) {
        // std::cout << "WAR race detected; Add Sync" << std::endl;
        has_war_hazard_sync_ = true;
        kir::IrBuilder ir_builder(GpuLower::current()->kernel());
        fl->body().push_back(ir_builder.create<kir::Sync>(true));
      }
    }
  }

  bool detect_intersection(
      std::unordered_set<const TensorView*>& left,
      std::unordered_set<const TensorView*>& right) {
    for (auto item : left) {
      if (right.find(item) != right.end()) {
        return true;
      }
    }
    return false;
  }

  void hasOutputSmemExpr(
      Expr* expr,
      std::unordered_set<const TensorView*>& set) {
    for (auto out : expr->outputs()) {
      if (ir_utils::isTV(out)) {
        auto tv = out->as<TensorView>();
        if (tv->getMemoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

  void hasInputSmemExpr(
      Expr* expr,
      std::unordered_set<const TensorView*>& set) {
    for (auto inp : expr->inputs()) {
      if (ir_utils::isTV(inp)) {
        auto tv = inp->as<TensorView>();
        if (tv->getMemoryType() == MemoryType::Shared) {
          set.insert(tv);
        }
      }
    }
  }

 private:
  // Track Shared Memory Inputs (Reads) for parent for-loop
  std::unordered_set<const TensorView*> all_smem_inputs_;

  // Track Shared Memory Outputs (Writes) for parent for-loop
  std::unordered_set<const TensorView*> all_smem_outputs_;

  // Shared Memory Writes at beginning of the for-loop
  // before first SyncThreads
  std::unordered_set<const TensorView*> initial_;

  // Shared Memory Reads at end of the for-loop
  // Cleared after each SyncThreads
  std::unordered_set<const TensorView*> final_;

  // Track first sync found in for-loop
  bool initial_sync_ = false;

  // Track sync was inserted for war hazard
  bool has_war_hazard_sync_ = false;
};

} // namespace

std::vector<Expr*> insertThreadSynchronization(
    Fusion* fusion,
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("insertThreadSynchronization");
  FusionGuard fg(fusion);
  std::vector<Expr*> mutated_exprs;
  for (auto expr : exprs) {
    LocalSyncInserter::InsertSyncs(expr);
    mutated_exprs.push_back(expr);
  }
  return mutated_exprs;
}

} // namespace fuser
} // namespace jit
} // namespace torch
