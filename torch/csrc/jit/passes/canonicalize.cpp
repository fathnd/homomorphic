#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/ir_views.h>

namespace torch {
namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
// keep_unique_names: If false, canonicalizes unique names by removing them
//   and replacing them with normal value names.
//   Otherwise, ignores values with unique names.
std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names) {
  auto r = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> rn_env;
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input);
    if (!keep_unique_names)
      r_input->setUniqueName("");
    rn_env[input] = r_input;
  }
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(node, rn_fn);
    if (!keep_unique_names) {
      for (auto* output : r_node->outputs()) {
        output->setUniqueName("");
      }
    }
    r->appendNode(r_node);
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }
    if (node->hasAttribute(attr::Subgraph)) {
      r_node->g_(
          attr::Subgraph,
          Canonicalize(node->g(attr::Subgraph), keep_unique_names));
    }
  }
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }

  return r;
}

// Which index in b's owning Node is b
size_t blockIndex(const Block* b) {
  auto n = b->owningNode();
  AT_ASSERT(n);
  for (size_t i = 0; i < n->blocks().size(); ++i) {
    if (n->blocks()[i] == b) {
      return i;
    }
  }
  AT_ASSERT(false);
}

size_t blocksFromGraphBlock(Node* n) {
  size_t dist = 0;
  while (n->owningBlock()->owningNode()) {
    n = n->owningBlock()->owningNode();
    ++dist;
  }
  return dist;
}

/*
 * This establishes a canonical ordering of nodes.
 * If n1 and n2 are in the same block, whichever node appears first
 * is before the other.
 * If n1 and n2 are contained in different blocks of an if node,
 * then whichever block is in the true block is ordered before the other.
 * If n1 contains n2, then n1 is before n2. This has the nice property that
 * whichever node appears first in a dump of the graph is before the other.
 * NB: this is not a topological index. Topologically, two nodes in
 * different blocks of an if node are not topologically < or > each other.
 */
bool isBefore(Node* n1, Node* n2) {
  // Invalid to call with the same node as both args
  AT_ASSERT(n1 != n2);

  // Set n1 and n2 to be the number of blocks from the Graph block
  size_t d_1 = blocksFromGraphBlock(n1);
  size_t d_2 = blocksFromGraphBlock(n2);

  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();
    // n2 contains n1
    if (n1 == n2) {
      return false;
    }
  }

  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();
    // n1 contains n2
    if (n2 == n1) {
      return true;
    }
  }

  // Now they are the same numer of blocks from the graph block,
  // recurse upwards, checking if they are on the same block
  while (true) {
    if (n1->owningBlock() == n2->owningBlock()) {
      return n1->isBefore(n2);
    }

    auto new_n1 = n1->owningBlock()->owningNode();
    auto new_n2 = n2->owningBlock()->owningNode();

    AT_ASSERT(new_n1 != nullptr);
    AT_ASSERT(new_n2 != nullptr);

    if (new_n1 == new_n2) {
      // take whichever node is in the earlier block
      auto index_1 = blockIndex(n1->owningBlock());
      auto index_2 = blockIndex(n2->owningBlock());
      return index_1 < index_2;
    }

    n1 = new_n1;
    n2 = new_n2;
  }
}

std::vector<size_t> sort_indexes(at::ArrayRef<Value*> values) {
  // initialize original index locations
  std::vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);

  // Sort values based on canonical ordering of their first usage
  std::sort(idx.begin(), idx.end(), [&values](size_t i1, size_t i2) {
    if (values[i1]->uses().size() == 0 && values[i2]->uses().size() == 0) {
      return i1 < i2;
    }
    if (values[i1]->uses().size() == 0) {
      return false;
    } else if (values[i2]->uses().size() == 0) {
      return true;
    }

    auto fst_v1 = values[i1]->uses()[0];
    auto fst_v2 = values[i2]->uses()[0];

    // If two values first usage is the same node, we order on offset
    if (fst_v1.user == fst_v2.user) {
      return fst_v1.offset < fst_v2.offset;
    }

    return isBefore(fst_v1.user, fst_v2.user);
  });

  return idx;
}

std::string uniqueName(Value* v) {
  return v->hasUniqueName() ? v->uniqueName() : "";
}

void CanonicalizeOutputs(Block* block);

void swapIfNodeOutputs(Node* n, const std::vector<size_t>& new_indices) {
  for (size_t index : new_indices) {
    auto orig = n->outputs().at(index);
    auto new_out =
        n->addOutput()->setUniqueName(uniqueName(orig))->setType(orig->type());
    orig->replaceAllUsesWith(new_out);
  }
  while (n->outputs().size() > new_indices.size()) {
    n->eraseOutput(0);
  }
}

void swapIfBlockOutputs(Block* b, const std::vector<size_t>& new_indices) {
  for (size_t index : new_indices) {
    b->registerOutput(b->outputs().at(index));
  }
  for (size_t i = 0; i < new_indices.size(); ++i) {
    b->eraseOutput(0);
  }
}

void swapLoopBlockOutputs(Node* n, const std::vector<size_t>& new_indices) {
  LoopView loop(n);
  Block* b = loop.bodyBlock();
  for (size_t index : new_indices) {
    b->registerOutput(loop.bodyCarriedOutputs().at(index));
    auto orig_inp = loop.bodyCarriedInputs().at(index);
    auto new_inp = b->addInput(uniqueName(orig_inp))->setType(orig_inp->type());
    orig_inp->replaceAllUsesWith(new_inp);
  }

  constexpr size_t body_carried_offset = 1;
  for (size_t i = 0; i < new_indices.size(); ++i) {
    b->eraseOutput(body_carried_offset);
    b->eraseInput(body_carried_offset);
  }
}

void swapLoopNodeInputs(Node* n, const std::vector<size_t>& new_indices) {
  LoopView loop(n);
  for (size_t index : new_indices) {
    auto orig = n->outputs().at(index);
    auto new_out =
        n->addOutput()->setUniqueName(uniqueName(orig))->setType(orig->type());
    orig->replaceAllUsesWith(new_out);
  }

  for (size_t index : new_indices) {
    auto orig = loop.carriedInputs().at(index);
    n->addInput(orig);
  }

  for (size_t i = 0; i < new_indices.size(); ++i) {
    n->eraseOutput(0);
    n->removeInput(/*carried_inputs_offset*/ 2);
  }
}

void CanonicalizeLoopOutputs(Node* n) {
  auto new_indices = sort_indexes(n->outputs());
  swapLoopBlockOutputs(n, new_indices);
  swapLoopNodeInputs(n, new_indices);
}

void CanonicalizeIfOutputs(Node* n) {
  auto new_indices = sort_indexes(n->outputs());
  swapIfBlockOutputs(n->blocks().at(0), new_indices);
  swapIfBlockOutputs(n->blocks().at(1), new_indices);
  swapIfNodeOutputs(n, new_indices);
}

void CanonicalizeOutputs(Block* block) {
  // We iterate in reverse since ordering of a node's outputs is dependent on
  // the value use following it in the graph
  for (Node* n : block->nodes().reverse()) {
    switch (n->kind()) {
      case prim::Loop: {
        CanonicalizeLoopOutputs(n);
      } break;
      case prim::If: {
        CanonicalizeIfOutputs(n);
      } break;
    }
    // Since an a control flow node's outputs are after
    // the values outputted within its blocks, first canonicalize
    // the nodes outputs and then recurse on its blocks
    for (Block* b : n->blocks()) {
      CanonicalizeOutputs(b);
    }
  }
}

// Canonicalize a graph's control flow node outputs. We do this to solve jitter
// issues with outputs added to control flow nodes after the first pass of
// compilation in compiler.cpp
void CanonicalizeOutputs(std::shared_ptr<Graph>& graph) {
  CanonicalizeOutputs(graph->block());
}
} // namespace jit
} // namespace torch
