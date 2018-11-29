#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/operator.h"

#include <sstream>
#include <regex>

namespace torch {
namespace jit {

namespace {
RegisterOperators reg({
    Operator(
        "aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("split_with_sizes");
          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        }),
    Operator(
        "aten::Size(int[] sizes) -> int[]",
        [](Stack& stack) {
          return 0;
        }),
    Operator(
        "aten::size(Tensor self) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto result = (std::move(pop(stack))).toTensor().sizes();
          pack(stack, std::move(result));
          return 0;
        }),
    // reference _list_with_default in utils.py
    Operator(
      "aten::list_with_default(int list, int[] defaults) -> int",
      [](Stack& stack) {
        IValue single_val;
        IValue defaults;
        pop(stack, single_val, defaults);
        push(stack, single_val);
        return 0;
      }),
    Operator(
        "aten::list_with_default(int?[] list, int[] defaults) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          std::vector<IValue> list;
          std::vector<int64_t> defaults;
          pop(stack, list, defaults);

          JIT_ASSERT(defaults.size() >= list.size());
          std::vector<int64_t> ret_list;
          for (size_t i = 0; i < list.size(); ++i) {
            if (!list[i].isNone()) {
              ret_list.push_back(list[i].toInt());
            } else {
              ret_list.push_back(defaults[i]);
            }
          }

          push(stack, ret_list);
          return 0;
        }),
    Operator(
        "aten::format(str self, ...) -> str",
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options("\\{(.*)\\}");
          return [num_inputs, unsupported_options](Stack& stack) {
            auto format = peek(stack, 0, num_inputs).toStringRef();

            if (std::regex_search(format, unsupported_options)) {
              AT_WARN("Format options are not supported.");
            }

            auto args = last(stack, num_inputs - 1);
            std::stringstream ss;
            for(size_t begin = 0, used_args = 0; true; ++used_args) {
              size_t loc = format.find("{}", begin);
              if(loc == std::string::npos) {
                ss << format.substr(begin);
                break;
              }
              ss << format.substr(begin, loc - begin);
              if(used_args >= args.size()) {
                AT_ERROR("Too few arguments for format string: ", format);
              }
              ss << args[used_args];
              begin = loc + 2;
            }

            drop(stack, num_inputs);
            push(stack, ss.str());
            return 0;
          };
        })
});
}
} // namespace jit
} // namespace torch
