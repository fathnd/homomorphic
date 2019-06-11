#include <torch/csrc/jit/print_handler.h>

namespace torch {
namespace jit {

std::atomic<PrintHandler> print_handler([](const std::string& str) {
  std::cout << str;
});

PrintHandler getPrintHandler() {
  return print_handler.load();
}

void setPrintHandler(PrintHandler ph) {
  print_handler.store(ph);
}

} // namespace jit
} // namespace torch
