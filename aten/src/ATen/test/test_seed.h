#pragma once

#include "ATen/ATen.h"

void manual_seed(uint64_t seed) {
  at::Generator & cpu_gen = at::globalContext().defaultGenerator(at::Backend::CPU);
  cpu_gen.manualSeed(seed);
  if (at::hasCUDA()) {
    at::Generator & cuda_gen = at::globalContext().defaultGenerator(at::Backend::CUDA);
    cuda_gen.manualSeed(seed);
  }
}
