#pragma once

#include <ATen/native/special/detail/is_complex_v.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename Tp>
Tp
exp2(Tp x) {
  if constexpr (is_complex_v < Tp >)
    return std::pow(numeric_t < Tp > {2}, x);
  else
    return std::exp2(x);
}
}
}
}
}
