#pragma once

#include "ATen/ATenGeneral.h"
#include <cstring>
#include <limits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace at {

/// Constructors

inline AT_HOSTDEVICE Half::Half(float value) {
#ifdef __CUDA_ARCH__
  x = __half_as_short(__float2half(value));
#else
  x = detail::float2halfbits(value);
#endif
}

/// Implicit conversions

inline AT_HOSTDEVICE Half::operator float() const {
#ifdef __CUDA_ARCH__
  return __half2float(*reinterpret_cast<const __half*>(x));
#else
  return detail::halfbits2float(x);
#endif
}

#ifdef __CUDACC__
inline AT_HOSTDEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline AT_HOSTDEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(this);
}
#endif

/// Arithmetic

inline AT_HOSTDEVICE Half operator+(const Half& a, const Half& b) {
  return (float)a + (float)b;
}

inline AT_HOSTDEVICE Half operator-(const Half& a, const Half& b) {
  return (float)a - (float)b;
}

inline AT_HOSTDEVICE Half operator*(const Half& a, const Half& b) {
  return (float)a * (float)b;
}

inline AT_HOSTDEVICE Half operator/(const Half& a, const Half& b) {
  return (float)a / (float)b;
}

inline AT_HOSTDEVICE Half operator-(const Half& a) {
  return -(float)a;
}

inline AT_HOSTDEVICE Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline AT_HOSTDEVICE Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline AT_HOSTDEVICE Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline AT_HOSTDEVICE Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline AT_HOSTDEVICE float operator+(Half a, float b) { return (float)a + b; }
inline AT_HOSTDEVICE float operator-(Half a, float b) { return (float)a - b; }
inline AT_HOSTDEVICE float operator*(Half a, float b) { return (float)a * b; }
inline AT_HOSTDEVICE float operator/(Half a, float b) { return (float)a / b; }

inline AT_HOSTDEVICE float operator+(float a, Half b) { return a + (float)b; }
inline AT_HOSTDEVICE float operator-(float a, Half b) { return a - (float)b; }
inline AT_HOSTDEVICE float operator*(float a, Half b) { return a * (float)b; }
inline AT_HOSTDEVICE float operator/(float a, Half b) { return a / (float)b; }

inline AT_HOSTDEVICE float& operator+=(float& a, const Half& b) { return a += (float)b; }
inline AT_HOSTDEVICE float& operator-=(float& a, const Half& b) { return a -= (float)b; }
inline AT_HOSTDEVICE float& operator*=(float& a, const Half& b) { return a *= (float)b; }
inline AT_HOSTDEVICE float& operator/=(float& a, const Half& b) { return a /= (float)b; }

/// Arithmetic with doubles

inline AT_HOSTDEVICE double operator+(Half a, double b) { return (double)a + b; }
inline AT_HOSTDEVICE double operator-(Half a, double b) { return (double)a - b; }
inline AT_HOSTDEVICE double operator*(Half a, double b) { return (double)a * b; }
inline AT_HOSTDEVICE double operator/(Half a, double b) { return (double)a / b; }

inline AT_HOSTDEVICE double operator+(double a, Half b) { return a + (double)b; }
inline AT_HOSTDEVICE double operator-(double a, Half b) { return a - (double)b; }
inline AT_HOSTDEVICE double operator*(double a, Half b) { return a * (double)b; }
inline AT_HOSTDEVICE double operator/(double a, Half b) { return a / (double)b; }

/// Arithmetic with ints

inline AT_HOSTDEVICE Half operator+(Half a, int b) { return a + (Half)b; }
inline AT_HOSTDEVICE Half operator-(Half a, int b) { return a - (Half)b; }
inline AT_HOSTDEVICE Half operator*(Half a, int b) { return a * (Half)b; }
inline AT_HOSTDEVICE Half operator/(Half a, int b) { return a / (Half)b; }

inline AT_HOSTDEVICE Half operator+(int a, Half b) { return (Half)a + b; }
inline AT_HOSTDEVICE Half operator-(int a, Half b) { return (Half)a - b; }
inline AT_HOSTDEVICE Half operator*(int a, Half b) { return (Half)a * b; }
inline AT_HOSTDEVICE Half operator/(int a, Half b) { return (Half)a / b; }

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from at::Half to float.

} // namespace at

namespace std {

template<> class numeric_limits<at::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr bool has_denorm = numeric_limits<float>::has_denorm;
  static constexpr bool has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr bool round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr bool digits = 11;
  static constexpr bool digits10 = 3;
  static constexpr bool max_digits10 = 5;
  static constexpr bool radix = 2;
  static constexpr bool min_exponent = -13;
  static constexpr bool min_exponent10 = -4;
  static constexpr bool max_exponent = 16;
  static constexpr bool max_exponent10 = 4;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = numeric_limits<float>::tinyness_before;
  static constexpr at::Half min() { return at::Half(0x400, at::Half::from_bits); }
  static constexpr at::Half lowest() { return at::Half(0xFBFF, at::Half::from_bits); }
  static constexpr at::Half max() { return at::Half(0x7BFF, at::Half::from_bits); }
  static constexpr at::Half epsilon() { return at::Half(0x1400, at::Half::from_bits); }
  static constexpr at::Half round_error() { return at::Half(0x3800, at::Half::from_bits); }
  static constexpr at::Half infinity() { return at::Half(0x7C00, at::Half::from_bits); }
  static constexpr at::Half quiet_NaN() { return at::Half(0x7E00, at::Half::from_bits); }
  static constexpr at::Half signaling_NaN() { return at::Half(0x7D00, at::Half::from_bits); }
  static constexpr at::Half denorm_min() { return at::Half(0x1, at::Half::from_bits); }
};

} // namespace std
