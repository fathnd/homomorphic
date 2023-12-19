#pragma once

#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

#include <c10/util/ArrayRef.h>
#include <c10/util/complex.h>

namespace c10 {

// NOTE: hash_combine and SHA1 hashing is based on implementation from Boost
//
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

// Creates the SHA1 hash of a string. A 160-bit hash.
// Based on the implementation in Boost (see notice above).
// Note that SHA1 hashes are no longer considered cryptographically
//   secure, but are the standard hash for generating unique ids.
// Usage:
//   // Let 'code' be a std::string
//   c10::sha1 sha1_hash{code};
//   const auto hash_code = sha1_hash.str();
// TODO: Compare vs OpenSSL and/or CryptoPP implementations
struct sha1 {
  typedef unsigned int(digest_type)[5];

  sha1(const std::string& s = "") {
    if (!s.empty()) {
      reset();
      process_bytes(s.c_str(), s.size());
    }
  }

  void reset() {
    h_[0] = 0x67452301;
    h_[1] = 0xEFCDAB89;
    h_[2] = 0x98BADCFE;
    h_[3] = 0x10325476;
    h_[4] = 0xC3D2E1F0;

    block_byte_index_ = 0;
    bit_count_low = 0;
    bit_count_high = 0;
  }

  std::string str() {
    unsigned int digest[5];
    get_digest(digest);

    std::ostringstream buf;
    for (unsigned int i : digest) {
      buf << std::hex << std::setfill('0') << std::setw(8) << i;
    }

    return buf.str();
  }

 private:
  unsigned int left_rotate(unsigned int x, std::size_t n) {
    return (x << n) ^ (x >> (32 - n));
  }

  void process_block_impl() {
    unsigned int w[80];

    for (std::size_t i = 0; i < 16; ++i) {
      w[i] = (block_[i * 4 + 0] << 24);
      w[i] |= (block_[i * 4 + 1] << 16);
      w[i] |= (block_[i * 4 + 2] << 8);
      w[i] |= (block_[i * 4 + 3]);
    }

    for (std::size_t i = 16; i < 80; ++i) {
      w[i] = left_rotate((w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]), 1);
    }

    unsigned int a = h_[0];
    unsigned int b = h_[1];
    unsigned int c = h_[2];
    unsigned int d = h_[3];
    unsigned int e = h_[4];

    for (std::size_t i = 0; i < 80; ++i) {
      unsigned int f = 0;
      unsigned int k = 0;

      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }

      unsigned temp = left_rotate(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = left_rotate(b, 30);
      b = a;
      a = temp;
    }

    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
  }

  void process_byte_impl(unsigned char byte) {
    block_[block_byte_index_++] = byte;

    if (block_byte_index_ == 64) {
      block_byte_index_ = 0;
      process_block_impl();
    }
  }

  void process_byte(unsigned char byte) {
    process_byte_impl(byte);

    // size_t max value = 0xFFFFFFFF
    // if (bit_count_low + 8 >= 0x100000000) { // would overflow
    // if (bit_count_low >= 0x100000000-8) {
    if (bit_count_low < 0xFFFFFFF8) {
      bit_count_low += 8;
    } else {
      bit_count_low = 0;

      if (bit_count_high <= 0xFFFFFFFE) {
        ++bit_count_high;
      } else {
        TORCH_CHECK(false, "sha1 too many bytes");
      }
    }
  }

  void process_block(void const* bytes_begin, void const* bytes_end) {
    unsigned char const* begin = static_cast<unsigned char const*>(bytes_begin);
    unsigned char const* end = static_cast<unsigned char const*>(bytes_end);
    for (; begin != end; ++begin) {
      process_byte(*begin);
    }
  }

  void process_bytes(void const* buffer, std::size_t byte_count) {
    unsigned char const* b = static_cast<unsigned char const*>(buffer);
    process_block(b, b + byte_count);
  }

  void get_digest(digest_type& digest) {
    // append the bit '1' to the message
    process_byte_impl(0x80);

    // append k bits '0', where k is the minimum number >= 0
    // such that the resulting message length is congruent to 56 (mod 64)
    // check if there is enough space for padding and bit_count
    if (block_byte_index_ > 56) {
      // finish this block
      while (block_byte_index_ != 0) {
        process_byte_impl(0);
      }

      // one more block
      while (block_byte_index_ < 56) {
        process_byte_impl(0);
      }
    } else {
      while (block_byte_index_ < 56) {
        process_byte_impl(0);
      }
    }

    // append length of message (before pre-processing)
    // as a 64-bit big-endian integer
    process_byte_impl(
        static_cast<unsigned char>((bit_count_high >> 24) & 0xFF));
    process_byte_impl(
        static_cast<unsigned char>((bit_count_high >> 16) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_high >> 8) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_high) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_low >> 24) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_low >> 16) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_low >> 8) & 0xFF));
    process_byte_impl(static_cast<unsigned char>((bit_count_low) & 0xFF));

    // get final digest
    digest[0] = h_[0];
    digest[1] = h_[1];
    digest[2] = h_[2];
    digest[3] = h_[3];
    digest[4] = h_[4];
  }

  unsigned int h_[5]{};
  unsigned char block_[64]{};
  std::size_t block_byte_index_{};
  std::size_t bit_count_low{};
  std::size_t bit_count_high{};
};

////////////////////////////////////////////////////////////////////////////////
// c10::hash implementation
////////////////////////////////////////////////////////////////////////////////

namespace _hash_detail {

// Use template argument deduction to shorten calls to c10::hash
template <typename T>
size_t simple_get_hash(const T& o);

template <typename T, typename V>
using type_if_not_enum =
    typename std::enable_if<!std::is_enum<T>::value, V>::type;

// Use SFINAE to dispatch to std::hash if possible, cast enum types to int
// automatically, and fall back to T::hash otherwise. NOTE: C++14 added support
// for hashing enum types to the standard, and some compilers implement it even
// when C++14 flags aren't specified. This is why we have to disable this
// overload if T is an enum type (and use the one below in this case).
template <typename T>
auto dispatch_hash(const T& o)
    -> decltype(std::hash<T>()(o), type_if_not_enum<T, size_t>()) {
  return std::hash<T>()(o);
}

template <typename T>
typename std::enable_if<std::is_enum<T>::value, size_t>::type dispatch_hash(
    const T& o) {
  using R = typename std::underlying_type<T>::type;
  return std::hash<R>()(static_cast<R>(o));
}

template <typename T>
auto dispatch_hash(const T& o) -> decltype(T::hash(o), size_t()) {
  return T::hash(o);
}

} // namespace _hash_detail

// Hasher struct
template <typename T>
struct hash {
  size_t operator()(const T& o) const {
    return _hash_detail::dispatch_hash(o);
  };
};

// Specialization for std::tuple
template <typename... Types>
struct hash<std::tuple<Types...>> {
  template <size_t idx, typename... Ts>
  struct tuple_hash {
    size_t operator()(const std::tuple<Ts...>& t) const {
      return hash_combine(
          _hash_detail::simple_get_hash(std::get<idx>(t)),
          tuple_hash<idx - 1, Ts...>()(t));
    }
  };

  template <typename... Ts>
  struct tuple_hash<0, Ts...> {
    size_t operator()(const std::tuple<Ts...>& t) const {
      return _hash_detail::simple_get_hash(std::get<0>(t));
    }
  };

  size_t operator()(const std::tuple<Types...>& t) const {
    return tuple_hash<sizeof...(Types) - 1, Types...>()(t);
  }
};

template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
  size_t operator()(const std::pair<T1, T2>& pair) const {
    std::tuple<T1, T2> tuple = std::make_tuple(pair.first, pair.second);
    return _hash_detail::simple_get_hash(tuple);
  }
};

template <typename T>
struct hash<c10::ArrayRef<T>> {
  size_t operator()(c10::ArrayRef<T> v) const {
    size_t seed = 0;
    for (const auto& elem : v) {
      seed = hash_combine(seed, _hash_detail::simple_get_hash(elem));
    }
    return seed;
  }
};

// Specialization for std::vector
template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(const std::vector<T>& v) const {
    return hash<c10::ArrayRef<T>>()(v);
  }
};

namespace _hash_detail {

template <typename T>
size_t simple_get_hash(const T& o) {
  return c10::hash<T>()(o);
}

} // namespace _hash_detail

// Use this function to actually hash multiple things in one line.
// Dispatches to c10::hash, so it can hash containers.
// Example:
//
// static size_t hash(const MyStruct& s) {
//   return get_hash(s.member1, s.member2, s.member3);
// }
template <typename... Types>
size_t get_hash(const Types&... args) {
  return c10::hash<decltype(std::tie(args...))>()(std::tie(args...));
}

// Specialization for c10::complex
template <typename T>
struct hash<c10::complex<T>> {
  size_t operator()(const c10::complex<T>& c) const {
    return get_hash(c.real(), c.imag());
  }
};

} // namespace c10
