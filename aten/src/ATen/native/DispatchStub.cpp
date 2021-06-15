#include <ATen/native/DispatchStub.h>

#include <c10/util/Exception.h>

#include <cpuinfo.h>
#include <cstdlib>
#include <cstring>

namespace at { namespace native {

static CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
#ifdef HAVE_VSX_CPU_DEFINITION
    if (strcmp(envar, "vsx") == 0) {
      return CPUCapability::VSX;
    }
#else
    if (strcmp(envar, "avx512_256") == 0) {
      return CPUCapability::AVX512_256;
    }
    if (strcmp(envar, "avx512") == 0) {
      return CPUCapability::AVX512;
    }
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
#endif
    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

#if !defined(__powerpc__) && !defined(__s390x__)
  if (cpuinfo_initialize()) {
    // 'AVX512' can only be used if ATEN_CPU_CAPABILITY is set to AVX512.
    // Otherwise AVX512_256 would be used.
    // AVX512_256 doesn't require avx512bw & avx512dq
    // but some old compilers don't support all AVX512 instruction sets,
    // or have bugs
    if (cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() &&  \
        cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_fma3()) {
#ifdef _WIN32
      _putenv("ATEN_CPU_CAPABILITY=avx512_256");
#else
      setenv("ATEN_CPU_CAPABILITY", "avx512_256", 1);
#endif
      return CPUCapability::AVX512_256;
    }
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
#ifdef _WIN32
      _putenv("ATEN_CPU_CAPABILITY=avx2");
#else
      setenv("ATEN_CPU_CAPABILITY", "avx2", 1);
#endif
      return CPUCapability::AVX2;
    }
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
#ifdef _WIN32
      _putenv("ATEN_CPU_CAPABILITY=vsx");
#else
      setenv("ATEN_CPU_CAPABILITY", "vsx", 1);
#endif
  return CPUCapability::VSX;
#else
#ifdef _WIN32
      _putenv("ATEN_CPU_CAPABILITY=default");
#else
      setenv("ATEN_CPU_CAPABILITY", "default", 1);
#endif
  return CPUCapability::DEFAULT;
#endif
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

void* DispatchStubImpl::get_call_ptr(
  DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_256_CPU_DEFINITION
  , void *AVX512_256
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
) {
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        fptr = choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_256_CPU_DEFINITION
          , AVX512_256
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
        );
        cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
      }
      return fptr;
    }

    case DeviceType::CUDA:
      TORCH_INTERNAL_ASSERT(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
      return cuda_dispatch_ptr;

    case DeviceType::HIP:
      TORCH_INTERNAL_ASSERT(hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
      return hip_dispatch_ptr;

    default:
      AT_ERROR("DispatchStub: unsupported device type", device_type);
  }
}

void* DispatchStubImpl::choose_cpu_impl(
  void *DEFAULT
#ifdef HAVE_AVX512_256_CPU_DEFINITION
  , void *AVX512_256
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_256_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512_256)) {
    TORCH_INTERNAL_ASSERT(AVX512_256, "DispatchStub: missing AVX512_256 kernel");
    return AVX512_256;
  }
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    TORCH_INTERNAL_ASSERT(AVX512, "DispatchStub: missing AVX512 kernel");
    return AVX512;
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
    return AVX2;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    TORCH_INTERNAL_ASSERT(VSX, "DispatchStub: missing VSX kernel");
    return VSX;
  }
#endif
  TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
  return DEFAULT;
}

}}  // namespace at::native
