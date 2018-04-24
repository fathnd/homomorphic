#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCAtomics.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>

const int WARP_SIZE = 32;
const int MODE_SUM = 0;
const int MODE_MEAN = 1;
const int MODE_MAX = 2;

namespace at {
namespace native {

namespace {

template <typename scalar_t>
__global__ void EmbeddingBag_updateOutputKernel(
    int64_t *input, int64_t *offsets, scalar_t *weight, scalar_t *output,
    int64_t *offset2bag, int64_t numIndices, int64_t numBags, int64_t stride,
    int mode, int64_t *bag_size, int64_t *max_indices) {

  // the strategy here is that each bag x feature is handled by a single thread

  using accscalar_t = acc_type<scalar_t, true>;
  int64_t chunksPerBag = THCCeilDiv(stride, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;
      scalar_t *weightFeat = weight + featureDim;
      int64_t begin = offsets[bag];
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
      assert(end >= begin);

      accscalar_t weightFeatSum = scalar_cast<accscalar_t>(0);
      scalar_t weightFeatMax;

      int64_t bag_size_ = 0;
      int64_t maxWord = -1;
      for (int64_t emb = begin; emb < end; emb++) {
        const int weightRow = ((int)input[emb]) * stride;
        scalar_t weightValue = weightFeat[weightRow];

        if (mode == MODE_MAX) {
          if (emb == begin || weightValue > weightFeatMax) {
            weightFeatMax = weightValue;
            maxWord = input[emb]; 
          }
        } else {
          weightFeatSum += scalar_cast<accscalar_t>(weightValue);
        }

        bag_size_++;
        if (featureDim == 0) {
          offset2bag[emb] = bag;
        }
      }
      if (mode == MODE_MEAN) {
        weightFeatSum = weightFeatSum / scalar_cast<accscalar_t>(bag_size_);
        bag_size[bag] = bag_size_;
      }

      if (mode == MODE_MEAN || mode == MODE_SUM) {
        output[bag * stride + featureDim] = scalar_cast<scalar_t>(weightFeatSum);
      }
      else if (mode == MODE_MAX) {
        max_indices[bag * stride + featureDim] = maxWord;
        output[bag * stride + featureDim] = weightFeatMax;
      }
    }
  }
}

// FIXME: removed the accGradParametersKernelByFeature case present in
// LookupTable. That kernel is faster at small sizes (<768 indices), which
// does not need EmbeddingBag (LookupTable + Sum works fine), but would
// still be nice to not be slow in that case.

template <typename scalar_t>
__global__ void EmbeddingBag_accGradParametersKernel_sum_avg(
    int64_t *input, int64_t *indices, scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t *offset2bag, int64_t *count, ptrdiff_t numel,
    int64_t stride, int mode, int64_t *bag_size) {

  using accscalar_t = acc_type<scalar_t, true>;
  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel && (idx == 0 || input[idx] != input[idx - 1])) {
    do {
      const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weightRow = ((int)input[idx]) * stride;

      // Note: only this line changes from LookupTable_accgradParametersKernel
      const int origRow = ((int)indices[idx]);
      const int seq_number = offset2bag[origRow];
      const int gradOutputRow = ((int)seq_number) * stride;

      const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;

      accscalar_t gradient[SZ];
      accscalar_t weight[SZ];

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride) {
          gradient[ii] =
              scalar_cast<accscalar_t>(gradOutput[gradOutputRow + featureDim]);
          if (mode == MODE_MEAN) {
            gradient[ii] /= bag_size[seq_number];
          }
          weight[ii] =
              scalar_cast<accscalar_t>(gradWeight[weightRow + featureDim]);
        }
      }

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        weight[ii] += gradient[ii] * scale;
      }

#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride) {
          gradWeight[weightRow + featureDim] =
              scalar_cast<scalar_t>(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}


Tensor embedding_bag_backward_cuda_sum_avg(
                                   const Tensor &grad, 
                                   const Tensor &indices,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size_,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode) {

  Tensor &bag_size = const_cast<Tensor &>(bag_size_);

  auto grad_weight = at::zeros(grad.type(), {num_weights, grad.size(1)});

  cudaStream_t stream = globalContext().getCurrentCUDAStream();

  ptrdiff_t numel = indices.numel();
  int64_t stride = grad_weight.stride(0);

  auto sorted_indices = indices.type().tensor(indices.sizes());
  auto orig_indices = indices.type().tensor(indices.sizes());
  using device_ptr = thrust::device_ptr<int64_t>;

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    sorted_indices.copy_(indices);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential indices
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_indices.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + numel, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + numel, orig_data,
                        ThrustLTOp<int64_t>());
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = indices.type().tensor(indices.sizes());

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Compute an increasing sequence per unique item in sortedIndices:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    auto count_data = device_ptr(count.data<int64_t>());
    thrust::inclusive_scan_by_key(policy, sorted_data, sorted_data + numel,
                                  thrust::make_constant_iterator(1),
                                  count_data);

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
        policy, thrust::make_reverse_iterator(sorted_data + numel),
        thrust::make_reverse_iterator(sorted_data),
        thrust::make_reverse_iterator(count_data + numel),
        thrust::make_reverse_iterator(count_data + numel),
        thrust::equal_to<int64_t>(), thrust::maximum<int64_t>());
  }

  dim3 grid(THCCeilDiv(numel, (ptrdiff_t)4), THCCeilDiv(stride, (int64_t)128));
  dim3 block(32, 4);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.type(), "embedding_bag_backward_cuda_sum_avg_kernel", [&] {
        using cuda_scalar_t = cuda::type<scalar_t>;
        EmbeddingBag_accGradParametersKernel_sum_avg<
            cuda_scalar_t><<<grid, block, 0, stream>>>(
            sorted_indices.data<int64_t>(), orig_indices.data<int64_t>(),
            grad.data<cuda_scalar_t>(), grad_weight.data<cuda_scalar_t>(),
            offset2bag.data<int64_t>(),
            count.defined() ? count.data<int64_t>() : nullptr, numel, stride,
            mode, bag_size.data<int64_t>());
      });

  THCudaCheck(cudaGetLastError());
  return grad_weight;
}

template <typename scalar_t>
__global__ void EmbeddingBag_accGradParametersKernel_max(
    int64_t *max_indices, scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t stride, int64_t numBags) {

  using accscalar_t = cuda::acc_type<scalar_t>;

  int64_t chunksPerBag = THCCeilDiv(stride, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;

      int64_t word_idx = max_indices[bag * stride + featureDim];
      
      atomicAdd(&(gradWeight[word_idx * stride + featureDim]), gradOutput[bag * stride + featureDim]);
    }
  }
}

Tensor embedding_bag_backward_cuda_max(const Tensor &grad,
                                   const Tensor &max_indices,
                                   int64_t num_weights) {

  auto grad_weight = at::zeros(grad.type(), {num_weights, grad.size(1)});

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  cudaStream_t stream = globalContext().getCurrentCUDAStream();

  dim3 block = dim3(32, 8);
  int grid = 1024;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.type(), "embedding_bag_backward_cuda_max", [&] {
        using cuda_scalar_t = cuda::type<scalar_t>;
        EmbeddingBag_accGradParametersKernel_max<
            cuda_scalar_t><<<grid, block, 0, stream>>>(
            max_indices.data<int64_t>(), grad.data<cuda_scalar_t>(), 
            grad_weight.data<cuda_scalar_t>(), stride, numBags);
      });

  THCudaCheck(cudaGetLastError());
  return grad_weight;
}
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag_cuda(const Tensor &weight, const Tensor &indices,
                   const Tensor &offsets, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag_cuda", indices_arg, kLong);
  checkContiguous("embedding_bag_cuda", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag_cuda", offsets_arg, kLong);
  checkContiguous("embedding_bag_cuda", offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkContiguous("embedding_bag_cuda", weight_arg);
  checkSameGPU("embedding_bag_cuda", weight_arg, indices_arg);
  checkSameGPU("embedding_bag_cuda", weight_arg, offsets_arg);

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);
  int64_t stride = weight.size(1);

  auto bag_size = at::zeros(indices.type(), offsets.sizes());
  auto offset2bag =
      at::zeros(indices.type(), {indices.size(0)}); // offset2bag = [0 0 0 0 0]

  cudaStream_t stream = globalContext().getCurrentCUDAStream();

  auto output = at::zeros(weight.type(), {offsets.size(0), weight.size(1)});
  
  Tensor max_indices;
  
  if (mode == MODE_MAX) {
    max_indices = at::zeros(indices.type(), {offsets.size(0), weight.size(1)});
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::zeros(indices.type(), {0});
  }

  dim3 block = dim3(32, 8);
  int grid = 1024;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.type(), "embedding_bag_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    EmbeddingBag_updateOutputKernel<cuda_scalar_t><<<grid, block, 0, stream>>>(
        indices.data<int64_t>(), offsets.data<int64_t>(),
        weight.data<cuda_scalar_t>(), output.data<cuda_scalar_t>(),
        offset2bag.data<int64_t>(), numIndices, numBags, stride, mode,
        bag_size.data<int64_t>(), mode == MODE_MAX ? max_indices.data<int64_t>() : NULL);
  });

  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

Tensor embedding_bag_backward_cuda(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offsets,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size_,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode) {
  Tensor grad = grad_.contiguous();
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag_cuda", indices_arg, kLong);
  checkContiguous("embedding_bag_cuda", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag_cuda", offsets_arg, kLong);
  checkContiguous("embedding_bag_cuda", offsets_arg);
  auto grad_arg = TensorArg(grad, "grad", 1);
  checkContiguous("embedding_bag_cuda", grad_arg);
  checkSameGPU("embedding_bag_cuda", grad_arg, offsets_arg);
  checkSameGPU("embedding_bag_cuda", grad_arg, indices_arg);

  switch (mode) {
    case MODE_SUM:
    case MODE_MEAN:
      return embedding_bag_backward_cuda_sum_avg(grad, indices, offset2bag, bag_size_, num_weights, scale_grad_by_freq, mode);

    case MODE_MAX:
      return embedding_bag_backward_cuda_max(grad, max_indices, num_weights);

    default:
      AT_ERROR(                                               
          "Unknown mode for embedding_bag_backward_cuda %d", mode);
  }
}

}
}