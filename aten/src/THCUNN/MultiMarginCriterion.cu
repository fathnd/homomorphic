#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#define MULTIMARGIN_THREADS 128

template <int64_t P, typename Dtype, typename Acctype>
__global__ void cunn_MultiMarginCriterion_updateOutput_kernel(Dtype *output, Dtype *input, THCIndex_t *target, Dtype *weights, int64_t nframe, int64_t dim, bool sizeAverage, Dtype margin)
{
  __shared__ Acctype buffer[MULTIMARGIN_THREADS];
  int64_t k = blockIdx.x;
  Dtype *input_k = input + k*dim;
  Dtype *output_k = output + k;
  int64_t target_k = ((int64_t)target[k]) - TH_INDEX_BASE;
  Dtype input_target_k = input_k[target_k];

  int64_t i_start = threadIdx.x;
  int64_t i_end = dim;
  int64_t i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int64_t i = i_start; i < i_end; i += i_step)
  {
    Dtype z = margin - input_target_k + input_k[i];
    if (i == target_k)
      continue;

    if (z > 0) {
      Dtype h = (P==1) ? z : z*z;
      if(weights)
        h *= weights[target_k];
      buffer[threadIdx.x] += h;
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    Acctype sum = 0;
    for (int64_t i=0; i < blockDim.x; i++)
      sum += buffer[i];

    *output_k = ScalarConvert<Acctype, Dtype>::to(sum/dim);
    if(sizeAverage)
      *output_k /= nframe;
  }
}

template <int64_t P, typename Dtype, typename Acctype>
__global__ void cunn_MultiMarginCriterion_updateGradInput_kernel(Dtype *gradInput,
                                                                 Dtype *gradOutput,
                                                                 Dtype *input,
                                                                 THCIndex_t *target,
                                                                 Dtype *weights,
                                                                 int64_t nframe,
                                                                 int64_t dim,
                                                                 bool sizeAverage,
                                                                 Dtype margin,
                                                                 int64_t reduce)
{
  __shared__ Acctype buffer[MULTIMARGIN_THREADS];
  int64_t k = blockIdx.x;
  Dtype *input_k = input + k*dim;
  Dtype *gradInput_k = gradInput + k*dim;
  int64_t target_k = ((int64_t)target[k]) - TH_INDEX_BASE;
  Dtype input_target_k = input_k[target_k];

  Dtype *gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }

  Acctype g = (sizeAverage && reduce ? 1./((Acctype)(nframe*dim)) : 1./((Acctype)dim));

  int64_t i_start = threadIdx.x;
  int64_t i_end = dim;
  int64_t i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int64_t i=i_start; i<i_end; i+=i_step)
  {
    Dtype z = margin - input_target_k + input_k[i];
    if (i == target_k)
      continue;

    if (z > 0)
    {
      Dtype h = ScalarConvert<Acctype, Dtype>::to((P == 1) ? g : 2*g*z);
      if(weights)
        h *= weights[target_k];
      buffer[threadIdx.x] -= h;
      gradInput_k[i] = h;
    }
    else
      gradInput_k[i] = ScalarConvert<int, Dtype>::to(0);
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    Acctype gradInput_target_k = 0;
    for (int64_t i=0; i<blockDim.x; i++)
      gradInput_target_k += buffer[i];
    gradInput_k[target_k] = ScalarConvert<Acctype, Dtype>::to(gradInput_target_k);
  }

  for (int64_t i=i_start; i<i_end; i+= i_step)
  {
    gradInput_k[i] *= * gradOutput_k;
  }
}

#include "generic/MultiMarginCriterion.cu"
#include "THCGenerateFloatTypes.h"

#undef MULTIMARGIN_THREADS
