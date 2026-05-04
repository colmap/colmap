/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "solver_tools.h"

namespace caspar {

template <typename T>
void Zero(T* start, T* end) {
  const size_t num_bytes = (end - start) * sizeof(T);
  cudaMemset(start, 0, num_bytes);
}

template <typename T>
void Copy(const T* start, const T* end, T* target) {
  const size_t num_bytes = (end - start) * sizeof(T);
  cudaMemcpy(target, start, num_bytes, cudaMemcpyDeviceToDevice);
}

template <typename T>
T Sum(const T* start, const T* end, T* target_ptr, T* scratch_ptr, const bool copy_to_host) {
  const size_t num_el = (end - start);
  size_t tmp_storage_bytes;
  cudaMemset(target_ptr, 0, sizeof(T));
  cub::DeviceReduce::Sum(nullptr, tmp_storage_bytes, start, target_ptr, num_el);
  cub::DeviceReduce::Sum(scratch_ptr, tmp_storage_bytes, start, target_ptr, num_el);
  T result = T{0};
  if (copy_to_host) {
    cudaMemcpy(&result, target_ptr, sizeof(T), cudaMemcpyDeviceToHost);
  }
  return result;
}

template <typename T>
T ReadCuMem(const T* const data) {
  T result;
  cudaMemcpy(&result, data, sizeof(T), cudaMemcpyDeviceToHost);
  return result;
}

template <typename T>
__global__ void AlphaFromNumDenomKernel(const T* alpha_numerator, const T* alpha_denominator,
                                        T* alpha, T* neg_alpha) {
  *alpha = *alpha_numerator / *alpha_denominator;
  *neg_alpha = -*alpha;
}

template <typename T>
void AlphaFromNumDenom(const T* alpha_numerator, const T* alpha_denominator, T* alpha,
                       T* neg_alpha) {
  AlphaFromNumDenomKernel<T><<<1, 1>>>(alpha_numerator, alpha_denominator, alpha, neg_alpha);
}

template <typename T>
__global__ void BetaFromNumDenomKernel(const T* beta_num, const T* beta_denum, T* beta) {
  *beta = *beta_num / *beta_denum;
}

template <typename T>
void BetaFromNumDenom(const T* beta_num, const T* beta_denum, T* beta) {
  BetaFromNumDenomKernel<T><<<1, 1>>>(beta_num, beta_denum, beta);
}

// Explicit instantiations
template void Zero<float>(float* start, float* end);
template void Zero<double>(double* start, double* end);

template void Copy<float>(const float* start, const float* end, float* target);
template void Copy<double>(const double* start, const double* end, double* target);

template void AlphaFromNumDenom<float>(const float* alpha_numerator, const float* alpha_denominator,
                                       float* alpha, float* neg_alpha);
template void AlphaFromNumDenom<double>(const double* alpha_numerator,
                                        const double* alpha_denominator, double* alpha,
                                        double* neg_alpha);

template void BetaFromNumDenom<float>(const float* beta_num, const float* beta_denum, float* beta);
template void BetaFromNumDenom<double>(const double* beta_num, const double* beta_denum,
                                       double* beta);

template float Sum<float>(const float* start, const float* end, float* target_ptr,
                          float* scratch_ptr, bool copy_to_host);
template double Sum<double>(const double* start, const double* end, double* target_ptr,
                            double* scratch_ptr, bool copy_to_host);

template float ReadCuMem<float>(const float* const data);
template double ReadCuMem<double>(const double* const data);

}  // namespace caspar
