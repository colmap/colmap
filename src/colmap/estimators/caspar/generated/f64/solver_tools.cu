/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "solver_tools.h"

namespace caspar {

template <typename T>
void zero(T* start, T* end) {
  const size_t num_bytes = (end - start) * sizeof(T);
  cudaMemset(start, 0, num_bytes);
}

template <typename T>
void copy(const T* start, const T* end, T* target) {
  const size_t num_bytes = (end - start) * sizeof(T);
  cudaMemcpy(target, start, num_bytes, cudaMemcpyDeviceToDevice);
}

template <typename T>
T sum(const T* start, const T* end, T* target_ptr, T* scratch_ptr, const bool copy_to_host) {
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
T read_cumem(const T* const data) {
  T result;
  cudaMemcpy(&result, data, sizeof(T), cudaMemcpyDeviceToHost);
  return result;
}

template <typename T>
__global__ void alpha_from_num_denum_kernel(const T* alpha_numerator, const T* alpha_denumerator,
                                            T* alpha, T* neg_alpha) {
  *alpha = *alpha_numerator / *alpha_denumerator;
  *neg_alpha = -*alpha;
}

template <typename T>
void alpha_from_num_denum(const T* alpha_numerator, const T* alpha_denumerator, T* alpha,
                          T* neg_alpha) {
  alpha_from_num_denum_kernel<T><<<1, 1>>>(alpha_numerator, alpha_denumerator, alpha, neg_alpha);
}

template <typename T>
__global__ void beta_from_num_denum_kernel(const T* beta_num, const T* beta_denum, T* beta) {
  *beta = *beta_num / *beta_denum;
}

template <typename T>
void beta_from_num_denum(const T* beta_num, const T* beta_denum, T* beta) {
  beta_from_num_denum_kernel<T><<<1, 1>>>(beta_num, beta_denum, beta);
}

// Explicit instantiations
template void zero<float>(float* start, float* end);
template void zero<double>(double* start, double* end);

template void copy<float>(const float* start, const float* end, float* target);
template void copy<double>(const double* start, const double* end, double* target);

template void alpha_from_num_denum<float>(const float* alpha_numerator,
                                          const float* alpha_denumerator, float* alpha,
                                          float* neg_alpha);
template void alpha_from_num_denum<double>(const double* alpha_numerator,
                                           const double* alpha_denumerator, double* alpha,
                                           double* neg_alpha);

template void beta_from_num_denum<float>(const float* beta_num, const float* beta_denum,
                                         float* beta);
template void beta_from_num_denum<double>(const double* beta_num, const double* beta_denum,
                                          double* beta);

template float sum<float>(const float* start, const float* end, float* target_ptr,
                          float* scratch_ptr, bool copy_to_host);
template double sum<double>(const double* start, const double* end, double* target_ptr,
                            double* scratch_ptr, bool copy_to_host);

template float read_cumem<float>(const float* const data);
template double read_cumem<double>(const double* const data);

}  // namespace caspar
