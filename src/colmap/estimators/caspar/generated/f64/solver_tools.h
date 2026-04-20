/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

template <typename T>
void zero(T* start, T* end);

template <typename T>
void copy(const T* start, const T* end, T* target);

template <typename T>
void alpha_from_num_denum(const T* alpha_numerator, const T* alpha_denumerator, T* alpha,
                          T* neg_alpha);

template <typename T>
void beta_from_num_denum(const T* beta_num, const T* beta_denum, T* beta);

template <typename T>
T sum(const T* start, const T* end, T* target_ptr, T* scratch_ptr, bool copy_to_host = true);

template <typename T>
T read_cumem(const T* const data);

// Explicit instantiations
extern template void zero<float>(float* start, float* end);
extern template void zero<double>(double* start, double* end);

extern template void copy<float>(const float* start, const float* end, float* target);
extern template void copy<double>(const double* start, const double* end, double* target);

extern template void alpha_from_num_denum<float>(const float* alpha_numerator,
                                                 const float* alpha_denumerator, float* alpha,
                                                 float* neg_alpha);
extern template void alpha_from_num_denum<double>(const double* alpha_numerator,
                                                  const double* alpha_denumerator, double* alpha,
                                                  double* neg_alpha);

extern template void beta_from_num_denum<float>(const float* beta_num, const float* beta_denum,
                                                float* beta);
extern template void beta_from_num_denum<double>(const double* beta_num, const double* beta_denum,
                                                 double* beta);

extern template float sum<float>(const float* start, const float* end, float* target_ptr,
                                 float* scratch_ptr, bool copy_to_host);
extern template double sum<double>(const double* start, const double* end, double* target_ptr,
                                   double* scratch_ptr, bool copy_to_host);

extern template float read_cumem<float>(const float* const data);
extern template double read_cumem<double>(const double* const data);

}  // namespace caspar
