/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

namespace caspar {

template <typename T>
void Zero(T* start, T* end);

template <typename T>
void Copy(const T* start, const T* end, T* target);

template <typename T>
void AlphaFromNumDenom(const T* alpha_numerator, const T* alpha_denominator, T* alpha,
                       T* neg_alpha);

template <typename T>
void BetaFromNumDenom(const T* beta_num, const T* beta_denum, T* beta);

template <typename T>
T Sum(const T* start, const T* end, T* target_ptr, T* scratch_ptr, bool copy_to_host = true);

template <typename T>
T ReadCuMem(const T* const data);

// Explicit instantiations
extern template void Zero<float>(float* start, float* end);
extern template void Zero<double>(double* start, double* end);

extern template void Copy<float>(const float* start, const float* end, float* target);
extern template void Copy<double>(const double* start, const double* end, double* target);

extern template void AlphaFromNumDenom<float>(const float* alpha_numerator,
                                              const float* alpha_denominator, float* alpha,
                                              float* neg_alpha);
extern template void AlphaFromNumDenom<double>(const double* alpha_numerator,
                                               const double* alpha_denominator, double* alpha,
                                               double* neg_alpha);

extern template void BetaFromNumDenom<float>(const float* beta_num, const float* beta_denum,
                                             float* beta);
extern template void BetaFromNumDenom<double>(const double* beta_num, const double* beta_denum,
                                              double* beta);

extern template float Sum<float>(const float* start, const float* end, float* target_ptr,
                                 float* scratch_ptr, bool copy_to_host);
extern template double Sum<double>(const double* start, const double* end, double* target_ptr,
                                   double* scratch_ptr, bool copy_to_host);

extern template float ReadCuMem<float>(const float* const data);
extern template double ReadCuMem<double>(const double* const data);

}  // namespace caspar
