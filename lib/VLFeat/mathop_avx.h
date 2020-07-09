/** @file mathop_avx.h
 ** @brief mathop for avx
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/* ---------------------------------------------------------------- */
#ifndef VL_MATHOP_AVX_H_INSTANTIATING

#ifndef VL_MATHOP_AVX_H
#define VL_MATHOP_AVX_H

#undef FLT
#define FLT VL_TYPE_DOUBLE
#define VL_MATHOP_AVX_H_INSTANTIATING
#include "mathop_avx.h"

#undef FLT
#define FLT VL_TYPE_FLOAT
#define VL_MATHOP_AVX_H_INSTANTIATING
#include "mathop_avx.h"

/* VL_MATHOP_AVX_H */
#endif

/* ---------------------------------------------------------------- */
/* VL_MATHOP_AVX_H_INSTANTIATING */
#else

#ifndef VL_DISABLE_AVX
#include "generic.h"
#include "float.h"

VL_EXPORT T
VL_XCAT(_vl_distance_mahalanobis_sq_avx_, SFX)
(vl_size dimension, T const * X, T const * MU, T const * S);

VL_EXPORT T
VL_XCAT(_vl_distance_l2_avx_, SFX)
(vl_size dimension, T const * X, T const * Y);

VL_EXPORT void
VL_XCAT(_vl_weighted_sigma_avx_, SFX)
(vl_size dimension, T * S, T const * X, T const * Y, T const W);

VL_EXPORT void
VL_XCAT(_vl_weighted_mean_avx_, SFX)
(vl_size dimension, T * MU, T const * X, T const W);

/* ! VL_DISABLE_AVX */
#endif

#undef VL_MATHOP_AVX_H_INSTANTIATING
#endif
