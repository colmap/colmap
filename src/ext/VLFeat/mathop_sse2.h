/** @file mathop_sse2.h
 ** @brief mathop for sse2
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/* ---------------------------------------------------------------- */
#ifndef VL_MATHOP_SSE2_H_INSTANTIATING

#ifndef VL_MATHOP_SSE2_H
#define VL_MATHOP_SSE2_H

#undef FLT
#define FLT VL_TYPE_DOUBLE
#define VL_MATHOP_SSE2_H_INSTANTIATING
#include "mathop_sse2.h"

#undef FLT
#define FLT VL_TYPE_FLOAT
#define VL_MATHOP_SSE2_H_INSTANTIATING
#include "mathop_sse2.h"

/* VL_MATHOP_SSE2_H */
#endif

/* ---------------------------------------------------------------- */
/* VL_MATHOP_SSE2_H_INSTANTIATING */
#else

#ifndef VL_DISABLE_SSE2

#include "generic.h"
#include "float.th"

VL_EXPORT T
VL_XCAT(_vl_dot_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_distance_l2_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_distance_l1_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_distance_chi2_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_kernel_l2_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_kernel_l1_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_kernel_chi2_sse2_, SFX)
(vl_size dimension, T const * X, T const * Y) ;

VL_EXPORT T
VL_XCAT(_vl_distance_mahalanobis_sq_sse2_, SFX)
(vl_size dimension, T const * X, T const * MU, T const * S);

VL_EXPORT void
VL_XCAT(_vl_weighted_sigma_sse2_, SFX)
(vl_size dimension, T * S, T const * X, T const * Y, T const W);

VL_EXPORT void
VL_XCAT(_vl_weighted_mean_sse2_, SFX)
(vl_size dimension, T * MU, T const * X, T const W);

/* ! VL_DISABLE_SSE2 */
#endif
#undef VL_MATHOP_SSE2_INSTANTIATING
#endif
