/** @file imopv_sse2.h
 ** @brief Vectorized image operations - SSE2
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_IMOPV_SSE2_H
#define VL_IMOPV_SSE2_H

#include "generic.h"

#ifndef VL_DISABLE_SSE2

VL_EXPORT
void _vl_imconvcol_vf_sse2 (float* dst, vl_size dst_stride,
                            float const* src,
                            vl_size src_width, vl_size src_height, vl_size src_stride,
                            float const* filt, vl_index filt_begin, vl_index filt_end,
                            int step, unsigned int flags) ;

VL_EXPORT
void _vl_imconvcol_vd_sse2 (double* dst, vl_size dst_stride,
                            double const* src,
                            vl_size src_width, vl_size src_height, vl_size src_stride,
                            double const* filt, vl_index filt_begin, vl_index filt_end,
                            int step, unsigned int flags) ;

/*
VL_EXPORT
void _vl_imconvcoltri_vf_sse2 (float* dst, int dst_stride,
                               float const* src,
                               int src_width, int src_height, int src_stride,
                               int filt_size,
                               int step, unsigned int flags) ;

VL_EXPORT
void _vl_imconvcoltri_vd_sse2 (double* dst, int dst_stride,
                               double const* src,
                               int src_width, int src_height, int src_stride,
                               int filt_size,
                               int step, unsigned int flags) ;
*/

#endif

/* VL_IMOPV_SSE2_H */
#endif
