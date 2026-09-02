/** @file imopv_neon.h
 ** @brief Vectorized image operations - ARM NEON
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_IMOPV_NEON_H
#define VL_IMOPV_NEON_H

#include "generic.h"

#ifndef VL_DISABLE_NEON

VL_EXPORT
void _vl_imconvcol_vf_neon (float* dst, vl_size dst_stride,
                            float const* src,
                            vl_size src_width, vl_size src_height, vl_size src_stride,
                            float const* filt, vl_index filt_begin, vl_index filt_end,
                            int step, unsigned int flags) ;

#endif

/* VL_IMOPV_NEON_H */
#endif
