/** @file imopv_neon.c
 ** @brief Vectorized image operations - ARM NEON
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "imopv_neon.h"
#include "imopv.h"

#if !defined(VL_DISABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))

#include <arm_neon.h>

#if defined(__aarch64__)
#define VL_NEON_FMA_N(acc, vec, c) vfmaq_n_f32((acc), (vec), (c))
#else
#define VL_NEON_FMA_N(acc, vec, c) vmlaq_n_f32((acc), (vec), (c))
#endif

VL_INLINE float32x4_t
vl_imconvcol_accum_vec_f32 (float const* src,
                            vl_size src_height,
                            vl_size src_stride,
                            float const* filt,
                            vl_index filt_begin,
                            vl_index filt_end,
                            vl_index y,
                            vl_size x,
                            vl_bool zeropad)
{
  float32x4_t acc = vdupq_n_f32(0.0f) ;
  vl_index valid_k_hi = VL_MIN(filt_end, y) ;
  vl_index valid_k_lo = VL_MAX(filt_begin, y - (vl_index)src_height + 1) ;
  vl_index k ;

  if (!zeropad) {
    float32x4_t top = vld1q_f32(src + x) ;
    for (k = filt_end ; k > valid_k_hi ; --k) {
      acc = VL_NEON_FMA_N(acc, top, filt[(vl_size)(k - filt_begin)]) ;
    }
  }

  if (valid_k_hi >= valid_k_lo) {
    for (k = valid_k_hi ; k >= valid_k_lo ; --k) {
      vl_index sy = y - k ;
      float32x4_t s = vld1q_f32(src + (vl_size)sy * src_stride + x) ;
      acc = VL_NEON_FMA_N(acc, s, filt[(vl_size)(k - filt_begin)]) ;
    }
  }

  if (!zeropad) {
    float32x4_t bottom = vld1q_f32(src + (src_height - 1) * src_stride + x) ;
    for (k = valid_k_lo - 1 ; k >= filt_begin ; --k) {
      acc = VL_NEON_FMA_N(acc, bottom, filt[(vl_size)(k - filt_begin)]) ;
    }
  }

  return acc ;
}

VL_INLINE float
vl_imconvcol_accum_scalar_f32 (float const* src,
                               vl_size src_height,
                               vl_size src_stride,
                               float const* filt,
                               vl_index filt_begin,
                               vl_index filt_end,
                               vl_index y,
                               vl_size x,
                               vl_bool zeropad)
{
  float acc = 0.0f ;
  vl_index valid_k_hi = VL_MIN(filt_end, y) ;
  vl_index valid_k_lo = VL_MAX(filt_begin, y - (vl_index)src_height + 1) ;
  vl_index k ;

  if (!zeropad) {
    float top = src[x] ;
    for (k = filt_end ; k > valid_k_hi ; --k) {
      acc += top * filt[(vl_size)(k - filt_begin)] ;
    }
  }

  if (valid_k_hi >= valid_k_lo) {
    for (k = valid_k_hi ; k >= valid_k_lo ; --k) {
      vl_index sy = y - k ;
      acc += src[(vl_size)sy * src_stride + x] * filt[(vl_size)(k - filt_begin)] ;
    }
  }

  if (!zeropad) {
    float bottom = src[(src_height - 1) * src_stride + x] ;
    for (k = valid_k_lo - 1 ; k >= filt_begin ; --k) {
      acc += bottom * filt[(vl_size)(k - filt_begin)] ;
    }
  }

  return acc ;
}

VL_EXPORT void
_vl_imconvcol_vf_neon (float* dst, vl_size dst_stride,
                       float const* src,
                       vl_size src_width, vl_size src_height, vl_size src_stride,
                       float const* filt, vl_index filt_begin, vl_index filt_end,
                       int step, unsigned int flags)
{
  vl_size yi ;
  vl_size dheight = (src_height - 1) / (vl_size)step + 1 ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;

  /* Hot path in SIFT: step == 1 and transpose output. */
  if (step == 1 && transp) {
    for (yi = 0 ; yi + 3 < dheight ; yi += 4) {
      vl_index y0 = (vl_index)yi ;
      vl_index y1 = (vl_index)(yi + 1) ;
      vl_index y2 = (vl_index)(yi + 2) ;
      vl_index y3 = (vl_index)(yi + 3) ;
      vl_size x = 0 ;

      for (; x + 4 <= src_width ; x += 4) {
        float32x4_t a0 = vl_imconvcol_accum_vec_f32(src,
                                                    src_height,
                                                    src_stride,
                                                    filt,
                                                    filt_begin,
                                                    filt_end,
                                                    y0,
                                                    x,
                                                    zeropad) ;
        float32x4_t a1 = vl_imconvcol_accum_vec_f32(src,
                                                    src_height,
                                                    src_stride,
                                                    filt,
                                                    filt_begin,
                                                    filt_end,
                                                    y1,
                                                    x,
                                                    zeropad) ;
        float32x4_t a2 = vl_imconvcol_accum_vec_f32(src,
                                                    src_height,
                                                    src_stride,
                                                    filt,
                                                    filt_begin,
                                                    filt_end,
                                                    y2,
                                                    x,
                                                    zeropad) ;
        float32x4_t a3 = vl_imconvcol_accum_vec_f32(src,
                                                    src_height,
                                                    src_stride,
                                                    filt,
                                                    filt_begin,
                                                    filt_end,
                                                    y3,
                                                    x,
                                                    zeropad) ;

        {
          float32x4x2_t t0 = vtrnq_f32(a0, a1) ;
          float32x4x2_t t1 = vtrnq_f32(a2, a3) ;
          float32x4_t c0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])) ;
          float32x4_t c1 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])) ;
          float32x4_t c2 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])) ;
          float32x4_t c3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])) ;
          vst1q_f32(dst + (x + 0) * dst_stride + yi, c0) ;
          vst1q_f32(dst + (x + 1) * dst_stride + yi, c1) ;
          vst1q_f32(dst + (x + 2) * dst_stride + yi, c2) ;
          vst1q_f32(dst + (x + 3) * dst_stride + yi, c3) ;
        }
      }

      for (; x < src_width ; ++x) {
        float* out = dst + x * dst_stride + yi ;
        out[0] = vl_imconvcol_accum_scalar_f32(src,
                                               src_height,
                                               src_stride,
                                               filt,
                                               filt_begin,
                                               filt_end,
                                               y0,
                                               x,
                                               zeropad) ;
        out[1] = vl_imconvcol_accum_scalar_f32(src,
                                               src_height,
                                               src_stride,
                                               filt,
                                               filt_begin,
                                               filt_end,
                                               y1,
                                               x,
                                               zeropad) ;
        out[2] = vl_imconvcol_accum_scalar_f32(src,
                                               src_height,
                                               src_stride,
                                               filt,
                                               filt_begin,
                                               filt_end,
                                               y2,
                                               x,
                                               zeropad) ;
        out[3] = vl_imconvcol_accum_scalar_f32(src,
                                               src_height,
                                               src_stride,
                                               filt,
                                               filt_begin,
                                               filt_end,
                                               y3,
                                               x,
                                               zeropad) ;
      }
    }

    /* Tail rows (< 4): generic transpose writeback. */
    for (; yi < dheight ; ++yi) {
      vl_index y = (vl_index)yi ;
      vl_size x = 0 ;
      for (; x + 4 <= src_width ; x += 4) {
        float32x4_t acc = vl_imconvcol_accum_vec_f32(src,
                                                     src_height,
                                                     src_stride,
                                                     filt,
                                                     filt_begin,
                                                     filt_end,
                                                     y,
                                                     x,
                                                     zeropad) ;
        dst[(x + 0) * dst_stride + yi] = vgetq_lane_f32(acc, 0) ;
        dst[(x + 1) * dst_stride + yi] = vgetq_lane_f32(acc, 1) ;
        dst[(x + 2) * dst_stride + yi] = vgetq_lane_f32(acc, 2) ;
        dst[(x + 3) * dst_stride + yi] = vgetq_lane_f32(acc, 3) ;
      }
      for (; x < src_width ; ++x) {
        dst[x * dst_stride + yi] = vl_imconvcol_accum_scalar_f32(src,
                                                                 src_height,
                                                                 src_stride,
                                                                 filt,
                                                                 filt_begin,
                                                                 filt_end,
                                                                 y,
                                                                 x,
                                                                 zeropad) ;
      }
    }
    return ;
  }

  for (yi = 0 ; yi < dheight ; ++ yi) {
    vl_index y = (vl_index)(yi * (vl_size)step) ;
    vl_size x = 0 ;

    for (; x + 4 <= src_width ; x += 4) {
      float32x4_t acc = vl_imconvcol_accum_vec_f32(src,
                                                   src_height,
                                                   src_stride,
                                                   filt,
                                                   filt_begin,
                                                   filt_end,
                                                   y,
                                                   x,
                                                   zeropad) ;

      if (transp) {
        dst[(x + 0) * dst_stride + yi] = vgetq_lane_f32(acc, 0) ;
        dst[(x + 1) * dst_stride + yi] = vgetq_lane_f32(acc, 1) ;
        dst[(x + 2) * dst_stride + yi] = vgetq_lane_f32(acc, 2) ;
        dst[(x + 3) * dst_stride + yi] = vgetq_lane_f32(acc, 3) ;
      } else {
        vst1q_f32(dst + yi * dst_stride + x, acc) ;
      }
    }

    for (; x < src_width ; ++ x) {
      float acc = vl_imconvcol_accum_scalar_f32(src,
                                                src_height,
                                                src_stride,
                                                filt,
                                                filt_begin,
                                                filt_end,
                                                y,
                                                x,
                                                zeropad) ;
      if (transp) {
        dst[x * dst_stride + yi] = acc ;
      } else {
        dst[yi * dst_stride + x] = acc ;
      }
    }
  }
}

#undef VL_NEON_FMA_N

#endif
