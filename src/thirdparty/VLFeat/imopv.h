/** @file imopv.h
 ** @brief Vectorized image operations
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_IMOPV_H
#define VL_IMOPV_H

#include "generic.h"

/** @name Image convolution flags
 ** @{ */
#define VL_PAD_BY_ZERO       (0x0 << 0) /**< @brief Pad with zeroes. */
#define VL_PAD_BY_CONTINUITY (0x1 << 0) /**< @brief Pad by continuity. */
#define VL_PAD_MASK          (0x3)      /**< @brief Padding field selector. */
#define VL_TRANSPOSE         (0x1 << 2) /**< @brief Transpose result. */
/** @} */

/** @name Image convolution
 ** @{ */
VL_EXPORT
void vl_imconvcol_vf (float* dst, vl_size dst_stride,
                      float const* src,
                      vl_size src_width, vl_size src_height, vl_size src_stride,
                      float const* filt, vl_index filt_begin, vl_index filt_end,
                      int step, unsigned int flags) ;

VL_EXPORT
void vl_imconvcol_vd (double* dst, vl_size dst_stride,
                      double const* src,
                      vl_size src_width, vl_size src_height, vl_size src_stride,
                      double const* filt, vl_index filt_begin, vl_index filt_end,
                      int step, unsigned int flags) ;

VL_EXPORT
void vl_imconvcoltri_f (float * dest, vl_size destStride,
                        float const * image,
                        vl_size imageWidth, vl_size imageHeight, vl_size imageStride,
                        vl_size filterSize,
                        vl_size step, int unsigned flags) ;

VL_EXPORT
void vl_imconvcoltri_d (double * dest, vl_size destStride,
                        double const * image,
                        vl_size imageWidth, vl_size imageHeight, vl_size imageStride,
                        vl_size filterSize,
                        vl_size step, int unsigned flags) ;
/** @} */

/** @name Integral image
 ** @{ */
VL_EXPORT
void vl_imintegral_f (float * integral,  vl_size integralStride,
                      float const * image,
                      vl_size imageWidth, vl_size imageHeight, vl_size imageStride) ;

VL_EXPORT
void vl_imintegral_d (double * integral,  vl_size integralStride,
                      double const * image,
                      vl_size imageWidth, vl_size imageHeight, vl_size imageStride) ;

VL_EXPORT
void vl_imintegral_i32 (vl_int32 * integral,  vl_size integralStride,
                        vl_int32 const * image,
                        vl_size imageWidth, vl_size imageHeight, vl_size imageStride) ;

VL_EXPORT
void vl_imintegral_ui32 (vl_uint32 * integral,  vl_size integralStride,
                         vl_uint32 const * image,
                         vl_size imageWidth, vl_size imageHeight, vl_size imageStride) ;
/** @} */

/** @name Distance transform */
/** @{ */

VL_EXPORT void
vl_image_distance_transform_d (double const * image,
                               vl_size numColumns,
                               vl_size numRows,
                               vl_size columnStride,
                               vl_size rowStride,
                               double * distanceTransform,
                               vl_uindex * indexes,
                               double coeff,
                               double offset) ;

VL_EXPORT void
vl_image_distance_transform_f (float const * image,
                               vl_size numColumns,
                               vl_size numRows,
                               vl_size columnStride,
                               vl_size rowStride,
                               float * distanceTransform,
                               vl_uindex * indexes,
                               float coeff,
                               float offset) ;

/** @} */

/* ---------------------------------------------------------------- */
/** @name Image smoothing */
/** @{ */

VL_EXPORT void
vl_imsmooth_f (float *smoothed, vl_size smoothedStride,
               float const *image, vl_size width, vl_size height, vl_size stride,
               double sigmax, double sigmay) ;

VL_EXPORT void
vl_imsmooth_d (double *smoothed, vl_size smoothedStride,
               double const *image, vl_size width, vl_size height, vl_size stride,
               double sigmax, double sigmay) ;

/** @} */

/* ---------------------------------------------------------------- */
/** @name Image gradients */
/** @{ */
VL_EXPORT void
vl_imgradient_polar_f (float* amplitudeGradient, float* angleGradient,
                       vl_size gradWidthStride, vl_size gradHeightStride,
                       float const* image,
                       vl_size imageWidth, vl_size imageHeight,
                       vl_size imageStride);

VL_EXPORT void
vl_imgradient_polar_d (double* amplitudeGradient, double* angleGradient,
                       vl_size gradWidthStride, vl_size gradHeightStride,
                       double const* image,
                       vl_size imageWidth, vl_size imageHeight,
                       vl_size imageStride);

VL_EXPORT void
vl_imgradient_f (float* xGradient, float* yGradient,
                 vl_size gradWidthStride, vl_size gradHeightStride,
                 float const *image,
                 vl_size imageWidth, vl_size imageHeight, vl_size imageStride);

VL_EXPORT void
vl_imgradient_d(double* xGradient, double* yGradient,
                vl_size gradWidthStride, vl_size gradHeightStride,
                double const *image,
                vl_size imageWidth, vl_size imageHeight, vl_size imageStride);

VL_EXPORT void
vl_imgradient_polar_f_callback(float const *sourceImage,
                               int sourceImageWidth, int sourceImageHeight,
                               float *dstImage,
                               int dstWidth, int dstHeight,
                               int octave, int level,
                               void *params);

/** @} */

/* VL_IMOPV_H */
#endif
