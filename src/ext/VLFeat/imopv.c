/** @file imopv.c
 ** @brief Vectorized image operations - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file imopv.h
 **
 ** This module provides the following image operations:
 **
 ** - <b>Separable convolution.</b> The function ::vl_imconvcol_vf()
 **   can be used to compute separable convolutions.
 **
 ** - <b>Convolution by a triangular kernel.</b> The function
 **   vl_imconvcoltri_vf() is an optimized convolution routine for
 **   triangular kernels.
 **
 ** - <b>Distance transform.</b> ::vl_image_distance_transform_f() is
 **   a linear algorithm to compute the distance transform of an
 **   image.
 **
 ** @remark  Some operations are optimized to exploit possible SIMD
 ** instructions. This requires image data to be properly aligned (typically
 ** to 16 bytes). Similalry, the image stride (the number of bytes to skip to move
 ** to the next image row), must be aligned.
  **/

#ifndef VL_IMOPV_INSTANTIATING

#include "imopv.h"
#include "imopv_sse2.h"
#include "mathop.h"

#define FLT VL_TYPE_FLOAT
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_DOUBLE
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_UINT32
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

#define FLT VL_TYPE_INT32
#define VL_IMOPV_INSTANTIATING
#include "imopv.c"

/* VL_IMOPV_INSTANTIATING */
#endif

#if defined(VL_IMOPV_INSTANTIATING) || defined(__DOXYGEN__)

#include "float.h"

/* ---------------------------------------------------------------- */
/*                                                Image Convolution */
/* ---------------------------------------------------------------- */

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

/** @fn vl_imconvcol_vd(double*,vl_size,double const*,vl_size,vl_size,vl_size,double const*,vl_index,vl_index,int,unsigned int)
 ** @brief Convolve image along columns
 **
 ** @param dst destination image.
 ** @param dst_stride width of the destination image including padding.
 ** @param src source image.
 ** @param src_width width of the source image.
 ** @param src_height height of the source image.
 ** @param src_stride width of the source image including padding.
 ** @param filt filter kernel.
 ** @param filt_begin coordinate of the first filter element.
 ** @param filt_end coordinate of the last filter element.
 ** @param step sub-sampling step.
 ** @param flags operation modes.
 **
 ** The function convolves the column of the image @a src by the
 ** filter @a filt and saves the result to the image @a dst. The size
 ** of @a dst must be equal to the size of @a src.  Formally, this
 ** results in the calculation
 **
 ** @f[
 ** \mathrm{dst} [x,y] = \sum_{p=y-\mathrm{filt\_end}}^{y-\mathrm{filt\_begin}}
 ** \mathrm{src}[x,y] \mathrm{filt}[y - p - \mathrm{filt\_begin}]
 ** @f]
 **
 ** The function subsamples the image along the columns according to
 ** the parameter @a step. Setting @a step to 1 (one) computes the
 ** elements @f$\mathrm{dst}[x,y]@f$ for all pairs (x,0), (x,1), (x,2)
 ** and so on. Setting @a step two 2 (two) computes only (x,0), (x,2)
 ** and so on (in this case the height of the destination image is
 ** <code>floor(src_height/step)+1)</code>.
 **
 ** Calling twice the function can be used to compute 2-D separable
 ** convolutions.  Use the flag ::VL_TRANSPOSE to transpose the result
 ** (in this case @a dst has transposed dimension as well).
 **
 ** The function allows the support of the filter to be any range.
 ** Usually the support is <code>@a filt_end = -@a filt_begin</code>.
 **
 ** The convolution operation may pick up values outside the image
 ** boundary. To cope with this edge cases, the function either pads
 ** the image by zero (::VL_PAD_BY_ZERO) or with the values at the
 ** boundary (::VL_PAD_BY_CONTINUITY).
 **/

/** @fn vl_imconvcol_vf(float*,vl_size,float const*,vl_size,vl_size,vl_size,float const*,vl_index,vl_index,int,unsigned int)
 ** @see ::vl_imconvcol_vd
 **/

VL_EXPORT void
VL_XCAT(vl_imconvcol_v, SFX)
(T* dst, vl_size dst_stride,
 T const* src,
 vl_size src_width, vl_size src_height, vl_size src_stride,
 T const* filt, vl_index filt_begin, vl_index filt_end,
 int step, unsigned int flags)
{
  vl_index x = 0 ;
  vl_index y ;
  vl_index dheight = (src_height - 1) / step + 1 ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;

  /* dispatch to accelerated version */
#ifndef VL_DISABLE_SSE2
  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    VL_XCAT3(_vl_imconvcol_v,SFX,_sse2)
    (dst,dst_stride,
     src,src_width,src_height,src_stride,
     filt,filt_begin,filt_end,
     step,flags) ;
    return ;
  }
#endif

  /* let filt point to the last sample of the filter */
  filt += filt_end - filt_begin ;

  while (x < (signed)src_width) {
    /* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
     * where supp(filt) = [filt_begin, filt_end] = [fb,fe].
     *
     * CHUNK_A: y - fe <= p < 0
     *          completes VL_MAX(fe - y, 0) samples
     * CHUNK_B: VL_MAX(y - fe, 0) <= p < VL_MIN(y - fb, height - 1)
     *          completes fe - VL_MAX(fb, height - y) + 1 samples
     * CHUNK_C: completes all samples
     */
    T const *filti ;
    vl_index stop ;

    for (y = 0 ; y < (signed)src_height ; y += step) {
      T acc = 0 ;
      T v = 0, c ;
      T const* srci ;

      filti = filt ;
      stop = filt_end - y ;
      srci = src + x - stop * src_stride ;

      if (stop > 0) {
        if (zeropad) {
          v = 0 ;
        } else {
          v = *(src + x) ;
        }
        while (filti > filt - stop) {
          c = *filti-- ;
          acc += v * c ;
          srci += src_stride ;
        }
      }

      stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
      while (filti > filt - stop) {
        v = *srci ;
        c = *filti-- ;
        acc += v * c ;
        srci += src_stride ;
      }

      if (zeropad) v = 0 ;

      stop = filt_end - filt_begin + 1 ;
      while (filti > filt - stop) {
        c = *filti-- ;
        acc += v * c ;
      }

      if (transp) {
        *dst = acc ; dst += 1 ;
      } else {
        *dst = acc ; dst += dst_stride ;
      }
    } /* next y */
    if (transp) {
      dst += 1 * dst_stride - dheight * 1 ;
    } else {
      dst += 1 * 1 - dheight * dst_stride ;
    }
    x += 1 ;
  } /* next x */
}

/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif

/* ---------------------------------------------------------------- */
/*                                         Image distance transform */
/* ---------------------------------------------------------------- */

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

/** @fn ::vl_image_distance_transform_d(double const*,vl_size,vl_size,vl_size,vl_size,double*,vl_uindex*,double,double)
 ** @brief Compute the distance transform of an image
 ** @param image image.
 ** @param numColumns number of columns of the image.
 ** @param numRows number of rows of the image.
 ** @param columnStride offset from one column to the next.
 ** @param rowStride offset from one row to the next.
 ** @param distanceTransform distance transform (out).
 ** @param indexes nearest neighbor indexes (in/out).
 ** @param coeff quadratic cost coefficient (non-negative).
 ** @param offset quadratic cost offset.
 **
 ** The function computes the distance transform along the first
 ** dimension of the image @a image. Let @f$ I(u,v) @f$ be @a image.
 ** Its distance transfrom @f$ D(u,v) @f$ is given by:
 **
 ** @f[
 **   u^*(u,v) = \min_{u'} I(u',v) + \mathtt{coeff} (u' - u - \mathtt{offset})^2,
 **   \quad D(u,v) = I(u^*(u,v),v).
 ** @f]
 **
 ** Notice that @a coeff must be non negative.
 **
 ** The function fills in the buffer @a distanceTransform with @f$ D
 ** @f$.  This buffer must have the same size as @a image.
 **
 ** If @a indexes is not @c NULL, it must be a matrix of the same size
 ** o the image. The function interprets the value of this matrix as
 ** indexes of the pixels, i.e @f$ \mathtt{indexes}(u,v) @f$ is the
 ** index of pixel @f$ (u,v) @f$. On output, the matrix @a indexes
 ** contains @f$ \mathtt{indexes}(u^*(u,v),v) @f$. This information
 ** can be used to determine for each pixel @f$ (u,v) @f$ its
 ** &ldquo;nearest neighbor&rdquo.
 **
 ** Notice that by swapping @a numRows and @a numColumns and @a
 ** columnStride and @a rowStride, the function can be made to operate
 ** along the other image dimension. Specifically, to compute the
 ** distance transform along columns and rows, call the functinon
 ** twice:
 ***
 ** @code
 **   for (i = 0 ; i < numColumns * numRows ; ++i) indexes[i] = i ;
 **   vl_image_distance_transform_d(image,numColumns,numRows,1,numColumns,
 **                                 distanceTransform,indexes,u_coeff,u_offset) ;
 **   vl_image_distance_transform_d(distanceTransform,numRows,numColumns,numColumns,1,
 **                                 distanceTransform,indexes,u_coeff,u_offset) ;
 ** @endcode
 **
 ** @par Algorithm
 **
 ** The function implements the algorithm described in:
 ** P. F. Felzenszwalb and D. P. Huttenlocher, <em>Distance Transforms
 ** of Sampled Functions,</em> Technical Report, Cornell University,
 ** 2004.
 **
 ** Since the algorithm operates along one dimension per time,
 ** consider the 1D version of the problem for simplicity:
 **
 ** @f[
 **  d(y) = \min_{x} g(y;x), \quad g(y;x) = f(x) + \alpha (y - x - \beta)^2,
 **  \quad x,y \in \{0,1,\dots,N-1\}.
 ** @f]
 **
 ** Hence the distance transform @f$ d(y) @f$ is the lower envelope of
 ** the family of parabolas @f$ g(y;x) @f$ indexed by @f$ x
 ** @f$. Notice that all parabolas have the same curvature and that
 ** their centers are located at @f$ x + \beta, @f$ @f$ x=0,\dots,N-1
 ** @f$. The algorithm considers one parabola per time, from left to
 ** right, and finds the interval for which the parabola belongs to
 ** the lower envelope (if any).
 **
 ** Initially, only the leftmost parabola @f$ g(y;0) @f$ has been
 ** considered, and its validity interval is @f$(-\infty, \infty) @f$.
 ** Then the second parabola @f$ g(y;1) @f$ is considered. As long as
 ** @f$ \alpha > 0 @f$, the two parabolas @f$ g(y;0),\ g(y;1) @f$
 ** intersect at a unique point @f$ \bar y @f$. Then the first
 ** parabola belongs to the envelope in the interval @f$ (-\infty,
 ** \bar y] @f$ and the second one in the interval @f$ (\bar y,
 ** +\infty] @f$. When the third parabola @f$ g(y;2) @f$ is
 ** considered, the intersection point @f$ \hat y @f$ with the
 ** previously added parabola @f$ g(y;1) @f$ is found. Now two cases
 ** may arise:
 **
 ** - @f$ \hat y > \bar y @f$, in which case all three parabolas
 **   belong to the envelope in the intervals @f$ (-\infty,\bar y],
 **   (\bar y, \hat y], (\hat y, +\infty] @f$.
 **
 ** - @f$ \hat y \leq \bar y @f$, in which case the second parabola
 **   @f$ g(y;1) @f$ has no point beloning to the envelope, and it is
 **   removed.  One then remains with the two parabolas @f$ g(y;0),\
 **   g(y;2) @f$ and the algorithm is re-iterated.
 **
 ** The algorithm proceeds in this fashion. Every time a new parabola
 ** is considered, its intersection point with the previously added
 ** parabola on the left is computed, and that parabola is potentially
 ** removed.  The cost of an iteration is 1 plus the number of deleted
 ** parabolas. Since there are @f$ N @f$ iterations and at most @f$ N
 ** @f$ parabolas to delete overall, the complexity is linear,
 ** i.e. @f$ O(N) @f$.
 **/

/** @fn ::vl_image_distance_transform_f(float const*,vl_size,vl_size,vl_size,vl_size,float*,vl_uindex*,float,float)
 ** @see ::vl_image_distance_transform_d
 **/

VL_EXPORT void
VL_XCAT(vl_image_distance_transform_,SFX)
(T const * image,
 vl_size numColumns,
 vl_size numRows,
 vl_size columnStride,
 vl_size rowStride,
 T * distanceTransform,
 vl_uindex * indexes,
 T coeff,
 T offset)
{
  /* Each image pixel corresponds to a parabola. The algorithm scans
   such parabolas from left to right, keeping track of which
   parabolas belong to the lower envelope and in which interval. There are
   NUM active parabolas, FROM stores the beginning of the interval
   for which a certain parabola is part of the envoelope, and WHICH store
   the index of the parabola (that is, the pixel x from which the parabola
   originated).
   */
  vl_uindex x, y ;
  T * from = vl_malloc (sizeof(T) * (numColumns + 1)) ;
  T * base = vl_malloc (sizeof(T) * numColumns) ;
  vl_uindex * baseIndexes = vl_malloc (sizeof(vl_uindex) * numColumns) ;
  vl_uindex * which = vl_malloc (sizeof(vl_uindex) * numColumns) ;
  vl_uindex num = 0 ;

  for (y = 0 ; y < numRows ; ++y) {
    num = 0 ;
    for (x = 0 ; x < numColumns ; ++x) {
      T r = image[x  * columnStride + y * rowStride] ;
      T x2 = x * x ;
#if (FLT == VL_TYPE_FLOAT)
      T from_ = - VL_INFINITY_F ;
#else
      T from_ = - VL_INFINITY_D ;
#endif

      /*
       Add next parabola (there are NUM so far). The algorithm finds
       intersection INTERS with the previously added parabola. If
       the intersection is on the right of the "starting point" of
       this parabola, then the previous parabola is kept, and the
       new one is added to its right. Otherwise the new parabola
       "eats" the old one, which gets deleted and the check is
       repeated with the parabola added before the deleted one.
       */

      while (num >= 1) {
        vl_uindex x_ = which[num - 1] ;
        T x2_ = x_ * x_ ;
        T r_ = image[x_ * columnStride + y * rowStride] ;
        T inters ;
        if (r == r_) {
          /* handles the case r = r_ = \pm inf */
          inters = (x + x_) / 2.0 + offset ;
        }
#if (FLT == VL_TYPE_FLOAT)
        else if (coeff > VL_EPSILON_F)
#else
        else if (coeff > VL_EPSILON_D)
#endif
        {
          inters = ((r - r_) + coeff * (x2 - x2_)) / (x - x_) / (2*coeff) + offset ;
        } else {
          /* If coeff is very small, the parabolas are flat (= lines).
           In this case the previous parabola should be deleted if the current
           pixel has lower score */
#if (FLT == VL_TYPE_FLOAT)
          inters = (r < r_) ? - VL_INFINITY_F : VL_INFINITY_F ;
#else
          inters = (r < r_) ? - VL_INFINITY_D : VL_INFINITY_D ;
#endif
        }
        if (inters <= from [num - 1]) {
          /* delete a previous parabola */
          -- num ;
        } else {
          /* accept intersection */
          from_ = inters ;
          break ;
        }
      }

      /* add a new parabola */
      which[num] = x ;
      from[num] = from_ ;
      base[num] = r ;
      if (indexes) baseIndexes[num] = indexes[x  * columnStride + y * rowStride] ;
      num ++ ;
    } /* next column */

#if (FLT == VL_TYPE_FLOAT)
    from[num] = VL_INFINITY_F ;
#else
    from[num] = VL_INFINITY_D ;
#endif

    /* fill in */
    num = 0 ;
    for (x = 0 ; x < numColumns ; ++x) {
      double delta ;
      while (x >= from[num + 1]) ++ num ;
      delta = (double) x - (double) which[num] - offset ;
      distanceTransform[x  * columnStride + y * rowStride]
      = base[num] + coeff * delta * delta ;
      if (indexes) {
        indexes[x  * columnStride + y * rowStride]
        = baseIndexes[num] ;
      }
    }
  } /* next row */

  vl_free (from) ;
  vl_free (which) ;
  vl_free (base) ;
  vl_free (baseIndexes) ;
}

/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif

/* ---------------------------------------------------------------- */
/*                         Image convolution by a triangular kernel */
/* ---------------------------------------------------------------- */

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

/** @fn vl_imconvcoltri_d(double*,vl_size,double const*,vl_size,vl_size,vl_size,vl_size,vl_size,int unsigned)
 ** @brief Convolve an image along the columns with a triangular kernel
 ** @param dest destination image.
 ** @param destStride destination image stride.
 ** @param image image to convolve.
 ** @param imageWidth width of the image.
 ** @param imageHeight height of the image.
 ** @param imageStride width of the image including padding.
 ** @param filterSize size of the triangular filter.
 ** @param step sub-sampling step.
 ** @param flags operation modes.
 **
 ** The function convolves the columns of the image @a image with the
 ** triangular kernel
 **
 ** @f[
 **   k(t) = \frac{1}{\Delta^2} \max\{ \Delta -  |t|, 0 \},
 **   \quad t \in \mathbb{Z}
 ** @f]
 **
 ** The paramter @f$ \Delta @f$, equal to the function argument @a
 ** filterSize, controls the width of the kernel. Notice that the
 ** support of @f$ k(x) @f$ as a continuous function of @f$ x @f$ is
 ** the open interval @f$ (-\Delta,\Delta) @f$, which has length @f$
 ** 2\Delta @f$.  However, @f$ k(x) @f$ restricted to the ingeter
 ** domain @f$ x \in \mathcal{Z} @f$ has support @f$ \{ -\Delta + 1,
 ** \Delta +2, \dots, \Delta-1 \} @f$, which counts @f$ 2 \Delta - 1
 ** @f$ elements only. In particular, the discrete kernel is symmetric
 ** about the origin for all values of @f$ \Delta @f$.
 **
 ** The normalization factor @f$ 1 / \Delta^2 @f$ guaratnees that the
 ** filter is normalized to one, i.e.:
 **
 ** @f[
 **   \sum_{t=-\infty}^{+\infty} k(t) = 1
 ** @f]
 **
 ** @par Algorithm
 **
 ** The function exploits the fact that convolution by a triangular
 ** kernel can be expressed as the repeated convolution by a
 ** rectangular kernel, and that the latter can be performed in time
 ** indepenedent on the fiter width by using an integral-image type
 ** trick. Overall, the algorithm complexity is independent on the
 ** parameter @a filterSize and linear in the nubmer of image pixels.
 **
 ** @see ::vl_imconvcol_vd for details on the meaning of the other parameters.
 **/

/** @fn vl_imconvcoltri_f(float*,vl_size,float const*,vl_size,vl_size,vl_size,vl_size,vl_size,int unsigned)
 ** @brief Convolve an image along the columns with a triangular kernel
 ** @see ::vl_imconvcoltri_d()
 **/

VL_EXPORT void
VL_XCAT(vl_imconvcoltri_, SFX)
(T * dest, vl_size destStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride,
 vl_size filterSize,
 vl_size step, unsigned int flags)
{
  vl_index x, y, dheight ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;
  T scale = (T) (1.0 / ((double)filterSize * (double)filterSize)) ;
  T * buffer = vl_malloc (sizeof(T) * (imageHeight + filterSize)) ;
  buffer += filterSize ;

  if (imageHeight == 0) {
    return  ;
  }

  x = 0 ;
  dheight = (imageHeight - 1) / step + 1 ;

  while (x < (signed)imageWidth) {
    T const * imagei ;
    imagei = image + x + imageStride * (imageHeight - 1) ;

    /* We decompose the convolution by a triangluar signal as the convolution
     * by two rectangular signals. The rectangular convolutions are computed
     * quickly by computing the integral signals. Each rectangular convolution
     * introduces a delay, which is compensated by convolving each in opposite
     * directions.
     */

    /* integrate backward the column */
    buffer[imageHeight - 1] = *imagei ;
    for (y = (signed)imageHeight - 2 ; y >=  0 ; --y) {
      imagei -= imageStride ;
      buffer[y] = buffer[y + 1] + *imagei ;
    }
    if (zeropad) {
      for ( ; y >= - (signed)filterSize ; --y) {
        buffer[y] = buffer[y + 1] ;
      }
    } else {
      for ( ; y >= - (signed)filterSize ; --y) {
        buffer[y] = buffer[y + 1] + *imagei ;
      }
    }

    /* compute the filter forward */
    for (y = - (signed)filterSize ;
         y < (signed)imageHeight - (signed)filterSize ; ++y) {
      buffer[y] = buffer[y] - buffer[y + filterSize] ;
    }
    if (! zeropad) {
      for (y = (signed)imageHeight - (signed)filterSize ;
           y < (signed)imageHeight ;
           ++y) {
        buffer[y] = buffer[y] - buffer[imageHeight - 1]  *
        ((signed)imageHeight - (signed)filterSize - y) ;
      }
    }

    /* integrate forward the column */
    for (y = - (signed)filterSize + 1 ;
         y < (signed)imageHeight ; ++y) {
      buffer[y] += buffer[y - 1] ;
    }

    /* compute the filter backward */
    {
      vl_size stride = transp ? 1 : destStride ;
      dest += dheight * stride ;
      for (y = step * (dheight - 1) ; y >= 0 ; y -= step) {
        dest -= stride ;
        *dest = scale * (buffer[y] - buffer[y - (signed)filterSize]) ;
      }
      dest += transp ? destStride : 1 ;
    }
    x += 1 ;
  } /* next x */
  vl_free (buffer - filterSize) ;
}

/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif

/* ---------------------------------------------------------------- */
/*                                               Gaussian Smoothing */
/* ---------------------------------------------------------------- */

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

/** @fn vl_imsmooth_d(double*,vl_size,double const*,vl_size,vl_size,vl_size,double,double)
 ** @brief Smooth an image with a Gaussian filter
 ** @param smoothed
 ** @param smoothedStride
 ** @param image
 ** @param width
 ** @param height
 ** @param stride
 ** @param sigmax
 ** @param sigmay
 **/

/** @fn vl_imsmooth_f(float*,vl_size,float const*,vl_size,vl_size,vl_size,double,double)
 ** @brief Smooth an image with a Gaussian filter
 ** @see ::vl_imsmooth_d
 **/

static T*
VL_XCAT(_vl_new_gaussian_fitler_,SFX)(vl_size *size, double sigma)
{
  T* filter ;
  T mass = (T)1.0 ;
  vl_index i ;
  vl_size width = vl_ceil_d(sigma * 3.0) ;
  *size = 2 * width + 1 ;

  assert(size) ;

  filter = vl_malloc((*size) * sizeof(T)) ;
  filter[width] = 1.0 ;
  for (i = 1 ; i <= (signed)width ; ++i) {
    double x = (double)i / sigma ;
    double g = exp(-0.5 * x * x) ;
    mass += g + g ;
    filter[width-i] = g ;
    filter[width+i] = g ;
  }
  for (i = 0 ; i < (signed)(*size) ; ++i) {filter[i] /= mass ;}
  return filter ;
}

VL_EXPORT void
VL_XCAT(vl_imsmooth_, SFX)
(T * smoothed, vl_size smoothedStride,
 T const *image, vl_size width, vl_size height, vl_size stride,
 double sigmax, double sigmay)
{
  T *filterx, *filtery, *buffer ;
  vl_size sizex, sizey ;

  filterx = VL_XCAT(_vl_new_gaussian_fitler_,SFX)(&sizex,sigmax) ;
  if (sigmax == sigmay) {
    filtery = filterx ;
    sizey = sizex ;
  } else {
    filtery = VL_XCAT(_vl_new_gaussian_fitler_,SFX)(&sizey,sigmay) ;
  }
  buffer = vl_malloc(width*height*sizeof(T)) ;

  VL_XCAT(vl_imconvcol_v,SFX) (buffer, height,
                               image, width, height, stride,
                               filtery,
                               -((signed)sizey-1)/2, ((signed)sizey-1)/2,
                               1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  VL_XCAT(vl_imconvcol_v,SFX) (smoothed, smoothedStride,
                               buffer, height, width, height,
                               filterx,
                               -((signed)sizex-1)/2, ((signed)sizex-1)/2,
                               1, VL_PAD_BY_CONTINUITY | VL_TRANSPOSE) ;

  vl_free(buffer) ;
  vl_free(filterx) ;
  if (sigmax != sigmay) {
    vl_free(filtery) ;
  }
}

/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif

/* ---------------------------------------------------------------- */
/*                                                   Image Gradient */
/* ---------------------------------------------------------------- */

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

/** @fn vl_imgradient_d(double*,double*,vl_size,vl_size,double*,vl_size,vl_size,vl_size)
 ** @brief Compute image gradient
 ** @param xGradient Pointer to amplitude gradient plane
 ** @param yGradient Pointer to angle gradient plane
 ** @param gradWidthStride Width of the gradient plane including padding
 ** @param gradHeightStride Height of the gradient plane including padding
 ** @param image Pointer to the source image
 ** @param imageWidth Source image width
 ** @param imageHeight Source image height
 ** @param imageStride Width of the image including padding.
 **
 ** This functions computes the amplitudes and angles of input image gradient.
 **
 ** Gradient is computed simple by gradient kernel \f$ (-1 ~ 1) \f$,
 ** \f$ (-1 ~ 1)^T \f$ for border pixels and with sobel filter kernel
 ** \f$ (-0.5 ~ 0 ~ 0.5) \f$, \f$ (-0.5 ~ 0 ~ 0.5)^T \f$ otherwise on the input
 ** image @a image yielding x-gradient \f$ dx \f$, stored in @a xGradient and
 ** y-gradient \f$ dy \f$, stored in @a yGradient, respectively.
 **
 ** This function also allows to process only part of the input image
 ** defining the @a imageStride as original image width and @a width as
 ** width of the sub-image.
 **
 ** Also it allows to easily align the output data by definition
 ** of the @a gradWidthStride and @a gradHeightStride .
 **/

/** @fn vl_imgradient_f(float*,float*,vl_size,vl_size,float*,vl_size,vl_size,vl_size)
 ** @brief Compute image gradient
 ** @see ::vl_imgradient_d
 **/

VL_EXPORT void
VL_XCAT(vl_imgradient_, SFX)
(T * xGradient, T * yGradient,
 vl_size gradWidthStride, vl_size gradHeightStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight,
 vl_size imageStride)
{
  /* Shortcuts */
  vl_index const xo = 1 ;
  vl_index const yo = imageStride ;
  vl_size const w = imageWidth;
  vl_size const h = imageHeight;

  T const *src, *end ;
  T *pgrad_x, *pgrad_y;
  vl_size y;

  src  = image ;
  pgrad_x = xGradient ;
  pgrad_y = yGradient ;

  /* first pixel of the first row */
  *pgrad_x = src[+xo] - src[0] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[+yo] - src[0] ;
  pgrad_y += gradWidthStride;
  src++;

  /* middle pixels of the  first row */
  end = (src - 1) + w - 1 ;
  while (src < end) {
    *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
    pgrad_x += gradWidthStride;
    *pgrad_y =        src[+yo] - src[0] ;
    pgrad_y += gradWidthStride;
    src++;
  }

  /* last pixel of the first row */
  *pgrad_x = src[0]   - src[-xo] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[+yo] - src[0] ;
  pgrad_y += gradWidthStride;
  src++;

  xGradient += gradHeightStride;
  pgrad_x = xGradient;
  yGradient += gradHeightStride;
  pgrad_y = yGradient;
  image += yo;
  src = image;

  for (y = 1 ; y < h -1 ; ++y) {

    /* first pixel of the middle rows */
    *pgrad_x =        src[+xo] - src[0] ;
    pgrad_x += gradWidthStride;
    *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
    pgrad_y += gradWidthStride;
    src++;

    /* middle pixels of the middle rows */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
      pgrad_x += gradWidthStride;
      *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
      pgrad_y += gradWidthStride;
      src++;
    }

    /* last pixel of the middle row */
    *pgrad_x =        src[0]   - src[-xo] ;
    pgrad_x += gradWidthStride;
    *pgrad_y = 0.5 * (src[+yo] - src[-yo]) ;
    pgrad_y += gradWidthStride;
    src++;

    xGradient += gradHeightStride;
    pgrad_x = xGradient;
    yGradient += gradHeightStride;
    pgrad_y = yGradient;
    image += yo;
    src = image;
  }

  /* first pixel of the last row */
  *pgrad_x = src[+xo] - src[0] ;
  pgrad_x += gradWidthStride;
  *pgrad_y = src[  0] - src[-yo] ;
  pgrad_y += gradWidthStride;
  src++;

  /* middle pixels of the last row */
  end = (src - 1) + w - 1 ;
  while (src < end) {
    *pgrad_x = 0.5 * (src[+xo] - src[-xo]) ;
    pgrad_x += gradWidthStride;
    *pgrad_y =        src[0]   - src[-yo] ;
    pgrad_y += gradWidthStride;
    src++;
  }

  /* last pixel of the last row */
  *pgrad_x = src[0]   - src[-xo] ;
  *pgrad_y = src[0]   - src[-yo] ;
}
/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif


/** @fn vl_imgradient_polar_d(double*,double*,vl_size,vl_size,double const*,vl_size,vl_size,vl_size)
 ** @brief Compute gradient mangitudes and directions of an image.
 ** @param amplitudeGradient Pointer to amplitude gradient plane
 ** @param angleGradient Pointer to angle gradient plane
 ** @param gradWidthStride Width of the gradient plane including padding
 ** @param gradHeightStride Height of the gradient plane including padding
 ** @param image Pointer to the source image
 ** @param imageWidth Source image width
 ** @param imageHeight Source image height
 ** @param imageStride Width of the source image including padding.
 **
 ** This functions computes the amplitudes and angles of input image gradient.
 **
 ** Gradient is computed simple by gradient kernel \f$ (-1 ~ 1) \f$,
 ** \f$ (-1 ~ 1)^T \f$ for border pixels and with sobel filter kernel
 ** \f$ (-0.5 ~ 0 ~ 0.5) \f$, \f$ (-0.5 ~ 0 ~ 0.5)^T \f$ otherwise on
 ** the input image @a image yielding x-gradient \f$ dx \f$, stored in
 ** @a xGradient and y-gradient \f$ dy \f$, stored in @a yGradient,
 ** respectively.
 **
 ** The amplitude of the gradient, stored in plane @a
 ** amplitudeGradient, is then calculated as \f$ \sqrt(dx^2+dy^2) \f$
 ** and the angle of the gradient, stored in @a angleGradient is \f$
 ** atan(\frac{dy}{dx}) \f$ normalised into interval 0 and @f$ 2\pi
 ** @f$.
 **
 ** This function also allows to process only part of the input image
 ** defining the @a imageStride as original image width and @a width
 ** as width of the sub-image.
 **
 ** Also it allows to easily align the output data by definition
 ** of the @a gradWidthStride and @a gradHeightStride .
 **/

/** @fn vl_imgradient_polar_f(float*,float*,vl_size,vl_size,float const*,vl_size,vl_size,vl_size)
 ** @see ::vl_imgradient_polar_d
 **/

#if (FLT == VL_TYPE_FLOAT || FLT == VL_TYPE_DOUBLE)

VL_EXPORT void
VL_XCAT(vl_imgradient_polar_, SFX)
(T * gradientModulus, T * gradientAngle,
 vl_size gradientHorizontalStride, vl_size gradHeightStride,
 T const* image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride)
{
  /* Shortcuts */
  vl_index const xo = 1 ;
  vl_index const yo = imageStride ;
  vl_size const w = imageWidth;
  vl_size const h = imageHeight;

  T const *src, *end;
  T *pgrad_angl, *pgrad_ampl;
  T gx, gy ;
  vl_size y;

#define SAVE_BACK                                                    \
*pgrad_ampl = vl_fast_sqrt_f (gx*gx + gy*gy) ;                       \
pgrad_ampl += gradientHorizontalStride ;                             \
*pgrad_angl = vl_mod_2pi_f   (vl_fast_atan2_f (gy, gx) + 2*VL_PI) ;  \
pgrad_angl += gradientHorizontalStride ;                             \
++src ;                                                              \

  src  = image ;
  pgrad_angl = gradientAngle ;
  pgrad_ampl = gradientModulus ;

  /* first pixel of the first row */
  gx = src[+xo] - src[0] ;
  gy = src[+yo] - src[0] ;
  SAVE_BACK ;

  /* middle pixels of the  first row */
  end = (src - 1) + w - 1 ;
  while (src < end) {
    gx = 0.5 * (src[+xo] - src[-xo]) ;
    gy =        src[+yo] - src[0] ;
    SAVE_BACK ;
  }

  /* last pixel of the first row */
  gx = src[0]   - src[-xo] ;
  gy = src[+yo] - src[0] ;
  SAVE_BACK ;

  gradientModulus += gradHeightStride;
  pgrad_ampl = gradientModulus;
  gradientAngle += gradHeightStride;
  pgrad_angl = gradientAngle;
  image += imageStride;
  src = image;

  for (y = 1 ; y < h -1 ; ++y) {

    /* first pixel of the middle rows */
    gx =        src[+xo] - src[0] ;
    gy = 0.5 * (src[+yo] - src[-yo]) ;
    SAVE_BACK ;

    /* middle pixels of the middle rows */
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }

    /* last pixel of the middle row */
    gx =        src[0]   - src[-xo] ;
    gy = 0.5 * (src[+yo] - src[-yo]) ;
    SAVE_BACK ;

    gradientModulus += gradHeightStride;
    pgrad_ampl = gradientModulus;
    gradientAngle += gradHeightStride;
    pgrad_angl = gradientAngle;
    image += imageStride;
    src = image;
  }

  /* first pixel of the last row */
  gx = src[+xo] - src[0] ;
  gy = src[  0] - src[-yo] ;
  SAVE_BACK ;

  /* middle pixels of the last row */
  end = (src - 1) + w - 1 ;
  while (src < end) {
    gx = 0.5 * (src[+xo] - src[-xo]) ;
    gy =        src[0]   - src[-yo] ;
    SAVE_BACK ;
  }

  /* last pixel of the last row */
  gx = src[0]   - src[-xo] ;
  gy = src[0]   - src[-yo] ;
  SAVE_BACK ;

}
/* VL_TYPE_FLOAT, VL_TYPE_DOUBLE */
#endif

/* ---------------------------------------------------------------- */
/*                                                   Integral Image */
/* ---------------------------------------------------------------- */

/** @fn vl_imintegral_d(double*,vl_size,double const*,vl_size,vl_size,vl_size)
 ** @brief Compute integral image
 **
 ** @param integral integral image.
 ** @param integralStride integral image stride.
 ** @param image source image.
 ** @param imageWidth source image width.
 ** @param imageHeight source image height.
 ** @param imageStride source image stride.
 **
 ** Let @f$ I(x,y), (x,y) \in [0, W-1] \times [0, H-1] @f$. The
 ** function computes the integral image @f$ J(x,y) @f$ of @f$ I(x,g)
 ** @f$:
 **
 ** @f[
 **   J(x,y) = \sum_{x'=0}^{x} \sum_{y'=0}^{y} I(x',y')
 ** @f]
 **
 ** The integral image @f$ J(x,y) @f$ can be used to compute quickly
 ** the integral of of @f$ I(x,y) @f$ in a rectangular region @f$ R =
 ** [x',x'']\times[y',y''] @f$:
 **
 ** @f[
 **  \sum_{(x,y)\in[x',x'']\times[y',y'']} I(x,y) =
 **  (J(x'',y'') - J(x'-1, y'')) - (J(x'',y'-1) - J(x'-1,y'-1)).
 ** @f]
 **
 ** Note that the order of operations is important when the integral image
 ** has an unsigned data type (e.g. ::vl_uint32). The formula
 ** is easily derived as follows:
 **
 ** @f{eqnarray*}
 **   \sum_{(x,y)\in R} I(x,y)
 **   &=& \sum_{x=x'}^{x''} \sum_{y=y'}^{y''} I(x,y)\\
 **   &=& \sum_{x=0}^{x''}  \sum_{y=y'}^{y''} I(x,y)
 **     - \sum_{x=0}^{x'-1} \sum_{y=y'}^{y''} I(x,y)\\
 **   &=& \sum_{x=0}^{x''}  \sum_{y=0}^{y''}  I(x,y)
 **     - \sum_{x=0}^{x''}  \sum_{y=0}^{y'-1} I(x,y)
 **     - \sum_{x=0}^{x'-1} \sum_{y=0}^{y''}  I(x,y)
 **     + \sum_{x=0}^{x'-1} \sum_{y=0}^{y'-1} I(x,y)\\
 **   &=& J(x'',y'') - J(x'-1,y'') - J(x'',y'-1) + J(x'-1,y'-1).
 ** @f}
 **/

/** @fn vl_imintegral_f(float*,vl_size,float const*,vl_size,vl_size,vl_size)
 ** @brief Compute integral image
 ** @see ::vl_imintegral_d.
 **/

/** @fn vl_imintegral_ui32(vl_uint32*,vl_size,vl_uint32 const*,vl_size,vl_size,vl_size)
 ** @brief Compute integral image
 ** @see ::vl_imintegral_d.
 **/

/** @fn vl_imintegral_i32(vl_int32*,vl_size,vl_int32 const*,vl_size,vl_size,vl_size)
 ** @brief Compute integral image
 ** @see ::vl_imintegral_d.
 **/

VL_EXPORT void
VL_XCAT(vl_imintegral_, SFX)
(T * integral, vl_size integralStride,
 T const * image,
 vl_size imageWidth, vl_size imageHeight, vl_size imageStride)
{
  vl_uindex x, y ;
  T temp  = 0 ;

  if (imageHeight > 0) {
    for (x = 0 ; x < imageWidth ; ++ x) {
      temp += *image++ ;
      *integral++ = temp ;
    }
  }

  for (y = 1 ; y < imageHeight ; ++ y) {
    T * integralPrev ;
    integral += integralStride - imageWidth ;
    image += imageStride - imageWidth ;
    integralPrev = integral - integralStride ;

    temp = 0 ;
    for (x = 0 ; x < imageWidth ; ++ x) {
      temp += *image++ ;
      *integral++ = *integralPrev++ + temp ;
    }
  }
}

/* endif VL_IMOPV_INSTANTIATING */
#undef FLT
#undef VL_IMOPV_INSTANTIATING
#endif
