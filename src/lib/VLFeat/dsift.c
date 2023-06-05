/** @file dsift.c
 ** @brief Dense SIFT - Definition
 ** @author Andrea Vedaldi
 ** @author Brian Fulkerson
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "dsift.h"
#include "pgm.h"
#include "mathop.h"
#include "imopv.h"
#include <math.h>
#include <string.h>

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page dsift Dense Scale Invariant Feature Transform (DSIFT)
@author Andrea Vedaldi
@author Brian Fulkerson
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref dsift.h implements a dense version of @ref sift.h "SIFT". This is
an object that can quickly compute descriptors for densely sampled
keypoints with identical size and orientation. It can be reused for
multiple images of the same size.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section dsift-intro Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@sa @ref sift "The SIFT module", @ref dsift-tech "Technical details"

This module implements a fast algorithm for the calculation of a large
number of SIFT descriptors of densely sampled features of the same
scale and orientation. See the @ref sift "SIFT module" for an
overview of SIFT.

The feature frames (keypoints) are indirectly specified by the
sampling steps (::vl_dsift_set_steps) and the sampling bounds
(::vl_dsift_set_bounds).  The descriptor geometry (number and size of
the spatial bins and number of orientation bins) can be customized
(::vl_dsift_set_geometry, ::VlDsiftDescriptorGeometry).

@image html dsift-geom.png "Dense SIFT descriptor geometry"

By default, SIFT uses a Gaussian windowing function that discounts
contributions of gradients further away from the descriptor
centers. This function can be changed to a flat window by invoking
::vl_dsift_set_flat_window. In this case, gradients are accumulated
using only bilinear interpolation, but instad of being reweighted by a
Gassuain window, they are all weighted equally. However, after
gradients have been accumulated into a spatial bin, the whole bin is
reweighted by the average of the Gaussian window over the spatial
support of that bin. This &ldquo;approximation&rdquo; substantially
improves speed with little or no loss of performance in applications.

Keypoints are sampled in such a way that the centers of the spatial
bins are at integer coordinates within the image boundaries. For
instance, the top-left bin of the top-left descriptor is centered on
the pixel (0,0). The bin immediately to the right at
(<code>binSizeX</code>,0), where <code>binSizeX</code> is a paramtere
in the ::VlDsiftDescriptorGeometry structure. ::vl_dsift_set_bounds
can be used to further restrict sampling to the keypoints in an image.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
 @section dsift-usage Usage
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

DSIFT is implemented by a ::VlDsiftFilter object that can be used
to process a sequence of images of a given geometry.
To use the <b>DSIFT filter</b>:

- Initialize a new DSIFT filter object by ::vl_dsift_new (or the simplified
::vl_dsift_new_basic). Customize the descriptor parameters by
::vl_dsift_set_steps, ::vl_dsift_set_geometry, etc.
- Process an image by ::vl_dsift_process.
- Retrieve the number of keypoints (::vl_dsift_get_keypoint_num), the
  keypoints (::vl_dsift_get_keypoints), and their descriptors
  (::vl_dsift_get_descriptors).
- Optionally repeat for more images.
- Delete the DSIFT filter by ::vl_dsift_delete.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section dsift-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This section extends the @ref sift-tech-descriptor "SIFT descriptor section"
and specialzies it to the case of dense keypoints.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection dsift-tech-descriptor-dense Dense descriptors
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

When computing descriptors for many keypoints differing only by their
position (and with null rotation), further simplifications are
possible. In this case, in fact,

@f{eqnarray*}
     \mathbf{x} &=& m \sigma \hat{\mathbf{x}} + T,\\
 h(t,i,j)
 &=&
 m \sigma \int
 g_{\sigma_\mathrm{win}}(\mathbf{x} - T)\,
 w_\mathrm{ang}(\angle J(\mathbf{x}) - \theta_t)\,
 w\left(\frac{x - T_x}{m\sigma} - \hat{x}_i\right)\,
 w\left(\frac{y - T_y}{m\sigma} - \hat{y}_j\right)\,
 |J(\mathbf{x})|\,
 d\mathbf{x}.
@f}

Since many different values of @e T are sampled, this is conveniently
expressed as a separable convolution. First, we translate by @f$
\mathbf{x}_{ij} = m\sigma(\hat x_i,\ \hat y_i)^\top @f$ and we use the
symmetry of the various binning and windowing functions to write

@f{eqnarray*}
 h(t,i,j)
 &=&
 m \sigma \int
 g_{\sigma_\mathrm{win}}(T' - \mathbf{x} - \mathbf{x}_{ij})\,
 w_\mathrm{ang}(\angle J(\mathbf{x}) - \theta_t)\,
 w\left(\frac{T'_x - x}{m\sigma}\right)\,
 w\left(\frac{T'_y - y}{m\sigma}\right)\,
 |J(\mathbf{x})|\,
 d\mathbf{x},
\\
T' &=& T + m\sigma
\left[\begin{array}{cc} x_i \\ y_j \end{array}\right].
@f}

Then we define kernels

@f{eqnarray*}
 k_i(x) &=&
 \frac{1}{\sqrt{2\pi} \sigma_{\mathrm{win}}}
 \exp\left(
 -\frac{1}{2}
 \frac{(x-x_i)^2}{\sigma_{\mathrm{win}}^2}
 \right)
 w\left(\frac{x}{m\sigma}\right),
 \\
 k_j(y) &=&
 \frac{1}{\sqrt{2\pi} \sigma_{\mathrm{win}}}
 \exp\left(
 -\frac{1}{2}
 \frac{(y-y_j)^2}{\sigma_{\mathrm{win}}^2}
 \right)
 w\left(\frac{y}{m\sigma}\right),
@f}

and obtain

@f{eqnarray*}
 h(t,i,j) &=& (k_ik_j * \bar J_t)\left( T + m\sigma
\left[\begin{array}{cc} x_i \\ y_j \end{array}\right] \right),
\\
\bar J_t(\mathbf{x}) &=&  w_\mathrm{ang}(\angle J(\mathbf{x}) - \theta_t)\,|J(\mathbf{x})|.
@f}

Furthermore, if we use a flat rather than Gaussian windowing function,
the kernels do not depend on the bin, and we have

@f{eqnarray*}
 k(z) &=&
 \frac{1}{\sigma_{\mathrm{win}}}
 w\left(\frac{z}{m\sigma}\right),
\\
 h(t,i,j) &=& (k(x)k(y) * \bar J_t)\left( T + m\sigma
\left[\begin{array}{cc} x_i \\ y_j \end{array}\right] \right),
@f}

(here @f$ \sigma_\mathrm{win} @f$ is the side of the flat window).

@note In this case the binning functions @f$ k(z) @f$ are triangular
and the convolution can be computed in time independent on the filter
(i.e. descriptor bin) support size by integral signals.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection dsift-tech-sampling Sampling
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

To avoid resampling and dealing with special boundary conditions, we
impose some mild restrictions on the geometry of the descriptors that
can be computed. In particular, we impose that the bin centers @f$ T +
m\sigma (x_i,\ y_j) @f$ are always at integer coordinates within the
image boundaries. This eliminates the need for costly interpolation.
This condition amounts to (expressed in terms of the @e x coordinate,
and equally applicable to @e y)

@f[
 \{0,\dots, W-1\} \ni T_x + m\sigma x_i =
 T_x + m\sigma i - \frac{N_x-1}{2}
 = \bar T_x + m\sigma i,
 \qquad i = 0,\dots,N_x-1.
@f]

Notice that for this condition to be satisfied, the @em descriptor
center @f$ T_x @f$ needs to be either fractional or integer depending
on @f$ N_x @f$ being even or odd. To eliminate this complication,
it is simpler to use as a reference not the descriptor center @e T,
but the coordinates of the upper-left bin @f$ \bar T @f$. Thus we
sample the latter on a regular (integer) grid

@f[
 \left[\begin{array}{cc}
   0 \\
   0
 \end{array}\right]
 \leq
 \bar T =
 \left[\begin{array}{cc}
   \bar T_x^{\min} + p \Delta_x \\
   \bar T_y^{\min} + q \Delta_y \\
 \end{array}\right]
 \leq
 \left[\begin{array}{cc}
   W - 1 - m\sigma N_x \\
   H - 1 - m\sigma N_y
 \end{array}\right],
 \quad
 \bar T =
 \left[\begin{array}{cc}
   T_x - \frac{N_x - 1}{2} \\
   T_y - \frac{N_y - 1}{2} \\
  \end{array}\right]
@f]

and we impose that the bin size @f$ m \sigma @f$ is integer as well.

**/

/** ------------------------------------------------------------------
 ** @internal @brief Initialize new convolution kernel
 ** @param binSize
 ** @param numBins
 ** @param binIndex negative to use flat window.
 ** @param windowSize
 ** @return a pointer to new filter.
 **/

float *
_vl_dsift_new_kernel (int binSize, int numBins, int binIndex, double windowSize)
{
  int filtLen = 2 * binSize - 1 ;
  float * ker = vl_malloc (sizeof(float) * filtLen) ;
  float * kerIter = ker ;
  float delta = binSize * (binIndex - 0.5F * (numBins - 1)) ;
  /*
  float sigma = 0.5F * ((numBins - 1) * binSize + 1) ;
  float sigma = 0.5F * ((numBins) * binSize) ;
  */
  float sigma = (float) binSize * (float) windowSize ;
  int x ;

  for (x = - binSize + 1 ; x <= + binSize - 1 ; ++ x) {
    float z = (x - delta) / sigma ;
    *kerIter++ = (1.0F - fabsf(x) / binSize) *
      ((binIndex >= 0) ? expf(- 0.5F * z*z) : 1.0F) ;
  }
  return ker ;
}

static float
_vl_dsift_get_bin_window_mean
(int binSize, int numBins, int binIndex, double windowSize)
{
  float delta = binSize * (binIndex - 0.5F * (numBins - 1)) ;
  /*float sigma = 0.5F * ((numBins - 1) * binSize + 1) ;*/
  float sigma = (float) binSize * (float) windowSize ;
  int x ;

  float acc = 0.0 ;
  for (x = - binSize + 1 ; x <= + binSize - 1 ; ++ x) {
    float z = (x - delta) / sigma ;
    acc += ((binIndex >= 0) ? expf(- 0.5F * z*z) : 1.0F) ;
  }
  return acc /= (2 * binSize - 1) ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Normalize histogram
 ** @param begin first element of the histogram.
 ** @param end last plus one element of the histogram.
 **
 ** The function divides the specified histogram by its l2 norm.
 **/

VL_INLINE float
_vl_dsift_normalize_histogram (float * begin, float * end)
{
  float * iter ;
  float  norm = 0.0F ;

  for (iter = begin ; iter < end ; ++ iter) {
    norm += (*iter) * (*iter) ;
  }
  norm = vl_fast_sqrt_f (norm) + VL_EPSILON_F ;

  for (iter = begin; iter < end ; ++ iter) {
    *iter /= norm ;
  }
  return norm ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Free internal buffers
 ** @param self DSIFT filter.
 **/

static void
_vl_dsift_free_buffers (VlDsiftFilter* self)
{
  if (self->frames) {
    vl_free(self->frames) ;
    self->frames = NULL ;
  }
  if (self->descrs) {
    vl_free(self->descrs) ;
    self->descrs = NULL ;
  }
  if (self->grads) {
    int t ;
    for (t = 0 ; t < self->numGradAlloc ; ++t)
      if (self->grads[t]) vl_free(self->grads[t]) ;
    vl_free(self->grads) ;
    self->grads = NULL ;
  }
  self->numFrameAlloc = 0 ;
  self->numBinAlloc = 0 ;
  self->numGradAlloc = 0 ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Updates internal buffers to current geometry
 **/

VL_EXPORT void
_vl_dsift_update_buffers (VlDsiftFilter * self)
{
  int x1 = self->boundMinX ;
  int x2 = self->boundMaxX ;
  int y1 = self->boundMinY ;
  int y2 = self->boundMaxY ;

  int rangeX = x2 - x1 - (self->geom.numBinX - 1) * self->geom.binSizeX ;
  int rangeY = y2 - y1 - (self->geom.numBinY - 1) * self->geom.binSizeY ;

  int numFramesX = (rangeX >= 0) ? rangeX / self->stepX + 1 : 0 ;
  int numFramesY = (rangeY >= 0) ? rangeY / self->stepY + 1 : 0 ;

  self->numFrames = numFramesX * numFramesY ;
  self->descrSize = self->geom.numBinT *
                    self->geom.numBinX *
                    self->geom.numBinY ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Allocate internal buffers
 ** @param self DSIFT filter.
 **
 ** The function (re)allocates the internal buffers in accordance with
 ** the current image and descriptor geometry.
 **/

static void
_vl_dsift_alloc_buffers (VlDsiftFilter* self)
{
  _vl_dsift_update_buffers (self) ;
  {
    int numFrameAlloc = vl_dsift_get_keypoint_num (self) ;
    int numBinAlloc   = vl_dsift_get_descriptor_size (self) ;
    int numGradAlloc  = self->geom.numBinT ;

    /* see if we need to update the buffers */
    if (numBinAlloc != self->numBinAlloc ||
        numGradAlloc != self->numGradAlloc ||
        numFrameAlloc != self->numFrameAlloc) {

      int t ;

      _vl_dsift_free_buffers(self) ;

      self->frames = vl_malloc(sizeof(VlDsiftKeypoint) * numFrameAlloc) ;
      self->descrs = vl_malloc(sizeof(float) * numBinAlloc * numFrameAlloc) ;
      self->grads  = vl_malloc(sizeof(float*) * numGradAlloc) ;
      for (t = 0 ; t < numGradAlloc ; ++t) {
        self->grads[t] =
          vl_malloc(sizeof(float) * self->imWidth * self->imHeight) ;
      }
      self->numBinAlloc = numBinAlloc ;
      self->numGradAlloc = numGradAlloc ;
      self->numFrameAlloc = numFrameAlloc ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Create a new DSIFT filter
 **
 ** @param imWidth width of the image.
 ** @param imHeight height of the image
 **
 ** @return new filter.
 **/

VL_EXPORT VlDsiftFilter *
vl_dsift_new (int imWidth, int imHeight)
{
  VlDsiftFilter * self = vl_malloc (sizeof(VlDsiftFilter)) ;
  self->imWidth  = imWidth ;
  self->imHeight = imHeight ;

  self->stepX = 5 ;
  self->stepY = 5 ;

  self->boundMinX = 0 ;
  self->boundMinY = 0 ;
  self->boundMaxX = imWidth - 1 ;
  self->boundMaxY = imHeight - 1 ;

  self->geom.numBinX = 4 ;
  self->geom.numBinY = 4 ;
  self->geom.numBinT = 8 ;
  self->geom.binSizeX = 5 ;
  self->geom.binSizeY = 5 ;

  self->useFlatWindow = VL_FALSE ;
  self->windowSize = 2.0 ;

  self->convTmp1 = vl_malloc(sizeof(float) * self->imWidth * self->imHeight) ;
  self->convTmp2 = vl_malloc(sizeof(float) * self->imWidth * self->imHeight) ;

  self->numBinAlloc = 0 ;
  self->numFrameAlloc = 0 ;
  self->numGradAlloc = 0 ;

  self->descrSize = 0 ;
  self->numFrames = 0 ;
  self->grads = NULL ;
  self->frames = NULL ;
  self->descrs = NULL ;

  _vl_dsift_update_buffers(self) ;
  return self ;
}

/** ------------------------------------------------------------------
 ** @brief Create a new DSIFT filter (basic interface)
 ** @param imWidth width of the image.
 ** @param imHeight height of the image.
 ** @param step sampling step.
 ** @param binSize bin size.
 ** @return new filter.
 **
 ** The descriptor geometry matches the standard SIFT descriptor.
 **/

VL_EXPORT VlDsiftFilter *
vl_dsift_new_basic (int imWidth, int imHeight, int step, int binSize)
{
  VlDsiftFilter* self = vl_dsift_new(imWidth, imHeight) ;
  VlDsiftDescriptorGeometry geom = *vl_dsift_get_geometry(self) ;
  geom.binSizeX = binSize ;
  geom.binSizeY = binSize ;
  vl_dsift_set_geometry(self, &geom) ;
  vl_dsift_set_steps(self, step, step) ;
  return self ;
}

/** ------------------------------------------------------------------
 ** @brief Delete DSIFT filter
 ** @param self DSIFT filter.
 **/

VL_EXPORT void
vl_dsift_delete (VlDsiftFilter * self)
{
  _vl_dsift_free_buffers (self) ;
  if (self->convTmp2) vl_free (self->convTmp2) ;
  if (self->convTmp1) vl_free (self->convTmp1) ;
  vl_free (self) ;
}


/** ------------------------------------------------------------------
 ** @internal @brief Process with Gaussian window
 ** @param self DSIFT filter.
 **/

VL_INLINE void
_vl_dsift_with_gaussian_window (VlDsiftFilter * self)
{
  int binx, biny, bint ;
  int framex, framey ;
  float *xker, *yker ;

  int Wx = self->geom.binSizeX - 1 ;
  int Wy = self->geom.binSizeY - 1 ;

  for (biny = 0 ; biny < self->geom.numBinY ; ++biny) {

    yker = _vl_dsift_new_kernel (self->geom.binSizeY,
                                 self->geom.numBinY,
                                 biny,
                                 self->windowSize) ;

    for (binx = 0 ; binx < self->geom.numBinX ; ++binx) {

      xker = _vl_dsift_new_kernel(self->geom.binSizeX,
                                  self->geom.numBinX,
                                  binx,
                                  self->windowSize) ;

      for (bint = 0 ; bint < self->geom.numBinT ; ++bint) {

        vl_imconvcol_vf (self->convTmp1, self->imHeight,
                         self->grads[bint], self->imWidth, self->imHeight,
                         self->imWidth,
                         yker, -Wy, +Wy, 1,
                         VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;

        vl_imconvcol_vf (self->convTmp2, self->imWidth,
                         self->convTmp1, self->imHeight, self->imWidth,
                         self->imHeight,
                         xker, -Wx, +Wx, 1,
                         VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;

        {
          float *dst = self->descrs
            + bint
            + binx * self->geom.numBinT
            + biny * (self->geom.numBinX * self->geom.numBinT)  ;

          float *src = self->convTmp2 ;

          int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1 ;
          int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1 ;
          int descrSize = vl_dsift_get_descriptor_size (self) ;

          for (framey  = self->boundMinY ;
               framey <= self->boundMaxY - frameSizeY + 1 ;
               framey += self->stepY) {
            for (framex  = self->boundMinX ;
                 framex <= self->boundMaxX - frameSizeX + 1 ;
                 framex += self->stepX) {
              *dst = src [(framex + binx * self->geom.binSizeX) * 1 +
                          (framey + biny * self->geom.binSizeY) * self->imWidth]  ;
              dst += descrSize ;
            } /* framex */
          } /* framey */
        }

      } /* for bint */
      vl_free (xker) ;
    } /* for binx */
    vl_free (yker) ;
  } /* for biny */
}

/** ------------------------------------------------------------------
 ** @internal @brief Process with flat window.
 ** @param self DSIFT filter object.
 **/

VL_INLINE void
_vl_dsift_with_flat_window (VlDsiftFilter* self)
{
  int binx, biny, bint ;
  int framex, framey ;

  /* for each orientation bin */
  for (bint = 0 ; bint < self->geom.numBinT ; ++bint) {

    vl_imconvcoltri_f (self->convTmp1, self->imHeight,
                       self->grads [bint], self->imWidth, self->imHeight,
                       self->imWidth,
                       self->geom.binSizeY, /* filt size */
                       1, /* subsampling step */
                       VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;

    vl_imconvcoltri_f (self->convTmp2, self->imWidth,
                       self->convTmp1, self->imHeight, self->imWidth,
                       self->imHeight,
                       self->geom.binSizeX,
                       1,
                       VL_PAD_BY_CONTINUITY|VL_TRANSPOSE) ;

    for (biny = 0 ; biny < self->geom.numBinY ; ++biny) {

      /*
      This fast version of DSIFT does not use a proper Gaussian
      weighting scheme for the gradiens that are accumulated on the
      spatial bins. Instead each spatial bins is accumulated based on
      the triangular kernel only, equivalent to bilinear interpolation
      plus a flat, rather than Gaussian, window. Eventually, however,
      the magnitude of the spatial bins in the SIFT descriptor is
      reweighted by the average of the Gaussian window on each bin.
      */

      float wy = _vl_dsift_get_bin_window_mean
        (self->geom.binSizeY, self->geom.numBinY, biny,
         self->windowSize) ;

      /* The convolution functions vl_imconvcoltri_* convolve by a
       * triangular kernel with unit integral. Instead for SIFT the
       * triangular kernel should have unit height. This is
       * compensated for by multiplying by the bin size:
       */

      wy *= self->geom.binSizeY ;

      for (binx = 0 ; binx < self->geom.numBinX ; ++binx) {
        float w ;
        float wx = _vl_dsift_get_bin_window_mean (self->geom.binSizeX,
                                                  self->geom.numBinX,
                                                  binx,
                                                  self->windowSize) ;

        float *dst = self->descrs
          + bint
          + binx * self->geom.numBinT
          + biny * (self->geom.numBinX * self->geom.numBinT)  ;

        float *src = self->convTmp2 ;

        int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1 ;
        int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1 ;
        int descrSize = vl_dsift_get_descriptor_size (self) ;

        wx *= self->geom.binSizeX ;
        w = wx * wy ;

        for (framey  = self->boundMinY ;
             framey <= self->boundMaxY - frameSizeY + 1 ;
             framey += self->stepY) {
          for (framex  = self->boundMinX ;
               framex <= self->boundMaxX - frameSizeX + 1 ;
               framex += self->stepX) {
            *dst = w * src [(framex + binx * self->geom.binSizeX) * 1 +
                            (framey + biny * self->geom.binSizeY) * self->imWidth]  ;
            dst += descrSize ;
          } /* framex */
        } /* framey */
      } /* binx */
    } /* biny */
  } /* bint */
}

/** ------------------------------------------------------------------
 ** @brief Compute keypoints and descriptors
 **
 ** @param self DSIFT filter.
 ** @param im   image data.
 **/

void vl_dsift_process (VlDsiftFilter* self, float const* im)
{
  int t, x, y ;

  /* update buffers */
  _vl_dsift_alloc_buffers (self) ;

  /* clear integral images */
  for (t = 0 ; t < self->geom.numBinT ; ++t)
    memset (self->grads[t], 0,
            sizeof(float) * self->imWidth * self->imHeight) ;

#undef at
#define at(x,y) (im[(y)*self->imWidth+(x)])

  /* Compute gradients, their norm, and their angle */

  for (y = 0 ; y < self->imHeight ; ++ y) {
    for (x = 0 ; x < self->imWidth ; ++ x) {
      float gx, gy ;
      float angle, mod, nt, rbint ;
      int bint ;

      /* y derivative */
      if (y == 0) {
        gy = at(x,y+1) - at(x,y) ;
      } else if (y == self->imHeight - 1) {
        gy = at(x,y) - at(x,y-1) ;
      } else {
        gy = 0.5F * (at(x,y+1) - at(x,y-1)) ;
      }

      /* x derivative */
      if (x == 0) {
        gx = at(x+1,y) - at(x,y) ;
      } else if (x == self->imWidth - 1) {
        gx = at(x,y) - at(x-1,y) ;
      } else {
        gx = 0.5F * (at(x+1,y) - at(x-1,y)) ;
      }

      /* angle and modulus */
      angle = vl_fast_atan2_f (gy,gx) ;
      mod = vl_fast_sqrt_f (gx*gx + gy*gy) ;

      /* quantize angle */
      nt = vl_mod_2pi_f (angle) * (self->geom.numBinT / (2*VL_PI)) ;
      bint = (int) vl_floor_f (nt) ;
      rbint = nt - bint ;

      /* write it back */
      self->grads [(bint    ) % self->geom.numBinT][x + y * self->imWidth] = (1 - rbint) * mod ;
      self->grads [(bint + 1) % self->geom.numBinT][x + y * self->imWidth] = (    rbint) * mod ;
    }
  }

  if (self->useFlatWindow) {
    _vl_dsift_with_flat_window(self) ;
  } else {
    _vl_dsift_with_gaussian_window(self) ;
  }

  {
    VlDsiftKeypoint* frameIter = self->frames ;
    float * descrIter = self->descrs ;
    int framex, framey, bint ;

    int frameSizeX = self->geom.binSizeX * (self->geom.numBinX - 1) + 1 ;
    int frameSizeY = self->geom.binSizeY * (self->geom.numBinY - 1) + 1 ;
    int descrSize = vl_dsift_get_descriptor_size (self) ;

    float deltaCenterX = 0.5F * self->geom.binSizeX * (self->geom.numBinX - 1) ;
    float deltaCenterY = 0.5F * self->geom.binSizeY * (self->geom.numBinY - 1) ;

    float normConstant = frameSizeX * frameSizeY ;

    for (framey  = self->boundMinY ;
         framey <= self->boundMaxY - frameSizeY + 1 ;
         framey += self->stepY) {

      for (framex  = self->boundMinX ;
           framex <= self->boundMaxX - frameSizeX + 1 ;
           framex += self->stepX) {

        frameIter->x    = framex + deltaCenterX ;
        frameIter->y    = framey + deltaCenterY ;

        /* mass */
        {
          float mass = 0 ;
          for (bint = 0 ; bint < descrSize ; ++ bint)
            mass += descrIter[bint] ;
          mass /= normConstant ;
          frameIter->norm = mass ;
        }

        /* L2 normalize */
        _vl_dsift_normalize_histogram (descrIter, descrIter + descrSize) ;

        /* clamp */
        for(bint = 0 ; bint < descrSize ; ++ bint)
          if (descrIter[bint] > 0.2F) descrIter[bint] = 0.2F ;

        /* L2 normalize */
        _vl_dsift_normalize_histogram (descrIter, descrIter + descrSize) ;

        frameIter ++ ;
        descrIter += descrSize ;
      } /* for framex */
    } /* for framey */
  }
}
