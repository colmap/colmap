/** @file liop.c
 ** @brief Local Intensity Order Pattern (LIOP) descriptor - Definition
 ** @author Hana Sarbortova
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 Hana Sarbortova and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page liop Local Intensity Order Pattern (LIOP) descriptor
@author Hana Sarbortova
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref liop.h implements *Local Intensity Order Pattern descriptor*
(LIOP) of @cite{wang11local}. LIOP is a local image descriptor,
similarly to the @ref sift "SIFT descriptor".

@ref liop-starting demonstrates how to use the C API to compute the
LIOP descriptor of a patch. For further details refer to:

- @subpage liop-fundamentals - LIOP definition and parameters.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section liop-starting Getting started with LIOP
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The following code fragment demonstrates how tow to use @ref liop.h in
a C program in order to compute the LIOP descriptor of an image patch.

@code
#include <vl/liop.h>

// Create a new object instance (these numbers corresponds to parameter
// values proposed by authors of the paper, except for 41)
vl_size sideLength = 41 ;
VlLiopDesc * liop = vl_liopdesc_new_basic (sideLength);

// allocate the descriptor array
vl_size dimension = vl_liopdesc_get_dimension(liop) ;
float * desc = vl_malloc(sizeof(float) * dimension) ;

// compute descriptor from a patch (an array of length sideLegnth *
// sideLength)
vl_liopdesc_process(liop, desc, patch) ;

// delete the object
vl_liopdesc_delete(liop) ;
@endcode

The image patch must be of odd side length and in single
precision. There are several parameters affecting the LIOP
descriptor. An example is the @ref liop-weighing "threshold" used to
discard low-contrast oder pattern in the computation of the
statistics. This is changed by using ::vl_liopdesc_set_intensity_threshold.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page liop-fundamentals LIOP fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The *Local Invariant Order Pattern* (LIOP) descriptor
@cite{wang11local} is a local image descriptor based on the concept of
*local order pattern*. An order pattern is simply the order obtained
by sorting selected image samples by increasing intensity. Consider in
particular a pixel $\bx$ and $n$ neighbors
$\bx_1,\bx_2,\dots,\bx_n$. The local order pattern at $\bx$ is the
permutation $\sigma$ that sorts the neighbours by increasing intensity
$I(\bx_{\sigma(1)}) \leq I(\bx_{\sigma(2)}) \leq \dots \leq
I(\bx_{\sigma(2)})$.

An advantage of order patterns is that they are invariant to monotonic
changes of the image intensity. However, an order pattern describes
only a small portion of a patch and is not very distinctive. LIOP
assembles local order patterns computed at all image locations to
obtain a descriptor that at the same time distinctive and invariant to
monotonic intensity changes as well as image rotations.

In order to make order patterns rotation invariant, the neighborhood
of samples around $\bx$ is taken in a rotation-covariant manner. In
particular, the points $\bx_1,\dots,\bx_n$ are sampled anticlockwise
on a circle of radius $r$ around $\bx$, as shown in the following
figure:

@image html liop.png "LIOP descriptor layout: square input patch (shaded area), circular measurement region (white area), local neighborhood of a point (blue)."

Since the sample points do not necessarily have integer coordinates,
$I(\bx_i)$ is computed using bilinear interpolation.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section liop-spatial-binning Intensity rank spatial binning
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Once local order patterns are computed for all pixels $\bx$ in the
image, they can be pooled into a histogram to form an image
descriptor. Pooling discards spatial information resulting in a
warp-invariant statistics. In practice, there are two restriction on
which pixels can be used for this purpose:

- A margin of $r$ pixels from the image boundary must be maintained so
  that neighborhoods fall within the image boundaries.
- Rotation invariance requires the pooling regions to be rotation
  co-variant.  A way to do so is to make the shape of the pooling
  region rotation invariant.

For this reason, the histogram pooling region is restricted to the
circular region shown with a light color in the figure above.

In order to increase distinctiveness of the descriptor, LIOP pools
multiple histograms from a number of regions $R_1,\dots,R_m$ (spatial
pooling). These regions are selected in an illumination-invariant and
rotation-covariant manner by looking at level sets:
\[
R_t = \{\bx :\tau_{t} \leq I(\bx) < \tau_{t+1} \}.
\]
In order to be invariant to monotonic changes of the intensity, the
thresholds $\tau_t$ are selected so that all regions contain the same
number of pixels. This can be done efficiently by sorting pixels by
increasing intensity and then partitioning the resulting list into $m$
equal parts (when $m$ does not divide the number of pixels exactly,
the remaining pixels are incorporated into the last partition).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section liop-weighing Weighted pooling
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In order to compute a histogram of order pattern occurrences, one
needs to map permutations to histogram bins. This is obtained by
sorting permutation in lexycogrpahical order. For example, for $n=4$
neighbors one has the following $n!=24$ permutations:

Permutation   | Lexycographical rank
--------------|----------------------
1 2 3 4       | 1
1 2 4 3       | 2
1 3 2 4       | 3
1 3 4 2       | 4
...           | ...
4 3 1 2       | 23
4 3 2 1       | 24

In the following, $q(\bx) \in [1, n!]$ will denote the index of the
local order pattern $\sigma$ centered at pixel $\bx$.

The local order patterns $q(\bx)$ in a region $R_t$ are then pooled to
form a histogram of size $!n$. In this process, patterns are weighted
based on their stability. The latter is assumed to be proportional to
the number of pairs of pixels in the neighborhood that have a
sufficiently large intensity difference:

@f[
w(\bx) = \sum_{i=1}^n \sum_{j=1}^n [ |I(\bx_{i}) - I(\bx_{j})| >  \Theta) ]
@f]

where $[\cdot]$ is the indicator function.

In VLFeat LIOP implementation, the threshold $\Theta$ is either set as
an absolute value, or as a faction of the difference between the
maximum and minimum intensity in the image (restricted to the pixels
in the light area in the figure above).

Overall, LIOP consists of $m$ histograms of size $n!$ obtained as

\[
  h_{qt} = \sum_{\bx : q(\bx) = q \ \wedge\  \bx \in R_t} w(\bx).
\]


<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section liop-normalization Normalization
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

After computing the weighted counts $h_{qt}$, the LIOP descriptor is
obtained by stacking the values $\{h_{qt}\}$ into a vector
$\mathbf{h}$ and then normalising it:

\[
  \Phi = \frac{\mathbf{h}}{\|\mathbf{h}\|_2}
\]

The dimensionality is therefore $m n!$, where $m$ is the @c
numSpatialBins number of spatial bins and $n$ is the @c numNeighbours
number of neighbours (see ::vl_liopdesc_new). By default, this
descriptor is stored in @c single format. It can be stored as a
sequence of bytes by premultiplying the values by the constant 255 and
then rounding:

\[
 \operatorname{round}\left[ 255\, \times \Phi\right].
\]

*/

#include "liop.h"
#include "mathop.h"
#include "imopv.h"
#include <string.h>

#define DEFAULT_INTENSITY_THRESHOLD -(5.0/255)
#define DEFAULT_RADIUS 6.0
#define DEFAULT_NUM_SPATIAL_BINS 6
#define DEFAULT_NUM_NEIGHBOURS 4

/* ---------------------------------------------------------------- */
/*                                                 Helper functions */
/* ---------------------------------------------------------------- */

static
vl_int factorial(vl_int num)
{
  vl_int result = 1;
  while(num > 1){
    result = num*result;
    num--;
  }
  return result ;
}

/** @internal @brief Compute permutation index.
 ** @param permutation array containing all values from 0 to (size - 1) (input/output).
 ** @param size size of the permutation array.
 ** @return permutation index.
 **
 ** Compute the position of @a permutation in the lexycographcial
 ** sorting of permutations of the given @a size.
 **
 ** For example, in the lexicographical ordering, permutations of four elements
 ** are listed as [1 2 3 4], [1 2 4 3], [1 3 2 4], [1 3 4 2], [1 4 2 3],
 ** [1 4 3 2], [2 1 3 4], ..., [4 3 2 1].
 **
 ** The index can be computed as follows. First pick the first digit
 ** perm[1]. This is either 1,2,...,n. For each
 ** choice of the first digits, there are (n-1)! other permutations, separated
 ** therefore by (n-1)! elements in lexicographical order.
 **
 ** Process then the second digit perm[2]. This can be though as finding
 ** the lexycotraphical index of perm[2], ..., perm[n], a permutation of
 ** n-1 elements. This can be explicitly obtained by taking out 1 from
 ** all elements perm[i] > perm[1]. */

VL_INLINE vl_index get_permutation_index(vl_uindex *permutation, vl_size size){
  vl_index index = 0 ;
  vl_index i ;
  vl_index j ;

  for (i = 0 ; i < (signed)size ; ++i) {
    index = index * ((signed)size - i) + permutation[i] ;
    for (j = i + 1 ; j < (signed)size ; ++j) {
      if (permutation[j] > permutation[i]) { permutation[j] -- ; }
    }
  }
  return index ;
}

/* instantiate two quick sort algorithms */
VL_INLINE float patch_cmp (VlLiopDesc * liop, vl_index i, vl_index j)
{
  vl_index ii = liop->patchPermutation[i] ;
  vl_index jj = liop->patchPermutation[j] ;
  return liop->patchIntensities[ii] - liop->patchIntensities[jj] ;
}

VL_INLINE void patch_swap (VlLiopDesc * liop, vl_index i, vl_index j)
{
  vl_index tmp = liop->patchPermutation[i] ;
  liop->patchPermutation[i] = liop->patchPermutation[j] ;
  liop->patchPermutation[j] = tmp ;
}

#define VL_QSORT_prefix patch
#define VL_QSORT_array VlLiopDesc*
#define VL_QSORT_cmp patch_cmp
#define VL_QSORT_swap patch_swap
#include "qsort-def.h"

VL_INLINE float neigh_cmp (VlLiopDesc * liop, vl_index i, vl_index j)
{
  vl_index ii = liop->neighPermutation[i] ;
  vl_index jj = liop->neighPermutation[j] ;
  return liop->neighIntensities[ii] - liop->neighIntensities[jj] ;
}

VL_INLINE void neigh_swap (VlLiopDesc * liop, vl_index i, vl_index j)
{
  vl_index tmp = liop->neighPermutation[i] ;
  liop->neighPermutation[i] = liop->neighPermutation[j] ;
  liop->neighPermutation[j] = tmp ;
}

#define VL_QSORT_prefix neigh
#define VL_QSORT_array VlLiopDesc*
#define VL_QSORT_cmp neigh_cmp
#define VL_QSORT_swap neigh_swap
#include "qsort-def.h"

/* ---------------------------------------------------------------- */
/*                                            Construct and destroy */
/* ---------------------------------------------------------------- */

/** @brief Create a new LIOP object instance.
 ** @param numNeighbours number of neighbours.
 ** @param numSpatialBins number of bins.
 ** @param radius radius of the cirucal sample neighbourhoods.
 ** @param sideLength width of the input image patch (the patch is square).
 ** @return new object instance.
 **
 ** The value of @a radius should be at least less than half the @a
 ** sideLength of the patch.
 **/

VlLiopDesc *
vl_liopdesc_new (vl_int numNeighbours, vl_int numSpatialBins,
                 float radius, vl_size sideLength)
{
  vl_index i, t ;
  VlLiopDesc * self = vl_calloc(sizeof(VlLiopDesc), 1);

  assert(radius <= sideLength/2) ;

  self->numNeighbours = numNeighbours ;
  self->numSpatialBins = numSpatialBins ;
  self->neighRadius = radius ;
  self->intensityThreshold = DEFAULT_INTENSITY_THRESHOLD ;

  self->dimension = factorial(numNeighbours) * numSpatialBins ;

  /*
   Precompute a list of pixels within a circular patch inside
   the square image. Leave a suitable marging for sampling around
   these pixels.
   */

  self->patchSize = 0 ;
  self->patchPixels = vl_malloc(sizeof(vl_uindex)*sideLength*sideLength) ;
  self->patchSideLength = sideLength ;

  {
    vl_index x, y ;
    vl_index center = (sideLength - 1) / 2 ;
    double t = center - radius + 0.6 ;
    vl_index t2 = (vl_index) (t * t) ;
    for (y = 0 ; y < (signed)sideLength ; ++y) {
      for (x = 0 ; x < (signed)sideLength ; ++x) {
        vl_index dx = x - center ;
        vl_index dy = y - center ;
        if (x == 0 && y == 0) continue ;
        if (dx*dx + dy*dy <= t2) {
          self->patchPixels[self->patchSize++] = x + y * sideLength ;
        }
      }
    }
  }

  self->patchIntensities = vl_malloc(sizeof(vl_uindex)*self->patchSize) ;
  self->patchPermutation = vl_malloc(sizeof(vl_uindex)*self->patchSize) ;

  /*
   Precompute the samples in the circular neighbourhood of each
   measurement point.
   */

  self->neighPermutation = vl_malloc(sizeof(vl_uindex) * self->numNeighbours) ;
  self->neighIntensities = vl_malloc(sizeof(float) * self->numNeighbours) ;
  self->neighSamplesX = vl_calloc(sizeof(double), self->numNeighbours * self->patchSize) ;
  self->neighSamplesY = vl_calloc(sizeof(double), self->numNeighbours * self->patchSize) ;

  for (i = 0 ; i < (signed)self->patchSize ; ++i) {
    vl_index pixel ;
    double x, y ;
    double dangle = 2*VL_PI / (double)self->numNeighbours ;
    double angle0 ;
    vl_index center = (sideLength - 1) / 2 ;

    pixel = self->patchPixels[i] ;
    x = (pixel % (signed)self->patchSideLength) - center ;
    y = (pixel / (signed)self->patchSideLength) - center ;

    angle0 = atan2(y,x) ;

    for (t = 0 ; t < (signed)self->numNeighbours ; ++t) {
      double x1 = x + radius * cos(angle0 + dangle * t) + center ;
      double y1 = y + radius * sin(angle0 + dangle * t) + center ;
      self->neighSamplesX[t + (signed)self->numNeighbours * i] = x1 ;
      self->neighSamplesY[t + (signed)self->numNeighbours * i] = y1 ;
    }
  }
  return self ;
}

/** @brief Create a new object with default parameters
 ** @param sideLength size of the patches to be processed.
 ** @return new object.
 **
 ** @see ::vl_liopdesc_new. */

VlLiopDesc * vl_liopdesc_new_basic (vl_size sideLength)
{
  return vl_liopdesc_new(DEFAULT_NUM_NEIGHBOURS,
                         DEFAULT_NUM_SPATIAL_BINS,
                         DEFAULT_RADIUS,
                         sideLength) ;
}

/** @brief Delete object instance.
 ** @param self object instance. */

void
vl_liopdesc_delete (VlLiopDesc * self)
{
  vl_free (self->patchPixels) ;
  vl_free (self->patchIntensities) ;
  vl_free (self->patchPermutation) ;
  vl_free (self->neighPermutation) ;
  vl_free (self->neighIntensities) ;
  vl_free (self->neighSamplesX) ;
  vl_free (self->neighSamplesY) ;
  vl_free (self) ;
}

/* ---------------------------------------------------------------- */
/*                                          Compute LIOP descriptor */
/* ---------------------------------------------------------------- */

/** @brief Compute liop descriptor for a patch
 ** @param self object instance
 ** @param desc descriptor to be computed (output).
 ** @param patch patch to process
 **
 ** Use ::vl_liopdesc_get_dimension to get the size of the descriptor
 ** @a desc. */

void
vl_liopdesc_process (VlLiopDesc * self, float * desc, float const * patch)
{
  vl_index i,t ;
  vl_index offset,numPermutations ;
  vl_index spatialBinArea, spatialBinEnd, spatialBinIndex ;
  float threshold ;

  memset(desc, 0, sizeof(float) * self->dimension) ;

  /*
   * Sort pixels in the patch by increasing intensity.
   */

  for (i = 0 ; i < (signed)self->patchSize ; ++i) {
    vl_index pixel = self->patchPixels[i] ;
    self->patchIntensities[i] = patch[pixel] ;
    self->patchPermutation[i] = i ;
  }
  patch_sort(self, self->patchSize) ;

  /*
   * Tune the threshold if needed.
   */

  if (self->intensityThreshold < 0) {
    i = self->patchPermutation[0] ;
    t = self->patchPermutation[self->patchSize-1] ;
    threshold = - self->intensityThreshold
    * (self->patchIntensities[t] - self->patchIntensities[i]);
  } else {
    threshold = self->intensityThreshold ;
  }

  /*
   * Process pixels in order of increasing intenisity, dividing them into
   * spatial bins on the fly.
   */

  numPermutations = factorial(self->numNeighbours) ;
  spatialBinArea = self->patchSize / self->numSpatialBins ;
  spatialBinEnd = spatialBinArea ;
  spatialBinIndex = 0 ;
  offset = 0 ;

  for (i = 0 ; i < (signed)self->patchSize ; ++i) {
    vl_index permIndex ;
    double *sx, *sy ;

    /* advance to the next spatial bin if needed */
    if (i >= (signed)spatialBinEnd && spatialBinIndex < (signed)self->numSpatialBins - 1) {
      spatialBinEnd += spatialBinArea ;
      spatialBinIndex ++ ;
      offset += numPermutations ;
    }

    /* get intensities of neighbours of the current patch element and sort them */
    sx = self->neighSamplesX + self->numNeighbours * self->patchPermutation[i] ;
    sy = self->neighSamplesY + self->numNeighbours * self->patchPermutation[i] ;
    for (t = 0 ; t < self->numNeighbours ; ++t) {
      double x = *sx++ ;
      double y = *sy++ ;

      /* bilinear interpolation */
      vl_index ix = vl_floor_d(x) ;
      vl_index iy = vl_floor_d(y) ;

      double wx = x - ix ;
      double wy = y - iy ;

      double a = 0, b = 0, c = 0, d = 0 ;

      int L = (int) self->patchSideLength ;

      if (ix >= 0   && iy >= 0  ) { a = patch[ix   + iy * L] ; }
      if (ix <  L-1 && iy >= 0  ) { b = patch[ix+1 + iy * L] ; }
      if (ix >= 0   && iy <  L-1) { c = patch[ix   + (iy+1) * L] ; }
      if (ix <  L-1 && iy <  L-1) { d = patch[ix+1 + (iy+1) * L] ; }

      self->neighPermutation[t] = t;
      self->neighIntensities[t] = (1 - wy) * (a + (b - a) * wx) + wy * (c + (d - c) * wx) ;
    }
    neigh_sort (self, self->numNeighbours) ;

    /* get permutation index */
    permIndex = get_permutation_index(self->neighPermutation, self->numNeighbours);

    /*
     * Compute weight according to difference in intensity values and
     * accumulate.
     */
    {
      int k, t ;
      float weight = 0 ;
      for(k = 0; k < self->numNeighbours ; ++k) {
        for(t = k + 1; t < self->numNeighbours; ++t){
          double a = self->neighIntensities[k] ;
          double b = self->neighIntensities[t] ;
          weight += (a > b + threshold || b > a + threshold) ;
        }
      }
      desc[permIndex + offset] += weight ;
    }
  }

  /* normalization */
  {
    float norm = 0;
    for(i = 0; i < (signed)self->dimension; i++) {
      norm += desc[i]*desc[i];
    }
    norm = VL_MAX(sqrt(norm), 1e-12) ;
    for(i = 0; i < (signed)self->dimension; i++){
      desc[i] /= norm ;
    }
  }
}


/* ---------------------------------------------------------------- */
/*                                              Getters and setters */
/* ---------------------------------------------------------------- */

/** @brief Get the dimension of a LIOP descriptor.
 ** @param self object.
 ** @return dimension. */

vl_size
vl_liopdesc_get_dimension (VlLiopDesc const * self)
{
  return self->dimension ;
}


/** @brief Get the number of neighbours.
 ** @param self object.
 ** @return number of neighbours.
 **/

vl_size
vl_liopdesc_get_num_neighbours (VlLiopDesc const * self)
{
  assert(self) ;
  return self->numNeighbours ;
}

/** @brief Get the intensity threshold
 ** @param self object.
 ** @return intensity threshold.
 ** @see liop-weighing
 **/

float
vl_liopdesc_get_intensity_threshold (VlLiopDesc const * self)
{
  assert(self) ;
  return self->intensityThreshold ;
}

/** @brief Set the intensity threshold
 ** @param self object.
 ** @param x intensity threshold.
 **
 ** If non-negative, the threshold as is is used when comparing
 ** intensities. If negative, the absolute value of the specified
 ** number is multipled by the maximum intensity difference inside a
 ** patch to obtain the threshold.
 **
 ** @see liop-weighing
 **/

void
vl_liopdesc_set_intensity_threshold (VlLiopDesc * self, float x)
{
  assert(self) ;
  self->intensityThreshold = x ;
}

/** @brief Get the neighbourhood radius.
 ** @param self object.
 ** @return neighbourhood radius.
 **/

double
vl_liopdesc_get_neighbourhood_radius (VlLiopDesc const * self)
{
  assert(self) ;
  return self->neighRadius ;
}

/** @brief Get the number of spatial bins.
 ** @param self object.
 ** @return number of spatial bins.
 **/

vl_size
vl_liopdesc_get_num_spatial_bins (VlLiopDesc const * self)
{
  assert(self) ;
  return self->numSpatialBins ;
}
