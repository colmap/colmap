/** @file quickshift.c
 ** @brief Quick shift - Definition
 ** @author Brian Fulkerson
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page quickshift Quick shift image segmentation
@author Brian Fulkerson
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref quickshift.h implements an image segmentation algorithm based on
the quick shift clustering algorithm @cite{vedaldi08quick}.

- @ref quickshift-intro
- @ref quickshift-usage
- @ref quickshift-tech

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section quickshift-intro Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Quick shift @cite{vedaldi08quick} is a fast mode seeking algorithm,
similar to mean shift. The algorithm segments an RGB image (or any
image with more than one channel) by identifying clusters of pixels in
the joint spatial and color dimensions. Segments are local
(superpixels) and can be used as a basis for further processing.

Given an image, the algorithm calculates a forest of pixels whose
branches are labeled with a distance value
(::vl_quickshift_get_parents, ::vl_quickshift_get_dists). This
specifies a hierarchical segmentation of the image, with segments
corresponding to subtrees. Useful superpixels can be identified by
cutting the branches whose distance label is above a given threshold
(the threshold can be either fixed by hand, or determined by cross
validation).

Parameter influencing the algorithm are:

- <b>Kernel size.</b> The pixel density and its modes are estimated by
using a Parzen window estimator with a Gaussian kernel of the
specified size (::vl_quickshift_set_kernel_size). The larger the size,
the larger the neighborhoods of pixels considered.
- <b>Maximum distance.</b> This (::vl_quickshift_set_max_dist) is the
maximum distance between two pixels that the algorithm considers when
building the forest. In principle, it can be infinity (so that a tree
is returned), but in practice it is much faster to consider only
relatively small distances (the maximum distance can be set to a small
multiple of the kernel size).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section quickshift-usage Usage
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

- Create a new quick shift object (::vl_quickshift_new). The object
  can be reused for multiple images of the same size.
- Configure quick shift by setting the kernel size
  (::vl_quickshift_set_kernel_size) and the maximum gap
  (::vl_quickshift_set_max_dist). The latter is in principle not
  necessary, but useful to speedup processing.
- Process an image (::vl_quickshift_process).
- Retrieve the parents (::vl_quickshift_get_parents) and the distances
  (::vl_quickshift_get_dists). These can be used to segment
  the image in superpixels.
- Delete the quick shift object (::vl_quickshift_delete).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section quickshift-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

For each pixel <em>(x,y)</em>, quick shift regards @f$ (x,y,I(x,y))
@f$ as a sample from a <em>d + 2</em> dimensional vector space. It
then calculates the Parzen density estimate (with a Gaussian kernel of
standard deviation @f$ \sigma @f$)

@f[
E(x,y) = P(x,y,I(x,y)) = \sum_{x'y'}
\frac{1}{(2\pi\sigma)^{d+2}}
\exp
\left(
-\frac{1}{2\sigma^2}
\left[
\begin{array}{c}
x - x' \\
y - y' \\
I(x,y) - I(x',y') \\
\end{array}
\right]
\right)
@f]

Then quick shift construct a tree connecting each image pixel to its
nearest neighbor which has greater density value. Formally, write @f$
(x',y') >_P (x,y) @f$ if, and only if,

@f[
  P(x',y',I(x',y')) > P(x,y,I(x,y))}.
@f]

Each pixel <em>(x, y)</em> is connected to the closest higher density
pixel <em>parent(x, y)</em> that achieves the minimum distance in

@f[
 \mathrm{dist}(x,y) =
 \mathrm{min}_{(x',y') > P(x,y)}
\left(
(x - x')^2 +
(y - y')^2 +
\| I(x,y) - I(x',y') \|_2^2
\right).
@f]

**/

#include "quickshift.h"
#include "mathop.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/** -----------------------------------------------------------------
 ** @internal
 ** @brief Computes the accumulated channel L2 distance between
 **        i,j + the distance between i,j
 **
 ** @param I    input image buffer
 ** @param N1   size of the first dimension of the image
 ** @param N2   size of the second dimension of the image
 ** @param K    number of channels
 ** @param i1   first dimension index of the first pixel to compare
 ** @param i2   second dimension of the first pixel
 ** @param j1   index of the second pixel to compare
 ** @param j2   second dimension of the second pixel
 **
 ** Takes the L2 distance between the values in I at pixel i and j,
 ** accumulating along K channels and adding in the distance
 ** between i,j in the image.
 **
 ** @return the distance as described above
 **/

VL_INLINE
vl_qs_type
vl_quickshift_distance(vl_qs_type const * I,
         int N1, int N2, int K,
         int i1, int i2,
         int j1, int j2)
{
  vl_qs_type dist = 0 ;
  int d1 = j1 - i1 ;
  int d2 = j2 - i2 ;
  int k ;
  dist += d1*d1 + d2*d2 ;
  /* For k = 0...K-1, d+= L2 distance between I(i1,i2,k) and
   * I(j1,j2,k) */
  for (k = 0 ; k < K ; ++k) {
    vl_qs_type d =
      I [i1 + N1 * i2 + (N1*N2) * k] -
      I [j1 + N1 * j2 + (N1*N2) * k] ;
    dist += d*d ;
  }
  return dist ;
}

/** -----------------------------------------------------------------
 ** @internal
 ** @brief Computes the accumulated channel inner product between i,j + the
 **        distance between i,j
 **
 ** @param I    input image buffer
 ** @param N1   size of the first dimension of the image
 ** @param N2   size of the second dimension of the image
 ** @param K    number of channels
 ** @param i1   first dimension index of the first pixel to compare
 ** @param i2   second dimension of the first pixel
 ** @param j1   index of the second pixel to compare
 ** @param j2   second dimension of the second pixel
 **
 ** Takes the channel-wise inner product between the values in I at
 ** pixel i and j, accumulating along K channels and adding in the
 ** inner product between i,j in the image.
 **
 ** @return the inner product as described above
 **/

VL_INLINE
vl_qs_type
vl_quickshift_inner(vl_qs_type const * I,
      int N1, int N2, int K,
      int i1, int i2,
      int j1, int j2)
{
  vl_qs_type ker = 0 ;
  int k ;
  ker += i1*j1 + i2*j2 ;
  for (k = 0 ; k < K ; ++k) {
    ker +=
      I [i1 + N1 * i2 + (N1*N2) * k] *
      I [j1 + N1 * j2 + (N1*N2) * k] ;
  }
  return ker ;
}

/** -----------------------------------------------------------------
 ** @brief Create a quick shift object
 ** @param image the image.
 ** @param height the height (number of rows) of the image.
 ** @param width the width (number of columns) of the image.
 ** @param channels the number of channels of the image.
 ** @return new quick shift object.
 **
 ** The @c image is an array of ::vl_qs_type values with three
 ** dimensions (respectively @c widht, @c height, and @c
 ** channels). Typically, a color (e.g, RGB) image has three
 ** channels. The linear index of a pixel is computed with:
 ** @c channels * @c width* @c height + @c row + @c height * @c col.
 **/

VL_EXPORT
VlQS *
vl_quickshift_new(vl_qs_type const * image, int height, int width,
                       int channels)
{
  VlQS * q = vl_malloc(sizeof(VlQS));

  q->image    = (vl_qs_type *)image;
  q->height   = height;
  q->width    = width;
  q->channels = channels;

  q->medoid   = VL_FALSE;
  q->tau      = VL_MAX(height,width)/50;
  q->sigma    = VL_MAX(2, q->tau/3);

  q->dists    = vl_calloc(height*width, sizeof(vl_qs_type));
  q->parents  = vl_calloc(height*width, sizeof(int));
  q->density  = vl_calloc(height*width, sizeof(vl_qs_type)) ;

  return q;
}

/** -----------------------------------------------------------------
 ** @brief Create a quick shift objet
 ** @param q quick shift object.
 **/

VL_EXPORT
void vl_quickshift_process(VlQS * q)
{
  vl_qs_type const *I = q->image;
  int        *parents = q->parents;
  vl_qs_type *E = q->density;
  vl_qs_type *dists = q->dists;
  vl_qs_type *M = 0, *n = 0 ;
  vl_qs_type sigma = q->sigma ;
  vl_qs_type tau = q->tau;
  vl_qs_type tau2 = tau*tau;

  int K = q->channels, d;
  int N1 = q->height, N2 = q->width;
  int i1,i2, j1,j2, R, tR;

  d = 2 + K ; /* Total dimensions include spatial component (x,y) */

  if (q->medoid) { /* n and M are only used in mediod shift */
    M = (vl_qs_type *) vl_calloc(N1*N2*d, sizeof(vl_qs_type)) ;
    n = (vl_qs_type *) vl_calloc(N1*N2,   sizeof(vl_qs_type)) ;
  }

  R = (int) ceil (3 * sigma) ;
  tR = (int) ceil (tau) ;

  /* -----------------------------------------------------------------
   *                                                                 n
   * -------------------------------------------------------------- */

  /* If we are doing medoid shift, initialize n to the inner product of the
   * image with itself
   */
  if (n) {
    for (i2 = 0 ; i2 < N2 ; ++ i2) {
      for (i1 = 0 ; i1 < N1 ; ++ i1) {
        n [i1 + N1 * i2] = vl_quickshift_inner(I,N1,N2,K,
                                               i1,i2,
                                               i1,i2) ;
      }
    }
  }

  /* -----------------------------------------------------------------
   *                                                 E = - [oN'*F]', M
   * -------------------------------------------------------------- */

  /*
     D_ij = d(x_i,x_j)
     E_ij = exp(- .5 * D_ij / sigma^2) ;
     F_ij = - E_ij
     E_i  = sum_j E_ij
     M_di = sum_j X_j F_ij

     E is the parzen window estimate of the density
     0 = dissimilar to everything, windowsize = identical
  */

  for (i2 = 0 ; i2 < N2 ; ++ i2) {
    for (i1 = 0 ; i1 < N1 ; ++ i1) {

      int j1min = VL_MAX(i1 - R, 0   ) ;
      int j1max = VL_MIN(i1 + R, N1-1) ;
      int j2min = VL_MAX(i2 - R, 0   ) ;
      int j2max = VL_MIN(i2 + R, N2-1) ;

      /* For each pixel in the window compute the distance between it and the
       * source pixel */
      for (j2 = j2min ; j2 <= j2max ; ++ j2) {
        for (j1 = j1min ; j1 <= j1max ; ++ j1) {
          vl_qs_type Dij = vl_quickshift_distance(I,N1,N2,K, i1,i2, j1,j2) ;
          /* Make distance a similarity */
          vl_qs_type Fij = - exp(- Dij / (2*sigma*sigma)) ;

          /* E is E_i above */
          E [i1 + N1 * i2] -= Fij ;

          if (M) {
            /* Accumulate votes for the median */
            int k ;
            M [i1 + N1*i2 + (N1*N2) * 0] += j1 * Fij ;
            M [i1 + N1*i2 + (N1*N2) * 1] += j2 * Fij ;
            for (k = 0 ; k < K ; ++k) {
              M [i1 + N1*i2 + (N1*N2) * (k+2)] +=
                I [j1 + N1*j2 + (N1*N2) * k] * Fij ;
            }
          }

        } /* j1 */
      } /* j2 */

    }  /* i1 */
  } /* i2 */

  /* -----------------------------------------------------------------
   *                                               Find best neighbors
   * -------------------------------------------------------------- */

  if (q->medoid) {

    /*
       Qij = - nj Ei - 2 sum_k Gjk Mik
       n is I.^2
    */

    /* medoid shift */
    for (i2 = 0 ; i2 < N2 ; ++i2) {
      for (i1 = 0 ; i1 < N1 ; ++i1) {

        vl_qs_type sc_best = 0  ;
        /* j1/j2 best are the best indicies for each i */
        vl_qs_type j1_best = i1 ;
        vl_qs_type j2_best = i2 ;

        int j1min = VL_MAX(i1 - R, 0   ) ;
        int j1max = VL_MIN(i1 + R, N1-1) ;
        int j2min = VL_MAX(i2 - R, 0   ) ;
        int j2max = VL_MIN(i2 + R, N2-1) ;

        for (j2 = j2min ; j2 <= j2max ; ++ j2) {
          for (j1 = j1min ; j1 <= j1max ; ++ j1) {

            vl_qs_type Qij = - n [j1 + j2 * N1] * E [i1 + i2 * N1] ;
            int k ;

            Qij -= 2 * j1 * M [i1 + i2 * N1 + (N1*N2) * 0] ;
            Qij -= 2 * j2 * M [i1 + i2 * N1 + (N1*N2) * 1] ;
            for (k = 0 ; k < K ; ++k) {
              Qij -= 2 *
                I [j1 + j2 * N1 + (N1*N2) * k] *
                M [i1 + i2 * N1 + (N1*N2) * (k + 2)] ;
            }

            if (Qij > sc_best) {
              sc_best = Qij ;
              j1_best = j1 ;
              j2_best = j2 ;
            }
          }
        }

        /* parents_i is the linear index of j which is the best pair
         * dists_i is the score of the best match
         */
        parents [i1 + N1 * i2] = j1_best + N1 * j2_best ;
        dists[i1 + N1 * i2] = sc_best ;
      }
    }

  } else {

    /* Quickshift assigns each i to the closest j which has an increase in the
     * density (E). If there is no j s.t. Ej > Ei, then dists_i == inf (a root
     * node in one of the trees of merges).
     */
    for (i2 = 0 ; i2 < N2 ; ++i2) {
      for (i1 = 0 ; i1 < N1 ; ++i1) {

        vl_qs_type E0 = E [i1 + N1 * i2] ;
        vl_qs_type d_best = VL_QS_INF ;
        vl_qs_type j1_best = i1   ;
        vl_qs_type j2_best = i2   ;

        int j1min = VL_MAX(i1 - tR, 0   ) ;
        int j1max = VL_MIN(i1 + tR, N1-1) ;
        int j2min = VL_MAX(i2 - tR, 0   ) ;
        int j2max = VL_MIN(i2 + tR, N2-1) ;

        for (j2 = j2min ; j2 <= j2max ; ++ j2) {
          for (j1 = j1min ; j1 <= j1max ; ++ j1) {
            if (E [j1 + N1 * j2] > E0) {
              vl_qs_type Dij = vl_quickshift_distance(I,N1,N2,K, i1,i2, j1,j2) ;
              if (Dij <= tau2 && Dij < d_best) {
                d_best = Dij ;
                j1_best = j1 ;
                j2_best = j2 ;
              }
            }
          }
        }

        /* parents is the index of the best pair */
        /* dists_i is the minimal distance, inf implies no Ej > Ei within
         * distance tau from the point */
        parents [i1 + N1 * i2] = j1_best + N1 * j2_best ;
        dists[i1 + N1 * i2] = sqrt(d_best) ;
      }
    }
  }

  if (M) vl_free(M) ;
  if (n) vl_free(n) ;
}

/** -----------------------------------------------------------------
 ** @brief Delete quick shift object
 ** @param q quick shift object.
 **/

void vl_quickshift_delete(VlQS * q)
{
  if (q) {
    if (q->parents) vl_free(q->parents);
    if (q->dists)   vl_free(q->dists);
    if (q->density) vl_free(q->density);

    vl_free(q);
  }
}
