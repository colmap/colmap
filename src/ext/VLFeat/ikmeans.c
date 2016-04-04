/** @file ikmeans.c
 ** @brief Integer K-Means clustering - Definition
 ** @author Brian Fulkerson
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file ikmeans.h
 **
 ** Integer K-means (IKM) is an implementation of K-means clustering
 ** (or Vector Quantization, VQ) for integer data. This is
 ** particularly useful for clustering large collections of visual
 ** descriptors.
 **
 ** Use the function ::vl_ikm_new() to create a IKM
 ** quantizer. Initialize the IKM quantizer with @c K clusters by
 ** ::vl_ikm_init() or similar function. Use ::vl_ikm_train() to train
 ** the quantizer. Use ::vl_ikm_push() or ::vl_ikm_push_one() to
 ** quantize new data.
 **
 ** Given data @f$x_1,\dots,x_N\in R^d@f$ and a number of clusters
 ** @f$K@f$, the goal is to find assignments @f$a_i\in\{1,\dots,K\},@f$
 ** and centers @f$c_1,\dots,c_K\in R^d@f$ so that the <em>expected
 ** distortion</em>
 **
 ** @f[
 **   E(\{a_{i}, c_j\}) = \frac{1}{N} \sum_{i=1}^N d(x_i, c_{a_i})
 ** @f]
 **
 ** is minimized. Here @f$d(x_i, c_{a_i})@f$ is the
 ** <em>distortion</em>, i.e. the cost we pay for representing @f$ x_i
 ** @f$ by @f$ c_{a_i} @f$. IKM uses the squared distortion
 ** @f$d(x,y)=\|x-y\|^2_2@f$.
 **
 ** @section ikmeans-algo Algorithms
 **
 ** @subsection ikmeans-alg-init Initialization
 **
 ** Most K-means algorithms are iterative and needs an initialization
 ** in the form of an initial choice of the centers
 ** @f$c_1,\dots,c_K@f$. We include the following options:
 **
 ** - User specified centers (::vl_ikm_init);
 ** - Random centers (::vl_ikm_init_rand);
 ** - Centers from @c K randomly selected data points (::vl_ikm_init_rand_data).
 **
 ** @subsection ikmeans-alg-lloyd Lloyd
 **
 ** The Lloyd (also known as Lloyd-Max and LBG) algorithm iteratively:
 **
 ** - Fixes the centers, optimizing the assignments (minimizing by
 **   exhaustive search the association of each data point to the
 **   centers);
 ** - Fixes the assignments and optimizes the centers (by descending
 **   the distortion error function). For the squared distortion, this
 **   step is in closed form.
 **
 ** This algorithm is not particularly efficient because all data
 ** points need to be compared to all centers, for a complexity
 ** @f$O(dNKT)@f$, where <em>T</em> is the total number of iterations.
 **
 ** @subsection ikmeans-alg-elkan Elkan
 **
 ** The Elkan algorithm is an optimized variant of Lloyd. By making
 ** use of the triangle inequality, many comparisons of data points
 ** and centers are avoided, especially at later iterations.
 ** Usually 4-5 times less comparisons than Lloyd are preformed,
 ** providing a dramatic speedup in the execution time.
 **
 **/

#include "ikmeans.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h> /* memset */
#include "assert.h"

static void vl_ikm_init_lloyd (VlIKMFilt*) ;
static void vl_ikm_init_elkan (VlIKMFilt*) ;
static int vl_ikm_train_lloyd (VlIKMFilt*, vl_uint8 const*, vl_size) ;
static int vl_ikm_train_elkan (VlIKMFilt*, vl_uint8 const*, vl_size) ;
static void vl_ikm_push_lloyd (VlIKMFilt*, vl_uint32*, vl_uint8 const*, vl_size) ;
static void  vl_ikm_push_elkan  (VlIKMFilt*, vl_uint32*, vl_uint8 const*, vl_size) ;

/** @brief Create a new IKM quantizer
 ** @param method Clustering algorithm.
 ** @return new IKM quantizer.
 **
 ** The function allocates initializes a new IKM quantizer to
 ** operate based algorithm @a method.
 **
 ** @a method has values in the enumerations ::VlIKMAlgorithms.
 **/

VlIKMFilt *
vl_ikm_new (int method)
{
  VlIKMFilt *f = vl_calloc (sizeof(VlIKMFilt), 1) ;
  f -> method = method ;
  f -> max_niters = 200 ;
  return f ;
}

/** @brief Delete IKM quantizer
 ** @param f IKM quantizer.
 **/

void
vl_ikm_delete (VlIKMFilt* f)
{
  if (f) {
    if (f->centers) vl_free(f->centers) ;
    if (f->inter_dist) vl_free(f->inter_dist) ;
    vl_free(f) ;
  }
}

/** @brief Train clusters
 ** @param f IKM quantizer.
 ** @param data data.
 ** @param N number of data (@a N @c >= 1).
 ** @return -1 if an overflow may have occurred.
 **/

int
vl_ikm_train (VlIKMFilt *f, vl_uint8 const *data, vl_size N)
{
  int err ;

  if (f-> verb) {
    VL_PRINTF ("ikm: training with %d data\n",  N) ;
    VL_PRINTF ("ikm: %d clusters\n",  f -> K) ;
  }

  switch (f -> method) {
  case VL_IKM_LLOYD : err = vl_ikm_train_lloyd (f, data, N) ; break ;
  case VL_IKM_ELKAN : err = vl_ikm_train_elkan (f, data, N) ; break ;
  default :
    abort() ;
  }
  return err ;
}

/** @brief Project data to clusters
 ** @param f     IKM quantizer.
 ** @param asgn  Assignments (out).
 ** @param data  data.
 ** @param N     number of data (@a N @c >= 1).
 **
 ** The function projects the data @a data on the integer K-means
 ** clusters specified by the IKM quantizer @a f. Notice that the
 ** quantizer must be initialized.
 **/

void
vl_ikm_push (VlIKMFilt *f, vl_uint32 *asgn, vl_uint8 const *data, vl_size N) {
  switch (f -> method) {
  case VL_IKM_LLOYD : vl_ikm_push_lloyd (f, asgn, data, N) ; break ;
  case VL_IKM_ELKAN : vl_ikm_push_elkan (f, asgn, data, N) ; break ;
  default :
    abort() ;
  }
}

/** @brief Project one datum to clusters
 ** @param centers centers.
 ** @param data datum to project.
 ** @param K number of centers.
 ** @param M dimensionality of the datum.
 ** @return the cluster index.
 **
 ** The function projects the specified datum @a data on the clusters
 ** specified by the centers @a centers.
 **/

vl_uint32
vl_ikm_push_one (vl_ikmacc_t const *centers,
		 vl_uint8 const *data,
		 vl_size M, vl_size K)
{
  vl_uindex i,k ;

  /* assign data to centers */
  vl_uindex best = (vl_uindex) -1 ;
  vl_ikmacc_t best_dist = 0 ;

  for(k = 0 ; k < K ; ++k) {
    vl_ikmacc_t dist = 0 ;

    /* compute distance with this center */
    for(i = 0 ; i < M ; ++i) {
      vl_ikmacc_t delta = (vl_ikmacc_t)data[i] - centers[k*M + i] ;
      dist += delta * delta ;
    }

    /* compare with current best */
    if (best == (vl_uindex) -1 || dist < best_dist) {
      best = k  ;
      best_dist = dist ;
    }
  }
  return (vl_uint32)best;
}

/* ---------------------------------------------------------------- */
/*                                              Getters and setters */
/* ---------------------------------------------------------------- */

/** @brief Get data dimensionality
 ** @param f IKM filter.
 ** @return data dimensionality.
 **/

vl_size
vl_ikm_get_ndims (VlIKMFilt const* f)
{
  return f->M ;
}


/** @brief Get the number of centers K
 ** @param f IKM filter.
 ** @return number of centers K.
 **/

vl_size
vl_ikm_get_K (VlIKMFilt const* f)
{
  return f->K ;
}

/** @brief Get verbosity level
 ** @param f IKM filter.
 ** @return verbosity level.
 **/

int
vl_ikm_get_verbosity (VlIKMFilt const* f)
{
  return f->verb ;
}

/** @brief Get maximum number of iterations
 ** @param f IKM filter.
 ** @return maximum number of iterations.
 **/

vl_size
vl_ikm_get_max_niters (VlIKMFilt const* f)
{
  return f->max_niters ;
}

/** @brief Get maximum number of iterations
 ** @param f IKM filter.
 ** @return maximum number of iterations.
 **/

vl_ikmacc_t const *
vl_ikm_get_centers (VlIKMFilt const* f)
{
  return f-> centers ;
}

/** @brief Set verbosity level
 ** @param f IKM filter.
 ** @param verb verbosity level.
 **/

void
vl_ikm_set_verbosity (VlIKMFilt *f, int verb)
{
  f-> verb = VL_MAX(0,verb) ;
}

/** @brief Set maximum number of iterations
 ** @param f IKM filter.
 ** @param max_niters maximum number of iterations.
 **/

void
vl_ikm_set_max_niters (VlIKMFilt *f, vl_size max_niters)
{
  f-> max_niters = max_niters ;
}

#include "ikmeans_init.tc"
#include "ikmeans_lloyd.tc"
#include "ikmeans_elkan.tc"
