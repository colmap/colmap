/** @file kmeans.c
 ** @brief K-means - Declaration
 ** @author Andrea Vedaldi, David Novotny
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi and David Novotny.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page kmeans K-means clustering
@author Andrea Vedaldi
@author David Novotny
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref kmeans.h implements a number of algorithm for **K-means
quantization**: Lloyd @cite{lloyd82least}, an accelerated version by
Elkan @cite{elkan03using}, and a large scale algorithm based on
Approximate Nearest Neighbors (ANN). All algorithms support @c float
or @c double data and can use the $l^1$ or the $l^2$ distance for
clustering. Furthermore, all algorithms can take advantage of multiple
CPU cores.

Please see @subpage kmeans-fundamentals for a technical description of
K-means and of the algorithms implemented here.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section kmeans-starting Getting started
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The goal of K-means is to partition a dataset into $K$
&ldquo;compact&rdquo; clusters. The following example demonstrates
using @ref kmeans.h in the C programming language to partition @c
numData @c float vectors into compute @c numCenters clusters using
Lloyd's algorithm:

@code
#include <vl/kmeans.h>
double energy ;
double * centers ;

// Use float data and the L2 distance for clustering
KMeans * kmeans = vl_kmeans_new (VLDistanceL2, VL_TYPE_FLOAT) ;

// Use Lloyd algorithm
vl_kmeans_set_algorithm (kmeans, VlKMeansLloyd) ;

// Initialize the cluster centers by randomly sampling the data
vl_kmeans_init_centers_with_rand_data (kmeans, data, dimension, numData, numCenters) ;

// Run at most 100 iterations of cluster refinement using Lloyd algorithm
vl_kmeans_set_max_num_iterations (kmeans, 100) ;
vl_kmeans_refine_centers (kmeans, data, numData) ;

// Obtain the energy of the solution
energy = vl_kmeans_get_energy(kmeans) ;

// Obtain the cluster centers
centers = vl_kmeans_get_centers(kmeans) ;
@endcode

Once the centers have been obtained, new data points can be assigned
to clusters by using the ::vl_kmeans_quantize function:

@code
vl_uint32 * assignments = vl_malloc(sizeof(vl_uint32) * numData) ;
float * distances = vl_malloc(sizeof(float) * numData) ;
vl_kmeans_quantize(kmeans, assignments, distances, data, numData) ;
@endcode

Alternatively, one can directly assign new pointers to the closest
centers, without bothering with a ::VlKMeans object.

There are several considerations that may impact the performance of
KMeans. First, since K-means is usually based local optimization
algorithm, the **initialization method** is important. The following
initialization methods are supported:

Method         | Function                                | Description
---------------|-----------------------------------------|-----------------------------------------------
Random samples | ::vl_kmeans_init_centers_with_rand_data | Random data points
K-means++      | ::vl_kmeans_init_centers_plus_plus      | Random selection biased towards diversity
Custom         | ::vl_kmeans_set_centers                 | Choose centers (useful to run quantization only)

See @ref kmeans-init for further details. The initialization methods
use a randomized selection of the data points; the random number
generator init is controlled by ::vl_rand_init.

The second important choice is the **optimization algorithm**. The
following optimization algorithms are supported:

Algorithm   | Symbol           | See               | Description
------------|------------------|-------------------|-----------------------------------------------
Lloyd       | ::VlKMeansLloyd  | @ref kmeans-lloyd | Alternate EM-style optimization
Elkan       | ::VlKMeansElkan  | @ref kmeans-elkan | A speedup using triangular inequalities
ANN         | ::VlKMeansANN    | @ref kmeans-ann   | A speedup using approximated nearest neighbors

See the relative sections for further details. These algorithm are
iterative, and stop when either a **maximum number of iterations**
(::vl_kmeans_set_max_num_iterations) is reached, or when the energy
changes sufficiently slowly in one iteration (::vl_kmeans_set_min_energy_variation).


All the three algorithms support multithreaded computations. The number
of threads used is usually controlled globally by ::vl_set_num_threads.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page kmeans-fundamentals K-means fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Given $n$ points $\bx_1,\dots,\bx_n \in \real^d$, the goal of K-means
is find $K$ `centers` $\bc_1,\dots,\bc_m \in \real^d$ and
`assignments` $q_1,\dots,q_n \in \{1,\dots,K\}$ of the points to the
centers such that the sum of distances

\[
 E(\bc_1,\dots,\bc_k,q_1,\dots,q_n)
 = \sum_{i=1}^n \|\bx_i - \bc_{q_i} \|_p^p
\]

is minimized. $K$-means is obtained for the case $p=2$ ($l^2$ norm),
because in this case the optimal centers are the means of the input
vectors assigned to them. Here the generalization $p=1$ ($l^1$ norm)
will also be considered.

Up to normalization, the K-means objective $E$ is also the average
reconstruction error if the original points are approximated with the
cluster centers. Thus K-means is used not only to group the input
points into cluster, but also to `quantize` their values.

K-means is widely used in computer vision, for example in the
construction of vocabularies of visual features (visual words). In
these applications the number $n$ of points to cluster and/or the
number $K$ of clusters is often large. Unfortunately, minimizing the
objective $E$ is in general a difficult combinatorial problem, so
locally optimal or approximated solutions are sought instead.

The basic K-means algorithm alternate between re-estimating the
centers and the assignments (@ref kmeans-lloyd). Combined with a good
initialization strategy (@ref kmeans-init) and, potentially, by
re-running the optimization from a number of randomized starting
states, this algorithm may attain satisfactory solutions in practice.

However, despite its simplicity, Lloyd's algorithm is often too slow.
A good replacement is Elkan's algorithm (@ref kmeans-elkan), which
uses the triangular inequality to cut down significantly the cost of
Lloyd's algorithm. Since this algorithm is otherwise equivalent, it
should often be preferred.

For very large problems (millions of point to clusters and hundreds,
thousands, or more clusters to find), even Elkan's algorithm is not
sufficiently fast. In these cases, one can resort to a variant of
Lloyd's algorithm that uses an approximated nearest neighbors routine
(@ref kmeans-ann).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section kmeans-init Initialization methods
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

All the $K$-means algorithms considered here find locally optimal
solutions; as such the way they are initialized is important. @ref
kmeans.h supports the following initialization algorithms:

@par Random data samples

The simplest initialization method is to sample $K$ points at random
from the input data and use them as initial values for the cluster
centers.

@par K-means++

@cite{arthur07k-means} proposes a randomized initialization of the
centers which improves upon random selection. The first center $\bc_1$
is selected at random from the data points $\bx_1, \dots, \bx_n $ and
the distance from this center to all points $\|\bx_i - \bc_1\|_p^p$ is
computed. Then the second center $\bc_2$ is selected at random from
the data points with probability proportional to the distance. The
procedure is repeated to obtain the other centers by using the minimum
distance to the centers collected so far.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section kmeans-lloyd Lloyd's algorithm
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The most common K-means method is Lloyd's algorithm
@cite{lloyd82least}. This algorithm is based on the observation that,
while jointly optimizing clusters and assignment is difficult,
optimizing one given the other is easy. Lloyd's algorithm alternates
the steps:

1. **Quantization.** Each point $\bx_i$ is reassigned to the center
   $\bc_{q_j}$ closer to it. This requires finding for each point the
   closest among $K$ other points, which is potentially slow.
2. **Center estimation.** Each center $\bc_q$ is updated to minimize
   its average distances to the points assigned to it. It is easy to
   show that the best center is the mean or median of the points,
   respectively if the $l^2$ or $l^1$ norm is considered.

A naive implementation of the assignment step requires $O(dnK)$
operations, where $d$ is the dimensionality of the data, $n$ the
number of data points, and $K$ the number of centers. Updating the
centers is much cheaper: $O(dn)$ operations suffice to compute the $K$
means and a slightly higher cost is required for the medians. Clearly,
the bottleneck is the assignment computation, and this is what the
other K-means algorithm try to improve.

During the iterations, it can happen that a cluster becomes empty. In
this case, K-means automatically **&ldquo;restarts&rdquo; the
cluster** center by selecting a training point at random.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section kmeans-elkan Elkan's algorithm
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Elkan's algorithm @cite{elkan03using} is a variation of Lloyd
alternate optimization algorithm (@ref kmeans-lloyd) that uses the
triangular inequality to avoid many distance calculations when
assigning points to clusters. While much faster than Lloyd, Elkan's
method uses storage proportional to the umber of clusters by data
points, which makes it unpractical for a very large number of
clusters.

The idea of this algorithm is that, if a center update does not move
them much, then most of the point-to-center computations can be
avoided when the point-to-center assignments are recomputed. To detect
which distances need evaluation, the triangular inequality is used to
lower and upper bound distances after a center update.

Elkan algorithms uses two key observations. First, one has

\[
\|\bx_i - \bc_{q_i}\|_p \leq \|\bc - \bc_{q_i}\|_p / 2
\quad\Rightarrow\quad
\|\bx_i - \bc_{q_i}\|_p \leq \|\bx_i - \bc\|_p.
\]

Thus if the distance between $\bx_i$ and its current center
$\bc_{q_i}$ is less than half the distance of the center $\bc_{q_i}$
to another center $\bc$, then $\bc$ can be skipped when the new
assignment for $\bx_i$ is searched. Checking this requires keeping
track of all the inter-center distances, but centers are typically a
small fraction of the training data, so overall this can be a
significant saving. In particular, if this condition is satisfied for
all the centers $\bc \not= \bc_{q_i}$, the point $\bx_i$ can be
skipped completely. Furthermore, the condition can be tested also
based on an upper bound $UB_i$ of $\|\bx_i - \bc_{q_i}\|_p$.

Second, if a center $\bc$ is updated to $\hat{\bc}$, then the new
distance from $\bx$ to $\hat{\bc}$ is bounded from below and above by

\[
\|\bx - \bc\|_p - \|bc - \hat\bc\|_p
\leq
\|\bx - \hat{\bc}\|_p
\leq
\|\bx - \hat{\bc}\|_p + \|\bc + \hat{\bc}\|_p.
\]

This allows to maintain an upper bound on the distance of $\bx_i$ to
its current center $\bc_{q_i}$ and a lower bound to any other center
$\bc$:

@f{align*}
  UB_i      & \leftarrow UB_i + \|\bc_{q_i} - \hat{\bc}_{q_i} \|_p \\
  LB_i(\bc) & \leftarrow LB_i(\bc) - \|\bc -\hat \bc\|_p.
@f}

Thus the K-means algorithm becomes:

1.  **Initialization.** Compute $LB_i(\bc) = \|\bx_i -\hat \bc\|_p$ for
    all points and centers.  Find the current assignments $q_i$ and
    bounds $UB_i$ by finding the closest centers to each point: $UB_i =
    \min_{\bc} LB_i(\bc)$.
2.  **Center estimation.**
    1. Recompute all the centers based on the new means; call the updated
       version $\hat{\bc}$.
    2. Update all the bounds based on the distance $\|\bc - \hat\bc\|_p$
       as explained above.
    3. Set $\bc \leftarrow \hat\bc$ for all the centers and go to the next
       iteration.
3.  **Quantization.**
    1. Skip any point $\bx_i$ such that $UB_i \leq \frac{1}{2} \|\bc_{q_i} - \bc\|_p$
       for all centers $\bc \not= \bc_{q_i}$.
    2. For each remaining point $\bx_i$ and center $\bc \not= \bc_{q_i}$:
       1. Skip $\bc$ if
          \[
           UB_i \leq \frac{1}{2} \| \bc_{q_i} - \bc \|
           \quad\text{or}\quad
           UB_i \leq LB_i(\bc).
           \]
          The first condition reflects the first observation above; the
          second uses the bounds to decide if $\bc$ can be closer than the
          current center $\bc_{q_i}$ to the point $\bx_i$. If the center
          cannot be skipped, continue as follows.
       3. Skip $\bc$ if the condition above is satisfied after making the
          upper bound tight:
          \[
          UB_i = LB_i(\bc_{q_i}) = \| \bx_i - \bc_{q_i} \|_p.
          \]
          Note that the latter calculation can be done only once for $\bx_i$.
          If the center cannot be skipped still, continue as follows.
       4. Tighten the lower bound too:
          \[
          LB_i(\bc) = \| \bx_i - \bc \|_p.
          \]
          At this point both $UB_i$ and $LB_i(\bc)$ are tight. If $LB_i <
          UB_i$, then the point $\bx_i$ should be reassigned to
          $\bc$. Update $q_i$ to the index of center $\bc$ and reset $UB_i
          = LB_i(\bc)$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section kmeans-ann ANN algorithm
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The *Approximate Nearest Neighbor* (ANN) K-means algorithm
@cite{beis97shape} @cite{silpa-anan08optimised} @cite{muja09fast} is a
variant of Lloyd's algorithm (@ref kmeans-lloyd) uses a best-bin-first
randomized KD-tree algorithm to approximately (and quickly) find the
closest cluster center to each point. The KD-tree implementation is
based on @ref kdtree.

The algorithm can be summarized as follows:

1. **Quantization.** Each point $\bx_i$ is reassigned to the center
   $\bc_{q_j}$ closer to it. This starts by indexing the $K$ centers
   by a KD-tree and then using the latter to quickly find the closest
   center for every training point. The search is approximated to
   further improve speed. This opens up the possibility that a data
   point may receive an assignment that is *worse* than the current
   one. This is avoided by checking that the new assignment estimated
   by using ANN is an improvement; otherwise the old assignment is
   kept.
2. **Center estimation.** Each center $\bc_q$ is updated to minimize
   its average distances to the points assigned to it. It is easy to
   show that the best center is the mean or median of the points,
   respectively if the $l^2$ or $l^1$ norm is considered.

The key is to trade-off carefully the speedup obtained by using the
ANN algorithm and the loss in accuracy when retrieving neighbors.  Due
to the curse of dimensionality, KD-trees become less effective for
higher dimensional data, so that the search cost, which in the best
case is logarithmic with this data structure, may become effectively
linear. This is somehow mitigated by the fact that new a new KD-tree
is computed at each iteration, reducing the likelihood that points may
get stuck with sub-optimal assignments.

Experiments with the quantization of 128-dimensional SIFT features
show that the ANN algorithm may use one quarter of the comparisons of
Elkan's while retaining a similar solution accuracy.

*/

#include "kmeans.h"
#include "generic.h"
#include "mathop.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================================================================ */
#ifndef VL_KMEANS_INSTANTIATING


/** ------------------------------------------------------------------
 ** @brief Reset state
 **
 ** The function reset the state of the KMeans object. It deletes
 ** any stored centers, releasing the corresponding memory. This
 ** cancels the effect of seeding or setting the centers, but
 ** does not change the other configuration parameters.
 **/

VL_EXPORT void
vl_kmeans_reset (VlKMeans * self)
{
  self->numCenters = 0 ;
  self->dimension = 0 ;

  if (self->centers) vl_free(self->centers) ;
  if (self->centerDistances) vl_free(self->centerDistances) ;

  self->centers = NULL ;
  self->centerDistances = NULL ;
}

/** ------------------------------------------------------------------
 ** @brief Create a new KMeans object
 ** @param dataType type of data (::VL_TYPE_FLOAT or ::VL_TYPE_DOUBLE)
 ** @param distance distance.
 ** @return new KMeans object instance.
**/

VL_EXPORT VlKMeans *
vl_kmeans_new (vl_type dataType,
               VlVectorComparisonType distance)
{
  VlKMeans * self = vl_calloc(1, sizeof(VlKMeans)) ;

  self->algorithm = VlKMeansLloyd ;
  self->distance = distance ;
  self->dataType = dataType ;
  self->verbosity = 0 ;
  self->maxNumIterations = 100 ;
  self->minEnergyVariation = 1e-4 ;
  self->numRepetitions = 1 ;
  self->centers = NULL ;
  self->centerDistances = NULL ;
  self->numTrees = 3;
  self->maxNumComparisons = 100;

  vl_kmeans_reset (self) ;
  return self ;
}

/** ------------------------------------------------------------------
 ** @brief Create a new KMeans object by copy
 ** @param kmeans KMeans object to copy.
 ** @return new copy.
 **/

VL_EXPORT VlKMeans *
vl_kmeans_new_copy (VlKMeans const * kmeans)
{
  VlKMeans * self = vl_malloc(sizeof(VlKMeans)) ;

  self->algorithm = kmeans->algorithm ;
  self->distance = kmeans->distance ;
  self->dataType = kmeans->dataType ;

  self->verbosity = kmeans->verbosity ;
  self->maxNumIterations = kmeans->maxNumIterations ;
  self->numRepetitions = kmeans->numRepetitions ;

  self->dimension = kmeans->dimension ;
  self->numCenters = kmeans->numCenters ;
  self->centers = NULL ;
  self->centerDistances = NULL ;

  self->numTrees = kmeans->numTrees;
  self->maxNumComparisons = kmeans->maxNumComparisons;

  if (kmeans->centers) {
    vl_size dataSize = vl_get_type_size(self->dataType) * self->dimension * self->numCenters ;
    self->centers = vl_malloc(dataSize) ;
    memcpy (self->centers, kmeans->centers, dataSize) ;
  }

  if (kmeans->centerDistances) {
    vl_size dataSize = vl_get_type_size(self->dataType) * self->numCenters * self->numCenters ;
    self->centerDistances = vl_malloc(dataSize) ;
    memcpy (self->centerDistances, kmeans->centerDistances, dataSize) ;
  }

  return self ;
}

/** ------------------------------------------------------------------
 ** @brief Deletes a KMeans object
 ** @param self KMeans object instance.
 **
 ** The function deletes the KMeans object instance created
 ** by ::vl_kmeans_new.
 **/

VL_EXPORT void
vl_kmeans_delete (VlKMeans * self)
{
  vl_kmeans_reset (self) ;
  vl_free (self) ;
}

/* an helper structure */
typedef struct _VlKMeansSortWrapper {
  vl_uint32 * permutation ;
  void const * data ;
  vl_size stride ;
} VlKMeansSortWrapper ;


/* ---------------------------------------------------------------- */
/* Instantiate shuffle algorithm */

#define VL_SHUFFLE_type vl_uindex
#define VL_SHUFFLE_prefix _vl_kmeans
#include "shuffle-def.h"

/* #ifdef VL_KMEANS_INSTANTITATING */
#endif

/* ================================================================ */
#ifdef VL_KMEANS_INSTANTIATING

/* ---------------------------------------------------------------- */
/*                                                      Set centers */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_kmeans_set_centers_, SFX)
(VlKMeans * self,
 TYPE const * centers,
 vl_size dimension,
 vl_size numCenters)
{
  self->dimension = dimension ;
  self->numCenters = numCenters ;
  self->centers = vl_malloc (sizeof(TYPE) * dimension * numCenters) ;
  memcpy ((TYPE*)self->centers, centers,
          sizeof(TYPE) * dimension * numCenters) ;
}

/* ---------------------------------------------------------------- */
/*                                                   Random seeding */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_kmeans_init_centers_with_rand_data_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size dimension,
 vl_size numData,
 vl_size numCenters)
{
  vl_uindex i, j, k ;
  VlRand * rand = vl_get_rand () ;

  self->dimension = dimension ;
  self->numCenters = numCenters ;
  self->centers = vl_malloc (sizeof(TYPE) * dimension * numCenters) ;

  {
    vl_uindex * perm = vl_malloc (sizeof(vl_uindex) * numData) ;
#if (FLT == VL_TYPE_FLOAT)
    VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
    VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif
    TYPE * distances = vl_malloc (sizeof(TYPE) * numCenters) ;

    /* get a random permutation of the data point */
    for (i = 0 ; i < numData ; ++i) perm[i] = i ;
    _vl_kmeans_shuffle (perm, numData, rand) ;

    for (k = 0, i = 0 ; k < numCenters ; ++ i) {

      /* compare the next data point to all centers collected so far
       to detect duplicates (if there are enough left)
       */
      if (numCenters - k < numData - i) {
        vl_bool duplicateDetected = VL_FALSE ;
        VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)(distances,
            dimension,
            data + dimension * perm[i], 1,
            (TYPE*)self->centers, k,
            distFn) ;
        for (j = 0 ; j < k ; ++j) {
          duplicateDetected |= (distances[j] == 0) ;
        }
        if (duplicateDetected) continue ;
      }

      /* ok, it is not a duplicate so we can accept it! */
      memcpy ((TYPE*)self->centers + dimension * k,
              data + dimension * perm[i],
              sizeof(TYPE) * dimension) ;
      k ++ ;
    }
    vl_free(distances) ;
    vl_free(perm) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                 kmeans++ seeding */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_kmeans_init_centers_plus_plus_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size dimension,
 vl_size numData,
 vl_size numCenters)
{
  vl_uindex x, c ;
  VlRand * rand = vl_get_rand () ;
  TYPE * distances = vl_malloc (sizeof(TYPE) * numData) ;
  TYPE * minDistances = vl_malloc (sizeof(TYPE) * numData) ;
#if (FLT == VL_TYPE_FLOAT)
  VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
  VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif

  self->dimension = dimension ;
  self->numCenters = numCenters ;
  self->centers = vl_malloc (sizeof(TYPE) * dimension * numCenters) ;

  for (x = 0 ; x < numData ; ++x) {
    minDistances[x] = (TYPE) VL_INFINITY_D ;
  }

  /* select the first point at random */
  x = vl_rand_uindex (rand, numData) ;
  c = 0 ;
  while (1) {
    TYPE energy = 0 ;
    TYPE acc = 0 ;
    TYPE thresh = (TYPE) vl_rand_real1 (rand) ;

    memcpy ((TYPE*)self->centers + c * dimension,
            data + x * dimension,
            sizeof(TYPE) * dimension) ;

    c ++ ;
    if (c == numCenters) break ;

    VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)
    (distances,
     dimension,
     (TYPE*)self->centers + (c - 1) * dimension, 1,
     data, numData,
     distFn) ;

    for (x = 0 ; x < numData ; ++x) {
      minDistances[x] = VL_MIN(minDistances[x], distances[x]) ;
      energy += minDistances[x] ;
    }

    for (x = 0 ; x < numData - 1 ; ++x) {
      acc += minDistances[x] ;
      if (acc >= thresh * energy) break ;
    }
  }

  vl_free(distances) ;
  vl_free(minDistances) ;
}

/* ---------------------------------------------------------------- */
/*                                                     Quantization */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_kmeans_quantize_, SFX)
(VlKMeans * self,
 vl_uint32 * assignments,
 TYPE * distances,
 TYPE const * data,
 vl_size numData)
{
  vl_index i ;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
  VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif

#ifdef _OPENMP
#pragma omp parallel default(none) \
            shared(self, distances, assignments, numData, distFn, data) \
            num_threads(vl_get_max_threads())
#endif
  {
    /* vl_malloc cannot be used here if mapped to MATLAB malloc */
    TYPE * distanceToCenters = malloc(sizeof(TYPE) * self->numCenters) ;

#ifdef _OPENMP
#pragma omp for
#endif
    for (i = 0 ; i < (signed)numData ; ++i) {
      vl_uindex k ;
      TYPE bestDistance = (TYPE) VL_INFINITY_D ;
      VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)(distanceToCenters,
                                                            self->dimension,
                                                            data + self->dimension * i, 1,
                                                            (TYPE*)self->centers, self->numCenters,
                                                            distFn) ;
      for (k = 0 ; k < self->numCenters ; ++k) {
        if (distanceToCenters[k] < bestDistance) {
          bestDistance = distanceToCenters[k] ;
          assignments[i] = (vl_uint32)k ;
        }
      }
      if (distances) distances[i] = bestDistance ;
    }

    free(distanceToCenters) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                 ANN quantization */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_kmeans_quantize_ann_, SFX)
(VlKMeans * self,
 vl_uint32 * assignments,
 TYPE * distances,
 TYPE const * data,
 vl_size numData,
 vl_bool update)
{
#if (FLT == VL_TYPE_FLOAT)
  VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
  VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif

  VlKDForest * forest = vl_kdforest_new(self->dataType,self->dimension,self->numTrees, self->distance) ;
  vl_kdforest_set_max_num_comparisons(forest,self->maxNumComparisons);
  vl_kdforest_set_thresholding_method(forest,VL_KDTREE_MEDIAN);
  vl_kdforest_build(forest,self->numCenters,self->centers);

#ifdef _OPENMP
#pragma omp parallel default(none) \
  num_threads(vl_get_max_threads()) \
  shared(self, forest, update, assignments, distances, data, numData, distFn)
#endif
  {
    VlKDForestNeighbor neighbor ;
    VlKDForestSearcher * searcher ;
    vl_index x;

#ifdef _OPENMP
#pragma omp critical
#endif
    searcher = vl_kdforest_new_searcher (forest) ;

#ifdef _OPENMP
#pragma omp for
#endif
    for(x = 0 ; x < (signed)numData ; ++x) {
      vl_kdforestsearcher_query (searcher, &neighbor, 1, (TYPE const *) (data + x*self->dimension));

      if (distances) {
        if(!update) {
          distances[x] = (TYPE) neighbor.distance;
          assignments[x] = (vl_uint32) neighbor.index ;
        } else {
          TYPE prevDist = (TYPE) distFn(self->dimension,
                                        data + self->dimension * x,
                                        (TYPE*)self->centers + self->dimension *assignments[x]);
          if (prevDist > (TYPE) neighbor.distance) {
            distances[x] = (TYPE) neighbor.distance ;
            assignments[x] = (vl_uint32) neighbor.index ;
          } else {
            distances[x] = prevDist ;
          }
        }
      } else {
        assignments[x] = (vl_uint32) neighbor.index ;
      }
    } /* end for */
  } /* end of parallel region */

  vl_kdforest_delete(forest);
}

/* ---------------------------------------------------------------- */
/*                                                 Helper functions */
/* ---------------------------------------------------------------- */

/* The sorting routine is used to find increasing permutation of each
 * data dimension. This is used to quickly find the median for l1
 * distance clustering. */

VL_INLINE TYPE
VL_XCAT3(_vl_kmeans_, SFX, _qsort_cmp)
(VlKMeansSortWrapper * array, vl_uindex indexA, vl_uindex indexB)
{
  return
    ((TYPE*)array->data) [array->permutation[indexA] * array->stride]
    -
    ((TYPE*)array->data) [array->permutation[indexB] * array->stride] ;
}

VL_INLINE void
VL_XCAT3(_vl_kmeans_, SFX, _qsort_swap)
(VlKMeansSortWrapper * array, vl_uindex indexA, vl_uindex indexB)
{
  vl_uint32 tmp = array->permutation[indexA] ;
  array->permutation[indexA] = array->permutation[indexB] ;
  array->permutation[indexB] = tmp ;
}

#define VL_QSORT_prefix  VL_XCAT3(_vl_kmeans_, SFX, _qsort)
#define VL_QSORT_array   VlKMeansSortWrapper*
#define VL_QSORT_cmp     VL_XCAT3(_vl_kmeans_, SFX, _qsort_cmp)
#define VL_QSORT_swap    VL_XCAT3(_vl_kmeans_, SFX, _qsort_swap)
#include "qsort-def.h"

static void
VL_XCAT(_vl_kmeans_sort_data_helper_, SFX)
(VlKMeans * self, vl_uint32 * permutations, TYPE const * data, vl_size numData)
{
  vl_uindex d, x ;

  for (d = 0 ; d < self->dimension ; ++d) {
    VlKMeansSortWrapper array ;
    array.permutation = permutations + d * numData ;
    array.data = data + d ;
    array.stride = self->dimension ;
    for (x = 0 ; x < numData ; ++x) {
      array.permutation[x] = (vl_uint32)x ;
    }
    VL_XCAT3(_vl_kmeans_, SFX, _qsort_sort)(&array, numData) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                 Lloyd refinement */
/* ---------------------------------------------------------------- */

static double
VL_XCAT(_vl_kmeans_refine_centers_lloyd_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size numData)
{
  vl_size c, d, x, iteration ;
  double previousEnergy = VL_INFINITY_D ;
  double initialEnergy = VL_INFINITY_D ;
  double energy ;
  TYPE * distances = vl_malloc (sizeof(TYPE) * numData) ;

  vl_uint32 * assignments = vl_malloc (sizeof(vl_uint32) * numData) ;
  vl_size * clusterMasses = vl_malloc (sizeof(vl_size) * numData) ;
  vl_uint32 * permutations = NULL ;
  vl_size * numSeenSoFar = NULL ;
  VlRand * rand = vl_get_rand () ;
  vl_size totNumRestartedCenters = 0 ;
  vl_size numRestartedCenters = 0 ;

  if (self->distance == VlDistanceL1) {
    permutations = vl_malloc(sizeof(vl_uint32) * numData * self->dimension) ;
    numSeenSoFar = vl_malloc(sizeof(vl_size) * self->numCenters) ;
    VL_XCAT(_vl_kmeans_sort_data_helper_, SFX)(self, permutations, data, numData) ;
  }

  for (energy = VL_INFINITY_D,
       iteration = 0;
       1 ;
       ++ iteration) {

    /* assign data to cluters */
    VL_XCAT(_vl_kmeans_quantize_, SFX)(self, assignments, distances, data, numData) ;

    /* compute energy */
    energy = 0 ;
    for (x = 0 ; x < numData ; ++x) energy += distances[x] ;
    if (self->verbosity) {
      VL_PRINTF("kmeans: Lloyd iter %d: energy = %g\n", iteration,
                energy) ;
    }

    /* check termination conditions */
    if (iteration >= self->maxNumIterations) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: Lloyd terminating because maximum number of iterations reached\n") ;
      }
      break ;
    }
    if (energy == previousEnergy) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: Lloyd terminating because the algorithm fully converged\n") ;
      }
      break ;
    }
    
    if (iteration == 0) {
      initialEnergy = energy ;
    } else {
      double eps = (previousEnergy - energy) / (initialEnergy - energy) ;
      if (eps < self->minEnergyVariation) {
        if (self->verbosity) {
          VL_PRINTF("kmeans: ANN terminating because the energy relative variation was less than %f\n", self->minEnergyVariation) ;
        }
        break ;
      }
    }
    
    /* begin next iteration */
    previousEnergy = energy ;

    /* update clusters */
    memset(clusterMasses, 0, sizeof(vl_size) * numData) ;
    for (x = 0 ; x < numData ; ++x) {
      clusterMasses[assignments[x]] ++ ;
    }

    numRestartedCenters = 0 ;
    switch (self->distance) {
      case VlDistanceL2:
        memset(self->centers, 0, sizeof(TYPE) * self->dimension * self->numCenters) ;
        for (x = 0 ; x < numData ; ++x) {
          TYPE * cpt = (TYPE*)self->centers + assignments[x] * self->dimension ;
          TYPE const * xpt = data + x * self->dimension ;
          for (d = 0 ; d < self->dimension ; ++d) {
            cpt[d] += xpt[d] ;
          }
        }
        for (c = 0 ; c < self->numCenters ; ++c) {
          TYPE * cpt = (TYPE*)self->centers + c * self->dimension ;
          if (clusterMasses[c] > 0) {
            TYPE mass = clusterMasses[c] ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] /= mass ;
            }
          } else {
            vl_uindex x = vl_rand_uindex(rand, numData) ;
            numRestartedCenters ++ ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] = data[x * self->dimension + d] ;
            }
          }
        }
        break ;
      case VlDistanceL1:
        for (d = 0 ; d < self->dimension ; ++d) {
          vl_uint32 * perm = permutations + d * numData ;
          memset(numSeenSoFar, 0, sizeof(vl_size) * self->numCenters) ;
          for (x = 0; x < numData ; ++x) {
            c = assignments[perm[x]] ;
            if (2 * numSeenSoFar[c] < clusterMasses[c]) {
              ((TYPE*)self->centers) [d + c * self->dimension] =
                data [d + perm[x] * self->dimension] ;
            }
            numSeenSoFar[c] ++ ;
          }
          /* restart the centers as required  */
          for (c = 0 ; c < self->numCenters ; ++c) {
            if (clusterMasses[c] == 0) {
              TYPE * cpt = (TYPE*)self->centers + c * self->dimension ;
              vl_uindex x = vl_rand_uindex(rand, numData) ;
              numRestartedCenters ++ ;
              for (d = 0 ; d < self->dimension ; ++d) {
                cpt[d] = data[x * self->dimension + d] ;
              }
            }
          }
        }
        break ;
      default:
        abort();
    } /* done compute centers */

    totNumRestartedCenters += numRestartedCenters ;
    if (self->verbosity && numRestartedCenters) {
      VL_PRINTF("kmeans: Lloyd iter %d: restarted %d centers\n", iteration,
                numRestartedCenters) ;
    }
  } /* next Lloyd iteration */

  if (permutations) {
    vl_free(permutations) ;
  }
  if (numSeenSoFar) {
    vl_free(numSeenSoFar) ;
  }
  vl_free(distances) ;
  vl_free(assignments) ;
  vl_free(clusterMasses) ;
  return energy ;
}

static double
VL_XCAT(_vl_kmeans_update_center_distances_, SFX)
(VlKMeans * self)
{
#if (FLT == VL_TYPE_FLOAT)
  VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
  VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif

  if (! self->centerDistances) {
    self->centerDistances = vl_malloc (sizeof(TYPE) *
                                       self->numCenters *
                                       self->numCenters) ;
  }
  VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)(self->centerDistances,
      self->dimension,
      self->centers, self->numCenters,
      NULL, 0,
      distFn) ;
  return self->numCenters * (self->numCenters - 1) / 2 ;
}

static double
VL_XCAT(_vl_kmeans_refine_centers_ann_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size numData)
{
  vl_size c, d, x, iteration ;
  double initialEnergy = VL_INFINITY_D ;
  double previousEnergy = VL_INFINITY_D ;
  double energy ;

  vl_uint32 * permutations = NULL ;
  vl_size * numSeenSoFar = NULL ;
  VlRand * rand = vl_get_rand () ;
  vl_size totNumRestartedCenters = 0 ;
  vl_size numRestartedCenters = 0 ;

  vl_uint32 * assignments = vl_malloc (sizeof(vl_uint32) * numData) ;
  vl_size * clusterMasses = vl_malloc (sizeof(vl_size) * numData) ;
  TYPE * distances = vl_malloc (sizeof(TYPE) * numData) ;

  if (self->distance == VlDistanceL1) {
    permutations = vl_malloc(sizeof(vl_uint32) * numData * self->dimension) ;
    numSeenSoFar = vl_malloc(sizeof(vl_size) * self->numCenters) ;
    VL_XCAT(_vl_kmeans_sort_data_helper_, SFX)(self, permutations, data, numData) ;
  }

  for (energy = VL_INFINITY_D,
       iteration = 0;
       1 ;
       ++ iteration) {

    /* assign data to cluters */
    VL_XCAT(_vl_kmeans_quantize_ann_, SFX)(self, assignments, distances, data, numData, iteration > 0) ;

    /* compute energy */
    energy = 0 ;
    for (x = 0 ; x < numData ; ++x) energy += distances[x] ;
    if (self->verbosity) {
      VL_PRINTF("kmeans: ANN iter %d: energy = %g\n", iteration,
                energy) ;
    }

    /* check termination conditions */
    if (iteration >= self->maxNumIterations) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: ANN terminating because the maximum number of iterations has been reached\n") ;
      }
      break ;
    }
    if (energy == previousEnergy) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: ANN terminating because the algorithm fully converged\n") ;
      }
      break ;
    }
    
    if (iteration == 0) {
      initialEnergy = energy ;
    } else {
      double eps = (previousEnergy - energy) / (initialEnergy - energy) ;
      if (eps < self->minEnergyVariation) {
        if (self->verbosity) {
          VL_PRINTF("kmeans: ANN terminating because the energy relative variation was less than %f\n", self->minEnergyVariation) ;
        }
        break ;
      }
    }

    /* begin next iteration */
    previousEnergy = energy ;

    /* update clusters */
    memset(clusterMasses, 0, sizeof(vl_size) * numData) ;
    for (x = 0 ; x < numData ; ++x) {
      clusterMasses[assignments[x]] ++ ;
    }

    numRestartedCenters = 0 ;
    switch (self->distance) {
      case VlDistanceL2:
        memset(self->centers, 0, sizeof(TYPE) * self->dimension * self->numCenters) ;
        for (x = 0 ; x < numData ; ++x) {
          TYPE * cpt = (TYPE*)self->centers + assignments[x] * self->dimension ;
          TYPE const * xpt = data + x * self->dimension ;
          for (d = 0 ; d < self->dimension ; ++d) {
            cpt[d] += xpt[d] ;
          }
        }
        for (c = 0 ; c < self->numCenters ; ++c) {
          TYPE * cpt = (TYPE*)self->centers + c * self->dimension ;
          if (clusterMasses[c] > 0) {
            TYPE mass = clusterMasses[c] ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] /= mass ;
            }
          } else {
            vl_uindex x = vl_rand_uindex(rand, numData) ;
            numRestartedCenters ++ ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] = data[x * self->dimension + d] ;
            }
          }
        }
        break ;
      case VlDistanceL1:
        for (d = 0 ; d < self->dimension ; ++d) {
          vl_uint32 * perm = permutations + d * numData ;
          memset(numSeenSoFar, 0, sizeof(vl_size) * self->numCenters) ;
          for (x = 0; x < numData ; ++x) {
            c = assignments[perm[x]] ;
            if (2 * numSeenSoFar[c] < clusterMasses[c]) {
              ((TYPE*)self->centers) [d + c * self->dimension] =
                data [d + perm[x] * self->dimension] ;
            }
            numSeenSoFar[c] ++ ;
          }
          /* restart the centers as required  */
          for (c = 0 ; c < self->numCenters ; ++c) {
            if (clusterMasses[c] == 0) {
              TYPE * cpt = (TYPE*)self->centers + c * self->dimension ;
              vl_uindex x = vl_rand_uindex(rand, numData) ;
              numRestartedCenters ++ ;
              for (d = 0 ; d < self->dimension ; ++d) {
                cpt[d] = data[x * self->dimension + d] ;
              }
            }
          }
        }
        break ;
      default:
        VL_PRINT("bad distance set: %d\n",self->distance);
        abort();
    } /* done compute centers */

    totNumRestartedCenters += numRestartedCenters ;
    if (self->verbosity && numRestartedCenters) {
      VL_PRINTF("kmeans: ANN iter %d: restarted %d centers\n", iteration,
                numRestartedCenters) ;
    }
  }

  if (permutations) {
    vl_free(permutations) ;
  }
  if (numSeenSoFar) {
    vl_free(numSeenSoFar) ;
  }

  vl_free(distances) ;
  vl_free(assignments) ;
  vl_free(clusterMasses) ;
  return energy ;
}

/* ---------------------------------------------------------------- */
/*                                                 Elkan refinement */
/* ---------------------------------------------------------------- */

static double
VL_XCAT(_vl_kmeans_refine_centers_elkan_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size numData)
{
  vl_size d, iteration ;
  vl_index x ;
  vl_uint32 c, j ;
  vl_bool allDone ;
  TYPE * distances = vl_malloc (sizeof(TYPE) * numData) ;
  vl_uint32 * assignments = vl_malloc (sizeof(vl_uint32) * numData) ;
  vl_size * clusterMasses = vl_malloc (sizeof(vl_size) * numData) ;
  VlRand * rand = vl_get_rand () ;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVectorComparisonFunction distFn = vl_get_vector_comparison_function_f(self->distance) ;
#else
  VlDoubleVectorComparisonFunction distFn = vl_get_vector_comparison_function_d(self->distance) ;
#endif

  TYPE * nextCenterDistances = vl_malloc (sizeof(TYPE) * self->numCenters) ;
  TYPE * pointToClosestCenterUB = vl_malloc (sizeof(TYPE) * numData) ;
  vl_bool * pointToClosestCenterUBIsStrict = vl_malloc (sizeof(vl_bool) * numData) ;
  TYPE * pointToCenterLB = vl_malloc (sizeof(TYPE) * numData * self->numCenters) ;
  TYPE * newCenters = vl_malloc(sizeof(TYPE) * self->dimension * self->numCenters) ;
  TYPE * centerToNewCenterDistances = vl_malloc (sizeof(TYPE) * self->numCenters) ;

  vl_uint32 * permutations = NULL ;
  vl_size * numSeenSoFar = NULL ;

  double energy ;

  vl_size totDistanceComputationsToInit = 0 ;
  vl_size totDistanceComputationsToRefreshUB = 0 ;
  vl_size totDistanceComputationsToRefreshLB = 0 ;
  vl_size totDistanceComputationsToRefreshCenterDistances = 0 ;
  vl_size totDistanceComputationsToNewCenters = 0 ;
  vl_size totDistanceComputationsToFinalize = 0 ;
  vl_size totNumRestartedCenters = 0 ;

  if (self->distance == VlDistanceL1) {
    permutations = vl_malloc(sizeof(vl_uint32) * numData * self->dimension) ;
    numSeenSoFar = vl_malloc(sizeof(vl_size) * self->numCenters) ;
    VL_XCAT(_vl_kmeans_sort_data_helper_, SFX)(self, permutations, data, numData) ;
  }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*                          Initialization                        */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  /* An iteration is: get_new_centers + reassign + get_energy.
   This counts as iteration 0, where get_new_centers is assumed
   to be performed before calling the train function by
   the initialization function */

  /* update distances between centers */
  totDistanceComputationsToInit +=
  VL_XCAT(_vl_kmeans_update_center_distances_, SFX)(self) ;

  /* assigmen points to the initial centers and initialize bounds */
  memset(pointToCenterLB, 0, sizeof(TYPE) * self->numCenters *  numData) ;
  for (x = 0 ; x < (signed)numData ; ++x) {
    TYPE distance ;

    /* do the first center */
    assignments[x] = 0 ;
    distance = distFn(self->dimension,
                      data + x * self->dimension,
                      (TYPE*)self->centers + 0) ;
    pointToClosestCenterUB[x] = distance ;
    pointToClosestCenterUBIsStrict[x] = VL_TRUE ;
    pointToCenterLB[0 + x * self->numCenters] = distance ;
    totDistanceComputationsToInit += 1 ;

    /* do other centers */
    for (c = 1 ; c < self->numCenters ; ++c) {

      /* Can skip if the center assigned so far is twice as close
       as its distance to the center under consideration */

      if (((self->distance == VlDistanceL1) ? 2.0 : 4.0) *
          pointToClosestCenterUB[x] <=
          ((TYPE*)self->centerDistances)
          [c + assignments[x] * self->numCenters]) {
        continue ;
      }

      distance = distFn(self->dimension,
                        data + x * self->dimension,
                        (TYPE*)self->centers + c * self->dimension) ;
      pointToCenterLB[c + x * self->numCenters] = distance ;
      totDistanceComputationsToInit += 1 ;
      if (distance < pointToClosestCenterUB[x]) {
        pointToClosestCenterUB[x] = distance ;
        assignments[x] = c ;
      }
    }
  }

  /* compute UB on energy */
  energy = 0 ;
  for (x = 0 ; x < (signed)numData ; ++x) {
    energy += pointToClosestCenterUB[x] ;
  }

  if (self->verbosity) {
    VL_PRINTF("kmeans: Elkan iter 0: energy = %g, dist. calc. = %d\n",
              energy, totDistanceComputationsToInit) ;
  }

  /* #define SANITY*/
#ifdef SANITY
  {
    int xx ;
    int cc ;
    TYPE tol = 1e-5 ;
    VL_PRINTF("inconsistencies after initial assignments:\n");
    for (xx = 0 ; xx < numData ; ++xx) {
      for (cc = 0 ; cc < self->numCenters ; ++cc) {
        TYPE a = pointToCenterLB[cc + xx * self->numCenters] ;
        TYPE b = distFn(self->dimension,
                        data + self->dimension * xx,
                        (TYPE*)self->centers + self->dimension * cc) ;
        if (cc == assignments[xx]) {
          TYPE z = pointToClosestCenterUB[xx] ;
          if (z+tol<b) VL_PRINTF("UB %d %d = %f < %f\n",
                                 cc, xx, z, b) ;
        }
        if (a>b+tol) VL_PRINTF("LB %d %d = %f  > %f\n",
                               cc, xx, a, b) ;
      }
    }
  }
#endif

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  /*                          Iterations                            */
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  for (iteration = 1 ; 1; ++iteration) {

    vl_size numDistanceComputationsToRefreshUB = 0 ;
    vl_size numDistanceComputationsToRefreshLB = 0 ;
    vl_size numDistanceComputationsToRefreshCenterDistances = 0 ;
    vl_size numDistanceComputationsToNewCenters = 0 ;
    vl_size numRestartedCenters = 0 ;

    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    /*                         Compute new centers                  */
    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

    memset(clusterMasses, 0, sizeof(vl_size) * numData) ;
    for (x = 0 ; x < (signed)numData ; ++x) {
      clusterMasses[assignments[x]] ++ ;
    }

    switch (self->distance) {
      case VlDistanceL2:
        memset(newCenters, 0, sizeof(TYPE) * self->dimension * self->numCenters) ;
        for (x = 0 ; x < (signed)numData ; ++x) {
          TYPE * cpt = newCenters + assignments[x] * self->dimension ;
          TYPE const * xpt = data + x * self->dimension ;
          for (d = 0 ; d < self->dimension ; ++d) {
            cpt[d] += xpt[d] ;
          }
        }
        for (c = 0 ; c < self->numCenters ; ++c) {
          TYPE * cpt = newCenters + c * self->dimension ;
          if (clusterMasses[c] > 0) {
            TYPE mass = clusterMasses[c] ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] /= mass ;
            }
          } else {
            /* restart the center */
            vl_uindex x = vl_rand_uindex(rand, numData) ;
            numRestartedCenters ++ ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] = data[x * self->dimension + d] ;
            }
          }
        }
        break ;
      case VlDistanceL1:
        for (d = 0 ; d < self->dimension ; ++d) {
          vl_uint32 * perm = permutations + d * numData ;
          memset(numSeenSoFar, 0, sizeof(vl_size) * self->numCenters) ;
          for (x = 0; x < (signed)numData ; ++x) {
            c = assignments[perm[x]] ;
            if (2 * numSeenSoFar[c] < clusterMasses[c]) {
              newCenters [d + c * self->dimension] =
              data [d + perm[x] * self->dimension] ;
            }
            numSeenSoFar[c] ++ ;
          }
        }
        /* restart the centers as required  */
        for (c = 0 ; c < self->numCenters ; ++c) {
          if (clusterMasses[c] == 0) {
            TYPE * cpt = newCenters + c * self->dimension ;
            vl_uindex x = vl_rand_uindex(rand, numData) ;
            numRestartedCenters ++ ;
            for (d = 0 ; d < self->dimension ; ++d) {
              cpt[d] = data[x * self->dimension + d] ;
            }
          }
        }
        break ;
      default:
        abort();
    } /* done compute centers */

    /* compute the distance from the old centers to the new centers */
    for (c = 0 ; c < self->numCenters ; ++c) {
      TYPE distance = distFn(self->dimension,
                             newCenters + c * self->dimension,
                             (TYPE*)self->centers + c * self->dimension) ;
      centerToNewCenterDistances[c] = distance ;
      numDistanceComputationsToNewCenters += 1 ;
    }

    /* make the new centers current */
    {
      TYPE * tmp = self->centers ;
      self->centers = newCenters ;
      newCenters = tmp ;
    }

    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
    /*                Reassign points to a centers                  */
    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

    /*
     Update distances between centers.
     */
    numDistanceComputationsToRefreshCenterDistances
    += VL_XCAT(_vl_kmeans_update_center_distances_, SFX)(self) ;

    for (c = 0 ; c < self->numCenters ; ++c) {
      nextCenterDistances[c] = (TYPE) VL_INFINITY_D ;
      for (j = 0 ; j < self->numCenters ; ++j) {
        if (j == c) continue ;
        nextCenterDistances[c] = VL_MIN(nextCenterDistances[c],
                                        ((TYPE*)self->centerDistances)
                                        [j + c * self->numCenters]) ;
      }
    }

    /*
     Update upper bounds on point-to-closest-center distances
     based on the center variation.
     */
    for (x = 0 ; x < (signed)numData ; ++x) {
      TYPE a = pointToClosestCenterUB[x] ;
      TYPE b = centerToNewCenterDistances[assignments[x]] ;
      if (self->distance == VlDistanceL1) {
        pointToClosestCenterUB[x] = a + b ;
      } else {
#if (FLT == VL_TYPE_FLOAT)
        TYPE sqrtab =  sqrtf (a * b) ;
#else
        TYPE sqrtab =  sqrt (a * b) ;
#endif
        pointToClosestCenterUB[x] = a + b + 2.0 * sqrtab ;
      }
      pointToClosestCenterUBIsStrict[x] = VL_FALSE ;
    }

    /*
     Update lower bounds on point-to-center distances
     based on the center variation.
     */

#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(x,c) num_threads(vl_get_max_threads())
#endif
    for (x = 0 ; x < (signed)numData ; ++x) {
      for (c = 0 ; c < self->numCenters ; ++c) {
        TYPE a = pointToCenterLB[c + x * self->numCenters] ;
        TYPE b = centerToNewCenterDistances[c] ;
        if (a < b) {
          pointToCenterLB[c + x * self->numCenters] = 0 ;
        } else {
          if (self->distance == VlDistanceL1) {
            pointToCenterLB[c + x * self->numCenters]  = a - b ;
          } else {
#if (FLT == VL_TYPE_FLOAT)
            TYPE sqrtab =  sqrtf (a * b) ;
#else
            TYPE sqrtab =  sqrt (a * b) ;
#endif
            pointToCenterLB[c + x * self->numCenters]  = a + b - 2.0 * sqrtab ;
          }
        }
      }
    }

#ifdef SANITY
    {
      int xx ;
      int cc ;
      TYPE tol = 1e-5 ;
      VL_PRINTF("inconsistencies before assignments:\n");
      for (xx = 0 ; xx < numData ; ++xx) {
        for (cc = 0 ; cc < self->numCenters ; ++cc) {
          TYPE a = pointToCenterLB[cc + xx * self->numCenters] ;
          TYPE b = distFn(self->dimension,
                          data + self->dimension * xx,
                          (TYPE*)self->centers + self->dimension * cc) ;
          if (cc == assignments[xx]) {
            TYPE z = pointToClosestCenterUB[xx] ;
            if (z+tol<b) VL_PRINTF("UB %d %d = %f < %f\n",
                                   cc, xx, z, b) ;
          }
          if (a>b+tol) VL_PRINTF("LB %d %d = %f  > %f (assign = %d)\n",
                                 cc, xx, a, b, assignments[xx]) ;
        }
      }
    }
#endif

    /*
     Scan the data and do the reassignments. Use the bounds to
     skip as many point-to-center distance calculations as possible.
     */
    allDone = VL_TRUE ;

#if defined(_OPENMP)
#pragma omp parallel for \
            default(none) \
            shared(self,numData, \
              pointToClosestCenterUB,pointToCenterLB, \
              nextCenterDistances,pointToClosestCenterUBIsStrict, \
              assignments,data,distFn,allDone) \
            private(c,x) \
            reduction(+:numDistanceComputationsToRefreshUB,numDistanceComputationsToRefreshLB) \
            num_threads(vl_get_max_threads())
#endif
    for (x = 0 ; x < (signed)numData ; ++ x) {
      /*
       A point x sticks with its current center assignmets[x]
       the UB to d(x, c[assigmnets[x]]) is not larger than half
       the distance of c[assigments[x]] to any other center c.
       */
      if (((self->distance == VlDistanceL1) ? 2.0 : 4.0) *
          pointToClosestCenterUB[x] <= nextCenterDistances[assignments[x]]) {
        continue ;
      }

      for (c = 0 ; c < self->numCenters ; ++c) {
        vl_uint32 cx = assignments[x] ;
        TYPE distance ;

        /* The point is not reassigned to a given center c
         if either:

         0 - c is already the assigned center
         1 - The UB of d(x, c[assignments[x]]) is smaller than half
         the distance of c[assigments[x]] to c, OR
         2 - The UB of d(x, c[assignmets[x]]) is smaller than the
         LB of the distance of x to c.
         */
        if (cx == c) {
          continue ;
        }
        if (((self->distance == VlDistanceL1) ? 2.0 : 4.0) *
            pointToClosestCenterUB[x] <= ((TYPE*)self->centerDistances)
            [c + cx * self->numCenters]) {
          continue ;
        }
        if (pointToClosestCenterUB[x] <= pointToCenterLB
            [c + x * self->numCenters]) {
          continue ;
        }

        /* If the UB is loose, try recomputing it and test again */
        if (! pointToClosestCenterUBIsStrict[x]) {
          distance = distFn(self->dimension,
                            data + self->dimension * x,
                            (TYPE*)self->centers + self->dimension * cx) ;
          pointToClosestCenterUB[x] = distance ;
          pointToClosestCenterUBIsStrict[x] = VL_TRUE ;
          pointToCenterLB[cx + x * self->numCenters] = distance ;
          numDistanceComputationsToRefreshUB += 1 ;

          if (((self->distance == VlDistanceL1) ? 2.0 : 4.0) *
              pointToClosestCenterUB[x] <= ((TYPE*)self->centerDistances)
              [c + cx * self->numCenters]) {
            continue ;
          }
          if (pointToClosestCenterUB[x] <= pointToCenterLB
              [c + x * self->numCenters]) {
            continue ;
          }
        }

        /*
         Now the UB is strict (equal to d(x, assignments[x])), but
         we still could not exclude that x should be reassigned to
         c. We therefore compute the distance, update the LB,
         and check if a reassigmnet must be made
         */
        distance = distFn(self->dimension,
                          data + x * self->dimension,
                          (TYPE*)self->centers + c *  self->dimension) ;
        numDistanceComputationsToRefreshLB += 1 ;
        pointToCenterLB[c + x * self->numCenters] = distance ;

        if (distance < pointToClosestCenterUB[x]) {
          assignments[x] = c ;
          pointToClosestCenterUB[x] = distance ;
          allDone = VL_FALSE ;
          /* the UB strict flag is already set here */
        }

      } /* assign center */
    } /* next data point */


    totDistanceComputationsToRefreshUB
    += numDistanceComputationsToRefreshUB ;

    totDistanceComputationsToRefreshLB
    += numDistanceComputationsToRefreshLB ;

    totDistanceComputationsToRefreshCenterDistances
    += numDistanceComputationsToRefreshCenterDistances ;

    totDistanceComputationsToNewCenters
    += numDistanceComputationsToNewCenters ;

    totNumRestartedCenters
    += numRestartedCenters ;

#ifdef SANITY
    {
      int xx ;
      int cc ;
      TYPE tol = 1e-5 ;
      VL_PRINTF("inconsistencies after assignments:\n");
      for (xx = 0 ; xx < numData ; ++xx) {
        for (cc = 0 ; cc < self->numCenters ; ++cc) {
          TYPE a = pointToCenterLB[cc + xx * self->numCenters] ;
          TYPE b = distFn(self->dimension,
                          data + self->dimension * xx,
                          (TYPE*)self->centers + self->dimension * cc) ;
          if (cc == assignments[xx]) {
            TYPE z = pointToClosestCenterUB[xx] ;
            if (z+tol<b) VL_PRINTF("UB %d %d = %f < %f\n",
                                   cc, xx, z, b) ;
          }
          if (a>b+tol) VL_PRINTF("LB %d %d = %f  > %f (assign = %d)\n",
                                 cc, xx, a, b, assignments[xx]) ;
        }
      }
    }
#endif

    /* compute UB on energy */
    energy = 0 ;
    for (x = 0 ; x < (signed)numData ; ++x) {
      energy += pointToClosestCenterUB[x] ;
    }

    if (self->verbosity) {
      vl_size numDistanceComputations =
      numDistanceComputationsToRefreshUB +
      numDistanceComputationsToRefreshLB +
      numDistanceComputationsToRefreshCenterDistances +
      numDistanceComputationsToNewCenters ;
      VL_PRINTF("kmeans: Elkan iter %d: energy <= %g, dist. calc. = %d\n",
                iteration,
                energy,
                numDistanceComputations) ;
      if (numRestartedCenters) {
        VL_PRINTF("kmeans: Elkan iter %d: restarted %d centers\n",
                  iteration,
                  energy,
                  numRestartedCenters) ;
      }
      if (self->verbosity > 1) {
        VL_PRINTF("kmeans: Elkan iter %d: total dist. calc. per type: "
                  "UB: %.1f%% (%d), LB: %.1f%% (%d), "
                  "intra_center: %.1f%% (%d), "
                  "new_center: %.1f%% (%d)\n",
                  iteration,
                  100.0 * numDistanceComputationsToRefreshUB / numDistanceComputations,
                  numDistanceComputationsToRefreshUB,
                  100.0 *numDistanceComputationsToRefreshLB / numDistanceComputations,
                  numDistanceComputationsToRefreshLB,
                  100.0 * numDistanceComputationsToRefreshCenterDistances / numDistanceComputations,
                  numDistanceComputationsToRefreshCenterDistances,
                  100.0 * numDistanceComputationsToNewCenters / numDistanceComputations,
                  numDistanceComputationsToNewCenters) ;
      }
    }

    /* check termination conditions */
    if (iteration >= self->maxNumIterations) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: Elkan terminating because maximum number of iterations reached\n") ;
      }
      break ;
    }
    if (allDone) {
      if (self->verbosity) {
        VL_PRINTF("kmeans: Elkan terminating because the algorithm fully converged\n") ;
      }
      break ;
    }

  } /* next Elkan iteration */

  /* compute true energy */
  energy = 0 ;
  for (x = 0 ; x < (signed)numData ; ++ x) {
    vl_uindex cx = assignments [x] ;
    energy += distFn(self->dimension,
                     data + self->dimension * x,
                     (TYPE*)self->centers + self->dimension * cx) ;
    totDistanceComputationsToFinalize += 1 ;
  }

  {
    vl_size totDistanceComputations =
    totDistanceComputationsToInit +
    totDistanceComputationsToRefreshUB +
    totDistanceComputationsToRefreshLB +
    totDistanceComputationsToRefreshCenterDistances +
    totDistanceComputationsToNewCenters +
    totDistanceComputationsToFinalize ;

    double saving = (double)totDistanceComputations
    / (iteration * self->numCenters * numData) ;

    if (self->verbosity) {
      VL_PRINTF("kmeans: Elkan: total dist. calc.: %d (%.2f %% of Lloyd)\n",
                totDistanceComputations, saving * 100.0) ;
      if (totNumRestartedCenters) {
        VL_PRINTF("kmeans: Elkan: there have been %d restarts\n",
                  totNumRestartedCenters) ;
      }
    }

    if (self->verbosity > 1) {
      VL_PRINTF("kmeans: Elkan: total dist. calc. per type: "
                "init: %.1f%% (%d), UB: %.1f%% (%d), LB: %.1f%% (%d), "
                "intra_center: %.1f%% (%d), "
                "new_center: %.1f%% (%d), "
                "finalize: %.1f%% (%d)\n",
                100.0 * totDistanceComputationsToInit / totDistanceComputations,
                totDistanceComputationsToInit,
                100.0 * totDistanceComputationsToRefreshUB / totDistanceComputations,
                totDistanceComputationsToRefreshUB,
                100.0 *totDistanceComputationsToRefreshLB / totDistanceComputations,
                totDistanceComputationsToRefreshLB,
                100.0 * totDistanceComputationsToRefreshCenterDistances / totDistanceComputations,
                totDistanceComputationsToRefreshCenterDistances,
                100.0 * totDistanceComputationsToNewCenters / totDistanceComputations,
                totDistanceComputationsToNewCenters,
                100.0 * totDistanceComputationsToFinalize / totDistanceComputations,
                totDistanceComputationsToFinalize) ;
    }
  }

  if (permutations) {
    vl_free(permutations) ;
  }
  if (numSeenSoFar) {
    vl_free(numSeenSoFar) ;
  }

  vl_free(distances) ;
  vl_free(assignments) ;
  vl_free(clusterMasses) ;

  vl_free(nextCenterDistances) ;
  vl_free(pointToClosestCenterUB) ;
  vl_free(pointToClosestCenterUBIsStrict) ;
  vl_free(pointToCenterLB) ;
  vl_free(newCenters) ;
  vl_free(centerToNewCenterDistances) ;

  return energy ;
}

/* ---------------------------------------------------------------- */
static double
VL_XCAT(_vl_kmeans_refine_centers_, SFX)
(VlKMeans * self,
 TYPE const * data,
 vl_size numData)
{
  switch (self->algorithm) {
    case VlKMeansLloyd:
      return
        VL_XCAT(_vl_kmeans_refine_centers_lloyd_, SFX)(self, data, numData) ;
      break ;
    case VlKMeansElkan:
      return
        VL_XCAT(_vl_kmeans_refine_centers_elkan_, SFX)(self, data, numData) ;
      break ;
    case VlKMeansANN:
      return
        VL_XCAT(_vl_kmeans_refine_centers_ann_, SFX)(self, data, numData) ;
      break ;
    default:
      abort() ;
  }
}

/* VL_KMEANS_INSTANTIATING */
#else

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_KMEANS_INSTANTIATING
#include "kmeans.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_KMEANS_INSTANTIATING
#include "kmeans.c"
#endif

/* VL_KMEANS_INSTANTIATING */
#endif

/* ================================================================ */
#ifndef VL_KMEANS_INSTANTIATING

/** ------------------------------------------------------------------
 ** @brief Set centers
 ** @param self KMeans object.
 ** @param centers centers to copy.
 ** @param dimension data dimension.
 ** @param numCenters number of centers.
 **/

VL_EXPORT void
vl_kmeans_set_centers
(VlKMeans * self,
 void const * centers,
 vl_size dimension,
 vl_size numCenters)
{
  vl_kmeans_reset (self) ;

  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_kmeans_set_centers_f
      (self, (float const *)centers, dimension, numCenters) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_kmeans_set_centers_d
      (self, (double const *)centers, dimension, numCenters) ;
      break ;
    default:
      abort() ;
  }
}

/** ------------------------------------------------------------------
 ** @brief init centers by randomly sampling data
 ** @param self KMeans object.
 ** @param data data to sample from.
 ** @param dimension data dimension.
 ** @param numData nmber of data points.
 ** @param numCenters number of centers.
 **
 ** The function inits the KMeans centers by randomly sampling
 ** the data @a data.
 **/

VL_EXPORT void
vl_kmeans_init_centers_with_rand_data
(VlKMeans * self,
 void const * data,
 vl_size dimension,
 vl_size numData,
 vl_size numCenters)
{
  vl_kmeans_reset (self) ;

  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_kmeans_init_centers_with_rand_data_f
      (self, (float const *)data, dimension, numData, numCenters) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_kmeans_init_centers_with_rand_data_d
      (self, (double const *)data, dimension, numData, numCenters) ;
      break ;
    default:
      abort() ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Seed centers by the KMeans++ algorithm
 ** @param self KMeans object.
 ** @param data data to sample from.
 ** @param dimension data dimension.
 ** @param numData nmber of data points.
 ** @param numCenters number of centers.
 **/

VL_EXPORT void
vl_kmeans_init_centers_plus_plus
(VlKMeans * self,
 void const * data,
 vl_size dimension,
 vl_size numData,
 vl_size numCenters)
{
  vl_kmeans_reset (self) ;

  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_kmeans_init_centers_plus_plus_f
      (self, (float const *)data, dimension, numData, numCenters) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_kmeans_init_centers_plus_plus_d
      (self, (double const *)data, dimension, numData, numCenters) ;
      break ;
    default:
      abort() ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Quantize data
 ** @param self KMeans object.
 ** @param assignments data to closest center assignments (output).
 ** @param distances data to closest center distance (output).
 ** @param data data to quantize.
 ** @param numData number of data points to quantize.
 **/

VL_EXPORT void
vl_kmeans_quantize
(VlKMeans * self,
 vl_uint32 * assignments,
 void * distances,
 void const * data,
 vl_size numData)
{
  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_kmeans_quantize_f
      (self, assignments, distances, (float const *)data, numData) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_kmeans_quantize_d
      (self, assignments, distances, (double const *)data, numData) ;
      break ;
    default:
      abort() ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Quantize data using approximate nearest neighbours (ANN).
 ** @param self KMeans object.
 ** @param assignments data to centers assignments (output).
 ** @param distances data to closes center distance (output)
 ** @param data data to quantize.
 ** @param numData number of data points.
 ** @param update choose wether to update current assignments.
 **
 ** The function uses an ANN procedure to compute the approximate
 ** nearest neighbours of the input data point.
 **
 ** Setting @a update to ::VL_TRUE will cause the algorithm
 ** to *update existing assignments*. This means that each
 ** element of @a assignments and @a distances is updated ony if the
 ** ANN procedure can find a better assignment of the existing one.
 **/

VL_EXPORT void
vl_kmeans_quantize_ann
(VlKMeans * self,
 vl_uint32 * assignments,
 void * distances,
 void const * data,
 vl_size numData,
 vl_bool update)
{
  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_kmeans_quantize_ann_f
      (self, assignments, distances, (float const *)data, numData, update) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_kmeans_quantize_ann_d
      (self, assignments, distances, (double const *)data, numData, update) ;
      break ;
    default:
      abort() ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Refine center locations.
 ** @param self KMeans object.
 ** @param data data to quantize.
 ** @param numData number of data points.
 ** @return K-means energy at the end of optimization.
 **
 ** The function calls the underlying K-means quantization algorithm
 ** (@ref VlKMeansAlgorithm) to quantize the specified data @a data.
 ** The function assumes that the cluster centers have already
 ** been assigned by using one of the seeding functions, or by
 ** setting them.
 **/

VL_EXPORT double
vl_kmeans_refine_centers
(VlKMeans * self,
 void const * data,
 vl_size numData)
{
  assert (self->centers) ;

  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      return
        _vl_kmeans_refine_centers_f
        (self, (float const *)data, numData) ;
    case VL_TYPE_DOUBLE :
      return
        _vl_kmeans_refine_centers_d
        (self, (double const *)data, numData) ;
    default:
      abort() ;
  }
}


/** ------------------------------------------------------------------
 ** @brief Cluster data.
 ** @param self KMeans object.
 ** @param data data to quantize.
 ** @param dimension data dimension.
 ** @param numData number of data points.
 ** @param numCenters number of clusters.
 ** @return K-means energy at the end of optimization.
 **
 ** The function initializes the centers by using the initialization
 ** algorithm set by ::vl_kmeans_set_initialization and refines them
 ** by the quantization algorithm set by ::vl_kmeans_set_algorithm.
 ** The process is repeated one or more times (see
 ** ::vl_kmeans_set_num_repetitions) and the resutl with smaller
 ** energy is retained.
 **/

VL_EXPORT double
vl_kmeans_cluster (VlKMeans * self,
                   void const * data,
                   vl_size dimension,
                   vl_size numData,
                   vl_size numCenters)
{
  vl_uindex repetition ;
  double bestEnergy = VL_INFINITY_D ;
  void * bestCenters = NULL ;

  for (repetition = 0 ; repetition < self->numRepetitions ; ++ repetition) {
    double energy ;
    double timeRef ;

    if (self->verbosity) {
      VL_PRINTF("kmeans: repetition %d of %d\n", repetition + 1, self->numRepetitions) ;
    }

    timeRef = vl_get_cpu_time() ;
    switch (self->initialization) {
      case VlKMeansRandomSelection :
        vl_kmeans_init_centers_with_rand_data (self,
                                               data, dimension, numData,
                                               numCenters) ;
        break ;
      case VlKMeansPlusPlus :
        vl_kmeans_init_centers_plus_plus (self,
                                          data, dimension, numData,
                                          numCenters) ;
        break ;
      default:
        abort() ;
    }

    if (self->verbosity) {
      VL_PRINTF("kmeans: K-means initialized in %.2f s\n",
                vl_get_cpu_time() - timeRef) ;
    }

    timeRef = vl_get_cpu_time () ;
    energy = vl_kmeans_refine_centers (self, data, numData) ;
    if (self->verbosity) {
      VL_PRINTF("kmeans: K-means terminated in %.2f s with energy %g\n",
                vl_get_cpu_time() - timeRef, energy) ;
    }

    /* copy centers to output if current solution is optimal */
    /* check repetition == 0 as well in case energy = NaN, which */
    /* can happen if the data contain NaNs */
    if (energy < bestEnergy || repetition == 0) {
      void * temp ;
      bestEnergy = energy ;

      if (bestCenters == NULL) {
        bestCenters = vl_malloc(vl_get_type_size(self->dataType) *
                                self->dimension *
                                self->numCenters) ;
      }

      /* swap buffers */
      temp = bestCenters ;
      bestCenters = self->centers ;
      self->centers = temp ;
    } /* better energy */
  } /* next repetition */

  vl_free (self->centers) ;
  self->centers = bestCenters ;
  return bestEnergy ;
}

/* VL_KMEANS_INSTANTIATING */
#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_KMEANS_INSTANTIATING
