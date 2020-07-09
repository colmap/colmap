/** @file vlad.c
 ** @brief VLAD - Declaration
 ** @author David Novotny
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 David Novotny and Andera Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page vlad Vector of Locally Aggregated Descriptors (VLAD) encoding
@author David Novotny
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref vlad.h implements the *Vector of Linearly Aggregated Descriptors*
(VLAD) image representation @cite{jegou10aggregating}
@cite{arandjelovic13all-about}.

@ref vlad-starting demonstreates how to use the C API to compute the
VLAD representation of an image. For further details on the VLAD image
representation refer to:

- @subpage vlad-fundamentals - VLAD definition and computation.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section vlad-starting Getting started with VLAD
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The VLAD encoding of a set of features is obtained by using the
function ::vl_vlad_encode.  The function can be applied to both @c
float or @c double data types.

::vl_vlad_encode requires a visual dictionary, for example obtained by
using @ref kmeans. Furthermore, the assignments of features to
dictionary elements must be pre-computed, for example by using @ref
kdtree.

In the following example code, the vocabulary is first created using
the KMeans clustering, then the points, that are to be encoded are
assigned to its corresponding nearest vocabulary words, after that the
original vlad encoding routine without any normalization option takes place.
At the end of the process the encoding is stored in the @c enc variable.

@code
vl_uint32 * indexes;
float * assignments;
float * enc
int i;

// create a KMeans object and run clustering to get vocabulary words (centers)
kmeans = vl_kmeans_new (VLDistanceL2, VL_TYPE_FLOAT) ;
vl_kmeans_cluster (kmeans,
                   data,
                   dimension,
                   numData,
                   numCenters) ;

// find nearest cliuster centers for the data that should be encoded
indexes = vl_malloc(sizeof(vl_uint32) * numDataToEncode);
vl_kmeans_quantize(kmeans,indexes,dataToEncode,numDataToEncode);

// convert indexes array to assignments array,
// which can be processed by vl_vlad_encode
assignments = vl_malloc(sizeof(float) * numDataToEncode * numCenters);
memset(assignments, 0, sizeof(float) * numDataToEncode * numCenters);
for(i = 0; i < numDataToEncode; i++) {
  assignments[i * numCenters + indexes[i]] = 1.;
}

// allocate space for vlad encoding
enc = vl_malloc(sizeof(TYPE) * dimension * numCenters);

// do the encoding job
vl_vlad_encode (enc, VL_F_TYPE,
                vl_kmeans_get_centers(kmeans), dimension, numCenters,
                data, numData,
                assignments,
                0) ;
@endcode

Various @ref vlad-normalization normalizations can be applied to the
VLAD vectors. These are controlled by the parameter @a flag of
::vl_vlad_encode.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page vlad-fundamentals VLAD fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This page describes the *Vector of Locally Aggregated Descriptors*
(VLAD) image encoding of @cite{jegou10aggregating}. See @ref vlad for
an overview of the C API.

VLAD is a *feature encoding and pooling* method, similar to @ref
fisher "Fisher vectors". VLAD encodes a set of local feature
descriptors $I=(\bx_1,\dots,\bx_n)$ extracted from an image using a
dictionary built using a clustering method such as @ref gmm or @ref
kmeans. Let $q_{ik}$ be the strength of the association of data vector
$\bx_i$ to cluster $\mu_k$, such that $q_{ik} \geq 0$ and
$\sum_{k=1}^K q_{ik} = 1$. The association may be either soft
(e.g. obtained as the posterior probabilities of the GMM clusters) or
hard (e.g. obtained by vector quantization with K-means).

$\mu_k$ are the cluster *means*, vectors of the same dimension as the
data $\bx_i$. VLAD encodes feature $\bx$ by considering the *residuals*
\[
 \bv_k = \sum_{i=1}^{N} q_{ik} (\bx_{i} - \mu_k).
\]
The residulas are stacked together to obtain the vector
\[
\hat\Phi(I) =
\begin{bmatrix}
\vdots \\
\bv_k \\
\vdots
\end{bmatrix}
\]

Before the VLAD encoding is used it is usually normalized, as
explained @ref vlad-normalization next.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section vlad-normalization VLAD normalization
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat VLAD implementation supports a number of different
normalization strategies. These are optionally applied in this order:

- **Component-wise mass normalization.** Each vector $\bv_k$ is
  divided by the total mass of features associated to it $\sum_{i=1}^N
  q_{ik}$.

- **Square-rooting.** The function $\sign(z)\sqrt{|z|}$ is applied to
  all scalar components of the VLAD descriptor.

- **Component-wise $l^2$ normalization.** The vectors $\bv_k$ are
  divided by their norm $\|\bv_k\|_2$.

- **Global $l^2$ normalization.** The VLAD descriptor $\hat\Phi(I)$ is
  divided by its norm $\|\hat\Phi(I)\|_2$.
*/

#include "vlad.h"
#include "mathop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ================================================================ */
#ifdef VL_VLAD_INSTANTIATING

static void
VL_XCAT(_vl_vlad_encode_, SFX)
(TYPE * enc,
 TYPE const * means, vl_size dimension, vl_size numClusters,
 TYPE const * data, vl_size numData,
 TYPE const * assignments,
 int flags)
{
  vl_uindex dim ;
  vl_index i_cl, i_d ;

  memset(enc, 0, sizeof(TYPE) * dimension * numClusters) ;

#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl,i_d,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0; i_cl < (signed)numClusters; i_cl++) {
    double clusterMass = 0 ;
    for (i_d = 0; i_d < (signed)numData; i_d++) {
      if (assignments[i_d*numClusters + i_cl] > 0) {
        double q = assignments[i_d*numClusters+i_cl] ;
        clusterMass +=  q ;
        for(dim = 0; dim < dimension; dim++) {
          enc [i_cl * dimension + dim] += q * data [i_d  * dimension + dim] ;
        }
      }
    }

    if (clusterMass > 0) {
      if (flags & VL_VLAD_FLAG_NORMALIZE_MASS) {
        for(dim = 0; dim < dimension; dim++) {
          enc[i_cl*dimension + dim] /= clusterMass ;
          enc[i_cl*dimension + dim] -= means[i_cl*dimension+dim];
        }
      } else {
        for(dim = 0; dim < dimension; dim++) {
          enc[i_cl*dimension + dim] -= clusterMass * means[i_cl*dimension+dim];
        }
      }
    }

    if (flags & VL_VLAD_FLAG_SQUARE_ROOT) {
      for(dim = 0; dim < dimension; dim++) {
        TYPE z = enc[i_cl*dimension + dim] ;
        if (z >= 0) {
          enc[i_cl*dimension + dim] = VL_XCAT(vl_sqrt_, SFX)(z) ;
        } else {
          enc[i_cl*dimension + dim] = - VL_XCAT(vl_sqrt_, SFX)(- z) ;
        }
      }
    }

    if (flags & VL_VLAD_FLAG_NORMALIZE_COMPONENTS) {
      TYPE n = 0 ;
      dim = 0 ;
      for(dim = 0; dim < dimension; dim++) {
        TYPE z = enc[i_cl*dimension + dim] ;
        n += z * z ;
      }
      n = VL_XCAT(vl_sqrt_, SFX)(n) ;
      n = VL_MAX(n, 1e-12) ;
      for(dim = 0; dim < dimension; dim++) {
        enc[i_cl*dimension + dim] /= n ;
      }
    }
  }

  if (! (flags & VL_VLAD_FLAG_UNNORMALIZED)) {
    TYPE n = 0 ;
    for(dim = 0 ; dim < dimension * numClusters ; dim++) {
      TYPE z = enc [dim] ;
      n += z * z ;
    }
    n = VL_XCAT(vl_sqrt_, SFX)(n) ;
    n = VL_MAX(n, 1e-12) ;
    for(dim = 0 ; dim < dimension * numClusters ; dim++) {
      enc[dim] /= n ;
    }
  }
}

/* VL_FISHER_INSTANTIATING */
#else

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_VLAD_INSTANTIATING
#include "vlad.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_VLAD_INSTANTIATING
#include "vlad.c"
#endif

/* VL_VLAD_INSTANTIATING */
#endif

/* ================================================================ */
#ifndef VL_VLAD_INSTANTIATING

/** @brief VLAD encoding of a set of vectors.
 ** @param enc output VLAD encoding (out).
 ** @param dataType the type of the input data (::VL_TYPE_DOUBLE or ::VL_TYPE_FLOAT).
 ** @param numData number of data vectors to encode.
 ** @param means cluster means.
 ** @param numClusters number of clusters.
 ** @param data the data vectors to encode.
 ** @param dimension dimensionality of the data.
 ** @param assignments data to cluster soft assignments.
 ** @param flags options.
 **
 ** @a enc is the VLAD vector of size @a numClusters by
 ** @a dimension. @a means is a matrix with @a numClusters columns and
 ** @a dimension rows. @a data is the matrix of vectors to be encoded,
 ** with @a dimension rows and @a numData columns. @a assignments is a
 ** matrix with @a numClusters rows and @a numData columns.
 ** All the matrices should be stored in column-major order.
 **
 ** @a flag allows controlling further options:
 ** ::VL_VLAD_FLAG_NORMALIZE_COMPONENTS, ::VL_VLAD_FLAG_SQUARE_ROOT,
 ** ::VL_VLAD_FLAG_UNNORMALIZED, and ::VL_VLAD_FLAG_NORMALIZE_MASS.
 **
 ** @sa @ref vlad
 **/

void
vl_vlad_encode (void * enc, vl_type dataType,
                void const * means, vl_size dimension, vl_size numClusters,
                void const * data, vl_size numData,
                void const * assignments,
                int flags)
{
  switch(dataType) {
    case VL_TYPE_FLOAT:
      _vl_vlad_encode_f ((float *) enc,
                         (float const *) means, dimension, numClusters,
                         (float const *) data,  numData,
                         (float const *) assignments, flags) ;
      break;
    case VL_TYPE_DOUBLE:
      _vl_vlad_encode_d ((double *) enc,
                         (double const *) means, dimension, numClusters,
                         (double const *) data, numData,
                         (double const *) assignments, flags) ;
      break;
    default:
      abort();
  }
}

/* ! VL_VLAD_INSTANTIATING */
#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_VLAD_INSTANTIATING
