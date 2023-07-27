/** @file ikmeans.h
 ** @brief Integer K-Means clustering
 ** @author Brian Fulkerson
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_IKMEANS_H
#define VL_IKMEANS_H

#include "generic.h"
#include "random.h"

#if 0
typedef vl_int64 vl_ikmacc_t ; /**< IKM accumulator data type */
#define VL_IKMACC_MAX 0x7fffffffffffffffULL
#else
typedef vl_int32 vl_ikmacc_t ; /**< IKM accumulator data type */
#define VL_IKMACC_MAX 0x7fffffffUL
#endif


/** ------------------------------------------------------------------
 ** @brief IKM algorithms
 **/

enum VlIKMAlgorithms {
  VL_IKM_LLOYD, /**< Lloyd algorithm */
  VL_IKM_ELKAN, /**< Elkan algorithm */
} ;

/** ------------------------------------------------------------------
 ** @brief IKM quantizer
 **/

typedef struct _VlIKMFilt
{
  vl_size M ; /**< data dimensionality */
  vl_size K ; /**< number of centers   */
  vl_size max_niters ; /**< Lloyd: maximum number of iterations */
  int method ; /**< Learning method */
  int verb ; /**< verbosity level */
  vl_ikmacc_t *centers ; /**< centers */
  vl_ikmacc_t *inter_dist ; /**< centers inter-distances */
} VlIKMFilt ;

/** @name Create and destroy
 ** @{ */
VL_EXPORT VlIKMFilt *vl_ikm_new (int method) ;
VL_EXPORT void vl_ikm_delete (VlIKMFilt *f) ;
/** @} */

/** @name Process data
 ** @{ */
VL_EXPORT void vl_ikm_init (VlIKMFilt *f, vl_ikmacc_t const *centers, vl_size M, vl_size K) ;
VL_EXPORT void vl_ikm_init_rand (VlIKMFilt *f, vl_size M, vl_size K) ;
VL_EXPORT void vl_ikm_init_rand_data (VlIKMFilt *f, vl_uint8 const *data, vl_size M, vl_size N, vl_size K) ;
VL_EXPORT int  vl_ikm_train (VlIKMFilt *f, vl_uint8 const *data, vl_size N) ;
VL_EXPORT void vl_ikm_push (VlIKMFilt *f, vl_uint32 *asgn, vl_uint8 const *data, vl_size N) ;
VL_EXPORT vl_uint vl_ikm_push_one (vl_ikmacc_t const *centers, vl_uint8 const *data, vl_size M, vl_size K) ;
/** @} */

/** @name Retrieve data and parameters
 ** @{ */
VL_EXPORT vl_size vl_ikm_get_ndims (VlIKMFilt const *f) ;
VL_EXPORT vl_size vl_ikm_get_K (VlIKMFilt const *f) ;
VL_EXPORT int vl_ikm_get_verbosity (VlIKMFilt const *f) ;
VL_EXPORT vl_size vl_ikm_get_max_niters (VlIKMFilt const *f) ;
VL_EXPORT vl_ikmacc_t const *vl_ikm_get_centers (VlIKMFilt const *f) ;
/** @} */

/** @name Set parameters
 ** @{ */
VL_EXPORT void vl_ikm_set_verbosity (VlIKMFilt *f, int verb) ;
VL_EXPORT void vl_ikm_set_max_niters (VlIKMFilt *f, vl_size max_niters) ;
/** @} */

/* VL_IKMEANS_H */
#endif
