/** @file hikmeans.h
 ** @brief Hierarchical Integer K-Means Clustering
 ** @author Brian Fulkerson
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_HIKMEANS_H
#define VL_HIKMEANS_H

#include "generic.h"
#include "ikmeans.h"

struct _VLHIKMTree ;
struct _VLHIKMNode ;

/** @brief HIKM tree node
 **
 ** The number of children @a K is not bigger than the @a K parameter
 ** of the HIKM tree.
 **/
typedef struct _VlHIKMNode
{
  VlIKMFilt *filter ; /**< IKM filter for this node*/
  struct _VlHIKMNode **children ; /**< Node children (if any) */
} VlHIKMNode ;

/** @brief HIKM tree */
typedef struct _VlHIKMTree {
  vl_size M ; /**< IKM: data dimensionality */
  vl_size K ; /**< IKM: K */
  vl_size depth ; /**< Depth of the tree */
  vl_size max_niters ;  /**< IKM: maximum # of iterations */
  int method ; /**< IKM: method */
  int verb ; /**< Verbosity level */
  VlHIKMNode * root; /**< Tree root node */
} VlHIKMTree ;

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT VlHIKMTree *vl_hikm_new (int method) ;
VL_EXPORT void vl_hikm_delete (VlHIKMTree *f) ;
/** @} */

/** @name Retrieve data and parameters
 ** @{
 **/
VL_EXPORT vl_size vl_hikm_get_ndims (VlHIKMTree const *f) ;
VL_EXPORT vl_size vl_hikm_get_K (VlHIKMTree const *f) ;
VL_EXPORT vl_size vl_hikm_get_depth (VlHIKMTree const *f) ;
VL_EXPORT int vl_hikm_get_verbosity (VlHIKMTree const *f) ;
VL_EXPORT vl_size vl_hikm_get_max_niters (VlHIKMTree const *f) ;
VL_EXPORT VlHIKMNode const * vl_hikm_get_root (VlHIKMTree const *f) ;
/** @} */

/** @name Set parameters
 ** @{
 **/
VL_EXPORT void vl_hikm_set_verbosity (VlHIKMTree *f, int verb) ;
VL_EXPORT void vl_hikm_set_max_niters (VlHIKMTree *f, int max_niters) ;
/** @} */

/** @name Process data
 ** @{
 **/
VL_EXPORT void vl_hikm_init (VlHIKMTree *f, vl_size M, vl_size K, vl_size depth) ;
VL_EXPORT void vl_hikm_train (VlHIKMTree *f, vl_uint8 const *data, vl_size N) ;
VL_EXPORT void vl_hikm_push (VlHIKMTree *f, vl_uint32 *asgn, vl_uint8 const *data, vl_size N) ;
/** @} */


/* VL_HIKMEANS_H */
#endif
