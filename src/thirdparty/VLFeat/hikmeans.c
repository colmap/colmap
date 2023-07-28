/** @file hikmeans.c
 ** @brief Hierarchical Integer K-Means Clustering - Declaration
 ** @author Brian Fulkerson
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file hikmeans.h
 ** @brief Hierarchical integer K-Means clustering
 **
 ** Hierarchical integer K-Means clustering (HIKM) is a simple
 ** hierarchical version of integer K-Means (@ref ikmeans.h
 ** "IKM"). The algorithm recursively applies integer K-means to create
 ** more refined partitions of the data.
 **
 ** Create a tree with ::vl_hikm_new() and delete it with
 ** ::vl_hikm_delete(). Use ::vl_hikm_train() to build the tree
 ** from training data and ::vl_hikm_push() to project new data down
 ** a HIKM tree.
 **
 ** @section hikm-tree HIKM tree
 **
 ** The HIKM tree is represented by a ::VlHIKMTree structure, which
 ** contains a tree composed of ::VlHIKMNode. Each node is an
 ** integer K-means filter which partitions the data into @c K
 ** clusters.
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hikmeans.h"

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy a subset of the data to a buffer
 ** @param data Data
 ** @param ids Data labels
 ** @param N Number of indices
 ** @param M Data dimensionality
 ** @param id Label of data to copy
 ** @param N2 Number of data copied (out)
 ** @return a new buffer with a copy of the selected data.
 **/

vl_uint8*
vl_hikm_copy_subset (vl_uint8 const * data,
                     vl_uint32 *ids,
                     vl_size N, vl_size M,
                     vl_uint32 id, vl_size *N2)
{
  vl_uindex i ;
  vl_size count = 0;

  /* count how many data points with this label there are */
  for (i = 0 ; i < N ; i++) {
    if (ids[i] == id) {
      count ++ ;
    }
  }
  *N2 = count ;

  /* copy each datum to the buffer */
  {
    vl_uint8 *new_data = vl_malloc (sizeof(*new_data) * M * count);
    count = 0;
    for (i = 0 ; i < N ; i ++) {
      if (ids[i] == id) {
        memcpy(new_data + count * M,
               data + i * M,
               sizeof(*new_data) * M);
        count ++ ;
      }
    }
    *N2 = count ;
    return new_data ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Compute HIKM clustering.
 **
 ** @param tree   HIKM tree to initialize.
 ** @param data   Data to cluster.
 ** @param N      Number of data points.
 ** @param K      Number of clusters for this node.
 ** @param height Tree height.
 **
 ** @remark height cannot be smaller than 1.
 **
 ** @return a new HIKM node representing a sub-clustering.
 **/

static VlHIKMNode *
xmeans (VlHIKMTree *tree,
        vl_uint8 const *data,
        vl_size N, vl_size K, vl_size height)
{
  VlHIKMNode *node = vl_malloc (sizeof(*node)) ;
  vl_uint32 *ids = vl_malloc (sizeof(*ids) * N) ;

  node->filter = vl_ikm_new (tree -> method) ;
  node->children = (height == 1) ? 0 : vl_malloc (sizeof(*node->children) * K) ;

  vl_ikm_set_max_niters (node->filter, tree->max_niters) ;
  vl_ikm_set_verbosity  (node->filter, tree->verb - 1  ) ;
  vl_ikm_init_rand_data (node->filter, data, tree->M, N, K) ;
  vl_ikm_train (node->filter, data, N) ;
  vl_ikm_push (node->filter, ids, data, N) ;

  /* recursively process each child */
  if (height > 1) {
    vl_uindex k ;
    for (k = 0 ; k < K ; ++k) {
      vl_size partition_N ;
      vl_size partition_K ;
      vl_uint8 *partition ;

      partition = vl_hikm_copy_subset
        (data, ids, N, tree->M, (vl_uint32)k, &partition_N) ;

      partition_K = VL_MIN (K, partition_N) ;

      node->children [k] = xmeans
        (tree, partition, partition_N, partition_K, height - 1) ;

      vl_free (partition) ;

      if (tree->verb > (signed)tree->depth - (signed)height) {
        VL_PRINTF("hikmeans: branch at depth %d: %6.1f %% completed\n",
                  tree->depth - height,
                  (double) (k+1) / K * 100) ;
      }
    }
  }

  vl_free (ids) ;
  return node ;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Delete node
 **
 ** @param node to delete.
 **
 ** The function deletes recursively @a node and all its descendent.
 **/

static void
xdelete (VlHIKMNode *node)
{
  if(node) {
    vl_uindex k ;
    if (node->children) {
      for(k = 0 ; k < vl_ikm_get_K (node->filter) ; ++k)
        xdelete (node->children[k]) ;
      vl_free (node->children) ;
    }
    if (node->filter) {
      vl_ikm_delete (node->filter) ;
    }
    vl_free(node);
  }
}

/** ------------------------------------------------------------------
 ** @brief New HIKM tree
 ** @param method clustering method.
 ** @return new HIKM tree.
 **/

VlHIKMTree *
vl_hikm_new (int method)
{
  VlHIKMTree *f = vl_calloc (sizeof(VlHIKMTree), 1) ;
  f->max_niters = 200 ;
  f->method = method ;
  return f ;
}

/** ------------------------------------------------------------------
 ** @brief Delete HIKM tree
 ** @param f HIKM tree.
 **/

void
vl_hikm_delete (VlHIKMTree *f)
{
  if (f) {
    xdelete (f->root) ;
    vl_free (f) ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Initialize HIKM tree
 ** @param f HIKM tree.
 ** @param M Data dimensionality.
 ** @param K Number of clusters per node.
 ** @param depth Tree depth.
 ** @return a new HIKM tree representing the clustering.
 **
 ** @remark @a depth cannot be smaller than 1.
 **/

void
vl_hikm_init (VlHIKMTree *f, vl_size M, vl_size K, vl_size depth)
{
  assert(depth > 0) ;
  assert(M > 0) ;
  assert(K > 0) ;

  xdelete (f -> root) ;
  f->root = 0;
  f->M = M ;
  f->K = K ;
  f->depth = depth ;
}

/** ------------------------------------------------------------------
 ** @brief Train HIKM tree
 ** @param f       HIKM tree.
 ** @param data    Data to cluster.
 ** @param N       Number of data.
 **/

void
vl_hikm_train (VlHIKMTree *f, vl_uint8 const *data, vl_size N)
{
  f->root= xmeans (f, data, N, VL_MIN(f->K, N), f->depth) ;
}

/** ------------------------------------------------------------------
 ** @brief Project data down HIKM tree
 ** @param f HIKM tree.
 ** @param asgn Path down the tree (out).
 ** @param data Data to project.
 ** @param N Number of data.
 **
 ** The function writes to @a asgn the path of the data @a data
 ** down the HIKM tree @a f. The parameter @a asgn must point to
 ** an array of @c M by @c N elements, where @c M is the depth of
 ** the HIKM tree and @c N is the number of data point to process.
 **/

void
vl_hikm_push (VlHIKMTree *f, vl_uint32 *asgn, vl_uint8 const *data, vl_size N)
{
  vl_uindex i, d ;
  vl_size M = vl_hikm_get_ndims (f) ;
  vl_size depth = vl_hikm_get_depth (f) ;

  /* for each datum */
  for(i = 0 ; i < N ; i++) {
    VlHIKMNode *node = f->root ;
    d = 0 ;
    while (node) {
      vl_uint32 best ;
      vl_ikm_push (node->filter,
                   &best,
                   data + i * M, 1) ;
      asgn[i * depth + d] = best ;
      ++ d ;
      if (!node->children) break ;
      node = node->children [best] ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                              Setters and getters */
/* ---------------------------------------------------------------- */

/** @brief Get data dimensionality
 ** @param f HIKM tree.
 ** @return data dimensionality.
 **/

vl_size
vl_hikm_get_ndims (VlHIKMTree const* f)
{
  return f->M ;
}

/** @brief Get K
 ** @param f HIKM tree.
 ** @return K.
 **/

vl_size
vl_hikm_get_K (VlHIKMTree const *f)
{
  return f->K ;
}

/** @brief Get depth
 ** @param f HIKM tree.
 ** @return depth.
 **/

vl_size
vl_hikm_get_depth (VlHIKMTree const *f)
{
  return f->depth ;
}


/** @brief Get verbosity level
 ** @param f HIKM tree.
 ** @return verbosity level.
 **/

int
vl_hikm_get_verbosity (VlHIKMTree const *f)
{
  return f->verb ;
}

/** @brief Get maximum number of iterations
 ** @param f HIKM tree.
 ** @return maximum number of iterations.
 **/

vl_size
vl_hikm_get_max_niters (VlHIKMTree const *f)
{
  return f-> max_niters ;
}

/** @brief Get maximum number of iterations
 ** @param f HIKM tree.
 ** @return maximum number of iterations.
 **/

VlHIKMNode const *
vl_hikm_get_root (VlHIKMTree const *f)
{
  return f->root ;
}

/** @brief Set verbosity level
 ** @param f HIKM tree.
 ** @param verb verbosity level.
 **/

void
vl_hikm_set_verbosity (VlHIKMTree *f, int verb)
{
  f->verb = verb ;
}

/** @brief Set maximum number of iterations
 ** @param f HIKM tree.
 ** @param max_niters maximum number of iterations.
 **/

void
vl_hikm_set_max_niters (VlHIKMTree *f, int max_niters)
{
  f->max_niters = max_niters ;
}
