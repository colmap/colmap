/** @file kdtree.h
 ** @brief KD-tree (@ref kdtree)
 ** @author Andrea Vedaldi, David Novotny
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_KDTREE_H
#define VL_KDTREE_H

#include "generic.h"
#include "mathop.h"

#define VL_KDTREE_SPLIT_HEAP_SIZE 5
#define VL_KDTREE_VARIANCE_EST_NUM_SAMPLES 1024

typedef struct _VlKDTreeNode VlKDTreeNode ;
typedef struct _VlKDTreeSplitDimension VlKDTreeSplitDimension ;
typedef struct _VlKDTreeDataIndexEntry VlKDTreeDataIndexEntry ;
typedef struct _VlKDForestSearchState VlKDForestSearchState ;

struct _VlKDTreeNode
{
  vl_uindex parent ;
  vl_index lowerChild ;
  vl_index upperChild ;
  unsigned int splitDimension ;
  double splitThreshold ;
  double lowerBound ;
  double upperBound ;
} ;

struct _VlKDTreeSplitDimension
{
  unsigned int dimension ;
  double mean ;
  double variance ;
} ;

struct _VlKDTreeDataIndexEntry
{
  vl_index index ;
  double value ;
} ;

/** @brief Thresholding method */
typedef enum _VlKDTreeThresholdingMethod
{
  VL_KDTREE_MEDIAN,
  VL_KDTREE_MEAN
} VlKDTreeThresholdingMethod ;

/** @brief Neighbor of a query point */
typedef struct _VlKDForestNeighbor {
  double distance ;   /**< distance to the query point */
  vl_uindex index ;   /**< index of the neighbor in the KDTree data */
} VlKDForestNeighbor ;

typedef struct _VlKDTree
{
  VlKDTreeNode * nodes ;
  vl_size numUsedNodes ;
  vl_size numAllocatedNodes ;
  VlKDTreeDataIndexEntry * dataIndex ;
  unsigned int depth ;
} VlKDTree ;

struct _VlKDForestSearchState
{
  VlKDTree * tree ;
  vl_uindex nodeIndex ;
  double distanceLowerBound ;
} ;

struct _VlKDForestSearcher;

/** @brief KDForest object */
typedef struct _VlKDForest
{
  vl_size dimension ;

  /* random number generator */
  VlRand * rand ;

  /* indexed data */
  vl_type dataType ;
  void const * data ;
  vl_size numData ;
  VlVectorComparisonType distance;
  void (*distanceFunction)(void) ;

  /* tree structure */
  VlKDTree ** trees ;
  vl_size numTrees ;

  /* build */
  VlKDTreeThresholdingMethod thresholdingMethod ;
  VlKDTreeSplitDimension splitHeapArray [VL_KDTREE_SPLIT_HEAP_SIZE] ;
  vl_size splitHeapNumNodes ;
  vl_size splitHeapSize ;
  vl_size maxNumNodes;

  /* query */
  vl_size searchMaxNumComparisons ;
  vl_size numSearchers;
  struct _VlKDForestSearcher * headSearcher ;  /* head of the double linked list with searchers */

} VlKDForest ;

/** @brief ::VlKDForest searcher object */
typedef struct _VlKDForestSearcher
{
  /* maintain a linked list of searchers for later disposal*/
  struct _VlKDForestSearcher * next;
  struct _VlKDForestSearcher * previous;

  vl_uindex * searchIdBook ;
  VlKDForestSearchState * searchHeapArray ;
  VlKDForest * forest;

  vl_size searchNumComparisons;
  vl_size searchNumRecursions ;
  vl_size searchNumSimplifications ;

  vl_size searchHeapNumNodes ;
  vl_uindex searchId ;
} VlKDForestSearcher ;

/** @name Creating, copying and disposing
 ** @{ */
VL_EXPORT VlKDForest * vl_kdforest_new (vl_type dataType,
                                        vl_size dimension, vl_size numTrees, VlVectorComparisonType normType) ;
VL_EXPORT VlKDForestSearcher * vl_kdforest_new_searcher (VlKDForest * kdforest);
VL_EXPORT void vl_kdforest_delete (VlKDForest * self) ;
VL_EXPORT void vl_kdforestsearcher_delete (VlKDForestSearcher * searcher) ;
/** @} */

/** @name Building and querying
 ** @{ */
VL_EXPORT void vl_kdforest_build (VlKDForest * self,
                                  vl_size numData,
                                  void const * data) ;

VL_EXPORT vl_size vl_kdforest_query (VlKDForest * self,
                                     VlKDForestNeighbor * neighbors,
                                     vl_size numNeighbors,
                                     void const * query) ;

VL_EXPORT vl_size vl_kdforest_query_with_array (VlKDForest * self,
                                                vl_uint32 * index,
                                                vl_size numNeighbors,
                                                vl_size numQueries,
                                                void * distance,
                                                void const * queries) ;

VL_EXPORT vl_size vl_kdforestsearcher_query (VlKDForestSearcher * self,
                                             VlKDForestNeighbor * neighbors,
                                             vl_size numNeighbors,
                                             void const * query) ;
/** @} */

/** @name Retrieving and setting parameters
 ** @{ */
VL_EXPORT vl_size vl_kdforest_get_depth_of_tree (VlKDForest const * self, vl_uindex treeIndex) ;
VL_EXPORT vl_size vl_kdforest_get_num_nodes_of_tree (VlKDForest const * self, vl_uindex treeIndex) ;
VL_EXPORT vl_size vl_kdforest_get_num_trees (VlKDForest const * self) ;
VL_EXPORT vl_size vl_kdforest_get_data_dimension (VlKDForest const * self) ;
VL_EXPORT vl_type vl_kdforest_get_data_type (VlKDForest const * self) ;
VL_EXPORT void vl_kdforest_set_max_num_comparisons (VlKDForest * self, vl_size n) ;
VL_EXPORT vl_size vl_kdforest_get_max_num_comparisons (VlKDForest * self) ;
VL_EXPORT void vl_kdforest_set_thresholding_method (VlKDForest * self, VlKDTreeThresholdingMethod method) ;
VL_EXPORT VlKDTreeThresholdingMethod vl_kdforest_get_thresholding_method (VlKDForest const * self) ;
VL_EXPORT VlKDForest * vl_kdforest_searcher_get_forest (VlKDForestSearcher const * self) ;
VL_EXPORT VlKDForestSearcher * vl_kdforest_get_searcher (VlKDForest const * self, vl_uindex pos) ;
/** @} */


/* VL_KDTREE_H */
#endif
