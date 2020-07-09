/** @file   heap-def.h
 ** @brief  Heap preprocessor metaprogram
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file heap-def.h

 A heap organizes an array of objects in a priority queue. This module
 is a template metaprogram that defines heap operations on array of
 generic objects, or even generic object containers.

 - @ref heap-def-overview "Overview"
   - @ref heap-def-overview-general "General usage"
 - @ref heap-def-tech "Technical details"

 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
 @section heap-def-overview Overview
 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

 To use @ref heap-def.h one must specify at least a prefix and the data
 type for the heap elements:

 @code
 #define VL_HEAP_prefix  my_heap
 #define VL_HEAP_type    float
 #include <vl/heap-def.h>
 @endcode

 This code fragment defines a number of functions prefixed by
 ::VL_HEAP_prefix, such as @c my_heap_push (::VL_HEAP_push) and @c
 my_heap_pop (::VL_HEAP_pop), that implement the heap operations.
 These functions operate on an array that has type ::VL_HEAP_array.
 By default, this is defined to be:

 @code
 #define VL_HEAP_array VL_HEAP_type*
 #define VL_HEAP_array_const VL_HEAP_type const*
 @endcode

 The array itself is accessed uniquely by means of two functions:

 - ::VL_HEAP_cmp, that compares two array elements. The default
   implementation assumes that ::VL_HEAP_type is numeric.
 - ::VL_HEAP_swap, that swaps two array elements. The default
   implementation assumes that ::VL_HEAP_type can be copied by the @c
   = operator.

 The heap state is a integer @c numElements (of type ::vl_size) counting
 the number of elements of the array that are currently part of the heap
 and the content of the first @c numElements elements of the array. The
 portion of the array that constitutes the heap satisfies a certain
 invariant property (heap property, @ref heap-def-tech). From a user
 viewpoint, the most important consequence is that the first element
 of the array (the one of index 0) is also the smallest (according to
 ::VL_HEAP_cmp).

 Elements are added to the heap by ::VL_HEAP_push and removed from the
 heap by ::VL_HEAP_pop.  A push operation adds to the heap the array
 element immediately after the last element already in the heap
 (i.e. the element of index @c numElements) and increases the number of
 heap elements @c numElements. Elements in the heap are swapped as required in
 order to maintain the heap consistency.  Similarly, a pop operation
 removes the first (smaller) element from the heap and decreases the
 number of heap elements @c numElements.

 The values of nodes currently in the heap can be updated by
 ::VL_HEAP_update. Notice however that using this function requires
 knowing the index of the element that needs to be updated up to the
 swapping operations that the heap performs to maintain
 consistency. Typically, this requires redefining ::VL_HEAP_swap to
 keep track of such changes (@ref heap-def-overview-general).

 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
 @subsection heap-def-overview-general General usage
 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

 The heap container may be mapped to any type by reimplementing
 ::VL_HEAP_cmp and ::VL_HEAP_swap explicitly. For instance
 the following code redefines ::VL_HEAP_cmp to deal with the case
 in which the heap is an array of structures:

 @code
 typedef struct _S { int x ; } S ;
 int s_cmp (S const * v, vl_uindex a, vl_uindex b) {
   return v[a].x - v[b].x ;
 }
 #define VL_HEAP_prefix  s_heap
 #define VL_HEAP_type    S
 #define VL_HEAP_cmp     s_cmp
 #include <vl/heap-def.h>
 @endcode

 In the following example, the heap itself is an arbitrary structure:

 @code
 typedef struct _H { int* array ; } H ;
 int h_cmp (H const * h, vl_uindex a, vl_uindex b) {
   return h->array[a] - h->array[b] ;
 }
 int h_swap (H * h, vl_uindex a, vl_uindex b) {
   int t = h->array[a] ;
   h->array[a] = h->array[b] ;
   h->array[b] = t ;
 }
 #define VL_HEAP_prefix  h_heap
 #define VL_HEAP_swap    h_swap
 #define VL_HEAP_cmp     h_cmp
 #include <vl/heap-def.h>
 @endcode

 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
 @section heap-def-tech Technical details
 <!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

 The heap is organised as a binary tree with the property (<em>heap
 property</em>) that any node is not larger than any of its
 children. In particular, the root is the smallest node.

 @ref heap-def.h uses the standard binary tree representation as a linear
 array. Tree nodes are mapped to array elements as follows:
 <code>array[0]</code> corresponds to the root, <code>array[1]</code>
 and <code>array[2]</code> to the root left and right children and so
 on.  In this way, the tree structure is fully specified by the total
 number of nodes <code>N</code>.

 Assuming that the heap has <code>N</code> nodes (from
 <code>array[0]</code> to <code>array[N-1]</code>), adding the node
 <code>array[N]</code> to the heap is done by a <em>push down</em>
 operation: if the node <code>array[N]</code> is smaller than its
 parent (violating the heap property) it is pushed down by swapping it
 with the parent, and so on recursively.

 Removing the smallest element <code>array[0]</code> with an heap of
 <code>N</code> nodes is done by swapping <code>array[0]</code> with
 <code>array[N-1]</code>. If then <code>array[0]</code> is larger than
 any of its children, it is swapped with the smallest of the two and
 so on recursively (<em>push up</em> operation).

 Restoring the heap property after an element <code>array[i]</code>
 has been modified can be done by a push up or push down operation on
 that node.

 **/

#include "host.h"
#include <assert.h>

#ifndef VL_HEAP_prefix
#error "VL_HEAP_prefix must be defined"
#endif

#ifndef VL_HEAP_array
#ifndef VL_HEAP_type
#error "VL_HEAP_type must be defined if VL_HEAP_array is not"
#endif
#define VL_HEAP_array       VL_HEAP_type*
#define VL_HEAP_array_const VL_HEAP_type const*
#endif

#ifndef VL_HEAP_array_const
#define VL_HEAP_array_const VL_HEAP_array
#endif

#ifdef __DOXYGEN__
#define VL_HEAP_prefix  HeapObject       /**< Prefix of the heap functions */
#define VL_HEAP_type    HeapType         /**< Data type of the heap elements */
#define VL_HEAP_array   HeapType*        /**< Data type of the heap container */
#define VL_HEAP_array   HeapType const*  /**< Const data type of the heap container */
#endif

/* ---------------------------------------------------------------- */

#ifndef VL_HEAP_DEF_H
#define VL_HEAP_DEF_H

/** @internal @brief Get index of parent node
 ** @param index a node index.
 ** @return index of the parent node.
 **/

VL_INLINE vl_uindex
vl_heap_parent (vl_uindex index)
{
  if (index == 0) return 0 ;
  return (index - 1) / 2 ;
}

/** @internal @brief Get index of left child
 ** @param index a node index.
 ** @return index of the left child.
 **/

VL_INLINE vl_uindex
vl_heap_left_child (vl_uindex index)
{
  return 2 * index + 1 ;
}

/** @internal @brief Get index of right child
 ** @param index a node index.
 ** @return index of the right child.
 **/

VL_INLINE vl_uindex
vl_heap_right_child (vl_uindex index)
{
  return vl_heap_left_child (index) + 1 ;
}

/* VL_HEAP_DEF_H */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_cmp) || defined(__DOXYGEN__)
#define VL_HEAP_cmp VL_XCAT(VL_HEAP_prefix, _cmp)

/** @brief Compare two heap elements
 ** @param array heap array.
 ** @param indexA index of the first element @c A to compare.
 ** @param indexB index of the second element @c B to comapre.
 ** @return a negative number if @c A<B, 0 if @c A==B, and
 ** a positive number if if @c A>B.
 **/

VL_INLINE VL_HEAP_type
VL_HEAP_cmp
(VL_HEAP_array_const array,
 vl_uindex indexA,
 vl_uindex indexB)
{
  return array[indexA] - array[indexB] ;
}

/* VL_HEAP_cmp */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_swap) || defined(__DOXYGEN__)
#define VL_HEAP_swap VL_XCAT(VL_HEAP_prefix, _swap)

/** @brief Swap two heap elements
 ** @param array array of nodes.
 ** @param array heap array.
 ** @param indexA index of the first node to swap.
 ** @param indexB index of the second node to swap.
 **
 ** The function swaps the two heap elements @a a and @ b. The function
 ** uses a temporary element and the copy operator, which must be
 ** well defined for the heap elements.
 **/

VL_INLINE void
VL_HEAP_swap
(VL_HEAP_array array,
 vl_uindex indexA,
 vl_uindex indexB)
{
  VL_HEAP_type t = array [indexA] ;
  array [indexA] = array [indexB] ;
  array [indexB] = t ;
}

/* VL_HEAP_swap */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_up) || defined(__DOXYGEN__)
#define VL_HEAP_up VL_XCAT(VL_HEAP_prefix, _up)

/** @brief Heap up operation
 ** @param array pointer to the heap array.
 ** @param heapSize size of the heap.
 ** @param index index of the node to push up.
 **/

VL_INLINE void
VL_HEAP_up
(VL_HEAP_array array, vl_size heapSize, vl_uindex index)
{
  vl_uindex leftIndex  = vl_heap_left_child (index) ;
  vl_uindex rightIndex = vl_heap_right_child (index) ;

  /* no childer: stop */
  if (leftIndex >= heapSize) return ;

  /* only left childer: easy */
  if (rightIndex >= heapSize) {
    if (VL_HEAP_cmp (array, index, leftIndex) > 0) {
      VL_HEAP_swap (array, index, leftIndex) ;
    }
    return ;
  }

  /* both childern */
  {
    if (VL_HEAP_cmp (array, leftIndex, rightIndex) < 0) {
      /* swap with left */
      if (VL_HEAP_cmp (array, index, leftIndex) > 0) {
        VL_HEAP_swap (array, index, leftIndex) ;
        VL_HEAP_up (array, heapSize, leftIndex) ;
      }
    } else {
      /* swap with right */
      if (VL_HEAP_cmp (array, index, rightIndex) > 0) {
        VL_HEAP_swap (array, index, rightIndex) ;
        VL_HEAP_up (array, heapSize, rightIndex) ;
      }
    }
  }
}

/* VL_HEAP_up */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_down) || defined(__DOXYGEN__)
#define VL_HEAP_down VL_XCAT(VL_HEAP_prefix, _down)

/** @brief Heap down operation
 ** @param array pointer to the heap node array.
 ** @param index index of the node to push up.
 **/

VL_INLINE void
VL_HEAP_down
(VL_HEAP_array array, vl_uindex index)
{
  vl_uindex parentIndex ;

  if (index == 0) return  ;

  parentIndex = vl_heap_parent (index) ;

  if (VL_HEAP_cmp (array, index, parentIndex) < 0) {
    VL_HEAP_swap (array, index, parentIndex) ;
    VL_HEAP_down (array, parentIndex) ;
  }
}

/* VL_HEAP_down */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_push) || defined(__DOXYGEN__)
#define VL_HEAP_push VL_XCAT(VL_HEAP_prefix, _push)

/** @brief Heap push operation
 ** @param array pointer to the heap array.
 ** @param heapSize (in/out) size of the heap.
 **
 ** The function adds to the heap the element of index @c heapSize
 ** and increments @c heapSize.
 **/

VL_INLINE void
VL_HEAP_push
(VL_HEAP_array array, vl_size *heapSize)
{
  VL_HEAP_down (array, *heapSize) ;
  *heapSize += 1 ;
}

/* VL_HEAP_push */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_pop) || defined(__DOXYGEN__)
#define VL_HEAP_pop VL_XCAT(VL_HEAP_prefix, _pop)

/** @brief Heap pop operation
 ** @param array pointer to the heap array.
 ** @param heapSize (in/out) size of the heap.
 ** @return index of the popped element.
 **
 ** The function extracts from the heap the element of index 0
 ** (the smallest element) and decreases @c heapSize.
 **
 ** The element extracted is moved as the first element after
 ** the heap end (thus it has index @c heapSize). For convenience,
 ** this index is returned by the function.
 **
 ** Popping from an empty heap is undefined.
 **/

VL_INLINE vl_uindex
VL_HEAP_pop
(VL_HEAP_array array, vl_size *heapSize)
{
  assert (*heapSize) ;

  *heapSize -= 1 ;

  VL_HEAP_swap (array, 0, *heapSize) ;

  if (*heapSize > 1) {
    VL_HEAP_up (array, *heapSize, 0) ;
  }

  return *heapSize ;
}

/* VL_HEAP_pop */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_HEAP_update) || defined(__DOXYGEN__)
#define VL_HEAP_update VL_XCAT(VL_HEAP_prefix, _update)

/** @brief Heap update operation
 ** @param array pointer to the heap array.
 ** @param heapSize size of the heap.
 ** @param index index of the node to update.
 **
 ** The function updates the heap to account for a change to the
 ** element of index @c index in the heap.
 **
 ** Notice that using this
 ** function requires knowing the index of the heap index of
 ** element that was changed. Since the heap swaps elements in the
 ** array, this is in general different from the index that that
 ** element had originally.
 **/

VL_INLINE void
VL_HEAP_update
(VL_HEAP_array array,
 vl_size heapSize,
 vl_uindex index)
{
  VL_HEAP_up (array, heapSize, index) ;
  VL_HEAP_down (array, index) ;
}

/* VL_HEAP_update */
#endif

/* ---------------------------------------------------------------- */

#undef VL_HEAP_cmp
#undef VL_HEAP_swap
#undef VL_HEAP_up
#undef VL_HEAP_down
#undef VL_HEAP_push
#undef VL_HEAP_pop
#undef VL_HEAP_update
#undef VL_HEAP_prefix
#undef VL_HEAP_type
#undef VL_HEAP_array
#undef VL_HEAP_array_const
