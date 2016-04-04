/** @file   qsort-def.h
 ** @brief  QSort preprocessor metaprogram
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file qsort-def.h

@section qsort-def-overview Overview

@ref qsort-def.h is a metaprogram to define specialized instances
of the quick-sort algorithm.

@section qsort-def-usage Usage

@ref qsort-def.h is used to define a specialization of the
::VL_QSORT_sort function that operates
on a given type of array. For instance the code

@code
#define VL_QSORT_type float
#define VL_QSORT_prefix my_qsort
#include <vl/qsort-def.h>
@endcode

defines a function @c my_qsort_sort that operates on an array of floats.

@todo large array compatibility.
**/

#include "host.h"
#include <assert.h>

#ifndef VL_QSORT_prefix
#error "VL_QSORT_prefix must be defined"
#endif

#ifndef VL_QSORT_array
#ifndef VL_QSORT_type
#error "VL_QSORT_type must be defined if VL_QSORT_array is not"
#endif
#define VL_QSORT_array VL_QSORT_type*
#define VL_QSORT_array_const VL_QSORT_type const*
#endif

#ifdef __DOXYGEN__
#define VL_QSORT_prefix  QSortPrefix       /**< Prefix of the qsort functions */
#define VL_QSORT_type    QSortType         /**< Data type of the qsort elements */
#define VL_QSORT_array   QSortType*        /**< Data type of the qsort container */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_QSORT_cmp) || defined(__DOXYGEN__)
#define VL_QSORT_cmp VL_XCAT(VL_QSORT_prefix, _cmp)

/** @brief Compare two array elements
 ** @param array qsort array.
 ** @param indexA index of the first element @c A to compare.
 ** @param indexB index of the second element @c B to comapre.
 ** @return a negative number if @c A<B, 0 if @c A==B, and
 ** a positive number if if @c A>B.
 **/

VL_INLINE VL_QSORT_type
VL_QSORT_cmp
(VL_QSORT_array_const array,
 vl_uindex indexA,
 vl_uindex indexB)
{
  return array[indexA] - array[indexB] ;
}

/* VL_QSORT_cmp */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_QSORT_swap) || defined(__DOXYGEN__)
#define VL_QSORT_swap VL_XCAT(VL_QSORT_prefix, _swap)

/** @brief Swap two array elements
 ** @param array qsort array.
 ** @param indexA index of the first element to swap.
 ** @param indexB index of the second element to swap.
 **
 ** The function swaps the two elements @a a and @ b. The function
 ** uses a temporary element of type ::VL_QSORT_type
 ** and the copy operator @c =.
 **/

VL_INLINE void
VL_QSORT_swap
(VL_QSORT_array array,
 vl_uindex indexA,
 vl_uindex indexB)
{
  VL_QSORT_type t = array [indexA] ;
  array [indexA] = array [indexB] ;
  array [indexB] = t ;
}

/* VL_QSORT_swap */
#endif

/* ---------------------------------------------------------------- */
#if ! defined(VL_QSORT_sort_recursive) || defined(__DOXYGEN__)
#define VL_QSORT_sort_recursive VL_XCAT(VL_QSORT_prefix, _sort_recursive)

/** @brief Sort portion of an array using quicksort
 ** @param array (in/out) pointer to the array.
 ** @param begin first element of the array portion.
 ** @param end last element of the array portion.
 **
 ** The function sorts the array using quick-sort. Note that
 ** @c begin must be not larger than @c end.
 **/

VL_INLINE void
VL_QSORT_sort_recursive
(VL_QSORT_array array, vl_uindex begin, vl_uindex end)
{
  vl_uindex pivot = (end + begin) / 2 ;
  vl_uindex lowPart, i ;

  assert (begin <= end) ;

  /* swap pivot with last */
  VL_QSORT_swap (array, pivot, end) ;
  pivot = end ;

  /*
   Now scan from left to right, moving all element smaller
   or equal than the pivot to the low part
   array[0], array[1], ..., array[lowPart - 1].
   */
  lowPart = begin ;
  for (i = begin; i < end ; ++i) { /* one less */
    if (VL_QSORT_cmp (array, i, pivot) <= 0) {
      /* array[i] must be moved into the low part */
      VL_QSORT_swap (array, lowPart, i) ;
      lowPart ++ ;
    }
  }

  /* the pivot should also go into the low part */
  VL_QSORT_swap (array, lowPart, pivot) ;
  pivot = lowPart ;

  /* do recursion */
  if (pivot > begin) {
    /* note that pivot-1 stays non-negative */
    VL_QSORT_sort_recursive (array, begin, pivot - 1) ;
  }
  if (pivot < end) {
    VL_QSORT_sort_recursive (array, pivot + 1, end) ;
  }
}

/* VL_QSORT_sort_recursive */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_QSORT_sort) || defined(__DOXYGEN__)
#define VL_QSORT_sort VL_XCAT(VL_QSORT_prefix, _sort)

/** @brief Sort array using quicksort
 ** @param array (in/out) pointer to the array.
 ** @param size size of the array.
 **
 ** The function sorts the array using quick-sort.
 **/

VL_INLINE void
VL_QSORT_sort
(VL_QSORT_array array, vl_size size)
{
  assert (size >= 1) ;
  VL_QSORT_sort_recursive (array, 0, size - 1) ;
}

/* VL_QSORT_qsort */
#endif

#undef VL_QSORT_prefix
#undef VL_QSORT_swap
#undef VL_QSORT_sort
#undef VL_QSORT_sort_recursive
#undef VL_QSORT_type
#undef VL_QSORT_array
#undef VL_QSORT_cmp

