/** @file   shuffle-def.h
 ** @brief  Shuffle preprocessor metaprogram
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file shuffle-def.h

 @todo large array compatibility.
 **/

#include "host.h"
#include "random.h"
#include <assert.h>

#ifndef VL_SHUFFLE_prefix
#error "VL_SHUFFLE_prefix must be defined"
#endif

#ifndef VL_SHUFFLE_array
#ifndef VL_SHUFFLE_type
#error "VL_SHUFFLE_type must be defined if VL_SHUFFLE_array is not"
#endif
#define VL_SHUFFLE_array VL_SHUFFLE_type*
#endif

#ifdef __DOXYGEN__
#define VL_SHUFFLE_prefix  ShufflePrefix       /**< Prefix of the shuffle functions */
#define VL_SHUFFLE_type    ShuffleType         /**< Data type of the shuffle elements */
#define VL_SHUFFLE_array   ShuffleType*        /**< Data type of the shuffle container */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_SHUFFLE_swap) || defined(__DOXYGEN__)
#define VL_SHUFFLE_swap VL_XCAT(VL_SHUFFLE_prefix, _swap)

/** @brief Swap two array elements
 ** @param array shuffle array.
 ** @param indexA index of the first element to swap.
 ** @param indexB index of the second element to swap.
 **
 ** The function swaps the two elements @a a and @ b. The function
 ** uses a temporary element of type ::VL_SHUFFLE_type
 ** and the copy operator @c =.
 **/

VL_INLINE void
VL_SHUFFLE_swap
(VL_SHUFFLE_array array,
 vl_uindex indexA,
 vl_uindex indexB)
{
  VL_SHUFFLE_type t = array [indexA] ;
  array [indexA] = array [indexB] ;
  array [indexB] = t ;
}

/* VL_SHUFFLE_swap */
#endif

/* ---------------------------------------------------------------- */

#if ! defined(VL_SHUFFLE_shuffle) || defined(__DOXYGEN__)
#define VL_SHUFFLE_shuffle VL_XCAT(VL_SHUFFLE_prefix, _shuffle)

/** @brief Shuffle
 ** @param array (in/out) pointer to the array.
 ** @param size size of the array.
 ** @param rand random number generator to use.
 **
 ** The function randomly permutes the array.
 **/

VL_INLINE void
VL_SHUFFLE_shuffle
(VL_SHUFFLE_array array, vl_size size, VlRand * rand)
{
  vl_uindex n = size ;
  while (n > 1) {
    vl_uindex k = vl_rand_uindex (rand, n) ;
    n -- ;
    VL_SHUFFLE_swap (array, n, k) ;
  }
}

/* VL_SHUFFLE_shuffle */
#endif

#undef VL_SHUFFLE_prefix
#undef VL_SHUFFLE_swap
#undef VL_SHUFFLE_shuffle
#undef VL_SHUFFLE_type
#undef VL_SHUFFLE_array
