/** @file random.h
 ** @brief Random number generator (@ref random)
 ** @author Andrea Vedaldi
 ** @see @ref random
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_RANDOM_H
#define VL_RANDOM_H

#include "host.h"

/** @brief Random numbber generator state */
typedef struct _VlRand {
  vl_uint32 mt [624] ;
  vl_uint32 mti ;
} VlRand ;

/** @name Setting and reading the state
 **
 ** @{ */
VL_EXPORT void vl_rand_init (VlRand * self) ;
VL_EXPORT void vl_rand_seed (VlRand * self, vl_uint32 s) ;
VL_EXPORT void vl_rand_seed_by_array (VlRand * self,
                                      vl_uint32 const key [],
                                      vl_size keySize) ;
/** @} */

/** @name Generate random numbers
 **
 ** @{ */
VL_INLINE vl_uint64 vl_rand_uint64 (VlRand * self) ;
VL_INLINE vl_int64  vl_rand_int63  (VlRand * self) ;
VL_EXPORT vl_uint32 vl_rand_uint32 (VlRand * self) ;
VL_INLINE vl_int32  vl_rand_int31  (VlRand * self) ;
VL_INLINE double    vl_rand_real1  (VlRand * self) ;
VL_INLINE double    vl_rand_real2  (VlRand * self) ;
VL_INLINE double    vl_rand_real3  (VlRand * self) ;
VL_INLINE double    vl_rand_res53  (VlRand * self) ;
VL_INLINE vl_uindex vl_rand_uindex (VlRand * self, vl_uindex range) ;
/** @} */

VL_EXPORT void vl_rand_permute_indexes (VlRand * self, vl_index* array, vl_size size) ;

/* ---------------------------------------------------------------- */

/** @brief Generate a random index in a given range
 ** @param self random number generator.
 ** @param range range.
 ** @return an index sampled uniformly at random in the interval [0, @c range - 1]
 **
 ** @remark Currently, this function uses a simple algorithm that
 ** may yield slightly biased samples if @c range is not a power of
 ** two.
 **/

VL_INLINE vl_uindex
vl_rand_uindex (VlRand * self, vl_uindex range)
{
  if (range <= 0xffffffff) {
    /* 32-bit version */
    return (vl_rand_uint32 (self) % (vl_uint32)range) ;
  } else {
    /* 64-bit version */
    return (vl_rand_uint64 (self) % range) ;
  }
}

/** @brief Generate a random UINT64
 ** @param self random number generator.
 ** @return a random number in [0, 0xffffffffffffffff].
 **/

VL_INLINE vl_uint64
vl_rand_uint64 (VlRand * self)
{
  vl_uint64 a = vl_rand_uint32 (self) ;
  vl_uint64 b = vl_rand_uint32 (self) ;
  return (a << 32) | b ;
}

/** @brief Generate a random INT63
 ** @param self random number generator.
 ** @return a random number in [0, 0x7fffffffffffffff].
 **/

VL_INLINE vl_int64
vl_rand_int63 (VlRand * self)
{
  return (vl_int64)(vl_rand_uint64 (self) >> 1) ;
}

/** @brief Generate a random INT31
 ** @param self random number generator.
 ** @return a random number in [0, 0x7fffffff].
 **/

VL_INLINE vl_int32
vl_rand_int31 (VlRand * self)
{
  return (vl_int32)(vl_rand_uint32 (self) >> 1) ;
}

/** @brief Generate a random number in [0,1]
 ** @param self random number generator.
 ** @return a random number.
 **/

VL_INLINE double
vl_rand_real1 (VlRand * self)
{
  return vl_rand_uint32(self)*(1.0/4294967295.0);
  /* divided by 2^32-1 */
}

/** @brief Generate a random number in [0,1)
 ** @param self random number generator.
 ** @return a random number.
 **/

VL_INLINE double
vl_rand_real2 (VlRand * self)
{
  return vl_rand_uint32(self)*(1.0/4294967296.0);
  /* divided by 2^32 */
}

/** @brief Generate a random number in (0,1)
 ** @param self random number generator.
 ** @return a random number.
 **/

VL_INLINE double
vl_rand_real3 (VlRand * self)
{
  return (((double)vl_rand_uint32(self)) + 0.5)*(1.0/4294967296.0);
  /* divided by 2^32 */
}

/** @brief Generate a random number in [0,1) with 53-bit resolution
 ** @param self random number generator.
 ** @return a random number.
 **/

VL_INLINE double
vl_rand_res53 (VlRand * self)
{
  vl_uint32
  a = vl_rand_uint32(self) >> 5,
  b = vl_rand_uint32(self) >> 6 ;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0) ;
}

/* VL_RANDOM_H */
#endif
