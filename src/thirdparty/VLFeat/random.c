/** @file random.c
 ** @brief Random number generator - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ------------------------------------------------------------- -->
@page random Random number generator
@author Andrea Vedaldi
@tableofcontents
<!-- ------------------------------------------------------------- -->

The module @ref random.h implements random number generation in
VLFeat.  The generator is based on the popular Mersenne Twister
algorithm @cite{matsumoto98mersenne} (which is the same as MATLAB
random generator from MATLAB version 7.4 onwards).

<!-- ------------------------------------------------------------- -->
@section random-starting Getting started
<!-- ------------------------------------------------------------- -->

In VLFeat, a random number generator is implemented by an object of
type ::VlRand. The simplest way to obtain such an object is to get the
default random generator by

@code
VlRand * rand = vl_get_rand() ;
vl_int32 signedRandomUniformInteger = vl_rand_int31(rand) ;
@code

Note that there is one such generator per thread (see
::vl_get_rand). If more control is desired, a new ::VlRand object can
be easily created.  The object is lightweight, designed to be
allocated on the stack:

@code
VlRand rand ;
vl_rand_init (&rand) ;
@endcode

The generator can be seeded by ::vl_rand_seed and ::vl_rand_seed_by_array.
For instance:

@code
vl_rand_seed (&rand, clock()) ;
@endcode

The generator can be used to obtain random quantities of
various types:

- ::vl_rand_int31, ::vl_rand_uint32 for 32-bit random integers;
- ::vl_rand_real1 for a double in [0,1];
- ::vl_rand_real2 for a double in [0,1);
- ::vl_rand_real3 for a double in (0,1);
- ::vl_rand_res53 for a double in [0,1) with high resolution.

There is no need to explicitly destroy a ::VlRand instance.

**/

#include "random.h"

/*
A C-program for MT19937, with initialization improved 2002/1/26.
Coded by Takuji Nishimura and Makoto Matsumoto.

Before using, initialize the state by using init_genrand(seed)
or init_by_array(init_key, keySize).

Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. The names of its contributors may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Any feedback is very welcome.
http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <stdio.h>
#include <string.h>

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A VL_UINT32_C(0x9908b0df)   /* constant vector a */
#define UPPER_MASK VL_UINT32_C(0x80000000) /* most asignificant w-r bits */
#define LOWER_MASK VL_UINT32_C(0x7fffffff) /* least significant r bits */

/* initializes mt[N] with a seed */

/** @brief Initialise random number generator
 ** @param self number generator.
 **/

void
vl_rand_init (VlRand * self)
{
  memset (self->mt, 0, sizeof(self->mt[0]) * N) ;
  self->mti = N + 1 ;
}

/** @brief Seed the state of the random number generator
 ** @param self random number generator.
 ** @param s seed.
 **/

void
vl_rand_seed (VlRand * self, vl_uint32 s)
{
#define mti self->mti
#define mt self->mt
  mt[0]= s & VL_UINT32_C(0xffffffff);
  for (mti=1; mti<N; mti++) {
    mt[mti] =
      (VL_UINT32_C(1812433253) * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= VL_UINT32_C(0xffffffff);
    /* for >32 bit machines */
  }
#undef mti
#undef mt
}

/** @brief Seed the state of the random number generator by an array
 ** @param self     random number generator.
 ** @param key      array of numbers.
 ** @param keySize  length of the array.
 **/

void
vl_rand_seed_by_array (VlRand * self, vl_uint32 const key [], vl_size keySize)
{
#define mti self->mti
#define mt self->mt
  int i, j, k;
  vl_rand_seed (self, VL_UINT32_C(19650218));
  i=1; j=0;
  k = (N > keySize ? N : (int)keySize);
  for (; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * VL_UINT32_C(1664525)))
      + key[j] + j; /* non linear */
    mt[i] &= VL_UINT32_C(0xffffffff); /* for WORDSIZE > 32 machines */
    i++; j++;
    if (i>=N) { mt[0] = mt[N-1]; i=1; }
    if (j>=(signed)keySize) j=0;
  }
  for (k=N-1; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * VL_UINT32_C(1566083941)))
      - i; /* non linear */
    mt[i] &= VL_UINT32_C(0xffffffff) ; /* for WORDSIZE > 32 machines */
    i++;
    if (i>=N) { mt[0] = mt[N-1]; i=1; }
  }

  mt[0] = VL_UINT32_C(0x80000000); /* MSB is 1; assuring non-zero initial array */
#undef mti
#undef mt
}

/** @brief Randomly permute and array of indexes.
 ** @param self random number generator.
 ** @param array array of indexes.
 ** @param size number of element in the array.
 **
 ** The function uses *Algorithm P*, also known as *Knuth shuffle*.
 **/

void
vl_rand_permute_indexes (VlRand *self, vl_index *array, vl_size size)
{
  vl_index i, j, tmp;
  for (i = size - 1 ; i > 0; i--) {
    /* Pick a random index j in the range 0, i + 1 and swap it with i */
    j = (vl_int) vl_rand_uindex (self, i + 1) ;
    tmp = array[i] ; array[i] = array[j] ; array[j] = tmp ;
  }
}


/** @brief Generate a random UINT32
 ** @param self random number generator.
 ** @return a random number in [0, 0xffffffff].
 **/

vl_uint32
vl_rand_uint32 (VlRand * self)
{
  vl_uint32 y;
  static vl_uint32 mag01[2]={VL_UINT32_C(0x0), MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

#define mti self->mti
#define mt self->mt

  if (mti >= N) { /* generate N words at one time */
    int kk;

    if (mti == N+1)   /* if init_genrand() has not been called, */
      vl_rand_seed (self, VL_UINT32_C(5489)); /* a default initial seed is used */

    for (kk=0;kk<N-M;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];
    }
    for (;kk<N-1;kk++) {
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
      mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];
    }
    y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
    mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & VL_UINT32_C(0x1)];

    mti = 0;
  }

  y = mt[mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & VL_UINT32_C(0x9d2c5680);
  y ^= (y << 15) & VL_UINT32_C(0xefc60000);
  y ^= (y >> 18);

  return (vl_uint32)y;

#undef mti
#undef mt
}
