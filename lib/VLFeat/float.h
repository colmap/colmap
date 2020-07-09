/** @file float.h
 ** @brief Float - Template
 ** @author Andrea Vedaldi
 ** @author David Novotny
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
Copyright (C) 2013 David Novotny.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "generic.h"

#undef  T
#undef  SFX
#undef  VSIZE
#undef  VSFX
#undef  VTYPE
#undef  VSIZEavx
#undef  VSFXavx
#undef  VTYPEavx

#if (FLT == VL_TYPE_FLOAT)
#  define T float
#  define SFX f
#elif (FLT == VL_TYPE_DOUBLE)
#  define T double
#  define SFX d
#elif (FLT == VL_TYPE_UINT32)
#  define T vl_uint32
#  define SFX ui32
#elif (FLT == VL_TYPE_INT32)
#  define T vl_int32
#  define SFX i32
#endif

/* ---------------------------------------------------------------- */
/*                                                              AVX */
/* ---------------------------------------------------------------- */

#ifdef __AVX__

#if (FLT == VL_TYPE_FLOAT)
#  define VSIZEavx  8
#  define VSFXavx   s
#  define VTYPEavx  __m256
#elif (FLT == VL_TYPE_DOUBLE)
#  define VSIZEavx  4
#  define VSFXavx   d
#  define VTYPEavx  __m256d
#endif

#define VALIGNEDavx(x) (! (((vl_uintptr)(x)) & 0x1F))

#define VMULavx  VL_XCAT(_mm256_mul_p,     VSFX)
#define VDIVavx  VL_XCAT(_mm256_div_p,     VSFX)
#define VADDavx  VL_XCAT(_mm256_add_p,     VSFX)
#define VHADDavx  VL_XCAT(_mm_hadd_p,     VSFX)
#define VHADD2avx  VL_XCAT(_mm256_hadd_p,     VSFX)
#define VSUBavx  VL_XCAT(_mm256_sub_p,     VSFX)
#define VSTZavx  VL_XCAT(_mm256_setzero_p, VSFX)
#define VLD1avx  VL_XCAT(_mm256_broadcast_s,   VSFX)
#define VLDUavx  VL_XCAT(_mm256_loadu_p,   VSFX)
#define VST1avx  VL_XCAT(_mm256_store_s,   VSFX)
#define VST2avx  VL_XCAT(_mm256_store_p,   VSFX)
#define VST2Uavx VL_XCAT(_mm256_storeu_p,  VSFX)
#define VPERMavx VL_XCAT(_mm256_permute2f128_p,  VSFX)
//#define VCSTavx VL_XCAT( _mm256_castps256_ps128,  VSFX)
#define VCSTavx  VL_XCAT5(_mm256_castp,VSFX,256_p,VSFX,128)

/* __AVX__ */
#endif

/* ---------------------------------------------------------------- */
/*                                                             SSE2 */
/* ---------------------------------------------------------------- */

#ifdef __SSE2__

#if (FLT == VL_TYPE_FLOAT)
#  define VSIZE  4
#  define VSFX   s
#  define VTYPE  __m128
#elif (FLT == VL_TYPE_DOUBLE)
#  define VSIZE  2
#  define VSFX   d
#  define VTYPE  __m128d
#endif

#define VALIGNED(x) (! (((vl_uintptr)(x)) & 0xF))

#define VMAX  VL_XCAT(_mm_max_p,     VSFX)
#define VMUL  VL_XCAT(_mm_mul_p,     VSFX)
#define VDIV  VL_XCAT(_mm_div_p,     VSFX)
#define VADD  VL_XCAT(_mm_add_p,     VSFX)
#define VSUB  VL_XCAT(_mm_sub_p,     VSFX)
#define VSTZ  VL_XCAT(_mm_setzero_p, VSFX)
#define VLD1  VL_XCAT(_mm_load1_p,   VSFX)
#define VLDU  VL_XCAT(_mm_loadu_p,   VSFX)
#define VST1  VL_XCAT(_mm_store_s,   VSFX)
#define VSET1 VL_XCAT(_mm_set_s,     VSFX)
#define VSHU  VL_XCAT(_mm_shuffle_p, VSFX)
#define VNEQ  VL_XCAT(_mm_cmpneq_p,  VSFX)
#define VAND  VL_XCAT(_mm_and_p,     VSFX)
#define VANDN VL_XCAT(_mm_andnot_p,  VSFX)
#define VST2  VL_XCAT(_mm_store_p,   VSFX)
#define VST2U VL_XCAT(_mm_storeu_p,  VSFX)

/* __SSE2__ */
#endif

