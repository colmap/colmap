/** @file mathop_avx.c
 ** @brief mathop for AVX - Definition
 ** @author Andrea Vedaldi, David Novotny
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/* ---------------------------------------------------------------- */
#if ! defined(VL_MATHOP_AVX_INSTANTIATING)

#include "mathop_avx.h"

#undef FLT
#define FLT VL_TYPE_DOUBLE
#define VL_MATHOP_AVX_INSTANTIATING
#include "mathop_avx.c"

#undef FLT
#define FLT VL_TYPE_FLOAT
#define VL_MATHOP_AVX_INSTANTIATING
#include "mathop_avx.c"

/* ---------------------------------------------------------------- */
/* VL_MATHOP_AVX_INSTANTIATING */
#else
#ifndef VL_DISABLE_AVX

#ifndef __AVX__
#error Compiling AVX functions but AVX does not seem to be supported by the compiler.
#endif

#include <immintrin.h>
#include "generic.h"
#include "mathop.h"
#include "float.th"

VL_INLINE T
VL_XCAT(_vl_vhsum_avx_, SFX)(VTYPEavx x)
{
  T acc ;
#if (VSIZEavx == 8)
  {
    //VTYPEavx hsum = _mm256_hadd_ps(x, x);
    //hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
    //_mm_store_ss(&acc, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
    VTYPEavx hsum = VHADD2avx(x, x);
    hsum = VADDavx(hsum, VPERMavx(hsum, hsum, 0x1));
    VST1(&acc, VHADDavx( VCSTavx(hsum), VCSTavx(hsum) ) );
  }
#else
  {
    //VTYPEavx hsum = _mm256_add_pd(x, _mm256_permute2f128_pd(x, x, 0x1));
    VTYPEavx hsum = VADDavx(x, VPERMavx(x, x, 0x1));

    //_mm_store_sd(&acc, _mm_hadd_pd( _mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum) ) );
    VST1(&acc, VHADDavx( VCSTavx(hsum), VCSTavx(hsum) ) );
  }
#endif
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_l2_avx_, SFX)
(vl_size dimension, T const * X, T const * Y)
{

  T const * X_end = X + dimension ;
  T const * X_vec_end = X_end - VSIZEavx + 1 ;
  T acc ;
  VTYPEavx vacc = VSTZavx() ;
  vl_bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y) ;

  if (dataAligned) {
    while (X < X_vec_end) {
      VTYPEavx a = *(VTYPEavx*)X ;
      VTYPEavx b = *(VTYPEavx*)Y ;
      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      vacc = VADDavx(vacc, delta2) ;
      X += VSIZEavx ;
      Y += VSIZEavx ;
    }
  } else {
    while (X < X_vec_end) {
      VTYPEavx a = VLDUavx(X) ;
      VTYPEavx b = VLDUavx(Y) ;
      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      vacc = VADDavx(vacc, delta2) ;
      X += VSIZEavx ;
      Y += VSIZEavx ;
    }
  }

  acc = VL_XCAT(_vl_vhsum_avx_, SFX)(vacc) ;

  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T delta = a - b ;
    acc += delta * delta ;
  }

  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_mahalanobis_sq_avx_, SFX)
(vl_size dimension, T const * X, T const * MU, T const * S)
{
  T const * X_end = X + dimension ;
  T const * X_vec_end = X_end - VSIZEavx + 1 ;
  T acc ;
  VTYPEavx vacc = VSTZavx() ;
  vl_bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(MU) & VALIGNEDavx(S);

  if (dataAligned) {
    while (X < X_vec_end) {
      VTYPEavx a = *(VTYPEavx*)X ;
      VTYPEavx b = *(VTYPEavx*)MU ;
      VTYPEavx c = *(VTYPEavx*)S ;

      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      VTYPEavx delta2div = VMULavx(delta2,c);

      vacc = VADDavx(vacc, delta2div) ;

      X  += VSIZEavx ;
      MU += VSIZEavx ;
      S  += VSIZEavx ;
    }
  } else {
    while (X < X_vec_end) {

      VTYPEavx a = VLDUavx(X) ;
      VTYPEavx b = VLDUavx(MU) ;
      VTYPEavx c = VLDUavx(S) ;

      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      VTYPEavx delta2div = VMULavx(delta2,c);

      vacc = VADDavx(vacc, delta2div) ;

      X  += VSIZEavx ;
      MU += VSIZEavx ;
      S  += VSIZEavx ;
    }
  }

  acc = VL_XCAT(_vl_vhsum_avx_, SFX)(vacc) ;

  while (X < X_end) {
    T a = *X++ ;
    T b = *MU++ ;
    T c = *S++ ;
    T delta = a - b ;
    acc += (delta * delta) * c;
  }

  return acc ;
}

VL_EXPORT void
VL_XCAT(_vl_weighted_mean_avx_, SFX)
(vl_size dimension, T * MU, T const * X, T const  W)
{
  T const * X_end = X + dimension ;
  T const * X_vec_end = X_end - VSIZEavx + 1 ;

  vl_bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(MU);
  VTYPEavx w = VLD1avx (&W) ;

  if (dataAligned) {
    while (X < X_vec_end) {
      VTYPEavx a = *(VTYPEavx*)X ;
      VTYPEavx mu = *(VTYPEavx*)MU ;

      VTYPEavx aw = VMULavx(a, w) ;
      VTYPEavx meanStore = VADDavx(aw, mu);

      *(VTYPEavx *)MU = meanStore;

      X += VSIZEavx ;
      MU += VSIZEavx ;
    }
  } else {
    while (X < X_vec_end) {
      VTYPEavx a  = VLDUavx(X) ;
      VTYPEavx mu = VLDUavx(MU) ;

      VTYPEavx aw = VMULavx(a, w) ;
      VTYPEavx meanStore = VADDavx(aw, mu);

      VST2Uavx(MU,meanStore);

      X += VSIZEavx ;
      MU += VSIZEavx ;
    }
  }

  while (X < X_end) {
    T a = *X++ ;
    *MU += a * W ;
    MU++;
  }
}

VL_EXPORT void
VL_XCAT(_vl_weighted_sigma_avx_, SFX)
(vl_size dimension, T * S, T const * X, T const * Y, T const W)
{
  T const * X_end = X + dimension ;
  T const * X_vec_end = X_end - VSIZEavx + 1 ;

  vl_bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y) & VALIGNEDavx(S);

  VTYPEavx w = VLD1avx (&W) ;

  if (dataAligned) {
    while (X < X_vec_end) {
      VTYPEavx a = *(VTYPEavx*)X ;
      VTYPEavx b = *(VTYPEavx*)Y ;
      VTYPEavx s = *(VTYPEavx*)S ;

      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      VTYPEavx delta2w = VMULavx(delta2, w) ;
      VTYPEavx sigmaStore = VADDavx(s,delta2w);

      *(VTYPEavx *)S = sigmaStore;

      X += VSIZEavx ;
      Y += VSIZEavx ;
      S += VSIZEavx ;
    }
  } else {
    while (X < X_vec_end) {
      VTYPEavx a = VLDUavx(X) ;
      VTYPEavx b = VLDUavx(Y) ;
      VTYPEavx s = VLDUavx(S) ;

      VTYPEavx delta = VSUBavx(a, b) ;
      VTYPEavx delta2 = VMULavx(delta, delta) ;
      VTYPEavx delta2w = VMULavx(delta2, w) ;
      VTYPEavx sigmaStore = VADDavx(s,delta2w);

      VST2Uavx(S,sigmaStore);

      X += VSIZEavx ;
      Y += VSIZEavx ;
      S += VSIZEavx ;
    }
  }

  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T delta = a - b ;
    *S += ((delta * delta)*W) ;
    S++;
  }
}

/* VL_DISABLE_AVX */
#endif
#undef VL_MATHOP_AVX_INSTANTIATING
#endif
