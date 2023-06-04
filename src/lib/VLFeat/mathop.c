/** @file mathop.c
 ** @brief Math operations - Definition
 ** @author Andrea Vedaldi, David Novotny
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@page mathop Mathematical operations
@author Andrea Vedaldi
@author Brian Fulkerson
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

VLFeat include several low-level routines to speedup common
mathematical operations used throughout the library. Most are
collected in the @ref mathop.h module.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section mathop-usage-vector-comparison Comparing vectors
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

@ref mathop.h includes a number of functions to quickly compute
distances or similarity of pairs of vector. Applications include
clustering and evaluation of SVM-like classifiers.

Use ::vl_get_vector_comparison_function_f or
::vl_get_vector_comparison_function_d obtain an approprite function
to comprare vectors of floats or doubles, respectively.  Such
functions are usually optimized (for instance, on X86 platforms they
use the SSE vector extension) and are several times faster than a
naive implementation.  ::vl_eval_vector_comparison_on_all_pairs_f and
::vl_eval_vector_comparison_on_all_pairs_d can be used to evaluate
the comparison function on all pairs of one or two sequences of
vectors.

Let @f$ \mathbf{x} = (x_1,\dots,x_d) @f$ and @f$ \mathbf{y} =
(y_1,\dots,y_d) @f$ be two vectors.  The following comparison
functions are supported:

<table>
<tr>
<td>@f$ l^1 @f$</td>
<td>::VlDistanceL1</td>
<td>@f$ \sum_{i=1}^d |x_i - y_i| @f$</td>
<td>l1 distance (squared intersection metric)</td>
</tr>
<tr>
<td>@f$ l^2 @f$</td>
<td>::VlDistanceL2</td>
<td>@f$\sum_{i=1}^d (x_i - y_i)^2@f$</td>
<td>Squared Euclidean disance</td>
</tr>
<tr>
<td>@f$ \chi^2 @f$</td>
<td>::VlDistanceChi2</td>
<td>@f$\sum_{i=1}^d \frac{(x_i - y_i)^2}{x_i + y_i}@f$</td>
<td>Squared chi-square distance</td>
</tr>
<tr>
<td>-</td>
<td>::VlDistanceHellinger</td>
<td>@f$\sum_{i=1}^d (\sqrt{x_i} - \sqrt{y_i})^2@f$</td>
<td>Squared Hellinger's distance</td>
</tr>
<tr>
<td>-</td>
<td>::VlDistanceJS</td>
<td>@f$
\sum_{i=1}^d
\left(
  x_i \log\frac{2x_i}{x_i+y_i}
+ y_i \log\frac{2y_i}{x_i+y_i}
\right)
@f$
</td>
<td>Squared Jensen-Shannon distance</td>
</tr>
<tr>
<td>@f$ l^1 @f$</td>
<td>::VlKernelL1</td>
<td>@f$ \sum_{i=1}^d \min\{ x_i, y_i \} @f$</td>
<td>intersection kernel</td>
</tr>
<tr>
<td>@f$ l^2 @f$</td>
<td>::VlKernelL2</td>
<td>@f$\sum_{i=1}^d x_iy_i @f$</td>
<td>linear kernel</td>
</tr>
<tr>
<td>@f$ \chi^2 @f$</td>
<td>::VlKernelChi2</td>
<td>@f$\sum_{i=1}^d 2 \frac{x_iy_i}{x_i + y_i}@f$</td>
<td>chi-square kernel</td>
</tr>
<tr>
<td>-</td>
<td>::VlKernelHellinger</td>
<td>@f$\sum_{i=1}^d 2 \sqrt{x_i y_i}@f$</td>
<td>Hellinger's kernel (Bhattacharya coefficient)</td>
</tr>
<tr>
<td>-</td>
<td>::VlKernelJS</td>
<td>@f$
\sum_{i=1}^d
\left(
  \frac{x_i}{2} \log_2\frac{x_i+y_i}{x_i}
+ \frac{y_i}{2} \log_2\frac{x_i+y_i}{y_i}
\right)
@f$
</td>
<td>Jensen-Shannon kernel</td>
</tr>
</table>

@remark The definitions have been choosen so that corresponding kernels and
distances are related by the equation:
@f[
 d^2(\mathbf{x},\mathbf{y})
 =
 k(\mathbf{x},\mathbf{x})
 +k(\mathbf{y},\mathbf{y})
 -k(\mathbf{x},\mathbf{y})
 -k(\mathbf{y},\mathbf{x})
@f]
This means that each of these distances can be interpreted as a
squared distance or metric in the corresponding reproducing kernel
Hilbert space. Notice in particular that the @f$ l^1 @f$ or Manhattan
distance is also a <em>squared</em> distance in this sense.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section mathop-integer-ops Fast basic functions operations
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

In certain algorithm it is useful to quickly compute integer
approximation of certain mathematical operations. Presently, VLFeat
includes and implementations of:

- Fast single precision atan2: ::vl_fast_sqrt_f.
- Fast inverse square root: ::vl_fast_resqrt_f, ::vl_fast_resqrt_d.
- Fast square root: ::vl_fast_sqrt_f, ::vl_fast_sqrt_d.
- Fast integer square root: ::vl_fast_sqrt_ui16, ::vl_fast_sqrt_ui32,
  ::vl_fast_sqrt_ui64 (see also @subpage mathop-sqrti).
**/

/** @fn vl_get_vector_comparison_function_f(VlVectorComparisonType)
 **
 ** @brief Get vector comparison function from comparison type
 ** @param type vector comparison type.
 ** @return comparison function.
 **/

/** @fn vl_get_vector_comparison_function_d(VlVectorComparisonType)
 ** @brief Get vector comparison function from comparison type
 ** @sa vl_get_vector_comparison_function_f
 **/

/** @fn vl_eval_vector_comparison_on_all_pairs_f(float*,vl_size,
 **     float const*,vl_size,float const*,vl_size,VlFloatVectorComparisonFunction)
 **
 ** @brief Evaluate vector comparison function on all vector pairs
 ** @param result comparison matrix (output).
 ** @param dimension number of vector components (rows of @a X and @a Y).
 ** @param X data matrix X.
 ** @param Y data matrix Y.
 ** @param numDataX number of vectors in @a X (columns of @a X)
 ** @param numDataY number of vectros in @a Y (columns of @a Y)
 ** @param function vector comparison function.
 **
 ** The function evaluates @a function on all pairs of columns
 ** from matrices @a X and @a Y, filling a @a numDataX by @a numDataY
 ** matrix.
 **
 ** If @a Y is a null pointer the function compares all columns from
 ** @a X with themselves.
 **/

/** @fn vl_eval_vector_comparison_on_all_pairs_d(double*,vl_size,
 **     double const*,vl_size,double const*,vl_size,VlDoubleVectorComparisonFunction)
 ** @brief Evaluate vector comparison function on all vector pairs
 ** @sa vl_eval_vector_comparison_on_all_pairs_f
 **/

/**
@page mathop-sqrti Fast integer square root algorithm
@tableofcontents

This section describes the fast integer square root algorithm used by
vl_fast_sqrt_ui8, ::vl_fast_sqrt_ui16, ::vl_fast_sqrt_ui32,
::vl_fast_sqrt_ui64.

Given a non-negative integer $x \in \mathbb{Z}_+$, the goal of this
algorithm is to quickly compute the integer approximation of the
square root of an integer number:

\[
y = \max_{\bar y\in\mathbb{Z}} \bar y, \qquad \text{such that}\  \bar y^2 \leq x.
\]

Consider determining the k-th bit of $y$. To this end, decompose $y$
in three parts:

\[
y = y_{k+1} + q 2^k + r,
\qquad \text{where}\   y_{k+1} \geq 2^{k+1}, r < 2^k,
\]

and $q\in\{0,1\}$ is the bit to be determined. Here $y_{k+1}$ is a part
of the result $y$ that has already been determined, while the bit $q$
and the remainder $r$ are still unknown. Recall that the goal is to
find the largest $y^2$ such that $y^2 \leq x$. Expanding $y^2$ this
condition becomes

\[
q (2^{2k} + 2 y_{k+1} 2^k) + r(r + 2q 2^k + 2 y_{k+1}) \leq x - y_{k+1}^2.
\]

We can now determine if $q=1$ or $q=0$ based on the value of the
residual $x - y_{k+1}^2$. Specifically, $q=1$ requires that:

\[
\boxed{
2^{2k} + 2a2^k \leq x - y_{k+1}^2.
}
\]

On the other hand, if this equation is satisfied, then setting $r=0$
shows that there exists at least one $y$ such that $q=1$ and $y^2 \leq
x$. In particular, greedily choosing $q=1$ in $x=y_{k+1} + 2^k q + r$ is
optimal because $2^k > r$. This yields the algorithm:

1. Note that if $x$ is stored in $n$ bits and $n$ is even, then the
   integer square root $y$ does not require more than $m = n / 2$ bit
   to be stored. Thus the first bit to be determined is $k \leftarrow
   m - 1 = n/2 - 1$ and $y_{n/2}=0$.
2. The algorithm stores and updates $y_k/2^{k}$ and $x - y_{k}^2$ for
   convenience.
3. During iteration $k$, $y_k$ is determined. On entering the
   iteration, the first step is to compute $y_{k+1}/2^k = 2
   y_{k+1}/2^{k+1}$.
4. Then the bound $t = (2^{2k} + 2 y_{k+1})2^k = 2^{2k}(1 + 2
   y_{k+1}/2^k)$.
5. If $t \geq x - y_{k+1}$, the $k$-th bit of $y_k$ is set to
   one. This means applying the update $\hat y_{k}/2^k \leftarrow
   y_{k+1}/2^k + 1$. This also requires computing $x - y_{k}^2
   \leftarrow x - y_{k+1}^2 - t$.
6. Decrement $k \leftarrow k -1$ and, if $k\geq 0$, continue from 3.

**/

/* ---------------------------------------------------------------- */
#ifndef VL_MATHOP_INSTANTIATING

#include "mathop.h"
#include "mathop_sse2.h"
 #include "mathop_avx.h"
#include <math.h>

#undef FLT
#define FLT VL_TYPE_FLOAT
#define VL_MATHOP_INSTANTIATING
#include "mathop.c"

#undef FLT
#define FLT VL_TYPE_DOUBLE
#define VL_MATHOP_INSTANTIATING
#include "mathop.c"
#endif

/* ---------------------------------------------------------------- */
#ifdef VL_MATHOP_INSTANTIATING
#include "float.h"

#undef COMPARISONFUNCTION_TYPE
#undef COMPARISONFUNCTION3_TYPE
#if (FLT == VL_TYPE_FLOAT)
#  define COMPARISONFUNCTION_TYPE VlFloatVectorComparisonFunction
#  define COMPARISONFUNCTION3_TYPE VlFloatVector3ComparisonFunction
#else
#  define COMPARISONFUNCTION_TYPE VlDoubleVectorComparisonFunction
#  define COMPARISONFUNCTION3_TYPE VlDoubleVector3ComparisonFunction
#endif

/* ---------------------------------------------------------------- */

VL_EXPORT T
VL_XCAT(_vl_distance_l2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *Y++ ;
    acc += d * d ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_l1_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *Y++ ;
    acc += VL_MAX(d, -d) ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_chi2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T delta = a - b ;
    T denom = (a + b) ;
    T numer = delta * delta ;
    if (denom) {
      T ratio = numer / denom ;
      acc += ratio ;
    }
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_hellinger_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
#if (FLT == VL_TYPE_FLOAT)
    acc += a + b - 2.0 * sqrtf (a*b) ;
#else
    acc += a + b - 2.0 * sqrt (a*b) ;
#endif
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_js_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T x = *X++ ;
    T y = *Y++ ;
    if (x) acc += x - x * VL_XCAT(vl_log2_,SFX)(1 + y/x) ;
    if (y) acc += y - y * VL_XCAT(vl_log2_,SFX)(1 + x/y) ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_l2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    acc += a * b ;
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_l1_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T a_ = VL_XCAT(vl_abs_, SFX) (a) ;
    T b_ = VL_XCAT(vl_abs_, SFX) (b) ;
    acc += a_ + b_ - VL_XCAT(vl_abs_, SFX) (a - b) ;
  }
  return acc / ((T)2) ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_chi2_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
    T denom = (a + b) ;
    if (denom) {
      T numer = 2 * a * b ;
      T ratio = numer / denom ;
      acc += ratio ;
    }
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_hellinger_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T a = *X++ ;
    T b = *Y++ ;
#if (FLT == VL_TYPE_FLOAT)
    acc += sqrtf (a*b) ;
#else
    acc += sqrt (a*b) ;
#endif
  }
  return acc ;
}

VL_EXPORT T
VL_XCAT(_vl_kernel_js_, SFX)
(vl_size dimension, T const * X, T const * Y)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T x = *X++ ;
    T y = *Y++ ;
    if (x) acc += x * VL_XCAT(vl_log2_,SFX)(1 + y/x) ;
    if (y) acc += y * VL_XCAT(vl_log2_,SFX)(1 + x/y) ;
  }
  return (T)0.5 * acc ;
}

VL_EXPORT T
VL_XCAT(_vl_distance_mahalanobis_sq_, SFX)
(vl_size dimension, T const * X, T const * MU, T const * S)
{
  T const * X_end = X + dimension ;
  T acc = 0.0 ;
  while (X < X_end) {
    T d = *X++ - *MU++ ;
    acc += d * d * (*S++) ;
  }
  return acc ;
}

/* ---------------------------------------------------------------- */

VL_EXPORT COMPARISONFUNCTION_TYPE
VL_XCAT(vl_get_vector_comparison_function_, SFX)(VlVectorComparisonType type)
{
  COMPARISONFUNCTION_TYPE function = 0 ;
  switch (type) {
    case VlDistanceL2        : function = VL_XCAT(_vl_distance_l2_,             SFX) ; break ;
    case VlDistanceL1        : function = VL_XCAT(_vl_distance_l1_,             SFX) ; break ;
    case VlDistanceChi2      : function = VL_XCAT(_vl_distance_chi2_,           SFX) ; break ;
    case VlDistanceHellinger : function = VL_XCAT(_vl_distance_hellinger_,      SFX) ; break ;
    case VlDistanceJS        : function = VL_XCAT(_vl_distance_js_,             SFX) ; break ;
    case VlKernelL2          : function = VL_XCAT(_vl_kernel_l2_,               SFX) ; break ;
    case VlKernelL1          : function = VL_XCAT(_vl_kernel_l1_,               SFX) ; break ;
    case VlKernelChi2        : function = VL_XCAT(_vl_kernel_chi2_,             SFX) ; break ;
    case VlKernelHellinger   : function = VL_XCAT(_vl_kernel_hellinger_,        SFX) ; break ;
    case VlKernelJS          : function = VL_XCAT(_vl_kernel_js_,               SFX) ; break ;
    default: abort() ;
  }

#ifndef VL_DISABLE_SSE2
  /* if a SSE2 implementation is available, use it */
  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceL2    : function = VL_XCAT(_vl_distance_l2_sse2_,             SFX) ; break ;
      case VlDistanceL1    : function = VL_XCAT(_vl_distance_l1_sse2_,             SFX) ; break ;
      case VlDistanceChi2  : function = VL_XCAT(_vl_distance_chi2_sse2_,           SFX) ; break ;
      case VlKernelL2      : function = VL_XCAT(_vl_kernel_l2_sse2_,               SFX) ; break ;
      case VlKernelL1      : function = VL_XCAT(_vl_kernel_l1_sse2_,               SFX) ; break ;
      case VlKernelChi2    : function = VL_XCAT(_vl_kernel_chi2_sse2_,             SFX) ; break ;
      default: break ;
    }
  }
#endif

#ifndef VL_DISABLE_AVX
  /* if an AVX implementation is available, use it */
  if (vl_cpu_has_avx() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceL2    : function = VL_XCAT(_vl_distance_l2_avx_,             SFX) ; break ;
      default: break ;
    }
  }
#endif

  return function ;
}

/* ---------------------------------------------------------------- */

VL_EXPORT COMPARISONFUNCTION3_TYPE
VL_XCAT(vl_get_vector_3_comparison_function_, SFX)(VlVectorComparisonType type)
{
  COMPARISONFUNCTION3_TYPE function = 0 ;
  switch (type) {
    case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_, SFX) ; break ;
    default: abort() ;
  }

#ifndef VL_DISABLE_SSE2
  /* if a SSE2 implementation is available, use it */
  if (vl_cpu_has_sse2() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_sse2_, SFX) ; break ;
      default: break ;
    }
  }
#endif

#ifndef VL_DISABLE_AVX
  /* if an AVX implementation is available, use it */
  if (vl_cpu_has_avx() && vl_get_simd_enabled()) {
    switch (type) {
      case VlDistanceMahalanobis : function = VL_XCAT(_vl_distance_mahalanobis_sq_avx_, SFX) ; break ;
      default: break ;
    }
  }
#endif

  return function ;
}

/* ---------------------------------------------------------------- */

VL_EXPORT void
VL_XCAT(vl_eval_vector_comparison_on_all_pairs_, SFX)
(T * result, vl_size dimension,
 T const * X, vl_size numDataX,
 T const * Y, vl_size numDataY,
 COMPARISONFUNCTION_TYPE function)
{
  vl_uindex xi ;
  vl_uindex yi ;

  if (dimension == 0) return ;
  if (numDataX == 0) return ;
  assert (X) ;

  if (Y) {
    if (numDataY == 0) return ;
    for (yi = 0 ; yi < numDataY ; ++ yi) {
      for (xi = 0 ; xi < numDataX ; ++ xi) {
        *result++ = (*function)(dimension, X, Y) ;
        X += dimension ;
      }
      X -= dimension * numDataX ;
      Y += dimension ;
    }
  } else {
    T * resultTransp = result ;
    Y = X ;
    for (yi = 0 ; yi < numDataX ; ++ yi) {
      for (xi = 0 ; xi <= yi ; ++ xi) {
        T z = (*function)(dimension, X, Y) ;
        X += dimension ;
        *result       = z ;
        *resultTransp = z ;
        result        += 1 ;
        resultTransp  += numDataX ;
      }
      X -= dimension * (yi + 1) ;
      Y += dimension ;
      result       += numDataX - (yi + 1) ;
      resultTransp += 1        - (yi + 1) * numDataX ;
    }
  }
}

/* VL_MATHOP_INSTANTIATING */
#endif


/* ---------------------------------------------------------------- */
/*                                               Numerical analysis */
/* ---------------------------------------------------------------- */

#ifndef VL_MATHOP_INSTANTIATING

/** @brief SVD of a 2x2 real matrix
 ** @param S 2x2 real diagonal matrix of the singular values (out).
 ** @param U first 2x2 real orthonormal matrix (out).
 ** @param V second 2x2 real orthonormal matrix (out).
 ** @param M 2x2 matrix.
 **
 ** The function comptues the SVD decomposition of the 2x2
 ** real matrix @f$ M @f$:
 ** @f[
 **    M = U \operatorname S V^\top
 ** @f]
 ** where @f$ U @f$ and @f$ V @f$ are real orthonormal matrices
 ** and @f$ S @f$ is the diagonal matrix of the singular values
 ** in decreasing order.
 **
 ** @par Algorithm
 **
 ** The fist step is to find rotation matrices @f$ U_1 @f$ and
 ** @f$ V_1 @f$ such taht
 ** @f[
 **     M = U_1 R V_1^\top, \quad
 **     U_1 = \begin{barray} c_{u1} & - s_{u1} \\ s_{u1} & c_{u1} \end{barray}, \quad
 **     V_1 = \begin{barray} c_{v1} & - s_{v1} \\ s_{v1} & c_{v1} \end{barray}, \quad
 **     R = \begin{barray} f & g \\ 0 & h \end{barray}.
 ** @f]
 ** Gives a 2x2 triangular matrix. The second step is to call
 ** ::vl_lapack_dlasv2 on the matrix @f$ R @f$ obtaining
 ** @f[
 **   M = U_1 (U_2 S V_2^\top) V_2.
 ** @f]
 **/

void
vl_svd2 (double* S, double *U, double *V, double const *M)
{
  double m11 = M[0] ;
  double m21 = M[1] ;
  double m12 = M[2] ;
  double m22 = M[3] ;
  double cu1 = m11 ;
  double su1 = m21 ;
  double norm = sqrt(cu1*cu1 + su1*su1) ;
  double cu2, su2, cv2, sv2 ;
  double f, g, h ;
  double smin, smax ;
  cu1 /= norm ;
  su1 /= norm ;

  f = cu1 * m11 + su1 * m21 ;
  g = cu1 * m12 + su1 * m22 ;
  h = - su1 * m12 + cu1 * m22 ;

  vl_lapack_dlasv2 (&smin, &smax,
                    &sv2, &cv2,
                    &su2, &cu2,
                    f, g, h) ;

  assert(S) ;
  S[0] = smax ;
  S[1] = 0 ;
  S[2] = 0 ;
  S[3] = smin ;

  if (U) {
    U[0] = cu2*cu1 - su2*su1 ;
    U[1] = su2*cu1 + cu2*su1 ;
    U[2] = - cu2*su1 - su2*cu1 ;
    U[3] = - su2*su1 + cu2*cu1 ;
  }
  if (V) {
    V[0] = cv2 ;
    V[1] = sv2 ;
    V[2] = - sv2 ;
    V[3] = cv2 ;
  }
}

/** @brief SVD of a 2x2 upper triangular matrix (LAPACK @c dlasv2 equivalent)
 ** @param smin smallest (in modulus) singular value (out).
 ** @param smax largest (in modulus) singuarl value (out).
 ** @param sv second component of the right singular vector of @c smax (out).
 ** @param cv first component of the right singular vector of @c smax (out).
 ** @param su second component of the left singular vector of @c smax (out).
 ** @param cu first component of the left singular vector of @c smax (out).
 ** @param f first entry of the upper triangular matrix.
 ** @param g second entry of the upper triangular matrix.
 ** @param h third entry of the upper triangular matrix.
 **
 ** @f[
 **  \begin{bmatrix} f & g \\ 0 & h \end{bmatrix}
 **  =
 **  \begin{bmatrix} cv & - sv \\ sv & cv \end{bmatrix}
 **  \begon{bmatrix} smax & 0 \\ 0 & smin \end{bmatrix}
 **  \begin{bmatrix} cv & - sv \\ sv & cv \end{bmatrix}
 ** @f]
 **
 ** Z.Bai and J.Demmel,
 ** "Computing the Generalized Singular Value Decomposition",
 ** SIAM J. Sci. Comput., Vol. 14, No. 6, pp. 1464-1486, November 1993
 **/

#define isign(i) ((i)<0 ? (-1) : (+1))  /* integer sign function */
#define sign(x) ((x)<0.0 ? (-1) : (+1)) /* double sign function */

void
vl_lapack_dlasv2 (double *smin,
                  double *smax,
                  double *sv,
                  double *cv,
                  double *su,
                  double *cu,
                  double f,
                  double g,
                  double h)
{
  double svt, cvt, sut, cut; /* temporary sv, cv, su, and cu */
  double ft = f, gt = g, ht = h; /* temporary f, g, h */
  double fa = fabs(f), ga = fabs(g), ha = fabs(h); /* |f|, |g|, and |h| */
  int pmax = 1 ; /* pointer to max abs entry */
  int swap = 0 ; /* is swapped */
  int glarge = 0 ; /* is g very large */
  int tsign ; /* tmp sign */
  double fmh ; /* |f| -|h| */
  double d ; /* (|f| -|h|)/|f| */
  double dd ; /* d*d */
  double q ; /* g/f */
  double qq ; /* q*q */
  double s ; /* (|f| + |h|)/|f| */
  double ss ; /* s*s */
  double spq ; /* sqrt(ss + qq) */
  double dpq ; /* sqrt(dd + qq) */
  double a ; /* (spq + dpq)/2 */
  double tmp ; /* temporaries */
  double tt;

  /* make fa >= ha */
  if (fa < ha) {
    pmax = 3 ;
    tmp =ft ; ft = ht ; ht = tmp ; /* swap ft and ht */
    tmp =fa ; fa = ha ; ha = tmp ; /* swap fa and ha */
    swap = 1 ;
  }

  if (ga == 0.0) { /* diagonal */
    *smin = ha ;
    *smax = fa ;
    /* identity matrix */
    cut = 1.0 ; sut = 0.0 ;
    cvt = 1.0 ; svt = 0.0 ;
  }
  else { /* not diagonal */
    if (ga > fa) { /* g is the largest entry */
      pmax = 2 ;
      if ((fa / ga) < VL_EPSILON_D) { /* g is very large */
        glarge = 1 ;
        *smax = ga ; /* 1 ulp */
        if (ha > 1.0) {
          *smin = fa / (ga / ha) ; /* 2 ulps */
        } else {
          *smin = (fa / ga) * ha ; /* 2 ulps */
        }
        cut = 1.0 ; sut = ht / gt ;
        cvt = 1.0 ; svt = ft / gt ;
      }
    }

    if (glarge == 0) { /* normal case */
      fmh = fa - ha ; /* 1ulp */
      if (fmh == fa) {  /* cope with infinite f or h */
        d = 1.0 ;
      } else {
        d = fmh / fa ; /* note 0<=d<=1.0, 2 ulps */
      }
      q = gt / ft ; /* note |q|<1/EPS, 1 ulp */
      s = 2.0 - d ; /* note s>=1.0, 3 ulps */
      dd = d*d ;
      qq = q*q ;
      ss = s*s ;
      spq = sqrt(ss + qq) ; /* note 1<=spq<=1+1/EPS, 5 ulps */
      if (d == 0.0) {
        dpq = fabs(q) ; /* 0 ulp */
      } else {
        dpq = sqrt(dd + qq) ; /* note 0<=dpq<=1+1/EPS, 3.5 ulps */
      }
      a = 0.5 * (spq + dpq) ; /* note 1<=a<=1 + |q|, 6 ulps */
      *smin = ha / a; /* 7 ulps */
      *smax = fa * a; /* 7 ulps */
      if (qq==0.0) { /* qq underflow */
        if (d==0.0) {
          tmp = sign(ft)*2*sign(gt); /* 0ulp */
        }
        else {
          tmp = gt/(sign(ft)*fmh) + q/s; /* 6 ulps */
        }
      } else {
        tmp = (q/(spq + s) + q/(dpq + d))*(1.0 + a);  /* 17 ulps */
      }
      /* if qq */
      tt = sqrt(tmp*tmp + 4.0) ; /* 18.5 ulps */
      cvt = 2.0 / tt ; /* 19.5 ulps */
      svt = tmp / tt ; /* 36.5 ulps */
      cut = (cvt + svt*q) / a ; /* 46.5 ulps */
      sut = (ht / ft) * svt / a ; /* 45.5 ulps */
    } /* if g not large */
  } /* if ga */
  if (swap == 1) {
    *cu = svt ; *su = cvt ;
    *cv = sut ; *sv = cut ;
  } else {
    *cu = cut ; *su = sut ;
    *cv = cvt ; *sv = svt ;
  }
  /* correct the signs of smax and smin */
  if (pmax==1) { tsign = sign(*cv) * sign(*cu) * sign(f) ; }
  if (pmax==2) { tsign = sign(*sv) * sign(*cu) * sign(g) ; }
  if (pmax==3) { tsign = sign(*sv) * sign(*su) * sign(h) ; }
  *smax = isign(tsign) * (*smax);
  *smin = isign(tsign * sign(f) * sign(h)) * (*smin) ;
}


/** @brief Solve a 3x3 linear system
 ** @param x result.
 ** @param A system matrix.
 ** @param b coefficients.
 **
 ** The function computes a solution to @f$ Ax =b @f$ for a 3x3
 ** matrix.
 **/

VL_EXPORT int
vl_solve_linear_system_3 (double * x, double const * A, double const *b)
{
  int err ;
  double M[3*4] ;
  M[0] = A[0] ;
  M[1] = A[1] ;
  M[2] = A[2] ;
  M[3] = A[3] ;
  M[4] = A[4] ;
  M[5] = A[5] ;
  M[6] = A[6] ;
  M[7] = A[7] ;
  M[8] = A[8] ;
  M[9] = b[0] ;
  M[10] = b[1] ;
  M[11] = b[2] ;
  err = vl_gaussian_elimination(M,3,4) ;
  x[0] = M[9] ;
  x[1] = M[10] ;
  x[2] = M[11] ;
  return err ;
}

/** @brief Solve a 2x2 linear system
 ** @param x result.
 ** @param A system matrix.
 ** @param b coefficients.
 **
 ** The function computes a solution to @f$ Ax =b @f$ for a 2x2
 ** matrix.
 **/

VL_EXPORT int
vl_solve_linear_system_2 (double * x, double const * A, double const *b)
{
  int err ;
  double M[2*3] ;
  M[0] = A[0] ;
  M[1] = A[1] ;
  M[2] = A[2] ;
  M[3] = A[3] ;
  M[4] = b[0];
  M[5] = b[1] ;
  err = vl_gaussian_elimination(M,2,3) ;
  x[0] = M[4] ;
  x[1] = M[5] ;
  return err ;
}

/** @brief Gaussian elimination
 ** @param M matrix.
 ** @param numRows number of rows of @c M.
 ** @param numColumns number of columns of @c M.
 **
 ** The function runs Gaussian elimination with pivoting
 ** on the matrix @a M in place.
 ** @c numRows must be not larger than @c numColumns.
 **
 ** Let @f$ M = [A, b] @f$ to obtain the solution to the linear
 ** system @f$ Ax=b @f$ (as the last column of @c M after
 ** elimination).
 **
 ** Let @f$ M = [A, I] @f$ to compute the inverse of @c A in
 ** a similar manner.
 **/

VL_EXPORT vl_bool
vl_gaussian_elimination (double * A, vl_size numRows, vl_size numColumns)
{
  vl_index i, j, ii, jj ;
  assert(A) ;
  assert(numRows <= numColumns) ;

#define Aat(i,j) A[(i) + (j)*numRows]

  /* Gauss elimination */
  for(j = 0 ; j < (signed)numRows ; ++j) {
    double maxa = 0 ;
    double maxabsa = 0 ;
    vl_index maxi = -1 ;
    double tmp ;

#if 0
    {
      vl_index iii, jjj ;
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif

    /* look for the maximally stable pivot */
    for (i = j ; i < (signed)numRows ; ++i) {
      double a = Aat(i,j) ;
      double absa = vl_abs_d (a) ;
      if (absa > maxabsa) {
        maxa = a ;
        maxabsa = absa ;
        maxi = i ;
      }
    }
    i = maxi ;

    /* if singular give up */
    if (maxabsa < 1e-10) return VL_ERR_OVERFLOW ;

    /* swap j-th row with i-th row and normalize j-th row */
    for(jj = j ; jj < (signed)numColumns ; ++jj) {
      tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
      Aat(j,jj) /= maxa ;
    }

#if 0
    {
      vl_index iii, jjj ;
      VL_PRINTF("after swap %d %d\n", j, i);
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif

    /* elimination */
    for (ii = j+1 ; ii < (signed)numRows ; ++ii) {
      double x = Aat(ii,j) ;
      for (jj = j ; jj < (signed)numColumns ; ++jj) {
        Aat(ii,jj) -= x * Aat(j,jj) ;
      }
    }

#if 0
    {
      VL_PRINTF("after elimination\n");

      vl_index iii, jjj ;
      for (iii = 0 ; iii < 2 ; ++iii) {
        for (jjj = 0 ; jjj < 3 ; ++jjj) {
          VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

        }
        VL_PRINTF("\n") ;
      }
      VL_PRINTF("\n") ;
    }
#endif

  }

  /* backward substitution */
  for (i = numRows - 1 ; i > 0 ; --i) {
    /* substitute in all rows above */
    for (ii = i - 1 ; ii >= 0 ; --ii) {
      double x = Aat(ii,i) ;
      /* j = numRows */
      for (j = numRows ; j < (signed)numColumns ; ++j) {
        Aat(ii,j) -= x * Aat(i,j) ;
      }
    }
  }

#if 0
  {
    VL_PRINTF("after substitution\n");

    vl_index iii, jjj ;
    for (iii = 0 ; iii < 2 ; ++iii) {
      for (jjj = 0 ; jjj < 3 ; ++jjj) {
        VL_PRINTF("%5.2g ", Aat(iii,jjj)) ;

      }
      VL_PRINTF("\n") ;
    }
    VL_PRINTF("\n") ;
  }
#endif


  return VL_ERR_OK ;
}

/* VL_MATHOP_INSTANTIATING */
#endif

#undef VL_MATHOP_INSTANTIATING
