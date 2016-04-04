/** @file homkermap.c
 ** @brief Homogeneous kernel map - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file homkermap.h

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@page homkermap Homogeneous kernel map
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

@ref homkermap.h implements the homogeneous kernel maps introduced in
@cite{vedaldi10efficient},@cite{vedaldi12efficient}.  Such maps are
efficient linear representations of popular kernels such as the
intersection, $\chi^2$, and Jensen-Shannon ones.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section homkermap-starting Getting started
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The homogeneous kernel map is implemented as an object of type
::VlHomogeneousKernelMap. To use thois object, first create an
instance by using ::vl_homogeneouskernelmap_new, then use
::vl_homogeneouskernelmap_evaluate_d or
::vl_homogeneouskernelmap_evaluate_f (depdening on whether the data is
@c double or @c float) to compute the feature map $ \Psi(x)
$. When done, dispose of the object by calling
::vl_homogeneouskernelmap_delete.

@code
double gamma = 1.0 ;
int order = 1 ;
double period = -1 ; // use default
double psi [3] ;
vl_size psiStride = 1 ;
double x = 0.5 ;
VlHomogeneousKernelMap * hom = vl_homogeneouskernelmap_new(
  VlHomogeneousKernelChi2, gamma, order, period,
  VlHomogeneousKernelMapWindowRectangular) ;
vl_homogeneouskernelmap_evaluate_d(hom, psi, psiStride, x) ;
vl_homogeneouskernelmap_delete(x) ;
@endcode

The constructor ::vl_homogeneouskernelmap_new takes the kernel type @c
kernel (see ::VlHomogeneousKernelType), the homogeneity order @c gamma
(use one for the standard $1$-homogeneous kernels), the approximation
order @c order (usually order one is enough), the period @a period
(use a negative value to use the default period), and a window type @c
window (use ::VlHomogeneousKernelMapWindowRectangular if unsure). The
approximation order trades off the quality and dimensionality of the
approximation. The resulting feature map $ \Psi(x) $, computed by
::vl_homogeneouskernelmap_evaluate_d or
::vl_homogeneouskernelmap_evaluate_f , is <code>2*order+1</code>
dimensional.

The code pre-computes the map $ \Psi(x) $ for efficient
evaluation. The table spans values of $ x $ in the range
$[2^{-20}, 2^{8}) $. In particular, values smaller than $
2^{-20} $ are treated as zeroes (which results in a null feature).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section homkermap-fundamentals Fundamentals
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The <em>homogeneous kernel map</em> is a finite dimensional linear
approximation of homogeneous kernels, including the intersection,
$\chi^2$, and Jensen-Shannon kernels. These kernels are frequently
used in computer vision applications because they are particular
suited to data in the format of histograms, which includes many common
visual descriptors.

Let $x,y \in \mathbb{R}_+$ be non-negative scalars and let $k(x,y) \in
\mathbb{R}$ be an homogeneous kernel such as the $\chi^2$ and or the
intersection ones:

@f[
  k_{\mathrm{inters}}(x,y) = \min\{x, y\},
  \quad
  k_{\chi^2}(x,y) = 2 \frac{(x - y)^2}{x+y}.
@f]

For vectorial data $ \mathbf{x},\mathbf{y} \in \mathbb{R}_+^d $, the
homogeneous kernels is defined as an <em>additive combination</em> of
scalar kernels $K(\mathbf{x},\mathbf{y}) = \sum_{i=1}^d k(x_i,y_i)$.

The <em>homogeneous kernel map</em> of order $n$ is a vector function
$\Psi(x) \in \mathbb{R}^{2n+1}$ such that, for any choice of $x, y \in
\mathbb{R}_+$, the following approximation holds:

@f[
  k(x,y) \approx \langle \Psi(x), \Psi(y) \rangle.
@f]

Given the feature map for the scalar case, the corresponding feature
map $\Psi(\mathbf{x})$ for the vectorial case is obtained by stacking
$[\Psi(x_1), \dots, \Psi(x_n)]$.  Note that the stacked feature
$\Psi(\mathbf{x})$ has dimension $d(2n+1)$.

Using linear analysis tools (e.g. a linear support vector machine)
on top of dataset that has been encoded by the homogeneous kernel map
is therefore approximately equivalent to using a method based
on the corresponding non-linear kernel.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection homkermap-overview-negative Extension to the negative reals
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

Any positive (semi-)definite kernel $k(x,y)$ defined on the
non-negative reals $x,y \in \mathbb{R}_+$ can be extended to the
entire real line by using the definition:

@f[
k_\pm(x,y) = \operatorname{sign}(x) \operatorname{sign}(y) k(|x|,|y|).
@f]

The homogeneous kernel map implements this extension by defining
$\Psi_\pm(x) = \operatorname{sign}(x) \Psi(|x|)$. Note that other
extensions are possible, such as

@f[
k_\pm(x,y) = H(xy) \operatorname{sign}(y) k(|x|,|y|)
@f]

where $H$ is the Heaviside function, but may result in higher
dimensional feature maps.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection homkermap-overview-homogeneity Homogeneity degree
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

Any (1-)homogeneous kernel $k_1(x,y)$ can be extended to a so called
$\gamma$-homgeneous kernel $k_\gamma(x,y)$ by the definition

@f[
  k_\gamma(x,y) = (xy)^{\frac{\gamma}{2}} \frac{k_1(x,y)}{\sqrt{xy}}
@f]

Smaller values of $\gamma$ enhance the kernel non-linearity and are
sometimes beneficial in applications (see
@cite{vedaldi10efficient},@cite{vedaldi12efficient} for details).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection homkermap-overview-window Windowing and period
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

This section discusses aspects of the homogeneous kernel map which are
more technical and may be skipped. The homogeneous kernel map
approximation is based on periodizing the kernel; given the kernel
signature

@f[
    \mathcal{K}(\lambda) = k(e^{\frac{\lambda}{2}}, e^{-\frac{\lambda}{2}})
@f]

the homogeneous kernel map is a feature map for the windowed and
periodized kernel whose signature is given by

@f[
   \hat{\mathcal{K}}(\lambda)
   =
   \sum_{i=-\infty}^{+\infty} \mathcal{K}(\lambda + k \Lambda) W(\lambda + k \Lambda)
@f]

where $W(\lambda)$ is a windowing function and $\Lambda$ is the
period. This implementation of the homogeneous kernel map supports the
use of a <em>uniform window</em> ($ W(\lambda) = 1 $) or of a
<em>rectangular window</em> ($ W(\lambda) =
\operatorname{rect}(\lambda/\Lambda) $). Note that $ \lambda =
\log(y/x) $ is equal to the logarithmic ratio of the arguments of the
kernel. Empirically, the rectangular window seems to have a slight
edge in applications.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section homkermap-details Implementation details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

This implementation uses the expressions given in
@cite{vedaldi10efficient},@cite{vedaldi11efficient} to compute in
closed form the maps $\Psi(x)$ for the supported kernel types. For
efficiency reasons, it precomputes $\Psi(x)$ for a large range of
values of the argument when the homogeneous kernel map object is
created.

The internal table stores $\Psi(x) \in \mathbb{R}^{2n+1}$ by sampling
$x\geq 0$. This uses the internal decomposition of IEEE floating point
representations (@c float and @c double) in mantissa and exponent:
<pre>
  x = mantissa * (2**exponent),
  minExponent <= exponent <= maxExponent,
  1 <= matnissa < 2.
</pre>
Each octave is further sampled in @c numSubdivisions sublevels.

When the map $\Psi(x)$ is evaluated, @c x is decomposed again into
exponent and mantissa to index the table. The output is obtained by
bilinear interpolation from the appropriate table entries.

**/

/* ---------------------------------------------------------------- */
#ifndef VL_HOMKERMAP_INSTANTIATING
/* ---------------------------------------------------------------- */

#include "homkermap.h"
#include "mathop.h"
#include <math.h>

struct _VlHomogeneousKernelMap
{
  VlHomogeneousKernelType kernelType ;
  double gamma ;
  VlHomogeneousKernelMapWindowType windowType ;
  vl_size order ;
  double period ;
  vl_size numSubdivisions ;
  double subdivision  ;
  vl_index minExponent ;
  vl_index maxExponent ;
  double * table ;
} ;

/** @internal @brief Sample the kernel specturm
 ** @param self homogeneous kernel map.
 ** @param omega sampling frequency.
 ** @return the spectrum sampled at @a omega.
 **/

VL_INLINE double
vl_homogeneouskernelmap_get_spectrum (VlHomogeneousKernelMap const * self, double omega)
{
  assert (self) ;
  switch (self->kernelType) {
    case VlHomogeneousKernelIntersection:
      return (2.0 / VL_PI) / (1 + 4 * omega*omega) ;
    case VlHomogeneousKernelChi2:
      return 2.0 / (exp(VL_PI * omega) + exp(-VL_PI * omega)) ;
    case VlHomogeneousKernelJS:
      return (2.0 / log(4.0)) *
      2.0 / (exp(VL_PI * omega) + exp(-VL_PI * omega)) /
      (1 + 4 * omega*omega) ;
    default:
      abort() ;
  }
}

/* helper */
VL_INLINE double sinc(double x)
{
  if (x == 0.0) return 1.0 ;
  return sin(x) / x ;
}

/** @internal @brief Sample the smoothed kernel spectrum
 ** @param self homogeneous kernel map.
 ** @param omega sampling frequency.
 ** @return the spectrum sampled at @a omega after smoothing.
 **/

VL_INLINE double
vl_homogeneouskernelmap_get_smooth_spectrum (VlHomogeneousKernelMap const * self, double omega)
{
  double kappa_hat = 0 ;
  double omegap ;
  double epsilon = 1e-2 ;
  double const omegaRange = 2.0 / (self->period * epsilon) ;
  double const domega = 2 * omegaRange / (2 * 1024.0 + 1) ;
  assert (self) ;
  switch (self->windowType) {
    case VlHomogeneousKernelMapWindowUniform:
      kappa_hat = vl_homogeneouskernelmap_get_spectrum(self, omega) ;
      break ;
    case VlHomogeneousKernelMapWindowRectangular:
      for (omegap = - omegaRange ; omegap <= omegaRange ; omegap += domega) {
        double win = sinc((self->period/2.0) * omegap) ;
        win *= (self->period/(2.0*VL_PI)) ;
        kappa_hat += win * vl_homogeneouskernelmap_get_spectrum(self, omegap + omega) ;
      }
      kappa_hat *= domega ;
      /* project on the postivie orthant (see PAMI) */
      kappa_hat = VL_MAX(kappa_hat, 0.0) ;
      break ;
    default:
      abort() ;
  }
  return kappa_hat ;
}

/* ---------------------------------------------------------------- */
/*                                     Constructors and destructors */
/* ---------------------------------------------------------------- */

/** @brief Create a new homgeneous kernel map
 ** @param kernelType type of homogeneous kernel.
 ** @param gamma kernel homogeneity degree.
 ** @param order approximation order.
 ** @param period kernel period.
 ** @param windowType type of window used to truncate the kernel.
 ** @return the new homogeneous kernel map.
 **
 ** The function intializes a new homogeneous kernel map for the
 ** specified kernel type, homogeneity degree, approximation order,
 ** period, and truncation window. See @ref homkermap-fundamentals for
 ** details.
 **
 ** The homogeneity degree @c gamma must be positive (the standard
 ** kernels are obtained by setting @c gamma to 1). When unsure, set
 ** @c windowType to ::VlHomogeneousKernelMapWindowRectangular. The @c
 ** period should be non-negative; specifying a negative or null value
 ** causes the function to switch to a default value.
 **
 ** The function returns @c NULL if there is not enough free memory.
 **/

VlHomogeneousKernelMap *
vl_homogeneouskernelmap_new (VlHomogeneousKernelType kernelType,
                             double gamma,
                             vl_size order,
                             double period,
                             VlHomogeneousKernelMapWindowType windowType)
{
  int tableWidth, tableHeight ;
  VlHomogeneousKernelMap * self = vl_malloc(sizeof(VlHomogeneousKernelMap)) ;
  if (! self) return NULL ;

  assert(gamma > 0) ;

  assert(kernelType == VlHomogeneousKernelIntersection ||
         kernelType == VlHomogeneousKernelChi2 ||
         kernelType == VlHomogeneousKernelJS) ;

  assert(windowType == VlHomogeneousKernelMapWindowUniform ||
         windowType == VlHomogeneousKernelMapWindowRectangular) ;

  if (period < 0) {
    switch (windowType) {
    case VlHomogeneousKernelMapWindowUniform:
      switch (kernelType) {
      case VlHomogeneousKernelChi2:         period = 5.86 * sqrt(order + 0)  + 3.65 ; break ;
      case VlHomogeneousKernelJS:           period = 6.64 * sqrt(order + 0)  + 7.24 ; break ;
      case VlHomogeneousKernelIntersection: period = 2.38 * log(order + 0.8) + 5.6 ; break ;
      }
      break ;
    case VlHomogeneousKernelMapWindowRectangular:
      switch (kernelType) {
      case VlHomogeneousKernelChi2:         period = 8.80 * sqrt(order + 4.44) - 12.6 ; break ;
      case VlHomogeneousKernelJS:           period = 9.63 * sqrt(order + 1.00) - 2.93;  break ;
      case VlHomogeneousKernelIntersection: period = 2.00 * log(order + 0.99)  + 3.52 ; break ;
      }
      break ;
    }
    period = VL_MAX(period, 1.0) ;
  }

  self->kernelType = kernelType ;
  self->windowType = windowType ;
  self->gamma = gamma ;
  self->order = order ;
  self->period = period ;
  self->numSubdivisions = 8 + 8*order ;
  self->subdivision = 1.0 / self->numSubdivisions ;
  self->minExponent = -20 ;
  self->maxExponent = 8 ;

  tableHeight = (int) (2*self->order + 1) ;
  tableWidth = (int) (self->numSubdivisions * (self->maxExponent - self->minExponent + 1)) ;
  self->table = vl_malloc (sizeof(double) *
                           (tableHeight * tableWidth + 2*(1+self->order))) ;
  if (! self->table) {
    vl_free(self) ;
    return NULL ;
  }

  {
    vl_index exponent ;
    vl_uindex i, j ;
    double * tablep = self->table ;
    double * kappa = self->table + tableHeight * tableWidth ;
    double * freq = kappa + (1+self->order) ;
    double L = 2.0 * VL_PI / self->period ;

    /* precompute the sampled periodicized spectrum */
    j = 0 ;
    i = 0 ;
    while (i <= self->order) {
      freq[i] = j ;
      kappa[i] = vl_homogeneouskernelmap_get_smooth_spectrum(self, j * L) ;
      ++ j ;
      if (kappa[i] > 0 || j >= 3*i) ++ i ;
    }

    /* fill table */
    for (exponent  = self->minExponent ;
         exponent <= self->maxExponent ; ++ exponent) {

      double x, Lxgamma, Llogx, xgamma ;
      double sqrt2kappaLxgamma ;
      double mantissa = 1.0 ;

      for (i = 0 ; i < self->numSubdivisions ;
           ++i, mantissa += self->subdivision) {
        x = ldexp(mantissa, (int)exponent) ;
        xgamma = pow(x, self->gamma) ;
        Lxgamma = L * xgamma ;
        Llogx = L * log(x) ;

        *tablep++ = sqrt(Lxgamma * kappa[0]) ;
        for (j = 1 ; j <= self->order ; ++j) {
          sqrt2kappaLxgamma = sqrt(2.0 * Lxgamma * kappa[j]) ;
          *tablep++ = sqrt2kappaLxgamma * cos(freq[j] * Llogx) ;
          *tablep++ = sqrt2kappaLxgamma * sin(freq[j] * Llogx) ;
        }
      } /* next mantissa */
    } /* next exponent */
  }
  return self ;
}

/** @brief Delete an object instance.
 ** @param self object.
 ** The function deletes the specified map object.
 **/

void
vl_homogeneouskernelmap_delete (VlHomogeneousKernelMap * self)
{
  vl_free(self->table) ;
  self->table = NULL ;
  vl_free(self) ;
}

/* ---------------------------------------------------------------- */
/*                                     Retrieve data and parameters */
/* ---------------------------------------------------------------- */

/** @brief Get the map order.
 ** @param self object.
 ** @return the map order.
 **/

vl_size
vl_homogeneouskernelmap_get_order (VlHomogeneousKernelMap const * self)
{
  assert(self) ;
  return self->order ;
}

/** @brief Get the map dimension.
 ** @param self object.
 ** @return the map dimension (2 @c order  +1).
 **/

vl_size
vl_homogeneouskernelmap_get_dimension (VlHomogeneousKernelMap const * self)
{
  assert(self) ;
  return 2 * self->order + 1 ;
}

/** @brief Get the kernel type.
 ** @param self object.
 ** @return kernel type.
 **/

VlHomogeneousKernelType
vl_homogeneouskernelmap_get_kernel_type (VlHomogeneousKernelMap const * self)
{
  assert(self) ;
  return self->kernelType ;
}

/** @brief Get the window type.
 ** @param self object.
 ** @return window type.
 **/

VlHomogeneousKernelMapWindowType
vl_homogeneouskernelmap_get_window_type (VlHomogeneousKernelMap const * self)
{
  assert(self) ;
  return self->windowType ;
}

/* ---------------------------------------------------------------- */
/*                                                     Process data */
/* ---------------------------------------------------------------- */

/** @fn ::vl_homogeneouskernelmap_evaluate_d(VlHomogeneousKernelMap const*,double*,vl_size,double)
 ** @brief Evaluate map
 ** @param self map object.
 ** @param destination output buffer.
 ** @param stride stride of the output buffer.
 ** @param x value to expand.
 **
 ** The function evaluates the feature map on @a x and stores the
 ** resulting <code>2*order+1</code> dimensional vector to
 ** @a destination[0], @a destination[stride], @a destination[2*stride], ....
 **/

/** @fn ::vl_homogeneouskernelmap_evaluate_f(VlHomogeneousKernelMap const*,float*,vl_size,double)
 ** @copydetails ::vl_homogeneouskernelmap_evaluate_d(VlHomogeneousKernelMap const*,double*,vl_size,double)
 **/

#define FLT VL_TYPE_FLOAT
#define VL_HOMKERMAP_INSTANTIATING
#include "homkermap.c"

#define FLT VL_TYPE_DOUBLE
#define VL_HOMKERMAP_INSTANTIATING
#include "homkermap.c"

/* VL_HOMKERMAP_INSTANTIATING */
#endif

/* ---------------------------------------------------------------- */
#ifdef VL_HOMKERMAP_INSTANTIATING
/* ---------------------------------------------------------------- */

#include "float.th"

void
VL_XCAT(vl_homogeneouskernelmap_evaluate_,SFX)
(VlHomogeneousKernelMap const * self,
 T * destination,
 vl_size stride,
 double x)
{
  /* break value into exponent and mantissa */
  int exponent ;
  int unsigned j ;
  double mantissa = frexp(x, &exponent) ;
  double sign = (mantissa >= 0.0) ? +1.0 : -1.0 ;
  mantissa *= 2*sign ;
  exponent -- ;

  if (mantissa == 0 ||
      exponent <= self->minExponent ||
      exponent >= self->maxExponent) {
    for (j = 0 ; j < 2*self->order+1 ; ++j) {
      *destination = (T) 0.0 ;
      destination += stride ;
    }
    return  ;
  }
  {
    vl_size featureDimension = 2*self->order + 1 ;
    double const * v1 = self->table +
    (exponent - self->minExponent) * self->numSubdivisions * featureDimension ;
    double const * v2 ;
    double f1, f2 ;

    mantissa -= 1.0 ;
    while (mantissa >= self->subdivision) {
      mantissa -= self->subdivision ;
      v1 += featureDimension ;
    }
    v2 = v1 + featureDimension ;
    for (j = 0 ; j < featureDimension ; ++j) {
      f1 = *v1++ ;
      f2 = *v2++ ;
      *destination = (T) sign * ((f2 - f1) * (self->numSubdivisions * mantissa) + f1) ;
      destination += stride ;
    }
  }
}

#undef FLT
#undef VL_HOMKERMAP_INSTANTIATING
/* VL_HOMKERMAP_INSTANTIATING */
#endif
