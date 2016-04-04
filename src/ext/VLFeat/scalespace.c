/** @file scalespace.c
 ** @brief Scale Space - Definition
 ** @author Karel Lenc
 ** @author Andrea Vedaldi
 ** @author Michal Perdoch
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page scalespace Gaussian Scale Space (GSS)
@author Karel Lenc
@author Andrea Vedaldi
@author Michal Perdoch
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref scalespace.h implements a Gaussian scale space, a data structure
representing an image at multiple resolutions
@cite{witkin83scale-space} @cite{koenderink84the-structure}
@cite{lindeberg94scale-space}. Scale spaces have many use, including
the detection of co-variant local features
@cite{lindeberg98principles} such as SIFT, Hessian-Affine,
Harris-Affine, Harris-Laplace, etc. @ref scalespace-starting
demonstreates how to use the C API to compute the scalespace of an
image. For further details refer to:

- @subpage scalespace-fundamentals

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section scalespace-starting Getting started
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

Given an input image `image`, the following example uses the
::VlScaleSpace object to compute its Gaussian scale space and return
the image `level` at scale `(o,s)`, where `o` is the octave and `s` is
the octave subdivision or sublevel:

@code
float* level ;
VlScaleSpace ss = vl_scalespace_new(imageWidth, imageHeight) ;
vl_scalespace_put_image(ss, image) ;
level = vl_scalespace_get_level(ss, o, s) ;
@endcode

The image `level` is obtained by convolving `image` by a Gaussian
filter of isotropic standard deviation given by

@code
double sigma = vl_scalespace_get_sigma(ss, o, s) ;
@endcode

The resolution of `level` is in general different from the resolution
of `image` and is determined by the octave `o`. It can be obtained as
follows:

@code
VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(ss, o) ;
ogeom.width // width of level (in number of pixels)
ogeom.height // height of level (in number of pixels)
ogeom.step // spatial sampling step
@endcode

The parameter `ogeom.step` is the sampling step relatively to the
sampling of the input image `image`. The ranges of valid octaves and
scale sublevels can be obtained as

@code
VlScaleSpaceGeometry geom = vl_scalespace_get_geometry(ss) ;
geom.firstOctave // Index of the fisrt octave
geom.lastOctave // Index of the last octave
geom.octaveResolution ; // Number of octave subdivisions
geom.octaveFirstSubdivision // Index of the first octave subdivision
geom.octaveLastSubdivision  // Index of the last octave subdivision
@endcode

So for example `o` minimum value is `geom.firstOctave` and maximum
value is `geom.lastOctave`. The subdivision index `s` naturally spans
the range 0 to `geom.octaveResolution-1`. However, the scale space
object is flexible in that it allows different ranges of subdivisions
to be computed and `s` varies in the range
`geom.octaveFirstSubdivision` to `geom.octaveLastSubdivision`. See
@ref scalespace-fundamentals for further details.

The geometry of the scale space can be customized upon creation, as
follows:

@code
VlScaleSpaceGeometry geom = vl_scalespace_get_default_geometry(imageWidth, imageHeight) ;
geom.firstOctave = -1 ;
geom.octaveFirstSubdivision = -1 ;
geom.octaveLastSubdivision = geom.octaveResolution ;
VlScaleSpacae ss = vl_scalespace_new_with_geometry (geom) ;
@endcode

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page scalespace-fundamentals Gaussian scale space fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This page discusses the notion of *Gaussian scale space* and the
relative data structure. For the C API see @ref scalespace.h and @ref
scalespace-starting.

A *scale space* is representation of an image at multiple resolution
levels. An image is a function $\ell(x,y)$ of two coordinates $x$,
$y$; the scale space $\ell(x,y,\sigma)$ adds a third coordinate
$\sigma$ indexing the *scale*. Here the focus is the Gaussian scale
space, where the image $\ell(x,y,\sigma)$ is obtained by smoothing
$\ell(x,y)$ by a Gaussian kernel of isotropic standard deviation
$\sigma$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section scalespace-definition Scale space definition
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Formally, the *Gaussian scale space* of an image $\ell(x,y)$ is
defined as

\[
   \ell(x,y,\sigma) =
   [g_{\sigma} * \ell](x,y,\sigma)
\]

where $g_\sigma$ denotes a 2D Gaussian kernel of isotropic standard
deviation $\sigma$:

\[
  g_{\sigma}(x,y) = \frac{1}{2\pi\sigma^2}
  \exp\left(
  - \frac{x^2 + y^2}{2\sigma^2}
  \right).
\]

An important detail is that the algorithm computing the scale space
assumes that the input image $\ell(x,y)$ is pre-smoothed, roughly
capturing the effect of the finite pixel size in a CCD. This is
modelled by assuming that the input is not $\ell(x,y)$, but
$\ell(x,y,\sigma_n)$, where $\sigma_n$ is a *nominal smoothing*,
usually taken to be 0.5 (half a pixel standard deviation). This also
means that $\sigma = \sigma_n = 0.5$ is the *finest scale* that can
actually be computed.

The scale space structure stores samples of the function
$\ell(x,y,\sigma)$. The density of the sampling of the spatial
coordinates $x$ and $y$ is adjusted as a function of the scale
$\sigma$, corresponding to the intuition that images at a coarse
resolution can be sampled more coarsely without loss of
information. Thus, the scale space has the structure of a *pyramid*: a
collection of digital images sampled at progressively coarser spatial
resolution and hence of progressively smaller size (in pixels).

The following figure illustrates the scale space pyramid structure:

@image html scalespace-basic.png "A scalespace structure with 2 octaves and S=3 subdivisions per octave"

The pyramid is organised in a number of *octaves*, indexed by a
parameter `o`. Each octave is further subdivided into *sublevels*,
indexed by a parameter `s`. These are related to the scale $\sigma$ by
the equation

\[
  \sigma(s,o) = \sigma_o 2^{\displaystyle o + \frac{s}{\mathtt{octaveResolution}}}
\]

where `octaveResolution` is the resolution of the octave subsampling
$\sigma_0$ is the *base smoothing*.

At each octave the spatial resolution is doubled, in the sense that
samples are take with a step of
\[
\mathtt{step} = 2^o.
\]
Hence, denoting as `level[i,j]` the corresponding samples, one has
$\ell(x,y,\sigma) = \mathtt{level}[i,j]$, where
\[
 (x,y) = (i,j) \times \mathtt{step},
\quad
\sigma = \sigma(o,s),
 \quad
 0 \leq i < \mathtt{lwidth},
\quad
 0 \leq j < \mathtt{lheight},
\]
where
\[
  \mathtt{lwidth} = \lfloor \frac{\mathtt{width}}{2^\mathtt{o}}\rfloor, \quad
  \mathtt{lheight} = \lfloor \frac{\mathtt{height}}{2^\mathtt{o}}\rfloor.
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section scalespace-geometry Scale space geometry
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In addition to the parameters discussed above, the geometry of the
data stored in a scale space structure depends on the range of
allowable octaves `o` and scale sublevels `s`.

While `o` may range in any reasonable value given the size of the
input image `image`, usually its minimum value is either 0 or -1. The
latter corresponds to doubling the resolution of the image in the
first octave of the scale space and it is often used in feature
extraction. While there is no information added to the image by
upsampling in this manner, fine scale filters, including derivative
filters, are much easier to compute by upsalmpling first. The maximum
practical value is dictated by the image resolution, as it should be
$2^o\leq\min\{\mathtt{width},\mathtt{height}\}$. VLFeat has the
flexibility of specifying the range of `o` using the `firstOctave` and
`lastOctave` parameters of the ::VlScaleSpaceGeometry structure.

The sublevel `s` varies naturally in the range
$\{0,\dots,\mathtt{octaveResolution}-1\}$. However, it is often
convenient to store a few extra levels per octave (e.g. to compute the
local maxima of a function in scale or the Difference of Gaussian
cornerness measure). Thus VLFeat scale space structure allows this
parameter to vary in an arbitrary range, specified by the parameters
`octaveFirstSubdivision` and `octaveLastSubdivision` of
::VlScaleSpaceGeometry.

Overall the possible values of the indexes `o` and `s` are:

\[
\mathtt{firstOctave} \leq o \leq \mathtt{lastOctave},
\qquad
\mathtt{octaveFirstSubdivision} \leq s \leq \mathtt{octaveLastSubdivision}.
\]

Note that, depending on these ranges, there could be *redundant pairs*
of indexes `o` and `s` that represent the *same* pyramid level at more
than one sampling resolution. In practice, the ability to generate
such redundant information is very useful in algorithms using
scalespaces, as coding multiscale operations using a fixed sampling
resolution is far easier. For example, the DoG feature detector
computes the scalespace with three redundant levels per octave, as
follows:

@image html scalespace.png "A scalespace containing redundant representation of certain scale levels."

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section scalespace-algorithm Algorithm and limitations
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Given $\ell(x,y,\sigma_n)$, any of a vast number digitial filtering
techniques can be used to compute the scale levels. Presently, VLFeat
uses a basic FIR implementation of the Gaussian filters.

The FIR implementation is obtained by sampling the Gaussian function
and re-normalizing it to have unit norm. This simple construction does
not account properly for sampling effect, which may be a problem for
very small Gausisan kernels. As a rule of thumb, such filters work
sufficiently well for, say, standard deviation $\sigma$ at least 1.6
times the sampling step. A work around to apply this basic FIR
implementation to very small Gaussian filters is to upsample the image
first.

The limitations on the FIR filters have relatively important for the
pyramid construction, as the latter is obtained by *incremental
smoothing*: each successive level is obtained from the previous one by
adding the needed amount of smoothing. In this manner, the size of the
FIR filters remains small, which makes them efficient; at the same
time, for what discussed, excessively small filters are not
represented properly.

*/

#include "scalespace.h"
#include "mathop.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/** @file scalespace.h
 ** @struct VlScaleSpace
 ** @brief Scale space class
 **
 ** This is an opaque class used to compute the scale space of an
 ** image.
 **/

struct _VlScaleSpace
{
  VlScaleSpaceGeometry geom ; /**< Geometry of the scale space */
  float **octaves ; /**< Data */
} ;

/* ---------------------------------------------------------------- */
/** @brief Get the default geometry for a given image size.
 ** @param width image width.
 ** @param height image height.
 ** @return the default scale space geometry.
 **
 ** Both @a width and @a height must be at least one pixel wide.
 **/

VlScaleSpaceGeometry
vl_scalespace_get_default_geometry (vl_size width, vl_size height)
{
  VlScaleSpaceGeometry geom ;
  assert(width >= 1) ;
  assert(height >= 1) ;
  geom.width = width ;
  geom.height = height ;
  geom.firstOctave = 0 ;
  geom.lastOctave = VL_MAX(floor(vl_log2_d(VL_MIN(width, height))) - 3, 0) ;
  geom.octaveResolution= 3 ;
  geom.octaveFirstSubdivision = 0 ;
  geom.octaveLastSubdivision = geom.octaveResolution - 1 ;
  geom.baseScale = 1.6 * pow(2.0, 1.0 / geom.octaveResolution) ;
  geom.nominalScale = 0.5 ;
  return geom ;
}

#define is_valid_geometry(geom) (\
geom.firstOctave <= geom.lastOctave && \
geom.octaveResolution >= 1 && \
geom.octaveFirstSubdivision <= geom.octaveLastSubdivision && \
geom.baseScale >= 0.0 && \
geom.nominalScale >= 0.0)

/** @brief Check scale space geometries for equality
 ** @param a first geometry.
 ** @param b second geometry.
 ** @return true if equal.
 **/

vl_bool
vl_scalespacegeometry_is_equal (VlScaleSpaceGeometry a,
                                VlScaleSpaceGeometry b)
{
  return
  a.width == b.width &&
  a.height == b.height &&
  a.firstOctave == b.firstOctave &&
  a.lastOctave == b.lastOctave &&
  a.octaveResolution == b.octaveResolution &&
  a.octaveFirstSubdivision == b.octaveLastSubdivision &&
  a.baseScale == b.baseScale &&
  a.nominalScale == b.nominalScale ;
}

/** @brief Get the geometry of the scale space.
 ** @param self object.
 ** @return the scale space geometry.
 **/

VlScaleSpaceGeometry
vl_scalespace_get_geometry (VlScaleSpace const * self)
{
  return self->geom ;
}

/** @brief Get the geometry of an octave of the scalespace.
 ** @param self object.
 ** @param o octave index.
 ** @return the geometry of octave @a o.
 **/

VlScaleSpaceOctaveGeometry
vl_scalespace_get_octave_geometry (VlScaleSpace const * self, vl_index o)
{
  VlScaleSpaceOctaveGeometry ogeom ;
  ogeom.width = VL_SHIFT_LEFT(self->geom.width, -o) ;
  ogeom.height = VL_SHIFT_LEFT(self->geom.height, -o) ;
  ogeom.step = pow(2.0, o) ;
  return ogeom ;
}

/** @brief Get the data of a scale space level
 ** @param self object.
 ** @param o octave index.
 ** @param s level index.
 ** @return pointer to the data for octave @a o, level @a s.
 **
 ** The octave index @a o must be in the range @c firstOctave
 ** to @c lastOctave and the scale index @a s must be in the
 ** range @c octaveFirstSubdivision to @c octaveLastSubdivision.
 **/

float *
vl_scalespace_get_level (VlScaleSpace *self, vl_index o, vl_index s)
{
  VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self,o) ;
  float * octave ;
  assert(self) ;
  assert(o >= self->geom.firstOctave) ;
  assert(o <= self->geom.lastOctave) ;
  assert(s >= self->geom.octaveFirstSubdivision) ;
  assert(s <= self->geom.octaveLastSubdivision) ;

  octave = self->octaves[o - self->geom.firstOctave] ;
  return octave + ogeom.width * ogeom.height * (s - self->geom.octaveFirstSubdivision) ;
}

/** @brief Get the data of a scale space level (const)
 ** @param self object.
 ** @param o octave index.
 ** @param s level index.
 ** @return pointer to the data for octave @a o, level @a s.
 **
 ** This function is the same as ::vl_scalespace_get_level but reutrns
 ** a @c const pointer to the data.
 **/

float const *
vl_scalespace_get_level_const (VlScaleSpace const * self, vl_index o, vl_index s)
{
  return vl_scalespace_get_level((VlScaleSpace*)self, o, s) ;
}

/** ------------------------------------------------------------------
 ** @brief Get the scale of a given octave and sublevel
 ** @param self object.
 ** @param o octave index.
 ** @param s sublevel index.
 **
 ** The function returns the scale $\sigma(o,s)$ as a function of the
 ** octave index @a o and sublevel @a s.
 **/

double
vl_scalespace_get_level_sigma (VlScaleSpace const *self, vl_index o, vl_index s)
{
  return self->geom.baseScale * pow(2.0, o + (double) s / self->geom.octaveResolution) ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Upsample the rows and take the transpose
 ** @param destination output image.
 ** @param source input image.
 ** @param width input image width.
 ** @param height input image height.
 **
 ** The output image has dimensions @a height by 2 @a width (so the
 ** destination buffer must be at least as big as two times the
 ** input buffer).
 **
 ** Upsampling is performed by linear interpolation.
 **/

static void
copy_and_upsample
(float *destination,
 float const *source, vl_size width, vl_size height)
{
  vl_index x, y, ox, oy ;
  float v00, v10, v01, v11 ;

  assert(destination) ;
  assert(source) ;

  for(y = 0 ; y < (signed)height ; ++y) {
    oy = (y < ((signed)height - 1)) * width ;
    v10 = source[0] ;
    v11 = source[oy] ;
    for(x = 0 ; x < (signed)width ; ++x) {
      ox = x < ((signed)width - 1) ;
      v00 = v10 ;
      v01 = v11 ;
      v10 = source[ox] ;
      v11 = source[ox + oy] ;
      destination[0] = v00 ;
      destination[1] = 0.5f * (v00 + v10) ;
      destination[2*width] = 0.5f * (v00 + v01) ;
      destination[2*width+1] = 0.25f * (v00 + v01 + v10 + v11) ;
      destination += 2 ;
      source ++;
    }
    destination += 2*width ;
  }
}

/** ------------------------------------------------------------------
 ** @internal @brief Downsample
 ** @param destination output imgae buffer.
 ** @param source input image buffer.
 ** @param width input image width.
 ** @param height input image height.
 ** @param numOctaves octaves (non negative).
 **
 ** The function downsamples the image @a d times, reducing it to @c
 ** 1/2^d of its original size. The parameters @a width and @a height
 ** are the size of the input image. The destination image @a dst is
 ** assumed to be <code>floor(width/2^d)</code> pixels wide and
 ** <code>floor(height/2^d)</code> pixels high.
 **/

static void
copy_and_downsample
(float *destination,
 float const *source,
 vl_size width, vl_size height, vl_size numOctaves)
{
  vl_index x, y ;
  vl_size step = 1 << numOctaves ; /* step = 2^numOctaves */

  assert(destination) ;
  assert(source) ;

  if (numOctaves == 0) {
    memcpy(destination, source, sizeof(float) * width * height) ;
  } else {
    for(y = 0 ; y < (signed)height ; y += step) {
      float const *p = source + y * width ;
      for(x = 0 ; x < (signed)width - ((signed)step - 1) ; x += step) {
        *destination++ = *p ;
        p += step ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/** @brief Create a new scale space object
 ** @param width image width.
 ** @param height image height.
 ** @return new scale space object.
 **
 ** This function is the same as ::vl_scalespace_new_with_geometry()
 ** but it uses ::vl_scalespace_get_default_geometry to initialise
 ** the geometry of the scale space from the image size.
 **
 ** @sa ::vl_scalespace_new_with_geometry(), ::vl_scalespace_delete().
 **/

VlScaleSpace *
vl_scalespace_new (vl_size width, vl_size height)
{
  VlScaleSpaceGeometry geom ;
  geom = vl_scalespace_get_default_geometry(width, height) ;
  return vl_scalespace_new_with_geometry(geom) ;
}

/** ------------------------------------------------------------------
 ** @brief Create a new scale space with the specified geometry
 ** @param geom scale space geomerty.
 ** @return new scale space object.
 **
 ** If the geometry is not valid (see ::VlScaleSpaceGeometry), the
 ** result is unpredictable.
 **
 ** The function returns `NULL` if it was not possible to allocate the
 ** object because of an out-of-memory condition.
 **
 ** @sa ::VlScaleSpaceGeometry, ::vl_scalespace_delete().
 **/

VlScaleSpace *
vl_scalespace_new_with_geometry (VlScaleSpaceGeometry geom)
{

  vl_index o ;
  vl_size numSublevels = geom.octaveLastSubdivision - geom.octaveFirstSubdivision + 1 ;
  vl_size numOctaves = geom.lastOctave - geom.firstOctave + 1 ;
  VlScaleSpace *self ;

  assert(is_valid_geometry(geom)) ;
  numOctaves = geom.lastOctave - geom.firstOctave + 1 ;
  numSublevels = geom.octaveLastSubdivision - geom.octaveFirstSubdivision + 1 ;

  self = vl_calloc(1, sizeof(VlScaleSpace)) ;
  if (self == NULL) goto err_alloc_self ;
  self->geom = geom ;
  self->octaves = vl_calloc(numOctaves, sizeof(float*)) ;
  if (self->octaves == NULL) goto err_alloc_octave_list ;
  for (o = self->geom.firstOctave ; o <= self->geom.lastOctave ; ++o) {
    VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self,o) ;
    vl_size octaveSize = ogeom.width * ogeom.height * numSublevels ;
    self->octaves[o - self->geom.firstOctave] = vl_malloc(octaveSize * sizeof(float)) ;
    if (self->octaves[o - self->geom.firstOctave] == NULL) goto err_alloc_octaves;
  }
  return self ;

err_alloc_octaves:
  for (o = self->geom.firstOctave ; o <= self->geom.lastOctave ; ++o) {
    if (self->octaves[o - self->geom.firstOctave]) {
      vl_free(self->octaves[o - self->geom.firstOctave]) ;
    }
  }
err_alloc_octave_list:
  vl_free(self) ;
err_alloc_self:
  return NULL ;
}

/* ---------------------------------------------------------------- */
/** @brief Create a new copy of the object
 ** @param self object to copy from.
 **
 ** The function returns `NULL` if the copy cannot be made due to an
 ** out-of-memory condition.
 **/

VlScaleSpace *
vl_scalespace_new_copy (VlScaleSpace* self)
{
  vl_index o  ;
  VlScaleSpace * copy = vl_scalespace_new_shallow_copy(self) ;
  if (copy == NULL) return NULL ;

  for (o = self->geom.firstOctave ; o <= self->geom.lastOctave ; ++o) {
    VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self,o) ;
    vl_size numSubevels = self->geom.octaveLastSubdivision - self->geom.octaveFirstSubdivision + 1;
    memcpy(copy->octaves[o - self->geom.firstOctave],
           self->octaves[o - self->geom.firstOctave],
           ogeom.width * ogeom.height * numSubevels * sizeof(float)) ;
  }
  return copy ;
}

/* ---------------------------------------------------------------- */
/** @brief Create a new shallow copy of the object
 ** @param self object to copy from.
 **
 ** The function works like ::vl_scalespace_new_copy() but only allocates
 ** the scale space, without actually copying the data.
 **/

VlScaleSpace *
vl_scalespace_new_shallow_copy (VlScaleSpace* self)
{
  return vl_scalespace_new_with_geometry (self->geom) ;
}

/* ---------------------------------------------------------------- */
/** @brief Delete object
 ** @param self object to delete.
 ** @sa ::vl_scalespace_new()
 **/

void
vl_scalespace_delete (VlScaleSpace * self)
{
  if (self) {
    if (self->octaves) {
      vl_index o ;
      for (o = self->geom.firstOctave ; o <= self->geom.lastOctave ; ++o) {
        if (self->octaves[o - self->geom.firstOctave]) {
          vl_free(self->octaves[o - self->geom.firstOctave]) ;
        }
      }
      vl_free(self->octaves) ;
    }
    vl_free(self) ;
  }
}

/* ---------------------------------------------------------------- */

/** @internal @brief Fill octave starting from the first level
 ** @param self object instance.
 ** @param o octave to process.
 **
 ** The function takes the first sublevel of octave @a o (the one at
 ** sublevel `octaveFirstLevel` and iteratively
 ** smoothes it to obtain the other octave levels.
 **/

void
_vl_scalespace_fill_octave (VlScaleSpace *self, vl_index o)
{
  vl_index s ;
  VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self, o) ;

  for(s = self->geom.octaveFirstSubdivision + 1 ;
      s <= self->geom.octaveLastSubdivision ; ++s) {
    double sigma = vl_scalespace_get_level_sigma(self, o, s) ;
    double previousSigma = vl_scalespace_get_level_sigma(self, o, s - 1) ;
    double deltaSigma = sqrtf(sigma*sigma - previousSigma*previousSigma) ;

    float* level = vl_scalespace_get_level (self, o, s) ;
    float* previous = vl_scalespace_get_level (self, o, s-1) ;
    vl_imsmooth_f (level, ogeom.width,
                   previous, ogeom.width, ogeom.height, ogeom.width,
                   deltaSigma / ogeom.step, deltaSigma / ogeom.step) ;
  }
}

/** ------------------------------------------------------------------
 ** @internal @brief Initialize the first level of an octave from an image
 ** @param self ::VlScaleSpace object instance.
 ** @param image image data.
 ** @param o octave to start.
 **
 ** The function initializes the first level of octave @a o from
 ** image @a image. The dimensions of the image are the ones set
 ** during the creation of the ::VlScaleSpace object instance.
 **/

static void
_vl_scalespace_start_octave_from_image (VlScaleSpace *self,
                                        float const *image,
                                        vl_index o)
{
  float *level ;
  double sigma, imageSigma ;
  vl_index op ;

  assert(self) ;
  assert(image) ;
  assert(o >= self->geom.firstOctave) ;
  assert(o <= self->geom.lastOctave) ;

  /*
   * Copy the image to self->geom.octaveFirstSubdivision of octave o, upscaling or
   * downscaling as needed.
   */

  level = vl_scalespace_get_level(self, VL_MAX(0, o), self->geom.octaveFirstSubdivision) ;
  copy_and_downsample(level, image, self->geom.width, self->geom.height, VL_MAX(0, o)) ;

  for (op = -1 ; op >= o ; --op) {
    VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self, op + 1) ;
    float *succLevel = vl_scalespace_get_level(self, op + 1, self->geom.octaveFirstSubdivision) ;
    level = vl_scalespace_get_level(self, op, self->geom.octaveFirstSubdivision) ;
    copy_and_upsample(level, succLevel, ogeom.width, ogeom.height) ;
  }

  /*
   * Adjust the smoothing of the first level just initialised, accounting
   * for the fact that the input image is assumed to be a nominal scale
   * level.
   */

  sigma = vl_scalespace_get_level_sigma(self, o, self->geom.octaveFirstSubdivision) ;
  imageSigma = self->geom.nominalScale ;

  if (sigma > imageSigma) {
    VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self, o) ;
    double deltaSigma = sqrt (sigma*sigma - imageSigma*imageSigma) ;
    level = vl_scalespace_get_level (self, o, self->geom.octaveFirstSubdivision) ;
    vl_imsmooth_f (level, ogeom.width,
                   level, ogeom.width, ogeom.height, ogeom.width,
                   deltaSigma / ogeom.step, deltaSigma / ogeom.step) ;
  }
}

/** @internal @brief Initialize the first level of an octave from the previous octave
 ** @param self object.
 ** @param o octave to initialize.
 **
 ** The function initializes the first level of octave @a o from the
 ** content of octave <code>o - 1</code>.
 **/

static void
_vl_scalespace_start_octave_from_previous_octave (VlScaleSpace *self, vl_index o)
{
  double sigma, prevSigma ;
  float *level, *prevLevel ;
  vl_index prevLevelIndex ;
  VlScaleSpaceOctaveGeometry ogeom ;

  assert(self) ;
  assert(o > self->geom.firstOctave) ; /* must not be the first octave */
  assert(o <= self->geom.lastOctave) ;

  /*
   * From the previous octave pick the level which is closer to
   * self->geom.octaveFirstSubdivision in this octave.
   * The is self->geom.octaveFirstSubdivision + self->numLevels since there are
   * self->geom.octaveResolution levels in an octave, provided that
   * this value does not exceed self->geom.octaveLastSubdivision.
   */

  prevLevelIndex = VL_MIN(self->geom.octaveFirstSubdivision
                          + (signed)self->geom.octaveResolution,
                          self->geom.octaveLastSubdivision) ;
  prevLevel = vl_scalespace_get_level (self, o - 1, prevLevelIndex) ;
  level = vl_scalespace_get_level (self, o, self->geom.octaveFirstSubdivision) ;
  ogeom = vl_scalespace_get_octave_geometry(self, o - 1) ;

  copy_and_downsample (level, prevLevel, ogeom.width, ogeom.height, 1) ;

  /*
   * Add remaining smoothing, if any.
   */

  sigma = vl_scalespace_get_level_sigma(self, o, self->geom.octaveFirstSubdivision) ;
  prevSigma = vl_scalespace_get_level_sigma(self, o - 1, prevLevelIndex) ;

  if (sigma > prevSigma) {
    VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(self, o) ;
    double deltaSigma = sqrt (sigma*sigma - prevSigma*prevSigma) ;
    level = vl_scalespace_get_level (self, o, self->geom.octaveFirstSubdivision) ;

    /* todo: this may fail due to an out-of-memory condition */
    vl_imsmooth_f (level, ogeom.width,
                   level, ogeom.width, ogeom.height, ogeom.width,
                   deltaSigma / ogeom.step, deltaSigma / ogeom.step) ;
  }
}

/** @brief Initialise Scale space with new image
 ** @param self ::VlScaleSpace object instance.
 ** @param image image to process.
 **
 ** Compute the data of all the defined octaves and scales of the scale
 ** space @a self.
 **/

void
vl_scalespace_put_image (VlScaleSpace *self, float const *image)
{
  vl_index o ;
  _vl_scalespace_start_octave_from_image(self, image, self->geom.firstOctave) ;
  _vl_scalespace_fill_octave(self, self->geom.firstOctave) ;
  for (o = self->geom.firstOctave + 1 ; o <= self->geom.lastOctave ; ++o) {
    _vl_scalespace_start_octave_from_previous_octave(self, o) ;
    _vl_scalespace_fill_octave(self, o) ;
  }
}
