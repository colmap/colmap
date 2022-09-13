/** @file scalespace.h
 ** @brief Scale Space (@ref scalespace)
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 ** @author Michal Perdoch
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_SCALESPACE_H
#define VL_SCALESPACE_H

#include "generic.h"
#include "imopv.h"
#include "mathop.h"

/* ---------------------------------------------------------------- */
/*                                             VlScaleSpaceGeometry */
/* ---------------------------------------------------------------- */

/** @brief Geometry of a scale space
 **
 ** There are a few restrictions on the valid geometrties.
 */
typedef struct _VlScaleSpaceGeometry
{
  vl_size width ; /**< Image width */
  vl_size height ; /**< Image height */
  vl_index firstOctave ; /**< Index of the fisrt octave */
  vl_index lastOctave ; /**< Index of the last octave */
  vl_size octaveResolution ; /**< Number of octave subdivisions */
  vl_index octaveFirstSubdivision ; /**< Index of the first octave subdivision */
  vl_index octaveLastSubdivision ; /**< Index of the last octave subdivision */
  double baseScale ; /**< Base smoothing (smoothing of octave 0, level 0) */
  double nominalScale ; /**< Nominal smoothing of the original image */
} VlScaleSpaceGeometry ;

VL_EXPORT
vl_bool vl_scalespacegeometry_is_equal (VlScaleSpaceGeometry a,
                                        VlScaleSpaceGeometry b) ;

/* ---------------------------------------------------------------- */
/*                                       VlScaleSpaceOctaveGeometry */
/* ---------------------------------------------------------------- */

/** @brief Geometry of one octave of a scale space */
typedef struct _VlScaleSpaceOctaveGeometry
{
  vl_size width ; /**< Width (number of pixels) */
  vl_size height ; /**< Height (number of pixels) */
  double step ; /**< Sampling step (size of a pixel) */
} VlScaleSpaceOctaveGeometry ;

/* ---------------------------------------------------------------- */
/*                                                     VlScaleSpace */
/* ---------------------------------------------------------------- */

typedef struct _VlScaleSpace VlScaleSpace ;

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT VlScaleSpaceGeometry vl_scalespace_get_default_geometry(vl_size width, vl_size height) ;
VL_EXPORT VlScaleSpace * vl_scalespace_new (vl_size width, vl_size height) ;
VL_EXPORT VlScaleSpace * vl_scalespace_new_with_geometry (VlScaleSpaceGeometry geom) ;
VL_EXPORT VlScaleSpace * vl_scalespace_new_copy (VlScaleSpace* src);
VL_EXPORT VlScaleSpace * vl_scalespace_new_shallow_copy (VlScaleSpace* src);
VL_EXPORT void vl_scalespace_delete (VlScaleSpace *self) ;
/** @} */

/** @name Process data
 ** @{
 **/
VL_EXPORT void
vl_scalespace_put_image (VlScaleSpace *self, float const* image);
/** @} */

/** @name Retrieve data and parameters
 ** @{
 **/
VL_EXPORT VlScaleSpaceGeometry vl_scalespace_get_geometry (VlScaleSpace const * self) ;
VL_EXPORT VlScaleSpaceOctaveGeometry vl_scalespace_get_octave_geometry (VlScaleSpace const * self, vl_index o) ;
VL_EXPORT float *
vl_scalespace_get_level (VlScaleSpace * self, vl_index o, vl_index s) ;
VL_EXPORT float const *
vl_scalespace_get_level_const (VlScaleSpace const * self, vl_index o, vl_index s) ;
VL_EXPORT double
vl_scalespace_get_level_sigma (VlScaleSpace const *self, vl_index o, vl_index s) ;
/** @} */

/* VL_SCALESPACE_H */
#endif

