/** @file covdet.h
 ** @brief Covariant feature detectors (@ref covdet)
 ** @author Karel Lenc
 ** @author Andrea Vedaldi
 ** @author Michal Perdoch
 **/

/*
Copyright (C) 2013-14 Andrea Vedaldi.
Copyright (C) 2012 Karel Lenc, Andrea Vedaldi and Michal Perdoch.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_COVDET_H
#define VL_COVDET_H

#include "generic.h"
#include "stringop.h"
#include "scalespace.h"

#include <stdio.h>

/* ---------------------------------------------------------------- */
/*                                                   Feature Frames */
/* ---------------------------------------------------------------- */

/** @name Feature frames
 ** @{ */

/** @brief Types of feature frames */
typedef enum _VlFrameType {
  VL_FRAMETYPE_DISC = 1,         /**< A disc. */
  VL_FRAMETYPE_ORIENTED_DISC,    /**< An oriented disc. */
  VL_FRAMETYPE_ELLIPSE,          /**< An ellipse. */
  VL_FRAMETYPE_ORIENTED_ELLIPSE, /**< An oriented ellipse. */
  VL_FRAMETYPE_NUM
} VlFrameType ;

/** @brief Names of the frame types */
VL_EXPORT const char* vlFrameNames [VL_FRAMETYPE_NUM] ;

/** @brief Mapping between string values and VlFrameType values */
VL_EXPORT VlEnumerator vlFrameTypes [VL_FRAMETYPE_NUM] ;

/** @brief Disc feature frame */
typedef struct _VlFrameDisc
{
  float x ;     /**< center x-coordinate */
  float y ;     /**< center y-coordinate */
  float sigma ; /**< radius or scale */
} VlFrameDisc ;

/** @brief Oriented disc feature frame
 ** An upright frame has @c angle equal to zero.
 **/
typedef struct _VlFrameOrientedDisc {
  float x ;     /**< center x-coordinate */
  float y ;     /**< center y-coordinate */
  float sigma ; /**< radius or scale */
  float angle ; /**< rotation angle (rad) */
} VlFrameOrientedDisc ;

/** @brief Ellipse feature frame */
typedef struct _VlFrameEllipse {
  float x ;     /**< center x-coordinate */
  float y ;     /**< center y-coordinate */
  float e11 ;   /**< */
  float e12 ;
  float e22 ;
} VlFrameEllipse ;

/** @brief Oriented ellipse feature frame
 ** The affine transformation transforms the ellipse shape into
 ** a circular region. */
typedef struct _VlFrameOrientedEllipse {
  float x ;     /**< center x-coordinate */
  float y ;     /**< center y-coordinate */
  float a11 ;   /**< */
  float a12 ;
  float a21 ;
  float a22 ;
} VlFrameOrientedEllipse;

/** @brief Get the size of a frame structure
 ** @param frameType identifier of the type of frame.
 ** @return size of the corresponding frame structure in bytes.
 **/
VL_INLINE vl_size
vl_get_frame_size (VlFrameType frameType) {
  switch (frameType) {
    case VL_FRAMETYPE_DISC: return sizeof(VlFrameDisc);
    case VL_FRAMETYPE_ORIENTED_DISC: return sizeof(VlFrameOrientedDisc);
    case VL_FRAMETYPE_ELLIPSE: return sizeof(VlFrameEllipse);
    case VL_FRAMETYPE_ORIENTED_ELLIPSE: return sizeof(VlFrameOrientedEllipse);
    default:
      assert(0);
      break;
  }
  return 0;
}

/** @brief Get the size of a frame structure
 ** @param affineAdaptation whether the detector use affine adaptation.
 ** @param orientation whether the detector estimates the feature orientation.
 ** @return the type of extracted frame.
 **
 ** Depedning on whether the detector estimate the affine shape
 ** and orientation of a feature, different frame types
 ** are extracted. */

VL_INLINE VlFrameType
vl_get_frame_type (vl_bool affineAdaptation, vl_bool orientation)
{
  if (affineAdaptation) {
    if (orientation) {
      return VL_FRAMETYPE_ORIENTED_ELLIPSE;
    } else {
      return VL_FRAMETYPE_ELLIPSE;
    }
  } else {
    if (orientation) {
      return VL_FRAMETYPE_ORIENTED_DISC;
    } else {
      return VL_FRAMETYPE_DISC;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                       Covariant Feature Detector */
/* ---------------------------------------------------------------- */

/** @brief A detected feature shape and location */
typedef struct _VlCovDetFeature
{
  VlFrameOrientedEllipse frame ; /**< feature frame. */
  float peakScore ; /**< peak score. */
  float edgeScore ; /**< edge score. */
  float orientationScore ; /**< orientation score. */
  float laplacianScaleScore ; /**< Laplacian scale score. */
} VlCovDetFeature ;

/** @brief A detected feature orientation */
typedef struct _VlCovDetFeatureOrientation
{
  double angle ;
  double score ;
} VlCovDetFeatureOrientation ;

/** @brief A detected feature Laplacian scale */
typedef struct _VlCovDetFeatureLaplacianScale
{
  double scale ;
  double score ;
} VlCovDetFeatureLaplacianScale ;

/** @brief Covariant feature detection method */
typedef enum _VlCovDetMethod
{
  VL_COVDET_METHOD_DOG = 1,
  VL_COVDET_METHOD_HESSIAN,
  VL_COVDET_METHOD_HESSIAN_LAPLACE,
  VL_COVDET_METHOD_HARRIS_LAPLACE,
  VL_COVDET_METHOD_MULTISCALE_HESSIAN,
  VL_COVDET_METHOD_MULTISCALE_HARRIS,
  VL_COVDET_METHOD_NUM
} VlCovDetMethod;

/** @brief Mapping between strings and ::VlCovDetMethod values */
VL_EXPORT VlEnumerator vlCovdetMethods [VL_COVDET_METHOD_NUM] ;

#ifdef __DOXYGEN__
/** @brief Covariant feature detector
 ** @see @ref covdet */
struct _VlCovDet { }
#endif

/** @brief Covariant feature detector
 ** @see @ref covdet */
typedef struct _VlCovDet VlCovDet ;

/** @name Create and destroy
 ** @{ */
VL_EXPORT VlCovDet * vl_covdet_new (VlCovDetMethod method) ;
VL_EXPORT void vl_covdet_delete (VlCovDet * self) ;
VL_EXPORT void vl_covdet_reset (VlCovDet * self) ;
/** @} */

/** @name Process data
 ** @{ */
VL_EXPORT int vl_covdet_put_image (VlCovDet * self,
                                    float const * image,
                                    vl_size width, vl_size height) ;

VL_EXPORT void vl_covdet_detect (VlCovDet * self) ;
VL_EXPORT int vl_covdet_append_feature (VlCovDet * self, VlCovDetFeature const * feature) ;
VL_EXPORT void vl_covdet_extract_orientations (VlCovDet * self) ;
VL_EXPORT void vl_covdet_extract_laplacian_scales (VlCovDet * self) ;
VL_EXPORT void vl_covdet_extract_affine_shape (VlCovDet * self) ;

VL_EXPORT VlCovDetFeatureOrientation *
vl_covdet_extract_orientations_for_frame (VlCovDet * self,
                                          vl_size *numOrientations,
                                          VlFrameOrientedEllipse frame) ;

VL_EXPORT VlCovDetFeatureLaplacianScale *
vl_covdet_extract_laplacian_scales_for_frame (VlCovDet * self,
                                              vl_size * numScales,
                                              VlFrameOrientedEllipse frame) ;
VL_EXPORT int
vl_covdet_extract_affine_shape_for_frame (VlCovDet * self,
                                          VlFrameOrientedEllipse * adapted,
                                          VlFrameOrientedEllipse frame) ;

VL_EXPORT vl_bool
vl_covdet_extract_patch_for_frame (VlCovDet * self, float * patch,
                                   vl_size resolution,
                                   double extent,
                                   double sigma,
                                   VlFrameOrientedEllipse frame) ;

VL_EXPORT void
vl_covdet_drop_features_outside (VlCovDet * self, double margin) ;
/** @} */

/** @name Retrieve data and parameters
 ** @{ */
VL_EXPORT vl_size vl_covdet_get_num_features (VlCovDet const * self) ;
VL_EXPORT void * vl_covdet_get_features (VlCovDet * self) ;
VL_EXPORT vl_index vl_covdet_get_first_octave (VlCovDet const * self) ;
VL_EXPORT vl_size vl_covdet_get_octave_resolution (VlCovDet const * self) ;
VL_EXPORT double vl_covdet_get_peak_threshold (VlCovDet const * self) ;
VL_EXPORT double vl_covdet_get_edge_threshold (VlCovDet const * self) ;
VL_EXPORT double vl_covdeg_get_laplacian_peak_threshold (VlCovDet const * self) ;
VL_EXPORT vl_bool vl_covdet_get_transposed (VlCovDet const * self) ;
VL_EXPORT VlScaleSpace *  vl_covdet_get_gss (VlCovDet const * self) ;
VL_EXPORT VlScaleSpace *  vl_covdet_get_css (VlCovDet const * self) ;
VL_EXPORT vl_bool vl_covdet_get_aa_accurate_smoothing (VlCovDet const * self) ;
VL_EXPORT vl_size const * vl_covdet_get_laplacian_scales_statistics (VlCovDet const * self, vl_size * numScales) ;
VL_EXPORT double vl_covdet_get_non_extrema_suppression_threshold (VlCovDet const * self) ;
VL_EXPORT vl_size vl_covdet_get_num_non_extrema_suppressed (VlCovDet const * self) ;

/** @} */

/** @name Set parameters
 ** @{ */
VL_EXPORT void vl_covdet_set_first_octave (VlCovDet * self, vl_index o) ;
VL_EXPORT void vl_covdet_set_octave_resolution (VlCovDet * self, vl_size r) ;
VL_EXPORT void vl_covdet_set_peak_threshold (VlCovDet * self, double peakThreshold) ;
VL_EXPORT void vl_covdet_set_edge_threshold (VlCovDet * self, double edgeThreshold) ;
VL_EXPORT void vl_covdet_set_laplacian_peak_threshold (VlCovDet * self, double peakThreshold) ;
VL_EXPORT void vl_covdet_set_transposed (VlCovDet * self, vl_bool t) ;
VL_EXPORT void vl_covdet_set_aa_accurate_smoothing (VlCovDet * self, vl_bool x) ;
VL_EXPORT void vl_covdet_set_non_extrema_suppression_threshold (VlCovDet * self, double x) ;
/** @} */

/* VL_COVDET_H */
#endif
