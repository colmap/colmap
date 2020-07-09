/** @file hog.h
 ** @brief Histogram of Oriented Gradients (@ref hog)
 ** @author Andrea Vedaldi
 **/

/*
 Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_HOG_H
#define VL_HOG_H

#include "generic.h"

enum VlHogVariant_ { VlHogVariantDalalTriggs, VlHogVariantUoctti } ;

typedef enum VlHogVariant_ VlHogVariant ;

struct VlHog_
{
  VlHogVariant variant ;
  vl_size dimension ;
  vl_size numOrientations ;
  vl_bool transposed ;
  vl_bool useBilinearOrientationAssigment ;

  /* left-right flip permutation */
  vl_index * permutation ;

  /* glyphs */
  float * glyphs ;
  vl_size glyphSize ;

  /* helper vectors */
  float * orientationX ;
  float * orientationY ;

  /* buffers */
  float * hog ;
  float * hogNorm ;
  vl_size hogWidth ;
  vl_size hogHeight ;
} ;

typedef struct VlHog_ VlHog ;

VL_EXPORT VlHog * vl_hog_new (VlHogVariant variant, vl_size numOrientations, vl_bool transposed) ;
VL_EXPORT void vl_hog_delete (VlHog * self) ;
VL_EXPORT void vl_hog_process (VlHog * self,
                               float * features,
                               float const * image,
                               vl_size width, vl_size height, vl_size numChannels,
                               vl_size cellSize) ;

VL_EXPORT void vl_hog_put_image (VlHog * self,
                                 float const * image,
                                 vl_size width, vl_size height, vl_size numChannels,
                                 vl_size cellSize) ;

VL_EXPORT void vl_hog_put_polar_field (VlHog * self,
                                       float const * modulus,
                                       float const * angle,
                                       vl_bool directed,
                                       vl_size width, vl_size height, vl_size cellSize) ;

VL_EXPORT void vl_hog_extract (VlHog * self, float * features) ;
VL_EXPORT vl_size vl_hog_get_height (VlHog * self) ;
VL_EXPORT vl_size vl_hog_get_width (VlHog * self) ;


VL_EXPORT void vl_hog_render (VlHog const * self,
                              float * image,
                              float const * features,
                              vl_size width,
                              vl_size height) ;

VL_EXPORT vl_size vl_hog_get_dimension (VlHog const * self) ;
VL_EXPORT vl_index const * vl_hog_get_permutation (VlHog const * self) ;
VL_EXPORT vl_size vl_hog_get_glyph_size (VlHog const * self) ;

VL_EXPORT vl_bool vl_hog_get_use_bilinear_orientation_assignments (VlHog const * self) ;
VL_EXPORT void vl_hog_set_use_bilinear_orientation_assignments (VlHog * self, vl_bool x) ;

/* VL_HOG_H */
#endif
