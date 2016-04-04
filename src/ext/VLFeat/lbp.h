/** @file lbp.h
 ** @brief Local Binary Patterns (LBP) descriptor (@ref lbp)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_LBP_H
#define VL_LBP_H

#include "generic.h"

/** @brief Type of quantization for the LBP descriptors
 ** @see @ref lbp-quantization
 **/
typedef enum _VlLbpMappingType
{
  VlLbpUniform     /**< Uniform local binary patterns. */
} VlLbpMappingType ;

/** @brief Local Binary Pattern extractor */
typedef struct VlLbp_
{
  vl_size dimension ;
  vl_uint8 mapping [256] ;
  vl_bool transposed ;
} VlLbp ;

VL_EXPORT VlLbp * vl_lbp_new(VlLbpMappingType type, vl_bool transposed) ;
VL_EXPORT void vl_lbp_delete(VlLbp * self) ;
VL_EXPORT void vl_lbp_process(VlLbp * self,
                              float * features,
                              float * image, vl_size width, vl_size height,
                              vl_size cellSize) ;
VL_EXPORT vl_size vl_lbp_get_dimension(VlLbp * self) ;

/* VL_LBP_H */
#endif
