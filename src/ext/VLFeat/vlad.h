/** @file vlad.h
 ** @brief VLAD encoding (@ref vlad)
 ** @author David Novotny
 ** @author Andrea Vedaldi
 ** @see @ref vlad
 **/

/*
Copyright (C) 2013 David Novotny and Andera Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_VLAD_H
#define VL_VLAD_H

#include "generic.h"

/** @name VLAD options
 ** @{ */
#define VL_VLAD_FLAG_NORMALIZE_COMPONENTS (0x1 << 0)
#define VL_VLAD_FLAG_SQUARE_ROOT          (0x1 << 1)
#define VL_VLAD_FLAG_UNNORMALIZED         (0x1 << 2)
#define VL_VLAD_FLAG_NORMALIZE_MASS       (0x1 << 3)

/** @def VL_VLAD_FLAG_NORMALIZE_COMPONENTS
 ** @brief Normalize each VLAD component individually.
 **/

/** @def VL_VLAD_FLAG_SQUARE_ROOT
 ** @brief Use signed squared-root.
 **/

/** @def VL_VLAD_FLAG_UNNORMALIZED
 ** @brief Do not globally normalize the VLAD descriptor.
 **/

/** @def VL_VLAD_FLAG_NORMALIZE_MASS
 ** @brief Normalize each component by the number of features assigned to it.
 **/
/** @} */

VL_EXPORT void vl_vlad_encode
  (void * enc, vl_type dataType,
   void const * means, vl_size dimension, vl_size numClusters,
   void const * data, vl_size numData,
   void const * assignments,
   int flags) ;

/* VL_VLAD_H */
#endif
