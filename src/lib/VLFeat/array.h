/** @file array.h
 ** @brief Array - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_ARRAY_H
#define VL_ARRAY_H

#include "generic.h"

/** @brief Maximum number of array dimensions */
#define VL_ARRAY_MAX_NUM_DIMENSIONS 16

/** @brief Numeric array */
typedef struct _VlArray
{
  vl_type type ;
  vl_bool isEnvelope ;
  vl_bool isSparse ;
  vl_size numDimensions ;
  vl_size dimensions [VL_ARRAY_MAX_NUM_DIMENSIONS] ;
  void * data ;
  void * rowPointers ;
  void * columnPointers ;
} VlArray ;


/** @name Get data and parameters
 ** @{ */

/** @brief Get number of dimensions
 ** @param self array.
 ** @return number of dimensions.
 **/

VL_INLINE vl_size
vl_array_get_num_dimensions (VlArray const * self)
{
  return self->numDimensions ;
}

/** @brief Get dimensions
 ** @param self array.
 ** @return dimensions.
 **/

VL_INLINE vl_size const *
vl_array_get_dimensions (VlArray const * self)
{
  return self->dimensions ;
}

/** @brief Get data
 ** @param self array.
 ** @return data.
 **/

VL_INLINE void *
vl_array_get_data (VlArray const * self)
{
  return self->data;
}

/** @brief Get type
 ** @param self array.
 ** @return type.
 **/

VL_INLINE vl_type
vl_array_get_data_type (VlArray const * self)
{
  return self->type ;
}

VL_EXPORT vl_size vl_array_get_num_elements (VlArray const * self) ;

/** @{ */

/** @name Constructing and destroying
 ** @{ */

VL_EXPORT VlArray * vl_array_init (VlArray * self, vl_type type, vl_size numDimension, vl_size const * dimensions) ;
VL_EXPORT VlArray * vl_array_init_envelope (VlArray *self, void * data, vl_type type, vl_size numDimension, vl_size const * dimensions) ;
VL_EXPORT VlArray * vl_array_init_matrix (VlArray * self, vl_type type, vl_size numRows, vl_size numColumns) ;
VL_EXPORT VlArray * vl_array_init_matrix_envelope (VlArray * self, void * data, vl_type type, vl_size numRows, vl_size numColumns) ;

VL_EXPORT VlArray * vl_array_new (vl_type type, vl_size numDimension, vl_size const * dimensions) ;
VL_EXPORT VlArray * vl_array_new_envelope (void * data, vl_type type, vl_size numDimension, vl_size const * dimensions) ;
VL_EXPORT VlArray * vl_array_new_matrix (vl_type type, vl_size numRows, vl_size numColumns) ;
VL_EXPORT VlArray * vl_array_new_matrix_envelope (void * data, vl_type type, vl_size numRows, vl_size numColumns) ;

VL_EXPORT void vl_array_dealloc (VlArray * self) ;
VL_EXPORT void vl_array_delete (VlArray * self) ;
/** @} */

/* VL_ARRAY_H */
#endif
