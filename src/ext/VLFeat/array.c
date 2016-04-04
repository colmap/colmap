/** @file array.h
 ** @brief Array
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "array.h"
#include <string.h>

/** @brief Get number of elements in array
 ** @param self array.
 ** @return number of elements.
 **/

VL_EXPORT vl_size
vl_array_get_num_elements (VlArray const * self)
{
  vl_size numElements = 1 ;
  vl_uindex k ;
  if (self->numDimensions == 0) {
    return 0 ;
  }
  for (k = 0 ; k < self->numDimensions ; ++k) {
    numElements *= self->dimensions[k] ;
  }
  return numElements ;
}

/* ---------------------------------------------------------------- */
/*                                                  init &  dealloc */
/* ---------------------------------------------------------------- */

/** @brief New numeric array
 ** @param self array to initialize.
 ** @param type data type.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 **
 ** The function initializes the specified array and allocates
 ** the necessary memory for storage.
 **/

VL_EXPORT VlArray *
vl_array_init (VlArray* self, vl_type type,
               vl_size numDimensions, vl_size const * dimensions)
{
  assert (numDimensions <= VL_ARRAY_MAX_NUM_DIMENSIONS) ;
  self->type = type ;
  self->numDimensions = numDimensions ;
  memcpy(self->dimensions, dimensions, sizeof(vl_size) * numDimensions) ;
  self->data = vl_malloc(vl_get_type_size(type) * vl_array_get_num_elements (self)) ;
  self->isEnvelope = VL_FALSE ;
  self->isSparse = VL_FALSE ;
  return self ;
}

/** @brief New numeric array envelope
 ** @param self array to initialize.
 ** @param data data to envelople.
 ** @param type data type.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 **
 ** The function initializes the specified array wrapping the
 ** specified buffer.
 **/

VL_EXPORT VlArray *
vl_array_init_envelope (VlArray * self, void * data, vl_type type,
                        vl_size numDimensions, vl_size const * dimensions)
{
  assert (numDimensions <= VL_ARRAY_MAX_NUM_DIMENSIONS) ;
  self->type = type ;
  self->numDimensions = numDimensions ;
  memcpy(self->dimensions, dimensions, sizeof(vl_size) * numDimensions) ;
  self->data = data ;
  self->isEnvelope = VL_TRUE ;
  self->isSparse = VL_FALSE ;
  return self ;
}

/** @brief New numeric array with matrix shape
 ** @param self array to initialize.
 ** @param type type.
 ** @param numRows number of rows.
 ** @param numColumns number of columns.
 **/

VL_EXPORT VlArray *
vl_array_init_matrix (VlArray * self, vl_type type, vl_size numRows, vl_size numColumns)
{
  vl_size dimensions [2] = {numRows, numColumns} ;
  return vl_array_init (self, type, 2, dimensions) ;
}

/** @brief New numeric array envelpe with matrix shape
 ** @param self array to initialize.
 ** @param data data to envelope.
 ** @param type type.
 ** @param numRows number of rows.
 ** @param numColumns number of columns.
 **/

VL_EXPORT VlArray *
vl_array_init_matrix_envelope (VlArray * self, void * data,
                                vl_type type, vl_size numRows, vl_size numColumns)
{
  vl_size dimensions [2] = {numRows, numColumns} ;
  return vl_array_init_envelope (self, data, type, 2, dimensions) ;
}

/** @brief Delete array
 ** @param self array.
 **/

VL_EXPORT void
vl_array_dealloc (VlArray * self)
{
  if (! self->isEnvelope) {
    if (self->data) {
      vl_free(self->data) ;
      self->data = NULL ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                    new &  delete */
/* ---------------------------------------------------------------- */


/** @brief New numeric array
 ** @param type data type.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 **
 ** The function creates a new VLArray instance and allocates
 ** the necessary memory for storage.
 **/

VL_EXPORT VlArray *
vl_array_new (vl_type type, vl_size numDimensions, vl_size const * dimensions)
{
  VlArray * self = vl_malloc(sizeof(VlArray)) ;
  return vl_array_init(self, type, numDimensions, dimensions) ;
}

/** @brief New numeric array with matrix shape
 ** @param type type.
 ** @param numRows number of rows.
 ** @param numColumns number of columns.
 **/

VL_EXPORT VlArray *
vl_array_new_matrix (vl_type type, vl_size numRows, vl_size numColumns)
{
  vl_size dimensions [2] = {numRows, numColumns} ;
  return vl_array_new (type, 2, dimensions) ;
}

/** @brief New numeric array envelope
 ** @param data data to envelople.
 ** @param type data type.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 **/

VL_EXPORT VlArray *
vl_array_new_envelope (void * data, vl_type type,
                       vl_size numDimensions, vl_size const * dimensions)
{
  VlArray * self = vl_malloc(sizeof(VlArray)) ;
  return vl_array_init_envelope(self, data, type, numDimensions, dimensions) ;
}

/** @brief New numeric array envelpe with matrix shape
 ** @param data data to envelope.
 ** @param type type.
 ** @param numRows number of rows.
 ** @param numColumns number of columns.
 **/

VL_EXPORT VlArray *
vl_array_new_matrix_envelope (void * data, vl_type type, vl_size numRows, vl_size numColumns)
{
  vl_size dimensions [2] = {numRows, numColumns} ;
  return vl_array_new_envelope (data, type, 2, dimensions) ;
}

/** @brief Delete array
 ** @param self array.
 **/

VL_EXPORT void
vl_array_delete (VlArray * self)
{
  vl_array_dealloc(self) ;
  vl_free(self) ;
}
