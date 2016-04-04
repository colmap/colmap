/** @file svmdataset.c
 ** @brief SVM Dataset - Definition
 ** @author Daniele Perrone
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2012 Daniele Perrone.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
@file svmdataset.h
@tableofcontents
@author Daniele Perrone
@author Andrea Vedaldi

The SVM solver object ::VlSvm, supporting SVM learning in VLFeat,
uses an abstraction mechanism to work on arbitrary data types.
This module provides an helper object, ::VlSvmDataset,
that simplify taking advantage of this functionality, supporting for example
different data types and the computation of feature maps out of the box.

<!-- ------------------------------------------------------------- -->
@section svmdataset-starting Getting started
<!-- ------------------------------------------------------------- -->

As discussed in @ref svm-advanced, most linear SVM solvers,
such as the ones implemented in VLFeat in @ref svm, require only two
operations to be defined on the data:

- *Inner product* between a data point $\bx$ and the model vector $\bw$.
  This is implemented by a function of type ::VlSvmInnerProductFunction.
- *Accumulation* of a dataobint $\bx$ to the model vector $\bw$:
  $\bw \leftarrow \bw + \alpha \bx$. This is implemented
  by a function of the type ::VlSvmAccumulateFunction .

The SVM solver needs to know nothing about the data once these two
operations are defined. These functions can do any number of things,
such as supporting different formats for the data (dense or sparse,
float or double), computing feature maps, or expanding compressed
representations such as Product Quantization.

VLFeat provides the helper object ::VlSvmDataset to support some
of these functionalities out of the box (it is important to remark
that its use with the SVM solver ::VlSvm is entirely optional).

Presently, ::VlSvmDataset supports:

- @c float and @c double dense arrays.
- The on-the-fly application of the homogeneous kernel map to implement
  additive non-linear kernels (see @ref homkermap).

For example, to learn a linear SVM on SINGLE data:

@code
int main()
{
  vl_size const numData = 4 ;
  vl_size const dimension = 2 ;
  single x [dimension * numData] = {
    0.0, -0.5,
    0.6, -0.3,
    0.0,  0.5,
    0.6,  0.0} ;
  double y [numData] = {1, 1, -1, 1} ;
  double lambda = 0.01;
  double * const model ;
  double bias ;

  VlSvmDataset * dataset = vl_svmdataset_new (VL_TYPE_SINGLE, x, dimension, numData) ;
  VlSvm * svm = vl_svm_new_with_dataset (VlSvmSolverSgd, dataset, y, lambda) ;

  vl_svm_train(svm) ;

  model = vl_svm_get_model(svm) ;
  bias = vl_svm_get_bias(svm) ;

  printf("model w = [ %f , %f ] , bias b = %f \n",
         model[0],
         model[1],
         bias);

  vl_svm_delete(svm) ;
  vl_svmdataset_delete(dataset) ;
  return 0;
}
@endcode

**/

/* ---------------------------------------------------------------- */
#ifndef VL_SVMDATASET_INSTANTIATING
/* ---------------------------------------------------------------- */

#include "svmdataset.h"
#include <string.h>
#include <math.h>

struct VlSvmDataset_ {
  vl_type dataType ;                /**< Data type. */
  void * data ;                     /**< Pointer to data. */
  vl_size numData ;                 /**< Number of wrapped data. */
  vl_size dimension ;               /**< Data point dimension. */
  VlHomogeneousKernelMap * hom ;    /**< Homogeneous kernel map (optional). */
  void * homBuffer ;                /**< Homogeneous kernel map buffer. */
  vl_size homDimension ;            /**< Homogeneous kernel map dimension. */
} ;

/* templetized parts of the implementation */
#define FLT VL_TYPE_FLOAT
#define VL_SVMDATASET_INSTANTIATING
#include "svmdataset.c"

#define FLT VL_TYPE_DOUBLE
#define VL_SVMDATASET_INSTANTIATING
#include "svmdataset.c"

/** @brief Create a new object wrapping a dataset.
 ** @param dataType of data (@c float and @c double supported).
 ** @param data pointer to the data.
 ** @param dimension the dimension of a data vector.
 ** @param numData number of wrapped data vectors.
 ** @return new object.
 **
 ** The function allocates and returns a new SVM dataset object
 ** wrapping the data pointed by @a data. Note that no copy is made
 ** of data, so the caller should keep the data allocated as the object exists.
 **
 ** @sa ::vl_svmdataset_delete
 **/

VlSvmDataset*
vl_svmdataset_new (vl_type dataType, void *data, vl_size dimension, vl_size numData)
{
  VlSvmDataset * self ;
  assert(dataType == VL_TYPE_DOUBLE || dataType == VL_TYPE_FLOAT) ;
  assert(data) ;

  self = vl_calloc(1, sizeof(VlSvmDataset)) ;
  if (self == NULL) return NULL ;

  self->dataType = dataType ;
  self->data = data ;
  self->dimension = dimension ;
  self->numData = numData ;
  self->hom = NULL ;
  self->homBuffer = NULL ;
  return self ;
}

/** @brief Delete the object.
 ** @param self object to delete.
 **
 ** The function frees the resources allocated by
 ** ::vl_svmdataset_new(). Notice that the wrapped data will *not*
 ** be freed as it is not owned by the object.
 **/

void vl_svmdataset_delete (VlSvmDataset *self)
{
  if (self->homBuffer) {
    vl_free(self->homBuffer) ;
    self->homBuffer = 0 ;
  }
  vl_free (self) ;
}

/** @brief Get the wrapped data.
 ** @param self object.
 ** @return a pointer to the wrapped data.
 **/

void*
vl_svmdataset_get_data (VlSvmDataset const *self)
{
  return self->data ;
}

/** @brief Get the number of wrapped data elements.
 ** @param self object.
 ** @return number of wrapped data elements.
 **/

vl_size
vl_svmdataset_get_num_data (VlSvmDataset const *self)
{
  return self->numData ;
}

/** @brief Get the dimension of the wrapped data.
 ** @param self object.
 ** @return dimension of the wrapped data.
 **/

vl_size
vl_svmdataset_get_dimension (VlSvmDataset const *self)
{
  if (self->hom) {
    return self->dimension * vl_homogeneouskernelmap_get_dimension(self->hom) ;
  }
  return self->dimension ;
}

/** @brief Get the homogeneous kernel map object.
 ** @param self object.
 ** @return homogenoeus kernel map object (or @c NULL if any).
 **/

VlHomogeneousKernelMap *
vl_svmdataset_get_homogeneous_kernel_map (VlSvmDataset const *self)
{
  assert(self) ;
  return self->hom ;
}

/** @brief Set the homogeneous kernel map object.
 ** @param self object.
 ** @param hom homogeneous kernel map object to use.
 **
 ** After changing the kernel map, the inner product and accumulator
 ** function should be queried again (::vl_svmdataset_get_inner_product_function
 ** adn ::vl_svmdataset_get_accumulate_function).
 **
 ** Set this to @c NULL to avoid using a kernel map.
 **
 ** Note that this does *not* transfer the ownership of the object
 ** to the function. Furthermore, ::VlSvmDataset holds to the
 ** object until it is destroyed or the object is replaced or removed
 ** by calling this function again.
 **/

void
vl_svmdataset_set_homogeneous_kernel_map (VlSvmDataset * self,
                                          VlHomogeneousKernelMap * hom)
{
  assert(self) ;
  self->hom = hom ;
  self->homDimension = 0 ;
  if (self->homBuffer) {
    vl_free (self->homBuffer) ;
    self->homBuffer = 0 ;
  }
  if (self->hom) {
    self->homDimension = vl_homogeneouskernelmap_get_dimension(self->hom) ;
    self->homBuffer = vl_calloc(self->homDimension, vl_get_type_size(self->dataType)) ;
  }
}

/** @brief Get the accumulate function
 ** @param self object.
 ** @return a pointer to the accumulate function to use with this data.
 **/

VlSvmAccumulateFunction
vl_svmdataset_get_accumulate_function(VlSvmDataset const *self)
{
  if (self->hom == NULL) {
    switch (self->dataType) {
      case VL_TYPE_FLOAT:
        return (VlSvmAccumulateFunction) vl_svmdataset_accumulate_f ;
        break ;
      case VL_TYPE_DOUBLE:
        return (VlSvmAccumulateFunction) vl_svmdataset_accumulate_d ;
        break ;
    }
  } else {
    switch (self->dataType) {
      case VL_TYPE_FLOAT:
        return (VlSvmAccumulateFunction) vl_svmdataset_accumulate_hom_f ;
        break ;
      case VL_TYPE_DOUBLE:
        return (VlSvmAccumulateFunction) vl_svmdataset_accumulate_hom_d ;
        break ;
    }
  }
  assert(0) ;
  return NULL ;
}

/** @brief Get the inner product function.
 ** @param self object.
 ** @return a pointer to the inner product function to use with this data.
 **/

VlSvmInnerProductFunction
vl_svmdataset_get_inner_product_function (VlSvmDataset const *self)
{
  if (self->hom == NULL) {
    switch (self->dataType) {
      case VL_TYPE_FLOAT:
        return (VlSvmInnerProductFunction) _vl_svmdataset_inner_product_f ;
        break ;
      case VL_TYPE_DOUBLE:
        return (VlSvmInnerProductFunction) _vl_svmdataset_inner_product_d ;
        break ;
      default:
        assert(0) ;
    }
  } else {
    switch (self->dataType) {
      case VL_TYPE_FLOAT:
        return (VlSvmInnerProductFunction) _vl_svmdataset_inner_product_hom_f ;
        break ;
      case VL_TYPE_DOUBLE:
        return (VlSvmInnerProductFunction) _vl_svmdataset_inner_product_hom_d ;
        break ;
      default:
        assert(0) ;
    }
  }

  return NULL;
}

/* VL_SVMDATASET_INSTANTIATING */
#endif

/* ---------------------------------------------------------------- */
#ifdef VL_SVMDATASET_INSTANTIATING
/* ---------------------------------------------------------------- */

#include "float.th"

double
VL_XCAT(_vl_svmdataset_inner_product_,SFX) (VlSvmDataset const *self,
                                            vl_uindex element,
                                            double const *model)
{
  double product = 0 ;
  T* data = ((T*)self->data) + self->dimension * element ;
  T* end = data + self->dimension ;
  while (data != end) {
    product += (*data++) * (*model++) ;
  }
  return product ;
}

void
VL_XCAT(vl_svmdataset_accumulate_,SFX)(VlSvmDataset const *self,
                                       vl_uindex element,
                                       double *model,
                                       const double multiplier)
{
  T* data = ((T*)self->data) + self->dimension * element ;
  T* end = data + self->dimension ;
  while (data != end) {
    *model += (*data++) * multiplier ;
    model++ ;
  }
}

double
VL_XCAT(_vl_svmdataset_inner_product_hom_,SFX) (VlSvmDataset const *self,
                                                vl_uindex element,
                                                double const *model)
{
  double product = 0 ;
  T* data = ((T*)self->data) + self->dimension * element ;
  T* end = data + self->dimension ;
  T* bufEnd = ((T*)self->homBuffer)+ self->homDimension ;
  while (data != end) {
    /* TODO: zeros in data could be optimized by skipping over them */
    T* buf = self->homBuffer ;
    VL_XCAT(vl_homogeneouskernelmap_evaluate_,SFX)(self->hom,
                                                   self->homBuffer,
                                                   1,
                                                   (*data++)) ;
    while (buf != bufEnd) {
      product += (*buf++) * (*model++) ;
    }
  }
  return product ;
}

void
VL_XCAT(vl_svmdataset_accumulate_hom_,SFX)(VlSvmDataset const *self,
                                           vl_uindex element,
                                           double *model,
                                           const double multiplier)
{
  T* data = ((T*)self->data) + self->dimension * element ;
  T* end = data + self->dimension ;
  T* bufEnd = ((T*)self->homBuffer)+ self->homDimension ;
  while (data != end) {
    /* TODO: zeros in data could be optimized by skipping over them */
    T* buf = self->homBuffer ;
    VL_XCAT(vl_homogeneouskernelmap_evaluate_,SFX)(self->hom,
                                                   self->homBuffer,
                                                   1,
                                                   (*data++)) ;
    while (buf != bufEnd) {
      *model += (*buf++) * multiplier ;
      model++ ;
    }
  }
}

#undef FLT
#undef VL_SVMDATASET_INSTANTIATING

/* VL_SVMDATASET_INSTANTIATING */
#endif
