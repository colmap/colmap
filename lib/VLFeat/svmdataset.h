/** @file svmdataset.h
 ** @brief SVM Dataset
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

#ifndef VL_SVMDATASET_H
#define VL_SVMDATASET_H

#include "generic.h"
#include "homkermap.h"

struct VlSvm_ ;

/** @typedef VlSvmDataset
 ** @brief SVM dataset object
 **
 ** This objects contain a training set to be used in combination with
 ** the SVM solver object ::VlSvm. Its main purpose is to implement
 ** the two basic operations inner product (::VlSvmInnerProductFunction)
 ** and accumulation (::VlSvmAccumulateFunction).
 **
 ** See @ref svm and @ref svm-advanced for further information.
 **/

#ifndef __DOXYGEN__
struct VlSvmDataset_ ;
typedef struct VlSvmDataset_ VlSvmDataset ;
#else
typedef OPAQUE VlSvmDataset ;
#endif

/** @name SVM callbacks
 ** @{ */
typedef void (*VlSvmDiagnosticFunction) (struct VlSvm_ *svm, void *data) ;
typedef double (*VlSvmLossFunction) (double inner, double label) ;
typedef double (*VlSvmDcaUpdateFunction) (double alpha, double inner, double norm2, double label) ;
typedef double (*VlSvmInnerProductFunction)(const void *data, vl_uindex element, double *model) ;
typedef void (*VlSvmAccumulateFunction) (const void *data, vl_uindex element, double *model, double multiplier) ;
/* typedef double (*VlSvmSquareNormFunction) (const void *data, vl_uindex element) ; */
/** @} */

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT VlSvmDataset* vl_svmdataset_new (vl_type dataType, void *data, vl_size dimension, vl_size numData) ;
VL_EXPORT void vl_svmdataset_delete (VlSvmDataset * dataset) ;
/** @} */

/** @name Set parameters
 ** @{
 **/
VL_EXPORT void vl_svmdataset_set_homogeneous_kernel_map (VlSvmDataset * self,
                                                         VlHomogeneousKernelMap * hom) ;
/** @} */

/** @name Get data and parameters
 ** @{
 **/
VL_EXPORT void* vl_svmdataset_get_data (VlSvmDataset const *self) ;
VL_EXPORT vl_size vl_svmdataset_get_num_data (VlSvmDataset const *self) ;
VL_EXPORT vl_size vl_svmdataset_get_dimension (VlSvmDataset const *self) ;
VL_EXPORT void* vl_svmdataset_get_map (VlSvmDataset const *self) ;
VL_EXPORT vl_size vl_svmdataset_get_mapDim (VlSvmDataset const *self) ;
VL_EXPORT VlSvmAccumulateFunction vl_svmdataset_get_accumulate_function (VlSvmDataset const *self) ;
VL_EXPORT VlSvmInnerProductFunction vl_svmdataset_get_inner_product_function (VlSvmDataset const * self) ;
VL_EXPORT VlHomogeneousKernelMap * vl_svmdataset_get_homogeneous_kernel_map (VlSvmDataset const * self) ;
/** @} */

/* VL_SVMDATASET_H */
#endif


