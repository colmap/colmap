/** @file homkermap.h
 ** @brief Homogeneous kernel map (@ref homkermap)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_HOMKERMAP_H
#define VL_HOMKERMAP_H

#include "generic.h"

#include <math.h>

/** @brief Type of kernel */
typedef enum {
  VlHomogeneousKernelIntersection = 0, /**< intersection kernel */
  VlHomogeneousKernelChi2, /**< Chi2 kernel */
  VlHomogeneousKernelJS /**< Jensen-Shannon kernel */
} VlHomogeneousKernelType ;

/** @brief Type of spectral windowing function  */
typedef enum {
  VlHomogeneousKernelMapWindowUniform = 0, /**< uniform window */
  VlHomogeneousKernelMapWindowRectangular = 1, /**< rectangular window */
} VlHomogeneousKernelMapWindowType ;

#ifndef __DOXYGEN__
struct _VlHomogeneousKernelMap ;
typedef struct _VlHomogeneousKernelMap VlHomogeneousKernelMap ;
#else
/** @brief Homogeneous kernel map object */
typedef OPAQUE VlHomogeneousKernelMap ;
#endif

/** @name Create and destroy
 ** @{ */
VL_EXPORT VlHomogeneousKernelMap *
vl_homogeneouskernelmap_new (VlHomogeneousKernelType kernelType,
                             double gamma,
                             vl_size order,
                             double period,
                             VlHomogeneousKernelMapWindowType windowType) ;
VL_EXPORT void
vl_homogeneouskernelmap_delete (VlHomogeneousKernelMap * self) ;
/** @} */

/** @name Process data
 ** @{ */
VL_EXPORT void
vl_homogeneouskernelmap_evaluate_d (VlHomogeneousKernelMap const * self,
                                    double * destination,
                                    vl_size stride,
                                    double x) ;

VL_EXPORT void
vl_homogeneouskernelmap_evaluate_f (VlHomogeneousKernelMap const * self,
                                    float * destination,
                                    vl_size stride,
                                    double x) ;
/** @} */


/** @name Retrieve data and parameters
 ** @{ */
VL_EXPORT vl_size
vl_homogeneouskernelmap_get_order (VlHomogeneousKernelMap const * self) ;

VL_EXPORT vl_size
vl_homogeneouskernelmap_get_dimension (VlHomogeneousKernelMap const * self) ;

VL_EXPORT VlHomogeneousKernelType
vl_homogeneouskernelmap_get_kernel_type (VlHomogeneousKernelMap const * self) ;

VL_EXPORT VlHomogeneousKernelMapWindowType
vl_homogeneouskernelmap_get_window_type (VlHomogeneousKernelMap const * self) ;
/** @} */

/* VL_HOMKERMAP_H */
#endif
