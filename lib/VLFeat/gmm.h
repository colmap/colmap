/** @file gmm.h
 ** @brief GMM (@ref gmm)
 ** @author David Novotny
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 David Novotny and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_GMM_H
#define VL_GMM_H

#include "kmeans.h"

/** @brief GMM initialization algorithms */
typedef enum _VlGMMInitialization
{
  VlGMMKMeans, /**< Initialize GMM from KMeans clustering. */
  VlGMMRand,   /**< Initialize GMM parameters by selecting points at random. */
  VlGMMCustom  /**< User specifies the initial GMM parameters. */
} VlGMMInitialization ;


#ifndef __DOXYGEN__
struct _VlGMM ;
typedef struct _VlGMM VlGMM ;
#else
/** @brief GMM quantizer */
typedef OPAQUE VlGMM ;
#endif

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT VlGMM * vl_gmm_new (vl_type dataType, vl_size dimension, vl_size numComponents) ;
VL_EXPORT VlGMM * vl_gmm_new_copy (VlGMM const * gmm) ;
VL_EXPORT void vl_gmm_delete (VlGMM * self) ;
VL_EXPORT void vl_gmm_reset (VlGMM * self);
/** @} */

/** @name Basic data processing
 ** @{
 **/
VL_EXPORT double
vl_gmm_cluster
(VlGMM * self,
 void const * data,
 vl_size numData);
/** @} */

/** @name Fine grained data processing
 ** @{ */

VL_EXPORT void
vl_gmm_init_with_rand_data
(VlGMM * self,
 void const * data,
 vl_size numData) ;

VL_EXPORT void
vl_gmm_init_with_kmeans
(VlGMM * self,
 void const * data,
 vl_size numData,
 VlKMeans * kmeansInit);

VL_EXPORT double
vl_gmm_em
(VlGMM * self,
 void const * data,
 vl_size numData);
/** @} */

VL_EXPORT void
vl_gmm_set_means
(VlGMM * self,
 void const * means);

VL_EXPORT void
vl_gmm_set_covariances
(VlGMM * self,
 void const * covariances);

VL_EXPORT void
vl_gmm_set_priors
(VlGMM * self,
 void const * priors);

VL_EXPORT double
vl_get_gmm_data_posteriors_f(float * posteriors,
                             vl_size numClusters,
                             vl_size numData,
                             float const * priors,
                             float const * means,
                             vl_size dimension,
                             float const * covariances,
                             float const * data) ;

VL_EXPORT double
vl_get_gmm_data_posteriors_d(double * posteriors,
                             vl_size numClusters,
                             vl_size numData,
                             double const * priors,
                             double const * means,
                             vl_size dimension,
                             double const * covariances,
                             double const * data) ;
/** @} */

/** @name Set parameters
 ** @{
 **/
VL_EXPORT void vl_gmm_set_num_repetitions (VlGMM * self, vl_size numRepetitions) ;
VL_EXPORT void vl_gmm_set_max_num_iterations (VlGMM * self, vl_size maxNumIterations) ;
VL_EXPORT void vl_gmm_set_verbosity (VlGMM * self, int verbosity) ;
VL_EXPORT void vl_gmm_set_initialization (VlGMM * self, VlGMMInitialization init);
VL_EXPORT void vl_gmm_set_kmeans_init_object (VlGMM * self, VlKMeans * kmeans);
VL_EXPORT void vl_gmm_set_covariance_lower_bounds (VlGMM * self, double const * bounds);
VL_EXPORT void vl_gmm_set_covariance_lower_bound (VlGMM * self, double bound) ;
/** @} */

/** @name Get parameters
 ** @{
 **/
VL_EXPORT void const * vl_gmm_get_means (VlGMM const * self);
VL_EXPORT void const * vl_gmm_get_covariances (VlGMM const * self);
VL_EXPORT void const * vl_gmm_get_priors (VlGMM const * self);
VL_EXPORT void const * vl_gmm_get_posteriors (VlGMM const * self);
VL_EXPORT vl_type vl_gmm_get_data_type (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_dimension (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_num_repetitions (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_num_data (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_num_clusters (VlGMM const * self);
VL_EXPORT double vl_gmm_get_loglikelihood (VlGMM const * self);
VL_EXPORT int vl_gmm_get_verbosity (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_max_num_iterations (VlGMM const * self);
VL_EXPORT vl_size vl_gmm_get_num_repetitions (VlGMM const * self);
VL_EXPORT VlGMMInitialization vl_gmm_get_initialization (VlGMM const * self);
VL_EXPORT VlKMeans * vl_gmm_get_kmeans_init_object (VlGMM const * self);
VL_EXPORT double const * vl_gmm_get_covariance_lower_bounds (VlGMM const * self);
/** @} */

/* VL_GMM_H */
#endif
