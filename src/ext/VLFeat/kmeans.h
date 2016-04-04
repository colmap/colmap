/** @file kmeans.h
 ** @brief K-means (@ref kmeans)
 ** @author Andrea Vedaldi
 ** @author David Novotny
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi and David Novotny.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_KMEANS_H
#define VL_KMEANS_H

#include "generic.h"
#include "random.h"
#include "mathop.h"
#include "kdtree.h"

/* ---------------------------------------------------------------- */

/** @brief K-means algorithms */

typedef enum _VlKMeansAlgorithm {
  VlKMeansLloyd,       /**< Lloyd algorithm */
  VlKMeansElkan,       /**< Elkan algorithm */
  VlKMeansANN          /**< Approximate nearest neighbors */
} VlKMeansAlgorithm ;

/** @brief K-means initialization algorithms */

typedef enum _VlKMeansInitialization {
  VlKMeansRandomSelection,  /**< Randomized selection */
  VlKMeansPlusPlus          /**< Plus plus raondomized selection */
} VlKMeansInitialization ;

/** ------------------------------------------------------------------
 ** @brief K-means quantizer
 **/

typedef struct _VlKMeans
{

  vl_type dataType ;                      /**< Data type. */
  vl_size dimension ;                     /**< Data dimensionality. */
  vl_size numCenters ;                    /**< Number of centers. */
  vl_size numTrees ;                      /**< Number of trees in forest when using ANN-kmeans. */
  vl_size maxNumComparisons ;             /**< Maximum number of comparisons when using ANN-kmeans. */

  VlKMeansInitialization initialization ; /**< Initalization algorithm. */
  VlKMeansAlgorithm algorithm ;           /**< Clustring algorithm. */
  VlVectorComparisonType distance ;       /**< Distance. */
  vl_size maxNumIterations ;              /**< Maximum number of refinement iterations. */
  double minEnergyVariation ;             /**< Minimum energy variation. */
  vl_size numRepetitions ;                /**< Number of clustering repetitions. */
  int verbosity ;                         /**< Verbosity level. */

  void * centers ;                        /**< Centers */
  void * centerDistances ;                /**< Centers inter-distances. */

  double energy ;                         /**< Current solution energy. */
  VlFloatVectorComparisonFunction floatVectorComparisonFn ;
  VlDoubleVectorComparisonFunction doubleVectorComparisonFn ;
} VlKMeans ;

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT VlKMeans * vl_kmeans_new (vl_type dataType, VlVectorComparisonType distance) ;
VL_EXPORT VlKMeans * vl_kmeans_new_copy (VlKMeans const * kmeans) ;
VL_EXPORT void vl_kmeans_delete (VlKMeans * self) ;
/** @} */

/** @name Basic data processing
 ** @{
 **/
VL_EXPORT void vl_kmeans_reset (VlKMeans * self) ;

VL_EXPORT double vl_kmeans_cluster (VlKMeans * self,
                                    void const * data,
                                    vl_size dimension,
                                    vl_size numData,
                                    vl_size numCenters) ;

VL_EXPORT void vl_kmeans_quantize (VlKMeans * self,
                                   vl_uint32 * assignments,
                                   void * distances,
                                   void const * data,
                                   vl_size numData) ;

VL_EXPORT void vl_kmeans_quantize_ANN (VlKMeans * self,
                                   vl_uint32 * assignments,
                                   void * distances,
                                   void const * data,
                                   vl_size numData,
                                   vl_size iteration );
/** @} */

/** @name Advanced data processing
 ** @{
 **/
VL_EXPORT void vl_kmeans_set_centers (VlKMeans * self,
                                      void const * centers,
                                      vl_size dimension,
                                      vl_size numCenters) ;

VL_EXPORT void vl_kmeans_init_centers_with_rand_data
                  (VlKMeans * self,
                   void const * data,
                   vl_size dimensions,
                   vl_size numData,
                   vl_size numCenters) ;

VL_EXPORT void vl_kmeans_init_centers_plus_plus
                  (VlKMeans * self,
                   void const * data,
                   vl_size dimensions,
                   vl_size numData,
                   vl_size numCenters) ;

VL_EXPORT double vl_kmeans_refine_centers (VlKMeans * self,
                                           void const * data,
                                           vl_size numData) ;

/** @} */

/** @name Retrieve data and parameters
 ** @{
 **/
VL_INLINE vl_type vl_kmeans_get_data_type (VlKMeans const * self) ;
VL_INLINE VlVectorComparisonType vl_kmeans_get_distance (VlKMeans const * self) ;

VL_INLINE VlKMeansAlgorithm vl_kmeans_get_algorithm (VlKMeans const * self) ;
VL_INLINE VlKMeansInitialization vl_kmeans_get_initialization (VlKMeans const * self) ;
VL_INLINE vl_size vl_kmeans_get_num_repetitions (VlKMeans const * self) ;

VL_INLINE vl_size vl_kmeans_get_dimension (VlKMeans const * self) ;
VL_INLINE vl_size vl_kmeans_get_num_centers (VlKMeans const * self) ;

VL_INLINE int vl_kmeans_get_verbosity (VlKMeans const * self) ;
VL_INLINE vl_size vl_kmeans_get_max_num_iterations (VlKMeans const * self) ;
VL_INLINE double vl_kmeans_get_min_energy_variation (VlKMeans const * self) ;
VL_INLINE vl_size vl_kmeans_get_max_num_comparisons (VlKMeans const * self) ;
VL_INLINE vl_size vl_kmeans_get_num_trees (VlKMeans const * self) ;
VL_INLINE double vl_kmeans_get_energy (VlKMeans const * self) ;
VL_INLINE void const * vl_kmeans_get_centers (VlKMeans const * self) ;
/** @} */

/** @name Set parameters
 ** @{
 **/
VL_INLINE void vl_kmeans_set_algorithm (VlKMeans * self, VlKMeansAlgorithm algorithm) ;
VL_INLINE void vl_kmeans_set_initialization (VlKMeans * self, VlKMeansInitialization initialization) ;
VL_INLINE void vl_kmeans_set_num_repetitions (VlKMeans * self, vl_size numRepetitions) ;
VL_INLINE void vl_kmeans_set_max_num_iterations (VlKMeans * self, vl_size maxNumIterations) ;
VL_INLINE void vl_kmeans_set_min_energy_variation (VlKMeans * self, double minEnergyVariation) ;
VL_INLINE void vl_kmeans_set_verbosity (VlKMeans * self, int verbosity) ;
VL_INLINE void vl_kmeans_set_max_num_comparisons (VlKMeans * self, vl_size maxNumComparisons) ;
VL_INLINE void vl_kmeans_set_num_trees (VlKMeans * self, vl_size numTrees) ;
/** @} */

/** ------------------------------------------------------------------
 ** @brief Get data type
 ** @param self KMeans object instance.
 ** @return data type.
 **/

VL_INLINE vl_type
vl_kmeans_get_data_type (VlKMeans const * self)
{
  return self->dataType ;
}

/** @brief Get data dimension
 ** @param self KMeans object instance.
 ** @return data dimension.
 **/

VL_INLINE vl_size
vl_kmeans_get_dimension (VlKMeans const * self)
{
  return self->dimension ;
}

/** @brief Get data type
 ** @param self KMeans object instance.
 ** @return data type.
 **/

VL_INLINE VlVectorComparisonType
vl_kmeans_get_distance (VlKMeans const * self)
{
  return self->distance ;
}

/** @brief Get the number of centers (K)
 ** @param self KMeans object instance.
 ** @return number of centers.
 **/

VL_INLINE vl_size
vl_kmeans_get_num_centers (VlKMeans const * self)
{
  return self->numCenters ;
}

/** @brief Get the number energy of the current fit
 ** @param self KMeans object instance.
 ** @return energy.
 **/

VL_INLINE double
vl_kmeans_get_energy (VlKMeans const * self)
{
  return self->energy ;
}

/** ------------------------------------------------------------------
 ** @brief Get verbosity level
 ** @param self KMeans object instance.
 ** @return verbosity level.
 **/

VL_INLINE int
vl_kmeans_get_verbosity (VlKMeans const * self)
{
  return self->verbosity ;
}

/** @brief Set verbosity level
 ** @param self KMeans object instance.
 ** @param verbosity verbosity level.
 **/

VL_INLINE void
vl_kmeans_set_verbosity (VlKMeans * self, int verbosity)
{
  self->verbosity = verbosity ;
}

/** ------------------------------------------------------------------
 ** @brief Get centers
 ** @param self KMeans object instance.
 ** @return cluster centers.
 **/

VL_INLINE void const *
vl_kmeans_get_centers (VlKMeans const * self)
{
  return self->centers ;
}

/** ------------------------------------------------------------------
 ** @brief Get maximum number of iterations
 ** @param self KMeans object instance.
 ** @return maximum number of iterations.
 **/

VL_INLINE vl_size
vl_kmeans_get_max_num_iterations (VlKMeans const * self)
{
  return self->maxNumIterations ;
}

/** @brief Set maximum number of iterations
 ** @param self KMeans filter.
 ** @param maxNumIterations maximum number of iterations.
 **/

VL_INLINE void
vl_kmeans_set_max_num_iterations (VlKMeans * self, vl_size maxNumIterations)
{
  self->maxNumIterations = maxNumIterations ;
}

/** ------------------------------------------------------------------
 ** @brief Get maximum number of repetitions.
 ** @param self KMeans object instance.
 ** @return current number of repretitions for quantization.
 **/

VL_INLINE vl_size
vl_kmeans_get_num_repetitions (VlKMeans const * self)
{
  return self->numRepetitions ;
}

/** @brief Set maximum number of repetitions
 ** @param self KMeans object instance.
 ** @param numRepetitions maximum number of repetitions.
 ** The number of repetitions cannot be smaller than 1.
 **/

VL_INLINE void
vl_kmeans_set_num_repetitions (VlKMeans * self,
                               vl_size numRepetitions)
{
  assert (numRepetitions >= 1) ;
  self->numRepetitions = numRepetitions ;
}

/** ------------------------------------------------------------------
 ** @brief Get the minimum relative energy variation for convergence.
 ** @param self KMeans object instance.
 ** @return minimum energy variation.
 **/

VL_INLINE double
vl_kmeans_get_min_energy_variation (VlKMeans const * self)
{
  return self->minEnergyVariation ;
}

/** @brief Set the maximum relative energy variation for convergence.
 ** @param self KMeans object instance.
 ** @param minEnergyVariation maximum number of repetitions.
 ** The variation cannot be negative.
 **
 ** The relative energy variation is calculated after the $t$-th update
 ** to the parameters as:
 **
 ** \[ \epsilon_t =  \frac{E_{t-1} - E_t}{E_0 - E_t} \]
 **
 ** Note that this quantitiy is non-negative since $E_{t+1} \leq E_t$.
 ** Hence, $\epsilon_t$ is the improvement to the energy made in the last
 ** iteration compared to the total improvement so far. The algorithm
 ** stops if this value is less or equal than @a minEnergyVariation.
 **
 ** This test is applied only to the LLoyd and ANN algorithms.
 **/

VL_INLINE void
vl_kmeans_set_min_energy_variation (VlKMeans * self,
                                    double minEnergyVariation)
{
  assert (minEnergyVariation >= 0) ;
  self->minEnergyVariation = minEnergyVariation ;
}

/** ------------------------------------------------------------------
 ** @brief Get K-means algorithm
 ** @param self KMeans object.
 ** @return algorithm.
 **/

VL_INLINE VlKMeansAlgorithm
vl_kmeans_get_algorithm (VlKMeans const * self)
{
  return self->algorithm ;
}

/** @brief Set K-means algorithm
 ** @param self KMeans object.
 ** @param algorithm K-means algorithm.
 **/

VL_INLINE void
vl_kmeans_set_algorithm (VlKMeans * self, VlKMeansAlgorithm algorithm)
{
  self->algorithm = algorithm ;
}

/** ------------------------------------------------------------------
 ** @brief Get K-means initialization algorithm
 ** @param self KMeans object.
 ** @return algorithm.
 **/

VL_INLINE VlKMeansInitialization
vl_kmeans_get_initialization (VlKMeans const * self)
{
  return self->initialization ;
}

/** @brief Set K-means initialization algorithm
 ** @param self KMeans object.
 ** @param initialization initialization.
 **/

VL_INLINE void
vl_kmeans_set_initialization (VlKMeans * self,
                              VlKMeansInitialization initialization)
{
  self->initialization = initialization ;
}

/** ------------------------------------------------------------------
 ** @brief Get the maximum number of comparisons in the KD-forest ANN algorithm.
 ** @param self KMeans object instance.
 ** @return maximum number of comparisons.
 **/

VL_INLINE vl_size
vl_kmeans_get_max_num_comparisons (VlKMeans const * self)
{
  return self->maxNumComparisons ;
}

/** @brief Set maximum number of comparisons in ANN-KD-Tree.
 ** @param self KMeans filter.
 ** @param maxNumComparisons maximum number of comparisons.
 **/

VL_INLINE void
vl_kmeans_set_max_num_comparisons (VlKMeans * self,
                              vl_size maxNumComparisons)
{
    self->maxNumComparisons = maxNumComparisons;
}

/** ------------------------------------------------------------------
 ** @brief Set the number of trees in the KD-forest ANN algorithm
 ** @param self KMeans object instance.
 ** @param numTrees number of trees to use.
 **/

VL_INLINE void
vl_kmeans_set_num_trees (VlKMeans * self, vl_size numTrees)
{
    self->numTrees = numTrees;
}

VL_INLINE vl_size
vl_kmeans_get_num_trees (VlKMeans const * self)
{
    return self->numTrees;
}


/* VL_IKMEANS_H */
#endif
