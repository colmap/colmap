/** @file svm.h
 ** @brief Support Vector Machines (@ref svm)
 ** @author Milan Sulc
 ** @author Daniele Perrone
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 Milan Sulc.
Copyright (C) 2012 Daniele Perrone.
Copyright (C) 2011-13 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_SVM_H
#define VL_SVM_H

#include "generic.h"
#include "svmdataset.h"

/** @typedef VlSvm
 ** @brief SVM solver.
 ** This object implements VLFeat SVM solvers (see @ref svm.h).
 **/

#ifndef __DOXYGEN__
struct VlSvm_ ;
typedef struct VlSvm_ VlSvm ;
#else
typedef OPAQUE VlSvm ;
#endif

/** @brief Type of SVM solver */
typedef enum
{
  VlSvmSolverNone = 0, /**< No solver (used to evaluate an SVM). */
  VlSvmSolverSgd = 1,  /**< SGD algorithm (@ref svm-sgd). */
  VlSvmSolverSdca      /**< SDCA algorithm (@ref svm-sdca). */
} VlSvmSolverType ;

/** @brief Type of SVM loss
 **
 ** Default SVM loss types. The loss can be set by using ::vl_svm_set_loss.
 ** Note that custom losses can be used too by using ::vl_svm_set_loss_function,
 ** ::vl_svm_set_loss_derivative_function, etc.
 **
 ** @sa svm-loss-functions
 **/
typedef enum
{
  VlSvmLossHinge = 0,   /**< Standard hinge loss. */
  VlSvmLossHinge2 = 1,  /**< Hinge loss squared. */
  VlSvmLossL1,          /**< L1 loss. */
  VlSvmLossL2,          /**< L2 loss. */
  VlSvmLossLogistic     /**< Logistic loss. */
} VlSvmLossType ;

/** @brief Solver status */
typedef enum
{
  VlSvmStatusTraining = 1, /**< Optimization in progress. */
  VlSvmStatusConverged, /**< Optimization finished because the convergence criterion was met. */
  VlSvmStatusMaxNumIterationsReached /**< Optimization finished without convergence. */
} VlSvmSolverStatus ;

/** @brief SVM statistics
 ** This structure contains statistics characterising the state of
 ** the SVM solver, such as the current value of the objective function.
 **
 ** Not all fields are used by all solvers.
 **/
typedef struct VlSvmStatistics_ {
  VlSvmSolverStatus status ;    /**< Solver status. */
  vl_size iteration ;           /**< Solver iteration. */
  vl_size epoch ;               /**< Solver epoch (iteration / num samples). */
  double objective ;            /**< Objective function value. */
  double regularizer ;          /**< Regularizer value. */
  double loss ;                 /**< Loss value. */
  double dualObjective ;        /**< Dual objective value. */
  double dualLoss ;             /**< Dual loss value. */
  double dualityGap ;           /**< Duality gap = objective - dualObjective. */
  double scoresVariation ;      /**< Variance of the score updates. */
  double elapsedTime ;          /**< Time elapsed from the start of training. */
} VlSvmStatistics ;

/** @name Create and destroy
 ** @{ */
VL_EXPORT VlSvm * vl_svm_new (VlSvmSolverType type,
                              double const * data,
                              vl_size dimension,
                              vl_size numData,
                              double const * labels,
                              double lambda) ;

VL_EXPORT VlSvm * vl_svm_new_with_dataset (VlSvmSolverType type,
                                           VlSvmDataset * dataset,
                                           double const * labels,
                                           double lambda) ;

VL_EXPORT VlSvm * vl_svm_new_with_abstract_data (VlSvmSolverType type,
                                              void * data,
                                              vl_size dimension,
                                              vl_size numData,
                                              double const * labels,
                                              double lambda) ;

VL_EXPORT void vl_svm_delete (VlSvm * self) ;
/** @} */

/** @name Retrieve parameters and data
 ** @{ */
VL_EXPORT VlSvmStatistics const * vl_svm_get_statistics (VlSvm const *self) ;
VL_EXPORT double const * vl_svm_get_model (VlSvm const *self) ;
VL_EXPORT double vl_svm_get_bias (VlSvm const *self) ;
VL_EXPORT vl_size vl_svm_get_dimension (VlSvm *self) ;
VL_EXPORT vl_size vl_svm_get_num_data (VlSvm *self) ;
VL_EXPORT double vl_svm_get_epsilon (VlSvm const *self) ;
VL_EXPORT double vl_svm_get_bias_learning_rate (VlSvm const *self) ;
VL_EXPORT vl_size vl_svm_get_max_num_iterations (VlSvm const *self) ;
VL_EXPORT vl_size vl_svm_get_diagnostic_frequency (VlSvm const *self) ;
VL_EXPORT VlSvmSolverType vl_svm_get_solver (VlSvm const *self) ;
VL_EXPORT double vl_svm_get_bias_multiplier (VlSvm const *self) ;
VL_EXPORT double vl_svm_get_lambda (VlSvm const *self) ;
VL_EXPORT vl_size vl_svm_get_iteration_number (VlSvm const *self) ;
VL_EXPORT double const * vl_svm_get_scores (VlSvm const *self) ;
VL_EXPORT double const * vl_svm_get_weights (VlSvm const *self) ;
/** @} */

/** @name Set parameters
 ** @{ */
VL_EXPORT void vl_svm_set_epsilon (VlSvm *self, double epsilon) ;
VL_EXPORT void vl_svm_set_bias_learning_rate (VlSvm *self, double rate) ;
VL_EXPORT void vl_svm_set_max_num_iterations (VlSvm *self, vl_size maxNumIterations) ;
VL_EXPORT void vl_svm_set_diagnostic_frequency (VlSvm *self, vl_size f) ;
VL_EXPORT void vl_svm_set_bias_multiplier (VlSvm *self, double b) ;
VL_EXPORT void vl_svm_set_model (VlSvm *self, double const *model) ;
VL_EXPORT void vl_svm_set_bias (VlSvm *self, double b) ;
VL_EXPORT void vl_svm_set_iteration_number (VlSvm *self, vl_uindex n) ;
VL_EXPORT void vl_svm_set_weights (VlSvm *self, double const *weights) ;

VL_EXPORT void vl_svm_set_diagnostic_function (VlSvm *self, VlSvmDiagnosticFunction f, void *data) ;
VL_EXPORT void vl_svm_set_loss_function (VlSvm *self, VlSvmLossFunction f) ;
VL_EXPORT void vl_svm_set_loss_derivative_function (VlSvm *self, VlSvmLossFunction f) ;
VL_EXPORT void vl_svm_set_conjugate_loss_function (VlSvm *self, VlSvmLossFunction f) ;
VL_EXPORT void vl_svm_set_dca_update_function (VlSvm *self, VlSvmDcaUpdateFunction f) ;
VL_EXPORT void vl_svm_set_data_functions (VlSvm *self, VlSvmInnerProductFunction inner, VlSvmAccumulateFunction acc) ;
VL_EXPORT void vl_svm_set_loss (VlSvm *self, VlSvmLossType loss) ;
/** @} */

/** @name Process data
 ** @{ */
VL_EXPORT void vl_svm_train (VlSvm * self) ;
/** @} */

/** @name Loss functions
 ** @sa @ref svm-advanced
 ** @{ */

/* hinge */
VL_EXPORT double vl_svm_hinge_loss (double label, double inner) ;
VL_EXPORT double vl_svm_hinge_loss_derivative (double label, double inner) ;
VL_EXPORT double vl_svm_hinge_conjugate_loss (double label, double u) ;
VL_EXPORT double vl_svm_hinge_dca_update (double alpha, double inner, double norm2, double label) ;

/* square hinge */
VL_EXPORT double vl_svm_hinge2_loss (double label, double inner) ;
VL_EXPORT double vl_svm_hinge2_loss_derivative (double label, double inner) ;
VL_EXPORT double vl_svm_hinge2_conjugate_loss (double label, double u) ;
VL_EXPORT double vl_svm_hinge2_dca_update (double alpha, double inner, double norm2, double label) ;

/* l1 */
VL_EXPORT double vl_svm_l1_loss (double label, double inner) ;
VL_EXPORT double vl_svm_l1_loss_derivative (double label, double inner) ;
VL_EXPORT double vl_svm_l1_conjugate_loss (double label, double u) ;
VL_EXPORT double vl_svm_l1_dca_update (double alpha, double inner, double norm2, double label) ;

/* l2 */
VL_EXPORT double vl_svm_l2_loss (double label, double inner) ;
VL_EXPORT double vl_svm_l2_loss_derivative (double label, double inner) ;
VL_EXPORT double vl_svm_l2_conjugate_loss (double label, double u) ;
VL_EXPORT double vl_svm_l2_dca_update (double alpha, double inner, double norm2, double label) ;

/* logistic */
VL_EXPORT double vl_svm_logistic_loss (double label, double inner) ;
VL_EXPORT double vl_svm_logistic_loss_derivative (double label, double inner) ;
VL_EXPORT double vl_svm_logistic_conjugate_loss (double label, double u) ;
VL_EXPORT double vl_svm_logistic_dca_update (double alpha, double inner, double norm2, double label) ;
/** } */

/* VL_SVM_H */
#endif
