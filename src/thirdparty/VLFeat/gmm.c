/** @file gmm.c
 ** @brief Gaussian Mixture Models - Implementation
 ** @author David Novotny
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 David Novotny and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page gmm Gaussian Mixture Models (GMM)
@author David Novotny
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref gmm.h is an implementation of *Gaussian Mixture Models* (GMMs).
The main functionality provided by this module is learning GMMs from
data by maximum likelihood. Model optimization uses the Expectation
Maximization (EM) algorithm @cite{dempster77maximum}. The
implementation supports @c float or @c double data types, is
parallelized, and is tuned to work reliably and effectively on
datasets of visual features. Stability is obtained in part by
regularizing and restricting the parameters of the GMM.

@ref gmm-starting demonstreates how to use the C API to compute the FV
representation of an image. For further details refer to:

- @subpage gmm-fundamentals

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section gmm-starting Getting started
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In order to use @ref gmm.h to learn a GMM from training data, create a
new ::VlGMM object instance, set the parameters as desired, and run
the training code. The following example learns @c numClusters
Gaussian components from @c numData vectors of dimension @c dimension
and storage class @c float using at most 100 EM iterations:

@code
float * means ;
float * covariances ;
float * priors ;
float * posteriors ;

double loglikelihood ;

// create a new instance of a GMM object for float data
gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numClusters) ;

// set the maximum number of EM iterations to 100
vl_gmm_set_max_num_iterations (gmm, 100) ;

// set the initialization to random selection
vl_gmm_set_initialization (gmm,VlGMMRand);

// cluster the data, i.e. learn the GMM
vl_gmm_cluster (gmm, data, numData);

// get the means, covariances, and priors of the GMM
means = vl_gmm_get_means(gmm);
covariances = vl_gmm_get_covariances(gmm);
priors = vl_gmm_get_priors(gmm);

// get loglikelihood of the estimated GMM
loglikelihood = vl_gmm_get_loglikelihood(gmm) ;

// get the soft assignments of the data points to each cluster
posteriors = vl_gmm_get_posteriors(gmm) ;
@endcode

@note ::VlGMM assumes that the covariance matrices of the GMM are
diagonal. This reduces significantly the number of parameters to learn
and is usually an acceptable compromise in vision applications. If the
data is significantly correlated, it can be beneficial to de-correlate
it by PCA rotation or projection in pre-processing.

::vl_gmm_get_loglikelihood is used to get the final loglikelihood of
the estimated mixture, ::vl_gmm_get_means and ::vl_gmm_get_covariances
to obtain the means and the diagonals of the covariance matrices of
the estimated Gaussian modes, and ::vl_gmm_get_posteriors to get the
posterior probabilities that a given point is associated to each of
the modes (soft assignments).

The learning algorithm, which uses EM, finds a local optimum of the
objective function. Therefore the initialization is crucial in
obtaining a good model, measured in term of the final
loglikelihood. ::VlGMM supports a few methods (use
::vl_gmm_set_initialization to choose one) as follows:

Method                | ::VlGMMInitialization enumeration       | Description
----------------------|-----------------------------------------|-----------------------------------------------
Random initialization | ::VlGMMRand                             | Random initialization of the mixture parameters
KMeans                | ::VlGMMKMeans                           | Initialization of the mixture parameters using ::VlKMeans
Custom                | ::VlGMMCustom                           | User specified initialization

Note that in the case of ::VlGMMKMeans initialization, an object of
type ::VlKMeans object must be created and passed to the ::VlGMM
instance (see @ref kmeans to see how to correctly set up this object).

When a user wants to use the ::VlGMMCustom method, the initial means,
covariances and priors have to be specified using the
::vl_gmm_set_means, ::vl_gmm_set_covariances and ::vl_gmm_set_priors
methods.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page gmm-fundamentals GMM fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A *Gaussian Mixture Model* (GMM) is a mixture of $K$ multivariate
Gaussian distributions. In order to sample from a GMM, one samples
first the component index $k \in \{1,\dots,K\}$ with *prior
probability* $\pi_k$, and then samples the vector $\bx \in
\mathbb{R}^d$ from the $k$-th Gaussian distribution
$p(\bx|\mu_k,\Sigma_k)$. Here $\mu_k$ and $\Sigma_k$ are respectively
the *mean* and *covariance* of the distribution. The GMM is completely
specified by the parameters $\Theta=\{\pi_k,\mu_k,\Sigma_k; k =
1,\dots,K\}$

The density $p(\bx|\Theta)$ induced on the training data is obtained
by marginalizing the component selector $k$, obtaining
\[
p(\bx|\Theta)
= \sum_{k=1}^{K} \pi_k p( \bx_i |\mu_k,\Sigma_k),
\qquad
p( \bx |\mu_k,\Sigma_k)
=
\frac{1}{\sqrt{(2\pi)^d\det\Sigma_k}}
\exp\left[
-\frac{1}{2} (\bx-\mu_k)^\top\Sigma_k^{-1}(\bx-\mu_k)
\right].
\]
Learning a GMM to fit a dataset $X=(\bx_1, \dots, \bx_n)$ is usually
done by maximizing the log-likelihood of the data:
@f[
 \ell(\Theta;X)
 = E_{\bx\sim\hat p} [ \log p(\bx|\Theta) ]
 = \frac{1}{n}\sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k p(\bx_i|\mu_k, \Sigma_k)
@f]
where $\hat p$ is the empirical distribution of the data. An algorithm
to solve this problem is introduced next.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section gmm-em Learning a GMM by expectation maximization
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The direct maximization of the log-likelihood function of a GMM is
difficult due to the fact that the assignments of points to Gaussian
mode is not observable and, as such, must be treated as a latent
variable.

Usually, GMMs are learned by using the *Expectation Maximization* (EM)
algorithm @cite{dempster77maximum}. Consider in general the problem of
estimating to the maximum likelihood a distribution $p(x|\Theta) =
\int p(x,h|\Theta)\,dh$, where $x$ is a measurement, $h$ is a *latent
variable*, and $\Theta$ are the model parameters. By introducing an
auxiliary distribution $q(h|x)$ on the latent variable, one can use
Jensen inequality to obtain the following lower bound on the
log-likelihood:

@f{align*}
\ell(\Theta;X) =
E_{x\sim\hat p} \log p(x|\Theta)
&= E_{x\sim\hat p} \log \int p(x,h|\Theta) \,dh \\
&= E_{x\sim\hat p} \log \int \frac{p(x,h|\Theta)}{q(h|x)} q(h|x)\,dh \\
&\geq E_{x\sim\hat p} \int q(h) \log \frac{p(x,h|\Theta)}{q(h|x)}\,dh \\
&= E_{(x,q) \sim q(h|x) \hat p(x)} \log p(x,h|\Theta) -
   E_{(x,q) \sim q(h|x) \hat p(x)} \log q(h|x)
@f}

The first term of the last expression is the log-likelihood of the
model where both the $x$ and $h$ are observed and joinlty distributed
as $q(x|h)\hat p(x)$; the second term is the a average entropy of the
latent variable, which does not depend on $\Theta$. This lower bound
is maximized and becomes tight by setting $q(h|x) = p(h|x,\Theta)$ to
be the posterior distribution on the latent variable $h$ (given the
current estimate of the parameters $\Theta$). In fact:

\[
E_{x \sim \hat p} \log p(x|\Theta)
=
E_{(x,h) \sim p(h|x,\Theta) \hat p(x)}\left[ \log \frac{p(x,h|\Theta)}{p(h|x,\Theta)} \right]
=
E_{(x,h) \sim p(h|x,\Theta) \hat p(x)} [ \log p(x|\Theta) ]
=
\ell(\Theta;X).
\]

EM alternates between updating the latent variable auxiliary
distribution $q(h|x) = p(h|x,\Theta_t)$ (*expectation step*) given the
current estimate of the parameters $\Theta_t$, and then updating the
model parameters $\Theta_{t+1}$ by maximizing the log-likelihood lower
bound derived (*maximization step*). The simplification is that in the
maximization step both $x$ and $h$ are now ``observed'' quantities.
This procedure converges to a local optimum of the model
log-likelihood.

@subsection gmm-expectation-step Expectation step

In the case of a GMM, the latent variables are the point-to-cluster
assignments $k_i, i=1,\dots,n$, one for each of $n$ data points. The
auxiliary distribution $q(k_i|\bx_i) = q_{ik}$ is a matrix with $n
\times K$ entries. Each row $q_{i,:}$ can be thought of as a vector of
soft assignments of the data points $\bx_i$ to each of the Gaussian
modes. Setting $q_{ik} = p(k_i | \bx_i, \Theta)$ yields

\[
 q_{ik} =
\frac
{\pi_k p(\bx_i|\mu_k,\Sigma_k)}
{\sum_{l=1}^K \pi_l p(\bx_i|\mu_l,\Sigma_l)}
\]

where the Gaussian density $p(\bx_i|\mu_k,\Sigma_k)$ was given above.

One important point to keep in mind when these probabilities are
computed is the fact that the Gaussian densities may attain very low
values and underflow in a vanilla implementation. Furthermore, VLFeat
GMM implementation restricts the covariance matrices to be
diagonal. In this case, the computation of the determinant of
$\Sigma_k$ reduces to computing the trace of the matrix and the
inversion of $\Sigma_k$ could be obtained by inverting the elements on
the diagonal of the covariance matrix.

@subsection gmm-maximization-step  Maximization step

The M step estimates the parameters of the Gaussian mixture components
and the prior probabilities $\pi_k$ given the auxiliary distribution
on the point-to-cluster assignments computed in the E step. Since all
the variables are now ``observed'', the estimate is quite simple. For
example, the mean $\mu_k$ of a Gaussian mode is obtained as the mean
of the data points assigned to it (accounting for the strength of the
soft assignments). The other quantities are obtained in a similar
manner, yielding to:

@f{align*}
 \mu_k &= { { \sum_{i=1}^n q_{ik} \bx_{i} } \over { \sum_{i=1}^n q_{ik} } },
\\
 \Sigma_k &= { { \sum_{i=1}^n { q_{ik} (\bx_{i} - \mu_{k}) {(\bx_{i} - \mu_{k})}^T } } \over { \sum_{i=1}^n q_{ik} } },
\\
 \pi_k &= { \sum_{i=1}^n { q_{ik} } \over { \sum_{i=1}^n \sum_{l=1}^K q_{il} } }.
@f}

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section gmm-fundamentals-init Initialization algorithms
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The EM algorithm is a local optimization method. As such, the quality
of the solution strongly depends on the quality of the initial values
of the parameters (i.e.  of the locations and shapes of the Gaussian
modes).

@ref gmm.h supports the following cluster initialization algorithms:

- <b>Random data points.</b> (::vl_gmm_init_with_rand_data) This method
  sets the means of the modes by sampling at random a corresponding
  number of data points, sets the covariance matrices of all the modes
  are to the covariance of the entire dataset, and sets the prior
  probabilities of the Gaussian modes to be uniform. This
  initialization method is the fastest, simplest, as well as the one
  most likely to end in a bad local minimum.

- <b>KMeans initialization</b> (::vl_gmm_init_with_kmeans) This
  method uses KMeans to pre-cluster the points. It then sets the means
  and covariances of the Gaussian distributions the sample means and
  covariances of each KMeans cluster. It also sets the prior
  probabilities to be proportional to the mass of each cluster. In
  order to use this initialization method, a user can specify an
  instance of ::VlKMeans by using the function
  ::vl_gmm_set_kmeans_init_object, or let ::VlGMM create one
  automatically.

Alternatively, one can manually specify a starting point
(::vl_gmm_set_priors, ::vl_gmm_set_means, ::vl_gmm_set_covariances).
**/

#include "gmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef VL_DISABLE_SSE2
#include "mathop_sse2.h"
#endif

#ifndef VL_DISABLE_AVX
#include "mathop_avx.h"
#endif

/* ---------------------------------------------------------------- */
#ifndef VL_GMM_INSTANTIATING
/* ---------------------------------------------------------------- */

#define VL_GMM_MIN_VARIANCE 1e-6
#define VL_GMM_MIN_POSTERIOR 1e-2
#define VL_GMM_MIN_PRIOR 1e-6

struct _VlGMM
{
  vl_type dataType ;                  /**< Data type. */
  vl_size dimension ;                 /**< Data dimensionality. */
  vl_size numClusters ;               /**< Number of clusters  */
  vl_size numData ;                   /**< Number of last time clustered data points.  */
  vl_size maxNumIterations ;          /**< Maximum number of refinement iterations. */
  vl_size numRepetitions   ;          /**< Number of clustering repetitions. */
  int     verbosity ;                 /**< Verbosity level. */
  void *  means;                      /**< Means of Gaussian modes. */
  void *  covariances;                /**< Diagonals of covariance matrices of Gaussian modes. */
  void *  priors;                     /**< Weights of Gaussian modes. */
  void *  posteriors;                 /**< Probabilities of correspondences of points to clusters. */
  double * sigmaLowBound ;            /**< Lower bound on the diagonal covariance values. */
  VlGMMInitialization initialization; /**< Initialization option */
  VlKMeans * kmeansInit;              /**< Kmeans object for initialization of gaussians */
  double LL ;                         /**< Current solution loglikelihood */
  vl_bool kmeansInitIsOwner; /**< Indicates whether a user provided the kmeans initialization object */
} ;

/* ---------------------------------------------------------------- */
/*                                                       Life-cycle */
/* ---------------------------------------------------------------- */

static void
_vl_gmm_prepare_for_data (VlGMM* self, vl_size numData)
{
  if (self->numData < numData) {
    vl_free(self->posteriors) ;
    self->posteriors = vl_malloc(vl_get_type_size(self->dataType) * numData * self->numClusters) ;
  }
  self->numData = numData ;
}

/** @brief Create a new GMM object
 ** @param dataType type of data (::VL_TYPE_FLOAT or ::VL_TYPE_DOUBLE)
 ** @param dimension dimension of the data.
 ** @param numComponents number of Gaussian mixture components.
 ** @return new GMM object instance.
 **/

VlGMM *
vl_gmm_new (vl_type dataType, vl_size dimension, vl_size numComponents)
{
  vl_index i ;
  vl_size size = vl_get_type_size(dataType) ;
  VlGMM * self = vl_calloc(1, sizeof(VlGMM)) ;
  self->dataType = dataType;
  self->numClusters = numComponents ;
  self->numData = 0;
  self->dimension = dimension ;
  self->initialization = VlGMMRand;
  self->verbosity = 0 ;
  self->maxNumIterations = 50;
  self->numRepetitions = 1;
  self->sigmaLowBound =  NULL ;
  self->priors = NULL ;
  self->covariances = NULL ;
  self->means = NULL ;
  self->posteriors = NULL ;
  self->kmeansInit = NULL ;
  self->kmeansInitIsOwner = VL_FALSE;

  self->priors = vl_calloc (numComponents, size) ;
  self->means = vl_calloc (numComponents * dimension, size) ;
  self->covariances = vl_calloc (numComponents * dimension, size) ;
  self->sigmaLowBound = vl_calloc (dimension, sizeof(double)) ;

  for (i = 0 ; i < (unsigned)self->dimension ; ++i)  { self->sigmaLowBound[i] = 1e-4 ; }
  return self ;
}

/** @brief Reset state
 ** @param self object.
 **
 ** The function reset the state of the GMM object. It deletes
 ** any stored posterior and other internal state variables.
 **/

void
vl_gmm_reset (VlGMM * self)
{
  if (self->posteriors) {
    vl_free(self->posteriors) ;
    self->posteriors = NULL ;
    self->numData = 0 ;
  }
  if (self->kmeansInit && self->kmeansInitIsOwner) {
    vl_kmeans_delete(self->kmeansInit) ;
    self->kmeansInit = NULL ;
    self->kmeansInitIsOwner = VL_FALSE ;
  }
}

/** @brief Deletes a GMM object
 ** @param self GMM object instance.
 **
 ** The function deletes the GMM object instance created
 ** by ::vl_gmm_new.
 **/

void
vl_gmm_delete (VlGMM * self)
{
  if(self->means) vl_free(self->means);
  if(self->covariances) vl_free(self->covariances);
  if(self->priors) vl_free(self->priors);
  if(self->posteriors) vl_free(self->posteriors);
  if(self->kmeansInit && self->kmeansInitIsOwner) {
    vl_kmeans_delete(self->kmeansInit);
  }
  vl_free(self);
}

/* ---------------------------------------------------------------- */
/*                                              Getters and setters */
/* ---------------------------------------------------------------- */

/** @brief Get data type
 ** @param self object
 ** @return data type.
 **/

vl_type
vl_gmm_get_data_type (VlGMM const * self)
{
  return self->dataType ;
}

/** @brief Get the number of clusters
 ** @param self object
 ** @return number of clusters.
 **/

vl_size
vl_gmm_get_num_clusters (VlGMM const * self)
{
  return self->numClusters ;
}

/** @brief Get the number of data points
 ** @param self object
 ** @return number of data points.
 **/

vl_size
vl_gmm_get_num_data (VlGMM const * self)
{
  return self->numData ;
}

/** @brief Get the log likelihood of the current mixture
 ** @param self object
 ** @return loglikelihood.
 **/

double
vl_gmm_get_loglikelihood (VlGMM const * self)
{
  return self->LL ;
}

/** @brief Get verbosity level
 ** @param self object
 ** @return verbosity level.
 **/

int
vl_gmm_get_verbosity (VlGMM const * self)
{
  return self->verbosity ;
}

/** @brief Set verbosity level
 ** @param self object
 ** @param verbosity verbosity level.
 **/

void
vl_gmm_set_verbosity (VlGMM * self, int verbosity)
{
  self->verbosity = verbosity ;
}

/** @brief Get means
 ** @param self object
 ** @return cluster means.
 **/

void const *
vl_gmm_get_means (VlGMM const * self)
{
  return self->means ;
}

/** @brief Get covariances
 ** @param self object
 ** @return diagonals of cluster covariance matrices.
 **/

void const *
vl_gmm_get_covariances (VlGMM const * self)
{
  return self->covariances ;
}

/** @brief Get priors
 ** @param self object
 ** @return priors of cluster gaussians.
 **/

void const *
vl_gmm_get_priors (VlGMM const * self)
{
  return self->priors ;
}

/** @brief Get posteriors
 ** @param self object
 ** @return posterior probabilities of cluster memberships.
 **/

void const *
vl_gmm_get_posteriors (VlGMM const * self)
{
  return self->posteriors ;
}

/** @brief Get maximum number of iterations
 ** @param self object
 ** @return maximum number of iterations.
 **/

vl_size
vl_gmm_get_max_num_iterations (VlGMM const * self)
{
  return self->maxNumIterations ;
}

/** @brief Set maximum number of iterations
 ** @param self VlGMM filter.
 ** @param maxNumIterations maximum number of iterations.
 **/

void
vl_gmm_set_max_num_iterations (VlGMM * self, vl_size maxNumIterations)
{
  self->maxNumIterations = maxNumIterations ;
}

/** @brief Get maximum number of repetitions.
 ** @param self object
 ** @return current number of repretitions for quantization.
 **/

vl_size
vl_gmm_get_num_repetitions (VlGMM const * self)
{
  return self->numRepetitions ;
}

/** @brief Set maximum number of repetitions
 ** @param self object
 ** @param numRepetitions maximum number of repetitions.
 ** The number of repetitions cannot be smaller than 1.
 **/

void
vl_gmm_set_num_repetitions (VlGMM * self, vl_size numRepetitions)
{
  assert (numRepetitions >= 1) ;
  self->numRepetitions = numRepetitions ;
}

/** @brief Get data dimension
 ** @param self object
 ** @return data dimension.
 **/

vl_size
vl_gmm_get_dimension (VlGMM const * self)
{
  return self->dimension ;
}

/** @brief Get initialization algorithm
 ** @param self object
 ** @return initialization algorithm.
 **/

VlGMMInitialization
vl_gmm_get_initialization (VlGMM const * self)
{
  return self->initialization ;
}

/** @brief Set initialization algorithm.
 ** @param self object
 ** @param init initialization algorithm.
 **/
void
vl_gmm_set_initialization (VlGMM * self, VlGMMInitialization init)
{
  self->initialization = init;
}

/** @brief Get KMeans initialization object.
 ** @param self object
 ** @return kmeans initialization object.
 **/
VlKMeans * vl_gmm_get_kmeans_init_object (VlGMM const * self)
{
  return self->kmeansInit;
}

/** @brief Set KMeans initialization object.
 ** @param self object
 ** @param kmeans initialization KMeans object.
 **/
void vl_gmm_set_kmeans_init_object (VlGMM * self, VlKMeans * kmeans)
{
  if (self->kmeansInit && self->kmeansInitIsOwner) {
    vl_kmeans_delete(self->kmeansInit) ;
  }
  self->kmeansInit = kmeans;
  self->kmeansInitIsOwner = VL_FALSE;
}

/** @brief Get the lower bound on the diagonal covariance values.
 ** @param self object
 ** @return lower bound on covariances.
 **/
double const * vl_gmm_get_covariance_lower_bounds (VlGMM const * self)
{
  return self->sigmaLowBound;
}

/** @brief Set the lower bounds on diagonal covariance values.
 ** @param self object.
 ** @param bounds bounds.
 **
 ** There is one lower bound per dimension. Use ::vl_gmm_set_covariance_lower_bound
 ** to set all of them to a given scalar.
 **/
void vl_gmm_set_covariance_lower_bounds (VlGMM * self, double const * bounds)
{
  memcpy(self->sigmaLowBound, bounds, sizeof(double) * self->dimension) ;
}

/** @brief Set the lower bounds on diagonal covariance values.
 ** @param self object.
 ** @param bound bound.
 **
 ** While there is one lower bound per dimension, this function sets
 ** all of them to the specified scalar. Use ::vl_gmm_set_covariance_lower_bounds
 ** to set them individually.
 **/
void vl_gmm_set_covariance_lower_bound (VlGMM * self, double bound)
{
  int i ;
  for (i = 0 ; i < (signed)self->dimension ; ++i) {
    self->sigmaLowBound[i] = bound ;
  }
}

/* ---------------------------------------------------------------- */
/* Instantiate shuffle algorithm */

#define VL_SHUFFLE_type vl_uindex
#define VL_SHUFFLE_prefix _vl_gmm
#include "shuffle-def.h"

/* #ifdef VL_GMM_INSTANTITATING */
#endif

/* ---------------------------------------------------------------- */
#ifdef VL_GMM_INSTANTIATING
/* ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- */
/*                                            Posterior assignments */
/* ---------------------------------------------------------------- */

/** @fn vl_get_gmm_data_posterior_f(float*,vl_size,vl_size,float const*,float const*,vl_size,float const*,float const*)
 ** @brief Get Gaussian modes posterior probabilities
 ** @param posteriors posterior probabilities (output)/
 ** @param numClusters number of modes in the GMM model.
 ** @param numData number of data elements.
 ** @param priors prior mode probabilities of the GMM model.
 ** @param means means of the GMM model.
 ** @param dimension data dimension.
 ** @param covariances diagonal covariances of the GMM model.
 ** @param data data.
 ** @return data log-likelihood.
 **
 ** This is a helper function that does not require a ::VlGMM object
 ** instance to operate.
 **/

double
VL_XCAT(vl_get_gmm_data_posteriors_, SFX)
(TYPE * posteriors,
 vl_size numClusters,
 vl_size numData,
 TYPE const * priors,
 TYPE const * means,
 vl_size dimension,
 TYPE const * covariances,
 TYPE const * data)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

  logCovariances = vl_malloc(sizeof(TYPE) * numClusters) ;
  invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  logWeights = vl_malloc(sizeof(TYPE) * numClusters) ;

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
    if (priors[i_cl] < VL_GMM_MIN_PRIOR) {
      logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights[i_cl] = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    logCovariances[i_cl] = logSigma;
  } /* end of parallel region */

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE clusterPosteriorsSum = 0;
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE p =
      logWeights[i_cl]
      - halfDimLog2Pi
      - 0.5 * logCovariances[i_cl]
      - 0.5 * distFn (dimension,
                      data + i_d * dimension,
                      means + i_cl * dimension,
                      invCovariances + i_cl * dimension) ;
      posteriors[i_cl + i_d * numClusters] = p ;
      if (p > maxPosterior) { maxPosterior = p ; }
    }

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      TYPE p = posteriors[i_cl + i_d * numClusters] ;
      p =  exp(p - maxPosterior) ;
      posteriors[i_cl + i_d * numClusters] = p ;
      clusterPosteriorsSum += p ;
    }

    LL +=  log(clusterPosteriorsSum) + (double) maxPosterior ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum ;
    }
  } /* end of parallel region */

  vl_free(logCovariances);
  vl_free(logWeights);
  vl_free(invCovariances);

  return LL;
}

/* ---------------------------------------------------------------- */
/*                                 Restarts zero-weighted Gaussians */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_gmm_maximization_, SFX)
(VlGMM * self,
 TYPE * posteriors,
 TYPE * priors,
 TYPE * covariances,
 TYPE * means,
 TYPE const * data,
 vl_size numData) ;

static vl_size
VL_XCAT(_vl_gmm_restart_empty_modes_, SFX) (VlGMM * self, TYPE const * data)
{
  vl_size dimension = self->dimension;
  vl_size numClusters = self->numClusters;
  vl_index i_cl, j_cl, i_d, d;
  vl_size zeroWNum = 0;
  TYPE * priors = (TYPE*)self->priors ;
  TYPE * means = (TYPE*)self->means ;
  TYPE * covariances = (TYPE*)self->covariances ;
  TYPE * posteriors = (TYPE*)self->posteriors ;

  //VlRand * rand = vl_get_rand() ;

  TYPE * mass = vl_calloc(sizeof(TYPE), self->numClusters) ;

  if (numClusters <= 1) { return 0 ; }

  /* compute statistics */
  {
    vl_uindex i, k ;
    vl_size numNullAssignments = 0 ;
    for (i = 0 ; i < self->numData ; ++i) {
      for (k = 0 ; k < self->numClusters ; ++k) {
        TYPE p = ((TYPE*)self->posteriors)[k + i * self->numClusters] ;
        mass[k] += p ;
        if (p < VL_GMM_MIN_POSTERIOR) {
          numNullAssignments ++ ;
        }
      }
    }
    if (self->verbosity) {
      VL_PRINTF("gmm: sparsity of data posterior: %.1f%%\n", (double)numNullAssignments / (self->numData * self->numClusters) * 100) ;
    }
  }

#if 0
  /* search for cluster with negligible weight and reassign them to fat clusters */
  for (i_cl = 0 ; i_cl < numClusters ; ++i_cl) {
    if (priors[i_cl] < 0.00001/numClusters) {
      double mass = priors[0]  ;
      vl_index best = 0 ;

      for (j_cl = 1 ; j_cl < numClusters ; ++j_cl) {
        if (priors[j_cl] > mass) { mass = priors[j_cl] ; best = j_cl ; }
      }

      if (j_cl == i_cl) {
        /* this should never happen */
        continue ;
      }

      j_cl = best ;
      zeroWNum ++ ;

      VL_PRINTF("gmm: restarting mode %d by splitting mode %d (with prior %f)\n", i_cl,j_cl,mass) ;

      priors[i_cl] = mass/2 ;
      priors[j_cl] = mass/2 ;
      for (d = 0 ; d < dimension ; ++d) {
        TYPE sigma2 =  covariances[j_cl*dimension + d] ;
        TYPE sigma = VL_XCAT(vl_sqrt_,SFX)(sigma2) ;
        means[i_cl*dimension + d] = means[j_cl*dimension + d] + 0.001 * (vl_rand_real1(rand) - 0.5) * sigma ;
        covariances[i_cl*dimension + d] = sigma2 ;
      }
    }
  }
#endif

  /* search for cluster with negligible weight and reassign them to fat clusters */
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
    double size = - VL_INFINITY_D ;
    vl_index best = -1 ;

    if (mass[i_cl] >= VL_GMM_MIN_POSTERIOR *
        VL_MAX(1.0, (double) self->numData / self->numClusters))
    {
      continue ;
    }

    if (self->verbosity) {
      VL_PRINTF("gmm: mode %d is nearly empty (mass %f)\n", i_cl, mass[i_cl]) ;
    }

    /*
     Search for the Gaussian components that (approximately)
     maximally contribute to make the negative log-likelihood of the data
     large. Then split the worst offender.
     
     To do so, we approximate the exptected log-likelihood of the GMM:
     
     E[-log(f(x))] = H(f) = - log \int f(x) log f(x)
    
     where the density f(x) = sum_k pk gk(x) is a GMM. This is intractable
     but it is easy to approximate if we suppose that supp gk is disjoint with
     supp gq for all components k ~= q. In this canse
     
     H(f) ~= sum_k [ - pk log(pk) + pk H(gk) ]
     
     where H(gk) is the entropy of component k taken alone. The entropy of
     the latter is given by:
     
     H(gk) = D/2 (1 + log(2pi) + 1/2 sum_{i=0}^D log sigma_i^2

     */

    for (j_cl = 0 ; j_cl < (signed)numClusters ; ++j_cl) {
      double size_ ;
      if (priors[j_cl] < VL_GMM_MIN_PRIOR) { continue ; }
      size_ = + 0.5 * dimension * (1.0 + log(2*VL_PI)) ;
      for(d = 0 ; d < (signed)dimension ; d++) {
        double sigma2 = covariances[j_cl * dimension + d] ;
        size_ += 0.5 * log(sigma2) ;
      }
      size_ = priors[j_cl] * (size_ - log(priors[j_cl])) ;

      if (self->verbosity > 1) {
        VL_PRINTF("gmm: mode %d: prior %f, mass %f, entropy contribution %f\n",
                  j_cl, priors[j_cl], mass[j_cl], size_) ;
      }

      if (size_ > size) {
        size = size_ ;
        best = j_cl ;
      }
    }

    j_cl = best ;

    if (j_cl == i_cl || j_cl < 0) {
      if (self->verbosity) {
        VL_PRINTF("gmm: mode %d is empty, "
                  "but no other mode to split could be found\n", i_cl) ;
      }
      continue ;
    }

    if (self->verbosity) {
      VL_PRINTF("gmm: reinitializing empty mode %d with mode %d (prior %f, mass %f, score %f)\n",
                i_cl, j_cl, priors[j_cl], mass[j_cl], size) ;
    }

    /*
     Search for the dimension with maximum variance.
     */

    size = - VL_INFINITY_D ;
    best = - 1 ;

    for(d = 0; d < (signed)dimension; d++) {
      double sigma2 = covariances[j_cl * dimension + d] ;
      if (sigma2 > size) {
        size = sigma2 ;
        best = d ;
      }
    }

    /*
     Reassign points j_cl (mode to split) to i_cl (empty mode).
     */
    {
      TYPE mu = means[best + j_cl * self->dimension] ;
      for(i_d = 0 ; i_d < (signed)self->numData ; ++ i_d) {
        TYPE p = posteriors[j_cl + self->numClusters * i_d] ;
        TYPE q = posteriors[i_cl + self->numClusters * i_d] ; /* ~= 0 */
        if (data[best + i_d * self->dimension] < mu) {
          /* assign this point to i_cl */
          posteriors[i_cl + self->numClusters * i_d] = p + q ;
          posteriors[j_cl + self->numClusters * i_d] = 0 ;
        } else {
          /* assign this point to j_cl */
          posteriors[i_cl + self->numClusters * i_d] = 0 ;
          posteriors[j_cl + self->numClusters * i_d] = p + q ;
        }
      }
    }

    /*
     Re-estimate.
     */
    VL_XCAT(_vl_gmm_maximization_, SFX)
    (self,posteriors,priors,covariances,means,data,self->numData) ;
  }

  return zeroWNum;
}

/* ---------------------------------------------------------------- */
/*                                                          Helpers */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_gmm_apply_bounds_, SFX)(VlGMM * self)
{
  vl_uindex dim ;
  vl_uindex k ;
  vl_size numAdjusted = 0 ;
  TYPE * cov = (TYPE*)self->covariances ;
  double const * lbs = self->sigmaLowBound ;

  for (k = 0 ; k < self->numClusters ; ++k) {
    vl_bool adjusted = VL_FALSE ;
    for (dim = 0 ; dim < self->dimension ; ++dim) {
      if (cov[k * self->dimension + dim] < lbs[dim] ) {
        cov[k * self->dimension + dim] = lbs[dim] ;
        adjusted = VL_TRUE ;
      }
    }
    if (adjusted) { numAdjusted ++ ; }
  }

  if (numAdjusted > 0 && self->verbosity > 0) {
    VL_PRINT("gmm: detected %d of %d modes with at least one dimension "
             "with covariance too small (set to lower bound)\n",
             numAdjusted, self->numClusters) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                           EM - Maximization step */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_gmm_maximization_, SFX)
(VlGMM * self,
 TYPE * posteriors,
 TYPE * priors,
 TYPE * covariances,
 TYPE * means,
 TYPE const * data,
 vl_size numData)
{
  vl_size numClusters = self->numClusters;
  vl_index i_d, i_cl;
  vl_size dim ;
  TYPE * oldMeans ;
  double time = 0 ;

  if (self->verbosity > 1) {
    VL_PRINTF("gmm: em: entering maximization step\n") ;
    time = vl_get_cpu_time() ;
  }

  oldMeans = vl_malloc(sizeof(TYPE) * self->dimension * numClusters) ;
  memcpy(oldMeans, means, sizeof(TYPE) * self->dimension * numClusters) ;

  memset(priors, 0, sizeof(TYPE) * numClusters) ;
  memset(means, 0, sizeof(TYPE) * self->dimension * numClusters) ;
  memset(covariances, 0, sizeof(TYPE) * self->dimension * numClusters) ;

#if defined(_OPENMP)
#pragma omp parallel default(shared) private(i_d, i_cl, dim) \
                     num_threads(vl_get_max_threads())
#endif
  {
    TYPE * clusterPosteriorSum_, * means_, * covariances_ ;

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      clusterPosteriorSum_ = vl_calloc(sizeof(TYPE), numClusters) ;
      means_ = vl_calloc(sizeof(TYPE), self->dimension * numClusters) ;
      covariances_ = vl_calloc(sizeof(TYPE), self->dimension * numClusters) ;
    }

    /*
      Accumulate weighted sums and sum of square differences. Once normalized,
      these become the means and covariances of each Gaussian mode.

      The squared differences will be taken w.r.t. the old means however. In this manner,
      one avoids doing two passes across the data. Eventually, these are corrected to account
      for the new means properly. In principle, one could set the old means to zero, but
      this may cause numerical instabilities (by accumulating large squares).
    */

#if defined(_OPENMP)
#pragma omp for
#endif
    for (i_d = 0 ; i_d < (signed)numData ; ++i_d) {
      for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
        TYPE p = posteriors[i_cl + i_d * self->numClusters] ;
        vl_bool calculated = VL_FALSE ;

        /* skip very small associations for speed */
        if (p < VL_GMM_MIN_POSTERIOR / numClusters) { continue ; }

        clusterPosteriorSum_ [i_cl] += p ;

        #ifndef VL_DISABLE_AVX
        if (vl_get_simd_enabled() && vl_cpu_has_avx()) {
          VL_XCAT(_vl_weighted_mean_avx_, SFX)
          (self->dimension,
           means_+ i_cl * self->dimension,
           data + i_d * self->dimension,
           p) ;

          VL_XCAT(_vl_weighted_sigma_avx_, SFX)
          (self->dimension,
           covariances_ + i_cl * self->dimension,
           data + i_d * self->dimension,
           oldMeans + i_cl * self->dimension,
           p) ;

          calculated = VL_TRUE;
        }
        #endif
        #ifndef VL_DISABLE_SSE2
        if (vl_get_simd_enabled() && vl_cpu_has_sse2() && !calculated) {
          VL_XCAT(_vl_weighted_mean_sse2_, SFX)
          (self->dimension,
           means_+ i_cl * self->dimension,
           data + i_d * self->dimension,
           p) ;

           VL_XCAT(_vl_weighted_sigma_sse2_, SFX)
          (self->dimension,
           covariances_ + i_cl * self->dimension,
           data + i_d * self->dimension,
           oldMeans + i_cl * self->dimension,
           p) ;

          calculated = VL_TRUE;
        }
        #endif
        if(!calculated) {
          for (dim = 0 ; dim < self->dimension ; ++dim) {
            TYPE x = data[i_d * self->dimension + dim] ;
            TYPE mu = oldMeans[i_cl * self->dimension + dim] ;
            TYPE diff = x - mu ;
            means_ [i_cl * self->dimension + dim] += p * x ;
            covariances_ [i_cl * self->dimension + dim] += p * (diff*diff) ;
          }
        }
      }
    }

    /* accumulate */
#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
        priors [i_cl] += clusterPosteriorSum_ [i_cl];
        for (dim = 0 ; dim < self->dimension ; ++dim) {
          means [i_cl * self->dimension + dim] += means_ [i_cl * self->dimension + dim] ;
          covariances [i_cl * self->dimension + dim] += covariances_ [i_cl * self->dimension + dim] ;
        }
      }
      vl_free(means_);
      vl_free(covariances_);
      vl_free(clusterPosteriorSum_);
    }
  } /* parallel section */

  /* at this stage priors[] contains the total mass of each cluster */
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE mass = priors[i_cl] ;
    /* do not update modes that do not recieve mass */
    if (mass >= 1e-6 / numClusters) {
      for (dim = 0 ; dim < self->dimension ; ++dim) {
        means[i_cl * self->dimension + dim] /= mass ;
        covariances[i_cl * self->dimension + dim] /= mass ;
      }
    }
  }

  /* apply old to new means correction */
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE mass = priors[i_cl] ;
    if (mass >= 1e-6 / numClusters) {
      for (dim = 0 ; dim < self->dimension ; ++dim) {
        TYPE mu = means[i_cl * self->dimension + dim] ;
        TYPE oldMu = oldMeans[i_cl * self->dimension + dim] ;
        TYPE diff = mu - oldMu ;
        covariances[i_cl * self->dimension + dim] -= diff * diff ;
      }
    }
  }

  VL_XCAT(_vl_gmm_apply_bounds_,SFX)(self) ;

  {
    TYPE sum = 0;
    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      sum += priors[i_cl] ;
    }
    sum = VL_MAX(sum, 1e-12) ;
    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      priors[i_cl] /= sum ;
    }
  }

  if (self->verbosity > 1) {
    VL_PRINTF("gmm: em: maximization step completed in %.2f s\n",
              vl_get_cpu_time() - time) ;
  }

  vl_free(oldMeans);
}

/* ---------------------------------------------------------------- */
/*                                                    EM iterations */
/* ---------------------------------------------------------------- */


static double
VL_XCAT(_vl_gmm_em_, SFX)
(VlGMM * self,
 TYPE const * data,
 vl_size numData)
{
  vl_size iteration, restarted ;
  double previousLL = (TYPE)(-VL_INFINITY_D) ;
  double LL = (TYPE)(-VL_INFINITY_D) ;
  double time = 0 ;

  _vl_gmm_prepare_for_data (self, numData) ;

  VL_XCAT(_vl_gmm_apply_bounds_,SFX)(self) ;

  for (iteration = 0 ; 1 ; ++ iteration) {
    double eps ;

    /*
     Expectation: assign data to Gaussian modes
     and compute log-likelihood.
     */

    if (self->verbosity > 1) {
      VL_PRINTF("gmm: em: entering expectation step\n") ;
      time = vl_get_cpu_time() ;
    }

    LL = VL_XCAT(vl_get_gmm_data_posteriors_,SFX)
    (self->posteriors,
     self->numClusters,
     numData,
     self->priors,
     self->means,
     self->dimension,
     self->covariances,
     data) ;

    if (self->verbosity > 1) {
      VL_PRINTF("gmm: em: expectation step completed in %.2f s\n",
                vl_get_cpu_time() - time) ;
    }

    /*
     Check the termination conditions.
     */
    if (self->verbosity) {
      VL_PRINTF("gmm: em: iteration %d: loglikelihood = %f (variation = %f)\n",
                iteration, LL, LL - previousLL) ;
    }
    if (iteration >= self->maxNumIterations) {
      if (self->verbosity) {
        VL_PRINTF("gmm: em: terminating because "
                  "the maximum number of iterations "
                  "(%d) has been reached.\n", self->maxNumIterations) ;
      }
      break ;
    }

    eps = vl_abs_d ((LL - previousLL) / (LL));
    if ((iteration > 0) && (eps < 0.00001)) {
      if (self->verbosity) {
        VL_PRINTF("gmm: em: terminating because the algorithm "
                  "fully converged (log-likelihood variation = %f).\n", eps) ;
      }
      break ;
    }
    previousLL = LL ;

    /*
     Restart empty modes.
     */
    if (iteration > 1) {
      restarted = VL_XCAT(_vl_gmm_restart_empty_modes_, SFX)
        (self, data);
      if ((restarted > 0) & (self->verbosity > 0)) {
        VL_PRINTF("gmm: em: %d Gaussian modes restarted because "
                  "they had become empty.\n", restarted);
      }
    }

    /*
      Maximization: reestimate the GMM parameters.
    */
    VL_XCAT(_vl_gmm_maximization_, SFX)
      (self,self->posteriors,self->priors,self->covariances,self->means,data,numData) ;
  }
  return LL;
}


/* ---------------------------------------------------------------- */
/*                                Kmeans initialization of mixtures */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_gmm_init_with_kmeans_, SFX)
(VlGMM * self,
 TYPE const * data,
 vl_size numData,
 VlKMeans * kmeansInit)
{
  vl_size i_d ;
  vl_uint32 * assignments = vl_malloc(sizeof(vl_uint32) * numData);

  _vl_gmm_prepare_for_data (self, numData) ;

  memset(self->means,0,sizeof(TYPE) * self->numClusters * self->dimension) ;
  memset(self->priors,0,sizeof(TYPE) * self->numClusters) ;
  memset(self->covariances,0,sizeof(TYPE) * self->numClusters * self->dimension) ;
  memset(self->posteriors,0,sizeof(TYPE) * self->numClusters * numData) ;

  /* setup speified KMeans initialization object if any */
  if (kmeansInit) { vl_gmm_set_kmeans_init_object (self, kmeansInit) ; }

  /* if a KMeans initalization object is still unavailable, create one */
  if(self->kmeansInit == NULL) {
    vl_size ncomparisons = VL_MAX(numData / 4, 10) ;
    vl_size niter = 5 ;
    vl_size ntrees = 1 ;
    vl_size nrepetitions = 1 ;
    VlKMeansAlgorithm algorithm = VlKMeansANN ;
    VlKMeansInitialization initialization = VlKMeansRandomSelection ;

    VlKMeans * kmeansInitDefault = vl_kmeans_new(self->dataType,VlDistanceL2) ;
    vl_kmeans_set_initialization(kmeansInitDefault, initialization);
    vl_kmeans_set_max_num_iterations (kmeansInitDefault, niter) ;
    vl_kmeans_set_max_num_comparisons (kmeansInitDefault, ncomparisons) ;
    vl_kmeans_set_num_trees (kmeansInitDefault, ntrees);
    vl_kmeans_set_algorithm (kmeansInitDefault, algorithm);
    vl_kmeans_set_num_repetitions(kmeansInitDefault, nrepetitions);
    vl_kmeans_set_verbosity (kmeansInitDefault, self->verbosity);

    self->kmeansInit = kmeansInitDefault;
    self->kmeansInitIsOwner = VL_TRUE ;
  }

  /* Use k-means to assign data to clusters */
  vl_kmeans_cluster (self->kmeansInit, data, self->dimension, numData, self->numClusters);
  vl_kmeans_quantize (self->kmeansInit, assignments, NULL, data, numData) ;

  /* Transform the k-means assignments in posteriors and estimates the mode parameters */
  for(i_d = 0; i_d < numData; i_d++) {
    ((TYPE*)self->posteriors)[assignments[i_d] + i_d * self->numClusters] = (TYPE) 1.0 ;
  }

  /* Update cluster parameters */
  VL_XCAT(_vl_gmm_maximization_, SFX)
    (self,self->posteriors,self->priors,self->covariances,self->means,data,numData);
  vl_free(assignments) ;
}

/* ---------------------------------------------------------------- */
/*                                Random initialization of mixtures */
/* ---------------------------------------------------------------- */

static void
VL_XCAT(_vl_gmm_compute_init_sigma_, SFX)
(VlGMM * self,
 TYPE const * data,
 TYPE * initSigma,
 vl_size dimension,
 vl_size numData)
{
  vl_size dim;
  vl_uindex i;

  TYPE * dataMean ;

  memset(initSigma,0,sizeof(TYPE)*dimension) ;
  if (numData <= 1) return ;

  dataMean = vl_malloc(sizeof(TYPE)*dimension);
  memset(dataMean,0,sizeof(TYPE)*dimension) ;

  /* find mean of the whole dataset */
  for(dim = 0 ; dim < dimension ; dim++) {
    for(i = 0 ; i < numData ; i++) {
      dataMean[dim] += data[i*dimension + dim];
    }
    dataMean[dim] /= numData;
  }

  /* compute variance of the whole dataset */
  for(dim = 0; dim < dimension; dim++) {
    for(i = 0; i < numData; i++) {
      TYPE diff = (data[i*self->dimension + dim] - dataMean[dim]) ;
      initSigma[dim] += diff*diff ;
    }
    initSigma[dim] /= numData - 1 ;
  }

  vl_free(dataMean) ;
}

static void
VL_XCAT(_vl_gmm_init_with_rand_data_, SFX)
(VlGMM * self,
 TYPE const * data,
 vl_size numData)
{
  vl_uindex i, k, dim ;
  VlKMeans * kmeans ;

  _vl_gmm_prepare_for_data(self, numData) ;

  /* initilaize priors of gaussians so they are equal and sum to one */
  for (i = 0 ; i < self->numClusters ; ++i) { ((TYPE*)self->priors)[i] = (TYPE) (1.0 / self->numClusters) ; }

  /* initialize diagonals of covariance matrices to data covariance */
  VL_XCAT(_vl_gmm_compute_init_sigma_, SFX) (self, data, self->covariances, self->dimension, numData);
  for (k = 1 ; k < self->numClusters ; ++ k) {
    for(dim = 0; dim < self->dimension; dim++) {
      *((TYPE*)self->covariances + k * self->dimension + dim) =
      *((TYPE*)self->covariances + dim) ;
    }
  }

  /* use kmeans++ initialization to pick points at random */
  kmeans = vl_kmeans_new(self->dataType,VlDistanceL2) ;
  vl_kmeans_init_centers_plus_plus(kmeans, data, self->dimension, numData, self->numClusters) ;
  memcpy(self->means, vl_kmeans_get_centers(kmeans), sizeof(TYPE) * self->dimension * self->numClusters) ;
  vl_kmeans_delete(kmeans) ;
}

/* ---------------------------------------------------------------- */
#else /* VL_GMM_INSTANTIATING */
/* ---------------------------------------------------------------- */

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define VL_GMM_INSTANTIATING
#include "gmm.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define VL_GMM_INSTANTIATING
#include "gmm.c"
#endif

/* VL_GMM_INSTANTIATING */
#endif

/* ---------------------------------------------------------------- */
#ifndef VL_GMM_INSTANTIATING
/* ---------------------------------------------------------------- */

/** @brief Create a new GMM object by copy
 ** @param self object.
 ** @return new copy.
 **
 ** Most parameters, including the cluster priors, means, and
 ** covariances are copied. Data posteriors (available after
 ** initalization or EM) are not; nor is the KMeans object used for
 ** initialization, if any.
 **/

VlGMM *
vl_gmm_new_copy (VlGMM const * self)
{
  vl_size size = vl_get_type_size(self->dataType) ;
  VlGMM * gmm = vl_gmm_new(self->dataType, self->dimension, self->numClusters);
  gmm->initialization = self->initialization;
  gmm->maxNumIterations = self->maxNumIterations;
  gmm->numRepetitions = self->numRepetitions;
  gmm->verbosity = self->verbosity;
  gmm->LL = self->LL;

  memcpy(gmm->means, self->means, size*self->numClusters*self->dimension);
  memcpy(gmm->covariances, self->covariances, size*self->numClusters*self->dimension);
  memcpy(gmm->priors, self->priors, size*self->numClusters);
  return gmm ;
}

/** @brief Initialize mixture before EM takes place using random initialization
 ** @param self GMM object instance.
 ** @param data data points which should be clustered.
 ** @param numData number of data points.
 **/

void
vl_gmm_init_with_rand_data
(VlGMM * self,
 void const * data,
 vl_size numData)
{
  vl_gmm_reset (self) ;
  switch (self->dataType) {
    case VL_TYPE_FLOAT : _vl_gmm_init_with_rand_data_f (self, (float const *)data, numData) ; break ;
    case VL_TYPE_DOUBLE : _vl_gmm_init_with_rand_data_d (self, (double const *)data, numData) ; break ;
    default:
      abort() ;
  }
}

/** @brief Initializes the GMM using KMeans
 ** @param self GMM object instance.
 ** @param data data points which should be clustered.
 ** @param numData number of data points.
 ** @param kmeansInit KMeans object to use.
 **/

void
vl_gmm_init_with_kmeans
(VlGMM * self,
 void const * data,
 vl_size numData,
 VlKMeans * kmeansInit)
{
  vl_gmm_reset (self) ;
  switch (self->dataType) {
    case VL_TYPE_FLOAT :
      _vl_gmm_init_with_kmeans_f
      (self, (float const *)data, numData, kmeansInit) ;
      break ;
    case VL_TYPE_DOUBLE :
      _vl_gmm_init_with_kmeans_d
      (self, (double const *)data, numData, kmeansInit) ;
      break ;
    default:
      abort() ;
  }
}

#if 0
#include<fenv.h>
#endif

/** @brief Run GMM clustering - includes initialization and EM
 ** @param self GMM object instance.
 ** @param data data points which should be clustered.
 ** @param numData number of data points.
 **/

double vl_gmm_cluster (VlGMM * self,
                       void const * data,
                       vl_size numData)
{
  void * bestPriors = NULL ;
  void * bestMeans = NULL;
  void * bestCovariances = NULL;
  void * bestPosteriors = NULL;
  vl_size size = vl_get_type_size(self->dataType) ;
  double bestLL = -VL_INFINITY_D;
  vl_uindex repetition;

  assert(self->numRepetitions >=1) ;

  bestPriors = vl_malloc(size * self->numClusters) ;
  bestMeans = vl_malloc(size * self->dimension * self->numClusters) ;
  bestCovariances = vl_malloc(size * self->dimension * self->numClusters) ;
  bestPosteriors = vl_malloc(size * self->numClusters * numData) ;

#if 0
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  for (repetition = 0 ; repetition < self->numRepetitions ; ++ repetition) {
    double LL ;
    double timeRef ;

    if (self->verbosity) {
      VL_PRINTF("gmm: clustering: starting repetition %d of %d\n", repetition + 1, self->numRepetitions) ;
    }

    /* initialize a new mixture model */
    timeRef = vl_get_cpu_time() ;
    switch (self->initialization) {
      case VlGMMKMeans : vl_gmm_init_with_kmeans (self, data, numData, NULL) ; break ;
      case VlGMMRand : vl_gmm_init_with_rand_data (self, data, numData) ; break ;
      case VlGMMCustom : break ;
      default: abort() ;
    }
    if (self->verbosity) {
      VL_PRINTF("gmm: model initialized in %.2f s\n",
                vl_get_cpu_time() - timeRef) ;
    }

    /* fit the model to data by running EM */
    timeRef = vl_get_cpu_time () ;
    LL = vl_gmm_em (self, data, numData) ;
    if (self->verbosity) {
      VL_PRINTF("gmm: optimization terminated in %.2f s with loglikelihood %f\n",
                vl_get_cpu_time() - timeRef, LL) ;
    }

    if (LL > bestLL || repetition == 0) {
      void * temp ;

      temp = bestPriors ;
      bestPriors = self->priors ;
      self->priors = temp ;

      temp = bestMeans ;
      bestMeans = self->means ;
      self->means = temp ;

      temp = bestCovariances ;
      bestCovariances = self->covariances ;
      self->covariances = temp ;

      temp = bestPosteriors ;
      bestPosteriors = self->posteriors ;
      self->posteriors = temp ;

      bestLL = LL;
    }
  }

  vl_free (self->priors) ;
  vl_free (self->means) ;
  vl_free (self->covariances) ;
  vl_free (self->posteriors) ;

  self->priors = bestPriors ;
  self->means = bestMeans ;
  self->covariances = bestCovariances ;
  self->posteriors = bestPosteriors ;
  self->LL = bestLL;

  if (self->verbosity) {
    VL_PRINTF("gmm: all repetitions terminated with final loglikelihood %f\n", self->LL) ;
  }

  return bestLL ;
}

/** @brief Invoke the EM algorithm.
 ** @param self GMM object instance.
 ** @param data data points which should be clustered.
 ** @param numData number of data points.
 **/

double vl_gmm_em (VlGMM * self, void const * data, vl_size numData)
{
  switch (self->dataType) {
    case VL_TYPE_FLOAT:
      return _vl_gmm_em_f (self, (float const *)data, numData) ; break ;
    case VL_TYPE_DOUBLE:
      return _vl_gmm_em_d (self, (double const *)data, numData) ; break ;
    default:
      abort() ;
  }
  return 0 ;
}

/** @brief Explicitly set the initial means for EM.
 ** @param self GMM object instance.
 ** @param means initial values of means.
 **/

void
vl_gmm_set_means (VlGMM * self, void const * means)
{
  memcpy(self->means,means,
         self->dimension * self->numClusters * vl_get_type_size(self->dataType));
}

/** @brief Explicitly set the initial sigma diagonals for EM.
 ** @param self GMM object instance.
 ** @param covariances initial values of covariance matrix diagonals.
 **/

void vl_gmm_set_covariances (VlGMM * self, void const * covariances)
{
  memcpy(self->covariances,covariances,
         self->dimension * self->numClusters * vl_get_type_size(self->dataType));
}

/** @brief Explicitly set the initial priors of the gaussians.
 ** @param self GMM object instance.
 ** @param priors initial values of the gaussian priors.
 **/

void vl_gmm_set_priors (VlGMM * self, void const * priors)
{
  memcpy(self->priors,priors,
         self->numClusters * vl_get_type_size(self->dataType));
}

/* VL_GMM_INSTANTIATING */
#endif

#undef SFX
#undef TYPE
#undef FLT
#undef VL_GMM_INSTANTIATING
