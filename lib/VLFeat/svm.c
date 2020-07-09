/** @file svm.c
 ** @brief Support Vector Machines (SVM) - Implementation
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

/** @file svm.h
 ** @see @ref svm.
 **/

/**
<!-- ------------------------------------------------------------- -->
@page svm Support Vector Machines (SVM)
@author Milan Sulc
@author Daniele Perrone
@author Andrea Vedaldi
@tableofcontents
<!-- ------------------------------------------------------------- -->

*Support Vector Machines* (SVMs) are one of the most popular types of
discriminate classifiers. VLFeat implements two solvers, SGD and SDCA,
capable of learning linear SVMs on a large scale. These linear solvers
can be combined with explicit feature maps to learn non-linear models
as well. The solver supports a few variants of the standard
SVM formulation, including using loss functions other than the hinge
loss.

@ref svm-starting demonstrates how to use VLFeat to learn an SVM.
Information on SVMs and the corresponding optimization algorithms as
implemented by VLFeat are given in:

- @subpage svm-fundamentals - Linear SVMs and their learning.
- @subpage svm-advanced - Loss functions, dual objective, and other details.
- @subpage svm-sgd - The SGD algorithm.
- @subpage svm-sdca - The SDCA algorithm.

<!-- ------------------------------------------------------------- -->
@section svm-starting Getting started
<!-- ------------------------------------------------------------- -->

This section demonstrates how to learn an SVM by using VLFeat. SVM
learning is implemented by the ::VlSvm object type. Let's
start by a complete example:

@code
#include <stdio.h>
#include <vl/svm.h>

int main()
{
  vl_size const numData = 4 ;
  vl_size const dimension = 2 ;
  double x [dimension * numData] = {
    0.0, -0.5,
    0.6, -0.3,
    0.0,  0.5
    0.6,  0.0} ;
  double y [numData] = {1, 1, -1, 1} ;
  double lambda = 0.01;
  double * const model ;
  double bias ;

  VlSvm * svm = vl_svm_new(VlSvmSolverSgd,
                           x, dimension, numData,
                           y,
                           lambda) ;
  vl_svm_train(svm) ;

  model = vl_svm_get_model(svm) ;
  bias = vl_svm_get_bias(svm) ;

  printf("model w = [ %f , %f ] , bias b = %f \n",
         model[0],
         model[1],
         bias);

  vl_svm_delete(svm) ;
  return 0;
}
@endcode

This code learns a binary linear SVM using the SGD algorithm on
four two-dimensional points using 0.01 as regularization parameter.

::VlSvmSolverSdca can be specified in place of ::VlSvmSolverSdca
in orer to use the SDCA algorithm instead.

Convergence and other diagnostic information can be obtained after
training by using the ::vl_svm_get_statistics function. Algorithms
regularly check for convergence (usally after each pass over the data).
The ::vl_svm_set_diagnostic_function can be used to specify a callback
to be invoked when diagnostic is run. This can be used, for example,
to dump information on the screen as the algorithm progresses.

Convergence is reached after a maximum number of iterations
(::vl_svm_set_max_num_iterations) or after a given criterion falls
below a threshold (::vl_svm_set_epsilon). The meaning of these
may depend on the specific algorithm (see @ref svm for further details).

::VlSvm is a quite powerful object. Algorithms only need to perform
inner product and accumulation operation on the data (see @ref svm-advanced).
This is used to abstract from the data type and support almost anything
by speciying just two functions (::vl_svm_set_data_functions).

A simple interface to this advanced functionality is provided by the
::VlSvmDataset object. This supports natively @c float and @c double
data types, as well as applying on the fly the homogeneous kernel map
(@ref homkermap). This is exemplified in @ref svmdataset-starting.

*/

/**
<!-- ------------------------------------------------------------- -->
@page svm-fundamentals SVM fundamentals
@tableofcontents
<!-- ------------------------------------------------------------- -->

This page introduces the SVM formulation used in VLFeat. See @ref svm
for more information on VLFeat SVM support.

Let $ \bx \in \real^d $ be a vector representing, for example, an
image, an audio track, or a fragment of text. Our goal is to design a
*classifier*, i.e. a function that associates to each vector $\bx$ a
positive or negative label based on a desired criterion, for example
the fact that the image contains or not a cat, that the audio track
contains or not English speech, or that the text is or not a
scientific paper.

The vector $\bx$ is classified by looking at the sign of a *linear
scoring function* $\langle \bx, \bw \rangle$. The goal of learning is
to estimate the parameter $\bw \in \real^d$ in such a way that the
score is positive if the vector $\bx$ belongs to the positive class
and negative otherwise. In fact, in the standard SVM formulation the
the goal is to have a score of *at least 1* in the first case, and of
*at most -1* in the second one, imposing a *margin*.

The parameter $\bw$ is estimated or *learned* by fitting the scoring
function to a training set of $n$ example pairs $(\bx_i,y_i),
i=1,\dots,n$. Here $y_i \in \{-1,1\}$ are the *ground truth labels* of
the corresponding example vectors. The fit quality is measured by a
*loss function* which, in standard SVMs, is the *hinge loss*:

\[
\ell_i(\langle \bw,\bx\rangle) = \max\{0, 1 - y_i \langle \bw,\bx\rangle\}.
\]

Note that the hinge loss is zero only if the score $\langle
\bw,\bx\rangle$ is at least 1 or at most -1, depending on the label
$y_i$.

Fitting the training data is usually insufficient. In order for the
scoring function *generalize to future data* as well, it is usually
preferable to trade off the fitting accuracy with the *regularity* of
the learned scoring function $\langle \bx, \bw \rangle$. Regularity in
the standard formulation is measured by the norm of the parameter
vector $\|\bw\|^2$ (see @ref svm-advanced). Averaging the loss on all
training samples and adding to it the regularizer weighed by a
parameter $\lambda$ yields the *regularized loss objective*

@f{equation}{
\boxed{\displaystyle
E(\bw) =  \frac{\lambda}{2} \left\| \bw \right\|^2
+ \frac{1}{n} \sum_{i=1}^n \max\{0, 1 - y_i \langle \bw,\bx\rangle\}.
\label{e:svm-primal-hinge}
}
@f}

Note that this objective function is *convex*, so that there exists a
single global optimum.

The scoring function $\langle \bx, \bw \rangle$ considered so far has
been linear and unbiased. @ref svm-bias discusses how a bias term can
be added to the SVM and @ref svm-feature-maps shows how non-linear
SVMs can be reduced to the linear case by computing suitable feature
maps.

@ref svm-learning shows how VLFeat can be used to learn an SVM by
minimizing $E(\bw)$.

<!-- ------------------------------------------------------------- -->
@section svm-learning Learning
<!-- ------------------------------------------------------------- -->

Learning an SVM amounts to finding the minimizer $\bw^*$ of the cost
function $E(\bw)$. While there are dozens of methods that can be used
to do so, VLFeat implements two large scale methods, designed to work
with linear SVMs (see @ref svm-feature-maps to go beyond linear):

- @ref svm-sgd
- @ref svm-sdca

Using these solvers is exemplified in @ref svm-starting.

<!-- ------------------------------------------------------------- -->
@section svm-bias Adding a bias
<!-- ------------------------------------------------------------- -->

It is common to add to the SVM scoring function a *bias term* $b$, and
to consider the score $\langle \bx,\bw \rangle + b$. In practice the
bias term can be crucial to fit the training data optimally, as there
is no reason why the inner products $\langle \bx,\bw \rangle$ should
be naturally centered at zero.

Some SVM learning algorithms can estimate both $\bw$ and $b$
directly. However, other algorithms such as SGD and SDCA cannot. In
this case, a simple workaround is to add a constant component $B > 0$
(we call this constant the *bias multiplier*) to the data,
i.e. consider the extended data vectors:
\[
\bar \bx = \begin{bmatrix} \bx \\ B \end{bmatrix},
\quad
\bar \bw = \begin{bmatrix} \bw \\ w_b \end{bmatrix}.
\]
In this manner the scoring function incorporates implicitly a bias $b = B w_b$:
\[
\langle \bar\bx, \bar\bw \rangle =
\langle \bx, \bw \rangle + B w_b.
\]

The disadvantage of this reduction is that the term $w_b^2$ becomes
part of the SVM regularizer, which shrinks the bias $b$ towards
zero. This effect can be alleviated by making $B$ sufficiently large,
because in this case $\|\bw\|^2 \gg w_b^2$ and the shrinking effect is
negligible.

Unfortunately, making $B$ too large makes the problem numerically
unbalanced, so a reasonable trade-off between shrinkage and stability
is generally sought. Typically, a good trade-off is obtained by
normalizing the data to have unitary Euclidean norm and then choosing
$B \in [1, 10]$.

Specific implementations of SGD and SDCA may provide explicit support
to learn the bias in this manner, but it is important to understand
the implications on speed and accuracy of the learning if this is
done.

<!-- ------------------------------------------------------------- -->
@section svm-feature-maps Non-linear SVMs and feature maps
<!-- ------------------------------------------------------------- -->

So far only linear scoring function $\langle \bx,\bw \rangle$ have
been considered. Implicitly, however, this assumes that the objects to
be classified (e.g. images) have been encoded as vectors $\bx$ in a
way that makes linear classification possible. This encoding step can
be made explicit by introducing the *feature map* $\Phi(\bx) \in
\real^d$. Including the feature map yields a scoring function
*non-linear* in $\bx$:
\[
\bx\in\mathcal{X} \quad\longrightarrow\quad \langle \Phi(\bx), \bw \rangle.
\]
The nature of the input space $\mathcal{X}$ can be arbitrary and might
not have a vector space structure at all.

The representation or encoding captures a notion of *similarity*
between objects: if two vectors $\Phi(\bx_1)$ and $\Phi(\bx_2)$ are
similar, then their scores will also be similar. Note that choosing a
feature map amounts to incorporating this information in the model
*prior* to learning.

The relation of feature maps to similarity functions is formalized by
the notion of a *kernel*, a positive definite function $K(\bx,\bx')$
measuring the similarity of a pair of objects. A feature map defines a
kernel by

\[
K(\bx,\bx') = \langle \Phi(\bx),\Phi(\bx') \rangle.
\]

Viceversa, any kernel function can be represented by a feature map in
this manner, establishing an equivalence.

So far, all solvers in VLFeat assume that the feature map $\Psi(\bx)$
can be explicitly computed. Although classically kernels were
introduced to generalize solvers to non-linear SVMs for which a
feature map *cannot* be computed (e.g. for a Gaussian kernel the
feature map is infinite dimensional), in practice using explicit
feature representations allow to use much faster solvers, so it makes
sense to *reverse* this process.
*/

/**
<!-- ------------------------------------------------------------- -->
@page svm-advanced Advanced SVM topics
@tableofcontents
<!-- ------------------------------------------------------------- -->

This page discusses advanced SVM topics. For an introduction to SVMs,
please refer to @ref svm and @ref svm-fundamentals.

<!-- ------------------------------------------------------------- -->
@section svm-loss-functions Loss functions
<!-- ------------------------------------------------------------- -->

The SVM formulation given in @ref svm-fundamentals uses the
hinge loss, which is only one of a variety of loss functions that
are often used for SVMs. More in general, one
can consider the objective

@f{equation}{
E(\bw) =  \frac{\lambda}{2} \left\| \bw \right\|^2 + \frac{1}{n} \sum_{i=1}^n \ell_i(\langle \bw,\bx\rangle).
\label{e:svm-primal}
@f}

where the loss $\ell_i(z)$ is a convex function of the scalar variable
$z$. Losses differ by: (i) their purpose (some are suitable for
classification, other for regression), (ii) their smoothness (which
usually affects how quickly the SVM objective function can be
minimized), and (iii) their statistical interpretation (for example
the logistic loss can be used to learn logistic models).

Concrete examples are the:

<table>
<tr>
<td>Name</td>
<td>Loss $\ell_i(z)$</td>
<td>Description</td>
</tr>
<tr>
<td>Hinge</td>
<td>$\max\{0, 1-y_i z\}$</td>
<td>The standard SVM loss function.</td>
</tr>
<tr>
<td>Square hinge</td>
<td>$\max\{0, 1-y_i z\}^2$</td>
<td>The standard SVM loss function, but squared. This version is
smoother and may yield numerically easier problems.</td>
</tr>
<tr>
<td>Square or l2</td>
<td>$(y_i - z)^2$</td>
<td>This loss yields the ridge regression model (l2 regularised least
square).</td>
</tr>
<tr>
<td>Linear or l1</td>
<td>$|y_i - z|$</td>
<td>Another loss suitable for regression, usually more robust but
harder to optimize than the squared one.</td>
</tr>
<tr>
<td>Insensitive l1</td>
<td>$\max\{0, |y_i - z| - \epsilon\}$.</td>
<td>This is a variant of the previous loss, proposed in the original
Support Vector Regression formulation. Differently from the previous
two losses, the insensitivity may yield to a sparse selection of
support vectors.</td>
</tr>
<tr>
<td>Logistic</td>
<td>$\log(1 + e^{-y_i z})$</td>
<td>This corresponds to regularized logisitc regression. The loss can
be seen as a negative log-likelihood: $\ell_i(z) = -\log P[y_i | z] =
- \log \sigma(y_iz/2)$, where $\sigma(z) = e^z/(1 + e^z)$ is the
sigmoid function, mapping a score $z$ to a probability. The $1/2$
factor in the sigmoid is due to the fact that labels are in $\{-1,1\}$
rather than $\{0,1\}$ as more common for the standard sigmoid
model.</td>
</tr>
</table>

<!-- ------------------------------------------------------------- -->
@section svm-data-abstraction Data abstraction: working with compressed data
<!-- ------------------------------------------------------------- -->

VLFeat learning algorithms (SGD and SDCA) access the data by means of
only two operations:

- *inner product*: computing the inner product between the model and
a data vector, i.e. $\langle \bw, \bx \rangle$.
- *accumulation*: summing a data vector to the model, i.e. $\bw
\leftarrow \bw + \beta \bx$.

VLFeat learning algorithms are *parameterized* in these two
operations. As a consequence, the data can be stored in any format
suitable to the user (e.g. dense matrices, sparse matrices,
block-sparse matrices, disk caches, and so on) provided that these two
operations can be implemented efficiently. Differently from the data,
however, the model vector $\bw$ is represented simply as a dense array
of doubles. This choice is adequate in almost any case.

A particularly useful aspect of this design choice is that the
training data can be store in *compressed format* (for example by
using product quantization (PQ)). Furthermore, higher-dimensional
encodings such as the homogeneous kernel map (@ref homkermap) and the
intersection kernel map can be *computed on the fly*. Such techniques
are very important when dealing with GBs of data.

<!-- ------------------------------------------------------------- -->
@section svm-dual-problem Dual problem
<!-- ------------------------------------------------------------- -->

In optimization, the *dual objective* $D(\balpha)$ of the SVM
objective $E(\bw)$ is of great interest. To obtain the dual objective,
one starts by approximating each loss term from below by a family of planes:
\[
\ell_i(z) = \sup_{u} (u z - \ell_i^*(u) ),
\qquad
\ell_i^*(u) = \sup_{z} (z u - \ell_i(z) )
\]
where $\ell_i^*(u)$ is the *dual conjugate* of the loss and gives the
intercept of each approximating plane as a function of the slope. When
the loss function is convex, the approximation is in fact exact. Examples
include:

<table>
<tr>
<td>Name</td>
<td>Loss $\ell_i(z)$</td>
<td>Conjugate loss $\ell_i^*(u)$</td>
</tr>
<tr>
<td>Hinge</td>
<td>$\max\{0, 1-y_i z\}$</td>
<td>\[
\ell_i^*(u) =
\begin{cases}
y_i u, & -1 \leq y_i u \leq 0, \\
+\infty, & \text{otherwise}
\end{cases}
\]</td>
</tr>
<tr>
<td>Square hinge</td>
<td>$\max\{0, 1-y_i z\}^2$</td>
<td>\[\ell_i^*(u) =
\begin{cases}
y_i u + \frac{u^2}{4}, & y_i u \leq 0, \\
+\infty, & \text{otherwise} \\
\end{cases}\]</td>
</tr>
<tr>
<td>Linear or l1</td>
<td>$|y_i - z|$</td>
<td>\[\ell_i^*(u) =
\begin{cases}
y_i u, & -1 \leq y_i u \leq 1, \\
+\infty, & \text{otherwise} \\
\end{cases}\]</td>
</tr>
<tr>
<td>Square or l2</td>
<td>$(y_i - z)^2$</td>
<td>\[\ell_i^*(u)=y_iu + \frac{u^2}{4}\]</td>
</tr>
<tr>
<td>Insensitive l1</td>
<td>$\max\{0, |y_i - z| - \epsilon\}$.</td>
<td></td>
</tr>
<tr>
<td>Logistic</td>
<td>$\log(1 + e^{-y_i z})$</td>
<td>\[\ell_i^*(u) =
 \begin{cases}
 (1+u) \log(1+u) - u \log(-u), & -1 \leq y_i u \leq 0, \\
 +\infty, & \text{otherwise} \\
 \end{cases}\]
</td>
</tr>
</table>

Since each plane $- z \alpha_i - \ell^*_i(-\alpha_i) \leq \ell_i(z)$
bounds the loss from below, by substituting in $E(\bw)$ one can write
a lower bound for the SVM objective
\[
F(\bw,\balpha) = \frac{\lambda}{2} \|\bw\|^2 -
\frac{1}{n}\sum_{i=1}^n (\bw^\top \bx_i\alpha_i + \ell_i^*(-\alpha_i))
\leq E(\bw).
\]
for each setting of the *dual variables* $\alpha_i$. The dual
objective function $D(\balpha)$ is obtained by minimizing the lower
bound $F(\bw,\balpha)$ w.r.t. to $\bw$:
\[
D(\balpha) = \inf_{\bw} F(\bw,\balpha) \leq E(\bw).
\]
The minimizer and the dual objective are now easy to find:
\[
\boxed{\displaystyle
\bw(\balpha) =
\frac{1}{\lambda n}
\sum_{i=1}^n \bx_i \alpha_i = \frac{1}{\lambda n} X\balpha,
\quad
D(\balpha) = - \frac{1}{2\lambda n^2} \balpha^\top X^\top X \balpha +
\frac{1}{n} \sum_{i=1}^n - \ell_i^*(-\alpha_i)
}
\]
where $X = [\bx_1, \dots, \bx_n]$ is the data matrix. Since the dual
is uniformly smaller than the primal, one has the *duality gap* bound:
\[
D(\balpha) \leq P(\bw^*) \leq P(\bw(\balpha))
\]
This bound can be used to evaluate how far off $\bw(\balpha)$ is from
the primal minimizer $\bw^*$. In fact, due to convexity, this bound
can be shown to be zero when $\balpha^*$ is the dual maximizer (strong
duality):
\[
D(\balpha^*) = P(\bw^*) = P(\bw(\balpha^*)),
\quad \bw^* = \bw(\balpha^*).
\]

<!-- ------------------------------------------------------------- -->
@section svm-C Parametrization in C
<!-- ------------------------------------------------------------- -->

Often a slightly different form of the SVM objective is considered,
where a parameter $C$ is used to scale the loss instead of the regularizer:

\[
E_C(\bw) = \frac{1}{2} \|\bw\|^2 + C \sum_{i=1}^n \ell_i(\langle \bx_i, \bw\rangle)
\]

This and the objective function $E(\bw)$ in $\lambda$ are equivalent
(proportional) if

\[
\lambda = \frac{1}{nC},
\qquad C = \frac{1}{n\lambda}.
\] up to an overall scaling factor to the problem.

**/

/**

<!-- ------------------------------------------------------------- -->
@page svm-sdca Stochastic Dual Coordinate Ascent
@tableofcontents
<!-- ------------------------------------------------------------- -->

This page describes the *Stochastic Dual Coordinate Ascent* (SDCA)
linear SVM solver. Please see @ref svm for an overview of VLFeat SVM
support.

SDCA maximizes the dual SVM objective (see @ref svm-dual-problem
for a derivation of this expression):

\[
D(\balpha) = - \frac{1}{2\lambda n^2} \balpha^\top X^\top X \balpha +
\frac{1}{n} \sum_{i=1}^n - \ell_i^*(-\alpha_i)
\]

where $X$ is the data matrix. Recall that the primal parameter
corresponding to a given setting of the dual variables is:

\[
\bw(\balpha) = \frac{1}{\lambda n} \sum_{i=1}^n \bx_i \alpha_i = \frac{1}{\lambda n} X\balpha
\]

In its most basic form, the *SDCA algorithm* can be summarized as follows:

- Let $\balpha_0 = 0$.
- Until the duality gap $P(\bw(\balpha_t)) -  D(\balpha_t) < \epsilon$
  - Pick a dual variable $q$ uniformly at random in $1, \dots, n$.
  - Maximize the dual with respect to this variable: $\Delta\alpha_q = \max_{\Delta\alpha_q} D(\balpha_t + \Delta\alpha_q \be_q )$
  - Update $\balpha_{t+1} = \balpha_{t} + \be_q \Delta\alpha_q$.

In VLFeat, we partially use the nomenclature from @cite{shwartz13a-dual} and @cite{hsieh08a-dual}.

<!-- ------------------------------------------------------------- -->
@section svm-sdca-dual-max Dual coordinate maximization
<!-- ------------------------------------------------------------- -->

The updated dual objective can be expanded as:
\[
D(\balpha_t + \be_q \Delta\alpha_q) =
\text{const.}
- \frac{1}{2\lambda n^2} \bx_q^\top \bx_q (\Delta\alpha_q)^2
- \frac{1}{n} \bx_q^\top \frac{X\alpha_t}{\lambda n} \Delta\alpha_q
- \frac{1}{n} \ell^*_q(- \alpha_q - \Delta\alpha_q)
\]
This can also be written as
@f{align*}
D(\balpha_t + \be_q \Delta\alpha_q) &\propto
- \frac{A}{2} (\Delta\alpha_q)^2
- B \Delta\alpha_q
- \ell^*_q(- \alpha_q - \Delta\alpha_q),
\\
A &= \frac{1}{\lambda n} \bx_q^\top \bx_q = \frac{1}{\lambda n} \| \bx_q \|^2,
\\
B &= \bx_q^\top \frac{X\balpha_t}{\lambda n} = \bx_q^\top \bw_t.
@f}
Maximizing this quantity in the scalar variable $\Delta\balpha$ is usually
not difficult. It is convenient to store and incrementally
update the model $\bw_t$ after the optimal step $\Delta\balpha$ has been
determined:
\[
\bw_t = \frac{X \balpha_t}{\lambda n},
\quad \bw_{t+1} = \bw_t + \frac{1}{\lambda n }\bx_q \be_q \Delta\alpha_q.
\]

For example, consider the hinge loss as given in @ref svm-advanced :
\[
\ell_q^*(u) =
\begin{cases}
y_q u, & -1 \leq y_q u \leq 0, \\
+\infty, & \text{otherwise}.
\end{cases}
\]
The maximizer $\Delta\alpha_q$ of the update objective must be in the
range where the conjugate loss is not infinite. Ignoring such bounds,
the update can be obtained by setting the derivative of the objective
to zero, obtaining
\[
\tilde {\Delta \alpha_q}= \frac{y_q - B}{A}.
\]
Note that $B$ is simply current score associated by the SVM to
the sample $\bx_q$. Incorporating the constraint $-1 \leq - y_q
(\alpha_q + \Delta \alpha_q) \leq 0$,
i.e. $0 \leq y_q (\alpha_q + \Delta \alpha_q) \leq 1$, one obtains the update
\[
\Delta\alpha_q =  y_q \max\{0, \min\{1, y_q (\tilde {\Delta\alpha_q } + \alpha_q)\}\} - \alpha_q.
\]

<!-- ------------------------------------------------------------ --->
@section svm-sdca-details Implementation details
<!-- ------------------------------------------------------------ --->

Rather than visiting points completely at random, VLFeat SDCA follows
the best practice of visiting all the points at every epoch (pass
through the data), changing the order of the visit randomly by picking
every time a new random permutation.
**/

/**
<!-- ------------------------------------------------------------- -->
@page svm-sgd Stochastic Gradient Descent
@tableofcontents
<!-- ------------------------------------------------------------- -->

This page describes the *Stochastic Gradient Descent* (SGD) linear SVM
solver. SGD minimizes directly the primal SVM objective (see @ref svm):

\[
E(\bw) = \frac{\lambda}{2} \left\| \bw \right\|^2 + \frac{1}{n} \sum_{i=1}^n
\ell_i(\langle \bw,\bx\rangle)
\]

Firts, rewrite the objective as the average

\[
E(\bw) = \frac{1}{n} \sum_{i=1}^n E_i(\bw),
\quad
E_i(\bw) = \frac{\lambda}{2}  \left\| \bw \right\|^2 + \ell_i(\langle \bw,\bx\rangle).
\]

Then SGD performs gradient steps by considering at each iteration
one term $E_i(\bw)$ selected at random from this average.
In its most basic form, the algorithm is:

- Start with $\bw_0 = 0$.
- For $t=1,2,\dots T$:
  - Sample one index $i$ in $1,\dots,n$ uniformly at random.
  - Compute a subgradient $\bg_t$ of $E_i(\bw)$ at $\bw_t$.
  - Compute the learning rate $\eta_t$.
  - Update $\bw_{t+1} = \bw_t - \eta_t \bg_t$.

Provided that the learning rate $\eta_t$ is chosen correctly, this
simple algorithm is guaranteed to converge to the minimizer $\bw^*$ of
$E$.

<!-- ------------------------------------------------------------- -->
@section svm-sgd-convergence Convergence and speed
<!-- ------------------------------------------------------------- -->

The goal of the SGD algorithm is to bring the *primal suboptimality*
below a threshold $\epsilon_P$:
\[
E(\bw_t) - E(\bw^*) \leq \epsilon_P.
\]

If the learning rate $\eta_t$ is selected appropriately, SGD can be
shown to converge properly. For example,
@cite{shalev-shwartz07pegasos} show that, since $E(\bw)$ is
$\lambda$-strongly convex, then using the learning rate
\[
\boxed{\eta_t = \frac{1}{\lambda t}}
\]
guarantees that the algorithm reaches primal-suboptimality $\epsilon_P$ in
\[
\tilde O\left( \frac{1}{\lambda \epsilon_P} \right).
\]
iterations. This particular SGD variant is sometimes known as PEGASOS
@cite{shalev-shwartz07pegasos} and is the version implemented in
VLFeat.

The *convergence speed* is not sufficient to tell the *learning speed*,
i.e. how quickly an algorithm can learn an SVM that performs optimally
on the test set. The following two observations
can be used to link convergence speed to learning speed:

- The regularizer strength is often heuristically selected to be
  inversely proportional to the number of training samples: $\lambda =
  \lambda_0 /n$. This reflects the fact that with more training data
  the prior should count less.
- The primal suboptimality $\epsilon_P$ should be about the same as
  the estimation error of the SVM primal. This estimation error is due
  to the finite training set size and can be shown to be of the order
  of $1/\lambda n = 1 / \lambda_0$.

Under these two assumptions, PEGASOS can learn a linear SVM in time
$\tilde O(n)$, which is *linear in the number of training
examples*. This fares much better with $O(n^2)$ or worse of non-linear
SVM solvers.

<!-- ------------------------------------------------------------- -->
@section svm-sgd-bias The bias term
<!-- ------------------------------------------------------------- -->

Adding a bias $b$ to the SVM scoring function $\langle \bw, \bx
\rangle +b$ is done, as explained in @ref svm-bias, by appending a
constant feature $B$ (the *bias multiplier*) to the data vectors $\bx$
and a corresponding weight element $w_b$ to the weight vector $\bw$,
so that $b = B w_b$ As noted, the bias multiplier should be
relatively large in order to avoid shrinking the bias towards zero,
but small to make the optimization stable. In particular, setting $B$
to zero learns an unbiased SVM (::vl_svm_set_bias_multiplier).

To counter instability caused by a large bias multiplier, the learning
rate of the bias is slowed down by multiplying the overall learning
rate $\eta_t$ by a bias-specific rate coefficient
(::vl_svm_set_bias_learning_rate).

As a rule of thumb, if the data vectors $\bx$ are $l^2$ normalized (as
they typically should for optimal performance), then a reasonable bias
multiplier is in the range 1 to 10 and a reasonable bias learning rate
is somewhere in the range of the inverse of that (in this manner the
two parts of the extended feature vector $(\bx, B)$ are balanced).

<!-- ------------------------------------------------------------- -->
@section svm-sgd-starting-iteration Adjusting the learning rate
<!-- ------------------------------------------------------------- -->

Initially, the learning rate $\eta_t = 1/\lambda t$ is usually too
fast: as usually $\lambda \ll 1$, $\eta_1 \gg 1$. But this is clearly
excessive (for example, without a loss term, the best learning rate at
the first iteration is simply $\eta_1=1$, as this nails the optimum in
one step). Thus, the learning rate formula is modified to be $\eta_t =
1 / \lambda (t + t_0)$, where $t_0 \approx 2/\lambda$, which is
equivalent to start $t_0$ iterations later. In this manner $\eta_1
\approx 1/2$.

<!-- ------------------------------------------------------------ --->
@subsection svm-sgd-warm-start Warm start
<!-- ------------------------------------------------------------ --->

Starting from a given model $\bw$ is easy in SGD as the optimization
runs in the primal. However, the starting iteration index $t$ should
also be advanced for a warm start, as otherwise the initial setting of
$\bw$ is rapidly forgot (::vl_svm_set_model, ::vl_svm_set_bias,
::vl_svm_set_iteration_number).

<!-- ------------------------------------------------------------- -->
@section svm-sgd-details Implementation details
<!-- ------------------------------------------------------------- -->

@par "Random sampling of points"

Rather than visiting points completely at random, VLFeat SDCA follows
the best practice of visiting all the points at every epoch (pass
through the data), changing the order of the visit randomly by picking
every time a new random permutation.

@par "Factored representation"

At each iteration, the SGD algorithm updates the vector $\bw$
(including the additional bias component $w_b$) as $\bw_{t+1}
\leftarrow \bw_t - \lambda \eta_t \bw_t - \eta_t \bg_t$, where
$\eta_t$ is the learning rate. If the subgradient of the loss function
$\bg_t$ is zero at a given iteration, this amounts to simply shrink
$\bw$ towards the origin by multiplying it by the factor $1 - \lambda
\eta_t$. Thus such an iteration can be accelerated significantly by
representing internally $\bw_t = f_t \bu_t$, where $f_t$ is a scaling
factor. Then, the update becomes
\[
   f_{t+1} \bu_{t+1}
   = f_{t} \bu_{t} - \lambda \eta_t f_{t} \bu_{t} - \eta_t \bg_t
   = (1-\lambda \eta_t) f_{t} \bu_{t} - \eta_t \bg_t.
\]
Setting $f_{t+1} = (1-\lambda \eta_t) f_{t}$, this gives the update
equation for $\bu_t$
\[
\bu_{t+1} = \bu_{t} - \frac{\eta_t}{f_{t+1}} \bg_t.
\]
but this step can be skipped whenever $\bg_t$ is equal to zero.

When the bias component has a different learning rate, this scheme
must be adjusted slightly by adding a separated factor for the bias,
but it is otherwise identical.


**/

/*

<!-- ------------------------------------------------------------ --->
@section svm-pegasos PEGASOS
<!-- ------------------------------------------------------------ --->

<!-- ------------------------------------------------------------ --->
@subsection svm-pegasos-algorithm Algorithm
<!-- ------------------------------------------------------------ --->

PEGASOS @cite{shalev-shwartz07pegasos} is a stochastic subgradient
optimizer. At the <em>t</em>-th iteration the algorithm:

- Samples uniformly at random as subset @f$ A_t @f$ of <em>k</em> of
training pairs @f$(x,y)@f$ from the <em>m</em> pairs provided for
training (this subset is called mini batch).
- Computes a subgradient @f$ \nabla_t @f$ of the function @f$ E_t(w) =
\frac{1}{2}\|w\|^2 + \frac{1}{k} \sum_{(x,y) \in A_t} \ell(w;(x,y))
@f$ (this is the SVM objective function restricted to the
minibatch).
- Compute an intermediate weight vector @f$ w_{t+1/2} @f$ by doing a
step @f$ w_{t+1/2} = w_t - \alpha_t \nabla_t @f$ with learning rate
@f$ \alpha_t = 1/(\eta t) @f$ along the subgradient. Note that the
learning rate is inversely proportional to the iteration number.
- Back projects the weight vector @f$ w_{t+1/2} @f$ on the
hypersphere of radius @f$ \sqrt{\lambda} @f$ to obtain the next
model estimate @f$ w_{t+1} @f$:
@f[
w_t = \min\{1, \sqrt{\lambda}/\|w\|\} w_{t+1/2}.
@f]
The hypersphere is guaranteed to contain the optimal weight vector
@f$ w^* @f$.

VLFeat implementation fixes to one the size of the mini batches @f$ k
@f$.


<!-- ------------------------------------------------------------ --->
@subsection svm-pegasos-permutation Permutation
<!-- ------------------------------------------------------------ --->

VLFeat PEGASOS can use a user-defined permutation to decide the order
in which data points are visited (instead of using random
sampling). By specifying a permutation the algorithm is guaranteed to
visit each data point exactly once in each loop. The permutation needs
not to be bijective. This can be used to visit certain data samples
more or less often than others, implicitly reweighting their relative
importance in the SVM objective function. This can be used to balance
the data.

<!-- ------------------------------------------------------------ --->
@subsection svm-pegasos-kernels Non-linear kernels
<!-- ------------------------------------------------------------ --->

PEGASOS can be extended to non-linear kernels, but the algorithm is
not particularly efficient in this setting [1]. When possible, it may
be preferable to work with explicit feature maps.

Let @f$ k(x,y) @f$ be a positive definite kernel. A <em>feature
map</em> is a function @f$ \Psi(x) @f$ such that @f$ k(x,y) = \langle
\Psi(x), \Psi(y) \rangle @f$. Using this representation the non-linear
SVM learning objective function writes:

@f[
\min_{w} \frac{\lambda}{2} \|w\|^2 + \frac{1}{m} \sum_{i=1}^n
\ell(w; (\Psi(x)_i,y_i)).
@f]

Thus the only difference with the linear case is that the feature @f$
\Psi(x) @f$ is used in place of the data @f$ x @f$.

@f$ \Psi(x) @f$ can be learned off-line, for instance by using the
incomplete Cholesky decomposition @f$ V^\top V @f$ of the Gram matrix
@f$ K = [k(x_i,x_j)] @f$ (in this case @f$ \Psi(x_i) @f$ is the
<em>i</em>-th columns of <em>V</em>). Alternatively, for additive
kernels (e.g. intersection, Chi2) the explicit feature map computed by
@ref homkermap.h can be used.

For additive kernels it is also possible to perform the feature
expansion online inside the solver, setting the specific feature map
via ::vl_svmdataset_set_map. This is particular useful to keep the
size of the training data small, when the number of the samples is big
or the memory is limited.
*/

#include "svm.h"
#include "mathop.h"
#include <string.h>

struct VlSvm_ {
  VlSvmSolverType solver ;      /**< SVM solver type. */

  vl_size dimension ;           /**< Model dimension. */
  double * model ;              /**< Model ($\bw$ vector). */
  double bias ;                 /**< Bias. */
  double biasMultiplier ;       /**< Bias feature multiplier. */

  /* valid during a run */
  double lambda ;               /**< Regularizer multiplier. */
  void const * data ;
  vl_size numData ;
  double const * labels ;       /**< Data labels. */
  double const * weights ;      /**< Data weights. */

  VlSvmDataset * ownDataset ;   /**< Optional owned dataset. */

  VlSvmDiagnosticFunction diagnosticFn ;
  void * diagnosticFnData ;
  vl_size diagnosticFrequency ; /**< Frequency of diagnostic. */

  VlSvmLossFunction lossFn ;
  VlSvmLossFunction conjugateLossFn ;
  VlSvmLossFunction lossDerivativeFn ;
  VlSvmDcaUpdateFunction dcaUpdateFn ;
  VlSvmInnerProductFunction innerProductFn ;
  VlSvmAccumulateFunction accumulateFn ;

  vl_size iteration ;           /**< Current iterations number. */
  vl_size maxNumIterations ;    /**< Maximum number of iterations. */
  double epsilon ;              /**< Stopping threshold. */

  /* Book keeping */
  VlSvmStatistics statistics ;  /**< Statistcs. */
  double * scores ;

  /* SGD specific */
  double  biasLearningRate ;    /**< Bias learning rate. */

  /* SDCA specific */
  double * alpha ;              /**< Dual variables. */
} ;

/* ---------------------------------------------------------------- */

/** @brief Create a new object with plain data.
 ** @param type type of SMV solver.
 ** @param data a pointer to a matrix of data.
 ** @param dimension dimension of the SVM model.
 ** @param numData number of training samples.
 ** @param labels training labels.
 ** @param lambda regularizer parameter.
 ** @return the new object.
 **
 ** @a data has one column per sample, in @c double format.
 ** More advanced inputs can be used with ::vl_svm_new_with_dataset
 ** and ::vl_svm_new_with_abstract_data.
 **
 ** @sa ::vl_svm_delete
 **/

VlSvm *
vl_svm_new (VlSvmSolverType type,
            double const * data,
            vl_size dimension,
            vl_size numData,
            double const * labels,
            double lambda)
{
  VlSvmDataset * dataset = vl_svmdataset_new(VL_TYPE_DOUBLE, (void*)data, dimension, numData) ;
  VlSvm * self = vl_svm_new_with_dataset (type, dataset, labels, lambda) ;
  self->ownDataset = dataset ;
  return self ;
}

/** @brief Create a new object with a dataset.
 ** @param solver type of SMV solver.
 ** @param dataset SVM dataset object
 ** @param labels training samples labels.
 ** @param lambda regularizer parameter.
 ** @return the new object.
 ** @sa ::vl_svm_delete
 **/

VlSvm *
vl_svm_new_with_dataset (VlSvmSolverType solver,
                         VlSvmDataset * dataset,
                         double const * labels,
                         double lambda)
{
  VlSvm * self = vl_svm_new_with_abstract_data (solver,
                                             dataset,
                                             vl_svmdataset_get_dimension(dataset),
                                             vl_svmdataset_get_num_data(dataset),
                                             labels,
                                             lambda) ;
  vl_svm_set_data_functions (self,
                             vl_svmdataset_get_inner_product_function(dataset),
                             vl_svmdataset_get_accumulate_function(dataset)) ;
  return self ;
}

/** @brief Create a new object with abstract data.
 ** @param solver type of SMV solver.
 ** @param data pointer to the data.
 ** @param dimension dimension of the SVM model.
 ** @param numData num training samples.
 ** @param labels training samples labels.
 ** @param lambda regularizer parameter.
 ** @return the new object.
 **
 ** After calling this function, ::vl_svm_set_data_functions *must*
 ** be used to setup suitable callbacks for the inner product
 ** and accumulation operations (@see svm-data-abstraction).
 **
 ** @sa ::vl_svm_delete
 **/

VlSvm *
vl_svm_new_with_abstract_data (VlSvmSolverType solver,
                               void * data,
                               vl_size dimension,
                               vl_size numData,
                               double const * labels,
                               double lambda)
{
  VlSvm * self = vl_calloc(1,sizeof(VlSvm)) ;

  assert(dimension >= 1) ;
  assert(numData >= 1) ;
  assert(labels) ;

  self->solver = solver ;

  self->dimension = dimension ;
  self->model = 0 ;
  self->bias = 0 ;
  self->biasMultiplier = 1.0 ;

  self->lambda = lambda ;
  self->data = data ;
  self->numData = numData ;
  self->labels = labels ;

  self->diagnosticFrequency = numData ;
  self->diagnosticFn = 0 ;
  self->diagnosticFnData = 0 ;

  self->lossFn = vl_svm_hinge_loss ;
  self->conjugateLossFn = vl_svm_hinge_conjugate_loss ;
  self->lossDerivativeFn = vl_svm_hinge_loss_derivative ;
  self->dcaUpdateFn = vl_svm_hinge_dca_update ;

  self->innerProductFn = 0 ;
  self->accumulateFn = 0 ;

  self->iteration = 0 ;
  self->maxNumIterations = VL_MAX((double)numData, vl_ceil_f(10.0 / lambda)) ;
  self->epsilon = 1e-2 ;

  /* SGD */
  self->biasLearningRate = 0.01 ;

  /* SDCA */
  self->alpha = 0 ;

  /* allocations */
  self->model = vl_calloc(dimension, sizeof(double)) ;
  if (self->model == NULL) goto err_alloc ;

  if (self->solver == VlSvmSolverSdca) {
    self->alpha = vl_calloc(self->numData, sizeof(double)) ;
    if (self->alpha == NULL) goto err_alloc ;
  }

  self->scores = vl_calloc(numData, sizeof(double)) ;
  if (self->scores == NULL) goto err_alloc ;

  return self ;

err_alloc:
  if (self->scores) {
    vl_free (self->scores) ;
    self->scores = 0 ;
  }
  if (self->model) {
    vl_free (self->model) ;
    self->model = 0 ;
  }
  if (self->alpha) {
    vl_free (self->alpha) ;
    self->alpha = 0 ;
  }
  return 0 ;
}

/** @brief Delete object.
 ** @param self object.
 ** @sa ::vl_svm_new
 **/

void
vl_svm_delete (VlSvm * self)
{
  if (self->model) {
    vl_free (self->model) ;
    self->model = 0 ;
  }
  if (self->alpha) {
    vl_free (self->alpha) ;
    self->alpha = 0 ;
  }
  if (self->ownDataset) {
    vl_svmdataset_delete(self->ownDataset) ;
    self->ownDataset = 0 ;
  }
  vl_free (self) ;
}

/* ---------------------------------------------------------------- */
/*                                              Setters and getters */
/* ---------------------------------------------------------------- */

/** @brief Set the convergence threshold
 ** @param self object
 ** @param epsilon threshold (non-negative).
 **/

void vl_svm_set_epsilon (VlSvm *self, double epsilon)
{
  assert(self) ;
  assert(epsilon >= 0) ;
  self->epsilon = epsilon ;
}

/** @brief Get the convergence threshold
 ** @param self object
 ** @return epsilon threshold.
 **/

double vl_svm_get_epsilon (VlSvm const *self)
{
  assert(self) ;
  return self->epsilon ;
}

/** @brief Set the bias learning rate
 ** @param self object
 ** @param rate bias learning rate (positive).
 **
 ** This parameter applies only to the SGD solver.
 **/

void vl_svm_set_bias_learning_rate (VlSvm *self, double rate)
{
  assert(self) ;
  assert(rate > 0) ;
  self->biasLearningRate = rate ;
}

/** @brief Get the bias leraning rate.
 ** @param self object
 ** @return bias learning rate.
 **/

double vl_svm_get_bias_learning_rate (VlSvm const *self)
{
  assert(self) ;
  return self->biasLearningRate ;
}

/** @brief Set the bias multiplier.
 ** @param self object
 ** @param b bias multiplier.
 **
 ** The *bias multiplier* is the value of the constant feature
 ** appended to the data vectors to implement the bias (@ref svm-bias).
 **/

void vl_svm_set_bias_multiplier (VlSvm * self, double b)
{
  assert(self) ;
  assert(b >= 0) ;
  self->biasMultiplier = b ;
}

/** @brief Get the bias multiplier.
 ** @param self object.
 ** @return bias multiplier.
 **/

double vl_svm_get_bias_multiplier (VlSvm const * self)
{
  assert(self) ;
  return self->biasMultiplier ;
}

/** @brief Set the current iteratio number.
 ** @param self object.
 ** @param n iteration number.
 **
 ** If called before training,
 ** this can be used with SGD for a warm start, as the net
 ** effect is to slow down the learning rate.
 **/

void vl_svm_set_iteration_number (VlSvm *self, vl_uindex n)
{
  assert(self) ;
  self->iteration = n ;
}

/** @brief Get the current iteration number.
 ** @param self object.
 ** @return current iteration number.
 **/

vl_size vl_svm_get_iteration_number (VlSvm const *self)
{
  assert(self) ;
  return self->iteration ;
}

/** @brief Set the maximum number of iterations.
 ** @param self object.
 ** @param n maximum number of iterations.
 **/

void vl_svm_set_max_num_iterations (VlSvm *self, vl_size n)
{
  assert(self) ;
  self->maxNumIterations = n ;
}

/** @brief Get the maximum number of iterations.
 ** @param self object.
 ** @return maximum number of iterations.
 **/

vl_size vl_svm_get_max_num_iterations (VlSvm const *self)
{
  assert(self) ;
  return self->maxNumIterations ;
}

/** @brief Set the diagnostic frequency.
 ** @param self object.
 ** @param f diagnostic frequency (@c >= 1).
 **
 ** A diagnostic round (to test for convergence and to printout
 ** information) is performed every @a f iterations.
 **/

void vl_svm_set_diagnostic_frequency (VlSvm *self, vl_size f)
{
  assert(self) ;
  assert(f > 0) ;
  self->diagnosticFrequency = f ;
}

/** @brief Get the diagnostic frequency.
 ** @param self object.
 ** @return diagnostic frequency.
 **/

vl_size vl_svm_get_diagnostic_frequency (VlSvm const *self)
{
  assert(self) ;
  return self->diagnosticFrequency ;
}

/** @brief Get the SVM solver type.
 ** @param self object.
 ** @return SVM solver type.
 **/

VlSvmSolverType vl_svm_get_solver (VlSvm const * self)
{
  assert(self) ;
  return self->solver ;
}

/** @brief Set the regularizer parameter lambda.
 ** @param self object.
 ** @param lambda regularizer parameter.
 **
 ** Note that @a lambda is usually set when calling a
 ** constructor for ::VlSvm as certain parameters, such
 ** as the maximum number of iterations, are tuned accordingly.
 ** This tuning is not performed when @a lambda is changed
 ** using this function.
 **/

void vl_svm_set_lambda (VlSvm * self, double lambda)
{
  assert(self) ;
  assert(lambda >= 0) ;
  self->lambda = lambda ;
}

/** @brief Get the regularizer parameter lambda.
 ** @param self object.
 ** @return diagnostic frequency.
 **/

double vl_svm_get_lambda (VlSvm const * self)
{
  assert(self) ;
  return self->lambda ;
}

/** @brief Set the data weights.
 ** @param self object.
 ** @param weights data weights.
 **
 ** @a weights must be an array of non-negative weights.
 ** The loss of each data point is multiplied by the corresponding
 ** weight.
 **
 ** Set @a weights to @c NULL to weight the data uniformly by 1 (default).
 **
 ** Note that the @a weights array is *not* copied and must be valid
 ** througout the object lifetime (unless it is replaced).
 **/

void vl_svm_set_weights (VlSvm * self, double const *weights)
{
  assert(self) ;
  self->weights = weights ;
}

/** @brief Get the data weights.
 ** @param self object.
 ** @return data weights.
 **/

double const *vl_svm_get_weights (VlSvm const * self)
{
  assert(self) ;
  return self->weights ;
}

/* ---------------------------------------------------------------- */
/*                                                         Get data */
/* ---------------------------------------------------------------- */

/** @brief Get the model dimenison.
 ** @param self object.
 ** @return model dimension.
 **
 ** This is the dimensionality of the weight vector $\bw$.
 **/

vl_size vl_svm_get_dimension (VlSvm *self)
{
  assert(self) ;
  return self->dimension ;
}

/** @brief Get the number of data samples.
 ** @param self object.
 ** @return model number of data samples
 **
 ** This is the dimensionality of the weight vector $\bw$.
 **/

vl_size vl_svm_get_num_data (VlSvm *self)
{
  assert(self) ;
  return self->numData ;
}

/** @brief Get the SVM model.
 ** @param self object.
 ** @return model.
 **
 ** This is the weight vector $\bw$.
 **/

double const * vl_svm_get_model (VlSvm const *self)
{
  assert(self) ;
  return self->model ;
}

/** @brief Set the SVM model.
 ** @param self object.
 ** @param model model.
 **
 ** The function *copies* the content of the vector @a model to the
 ** internal model buffer. This operation can be used for warm start
 ** with the SGD algorithm, but has undefined effect with the SDCA algorithm.
 **/

void vl_svm_set_model (VlSvm *self, double const *model)
{
  assert(self) ;
  assert(model) ;
  memcpy(self->model, model, sizeof(double) * vl_svm_get_dimension(self)) ;
}

/** @brief Set the SVM bias.
 ** @param self object.
 ** @param b bias.
 **
 ** The function set the internal representation of the SVM bias to
 ** be equal to @a b (the bias multiplier
 ** is applied). The same remark
 ** that applies to ::vl_svm_set_model applies here too.
 **/

void vl_svm_set_bias (VlSvm *self, double b)
{
  assert(self);
  if (self->biasMultiplier) {
    self->bias = b / self->biasMultiplier ;
  }
}

/** @brief Get the value of the bias.
 ** @param self object.
 ** @return bias $b$.
 **
 ** The value of the bias returned already include the effect of
 ** bias mutliplier.
 **/

double vl_svm_get_bias (VlSvm const *self)
{
  assert(self) ;
  return self->bias * self->biasMultiplier ;
}

/** @brief Get the solver statistics.
 ** @param self object.
 ** @return statistics.
 **/

VlSvmStatistics const * vl_svm_get_statistics (VlSvm const *self)
{
  assert(self) ;
  return &self->statistics ;
}

/** @brief Get the scores of the data points.
 ** @param self object.
 ** @return vector of scores.
 **
 ** After training or during the diagnostic callback,
 ** this function can be used to retrieve the scores
 ** of the points, i.e. $\langle \bx_i, \bw \rangle + b$.
 **/

double const * vl_svm_get_scores (VlSvm const *self)
{
  return self->scores ;
}

/* ---------------------------------------------------------------- */
/*                                                        Callbacks */
/* ---------------------------------------------------------------- */

/** @typedef VlSvmDiagnosticFunction
 ** @brief SVM diagnostic function pointer.
 ** @param svm is an instance of ::VlSvm .
 **/

/** @typedef VlSvmAccumulateFunction
 ** @brief Pointer to a function that adds to @a model the data point at
 ** position @a element multiplied by the constant @a multiplier.
 **/

/** @typedef VlSvmInnerProductFunction
 ** @brief Pointer to a function that defines the inner product
 ** between the data point at position @a element and the SVM model
 **/

/** @brief Set the diagnostic function callback
 ** @param self object.
 ** @param f diagnostic function pointer.
 ** @param data pointer to data used by the diagnostic function.
 **/

void
vl_svm_set_diagnostic_function (VlSvm *self, VlSvmDiagnosticFunction f, void *data) {
  self->diagnosticFn = f ;
  self->diagnosticFnData = data ;
}

/** @brief Set the data functions.
 ** @param self object.
 ** @param inner inner product function.
 ** @param acc accumulate function.
 **
 ** See @ref svm-data-abstraction.
 **/

void vl_svm_set_data_functions (VlSvm *self, VlSvmInnerProductFunction inner, VlSvmAccumulateFunction acc)
{
  assert(self) ;
  assert(inner) ;
  assert(acc) ;
  self->innerProductFn = inner ;
  self->accumulateFn = acc ;
}

/** @brief Set the loss function callback.
 ** @param self object.
 ** @param f loss function callback.
 **
 ** Note that setting up a loss requires specifying more than just one
 ** callback. See @ref svm-loss-functions for details.
 **/

void vl_svm_set_loss_function (VlSvm *self, VlSvmLossFunction f)
{
  assert(self) ;
  self->lossFn = f ;
}

/** @brief Set the loss derivative function callback.
 ** @copydetails vl_svm_set_loss_function.
 **/

void vl_svm_set_loss_derivative_function (VlSvm *self, VlSvmLossFunction f)
{
  assert(self) ;
  self->lossDerivativeFn = f ;
}

/** @brief Set the conjugate loss function callback.
 ** @copydetails vl_svm_set_loss_function.
 **/

void vl_svm_set_conjugate_loss_function (VlSvm *self, VlSvmLossFunction f)
{
  assert(self) ;
  self->conjugateLossFn = f ;
}

/** @brief Set the DCA update function callback.
 ** @copydetails vl_svm_set_loss_function.
 **/

void vl_svm_set_dca_update_function (VlSvm *self, VlSvmDcaUpdateFunction f)
{
  assert(self) ;
  self->dcaUpdateFn = f ;
}

/** @brief Set the loss function to one of the default types.
 ** @param self object.
 ** @param loss type of loss function.
 ** @sa @ref svm-loss-functions.
 **/

void
vl_svm_set_loss (VlSvm *self, VlSvmLossType loss)
{
#define SETLOSS(x,y) \
case VlSvmLoss ## x: \
  vl_svm_set_loss_function(self, vl_svm_ ## y ## _loss) ; \
  vl_svm_set_loss_derivative_function(self, vl_svm_ ## y ## _loss_derivative) ; \
  vl_svm_set_conjugate_loss_function(self, vl_svm_ ## y ## _conjugate_loss) ; \
  vl_svm_set_dca_update_function(self, vl_svm_ ## y ## _dca_update) ; \
  break;

  switch (loss) {
      SETLOSS(Hinge, hinge) ;
      SETLOSS(Hinge2, hinge2) ;
      SETLOSS(L1, l1) ;
      SETLOSS(L2, l2) ;
      SETLOSS(Logistic, logistic) ;
    default:
      assert(0) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                               Pre-defined losses */
/* ---------------------------------------------------------------- */

/** @typedef VlSvmLossFunction
 ** @brief SVM loss function pointer.
 ** @param inner inner product between sample and model $\bw^\top \bx$.
 ** @param label sample label $y$.
 ** @return value of the loss.
 **
 ** The interface is the same for a loss function, its derivative,
 ** or the conjugate loss.
 **
 ** @sa @ref svm-fundamentals
 **/

/** @typedef VlSvmDcaUpdateFunction
 ** @brief SVM SDCA update function pointer.
 ** @param alpha current value of the dual variable.
 ** @param inner inner product $\bw^\top \bx$ of the sample with the SVM model.
 ** @param norm2 normalization factor $\|\bx\|^2/\lambda n$.
 ** @param label label $y$ of the sample.
 ** @return incremental update $\Delta\alpha$ of the dual variable.
 **
 ** @sa @ref svm-sdca
 **/

/** @brief SVM hinge loss
 ** @copydetails VlSvmLossFunction */
double
vl_svm_hinge_loss (double inner, double label)
{
  return VL_MAX(1 - label * inner, 0.0);
}

/** @brief SVM hinge loss derivative
 ** @copydetails VlSvmLossFunction */
double
vl_svm_hinge_loss_derivative (double inner, double label)
{
  if (label * inner < 1.0) {
    return - label ;
  } else {
    return 0.0 ;
  }
}

/** @brief SVM hinge loss conjugate
 ** @param u dual variable.
 ** @param label label value.
 ** @return conjugate loss.
 **/
double
vl_svm_hinge_conjugate_loss (double u, double label) {
  double z = label * u ;
  if (-1 <= z && z <= 0) {
    return label * u ;
  } else {
    return VL_INFINITY_D ;
  }
}

/** @brief SVM hinge loss DCA update
 ** @copydetails VlSvmDcaUpdateFunction */
double
vl_svm_hinge_dca_update (double alpha, double inner, double norm2, double label) {
  double palpha = (label - inner) / norm2 + alpha ;
  return label * VL_MAX(0, VL_MIN(1, label * palpha)) - alpha ;
}

/** @brief SVM square hinge loss
 ** @copydetails VlSvmLossFunction */
double
vl_svm_hinge2_loss (double inner,double label)
{
  double z = VL_MAX(1 - label * inner, 0.0) ;
  return z*z ;
}

/** @brief SVM square hinge loss derivative
 ** @copydetails VlSvmLossFunction */
double
vl_svm_hinge2_loss_derivative (double inner, double label)
{
  if (label * inner < 1.0) {
    return 2 * (inner - label) ;
  } else {
    return 0 ;
  }
}

/** @brief SVM square hinge loss conjugate
 ** @copydetails vl_svm_hinge_conjugate_loss */
double
vl_svm_hinge2_conjugate_loss (double u, double label) {
  if (label * u <= 0) {
    return (label + u/4) * u ;
  } else {
    return VL_INFINITY_D ;
  }
}

/** @brief SVM square hinge loss DCA update
 ** @copydetails VlSvmDcaUpdateFunction */
double
vl_svm_hinge2_dca_update (double alpha, double inner, double norm2, double label) {
  double palpha = (label - inner - 0.5*alpha) / (norm2 + 0.5) + alpha ;
  return label * VL_MAX(0, label * palpha) - alpha ;
}

/** @brief SVM l1 loss
 ** @copydetails VlSvmLossFunction */
double
vl_svm_l1_loss (double inner,double label)
{
  return vl_abs_d(label - inner) ;
}

/** @brief SVM l1 loss derivative
 ** @copydetails VlSvmLossFunction */
double
vl_svm_l1_loss_derivative (double inner, double label)
{
  if (label > inner) {
    return - 1.0 ;
  } else {
    return + 1.0 ;
  }
}

/** @brief SVM l1 loss conjugate
 ** @copydetails vl_svm_hinge_conjugate_loss */
double
vl_svm_l1_conjugate_loss (double u, double label) {
  if (vl_abs_d(u) <= 1) {
    return label*u ;
  } else {
    return VL_INFINITY_D ;
  }
}

/** @brief SVM l1 loss DCA update
 ** @copydetails VlSvmDcaUpdateFunction */
double
vl_svm_l1_dca_update (double alpha, double inner, double norm2, double label) {
  if (vl_abs_d(alpha) <= 1) {
    double palpha = (label - inner) / norm2 + alpha ;
    return VL_MAX(-1.0, VL_MIN(1.0, palpha)) - alpha ;
  } else {
    return VL_INFINITY_D ;
  }
}

/** @brief SVM l2 loss
 ** @copydetails VlSvmLossFunction */
double
vl_svm_l2_loss (double inner,double label)
{
  double z = label - inner ;
  return z*z ;
}

/** @brief SVM l2 loss derivative
 ** @copydetails VlSvmLossFunction */
double
vl_svm_l2_loss_derivative (double inner, double label)
{
  return - 2 * (label - inner) ;
}

/** @brief SVM l2 loss conjugate
 ** @copydetails vl_svm_hinge_conjugate_loss */
double
vl_svm_l2_conjugate_loss (double u, double label) {
  return (label + u/4) * u ;
}

/** @brief SVM l2 loss DCA update
 ** @copydetails VlSvmDcaUpdateFunction */
double
vl_svm_l2_dca_update (double alpha, double inner, double norm2, double label) {
  return (label - inner - 0.5*alpha) / (norm2 + 0.5) ;
}

/** @brief SVM l2 loss
 ** @copydetails VlSvmLossFunction */
double
vl_svm_logistic_loss (double inner,double label)
{
  double z = label * inner ;
  if (z >= 0) {
    return log(1.0 + exp(-z)) ;
  } else {
    return -z + log(exp(z) + 1.0) ;
  }
}

/** @brief SVM l2 loss derivative
 ** @copydetails VlSvmLossFunction */
double
vl_svm_logistic_loss_derivative (double inner, double label)
{
  double z = label * inner ;
  double t = 1 / (1 + exp(-z)) ; /* this is stable for z << 0 too */
  return label * (t - 1) ; /*  = -label exp(-z) / (1 + exp(-z)) */
}

VL_INLINE double xlogx(double x)
{
  if (x <= 1e-10) return 0 ;
  return x*log(x) ;
}

/** @brief SVM l2 loss conjugate
 ** @copydetails vl_svm_hinge_conjugate_loss */
double
vl_svm_logistic_conjugate_loss (double u, double label) {
  double z = label * u ;
  if (-1 <= z && z <= 0) {
    return xlogx(-z) + xlogx(1+z) ;
  } else {
    return VL_INFINITY_D ;
  }
}

/** @brief SVM l2 loss DCA update
 ** @copydetails VlSvmDcaUpdateFunction */
double
vl_svm_logistic_dca_update (double alpha, double inner, double norm2, double label) {
  /*
   The goal is to solve the problem

   min_delta A/2 delta^2 + B delta + l*(-alpha - delta|y),  -1 <= - y (alpha+delta) <= 0

   where A = norm2, B = inner, and y = label. To simplify the notation, we set

     f(beta) = beta * log(beta) + (1 - beta) * log(1 - beta)

   where beta = y(alpha + delta) such that

     l*(-alpha - delta |y) = f(beta).

   Hence 0 <= beta <= 1, delta = + y beta - alpha. Substituting

     min_beta A/2 beta^2 + y (B - A alpha) beta + f(beta) + const

   The Newton step is then given by

     beta = beta - (A beta + y(B - A alpha) + df) / (A + ddf).

   However, the function is singluar for beta=0 and beta=1 (infinite
   first and second order derivatives). Since the function is monotonic
   (second derivarive always strictly greater than zero) and smooth,
   we canuse bisection to find the zero crossing of the first derivative.
   Once one is sufficiently close to the optimum, a one or two Newton
   steps are sufficien to land on it with excellent accuracy.
   */

  double  df, ddf, der, dder ;
  vl_index t ;

  /* bisection */
  double beta1 = 0 ;
  double beta2 = 1 ;
  double beta = 0.5 ;

  for (t = 0 ; t < 5 ; ++t) {
    df = log(beta) - log(1-beta) ;
    der = norm2 * beta + label * (inner - norm2*alpha) + df ;
    if (der >= 0) {
      beta2 = beta ;
    } else {
      beta1 = beta ;
    }
    beta = 0.5 * (beta1 + beta2) ;
  }

#if 1
  /* a final Newton step, but not too close to the singularities */
  for (t = 0 ; (t < 2) & (beta > VL_EPSILON_D) & (beta < 1-VL_EPSILON_D) ; ++t) {
    df = log(beta) - log(1-beta) ;
    ddf = 1 / (beta * (1-beta)) ;
    der = norm2 * beta + label * (inner - norm2*alpha) + df ;
    dder = norm2 + ddf ;
    beta -= der / dder ;
    beta = VL_MAX(0, VL_MIN(1, beta)) ;
  }
#endif

  return label * beta - alpha ;
}

/* ---------------------------------------------------------------- */

/** @internal @brief Update SVM statistics
 ** @param self object.
 **/

void _vl_svm_update_statistics (VlSvm *self)
{
  vl_size i, k ;
  double inner, p ;

  memset(&self->statistics, 0, sizeof(VlSvmStatistics)) ;

  self->statistics.regularizer = self->bias * self->bias ;
  for (i = 0; i < self->dimension; i++) {
    self->statistics.regularizer += self->model[i] * self->model[i] ;
  }
  self->statistics.regularizer *= self->lambda * 0.5 ;

  for (k = 0; k < self->numData ; k++) {
    p = (self->weights) ? self->weights[k] : 1.0 ;
    if (p <= 0) continue ;
    inner = self->innerProductFn(self->data, k, self->model) ;
    inner += self->bias * self->biasMultiplier ;
    self->scores[k] = inner ;
    self->statistics.loss += p * self->lossFn(inner, self->labels[k]) ;
    if (self->solver == VlSvmSolverSdca) {

      self->statistics.dualLoss -= p * self->conjugateLossFn(- self->alpha[k] / p, self->labels[k]) ;
    }
  }

  self->statistics.loss /= self->numData ;
  self->statistics.objective = self->statistics.regularizer + self->statistics.loss ;

  if (self->solver == VlSvmSolverSdca) {
    self->statistics.dualLoss /= self->numData ;
    self->statistics.dualObjective = - self->statistics.regularizer + self->statistics.dualLoss ;
    self->statistics.dualityGap = self->statistics.objective - self->statistics.dualObjective ;
  }
}

/* ---------------------------------------------------------------- */
/*                                       Evaluate rather than solve */
/* ---------------------------------------------------------------- */

void _vl_svm_evaluate (VlSvm *self)
{
  double startTime = vl_get_cpu_time () ;

  _vl_svm_update_statistics (self) ;

  self->statistics.elapsedTime = vl_get_cpu_time() - startTime ;
  self->statistics.iteration = 0 ;
  self->statistics.epoch = 0 ;
  self->statistics.status = VlSvmStatusConverged ;

  if (self->diagnosticFn) {
    self->diagnosticFn(self, self->diagnosticFnData) ;
  }
}

/* ---------------------------------------------------------------- */
/*                         Stochastic Dual Coordinate Ascent Solver */
/* ---------------------------------------------------------------- */

void _vl_svm_sdca_train (VlSvm *self)
{
  double * norm2 ;
  vl_index * permutation ;
  vl_uindex i, t  ;
  double inner, delta, multiplier, p ;

  double startTime = vl_get_cpu_time () ;
  VlRand * rand = vl_get_rand() ;

  norm2 = (double*) vl_calloc(self->numData, sizeof(double));
  permutation = vl_calloc(self->numData, sizeof(vl_index)) ;

  {
    double * buffer = vl_calloc(self->dimension, sizeof(double)) ;
    for (i = 0 ; i < (unsigned)self->numData; i++) {
      double n2 ;
      permutation [i] = i ;
      memset(buffer, 0, self->dimension * sizeof(double)) ;
      self->accumulateFn (self->data, i, buffer, 1) ;
      n2 = self->innerProductFn (self->data, i, buffer) ;
      n2 += self->biasMultiplier * self->biasMultiplier ;
      norm2[i] = n2 / (self->lambda * self->numData) ;
    }
    vl_free(buffer) ;
  }

  for (t = 0 ; 1 ; ++t) {

    if (t % self->numData == 0) {
      /* once a new epoch is reached (all data have been visited),
       change permutation */
      vl_rand_permute_indexes(rand, permutation, self->numData) ;
    }

    /* pick a sample and compute update */
    i = permutation[t % self->numData] ;
    p = (self->weights) ? self->weights[i] : 1.0 ;
    if (p > 0) {
      inner = self->innerProductFn(self->data, i, self->model) ;
      inner += self->bias * self->biasMultiplier ;
      delta = p * self->dcaUpdateFn(self->alpha[i] / p, inner, p * norm2[i], self->labels[i]) ;
    } else {
      delta = 0 ;
    }

    /* apply update */
    if (delta != 0) {
      self->alpha[i] += delta ;
      multiplier = delta / (self->numData * self->lambda) ;
      self->accumulateFn(self->data,i,self->model,multiplier) ;
      self->bias += self->biasMultiplier * multiplier;
    }

    /* call diagnostic occasionally */
    if ((t + 1) % self->diagnosticFrequency == 0 || t + 1 == self->maxNumIterations) {
      _vl_svm_update_statistics (self) ;
      self->statistics.elapsedTime = vl_get_cpu_time() - startTime ;
      self->statistics.iteration = t ;
      self->statistics.epoch = t / self->numData ;

      self->statistics.status = VlSvmStatusTraining ;
      if (self->statistics.dualityGap < self->epsilon) {
        self->statistics.status = VlSvmStatusConverged ;
      }
      else if (t + 1 == self->maxNumIterations) {
        self->statistics.status = VlSvmStatusMaxNumIterationsReached ;
      }

      if (self->diagnosticFn) {
        self->diagnosticFn(self, self->diagnosticFnData) ;
      }

      if (self->statistics.status != VlSvmStatusTraining) {
        break ;
      }
    }
  } /* next iteration */

  vl_free (norm2) ;
  vl_free (permutation) ;
}

/* ---------------------------------------------------------------- */
/*                               Stochastic Gradient Descent Solver */
/* ---------------------------------------------------------------- */

void _vl_svm_sgd_train (VlSvm *self)
{
  vl_index * permutation ;
  double * scores ;
  double * previousScores ;
  vl_uindex i, t, k ;
  double inner, gradient, rate, biasRate, p ;
  double factor = 1.0 ;
  double biasFactor = 1.0 ; /* to allow slower bias learning rate */
  vl_index t0 = VL_MAX(2, vl_ceil_d(1.0 / self->lambda)) ;
  //t0=2 ;

  double startTime = vl_get_cpu_time () ;
  VlRand * rand = vl_get_rand() ;

  permutation = vl_calloc(self->numData, sizeof(vl_index)) ;
  scores = vl_calloc(self->numData * 2, sizeof(double)) ;
  previousScores = scores + self->numData ;

  for (i = 0 ; i < (unsigned)self->numData; i++) {
    permutation [i] = i ;
    previousScores [i] = - VL_INFINITY_D ;
  }

  /*
   We store the w vector as the product fw (factor * model).
   We also use a different factor for the bias: biasFactor * biasMultiplier
   to enable a slower learning rate for the bias.

   Given this representation, it is easy to carry the two key operations:

   * Inner product: <fw,x> = f <w,x>

   * Model update: fp wp = fw - rate * lambda * w - rate * g
                         = f(1 - rate * lambda) w - rate * g

     Thus the update equations are:

                   fp = f(1 - rate * lambda), and
                   wp = w + rate / fp * g ;

   * Realization of the scaling factor. Before the statistics function
     is called, or training finishes, the factor (and biasFactor)
     are explicitly applied to the model and the bias.
  */

  for (t = 0 ; 1 ; ++t) {

    if (t % self->numData == 0) {
      /* once a new epoch is reached (all data have been visited),
       change permutation */
      vl_rand_permute_indexes(rand, permutation, self->numData) ;
    }

    /* pick a sample and compute update */
    i = permutation[t % self->numData] ;
    p = (self->weights) ? self->weights[i] : 1.0 ;
    p = VL_MAX(0.0, p) ; /* we assume non-negative weights, so this is just for robustness */
    inner = factor * self->innerProductFn(self->data, i, self->model) ;
    inner += biasFactor * (self->biasMultiplier * self->bias) ;
    gradient = p * self->lossDerivativeFn(inner, self->labels[i]) ;
    previousScores[i] = scores[i] ;
    scores[i] = inner ;

    /* apply update */
    rate = 1.0 /  (self->lambda * (t + t0)) ;
    biasRate = rate * self->biasLearningRate ;
    factor *= (1.0 - self->lambda * rate) ;
    biasFactor *= (1.0 - self->lambda * biasRate) ;

    /* debug: realize the scaling factor all the times */
    /*
    for (k = 0 ; k < self->dimension ; ++k) self->model[k] *= factor ;
    self->bias *= biasFactor;
    factor = 1.0 ;
    biasFactor = 1.0 ;
    */

    if (gradient != 0) {
      self->accumulateFn(self->data, i, self->model, - gradient * rate / factor) ;
      self->bias += self->biasMultiplier * (- gradient * biasRate / biasFactor) ;
    }

    /* call diagnostic occasionally */
    if ((t + 1) % self->diagnosticFrequency == 0 || t + 1 == self->maxNumIterations) {

      /* realize factor before computing statistics or completing training */
      for (k = 0 ; k < self->dimension ; ++k) self->model[k] *= factor ;
      self->bias *= biasFactor;
      factor = 1.0 ;
      biasFactor = 1.0 ;

      _vl_svm_update_statistics (self) ;

      for (k = 0 ; k < self->numData ; ++k) {
        double delta = scores[k] - previousScores[k] ;
        self->statistics.scoresVariation += delta * delta ;
      }
      self->statistics.scoresVariation = sqrt(self->statistics.scoresVariation) / self->numData ;

      self->statistics.elapsedTime = vl_get_cpu_time() - startTime ;
      self->statistics.iteration = t ;
      self->statistics.epoch = t / self->numData ;

      self->statistics.status = VlSvmStatusTraining ;
      if (self->statistics.scoresVariation < self->epsilon) {
        self->statistics.status = VlSvmStatusConverged ;
      }
      else if (t + 1 == self->maxNumIterations) {
        self->statistics.status = VlSvmStatusMaxNumIterationsReached ;
      }

      if (self->diagnosticFn) {
        self->diagnosticFn(self, self->diagnosticFnData) ;
      }

      if (self->statistics.status != VlSvmStatusTraining) {
        break ;
      }
    }
  } /* next iteration */

  vl_free (scores) ;
  vl_free (permutation) ;
}

/* ---------------------------------------------------------------- */
/*                                                       Dispatcher */
/* ---------------------------------------------------------------- */

/** @brief Run the SVM solver
 ** @param self object.
 **
 ** The data on which the SVM operates is passed upon the cration of
 ** the ::VlSvm object. This function runs a solver to learn a
 ** corresponding model. See @ref svm-starting.
 **/

void vl_svm_train (VlSvm * self)
{
  assert (self) ;
  switch (self->solver) {
    case VlSvmSolverSdca:
      _vl_svm_sdca_train(self) ;
      break ;
    case VlSvmSolverSgd:
      _vl_svm_sgd_train(self) ;
      break ;
    case VlSvmSolverNone:
      _vl_svm_evaluate(self) ;
      break ;
    default:
      assert(0) ;
  }
}
