/** @file covdet.c
 ** @brief Covariant feature detectors - Definition
 ** @author Karel Lenc
 ** @author Andrea Vedaldi
 ** @author Michal Perdoch
 **/

/*
Copyright (C) 2013-14 Andrea Vedaldi.
Copyright (C) 2012 Karel Lenc, Andrea Vedaldi and Michal Perdoch.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page covdet Covariant feature detectors
@author Karel Lenc
@author Andrea Vedaldi
@author Michal Perdoch
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref covdet.h implements a number of covariant feature detectors, based
on three cornerness measures (determinant of the Hessian, trace of the Hessian
(aka Difference of Gaussians, and Harris). It supprots affine adaptation,
orientation estimation, as well as Laplacian scale detection.

- @subpage covdet-fundamentals
- @subpage covdet-principles
- @subpage covdet-differential
- @subpage covdet-corner-types

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-starting Getting started
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The ::VlCovDet object implements a number of covariant feature
detectors: Difference of Gaussian, Harris, determinant of Hessian.
Variant of the basic detectors support scale selection by maximizing
the Laplacian measure as well as affine normalization.

@code
// create a detector object
VlCovDet * covdet = vl_covdet_new(method) ;

// set various parameters (optional)
vl_covdet_set_first_octave(covdet, -1) ; // start by doubling the image resolution
vl_covdet_set_octave_resolution(covdet, octaveResolution) ;
vl_covdet_set_peak_threshold(covdet, peakThreshold) ;
vl_covdet_set_edge_threshold(covdet, edgeThreshold) ;

// process the image and run the detector
vl_covdet_put_image(covdet, image, numRows, numCols) ;
vl_covdet_detect(covdet) ;

// drop features on the margin (optional)
vl_covdet_drop_features_outside (covdet, boundaryMargin) ;

// compute the affine shape of the features (optional)
vl_covdet_extract_affine_shape(covdet) ;

// compute the orientation of the features (optional)
vl_covdet_extract_orientations(covdet) ;

// get feature frames back
vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
VlCovDetFeature const * feature = vl_covdet_get_features(covdet) ;

// get normalized feature appearance patches (optional)
vl_size w = 2*patchResolution + 1 ;
for (i = 0 ; i < numFeatures ; ++i) {
  float * patch = malloc(w*w*sizeof(*desc)) ;
  vl_covdet_extract_patch_for_frame(covdet,
                                    patch,
                                    patchResolution,
                                    patchRelativeExtent,
                                    patchRelativeSmoothing,
                                    feature[i].frame) ;
  // do something with patch
}
@endcode

This example code:

- Calls ::vl_covdet_new constructs a new detector object. @ref
  covdet.h supports a variety of different detectors (see
  ::VlCovDetMethod).
- Optionally calls various functions to set the detector parameters if
  needed (e.g. ::vl_covdet_set_peak_threshold).
- Calls ::vl_covdet_put_image to start processing a new image. It
  causes the detector to compute the scale space representation of the
  image, but does not compute the features yet.
- Calls ::vl_covdet_detect runs the detector. At this point features are
  ready to be extracted. However, one or all of the following steps
  may be executed in order to process the features further.
- Optionally calls ::vl_covdet_drop_features_outside to drop features
  outside the image boundary.
- Optionally calls ::vl_covdet_extract_affine_shape to compute the
  affine shape of features using affine adaptation.
- Optionally calls ::vl_covdet_extract_orientations to compute the
  dominant orientation of features looking for the dominant gradient
  orientation in patches.
- Optionally calls ::vl_covdet_extract_patch_for_frame to extract a
  normalized feature patch, for example to compute an invariant
  feature descriptor.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@page covdet-fundamentals Covariant detectors fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

This page describes the fundamental concepts required to understand a
covariant feature detector, the geometry of covariant features, and
the process of feature normalization.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section covdet-covariance Covariant detection
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The purpose of a *covariant detector* is to extract from an image a
set of local features in a manner which is consistent with spatial
transformations of the image itself. For instance, a covariant
detector that extracts interest points $\bx_1,\dots,\bx_n$ from image
$\ell$ extracts the translated points $\bx_1+T,\dots,\bx_n+T$ from the
translated image $\ell'(\bx) = \ell(\bx-T)$.

More in general, consider a image $\ell$ and a transformed version
$\ell' = \ell \circ w^{-1}$ of it, as in the following figure:

@image html covdet.png "Covariant detection of local features."

The transformation or <em>warp</em> $w : \real^2 \mapsto \real^2$ is a
deformation of the image domain which may capture a change of camera
viewpoint or similar imaging factor. Examples of warps typically
considered are translations, scaling, rotations, and general affine
transformations; however, in $w$ could be another type of continuous
and invertible transformation.

Given an image $\ell$, a **detector** selects features $R_1,\dots,R_n$
(one such features is shown in the example as a green circle). The
detector is said to be **covariant** with the warps $w$ if it extracts
the transformed features $w[R_1],\dots, w[R_n]$ from the transformed
image $w[\ell]$. Intuitively, this means that the &ldquo;same
features&rdquo; are extracted in both cases up to the transformation
$w$. This property is described more formally in @ref
covdet-principles.

Covariance is a key property of local feature detectors as it allows
extracting corresponding features from two or more images, making it
possible to match them in a meaningful way.

The @ref covdet.h module in VLFeat implements an array of feature
detection algorithm that have are covariant to different classes of
transformations.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section covdet-frame Feature geometry and feature frames
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

As we have seen, local features are subject to image transformations,
and they apply a fundamental role in matching and normalizing
images. To operates effectively with local features is therefore
necessary to understand their geometry.

The geometry of a local feature is captured by a <b>feature frame</b>
$R$. In VLFeat, depending on the specific detector, the frame can be
either a point, a disc, an ellipse, an oriented disc, or an oriented
ellipse.

A frame captures both the extent of the local features, useful to know
which portions of two images are put in correspondence, as well their
shape.  The latter can be used to associate to diagnose the
transformation that affects a feature and remove it through the
process of **normalization**.

More precisely, in covariant detection feature frames are constructed
to be compatible with a certain class of transformations. For example,
circles are compatible with similarity transformations as they are
closed under them. Likewise, ellipses are compatible with affine
transformations.

Beyond this closure property, the key idea here is that all feature
occurrences can be seen as transformed versions of a base or
<b>canonical</b> feature. For example, all discs $R$ can be obtained
by applying a similarity transformation to the unit disc $\bar R$
centered at the origin. $\bar R$ is an example of canonical frame
as any other disc can be written as $R = w[\bar R]$ for a suitable
similarity $w$.

@image html frame-canonical.png "The idea of canonical frame and normalization"

The equation $R = w[\bar R_0]$ matching the canonical and detected
feature frames establishes a constraint on the warp $w$, very similar
to the way two reference frames in geometry establish a transformation
between spaces. The transformation $w$ can be thought as a the
**pose** of the detected feature, a generalization of its location.

In the case of discs and similarity transformations, the equation $R =
w[\bar R_0]$ fixes $w$ up to a residual rotation. This can be
addressed by considering oriented discs instead. An **oriented disc**
is a disc with a radius highlighted to represent the feature
orientation.

While discs are appropriate for similarity transformations, they are
not closed under general affine transformations. In this case, one
should consider the more general class of (oriented) ellipses. The
following image illustrates the five types of feature frames used in
VLFeat:

@image html frame-types.png "Types of local feature frames: points, discs, oriented discs, ellipses, oriented ellipses."

Note that these frames are described respectively by 2, 3, 4, 5 and 6
parameters. The most general type are the oriented ellipses, which can
be used to represent all the other frame types as well.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section covdet-frame-transformation Transforming feature frames
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

Consider a warp $w$ mapping image $\ell$ into image $\ell'$ as in the
figure below. A feature $R$ in the first image co-variantly transform
into a feature $R'=w[R]$ in the second image:

@image html covdet-normalization.png "Normalization removes the effect of an image deformation."

The poses $u,u'$ of $R=u[R_0]$ and $R' = u'[R_0]$ are then related by
the simple expression:

\[
  u' = w \circ u.
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@section covdet-frame-normalization Normalizing feature frames
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

In the example above, the poses $u$ and $u'$ relate the two
occurrences $R$ and $R'$ of the feature to its canonical version
$R_0$. If the pose $u$ of the feature in image $\ell$ is known, the
canonical feature appearance can be computed by un-warping it:

\[
 \ell_0 = u^{-1}[\ell] = \ell \circ u.
\]

This process is known as **normalization** and is the key in the
computation of invariant feature descriptors as well as in the
construction of most co-variant detectors.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@page covdet-principles Principles of covariant detection
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The goals of a co-variant detector were discussed in @ref
covdet-fundamentals. This page introduces a few general principles
that are at the basis of most covariant detection algorithms. Consider
an input image $\ell$ and a two dimensional continuous and invertible
warp $w$. The *warped image* $w[\ell]$ is defined to be

\[
 w[\ell] = \ell \circ w^{-1},
\]

or, equivalently,

\[
 w[\ell](x,y) =  \ell(w^{-1}(x,y)), \qquad \forall (x,y)\in\real^2.
\]

Note that, while $w$ pushes pixels forward, from the original to the
transformed image domain, defining the transformed image $\ell'$
requires inverting the warp and composing $\ell$ with $w^{-1}$.

The goal a covariant detector is to extract the same local features
irregardless of image transformations. The detector is said to be
<b>covariant</b> or <b>equivariant</b> with a class of warps
$w\in\mathcal{W}$ if, when the feature $R$ is detected in image
$\ell$, then the transformed feature $w[R]$ is detected in the
transformed image $w[\ell]$.

The net effect is that a covariant feature detector appears to
&ldquo;track&rdquo; image transformations; however, it is important to
note that a detector *is not a tracker* because it processes images
individually rather than jointly as part of a sequence.

An intuitive way to construct a covariant feature detector is to
extract features in correspondence of images structures that are
easily identifiable even after a transformation. Example of specific
structures include dots, corners, and blobs. These will be generically
indicated as **corners** in the followup.

A covariant detector faces two challenges. First, corners have, in
practice, an infinite variety of individual appearances and the
detector must be able to capture them to be of general applicability.
Second, the way corners are identified and detected must remain stable
under transformations of the image. These two problems are addressed
in @ref covdet-cornerness-localmax and @ref
covdet-cornerness-normalization respectively.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-cornerness Detection using a cornerness measure
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

One way to decide whether an image region $R$ contains a corner is to
compare the local appearance to a model or template of the corner; the
result of this comparisons produces a *cornerness score* at that
location. This page describe general theoretical properties of the
cornerness and the detection process. Concrete examples of cornerness
are given in @ref covdet-corner-types.

A **cornerness measure** associate a score to all possible feature
locations in an image $\ell$. As described in @ref covdet-frame, the
location or, more in general, pose $u$ of a feature $R$ is the warp
$w$ that maps the canonical feature frame $R_0$ to $R$:

\[
    R = u[R_0].
\]

The goal of a cornerness measure is to associate a score $F(u;\ell)$
to all possible feature poses $u$ and use this score to extract a
finite number of co-variant features from any image.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-cornerness-localmax Local maxima of a cornerness measure
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Given the cornerness of each candidate feature, the detector must
extract a finite number of them. However, the cornerness of features
with nearly identical pose must be similar (otherwise the cornerness
measure would be unstable). As such, simply thresholding $F(w;\ell)$
would detect an infinite number of nearly identical features rather
than a finite number.

The solution is to detect features in correspondence of the local
maxima of the score measure:

\[
 \{w_1,\dots,w_n\} = \operatorname{localmax}_{w\in\mathcal{W}} F(w;\ell).
\]

This also means that features are never detected in isolation, but by
comparing neighborhoods of them.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-cornerness-normalization Covariant detection by normalization
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The next difficulty is to guarantee that detection is co-variant with
image transformations. Hence, if $u$ is the pose of a feature
extracted from image $\ell$, then the transformed pose $u' = w[u]$
must be detected in the transformed image $\ell' = w[\ell]$.

Since features are extracted in correspondence of the local maxima of
the cornerness score, a sufficient condition is that corresponding
features attain the same score in the two images:

\[
\forall u\in\mathcal{W}: \quad F(u;\ell) = F(w[u];w[\ell]),
\qquad\text{or}\qquad
F(u;\ell) = F(w \circ u ;\ell \circ w^{-1}).
\]

One simple way to satisfy this equation is to compute a cornerness
score *after normalizing the image* by the inverse of the candidate
feature pose warp $u$, as follows:

\[
  F(u;\ell) = F(1;u^{-1}[\ell]) = F(1;\ell \circ u) = \mathcal{F}(\ell \circ u),
\]

where $1 = u^{-1} \circ u$ is the identity transformation and
$\mathcal{F}$ is an arbitrary functional. Intuitively, co-variant
detection is obtained by looking if the appearance of the feature
resembles a corner only *after normalization*. Formally:

@f{align*}
F(w[u];w[\ell])
&=
F(w \circ u ;\ell \circ w^{-1})
\\
&=
F(1; \ell \circ w^{-1} \circ w \circ u)
\\
&=
\mathcal{F}(\ell\circ u)
\\
&=
F(u;\ell).
@f}

Concrete examples of the functional $\mathcal{F}$ are given in @ref
covdet-corner-types.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-locality Locality of the detected features
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In the definition above, the cornenress functional $\mathcal{F}$ is an
arbitrary functional of the entire normalized image $u^{-1}[\ell]$.
In practice, one is always interested in detecting **local features**
(at the very least because the image extent is finite).

This is easily obtained by considering a cornerness $\mathcal{F}$
which only looks in a small region of the normalized image, usually
corresponding to the extent of the canonical feature $R_0$ (e.g. a
unit disc centered at the origin).

In this case the extent of the local feature in the original image is
simply given by $R = u[R_0]$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-partial Partial and iterated normalization
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Practical detectors implement variants of the ideas above. Very often,
for instance, detection is an iterative process, in which successive
parameters of the pose of a feature are determined. For instance, it
is typical to first detect the location and scale of a feature using a
rotation-invariant cornerness score $\mathcal{F}$. Once these two
parameters are known, the rotation can be determined using a different
score, sensitive to the orientation of the local image structures.

Certain detectors (such as Harris-Laplace and Hessian-Laplace) use
even more sophisticated schemes, in which different scores are used to
jointly (rather than in succession) different parameters of the pose
of a feature, such as its translation and scale. While a formal
treatment of these cases is possible as well, we point to the original
papers.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page covdet-differential Differential and integral image operations
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Dealing with covariant interest point detector requires working a good
deal with derivatives, convolutions, and transformations of images.
The notation and fundamental properties of interest here are discussed
next.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-derivatives Derivative operations: gradients
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

For the derivatives, we borrow the notation of
@cite{kinghorn96integrals}. Let $f: \mathbb{R}^m \rightarrow
\mathbb{R}^n, \bx \mapsto f(\bx)$ be a vector function. The derivative
of the function with respect to $\bx$ is given by its *Jacobian
matrix* denoted by the symbol:

\[
\frac{\partial f}{\partial \bx^\top}
=
\begin{bmatrix}
  \frac{\partial f_1}{x_1} & \frac{\partial f_1}{x_2} & \dots \\
  \frac{\partial f_2}{x_1} & \frac{\partial f_2}{x_2} & \dots \\
  \vdots & \vdots & \ddots \\
\end{bmatrix}.
\]

When the function $ f $ is scalar ($n=1$), the Jacobian is the same as
the gradient of the function (or, in fact, its transpose). More
precisely, the <b>gradient</b> $\nabla f $ of $ f $ denotes the column
vector of partial derivatives:

\[
\nabla f
 = \frac{\partial f}{\partial \bx}
 =
 \begin{bmatrix}
  \frac{\partial f}{\partial x_1} \\
  \frac{\partial f}{\partial x_2} \\
  \vdots
\end{bmatrix}.
\]

The second derivative $H_f $ of a scalar function $ f $, or
<b>Hessian</b>, is denoted as

\[
H_f
= \frac{\partial f}{\partial \bx \partial \bx^\top}
= \frac{\partial \nabla f}{\partial \bx^\top}
=
\begin{bmatrix}
  \frac{\partial f}{\partial x_1 \partial x_1} & \frac{\partial f}{\partial x_1 \partial x_2} & \dots \\
  \frac{\partial f}{\partial x_2 \partial x_1} & \frac{\partial f}{\partial x_2 \partial x_2} & \dots \\
  \vdots & \vdots & \ddots \\
\end{bmatrix}.
\]

The determinant of the Hessian is also known as <b>Laplacian</b> and denoted as

\[
 \Delta f = \operatorname{det} H_f =
\frac{\partial f}{\partial x_1^2} +
\frac{\partial f}{\partial x_2^2} +
\dots
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-derivative-transformations Derivative and image warps
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In the following, we will often been interested in domain warpings $u:
\mathbb{R}^m \rightarrow \mathbb{R}^n, \bx \mapsto u(\bx)$ of a
function $f(\bar\bx) $ and its effect on the derivatives of the
function. The key transformation is the chain rule:

\[
\frac{\partial f \circ u}{\partial \bx^\top}
=
\left(\frac{\partial f}{\partial \bar\bx^\top} \circ u\right)
\frac{\partial u}{\partial \bx^\top}
\]

In particular, for an affine transformation $u = (A,T) : \bx \mapsto
A\bx + T$, one obtains the transformation rules:

\[
\begin{align*}
\frac{\partial f \circ (A,T)}{\partial \bx^\top}
&=
\left(\frac{\partial f}{\partial \bar\bx^\top} \circ (A,T)\right)A,
\\
\nabla (f \circ (A,T))
&= A^\top (\nabla f) \circ (A,T),
\\
H_{f \circ(A,T)}
&= A^\top (H_f \circ (A,T)) A,
\\
\Delta (f \circ(A,T))
&= \det(A)^2\, (\Delta f) \circ (A,T).
\end{align*}
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-smoothing Integral operations: smoothing
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

In practice, given an image $\ell$ expressed in digital format, good
derivative approximations can be computed only if the bandwidth of the
image is limited and, in particular, compatible with the sampling
density. Since it is unreasonable to expect real images to be
band-limited, the bandwidth is artificially constrained by suitably
smoothing the image prior to computing its derivatives. This is also
interpreted as a form of regularization or as a way of focusing on the
image content at a particular scale.

Formally, we will focus on Gaussian smoothing kernels. For the 2D case
$\bx\in\real^2$, the Gaussian kernel of covariance $\Sigma$ is given
by

\[
g_{\Sigma}(\bx) = \frac{1}{2\pi \sqrt{\det\Sigma}}
  \exp\left(
  - \frac{1}{2} \bx^\top \Sigma^{-1} \bx
  \right).
\]

The symbol $g_{\sigma^2}$ will be used to denote a Gaussian kernel
with isotropic standard deviation $\sigma$, i.e. $\Sigma = \sigma^2
I$. Given an image $\ell$, the symbol $\ell_\Sigma$ will be used to
denote the image smoothed by the Gaussian kernel of parameter
$\Sigma$:

\[
\ell_\Sigma(\bx) = (g_\Sigma * \ell)(\bx)
=
\int_{\real^m}
g_\Sigma(\bx - \by) \ell(\by)\,d\by.
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-smoothing-transformations Smoothing and image warps
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

One advantage of Gaussian kernels is that they are (up to
renormalization) closed under a linear warp:

\[
 |A|\, g_\Sigma \circ A = g_{A^{-1} \Sigma A^{-\top}}
\]

This also means that smoothing a warped image is the same as warping
the result of smoothing the original image by a suitably adjusted
Gaussian kernel:

\[
g_{\Sigma} * (\ell \circ (A,T))
=
(g_{A\Sigma A^\top} * \ell) \circ (A,T).
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page covdet-corner-types Cornerness measures
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The goal of a cornerness measure (@ref covdet-cornerness) is to
associate to an image patch a score proportional to how strongly the
patch contain a certain strucuture, for example a corner or a
blob. This page reviews the most important cornerness measures as
implemented in VLFeat:

- @ref covdet-harris
- @ref covdet-laplacian
- @ref covdet-hessian

This page makes use of notation introduced in @ref
covdet-differential.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-harris Harris corners
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This section introduces the fist of the cornerness measure
$\mathcal{F}[\ell]$. Recall (@ref covdet-cornerness) that the goal of
this functional is to respond strongly to images $\ell$ of corner-like
structure.

Rather than explicitly encoding the appearance of corners, the idea of
the Harris measure is to label as corner *any* image patch whose
appearance is sufficiently distinctive to allow accurate
localization. In particular, consider an image patch $\ell(\bx),
\bx\in\Omega$, where $\Omega$ is a smooth circular window of radius
approximately $\sigma_i$; at necessary condition for the patch to
allow accurate localization is that even a small translation
$\ell(\bx+\delta)$ causes the appearance to vary significantly (if not
the origin and location $\delta$ would not be distinguishable from the
image alone). This variation is measured by the sum of squared
differences

\[
E(\delta) = \int g_{\sigma_i^2}(\bx)
(\ell_{\sigma_d^2}(\bx+\delta) -
 \ell_{\sigma_d^2}(\bx))^2 \,d\bx
\]

Note that images are compared at scale $\sigma_d$, known as
 *differentiation scale* for reasons that will be clear in a moment,
and that the squared differences are summed over a window softly
defined by $\sigma_i$, also known as *integration scale*. This
function can be approximated as $E(\delta)\approx \delta^\top
M[\ell;\sigma_i^2,\sigma_d^2] \delta$ where

\[
  M[\ell;\sigma_i^2,\sigma_d^2]
= \int  g_{\sigma_i^2}(\bx)
 (\nabla \ell_{\sigma_d^2}(\bx))
 (\nabla \ell_{\sigma_d^2}(\bx))^\top \, d\bx.
\]

is the so called **structure tensor**.

A corner is identified when the sum of squared differences $E(\delta)$
is large for displacements $\delta$ in all directions. This condition
is obtained when both the eignenvalues $\lambda_1,\lambda_2$ of the
structure tensor $M$ are large. The **Harris cornerness measure**
captures this fact:

\[
 \operatorname{Harris}[\ell;\sigma_i^2,\sigma_d^2] =
 \det M - \kappa \operatorname{trace}^2 M =
 \lambda_1\lambda_2 - \kappa (\lambda_1+\lambda_2)^2
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-harris-warped Harris in the warped domain
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The cornerness measure of a feature a location $u$ (recall that
locations $u$ are in general defined as image warps) should be
computed after normalizing the image (by applying to it the warp
$u^{-1}$). This section shows that, for affine warps, the Harris
cornerness measure can be computed directly in the Gaussian affine
scale space of the image. In particular, for similarities, it can be
computed in the standard Gaussian scale space.

To this end, let $u=(A,T)$ be an affine warp identifying a feature
location in image $\ell(\bx)$. Let $\bar\ell(\bar\bx) =
\ell(A\bar\bx+T)$ be the normalized image and rewrite the structure
tensor of the normalized image as follows:

\[
 M[\bar\ell; \bar\Sigma_i, \bar\Sigma_d]
=
 M[\bar\ell; \bar\Sigma_i, \bar\Sigma_d](\mathbf{0})
=
\left[
g_{\bar\Sigma_i} *
(\nabla\bar\ell_{\bar\Sigma_d})
(\nabla\bar\ell_{\bar\Sigma_d})^\top
\right](\mathbf{0})
\]

This notation emphasizes that the structure tensor is obtained by
taking derivatives and convolutions of the image. Using the fact that
$\nabla g_{\bar\Sigma_d} * \bar\ell = A^\top (\nabla g_{A\bar\Sigma
A^\top} * \ell) \circ (A,T)$ and that $g_{\bar\Sigma} * \bar \ell =
(g_{A\bar\Sigma A^\top} * \ell) \circ (A,T)$, we get the equivalent
expression:

\[
 M[\bar\ell; \bar\Sigma_i, \bar\Sigma_d](\mathbf{0})
 =
A^\top
\left[
g_{A\bar\Sigma_i A^\top} *
(\nabla\ell_{A\bar\Sigma_dA^\top})(\nabla\ell_{A\bar\Sigma_d A^\top})^\top
\right](A\mathbf{0}+T)
A.
\]

In other words, the structure tensor of the normalized image can be
computed as:

\[
M[\bar\ell; \bar\Sigma_i, \bar\Sigma_d](\mathbf{0})
=
A^\top M[\ell; \Sigma_i, \Sigma_d](T) A,
\quad
\Sigma_{i} = A\bar\Sigma_{i}A^\top,
\quad
\Sigma_{d} = A\bar\Sigma_{d}A^\top.
\]

This equation allows to compute the structure tensor for feature at
all locations directly in the original image. In particular, features
at all translations $T$ can be evaluated efficiently by computing
convolutions and derivatives of the image
$\ell_{A\bar\Sigma_dA^\top}$.

A case of particular instance is when $\bar\Sigma_i= \bar\sigma_i^2 I$
and $\bar\Sigma_d = \bar\sigma_d^2$ are both isotropic covariance
matrices and the affine transformation is a similarity $A=sR$.  Using
the fact that $\det\left( s^2 R^\top M R \right)= s^4 \det M$ and
$\operatorname{tr}\left(s^2 R^\top M R\right) = s^2 \operatorname{tr}
M$, one obtains the relation

\[
 \operatorname{Harris}[\bar \ell;\bar\sigma_i^2,\bar\sigma_d^2] =
 s^4 \operatorname{Harris}[\ell;s^2\bar\sigma_i^2,s^2\bar\sigma_d^2](T).
\]

This equation indicates that, for similarity transformations, not only
the structure tensor, but directly the Harris cornerness measure can
be computed on the original image and then be transferred back to the
normalized domain. Note, however, that this requires rescaling the
measure by the factor $s^4$.

Another important consequence of this relation is that the Harris
measure is invariant to pure image rotations. It cannot, therefore, be
used to associate an orientation to the detected features.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-hessian Hessian blobs
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The *(determinant of the) Hessian* cornerness measure is given
determinant of the Hessian of the image:

\[
 \operatorname{DetHess}[\ell;\sigma_d^2]
 =
 \det H_{g_{\sigma_d^2} * \ell}(\mathbf{0})
\]

This number is large and positive if the image is locally curved
(peaked), roughly corresponding to blob-like structures in the image.
In particular, a large score requires the product of the eigenvalues
of the Hessian to be large, which requires both of them to have the
same sign and are large in absolute value.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection covdet-hessian-warped Hessian in the warped domain
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Similarly to the Harris measure, it is possible to work with the
Hessian measure on the original unnormalized image. As before, let
$\bar\ell(\bar\bx) = \ell(A\bar\bx+T)$ be the normalized image and
rewrite the Hessian of the normalized image as follows:

\[
H_{g_{\bar\Sigma_d} * \bar\ell}(\mathbf{0}) = A^\top \left(H_{g_{\Sigma_d} * \ell}(T)\right) A.
\]

Then

\[
 \operatorname{DetHess}[\bar\ell;\bar\Sigma_d]
 =
 (\det A)^2 \operatorname{DetHess}[\ell;A\bar\Sigma_d A^\top](T).
\]

In particular, for isotropic covariance matrices and similarity
transformations $A=sR$:

\[
 \operatorname{DetHess}[\bar\ell;\bar\sigma_d^2]
 =
 s^4 \operatorname{DetHess}[\ell;s^2\bar\sigma_d^2](T)
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section covdet-laplacian Laplacian and Difference of Gaussians blobs
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The **Laplacian of Gaussian (LoG)** or **trace of the Hessian**
cornerness measure is given by the trace of the Hessian of the image:

\[
 \operatorname{Lap}[\ell;\sigma_d^2]
 =
 \operatorname{tr} H_{g_{\sigma_d}^2 * \ell}
\]

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection covdet-laplacian-warped Laplacian in the warped domain
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

Similarly to the Hessian measure, the Laplacian cornenress can often
be efficiently computed for features at all locations in the original
unnormalized image domain. In particular, if the derivative covariance
matrix $\Sigma_d$ is isotropic and one considers as warpings
similarity transformations $A=sR$, where $R$ is a rotatin and $s$ a
rescaling, one has

\[
 \operatorname{Lap}[\bar\ell;\bar\sigma_d^2]
 =
 s^2 \operatorname{Lap}[\ell;s^2\bar\sigma_d^2](T)
\]

Note that, comparing to the Harris and determinant of Hessian
measures, the scaling for the Laplacian is $s^2$ rather than $s^4$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection covdet-laplacian-matched Laplacian as a matched filter
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The Laplacian is given by the trace of the Hessian
operator. Differently from the determinant of the Hessian, this is a
linear operation. This means that computing the Laplacian cornerness
measure can be seen as applying a linear filtering operator to the
image. This filter can then be interpreted as a *template* of a corner
being matched to the image. Hence, the Laplacian cornerness measure
can be interpreted as matching this corner template at all possible
image locations.

To see this formally, compute the Laplacian score in the input image domain:

\[
 \operatorname{Lap}[\bar\ell;\bar\sigma_d^2]
 =
 s^2 \operatorname{Lap}[\ell;s^2\bar\sigma_d^2](T)
 =
 s^2 (\Delta g_{s^2\bar\sigma_d^2} * \ell)(T)
\]

The Laplacian fitler is obtained by moving the Laplacian operator from
the image to the Gaussian smoothing kernel:

\[
 s^2 (\Delta g_{s^2\bar\sigma_d^2} * \ell)
=
 (s^2 \Delta g_{s^2\bar\sigma_d^2}) * \ell
\]

Note that the filter is rescaled by the $s^2$; sometimes, this factor
is incorporated in the Laplacian operator, yielding the so-called
normalized Laplacian.

The Laplacian of Gaussian is also called *top-hat function* and has
the expression:

\[
\Delta g_{\sigma^2}(x,y)
=
\frac{x^2+y^2 - 2 \sigma^2}{\sigma^4} g_{\sigma^2}(x,y).
\]

This filter, which acts as corner template, resembles a blob (a dark
disk surrounded by a bright ring).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
@subsection covdet-laplacian-dog Difference of Gaussians
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

The **Difference of Gaussian** (DoG) cornerness measure can be
interpreted as an approximation of the Laplacian that is easy to
obtain once a scalespace of the input image has been computed.

As noted above, the Laplacian cornerness of the normalized feature can
be computed directly from the input image by convolving the image by
the normalized Laplacian of Gaussian filter $s^2 \Delta
g_{s^2\bar\sigma_d^2}$.

Like the other derivative operators, this filter is simpe to
discriteize. However, it is often approximated by computing the the
*Difference of Gaussians* (DoG) approximation instead. This
approximation is obtained from the easily-proved identity:

\[
  \frac{\partial}{\partial \sigma} g_{\sigma^2} =
  \sigma \Delta g_{\sigma^2}.
\]

This indicates that computing the normalized Laplacian of a Gaussian
filter is, in the limit, the same as taking the difference between
Gaussian filters of slightly increasing standard deviation $\sigma$
and $\kappa\sigma$, where $\kappa \approx 1$:

\[
\sigma^2 \Delta g_{\sigma^2}
\approx
\sigma \frac{g_{(\kappa\sigma)^2} - g_{\sigma^2}}{\kappa\sigma - \sigma}
=
\frac{1}{\kappa - 1}
(g_{(\kappa\sigma)^2} - g_{\sigma^2}).
\]

One nice propery of this expression is that the factor $\sigma$
cancels out in the right-hand side. Usually, scales $\sigma$ and
$\kappa\sigma$ are pre-computed in the image scale-space and
successive scales are sampled with uniform geometric spacing, meaning
that the factor $\kappa$ is the same for all scales. Then, up to a
overall scaling factor, the LoG cornerness measure can be obtained by
taking the difference of successive scale space images
$\ell_{(\kappa\sigma)^2}$ and $\ell_{\sigma^2}$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page covdet-affine-adaptation Affine adaptation
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page covdet-dominant-orientation Dominant orientation
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
**/

#include "covdet.h"
#include <string.h>

/** @brief Reallocate buffer
 ** @param buffer
 ** @param bufferSize
 ** @param targetSize
 ** @return error code
 **/

static int
_vl_resize_buffer (void ** buffer, vl_size * bufferSize, vl_size targetSize) {
  void * newBuffer ;
  if (*buffer == NULL) {
    *buffer = vl_malloc(targetSize) ;
    if (*buffer) {
      *bufferSize = targetSize ;
      return VL_ERR_OK ;
    } else {
      *bufferSize = 0 ;
      return VL_ERR_ALLOC ;
    }
  }
  newBuffer = vl_realloc(*buffer, targetSize) ;
  if (newBuffer) {
    *buffer = newBuffer ;
    *bufferSize = targetSize ;
    return VL_ERR_OK ;
  } else {
    return VL_ERR_ALLOC ;
  }
}

/** @brief Enlarge buffer
 ** @param buffer
 ** @param bufferSize
 ** @param targetSize
 ** @return error code
 **/

static int
_vl_enlarge_buffer (void ** buffer, vl_size * bufferSize, vl_size targetSize) {
  if (*bufferSize >= targetSize) return VL_ERR_OK ;
  return _vl_resize_buffer(buffer,bufferSize,targetSize) ;
}

/* ---------------------------------------------------------------- */
/*                                            Finding local extrema */
/* ---------------------------------------------------------------- */

/* Todo: make this generally available in the library */

typedef struct _VlCovDetExtremum2
{
  vl_index xi ;
  vl_index yi ;
  float x ;
  float y ;
  float peakScore ;
  float edgeScore ;
} VlCovDetExtremum2 ;

typedef struct _VlCovDetExtremum3
{
  vl_index xi ;
  vl_index yi ;
  vl_index zi ;
  float x ;
  float y ;
  float z ;
  float peakScore ;
  float edgeScore ;
} VlCovDetExtremum3 ;

VL_EXPORT vl_size
vl_find_local_extrema_3 (vl_index ** extrema, vl_size * bufferSize,
                         float const * map,
                         vl_size width, vl_size height, vl_size depth,
                         double threshold) ;

VL_EXPORT vl_size
vl_find_local_extrema_2 (vl_index ** extrema, vl_size * bufferSize,
                         float const * map,
                         vl_size width, vl_size height,
                         double threshold) ;

VL_EXPORT vl_bool
vl_refine_local_extreum_3 (VlCovDetExtremum3 * refined,
                           float const * map,
                           vl_size width, vl_size height, vl_size depth,
                           vl_index x, vl_index y, vl_index z) ;

VL_EXPORT vl_bool
vl_refine_local_extreum_2 (VlCovDetExtremum2 * refined,
                           float const * map,
                           vl_size width, vl_size height,
                           vl_index x, vl_index y) ;

/** @internal
 ** @brief Find the extrema of a 3D function
 ** @param extrema buffer containing the extrema found (in/out).
 ** @param bufferSize size of the @a extrema buffer in bytes (in/out).
 ** @param map a 3D array representing the map.
 ** @param width of the map.
 ** @param height of the map.
 ** @param depth of the map.
 ** @param threshold minumum extremum value.
 ** @return number of extrema found.
 ** @see @ref ::vl_refine_local_extreum_2.
 **/

vl_size
vl_find_local_extrema_3 (vl_index ** extrema, vl_size * bufferSize,
                         float const * map,
                         vl_size width, vl_size height, vl_size depth,
                         double threshold)
{
  vl_index x, y, z ;
  vl_size const xo = 1 ;
  vl_size const yo = width ;
  vl_size const zo = width * height ;
  float const *pt = map + xo + yo + zo ;

  vl_size numExtrema = 0 ;
  vl_size requiredSize = 0 ;

#define CHECK_NEIGHBORS_3(v,CMP,SGN)     (\
v CMP ## = SGN threshold &&               \
v CMP *(pt + xo) &&                       \
v CMP *(pt - xo) &&                       \
v CMP *(pt + zo) &&                       \
v CMP *(pt - zo) &&                       \
v CMP *(pt + yo) &&                       \
v CMP *(pt - yo) &&                       \
\
v CMP *(pt + yo + xo) &&                  \
v CMP *(pt + yo - xo) &&                  \
v CMP *(pt - yo + xo) &&                  \
v CMP *(pt - yo - xo) &&                  \
\
v CMP *(pt + xo      + zo) &&             \
v CMP *(pt - xo      + zo) &&             \
v CMP *(pt + yo      + zo) &&             \
v CMP *(pt - yo      + zo) &&             \
v CMP *(pt + yo + xo + zo) &&             \
v CMP *(pt + yo - xo + zo) &&             \
v CMP *(pt - yo + xo + zo) &&             \
v CMP *(pt - yo - xo + zo) &&             \
\
v CMP *(pt + xo      - zo) &&             \
v CMP *(pt - xo      - zo) &&             \
v CMP *(pt + yo      - zo) &&             \
v CMP *(pt - yo      - zo) &&             \
v CMP *(pt + yo + xo - zo) &&             \
v CMP *(pt + yo - xo - zo) &&             \
v CMP *(pt - yo + xo - zo) &&             \
v CMP *(pt - yo - xo - zo) )

  for (z = 1 ; z < (signed)depth - 1 ; ++z) {
    for (y = 1 ; y < (signed)height - 1 ; ++y) {
      for (x = 1 ; x < (signed)width - 1 ; ++x) {
        float value = *pt ;
        if (CHECK_NEIGHBORS_3(value,>,+) || CHECK_NEIGHBORS_3(value,<,-)) {
          numExtrema ++ ;
          requiredSize += sizeof(vl_index) * 3 ;
          if (*bufferSize < requiredSize) {
            int err = _vl_resize_buffer((void**)extrema, bufferSize,
                                        requiredSize + 2000 * 3 * sizeof(vl_index)) ;
            if (err != VL_ERR_OK) abort() ;
          }
          (*extrema) [3 * (numExtrema - 1) + 0] = x ;
          (*extrema) [3 * (numExtrema - 1) + 1] = y ;
          (*extrema) [3 * (numExtrema - 1) + 2] = z ;
        }
        pt += xo ;
      }
      pt += 2*xo ;
    }
    pt += 2*yo ;
  }
  return numExtrema ;
}

/** @internal
 ** @brief Find extrema in a 2D function
 ** @param extrema buffer containing the found extrema (in/out).
 ** @param bufferSize size of the @a extrema buffer in bytes (in/out).
 ** @param map a 3D array representing the map.
 ** @param width of the map.
 ** @param height of the map.
 ** @param threshold minumum extremum value.
 ** @return number of extrema found.
 **
 ** An extremum contains 2 ::vl_index values; they are arranged
 ** sequentially.
 **
 ** The function can reuse an already allocated buffer if
 ** @a extrema and @a bufferSize are initialized on input.
 ** It may have to @a realloc the memory if the buffer is too small.
 **/

vl_size
vl_find_local_extrema_2 (vl_index ** extrema, vl_size * bufferSize,
                         float const* map,
                         vl_size width, vl_size height,
                         double threshold)
{
  vl_index x, y ;
  vl_size const xo = 1 ;
  vl_size const yo = width ;
  float const *pt = map + xo + yo ;

  vl_size numExtrema = 0 ;
  vl_size requiredSize = 0 ;
#define CHECK_NEIGHBORS_2(v,CMP,SGN)     (\
v CMP ## = SGN threshold &&               \
v CMP *(pt + xo) &&                       \
v CMP *(pt - xo) &&                       \
v CMP *(pt + yo) &&                       \
v CMP *(pt - yo) &&                       \
\
v CMP *(pt + yo + xo) &&                  \
v CMP *(pt + yo - xo) &&                  \
v CMP *(pt - yo + xo) &&                  \
v CMP *(pt - yo - xo) )

  for (y = 1 ; y < (signed)height - 1 ; ++y) {
    for (x = 1 ; x < (signed)width - 1 ; ++x) {
      float value = *pt ;
      if (CHECK_NEIGHBORS_2(value,>,+) || CHECK_NEIGHBORS_2(value,<,-)) {
        numExtrema ++ ;
        requiredSize += sizeof(vl_index) * 2 ;
        if (*bufferSize < requiredSize) {
          int err = _vl_resize_buffer((void**)extrema, bufferSize,
                                      requiredSize + 2000 * 2 * sizeof(vl_index)) ;
          if (err != VL_ERR_OK) abort() ;
        }
        (*extrema) [2 * (numExtrema - 1) + 0] = x ;
        (*extrema) [2 * (numExtrema - 1) + 1] = y ;
      }
      pt += xo ;
    }
    pt += 2*xo ;
  }
  return numExtrema ;
}

/** @internal
 ** @brief Refine the location of a local extremum of a 3D map
 ** @param refined refined extremum (out).
 ** @param map a 3D array representing the map.
 ** @param width of the map.
 ** @param height of the map.
 ** @param depth of the map.
 ** @param x initial x position.
 ** @param y initial y position.
 ** @param z initial z position.
 ** @return a flat indicating whether the extrema refinement was stable.
 **/

VL_EXPORT vl_bool
vl_refine_local_extreum_3 (VlCovDetExtremum3 * refined,
                           float const * map,
                           vl_size width, vl_size height, vl_size depth,
                           vl_index x, vl_index y, vl_index z)
{
  vl_size const xo = 1 ;
  vl_size const yo = width ;
  vl_size const zo = width * height ;

  double Dx=0,Dy=0,Dz=0,Dxx=0,Dyy=0,Dzz=0,Dxy=0,Dxz=0,Dyz=0 ;
  double A [3*3], b [3] ;

#define at(dx,dy,dz) (*(pt + (dx)*xo + (dy)*yo + (dz)*zo))
#define Aat(i,j) (A[(i)+(j)*3])

  float const * pt ;
  vl_index dx = 0 ;
  vl_index dy = 0 ;
  /*vl_index dz = 0 ;*/
  vl_index iter ;
  int err ;

  assert (map) ;
  assert (1 <= x && x <= (signed)width - 2) ;
  assert (1 <= y && y <= (signed)height - 2) ;
  assert (1 <= z && z <= (signed)depth - 2) ;

  for (iter = 0 ; iter < 5 ; ++iter) {
    x += dx ;
    y += dy ;
    pt = map + x*xo + y*yo + z*zo ;

    /* compute the gradient */
    Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
    Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
    Dz = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;

    /* compute the Hessian */
    Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
    Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
    Dzz = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;

    Dxy = 0.25 * (at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0)) ;
    Dxz = 0.25 * (at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1)) ;
    Dyz = 0.25 * (at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1)) ;

    /* solve linear system */
    Aat(0,0) = Dxx ;
    Aat(1,1) = Dyy ;
    Aat(2,2) = Dzz ;
    Aat(0,1) = Aat(1,0) = Dxy ;
    Aat(0,2) = Aat(2,0) = Dxz ;
    Aat(1,2) = Aat(2,1) = Dyz ;

    b[0] = - Dx ;
    b[1] = - Dy ;
    b[2] = - Dz ;

    err = vl_solve_linear_system_3(b, A, b) ;

    if (err != VL_ERR_OK) {
      b[0] = 0 ;
      b[1] = 0 ;
      b[2] = 0 ;
      break ;
    }

    /* Keep going if there is sufficient translation */

    dx = (b[0] > 0.6 && x < (signed)width - 2 ?  1 : 0)
    + (b[0] < -0.6 && x > 1 ? -1 : 0) ;

    dy = (b[1] > 0.6 && y < (signed)height - 2 ?  1 : 0)
    + (b[1] < -0.6 && y > 1 ? -1 : 0) ;

    if (dx == 0 && dy == 0) break ;
  }

  /* check threshold and other conditions */
  {
    double peakScore = at(0,0,0)
    + 0.5 * (Dx * b[0] + Dy * b[1] + Dz * b[2]) ;
    double alpha = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
    double edgeScore ;

    if (alpha < 0) {
      /* not an extremum */
      edgeScore = VL_INFINITY_D ;
    } else {
      edgeScore = (0.5*alpha - 1) + sqrt(VL_MAX(0.25*alpha - 1,0)*alpha) ;
    }

    refined->xi = x ;
    refined->yi = y ;
    refined->zi = z ;
    refined->x = x + b[0] ;
    refined->y = y + b[1] ;
    refined->z = z + b[2] ;
    refined->peakScore = peakScore ;
    refined->edgeScore = edgeScore ;

    return
    err == VL_ERR_OK &&
    vl_abs_d(b[0]) < 1.5 &&
    vl_abs_d(b[1]) < 1.5 &&
    vl_abs_d(b[2]) < 1.5 &&
    0 <= refined->x && refined->x <= (signed)width - 1 &&
    0 <= refined->y && refined->y <= (signed)height - 1 &&
    0 <= refined->z && refined->z <= (signed)depth - 1 ;
  }
#undef Aat
#undef at
}

/** @internal
 ** @brief Refine the location of a local extremum of a 2D map
 ** @param refined refined extremum (out).
 ** @param map a 2D array representing the map.
 ** @param width of the map.
 ** @param height of the map.
 ** @param x initial x position.
 ** @param y initial y position.
 ** @return a flat indicating whether the extrema refinement was stable.
 **/

VL_EXPORT vl_bool
vl_refine_local_extreum_2 (VlCovDetExtremum2 * refined,
                           float const * map,
                           vl_size width, vl_size height,
                           vl_index x, vl_index y)
{
  vl_size const xo = 1 ;
  vl_size const yo = width ;

  double Dx=0,Dy=0,Dxx=0,Dyy=0,Dxy=0;
  double A [2*2], b [2] ;

#define at(dx,dy) (*(pt + (dx)*xo + (dy)*yo ))
#define Aat(i,j) (A[(i)+(j)*2])

  float const * pt ;
  vl_index dx = 0 ;
  vl_index dy = 0 ;
  vl_index iter ;
  int err ;

  assert (map) ;
  assert (1 <= x && x <= (signed)width - 2) ;
  assert (1 <= y && y <= (signed)height - 2) ;

  for (iter = 0 ; iter < 5 ; ++iter) {
    x += dx ;
    y += dy ;
    pt = map + x*xo + y*yo  ;

    /* compute the gradient */
    Dx = 0.5 * (at(+1,0) - at(-1,0)) ;
    Dy = 0.5 * (at(0,+1) - at(0,-1));

    /* compute the Hessian */
    Dxx = (at(+1,0) + at(-1,0) - 2.0 * at(0,0)) ;
    Dyy = (at(0,+1) + at(0,-1) - 2.0 * at(0,0)) ;
    Dxy = 0.25 * (at(+1,+1) + at(-1,-1) - at(-1,+1) - at(+1,-1)) ;

    /* solve linear system */
    Aat(0,0) = Dxx ;
    Aat(1,1) = Dyy ;
    Aat(0,1) = Aat(1,0) = Dxy ;

    b[0] = - Dx ;
    b[1] = - Dy ;

    err = vl_solve_linear_system_2(b, A, b) ;

    if (err != VL_ERR_OK) {
      b[0] = 0 ;
      b[1] = 0 ;
      break ;
    }

    /* Keep going if there is sufficient translation */

    dx = (b[0] > 0.6 && x < (signed)width - 2 ?  1 : 0)
    + (b[0] < -0.6 && x > 1 ? -1 : 0) ;

    dy = (b[1] > 0.6 && y < (signed)height - 2 ?  1 : 0)
    + (b[1] < -0.6 && y > 1 ? -1 : 0) ;

    if (dx == 0 && dy == 0) break ;
  }

  /* check threshold and other conditions */
  {
    double peakScore = at(0,0) + 0.5 * (Dx * b[0] + Dy * b[1]) ;
    double alpha = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ;
    double edgeScore ;

    if (alpha < 0) {
      /* not an extremum */
      edgeScore = VL_INFINITY_D ;
    } else {
      edgeScore = (0.5*alpha - 1) + sqrt(VL_MAX(0.25*alpha - 1,0)*alpha) ;
    }

    refined->xi = x ;
    refined->yi = y ;
    refined->x = x + b[0] ;
    refined->y = y + b[1] ;
    refined->peakScore = peakScore ;
    refined->edgeScore = edgeScore ;

    return
    err == VL_ERR_OK &&
    vl_abs_d(b[0]) < 1.5 &&
    vl_abs_d(b[1]) < 1.5 &&
    0 <= refined->x && refined->x <= (signed)width - 1 &&
    0 <= refined->y && refined->y <= (signed)height - 1 ;
  }
#undef Aat
#undef at
}

/* ---------------------------------------------------------------- */
/*                                                Covarant detector */
/* ---------------------------------------------------------------- */

#define VL_COVDET_MAX_NUM_ORIENTATIONS 4
#define VL_COVDET_MAX_NUM_LAPLACIAN_SCALES 4
#define VL_COVDET_AA_PATCH_RESOLUTION 20
#define VL_COVDET_AA_MAX_NUM_ITERATIONS 15
#define VL_COVDET_OR_NUM_ORIENTATION_HISTOGAM_BINS 36
#define VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA 3
#define VL_COVDET_AA_RELATIVE_DERIVATIVE_SIGMA 1
#define VL_COVDET_AA_MAX_ANISOTROPY 5
#define VL_COVDET_AA_CONVERGENCE_THRESHOLD 1.001
#define VL_COVDET_AA_ACCURATE_SMOOTHING VL_FALSE
#define VL_COVDET_AA_PATCH_EXTENT (3*VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA)
#define VL_COVDET_OR_ADDITIONAL_PEAKS_RELATIVE_SIZE 0.8
#define VL_COVDET_LAP_NUM_LEVELS 10
#define VL_COVDET_LAP_PATCH_RESOLUTION 16
#define VL_COVDET_LAP_DEF_PEAK_THRESHOLD 0.01
#define VL_COVDET_DOG_DEF_PEAK_THRESHOLD VL_COVDET_LAP_DEF_PEAK_THRESHOLD
#define VL_COVDET_DOG_DEF_EDGE_THRESHOLD 10.0
#define VL_COVDET_HARRIS_DEF_PEAK_THRESHOLD 0.000002
#define VL_COVDET_HARRIS_DEF_EDGE_THRESHOLD 10.0
#define VL_COVDET_HESSIAN_DEF_PEAK_THRESHOLD 0.003
#define VL_COVDET_HESSIAN_DEF_EDGE_THRESHOLD 10.0

/** @brief Covariant feature detector */
struct _VlCovDet
{
  VlScaleSpace *gss ;        /**< Gaussian scale space. */
  VlScaleSpace *css ;        /**< Cornerness scale space. */
  VlCovDetMethod method ;    /**< feature extraction method. */
  double peakThreshold ;     /**< peak threshold. */
  double edgeThreshold ;     /**< edge threshold. */
  double lapPeakThreshold;   /**< peak threshold for Laplacian scale selection. */
  vl_size octaveResolution ; /**< resolution of each octave. */
  vl_index firstOctave ;     /**< index of the first octave. */

  double nonExtremaSuppression ;
  vl_size numNonExtremaSuppressed ;

  VlCovDetFeature *features ;
  vl_size numFeatures ;
  vl_size numFeatureBufferSize ;

  float * patch ;
  vl_size patchBufferSize ;

  vl_bool transposed ;
  VlCovDetFeatureOrientation orientations [VL_COVDET_MAX_NUM_ORIENTATIONS] ;
  VlCovDetFeatureLaplacianScale scales [VL_COVDET_MAX_NUM_LAPLACIAN_SCALES] ;

  vl_bool aaAccurateSmoothing ;
  float aaPatch [(2*VL_COVDET_AA_PATCH_RESOLUTION+1)*(2*VL_COVDET_AA_PATCH_RESOLUTION+1)] ;
  float aaPatchX [(2*VL_COVDET_AA_PATCH_RESOLUTION+1)*(2*VL_COVDET_AA_PATCH_RESOLUTION+1)] ;
  float aaPatchY [(2*VL_COVDET_AA_PATCH_RESOLUTION+1)*(2*VL_COVDET_AA_PATCH_RESOLUTION+1)] ;
  float aaMask [(2*VL_COVDET_AA_PATCH_RESOLUTION+1)*(2*VL_COVDET_AA_PATCH_RESOLUTION+1)] ;

  float lapPatch [(2*VL_COVDET_LAP_PATCH_RESOLUTION+1)*(2*VL_COVDET_LAP_PATCH_RESOLUTION+1)] ;
  float laplacians [(2*VL_COVDET_LAP_PATCH_RESOLUTION+1)*(2*VL_COVDET_LAP_PATCH_RESOLUTION+1)*VL_COVDET_LAP_NUM_LEVELS] ;
  vl_size numFeaturesWithNumScales [VL_COVDET_MAX_NUM_LAPLACIAN_SCALES + 1] ;
}  ;

VlEnumerator vlCovdetMethods [VL_COVDET_METHOD_NUM] = {
  {"DoG" ,              (vl_index)VL_COVDET_METHOD_DOG               },
  {"Hessian",           (vl_index)VL_COVDET_METHOD_HESSIAN           },
  {"HessianLaplace",    (vl_index)VL_COVDET_METHOD_HESSIAN_LAPLACE   },
  {"HarrisLaplace",     (vl_index)VL_COVDET_METHOD_HARRIS_LAPLACE    },
  {"MultiscaleHessian", (vl_index)VL_COVDET_METHOD_MULTISCALE_HESSIAN},
  {"MultiscaleHarris",  (vl_index)VL_COVDET_METHOD_MULTISCALE_HARRIS },
  {0,                   0                                            }
} ;

/** @brief Create a new object instance
 ** @param method method for covariant feature detection.
 ** @return new covariant detector.
 **/

VlCovDet *
vl_covdet_new (VlCovDetMethod method)
{
  VlCovDet * self = vl_calloc(sizeof(VlCovDet),1) ;
  self->method = method ;
  self->octaveResolution = 3 ;
  self->firstOctave = -1 ;
  switch (self->method) {
    case VL_COVDET_METHOD_DOG :
      self->peakThreshold = VL_COVDET_DOG_DEF_PEAK_THRESHOLD ;
      self->edgeThreshold = VL_COVDET_DOG_DEF_EDGE_THRESHOLD ;
      self->lapPeakThreshold = 0  ; /* not used */
      break ;
    case VL_COVDET_METHOD_HARRIS_LAPLACE:
    case VL_COVDET_METHOD_MULTISCALE_HARRIS:
      self->peakThreshold = VL_COVDET_HARRIS_DEF_PEAK_THRESHOLD ;
      self->edgeThreshold = VL_COVDET_HARRIS_DEF_EDGE_THRESHOLD ;
      self->lapPeakThreshold = VL_COVDET_LAP_DEF_PEAK_THRESHOLD ;
      break ;
    case VL_COVDET_METHOD_HESSIAN :
    case VL_COVDET_METHOD_HESSIAN_LAPLACE:
    case VL_COVDET_METHOD_MULTISCALE_HESSIAN:
      self->peakThreshold = VL_COVDET_HESSIAN_DEF_PEAK_THRESHOLD ;
      self->edgeThreshold = VL_COVDET_HESSIAN_DEF_EDGE_THRESHOLD ;
      self->lapPeakThreshold = VL_COVDET_LAP_DEF_PEAK_THRESHOLD ;
      break;
    default:
      assert(0) ;
  }

  self->nonExtremaSuppression = 0.5 ;
  self->features = NULL ;
  self->numFeatures = 0 ;
  self->numFeatureBufferSize = 0 ;
  self->patch = NULL ;
  self->patchBufferSize = 0 ;
  self->transposed = VL_FALSE ;
  self->aaAccurateSmoothing = VL_COVDET_AA_ACCURATE_SMOOTHING ;

  {
    vl_index const w = VL_COVDET_AA_PATCH_RESOLUTION ;
    vl_index i,j ;
    double step = (2.0 * VL_COVDET_AA_PATCH_EXTENT) / (2*w+1) ;
    double sigma = VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA ;
    for (j = -w ; j <= w ; ++j) {
      for (i = -w ; i <= w ; ++i) {
        double dx = i*step/sigma ;
        double dy = j*step/sigma ;
        self->aaMask[(i+w) + (2*w+1)*(j+w)] = exp(-0.5*(dx*dx+dy*dy)) ;
      }
    }
  }

  {
    /*
     Covers one octave of Laplacian filters, from sigma=1 to sigma=2.
     The spatial sampling step is 0.5.
     */
    vl_index s ;
    for (s = 0 ; s < VL_COVDET_LAP_NUM_LEVELS ; ++s) {
      double sigmaLap = pow(2.0, -0.5 +
                            (double)s / (VL_COVDET_LAP_NUM_LEVELS - 1)) ;
      double const sigmaImage = 1.0 / sqrt(2.0) ;
      double const step = 0.5 * sigmaImage ;
      double const sigmaDelta = sqrt(sigmaLap*sigmaLap - sigmaImage*sigmaImage) ;
      vl_size const w = VL_COVDET_LAP_PATCH_RESOLUTION ;
      vl_size const num = 2 * w + 1  ;
      float * pt = self->laplacians + s * (num * num) ;

      memset(pt, 0, num * num * sizeof(float)) ;

#define at(x,y) pt[(x+w)+(y+w)*(2*w+1)]
      at(0,0) = - 4.0 ;
      at(-1,0) = 1.0 ;
      at(+1,0) = 1.0 ;
      at(0,1) = 1.0 ;
      at(0,-1) = 1.0 ;
#undef at

      vl_imsmooth_f(pt, num,
                    pt, num, num, num,
                    sigmaDelta / step, sigmaDelta / step) ;

#if 0
      {
        char name [200] ;
        snprintf(name, 200, "/tmp/%f-lap.pgm", sigmaDelta) ;
        vl_pgm_write_f(name, pt, num, num) ;
      }
#endif

    }
  }
  return self ;
}

/** @brief Reset object
 ** @param self object.
 **
 ** This function removes any buffered features and frees other
 ** internal buffers.
 **/

void
vl_covdet_reset (VlCovDet * self)
{
  if (self->features) {
    vl_free(self->features) ;
    self->features = NULL ;
  }
  if (self->css) {
    vl_scalespace_delete(self->css) ;
    self->css = NULL ;
  }
  if (self->gss) {
    vl_scalespace_delete(self->gss) ;
    self->gss = NULL ;
  }
}

/** @brief Delete object instance
 ** @param self object.
 **/

void
vl_covdet_delete (VlCovDet * self)
{
  vl_covdet_reset(self) ;
  if (self->patch) vl_free (self->patch) ;
  vl_free(self) ;
}

/** @brief Append a feature to the internal buffer.
 ** @param self object.
 ** @param feature a pointer to the feature to append.
 ** @return status.
 **
 ** The feature is copied. The function may fail with @c status
 ** equal to ::VL_ERR_ALLOC if there is insufficient memory.
 **/

int
vl_covdet_append_feature (VlCovDet * self, VlCovDetFeature const * feature)
{
  vl_size requiredSize ;
  assert(self) ;
  assert(feature) ;
  self->numFeatures ++ ;
  requiredSize = self->numFeatures * sizeof(VlCovDetFeature) ;
  if (requiredSize > self->numFeatureBufferSize) {
    int err = _vl_resize_buffer((void**)&self->features, &self->numFeatureBufferSize,
                                (self->numFeatures + 1000) * sizeof(VlCovDetFeature)) ;
    if (err) {
      self->numFeatures -- ;
      return err ;
    }
  }
  self->features[self->numFeatures - 1] = *feature ;
  return VL_ERR_OK ;
}

/* ---------------------------------------------------------------- */
/*                                              Process a new image */
/* ---------------------------------------------------------------- */

/** @brief Detect features in an image
 ** @param self object.
 ** @param image image to process.
 ** @param width image width.
 ** @param height image height.
 ** @return status.
 **
 ** @a width and @a height must be at least one pixel. The function
 ** fails by returing ::VL_ERR_ALLOC if the memory is insufficient.
 **/

int
vl_covdet_put_image (VlCovDet * self,
                     float const * image,
                     vl_size width, vl_size height)
{
  vl_size const minOctaveSize = 16 ;
  vl_index lastOctave ;
  vl_index octaveFirstSubdivision ;
  vl_index octaveLastSubdivision ;
  VlScaleSpaceGeometry geom = vl_scalespace_get_default_geometry(width,height) ;

  assert (self) ;
  assert (image) ;
  assert (width >= 1) ;
  assert (height >= 1) ;

  /* (minOctaveSize - 1) 2^lastOctave <= min(width,height) - 1 */
  lastOctave = vl_floor_d(vl_log2_d(VL_MIN((double)width-1,(double)height-1) / (minOctaveSize - 1))) ;

  if (self->method == VL_COVDET_METHOD_DOG) {
    octaveFirstSubdivision = -1 ;
    octaveLastSubdivision = self->octaveResolution + 1 ;
  } else if (self->method == VL_COVDET_METHOD_HESSIAN) {
    octaveFirstSubdivision = -1 ;
    octaveLastSubdivision = self->octaveResolution ;
  } else {
    octaveFirstSubdivision = 0 ;
    octaveLastSubdivision = self->octaveResolution - 1 ;
  }

  geom.width = width ;
  geom.height = height ;
  geom.firstOctave = self->firstOctave ;
  geom.lastOctave = lastOctave ;
  geom.octaveResolution = self->octaveResolution ;
  geom.octaveFirstSubdivision = octaveFirstSubdivision ;
  geom.octaveLastSubdivision = octaveLastSubdivision ;

  if (self->gss == NULL ||
      ! vl_scalespacegeometry_is_equal (geom,
                                        vl_scalespace_get_geometry(self->gss)))
  {
    if (self->gss) vl_scalespace_delete(self->gss) ;
    self->gss = vl_scalespace_new_with_geometry(geom) ;
    if (self->gss == NULL) return VL_ERR_ALLOC ;
  }
  vl_scalespace_put_image(self->gss, image) ;
  return VL_ERR_OK ;
}

/* ---------------------------------------------------------------- */
/*                                              Cornerness measures */
/* ---------------------------------------------------------------- */

/** @brief Scaled derminant of the Hessian filter
 ** @param hessian output image.
 ** @param image input image.
 ** @param width image width.
 ** @param height image height.
 ** @param step image sampling step (pixel size).
 ** @param sigma Gaussian smoothing of the input image.
 **/

static void
_vl_det_hessian_response (float * hessian,
                          float const * image,
                          vl_size width, vl_size height,
                          double step, double sigma)
{
  float factor = (float) pow(sigma/step, 4.0) ;
  vl_index const xo = 1 ; /* x-stride */
  vl_index const yo = width;  /* y-stride */
  vl_size r, c;

  float p11, p12, p13, p21, p22, p23, p31, p32, p33;

  /* setup input pointer to be centered at 0,1 */
  float const *in = image + yo ;

  /* setup output pointer to be centered at 1,1 */
  float *out = hessian + xo + yo;

  /* move 3x3 window and convolve */
  for (r = 1; r < height - 1; ++r)
  {
    /* fill in shift registers at the beginning of the row */
    p11 = in[-yo]; p12 = in[xo - yo];
    p21 = in[  0]; p22 = in[xo     ];
    p31 = in[+yo]; p32 = in[xo + yo];
    /* setup input pointer to (2,1) of the 3x3 square */
    in += 2;
    for (c = 1; c < width - 1; ++c)
    {
      float Lxx, Lyy, Lxy;
      /* fetch remaining values (last column) */
      p13 = in[-yo]; p23 = *in; p33 = in[+yo];

      /* Compute 3x3 Hessian values from pixel differences. */
      Lxx = (-p21 + 2*p22 - p23);
      Lyy = (-p12 + 2*p22 - p32);
      Lxy = ((p11 - p31 - p13 + p33)/4.0f);

      /* normalize and write out */
      *out = (Lxx * Lyy - Lxy * Lxy) * factor ;

      /* move window */
      p11=p12; p12=p13;
      p21=p22; p22=p23;
      p31=p32; p32=p33;

      /* move input/output pointers */
      in++; out++;
    }
    out += 2;
  }

  /* Copy the computed values to borders */
  in = hessian + yo + xo ;
  out = hessian + xo ;

  /* Top row without corners */
  memcpy(out, in, (width - 2)*sizeof(float));
  out--;
  in -= yo;

  /* Left border columns without last row */
  for (r = 0; r < height - 1; r++){
    *out = *in;
    *(out + yo - 1) = *(in + yo - 3);
    in += yo;
    out += yo;
  }

  /* Bottom corners */
  in -= yo;
  *out = *in;
  *(out + yo - 1) = *(in + yo - 3);

  /* Bottom row without corners */
  out++;
  memcpy(out, in, (width - 2)*sizeof(float));
}

/** @brief Scale-normalised Harris response
 ** @param harris output image.
 ** @param image input image.
 ** @param width image width.
 ** @param height image height.
 ** @param step image sampling step (pixel size).
 ** @param sigma Gaussian smoothing of the input image.
 ** @param sigmaI integration scale.
 ** @param alpha factor in the definition of the Harris score.
 **/

static void
_vl_harris_response (float * harris,
                     float const * image,
                     vl_size width, vl_size height,
                     double step, double sigma,
                     double sigmaI, double alpha)
{
  float factor = (float) pow(sigma/step, 4.0) ;
  vl_index k ;

  float * LxLx ;
  float * LyLy ;
  float * LxLy ;

  LxLx = vl_malloc(sizeof(float) * width * height) ;
  LyLy = vl_malloc(sizeof(float) * width * height) ;
  LxLy = vl_malloc(sizeof(float) * width * height) ;

  vl_imgradient_f (LxLx, LyLy, 1, width, image, width, height, width) ;

  for (k = 0 ; k < (signed)(width * height) ; ++k) {
    float dx = LxLx[k] ;
    float dy = LyLy[k] ;
    LxLx[k] = dx*dx ;
    LyLy[k] = dy*dy ;
    LxLy[k] = dx*dy ;
  }

  vl_imsmooth_f(LxLx, width, LxLx, width, height, width,
                sigmaI / step, sigmaI / step) ;

  vl_imsmooth_f(LyLy, width, LyLy, width, height, width,
                sigmaI / step, sigmaI / step) ;

  vl_imsmooth_f(LxLy, width, LxLy, width, height, width,
                sigmaI / step, sigmaI / step) ;

  for (k = 0 ; k < (signed)(width * height) ; ++k) {
    float a = LxLx[k] ;
    float b = LyLy[k] ;
    float c = LxLy[k] ;

    float determinant = a * b - c * c ;
    float trace = a + b ;

    harris[k] = factor * (determinant - alpha * (trace * trace)) ;
  }

  vl_free(LxLy) ;
  vl_free(LyLy) ;
  vl_free(LxLx) ;
}

/** @brief Difference of Gaussian
 ** @param dog output image.
 ** @param level1 input image at the smaller Gaussian scale.
 ** @param level2 input image at the larger Gaussian scale.
 ** @param width image width.
 ** @param height image height.
 **/

static void
_vl_dog_response (float * dog,
                  float const * level1,
                  float const * level2,
                  vl_size width, vl_size height)
{
  vl_index k ;
  for (k = 0 ; k < (signed)(width*height) ; ++k) {
    dog[k] = level2[k] - level1[k] ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                  Detect features */
/* ---------------------------------------------------------------- */

/** @brief Detect scale-space features
 ** @param self object.
 **
 ** This function runs the configured feature detector on the image
 ** that was passed by using ::vl_covdet_put_image.
 **/

void
vl_covdet_detect (VlCovDet * self, vl_size max_num_features)
{
  VlScaleSpaceGeometry geom = vl_scalespace_get_geometry(self->gss) ;
  VlScaleSpaceGeometry cgeom ;
  float * levelxx = NULL ;
  float * levelyy = NULL ;
  float * levelxy = NULL ;
  vl_index o, s ;

  assert (self) ;
  assert (self->gss) ;

  /* clear previous detections if any */
  self->numFeatures = 0 ;

  /* prepare buffers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  cgeom = geom ;
  if (self->method == VL_COVDET_METHOD_DOG) {
    cgeom.octaveLastSubdivision -= 1 ;
  }
  if (!self->css ||
      !vl_scalespacegeometry_is_equal(cgeom,
                                      vl_scalespace_get_geometry(self->css)))
  {
    if (self->css) vl_scalespace_delete(self->css) ;
    self->css = vl_scalespace_new_with_geometry(cgeom) ;
  }
  if (self->method == VL_COVDET_METHOD_HARRIS_LAPLACE ||
      self->method == VL_COVDET_METHOD_MULTISCALE_HARRIS) {
    VlScaleSpaceOctaveGeometry oct = vl_scalespace_get_octave_geometry(self->gss, geom.firstOctave) ;
    levelxx = vl_malloc(oct.width * oct.height * sizeof(float)) ;
    levelyy = vl_malloc(oct.width * oct.height * sizeof(float)) ;
    levelxy = vl_malloc(oct.width * oct.height * sizeof(float)) ;
  }

  /* compute cornerness ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  for (o = cgeom.firstOctave ; o <= cgeom.lastOctave ; ++o) {
    VlScaleSpaceOctaveGeometry oct = vl_scalespace_get_octave_geometry(self->css, o) ;

    for (s = cgeom.octaveFirstSubdivision ; s <= cgeom.octaveLastSubdivision ; ++s) {
      float * level = vl_scalespace_get_level(self->gss, o, s) ;
      float * clevel = vl_scalespace_get_level(self->css, o, s) ;
      double sigma = vl_scalespace_get_level_sigma(self->css, o, s) ;
      switch (self->method) {
        case VL_COVDET_METHOD_DOG:
          _vl_dog_response(clevel,
                           vl_scalespace_get_level(self->gss, o, s + 1),
                           level,
                           oct.width, oct.height) ;
          break ;

        case VL_COVDET_METHOD_HARRIS_LAPLACE:
        case VL_COVDET_METHOD_MULTISCALE_HARRIS:
          _vl_harris_response(clevel,
                              level, oct.width, oct.height, oct.step,
                              sigma, 1.4 * sigma, 0.05) ;
          break ;

        case VL_COVDET_METHOD_HESSIAN:
        case VL_COVDET_METHOD_HESSIAN_LAPLACE:
        case VL_COVDET_METHOD_MULTISCALE_HESSIAN:
          _vl_det_hessian_response(clevel, level, oct.width, oct.height, oct.step, sigma) ;
          break ;

        default:
          assert(0) ;
      }
    }
  }

  /* find and refine local maxima ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
  {
    vl_index * extrema = NULL ;
    vl_size extremaBufferSize = 0 ;
    vl_size numExtrema ;
    vl_size index ;
    for (o = cgeom.lastOctave; o >= cgeom.firstOctave; --o) {
      VlScaleSpaceOctaveGeometry octgeom = vl_scalespace_get_octave_geometry(self->css, o) ;
      double step = octgeom.step ;
      vl_size width = octgeom.width ;
      vl_size height = octgeom.height ;
      vl_size depth = cgeom.octaveLastSubdivision - cgeom.octaveFirstSubdivision + 1 ;

      switch (self->method) {
        case VL_COVDET_METHOD_DOG:
        case VL_COVDET_METHOD_HESSIAN:
        {
          /* scale-space extrema */
          float const * octave =
          vl_scalespace_get_level(self->css, o, cgeom.octaveFirstSubdivision) ;
          numExtrema = vl_find_local_extrema_3(&extrema, &extremaBufferSize,
                                               octave, width, height, depth,
                                               0.8 * self->peakThreshold);
          for (index = 0 ; index < numExtrema ; ++index) {
            VlCovDetExtremum3 refined ;
            VlCovDetFeature feature ;
            vl_bool ok ;
            memset(&feature, 0, sizeof(feature)) ;
            ok = vl_refine_local_extreum_3(&refined,
                                           octave, width, height, depth,
                                           extrema[3*index+0],
                                           extrema[3*index+1],
                                           extrema[3*index+2]) ;
            ok &= fabs(refined.peakScore) > self->peakThreshold ;
            ok &= refined.edgeScore < self->edgeThreshold ;
            if (ok) {
              double sigma = cgeom.baseScale *
              pow(2.0, o + (refined.z + cgeom.octaveFirstSubdivision)
                  / cgeom.octaveResolution) ;
              feature.frame.x = refined.x * step ;
              feature.frame.y = refined.y * step ;
              feature.frame.a11 = sigma ;
              feature.frame.a12 = 0.0 ;
              feature.frame.a21 = 0.0 ;
              feature.frame.a22 = sigma ;
              feature.o = o ;
              feature.s = round(refined.z) ;
              feature.peakScore = refined.peakScore ;
              feature.edgeScore = refined.edgeScore ;
              vl_covdet_append_feature(self, &feature) ;
            }
          }
          break ;
        }

        default:
        {
          for (s = cgeom.octaveFirstSubdivision ; s < cgeom.octaveLastSubdivision ; ++s) {
            /* space extrema */
            float const * level = vl_scalespace_get_level(self->css,o,s) ;
            numExtrema = vl_find_local_extrema_2(&extrema, &extremaBufferSize,
                                                 level,
                                                 width, height,
                                                 0.8 * self->peakThreshold);
            for (index = 0 ; index < numExtrema ; ++index) {
              VlCovDetExtremum2 refined ;
              VlCovDetFeature feature ;
              vl_bool ok ;
              memset(&feature, 0, sizeof(feature)) ;
              ok = vl_refine_local_extreum_2(&refined,
                                             level, width, height,
                                             extrema[2*index+0],
                                             extrema[2*index+1]);
              ok &= fabs(refined.peakScore) > self->peakThreshold ;
              ok &= refined.edgeScore < self->edgeThreshold ;
              if (ok) {
                double sigma = cgeom.baseScale *
                pow(2.0, o + (double)s / cgeom.octaveResolution) ;
                feature.frame.x = refined.x * step ;
                feature.frame.y = refined.y * step ;
                feature.frame.a11 = sigma ;
                feature.frame.a12 = 0.0 ;
                feature.frame.a21 = 0.0 ;
                feature.frame.a22 = sigma ;
                feature.o = o ;
                feature.s = s ;
                feature.peakScore = refined.peakScore ;
                feature.edgeScore = refined.edgeScore ;
                vl_covdet_append_feature(self, &feature) ;
              }
            }
          }
          break ;
        }
      }
      if (self->numFeatures >= max_num_features) {
        break;
      }
    } /* next octave */

    if (extrema) { vl_free(extrema) ; extrema = 0 ; }
  }

  /* Laplacian scale selection for certain methods */
  switch (self->method) {
    case VL_COVDET_METHOD_HARRIS_LAPLACE :
    case VL_COVDET_METHOD_HESSIAN_LAPLACE :
      vl_covdet_extract_laplacian_scales (self) ;
      break ;
    default:
      break ;
  }

  if (self->nonExtremaSuppression) {
    vl_index i, j ;
    double tol = self->nonExtremaSuppression ;
    self->numNonExtremaSuppressed = 0 ;
    for (i = 0 ; i < (signed)self->numFeatures ; ++i) {
      double x = self->features[i].frame.x ;
      double y = self->features[i].frame.y ;
      double sigma = self->features[i].frame.a11 ;
      double score = self->features[i].peakScore ;
      if (score == 0) continue ;

      for (j = 0 ; j < (signed)self->numFeatures ; ++j) {
        double dx_ = self->features[j].frame.x - x ;
        double dy_ = self->features[j].frame.y - y ;
        double sigma_ = self->features[j].frame.a11 ;
        double score_ = self->features[j].peakScore ;
        if (score_ == 0) continue ;
        if (sigma < (1+tol) * sigma_ &&
            sigma_ < (1+tol) * sigma &&
            vl_abs_d(dx_) < tol * sigma &&
            vl_abs_d(dy_) < tol * sigma &&
            vl_abs_d(score) > vl_abs_d(score_)) {
          self->features[j].peakScore = 0 ;
          self->numNonExtremaSuppressed ++ ;
        }
      }
    }
    j = 0 ;
    for (i = 0 ; i < (signed)self->numFeatures ; ++i) {
      VlCovDetFeature feature = self->features[i] ;
      if (self->features[i].peakScore != 0) {
        self->features[j++] = feature ;
      }
    }
    self->numFeatures = j ;
  }

  if (levelxx) vl_free(levelxx) ;
  if (levelyy) vl_free(levelyy) ;
  if (levelxy) vl_free(levelxy) ;
}

/* ---------------------------------------------------------------- */
/*                                                  Extract patches */
/* ---------------------------------------------------------------- */

/** @internal
 ** @brief Helper for extracting patches
 ** @param self object.
 ** @param[out] sigma1 actual patch smoothing along the first axis.
 ** @param[out] sigma2 actual patch smoothing along the second axis.
 ** @param patch buffer.
 ** @param resolution patch resolution.
 ** @param extent patch extent.
 ** @param sigma desired smoothing in the patch frame.
 ** @param A_ linear transfomration from patch to image.
 ** @param T_ translation from patch to image.
 ** @param d1 first singular value @a A.
 ** @param d2 second singular value of @a A.
 **/

vl_bool
vl_covdet_extract_patch_helper (VlCovDet * self,
                                double * sigma1,
                                double * sigma2,
                                float * patch,
                                vl_size resolution,
                                double extent,
                                double sigma,
                                double A_ [4],
                                double T_ [2],
                                double d1, double d2)
{
  vl_index o, s ;
  double factor ;
  double sigma_ ;
  float const * level ;
  vl_size width, height ;
  double step ;

  double A [4] = {A_[0], A_[1], A_[2], A_[3]} ;
  double T [2] = {T_[0], T_[1]} ;

  VlScaleSpaceGeometry geom = vl_scalespace_get_geometry(self->gss) ;
  VlScaleSpaceOctaveGeometry oct ;

  /* Starting from a pre-smoothed image at scale sigma_
     because of the mapping A the resulting smoothing in
     the warped patch is S, where

        sigma_^2 I = A S A',

        S = sigma_^2 inv(A) inv(A)' = sigma_^2 V D^-2 V',

        A = U D V'.

     Thus we rotate A by V to obtain an axis-aligned smoothing:

        A = U*D,

        S = sigma_^2 D^-2.

     Then we search the scale-space for the best sigma_ such
     that the target smoothing is approximated from below:

        max sigma_(o,s) :    simga_(o,s) factor <= sigma,
        factor = max{abs(D11), abs(D22)}.
   */


  /*
   Determine the best level (o,s) such that sigma_(o,s) factor <= sigma.
   This can be obtained by scanning octaves from smallest to largest
   and stopping when no level in the octave satisfies the relation.

   Given the range of octave availables, do the best you can.
   */

  factor = 1.0 / VL_MIN(d1, d2) ;

  for (o = geom.firstOctave + 1 ; o <= geom.lastOctave ; ++o) {
    s = vl_floor_d(vl_log2_d(sigma / (factor * geom.baseScale)) - o) ;
    s = VL_MAX(s, geom.octaveFirstSubdivision) ;
    s = VL_MIN(s, geom.octaveLastSubdivision) ;
    sigma_ = geom.baseScale * pow(2.0, o + (double)s / geom.octaveResolution) ;
    /*VL_PRINTF(".. %d D=%g %g; sigma_=%g factor*sigma_=%g\n", o, d1, d2, sigma_, factor* sigma_) ;*/
    if (factor * sigma_ > sigma) {
      o -- ;
      break ;
    }
  }
  o = VL_MIN(o, geom.lastOctave) ;
  s = vl_floor_d(vl_log2_d(sigma / (factor * geom.baseScale)) - o) ;
  s = VL_MAX(s, geom.octaveFirstSubdivision) ;
  s = VL_MIN(s, geom.octaveLastSubdivision) ;
  sigma_ = geom.baseScale * pow(2.0, o + (double)s / geom.octaveResolution) ;
  if (sigma1) *sigma1 = sigma_ / d1 ;
  if (sigma2) *sigma2 = sigma_ / d2 ;

  /*VL_PRINTF("%d %d %g %g %g %g\n", o, s, factor, sigma_, factor * sigma_, sigma) ;*/

  /*
   Now the scale space level to be used for this warping has been
   determined.

   If the patch is partially or completely out of the image boundary,
   create a padded copy of the required region first.
   */

  level = vl_scalespace_get_level(self->gss, o, s) ;
  oct = vl_scalespace_get_octave_geometry(self->gss, o) ;
  width = oct.width ;
  height = oct.height ;
  step = oct.step ;

  A[0] /= step ;
  A[1] /= step ;
  A[2] /= step ;
  A[3] /= step ;
  T[0] /= step ;
  T[1] /= step ;

  {
    /*
     Warp the patch domain [x0hat,y0hat,x1hat,y1hat] to the image domain/
     Obtain a box [x0,y0,x1,y1] enclosing that wrapped box, and then
     an integer vertexes version [x0i, y0i, x1i, y1i], making room
     for one pixel at the boundary to simplify bilinear interpolation
     later on.
     */
    vl_index x0i, y0i, x1i, y1i ;
    double x0 = +VL_INFINITY_D ;
    double x1 = -VL_INFINITY_D ;
    double y0 = +VL_INFINITY_D ;
    double y1 = -VL_INFINITY_D ;
    double boxx [4] = {extent, extent, -extent, -extent} ;
    double boxy [4] = {-extent, extent, extent, -extent} ;
    int i ;
    for (i = 0 ; i < 4 ; ++i) {
      double x = A[0] * boxx[i] + A[2] * boxy[i] + T[0] ;
      double y = A[1] * boxx[i] + A[3] * boxy[i] + T[1] ;
      x0 = VL_MIN(x0, x) ;
      x1 = VL_MAX(x1, x) ;
      y0 = VL_MIN(y0, y) ;
      y1 = VL_MAX(y1, y) ;
    }

    /* Leave one pixel border for bilinear interpolation. */
    x0i = floor(x0) - 1 ;
    y0i = floor(y0) - 1 ;
    x1i = ceil(x1) + 1 ;
    y1i = ceil(y1) + 1 ;

    /*
     If the box [x0i,y0i,x1i,y1i] is not fully contained in the
     image domain, then create a copy of this region by padding
     the image. The image is extended by continuity.
     */

    if (x0i < 0 || x1i > (signed)width-1 ||
        y0i < 0 || y1i > (signed)height-1) {
      vl_index xi, yi ;

      /* compute the amount of l,r,t,b padding needed to complete the patch */
      vl_index padx0 = VL_MAX(0, - x0i) ;
      vl_index pady0 = VL_MAX(0, - y0i) ;
      vl_index padx1 = VL_MAX(0, x1i - ((signed)width - 1)) ;
      vl_index pady1 = VL_MAX(0, y1i - ((signed)height - 1)) ;

      /* make enough room for the patch */
      vl_index patchWidth = x1i - x0i + 1 ;
      vl_index patchHeight = y1i - y0i + 1 ;
      vl_size patchBufferSize = patchWidth * patchHeight * sizeof(float) ;
      if (patchBufferSize > self->patchBufferSize) {
        int err = _vl_resize_buffer((void**)&self->patch, &self->patchBufferSize, patchBufferSize) ;
        if (err) return vl_set_last_error(VL_ERR_ALLOC, NULL) ;
      }

      if (pady0 < patchHeight - pady1) {
        /* start by filling the central horizontal band */
        for (yi = y0i + pady0 ; yi < y0i + patchHeight - pady1 ; ++ yi) {
          float *dst = self->patch + (yi - y0i) * patchWidth ;
          float const *src = level + yi * width + VL_MIN(VL_MAX(0, x0i),(signed)width-1) ;
          for (xi = x0i ; xi < x0i + padx0 ; ++xi) *dst++ = *src ;
          for ( ; xi < x0i + patchWidth - padx1 - 2 ; ++xi) *dst++ = *src++ ;
          for ( ; xi < x0i + patchWidth ; ++xi) *dst++ = *src ;
        }
        /* now extend the central band up and down */
        for (yi = 0 ; yi < pady0 ; ++yi) {
          memcpy(self->patch + yi * patchWidth,
                 self->patch + pady0 * patchWidth,
                 patchWidth * sizeof(float)) ;
        }
        for (yi = patchHeight - pady1 ; yi < patchHeight ; ++yi) {
          memcpy(self->patch + yi * patchWidth,
                 self->patch + (patchHeight - pady1 - 1) * patchWidth,
                 patchWidth * sizeof(float)) ;
        }
      } else {
        /* should be handled better! */
        memset(self->patch, 0, self->patchBufferSize) ;
      }
#if 0
      {
        char name [200] ;
        snprintf(name, 200, "/tmp/%20.0f-ext.pgm", 1e10*vl_get_cpu_time()) ;
        vl_pgm_write_f(name, patch, patchWidth, patchWidth) ;
      }
#endif

      level = self->patch ;
      width = patchWidth ;
      height = patchHeight ;
      T[0] -= x0i ;
      T[1] -= y0i ;
    }
  }

  /*
   Resample by using bilinear interpolation.
   */
  {
    float * pt = patch ;
    double yhat = -extent ;
    vl_index xxi ;
    vl_index yyi ;
    double stephat = extent / resolution ;

    for (yyi = 0 ; yyi < 2 * (signed)resolution + 1 ; ++yyi) {
      double xhat = -extent ;
      double rx = A[2] * yhat + T[0] ;
      double ry = A[3] * yhat + T[1] ;
      for (xxi = 0 ; xxi < 2 * (signed)resolution + 1 ; ++xxi) {
        double x = A[0] * xhat + rx ;
        double y = A[1] * xhat + ry ;
        vl_index xi = vl_floor_d(x) ;
        vl_index yi = vl_floor_d(y) ;
        double i00 = level[yi * width + xi] ;
        double i10 = level[yi * width + xi + 1] ;
        double i01 = level[(yi + 1) * width + xi] ;
        double i11 = level[(yi + 1) * width + xi + 1] ;
        double wx = x - xi ;
        double wy = y - yi ;

        assert(xi >= 0 && xi <= (signed)width - 1) ;
        assert(yi >= 0 && yi <= (signed)height - 1) ;

        *pt++ =
        (1.0 - wy) * ((1.0 - wx) * i00 + wx * i10) +
        wy * ((1.0 - wx) * i01 + wx * i11) ;

        xhat += stephat ;
      }
      yhat += stephat ;
    }
  }
#if 0
    {
      char name [200] ;
      snprintf(name, 200, "/tmp/%20.0f.pgm", 1e10*vl_get_cpu_time()) ;
      vl_pgm_write_f(name, patch, 2*resolution+1, 2*resolution+1) ;
    }
#endif
  return VL_ERR_OK ;
}

/** @brief Helper for extracting patches
 ** @param self object.
 ** @param patch buffer.
 ** @param resolution patch resolution.
 ** @param extent patch extent.
 ** @param sigma desired smoothing in the patch frame.
 ** @param frame feature frame.
 **
 ** The function considers a patch of extent <code>[-extent,extent]</code>
 ** on each side, with a side counting <code>2*resolution+1</code> pixels.
 ** In attempts to extract from the scale space a patch
 ** based on the affine warping specified by @a frame in such a way
 ** that the resulting smoothing of the image is @a sigma (in the
 ** patch frame).
 **
 ** The transformation is specified by the matrices @c A and @c T
 ** embedded in the feature @a frame. Note that this transformation maps
 ** pixels from the patch frame to the image frame.
 **/

vl_bool
vl_covdet_extract_patch_for_frame (VlCovDet * self,
                                   float * patch,
                                   vl_size resolution,
                                   double extent,
                                   double sigma,
                                   VlFrameOrientedEllipse frame)
{
  double A[2*2] = {frame.a11, frame.a21, frame.a12, frame.a22} ;
  double T[2] = {frame.x, frame.y} ;
  double D[4], U[4], V[4] ;

  vl_svd2(D, U, V, A) ;

  return vl_covdet_extract_patch_helper
  (self, NULL, NULL, patch, resolution, extent, sigma, A, T, D[0], D[3]) ;
}

/* ---------------------------------------------------------------- */
/*                                                     Affine shape */
/* ---------------------------------------------------------------- */

/** @brief Extract the affine shape for a feature frame
 ** @param self object.
 ** @param adapted the shape-adapted frame.
 ** @param frame the input frame.
 ** @return ::VL_ERR_OK if affine adaptation is successful.
 **
 ** This function may fail if adaptation is unsuccessful or if
 ** memory is insufficient.
 **/

int
vl_covdet_extract_affine_shape_for_frame (VlCovDet * self,
                                          VlFrameOrientedEllipse * adapted,
                                          VlFrameOrientedEllipse frame)
{
  vl_index iter = 0 ;

  double A [2*2] = {frame.a11, frame.a21, frame.a12, frame.a22} ;
  double T [2] = {frame.x, frame.y} ;
  double U [2*2] ;
  double V [2*2] ;
  double D [2*2] ;
  double M [2*2] ;
  double P [2*2] ;
  double P_ [2*2] ;
  double Q [2*2] ;
  double sigma1, sigma2 ;
  double sigmaD = VL_COVDET_AA_RELATIVE_DERIVATIVE_SIGMA ;
  double factor ;
  double anisotropy ;
  double referenceScale ;
  vl_size const resolution = VL_COVDET_AA_PATCH_RESOLUTION ;
  vl_size const side = 2*VL_COVDET_AA_PATCH_RESOLUTION + 1 ;
  double const extent = VL_COVDET_AA_PATCH_EXTENT ;


  *adapted = frame ;

  while (1) {
    double lxx = 0, lxy = 0, lyy = 0 ;
    vl_index k ;
    int err ;

    /* A = U D V' */
    vl_svd2(D, U, V, A) ;
    anisotropy = VL_MAX(D[0]/D[3], D[3]/D[0]) ;

    /* VL_PRINTF("anisot: %g\n", anisotropy); */

    if (anisotropy > VL_COVDET_AA_MAX_ANISOTROPY) {
      /* diverged, give up with current solution */
      break ;
    }

    /* make sure that the smallest singluar value stays fixed
       after the first iteration */
    if (iter == 0) {
      referenceScale = VL_MIN(D[0], D[3]) ;
      factor = 1.0 ;
    } else {
      factor = referenceScale / VL_MIN(D[0],D[3]) ;
    }

    D[0] *= factor ;
    D[3] *= factor ;

    A[0] = U[0] * D[0] ;
    A[1] = U[1] * D[0] ;
    A[2] = U[2] * D[3] ;
    A[3] = U[3] * D[3] ;

    adapted->a11 = A[0] ;
    adapted->a21 = A[1] ;
    adapted->a12 = A[2] ;
    adapted->a22 = A[3] ;

    if (++iter >= VL_COVDET_AA_MAX_NUM_ITERATIONS) break ;

    err = vl_covdet_extract_patch_helper(self,
                                         &sigma1, &sigma2,
                                         self->aaPatch,
                                         resolution,
                                         extent,
                                         sigmaD,
                                         A, T, D[0], D[3]) ;
    if (err) return err ;

    if (self->aaAccurateSmoothing ) {
      double deltaSigma1 = sqrt(VL_MAX(sigmaD*sigmaD - sigma1*sigma1,0)) ;
      double deltaSigma2 = sqrt(VL_MAX(sigmaD*sigmaD - sigma2*sigma2,0)) ;
      double stephat = extent / resolution ;
      vl_imsmooth_f(self->aaPatch, side,
                    self->aaPatch, side, side, side,
                    deltaSigma1 / stephat, deltaSigma2 / stephat) ;
    }

    /* compute second moment matrix */
    vl_imgradient_f (self->aaPatchX, self->aaPatchY, 1, side,
                     self->aaPatch, side, side, side) ;

    for (k = 0 ; k < (signed)(side*side) ; ++k) {
      double lx = self->aaPatchX[k] ;
      double ly = self->aaPatchY[k] ;
      lxx += lx * lx * self->aaMask[k] ;
      lyy += ly * ly * self->aaMask[k] ;
      lxy += lx * ly * self->aaMask[k] ;
    }
    M[0] = lxx ;
    M[1] = lxy ;
    M[2] = lxy ;
    M[3] = lyy ;

    if (lxx == 0 || lyy == 0) {
      *adapted = frame ;
      break ;
    }

    /* decompose M = P * Q * P' */
    vl_svd2 (Q, P, P_, M) ;

    /*
     Setting A <- A * dA results in M to change approximatively as

     M --> dA'  M dA = dA' P Q P dA

     To make this proportional to the identity, we set

     dA ~= P Q^1/2

     we also make it so the smallest singular value of A is unchanged.
     */

    if (Q[3]/Q[0] < VL_COVDET_AA_CONVERGENCE_THRESHOLD &&
        Q[0]/Q[3] < VL_COVDET_AA_CONVERGENCE_THRESHOLD) {
      break ;
    }

    {
      double Ap [4] ;
      double q0 = sqrt(Q[0]) ;
      double q1 = sqrt(Q[3]) ;
      Ap[0] = (A[0] * P[0] + A[2] * P[1]) / q0 ;
      Ap[1] = (A[1] * P[0] + A[3] * P[1]) / q0 ;
      Ap[2] = (A[0] * P[2] + A[2] * P[3]) / q1 ;
      Ap[3] = (A[1] * P[2] + A[3] * P[3]) / q1 ;
      memcpy(A,Ap,4*sizeof(double)) ;
    }

  } /* next iteration */

  /*
   Make upright.

   Shape adaptation does not estimate rotation. This is fixed by default
   so that a selected axis is not rotated at all (usually this is the
   vertical axis for upright features). To do so, the frame is rotated
   as follows.
   */
  {
    double A [2*2] = {adapted->a11, adapted->a21, adapted->a12, adapted->a22} ;
    double ref [2] ;
    double ref_ [2] ;
    double angle ;
    double angle_ ;
    double dangle ;
    double r1, r2 ;

    if (self->transposed) {
      /* up is the x axis */
      ref[0] = 1 ;
      ref[1] = 0 ;
    } else {
      /* up is the y axis */
      ref[0] = 0 ;
      ref[1] = 1 ;
    }

    vl_solve_linear_system_2 (ref_, A, ref) ;
    angle = atan2(ref[1], ref[0]) ;
    angle_ = atan2(ref_[1], ref_[0]) ;
    dangle = angle_ - angle ;
    r1 = cos(dangle) ;
    r2 = sin(dangle) ;
    adapted->a11 = + A[0] * r1 + A[2] * r2 ;
    adapted->a21 = + A[1] * r1 + A[3] * r2 ;
    adapted->a12 = - A[0] * r2 + A[2] * r1 ;
    adapted->a22 = - A[1] * r2 + A[3] * r1 ;
  }

  return VL_ERR_OK ;
}

/** @brief Extract the affine shape for the stored features
 ** @param self object.
 **
 ** This function may discard features for which no affine
 ** shape can reliably be detected.
 **/

void
vl_covdet_extract_affine_shape (VlCovDet * self)
{
  vl_index i, j = 0 ;
  vl_size numFeatures = vl_covdet_get_num_features(self) ;
  VlCovDetFeature * feature = vl_covdet_get_features(self);
  for (i = 0 ; i < (signed)numFeatures ; ++i) {
    int status ;
    VlFrameOrientedEllipse adapted ;
    status = vl_covdet_extract_affine_shape_for_frame(self, &adapted, feature[i].frame) ;
    if (status == VL_ERR_OK) {
      feature[j] = feature[i] ;
      feature[j].frame = adapted ;
      ++ j ;
    }
  }
  self->numFeatures = j ;
}

/* ---------------------------------------------------------------- */
/*                                                      Orientation */
/* ---------------------------------------------------------------- */

static int
_vl_covdet_compare_orientations_descending (void const * a_,
                                            void const * b_)
{
  VlCovDetFeatureOrientation const * a = a_ ;
  VlCovDetFeatureOrientation const * b = b_ ;
  if (a->score > b->score) return -1 ;
  if (a->score < b->score) return +1 ;
  return 0 ;
}

/** @brief Extract the orientation(s) for a feature
 ** @param self object.
 ** @param numOrientations the number of detected orientations.
 ** @param frame pose of the feature.
 ** @return an array of detected orientations with their scores.
 **
 ** The returned array is a matrix of size @f$ 2 \times n @f$
 ** where <em>n</em> is the number of detected orientations.
 **
 ** The function returns @c NULL if memory is insufficient.
 **/

VlCovDetFeatureOrientation *
vl_covdet_extract_orientations_for_frame (VlCovDet * self,
                                          vl_size * numOrientations,
                                          VlFrameOrientedEllipse frame)
{
  int err ;
  vl_index k, i ;
  vl_index iter ;

  double extent = VL_COVDET_AA_PATCH_EXTENT ;
  vl_size resolution = VL_COVDET_AA_PATCH_RESOLUTION ;
  vl_size side = 2 * resolution + 1  ;

  vl_size const numBins = VL_COVDET_OR_NUM_ORIENTATION_HISTOGAM_BINS ;
  double hist [VL_COVDET_OR_NUM_ORIENTATION_HISTOGAM_BINS] ;
  double const binExtent = 2 * VL_PI / VL_COVDET_OR_NUM_ORIENTATION_HISTOGAM_BINS ;
  double const peakRelativeSize = VL_COVDET_OR_ADDITIONAL_PEAKS_RELATIVE_SIZE ;
  double maxPeakValue ;

  double A [2*2] = {frame.a11, frame.a21, frame.a12, frame.a22} ;
  double T [2] = {frame.x, frame.y} ;
  double U [2*2] ;
  double V [2*2] ;
  double D [2*2] ;
  double sigma1, sigma2 ;
  double sigmaD = 1.0 ;
  double theta0 ;

  assert(self);
  assert(numOrientations) ;

  /*
   The goal is to estimate a rotation R(theta) such that the patch given
   by the transformation A R(theta) has the strongest average
   gradient pointing right (or down for transposed conventions).

   To compensate for tha anisotropic smoothing due to warping,
   A is decomposed as A = U D V' and the patch is warped by
   U D only, meaning that the matrix R_(theta) will be estimated instead,
   where:

      A R(theta) = U D V' R(theta) = U D R_(theta)

   such that R(theta) = V R(theta). That is an extra rotation of
   theta0 = atan2(V(2,1),V(1,1)).
   */

  /* axis aligned anisotropic smoothing for easier compensation */
  vl_svd2(D, U, V, A) ;

  A[0] = U[0] * D[0] ;
  A[1] = U[1] * D[0] ;
  A[2] = U[2] * D[3] ;
  A[3] = U[3] * D[3] ;

  theta0 = atan2(V[1],V[0]) ;

  err = vl_covdet_extract_patch_helper(self,
                                       &sigma1, &sigma2,
                                       self->aaPatch,
                                       resolution,
                                       extent,
                                       sigmaD,
                                       A, T, D[0], D[3]) ;

  if (err) {
    *numOrientations = 0 ;
    return NULL ;
  }

  if (1) {
    double deltaSigma1 = sqrt(VL_MAX(sigmaD*sigmaD - sigma1*sigma1,0)) ;
    double deltaSigma2 = sqrt(VL_MAX(sigmaD*sigmaD - sigma2*sigma2,0)) ;
    double stephat = extent / resolution ;
    vl_imsmooth_f(self->aaPatch, side,
                  self->aaPatch, side, side, side,
                  deltaSigma1 / stephat, deltaSigma2 / stephat) ;
  }

  /* histogram of oriented gradients */
  vl_imgradient_polar_f (self->aaPatchX, self->aaPatchY, 1, side,
                         self->aaPatch, side, side, side) ;

  memset (hist, 0, sizeof(double) * numBins) ;

  for (k = 0 ; k < (signed)(side*side) ; ++k) {
    double modulus = self->aaPatchX[k] ;
    double angle = self->aaPatchY[k] ;
    double weight = self->aaMask[k] ;

    double x = angle / binExtent ;
    vl_index bin = vl_floor_d(x) ;
    double w2 = x - bin ;
    double w1 = 1.0 - w2 ;

    hist[(bin + numBins) % numBins] += w1 * (modulus * weight) ;
    hist[(bin + numBins + 1) % numBins] += w2 * (modulus * weight) ;
  }

  /* smooth histogram */
  for (iter = 0; iter < 6; iter ++) {
    double prev = hist [numBins - 1] ;
    double first = hist [0] ;
    vl_index i ;
    for (i = 0; i < (signed)numBins - 1; ++i) {
      double curr = (prev + hist[i] + hist[(i + 1) % numBins]) / 3.0 ;
      prev = hist[i] ;
      hist[i] = curr ;
    }
    hist[i] = (prev + hist[i] + first) / 3.0 ;
  }

  /* find the histogram maximum */
  maxPeakValue = 0 ;
  for (i = 0 ; i < (signed)numBins ; ++i) {
    maxPeakValue = VL_MAX (maxPeakValue, hist[i]) ;
  }

  /* find peaks within 80% from max */
  *numOrientations = 0 ;
  for(i = 0 ; i < (signed)numBins ; ++i) {
    double h0 = hist [i] ;
    double hm = hist [(i - 1 + numBins) % numBins] ;
    double hp = hist [(i + 1 + numBins) % numBins] ;

    /* is this a peak? */
    if (h0 > peakRelativeSize * maxPeakValue && h0 > hm && h0 > hp) {
      /* quadratic interpolation */
      double di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
      double th = binExtent * (i + di) + theta0 ;
      if (self->transposed) {
        /* the axis to the right is y, measure orientations from this */
        th = th - VL_PI/2 ;
      }
      self->orientations[*numOrientations].angle = th ;
      self->orientations[*numOrientations].score = h0 ;
      *numOrientations += 1 ;
      //VL_PRINTF("%d %g\n", *numOrientations, th) ;

      if (*numOrientations >= VL_COVDET_MAX_NUM_ORIENTATIONS) break ;
    }
  }

  /* sort the orientations by decreasing scores */
  qsort(self->orientations,
        *numOrientations,
        sizeof(VlCovDetFeatureOrientation),
        _vl_covdet_compare_orientations_descending) ;

  return self->orientations ;
}

/** @brief Extract the orientation(s) for the stored features.
 ** @param self object.
 **
 ** Note that, since more than one orientation can be detected
 ** for each feature, this function may create copies of them,
 ** one for each orientation.
 **/

void
vl_covdet_extract_orientations (VlCovDet * self)
{
  vl_index i, j  ;
  vl_size numFeatures = vl_covdet_get_num_features(self) ;
  for (i = 0 ; i < (signed)numFeatures ; ++i) {
    vl_size numOrientations ;
    VlCovDetFeature feature = self->features[i] ;
    VlCovDetFeatureOrientation* orientations =
    vl_covdet_extract_orientations_for_frame(self, &numOrientations, feature.frame) ;

    for (j = 0 ; j < (signed)numOrientations ; ++j) {
      double A [2*2] = {
        feature.frame.a11,
        feature.frame.a21,
        feature.frame.a12,
        feature.frame.a22} ;
      double r1 = cos(orientations[j].angle) ;
      double r2 = sin(orientations[j].angle) ;
      VlCovDetFeature * oriented ;

      if (j == 0) {
        oriented = & self->features[i] ;
      } else {
        vl_covdet_append_feature(self, &feature) ;
        oriented = & self->features[self->numFeatures -1] ;
      }

      oriented->orientationScore = orientations[j].score ;
      oriented->frame.a11 = + A[0] * r1 + A[2] * r2 ;
      oriented->frame.a21 = + A[1] * r1 + A[3] * r2 ;
      oriented->frame.a12 = - A[0] * r2 + A[2] * r1 ;
      oriented->frame.a22 = - A[1] * r2 + A[3] * r1 ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                 Laplacian scales */
/* ---------------------------------------------------------------- */

/** @brief Extract the Laplacian scale(s) for a feature frame.
 ** @param self object.
 ** @param numScales the number of detected scales.
 ** @param frame pose of the feature.
 ** @return an array of detected scales.
 **
 ** The function returns @c NULL if memory is insufficient.
 **/

VlCovDetFeatureLaplacianScale *
vl_covdet_extract_laplacian_scales_for_frame (VlCovDet * self,
                                              vl_size * numScales,
                                              VlFrameOrientedEllipse frame)
{
  /*
   We try to explore one octave, with the nominal detection scale 1.0
   (in the patch reference frame) in the middle. Thus the goal is to sample
   the response of the tr-Laplacian operator at logarithmically
   spaced scales in 1/sqrt(2), sqrt(2).

   To this end, the patch is warped with a smoothing of at most
   sigmaImage = 1 / sqrt(2) (beginning of the scale), sampled at
   roughly twice the Nyquist frequency (so step = 1 / (2*sqrt(2))).
   This maes it possible to approximate the Laplacian operator at
   that scale by simple finite differences.

   */
  int err ;
  double const sigmaImage = 1.0 / sqrt(2.0) ;
  double const step = 0.5 * sigmaImage ;
  double actualSigmaImage ;
  vl_size const resolution = VL_COVDET_LAP_PATCH_RESOLUTION ;
  vl_size const num = 2 * resolution + 1 ;
  double extent = step * resolution ;
  double scores [VL_COVDET_LAP_NUM_LEVELS] ;
  double factor = 1.0 ;
  float const * pt ;
  vl_index k ;

  double A[2*2] = {frame.a11, frame.a21, frame.a12, frame.a22} ;
  double T[2] = {frame.x, frame.y} ;
  double D[4], U[4], V[4] ;
  double sigma1, sigma2 ;

  assert(self) ;
  assert(numScales) ;

  *numScales = 0 ;

  vl_svd2(D, U, V, A) ;

  err = vl_covdet_extract_patch_helper
  (self, &sigma1, &sigma2, self->lapPatch, resolution, extent, sigmaImage, A, T, D[0], D[3]) ;
  if (err) return NULL ;

  /* the actual smoothing after warping is never the target one */
  if (sigma1 == sigma2) {
    actualSigmaImage = sigma1 ;
  } else {
    /* here we could compensate */
    actualSigmaImage = sqrt(sigma1*sigma2) ;
  }

  /* now multiply by the bank of Laplacians */
  pt = self->laplacians ;
  for (k = 0 ; k < VL_COVDET_LAP_NUM_LEVELS ; ++k) {
    vl_index q ;
    double score = 0 ;
    double sigmaLap = pow(2.0, -0.5 + (double)k / (VL_COVDET_LAP_NUM_LEVELS - 1)) ;
    /* note that the sqrt argument cannot be negative since by construction
     sigmaLap >= sigmaImage */
    sigmaLap = sqrt(sigmaLap*sigmaLap
                    - sigmaImage*sigmaImage
                    + actualSigmaImage*actualSigmaImage) ;

    for (q = 0 ; q < (signed)(num * num) ; ++q) {
      score += (*pt++) * self->lapPatch[q] ;
    }
    scores[k] = score * sigmaLap * sigmaLap ;
  }

  /* find and interpolate maxima */
  for (k = 1 ; k < VL_COVDET_LAP_NUM_LEVELS - 1 ; ++k) {
    double a = scores[k-1] ;
    double b = scores[k] ;
    double c = scores[k+1] ;
    double t = self->lapPeakThreshold ;

    if ((((b > a) && (b > c)) || ((b < a) && (b < c))) && (vl_abs_d(b) >= t)) {
      double dk = - 0.5 * (c - a) / (c + a - 2 * b) ;
      double s = k + dk ;
      double sigmaLap = pow(2.0, -0.5 + s / (VL_COVDET_LAP_NUM_LEVELS - 1)) ;
      double scale ;
      sigmaLap = sqrt(sigmaLap*sigmaLap
                      - sigmaImage*sigmaImage
                      + actualSigmaImage*actualSigmaImage) ;
      scale = sigmaLap / 1.0 ;
      /*
       VL_PRINTF("** k:%d, s:%f, sigmaLapFilter:%f, sigmaLap%f, scale:%f (%f %f %f)\n",
       k,s,sigmaLapFilter,sigmaLap,scale,a,b,c) ;
       */
      if (*numScales < VL_COVDET_MAX_NUM_LAPLACIAN_SCALES) {
        self->scales[*numScales].scale = scale * factor ;
        self->scales[*numScales].score = b + 0.5 * (c - a) * dk ;
        *numScales += 1 ;
      }
    }
  }
  return self->scales ;
}

/** @brief Extract the Laplacian scales for the stored features
 ** @param self object.
 **
 ** Note that, since more than one orientation can be detected
 ** for each feature, this function may create copies of them,
 ** one for each orientation.
 **/
void
vl_covdet_extract_laplacian_scales (VlCovDet * self)
{
  vl_index i, j  ;
  vl_bool dropFeaturesWithoutScale = VL_TRUE ;
  vl_size numFeatures = vl_covdet_get_num_features(self) ;
  memset(self->numFeaturesWithNumScales, 0,
         sizeof(self->numFeaturesWithNumScales)) ;

  for (i = 0 ; i < (signed)numFeatures ; ++i) {
    vl_size numScales ;
    VlCovDetFeature feature = self->features[i] ;
    VlCovDetFeatureLaplacianScale const * scales =
    vl_covdet_extract_laplacian_scales_for_frame(self, &numScales, feature.frame) ;

    self->numFeaturesWithNumScales[numScales] ++ ;

    if (numScales == 0 && dropFeaturesWithoutScale) {
      self->features[i].peakScore = 0 ;
    }

    for (j = 0 ; j < (signed)numScales ; ++j) {
      VlCovDetFeature * scaled ;

      if (j == 0) {
        scaled = & self->features[i] ;
      } else {
        vl_covdet_append_feature(self, &feature) ;
        scaled = & self->features[self->numFeatures -1] ;
      }

      scaled->laplacianScaleScore = scales[j].score ;
      scaled->frame.a11 *= scales[j].scale ;
      scaled->frame.a21 *= scales[j].scale ;
      scaled->frame.a12 *= scales[j].scale ;
      scaled->frame.a22 *= scales[j].scale ;
    }
  }
  if (dropFeaturesWithoutScale) {
    j = 0 ;
    for (i = 0 ; i < (signed)self->numFeatures ; ++i) {
      VlCovDetFeature feature = self->features[i] ;
      if (feature.peakScore) {
        self->features[j++] = feature ;
      }
    }
    self->numFeatures = j ;
  }

}

/* ---------------------------------------------------------------- */
/*                       Checking that features are inside an image */
/* ---------------------------------------------------------------- */

vl_bool
_vl_covdet_check_frame_inside (VlCovDet * self, VlFrameOrientedEllipse frame, double margin)
{
  double extent = margin ;
  double A [2*2] = {frame.a11, frame.a21, frame.a12, frame.a22} ;
  double T[2] = {frame.x, frame.y} ;
  double x0 = +VL_INFINITY_D ;
  double x1 = -VL_INFINITY_D ;
  double y0 = +VL_INFINITY_D ;
  double y1 = -VL_INFINITY_D ;
  double boxx [4] = {extent, extent, -extent, -extent} ;
  double boxy [4] = {-extent, extent, extent, -extent} ;
  VlScaleSpaceGeometry geom = vl_scalespace_get_geometry(self->gss) ;
  int i ;
  for (i = 0 ; i < 4 ; ++i) {
    double x = A[0] * boxx[i] + A[2] * boxy[i] + T[0] ;
    double y = A[1] * boxx[i] + A[3] * boxy[i] + T[1] ;
    x0 = VL_MIN(x0, x) ;
    x1 = VL_MAX(x1, x) ;
    y0 = VL_MIN(y0, y) ;
    y1 = VL_MAX(y1, y) ;
  }

  return
  0 <= x0 && x1 <= geom.width-1 &&
  0 <= y0 && y1 <= geom.height-1 ;
}

/** @brief Drop features (partially) outside the image
 ** @param self object.
 ** @param margin geometric marging.
 **
 ** The feature extent is defined by @c maring. A bounding box
 ** in the normalised feature frame containin a circle of radius
 ** @a maring is created and mapped to the image by
 ** the feature frame transformation. Then the feature
 ** is dropped if the bounding box is not contained in the image.
 **
 ** For example, setting @c margin to zero drops a feature only
 ** if its center is not contained.
 **
 ** Typically a valua of @c margin equal to 1 or 2 is used.
 **/

void
vl_covdet_drop_features_outside (VlCovDet * self, double margin)
{
  vl_index i, j = 0 ;
  vl_size numFeatures = vl_covdet_get_num_features(self) ;
  for (i = 0 ; i < (signed)numFeatures ; ++i) {
    vl_bool inside =
    _vl_covdet_check_frame_inside (self, self->features[i].frame, margin) ;
    if (inside) {
      self->features[j] = self->features[i] ;
      ++j ;
    }
  }
  self->numFeatures = j ;
}

/* ---------------------------------------------------------------- */
/*                                              Setters and getters */
/* ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- */
/** @brief Get wether images are passed in transposed
 ** @param self object.
 ** @return whether images are transposed.
 **/
vl_bool
vl_covdet_get_transposed (VlCovDet const  * self)
{
  return self->transposed ;
}

/** @brief Set the index of the first octave
 ** @param self object.
 ** @param t whether images are transposed.
 **/
void
vl_covdet_set_transposed (VlCovDet * self, vl_bool t)
{
  self->transposed = t ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the edge threshold
 ** @param self object.
 ** @return the edge threshold.
 **/
double
vl_covdet_get_edge_threshold (VlCovDet const * self)
{
  return self->edgeThreshold ;
}

/** @brief Set the edge threshold
 ** @param self object.
 ** @param edgeThreshold the edge threshold.
 **
 ** The edge threshold must be non-negative.
 **/
void
vl_covdet_set_edge_threshold (VlCovDet * self, double edgeThreshold)
{
  assert(edgeThreshold >= 0) ;
  self->edgeThreshold = edgeThreshold ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the peak threshold
 ** @param self object.
 ** @return the peak threshold.
 **/
double
vl_covdet_get_peak_threshold (VlCovDet const * self)
{
  return self->peakThreshold ;
}

/** @brief Set the peak threshold
 ** @param self object.
 ** @param peakThreshold the peak threshold.
 **
 ** The peak threshold must be non-negative.
 **/
void
vl_covdet_set_peak_threshold (VlCovDet * self, double peakThreshold)
{
  assert(peakThreshold >= 0) ;
  self->peakThreshold = peakThreshold ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the Laplacian peak threshold
 ** @param self object.
 ** @return the Laplacian peak threshold.
 **
 ** This parameter affects only the detecors using the Laplacian
 ** scale selectino method such as Harris-Laplace.
 **/
double
vl_covdet_get_laplacian_peak_threshold (VlCovDet const * self)
{
  return self->lapPeakThreshold ;
}

/** @brief Set the Laplacian peak threshold
 ** @param self object.
 ** @param peakThreshold the Laplacian peak threshold.
 **
 ** The peak threshold must be non-negative.
 **/
void
vl_covdet_set_laplacian_peak_threshold (VlCovDet * self, double peakThreshold)
{
  assert(peakThreshold >= 0) ;
  self->lapPeakThreshold = peakThreshold ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the index of the first octave
 ** @param self object.
 ** @return index of the first octave.
 **/
vl_index
vl_covdet_get_first_octave (VlCovDet const * self)
{
  return self->firstOctave ;
}

/** @brief Set the index of the first octave
 ** @param self object.
 ** @param o index of the first octave.
 **
 ** Calling this function resets the detector.
 **/
void
vl_covdet_set_first_octave (VlCovDet * self, vl_index o)
{
  self->firstOctave = o ;
  vl_covdet_reset(self) ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the octave resolution.
 ** @param self object.
 ** @return octave resolution.
 **/

vl_size
vl_covdet_get_octave_resolution (VlCovDet const * self)
{
  return self->octaveResolution ;
}

/** @brief Set the octave resolutuon.
 ** @param self object.
 ** @param r octave resoltuion.
 **
 ** Calling this function resets the detector.
 **/

void
vl_covdet_set_octave_resolution (VlCovDet * self, vl_size r)
{
  self->octaveResolution = r ;
  vl_covdet_reset(self) ;
}

/* ---------------------------------------------------------------- */
/** @brief Get whether affine adaptation uses accurate smoothing.
 ** @param self object.
 ** @return @c true if accurate smoothing is used.
 **/

vl_bool
vl_covdet_get_aa_accurate_smoothing (VlCovDet const * self)
{
  return self->aaAccurateSmoothing ;
}

/** @brief Set whether affine adaptation uses accurate smoothing.
 ** @param self object.
 ** @param x whether accurate smoothing should be usd.
 **/

void
vl_covdet_set_aa_accurate_smoothing (VlCovDet * self, vl_bool x)
{
  self->aaAccurateSmoothing = x ;
}

/* ---------------------------------------------------------------- */
/** @brief Get the non-extrema suppression threshold
 ** @param self object.
 ** @return threshold.
 **/

double
vl_covdet_get_non_extrema_suppression_threshold (VlCovDet const * self)
{
  return self->nonExtremaSuppression ;
}

/** @brief Set the non-extrema suppression threshod
 ** @param self object.
 ** @param x threshold.
 **/

void
vl_covdet_set_non_extrema_suppression_threshold (VlCovDet * self, double x)
{
  self->nonExtremaSuppression = x ;
}

/** @brief Get the number of non-extrema suppressed
 ** @param self object.
 ** @return number.
 **/

vl_size
vl_covdet_get_num_non_extrema_suppressed (VlCovDet const * self)
{
  return self->numNonExtremaSuppressed ;
}


/* ---------------------------------------------------------------- */
/** @brief Get number of stored frames
 ** @return number of frames stored in the detector.
 **/
vl_size
vl_covdet_get_num_features (VlCovDet const * self)
{
  return self->numFeatures ;
}

/** @brief Get the stored frames
 ** @return frames stored in the detector.
 **/
VlCovDetFeature *
vl_covdet_get_features (VlCovDet * self)
{
  return self->features ;
}

/** @brief Get the Gaussian scale space
 ** @return Gaussian scale space.
 **
 ** A Gaussian scale space exists only after calling ::vl_covdet_put_image.
 ** Otherwise the function returns @c NULL.
 **/

VlScaleSpace *
vl_covdet_get_gss (VlCovDet const * self)
{
  return self->gss ;
}

/** @brief Get the cornerness measure scale space
 ** @return cornerness measure scale space.
 **
 ** A cornerness measure scale space exists only after calling
 ** ::vl_covdet_detect. Otherwise the function returns @c NULL.
 **/

VlScaleSpace *
vl_covdet_get_css (VlCovDet const * self)
{
  return self->css ;
}

/** @brief Get the number of features found with a certain number of scales
 ** @param self object.
 ** @param numScales length of the histogram (out).
 ** @return histogram.
 **
 ** Calling this function makes sense only after running a detector
 ** that uses the Laplacian as a secondary measure for scale
 ** detection
 **/

vl_size const *
vl_covdet_get_laplacian_scales_statistics (VlCovDet const * self,
                                           vl_size * numScales)
{
  *numScales = VL_COVDET_MAX_NUM_LAPLACIAN_SCALES ;
  return self->numFeaturesWithNumScales ;
}
