/** @file hog.c
 ** @brief Histogram of Oriented Gradients (HOG) - Definition
 ** @author Andrea Vedaldi
 **/

/*
 Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
*/

#include "hog.h"
#include "mathop.h"
#include <string.h>

/**

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page hog Histogram of Oriented Gradients (HOG) features
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref hog.h implements the Histogram of Oriented Gradients (HOG) features
in the variants of Dalal Triggs @cite{dalal05histograms} and of UOCTTI
@cite{felzenszwalb09object}. Applications include object detection
and deformable object detection.

- @ref hog-overview
- @ref hog-tech

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section hog-overview Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

HOG is a standard image feature used, among others, in object detection
and deformable object detection. It decomposes the image into square cells
of a given size (typically eight pixels), compute a histogram of oriented
gradient in each cell (similar to @ref sift), and then renormalizes
the cells by looking into adjacent blocks.

VLFeat implements two HOG variants: the original one of Dalal-Triggs
@cite{dalal05histograms} and the one proposed in Felzenszwalb et al.
@cite{felzenszwalb09object}.

In order to use HOG, start by creating a new HOG object, set the desired
parameters, pass a (color or grayscale) image, and read off the results.

@code
VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, numOrientations, VL_FALSE) ;
vl_hog_put_image(hog, image, height, width, numChannels, cellSize) ;
hogWidth = vl_hog_get_width(hog) ;
hogHeight = vl_hog_get_height(hog) ;
hogDimenison = vl_hog_get_dimension(hog) ;
hogArray = vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float)) ;
vl_hog_extract(hog, hogArray) ;
vl_hog_delete(hog) ;
@endcode

HOG is a feature array of the dimension returned by ::vl_hog_get_width,
::vl_hog_get_height, with each feature (histogram) having
dimension ::vl_hog_get_dimension. The array is stored in row major order,
with the slowest varying dimension beying the dimension indexing the histogram
elements.

The number of entreis in the histogram as well as their meaning depends
on the HOG variant and is detailed later. However, it is usually
unnecessary to know such details. @ref hog.h provides support for
creating an inconic representation of a HOG feature array:

@code
glyphSize = vl_hog_get_glyph_size(hog) ;
imageHeight = glyphSize * hogArrayHeight ;
imageWidth = glyphSize * hogArrayWidth ;
image = vl_malloc(sizeof(float)*imageWidth*imageHeight) ;
vl_hog_render(hog, image, hogArray) ;
@endcode

It is often convenient to mirror HOG features from left to right. This
can be obtained by mirroring an array of HOG cells, but the content
of each cell must also be rearranged. This can be done by
the permutation obtaiend by ::vl_hog_get_permutation.

Furthermore, @ref hog.h suppots computing HOG features not from
images but from vector fields.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section hog-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

HOG divdes the input image into square cells of size @c cellSize,
fitting as many cells as possible, filling the image domain from
the upper-left corner down to the right one. For each row and column,
the last cell is at least half contained in the image.
More precisely, the number of cells obtained in this manner is:

@code
hogWidth = (width + cellSize/2) / cellSize ;
hogHeight = (height + cellSize/2) / cellSize ;
@endcode

Then the image gradient @f$ \nabla \ell(x,y) @f$
is computed by using central difference (for colour image
the channel with the largest gradient at that pixel is used).
The gradient @f$ \nabla \ell(x,y) @f$ is assigned to one of @c 2*numOrientations orientation in the
range @f$ [0,2\pi) @f$ (see @ref hog-conventions for details).
Contributions are then accumulated by using bilinear interpolation
to four neigbhour cells, as in @ref sift.
This results in an histogram  @f$h_d@f$ of dimension
2*numOrientations, called of @e directed orientations
since it accounts for the direction as well as the orientation
of the gradient. A second histogram @f$h_u@f$ of undirected orientations
of half the size is obtained by folding @f$ h_d @f$ into two.

Let a block of cell be a @f$ 2\times 2 @f$ sub-array of cells.
Let the norm of a block be the @f$ l^2 @f$ norm of the stacking of the
respective unoriented histogram. Given a HOG cell, four normalisation
factors are then obtained as the inverse of the norm of the four
blocks that contain the cell.

For the Dalal-Triggs variant, each histogram @f$ h_d @f$ is copied
four times, normalised using the four different normalisation factors,
the four vectors are stacked, saturated at 0.2, and finally stored as the descriptor
of the cell. This results in a @c numOrientations * 4 dimensional
cell descriptor. Blocks are visited from left to right and top to bottom
when forming the final descriptor.

For the UOCCTI descriptor, the same is done for both the undirected
as well as the directed orientation histograms. This would yield
a dimension of @c 4*(2+1)*numOrientations elements, but the resulting
vector is projected down to @c (2+1)*numOrientations elements
by averaging corresponding histogram dimensions. This was shown to
be an algebraic approximation of PCA for descriptors computed on natural
images.

In addition, for the UOCTTI variant the l1 norm of each of the
four l2 normalised undirected histograms is computed and stored
as additional four dimensions, for a total of
@c 4+3*numOrientations dimensions.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection hog-conventions Conventions
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The orientation of a gradient is expressed as the angle it forms with the
horizontal axis of the image. Angles are measured clock-wise (as the vertical
image axis points downards), and the null angle corresponds to
an horizontal vector pointing right. The quantized directed
orientations are @f$ \mathrm{k} \pi / \mathrm{numOrientations} @f$, where
@c k is an index that varies in the ingeger
range @f$ \{0, \dots, 2\mathrm{numOrientations} - 1\} @f$.

Note that the orientations capture the orientation of the gradeint;
image edges would be oriented at 90 degrees from these.

**/

/* ---------------------------------------------------------------- */
/** @brief Create a new HOG object
 ** @param variant HOG descriptor variant.
 ** @param numOrientations number of distinguished orientations.
 ** @param transposed wether images are transposed (column major).
 ** @return the new HOG object.
 **
 ** The function creates a new HOG object to extract descriptors of
 ** the prescribed @c variant. The angular resolution is set by
 ** @a numOrientations, which specifies the number of <em>undirected</em>
 ** orientations. The object can work with column major images
 ** by setting @a transposed to true.
 **/

VlHog *
vl_hog_new (VlHogVariant variant, vl_size numOrientations, vl_bool transposed)
{
  vl_index o, k ;
  VlHog * self = vl_calloc(1, sizeof(VlHog)) ;

  assert(numOrientations >= 1) ;

  self->variant = variant ;
  self->numOrientations = numOrientations ;
  self->glyphSize = 21 ;
  self->transposed = transposed ;
  self->useBilinearOrientationAssigment = VL_FALSE ;
  self->orientationX = vl_malloc(sizeof(float) * self->numOrientations) ;
  self->orientationY = vl_malloc(sizeof(float) * self->numOrientations) ;

  /*
   Create a vector along the center of each orientation bin. These
   are used to map gradients to bins. If the image is transposed,
   then this can be adjusted here by swapping X and Y in these
   vectors.
   */
  for(o = 0 ; o < (signed)self->numOrientations ; ++o) {
    double angle = o * VL_PI / self->numOrientations ;
    if (!self->transposed) {
      self->orientationX[o] = (float) cos(angle) ;
      self->orientationY[o] = (float) sin(angle) ;
    } else {
      self->orientationX[o] = (float) sin(angle) ;
      self->orientationY[o] = (float) cos(angle) ;
    }
  }

  /*
   If the number of orientation is equal to 9, one gets:

   Uoccti:: 18 directed orientations + 9 undirected orientations + 4 texture
   DalalTriggs:: 9 undirected orientations x 4 blocks.
   */
  switch (self->variant) {
    case VlHogVariantUoctti:
      self->dimension = 3*self->numOrientations + 4 ;
      break ;

    case VlHogVariantDalalTriggs:
      self->dimension = 4*self->numOrientations ;
      break ;

    default:
      assert(0) ;
  }

  /*
   A permutation specifies how to permute elements in a HOG
   descriptor to flip it horizontally. Since the first orientation
   of index 0 points to the right, this must be swapped with orientation
   self->numOrientation that points to the left (for the directed case,
   and to itself for the undirected one).
   */

  self->permutation = vl_malloc(self->dimension * sizeof(vl_index)) ;
  switch (self->variant) {
    case VlHogVariantUoctti:
      for(o = 0 ; o < (signed)self->numOrientations ; ++o) {
        vl_index op = self->numOrientations - o ;
        self->permutation[o] = op ;
        self->permutation[o + self->numOrientations] = (op + self->numOrientations) % (2*self->numOrientations) ;
        self->permutation[o + 2*self->numOrientations] = (op % self->numOrientations) + 2*self->numOrientations ;
      }
      for (k = 0 ; k < 4 ; ++k) {
        /* The texture features correspond to four displaced block around
         a cell. These permute with a lr flip as for DalalTriggs. */
        vl_index blockx = k % 2 ;
        vl_index blocky = k / 2 ;
        vl_index q = (1 - blockx) + blocky * 2 ;
        self->permutation[k + self->numOrientations * 3] = q + self->numOrientations * 3 ;
      }
      break ;

    case VlHogVariantDalalTriggs:
      for(k = 0 ; k < 4 ; ++k) {
        /* Find the corresponding block. Blocks are listed in order 1,2,3,4,...
           from left to right and top to bottom */
        vl_index blockx = k % 2 ;
        vl_index blocky = k / 2 ;
        vl_index q = (1 - blockx) + blocky * 2 ;
        for(o = 0 ; o < (signed)self->numOrientations ; ++o) {
          vl_index op = self->numOrientations - o ;
          self->permutation[o + k*self->numOrientations] = (op % self->numOrientations) + q*self->numOrientations ;
        }
      }
      break ;

    default:
      assert(0) ;
  }

  /*
   Create glyphs for representing the HOG features/ filters. The glyphs
   are simple bars, oriented orthogonally to the gradients to represent
   image edges. If the object is configured to work on transposed image,
   the glyphs images are also stored in column-major.
   */
  self->glyphs = vl_calloc(self->glyphSize * self->glyphSize * self->numOrientations, sizeof(float)) ;
#define atglyph(x,y,k) self->glyphs[(x) + self->glyphSize * (y) + self->glyphSize * self->glyphSize * (k)]
  for (o = 0 ; o < (signed)self->numOrientations ; ++o) {
    double angle = fmod(o * VL_PI / self->numOrientations + VL_PI/2, VL_PI) ;
    double x2 = self->glyphSize * cos(angle) / 2 ;
    double y2 = self->glyphSize * sin(angle) / 2 ;

    if (angle <= VL_PI / 4 || angle >= VL_PI * 3 / 4) {
      /* along horizontal direction */
      double slope = y2 / x2 ;
      double offset = (1 - slope) * (self->glyphSize - 1) / 2 ;
      vl_index skip = (1 - fabs(cos(angle))) / 2 * self->glyphSize ;
      vl_index i, j ;
      for (i = skip ; i < (signed)self->glyphSize - skip ; ++i) {
        j = vl_round_d(slope * i + offset) ;
        if (! self->transposed) {
          atglyph(i,j,o) = 1 ;
        } else {
          atglyph(j,i,o) = 1 ;
        }
      }
    } else {
      /* along vertical direction */
      double slope = x2 / y2 ;
      double offset = (1 - slope) * (self->glyphSize - 1) / 2 ;
      vl_index skip = (1 - sin(angle)) / 2 * self->glyphSize ;
      vl_index i, j ;
      for (j = skip ; j < (signed)self->glyphSize - skip; ++j) {
        i = vl_round_d(slope * j + offset) ;
        if (! self->transposed) {
          atglyph(i,j,o) = 1 ;
        } else {
          atglyph(j,i,o) = 1 ;
        }
      }
    }
  }
  return self ;
}

/* ---------------------------------------------------------------- */
/** @brief Delete a HOG object
 ** @param self HOG object to delete.
 **/

void
vl_hog_delete (VlHog * self)
{
  if (self->orientationX) {
    vl_free(self->orientationX) ;
    self->orientationX = NULL ;
  }

  if (self->orientationY) {
    vl_free(self->orientationY) ;
    self->orientationY = NULL ;
  }

  if (self->glyphs) {
    vl_free(self->glyphs) ;
    self->glyphs = NULL ;
  }

  if (self->permutation) {
    vl_free(self->permutation) ;
    self->permutation = NULL ;
  }

  if (self->hog) {
    vl_free(self->hog) ;
    self->hog = NULL ;
  }

  if (self->hogNorm) {
    vl_free(self->hogNorm) ;
    self->hogNorm = NULL ;
  }

  vl_free(self) ;
}


/* ---------------------------------------------------------------- */
/** @brief Get HOG glyph size
 ** @param self HOG object.
 ** @return size (height and width) of a glyph.
 **/

vl_size
vl_hog_get_glyph_size (VlHog const * self)
{
  return self->glyphSize ;
}

/* ---------------------------------------------------------------- */
/** @brief Get HOG left-right flip permutation
 ** @param self HOG object.
 ** @return left-right permutation.
 **
 ** The function returns a pointer to an array @c permutation of ::vl_hog_get_dimension
 ** elements. Given a HOG descriptor (for a cell) @c hog, which is also
 ** a vector of ::vl_hog_get_dimension elements, the
 ** descriptor obtained for the same image flipped horizotnally is
 ** given by <code>flippedHog[i] = hog[permutation[i]]</code>.
 **/

vl_index const *
vl_hog_get_permutation (VlHog const * self)
{
  return self->permutation ;
}

/* ---------------------------------------------------------------- */
/** @brief Turn bilinear interpolation of assignments on or off
 ** @param self HOG object.
 ** @param x @c true if orientations should be assigned with bilinear interpolation.
 **/

void
vl_hog_set_use_bilinear_orientation_assignments (VlHog * self, vl_bool x) {
  self->useBilinearOrientationAssigment = x ;
}

/** @brief Tell whether assignments use bilinear interpolation or not
 ** @param self HOG object.
 ** @return @c true if orientations are be assigned with bilinear interpolation.
 **/

vl_bool
vl_hog_get_use_bilinear_orientation_assignments (VlHog const * self) {
  return self->useBilinearOrientationAssigment ;
}

/* ---------------------------------------------------------------- */
/** @brief Render a HOG descriptor to a glyph image
 ** @param self HOG object.
 ** @param image glyph image (output).
 ** @param descriptor HOG descriptor.
 ** @param width HOG descriptor width.
 ** @param height HOG descriptor height.
 **
 ** The function renders the HOG descriptor or filter
 ** @a descriptor as an image (for visualization) and stores the result in
 ** the buffer @a image. This buffer
 ** must be an array of dimensions @c width*glyphSize
 ** by @c height*glyphSize elements, where @c glyphSize is
 ** obtained from ::vl_hog_get_glyph_size and is the size in pixels
 ** of the image element used to represent the descriptor of one
 ** HOG cell.
 **/

void
vl_hog_render (VlHog const * self,
               float * image,
               float const * descriptor,
               vl_size width,
               vl_size height)
{
  vl_index x, y, k, cx, cy ;
  vl_size hogStride = width * height ;

  assert(self) ;
  assert(image) ;
  assert(descriptor) ;
  assert(width > 0) ;
  assert(height > 0) ;

  for (y = 0 ; y < (signed)height ; ++y) {
    for (x = 0 ; x < (signed)width ; ++x) {
      float minWeight = 0 ;
      float maxWeight = 0 ;

      for (k = 0 ; k < (signed)self->numOrientations ; ++k) {
        float weight ;
        float const * glyph = self->glyphs + k * (self->glyphSize*self->glyphSize) ;
        float * glyphImage = image + self->glyphSize * x + y * width * (self->glyphSize*self->glyphSize) ;

        switch (self->variant) {
          case VlHogVariantUoctti:
            weight =
            descriptor[k * hogStride] +
            descriptor[(k + self->numOrientations) * hogStride] +
            descriptor[(k + 2 * self->numOrientations) * hogStride] ;
            break ;
          case VlHogVariantDalalTriggs:
            weight =
            descriptor[k * hogStride] +
            descriptor[(k + self->numOrientations) * hogStride] +
            descriptor[(k + 2 * self->numOrientations) * hogStride] +
            descriptor[(k + 3 * self->numOrientations) * hogStride] ;
            break ;
          default:
            abort() ;
        }
        maxWeight = VL_MAX(weight, maxWeight) ;
        minWeight = VL_MIN(weight, minWeight);

        for (cy = 0 ; cy < (signed)self->glyphSize ; ++cy) {
          for (cx = 0 ; cx < (signed)self->glyphSize ; ++cx) {
            *glyphImage++ += weight * (*glyph++) ;
          }
          glyphImage += (width - 1) * self->glyphSize  ;
        }
      } /* next orientation */

      {
        float * glyphImage = image + self->glyphSize * x + y * width * (self->glyphSize*self->glyphSize) ;
        for (cy = 0 ; cy < (signed)self->glyphSize ; ++cy) {
          for (cx = 0 ; cx < (signed)self->glyphSize ; ++cx) {
            float value = *glyphImage ;
            *glyphImage++ = VL_MAX(minWeight, VL_MIN(maxWeight, value)) ;
          }
          glyphImage += (width - 1) * self->glyphSize  ;
        }
      }

      ++ descriptor ;
    } /* next column of cells (x) */
  } /* next row of cells (y) */
}

/* ---------------------------------------------------------------- */
/** @brief Get the dimension of the HOG features
 ** @param self HOG object.
 ** @return imension of a HOG cell descriptors.
 **/

vl_size
vl_hog_get_dimension (VlHog const * self)
{
  return self->dimension ;
}

/** @brief Get the width of the HOG cell array
 ** @param self HOG object.
 ** @return number of HOG cells in the horizontal direction.
 **/

vl_size
vl_hog_get_width (VlHog * self)
{
  return self->hogWidth ;
}

/** @brief Get the height of the HOG cell array
 ** @param self HOG object.
 ** @return number of HOG cells in the vertical direction.
 **/

vl_size
vl_hog_get_height (VlHog * self)
{
  return self->hogHeight ;
}

/* ---------------------------------------------------------------- */
/** @internal @brief Prepare internal buffers
 ** @param self HOG object.
 ** @param width image width.
 ** @param height image height.
 ** @param cellSize size of a HOG cell.
 **/

static void
vl_hog_prepare_buffers (VlHog * self, vl_size width, vl_size height, vl_size cellSize)
{
  vl_size hogWidth = (width + cellSize/2) / cellSize ;
  vl_size hogHeight = (height + cellSize/2) / cellSize ;

  assert(width > 3) ;
  assert(height > 3) ;
  assert(hogWidth > 0) ;
  assert(hogHeight > 0) ;

  if (self->hog &&
      self->hogWidth == hogWidth &&
      self->hogHeight == hogHeight) {
    /* a suitable buffer is already allocated */
    memset(self->hog, 0, sizeof(float) * hogWidth * hogHeight * self->numOrientations * 2) ;
    memset(self->hogNorm, 0, sizeof(float) * hogWidth * hogHeight) ;
    return ;
  }

  if (self->hog) {
    vl_free(self->hog) ;
    self->hog = NULL ;
  }

  if (self->hogNorm) {
    vl_free(self->hogNorm) ;
    self->hogNorm = NULL ;
  }

  self->hog = vl_calloc(hogWidth * hogHeight * self->numOrientations * 2, sizeof(float)) ;
  self->hogNorm = vl_calloc(hogWidth * hogHeight, sizeof(float)) ;
  self->hogWidth = hogWidth ;
  self->hogHeight = hogHeight ;
}

/* ---------------------------------------------------------------- */
/** @brief Process features starting from an image
 ** @param self HOG object.
 ** @param image image to process.
 ** @param width image width.
 ** @param height image height.
 ** @param numChannels number of image channles.
 ** @param cellSize size of a HOG cell.
 **
 ** The buffer @c hog must be a three-dimensional array.
 ** The first two dimensions are @c (width + cellSize/2)/cellSize and
 ** @c (height + cellSize/2)/cellSize, where divisions are integer.
 ** This is approximately @c width/cellSize and @c height/cellSize,
 ** adjusted so that the last cell is at least half contained in the
 ** image.
 **
 ** The image @c width and @c height must be not smaller than three
 ** pixels and not smaller than @c cellSize.
 **/

void
vl_hog_put_image (VlHog * self,
                  float const * image,
                  vl_size width, vl_size height, vl_size numChannels,
                  vl_size cellSize)
{
  vl_size hogStride ;
  vl_size channelStride = width * height ;
  vl_index x, y ;
  vl_uindex k ;

  assert(self) ;
  assert(image) ;

  /* clear features */
  vl_hog_prepare_buffers(self, width, height, cellSize) ;
  hogStride = self->hogWidth * self->hogHeight ;

#define at(x,y,k) (self->hog[(x) + (y) * self->hogWidth + (k) * hogStride])

  /* compute gradients and map the to HOG cells by bilinear interpolation */
  for (y = 1 ; y < (signed)height - 1 ; ++y) {
    for (x = 1 ; x < (signed)width - 1 ; ++x) {
      float gradx = 0 ;
      float grady = 0 ;
      float gradNorm ;
      float orientationWeights [2] = {-1, -1} ;
      vl_index orientationBins [2] = {-1, -1} ;
      vl_index orientation = 0 ;
      float hx, hy, wx1, wx2, wy1, wy2 ;
      vl_index binx, biny, o ;

      /*
       Compute the gradient at (x,y). The image channel with
       the maximum gradient at each location is selected.
       */
      {
        float const * iter = image + y * width + x ;
        float gradNorm2 = 0 ;
        for (k = 0 ; k < numChannels ; ++k) {
          float gradx_ = *(iter + 1) - *(iter - 1) ;
          float grady_ = *(iter + width)  - *(iter - width) ;
          float gradNorm2_ = gradx_ * gradx_ + grady_ * grady_ ;
          if (gradNorm2_ > gradNorm2) {
            gradx = gradx_ ;
            grady = grady_ ;
            gradNorm2 = gradNorm2_ ;
          }
          iter += channelStride ;
        }
        gradNorm = sqrtf(gradNorm2) ;
      }

      /*
       Map the gradient to the closest and second closets orientation bins.
       There are numOrientations orientation in the interval [0,pi).
       The next numOriantations are the symmetric ones, for a total
       of 2*numOrientation directed orientations.
       */
      for (k = 0 ; k < self->numOrientations ; ++k) {
        float orientationScore_ = gradx * self->orientationX[k] +  grady * self->orientationY[k] ;
        vl_index orientationBin_ = k ;
        if (orientationScore_ < 0) {
          orientationScore_ = - orientationScore_ ;
          orientationBin_ += self->numOrientations ;
        }
        if (orientationScore_ > orientationWeights[0]) {
          orientationBins[1] = orientationBins[0] ;
          orientationWeights[1] = orientationWeights[0] ;
          orientationBins[0] = orientationBin_ ; ;
          orientationWeights[0] = orientationScore_ ;
        } else if (orientationScore_ > orientationWeights[1]) {
          orientationBins[1] = orientationBin_ ;
          orientationWeights[1] = orientationScore_ ;
        }
      }

      if (self->useBilinearOrientationAssigment) {
        /* min(1.0,...) guards against small overflows causing NaNs */
        float angle0 = acosf(VL_MIN(orientationWeights[0] / VL_MAX(gradNorm, 1e-10),1.0)) ;
        orientationWeights[1] = angle0 / (VL_PI / self->numOrientations) ;
        orientationWeights[0] = 1 - orientationWeights[1] ;
      } else {
        orientationWeights[0] = 1 ;
        orientationBins[1] = -1 ;
      }

      for (o = 0 ; o < 2 ; ++o) {
        float ow ;
        /*
         Accumulate the gradient. hx is the distance of the
         pixel x to the cell center at its left, in units of cellSize.
         With this parametrixation, a pixel on the cell center
         has hx = 0, which gradually increases to 1 moving to the next
         center.
         */

        orientation = orientationBins[o] ;
        if (orientation < 0) continue ;

        /*  (x - (w-1)/2) / w = (x + 0.5)/w - 0.5 */
        hx = (x + 0.5) / cellSize - 0.5 ;
        hy = (y + 0.5) / cellSize - 0.5 ;
        binx = vl_floor_f(hx) ;
        biny = vl_floor_f(hy) ;
        wx2 = hx - binx ;
        wy2 = hy - biny ;
        wx1 = 1.0 - wx2 ;
        wy1 = 1.0 - wy2 ;

        ow = orientationWeights[o] ;

        /*VL_PRINTF("%d %d - %d %d %f %f - %f %f %f %f - %d \n ",x,y,binx,biny,hx,hy,wx1,wx2,wy1,wy2,o);*/

        if (binx >= 0 && biny >=0) {
          at(binx,biny,orientation) += gradNorm * ow * wx1 * wy1 ;
        }
        if (binx < (signed)self->hogWidth - 1 && biny >=0) {
          at(binx+1,biny,orientation) += gradNorm * ow * wx2 * wy1 ;
        }
        if (binx < (signed)self->hogWidth - 1 && biny < (signed)self->hogHeight - 1) {
          at(binx+1,biny+1,orientation) += gradNorm * ow * wx2 * wy2 ;
        }
        if (binx >= 0 && biny < (signed)self->hogHeight - 1) {
          at(binx,biny+1,orientation) += gradNorm * ow * wx1 * wy2 ;
        }
      } /* next o */
    } /* next x */
  } /* next y */
}

/* ---------------------------------------------------------------- */
/** @brief Process features starting from a field in polar notation
 ** @param self HOG object.
 ** @param modulus image gradient modulus.
 ** @param angle image gradient angle.
 ** @param directed wrap the gradient angles at 2pi (directed) or pi (undirected).
 ** @param width image width.
 ** @param height image height.
 ** @param cellSize size of a HOG cell.
 **
 ** The function behaves like ::vl_hog_put_image, but foregoes the internal
 ** computation of the gradient field, allowing the user to specify
 ** their own. Angles are measure clockwise, the y axis pointing downwards,
 ** starting from the x axis (pointing to the right).
 **/

void vl_hog_put_polar_field (VlHog * self,
                             float const * modulus,
                             float const * angle,
                             vl_bool directed,
                             vl_size width, vl_size height,
                             vl_size cellSize)
{
  vl_size hogStride ;
  vl_index x, y, o ;
  vl_index period = self->numOrientations * (directed ? 2 : 1) ;
  double angleStep = VL_PI / self->numOrientations ;

  assert(self) ;
  assert(modulus) ;
  assert(angle) ;

  /* clear features */
  vl_hog_prepare_buffers(self, width, height, cellSize) ;
  hogStride = self->hogWidth * self->hogHeight ;

#define at(x,y,k) (self->hog[(x) + (y) * self->hogWidth + (k) * hogStride])
#define atNorm(x,y) (self->hogNorm[(x) + (y) * self->hogWidth])

  /* fill HOG cells from gradient field */
  for (y = 0 ; y < (signed)height ; ++y) {
    for (x = 0 ; x < (signed)width ; ++x) {
      float ho, hx, hy, wo1, wo2, wx1, wx2, wy1, wy2 ;
      vl_index bino, binx, biny ;
      float orientationWeights [2] = {0,0} ;
      vl_index orientationBins [2] = {-1,-1} ;
      vl_index orientation = 0 ;
      float thisAngle = *angle++ ;
      float thisModulus = *modulus++ ;

      if (thisModulus <= 0.0f) continue ;

      /*  (x - (w-1)/2) / w = (x + 0.5)/w - 0.5 */

      ho = (float)thisAngle / angleStep ;
      bino = vl_floor_f(ho) ;
      wo2 = ho - bino ;
      wo1 = 1.0f - wo2 ;

      while (bino < 0) { bino += self->numOrientations * 2 ; }

      if (self->useBilinearOrientationAssigment) {
        orientationBins[0] = bino % period ;
        orientationBins[1] = (bino + 1) % period ;
        orientationWeights[0] = wo1 ;
        orientationWeights[1] = wo2 ;
      } else {
        orientationBins[0] = (bino + ((wo1 > wo2) ? 0 : 1)) % period ;
        orientationWeights[0] = 1 ;
        orientationBins[1] = -1 ;
      }

      for (o = 0 ; o < 2 ; ++o) {
        /*
         Accumulate the gradient. hx is the distance of the
         pixel x to the cell center at its left, in units of cellSize.
         With this parametrixation, a pixel on the cell center
         has hx = 0, which gradually increases to 1 moving to the next
         center.
         */

        orientation = orientationBins[o] ;
        if (orientation < 0) continue ;

        hx = (x + 0.5) / cellSize - 0.5 ;
        hy = (y + 0.5) / cellSize - 0.5 ;
        binx = vl_floor_f(hx) ;
        biny = vl_floor_f(hy) ;
        wx2 = hx - binx ;
        wy2 = hy - biny ;
        wx1 = 1.0 - wx2 ;
        wy1 = 1.0 - wy2 ;

        wx1 *= orientationWeights[o] ;
        wx2 *= orientationWeights[o] ;
        wy1 *= orientationWeights[o] ;
        wy2 *= orientationWeights[o] ;

        /*VL_PRINTF("%d %d - %d %d %f %f - %f %f %f %f - %d \n ",x,y,binx,biny,hx,hy,wx1,wx2,wy1,wy2,o);*/

        if (binx >= 0 && biny >=0) {
          at(binx,biny,orientation) += thisModulus * wx1 * wy1 ;
        }
        if (binx < (signed)self->hogWidth - 1 && biny >=0) {
          at(binx+1,biny,orientation) += thisModulus * wx2 * wy1 ;
        }
        if (binx < (signed)self->hogWidth - 1 && biny < (signed)self->hogHeight - 1) {
          at(binx+1,biny+1,orientation) += thisModulus * wx2 * wy2 ;
        }
        if (binx >= 0 && biny < (signed)self->hogHeight - 1) {
          at(binx,biny+1,orientation) += thisModulus * wx1 * wy2 ;
        }
      } /* next o */
    } /* next x */
  } /* next y */
}

/* ---------------------------------------------------------------- */
/** @brief Extract HOG features
 ** @param self HOG object.
 ** @param features HOG features (output).
 **
 ** This method is called after ::vl_hog_put_image or ::vl_hog_put_polar_field
 ** in order to retrieve the computed HOG features. The buffer @c features must have the dimensions returned by
 ** ::vl_hog_get_width, ::vl_hog_get_height, and ::vl_hog_get_dimension.
 **/

void
vl_hog_extract (VlHog * self, float * features)
{
  vl_index x, y ;
  vl_uindex k ;
  vl_size hogStride = self->hogWidth * self->hogHeight ;

  assert(features) ;

#define at(x,y,k) (self->hog[(x) + (y) * self->hogWidth + (k) * hogStride])
#define atNorm(x,y) (self->hogNorm[(x) + (y) * self->hogWidth])

  /*
   Compute the squared L2 norm of the unoriented version of each HOG
   cell histogram. The unoriented version is obtained by folding
   the 2*numOrientations compotnent into numOrientations only.
   */
  {
    float const * iter = self->hog ;
    for (k = 0 ; k < self->numOrientations ; ++k) {
      float * niter = self->hogNorm ;
      float * niterEnd = self->hogNorm + self->hogWidth * self->hogHeight ;
      vl_size stride = self->hogWidth*self->hogHeight*self->numOrientations ;
      while (niter != niterEnd) {
        float h1 = *iter ;
        float h2 = *(iter + stride) ;
        float h = h1 + h2 ;
        *niter += h * h ;
        niter++ ;
        iter++ ;
      }
    }
  }

  /*
   HOG block-normalisation.

   The Dalal-Triggs implementation computes a normalized descriptor for
   each block of 2x2 cells, by stacking the histograms of each cell
   into a vector and L2-normalizing and truncating the result.
   
   Each block-level descriptor is then decomposed back into cells
   and corresponding parts are stacked into cell-level descritpors.
   Each HOG cell is contained in exactly
   four 2x2 cell blocks. For example, the cell number 5 in the following
   figure is contained in blocks 1245, 2356, 4578, 5689:

   +---+---+---+
   | 1 | 2 | 3 |
   +---+---+---+
   | 4 | 5 | 6 |
   +---+---+---+
   | 7 | 8 | 9 |
   +---+---+---+

   Hence, when block-level descriptors are decomposed back
   into cells, each cell receives contributions from four blocks. So,
   if each cell started with a D-dimensional histogram, it
   ends up with a 4D dimesional descriptor vector.

   Note however that this is just a convenient way of rewriting the 
   blocks as per-cell contributions, but the block information
   is unchanged. In particular, barring boundary effects,
   in an array of H x W cells there are approximately HW blocks;
   hence the L2 norm of all the blocks stacked is approximately HW
   (because individual blocks are L2-normalized). Since this does
   not change in the final HOG descriptor,
   the L2 norm of the HOG descriptor of an image should be approximately
   the same as the area of the image divided by the
   area of a HOG cell. This can be used as a sanity check.

   The UoCTTI variant differs in some non-negligible ways. First, 
   it includes both oriented and unoriented histograms, as well
   as four components capturing texture. Second, and most importantly, 
   it merges the four chunks of block-level descirptors landing in
   each cell into one by taking their average. This makes sense
   because, ultimately, these four sub-descriptors are identical
   to the original cell histogram, just with four different normalisations
   applied.
   */
  {
    float const * iter = self->hog ;
    for (y = 0 ; y < (signed)self->hogHeight ; ++y) {
      for (x = 0 ; x < (signed)self->hogWidth ; ++x) {

        /* norm of upper-left, upper-right, ... cells */
        vl_index xm = VL_MAX(x - 1, 0) ;
        vl_index xp = VL_MIN(x + 1, (signed)self->hogWidth - 1) ;
        vl_index ym = VL_MAX(y - 1, 0) ;
        vl_index yp = VL_MIN(y + 1, (signed)self->hogHeight - 1) ;

        double norm1 = atNorm(xm,ym) ;
        double norm2 = atNorm(x,ym) ;
        double norm3 = atNorm(xp,ym) ;
        double norm4 = atNorm(xm,y) ;
        double norm5 = atNorm(x,y) ;
        double norm6 = atNorm(xp,y) ;
        double norm7 = atNorm(xm,yp) ;
        double norm8 = atNorm(x,yp) ;
        double norm9 = atNorm(xp,yp) ;

        double factor1, factor2, factor3, factor4 ;

        double t1 = 0 ;
        double t2 = 0 ;
        double t3 = 0 ;
        double t4 = 0 ;

        float * oiter = features + x + self->hogWidth * y ;

        /* each factor is the inverse of the l2 norm of one of the 2x2 blocks surrounding
           cell x,y */
#if 0
        if (self->transposed) {
          /* if the image is transposed, y and x are swapped */
          factor1 = 1.0 / VL_MAX(sqrt(norm1 + norm2 + norm4 + norm5), 1e-10) ;
          factor3 = 1.0 / VL_MAX(sqrt(norm2 + norm3 + norm5 + norm6), 1e-10) ;
          factor2 = 1.0 / VL_MAX(sqrt(norm4 + norm5 + norm7 + norm8), 1e-10) ;
          factor4 = 1.0 / VL_MAX(sqrt(norm5 + norm6 + norm8 + norm9), 1e-10) ;
        } else {
          factor1 = 1.0 / VL_MAX(sqrt(norm1 + norm2 + norm4 + norm5), 1e-10) ;
          factor2 = 1.0 / VL_MAX(sqrt(norm2 + norm3 + norm5 + norm6), 1e-10) ;
          factor3 = 1.0 / VL_MAX(sqrt(norm4 + norm5 + norm7 + norm8), 1e-10) ;
          factor4 = 1.0 / VL_MAX(sqrt(norm5 + norm6 + norm8 + norm9), 1e-10) ;
        }
#else
        /* as implemented in UOCTTI code */
        if (self->transposed) {
          /* if the image is transposed, y and x are swapped */
          factor1 = 1.0 / sqrt(norm1 + norm2 + norm4 + norm5 + 1e-4) ;
          factor3 = 1.0 / sqrt(norm2 + norm3 + norm5 + norm6 + 1e-4) ;
          factor2 = 1.0 / sqrt(norm4 + norm5 + norm7 + norm8 + 1e-4) ;
          factor4 = 1.0 / sqrt(norm5 + norm6 + norm8 + norm9 + 1e-4) ;
        } else {
          factor1 = 1.0 / sqrt(norm1 + norm2 + norm4 + norm5 + 1e-4) ;
          factor2 = 1.0 / sqrt(norm2 + norm3 + norm5 + norm6 + 1e-4) ;
          factor3 = 1.0 / sqrt(norm4 + norm5 + norm7 + norm8 + 1e-4) ;
          factor4 = 1.0 / sqrt(norm5 + norm6 + norm8 + norm9 + 1e-4) ;
        }
#endif

        for (k = 0 ; k < self->numOrientations ; ++k) {
          double ha = iter[hogStride * k] ;
          double hb = iter[hogStride * (k + self->numOrientations)] ;
          double hc ;

          double ha1 = factor1 * ha ;
          double ha2 = factor2 * ha ;
          double ha3 = factor3 * ha ;
          double ha4 = factor4 * ha ;

          double hb1 = factor1 * hb ;
          double hb2 = factor2 * hb ;
          double hb3 = factor3 * hb ;
          double hb4 = factor4 * hb ;

          double hc1 = ha1 + hb1 ;
          double hc2 = ha2 + hb2 ;
          double hc3 = ha3 + hb3 ;
          double hc4 = ha4 + hb4 ;

          ha1 = VL_MIN(0.2, ha1) ;
          ha2 = VL_MIN(0.2, ha2) ;
          ha3 = VL_MIN(0.2, ha3) ;
          ha4 = VL_MIN(0.2, ha4) ;

          hb1 = VL_MIN(0.2, hb1) ;
          hb2 = VL_MIN(0.2, hb2) ;
          hb3 = VL_MIN(0.2, hb3) ;
          hb4 = VL_MIN(0.2, hb4) ;

          hc1 = VL_MIN(0.2, hc1) ;
          hc2 = VL_MIN(0.2, hc2) ;
          hc3 = VL_MIN(0.2, hc3) ;
          hc4 = VL_MIN(0.2, hc4) ;

          t1 += hc1 ;
          t2 += hc2 ;
          t3 += hc3 ;
          t4 += hc4 ;

          switch (self->variant) {
            case VlHogVariantUoctti :
              ha = 0.5 * (ha1 + ha2 + ha3 + ha4) ;
              hb = 0.5 * (hb1 + hb2 + hb3 + hb4) ;
              hc = 0.5 * (hc1 + hc2 + hc3 + hc4) ;
              *oiter = ha ;
              *(oiter + hogStride * self->numOrientations) = hb ;
              *(oiter + 2 * hogStride * self->numOrientations) = hc ;
              break ;

            case VlHogVariantDalalTriggs :
              *oiter = hc1 ;
              *(oiter + hogStride * self->numOrientations) = hc2 ;
              *(oiter + 2 * hogStride * self->numOrientations) = hc3 ;
              *(oiter + 3 * hogStride * self->numOrientations) = hc4 ;
              break ;
          }
          oiter += hogStride ;

        } /* next orientation */

        switch (self->variant) {
          case VlHogVariantUoctti :
            oiter += 2 * hogStride * self->numOrientations ;
            *oiter = (1.0f/sqrtf(18.0f)) * t1 ; oiter += hogStride ;
            *oiter = (1.0f/sqrtf(18.0f)) * t2 ; oiter += hogStride ;
            *oiter = (1.0f/sqrtf(18.0f)) * t3 ; oiter += hogStride ;
            *oiter = (1.0f/sqrtf(18.0f)) * t4 ; oiter += hogStride ;
            break ;

          case VlHogVariantDalalTriggs :
            break ;
        }
        ++iter ;
      } /* next x */
    } /* next y */
  } /* block normalization */
}

