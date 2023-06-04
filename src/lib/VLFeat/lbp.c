/** @file lbp.c
 ** @brief Local Binary Patterns (LBP) - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 Andrea Vedaldi.
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page lbp Local Binary Patterns (LBP) descriptor
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref lbp.h implements the Local Binary Pattern (LBP) feature
descriptor.  The LBP descriptor @cite{ojala10multiresolution} is a
histogram of quantized LBPs pooled in a local image neighborhood. @ref
lbp-starting demonstrates how to use the C API to compute the LBP
descriptors of an image. For further details refer to:

- @subpage lbp-fundamentals - LBP definition and parameters.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section lbp-starting Getting started with LBP
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

To compute the LBP descriptor of an image, start by creating a ::VlLbp
object instance by specifying the type of LBP quantization. Given the
configure LBP object, then call ::vl_lbp_process to process a
grayscale image and obtain the corresponding LBP descriptors. This
function expects as input a buffer large enough to contain the
computed features. If the image has size @c width x @c height, there
are exactly @c floor(width/cellSize) x @c floor(height/cellSize)
cells, each of which has a histogram of LBPs of size @c dimension (as
returned by ::vl_lbp_get_dimension). Thus the required buffer has size
@c floor(width/cellSize) x @c floor(height/cellSize) x @c dimension.

::VlLbp supports computing transposed LPBs as well. A transposed LBP
is the LBP obtained by transposing the input image (regarded as a
matrix). This functionality can be useful to compute the features when
the input image is stored in column major format (e.g. MATLAB) rather
than row major.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page lbp-fundamentals Local Binary Patterns fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A *Locally Binary Pattern* (LBP) is a local descriptor that captures
the appearance of an image in a small neighborhood around a pixel.  An
LBP is a string of bits, with one bit for each of the pixels in the
neighborhood. Each bit is turned on or off depending on whether the
intensity of the corresponding pixel is greater than the intensity of
the central pixel. LBP are seldom used directly, however. Instead, the
binary string thus produced are further quantized (@ref
lbp-quantization) and pooled in local histograms (@ref
lbp-histograms).

While many variants are possible, ::VlLbp implements only the case of
3 &times; 3 pixel neighborhoods (this setting was found to be optimal
in several applications). In particular, the LBP centered on pixel
$(x,y)$ is a string of eight bits. Each bit is equal to one if the
corresponding pixel is brighter than the central one. Pixels are
scanned starting from the one to the right in anti-clockwise order.
For example the first bit is one if, and only if, $I(x+1,y) >
I(x,y)$, and the second bit is one if, and only if, $I(x+1,y-1) >
I(x,y)$.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section lbp-quantization Quantized LBP
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

For a 3 &times; 3 neighborhood, an LBP is a string of eight bits and
so there are 256 possible LBPs. These are usually too many for a
reliable statistics (histogram) to be computed. Therefore the 256
patterns are further quantized into a smaller number of patterns
according to one of the following rules:

- <b>Uniform</b> (::VlLbpUniform) There is one quantized pattern for
  each LBP that has exactly a transitions from 0 to 1 and one from 1
  to 0 when scanned in anti-clockwise order, plus one quantized
  pattern comprising the two uniform LBPs, and one quantized pattern
  comprising all the other LBPs. This yields a total of 58 quantized
  patterns.

  @image html lbp.png "LBP quantized patterns."

The number of quantized LBPs, which depends on the quantization type,
can be obtained by ::vl_lbp_get_dimension.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section lbp-histograms Histograms of LBPs
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The quantized LBP patterns are further grouped into local
histograms. The image is divided into a number of cells of a
prescribed size (as specified by the parameter @c cellSize passed to
::vl_lbp_process as described in @ref lbp-starting). Then the
quantized LBPs are aggregated into histogram by using bilinear
interpolation along the two spatial dimensions (similar to HOG and
SIFT).
**/

#include "lbp.h"
#include "mathop.h"
#include "string.h"

/* ---------------------------------------------------------------- */
/*                                           Initialization helpers */
/* ---------------------------------------------------------------- */

/*
 This function creates the LBP quantization table for the uniform LBP
 patterns. The purpose of this lookup table is to map a 8-bit LBP
 strings to one of 58 uniform pattern codes.

 Pixels in the 8-neighbourhoods are read in counterclockwise order
 starting from the east direction, as follows:

 NW(5)  N(6) NE(7)
 W(4)         E(0)  -> b0 b1 b2 b3 b4 b5 b6 b7
 SW(3)  S(2) SE(1)

 There are 256 such strings, indexing the lookup table. The table
 contains the corresponding code, effectively quantizing the 256
 patterns into 58. There is one bin for constant patterns (all zeros
 or ones), 8*7 for the uniform ones, and one for all other.

 A uniform pattern is a circular sequence of bit b0b1...b7 such that
 there is exactly one switch from 0 to 1 and one from 1 to 0.  These
 uniform patterns are enumerated as follows. The slowest varying index
 i (0...7) points to the first bit that is on and the slowest varying
 index j (1...7) to the length of the run of bits equal to one,
 resulting in the sequence:

 0:  1000 0000
 1:  1100 0000
 ...
 7:  1111 1110
 8:  0100 0000
 9:  0110 0000
 ...
 56: 1111 1101

 The function also accounts for when the image is stored in transposed
 format. The sampling function is unchanged, so that the first bit to
 be read is not the one to the east, but the one to the south, and
 overall the following sequence is read:

 NW(5)  W(4) SW(3)
 N(6)         S(2)  -> b2 b1 b0 b7 b6 b5 b4 b3
 NE(7)  E(0) SE(1)

 In enumerating the uniform patterns, the index j is unchanged as it
 encodes the runlenght. On the contrary, the index i changes to
 account for the transposition and for the fact that the beginning and
 ending of the run are swapped. With modular arithmetic, the i must be
 transformed as

 ip = - i + 2 - (j - 1)
 */

static void
_vl_lbp_init_uniform(VlLbp * self)
{
  int i, j ;

  /* overall number of quantized LBPs */
  self->dimension = 58 ;

  /* all but selected patterns map to bin 57 (the first bin has index 0) */
  for (i = 0 ; i < 256 ; ++i) {
    self->mapping[i] = 57 ;
  }

  /* the uniform (all zeros or ones) patterns map to bin 56 */
  self->mapping[0x00] = 56 ;
  self->mapping[0xff] = 56 ;

  /* 56 uniform patterns */
  for (i = 0 ; i < 8 ; ++i) {
    for (j = 1 ; j <= 7 ; ++j) {
      int ip ;
      int unsigned string ;
      if (self->transposed) {
        ip = (- i + 2 - (j - 1) + 16) % 8 ;
      } else {
        ip = i ;
      }

      /* string starting with j ones */
      string = (1 << j) - 1 ;
      string <<= ip ;
      string = (string | (string >> 8)) & 0xff ;

      self->mapping[string] = i * 7 + (j-1) ;
    }
  }
}

/* ---------------------------------------------------------------- */

/** @brief Create a new LBP object
 ** @param type type of LBP features.
 ** @param transposed if @c true, then transpose each LBP pattern.
 ** @return new VlLbp object instance.
 **/

VlLbp *
vl_lbp_new(VlLbpMappingType type, vl_bool transposed)
{
  VlLbp * self = vl_malloc(sizeof(VlLbp)) ;
  if (self == NULL) {
    vl_set_last_error(VL_ERR_ALLOC, NULL) ;
    return NULL ;
  }
  self->transposed = transposed ;
  switch (type) {
    case VlLbpUniform: _vl_lbp_init_uniform(self) ; break ;
    default: exit(1) ;
  }
  return self ;
}

/** @brief Delete VlLbp object
 ** @param self object to delete.
 **/

void
vl_lbp_delete(VlLbp * self) {
  vl_free(self) ;
}

/** @brief Get the dimension of the LBP histograms
 ** @return dimension of the LBP histograms.
 ** The dimension depends on the type of quantization used.
 ** @see ::vl_lbp_new().
 **/

VL_EXPORT vl_size vl_lbp_get_dimension(VlLbp * self)
{
  return self->dimension ;
}

/* ---------------------------------------------------------------- */

/** @brief Extract LBP features
 ** @param self LBP object.
 ** @param features buffer to write the features to.
 ** @param image image.
 ** @param width image width.
 ** @param height image height.
 ** @param cellSize size of the LBP cells.
 **
 ** @a features is a  @c numColumns x @c numRows x @c dimension where
 ** @c dimension is the dimension of a LBP feature obtained from ::vl_lbp_get_dimension,
 ** @c numColumns is equal to @c floor(width / cellSize), and similarly
 ** for @c numRows.
 **/

VL_EXPORT void
vl_lbp_process (VlLbp * self,
                float * features,
                float * image, vl_size width, vl_size height,
                vl_size cellSize)
{
  vl_size cwidth = width / cellSize;
  vl_size cheight = height / cellSize ;
  vl_size cstride = cwidth * cheight ;
  vl_size cdimension = vl_lbp_get_dimension(self) ;
  vl_index x,y,cx,cy,k,bin ;

#define at(u,v) (*(image + width * (v) + (u)))
#define to(u,v,w) (*(features + cstride * (w) + cwidth * (v) + (u)))

  /* clear the output buffer */
  memset(features, 0, sizeof(float)*cdimension*cstride) ;

  /* accumulate pixel-level measurements into cells */
  for (y = 1 ; y < (signed)height - 1 ; ++y) {
    float wy1 = (y + 0.5f) / (float)cellSize - 0.5f ;
    int cy1 = (int) vl_floor_f(wy1) ;
    int cy2 = cy1 + 1 ;
    float wy2 = wy1 - (float)cy1 ;
    wy1 = 1.0f - wy2 ;
    if (cy1 >= (signed)cheight) continue ;

    for (x = 1 ; x < (signed)width - 1; ++x) {
      float wx1 = (x + 0.5f) / (float)cellSize - 0.5f ;
      int cx1 = (int) vl_floor_f(wx1) ;
      int cx2 = cx1 + 1 ;
      float wx2 = wx1 - (float)cx1 ;
      wx1 = 1.0f - wx2 ;
      if (cx1 >= (signed)cwidth) continue ;

      {
        int unsigned bitString = 0 ;
        float center = at(x,y) ;
        if(at(x+1,y+0) > center) bitString |= 0x1 << 0; /*  E */
        if(at(x+1,y+1) > center) bitString |= 0x1 << 1; /* SE */
        if(at(x+0,y+1) > center) bitString |= 0x1 << 2; /* S  */
        if(at(x-1,y+1) > center) bitString |= 0x1 << 3; /* SW */
        if(at(x-1,y+0) > center) bitString |= 0x1 << 4; /*  W */
        if(at(x-1,y-1) > center) bitString |= 0x1 << 5; /* NW */
        if(at(x+0,y-1) > center) bitString |= 0x1 << 6; /* N  */
        if(at(x+1,y-1) > center) bitString |= 0x1 << 7; /* NE */
        bin = self->mapping[bitString] ;
      }

      if ((cx1 >= 0) & (cy1 >=0)) {
        to(cx1,cy1,bin) += wx1 * wy1;
      }
      if ((cx2 < (signed)cwidth)  & (cy1 >=0)) {
        to(cx2,cy1,bin) += wx2 * wy1 ;
      }
      if ((cx1 >= 0) & (cy2 < (signed)cheight)) {
        to(cx1,cy2,bin) += wx1 * wy2 ;
      }
      if ((cx2 < (signed)cwidth) & (cy2 < (signed)cheight)) {
        to(cx2,cy2,bin) += wx2 * wy2 ;
      }
    } /* x */
  } /* y */

  /* normalize cells */
  for (cy = 0 ; cy < (signed)cheight ; ++cy) {
    for (cx = 0 ; cx < (signed)cwidth ; ++ cx) {
      float norm = 0 ;
      for (k = 0 ; k < (signed)cdimension ; ++k) {
        norm += features[k * cstride] ;
      }
      norm = sqrtf(norm) + 1e-10f; ;
      for (k = 0 ; k < (signed)cdimension ; ++k) {
        features[k * cstride] = sqrtf(features[k * cstride]) / norm  ;
      }
      features += 1 ;
    }
  } /* next cell to normalize */
}
