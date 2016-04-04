/** @file slic.c
 ** @brief SLIC superpixels - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page slic Simple Linear Iterative Clustering (SLIC)
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref slic.h implements the *Simple Linear Iterative Clustering* (SLIC)
algorithm, an image segmentation method described in @cite{achanta10slic}.

- @ref slic-overview
- @ref slic-usage
- @ref slic-tech

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section slic-overview Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

SLIC @cite{achanta10slic} is a simple and efficient method to decompose
an image in visually homogeneous regions. It is based on a spatially
localized version of k-means clustering. Similar to mean shift or
quick shift (@ref quickshift.h), each pixel is associated to a feature
vector

@f[
\Psi(x,y) =
\left[
\begin{array}{c}
\lambda x \\
\lambda y \\
I(x,y)
\end{array}
\right]
@f]

and then k-means clustering is run on those. As discussed below, the
coefficient @f$ \lambda @f$ balances the spatial and appearance
components of the feature vectors, imposing a degree of spatial
regularization to the extracted regions.

SLIC takes two parameters: the nominal size of the regions
(superpixels) @c regionSize and the strength of the spatial
regularization @c regularizer. The image is first divided into a grid
with step @c regionSize. The center of each grid tile is then used to
initialize a corresponding k-means (up to a small shift to avoid
image edges). Finally, the k-means centers and clusters are refined by
using the Lloyd algorithm, yielding segmenting the image. As a
further restriction and simplification, during the k-means iterations
each pixel can be assigned to only the <em>2 x 2</em> centers
corresponding to grid tiles adjacent to the pixel.

The parameter @c regularizer sets the trade-off between clustering
appearance and spatial regularization. This is obtained by setting

@f[
 \lambda = \frac{\mathtt{regularizer}}{\mathtt{regionSize}}
@f]

in the definition of the feature @f$ \psi(x,y) @f$.

After the k-means step, SLIC optionally
removes any segment whose area is smaller than a threshld @c minRegionSize
by merging them into larger ones.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section slic-usage Usage from the C library
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

To compute the SLIC superpixels of an image use the function
::vl_slic_segment.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section slic-tech Technical details
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

SLIC starts by dividing the image domain into a regular grid with @f$
M \times N @f$ tiles, where

@f[
   M = \lceil \frac{\mathtt{imageWidth}}{\mathtt{regionSize}} \rceil,
   \quad
   N = \lceil \frac{\mathtt{imageHeight}}{\mathtt{regionSize}} \rceil.
@f]

A region (superpixel or k-means cluster) is initialized from each grid
center

@f[
  x_i = \operatorname{round} i \frac{\mathtt{imageWidth}}{\mathtt{regionSize}}
  \quad
  y_j = \operatorname{round} j \frac{\mathtt{imageWidth}}{\mathtt{regionSize}}.
@f]

In order to avoid placing these centers on top of image
discontinuities, the centers are then moved in a 3 x 3
neighbourohood to minimize the edge strength

@f[
   \operatorname{edge}(x,y) =
   \| I(x+1,y) - I(x-1,y) \|_2^2 +
   \| I(x,y+1) - I(x,y-1) \|_2^2.
@f]

Then the regions are obtained by running k-means clustering, started
from the centers

@f[
  C = \{ \Psi(x_i,y_j), i=0,1,\dots,M-1\ j=0,1,\dots,N-1 \}
@f]

thus obtained. K-means uses the standard LLoyd algorithm alternating
assigning pixels to the clostest centers a re-estiamting the centers
as the average of the corresponding feature vectors of the pixel
assigned to them. The only difference compared to standard k-means is
that each pixel can be assigned only to the center originated from the
neighbour tiles. This guarantees that there are exactly four
pixel-to-center comparisons at each round of minimization, which
threfore cost @f$ O(n) @f$, where @f$ n @f$ is the number of
superpixels.

After k-means has converged, SLIC eliminates any connected region whose
area is less than @c minRegionSize pixels. This is done by greedily
merging regions to neighbour ones: the pixels @f$ p @f$ are scanned in
lexicographical order and the corresponding connected components
are visited. If a region has already been visited, it is skipped; if not,
its area is computed and if this is less than  @c minRegionSize its label
is changed to the one of a neighbour
region at @f$ p @f$ that has already been vistied (there is always one
except for the very first pixel).

*/

#include "slic.h"
#include "mathop.h"
#include <math.h>
#include <string.h>

/** @brief SLIC superpixel segmentation
 ** @param segmentation segmentation.
 ** @param image image to segment.
 ** @param width image width.
 ** @param height image height.
 ** @param numChannels number of image channels (depth).
 ** @param regionSize nominal size of the regions.
 ** @param regularization trade-off between appearance and spatial terms.
 ** @param minRegionSize minimum size of a segment.
 **
 ** The function computes the SLIC superpixels of the specified image @a image.
 ** @a image is a pointer to an @c width by @c height by @c by numChannles array of @c float.
 ** @a segmentation is a pointer to a @c width by @c height array of @c vl_uint32.
 ** @a segmentation contain the labels of each image pixels, from 0 to
 ** the number of regions minus one.
 **
 ** @sa @ref slic-overview, @ref slic-tech
 **/

void
vl_slic_segment (vl_uint32 * segmentation,
                 float const * image,
                 vl_size width,
                 vl_size height,
                 vl_size numChannels,
                 vl_size regionSize,
                 float regularization,
                 vl_size minRegionSize)
{
  vl_index i, x, y, u, v, k, region ;
  vl_uindex iter ;
  vl_size const numRegionsX = (vl_size) ceil((double) width / regionSize) ;
  vl_size const numRegionsY = (vl_size) ceil((double) height / regionSize) ;
  vl_size const numRegions = numRegionsX * numRegionsY ;
  vl_size const numPixels = width * height ;
  float * centers ;
  float * edgeMap ;
  float previousEnergy = VL_INFINITY_F ;
  float startingEnergy ;
  vl_uint32 * masses ;
  vl_size const maxNumIterations = 100 ;

  assert(segmentation) ;
  assert(image) ;
  assert(width >= 1) ;
  assert(height >= 1) ;
  assert(numChannels >= 1) ;
  assert(regionSize >= 1) ;
  assert(regularization >= 0) ;

#define atimage(x,y,k) image[(x)+(y)*width+(k)*width*height]
#define atEdgeMap(x,y) edgeMap[(x)+(y)*width]

  edgeMap = vl_calloc(numPixels, sizeof(float)) ;
  masses = vl_malloc(sizeof(vl_uint32) * numPixels) ;
  centers = vl_malloc(sizeof(float) * (2 + numChannels) * numRegions) ;

  /* compute edge map (gradient strength) */
  for (k = 0 ; k < (signed)numChannels ; ++k) {
    for (y = 1 ; y < (signed)height-1 ; ++y) {
      for (x = 1 ; x < (signed)width-1 ; ++x) {
        float a = atimage(x-1,y,k) ;
        float b = atimage(x+1,y,k) ;
        float c = atimage(x,y+1,k) ;
        float d = atimage(x,y-1,k) ;
        atEdgeMap(x,y) += (a - b)  * (a - b) + (c - d) * (c - d) ;
      }
    }
  }

  /* initialize K-means centers */
  i = 0 ;
  for (v = 0 ; v < (signed)numRegionsY ; ++v) {
    for (u = 0 ; u < (signed)numRegionsX ; ++u) {
      vl_index xp ;
      vl_index yp ;
      vl_index centerx = 0 ;
      vl_index centery = 0 ;
      float minEdgeValue = VL_INFINITY_F ;

      x = (vl_index) vl_round_d(regionSize * (u + 0.5)) ;
      y = (vl_index) vl_round_d(regionSize * (v + 0.5)) ;

      x = VL_MAX(VL_MIN(x, (signed)width-1),0) ;
      y = VL_MAX(VL_MIN(y, (signed)height-1),0) ;

      /* search in a 3x3 neighbourhood the smallest edge response */
      for (yp = VL_MAX(0, y-1) ; yp <= VL_MIN((signed)height-1, y+1) ; ++ yp) {
        for (xp = VL_MAX(0, x-1) ; xp <= VL_MIN((signed)width-1, x+1) ; ++ xp) {
          float thisEdgeValue = atEdgeMap(xp,yp) ;
          if (thisEdgeValue < minEdgeValue) {
            minEdgeValue = thisEdgeValue ;
            centerx = xp ;
            centery = yp ;
          }
        }
      }

      /* initialize the new center at this location */
      centers[i++] = (float) centerx ;
      centers[i++] = (float) centery ;
      for (k  = 0 ; k < (signed)numChannels ; ++k) {
        centers[i++] = atimage(centerx,centery,k) ;
      }
    }
  }

  /* run k-means iterations */
  for (iter = 0 ; iter < maxNumIterations ; ++iter) {
    float factor = regularization / (regionSize * regionSize) ;
    float energy = 0 ;

    /* assign pixels to centers */
    for (y = 0 ; y < (signed)height ; ++y) {
      for (x = 0 ; x < (signed)width ; ++x) {
        vl_index u = floor((double)x / regionSize - 0.5) ;
        vl_index v = floor((double)y / regionSize - 0.5) ;
        vl_index up, vp ;
        float minDistance = VL_INFINITY_F ;

        for (vp = VL_MAX(0, v) ; vp <= VL_MIN((signed)numRegionsY-1, v+1) ; ++vp) {
          for (up = VL_MAX(0, u) ; up <= VL_MIN((signed)numRegionsX-1, u+1) ; ++up) {
            vl_index region = up  + vp * numRegionsX ;
            float centerx = centers[(2 + numChannels) * region + 0]  ;
            float centery = centers[(2 + numChannels) * region + 1] ;
            float spatial = (x - centerx) * (x - centerx) + (y - centery) * (y - centery) ;
            float appearance = 0 ;
            float distance ;
            for (k = 0 ; k < (signed)numChannels ; ++k) {
              float centerz = centers[(2 + numChannels) * region + k + 2]  ;
              float z = atimage(x,y,k) ;
              appearance += (z - centerz) * (z - centerz) ;
            }
            distance = appearance + factor * spatial ;
            if (minDistance > distance) {
              minDistance = distance ;
              segmentation[x + y * width] = (vl_uint32)region ;
            }
          }
        }
        energy += minDistance ;
      }
    }

    /*
     VL_PRINTF("vl:slic: iter %d: energy: %g\n", iter, energy) ;
    */

    /* check energy termination conditions */
    if (iter == 0) {
      startingEnergy = energy ;
    } else {
      if ((previousEnergy - energy) < 1e-5 * (startingEnergy - energy)) {
        break ;
      }
    }
    previousEnergy = energy ;

    /* recompute centers */
    memset(masses, 0, sizeof(vl_uint32) * width * height) ;
    memset(centers, 0, sizeof(float) * (2 + numChannels) * numRegions) ;

    for (y = 0 ; y < (signed)height ; ++y) {
      for (x = 0 ; x < (signed)width ; ++x) {
        vl_index pixel = x + y * width ;
        vl_index region = segmentation[pixel] ;
        masses[region] ++ ;
        centers[region * (2 + numChannels) + 0] += x ;
        centers[region * (2 + numChannels) + 1] += y ;
        for (k = 0 ; k < (signed)numChannels ; ++k) {
          centers[region * (2 + numChannels) + k + 2] += atimage(x,y,k) ;
        }
      }
    }

    for (region = 0 ; region < (signed)numRegions ; ++region) {
      float mass = VL_MAX(masses[region], 1e-8) ;
      for (i = (2 + numChannels) * region ;
           i < (signed)(2 + numChannels) * (region + 1) ;
           ++i) {
        centers[i] /= mass ;
      }
    }
  }

  vl_free(masses) ;
  vl_free(centers) ;
  vl_free(edgeMap) ;

  /* elimiate small regions */
  {
    vl_uint32 * cleaned = vl_calloc(numPixels, sizeof(vl_uint32)) ;
    vl_uindex * segment = vl_malloc(sizeof(vl_uindex) * numPixels) ;
    vl_size segmentSize ;
    vl_uint32 label ;
    vl_uint32 cleanedLabel ;
    vl_size numExpanded ;
    vl_index const dx [] = {+1, -1,  0,  0} ;
    vl_index const dy [] = { 0,  0, +1, -1} ;
    vl_index direction ;
    vl_index pixel ;

    for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel) {
      if (cleaned[pixel]) continue ;
      label = segmentation[pixel] ;
      numExpanded = 0 ;
      segmentSize = 0 ;
      segment[segmentSize++] = pixel ;

      /*
       find cleanedLabel as the label of an already cleaned
       region neihbour of this pixel
       */
      cleanedLabel = label + 1 ;
      cleaned[pixel] = label + 1 ;
      x = pixel % width ;
      y = pixel / width ;
      for (direction = 0 ; direction < 4 ; ++direction) {
        vl_index xp = x + dx[direction] ;
        vl_index yp = y + dy[direction] ;
        vl_index neighbor = xp + yp * width ;
        if (0 <= xp && xp < (signed)width &&
            0 <= yp && yp < (signed)height &&
            cleaned[neighbor]) {
          cleanedLabel = cleaned[neighbor] ;
        }
      }

      /* expand the segment */
      while (numExpanded < segmentSize) {
        vl_index open = segment[numExpanded++] ;
        x = open % width ;
        y = open / width ;
        for (direction = 0 ; direction < 4 ; ++direction) {
          vl_index xp = x + dx[direction] ;
          vl_index yp = y + dy[direction] ;
          vl_index neighbor = xp + yp * width ;
          if (0 <= xp && xp < (signed)width &&
              0 <= yp && yp < (signed)height &&
              cleaned[neighbor] == 0 &&
              segmentation[neighbor] == label) {
            cleaned[neighbor] = label + 1 ;
            segment[segmentSize++] = neighbor ;
          }
        }
      }

      /* change label to cleanedLabel if the semgent is too small */
      if (segmentSize < minRegionSize) {
        while (segmentSize > 0) {
          cleaned[segment[--segmentSize]] = cleanedLabel ;
        }
      }
    }
    /* restore base 0 indexing of the regions */
    for (pixel = 0 ; pixel < (signed)numPixels ; ++pixel) cleaned[pixel] -- ;

    memcpy(segmentation, cleaned, numPixels * sizeof(vl_uint32)) ;
    vl_free(cleaned) ;
    vl_free(segment) ;
  }
}
