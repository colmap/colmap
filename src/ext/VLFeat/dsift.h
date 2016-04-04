/** @file dsift.h
 ** @brief Dense SIFT (@ref dsift)
 ** @author Andrea Vedaldi
 ** @author Brian Fulkerson
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_DSIFT_H
#define VL_DSIFT_H

#include "generic.h"

/** @brief Dense SIFT keypoint */
typedef struct VlDsiftKeypoint_
{
  double x ; /**< x coordinate */
  double y ; /**< y coordinate */
  double s ; /**< scale */
  double norm ; /**< SIFT descriptor norm */
} VlDsiftKeypoint ;

/** @brief Dense SIFT descriptor geometry */
typedef struct VlDsiftDescriptorGeometry_
{
  int numBinT ;  /**< number of orientation bins */
  int numBinX ;  /**< number of bins along X */
  int numBinY ;  /**< number of bins along Y */
  int binSizeX ; /**< size of bins along X */
  int binSizeY ; /**< size of bins along Y */
} VlDsiftDescriptorGeometry ;

/** @brief Dense SIFT filter */
typedef struct VlDsiftFilter_
{
  int imWidth ;            /**< @internal @brief image width */
  int imHeight ;           /**< @internal @brief image height */

  int stepX ;              /**< frame sampling step X */
  int stepY ;              /**< frame sampling step Y */

  int boundMinX ;          /**< frame bounding box min X */
  int boundMinY ;          /**< frame bounding box min Y */
  int boundMaxX ;          /**< frame bounding box max X */
  int boundMaxY ;          /**< frame bounding box max Y */

  /** descriptor parameters */
  VlDsiftDescriptorGeometry geom ;

  int useFlatWindow ;      /**< flag: whether to approximate the Gaussian window with a flat one */
  double windowSize ;      /**< size of the Gaussian window */

  int numFrames ;          /**< number of sampled frames */
  int descrSize ;          /**< size of a descriptor */
  VlDsiftKeypoint *frames ; /**< frame buffer */
  float *descrs ;          /**< descriptor buffer */

  int numBinAlloc ;        /**< buffer allocated: descriptor size */
  int numFrameAlloc ;      /**< buffer allocated: number of frames  */
  int numGradAlloc ;       /**< buffer allocated: number of orientations */

  float **grads ;          /**< gradient buffer */
  float *convTmp1 ;        /**< temporary buffer */
  float *convTmp2 ;        /**< temporary buffer */
}  VlDsiftFilter ;

VL_EXPORT VlDsiftFilter *vl_dsift_new (int width, int height) ;
VL_EXPORT VlDsiftFilter *vl_dsift_new_basic (int width, int height, int step, int binSize) ;
VL_EXPORT void vl_dsift_delete (VlDsiftFilter *self) ;
VL_EXPORT void vl_dsift_process (VlDsiftFilter *self, float const* im) ;
VL_INLINE void vl_dsift_transpose_descriptor (float* dst,
                                             float const* src,
                                             int numBinT,
                                             int numBinX,
                                             int numBinY) ;

/** @name Setting parameters
 ** @{
 **/
VL_INLINE void vl_dsift_set_steps (VlDsiftFilter *self,
                                  int stepX,
                                  int stepY) ;
VL_INLINE void vl_dsift_set_bounds (VlDsiftFilter *self,
                                   int minX,
                                   int minY,
                                   int maxX,
                                   int maxY) ;
VL_INLINE void vl_dsift_set_geometry (VlDsiftFilter *self,
                                      VlDsiftDescriptorGeometry const* geom) ;
VL_INLINE void vl_dsift_set_flat_window (VlDsiftFilter *self, vl_bool useFlatWindow) ;
VL_INLINE void vl_dsift_set_window_size (VlDsiftFilter *self, double windowSize) ;
/** @} */

/** @name Retrieving data and parameters
 ** @{
 **/
VL_INLINE float const    *vl_dsift_get_descriptors     (VlDsiftFilter const *self) ;
VL_INLINE int             vl_dsift_get_descriptor_size (VlDsiftFilter const *self) ;
VL_INLINE int             vl_dsift_get_keypoint_num    (VlDsiftFilter const *self) ;
VL_INLINE VlDsiftKeypoint const *vl_dsift_get_keypoints (VlDsiftFilter const *self) ;
VL_INLINE void            vl_dsift_get_bounds          (VlDsiftFilter const *self,
                                                       int* minX,
                                                       int* minY,
                                                       int* maxX,
                                                       int* maxY) ;
VL_INLINE void            vl_dsift_get_steps           (VlDsiftFilter const* self,
                                                       int* stepX,
                                                       int* stepY) ;
VL_INLINE VlDsiftDescriptorGeometry const* vl_dsift_get_geometry (VlDsiftFilter const *self) ;
VL_INLINE vl_bool         vl_dsift_get_flat_window     (VlDsiftFilter const *self) ;
VL_INLINE double          vl_dsift_get_window_size     (VlDsiftFilter const *self) ;
/** @} */

VL_EXPORT
void _vl_dsift_update_buffers (VlDsiftFilter *self) ;

/** ------------------------------------------------------------------
 ** @brief Get descriptor size.
 ** @param self DSIFT filter object.
 ** @return size of a descriptor.
 **/

int
vl_dsift_get_descriptor_size (VlDsiftFilter const *self)
{
  return self->descrSize ;
}

/** ------------------------------------------------------------------
 ** @brief Get descriptors.
 ** @param self DSIFT filter object.
 ** @return descriptors.
 **/

float const *
vl_dsift_get_descriptors (VlDsiftFilter const *self)
{
  return self->descrs ;
}

/** ------------------------------------------------------------------
 ** @brief Get keypoints
 ** @param self DSIFT filter object.
 **/

VlDsiftKeypoint const *
vl_dsift_get_keypoints (VlDsiftFilter const *self)
{
  return self->frames ;
}

/** ------------------------------------------------------------------
 ** @brief Get number of keypoints
 ** @param self DSIFT filter object.
 **/

int
vl_dsift_get_keypoint_num (VlDsiftFilter const *self)
{
  return self->numFrames ;
}

/** ------------------------------------------------------------------
 ** @brief Get SIFT descriptor geometry
 ** @param self DSIFT filter object.
 ** @return DSIFT descriptor geometry.
 **/

VlDsiftDescriptorGeometry const* vl_dsift_get_geometry (VlDsiftFilter const *self)
{
  return &self->geom ;
}

/** ------------------------------------------------------------------
 ** @brief Get bounds
 ** @param self DSIFT filter object.
 ** @param minX bounding box minimum X coordinate.
 ** @param minY bounding box minimum Y coordinate.
 ** @param maxX bounding box maximum X coordinate.
 ** @param maxY bounding box maximum Y coordinate.
 **/

void
vl_dsift_get_bounds (VlDsiftFilter const* self,
                    int *minX, int *minY, int *maxX, int *maxY)
{
  *minX = self->boundMinX ;
  *minY = self->boundMinY ;
  *maxX = self->boundMaxX ;
  *maxY = self->boundMaxY ;
}

/** ------------------------------------------------------------------
 ** @brief Get flat window flag
 ** @param self DSIFT filter object.
 ** @return @c TRUE if the DSIFT filter uses a flat window.
 **/

int
vl_dsift_get_flat_window (VlDsiftFilter const* self)
{
  return self->useFlatWindow ;
}

/** ------------------------------------------------------------------
 ** @brief Get steps
 ** @param self DSIFT filter object.
 ** @param stepX sampling step along X.
 ** @param stepY sampling step along Y.
 **/

void
vl_dsift_get_steps (VlDsiftFilter const* self,
                   int* stepX,
                   int* stepY)
{
  *stepX = self->stepX ;
  *stepY = self->stepY ;
}

/** ------------------------------------------------------------------
 ** @brief Set steps
 ** @param self DSIFT filter object.
 ** @param stepX sampling step along X.
 ** @param stepY sampling step along Y.
 **/

void
vl_dsift_set_steps (VlDsiftFilter* self,
                   int stepX,
                   int stepY)
{
  self->stepX = stepX ;
  self->stepY = stepY ;
  _vl_dsift_update_buffers(self) ;
}

/** ------------------------------------------------------------------
 ** @brief Set bounds
 ** @param self DSIFT filter object.
 ** @param minX bounding box minimum X coordinate.
 ** @param minY bounding box minimum Y coordinate.
 ** @param maxX bounding box maximum X coordinate.
 ** @param maxY bounding box maximum Y coordinate.
 **/

void
vl_dsift_set_bounds (VlDsiftFilter* self,
                    int minX, int minY, int maxX, int maxY)
{
  self->boundMinX = minX ;
  self->boundMinY = minY ;
  self->boundMaxX = maxX ;
  self->boundMaxY = maxY ;
  _vl_dsift_update_buffers(self) ;
}

/** ------------------------------------------------------------------
 ** @brief Set SIFT descriptor geometry
 ** @param self DSIFT filter object.
 ** @param geom descriptor geometry parameters.
 **/

void
vl_dsift_set_geometry (VlDsiftFilter *self,
                       VlDsiftDescriptorGeometry const *geom)
{
  self->geom = *geom ;
  _vl_dsift_update_buffers(self) ;
}

/** ------------------------------------------------------------------
 ** @brief Set flat window flag
 ** @param self DSIFT filter object.
 ** @param useFlatWindow @c true if the DSIFT filter should use a flat window.
 **/

void
vl_dsift_set_flat_window (VlDsiftFilter* self,
                         vl_bool useFlatWindow)
{
  self->useFlatWindow = useFlatWindow ;
}

/** ------------------------------------------------------------------
 ** @brief Transpose descriptor
 **
 ** @param dst destination buffer.
 ** @param src source buffer.
 ** @param numBinT
 ** @param numBinX
 ** @param numBinY
 **
 ** The function writes to @a dst the transpose of the SIFT descriptor
 ** @a src. Let <code>I</code> be an image. The transpose operator
 ** satisfies the equation <code>transpose(dsift(I,x,y)) =
 ** dsift(transpose(I),y,x)</code>
 **/

VL_INLINE void
vl_dsift_transpose_descriptor (float* dst,
                              float const* src,
                              int numBinT,
                              int numBinX,
                              int numBinY)
{
  int t, x, y ;

  for (y = 0 ; y < numBinY ; ++y) {
    for (x = 0 ; x < numBinX ; ++x) {
      int offset  = numBinT * (x + y * numBinX) ;
      int offsetT = numBinT * (y + x * numBinY) ;

      for (t = 0 ; t < numBinT ; ++t) {
        int tT = numBinT / 4 - t ;
        dst [offsetT + (tT + numBinT) % numBinT] = src [offset + t] ;
      }
    }
  }
}

/** ------------------------------------------------------------------
 ** @brief Set SIFT descriptor Gaussian window size
 ** @param self DSIFT filter object.
 ** @param windowSize window size.
 **/

void
vl_dsift_set_window_size(VlDsiftFilter * self, double windowSize)
{
  assert(windowSize >= 0.0) ;
  self->windowSize = windowSize ;
}

/** ------------------------------------------------------------------
 ** @brief Get SIFT descriptor Gaussian window size
 ** @param self DSIFT filter object.
 ** @return window size.
 **/

VL_INLINE double
vl_dsift_get_window_size(VlDsiftFilter const * self)
{
  return self->windowSize ;
}

/*  VL_DSIFT_H */
#endif
