/** @file mser.h
 ** @brief MSER (@ref mser)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_MSER
#define VL_MSER

#include "generic.h"

/** @brief MSER image data type
 **
 ** This is the data type of the image pixels. It has to be an
 ** integer.
 **/
typedef vl_uint8 vl_mser_pix ;

/** @brief Maximum value
 **
 ** Maximum value of the integer type ::vl_mser_pix.
 **/
#define VL_MSER_PIX_MAXVAL 256

/** @brief MSER Filter
 **
 ** The MSER filter computes the Maximally Stable Extremal Regions of
 ** an image.
 **
 ** @sa @ref mser
 **/
typedef struct _VlMserFilt VlMserFilt ;

/** @brief MSER filter statistics */
typedef struct _VlMserStats VlMserStats ;

/** @brief MSER filter statistics definition */
struct _VlMserStats
{
  int num_extremal ;      /**< number of extremal regions                                */
  int num_unstable ;      /**< number of unstable extremal regions                       */
  int num_abs_unstable ;  /**< number of regions that failed the absolute stability test */
  int num_too_big ;       /**< number of regions that failed the maximum size test       */
  int num_too_small ;     /**< number of regions that failed the minimum size test       */
  int num_duplicates ;    /**< number of regions that failed the duplicate test          */
} ;

/** @name Construction and Destruction
 ** @{
 **/
VL_EXPORT VlMserFilt*      vl_mser_new     (int ndims, int const* dims) ;
VL_EXPORT void             vl_mser_delete  (VlMserFilt *f) ;
/** @} */

/** @name Processing
 ** @{
 **/
VL_EXPORT void             vl_mser_process (VlMserFilt *f,
                                            vl_mser_pix const *im) ;
VL_EXPORT void             vl_mser_ell_fit (VlMserFilt *f) ;
/** @} */

/** @name Retrieving data
 ** @{
 **/
VL_INLINE vl_uint          vl_mser_get_regions_num  (VlMserFilt const *f) ;
VL_INLINE vl_uint const*   vl_mser_get_regions      (VlMserFilt const *f) ;
VL_INLINE float const*     vl_mser_get_ell          (VlMserFilt const *f) ;
VL_INLINE vl_uint          vl_mser_get_ell_num      (VlMserFilt const *f) ;
VL_INLINE vl_uint          vl_mser_get_ell_dof      (VlMserFilt const *f) ;
VL_INLINE VlMserStats const*  vl_mser_get_stats     (VlMserFilt const *f) ;
/** @} */

/** @name Retrieving parameters
 ** @{
 **/
VL_INLINE vl_mser_pix  vl_mser_get_delta          (VlMserFilt const *f) ;
VL_INLINE double       vl_mser_get_min_area       (VlMserFilt const *f) ;
VL_INLINE double       vl_mser_get_max_area       (VlMserFilt const *f) ;
VL_INLINE double       vl_mser_get_max_variation  (VlMserFilt const *f) ;
VL_INLINE double       vl_mser_get_min_diversity  (VlMserFilt const *f) ;
/** @} */

/** @name Setting parameters
 ** @{
 **/
VL_INLINE void  vl_mser_set_delta           (VlMserFilt *f, vl_mser_pix x) ;
VL_INLINE void  vl_mser_set_min_area        (VlMserFilt *f, double      x) ;
VL_INLINE void  vl_mser_set_max_area        (VlMserFilt *f, double      x) ;
VL_INLINE void  vl_mser_set_max_variation   (VlMserFilt *f, double      x) ;
VL_INLINE void  vl_mser_set_min_diversity   (VlMserFilt *f, double      x) ;
/** @} */

/* ====================================================================
 *                                                   INLINE DEFINITIONS
 * ================================================================== */

/** @internal
 ** @brief MSER accumulator data type
 **
 ** This is a large integer type. It should be large enough to contain
 ** a number equal to the area (volume) of the image by the image
 ** width by the image height (for instance, if the image is a square
 ** of side 256, the maximum value is 256 x 256 x 256).
 **/
typedef float vl_mser_acc ;

/** @internal @brief Basic region flag: null region */
#ifdef VL_COMPILER_MSC
#define VL_MSER_VOID_NODE ((1ui64<<32) - 1)
#else
#define VL_MSER_VOID_NODE ((1ULL<<32) - 1)
#endif

/* ----------------------------------------------------------------- */
/** @internal
 ** @brief MSER: basic region (declaration)
 **
 ** Extremal regions and maximally stable extremal regions are
 ** instances of image regions.
 **
 ** There is an image region for each pixel of the image. Each region
 ** is represented by an instance of this structure.  Regions are
 ** stored into an array in pixel order.
 **
 ** Regions are arranged into a forest. VlMserReg::parent points to
 ** the parent node, or to the node itself if the node is a root.
 ** VlMserReg::parent is the index of the node in the node array
 ** (which therefore is also the index of the corresponding
 ** pixel). VlMserReg::height is the distance of the fartest leaf. If
 ** the node itself is a leaf, then VlMserReg::height is zero.
 **
 ** VlMserReg::area is the area of the image region corresponding to
 ** this node.
 **
 ** VlMserReg::region is the extremal region identifier. Not all
 ** regions are extremal regions however; if the region is NOT
 ** extremal, this field is set to ....
 **/
struct _VlMserReg
{
  vl_uint parent ;   /**< points to the parent region.            */
  vl_uint shortcut ; /**< points to a region closer to a root.    */
  vl_uint height ;   /**< region height in the forest.            */
  vl_uint area ;     /**< area of the region.                     */
} ;

/** @internal @brief MSER: basic region */
typedef struct _VlMserReg VlMserReg ;

/* ----------------------------------------------------------------- */
/** @internal
 ** @brief MSER: extremal region (declaration)
 **
 ** Extremal regions (ER) are extracted from the region forest. Each
 ** region is represented by an instance of this structure. The
 ** structures are stored into an array, in arbitrary order.
 **
 ** ER are arranged into a tree. @a parent points to the parent ER, or
 ** to itself if the ER is the root.
 **
 ** An instance of the structure represents the extremal region of the
 ** level set of intensity VlMserExtrReg::value and containing the
 ** pixel VlMserExtReg::index.
 **
 ** VlMserExtrReg::area is the area of the extremal region and
 ** VlMserExtrReg::area_top is the area of the extremal region
 ** containing this region in the level set of intensity
 ** VlMserExtrReg::area + @c delta.
 **
 ** VlMserExtrReg::variation is the relative area variation @c
 ** (area_top-area)/area.
 **
 ** VlMserExtrReg::max_stable is a flag signaling whether this extremal
 ** region is also maximally stable.
 **/
struct _VlMserExtrReg
{
  int          parent ;     /**< index of the parent region                   */
  int          index ;      /**< index of pivot pixel                         */
  vl_mser_pix  value ;      /**< value of pivot pixel                         */
  vl_uint      shortcut ;   /**< shortcut used when building a tree           */
  vl_uint      area ;       /**< area of the region                           */
  float        variation ;  /**< rel. area variation                          */
  vl_uint      max_stable ; /**< max stable number (=0 if not maxstable)      */
} ;

/** @internal
 ** @brief MSER: extremal region */
typedef struct _VlMserExtrReg VlMserExtrReg ;

/* ----------------------------------------------------------------- */
/** @internal
 ** @brief MSER filter
 ** @see @ref mser
 **/
struct _VlMserFilt
{

  /** @name Image data and meta data @internal */
  /*@{*/
  int                ndims ;   /**< number of dimensions                    */
  int               *dims ;    /**< dimensions                              */
  int                nel ;     /**< number of image elements (pixels)       */
  int               *subs ;    /**< N-dimensional subscript                 */
  int               *dsubs ;   /**< another subscript                       */
  int               *strides ; /**< strides to move in image data           */
  /*@}*/

  vl_uint           *perm ;    /**< pixel ordering                          */
  vl_uint           *joins ;   /**< sequence of join ops                    */
  int                njoins ;  /**< number of join ops                      */

  /** @name Regions */
  /*@{*/
  VlMserReg         *r ;       /**< basic regions                           */
  VlMserExtrReg     *er ;      /**< extremal tree                           */
  vl_uint           *mer ;     /**< maximally stable extremal regions       */
  int                ner ;     /**< number of extremal regions              */
  int                nmer ;    /**< number of maximally stable extr. reg.   */
  int                rer ;     /**< size of er buffer                       */
  int                rmer ;    /**< size of mer buffer                      */
  /*@}*/

  /** @name Ellipsoids fitting */
  /*@{*/
  float             *acc ;     /**< moment accumulator.                    */
  float             *ell ;     /**< ellipsoids list.                       */
  int                rell ;    /**< size of ell buffer                     */
  int                nell ;    /**< number of ellipsoids extracted         */
  int                dof ;     /**< number of dof of ellipsoids.           */

  /*@}*/

  /** @name Configuration */
  /*@{*/
  vl_bool   verbose ;          /**< be verbose                             */
  int       delta ;            /**< delta filter parameter                 */
  double    max_area ;         /**< badness test parameter                 */
  double    min_area ;         /**< badness test parameter                 */
  double    max_variation ;    /**< badness test parameter                 */
  double    min_diversity ;    /**< minimum diversity                      */
  /*@}*/

  VlMserStats stats ;          /** run statistic                           */
} ;

/* ----------------------------------------------------------------- */
/** @brief Get delta
 ** @param f MSER filter.
 ** @return value of @c delta.
 **/
VL_INLINE vl_mser_pix
vl_mser_get_delta (VlMserFilt const *f)
{
  return f-> delta ;
}

/** @brief Set delta
 ** @param f MSER filter.
 ** @param x value of @c delta.
 **/
VL_INLINE void
vl_mser_set_delta (VlMserFilt *f, vl_mser_pix x)
{
  f-> delta = x ;
}

/* ----------------------------------------------------------------- */
/** @brief Get minimum diversity
 ** @param  f MSER filter.
 ** @return value of @c minimum diversity.
 **/
VL_INLINE double
vl_mser_get_min_diversity (VlMserFilt const *f)
{
  return f-> min_diversity ;
}

/** @brief Set minimum diversity
 ** @param f MSER filter.
 ** @param x value of @c minimum diversity.
 **/
VL_INLINE void
vl_mser_set_min_diversity (VlMserFilt *f, double x)
{
  f-> min_diversity = x ;
}

/* ----------------------------------------------------------------- */
/** @brief Get statistics
 ** @param f MSER filter.
 ** @return statistics.
 **/
VL_INLINE VlMserStats const*
vl_mser_get_stats (VlMserFilt const *f)
{
  return & f-> stats ;
}

/* ----------------------------------------------------------------- */
/** @brief Get maximum region area
 ** @param f MSER filter.
 ** @return maximum region area.
 **/
VL_INLINE double
vl_mser_get_max_area (VlMserFilt const *f)
{
  return f-> max_area ;
}

/** @brief Set maximum region area
 ** @param f MSER filter.
 ** @param x maximum region area.
 **/
VL_INLINE void
vl_mser_set_max_area (VlMserFilt *f, double x)
{
  f-> max_area = x ;
}

/* ----------------------------------------------------------------- */
/** @brief Get minimum region area
 ** @param f MSER filter.
 ** @return minimum region area.
 **/
VL_INLINE double
vl_mser_get_min_area (VlMserFilt const *f)
{
  return f-> min_area ;
}

/** @brief Set minimum region area
 ** @param f MSER filter.
 ** @param x minimum region area.
 **/
VL_INLINE void
vl_mser_set_min_area (VlMserFilt *f, double x)
{
  f-> min_area = x ;
}

/* ----------------------------------------------------------------- */
/** @brief Get maximum region variation
 ** @param f MSER filter.
 ** @return maximum region variation.
 **/
VL_INLINE double
vl_mser_get_max_variation (VlMserFilt const *f)
{
  return f-> max_variation ;
}

/** @brief Set maximum region variation
 ** @param f MSER filter.
 ** @param x maximum region variation.
 **/
VL_INLINE void
vl_mser_set_max_variation (VlMserFilt *f, double x)
{
  f-> max_variation = x ;
}

/* ----------------------------------------------------------------- */
/** @brief Get maximally stable extremal regions
 ** @param f MSER filter.
 ** @return array of MSER pivots.
 **/
VL_INLINE vl_uint const *
vl_mser_get_regions (VlMserFilt const* f)
{
  return f-> mer ;
}

/** @brief Get number of maximally stable extremal regions
 ** @param f MSER filter.
 ** @return number of MSERs.
 **/
VL_INLINE vl_uint
vl_mser_get_regions_num (VlMserFilt const* f)
{
  return f-> nmer ;
}

/* ----------------------------------------------------------------- */
/** @brief Get ellipsoids
 ** @param f MSER filter.
 ** @return ellipsoids.
 **/
VL_INLINE float const *
vl_mser_get_ell (VlMserFilt const* f)
{
  return f-> ell ;
}

/** @brief Get number of degrees of freedom of ellipsoids
 ** @param f MSER filter.
 ** @return number of degrees of freedom.
 **/
VL_INLINE vl_uint
vl_mser_get_ell_dof (VlMserFilt const* f)
{
  return f-> dof ;
}

/** @brief Get number of ellipsoids
 ** @param f MSER filter.
 ** @return number of ellipsoids
 **/
VL_INLINE vl_uint
vl_mser_get_ell_num (VlMserFilt const* f)
{
  return f-> nell ;
}

/* VL_MSER */
#endif
