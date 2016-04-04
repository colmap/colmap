/** @file pgm.h
 ** @brief Portable graymap format (PGM) parser
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_PGM_H
#define VL_PGM_H

#include "generic.h"
#include "mathop.h"
#include <stdio.h>

/** @name PGM parser error codes
 ** @{ */
#define VL_ERR_PGM_INV_HEAD  101 /**< Invalid PGM header section. */
#define VL_ERR_PGM_INV_META  102 /**< Invalid PGM meta section. */
#define VL_ERR_PGM_INV_DATA  103 /**< Invalid PGM data section.*/
#define VL_ERR_PGM_IO        104 /**< Generic I/O error. */
/** @} */

/** @brief PGM image meta data
 **
 ** A PGM image is a 2-D array of pixels of width #width and height
 ** #height. Each pixel is an integer one or two bytes wide, depending
 ** whether #max_value is smaller than 256.
 **/

typedef struct _VlPgmImage
{
  vl_size width ;      /**< image width.                     */
  vl_size height ;     /**< image height.                    */
  vl_size max_value ;  /**< pixel maximum value (<= 2^16-1). */
  vl_bool is_raw ;     /**< is RAW format?                   */
} VlPgmImage ;

/** @name Core operations
 ** @{ */
VL_EXPORT int vl_pgm_extract_head (FILE *f, VlPgmImage *im) ;
VL_EXPORT int vl_pgm_extract_data (FILE *f, VlPgmImage const *im, void *data) ;
VL_EXPORT int vl_pgm_insert (FILE *f,
                             VlPgmImage const *im,
                             void const*data ) ;
VL_EXPORT vl_size vl_pgm_get_npixels (VlPgmImage const *im) ;
VL_EXPORT vl_size vl_pgm_get_bpp (VlPgmImage const *im) ;
/** @} */

/** @name Helper functions
 ** @{ */
VL_EXPORT int vl_pgm_write (char const *name,
                            vl_uint8 const *data,
                            int width, int height) ;
VL_EXPORT int vl_pgm_write_f (char const *name,
                              float const *data,
                              int width, int height) ;
VL_EXPORT int vl_pgm_read_new (char const *name,
                               VlPgmImage *im,
                               vl_uint8 **data) ;
VL_EXPORT int vl_pgm_read_new_f (char const *name,
                                 VlPgmImage *im,
                                 float **data) ;

/** @} */

/* VL_PGM_H */
#endif
