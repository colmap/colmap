/** @file pgm.c
 ** @brief Portable graymap format (PGM) parser - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** @file pgm.h

This module implements basic input and ouptut of images in PGM
format.

Extracting an image encoded in PGM format from an imput
file stream involves the following steps:

- use ::vl_pgm_extract_head to extract the image meta data
  (size and bit depth);
- allocate a buffer to store the image data;
- use ::vl_pgm_extract_data to extract the image data to the allocated
  buffer.

Writing an image in PGM format to an ouptut file stream
can be done by using ::vl_pgm_insert.

To quickly read/write a PGM image from/to a given file, use
::vl_pgm_read_new() and ::vl_pgm_write(). To to the same from a
buffer in floating point format use ::vl_pgm_read_new_f() and
::vl_pgm_write_f().

**/

#include "pgm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** ------------------------------------------------------------------
 ** @internal @brief Remove all characters to the next new-line.
 ** @param f file to strip.
 ** @return number of characters removed.
 **/

static int
remove_line(FILE* f)
{
  int count = 0 ;
  int c ;

  while (1) {
    c = fgetc(f) ;
    ++ count ;

    switch(c) {
    case '\n' :
      goto quit_remove_line ;

    case EOF :
      -- count ;
      goto quit_remove_line ;
    }
  }
 quit_remove_line :
  return count ;
}

/** ------------------------------------------------------------------
 ** @internal @brief Remove white-spaces and comments.
 ** @param f file to strip.
 ** @return number of characters removed.
 **/

static int
remove_blanks(FILE* f)
{
  int count = 0 ;
  int c ;

  while (1) {
    c = fgetc(f) ;

    switch(c) {

    case '\t' : case '\n' :
    case '\r' : case ' '  :
      ++ count ;
      break ;

    case '#' :
      count += 1 + remove_line(f) ;
      break ;

    case EOF :
      goto quit_remove_blanks ;

    default:
      ungetc(c, f) ;
      goto quit_remove_blanks ;
    }
  }
 quit_remove_blanks:
  return count ;
}

/** ------------------------------------------------------------------
 ** @brief Get PGM image number of pixels.
 ** @param im PGM image descriptor.
 ** @return number of pixels of the image.
 **
 ** The functions returns the number of pixels of the PGM image @a im.
 **
 ** To calculate the image data size in bytes, this value must be
 ** multiplied by the number of byte per pixels (see
 ** ::vl_pgm_get_bpp()).
 **/

VL_EXPORT vl_size
vl_pgm_get_npixels (VlPgmImage const *im)
{
  return im->width * im->height ;
}

/** ------------------------------------------------------------------
 ** @brief Get PGM image bytes per pixel.
 ** @param im PGM image descriptor.
 ** @return number of bytes per pixel.
 **
 ** The function returns the number of bytes for each pixel of the
 ** PGM image @a im.
 **/

VL_EXPORT vl_size
vl_pgm_get_bpp (VlPgmImage const *im)
{
  return (im->max_value >= 256) + 1 ;
}

/** ------------------------------------------------------------------
 ** @brief Extract PGM header from stream.
 ** @param f  input file.
 ** @param im image structure to fill.
 ** @return error code.
 **
 ** The function extracts from the file @a f the meta-data section of
 ** an image encoded in PGM format. The function fills the structure
 ** ::VlPgmImage accordingly.
 **
 ** The error may be either ::VL_ERR_PGM_INV_HEAD or ::VL_ERR_PGM_INV_META
 ** depending whether the error occurred in decoding the header or
 ** meta section of the PGM file.
 **/

VL_EXPORT int
vl_pgm_extract_head (FILE* f, VlPgmImage *im)
{
  char magic [2] ;
  int c ;
  int is_raw ;
  int width ;
  int height ;
  int max_value ;
  size_t sz ;
  vl_bool good ;

  /* -----------------------------------------------------------------
   *                                                check magic number
   * -------------------------------------------------------------- */
  sz = fread(magic, 1, 2, f) ;

  if (sz < 2) {
    return vl_set_last_error(VL_ERR_PGM_INV_HEAD, "Invalid PGM header") ;
  }

  good = magic [0] == 'P' ;

  switch (magic [1]) {
  case '2' : /* ASCII format */
    is_raw = 0 ;
    break ;

  case '5' : /* RAW format */
    is_raw = 1 ;
    break ;

  default :
    good = 0 ;
    break ;
  }

  if( ! good ) {
    return vl_set_last_error(VL_ERR_PGM_INV_HEAD, "Invalid PGM header") ;
  }

  /* -----------------------------------------------------------------
   *                                    parse width, height, max_value
   * -------------------------------------------------------------- */
  good = 1 ;

  c = remove_blanks(f) ;
  good &= c > 0 ;

  c = fscanf(f, "%d", &width) ;
  good &= c == 1 ;

  c = remove_blanks(f) ;
  good &= c > 0 ;

  c = fscanf(f, "%d", &height) ;
  good &= c == 1 ;

  c = remove_blanks(f) ;
  good &= c > 0 ;

  c = fscanf(f, "%d", &max_value) ;
  good &= c == 1 ;

  /* must end with a single blank */
  c = fgetc(f) ;
  good &=
    c == '\n' ||
    c == '\t' ||
    c == ' '  ||
    c == '\r' ;

  if(! good) {
    return vl_set_last_error(VL_ERR_PGM_INV_META, "Invalid PGM meta information");
  }

  if(! max_value >= 65536) {
    return vl_set_last_error(VL_ERR_PGM_INV_META, "Invalid PGM meta information");
  }

  /* exit */
  im-> width     = width ;
  im-> height    = height ;
  im-> max_value = max_value ;
  im-> is_raw    = is_raw ;
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @brief Extract PGM data from stream.
 ** @param f input file.
 ** @param im PGM image descriptor.
 ** @param data data buffer to fill.
 ** @return error code.
 **
 ** The function extracts from the file @a f the data section of an
 ** image encoded in PGM format. The function fills the buffer @a data
 ** according. The buffer @a data should be ::vl_pgm_get_npixels() by
 ** ::vl_pgm_get_bpp() bytes large.
 **/

VL_EXPORT
int
vl_pgm_extract_data (FILE* f, VlPgmImage const *im, void *data)
{
  vl_size bpp = vl_pgm_get_bpp(im) ;
  vl_size data_size = vl_pgm_get_npixels(im) ;
  vl_bool good = 1 ;
  size_t c ;

  /* -----------------------------------------------------------------
   *                                                         read data
   * -------------------------------------------------------------- */

  /*
     In RAW mode we read directly an array of bytes or shorts.  In
     the latter case, however, we must take care of the
     endianess. PGM files are sorted in big-endian format. If our
     architecture is little endian, we must do a conversion.
  */
  if (im->is_raw) {

    c = fread( data,
               bpp,
               data_size,
               f ) ;
    good = (c == data_size) ;

    /* adjust endianess */
#if defined(VL_ARCH_LITTLE_ENDIAN)
    if (bpp == 2) {
      vl_uindex i ;
      vl_uint8 *pt = (vl_uint8*) data ;
      for(i = 0 ; i < 2 * data_size ; i += 2) {
        vl_uint8 tmp = pt [i] ;
        pt [i]   = pt [i+1] ;
        pt [i+1] = tmp ;
      }
    }
#endif
  }
  /*
     In ASCII mode we read a sequence of decimal numbers separated
     by whitespaces.
  */
  else {
    vl_uindex i ;
    int unsigned v ;
    for(good = 1, i = 0 ;
        i < data_size && good ;
        ++i) {
      c = fscanf(f, " %ud", &v) ;
      if (bpp == 1) {
        * ((vl_uint8* )  data + i) = (vl_uint8)  v ;
      } else {
        * ((vl_uint16*)  data + i) = (vl_uint16) v ;
      }
      good &= c == 1 ;
    }
  }

  if(! good ) {
    return vl_set_last_error(VL_ERR_PGM_INV_DATA, "Invalid PGM data") ;
  }
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @brief Insert a PGM image into a stream.
 ** @param f output file.
 ** @param im   PGM image meta-data.
 ** @param data image data.
 ** @return error code.
 **/

VL_EXPORT
int
vl_pgm_insert(FILE* f, VlPgmImage const *im, void const *data)
{
  vl_size bpp = vl_pgm_get_bpp (im) ;
  vl_size data_size = vl_pgm_get_npixels (im) ;
  size_t c ;

  /* write preamble */
  fprintf(f,
          "P5\n%d\n%d\n%d\n",
          (signed)im->width,
          (signed)im->height,
          (signed)im->max_value) ;

  /* take care of endianness */
#if defined(VL_ARCH_LITTLE_ENDIAN)
  if (bpp == 2) {
    vl_uindex i ;
    vl_uint8* temp = vl_malloc (2 * data_size) ;
    memcpy(temp, data, 2 * data_size) ;
    for(i = 0 ; i < 2 * data_size ; i += 2) {
      vl_uint8 tmp = temp [i] ;
      temp [i]   = temp [i+1] ;
      temp [i+1] = tmp ;
    }
    c = fwrite(temp, 2, data_size, f) ;
    vl_free (temp) ;
  }
  else {
#endif
    c = fwrite(data, bpp, data_size, f) ;
#if defined(VL_ARCH_LITTLE_ENDIAN)
  }
#endif

  if(c != data_size) {
    return vl_set_last_error(VL_ERR_PGM_IO, "Error writing PGM data") ;
  }
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @brief Read a PGM file.
 ** @param name file name.
 ** @param im a pointer to the PGM image structure to fill.
 ** @param data a pointer to the pointer to the allocated buffer.
 ** @return error code.
 **
 ** The function reads a PGM image from file @a name and initializes
 ** the structure @a im and the buffer @a data accordingly.
 **
 ** The ownership of the buffer @a data is transfered to the caller.
 ** @a data should be freed by means of ::vl_free().
 **
 ** @bug Only PGM files with 1 BPP are supported.
 **/

VL_EXPORT
int vl_pgm_read_new (char const *name, VlPgmImage *im, vl_uint8** data)
{
  int err = 0 ;
  FILE *f = fopen (name, "rb") ;

  if (! f) {
    return vl_set_last_error(VL_ERR_PGM_IO, "Error opening PGM file `%s' for reading", name) ;
  }

  err = vl_pgm_extract_head(f, im) ;
  if (err) {
    fclose (f) ;
    return err ;
  }

  if (vl_pgm_get_bpp(im) > 1) {
    return vl_set_last_error(VL_ERR_BAD_ARG, "PGM with BPP > 1 not supported") ;
  }

  *data = vl_malloc (vl_pgm_get_npixels(im) * sizeof(vl_uint8)) ;
  err = vl_pgm_extract_data(f, im, *data) ;

  if (err) {
    vl_free (data) ;
    fclose (f) ;
  }

  fclose (f) ;
  return err ;
}

/** ------------------------------------------------------------------
 ** @brief Read floats from a PGM file.
 ** @param name file name.
 ** @param im a pointer to the PGM image structure to fill.
 ** @param data a pointer to the pointer to the allocated buffer.
 ** @return error code.
 **
 ** The function reads a PGM image from file @a name and initializes
 ** the structure @a im and the buffer @a data accordingly. The buffer
 ** @a data is an array of floats in the range [0, 1].
 **
 ** The ownership of the buffer @a data is transfered to the caller.
 ** @a data should be freed by means of ::vl_free().
 **
 ** @bug Only PGM files with 1 BPP are supported.
 **/

VL_EXPORT
int vl_pgm_read_new_f (char const *name,  VlPgmImage *im, float** data)
{
  int err = 0 ;
  size_t npixels ;
  vl_uint8 *idata ;

  err = vl_pgm_read_new (name, im, &idata) ;
  if (err) {
    return err ;
  }

  npixels = vl_pgm_get_npixels(im) ;
  *data = vl_malloc (sizeof(float) * npixels) ;
  {
    size_t k ;
    float scale = 1.0f / (float)im->max_value ;
    for (k = 0 ; k < npixels ; ++ k) (*data)[k] = scale * idata[k] ;
  }

  vl_free (idata) ;
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @brief Write bytes to a PGM file.
 ** @param name file name.
 ** @param data data to write.
 ** @param width width of the image.
 ** @param height height of the image.
 ** @return error code.
 **
 ** The function dumps the image @a data to the PGM file of the specified
 ** name. This is an helper function simplifying the usage of
 ** vl_pgm_insert().
 **/

VL_EXPORT
int vl_pgm_write (char const *name, vl_uint8 const* data, int width, int height)
{
  int err = 0 ;
  VlPgmImage pgm ;
  FILE *f = fopen (name, "wb") ;

  if (! f) {
    return vl_set_last_error(VL_ERR_PGM_IO,
             "Error opening PGM file '%s' for writing", name) ;
  }

  pgm.width = width ;
  pgm.height = height ;
  pgm.is_raw = 1 ;
  pgm.max_value = 255 ;

  err = vl_pgm_insert (f, &pgm, data) ;
  fclose (f) ;

  return err ;
}

/** -------------------------------------------------------------------
 ** @brief Write floats to PGM file
 ** @param name file name.
 ** @param data data to write.
 ** @param width width of the image.
 ** @param height height of the image.
 ** @return error code.
 **
 ** The function dumps the image @a data to the PGM file of the
 ** specified name. The data is re-scaled to fit in the range 0-255.
 ** This is an helper function simplifying the usage of
 ** vl_pgm_insert().
 **/

VL_EXPORT
int vl_pgm_write_f (char const *name, float const* data, int width, int height)
{
  int err = 0 ;
  int k ;
  float min = + VL_INFINITY_F ;
  float max = - VL_INFINITY_F ;
  float scale ;

  vl_uint8 * buffer = vl_malloc (sizeof(float) * width * height) ;

  for (k = 0 ; k < width * height ; ++k) {
    min = VL_MIN(min, data [k]) ;
    max = VL_MAX(max, data [k]) ;
  }

  scale = 255 / (max - min + VL_EPSILON_F) ;

  for (k = 0 ; k < width * height ; ++k) {
    buffer [k] = (vl_uint8) ((data [k] - min) * scale) ;
  }

  err = vl_pgm_write (name, buffer, width, height) ;

  vl_free (buffer) ;
  return err ;
}
