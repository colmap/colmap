/** @file stringop.h
 ** @brief String operations
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_STRINGOP_H
#define VL_STRINGOP_H

#include "generic.h"

/** @brief File protocols */
enum {
  VL_PROT_UNKNOWN = -1, /**< unknown protocol */
  VL_PROT_NONE    =  0, /**< no protocol      */
  VL_PROT_ASCII,        /**< ASCII protocol   */
  VL_PROT_BINARY        /**< Binary protocol  */
} ;


VL_EXPORT vl_size vl_string_copy (char *destination, vl_size destinationSize, char const *source) ;
VL_EXPORT vl_size vl_string_copy_sub (char *destination, vl_size destinationSize,
                                      char const *beginning, char const *end) ;
VL_EXPORT char *vl_string_parse_protocol (char const *string, int *protocol) ;
VL_EXPORT char const *vl_string_protocol_name (int prot) ;
VL_EXPORT vl_size vl_string_basename (char *destination, vl_size destinationSize,
                                      char const *source, vl_size maxNumStrippedExtension) ;
VL_EXPORT vl_size vl_string_replace_wildcard (char * destination, vl_size destinationSize,
                                              char const *src, char wildcardChar, char escapeChar,
                                              char const *replacement) ;
VL_EXPORT char *vl_string_find_char_rev (char const *beginning, char const *end, char c) ;
VL_EXPORT vl_size vl_string_length (char const *string) ;
VL_EXPORT int vl_string_casei_cmp (const char *string1, const char *string2) ;

/** @name String enumerations
 ** @{ */

/** @brief Member of an enumeration */
typedef struct _VlEnumerator
{
  char const *name ; /**< enumeration member name. */
  vl_index value ;   /**< enumeration member value. */
} VlEnumerator ;

VL_EXPORT VlEnumerator *vl_enumeration_get (VlEnumerator const *enumeration, char const *name) ;
VL_EXPORT VlEnumerator *vl_enumeration_get_casei (VlEnumerator const *enumeration, char const *name) ;
VL_EXPORT VlEnumerator *vl_enumeration_get_by_value (VlEnumerator const *enumeration, vl_index value) ;
/** @} */

/* VL_STRINGOP_H */
#endif
