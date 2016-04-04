/** @file stringop.c
 ** @brief String operations - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
@file stringop.h
@brief String operations
@author Andrea Vedaldi
@tableofcontents

@ref stringop.h implements basic string operations. All functions that
write to strings use range checking, which makes them safer than some
standard POSIX equivalent (see @ref vl-stringop-err).

@section vl-stringop-enumeration Enumerations

@ref stringop.h defines a simple enumeration data type. This is given
by an array of enumeration members, represented by
instances of the ::VlEnumerator strucutre, each storing a
name-value pair. The enumeration must end by a member whose
name is set to @c NULL.

Use ::vl_enumeration_get and ::vl_enumeration_get_casei
to retrieve an enumeration member by name.

@section vl-stringop-file-protocols File protocols

@ref stringop.h defines a few file "protocols" and helps parsing them
from URL-like formatted strings. The supported protocols are:

<table>
<caption>File protocols</caption>
<tr><td>Protocol</td><td>Code</td><td>URL prefix</td></tr>
<tr><td>ASCII</td><td>::VL_PROT_ASCII</td><td><code>ascii://</code></td></tr>
<tr><td>BINARY</td><td>::VL_PROT_BINARY</td><td><code>binary://</code></td></tr>
</table>

@section vl-stringop-err Detecting overflow

@ref stringop.h functions that write a string to a character buffer take
both the buffer and its size @c n as input. If @c n is not large
enough, the output may be truncated but it is always a null terminated
string (provided that @c n &gt;= 1). Such functions also return the
length of the string that would have been written @c r (which does not
include the terminating null character) had the buffer been large
enough.  Hence an <em>overflow</em> can be detected by testing if @c r
&gt;= @c n, @c r can be used to re-allocate a buffer large enough to
contain the result, and the operation can be repeated.
**/

#include "stringop.h"

#include <string.h>
#include <ctype.h>

/** ------------------------------------------------------------------
 ** @brief Extract the protocol prefix from a string
 ** @param string string.
 ** @param protocol protocol code (output).
 ** @return pointer to the first character after the protocol prefix.
 **
 ** The function extracts the prefix of the string @a string
 ** terminated by the first occurrence of the @c :// substring (if
 ** any). It then matches the suffix terminated by @c :// to the
 ** supported @ref vl-stringop-file-protocols protocols. If @c protocol is not
 ** @c NULL, the corresponding protocol code is written to @a protocol
 **
 ** The function writes to @a protocol the value ::VL_PROT_NONE if no
 ** suffix is detected and ::VL_PROT_UNKNOWN if there is a suffix but
 ** it cannot be matched to any of the supported protocols.
 **/

VL_EXPORT char *
vl_string_parse_protocol (char const *string, int *protocol)
{
  char const * cpt ;
  int dummy ;

  /* handle the case prot = 0 */
  if (protocol == 0)
    protocol = &dummy ;

  /* look for :// */
  cpt = strstr(string, "://") ;

  if (cpt == 0) {
    *protocol = VL_PROT_NONE ;
    cpt = string ;
  }
  else {
    if (strncmp(string, "ascii", cpt - string) == 0) {
      *protocol = VL_PROT_ASCII ;
    }
    else if (strncmp(string, "bin",   cpt - string) == 0) {
      *protocol = VL_PROT_BINARY ;
    }
    else {
      *protocol = VL_PROT_UNKNOWN ;
    }
    cpt += 3 ;
  }
  return (char*) cpt ;
}

/** ------------------------------------------------------------------
 ** @brief Get protocol name
 ** @param protocol protocol code.
 ** @return pointer protocol name string.
 **
 ** The function returns a pointer to a string containing the name of
 ** the protocol @a protocol (see the @a vl-file-protocols protocols
 ** list).  If the protocol is unknown the function returns the empty
 ** string.
 **/

VL_EXPORT char const *
vl_string_protocol_name (int protocol)
{
  switch (protocol) {
  case VL_PROT_ASCII:
    return "ascii" ;
  case VL_PROT_BINARY:
    return "bin" ;
  case VL_PROT_NONE :
    return "" ;
  default:
    return 0 ;
  }
}


/** ------------------------------------------------------------------
 ** @brief Extract base of file name
 ** @param destination destination buffer.
 ** @param destinationSize size of destination buffer.
 ** @param source input string.
 ** @param maxNumStrippedExtensions maximum number of extensions to strip.
 ** @return length of the destination string.
 **
 ** The function removes the leading path and up to @c
 ** maxNumStrippedExtensions trailing extensions from the string @a
 ** source and writes the result to the buffer @a destination.
 **
 ** The leading path is the longest suffix that ends with either the
 ** @c \ or @c / characters. An extension is a string starting with
 ** the <code>.</code> character not containing it. For instance, the string @c
 ** file.png contains the extension <code>.png</code> and the string @c
 ** file.tar.gz contains two extensions (<code>.tar</code> and @c <code>.gz</code>).
 **
 ** @sa @ref vl-stringop-err.
 **/

VL_EXPORT vl_size
vl_string_basename (char * destination,
                    vl_size destinationSize,
                    char const * source,
                    vl_size maxNumStrippedExtensions)
{
  char c ;
  vl_uindex k = 0, beg, end ;

  /* find beginning */
  beg = 0 ;
  for (k = 0 ; (c = source[k]) ; ++ k) {
    if (c == '\\' || c == '/') beg = k + 1 ;
  }

  /* find ending */
  end = strlen (source) ;
  for (k = end ; k > beg ; --k) {
    if (source[k - 1] == '.' && maxNumStrippedExtensions > 0) {
      -- maxNumStrippedExtensions ;
      end = k - 1 ;
    }
  }

  return vl_string_copy_sub (destination, destinationSize,
                             source + beg, source + end) ;
}

/** ------------------------------------------------------------------
 ** @brief Replace wildcard characters by a string
 ** @param destination output buffer.
 ** @param destinationSize size of the output buffer.
 ** @param source input string.
 ** @param wildcardChar wildcard character.
 ** @param escapeChar escape character.
 ** @param replacement replacement string.
 **
 ** The function replaces the occurrence of the specified wildcard
 ** character @a wildcardChar by the string @a replacement. The result
 ** is written to the buffer @a destination of size @a
 ** destinationSize.
 **
 ** Wildcard characters may be escaped by preceding them by the @a esc
 ** character. More in general, anything following an occurrence of @a
 ** esc character is copied verbatim. To disable the escape characters
 ** simply set @a esc to 0.
 **
 ** @return length of the result.
 ** @sa @ref vl-stringop-err.
 **/

VL_EXPORT vl_size
vl_string_replace_wildcard (char * destination,
                            vl_size destinationSize,
                            char const * source,
                            char wildcardChar,
                            char escapeChar,
                            char const * replacement)
{
  char c ;
  vl_uindex k = 0 ;
  vl_bool escape = 0 ;

  while ((c = *source++)) {

    /* enter escape mode ? */
    if (! escape && c == escapeChar) {
      escape = 1 ;
      continue ;
    }

    /* wildcard or regular? */
    if (! escape && c == wildcardChar) {
      char const * repl = replacement ;
      while ((c = *repl++)) {
        if (destination && k + 1 < destinationSize) {
          destination[k] = c ;
        }
        ++ k ;
      }
    }
    /* regular character */
    else {
      if (destination && k + 1 < destinationSize) {
        destination[k] = c ;
      }
      ++ k ;
    }
    escape = 0 ;
  }

  /* add trailing 0 */
  if (destinationSize > 0) {
    destination[VL_MIN(k, destinationSize - 1)] = 0 ;
  }
  return  k ;
}

/** ------------------------------------------------------------------
 ** @brief Copy string
 ** @param destination output buffer.
 ** @param destinationSize size of the output buffer.
 ** @param source string to copy.
 ** @return length of the source string.
 **
 ** The function copies the string @a source to the buffer @a
 ** destination of size @a destinationSize.
 **
 ** @sa @ref vl-stringop-err.
 **/

VL_EXPORT vl_size
vl_string_copy (char * destination, vl_size destinationSize,
                char const * source)
{
  char c ;
  vl_uindex k = 0 ;

  while ((c = *source++)) {
    if (destination && k + 1 < destinationSize) {
      destination[k] = c ;
    }
    ++ k ;
  }

  /* finalize */
  if (destinationSize > 0) {
    destination[VL_MIN(k, destinationSize - 1)] = 0 ;
  }
  return  k ;
}

/** ------------------------------------------------------------------
 ** @brief Copy substring
 ** @param destination output buffer.
 ** @param destinationSize  size of output buffer.
 ** @param beginning start of the substring.
 ** @param end end of the substring.
 ** @return length of the destination string.
 **
 ** The function copies the substring from at @a beginning to @a end
 ** (not included) to the buffer @a destination of size @a
 ** destinationSize. If, however, the null character is found before
 ** @a end, the substring terminates there.
 **
 ** @sa @ref vl-stringop-err.
 **/

VL_EXPORT vl_size
vl_string_copy_sub (char * destination,
                    vl_size destinationSize,
                    char const * beginning,
                    char const * end)
{
  char c ;
  vl_uindex k = 0 ;

  while (beginning < end && (c = *beginning++)) {
    if (destination && k + 1 < destinationSize) {
      destination[k] = c ;
    }
    ++ k ;
  }

  /* finalize */
  if (destinationSize > 0) {
    destination[VL_MIN(k, destinationSize - 1)] = 0 ;
  }
  return  k ;
}

/** ------------------------------------------------------------------
 ** @brief Search character in reversed order
 ** @param beginning pointer to the substring beginning.
 ** @param end pointer to the substring end.
 ** @param c character to search for.
 ** @return pointer to last occurrence of @a c, or 0 if none.
 **
 ** The function searches for the last occurrence of the character @a c
 ** in the substring from @a beg to @a end (the latter not being included).
 **/

VL_EXPORT char *
vl_string_find_char_rev (char const *beginning, char const* end, char c)
{
  while (end -- != beginning) {
    if (*end == c) {
      return (char*) end ;
    }
  }
  return 0 ;
}

/** ------------------------------------------------------------------
 ** @brief Calculate string length
 ** @param string string.
 ** @return string length.
 **/

VL_EXPORT vl_size
vl_string_length (char const *string)
{
  vl_uindex i ;
  for (i = 0 ; string[i] ; ++i) ;
  return i ;
}

/** ------------------------------------------------------------------
 ** @brief Compare strings case-insensitive
 ** @param string1 fisrt string.
 ** @param string2 second string.
 ** @return an integer =,<,> 0 if @c string1 =,<,> @c string2
 **/

VL_EXPORT int
vl_string_casei_cmp (const char * string1, const char * string2)
{
  while (tolower((char unsigned)*string1) ==
         tolower((char unsigned)*string2))
  {
    if (*string1 == 0) {
      return 0 ;
    }
    string1 ++ ;
    string2 ++ ;
  }
  return
    (int)tolower((char unsigned)*string1) -
    (int)tolower((char unsigned)*string2) ;
}

/* -------------------------------------------------------------------
 *                                                       VlEnumeration
 * ---------------------------------------------------------------- */

/** @brief Get a member of an enumeration by name
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

VL_EXPORT VlEnumerator *
vl_enumeration_get (VlEnumerator const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (strcmp(name, enumeration->name) == 0) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by name (case insensitive)
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned. @a string is matched case insensitive.
 **
 **  @sa vl-stringop-enumeration
 **/

VL_EXPORT VlEnumerator *
vl_enumeration_get_casei (VlEnumerator const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (vl_string_casei_cmp(name, enumeration->name) == 0) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by value
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param value value of the desired member.
 ** @return enumerator matching @a value.
 **
 ** If @a value is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

VL_EXPORT VlEnumerator *
vl_enumeration_get_by_value (VlEnumerator const *enumeration, vl_index value)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (enumeration->value == value) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

