/** @file getopt_long.h
 ** @brief getopt_long
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_GETOPT_LONG_H
#define VL_GETOPT_LONG_H

#include "generic.h"

VL_EXPORT int    opterr ;   /**< code of the last error occured while parsing an option */
VL_EXPORT int    optind ;   /**< index of the next option to process in @c argv */
VL_EXPORT int    optopt ;   /**< current option */
VL_EXPORT char * optarg ;   /**< argument of the current option */
VL_EXPORT int    optreset ; /**< reset flag */

/** @brief ::getopt_long option */
struct option
{
  const char *name ;  /**< option long name */
  int	      has_arg ; /**< flag indicating whether the option has no, required or optional argument */
  int	     *flag ;    /**< pointer to a variable to set (if @c NULL, the value is returned instead) */
  int	      val ;     /**< value to set or to return */
} ;

#define no_argument       0 /**< ::option with no argument */
#define required_argument 1 /**< ::option with required argument */
#define optional_argument 2 /**< ::option with optional argument */

VL_EXPORT int getopt_long(int argc, char * const argv[],
                          const char * optstring,
                          const struct option * longopts, int * longindex);

/* VL_GETOPT_LONG_H */
#endif
