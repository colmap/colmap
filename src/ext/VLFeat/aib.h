/** @file aib.h
 ** @brief AIB (@ref aib)
 ** @author Brian Fulkerson
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_AIB_H
#define VL_AIB_H

#include "generic.h"
#include "mathop.h"

/** ------------------------------------------------------------------
 ** @internal
 ** @brief AIB algorithm data
 **
 ** The implementation is quite straightforward, but the way feature
 ** values are handled in order to support efficient joins,
 ** deletions and re-arrangement needs to be explained. This is
 ** achieved by adding a layer of indirection:
 ** - Call each feature value (either original or obtained by a join
 **   operation) a <em>node</em>. Nodes are identified by numbers.
 ** - Call each element of the various arrays (such as VlAIB::Px)
 **    an <em>entry</em>.
 ** - Entries are dynamically associated to nodes as specified by
 **   VlAIB::nodes. For example, @c Px[i] refers to the node @c
 **   nodes[i].
 **/

typedef struct _VlAIB
{
  vl_uint   *nodes ;    /**< Entires to nodes */
  vl_uint    nentries ; /**< Total number of entries (= # active nodes) */
  double    *beta ;     /**< Minimum distance to an entry  */
  vl_uint   *bidx ;     /**< Closest entry */


  vl_uint   *which ;    /**< List of entries to update */
  vl_uint    nwhich ;   /**< Number of entries to update */

  double    *Pcx;       /**< Joint probability table */
  double    *Px;        /**< Marginal. */
  double    *Pc;        /**< Marginal. */
  vl_uint    nvalues;   /**< Number of feature values */
  vl_uint    nlabels;   /**< Number of labels */

  vl_uint   *parents;   /**< Array of parents */
  double    *costs;     /**< Cost of each merge */

  vl_uint verbosity ; /** Verbosity level */
} VlAIB;

/** @name Create and destroy
 ** @{
 **/
VL_EXPORT
VlAIB * vl_aib_new(double * Pcx, vl_uint nvalues, vl_uint nlabels);

VL_EXPORT
void vl_aib_delete (VlAIB * aib);
/** @} */

/** @name Process data
 ** @{
 **/
VL_EXPORT
void vl_aib_process(VlAIB * aib);
/** @} */

/** @name Retrieve results
 ** @{
 **/
VL_INLINE vl_uint * vl_aib_get_parents(VlAIB const * aib);
VL_INLINE double  * vl_aib_get_costs(VlAIB const * aib);
/** @} */


/* -------------------------------------------------------------------
 *                                     Inline functions implementation
 * ---------------------------------------------------------------- */

/** ------------------------------------------------------------------
 ** @brief Get resulting list of parents
 ** @param aib AIB filter.
 ** @return An array of parents
 **/
VL_INLINE
vl_uint * vl_aib_get_parents(VlAIB const * aib)
{
  return aib->parents;
}

/** ------------------------------------------------------------------
 ** @brief Get a list of merge costs
 ** @param aib AIB filter.
 ** @return An array of costs
 **/
VL_INLINE
double * vl_aib_get_costs(VlAIB const * aib)
{
  return aib->costs;
}

/* ----------------------------------------------------------------- */
/** @brief Set the verbosity
 ** @param self AIB object.
 ** @param verbosity a non-negative integer.
 **/
VL_INLINE void
vl_aib_set_verbosity (VlAIB * self, int verbosity)
{
  self->verbosity = verbosity ;
}

/** @brief Get the verbosity
 ** @param self AIB object.
 ** @return the verbosity level.
 **/
VL_INLINE int
vl_aib_get_verbosity (VlAIB const * self)
{
  return self->verbosity ;
}

/* VL_AIB_H */
#endif
