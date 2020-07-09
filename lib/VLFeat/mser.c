/** @file mser.c
 ** @brief MSER - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-13 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page mser Maximally Stable Extremal Regions (MSER)
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref mser.h implements the *Maximally Stable Extremal Regions* (MSER)
local feature detector of @cite{matas03robust}. This detector extracts
as features the the connected components of the level sets of the
input intensity image. Among all such regions, the ones that are
locally maximally stable are selected. MSERs are affine co-variant, as
well as largely co-variant to generic diffeomorphic transformations.

See @ref mser-starting for an introduction on how to use the detector
from the C API. For further details refer to:

- @subpage mser-fundamentals - MSER definition and parameters.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section mser-starting Getting started with the MSER detector
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Running the MSER filter usually involves the following steps:

- Initialize the MSER filter by ::vl_mser_new(). The
  filter can be reused for images of the same size.
- Compute the MSERs by ::vl_mser_process().
- Optionally fit ellipsoids to the MSERs by  ::vl_mser_ell_fit().
- Retrieve the results by ::vl_mser_get_regions() (and optionally ::vl_mser_get_ell()).
- Optionally retrieve filter statistics by ::vl_mser_get_stats().
- Delete the MSER filter by ::vl_mser_delete().

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page mser-fundamentals MSER fundamentals
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The *extermal regions* of an image are the connected components of the
level sets $S_l = \{ x : I(x) \leq l \}, l \in \real$ of the image
$I(x)$. Consider a discretization of the intensity levels $l$
consisting of $M$ samples $\mathcal{L}=\{0,\dots,M-1\}$. The extremal
regions $R_l \subset S_l$ of the level sets $S_l, l \in \mathcal{L}$
can be arranged in a tree, where a region $R_l$ is a children of a
region $R_{l+1}$ if $R_l \subset R_{l+1}$. The following figures shows
a 1D example where the regions are denoted by dark thick lines:

@image html mser-tree.png "Connected components of the image level sets arranged in a tree."

Note that, depending on the image, regions at different levels can be
identical as sets:

@image html mser-er-step.png "Connected components when the image contains step changes."

A *stable extremal region* is an extremal region that does not change
much as the index $l$ is varied. Here we use a criterion which is
similar but not identical to the original paper. This definition is
somewhat simpler both to understand and code.

Let $B(R_l)=(R_l,R_{l+1},\dots,R_{l+\Delta})$ be the branch of the
tree $R_l \subset R_{l+1} \subset \dots \subset R_{l + \Delta}$
rooted at $R_l$. We associate to the branch the (in)stability score

@f[
  v(R_l) = \frac{|R_{l+\Delta} - R_l|}{|R_l|}.
@f]

This score is a relative measure of how much $R_l$ changes as the
index is increased from $l$ to $l+\Delta$, as illustrated in the
following figure.

@image html mser-er.png "Stability is measured by looking at how much a region changes with the intensity level."

The score is low if the regions along the branch have similar area
(and thus similar shape). We aim to select maximally stable
branches; then a maximally stable region is just a representative
region selected from a maximally stable branch (for simplicity we
select $R_l$, but one could choose for example
$R_{l+\Delta/2}$).

Roughly speaking, a branch is maximally stable if it is a local
minimum of the (in)stability score. More accurately, we start by
assuming that all branches are maximally stable. Then we consider
each branch $B(R_{l})$ and its parent branch
$B(R_{l+1}):R_{l+1}\supset R_l$ (notice that, due to the
discrete nature of the calculations, they might be geometrically
identical) and we mark as unstable the less stable one, i.e.:

  - if $v(R_l)<v(R_{l+1})$, mark $R_{l+1}$ as unstable;
  - if $v(R_l)>v(R_{l+1})$, mark $R_{l}$ as unstable;
  - otherwise, do nothing.

This criterion selects among nearby regions the ones that are more
stable. We optionally refine the selection by running (starting
from the bigger and going to the smaller regions) the following
tests:

- $a_- \leq |R_{l}|/|R_{\infty}| \leq a_+$: exclude MSERs too
  small or too big ($|R_{\infty}|$ is the area of the image).

- $v(R_{l}) < v_+$: exclude MSERs too unstable.

- For any MSER $R_l$, find the parent MSER $R_{l'}$ and check
  if
  $|R_{l'} - R_l|/|R_l'| < d_+$: remove duplicated MSERs.

 <table>
 <tr>
  <td>parameter</td>
  <td>alt. name</td>
  <td>standard value</td>
  <td>set by</td>
 </tr>
 <tr>
   <td>$\Delta$</td>
   <td>@c delta</td>
   <td>5</td>
   <td>::vl_mser_set_delta()</td>
 </tr>
 <tr>
   <td>$a_+$</td>
   <td>@c max_area</td>
   <td>0.75</td>
   <td>::vl_mser_set_max_area()</td>
 </tr>
 <tr>
   <td>$a_-$</td>
   <td>@c min_area</td>
   <td>3.0/$|R_\infty|$</td>
   <td>::vl_mser_set_min_area()</td>
 </tr>
 <tr>
   <td>$v_+$</td>
   <td>@c max_var</td>
   <td>0.25</td>
   <td>::vl_mser_set_max_variation()</td>
 </tr>
 <tr>
   <td>$d_+$</td>
   <td>@c min_diversity</td>
   <td>0.2</td>
   <td>::vl_mser_set_min_diversity()</td>
 </tr>
</table>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section mser-vol Volumetric images
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The code supports images of arbitrary dimension. For instance, it
is possible to find the MSER regions of volumetric images or time
sequences. See ::vl_mser_new() for further details

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section mser-ell Ellipsoids
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Usually extremal regions are returned as a set of ellipsoids
fitted to the actual regions (which have arbitrary shape). The fit
is done by calculating the mean and variance of the pixels
composing the region:
@f[
\mu_l = \frac{1}{|R_l|}\sum_{x\in R_l}x,
\qquad
\Sigma_l = \frac{1}{|R_l|}\sum_{x\in R_l} (x-\mu_l)^\top(x-\mu_l)
@f]
Ellipsoids are fitted by ::vl_mser_ell_fit().  Notice that for a
<em>n</em> dimensional image, the mean has <em>n</em> components
and the variance has <em>n(n+1)/2</em> independent components. The
total number of components is obtained by ::vl_mser_get_ell_dof()
and the total number of fitted ellipsoids by
::vl_mser_get_ell_num(). A matrix with an ellipsoid per column is
returned by ::vl_mser_get_ell(). The column is the stacking of the
mean and of the independent components of the variance, in the
order <em>(1,1),(1,2),..,(1,n), (2,2),(2,3)...</em>. In the
calculations, the pixel coordinate $x=(x_1,...,x_n)$ use the
standard index order and ranges.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section mser-algo Algorithm
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The algorithm is quite efficient. While some details may be
tricky, the overall idea is easy to grasp.

- Pixels are sorted by increasing intensity.
- Pixels are added to a forest by increasing intensity. The forest has the
  following properties:
  - All the descendent of a certain pixels are subset of an extremal region.
  - All the extremal regions are the descendants of some pixels.
- Extremal regions are extracted from the region tree and the extremal regions tree is
  calculated.
- Stable regions are marked.
- Duplicates and other bad regions are removed.

@remark The extremal region tree which is calculated is a subset
of the actual extremal region tree. In particular, it does not
contain redundant entries extremal regions that coincide as
sets. So, for example, in the calculated extremal region tree, the
parent $R_q$ of an extremal region $R_{l}$ may or may
<em>not</em> correspond to $R_{l+1}$, depending whether
$q\leq l+1$ or not. These subtleties are important when
calculating the stability tests.

**/

#include "mser.h"
#include<stdlib.h>
#include<string.h>
#include<assert.h>

/** -------------------------------------------------------------------
 ** @brief Advance N-dimensional subscript
 **
 ** The function increments by one the subscript @a subs indexing an
 ** array the @a ndims dimensions @a dims.
 **
 ** @param ndims number of dimensions.
 ** @param dims dimensions.
 ** @param subs subscript to advance.
 **/

VL_INLINE void
adv(int ndims, int const *dims, int *subs)
{
  int d = 0 ;
  while(d < ndims) {
    if( ++subs[d]  < dims[d] ) return ;
    subs[d++] = 0 ;
  }
}

/** -------------------------------------------------------------------
 ** @brief Climb the region forest to reach aa root
 **
 ** The function climbs the regions forest @a r starting from the node
 ** @a idx to the corresponding root.
 **
 ** To speed-up the operation, the function uses the
 ** VlMserReg::shortcut field to quickly jump to the root. After the
 ** root is reached, all the used shortcut are updated.
 **
 ** @param r regions' forest.
 ** @param idx stating node.
 ** @return index of the reached root.
 **/

VL_INLINE vl_uint
climb (VlMserReg* r, vl_uint idx)
{

  vl_uint prev_idx = idx ;
  vl_uint next_idx ;
  vl_uint root_idx ;

  /* move towards root to find it */
  while (1) {

    /* next jump to the root */
    next_idx = r [idx] .shortcut ;

    /* recycle shortcut to remember how we came here */
    r [idx] .shortcut = prev_idx ;

    /* stop if the root is found */
    if( next_idx == idx ) break ;

    /* next guy */
    prev_idx = idx ;
    idx      = next_idx ;
  }

  root_idx = idx ;

  /* move backward to update shortcuts */
  while (1) {

    /* get previously visited one */
    prev_idx = r [idx] .shortcut ;

    /* update shortcut to point to the new root */
    r [idx] .shortcut = root_idx ;

    /* stop if the first visited node is reached */
    if( prev_idx == idx ) break ;

    /* next guy */
    idx = prev_idx ;
  }

  return root_idx ;
}

/** -------------------------------------------------------------------
 ** @brief Create a new MSER filter
 **
 ** Initializes a new MSER filter for images of the specified
 ** dimensions. Images are @a ndims -dimensional arrays of dimensions
 ** @a dims.
 **
 ** @param ndims number of dimensions.
 ** @param dims  dimensions.
 **/
VL_EXPORT
VlMserFilt*
vl_mser_new (int ndims, int const* dims)
{
  VlMserFilt* f ;
  int *strides, k ;

  f = vl_calloc (sizeof(VlMserFilt), 1) ;

  f-> ndims   = ndims ;
  f-> dims    = vl_malloc (sizeof(int) * ndims) ;
  f-> subs    = vl_malloc (sizeof(int) * ndims) ;
  f-> dsubs   = vl_malloc (sizeof(int) * ndims) ;
  f-> strides = vl_malloc (sizeof(int) * ndims) ;

  /* shortcuts */
  strides = f-> strides ;

  /* copy dims to f->dims */
  for(k = 0 ; k < ndims ; ++k) {
    f-> dims [k] = dims [k] ;
  }

  /* compute strides to move into the N-dimensional image array */
  strides [0] = 1 ;
  for(k = 1 ; k < ndims ; ++k) {
    strides [k] = strides [k-1] * dims [k-1] ;
  }

  /* total number of pixels */
  f-> nel = strides [ndims-1] * dims [ndims-1] ;

  /* dof of ellipsoids */
  f-> dof = ndims * (ndims + 1) / 2 + ndims ;

  /* more buffers */
  f-> perm   = vl_malloc (sizeof(vl_uint)   * f-> nel) ;
  f-> joins  = vl_malloc (sizeof(vl_uint)   * f-> nel) ;
  f-> r      = vl_malloc (sizeof(VlMserReg) * f-> nel) ;

  f-> er     = 0 ;
  f-> rer    = 0 ;
  f-> mer    = 0 ;
  f-> rmer   = 0 ;
  f-> ell    = 0 ;
  f-> rell   = 0 ;

  /* other parameters */
  f-> delta         = 5 ;
  f-> max_area      = 0.75 ;
  f-> min_area      = 3.0 / f-> nel ;
  f-> max_variation = 0.25 ;
  f-> min_diversity = 0.2 ;

  return f ;
}

/** -------------------------------------------------------------------
 ** @brief Delete MSER filter
 **
 ** The function releases the MSER filter @a f and all its resources.
 **
 ** @param f MSER filter to be deleted.
 **/
VL_EXPORT
void
vl_mser_delete (VlMserFilt* f)
{
  if(f) {
    if(f-> acc   )  vl_free( f-> acc    ) ;
    if(f-> ell   )  vl_free( f-> ell    ) ;

    if(f-> er    )  vl_free( f-> er     ) ;
    if(f-> r     )  vl_free( f-> r      ) ;
    if(f-> joins )  vl_free( f-> joins  ) ;
    if(f-> perm  )  vl_free( f-> perm   ) ;

    if(f-> strides) vl_free( f-> strides) ;
    if(f-> dsubs  ) vl_free( f-> dsubs  ) ;
    if(f-> subs   ) vl_free( f-> subs   ) ;
    if(f-> dims   ) vl_free( f-> dims   ) ;

    if(f-> mer    ) vl_free( f-> mer    ) ;
    vl_free (f) ;
  }
}


/** -------------------------------------------------------------------
 ** @brief Process image
 **
 ** The functions calculates the Maximally Stable Extremal Regions
 ** (MSERs) of image @a im using the MSER filter @a f.
 **
 ** The filter @a f must have been initialized to be compatible with
 ** the dimensions of @a im.
 **
 ** @param f MSER filter.
 ** @param im image data.
 **/
VL_EXPORT
void
vl_mser_process (VlMserFilt* f, vl_mser_pix const* im)
{
  /* shortcuts */
  vl_uint        nel     = f-> nel  ;
  vl_uint       *perm    = f-> perm ;
  vl_uint       *joins   = f-> joins ;
  int            ndims   = f-> ndims ;
  int           *dims    = f-> dims ;
  int           *subs    = f-> subs ;
  int           *dsubs   = f-> dsubs ;
  int           *strides = f-> strides ;
  VlMserReg     *r       = f-> r ;
  VlMserExtrReg *er      = f-> er ;
  vl_uint       *mer     = f-> mer ;
  int            delta   = f-> delta ;

  int njoins = 0 ;
  int ner    = 0 ;
  int nmer   = 0 ;
  int nbig   = 0 ;
  int nsmall = 0 ;
  int nbad   = 0 ;
  int ndup   = 0 ;

  int i, j, k ;

  /* delete any previosuly computed ellipsoid */
  f-> nell = 0 ;

  /* -----------------------------------------------------------------
   *                                          Sort pixels by intensity
   * -------------------------------------------------------------- */

  {
    vl_uint buckets [ VL_MSER_PIX_MAXVAL ] ;

    /* clear buckets */
    memset (buckets, 0, sizeof(vl_uint) * VL_MSER_PIX_MAXVAL ) ;

    /* compute bucket size (how many pixels for each intensity
       value) */
    for(i = 0 ; i < (int) nel ; ++i) {
      vl_mser_pix v = im [i] ;
      ++ buckets [v] ;
    }

    /* cumulatively add bucket sizes */
    for(i = 1 ; i < VL_MSER_PIX_MAXVAL ; ++i) {
      buckets [i] += buckets [i-1] ;
    }

    /* empty buckets computing pixel ordering */
    for(i = nel ; i >= 1 ; ) {
      vl_mser_pix v = im [ --i ] ;
      vl_uint j = -- buckets [v] ;
      perm [j] = i ;
    }
  }

  /* initialize the forest with all void nodes */
  for(i = 0 ; i < (int) nel ; ++i) {
    r [i] .parent = VL_MSER_VOID_NODE ;
  }

  /* -----------------------------------------------------------------
   *                        Compute regions and count extremal regions
   * -------------------------------------------------------------- */
  /*
     In the following:

     idx    : index of the current pixel
     val    : intensity of the current pixel
     r_idx  : index of the root of the current pixel
     n_idx  : index of the neighbors of the current pixel
     nr_idx : index of the root of the neighbor of the current pixel

  */

  /* process each pixel by increasing intensity */
  for(i = 0 ; i < (int) nel ; ++i) {

    /* pop next node xi */
    vl_uint     idx = perm [i] ;
    vl_mser_pix val = im [idx] ;
    vl_uint     r_idx ;

    /* add the pixel to the forest as a root for now */
    r [idx] .parent   = idx ;
    r [idx] .shortcut = idx ;
    r [idx] .area     = 1 ;
    r [idx] .height   = 1 ;

    r_idx = idx ;

    /* convert the index IDX into the subscript SUBS; also initialize
       DSUBS to (-1,-1,...,-1) */
    {
      vl_uint temp = idx ;
      for(k = ndims - 1 ; k >= 0 ; --k) {
        dsubs [k] = -1 ;
        subs  [k] = temp / strides [k] ;
        temp      = temp % strides [k] ;
      }
    }

    /* examine the neighbors of the current pixel */
    while (1) {
      vl_uint n_idx = 0 ;
      vl_bool good = 1 ;

      /*
         Compute the neighbor subscript as NSUBS+SUB, the
         corresponding neighbor index NINDEX and check that the
         neighbor is within the image domain.
      */
      for(k = 0 ; k < ndims && good ; ++k) {
        int temp  = dsubs [k] + subs [k] ;
        good     &= (0 <= temp) && (temp < dims [k]) ;
        n_idx    += temp * strides [k] ;
      }

      /*
         The neighbor should be processed if the following conditions
         are met:

         1. The neighbor is within image boundaries.

         2. The neighbor is indeed different from the current node
            (the opposite happens when DSUB=(0,0,...,0)).

         3. The neighbor is already in the forest, meaning that it has
            already been processed.
      */
      if (good &&
          n_idx != idx &&
          r [n_idx] .parent != VL_MSER_VOID_NODE ) {

        vl_mser_pix nr_val = 0 ;
        vl_uint     nr_idx = 0 ;
        int         hgt   = r [ r_idx] .height ;
        int         n_hgt = r [nr_idx] .height ;

        /*
          Now we join the two subtrees rooted at

           R_IDX = ROOT(  IDX)
          NR_IDX = ROOT(N_IDX).

          Note that R_IDX = ROOT(IDX) might change as we process more
          neighbors, so we need keep updating it.
        */

         r_idx = climb(r,   idx) ;
        nr_idx = climb(r, n_idx) ;

        /*
          At this point we have three possibilities:

          (A) ROOT(IDX) == ROOT(NR_IDX). In this case the two trees
              have already been joined and we do not do anything.

          (B) I(ROOT(IDX)) == I(ROOT(NR_IDX)). In this case the pixel
              IDX is extending an extremal region with the same
              intensity value. Since ROOT(NR_IDX) will NOT be an
              extremal region of the full image, ROOT(IDX) can be
              safely added as children of ROOT(NR_IDX) if this
              reduces the height according to the union rank
              heuristic.

          (C) I(ROOT(IDX)) > I(ROOT(NR_IDX)). In this case the pixel
              IDX is starting a new extremal region. Thus ROOT(NR_IDX)
              WILL be an extremal region of the final image and the
              only possibility is to add ROOT(NR_IDX) as children of
              ROOT(IDX), which becomes parent.
        */

        if( r_idx != nr_idx ) { /* skip if (A) */

          nr_val = im [nr_idx] ;

          if( nr_val == val && hgt < n_hgt ) {

            /* ROOT(IDX) becomes the child */
            r [r_idx]  .parent   = nr_idx ;
            r [r_idx]  .shortcut = nr_idx ;
            r [nr_idx] .area    += r [r_idx] .area ;
            r [nr_idx] .height   = VL_MAX(n_hgt, hgt+1) ;

            joins [njoins++] = r_idx ;

          } else {

            /* cases ROOT(IDX) becomes the parent */
            r [nr_idx] .parent   = r_idx ;
            r [nr_idx] .shortcut = r_idx ;
            r [r_idx]  .area    += r [nr_idx] .area ;
            r [r_idx]  .height   = VL_MAX(hgt, n_hgt + 1) ;

            joins [njoins++] = nr_idx ;

            /* count if extremal */
            if (nr_val != val) ++ ner ;

          } /* check b vs c */
        } /* check a vs b or c */
      } /* neighbor done */

      /* move to next neighbor */
      k = 0 ;
      while(++ dsubs [k] > 1) {
        dsubs [k++] = -1 ;
        if(k == ndims) goto done_all_neighbors ;
      }
    } /* next neighbor */
  done_all_neighbors : ;
  } /* next pixel */

  /* the last root is extremal too */
  ++ ner ;

  /* save back */
  f-> njoins = njoins ;

  f-> stats. num_extremal = ner ;

  /* -----------------------------------------------------------------
   *                                          Extract extremal regions
   * -------------------------------------------------------------- */

  /*
     Extremal regions are extracted and stored into the array ER.  The
     structure R is also updated so that .SHORTCUT indexes the
     corresponding extremal region if any (otherwise it is set to
     VOID).
  */

  /* make room */
  if (f-> rer < ner) {
    if (er) vl_free (er) ;
    f->er  = er = vl_malloc (sizeof(VlMserExtrReg) * ner) ;
    f->rer = ner ;
  } ;

  /* save back */
  f-> nmer = ner ;

  /* count again */
  ner = 0 ;

  /* scan all regions Xi */
  for(i = 0 ; i < (int) nel ; ++i) {

    /* pop next node xi */
    vl_uint     idx = perm [i] ;

    vl_mser_pix val   = im [idx] ;
    vl_uint     p_idx = r  [idx] .parent ;
    vl_mser_pix p_val = im [p_idx] ;

    /* is extremal ? */
    vl_bool is_extr = (p_val > val) || idx == p_idx ;

    if( is_extr ) {

      /* if so, add it */
      er [ner] .index      = idx ;
      er [ner] .parent     = ner ;
      er [ner] .value      = im [idx] ;
      er [ner] .area       = r  [idx] .area ;

      /* link this region to this extremal region */
      r [idx] .shortcut = ner ;

      /* increase count */
      ++ ner ;
    } else {
      /* link this region to void */
      r [idx] .shortcut =   VL_MSER_VOID_NODE ;
    }
  }

  /* -----------------------------------------------------------------
   *                                   Link extremal regions in a tree
   * -------------------------------------------------------------- */

  for(i = 0 ; i < ner ; ++i) {

    vl_uint idx = er [i] .index ;

    do {
      idx = r[idx] .parent ;
    } while (r[idx] .shortcut == VL_MSER_VOID_NODE) ;

    er[i] .parent   = r[idx] .shortcut ;
    er[i] .shortcut = i ;
  }

  /* -----------------------------------------------------------------
   *                            Compute variability of +DELTA branches
   * -------------------------------------------------------------- */
  /* For each extremal region Xi of value VAL we look for the biggest
   * parent that has value not greater than VAL+DELTA. This is dubbed
   * `top parent'. */

  for(i = 0 ; i < ner ; ++i) {

    /* Xj is the current region the region and Xj are the parents */
    int     top_val = er [i] .value + delta ;
    int     top     = er [i] .shortcut ;

    /* examine all parents */
    while (1) {
      int next     = er [top]  .parent ;
      int next_val = er [next] .value ;

      /* Break if:
       * - there is no node above the top or
       * - the next node is above the top value.
       */
      if (next == top || next_val > top_val) break ;

      /* so next could be the top */
      top = next ;
    }

    /* calculate branch variation */
    {
      int area     = er [i  ] .area ;
      int area_top = er [top] .area ;
      er [i] .variation  = (float) (area_top - area) / area ;
      er [i] .max_stable = 1 ;
    }

    /* Optimization: since extremal regions are processed by
     * increasing intensity, all next extremal regions being processed
     * have value at least equal to the one of Xi. If any of them has
     * parent the parent of Xi (this comprises the parent itself), we
     * can safely skip most intermediate node along the branch and
     * skip directly to the top to start our search. */
    {
      int parent = er [i] .parent ;
      int curr   = er [parent] .shortcut ;
      er [parent] .shortcut =  VL_MAX (top, curr) ;
    }
  }

  /* -----------------------------------------------------------------
   *                                  Select maximally stable branches
   * -------------------------------------------------------------- */

  nmer = ner ;
  for(i = 0 ; i < ner ; ++i) {
    vl_uint    parent = er [i     ] .parent ;
    vl_mser_pix   val = er [i     ] .value ;
    float     var = er [i     ] .variation ;
    vl_mser_pix p_val = er [parent] .value ;
    float   p_var = er [parent] .variation ;
    vl_uint     loser ;

    /*
       Notice that R_parent = R_{l+1} only if p_val = val + 1. If not,
       this and the parent region coincide and there is nothing to do.
    */
    if(p_val > val + 1) continue ;

    /* decide which one to keep and put that in loser */
    if(var < p_var) loser = parent ; else loser = i ;

    /* make loser NON maximally stable */
    if(er [loser] .max_stable) {
      -- nmer ;
      er [loser] .max_stable = 0 ;
    }
  }

  f-> stats. num_unstable = ner - nmer ;

  /* -----------------------------------------------------------------
   *                                                 Further filtering
   * -------------------------------------------------------------- */
  /* It is critical for correct duplicate detection to remove regions
   * from the bottom (smallest one first).                          */
  {
    float max_area = (float) f-> max_area * nel ;
    float min_area = (float) f-> min_area * nel ;
    float max_var  = (float) f-> max_variation ;
    float min_div  = (float) f-> min_diversity ;

    /* scan all extremal regions (intensity value order) */
    for(i = ner-1 ; i >= 0L  ; --i) {

      /* process only maximally stable extremal regions */
      if (! er [i] .max_stable) continue ;

      if (er [i] .variation >= max_var ) { ++ nbad ;   goto remove ; }
      if (er [i] .area      >  max_area) { ++ nbig ;   goto remove ; }
      if (er [i] .area      <  min_area) { ++ nsmall ; goto remove ; }

      /*
       * Remove duplicates
       */
      if (min_div < 1.0) {
        vl_uint   parent = er [i] .parent ;
        int       area, p_area ;
        float div ;

        /* check all but the root mser */
        if((int) parent != i) {

          /* search for the maximally stable parent region */
          while(! er [parent] .max_stable) {
            vl_uint next = er [parent] .parent ;
            if(next == parent) break ;
            parent = next ;
          }

          /* Compare with the parent region; if the current and parent
           * regions are too similar, keep only the parent. */
          area    = er [i]      .area ;
          p_area  = er [parent] .area ;
          div     = (float) (p_area - area) / (float) p_area ;

          if (div < min_div) { ++ ndup ; goto remove ; }
        } /* remove dups end */

      }
      continue ;
    remove :
      er [i] .max_stable = 0 ;
      -- nmer ;
    } /* check next region */

    f-> stats .num_abs_unstable = nbad ;
    f-> stats .num_too_big      = nbig ;
    f-> stats .num_too_small    = nsmall ;
    f-> stats .num_duplicates   = ndup ;
  }
  /* -----------------------------------------------------------------
   *                                                   Save the result
   * -------------------------------------------------------------- */

  /* make room */
  if (f-> rmer < nmer) {
    if (mer) vl_free (mer) ;
    f->mer  = mer = vl_malloc( sizeof(vl_uint) * nmer) ;
    f->rmer = nmer ;
  }

  /* save back */
  f-> nmer = nmer ;

  j = 0 ;
  for (i = 0 ; i < ner ; ++i) {
    if (er [i] .max_stable) mer [j++] = er [i] .index ;
  }
}

/** -------------------------------------------------------------------
 ** @brief Fit ellipsoids
 **
 ** @param f MSER filter.
 **
 ** @sa @ref mser-ell
 **/

VL_EXPORT
void
vl_mser_ell_fit (VlMserFilt* f)
{
  /* shortcuts */
  int                nel = f-> nel ;
  int                dof = f-> dof ;
  int              *dims = f-> dims ;
  int              ndims = f-> ndims ;
  int              *subs = f-> subs ;
  int             njoins = f-> njoins ;
  vl_uint         *joins = f-> joins ;
  VlMserReg           *r = f-> r ;
  vl_uint           *mer = f-> mer ;
  int               nmer = f-> nmer ;
  vl_mser_acc       *acc = f-> acc ;
  vl_mser_acc       *ell = f-> ell ;

  int d, index, i, j ;

  /* already fit ? */
  if (f->nell == f->nmer) return ;

  /* make room */
  if (f->rell < f->nmer) {
    if (f->ell) vl_free (f->ell) ;
    f->ell  = vl_malloc (sizeof(float) * f->nmer * f->dof) ;
    f->rell = f-> nmer ;
  }

  if (f->acc == 0) {
    f->acc = vl_malloc (sizeof(float) * f->nel) ;
  }

  acc = f-> acc ;
  ell = f-> ell ;

  /* -----------------------------------------------------------------
   *                                                 Integrate moments
   * -------------------------------------------------------------- */

  /* for each dof */
  for(d = 0 ; d < f->dof ; ++d) {

    /* start from the upper-left pixel (0,0,...,0) */
    memset (subs, 0, sizeof(int) * ndims) ;

    /* step 1: fill acc pretending that each region has only one pixel */
    if(d < ndims) {
      /* 1-order ................................................... */

      for(index = 0 ; index < nel ; ++ index) {
        acc [index] = subs [d] ;
        adv(ndims, dims, subs) ;
      }
    }
    else {
      /* 2-order ................................................... */

      /* map the dof d to a second order moment E[x_i x_j] */
      i = d - ndims ;
      j = 0 ;
      while(i > j) {
        i -= j + 1 ;
        j ++ ;
      }
      /* initialize acc with  x_i * x_j */
      for(index = 0 ; index < nel ; ++ index){
        acc [index] = subs [i] * subs [j] ;
        adv(ndims, dims, subs) ;
      }
    }

    /* step 2: integrate */
    for(i = 0 ; i < njoins ; ++i) {
      vl_uint index  = joins [i] ;
      vl_uint parent = r [index] .parent ;
      acc [parent] += acc [index] ;
    }

    /* step 3: save back to ellpises */
    for(i = 0 ; i < nmer ; ++i) {
      vl_uint idx = mer [i] ;
      ell [d + dof*i] = acc [idx] ;
    }

  }  /* next dof */

  /* -----------------------------------------------------------------
   *                                           Compute central moments
   * -------------------------------------------------------------- */

  for(index = 0 ; index < nmer ; ++index) {
    float  *pt  = ell + index * dof ;
    vl_uint    idx  = mer [index] ;
    float  area = r [idx] .area ;

    for(d = 0 ; d < dof ; ++d) {

      pt [d] /= area ;

      if(d >= ndims) {
        /* remove squared mean from moment to get variance */
        i = d - ndims ;
        j = 0 ;
        while(i > j) {
          i -= j + 1 ;
          j ++ ;
        }
        pt [d] -= pt [i] * pt [j] ;
      }

    }
  }

  /* save back */
  f-> nell = nmer ;
}
