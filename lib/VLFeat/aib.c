/** @file aib.c
 ** @brief AIB - Definition
 ** @author Brian Fulkerson
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
@page aib Agglomerative Information Bottleneck (AIB)
@author Brian Fulkerson
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

@ref aib.h implemens the Agglomerative Information Bottleneck (AIB)
algorithm as first described in @cite{slonim99agglomerative}.

AIB takes a discrete valued feature @f$x@f$ and a label @f$c@f$ and
gradually compresses @f$x@f$ by iteratively merging values which
minimize the loss in mutual information @f$I(x,c)@f$.

While the algorithm is equivalent to the one described in
@cite{slonim99agglomerative}, it has some speedups that enable
handling much larger datasets. Let <em>N</em> be the number of feature
values and <em>C</em> the number of labels. The algorithm of
@cite{slonim99agglomerative} is @f$O(N^2)@f$ in space and @f$O(C
N^3)@f$ in time. This algorithm is @f$O(N)@f$ space and @f$O(C N^2)@f$
time in common cases (@f$O(C N^3)@f$ in the worst case).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section aib-overview Overview
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Given a discrete feature @f$x \in \mathcal{X} = \{x_1,\dots,x_N\}@f$
and a category label @f$c = 1,\dots,C@f$ with joint probability
@f$p(x,c)@f$, AIB computes a compressed feature @f$[x]_{ij}@f$ by
merging two values @f$x_i@f$ and @f$x_j@f$.  Among all the pairs
@f$ij@f$, AIB chooses the one that yields the smallest loss in the
mutual information

@f[
   D_{ij} = I(x,c) - I([x]_{ij},c) =
   \sum_c p(x_i) \log \frac{p(x_i,c)}{p(x_i)p(c)}   +
   \sum_c p(x_i) \log \frac{p(x_i,c)}{p(x_i)p(c)}   -
   \sum_c (p(x_i)+p(x_j)) \log \frac {p(x_i,c)+p(x_i,c)}{(p(x_i)+p(x_j))p(c)}
@f]

AIB iterates this procedure until the desired level of
compression is achieved.

@section aib-algorithm Algorithm details

Computing @f$D_{ij}@f$ requires @f$O(C)@f$ operations. For example, in
standard AIB we need to calculate

@f[
   D_{ij} = I(x,c) - I([x]_{ij},c) =
   \sum_c p(x_i) \log \frac{p(x_i,c)}{p(x_i)p(c)}   +
   \sum_c p(x_i) \log \frac{p(x_i,c)}{p(x_i)p(c)}   -
   \sum_c (p(x_i)+p(x_j)) \log \frac {p(x_i,c)+p(x_i,c)}{(p(x_i)+p(x_j))p(c)}
@f]

Thus in a basic implementation of AIB, finding the optimal pair
@f$ij@f$ of feature values requires @f$O(CN^2)@f$ operations in
total. In order to join all the @f$N@f$ values, we repeat this
procedure @f$O(N)@f$ times, yielding @f$O(N^3 C)@f$ time and
@f$O(1)@f$ space complexity (this does not account for the space need
to store the input).

The complexity can be improved by reusing computations. For instance,
we can store the matrix @f$D = [ D_{ij} ]@f$ (which requires
@f$O(N^2)@f$ space). Then, after joining @f$ij@f$, all of the matrix
<em>D</em> except the rows and columns (the matrix is symmetric) of
indexes <em>i</em> and <em>j</em> is unchanged. These two rows and
columns are deleted and a new row and column, whose computation
requires @f$O(NC)@f$ operations, are added for the merged value
@f$x_{ij}@f$.  Finding the minimal element of the matrix still
requires @f$O(N^2)@f$ operations, so the complexity of this algorithm
is @f$O(N^2C + N^3)@f$ time and @f$O(N^2)@f$ space.

We can obtain a much better expected complexity as follows. First,
instead of storing the whole matrix <em>D</em>, we store the smallest
element (index and value) of each row as @f$(q_i, D_i)@f$ (notice that
this is also the best element of each column since <em>D</em> is
symmetric). This requires @f$O(N)@f$ space and finding the minimal
element of the matrix requires @f$O(N)@f$ operations.  After joining
@f$ij@f$, we have to efficiently update this representation. This is
done as follows:

- The entries @f$(q_i,D_i)@f$ and @f$(q_j,D_j)@f$ are deleted.
- A new entry @f$(q_{ij},D_{ij})@f$ for the joint value @f$x_{ij}@f$
  is added. This requires @f$O(CN)@f$ operations.
- We test which other entries @f$(q_{k},D_{k})@f$ need to
  be updated. Recall that @f$(q_{k},D_{k})@f$ means that, before the
  merge, the value
  closest to @f$x_k@f$ was @f$x_{q_k}@f$ at a distance @f$D_k@f$. Then
  - If @f$q_k \not = i@f$, @f$q_k \not = j@f$ and @f$D_{k,ij} \geq D_k@f$, then
    @f$q_k@f$ is still the closest element and we do not do anything.
  - If @f$q_k \not = i@f$, @f$q_k \not = j@f$ and @f$D_{k,ij} <
    D_k@f$, then the closest element is @f$ij@f$ and we update the
    entry in constant time.
  - If @f$q_k = i@f$ or @f$q_k = j@f$, then we need to re-compute
    the closest element in @f$O(CN)@f$ operations.

This algorithm requires only @f$O(N)@f$ space and @f$O(\gamma(N) C
N^2)@f$ time, where @f$\gamma(N)@f$ is the expected number of times we
fall in the last case. In common cases one has @f$\gamma(N) \approx
\mathrm{const.}@f$, so the time saving is significant.

**/

#include "aib.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

/* The maximum value which beta may take */
#define BETA_MAX DBL_MAX

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Normalizes an array of probabilities to sum to 1
 **
 ** @param P        The array of probabilities
 ** @param nelem    The number of elements in the array
 **
 ** @return Modifies P to contain values which sum to 1
 **/

void vl_aib_normalize_P (double * P, vl_uint nelem)
{
    vl_uint i;
    double sum = 0;
    for(i=0; i<nelem; i++)
        sum += P[i];
    for(i=0; i<nelem; i++)
        P[i] /= sum;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Allocates and creates a list of nodes
 **
 ** @param nentries   The size of the list which will be created
 **
 ** @return an array containing elements 0...nentries
 **/

vl_uint *vl_aib_new_nodelist (vl_uint nentries)
{
    vl_uint * nodelist = vl_malloc(sizeof(vl_uint)*nentries);
    vl_uint n;
    for(n=0; n<nentries; n++)
        nodelist[n] = n;

    return nodelist;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Allocates and creates the marginal distribution Px
 **
 ** @param Pcx   A two-dimensional array of probabilities
 ** @param nvalues The number of rows in Pcx
 ** @param nlabels The number of columns in Pcx
 **
 ** @return an array of size @a nvalues which contains the marginal
 **         distribution over the rows.
 **/

double * vl_aib_new_Px(double * Pcx, vl_uint nvalues, vl_uint nlabels)
{
    double * Px = vl_malloc(sizeof(double)*nvalues);
    vl_uint r,c;
    for(r=0; r<nvalues; r++)
    {
        double sum = 0;
        for(c=0; c<nlabels; c++)
            sum += Pcx[r*nlabels+c];
        Px[r] = sum;
    }
    return Px;
}

/** ------------------------------------------------------------------
 ** @internal @brief Allocates and creates the marginal distribution Pc
 **
 ** @param Pcx      A two-dimensional array of probabilities
 ** @param nvalues    The number of rows in Pcx
 ** @param nlabels    The number of columns in Pcx
 **
 ** @return an array of size @a nlabels which contains the marginal distribution
 **         over the columns
 **/

double * vl_aib_new_Pc(double * Pcx, vl_uint nvalues, vl_uint nlabels)
{
    double * Pc = vl_malloc(sizeof(double)*nlabels);
    vl_uint r, c;
    for(c=0; c<nlabels; c++)
    {
        double sum = 0;
        for(r=0; r<nvalues; r++)
            sum += Pcx[r*nlabels+c];
        Pc[c] = sum;
    }
    return Pc;
}

/** ------------------------------------------------------------------
 ** @internal @brief Find the two nodes which have minimum beta.
 **
 ** @param aib      A pointer to the internal data structure
 ** @param besti    The index of one member of the pair which has mininum beta
 ** @param bestj    The index of the other member of the pair which
 **                 minimizes beta
 ** @param minbeta  The minimum beta value corresponding to (@a i, @a j)
 **
 ** Searches @a aib->beta to find the minimum value and fills @a minbeta and
 ** @a besti and @a bestj with this information.
 **/

void vl_aib_min_beta
(VlAIB * aib, vl_uint * besti, vl_uint * bestj, double * minbeta)
{
    vl_uint i;
    *minbeta = aib->beta[0];
    *besti   = 0;
    *bestj   = aib->bidx[0];

    for(i=0; i<aib->nentries; i++)
    {
        if(aib->beta[i] < *minbeta)
        {
            *minbeta = aib->beta[i];
            *besti = i;
            *bestj = aib->bidx[i];
        }
    }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Merges two nodes i,j in the internal datastructure
 **
 ** @param aib  A pointer to the internal data structure
 ** @param i    The index of one member of the pair to merge
 ** @param j    The index of the other member of the pair to merge
 ** @param new  The index of the new node which corresponds to the union of
 **             (@a i, @a j).
 **
 ** Nodes are merged by replacing the entry @a i with the union of @c
 ** ij, moving the node stored in last position (called @c lastnode)
 ** back to jth position and the entry at the end.
 **
 ** After the nodes have been merged, it updates which nodes should be
 ** considered on the next iteration based on which beta values could
 ** potentially change. The merged node will always be part of this
 ** list.
 **/

void
vl_aib_merge_nodes (VlAIB * aib, vl_uint i, vl_uint j, vl_uint new)
{
  vl_uint last_entry = aib->nentries - 1 ;
  vl_uint c, n ;

  /* clear the list of nodes to update */
  aib->nwhich = 0;

  /* make sure that i is smaller than j */
  if(i > j) { vl_uint tmp = j; j = i; i = tmp; }

  /* -----------------------------------------------------------------
   *                    Merge entries i and j, storing the result in i
   * -------------------------------------------------------------- */

  aib-> Px   [i] += aib->Px[j] ;
  aib-> beta [i]  = BETA_MAX ;
  aib-> nodes[i]  = new ;

  for (c = 0; c < aib->nlabels; c++)
    aib-> Pcx [i*aib->nlabels + c] += aib-> Pcx [j*aib->nlabels + c] ;

  /* -----------------------------------------------------------------
   *                                              Move last entry to j
   * -------------------------------------------------------------- */

  aib-> Px    [j]  = aib-> Px    [last_entry];
  aib-> beta  [j]  = aib-> beta  [last_entry];
  aib-> bidx  [j]  = aib-> bidx  [last_entry];
  aib-> nodes [j]  = aib-> nodes [last_entry];

  for (c = 0 ;  c < aib->nlabels ; c++)
    aib-> Pcx[j*aib->nlabels + c] = aib-> Pcx [last_entry*aib->nlabels + c] ;

  /* delete last entry */
  aib-> nentries -- ;

  /* -----------------------------------------------------------------
   *                                        Scan for entries to update
   * -------------------------------------------------------------- */

  /*
   * After mergin entries i and j, we need to update all other entries
   * that had one of these two as closest match. We also need to
   * update the renewend entry i. This is added by the loop below
   * since bidx [i] = j exactly because i was merged.
   *
   * Additionaly, since we moved the last entry back to the entry j,
   * we need to adjust the valeus of bidx to reflect this.
   */

  for (n = 0 ; n < aib->nentries; n++) {
    if(aib->bidx[n] == i || aib->bidx[n] == j) {
        aib->bidx  [n] = 0;
        aib->beta  [n] = BETA_MAX;
        aib->which [aib->nwhich++] = n ;
      }
    else if(aib->bidx[n] == last_entry) {
      aib->bidx[n] = j ;
    }
  }
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Updates @c aib->beta and @c aib->bidx according to @c aib->which
 **
 ** @param aib AIB data structure.
 **
 ** The function calculates @c beta[i] and @c bidx[i] for the nodes @c
 ** i listed in @c aib->which.  @c beta[i] is the minimal variation of mutual
 ** information (or other score) caused by merging entry @c i with another entry
 ** and @c bidx[i] is the index of this best matching entry.
 **
 ** Notice that for each entry @c i that we need to update, a full
 ** scan of all the other entries must be performed.
 **/

void
vl_aib_update_beta (VlAIB * aib)
{

#define PLOGP(x) ((x)*log((x)))

  vl_uint i;
  double * Px  = aib->Px;
  double * Pcx = aib->Pcx;
  double * tmp = vl_malloc(sizeof(double)*aib->nentries);
  vl_uint a, b, c ;

  /*
   * T1 = I(x,c) - I([x]_ij) = A + B - C
   *
   * A  = \sum_c p(xa,c)           \log ( p(xa,c)          /  p(xa)       )
   * B  = \sum_c p(xb,c)           \log ( p(xb,c)          /  p(xb)       )
   * C  = \sum_c (p(xa,c)+p(xb,c)) \log ((p(xa,c)+p(xb,c)) / (p(xa)+p(xb)))
   *
   * C  = C1 + C2
   * C1 = \sum_c (p(xa,c)+p(xb,c)) \log (p(xa,c)+p(xb,c))
   * C2 = - (p(xa)+p(xb) \log (p(xa)+p(xb))
   */

  /* precalculate A and B */
  for (a = 0; a < aib->nentries; a++) {
    tmp[a] = 0;
    for (c = 0; c < aib->nlabels; c++) {
        double Pac = Pcx [a*aib->nlabels + c] ;
        if(Pac != 0) tmp[a] += Pac * log (Pac / Px[a]) ;
    }
  }

  /* for each entry listed in which */
  for (i = 0 ; i < aib->nwhich; i++) {
    a = aib->which[i];

    /* for each other entry */
    for(b = 0 ; b < aib->nentries ; b++) {
      double T1 = 0 ;

      if (a == b || Px [a] == 0 || Px [b] == 0) continue ;


      T1 = PLOGP ((Px[a] + Px[b])) ;                  /* - C2 */
      T1 += tmp[a] + tmp[b] ;                         /* + A + B */

      for (c = 0 ; c < aib->nlabels; ++ c) {
        double Pac = Pcx [a*aib->nlabels + c] ;
        double Pbc = Pcx [b*aib->nlabels + c] ;
        if (Pac == 0 && Pbc == 0) continue;
        T1 += - PLOGP ((Pac + Pbc)) ;                 /* - C1 */
      }

      /*
       * Now we have beta(a,b). We check wether this is the best beta
       * for entries a and b.
       */
      {
        double beta = T1 ;

        if (beta < aib->beta[a])
          {
            aib->beta[a] = beta;
            aib->bidx[a] = b;
          }
        if (beta < aib->beta[b])
          {
            aib->beta[b] = beta;
            aib->bidx[b] = a;
          }
      }
    }
  }
  vl_free(tmp);
}

/** ------------------------------------------------------------------
 ** @internal @brief Calculates the current information and entropy
 **
 ** @param aib      A pointer to the internal data structure
 ** @param I        The current mutual information (out).
 ** @param H        The current entropy (out).
 **
 ** Calculates the current mutual information and entropy of Pcx and sets
 ** @a I and @a H to these new values.
 **/
void vl_aib_calculate_information(VlAIB * aib, double * I, double * H)
{
    vl_uint r, c;
    *H = 0;
    *I = 0;

    /*
     * H(x)   = - sum_x p(x)    \ log p(x)
     * I(x,c) =   sum_xc p(x,c) \ log (p(x,c) / p(x)p(c))
     */

    /* for each entry */
    for(r = 0 ; r< aib->nentries ; r++) {

      if (aib->Px[r] == 0) continue ;
      *H += -log(aib->Px[r]) * aib->Px[r] ;

      for(c=0; c<aib->nlabels; c++) {
        if (aib->Pcx[r*aib->nlabels+c] == 0) continue;
        *I += aib->Pcx[r*aib->nlabels+c] *
          log (aib->Pcx[r*aib->nlabels+c] / (aib->Px[r]*aib->Pc[c])) ;
      }
    }
}

/** ------------------------------------------------------------------
 ** @brief Allocates and initializes the internal data structure
 **
 ** @param Pcx      A pointer to a 2D array of probabilities
 ** @param nvalues    The number of rows in the array
 ** @param nlabels    The number of columns in the array
 **
 ** Creates a new @a VlAIB struct containing pointers to all the data that
 ** will be used during the AIB process.
 **
 ** Allocates memory for the following:
 ** - Px (nvalues*sizeof(double))
 ** - Pc (nlabels*sizeof(double))
 ** - nodelist (nvalues*sizeof(vl_uint))
 ** - which (nvalues*sizeof(vl_uint))
 ** - beta (nvalues*sizeof(double))
 ** - bidx (nvalues*sizeof(vl_uint))
 ** - parents ((2*nvalues-1)*sizeof(vl_uint))
 ** - costs (nvalues*sizeof(double))
 **
 ** Since it simply copies to pointer to Pcx, the total additional memory
 ** requirement is:
 **
 ** (3*nvalues+nlabels)*sizeof(double) + 4*nvalues*sizeof(vl_uint)
 **
 ** @returns An allocated and initialized @a VlAIB pointer
 **/
VlAIB * vl_aib_new(double * Pcx, vl_uint nvalues, vl_uint nlabels)
{
    VlAIB * aib = vl_malloc(sizeof(VlAIB));
    vl_uint i ;

    aib->verbosity = 0 ;
    aib->Pcx   = Pcx ;
    aib->nvalues = nvalues ;
    aib->nlabels = nlabels ;

    vl_aib_normalize_P (aib->Pcx, aib->nvalues * aib->nlabels) ;

    aib->Px = vl_aib_new_Px (aib->Pcx, aib->nvalues, aib->nlabels) ;
    aib->Pc = vl_aib_new_Pc (aib->Pcx, aib->nvalues, aib->nlabels) ;

    aib->nentries = aib->nvalues ;
    aib->nodes    = vl_aib_new_nodelist(aib->nentries) ;
    aib->beta     = vl_malloc(sizeof(double) * aib->nentries) ;
    aib->bidx     = vl_malloc(sizeof(vl_uint)   * aib->nentries) ;

    for(i = 0 ; i < aib->nentries ; i++)
      aib->beta [i] = BETA_MAX ;

    /* Initially we must consider all nodes */
    aib->nwhich = aib->nvalues;
    aib->which  = vl_aib_new_nodelist (aib->nwhich) ;

    aib->parents = vl_malloc(sizeof(vl_uint)*(aib->nvalues*2-1));
    /* Initially, all parents point to a nonexistent node */
    for (i = 0 ; i < 2 * aib->nvalues - 1 ; i++)
      aib->parents [i] = 2 * aib->nvalues ;

    /* Allocate cost output vector */
    aib->costs = vl_malloc (sizeof(double) * (aib->nvalues - 1 + 1)) ;


    return aib ;
}

/** ------------------------------------------------------------------
 ** @brief Deletes AIB data structure
 ** @param aib data structure to delete.
 **/

void
vl_aib_delete (VlAIB * aib)
{
  if (aib) {
    if (aib-> nodes)   vl_free (aib-> nodes);
    if (aib-> beta)    vl_free (aib-> beta);
    if (aib-> bidx)    vl_free (aib-> bidx);
    if (aib-> which)   vl_free (aib-> which);
    if (aib-> Px)      vl_free (aib-> Px);
    if (aib-> Pc)      vl_free (aib-> Pc);
    if (aib-> parents) vl_free (aib-> parents);
    if (aib-> costs)   vl_free (aib-> costs);

    vl_free (aib) ;
  }
}

/** ------------------------------------------------------------------
 ** @brief Runs AIB on Pcx
 **
 ** @param aib     AIB object to process
 **
 ** The function runs Agglomerative Information Bottleneck (AIB) on
 ** the joint probability table @a aib->Pcx which has labels along the
 ** columns and feature values along the rows. AIB iteratively merges
 ** the two values of the feature @c x that causes the smallest
 ** decrease in mutual information between the random variables @c x
 ** and @c c.
 **
 ** Merge operations are arranged in a binary tree. The nodes of the
 ** tree correspond to the original feature values and any other value
 ** obtained as a result of a merge operation. The nodes are indexed
 ** in breadth-first order, starting from the leaves. The first index
 ** is zero. In this way, the leaves correspond directly to the
 ** original feature values.  In total there are @c 2*nvalues-1 nodes.
 **
 ** The results may be accessed through vl_aib_get_parents which
 ** returns an array with one element per tree node. Each
 ** element is the index the parent node. The root parent is equal to
 ** zero. The array has @c 2*nvalues-1 elements.
 **
 ** Feature values with null probability are ignored by the algorithm
 ** and their nodes have parents indexing a non-existent tree node (a
 ** value bigger than @c 2*nvalues-1).
 **
 ** Then the function will also compute the information level after each
 ** merge. vl_get_costs will return a vector with the information level
 ** after each merge. @a
 ** cost has @c nvalues entries: The first is the value of the cost
 ** functional before any merge, and the others are the cost after the
 ** @c nvalues-1 merges.
 **
 **/

VL_EXPORT
void vl_aib_process(VlAIB *aib)
{
    vl_uint i, besti, bestj, newnode, nodei, nodej;
    double I, H;
    double minbeta;

    /* Calculate initial value of cost function */
    vl_aib_calculate_information (aib, &I, &H) ;
    aib->costs[0] = I;

    /* Initially which = all */

    /* For each merge */
    for(i = 0 ; i < aib->nvalues - 1 ; i++) {

      /* update entries in aib-> which */
      vl_aib_update_beta(aib);

      /* find best pair of nodes to merge */
      vl_aib_min_beta (aib, &besti, &bestj, &minbeta);

      if(minbeta == BETA_MAX)
        /* only null-probability entries remain */
        break;

      /* Add the parent pointers for the new node */
      newnode = aib->nvalues + i ;
      nodei = aib->nodes[besti];
      nodej = aib->nodes[bestj];

      aib->parents [nodei] = newnode ;
      aib->parents [nodej] = newnode ;
      aib->parents [newnode] = 0 ;

      /* Merge the nodes which produced the minimum beta */
      vl_aib_merge_nodes (aib, besti, bestj, newnode) ;
      vl_aib_calculate_information (aib, &I, &H) ;

      aib->costs[i+1] = I;

      if (aib->verbosity > 0) {
        VL_PRINTF ("aib: (%5d,%5d)=%5d dE: %10.3g I: %6.4g H: %6.4g updt: %5d\n",
                   nodei,
                   nodej,
                   newnode,
                   minbeta,
                   I,
                   H,
                   aib->nwhich) ;
      }
    }

    /* fill ignored entries with NaNs */
    for(; i < aib->nvalues - 1 ; i++)
        aib->costs[i+1] = VL_NAN_D ;
}
