/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * separator.c
 *
 * This file contains code for separator extraction
 *
 * Started 8/1/97
 * George
 *
 * $Id: separator.c,v 1.1 1998/11/27 17:59:30 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function takes a bisection and constructs a minimum weight vertex
* separator out of it. It uses the node-based separator refinement for it.
**************************************************************************/
void ConstructSeparator(CtrlType *ctrl, GraphType *graph, float ubfactor)
{
  int i, j, k, nvtxs, nbnd;
  idxtype *xadj, *where, *bndind;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  nbnd = graph->nbnd;
  bndind = graph->bndind;

  where = idxcopy(nvtxs, graph->where, idxwspacemalloc(ctrl, nvtxs));

  /* Put the nodes in the boundary into the separator */
  for (i=0; i<nbnd; i++) {
    j = bndind[i];
    if (xadj[j+1]-xadj[j] > 0)  /* Ignore islands */
      where[j] = 2;
  }

  GKfree((void**) &graph->rdata, LTERM);
  Allocate2WayNodePartitionMemory(ctrl, graph);
  idxcopy(nvtxs, where, graph->where);
  idxwspacefree(ctrl, nvtxs);

  ASSERT(IsSeparable(graph));

  Compute2WayNodePartitionParams(ctrl, graph);

  ASSERT(CheckNodePartitionParams(graph));

  FM_2WayNodeRefine(ctrl, graph, ubfactor, 8);

  ASSERT(IsSeparable(graph));
}



/*************************************************************************
* This function takes a bisection and constructs a minimum weight vertex
* separator out of it. It uses an unweighted minimum-cover algorithm
* followed by node-based separator refinement.
**************************************************************************/
void ConstructMinCoverSeparator0(CtrlType *ctrl, GraphType *graph, float ubfactor)
{
  int i, ii, j, jj, k, l, nvtxs, nbnd, bnvtxs[3], bnedges[2], csize;
  idxtype *xadj, *adjncy, *bxadj, *badjncy;
  idxtype *where, *bndind, *bndptr, *vmap, *ivmap, *cover;


  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;

  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;

  vmap = idxwspacemalloc(ctrl, nvtxs);
  ivmap = idxwspacemalloc(ctrl, nbnd);
  cover = idxwspacemalloc(ctrl, nbnd);

  if (nbnd > 0) {
    /* Go through the boundary and determine the sizes of the bipartite graph */
    bnvtxs[0] = bnvtxs[1] = bnedges[0] = bnedges[1] = 0;
    for (i=0; i<nbnd; i++) {
      j = bndind[i];
      k = where[j];
      if (xadj[j+1]-xadj[j] > 0) {
        bnvtxs[k]++;
        bnedges[k] += xadj[j+1]-xadj[j];
      }
    }

    bnvtxs[2] = bnvtxs[0]+bnvtxs[1];
    bnvtxs[1] = bnvtxs[0];
    bnvtxs[0] = 0;

    bxadj = idxmalloc(bnvtxs[2]+1, "ConstructMinCoverSeparator: bxadj");
    badjncy = idxmalloc(bnedges[0]+bnedges[1]+1, "ConstructMinCoverSeparator: badjncy");

    /* Construct the ivmap and vmap */
    ASSERT(idxset(nvtxs, -1, vmap) == vmap);
    for (i=0; i<nbnd; i++) {
      j = bndind[i];
      k = where[j];
      if (xadj[j+1]-xadj[j] > 0) {
        vmap[j] = bnvtxs[k];
        ivmap[bnvtxs[k]++] = j;
      }
    }

    /* OK, go through and put the vertices of each part starting from 0 */
    bnvtxs[1] = bnvtxs[0];
    bnvtxs[0] = 0;
    bxadj[0] = l = 0;
    for (k=0; k<2; k++) {
      for (ii=0; ii<nbnd; ii++) {
        i = bndind[ii];
        if (where[i] == k && xadj[i] < xadj[i+1]) {
          for (j=xadj[i]; j<xadj[i+1]; j++) {
            jj = adjncy[j];
            if (where[jj] != k) {
              ASSERT(bndptr[jj] != -1);
              ASSERTP(vmap[jj] != -1, ("%d %d %d\n", jj, vmap[jj], graph->bndptr[jj]));
              badjncy[l++] = vmap[jj];
            }
          }
          bxadj[++bnvtxs[k]] = l;
        }
      }
    }

    ASSERT(l <= bnedges[0]+bnedges[1]);

    MinCover(bxadj, badjncy, bnvtxs[0], bnvtxs[1], cover, &csize);

    IFSET(ctrl->dbglvl, DBG_SEPINFO,
      printf("Nvtxs: %6d, [%5d %5d], Cut: %6d, SS: [%6d %6d], Cover: %6d\n", nvtxs, graph->pwgts[0], graph->pwgts[1], graph->mincut, bnvtxs[0], bnvtxs[1]-bnvtxs[0], csize));

    for (i=0; i<csize; i++) {
      j = ivmap[cover[i]];
      where[j] = 2;
    }

    GKfree((void**) &bxadj, (void**) &badjncy, LTERM);

    for (i=0; i<nbnd; i++)
      bndptr[bndind[i]] = -1;
    for (nbnd=i=0; i<nvtxs; i++) {
      if (where[i] == 2) {
        bndind[nbnd] = i;
        bndptr[i] = nbnd++;
      }
    }
  }
  else {
    IFSET(ctrl->dbglvl, DBG_SEPINFO,
      printf("Nvtxs: %6d, [%5d %5d], Cut: %6d, SS: [%6d %6d], Cover: %6d\n", nvtxs, graph->pwgts[0], graph->pwgts[1], graph->mincut, 0, 0, 0));
  }

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, graph->nbnd);
  idxwspacefree(ctrl, graph->nbnd);
  graph->nbnd = nbnd;


  ASSERT(IsSeparable(graph));
}



/*************************************************************************
* This function takes a bisection and constructs a minimum weight vertex
* separator out of it. It uses an unweighted minimum-cover algorithm
* followed by node-based separator refinement.
**************************************************************************/
void ConstructMinCoverSeparator(CtrlType *ctrl, GraphType *graph, float ubfactor)
{
  int i, ii, j, jj, k, l, nvtxs, nbnd, bnvtxs[3], bnedges[2], csize;
  idxtype *xadj, *adjncy, *bxadj, *badjncy;
  idxtype *where, *bndind, *bndptr, *vmap, *ivmap, *cover;


  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;

  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;

  vmap = idxwspacemalloc(ctrl, nvtxs);
  ivmap = idxwspacemalloc(ctrl, nbnd);
  cover = idxwspacemalloc(ctrl, nbnd);

  if (nbnd > 0) {
    /* Go through the boundary and determine the sizes of the bipartite graph */
    bnvtxs[0] = bnvtxs[1] = bnedges[0] = bnedges[1] = 0;
    for (i=0; i<nbnd; i++) {
      j = bndind[i];
      k = where[j];
      if (xadj[j+1]-xadj[j] > 0) {
        bnvtxs[k]++;
        bnedges[k] += xadj[j+1]-xadj[j];
      }
    }

    bnvtxs[2] = bnvtxs[0]+bnvtxs[1];
    bnvtxs[1] = bnvtxs[0];
    bnvtxs[0] = 0;

    bxadj = idxmalloc(bnvtxs[2]+1, "ConstructMinCoverSeparator: bxadj");
    badjncy = idxmalloc(bnedges[0]+bnedges[1]+1, "ConstructMinCoverSeparator: badjncy");

    /* Construct the ivmap and vmap */
    ASSERT(idxset(nvtxs, -1, vmap) == vmap);
    for (i=0; i<nbnd; i++) {
      j = bndind[i];
      k = where[j];
      if (xadj[j+1]-xadj[j] > 0) {
        vmap[j] = bnvtxs[k];
        ivmap[bnvtxs[k]++] = j;
      }
    }

    /* OK, go through and put the vertices of each part starting from 0 */
    bnvtxs[1] = bnvtxs[0];
    bnvtxs[0] = 0;
    bxadj[0] = l = 0;
    for (k=0; k<2; k++) {
      for (ii=0; ii<nbnd; ii++) {
        i = bndind[ii];
        if (where[i] == k && xadj[i] < xadj[i+1]) {
          for (j=xadj[i]; j<xadj[i+1]; j++) {
            jj = adjncy[j];
            if (where[jj] != k) {
              ASSERT(bndptr[jj] != -1);
              ASSERTP(vmap[jj] != -1, ("%d %d %d\n", jj, vmap[jj], graph->bndptr[jj]));
              badjncy[l++] = vmap[jj];
            }
          }
          bxadj[++bnvtxs[k]] = l;
        }
      }
    }

    ASSERT(l <= bnedges[0]+bnedges[1]);

    MinCover(bxadj, badjncy, bnvtxs[0], bnvtxs[1], cover, &csize);

    IFSET(ctrl->dbglvl, DBG_SEPINFO,
      printf("Nvtxs: %6d, [%5d %5d], Cut: %6d, SS: [%6d %6d], Cover: %6d\n", nvtxs, graph->pwgts[0], graph->pwgts[1], graph->mincut, bnvtxs[0], bnvtxs[1]-bnvtxs[0], csize));

    for (i=0; i<csize; i++) {
      j = ivmap[cover[i]];
      where[j] = 2;
    }

    GKfree((void**) &bxadj, (void**) &badjncy, LTERM);
  }
  else {
    IFSET(ctrl->dbglvl, DBG_SEPINFO,
      printf("Nvtxs: %6d, [%5d %5d], Cut: %6d, SS: [%6d %6d], Cover: %6d\n", nvtxs, graph->pwgts[0], graph->pwgts[1], graph->mincut, 0, 0, 0));
  }

  /* Prepare to refine the vertex separator */
  idxcopy(nvtxs, graph->where, vmap);
  GKfree((void**) &graph->rdata, LTERM);

  Allocate2WayNodePartitionMemory(ctrl, graph);
  idxcopy(nvtxs, vmap, graph->where);
  idxwspacefree(ctrl, nvtxs+2*graph->nbnd);

  Compute2WayNodePartitionParams(ctrl, graph);

  ASSERT(CheckNodePartitionParams(graph));

  FM_2WayNodeRefine_OneSided(ctrl, graph, ubfactor, 6);

  ASSERT(IsSeparable(graph));
}

