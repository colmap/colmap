/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * sfm.c
 *
 * This file contains code that implementes an FM-based separator refinement
 *
 * Started 8/1/97
 * George
 *
 * $Id: sfm.c,v 1.1 1998/11/27 17:59:30 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function performs a node-based FM refinement
**************************************************************************/
void FM_2WayNodeRefine(CtrlType *ctrl, GraphType *graph, float ubfactor, int npasses)
{
  int i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
  idxtype *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
  idxtype *mptr, *mind, *moved, *swaps, *perm;
  PQueueType parts[2];
  NRInfoType *rinfo;
  int higain, oldgain, mincut, initcut, mincutorder;
  int pass, to, other, limit;
  int badmaxpwgt, mindiff, newdiff;
  int u[2], g[2];

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;
  pwgts = graph->pwgts;
  rinfo = graph->nrinfo;


  i = ComputeMaxNodeGain(nvtxs, xadj, adjncy, vwgt);
  PQueueInit(ctrl, &parts[0], nvtxs, i);
  PQueueInit(ctrl, &parts[1], nvtxs, i);

  moved = idxwspacemalloc(ctrl, nvtxs);
  swaps = idxwspacemalloc(ctrl, nvtxs);
  mptr = idxwspacemalloc(ctrl, nvtxs+1);
  mind = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("Partitions: [%6d %6d] Nv-Nb[%6d %6d]. ISep: %6d\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  badmaxpwgt = (int)(ubfactor*(pwgts[0]+pwgts[1]+pwgts[2])/2);

  for (pass=0; pass<npasses; pass++) {
    idxset(nvtxs, -1, moved);
    PQueueReset(&parts[0]);
    PQueueReset(&parts[1]);

    mincutorder = -1;
    initcut = mincut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      ASSERT(where[i] == 2);
      PQueueInsert(&parts[0], i, vwgt[i]-rinfo[i].edegrees[1]);
      PQueueInsert(&parts[1], i, vwgt[i]-rinfo[i].edegrees[0]);
    }

    ASSERT(CheckNodeBnd(graph, nbnd));
    ASSERT(CheckNodePartitionParams(graph));

    limit = (ctrl->oflags&OFLAG_COMPRESS ? amin(5*nbnd, 400) : amin(2*nbnd, 300));

    /******************************************************
    * Get into the FM loop
    *******************************************************/
    mptr[0] = nmind = 0;
    mindiff = abs(pwgts[0]-pwgts[1]);
    to = (pwgts[0] < pwgts[1] ? 0 : 1);
    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      u[0] = PQueueSeeMax(&parts[0]);
      u[1] = PQueueSeeMax(&parts[1]);
      if (u[0] != -1 && u[1] != -1) {
        g[0] = vwgt[u[0]]-rinfo[u[0]].edegrees[1];
        g[1] = vwgt[u[1]]-rinfo[u[1]].edegrees[0];

        to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : pass%2));
        /* to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : (pwgts[0] < pwgts[1] ? 0 : 1))); */

        if (pwgts[to]+vwgt[u[to]] > badmaxpwgt)
          to = (to+1)%2;
      }
      else if (u[0] == -1 && u[1] == -1) {
        break;
      }
      else if (u[0] != -1 && pwgts[0]+vwgt[u[0]] <= badmaxpwgt) {
        to = 0;
      }
      else if (u[1] != -1 && pwgts[1]+vwgt[u[1]] <= badmaxpwgt) {
        to = 1;
      }
      else
        break;

      other = (to+1)%2;

      higain = PQueueGetMax(&parts[to]);
      if (moved[higain] == -1) /* Delete if it was in the separator originally */
        PQueueDelete(&parts[other], higain, vwgt[higain]-rinfo[higain].edegrees[to]);

      ASSERT(bndptr[higain] != -1);

      pwgts[2] -= (vwgt[higain]-rinfo[higain].edegrees[other]);

      newdiff = abs(pwgts[to]+vwgt[higain] - (pwgts[other]-rinfo[higain].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      }
      else {
        if (nswaps - mincutorder > limit) {
          pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      BNDDelete(nbnd, bndind, bndptr, higain);
      pwgts[to] += vwgt[higain];
      where[higain] = to;
      moved[higain] = nswaps;
      swaps[nswaps] = higain;


      /**********************************************************
      * Update the degrees of the affected nodes
      ***********************************************************/
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2) { /* For the in-separator vertices modify their edegree[to] */
          oldgain = vwgt[k]-rinfo[k].edegrees[to];
          rinfo[k].edegrees[to] += vwgt[higain];
          if (moved[k] == -1 || moved[k] == -(2+other))
            PQueueUpdate(&parts[other], k, oldgain, oldgain-vwgt[higain]);
        }
        else if (where[k] == other) { /* This vertex is pulled into the separator */
          ASSERTP(bndptr[k] == -1, ("%d %d %d\n", k, bndptr[k], where[k]));
          BNDInsert(nbnd, bndind, bndptr, k);

          mind[nmind++] = k;  /* Keep track for rollback */
          where[k] = 2;
          pwgts[other] -= vwgt[k];

          edegrees = rinfo[k].edegrees;
          edegrees[0] = edegrees[1] = 0;
          for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
            kk = adjncy[jj];
            if (where[kk] != 2)
              edegrees[where[kk]] += vwgt[kk];
            else {
              oldgain = vwgt[kk]-rinfo[kk].edegrees[other];
              rinfo[kk].edegrees[other] -= vwgt[k];
              if (moved[kk] == -1 || moved[kk] == -(2+to))
                PQueueUpdate(&parts[to], kk, oldgain, oldgain+vwgt[k]);
            }
          }

          /* Insert the new vertex into the priority queue. Only one side! */
          if (moved[k] == -1) {
            PQueueInsert(&parts[to], k, vwgt[k]-edegrees[other]);
            moved[k] = -(2+to);
          }
        }
      }
      mptr[nswaps+1] = nmind;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO,
            printf("Moved %6d to %3d, Gain: %5d [%5d] [%4d %4d] \t[%5d %5d %5d]\n", higain, to, g[to], g[other], vwgt[u[to]], vwgt[u[other]], pwgts[0], pwgts[1], pwgts[2]));

    }


    /****************************************************************
    * Roll back computation
    *****************************************************************/
    for (nswaps--; nswaps>mincutorder; nswaps--) {
      higain = swaps[nswaps];

      ASSERT(CheckNodePartitionParams(graph));

      to = where[higain];
      other = (to+1)%2;
      INC_DEC(pwgts[2], pwgts[to], vwgt[higain]);
      where[higain] = 2;
      BNDInsert(nbnd, bndind, bndptr, higain);

      edegrees = rinfo[higain].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2)
          rinfo[k].edegrees[to] -= vwgt[higain];
        else
          edegrees[where[k]] += vwgt[k];
      }

      /* Push nodes out of the separator */
      for (j=mptr[nswaps]; j<mptr[nswaps+1]; j++) {
        k = mind[j];
        ASSERT(where[k] == 2);
        where[k] = other;
        INC_DEC(pwgts[other], pwgts[2], vwgt[k]);
        BNDDelete(nbnd, bndind, bndptr, k);
        for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
          kk = adjncy[jj];
          if (where[kk] == 2)
            rinfo[kk].edegrees[other] += vwgt[k];
        }
      }
    }

    ASSERT(mincut == pwgts[2]);

    IFSET(ctrl->dbglvl, DBG_REFINE,
      printf("\tMinimum sep: %6d at %5d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd));

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut >= initcut)
      break;
  }

  PQueueFree(ctrl, &parts[0]);
  PQueueFree(ctrl, &parts[1]);

  idxwspacefree(ctrl, nvtxs+1);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function performs a node-based FM refinement
**************************************************************************/
void FM_2WayNodeRefine2(CtrlType *ctrl, GraphType *graph, float ubfactor, int npasses)
{
  int i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
  idxtype *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
  idxtype *mptr, *mind, *moved, *swaps, *perm;
  PQueueType parts[2];
  NRInfoType *rinfo;
  int higain, oldgain, mincut, initcut, mincutorder;
  int pass, to, other, limit;
  int badmaxpwgt, mindiff, newdiff;
  int u[2], g[2];

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;
  pwgts = graph->pwgts;
  rinfo = graph->nrinfo;


  i = ComputeMaxNodeGain(nvtxs, xadj, adjncy, vwgt);
  PQueueInit(ctrl, &parts[0], nvtxs, i);
  PQueueInit(ctrl, &parts[1], nvtxs, i);

  moved = idxwspacemalloc(ctrl, nvtxs);
  swaps = idxwspacemalloc(ctrl, nvtxs);
  mptr = idxwspacemalloc(ctrl, nvtxs+1);
  mind = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("Partitions: [%6d %6d] Nv-Nb[%6d %6d]. ISep: %6d\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  badmaxpwgt = (int)(ubfactor*(pwgts[0]+pwgts[1]+pwgts[2])/2);

  for (pass=0; pass<npasses; pass++) {
    idxset(nvtxs, -1, moved);
    PQueueReset(&parts[0]);
    PQueueReset(&parts[1]);

    mincutorder = -1;
    initcut = mincut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      ASSERT(where[i] == 2);
      PQueueInsert(&parts[0], i, vwgt[i]-rinfo[i].edegrees[1]);
      PQueueInsert(&parts[1], i, vwgt[i]-rinfo[i].edegrees[0]);
    }

    ASSERT(CheckNodeBnd(graph, nbnd));
    ASSERT(CheckNodePartitionParams(graph));

    limit = (ctrl->oflags&OFLAG_COMPRESS ? amin(5*nbnd, 400) : amin(2*nbnd, 300));

    /******************************************************
    * Get into the FM loop
    *******************************************************/
    mptr[0] = nmind = 0;
    mindiff = abs(pwgts[0]-pwgts[1]);
    to = (pwgts[0] < pwgts[1] ? 0 : 1);
    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      badmaxpwgt = (int)(ubfactor*(pwgts[0]+pwgts[1]+pwgts[2]/2)/2);

      u[0] = PQueueSeeMax(&parts[0]);
      u[1] = PQueueSeeMax(&parts[1]);
      if (u[0] != -1 && u[1] != -1) {
        g[0] = vwgt[u[0]]-rinfo[u[0]].edegrees[1];
        g[1] = vwgt[u[1]]-rinfo[u[1]].edegrees[0];

        to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : pass%2));
        /* to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : (pwgts[0] < pwgts[1] ? 0 : 1))); */

        if (pwgts[to]+vwgt[u[to]] > badmaxpwgt)
          to = (to+1)%2;
      }
      else if (u[0] == -1 && u[1] == -1) {
        break;
      }
      else if (u[0] != -1 && pwgts[0]+vwgt[u[0]] <= badmaxpwgt) {
        to = 0;
      }
      else if (u[1] != -1 && pwgts[1]+vwgt[u[1]] <= badmaxpwgt) {
        to = 1;
      }
      else
        break;

      other = (to+1)%2;

      higain = PQueueGetMax(&parts[to]);
      if (moved[higain] == -1) /* Delete if it was in the separator originally */
        PQueueDelete(&parts[other], higain, vwgt[higain]-rinfo[higain].edegrees[to]);

      ASSERT(bndptr[higain] != -1);

      pwgts[2] -= (vwgt[higain]-rinfo[higain].edegrees[other]);

      newdiff = abs(pwgts[to]+vwgt[higain] - (pwgts[other]-rinfo[higain].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      }
      else {
        if (nswaps - mincutorder > limit) {
          pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      BNDDelete(nbnd, bndind, bndptr, higain);
      pwgts[to] += vwgt[higain];
      where[higain] = to;
      moved[higain] = nswaps;
      swaps[nswaps] = higain;


      /**********************************************************
      * Update the degrees of the affected nodes
      ***********************************************************/
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2) { /* For the in-separator vertices modify their edegree[to] */
          oldgain = vwgt[k]-rinfo[k].edegrees[to];
          rinfo[k].edegrees[to] += vwgt[higain];
          if (moved[k] == -1 || moved[k] == -(2+other))
            PQueueUpdate(&parts[other], k, oldgain, oldgain-vwgt[higain]);
        }
        else if (where[k] == other) { /* This vertex is pulled into the separator */
          ASSERTP(bndptr[k] == -1, ("%d %d %d\n", k, bndptr[k], where[k]));
          BNDInsert(nbnd, bndind, bndptr, k);

          mind[nmind++] = k;  /* Keep track for rollback */
          where[k] = 2;
          pwgts[other] -= vwgt[k];

          edegrees = rinfo[k].edegrees;
          edegrees[0] = edegrees[1] = 0;
          for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
            kk = adjncy[jj];
            if (where[kk] != 2)
              edegrees[where[kk]] += vwgt[kk];
            else {
              oldgain = vwgt[kk]-rinfo[kk].edegrees[other];
              rinfo[kk].edegrees[other] -= vwgt[k];
              if (moved[kk] == -1 || moved[kk] == -(2+to))
                PQueueUpdate(&parts[to], kk, oldgain, oldgain+vwgt[k]);
            }
          }

          /* Insert the new vertex into the priority queue. Only one side! */
          if (moved[k] == -1) {
            PQueueInsert(&parts[to], k, vwgt[k]-edegrees[other]);
            moved[k] = -(2+to);
          }
        }
      }
      mptr[nswaps+1] = nmind;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO,
            printf("Moved %6d to %3d, Gain: %5d [%5d] [%4d %4d] \t[%5d %5d %5d]\n", higain, to, g[to], g[other], vwgt[u[to]], vwgt[u[other]], pwgts[0], pwgts[1], pwgts[2]));

    }


    /****************************************************************
    * Roll back computation
    *****************************************************************/
    for (nswaps--; nswaps>mincutorder; nswaps--) {
      higain = swaps[nswaps];

      ASSERT(CheckNodePartitionParams(graph));

      to = where[higain];
      other = (to+1)%2;
      INC_DEC(pwgts[2], pwgts[to], vwgt[higain]);
      where[higain] = 2;
      BNDInsert(nbnd, bndind, bndptr, higain);

      edegrees = rinfo[higain].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2)
          rinfo[k].edegrees[to] -= vwgt[higain];
        else
          edegrees[where[k]] += vwgt[k];
      }

      /* Push nodes out of the separator */
      for (j=mptr[nswaps]; j<mptr[nswaps+1]; j++) {
        k = mind[j];
        ASSERT(where[k] == 2);
        where[k] = other;
        INC_DEC(pwgts[other], pwgts[2], vwgt[k]);
        BNDDelete(nbnd, bndind, bndptr, k);
        for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
          kk = adjncy[jj];
          if (where[kk] == 2)
            rinfo[kk].edegrees[other] += vwgt[k];
        }
      }
    }

    ASSERT(mincut == pwgts[2]);

    IFSET(ctrl->dbglvl, DBG_REFINE,
      printf("\tMinimum sep: %6d at %5d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd));

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut >= initcut)
      break;
  }

  PQueueFree(ctrl, &parts[0]);
  PQueueFree(ctrl, &parts[1]);

  idxwspacefree(ctrl, nvtxs+1);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function performs a node-based FM refinement
**************************************************************************/
void FM_2WayNodeRefineEqWgt(CtrlType *ctrl, GraphType *graph, int npasses)
{
  int i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
  idxtype *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
  idxtype *mptr, *mind, *moved, *swaps, *perm;
  PQueueType parts[2];
  NRInfoType *rinfo;
  int higain, oldgain, mincut, initcut, mincutorder;
  int pass, to, other, limit;
  int mindiff, newdiff;
  int u[2], g[2];

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;
  pwgts = graph->pwgts;
  rinfo = graph->nrinfo;


  i = ComputeMaxNodeGain(nvtxs, xadj, adjncy, vwgt);
  PQueueInit(ctrl, &parts[0], nvtxs, i);
  PQueueInit(ctrl, &parts[1], nvtxs, i);

  moved = idxwspacemalloc(ctrl, nvtxs);
  swaps = idxwspacemalloc(ctrl, nvtxs);
  mptr = idxwspacemalloc(ctrl, nvtxs+1);
  mind = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("Partitions: [%6d %6d] Nv-Nb[%6d %6d]. ISep: %6d\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  for (pass=0; pass<npasses; pass++) {
    idxset(nvtxs, -1, moved);
    PQueueReset(&parts[0]);
    PQueueReset(&parts[1]);

    mincutorder = -1;
    initcut = mincut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      ASSERT(where[i] == 2);
      PQueueInsert(&parts[0], i, vwgt[i]-rinfo[i].edegrees[1]);
      PQueueInsert(&parts[1], i, vwgt[i]-rinfo[i].edegrees[0]);
    }

    ASSERT(CheckNodeBnd(graph, nbnd));
    ASSERT(CheckNodePartitionParams(graph));

    limit = (ctrl->oflags&OFLAG_COMPRESS ? amin(5*nbnd, 400) : amin(2*nbnd, 300));

    /******************************************************
    * Get into the FM loop
    *******************************************************/
    mptr[0] = nmind = 0;
    mindiff = abs(pwgts[0]-pwgts[1]);
    to = (pwgts[0] < pwgts[1] ? 0 : 1);
    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      to = (pwgts[0] < pwgts[1] ? 0 : 1);

      if (pwgts[0] == pwgts[1]) {
        u[0] = PQueueSeeMax(&parts[0]);
        u[1] = PQueueSeeMax(&parts[1]);
        if (u[0] != -1 && u[1] != -1) {
          g[0] = vwgt[u[0]]-rinfo[u[0]].edegrees[1];
          g[1] = vwgt[u[1]]-rinfo[u[1]].edegrees[0];

          to = (g[0] > g[1] ? 0 : (g[0] < g[1] ? 1 : pass%2));
        }
      }
      other = (to+1)%2;

      if ((higain = PQueueGetMax(&parts[to])) == -1)
        break;

      if (moved[higain] == -1) /* Delete if it was in the separator originally */
        PQueueDelete(&parts[other], higain, vwgt[higain]-rinfo[higain].edegrees[to]);

      ASSERT(bndptr[higain] != -1);

      pwgts[2] -= (vwgt[higain]-rinfo[higain].edegrees[other]);

      newdiff = abs(pwgts[to]+vwgt[higain] - (pwgts[other]-rinfo[higain].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      }
      else {
        if (nswaps - mincutorder > limit) {
          pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      BNDDelete(nbnd, bndind, bndptr, higain);
      pwgts[to] += vwgt[higain];
      where[higain] = to;
      moved[higain] = nswaps;
      swaps[nswaps] = higain;


      /**********************************************************
      * Update the degrees of the affected nodes
      ***********************************************************/
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2) { /* For the in-separator vertices modify their edegree[to] */
          oldgain = vwgt[k]-rinfo[k].edegrees[to];
          rinfo[k].edegrees[to] += vwgt[higain];
          if (moved[k] == -1 || moved[k] == -(2+other))
            PQueueUpdate(&parts[other], k, oldgain, oldgain-vwgt[higain]);
        }
        else if (where[k] == other) { /* This vertex is pulled into the separator */
          ASSERTP(bndptr[k] == -1, ("%d %d %d\n", k, bndptr[k], where[k]));
          BNDInsert(nbnd, bndind, bndptr, k);

          mind[nmind++] = k;  /* Keep track for rollback */
          where[k] = 2;
          pwgts[other] -= vwgt[k];

          edegrees = rinfo[k].edegrees;
          edegrees[0] = edegrees[1] = 0;
          for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
            kk = adjncy[jj];
            if (where[kk] != 2)
              edegrees[where[kk]] += vwgt[kk];
            else {
              oldgain = vwgt[kk]-rinfo[kk].edegrees[other];
              rinfo[kk].edegrees[other] -= vwgt[k];
              if (moved[kk] == -1 || moved[kk] == -(2+to))
                PQueueUpdate(&parts[to], kk, oldgain, oldgain+vwgt[k]);
            }
          }

          /* Insert the new vertex into the priority queue. Only one side! */
          if (moved[k] == -1) {
            PQueueInsert(&parts[to], k, vwgt[k]-edegrees[other]);
            moved[k] = -(2+to);
          }
        }
      }
      mptr[nswaps+1] = nmind;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO,
            printf("Moved %6d to %3d, Gain: %5d [%5d] [%4d %4d] \t[%5d %5d %5d]\n", higain, to, g[to], g[other], vwgt[u[to]], vwgt[u[other]], pwgts[0], pwgts[1], pwgts[2]));

    }


    /****************************************************************
    * Roll back computation
    *****************************************************************/
    for (nswaps--; nswaps>mincutorder; nswaps--) {
      higain = swaps[nswaps];

      ASSERT(CheckNodePartitionParams(graph));

      to = where[higain];
      other = (to+1)%2;
      INC_DEC(pwgts[2], pwgts[to], vwgt[higain]);
      where[higain] = 2;
      BNDInsert(nbnd, bndind, bndptr, higain);

      edegrees = rinfo[higain].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2)
          rinfo[k].edegrees[to] -= vwgt[higain];
        else
          edegrees[where[k]] += vwgt[k];
      }

      /* Push nodes out of the separator */
      for (j=mptr[nswaps]; j<mptr[nswaps+1]; j++) {
        k = mind[j];
        ASSERT(where[k] == 2);
        where[k] = other;
        INC_DEC(pwgts[other], pwgts[2], vwgt[k]);
        BNDDelete(nbnd, bndind, bndptr, k);
        for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
          kk = adjncy[jj];
          if (where[kk] == 2)
            rinfo[kk].edegrees[other] += vwgt[k];
        }
      }
    }

    ASSERT(mincut == pwgts[2]);

    IFSET(ctrl->dbglvl, DBG_REFINE,
      printf("\tMinimum sep: %6d at %5d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd));

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut >= initcut)
      break;
  }

  PQueueFree(ctrl, &parts[0]);
  PQueueFree(ctrl, &parts[1]);

  idxwspacefree(ctrl, nvtxs+1);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function performs a node-based FM refinement. This is the
* one-way version
**************************************************************************/
void FM_2WayNodeRefine_OneSided(CtrlType *ctrl, GraphType *graph, float ubfactor, int npasses)
{
  int i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps, nmind;
  idxtype *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
  idxtype *mptr, *mind, *swaps, *perm;
  PQueueType parts;
  NRInfoType *rinfo;
  int higain, oldgain, mincut, initcut, mincutorder;
  int pass, to, other, limit;
  int badmaxpwgt, mindiff, newdiff;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;
  pwgts = graph->pwgts;
  rinfo = graph->nrinfo;

  PQueueInit(ctrl, &parts, nvtxs, ComputeMaxNodeGain(nvtxs, xadj, adjncy, vwgt));

  perm = idxwspacemalloc(ctrl, nvtxs);
  swaps = idxwspacemalloc(ctrl, nvtxs);
  mptr = idxwspacemalloc(ctrl, nvtxs);
  mind = idxwspacemalloc(ctrl, nvtxs+1);

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("Partitions-N1: [%6d %6d] Nv-Nb[%6d %6d]. ISep: %6d\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  badmaxpwgt = (int)(ubfactor*(pwgts[0]+pwgts[1]+pwgts[2])/2);

  to = (pwgts[0] < pwgts[1] ? 1 : 0);
  for (pass=0; pass<npasses; pass++) {
    other = to;
    to = (to+1)%2;

    PQueueReset(&parts);

    mincutorder = -1;
    initcut = mincut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      ASSERT(where[i] == 2);
      PQueueInsert(&parts, i, vwgt[i]-rinfo[i].edegrees[other]);
    }

    ASSERT(CheckNodeBnd(graph, nbnd));
    ASSERT(CheckNodePartitionParams(graph));

    limit = (ctrl->oflags&OFLAG_COMPRESS ? amin(5*nbnd, 400) : amin(2*nbnd, 300));

    /******************************************************
    * Get into the FM loop
    *******************************************************/
    mptr[0] = nmind = 0;
    mindiff = abs(pwgts[0]-pwgts[1]);
    for (nswaps=0; nswaps<nvtxs; nswaps++) {

      if ((higain = PQueueGetMax(&parts)) == -1)
        break;

      ASSERT(bndptr[higain] != -1);

      if (pwgts[to]+vwgt[higain] > badmaxpwgt)
        break;  /* No point going any further. Balance will be bad */

      pwgts[2] -= (vwgt[higain]-rinfo[higain].edegrees[other]);

      newdiff = abs(pwgts[to]+vwgt[higain] - (pwgts[other]-rinfo[higain].edegrees[other]));
      if (pwgts[2] < mincut || (pwgts[2] == mincut && newdiff < mindiff)) {
        mincut = pwgts[2];
        mincutorder = nswaps;
        mindiff = newdiff;
      }
      else {
        if (nswaps - mincutorder > limit) {
          pwgts[2] += (vwgt[higain]-rinfo[higain].edegrees[other]);
          break; /* No further improvement, break out */
        }
      }

      BNDDelete(nbnd, bndind, bndptr, higain);
      pwgts[to] += vwgt[higain];
      where[higain] = to;
      swaps[nswaps] = higain;


      /**********************************************************
      * Update the degrees of the affected nodes
      ***********************************************************/
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2) { /* For the in-separator vertices modify their edegree[to] */
          rinfo[k].edegrees[to] += vwgt[higain];
        }
        else if (where[k] == other) { /* This vertex is pulled into the separator */
          ASSERTP(bndptr[k] == -1, ("%d %d %d\n", k, bndptr[k], where[k]));
          BNDInsert(nbnd, bndind, bndptr, k);

          mind[nmind++] = k;  /* Keep track for rollback */
          where[k] = 2;
          pwgts[other] -= vwgt[k];

          edegrees = rinfo[k].edegrees;
          edegrees[0] = edegrees[1] = 0;
          for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
            kk = adjncy[jj];
            if (where[kk] != 2)
              edegrees[where[kk]] += vwgt[kk];
            else {
              oldgain = vwgt[kk]-rinfo[kk].edegrees[other];
              rinfo[kk].edegrees[other] -= vwgt[k];

              /* Since the moves are one-sided this vertex has not been moved yet */
              PQueueUpdateUp(&parts, kk, oldgain, oldgain+vwgt[k]);
            }
          }

          /* Insert the new vertex into the priority queue. Safe due to one-sided moves */
          PQueueInsert(&parts, k, vwgt[k]-edegrees[other]);
        }
      }
      mptr[nswaps+1] = nmind;


      IFSET(ctrl->dbglvl, DBG_MOVEINFO,
            printf("Moved %6d to %3d, Gain: %5d [%5d] \t[%5d %5d %5d] [%3d %2d]\n",
                       higain, to, (vwgt[higain]-rinfo[higain].edegrees[other]), vwgt[higain], pwgts[0], pwgts[1], pwgts[2], nswaps, limit));

    }


    /****************************************************************
    * Roll back computation
    *****************************************************************/
    for (nswaps--; nswaps>mincutorder; nswaps--) {
      higain = swaps[nswaps];

      ASSERT(CheckNodePartitionParams(graph));
      ASSERT(where[higain] == to);

      INC_DEC(pwgts[2], pwgts[to], vwgt[higain]);
      where[higain] = 2;
      BNDInsert(nbnd, bndind, bndptr, higain);

      edegrees = rinfo[higain].edegrees;
      edegrees[0] = edegrees[1] = 0;
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        if (where[k] == 2)
          rinfo[k].edegrees[to] -= vwgt[higain];
        else
          edegrees[where[k]] += vwgt[k];
      }

      /* Push nodes out of the separator */
      for (j=mptr[nswaps]; j<mptr[nswaps+1]; j++) {
        k = mind[j];
        ASSERT(where[k] == 2);
        where[k] = other;
        INC_DEC(pwgts[other], pwgts[2], vwgt[k]);
        BNDDelete(nbnd, bndind, bndptr, k);
        for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
          kk = adjncy[jj];
          if (where[kk] == 2)
            rinfo[kk].edegrees[other] += vwgt[k];
        }
      }
    }

    ASSERT(mincut == pwgts[2]);

    IFSET(ctrl->dbglvl, DBG_REFINE,
      printf("\tMinimum sep: %6d at %5d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd));

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (pass%2 == 1 && (mincutorder == -1 || mincut >= initcut))
      break;
  }

  PQueueFree(ctrl, &parts);

  idxwspacefree(ctrl, nvtxs+1);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function performs a node-based FM refinement
**************************************************************************/
void FM_2WayNodeBalance(CtrlType *ctrl, GraphType *graph, float ubfactor)
{
  int i, ii, j, k, jj, kk, nvtxs, nbnd, nswaps;
  idxtype *xadj, *vwgt, *adjncy, *where, *pwgts, *edegrees, *bndind, *bndptr;
  idxtype *perm, *moved;
  PQueueType parts;
  NRInfoType *rinfo;
  int higain, oldgain;
  int pass, to, other;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;
  where = graph->where;
  pwgts = graph->pwgts;
  rinfo = graph->nrinfo;

  if (abs(pwgts[0]-pwgts[1]) < (int)((ubfactor-1.0)*(pwgts[0]+pwgts[1])))
    return;
  if (abs(pwgts[0]-pwgts[1]) < 3*idxsum(nvtxs, vwgt)/nvtxs)
    return;

  to = (pwgts[0] < pwgts[1] ? 0 : 1);
  other = (to+1)%2;

  PQueueInit(ctrl, &parts, nvtxs, ComputeMaxNodeGain(nvtxs, xadj, adjncy, vwgt));

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxset(nvtxs, -1, idxwspacemalloc(ctrl, nvtxs));

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("Partitions: [%6d %6d] Nv-Nb[%6d %6d]. ISep: %6d [B]\n", pwgts[0], pwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  nbnd = graph->nbnd;
  RandomPermute(nbnd, perm, 1);
  for (ii=0; ii<nbnd; ii++) {
    i = bndind[perm[ii]];
    ASSERT(where[i] == 2);
    PQueueInsert(&parts, i, vwgt[i]-rinfo[i].edegrees[other]);
  }

  ASSERT(CheckNodeBnd(graph, nbnd));
  ASSERT(CheckNodePartitionParams(graph));

  /******************************************************
  * Get into the FM loop
  *******************************************************/
  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if ((higain = PQueueGetMax(&parts)) == -1)
      break;

    moved[higain] = 1;

    if (pwgts[other] - rinfo[higain].edegrees[other] < (pwgts[0]+pwgts[1])/2)
      continue;
#ifdef XXX
    if (pwgts[other] - rinfo[higain].edegrees[other] < pwgts[to]+vwgt[higain])
      break;
#endif

    ASSERT(bndptr[higain] != -1);

    pwgts[2] -= (vwgt[higain]-rinfo[higain].edegrees[other]);

    BNDDelete(nbnd, bndind, bndptr, higain);
    pwgts[to] += vwgt[higain];
    where[higain] = to;

    IFSET(ctrl->dbglvl, DBG_MOVEINFO,
          printf("Moved %6d to %3d, Gain: %3d, \t[%5d %5d %5d]\n", higain, to, vwgt[higain]-rinfo[higain].edegrees[other], pwgts[0], pwgts[1], pwgts[2]));


    /**********************************************************
    * Update the degrees of the affected nodes
    ***********************************************************/
    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];
      if (where[k] == 2) { /* For the in-separator vertices modify their edegree[to] */
        rinfo[k].edegrees[to] += vwgt[higain];
      }
      else if (where[k] == other) { /* This vertex is pulled into the separator */
        ASSERTP(bndptr[k] == -1, ("%d %d %d\n", k, bndptr[k], where[k]));
        BNDInsert(nbnd, bndind, bndptr, k);

        where[k] = 2;
        pwgts[other] -= vwgt[k];

        edegrees = rinfo[k].edegrees;
        edegrees[0] = edegrees[1] = 0;
        for (jj=xadj[k]; jj<xadj[k+1]; jj++) {
          kk = adjncy[jj];
          if (where[kk] != 2)
            edegrees[where[kk]] += vwgt[kk];
          else {
            ASSERT(bndptr[kk] != -1);
            oldgain = vwgt[kk]-rinfo[kk].edegrees[other];
            rinfo[kk].edegrees[other] -= vwgt[k];

            if (moved[kk] == -1)
              PQueueUpdateUp(&parts, kk, oldgain, oldgain+vwgt[k]);
          }
        }

        /* Insert the new vertex into the priority queue */
        PQueueInsert(&parts, k, vwgt[k]-edegrees[other]);
      }
    }

    if (pwgts[to] > pwgts[other])
      break;
  }

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("\tBalanced sep: %6d at %4d, PWGTS: [%6d %6d], NBND: %6d\n", pwgts[2], nswaps, pwgts[0], pwgts[1], nbnd));

  graph->mincut = pwgts[2];
  graph->nbnd = nbnd;


  PQueueFree(ctrl, &parts);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function computes the maximum possible gain for a vertex
**************************************************************************/
int ComputeMaxNodeGain(int nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt)
{
  int i, j, k, max;

  max = 0;
  for (j=xadj[0]; j<xadj[1]; j++)
    max += vwgt[adjncy[j]];

  for (i=1; i<nvtxs; i++) {
    for (k=0, j=xadj[i]; j<xadj[i+1]; j++)
      k += vwgt[adjncy[j]];
    if (max < k)
      max = k;
  }

  return max;
}


