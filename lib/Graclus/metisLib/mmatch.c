/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mmatch.c
 *
 * This file contains the code that computes matchings and creates the next
 * level coarse graph.
 *
 * Started 7/23/97
 * George
 *
 * $Id: mmatch.c,v 1.3 1998/11/30 14:50:44 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void MCMatch_RM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, ncon, cnvtxs, maxidx;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *match, *cmap, *perm;
  float *nvwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  RandomPermute(nvtxs, perm, 1);

  cnvtxs = 0;
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;

      /* Find a random matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (match[k] == UNMATCHED && AreAllVwgtsBelowFast(ncon, nvwgt+i*ncon, nvwgt+k*ncon, ctrl->nmaxvwgt)) {
          maxidx = k;
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  CreateCoarseGraph(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void MCMatch_HEM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, l, nvtxs, cnvtxs, ncon, maxidx, maxwgt;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *match, *cmap, *perm;
  float *nvwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  RandomPermute(nvtxs, perm, 1);

  cnvtxs = 0;
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;
      maxwgt = 0;

      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (match[k] == UNMATCHED && maxwgt <= adjwgt[j] &&
               AreAllVwgtsBelowFast(ncon, nvwgt+i*ncon, nvwgt+k*ncon, ctrl->nmaxvwgt)) {
          maxwgt = adjwgt[j];
          maxidx = adjncy[j];
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  CreateCoarseGraph(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void MCMatch_SHEM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, cnvtxs, ncon, maxidx, maxwgt, avgdegree;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *match, *cmap, *degrees, *perm, *tperm;
  float *nvwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  tperm = idxwspacemalloc(ctrl, nvtxs);
  degrees = idxwspacemalloc(ctrl, nvtxs);

  RandomPermute(nvtxs, tperm, 1);
  avgdegree = (int) 0.7*(xadj[nvtxs]/nvtxs);
  for (i=0; i<nvtxs; i++)
    degrees[i] = (xadj[i+1]-xadj[i] > avgdegree ? avgdegree : xadj[i+1]-xadj[i]);
  BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

  cnvtxs = 0;

  /* Take care any islands. Islands are matched with non-islands due to coarsening */
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      if (xadj[i] < xadj[i+1])
        break;

      maxidx = i;
      for (j=nvtxs-1; j>ii; j--) {
        k = perm[j];
        if (match[k] == UNMATCHED && xadj[k] < xadj[k+1]) {
          maxidx = k;
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  /* Continue with normal matching */
  for (; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;
      maxwgt = 0;

      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (match[k] == UNMATCHED && maxwgt <= adjwgt[j] &&
               AreAllVwgtsBelowFast(ncon, nvwgt+i*ncon, nvwgt+k*ncon, ctrl->nmaxvwgt)) {
          maxwgt = adjwgt[j];
          maxidx = adjncy[j];
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  idxwspacefree(ctrl, nvtxs);  /* degrees */
  idxwspacefree(ctrl, nvtxs);  /* tperm */

  CreateCoarseGraph(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void MCMatch_SHEBM(CtrlType *ctrl, GraphType *graph, int norm)
{
  int i, ii, j, k, nvtxs, cnvtxs, ncon, maxidx, maxwgt, avgdegree;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *match, *cmap, *degrees, *perm, *tperm;
  float *nvwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  tperm = idxwspacemalloc(ctrl, nvtxs);
  degrees = idxwspacemalloc(ctrl, nvtxs);

  RandomPermute(nvtxs, tperm, 1);
  avgdegree = (int) 0.7*(xadj[nvtxs]/nvtxs);
  for (i=0; i<nvtxs; i++)
    degrees[i] = (xadj[i+1]-xadj[i] > avgdegree ? avgdegree : xadj[i+1]-xadj[i]);
  BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

  cnvtxs = 0;

  /* Take care any islands. Islands are matched with non-islands due to coarsening */
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      if (xadj[i] < xadj[i+1])
        break;

      maxidx = i;
      for (j=nvtxs-1; j>ii; j--) {
        k = perm[j];
        if (match[k] == UNMATCHED && xadj[k] < xadj[k+1]) {
          maxidx = k;
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  /* Continue with normal matching */
  for (; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;
      maxwgt = -1;

      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];

        if (match[k] == UNMATCHED &&
            AreAllVwgtsBelowFast(ncon, nvwgt+i*ncon, nvwgt+k*ncon, ctrl->nmaxvwgt) &&
            (maxwgt < adjwgt[j] ||
              (maxwgt == adjwgt[j] &&
               BetterVBalance(ncon, norm, nvwgt+i*ncon, nvwgt+maxidx*ncon, nvwgt+k*ncon) >= 0
              )
            )
           ) {
          maxwgt = adjwgt[j];
          maxidx = k;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  idxwspacefree(ctrl, nvtxs);  /* degrees */
  idxwspacefree(ctrl, nvtxs);  /* tperm */

  CreateCoarseGraph(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void MCMatch_SBHEM(CtrlType *ctrl, GraphType *graph, int norm)
{
  int i, ii, j, k, nvtxs, cnvtxs, ncon, maxidx, maxwgt, avgdegree;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *match, *cmap, *degrees, *perm, *tperm;
  float *nvwgt, vbal;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  tperm = idxwspacemalloc(ctrl, nvtxs);
  degrees = idxwspacemalloc(ctrl, nvtxs);

  RandomPermute(nvtxs, tperm, 1);
  avgdegree = (int) 0.7*(xadj[nvtxs]/nvtxs);
  for (i=0; i<nvtxs; i++)
    degrees[i] = (xadj[i+1]-xadj[i] > avgdegree ? avgdegree : xadj[i+1]-xadj[i]);
  BucketSortKeysInc(nvtxs, avgdegree, degrees, tperm, perm);

  cnvtxs = 0;

  /* Take care any islands. Islands are matched with non-islands due to coarsening */
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      if (xadj[i] < xadj[i+1])
        break;

      maxidx = i;
      for (j=nvtxs-1; j>ii; j--) {
        k = perm[j];
        if (match[k] == UNMATCHED && xadj[k] < xadj[k+1]) {
          maxidx = k;
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  /* Continue with normal matching */
  for (; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;
      maxwgt = -1;
      vbal = 0.0;

      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (match[k] == UNMATCHED && AreAllVwgtsBelowFast(ncon, nvwgt+i*ncon, nvwgt+k*ncon, ctrl->nmaxvwgt)) {
          if (maxidx != i)
            vbal = BetterVBalance(ncon, norm, nvwgt+i*ncon, nvwgt+maxidx*ncon, nvwgt+k*ncon);

          if (vbal > 0 || (vbal > -.01 && maxwgt < adjwgt[j])) {
            maxwgt = adjwgt[j];
            maxidx = k;
          }
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  idxwspacefree(ctrl, nvtxs);  /* degrees */
  idxwspacefree(ctrl, nvtxs);  /* tperm */

  CreateCoarseGraph(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}





/*************************************************************************
* This function checks if v+u2 provides a better balance in the weight
* vector that v+u1
**************************************************************************/
float BetterVBalance(int ncon, int norm, float *vwgt, float *u1wgt, float *u2wgt)
{
  int i;
  float sum1, sum2, max1, max2, min1, min2, diff1, diff2;

  if (norm == -1) {
    max1 = min1 = vwgt[0]+u1wgt[0];
    max2 = min2 = vwgt[0]+u2wgt[0];
    sum1 = vwgt[0]+u1wgt[0];
    sum2 = vwgt[0]+u2wgt[0];

    for (i=1; i<ncon; i++) {
      if (max1 < vwgt[i]+u1wgt[i])
        max1 = vwgt[i]+u1wgt[i];
      if (min1 > vwgt[i]+u1wgt[i])
        min1 = vwgt[i]+u1wgt[i];

      if (max2 < vwgt[i]+u2wgt[i])
        max2 = vwgt[i]+u2wgt[i];
      if (min2 > vwgt[i]+u2wgt[i])
        min2 = vwgt[i]+u2wgt[i];

      sum1 += vwgt[i]+u1wgt[i];
      sum2 += vwgt[i]+u2wgt[i];
    }

    if (sum1 == 0.0)
      return 1;
    else if (sum2 == 0.0)
      return -1;
    else
      return ((max1-min1)/sum1) - ((max2-min2)/sum2);
  }
  else if (norm == 1) {
    sum1 = sum2 = 0.0;
    for (i=0; i<ncon; i++) {
      sum1 += vwgt[i]+u1wgt[i];
      sum2 += vwgt[i]+u2wgt[i];
    }
    sum1 = sum1/(1.0*ncon);
    sum2 = sum2/(1.0*ncon);

    diff1 = diff2 = 0.0;
    for (i=0; i<ncon; i++) {
      diff1 += fabs(sum1 - (vwgt[i]+u1wgt[i]));
      diff2 += fabs(sum2 - (vwgt[i]+u2wgt[i]));
    }

    return diff1 - diff2;
  }
  else {
    errexit("Unknown norm: %d\n", norm);
  }
  return 0.0;
}


/*************************************************************************
* This function checks if the vertex weights of two vertices are below
* a given set of values
**************************************************************************/
int AreAllVwgtsBelowFast(int ncon, float *vwgt1, float *vwgt2, float limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (vwgt1[i] + vwgt2[i] > limit)
      return 0;

  return 1;
}

