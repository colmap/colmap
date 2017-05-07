/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * match.c
 *
 * This file contains the code that computes matchings and creates the next
 * level coarse graph.
 *
 * Started 7/23/97
 * George
 *
 * $Id: match.c,v 1.1 1998/11/27 17:59:18 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void Match_SHEMN(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, cnvtxs, maxidx, avgdegree;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *adjwgtsum;
  idxtype *match, *cmap, *degrees, *perm, *tperm;
  float rtemp1, rtemp2, maxwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  adjwgtsum = graph->adjwgtsum;

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
      //rtemp1 = 1.0/vwgt[i];
      rtemp1 = 1.0/adjwgtsum[i];
      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
	k = adjncy[j];
	//rtemp2 = adjwgt[j] *(rtemp1 + 1.0/vwgt[k]);
	rtemp2 = adjwgt[j] *(rtemp1 + 1.0/adjwgtsum[k]);
        if (match[k] == UNMATCHED && maxwgt < rtemp2 && vwgt[i]+vwgt[k] <= ctrl->maxvwgt) {
          maxwgt = rtemp2;
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
void Match_HEMN(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, cnvtxs, maxidx;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *adjwgtsum;
  idxtype *match, *cmap, *perm;
  float rtemp1, rtemp2, maxwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  adjwgtsum = graph->adjwgtsum;

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
      rtemp1 = 1.0/adjwgtsum[i];
      //rtemp1 = 1.0/vwgt[i];
      /* Find a heavy-edge matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
	rtemp2 = adjwgt[j] *(rtemp1 + 1.0/adjwgtsum[k]);
	//rtemp2 = adjwgt[j] *(rtemp1 + 1.0/vwgt[k]);
        if (match[k] == UNMATCHED && maxwgt < rtemp2 && vwgt[i]+vwgt[k] <= ctrl->maxvwgt) {
          maxwgt = rtemp2;
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
void Match_RM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, nvtxs, cnvtxs, maxidx;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt;
  idxtype *match, *cmap, *perm;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
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
        if (match[adjncy[j]] == UNMATCHED && vwgt[i]+vwgt[adjncy[j]] <= ctrl->maxvwgt) {
          maxidx = adjncy[j];
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
void Match_RM_NVW(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, nvtxs, cnvtxs, maxidx;
  idxtype *xadj, *adjncy;
  idxtype *match, *cmap, *perm;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  RandomPermute(nvtxs, perm, 1);

  cnvtxs = 0;
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  // Unmatched
      maxidx = i;

      // Find a random matching, subject to maxvwgt constraints
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (match[adjncy[j]] == UNMATCHED) {
          maxidx = adjncy[j];
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));

  CreateCoarseGraph_NVW(ctrl, graph, cnvtxs, match, perm);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}




/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void Match_HEM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, cnvtxs, maxidx, maxwgt;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt;
  idxtype *match, *cmap, *perm;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
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
        if (match[k] == UNMATCHED && maxwgt < adjwgt[j] && vwgt[i]+vwgt[k] <= ctrl->maxvwgt) {
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
void Match_SHEM(CtrlType *ctrl, GraphType *graph)
{
  int i, ii, j, k, nvtxs, cnvtxs, maxidx, maxwgt, avgdegree;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt;
  idxtype *match, *cmap, *degrees, *perm, *tperm;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  cmap = graph->cmap;
  match = idxset(nvtxs, UNMATCHED, idxwspacemalloc(ctrl, nvtxs));

  perm = idxwspacemalloc(ctrl, nvtxs);
  tperm = idxwspacemalloc(ctrl, nvtxs);
  degrees =  idxwspacemalloc(ctrl, nvtxs);

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
        if (match[adjncy[j]] == UNMATCHED && maxwgt < adjwgt[j] && vwgt[i]+vwgt[adjncy[j]] <= ctrl->maxvwgt) {
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

