/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * balance.c
 *
 * This file contains code that is used to forcefully balance either
 * bisections or k-sections
 *
 * Started 7/29/97
 * George
 *
 * $Id: balance.c,v 1.1 1998/11/27 17:59:10 karypis Exp $
 *
 */

#include "metis.h"

void Bnd2WayBalance(CtrlType *ctrl, GraphType *graph, int *tpwgts);
void General2WayBalance(CtrlType *ctrl, GraphType *graph, int *tpwgts);

/*************************************************************************
* This function is the entry point of the bisection balancing algorithms.
**************************************************************************/
void Balance2Way(CtrlType *ctrl, GraphType *graph, int *tpwgts, float ubfactor)
{
  int i, j, nvtxs, from, imax, gain, mindiff;
  idxtype *id, *ed;

  /* Return right away if the balance is OK */
  mindiff = abs(tpwgts[0]-graph->pwgts[0]);
  if (mindiff < 3*(graph->pwgts[0]+graph->pwgts[1])/graph->nvtxs)
    return;
  if (graph->pwgts[0] > tpwgts[0] && graph->pwgts[0] < (int)(ubfactor*tpwgts[0]))
    return;
  if (graph->pwgts[1] > tpwgts[1] && graph->pwgts[1] < (int)(ubfactor*tpwgts[1]))
    return;

  if (graph->nbnd > 0)
    Bnd2WayBalance(ctrl, graph, tpwgts);
  else
    General2WayBalance(ctrl, graph, tpwgts);

}



/*************************************************************************
* This function balances two partitions by moving boundary nodes
* from the domain that is overweight to the one that is underweight.
**************************************************************************/
void Bnd2WayBalance(CtrlType *ctrl, GraphType *graph, int *tpwgts)
{
  int i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, tmp;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
  idxtype *moved, *perm;
  PQueueType parts;
  int higain, oldgain, mincut, mindiff;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->id;
  ed = graph->ed;
  pwgts = graph->pwgts;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  moved = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  /* Determine from which domain you will be moving data */
  mindiff = abs(tpwgts[0]-pwgts[0]);
  from = (pwgts[0] < tpwgts[0] ? 1 : 0);
  to = (from+1)%2;

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d] T[%6d %6d], Nv-Nb[%6d %6d]. ICut: %6d [B]\n",
             pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  tmp = graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)];
  PQueueInit(ctrl, &parts, nvtxs, tmp);

  idxset(nvtxs, -1, moved);

  ASSERT(ComputeCut(graph, where) == graph->mincut);
  ASSERT(CheckBnd(graph));

  /* Insert the boundary nodes of the proper partition whose size is OK in the priority queue */
  nbnd = graph->nbnd;
  RandomPermute(nbnd, perm, 1);
  for (ii=0; ii<nbnd; ii++) {
    i = perm[ii];
    ASSERT(ed[bndind[i]] > 0 || id[bndind[i]] == 0);
    ASSERT(bndptr[bndind[i]] != -1);
    if (where[bndind[i]] == from && vwgt[bndind[i]] <= mindiff)
      PQueueInsert(&parts, bndind[i], ed[bndind[i]]-id[bndind[i]]);
  }

  mincut = graph->mincut;
  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if ((higain = PQueueGetMax(&parts)) == -1)
      break;
    ASSERT(bndptr[higain] != -1);

    if (pwgts[to]+vwgt[higain] > tpwgts[to])
      break;

    mincut -= (ed[higain]-id[higain]);
    INC_DEC(pwgts[to], pwgts[from], vwgt[higain]);

    where[higain] = to;
    moved[higain] = nswaps;

    IFSET(ctrl->dbglvl, DBG_MOVEINFO,
      printf("Moved %6d from %d. [%3d %3d] %5d [%4d %4d]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]));

    /**************************************************************
    * Update the id[i]/ed[i] values of the affected nodes
    ***************************************************************/
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);

    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];
      oldgain = ed[k]-id[k];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      /* Update its boundary information and queue position */
      if (bndptr[k] != -1) { /* If k was a boundary vertex */
        if (ed[k] == 0) { /* Not a boundary vertex any more */
          BNDDelete(nbnd, bndind, bndptr, k);
          if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)  /* Remove it if in the queues */
            PQueueDelete(&parts, k, oldgain);
        }
        else { /* If it has not been moved, update its position in the queue */
          if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
            PQueueUpdate(&parts, k, oldgain, ed[k]-id[k]);
        }
      }
      else {
        if (ed[k] > 0) {  /* It will now become a boundary vertex */
          BNDInsert(nbnd, bndind, bndptr, k);
          if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
            PQueueInsert(&parts, k, ed[k]-id[k]);
        }
      }
    }
  }

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("\tMinimum cut: %6d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, pwgts[0], pwgts[1], nbnd));

  graph->mincut = mincut;
  graph->nbnd = nbnd;

  PQueueFree(ctrl, &parts);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function balances two partitions by moving the highest gain
* (including negative gain) vertices to the other domain.
* It is used only when tha unbalance is due to non contigous
* subdomains. That is, the are no boundary vertices.
* It moves vertices from the domain that is overweight to the one that
* is underweight.
**************************************************************************/
void General2WayBalance(CtrlType *ctrl, GraphType *graph, int *tpwgts)
{
  int i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, tmp;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
  idxtype *moved, *perm;
  PQueueType parts;
  int higain, oldgain, mincut, mindiff;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->id;
  ed = graph->ed;
  pwgts = graph->pwgts;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  moved = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  /* Determine from which domain you will be moving data */
  mindiff = abs(tpwgts[0]-pwgts[0]);
  from = (pwgts[0] < tpwgts[0] ? 1 : 0);
  to = (from+1)%2;

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d] T[%6d %6d], Nv-Nb[%6d %6d]. ICut: %6d [B]\n",
             pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  tmp = graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)];
  PQueueInit(ctrl, &parts, nvtxs, tmp);

  idxset(nvtxs, -1, moved);

  ASSERT(ComputeCut(graph, where) == graph->mincut);
  ASSERT(CheckBnd(graph));

  /* Insert the nodes of the proper partition whose size is OK in the priority queue */
  RandomPermute(nvtxs, perm, 1);
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];
    if (where[i] == from && vwgt[i] <= mindiff)
      PQueueInsert(&parts, i, ed[i]-id[i]);
  }

  mincut = graph->mincut;
  nbnd = graph->nbnd;
  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if ((higain = PQueueGetMax(&parts)) == -1)
      break;

    if (pwgts[to]+vwgt[higain] > tpwgts[to])
      break;

    mincut -= (ed[higain]-id[higain]);
    INC_DEC(pwgts[to], pwgts[from], vwgt[higain]);

    where[higain] = to;
    moved[higain] = nswaps;

    IFSET(ctrl->dbglvl, DBG_MOVEINFO,
      printf("Moved %6d from %d. [%3d %3d] %5d [%4d %4d]\n", higain, from, ed[higain]-id[higain], vwgt[higain], mincut, pwgts[0], pwgts[1]));

    /**************************************************************
    * Update the id[i]/ed[i] values of the affected nodes
    ***************************************************************/
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);
    if (ed[higain] > 0 && bndptr[higain] == -1)
      BNDInsert(nbnd, bndind,  bndptr, higain);

    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];
      oldgain = ed[k]-id[k];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      /* Update the queue position */
      if (moved[k] == -1 && where[k] == from && vwgt[k] <= mindiff)
        PQueueUpdate(&parts, k, oldgain, ed[k]-id[k]);

      /* Update its boundary information */
      if (ed[k] == 0 && bndptr[k] != -1)
        BNDDelete(nbnd, bndind, bndptr, k);
      else if (ed[k] > 0 && bndptr[k] == -1)
        BNDInsert(nbnd, bndind, bndptr, k);
    }
  }

  IFSET(ctrl->dbglvl, DBG_REFINE,
    printf("\tMinimum cut: %6d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, pwgts[0], pwgts[1], nbnd));

  graph->mincut = mincut;
  graph->nbnd = nbnd;

  PQueueFree(ctrl, &parts);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}
