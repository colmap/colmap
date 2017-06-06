/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mbalance2.c
 *
 * This file contains code that is used to forcefully balance either
 * bisections or k-sections
 *
 * Started 7/29/97
 * George
 *
 * $Id: mbalance2.c,v 1.1 1998/11/27 17:59:19 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of the bisection balancing algorithms.
**************************************************************************/
void MocBalance2Way2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i;
  float tvec[MAXNCON];

  Compute2WayHLoadImbalanceVec(graph->ncon, graph->npwgts, tpwgts, tvec);
  if (!AreAllBelow(graph->ncon, tvec, ubvec))
    MocGeneral2WayBalance2(ctrl, graph, tpwgts, ubvec);
}



/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void MocGeneral2WayBalance2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i, ii, j, k, l, kwgt, nvtxs, ncon, nbnd, nswaps, from, to, pass, me, limit, tmp, cnum;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *swaps, *perm, *qnum;
  float *nvwgt, *npwgts, origbal[MAXNCON], minbal[MAXNCON], newbal[MAXNCON];
  PQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut, newcut, mincutorder;
  float *maxwgt, *minwgt, tvec[MAXNCON];


  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->id;
  ed = graph->ed;
  npwgts = graph->npwgts;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  moved = idxwspacemalloc(ctrl, nvtxs);
  swaps = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);
  qnum = idxwspacemalloc(ctrl, nvtxs);

  limit = (int) amin(amax(0.01*nvtxs, 15), 100);

  /* Setup the weight intervals of the two subdomains */
  minwgt = fwspacemalloc(ctrl, 2*ncon);
  maxwgt = fwspacemalloc(ctrl, 2*ncon);

  for (i=0; i<2; i++) {
    for (j=0; j<ncon; j++) {
      maxwgt[i*ncon+j] = tpwgts[i]*ubvec[j];
      minwgt[i*ncon+j] = tpwgts[i]*(1.0/ubvec[j]);
    }
  }


  /* Initialize the queues */
  for (i=0; i<ncon; i++) {
    PQueueInit(ctrl, &parts[i][0], nvtxs, PLUS_GAINSPAN+1);
    PQueueInit(ctrl, &parts[i][1], nvtxs, PLUS_GAINSPAN+1);
  }
  for (i=0; i<nvtxs; i++)
    qnum[i] = samax(ncon, nvwgt+i*ncon);

  Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, origbal);
  for (i=0; i<ncon; i++)
    minbal[i] = origbal[i];

  newcut = mincut = graph->mincut;
  mincutorder = -1;

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Parts: [");
    for (l=0; l<ncon; l++)
      printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
    printf("] T[%.3f %.3f], Nv-Nb[%5d, %5d]. ICut: %6d, LB: ", tpwgts[0], tpwgts[1],
            graph->nvtxs, graph->nbnd, graph->mincut);
    for (i=0; i<ncon; i++)
      printf("%.3f ", origbal[i]);
    printf("[B]\n");
  }

  idxset(nvtxs, -1, moved);

  ASSERT(ComputeCut(graph, where) == graph->mincut);
  ASSERT(CheckBnd(graph));

  /* Insert all nodes in the priority queues */
  nbnd = graph->nbnd;
  RandomPermute(nvtxs, perm, 1);
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];
    PQueueInsert(&parts[qnum[i]][where[i]], i, ed[i]-id[i]);
  }


  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if (AreAllBelow(ncon, minbal, ubvec))
      break;

    SelectQueue3(ncon, npwgts, tpwgts, &from, &cnum, parts, maxwgt);
    to = (from+1)%2;

    if (from == -1 || (higain = PQueueGetMax(&parts[cnum][from])) == -1)
      break;

    saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);
    newcut -= (ed[higain]-id[higain]);
    Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, newbal);

    if (IsBetter2wayBalance(ncon, newbal, minbal, ubvec) ||
        (IsBetter2wayBalance(ncon, newbal, origbal, ubvec) && newcut < mincut)) {
      mincut = newcut;
      for (i=0; i<ncon; i++)
        minbal[i] = newbal[i];
      mincutorder = nswaps;
    }
    else if (nswaps-mincutorder > limit) { /* We hit the limit, undo last move */
      newcut += (ed[higain]-id[higain]);
      saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);
      saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      break;
    }

    where[higain] = to;
    moved[higain] = nswaps;
    swaps[nswaps] = higain;

    if (ctrl->dbglvl&DBG_MOVEINFO) {
      printf("Moved %6d from %d(%d). Gain: %5d, Cut: %5d, NPwgts: ", higain, from, cnum, ed[higain]-id[higain], newcut);
      for (i=0; i<ncon; i++)
        printf("(%.3f, %.3f) ", npwgts[i], npwgts[ncon+i]);

      Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, tvec);
      printf(", LB: ");
      for (i=0; i<ncon; i++)
        printf("%.3f ", tvec[i]);
      if (mincutorder == nswaps)
        printf(" *\n");
      else
        printf("\n");
    }


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
      if (moved[k] == -1)
        PQueueUpdate(&parts[qnum[k]][where[k]], k, oldgain, ed[k]-id[k]);

      /* Update its boundary information */
      if (ed[k] == 0 && bndptr[k] != -1)
        BNDDelete(nbnd, bndind, bndptr, k);
      else if (ed[k] > 0 && bndptr[k] == -1)
        BNDInsert(nbnd, bndind, bndptr, k);
    }

  }



  /****************************************************************
  * Roll back computations
  *****************************************************************/
  for (i=0; i<nswaps; i++)
    moved[swaps[i]] = -1;  /* reset moved array */
  for (nswaps--; nswaps>mincutorder; nswaps--) {
    higain = swaps[nswaps];

    to = where[higain] = (where[higain]+1)%2;
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);
    else if (ed[higain] > 0 && bndptr[higain] == -1)
      BNDInsert(nbnd, bndind,  bndptr, higain);

    saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+((to+1)%2)*ncon, 1);
    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      if (bndptr[k] != -1 && ed[k] == 0)
        BNDDelete(nbnd, bndind, bndptr, k);
      if (bndptr[k] == -1 && ed[k] > 0)
        BNDInsert(nbnd, bndind, bndptr, k);
    }
  }

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("\tMincut: %6d at %5d, NBND: %6d, NPwgts: [", mincut, mincutorder, nbnd);
    for (i=0; i<ncon; i++)
      printf("(%.3f, %.3f) ", npwgts[i], npwgts[ncon+i]);
    printf("], LB: ");
    Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, tvec);
    for (i=0; i<ncon; i++)
      printf("%.3f ", tvec[i]);
    printf("\n");
  }

  graph->mincut = mincut;
  graph->nbnd = nbnd;


  for (i=0; i<ncon; i++) {
    PQueueFree(ctrl, &parts[i][0]);
    PQueueFree(ctrl, &parts[i][1]);
  }

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  fwspacefree(ctrl, 2*ncon);
  fwspacefree(ctrl, 2*ncon);

}




/*************************************************************************
* This function selects the partition number and the queue from which
* we will move vertices out
**************************************************************************/
void SelectQueue3(int ncon, float *npwgts, float *tpwgts, int *from, int *cnum,
       PQueueType queues[MAXNCON][2], float *maxwgt)
{
  int i, j, maxgain=0;
  float maxdiff=0.0, diff;

  *from = -1;
  *cnum = -1;

  /* First determine the side and the queue, irrespective of the presence of nodes */
  for (j=0; j<2; j++) {
    for (i=0; i<ncon; i++) {
      diff = npwgts[j*ncon+i]-maxwgt[j*ncon+i];
      if (diff >= maxdiff) {
        maxdiff = diff;
        *from = j;
        *cnum = i;
      }
    }
  }

/* DELETE
j = *from;
for (i=0; i<ncon; i++)
  printf("[%5d %5d %.4f %.4f] ", i, PQueueGetSize(&queues[i][j]), npwgts[j*ncon+i], maxwgt[j*ncon+i]);
printf("***[%5d %5d]\n", *cnum, *from);
*/

  /* If the desired queue is empty, select a node from that side anyway */
  if (*from != -1 && PQueueGetSize(&queues[*cnum][*from]) == 0) {
    for (i=0; i<ncon; i++) {
      if (PQueueGetSize(&queues[i][*from]) > 0) {
        maxdiff = (npwgts[(*from)*ncon+i] - maxwgt[(*from)*ncon+i]);
        *cnum = i;
        break;
      }
    }

    for (i++; i<ncon; i++) {
      diff = npwgts[(*from)*ncon+i] - maxwgt[(*from)*ncon+i];
      if (diff > maxdiff && PQueueGetSize(&queues[i][*from]) > 0) {
        maxdiff = diff;
        *cnum = i;
      }
    }
  }

  /* If the constraints ar OK, select a high gain vertex */
  if (*from == -1) {
    maxgain = -100000;
    for (j=0; j<2; j++) {
      for (i=0; i<ncon; i++) {
        if (PQueueGetSize(&queues[i][j]) > 0 && PQueueGetKey(&queues[i][j]) > maxgain) {
          maxgain = PQueueGetKey(&queues[i][0]);
          *from = j;
          *cnum = i;
        }
      }
    }

    /* printf("(%2d %2d) %3d\n", *from, *cnum, maxgain); */
  }
}
