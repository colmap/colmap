/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mfm2.c
 *
 * This file contains code that implements the edge-based FM refinement
 *
 * Started 7/23/97
 * George
 *
 * $Id: mfm2.c,v 1.2 1998/11/30 14:50:44 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void MocFM_2WayEdgeRefine2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *orgubvec,
       int npasses)
{
  int i, ii, j, k, l, kwgt, nvtxs, ncon, nbnd, nswaps, from, to, pass, me, limit, tmp, cnum;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *swaps, *perm, *qnum;
  float *nvwgt, *npwgts, origdiff[MAXNCON], origbal[MAXNCON], minbal[MAXNCON];
  PQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut, initcut, newcut, mincutorder;
  float *maxwgt, *minwgt, ubvec[MAXNCON], tvec[MAXNCON];

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

  Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, origbal);
  for (i=0; i<ncon; i++) {
    origdiff[i] = fabs(tpwgts[0]-npwgts[i]);
    ubvec[i] = amax(origbal[i], orgubvec[i]);
  }

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


  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Parts: [");
    for (l=0; l<ncon; l++)
      printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
    printf("] T[%.3f %.3f], Nv-Nb[%5d, %5d]. ICut: %6d, LB: ", tpwgts[0], tpwgts[1],
            graph->nvtxs, graph->nbnd, graph->mincut);
    for (i=0; i<ncon; i++)
      printf("%.3f ", origbal[i]);
    printf("\n");
  }

  idxset(nvtxs, -1, moved);
  for (pass=0; pass<npasses; pass++) { /* Do a number of passes */
    for (i=0; i<ncon; i++) {
      PQueueReset(&parts[i][0]);
      PQueueReset(&parts[i][1]);
    }

    mincutorder = -1;
    newcut = mincut = initcut = graph->mincut;
    Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, minbal);

    ASSERT(ComputeCut(graph, where) == graph->mincut);
    ASSERT(CheckBnd(graph));

    /* Insert boundary nodes in the priority queues */
    nbnd = graph->nbnd;
    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      ASSERT(ed[i] > 0 || id[i] == 0);
      ASSERT(bndptr[i] != -1);
      PQueueInsert(&parts[qnum[i]][where[i]], i, ed[i]-id[i]);
    }

    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      SelectQueue2(ncon, npwgts, tpwgts, &from, &cnum, parts, maxwgt);
      to = (from+1)%2;

      if (from == -1 || (higain = PQueueGetMax(&parts[cnum][from])) == -1)
        break;
      ASSERT(bndptr[higain] != -1);

      newcut -= (ed[higain]-id[higain]);
      saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);

      Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, tvec);
      if ((newcut < mincut && AreAllBelow(ncon, tvec, ubvec)) ||
          (newcut == mincut && IsBetter2wayBalance(ncon, tvec, minbal, ubvec))) {
        mincut = newcut;
        for (i=0; i<ncon; i++)
          minbal[i] = tvec[i];
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
        for (l=0; l<ncon; l++)
          printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);

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
            if (moved[k] == -1)  /* Remove it if in the queues */
              PQueueDelete(&parts[qnum[k]][where[k]], k, oldgain);
          }
          else { /* If it has not been moved, update its position in the queue */
            if (moved[k] == -1)
              PQueueUpdate(&parts[qnum[k]][where[k]], k, oldgain, ed[k]-id[k]);
          }
        }
        else {
          if (ed[k] > 0) {  /* It will now become a boundary vertex */
            BNDInsert(nbnd, bndind, bndptr, k);
            if (moved[k] == -1)
              PQueueInsert(&parts[qnum[k]][where[k]], k, ed[k]-id[k]);
          }
        }
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
      for (l=0; l<ncon; l++)
        printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
      printf("], LB: ");
      Compute2WayHLoadImbalanceVec(ncon, npwgts, tpwgts, tvec);
      for (i=0; i<ncon; i++)
        printf("%.3f ", tvec[i]);
      printf("\n");
    }

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut == initcut)
      break;
  }

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
void SelectQueue2(int ncon, float *npwgts, float *tpwgts, int *from, int *cnum,
       PQueueType queues[MAXNCON][2], float *maxwgt)
{
  int i, j, maxgain=0;
  float diff, max, maxdiff=0.0;

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

  if (*from != -1 && PQueueGetSize(&queues[*cnum][*from]) == 0) {
    /* The desired queue is empty, select a node from that side anyway */
    for (i=0; i<ncon; i++) {
      if (PQueueGetSize(&queues[i][*from]) > 0) {
        max = (npwgts[(*from)*ncon+i] - maxwgt[(*from)*ncon+i]);
        *cnum = i;
        break;
      }
    }

    for (i++; i<ncon; i++) {
      diff = npwgts[(*from)*ncon+i] - maxwgt[(*from)*ncon+i];
      if (diff > max && PQueueGetSize(&queues[i][*from]) > 0) {
        max = diff;
        *cnum = i;
      }
    }
  }

  /* Check to see if you can focus on the cut */
  if (maxdiff <= 0.0 || *from == -1) {
    maxgain = -100000;

    for (j=0; j<2; j++) {
      for (i=0; i<ncon; i++) {
        if (PQueueGetSize(&queues[i][j]) > 0 && PQueueGetKey(&queues[i][j]) > maxgain) {
          maxgain = PQueueGetKey(&queues[i][j]);
          *from = j;
          *cnum = i;
        }
      }
    }

    /* printf("(%2d %2d) %3d\n", *from, *cnum, maxgain); */
  }
}


/*************************************************************************
* This function checks if the newbal is better than oldbal given the
* ubvector ubvec
**************************************************************************/
int IsBetter2wayBalance(int ncon, float *newbal, float *oldbal, float *ubvec)
{
  int i, j;
  float max1=0.0, max2=0.0, sum1=0.0, sum2=0.0, tmp;

  for (i=0; i<ncon; i++) {
    tmp = (newbal[i]-1)/(ubvec[i]-1);
    max1 = (max1 < tmp ? tmp : max1);
    sum1 += tmp;

    tmp = (oldbal[i]-1)/(ubvec[i]-1);
    max2 = (max2 < tmp ? tmp : max2);
    sum2 += tmp;
  }

  if (max1 < max2)
    return 1;
  else if (max1 > max2)
    return 0;
  else
    return sum1 <= sum2;
}


