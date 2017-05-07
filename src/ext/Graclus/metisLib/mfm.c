/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mfm.c
 *
 * This file contains code that implements the edge-based FM refinement
 *
 * Started 7/23/97
 * George
 *
 * $Id: mfm.c,v 1.3 1998/11/30 14:50:44 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void MocFM_2WayEdgeRefine(CtrlType *ctrl, GraphType *graph, float *tpwgts, int npasses)
{
  int i, ii, j, k, l, kwgt, nvtxs, ncon, nbnd, nswaps, from, to, pass, me, limit, tmp, cnum;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *swaps, *perm, *qnum;
  float *nvwgt, *npwgts, mindiff[MAXNCON], origbal, minbal, newbal;
  PQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut, initcut, newcut, mincutorder;
  float rtpwgts[2];

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

  limit = amin(amax(0.01*nvtxs, 25), 150);

  /* Initialize the queues */
  for (i=0; i<ncon; i++) {
    PQueueInit(ctrl, &parts[i][0], nvtxs, PLUS_GAINSPAN+1);
    PQueueInit(ctrl, &parts[i][1], nvtxs, PLUS_GAINSPAN+1);
  }
  for (i=0; i<nvtxs; i++)
    qnum[i] = samax(ncon, nvwgt+i*ncon);

  origbal = Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

  rtpwgts[0] = origbal*tpwgts[0];
  rtpwgts[1] = origbal*tpwgts[1];


  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Parts: [");
    for (l=0; l<ncon; l++)
      printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
    printf("] T[%.3f %.3f], Nv-Nb[%5d, %5d]. ICut: %6d, LB: %.3f\n", tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut, origbal);
  }

  idxset(nvtxs, -1, moved);
  for (pass=0; pass<npasses; pass++) { /* Do a number of passes */
    for (i=0; i<ncon; i++) {
      PQueueReset(&parts[i][0]);
      PQueueReset(&parts[i][1]);
    }

    mincutorder = -1;
    newcut = mincut = initcut = graph->mincut;
    for (i=0; i<ncon; i++)
      mindiff[i] = fabs(tpwgts[0]-npwgts[i]);
    minbal = Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

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
      SelectQueue(ncon, npwgts, rtpwgts, &from, &cnum, parts);
      to = (from+1)%2;

      if (from == -1 || (higain = PQueueGetMax(&parts[cnum][from])) == -1)
        break;
      ASSERT(bndptr[higain] != -1);

      saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);

      newcut -= (ed[higain]-id[higain]);
      newbal = Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

      if ((newcut < mincut && newbal-origbal <= .00001) ||
          (newcut == mincut && (newbal < minbal ||
                                (newbal == minbal && BetterBalance(ncon, npwgts, tpwgts, mindiff))))) {
        mincut = newcut;
        minbal = newbal;
        mincutorder = nswaps;
        for (i=0; i<ncon; i++)
          mindiff[i] = fabs(tpwgts[0]-npwgts[i]);
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
        printf(", %.3f LB: %.3f\n", minbal, newbal);
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
      printf("], LB: %.3f\n", Compute2WayHLoadImbalance(ncon, npwgts, tpwgts));
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

}


/*************************************************************************
* This function selects the partition number and the queue from which
* we will move vertices out
**************************************************************************/
void SelectQueue(int ncon, float *npwgts, float *tpwgts, int *from, int *cnum, PQueueType queues[MAXNCON][2])
{
  int i, part, maxgain=0;
  float max, maxdiff=0.0;

  *from = -1;
  *cnum = -1;

  /* First determine the side and the queue, irrespective of the presence of nodes */
  for (part=0; part<2; part++) {
    for (i=0; i<ncon; i++) {
      if (npwgts[part*ncon+i]-tpwgts[part] >= maxdiff) {
        maxdiff = npwgts[part*ncon+i]-tpwgts[part];
        *from = part;
        *cnum = i;
      }
    }
  }

  /* printf("Selected1 %d(%d) -> %d [%5f]\n", *from, *cnum, PQueueGetSize(&queues[*cnum][*from]), maxdiff); */

  if (*from != -1 && PQueueGetSize(&queues[*cnum][*from]) == 0) {
    /* The desired queue is empty, select a node from that side anyway */
    for (i=0; i<ncon; i++) {
      if (PQueueGetSize(&queues[i][*from]) > 0) {
        max = npwgts[(*from)*ncon + i];
        *cnum = i;
        break;
      }
    }

    for (i++; i<ncon; i++) {
      if (npwgts[(*from)*ncon + i] > max && PQueueGetSize(&queues[i][*from]) > 0) {
        max = npwgts[(*from)*ncon + i];
        *cnum = i;
      }
    }
  }

  /* Check to see if you can focus on the cut */
  if (maxdiff <= 0.0 || *from == -1) {
    maxgain = -100000;

    for (part=0; part<2; part++) {
      for (i=0; i<ncon; i++) {
        if (PQueueGetSize(&queues[i][part]) > 0 && PQueueGetKey(&queues[i][part]) > maxgain) {
          maxgain = PQueueGetKey(&queues[i][part]);
          *from = part;
          *cnum = i;
        }
      }
    }
  }

  /* printf("Selected2 %d(%d) -> %d\n", *from, *cnum, PQueueGetSize(&queues[*cnum][*from])); */
}





/*************************************************************************
* This function checks if the balance achieved is better than the diff
* For now, it uses a 2-norm measure
**************************************************************************/
int BetterBalance(int ncon, float *npwgts, float *tpwgts, float *diff)
{
  int i;
  float ndiff[MAXNCON];

  for (i=0; i<ncon; i++)
    ndiff[i] = fabs(tpwgts[0]-npwgts[i]);

  return snorm2(ncon, ndiff) < snorm2(ncon, diff);
}



/*************************************************************************
* This function computes the load imbalance over all the constrains
**************************************************************************/
float Compute2WayHLoadImbalance(int ncon, float *npwgts, float *tpwgts)
{
  int i;
  float max=0.0, temp;

  for (i=0; i<ncon; i++) {
    /* temp = amax(npwgts[i]/tpwgts[0], npwgts[ncon+i]/tpwgts[1]); */
    temp = fabs(tpwgts[0]-npwgts[i])/tpwgts[0];
    max = (max < temp ? temp : max);
  }
  return 1.0+max;
}


/*************************************************************************
* This function computes the load imbalance over all the constrains
* For now assume that we just want balanced partitionings
**************************************************************************/
void Compute2WayHLoadImbalanceVec(int ncon, float *npwgts, float *tpwgts, float *lbvec)
{
  int i;

  for (i=0; i<ncon; i++)
    lbvec[i] = 1.0 + fabs(tpwgts[0]-npwgts[i])/tpwgts[0];
}

