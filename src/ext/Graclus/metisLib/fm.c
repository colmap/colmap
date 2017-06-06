/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * fm.c
 *
 * This file contains code that implements the edge-based FM refinement
 *
 * Started 7/23/97
 * George
 *
 * $Id: fm.c,v 1.1 1998/11/27 17:59:14 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void FM_2WayEdgeRefine(CtrlType *ctrl, GraphType *graph, int *tpwgts, int npasses)
{
  int i, ii, j, k, kwgt, nvtxs, nbnd, nswaps, from, to, pass, me, limit, tmp;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind, *pwgts;
  idxtype *moved, *swaps, *perm;
  PQueueType parts[2];
  int higain, oldgain, mincut, mindiff, origdiff, initcut, newcut, mincutorder, avgvwgt;

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
  swaps = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);

  limit = (int) amin(amax(0.01*nvtxs, 15), 100);
  avgvwgt = amin((pwgts[0]+pwgts[1])/20, 2*(pwgts[0]+pwgts[1])/nvtxs);

  tmp = graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)];
  PQueueInit(ctrl, &parts[0], nvtxs, tmp);
  PQueueInit(ctrl, &parts[1], nvtxs, tmp);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d] T[%6d %6d], Nv-Nb[%6d %6d]. ICut: %6d\n",
             pwgts[0], pwgts[1], tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut));

  origdiff = abs(tpwgts[0]-pwgts[0]);
  idxset(nvtxs, -1, moved);
  for (pass=0; pass<npasses; pass++) { /* Do a number of passes */
    PQueueReset(&parts[0]);
    PQueueReset(&parts[1]);

    mincutorder = -1;
    newcut = mincut = initcut = graph->mincut;
    mindiff = abs(tpwgts[0]-pwgts[0]);

    ASSERT(ComputeCut(graph, where) == graph->mincut);
    ASSERT(CheckBnd(graph));

    /* Insert boundary nodes in the priority queues */
    nbnd = graph->nbnd;
    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = perm[ii];
      ASSERT(ed[bndind[i]] > 0 || id[bndind[i]] == 0);
      ASSERT(bndptr[bndind[i]] != -1);
      PQueueInsert(&parts[where[bndind[i]]], bndind[i], ed[bndind[i]]-id[bndind[i]]);
    }

    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      from = (tpwgts[0]-pwgts[0] < tpwgts[1]-pwgts[1] ? 0 : 1);
      to = (from+1)%2;

      if ((higain = PQueueGetMax(&parts[from])) == -1)
        break;
      ASSERT(bndptr[higain] != -1);

      newcut -= (ed[higain]-id[higain]);
      INC_DEC(pwgts[to], pwgts[from], vwgt[higain]);

      if ((newcut < mincut && abs(tpwgts[0]-pwgts[0]) <= origdiff+avgvwgt) ||
          (newcut == mincut && abs(tpwgts[0]-pwgts[0]) < mindiff)) {
        mincut = newcut;
        mindiff = abs(tpwgts[0]-pwgts[0]);
        mincutorder = nswaps;
      }
      else if (nswaps-mincutorder > limit) { /* We hit the limit, undo last move */
        newcut += (ed[higain]-id[higain]);
        INC_DEC(pwgts[from], pwgts[to], vwgt[higain]);
        break;
      }

      where[higain] = to;
      moved[higain] = nswaps;
      swaps[nswaps] = higain;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO,
        printf("Moved %6d from %d. [%3d %3d] %5d [%4d %4d]\n", higain, from, ed[higain]-id[higain], vwgt[higain], newcut, pwgts[0], pwgts[1]));

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
              PQueueDelete(&parts[where[k]], k, oldgain);
          }
          else { /* If it has not been moved, update its position in the queue */
            if (moved[k] == -1)
              PQueueUpdate(&parts[where[k]], k, oldgain, ed[k]-id[k]);
          }
        }
        else {
          if (ed[k] > 0) {  /* It will now become a boundary vertex */
            BNDInsert(nbnd, bndind, bndptr, k);
            if (moved[k] == -1)
              PQueueInsert(&parts[where[k]], k, ed[k]-id[k]);
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

      INC_DEC(pwgts[to], pwgts[(to+1)%2], vwgt[higain]);
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

    IFSET(ctrl->dbglvl, DBG_REFINE,
      printf("\tMinimum cut: %6d at %5d, PWGTS: [%6d %6d], NBND: %6d\n", mincut, mincutorder, pwgts[0], pwgts[1], nbnd));

    graph->mincut = mincut;
    graph->nbnd = nbnd;

    if (mincutorder == -1 || mincut == initcut)
      break;
  }

  PQueueFree(ctrl, &parts[0]);
  PQueueFree(ctrl, &parts[1]);

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}


