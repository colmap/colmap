/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * minitpart2.c
 *
 * This file contains code that performs the initial partition of the
 * coarsest graph
 *
 * Started 7/23/97
 * George
 *
 * $Id: minitpart2.c,v 1.1 1998/11/27 17:59:23 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function computes the initial bisection of the coarsest graph
**************************************************************************/
void MocInit2WayPartition2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int dbglvl;

  dbglvl = ctrl->dbglvl;
  IFSET(ctrl->dbglvl, DBG_REFINE, ctrl->dbglvl -= DBG_REFINE);
  IFSET(ctrl->dbglvl, DBG_MOVEINFO, ctrl->dbglvl -= DBG_MOVEINFO);

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->InitPartTmr));

  switch (ctrl->IType) {
    case IPART_GGPKL:
    case IPART_RANDOM:
      MocGrowBisection2(ctrl, graph, tpwgts, ubvec);
      break;
    case 3:
      MocGrowBisectionNew2(ctrl, graph, tpwgts, ubvec);
      break;
    default:
      graclus_errexit("Unknown initial partition type: %d\n", ctrl->IType);
  }

  IFSET(ctrl->dbglvl, DBG_IPART, printf("Initial Cut: %d\n", graph->mincut));
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->InitPartTmr));
  ctrl->dbglvl = dbglvl;

}




/*************************************************************************
* This function takes a graph and produces a bisection by using a region
* growing algorithm. The resulting partition is returned in
* graph->where
**************************************************************************/
void MocGrowBisection2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i, j, k, nvtxs, ncon, from, bestcut, mincut, nbfs;
  idxtype *bestwhere, *where;

  nvtxs = graph->nvtxs;

  MocAllocate2WayPartitionMemory(ctrl, graph);
  where = graph->where;

  bestwhere = idxmalloc(nvtxs, "BisectGraph: bestwhere");
  nbfs = 2*(nvtxs <= ctrl->CoarsenTo ? SMALLNIPARTS : LARGENIPARTS);
  bestcut = idxsum(graph->nedges, graph->adjwgt);

  for (; nbfs>0; nbfs--) {
    idxset(nvtxs, 1, where);
    where[RandomInRange(nvtxs)] = 0;

    MocCompute2WayPartitionParams(ctrl, graph);

    MocBalance2Way2(ctrl, graph, tpwgts, ubvec);

    MocFM_2WayEdgeRefine2(ctrl, graph, tpwgts, ubvec, 4);

    MocBalance2Way2(ctrl, graph, tpwgts, ubvec);
    MocFM_2WayEdgeRefine2(ctrl, graph, tpwgts, ubvec, 4);

    if (bestcut > graph->mincut) {
      bestcut = graph->mincut;
      idxcopy(nvtxs, where, bestwhere);
      if (bestcut == 0)
        break;
    }
  }

  graph->mincut = bestcut;
  idxcopy(nvtxs, bestwhere, where);

  GKfree((void **) &bestwhere, LTERM);
}






/*************************************************************************
* This function takes a graph and produces a bisection by using a region
* growing algorithm. The resulting partition is returned in
* graph->where
**************************************************************************/
void MocGrowBisectionNew2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i, j, k, nvtxs, ncon, from, bestcut, mincut, nbfs;
  idxtype *bestwhere, *where;

  nvtxs = graph->nvtxs;

  MocAllocate2WayPartitionMemory(ctrl, graph);
  where = graph->where;

  bestwhere = idxmalloc(nvtxs, "BisectGraph: bestwhere");
  nbfs = 2*(nvtxs <= ctrl->CoarsenTo ? SMALLNIPARTS : LARGENIPARTS);
  bestcut = idxsum(graph->nedges, graph->adjwgt);

  for (; nbfs>0; nbfs--) {
    idxset(nvtxs, 1, where);
    where[RandomInRange(nvtxs)] = 0;

    MocCompute2WayPartitionParams(ctrl, graph);

    MocInit2WayBalance2(ctrl, graph, tpwgts, ubvec);

    MocFM_2WayEdgeRefine2(ctrl, graph, tpwgts, ubvec, 4);

    if (bestcut > graph->mincut) {
      bestcut = graph->mincut;
      idxcopy(nvtxs, where, bestwhere);
      if (bestcut == 0)
        break;
    }
  }

  graph->mincut = bestcut;
  idxcopy(nvtxs, bestwhere, where);

  GKfree((void **) &bestwhere, LTERM);
}



/*************************************************************************
* This function balances two partitions by moving the highest gain
* (including negative gain) vertices to the other domain.
* It is used only when tha unbalance is due to non contigous
* subdomains. That is, the are no boundary vertices.
* It moves vertices from the domain that is overweight to the one that
* is underweight.
**************************************************************************/
void MocInit2WayBalance2(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i, ii, j, k, l, kwgt, nvtxs, nbnd, ncon, nswaps, from, to, pass, me, cnum, tmp, imin;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *perm, *qnum;
  float *nvwgt, *npwgts, minwgt;
  PQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  nvwgt = graph->nvwgt;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->id;
  ed = graph->ed;
  npwgts = graph->npwgts;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  moved = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);
  qnum = idxwspacemalloc(ctrl, nvtxs);

  /* This is called for initial partitioning so we know from where to pick nodes */
  from = 1;
  to = (from+1)%2;

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Parts: [");
    for (l=0; l<ncon; l++)
      printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
    printf("] T[%.3f %.3f], Nv-Nb[%5d, %5d]. ICut: %6d, LB: %.3f [B]\n", tpwgts[0], tpwgts[1], graph->nvtxs, graph->nbnd, graph->mincut, ComputeLoadImbalance(ncon, 2, npwgts, tpwgts));
  }

  for (i=0; i<ncon; i++) {
    PQueueInit(ctrl, &parts[i][0], nvtxs, PLUS_GAINSPAN+1);
    PQueueInit(ctrl, &parts[i][1], nvtxs, PLUS_GAINSPAN+1);
  }

  idxset(nvtxs, -1, moved);

  ASSERT(ComputeCut(graph, where) == graph->mincut);
  ASSERT(CheckBnd(graph));
  ASSERT(CheckGraph(graph));

  /* Compute the queues in which each vertex will be assigned to */
  for (i=0; i<nvtxs; i++)
    qnum[i] = samax(ncon, nvwgt+i*ncon);

  /* Insert the nodes of the proper partition in the appropriate priority queue */
  RandomPermute(nvtxs, perm, 1);
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];
    if (where[i] == from) {
      if (ed[i] > 0)
        PQueueInsert(&parts[qnum[i]][0], i, ed[i]-id[i]);
      else
        PQueueInsert(&parts[qnum[i]][1], i, ed[i]-id[i]);
    }
  }

/*
  for (i=0; i<ncon; i++)
    printf("Queue #%d has %d %d\n", i, parts[i][0].nnodes, parts[i][1].nnodes);
*/

  /* Determine the termination criterion */
  imin = 0;
  for (i=1; i<ncon; i++)
    imin = (ubvec[i] < ubvec[imin] ? i : imin);
  minwgt = .5/ubvec[imin];

  mincut = graph->mincut;
  nbnd = graph->nbnd;
  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    /* Exit as soon as the minimum weight crossed over */
    if (npwgts[to*ncon+imin] > minwgt)
      break;

    if ((cnum = SelectQueueOneWay2(ncon, npwgts+to*ncon, parts, ubvec)) == -1)
      break;

    if ((higain = PQueueGetMax(&parts[cnum][0])) == -1)
      higain = PQueueGetMax(&parts[cnum][1]);

    mincut -= (ed[higain]-id[higain]);
    saxpy(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);

    where[higain] = to;
    moved[higain] = nswaps;

    if (ctrl->dbglvl&DBG_MOVEINFO) {
      printf("Moved %6d from %d(%d). [%5d] %5d, NPwgts: ", higain, from, cnum, ed[higain]-id[higain], mincut);
      for (l=0; l<ncon; l++)
        printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
      printf(", LB: %.3f\n", ComputeLoadImbalance(ncon, 2, npwgts, tpwgts));
      if (ed[higain] == 0 && id[higain] > 0)
        printf("\t Pulled from the interior!\n");
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
      if (moved[k] == -1 && where[k] == from) {
        if (ed[k] > 0 && bndptr[k] == -1) {  /* It moves in boundary */
          PQueueDelete(&parts[qnum[k]][1], k, oldgain);
          PQueueInsert(&parts[qnum[k]][0], k, ed[k]-id[k]);
        }
        else { /* It must be in the boundary already */
          if (bndptr[k] == -1)
            printf("What you thought was wrong!\n");
          PQueueUpdate(&parts[qnum[k]][0], k, oldgain, ed[k]-id[k]);
        }
      }

      /* Update its boundary information */
      if (ed[k] == 0 && bndptr[k] != -1)
        BNDDelete(nbnd, bndind, bndptr, k);
      else if (ed[k] > 0 && bndptr[k] == -1)
        BNDInsert(nbnd, bndind, bndptr, k);
    }

    ASSERTP(ComputeCut(graph, where) == mincut, ("%d != %d\n", ComputeCut(graph, where), mincut));

  }

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("\tMincut: %6d, NBND: %6d, NPwgts: ", mincut, nbnd);
    for (l=0; l<ncon; l++)
      printf("(%.3f, %.3f) ", npwgts[l], npwgts[ncon+l]);
    printf(", LB: %.3f\n", ComputeLoadImbalance(ncon, 2, npwgts, tpwgts));
  }

  graph->mincut = mincut;
  graph->nbnd = nbnd;

  for (i=0; i<ncon; i++) {
    PQueueFree(ctrl, &parts[i][0]);
    PQueueFree(ctrl, &parts[i][1]);
  }

  ASSERT(ComputeCut(graph, where) == graph->mincut);
  ASSERT(CheckBnd(graph));

  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function selects the partition number and the queue from which
* we will move vertices out
**************************************************************************/
int SelectQueueOneWay2(int ncon, float *pto, PQueueType queues[MAXNCON][2], float *ubvec)
{
  int i, cnum=-1, imax, maxgain;
  float max=0.0;
  float twgt[MAXNCON];

  for (i=0; i<ncon; i++) {
    if (max < pto[i]) {
      imax = i;
      max = pto[i];
    }
  }
  for (i=0; i<ncon; i++)
    twgt[i] = (max/(ubvec[imax]*ubvec[i]))/pto[i];
  twgt[imax] = 0.0;

  max = 0.0;
  for (i=0; i<ncon; i++) {
    if (max < twgt[i] && (PQueueGetSize(&queues[i][0]) > 0 || PQueueGetSize(&queues[i][1]) > 0)) {
      max = twgt[i];
      cnum = i;
    }
  }
  if (max > 1)
    return cnum;

  /* optimize of cut */
  maxgain = -10000000;
  for (i=0; i<ncon; i++) {
    if (PQueueGetSize(&queues[i][0]) > 0 && PQueueGetKey(&queues[i][0]) > maxgain) {
      maxgain = PQueueGetKey(&queues[i][0]);
      cnum = i;
    }
  }

  return cnum;

}

