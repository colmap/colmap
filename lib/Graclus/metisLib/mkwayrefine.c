/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mkwayrefine.c
 *
 * This file contains the driving routines for multilevel k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: mkwayrefine.c,v 1.2 1998/11/27 18:16:19 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of refinement
**************************************************************************/
void MocRefineKWayHorizontal(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, int nparts,
       float *ubvec)
{

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Compute the parameters of the coarsest graph */
  MocComputeKWayPartitionParams(ctrl, graph, nparts);

  for (;;) {
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RefTmr));

    if (!MocIsHBalanced(graph->ncon, nparts, graph->npwgts, ubvec)) {
      MocComputeKWayBalanceBoundary(ctrl, graph, nparts);
      MCGreedy_KWayEdgeBalanceHorizontal(ctrl, graph, nparts, ubvec, 4);
      ComputeKWayBoundary(ctrl, graph, nparts);
    }

    MCRandom_KWayEdgeRefineHorizontal(ctrl, graph, nparts, ubvec, 10);

    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RefTmr));

    if (graph == orggraph)
      break;

    graph = graph->finer;
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    MocProjectKWayPartition(ctrl, graph, nparts);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  if (!MocIsHBalanced(graph->ncon, nparts, graph->npwgts, ubvec)) {
    MocComputeKWayBalanceBoundary(ctrl, graph, nparts);
    MCGreedy_KWayEdgeBalanceHorizontal(ctrl, graph, nparts, ubvec, 4);
    ComputeKWayBoundary(ctrl, graph, nparts);
    MCRandom_KWayEdgeRefineHorizontal(ctrl, graph, nparts, ubvec, 10);
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}




/*************************************************************************
* This function allocates memory for k-way edge refinement
**************************************************************************/
void MocAllocateKWayPartitionMemory(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int nvtxs, ncon, pad64;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;

  pad64 = (3*nvtxs)%2;

  graph->rdata = idxmalloc(3*nvtxs+(sizeof(RInfoType)/sizeof(idxtype))*nvtxs+pad64, "AllocateKWayPartitionMemory: rdata");
  graph->where          = graph->rdata;
  graph->bndptr         = graph->rdata + nvtxs;
  graph->bndind         = graph->rdata + 2*nvtxs;
  graph->rinfo          = (RInfoType *)(graph->rdata + 3*nvtxs + pad64);

  graph->npwgts         = fmalloc(ncon*nparts, "MocAllocateKWayPartitionMemory: npwgts");
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void MocComputeKWayPartitionParams(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, j, k, l, nvtxs, ncon, nbnd, mincut, me, other;
  idxtype *xadj, *adjncy, *adjwgt, *where, *bndind, *bndptr;
  RInfoType *rinfo, *myrinfo;
  EDegreeType *myedegrees;
  float *nvwgt, *npwgts;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  npwgts = sset(ncon*nparts, 0.0, graph->npwgts);
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);
  rinfo = graph->rinfo;


  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  ctrl->wspace.cdegree = 0;
  nbnd = mincut = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    saxpy(ncon, 1.0, nvwgt+i*ncon, 1, npwgts+me*ncon, 1);

    myrinfo = rinfo+i;
    myrinfo->id = myrinfo->ed = myrinfo->ndegrees = 0;
    myrinfo->edegrees = NULL;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me != where[adjncy[j]])
        myrinfo->ed += adjwgt[j];
    }
    myrinfo->id = graph->adjwgtsum[i] - myrinfo->ed;

    if (myrinfo->ed > 0)
      mincut += myrinfo->ed;

    if (myrinfo->ed-myrinfo->id >= 0)
      BNDInsert(nbnd, bndind, bndptr, i);

    /* Time to compute the particular external degrees */
    if (myrinfo->ed > 0) {
      myedegrees = myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += xadj[i+1]-xadj[i];

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = where[adjncy[j]];
        if (me != other) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (myedegrees[k].pid == other) {
              myedegrees[k].ed += adjwgt[j];
              break;
            }
          }
          if (k == myrinfo->ndegrees) {
            myedegrees[myrinfo->ndegrees].pid = other;
            myedegrees[myrinfo->ndegrees++].ed = adjwgt[j];
          }
        }
      }

      ASSERT(myrinfo->ndegrees <= xadj[i+1]-xadj[i]);
    }
  }

  graph->mincut = mincut/2;
  graph->nbnd = nbnd;

}



/*************************************************************************
* This function projects a partition, and at the same time computes the
* parameters for refinement.
**************************************************************************/
void MocProjectKWayPartition(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, j, k, nvtxs, nbnd, me, other, istart, iend, ndegrees;
  idxtype *xadj, *adjncy, *adjwgt, *adjwgtsum;
  idxtype *cmap, *where, *bndptr, *bndind;
  idxtype *cwhere;
  GraphType *cgraph;
  RInfoType *crinfo, *rinfo, *myrinfo;
  EDegreeType *myedegrees;
  idxtype *htable;

  cgraph = graph->coarser;
  cwhere = cgraph->where;
  crinfo = cgraph->rinfo;

  nvtxs = graph->nvtxs;
  cmap = graph->cmap;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  adjwgtsum = graph->adjwgtsum;

  MocAllocateKWayPartitionMemory(ctrl, graph, nparts);
  where = graph->where;
  rinfo = graph->rinfo;
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);

  /* Go through and project partition and compute id/ed for the nodes */
  for (i=0; i<nvtxs; i++) {
    k = cmap[i];
    where[i] = cwhere[k];
    cmap[i] = crinfo[k].ed;  /* For optimization */
  }

  htable = idxset(nparts, -1, idxwspacemalloc(ctrl, nparts));

  ctrl->wspace.cdegree = 0;
  for (nbnd=0, i=0; i<nvtxs; i++) {
    me = where[i];

    myrinfo = rinfo+i;
    myrinfo->id = myrinfo->ed = myrinfo->ndegrees = 0;
    myrinfo->edegrees = NULL;

    myrinfo->id = adjwgtsum[i];

    if (cmap[i] > 0) { /* If it is an interface node. Note cmap[i] = crinfo[cmap[i]].ed */
      istart = xadj[i];
      iend = xadj[i+1];

      myedegrees = myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += iend-istart;

      ndegrees = 0;
      for (j=istart; j<iend; j++) {
        other = where[adjncy[j]];
        if (me != other) {
          myrinfo->ed += adjwgt[j];
          if ((k = htable[other]) == -1) {
            htable[other] = ndegrees;
            myedegrees[ndegrees].pid = other;
            myedegrees[ndegrees++].ed = adjwgt[j];
          }
          else {
            myedegrees[k].ed += adjwgt[j];
          }
        }
      }
      myrinfo->id -= myrinfo->ed;

      /* Remove space for edegrees if it was interior */
      if (myrinfo->ed == 0) {
        myrinfo->edegrees = NULL;
        ctrl->wspace.cdegree -= iend-istart;
      }
      else {
        if (myrinfo->ed-myrinfo->id >= 0)
          BNDInsert(nbnd, bndind, bndptr, i);

        myrinfo->ndegrees = ndegrees;

        for (j=0; j<ndegrees; j++)
          htable[myedegrees[j].pid] = -1;
      }
    }
  }

  scopy(graph->ncon*nparts, cgraph->npwgts, graph->npwgts);
  graph->mincut = cgraph->mincut;
  graph->nbnd = nbnd;

  FreeGraph(graph->coarser);
  graph->coarser = NULL;

  idxwspacefree(ctrl, nparts);

  ASSERT(CheckBnd2(graph));

}



/*************************************************************************
* This function computes the boundary definition for balancing
**************************************************************************/
void MocComputeKWayBalanceBoundary(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, nvtxs, nbnd;
  idxtype *bndind, *bndptr;

  nvtxs = graph->nvtxs;
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);


  /* Compute the new boundary */
  nbnd = 0;
  for (i=0; i<nvtxs; i++) {
    if (graph->rinfo[i].ed > 0)
      BNDInsert(nbnd, bndind, bndptr, i);
  }

  graph->nbnd = nbnd;
}

