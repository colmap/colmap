/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * refine.c
 *
 * This file contains the driving routines for multilevel refinement
 *
 * Started 7/24/97
 * George
 *
 * $Id: mrefine.c,v 1.2 1998/11/27 18:25:09 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of refinement
**************************************************************************/
void MocRefine2Way(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, float *tpwgts, float ubfactor)
{
  int i;
  float tubvec[MAXNCON];

  for (i=0; i<graph->ncon; i++)
    tubvec[i] = 1.0;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Compute the parameters of the coarsest graph */
  MocCompute2WayPartitionParams(ctrl, graph);

  for (;;) {
    ASSERT(CheckBnd(graph));

    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RefTmr));
    switch (ctrl->RType) {
      case RTYPE_FM:
        MocBalance2Way(ctrl, graph, tpwgts, 1.03);
        MocFM_2WayEdgeRefine(ctrl, graph, tpwgts, 8);
        break;
      case 2:
        MocBalance2Way(ctrl, graph, tpwgts, 1.03);
        MocFM_2WayEdgeRefine2(ctrl, graph, tpwgts, tubvec, 8);
        break;
      default:
        graclus_errexit("Unknown refinement type: %d\n", ctrl->RType);
    }
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RefTmr));

    if (graph == orggraph)
      break;

    graph = graph->finer;
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    MocProject2WayPartition(ctrl, graph);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  MocBalance2Way(ctrl, graph, tpwgts, 1.01);
  MocFM_2WayEdgeRefine(ctrl, graph, tpwgts, 8);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}


/*************************************************************************
* This function allocates memory for 2-way edge refinement
**************************************************************************/
void MocAllocate2WayPartitionMemory(CtrlType *ctrl, GraphType *graph)
{
  int nvtxs, ncon;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;

  graph->rdata = idxmalloc(5*nvtxs, "Allocate2WayPartitionMemory: rdata");
  graph->where		= graph->rdata;
  graph->id		= graph->rdata + nvtxs;
  graph->ed		= graph->rdata + 2*nvtxs;
  graph->bndptr		= graph->rdata + 3*nvtxs;
  graph->bndind		= graph->rdata + 4*nvtxs;

  graph->npwgts 	= fmalloc(2*ncon, "npwgts");
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void MocCompute2WayPartitionParams(CtrlType *ctrl, GraphType *graph)
{
  int i, j, k, l, nvtxs, ncon, nbnd, mincut;
  idxtype *xadj, *adjncy, *adjwgt;
  float *nvwgt, *npwgts;
  idxtype *id, *ed, *where;
  idxtype *bndptr, *bndind;
  int me, other;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  npwgts = sset(2*ncon, 0.0, graph->npwgts);
  id = idxset(nvtxs, 0, graph->id);
  ed = idxset(nvtxs, 0, graph->ed);
  bndptr = idxset(nvtxs, -1, graph->bndptr);
  bndind = graph->bndind;


  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  nbnd = mincut = 0;
  for (i=0; i<nvtxs; i++) {
    ASSERT(where[i] >= 0 && where[i] <= 1);
    me = where[i];
    saxpy(ncon, 1.0, nvwgt+i*ncon, 1, npwgts+me*ncon, 1);

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me == where[adjncy[j]])
        id[i] += adjwgt[j];
      else
        ed[i] += adjwgt[j];
    }

    if (ed[i] > 0 || xadj[i] == xadj[i+1]) {
      mincut += ed[i];
      bndptr[i] = nbnd;
      bndind[nbnd++] = i;
    }
  }

  graph->mincut = mincut/2;
  graph->nbnd = nbnd;

}



/*************************************************************************
* This function projects a partition, and at the same time computes the
* parameters for refinement.
**************************************************************************/
void MocProject2WayPartition(CtrlType *ctrl, GraphType *graph)
{
  int i, j, k, nvtxs, nbnd, me;
  idxtype *xadj, *adjncy, *adjwgt, *adjwgtsum;
  idxtype *cmap, *where, *id, *ed, *bndptr, *bndind;
  idxtype *cwhere, *cid, *ced, *cbndptr;
  GraphType *cgraph;

  cgraph = graph->coarser;
  cwhere = cgraph->where;
  cid = cgraph->id;
  ced = cgraph->ed;
  cbndptr = cgraph->bndptr;

  nvtxs = graph->nvtxs;
  cmap = graph->cmap;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  adjwgtsum = graph->adjwgtsum;

  MocAllocate2WayPartitionMemory(ctrl, graph);

  where = graph->where;
  id = idxset(nvtxs, 0, graph->id);
  ed = idxset(nvtxs, 0, graph->ed);
  bndptr = idxset(nvtxs, -1, graph->bndptr);
  bndind = graph->bndind;


  /* Go through and project partition and compute id/ed for the nodes */
  for (i=0; i<nvtxs; i++) {
    k = cmap[i];
    where[i] = cwhere[k];
    cmap[i] = cbndptr[k];
  }

  for (nbnd=0, i=0; i<nvtxs; i++) {
    me = where[i];

    id[i] = adjwgtsum[i];

    if (xadj[i] == xadj[i+1]) {
      bndptr[i] = nbnd;
      bndind[nbnd++] = i;
    }
    else {
      if (cmap[i] != -1) { /* If it is an interface node. Note that cmap[i] = cbndptr[cmap[i]] */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          if (me != where[adjncy[j]])
            ed[i] += adjwgt[j];
        }
        id[i] -= ed[i];

        if (ed[i] > 0 || xadj[i] == xadj[i+1]) {
          bndptr[i] = nbnd;
          bndind[nbnd++] = i;
        }
      }
    }
  }

  graph->mincut = cgraph->mincut;
  graph->nbnd = nbnd;
  scopy(2*graph->ncon, cgraph->npwgts, graph->npwgts);

  FreeGraph(graph->coarser);
  graph->coarser = NULL;

}

