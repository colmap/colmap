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
 * $Id: refine.c,v 1.1 1998/11/27 17:59:29 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of refinement
**************************************************************************/
void Refine2Way(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, int *tpwgts, float ubfactor)
{

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Compute the parameters of the coarsest graph */
  Compute2WayPartitionParams(ctrl, graph);

  for (;;) {
    ASSERT(CheckBnd(graph));

    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RefTmr));
    switch (ctrl->RType) {
      case 1:
        Balance2Way(ctrl, graph, tpwgts, ubfactor);
        FM_2WayEdgeRefine(ctrl, graph, tpwgts, 8);
        break;
      default:
        graclus_errexit("Unknown refinement type: %d\n", ctrl->RType);
    }
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RefTmr));

    if (graph == orggraph)
      break;

    graph = graph->finer;
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    Project2WayPartition(ctrl, graph);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}


/*************************************************************************
* This function allocates memory for 2-way edge refinement
**************************************************************************/
void Allocate2WayPartitionMemory(CtrlType *ctrl, GraphType *graph)
{
  int nvtxs;

  nvtxs = graph->nvtxs;

  graph->rdata = idxmalloc(5*nvtxs+2, "Allocate2WayPartitionMemory: rdata");
  graph->pwgts 		= graph->rdata;
  graph->where		= graph->rdata + 2;
  graph->id		= graph->rdata + nvtxs + 2;
  graph->ed		= graph->rdata + 2*nvtxs + 2;
  graph->bndptr		= graph->rdata + 3*nvtxs + 2;
  graph->bndind		= graph->rdata + 4*nvtxs + 2;
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void Compute2WayPartitionParams(CtrlType *ctrl, GraphType *graph)
{
  int i, j, k, l, nvtxs, nbnd, mincut;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *pwgts;
  idxtype *id, *ed, *where;
  idxtype *bndptr, *bndind;
  int me, other;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = idxset(2, 0, graph->pwgts);
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
    pwgts[me] += vwgt[i];

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

  ASSERT(pwgts[0]+pwgts[1] == idxsum(nvtxs, vwgt));
}



/*************************************************************************
* This function projects a partition, and at the same time computes the
* parameters for refinement.
**************************************************************************/
void Project2WayPartition(CtrlType *ctrl, GraphType *graph)
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

  Allocate2WayPartitionMemory(ctrl, graph);

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
  idxcopy(2, cgraph->pwgts, graph->pwgts);

  FreeGraph(graph->coarser);
  graph->coarser = NULL;

}

