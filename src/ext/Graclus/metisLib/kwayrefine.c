/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * kwayrefine.c
 *
 * This file contains the driving routines for multilevel k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: kwayrefine.c,v 1.1 1998/11/27 17:59:17 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of refinement
**************************************************************************/
void RefineKWay(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, int nparts, float *tpwgts, float ubfactor)
{
  int i, nlevels, mustfree=0;
  GraphType *ptr;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Compute the parameters of the coarsest graph */
  ComputeKWayPartitionParams(ctrl, graph, nparts);

  /* Take care any non-contiguity */
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->AuxTmr1));
  if (ctrl->RType == RTYPE_KWAYRANDOM_MCONN) {
    EliminateComponents(ctrl, graph, nparts, tpwgts, 1.25);
    EliminateSubDomainEdges(ctrl, graph, nparts, tpwgts);
    EliminateComponents(ctrl, graph, nparts, tpwgts, 1.25);
  }
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->AuxTmr1));

  /* Determine how many levels are there */
  for (ptr=graph, nlevels=0; ptr!=orggraph; ptr=ptr->finer, nlevels++);

  for (i=0; ;i++) {
    /* PrintSubDomainGraph(graph, nparts, graph->where); */
    if (ctrl->RType == RTYPE_KWAYRANDOM_MCONN && (i == nlevels/2 || i == nlevels/2+1))
      EliminateSubDomainEdges(ctrl, graph, nparts, tpwgts);

    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RefTmr));

    if (2*i >= nlevels && !IsBalanced(graph->pwgts, nparts, tpwgts, 1.04*ubfactor)) {
      ComputeKWayBalanceBoundary(ctrl, graph, nparts);
      if (ctrl->RType == RTYPE_KWAYRANDOM_MCONN)
        Greedy_KWayEdgeBalanceMConn(ctrl, graph, nparts, tpwgts, ubfactor, 1);
      else
        Greedy_KWayEdgeBalance(ctrl, graph, nparts, tpwgts, ubfactor, 1);
      ComputeKWayBoundary(ctrl, graph, nparts);
    }

    switch (ctrl->RType) {
      case RTYPE_KWAYRANDOM:
        Random_KWayEdgeRefine(ctrl, graph, nparts, tpwgts, ubfactor, 10, 1);
        break;
      case RTYPE_KWAYGREEDY:
        Greedy_KWayEdgeRefine(ctrl, graph, nparts, tpwgts, ubfactor, 10);
        break;
      case RTYPE_KWAYRANDOM_MCONN:
        Random_KWayEdgeRefineMConn(ctrl, graph, nparts, tpwgts, ubfactor, 10, 1);
        break;
    }
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RefTmr));

    if (graph == orggraph)
      break;

    GKfree((void**) &graph->gdata, LTERM);  /* Deallocate the graph related arrays */

    graph = graph->finer;

    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    if (graph->vwgt == NULL) {
      graph->vwgt = idxsmalloc(graph->nvtxs, 1, "RefineKWay: graph->vwgt");
      graph->adjwgt = idxsmalloc(graph->nedges, 1, "RefineKWay: graph->adjwgt");
      mustfree = 1;
    }
    ProjectKWayPartition(ctrl, graph, nparts);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  if (!IsBalanced(graph->pwgts, nparts, tpwgts, ubfactor)) {
    ComputeKWayBalanceBoundary(ctrl, graph, nparts);
    if (ctrl->RType == RTYPE_KWAYRANDOM_MCONN) {
      Greedy_KWayEdgeBalanceMConn(ctrl, graph, nparts, tpwgts, ubfactor, 8);
      Random_KWayEdgeRefineMConn(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
    }
    else {
      Greedy_KWayEdgeBalance(ctrl, graph, nparts, tpwgts, ubfactor, 8);
      Random_KWayEdgeRefine(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
    }
  }

  /* Take care any trivial non-contiguity */
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->AuxTmr2));
  EliminateComponents(ctrl, graph, nparts, tpwgts, ubfactor);
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->AuxTmr2));

  if (mustfree)
    GKfree((void**) &graph->vwgt, (void**) &graph->adjwgt, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}


/*************************************************************************
* This function allocates memory for k-way edge refinement
**************************************************************************/
void AllocateKWayPartitionMemory(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int nvtxs, pad64;

  nvtxs = graph->nvtxs;

  pad64 = (3*nvtxs+nparts)%2;

  graph->rdata = idxmalloc(3*nvtxs+nparts+(sizeof(RInfoType)/sizeof(idxtype))*nvtxs+pad64, "AllocateKWayPartitionMemory: rdata");
  graph->pwgts          = graph->rdata;
  graph->where          = graph->rdata + nparts;
  graph->bndptr         = graph->rdata + nvtxs + nparts;
  graph->bndind         = graph->rdata + 2*nvtxs + nparts;
  graph->rinfo          = (RInfoType *)(graph->rdata + 3*nvtxs+nparts + pad64);

/*
  if (ctrl->wspace.edegrees != NULL)
    free(ctrl->wspace.edegrees);
  ctrl->wspace.edegrees = (EDegreeType *)GKmalloc(graph->nedges*sizeof(EDegreeType), "AllocateKWayPartitionMemory: edegrees");
*/
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void ComputeKWayPartitionParams(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, j, k, l, nvtxs, nbnd, mincut, me, other;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *pwgts, *where, *bndind, *bndptr;
  RInfoType *rinfo, *myrinfo;
  EDegreeType *myedegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = idxset(nparts, 0, graph->pwgts);
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
    pwgts[me] += vwgt[i];

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
void ProjectKWayPartition(CtrlType *ctrl, GraphType *graph, int nparts)
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

  AllocateKWayPartitionMemory(ctrl, graph, nparts);
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

  idxcopy(nparts, cgraph->pwgts, graph->pwgts);
  graph->mincut = cgraph->mincut;
  graph->nbnd = nbnd;

  FreeGraph(graph->coarser);
  graph->coarser = NULL;

  idxwspacefree(ctrl, nparts);

  ASSERT(CheckBnd2(graph));

}



/*************************************************************************
* This function checks if the partition weights are within the balance
* contraints
**************************************************************************/
int IsBalanced(idxtype *pwgts, int nparts, float *tpwgts, float ubfactor)
{
  int i, j, tvwgt;

  tvwgt = idxsum(nparts, pwgts);
  for (i=0; i<nparts; i++) {
    if (pwgts[i] > tpwgts[i]*tvwgt*(ubfactor+0.005))
      return 0;
  }

  return 1;
}


/*************************************************************************
* This function computes the boundary definition for balancing
**************************************************************************/
void ComputeKWayBoundary(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, nvtxs, nbnd;
  idxtype *bndind, *bndptr;

  nvtxs = graph->nvtxs;
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);


  /*------------------------------------------------------------
  / Compute the new boundary
  /------------------------------------------------------------*/
  nbnd = 0;
  for (i=0; i<nvtxs; i++) {
    if (graph->rinfo[i].ed-graph->rinfo[i].id >= 0)
      BNDInsert(nbnd, bndind, bndptr, i);
  }

  graph->nbnd = nbnd;
}

/*************************************************************************
* This function computes the boundary definition for balancing
**************************************************************************/
void ComputeKWayBalanceBoundary(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, nvtxs, nbnd;
  idxtype *bndind, *bndptr;

  nvtxs = graph->nvtxs;
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);


  /*------------------------------------------------------------
  / Compute the new boundary
  /------------------------------------------------------------*/
  nbnd = 0;
  for (i=0; i<nvtxs; i++) {
    if (graph->rinfo[i].ed > 0)
      BNDInsert(nbnd, bndind, bndptr, i);
  }

  graph->nbnd = nbnd;
}

