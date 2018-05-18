/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * kwayvolrefine.c
 *
 * This file contains the driving routines for multilevel k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: kwayvolrefine.c,v 1.2 1998/11/30 16:13:57 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point of refinement
**************************************************************************/
void RefineVolKWay(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, int nparts,
                   float *tpwgts, float ubfactor)
{
  int i, nlevels;
  GraphType *ptr;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Take care any non-contiguity */
  if (ctrl->RType == RTYPE_KWAYRANDOM_MCONN) {
    ComputeVolKWayPartitionParams(ctrl, graph, nparts);
    EliminateVolComponents(ctrl, graph, nparts, tpwgts, 1.25);
    EliminateVolSubDomainEdges(ctrl, graph, nparts, tpwgts);
    EliminateVolComponents(ctrl, graph, nparts, tpwgts, 1.25);
  }


  /* Determine how many levels are there */
  for (ptr=graph, nlevels=0; ptr!=orggraph; ptr=ptr->finer, nlevels++);

  /* Compute the parameters of the coarsest graph */
  ComputeVolKWayPartitionParams(ctrl, graph, nparts);

  for (i=0; ;i++) {
    /*PrintSubDomainGraph(graph, nparts, graph->where);*/
    MALLOC_CHECK(NULL);
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RefTmr));

    if (2*i >= nlevels && !IsBalanced(graph->pwgts, nparts, tpwgts, 1.04*ubfactor)) {
      ComputeVolKWayBalanceBoundary(ctrl, graph, nparts);
      switch (ctrl->RType) {
        case RTYPE_KWAYRANDOM:
          Greedy_KWayVolBalance(ctrl, graph, nparts, tpwgts, ubfactor, 1);
          break;
        case RTYPE_KWAYRANDOM_MCONN:
          Greedy_KWayVolBalanceMConn(ctrl, graph, nparts, tpwgts, ubfactor, 1);
          break;
      }
      ComputeVolKWayBoundary(ctrl, graph, nparts);
    }

    switch (ctrl->RType) {
      case RTYPE_KWAYRANDOM:
        Random_KWayVolRefine(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
        break;
      case RTYPE_KWAYRANDOM_MCONN:
        Random_KWayVolRefineMConn(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
        break;
    }
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RefTmr));

    if (graph == orggraph)
      break;

    GKfree((void **) &graph->gdata, LTERM);  /* Deallocate the graph related arrays */

    graph = graph->finer;

    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    ProjectVolKWayPartition(ctrl, graph, nparts);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  if (!IsBalanced(graph->pwgts, nparts, tpwgts, ubfactor)) {
    ComputeVolKWayBalanceBoundary(ctrl, graph, nparts);
    switch (ctrl->RType) {
      case RTYPE_KWAYRANDOM:
        Greedy_KWayVolBalance(ctrl, graph, nparts, tpwgts, ubfactor, 8);
        Random_KWayVolRefine(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
        break;
      case RTYPE_KWAYRANDOM_MCONN:
        Greedy_KWayVolBalanceMConn(ctrl, graph, nparts, tpwgts, ubfactor, 8);
        Random_KWayVolRefineMConn(ctrl, graph, nparts, tpwgts, ubfactor, 10, 0);
        break;
    }
  }

  EliminateVolComponents(ctrl, graph, nparts, tpwgts, ubfactor);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}



/*************************************************************************
* This function allocates memory for k-way edge refinement
**************************************************************************/
void AllocateVolKWayPartitionMemory(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int nvtxs, pad64;

  nvtxs = graph->nvtxs;

  pad64 = (3*nvtxs+nparts)%2;

  graph->rdata = idxmalloc(3*nvtxs+nparts+(sizeof(VRInfoType)/sizeof(idxtype))*nvtxs+pad64, "AllocateVolKWayPartitionMemory: rdata");
  graph->pwgts          = graph->rdata;
  graph->where          = graph->rdata + nparts;
  graph->bndptr         = graph->rdata + nvtxs + nparts;
  graph->bndind         = graph->rdata + 2*nvtxs + nparts;
  graph->vrinfo          = (VRInfoType *)(graph->rdata + 3*nvtxs+nparts + pad64);

}



/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void ComputeVolKWayPartitionParams(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, ii, j, k, kk, l, nvtxs, nbnd, mincut, minvol, me, other, pid;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *pwgts, *where;
  VRInfoType *rinfo, *myrinfo, *orinfo;
  VEDegreeType *myedegrees, *oedegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = idxset(nparts, 0, graph->pwgts);
  rinfo = graph->vrinfo;


  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  ctrl->wspace.cdegree = 0;
  mincut = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    pwgts[me] += vwgt[i];

    myrinfo = rinfo+i;
    myrinfo->id = myrinfo->ed = myrinfo->nid = myrinfo->ndegrees = 0;
    myrinfo->edegrees = NULL;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me == where[adjncy[j]]) {
        myrinfo->id += adjwgt[j];
        myrinfo->nid++;
      }
    }
    myrinfo->ed = graph->adjwgtsum[i] - myrinfo->id;

    mincut += myrinfo->ed;

    /* Time to compute the particular external degrees */
    if (myrinfo->ed > 0) {
      myedegrees = myrinfo->edegrees = ctrl->wspace.vedegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += xadj[i+1]-xadj[i];

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = where[adjncy[j]];
        if (me != other) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (myedegrees[k].pid == other) {
              myedegrees[k].ed += adjwgt[j];
              myedegrees[k].ned++;
              break;
            }
          }
          if (k == myrinfo->ndegrees) {
            myedegrees[myrinfo->ndegrees].gv = 0;
            myedegrees[myrinfo->ndegrees].pid = other;
            myedegrees[myrinfo->ndegrees].ed = adjwgt[j];
            myedegrees[myrinfo->ndegrees++].ned = 1;
          }
        }
      }

      ASSERT(myrinfo->ndegrees <= xadj[i+1]-xadj[i]);
    }
  }
  graph->mincut = mincut/2;


  ComputeKWayVolGains(ctrl, graph, nparts);

}



/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void ComputeKWayVolGains(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, ii, j, k, kk, l, nvtxs, me, other, pid, myndegrees;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *where, *bndind, *bndptr, *ophtable;
  VRInfoType *rinfo, *myrinfo, *orinfo;
  VEDegreeType *myedegrees, *oedegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  bndind = graph->bndind;
  bndptr = idxset(nvtxs, -1, graph->bndptr);
  rinfo = graph->vrinfo;

  ophtable = idxset(nparts, -1, idxwspacemalloc(ctrl, nparts));

  /*------------------------------------------------------------
  / Compute now the iv/ev degrees
  /------------------------------------------------------------*/
  graph->minvol = graph->nbnd = 0;
  for (i=0; i<nvtxs; i++) {
    myrinfo = rinfo+i;
    myrinfo->gv = -MAXIDX;

    if (myrinfo->ndegrees > 0) {
      me = where[i];
      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      graph->minvol += myndegrees*vsize[i];

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        ii = adjncy[j];
        other = where[ii];
        orinfo = rinfo+ii;
        oedegrees = orinfo->edegrees;

        for (k=0; k<orinfo->ndegrees; k++)
          ophtable[oedegrees[k].pid] = k;
        ophtable[other] = 1;  /* this is to simplify coding */

        if (me == other) {
          /* Find which domains 'i' is connected and 'ii' is not and update their gain */
          for (k=0; k<myndegrees; k++) {
            if (ophtable[myedegrees[k].pid] == -1)
              myedegrees[k].gv -= vsize[ii];
          }
        }
        else {
          ASSERT(ophtable[me] != -1);

          if (oedegrees[ophtable[me]].ned == 1) { /* I'm the only connection of 'ii' in 'me' */
            /* Increase the gains for all the common domains between 'i' and 'ii' */
            for (k=0; k<myndegrees; k++) {
              if (ophtable[myedegrees[k].pid] != -1)
                myedegrees[k].gv += vsize[ii];
            }
          }
          else {
            /* Find which domains 'i' is connected and 'ii' is not and update their gain */
            for (k=0; k<myndegrees; k++) {
              if (ophtable[myedegrees[k].pid] == -1)
                myedegrees[k].gv -= vsize[ii];
            }
          }
        }

        for (kk=0; kk<orinfo->ndegrees; kk++)
          ophtable[oedegrees[kk].pid] = -1;
        ophtable[other] = -1;
      }

      /* Compute the max vgain */
      for (k=0; k<myndegrees; k++) {
        if (myedegrees[k].gv > myrinfo->gv)
          myrinfo->gv = myedegrees[k].gv;
      }
    }

    if (myrinfo->ed > 0 && myrinfo->id == 0)
      myrinfo->gv += vsize[i];

    if (myrinfo->gv >= 0 || myrinfo->ed-myrinfo->id >= 0)
      BNDInsert(graph->nbnd, bndind, bndptr, i);
  }

  idxwspacefree(ctrl, nparts);

}



/*************************************************************************
* This function projects a partition, and at the same time computes the
* parameters for refinement.
**************************************************************************/
void ProjectVolKWayPartition(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, j, k, nvtxs, me, other, istart, iend, ndegrees;
  idxtype *xadj, *adjncy, *adjwgt, *adjwgtsum;
  idxtype *cmap, *where;
  idxtype *cwhere;
  GraphType *cgraph;
  VRInfoType *crinfo, *rinfo, *myrinfo;
  VEDegreeType *myedegrees;
  idxtype *htable;

  cgraph = graph->coarser;
  cwhere = cgraph->where;
  crinfo = cgraph->vrinfo;

  nvtxs = graph->nvtxs;
  cmap = graph->cmap;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  adjwgtsum = graph->adjwgtsum;

  AllocateVolKWayPartitionMemory(ctrl, graph, nparts);
  where = graph->where;
  rinfo = graph->vrinfo;

  /* Go through and project partition and compute id/ed for the nodes */
  for (i=0; i<nvtxs; i++) {
    k = cmap[i];
    where[i] = cwhere[k];
    cmap[i] = crinfo[k].ed;  /* For optimization */
  }

  htable = idxset(nparts, -1, idxwspacemalloc(ctrl, nparts));

  ctrl->wspace.cdegree = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];

    myrinfo = rinfo+i;
    myrinfo->id = myrinfo->ed = myrinfo->nid = myrinfo->ndegrees = 0;
    myrinfo->edegrees = NULL;

    myrinfo->id = adjwgtsum[i];
    myrinfo->nid = xadj[i+1]-xadj[i];

    if (cmap[i] > 0) { /* If it is an interface node. Note cmap[i] = crinfo[cmap[i]].ed */
      istart = xadj[i];
      iend = xadj[i+1];

      myedegrees = myrinfo->edegrees = ctrl->wspace.vedegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += iend-istart;

      ndegrees = 0;
      for (j=istart; j<iend; j++) {
        other = where[adjncy[j]];
        if (me != other) {
          myrinfo->ed += adjwgt[j];
          myrinfo->nid--;
          if ((k = htable[other]) == -1) {
            htable[other] = ndegrees;
            myedegrees[ndegrees].gv = 0;
            myedegrees[ndegrees].pid = other;
            myedegrees[ndegrees].ed = adjwgt[j];
            myedegrees[ndegrees++].ned = 1;
          }
          else {
            myedegrees[k].ed += adjwgt[j];
            myedegrees[k].ned++;
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
        myrinfo->ndegrees = ndegrees;

        for (j=0; j<ndegrees; j++)
          htable[myedegrees[j].pid] = -1;
      }
    }
  }

  ComputeKWayVolGains(ctrl, graph, nparts);

  idxcopy(nparts, cgraph->pwgts, graph->pwgts);
  graph->mincut = cgraph->mincut;

  FreeGraph(graph->coarser);
  graph->coarser = NULL;

  idxwspacefree(ctrl, nparts);

}



/*************************************************************************
* This function computes the boundary definition for balancing
**************************************************************************/
void ComputeVolKWayBoundary(CtrlType *ctrl, GraphType *graph, int nparts)
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
    if (graph->vrinfo[i].gv >=0 || graph->vrinfo[i].ed-graph->vrinfo[i].id >= 0)
      BNDInsert(nbnd, bndind, bndptr, i);
  }

  graph->nbnd = nbnd;
}

/*************************************************************************
* This function computes the boundary definition for balancing
**************************************************************************/
void ComputeVolKWayBalanceBoundary(CtrlType *ctrl, GraphType *graph, int nparts)
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
    if (graph->vrinfo[i].ed > 0)
      BNDInsert(nbnd, bndind, bndptr, i);
  }

  graph->nbnd = nbnd;
}

