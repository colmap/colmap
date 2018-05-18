/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * debug.c
 *
 * This file contains code that performs self debuging
 *
 * Started 7/24/97
 * George
 *
 * $Id: debug.c,v 1.1 1998/11/27 17:59:13 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function computes the ratio assoc. given the graph and a where vector
**************************************************************************/
float ComputeRAsso(GraphType *graph, idxtype *where, int npart)
{
  int i, j, cm, nvtxs;
  idxtype *rasso, *clusterSize, *xadj, *adjncy;
  float result;
  idxtype * adjwgt;

  rasso = idxsmalloc(npart, 0, "ComputeNCut: ncut");
  clusterSize = idxsmalloc(npart, 0, "ComputeNCut: degree");
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  for (i=0; i<nvtxs; i++)
    clusterSize[where[i]] ++;

  if (graph->adjwgt == NULL) {
    for (i=0; i<nvtxs; i++) {
      cm = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (cm == where[adjncy[j]])
	  rasso[where[adjncy[j]]] ++;
    }
  }
  else {
    for (i=0; i<nvtxs; i++){
      cm = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (cm == where[adjncy[j]])
	  rasso[where[adjncy[j]]] += adjwgt[j];
    }
  }

  result =0;
  for (i=0; i<npart; i++){
    if (clusterSize[i] >0)
      result +=  rasso[i] *1.0/ clusterSize[i];
  }
  free(rasso);
  free(clusterSize);
  return result;
}

/*************************************************************************
* This function computes the normalized cut given the graph and a where vector
**************************************************************************/
float ComputeNCut(GraphType *graph, idxtype *where, int npart)
{
  int i, j, cm, nvtxs;
  idxtype *ncut, *degree, *xadj, *adjncy;
  float result;
  idxtype * adjwgt;

  ncut = idxsmalloc(npart, 0, "ComputeNCut: ncut");
  degree = idxsmalloc(npart, 0, "ComputeNCut: degree");
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  if (graph->adjwgt == NULL) {
    for (i=0; i<nvtxs; i++) {
      cm = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++){
       	degree[cm] ++;
        if (cm != where[adjncy[j]])
          ncut[cm] ++;
      }
    }
  }
  else {
    for (i=0; i<nvtxs; i++) {
      cm = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++){
	degree[cm] += adjwgt[j];
        if (cm != where[adjncy[j]])
          ncut[cm] += adjwgt[j];
      }
    }
  }
  int empty = 0;
  result =0;
  for (i=0; i<npart; i++){
    if (degree[i] == 0)
      empty++;
    if (degree[i] >0)
      result +=  ncut[i] *1.0/ degree[i];
  }
  //printf("Empty clusters: %d\n", empty);
  free(ncut);
  free(degree);
  return result+empty;
}


/*************************************************************************
* This function computes the cut given the graph and a where vector
**************************************************************************/
int ComputeCut(GraphType *graph, idxtype *where)
{
  int i, j, cut;

  if (graph->adjwgt == NULL) {
    for (cut=0, i=0; i<graph->nvtxs; i++) {
      for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++)
        if (where[i] != where[graph->adjncy[j]])
          cut++;
    }
  }
  else {
    for (cut=0, i=0; i<graph->nvtxs; i++) {
      for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++)
        if (where[i] != where[graph->adjncy[j]])
          cut += graph->adjwgt[j];
    }
  }

  return cut/2;
}


/*************************************************************************
* This function checks whether or not the boundary information is correct
**************************************************************************/
int CheckBnd(GraphType *graph)
{
  int i, j, nvtxs, nbnd;
  idxtype *xadj, *adjncy, *where, *bndptr, *bndind;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  for (nbnd=0, i=0; i<nvtxs; i++) {
    if (xadj[i+1]-xadj[i] == 0)
      nbnd++;   /* Islands are considered to be boundary vertices */

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (where[i] != where[adjncy[j]]) {
        nbnd++;
        ASSERT(bndptr[i] != -1);
        ASSERT(bndind[bndptr[i]] == i);
        break;
      }
    }
  }

  ASSERTP(nbnd == graph->nbnd, ("%d %d\n", nbnd, graph->nbnd));

  return 1;
}



/*************************************************************************
* This function checks whether or not the boundary information is correct
**************************************************************************/
int CheckBnd2(GraphType *graph)
{
  int i, j, nvtxs, nbnd, id, ed;
  idxtype *xadj, *adjncy, *where, *bndptr, *bndind;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  for (nbnd=0, i=0; i<nvtxs; i++) {
    id = ed = 0;
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (where[i] != where[adjncy[j]])
        ed += graph->adjwgt[j];
      else
        id += graph->adjwgt[j];
    }
    if (ed - id >= 0 && xadj[i] < xadj[i+1]) {
      nbnd++;
      ASSERTP(bndptr[i] != -1, ("%d %d %d\n", i, id, ed));
      ASSERT(bndind[bndptr[i]] == i);
    }
  }

  ASSERTP(nbnd == graph->nbnd, ("%d %d\n", nbnd, graph->nbnd));

  return 1;
}

/*************************************************************************
* This function checks whether or not the boundary information is correct
**************************************************************************/
int CheckNodeBnd(GraphType *graph, int onbnd)
{
  int i, j, nvtxs, nbnd;
  idxtype *xadj, *adjncy, *where, *bndptr, *bndind;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  for (nbnd=0, i=0; i<nvtxs; i++) {
    if (where[i] == 2)
      nbnd++;
  }

  ASSERTP(nbnd == onbnd, ("%d %d\n", nbnd, onbnd));

  for (i=0; i<nvtxs; i++) {
    if (where[i] != 2) {
      ASSERTP(bndptr[i] == -1, ("%d %d\n", i, bndptr[i]));
    }
    else {
      ASSERTP(bndptr[i] != -1, ("%d %d\n", i, bndptr[i]));
    }
  }

  return 1;
}



/*************************************************************************
* This function checks whether or not the rinfo of a vertex is consistent
**************************************************************************/
int CheckRInfo(RInfoType *rinfo)
{
  int i, j;

  for (i=0; i<rinfo->ndegrees; i++) {
    for (j=i+1; j<rinfo->ndegrees; j++)
      ASSERTP(rinfo->edegrees[i].pid != rinfo->edegrees[j].pid, ("%d %d %d %d\n", i, j, rinfo->edegrees[i].pid, rinfo->edegrees[j].pid));
  }

  return 1;
}



/*************************************************************************
* This function checks the correctness of the NodeFM data structures
**************************************************************************/
int CheckNodePartitionParams(GraphType *graph)
{
  int i, j, k, l, nvtxs, me, other;
  idxtype *xadj, *adjncy, *adjwgt, *vwgt, *where;
  idxtype edegrees[2], pwgts[3];

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;

  /*------------------------------------------------------------
  / Compute now the separator external degrees
  /------------------------------------------------------------*/
  pwgts[0] = pwgts[1] = pwgts[2] = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    pwgts[me] += vwgt[i];

    if (me == 2) { /* If it is on the separator do some computations */
      edegrees[0] = edegrees[1] = 0;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = where[adjncy[j]];
        if (other != 2)
          edegrees[other] += vwgt[adjncy[j]];
      }
      if (edegrees[0] != graph->nrinfo[i].edegrees[0] || edegrees[1] != graph->nrinfo[i].edegrees[1]) {
        printf("Something wrong with edegrees: %d %d %d %d %d\n", i, edegrees[0], edegrees[1], graph->nrinfo[i].edegrees[0], graph->nrinfo[i].edegrees[1]);
        return 0;
      }
    }
  }

  if (pwgts[0] != graph->pwgts[0] || pwgts[1] != graph->pwgts[1] || pwgts[2] != graph->pwgts[2])
    printf("Something wrong with part-weights: %d %d %d %d %d %d\n", pwgts[0], pwgts[1], pwgts[2], graph->pwgts[0], graph->pwgts[1], graph->pwgts[2]);

  return 1;
}


/*************************************************************************
* This function checks if the separator is indeed a separator
**************************************************************************/
int IsSeparable(GraphType *graph)
{
  int i, j, nvtxs, other;
  idxtype *xadj, *adjncy, *where;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;

  for (i=0; i<nvtxs; i++) {
    if (where[i] == 2)
      continue;
    other = (where[i]+1)%2;
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      ASSERTP(where[adjncy[j]] != other, ("%d %d %d %d %d %d\n", i, where[i], adjncy[j], where[adjncy[j]], xadj[i+1]-xadj[i], xadj[adjncy[j]+1]-xadj[adjncy[j]]));
    }
  }

  return 1;
}


