/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * graph.c
 *
 * This file contains functions that deal with setting up the graphs
 * for METIS.
 *
 * Started 7/25/97
 * George
 *
 * $Id: graph.c,v 1.1 1998/11/27 17:59:15 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function sets up the graph from the user input
**************************************************************************/
void SetUpGraph(GraphType *graph, int OpType, int nvtxs, int ncon,
       idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int wgtflag)
{
  int i, j, k, sum, gsize;
  float *nvwgt;
  idxtype tvwgt[MAXNCON];

  if (OpType == OP_KMETIS && ncon == 1 && (wgtflag&2) == 0 && (wgtflag&1) == 0) {
    SetUpGraphKway(graph, nvtxs, xadj, adjncy);
    return;
  }

  InitGraph(graph);

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->ncon = ncon;
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  if (ncon == 1) { /* We are in the non mC mode */
    gsize = 0;
    if ((wgtflag&2) == 0)
      gsize += nvtxs;
    if ((wgtflag&1) == 0)
      gsize += graph->nedges;

    gsize += 2*nvtxs;

    graph->gdata = idxmalloc(gsize, "SetUpGraph: gdata");

    /* Create the vertex/edge weight vectors if they are not supplied */
    gsize = 0;
    if ((wgtflag&2) == 0) {
      vwgt = graph->vwgt = idxset(nvtxs, 1, graph->gdata);
      gsize += nvtxs;
    }
    else
      graph->vwgt = vwgt;

    if ((wgtflag&1) == 0) {
      adjwgt = graph->adjwgt = idxset(graph->nedges, 1, graph->gdata+gsize);
      gsize += graph->nedges;
    }
    else
      graph->adjwgt = adjwgt;


    /* Compute the initial values of the adjwgtsum */
    graph->adjwgtsum = graph->gdata + gsize;
    gsize += nvtxs;

    for (i=0; i<nvtxs; i++) {
      sum = 0;
      for (j=xadj[i]; j<xadj[i+1]; j++)
        sum += adjwgt[j];
      graph->adjwgtsum[i] = sum;
    }

    graph->cmap = graph->gdata + gsize;
    gsize += nvtxs;

  }
  else {  /* Set up the graph in MOC mode */
    gsize = 0;
    if ((wgtflag&1) == 0)
      gsize += graph->nedges;

    gsize += 2*nvtxs;

    graph->gdata = idxmalloc(gsize, "SetUpGraph: gdata");
    gsize = 0;

    for (i=0; i<ncon; i++)
      tvwgt[i] = idxsum_strd(nvtxs, vwgt+i, ncon);

    nvwgt = graph->nvwgt = fmalloc(ncon*nvtxs, "SetUpGraph: nvwgt");

    for (i=0; i<nvtxs; i++) {
      for (j=0; j<ncon; j++)
        nvwgt[i*ncon+j] = (1.0*vwgt[i*ncon+j])/(1.0*tvwgt[j]);
    }


    /* Create the edge weight vectors if they are not supplied */
    if ((wgtflag&1) == 0) {
      adjwgt = graph->adjwgt = idxset(graph->nedges, 1, graph->gdata+gsize);
      gsize += graph->nedges;
    }
    else
      graph->adjwgt = adjwgt;

    /* Compute the initial values of the adjwgtsum */
    graph->adjwgtsum = graph->gdata + gsize;
    gsize += nvtxs;

    for (i=0; i<nvtxs; i++) {
      sum = 0;
      for (j=xadj[i]; j<xadj[i+1]; j++)
        sum += adjwgt[j];
      graph->adjwgtsum[i] = sum;
    }

    graph->cmap = graph->gdata + gsize;
    gsize += nvtxs;

  }

  if (OpType != OP_KMETIS && OpType != OP_KVMETIS) {
    graph->label = idxmalloc(nvtxs, "SetUpGraph: label");

    for (i=0; i<nvtxs; i++)
      graph->label[i] = i;
  }

}


/*************************************************************************
* This function sets up the graph from the user input
**************************************************************************/
void SetUpGraphKway(GraphType *graph, int nvtxs, idxtype *xadj, idxtype *adjncy)
{
  int i;

  InitGraph(graph);

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->ncon = 1;
  graph->xadj = xadj;
  graph->vwgt = NULL;
  graph->adjncy = adjncy;
  graph->adjwgt = NULL;

  graph->gdata = idxmalloc(2*nvtxs, "SetUpGraph: gdata");
  graph->adjwgtsum = graph->gdata;
  graph->cmap = graph->gdata + nvtxs;

  /* Compute the initial values of the adjwgtsum */
  for (i=0; i<nvtxs; i++)
    graph->adjwgtsum[i] = xadj[i+1]-xadj[i];

}



/*************************************************************************
* This function sets up the graph from the user input
**************************************************************************/
void SetUpGraph2(GraphType *graph, int nvtxs, int ncon, idxtype *xadj,
       idxtype *adjncy, float *nvwgt, idxtype *adjwgt)
{
  int i, j, sum;

  InitGraph(graph);

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->ncon = ncon;
  graph->xadj = xadj;
  graph->adjncy = adjncy;
  graph->adjwgt = adjwgt;

  graph->nvwgt = fmalloc(nvtxs*ncon, "SetUpGraph2: graph->nvwgt");
  scopy(nvtxs*ncon, nvwgt, graph->nvwgt);

  graph->gdata = idxmalloc(2*nvtxs, "SetUpGraph: gdata");

  /* Compute the initial values of the adjwgtsum */
  graph->adjwgtsum = graph->gdata;
  for (i=0; i<nvtxs; i++) {
    sum = 0;
    for (j=xadj[i]; j<xadj[i+1]; j++)
      sum += adjwgt[j];
    graph->adjwgtsum[i] = sum;
  }

  graph->cmap = graph->gdata+nvtxs;

  graph->label = idxmalloc(nvtxs, "SetUpGraph: label");
  for (i=0; i<nvtxs; i++)
    graph->label[i] = i;

}


/*************************************************************************
* This function sets up the graph from the user input
**************************************************************************/
void VolSetUpGraph(GraphType *graph, int OpType, int nvtxs, int ncon, idxtype *xadj,
                   idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int wgtflag)
{
  int i, j, k, sum, gsize;
  idxtype *adjwgt;
  float *nvwgt;
  idxtype tvwgt[MAXNCON];

  InitGraph(graph);

  graph->nvtxs = nvtxs;
  graph->nedges = xadj[nvtxs];
  graph->ncon = ncon;
  graph->xadj = xadj;
  graph->adjncy = adjncy;

  if (ncon == 1) { /* We are in the non mC mode */
    gsize = graph->nedges;  /* This is for the edge weights */
    if ((wgtflag&2) == 0)
      gsize += nvtxs; /* vwgts */
    if ((wgtflag&1) == 0)
      gsize += nvtxs; /* vsize */

    gsize += 2*nvtxs;

    graph->gdata = idxmalloc(gsize, "SetUpGraph: gdata");

    /* Create the vertex/edge weight vectors if they are not supplied */
    gsize = 0;
    if ((wgtflag&2) == 0) {
      vwgt = graph->vwgt = idxset(nvtxs, 1, graph->gdata);
      gsize += nvtxs;
    }
    else
      graph->vwgt = vwgt;

    if ((wgtflag&1) == 0) {
      vsize = graph->vsize = idxset(nvtxs, 1, graph->gdata);
      gsize += nvtxs;
    }
    else
      graph->vsize = vsize;

    /* Allocate memory for edge weights and initialize them to the sum of the vsize */
    adjwgt = graph->adjwgt = graph->gdata+gsize;
    gsize += graph->nedges;

    for (i=0; i<nvtxs; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++)
        adjwgt[j] = 1+vsize[i]+vsize[adjncy[j]];
    }


    /* Compute the initial values of the adjwgtsum */
    graph->adjwgtsum = graph->gdata + gsize;
    gsize += nvtxs;

    for (i=0; i<nvtxs; i++) {
      sum = 0;
      for (j=xadj[i]; j<xadj[i+1]; j++)
        sum += adjwgt[j];
      graph->adjwgtsum[i] = sum;
    }

    graph->cmap = graph->gdata + gsize;
    gsize += nvtxs;

  }
  else {  /* Set up the graph in MOC mode */
    gsize = graph->nedges;
    if ((wgtflag&1) == 0)
      gsize += nvtxs;

    gsize += 2*nvtxs;

    graph->gdata = idxmalloc(gsize, "SetUpGraph: gdata");
    gsize = 0;

    /* Create the normalized vertex weights along each constrain */
    if ((wgtflag&2) == 0)
      vwgt = idxsmalloc(nvtxs, 1, "SetUpGraph: vwgt");

    for (i=0; i<ncon; i++)
      tvwgt[i] = idxsum_strd(nvtxs, vwgt+i, ncon);

    nvwgt = graph->nvwgt = fmalloc(ncon*nvtxs, "SetUpGraph: nvwgt");

    for (i=0; i<nvtxs; i++) {
      for (j=0; j<ncon; j++)
        nvwgt[i*ncon+j] = (1.0*vwgt[i*ncon+j])/(1.0*tvwgt[j]);
    }
    if ((wgtflag&2) == 0)
      free(vwgt);


    /* Create the vsize vector if it is not supplied */
    if ((wgtflag&1) == 0) {
      vsize = graph->vsize = idxset(nvtxs, 1, graph->gdata);
      gsize += nvtxs;
    }
    else
      graph->vsize = vsize;

    /* Allocate memory for edge weights and initialize them to the sum of the vsize */
    adjwgt = graph->adjwgt = graph->gdata+gsize;
    gsize += graph->nedges;

    for (i=0; i<nvtxs; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++)
        adjwgt[j] = 1+vsize[i]+vsize[adjncy[j]];
    }

    /* Compute the initial values of the adjwgtsum */
    graph->adjwgtsum = graph->gdata + gsize;
    gsize += nvtxs;

    for (i=0; i<nvtxs; i++) {
      sum = 0;
      for (j=xadj[i]; j<xadj[i+1]; j++)
        sum += adjwgt[j];
      graph->adjwgtsum[i] = sum;
    }

    graph->cmap = graph->gdata + gsize;
    gsize += nvtxs;

  }

  if (OpType != OP_KVMETIS) {
    graph->label = idxmalloc(nvtxs, "SetUpGraph: label");

    for (i=0; i<nvtxs; i++)
      graph->label[i] = i;
  }

}


/*************************************************************************
* This function randomly permutes the adjacency lists of a graph
**************************************************************************/
void RandomizeGraph(GraphType *graph)
{
  int i, j, k, l, tmp, nvtxs;
  idxtype *xadj, *adjncy, *adjwgt;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  for (i=0; i<nvtxs; i++) {
    l = xadj[i+1]-xadj[i];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = xadj[i] + RandomInRange(l);
      SWAP(adjncy[j], adjncy[k], tmp);
      SWAP(adjwgt[j], adjwgt[k], tmp);
    }
  }
}


/*************************************************************************
* This function checks whether or not partition pid is contigous
**************************************************************************/
int IsConnectedSubdomain(CtrlType *ctrl, GraphType *graph, int pid, int report)
{
  int i, j, k, nvtxs, first, last, nleft, ncmps, wgt;
  idxtype *xadj, *adjncy, *where, *touched, *queue;
  idxtype *cptr;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;

  touched = idxsmalloc(nvtxs, 0, "IsConnected: touched");
  queue = idxmalloc(nvtxs, "IsConnected: queue");
  cptr = idxmalloc(nvtxs, "IsConnected: cptr");

  nleft = 0;
  for (i=0; i<nvtxs; i++) {
    if (where[i] == pid)
      nleft++;
  }

  for (i=0; i<nvtxs; i++) {
    if (where[i] == pid)
      break;
  }

  touched[i] = 1;
  queue[0] = i;
  first = 0; last = 1;

  cptr[0] = 0;  /* This actually points to queue */
  ncmps = 0;
  while (first != nleft) {
    if (first == last) { /* Find another starting vertex */
      cptr[++ncmps] = first;
      for (i=0; i<nvtxs; i++) {
        if (where[i] == pid && !touched[i])
          break;
      }
      queue[last++] = i;
      touched[i] = 1;
    }

    i = queue[first++];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (where[k] == pid && !touched[k]) {
        queue[last++] = k;
        touched[k] = 1;
      }
    }
  }
  cptr[++ncmps] = first;

  if (ncmps > 1 && report) {
    printf("The graph has %d connected components in partition %d:\t", ncmps, pid);
    for (i=0; i<ncmps; i++) {
      wgt = 0;
      for (j=cptr[i]; j<cptr[i+1]; j++)
        wgt += graph->vwgt[queue[j]];
      printf("[%5d %5d] ", cptr[i+1]-cptr[i], wgt);
      /*
      if (cptr[i+1]-cptr[i] == 1)
        printf("[%d %d] ", queue[cptr[i]], xadj[queue[cptr[i]]+1]-xadj[queue[cptr[i]]]);
      */
    }
    printf("\n");
  }

  GKfree((void**) &touched, (void**) &queue, (void**) &cptr, LTERM);

  return (ncmps == 1 ? 1 : 0);
}


/*************************************************************************
* This function checks whether a graph is contigous or not
**************************************************************************/
int IsConnected(CtrlType *ctrl, GraphType *graph, int report)
{
  int i, j, k, nvtxs, first, last;
  idxtype *xadj, *adjncy, *touched, *queue;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;

  touched = idxsmalloc(nvtxs, 0, "IsConnected: touched");
  queue = idxmalloc(nvtxs, "IsConnected: queue");

  touched[0] = 1;
  queue[0] = 0;
  first = 0; last = 1;

  while (first < last) {
    i = queue[first++];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (!touched[k]) {
        queue[last++] = k;
        touched[k] = 1;
      }
    }
  }

  if (first != nvtxs && report)
    printf("The graph is not connected. It has %d disconnected vertices!\n", nvtxs-first);

  return (first == nvtxs ? 1 : 0);
}


/*************************************************************************
* This function checks whether or not partition pid is contigous
**************************************************************************/
int IsConnected2(GraphType *graph, int report)
{
  int i, j, k, nvtxs, first, last, nleft, ncmps, wgt;
  idxtype *xadj, *adjncy, *where, *touched, *queue;
  idxtype *cptr;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;

  touched = idxsmalloc(nvtxs, 0, "IsConnected: touched");
  queue = idxmalloc(nvtxs, "IsConnected: queue");
  cptr = idxmalloc(nvtxs, "IsConnected: cptr");

  nleft = nvtxs;
  touched[0] = 1;
  queue[0] = 0;
  first = 0; last = 1;

  cptr[0] = 0;  /* This actually points to queue */
  ncmps = 0;
  while (first != nleft) {
    if (first == last) { /* Find another starting vertex */
      cptr[++ncmps] = first;
      for (i=0; i<nvtxs; i++) {
        if (!touched[i])
          break;
      }
      queue[last++] = i;
      touched[i] = 1;
    }

    i = queue[first++];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (!touched[k]) {
        queue[last++] = k;
        touched[k] = 1;
      }
    }
  }
  cptr[++ncmps] = first;

  if (ncmps > 1 && report) {
    printf("%d connected components:\t", ncmps);
    for (i=0; i<ncmps; i++) {
      if (cptr[i+1]-cptr[i] > 200)
        printf("[%5d] ", cptr[i+1]-cptr[i]);
    }
    printf("\n");
  }

  GKfree((void**) &touched, (void**) &queue, (void**) &cptr, LTERM);

  return (ncmps == 1 ? 1 : 0);
}


/*************************************************************************
* This function returns the number of connected components in cptr,cind
* The separator of the graph is used to split it and then find its components.
**************************************************************************/
int FindComponents(CtrlType *ctrl, GraphType *graph, idxtype *cptr, idxtype *cind)
{
  int i, j, k, nvtxs, first, last, nleft, ncmps, wgt;
  idxtype *xadj, *adjncy, *where, *touched, *queue;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;

  touched = idxsmalloc(nvtxs, 0, "IsConnected: queue");

  for (i=0; i<graph->nbnd; i++)
    touched[graph->bndind[i]] = 1;

  queue = cind;

  nleft = 0;
  //for (i=0; i<nvtxs; i++) {
  //  if (where[i] != 2)
  //    nleft++;
  //}
  //printf("Doing OK1.\n");
  //for (i=0; i<nvtxs; i++) {
  //  if (where[i] != 2)
  //    break;
  //}
  touched[i] = 1;
  queue[0] = i;
  first = 0; last = 1;

  nleft = nvtxs-1;
  int scanned = 0;

  cptr[0] = 0;  /* This actually points to queue */
  ncmps = 0;
  while (first != nleft) {
    if (first == last) { /* Find another starting vertex */
      cptr[++ncmps] = first;
      for (i=scanned; i<nvtxs; i++) {
        if (!touched[i])
	  {
	    scanned = i;
	    break;
	  }
      }
      queue[last++] = i;
      touched[i] = 1;
    }

    i = queue[first++];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (!touched[k]) {
        queue[last++] = k;
        touched[k] = 1;
      }
    }
  }
  cptr[++ncmps] = first;

  //FILE *fp;
  //fp = fopen("tmp_cc", "w");
  //fprintf(fp, "%d\n", ncmps);
  //for(i=0; i < ncmps; i++)
  //  fprintf(fp, "%d\n", cptr[i]);
  //for(i=0; i < nvtxs; i++)
  //  fprintf(fp, "%d\n", queue[i]);
  //fclose(fp);

  free(touched);

  return ncmps;
}



