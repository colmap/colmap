/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * compress.c
 *
 * This file contains code for compressing nodes with identical adjacency
 * structure and for prunning dense columns
 *
 * Started 9/17/97
 * George
 *
 * $Id: compress.c,v 1.1 1998/11/27 17:59:12 karypis Exp $
 */

#include "metis.h"

/*************************************************************************
* This function compresses a graph by merging identical vertices
* The compression should lead to at least 10% reduction.
**************************************************************************/
void CompressGraph(CtrlType *ctrl, GraphType *graph, int nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *cptr, idxtype *cind)
{
  int i, ii, iii, j, jj, k, l, cnvtxs, cnedges;
  idxtype *cxadj, *cadjncy, *cvwgt, *mark, *map;
  KeyValueType *keys;

  mark = idxsmalloc(nvtxs, -1, "CompressGraph: mark");
  map = idxsmalloc(nvtxs, -1, "CompressGraph: map");
  keys = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "CompressGraph: keys");

  /* Compute a key for each adjacency list */
  for (i=0; i<nvtxs; i++) {
    k = 0;
    for (j=xadj[i]; j<xadj[i+1]; j++)
      k += adjncy[j];
    keys[i].key = k+i; /* Add the diagonal entry as well */
    keys[i].val = i;
  }

  ikeysort(nvtxs, keys);

  l = cptr[0] = 0;
  for (cnvtxs=i=0; i<nvtxs; i++) {
    ii = keys[i].val;
    if (map[ii] == -1) {
      mark[ii] = i;  /* Add the diagonal entry */
      for (j=xadj[ii]; j<xadj[ii+1]; j++)
        mark[adjncy[j]] = i;

      cind[l++] = ii;
      map[ii] = cnvtxs;

      for (j=i+1; j<nvtxs; j++) {
        iii = keys[j].val;

        if (keys[i].key != keys[j].key || xadj[ii+1]-xadj[ii] != xadj[iii+1]-xadj[iii])
          break; /* Break if keys or degrees are different */

        if (map[iii] == -1) { /* Do a comparison if iii has not been mapped */
          for (jj=xadj[iii]; jj<xadj[iii+1]; jj++) {
            if (mark[adjncy[jj]] != i)
              break;
          }

          if (jj == xadj[iii+1]) { /* Identical adjacency structure */
            map[iii] = cnvtxs;
            cind[l++] = iii;
          }
        }
      }

      cptr[++cnvtxs] = l;
    }
  }

  /* printf("Original: %6d, Compressed: %6d\n", nvtxs, cnvtxs); */


  InitGraph(graph);

  if (cnvtxs >= COMPRESSION_FRACTION*nvtxs) {
    graph->nvtxs = nvtxs;
    graph->nedges = xadj[nvtxs];
    graph->ncon = 1;
    graph->xadj = xadj;
    graph->adjncy = adjncy;

    graph->gdata = idxmalloc(3*nvtxs+graph->nedges, "CompressGraph: gdata");
    graph->vwgt    	= graph->gdata;
    graph->adjwgtsum    = graph->gdata+nvtxs;
    graph->cmap		= graph->gdata+2*nvtxs;
    graph->adjwgt	= graph->gdata+3*nvtxs;

    idxset(nvtxs, 1, graph->vwgt);
    idxset(graph->nedges, 1, graph->adjwgt);
    for (i=0; i<nvtxs; i++)
      graph->adjwgtsum[i] = xadj[i+1]-xadj[i];

    graph->label = idxmalloc(nvtxs, "CompressGraph: label");
    for (i=0; i<nvtxs; i++)
      graph->label[i] = i;
  }
  else { /* Ok, form the compressed graph  */
    cnedges = 0;
    for (i=0; i<cnvtxs; i++) {
      ii = cind[cptr[i]];
      cnedges += xadj[ii+1]-xadj[ii];
    }

    /* Allocate memory for the compressed graph*/
    graph->gdata = idxmalloc(4*cnvtxs+1 + 2*cnedges, "CompressGraph: gdata");
    cxadj = graph->xadj		= graph->gdata;
    cvwgt = graph->vwgt         = graph->gdata + cnvtxs+1;
    graph->adjwgtsum        	= graph->gdata + 2*cnvtxs+1;
    graph->cmap                 = graph->gdata + 3*cnvtxs+1;
    cadjncy = graph->adjncy     = graph->gdata + 4*cnvtxs+1;
    graph->adjwgt            	= graph->gdata + 4*cnvtxs+1 + cnedges;

    /* Now go and compress the graph */
    idxset(nvtxs, -1, mark);
    l = cxadj[0] = 0;
    for (i=0; i<cnvtxs; i++) {
      cvwgt[i] = cptr[i+1]-cptr[i];
      mark[i] = i;  /* Remove any dioganal entries in the compressed graph */
      for (j=cptr[i]; j<cptr[i+1]; j++) {
        ii = cind[j];
        for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
          k = map[adjncy[jj]];
          if (mark[k] != i)
            cadjncy[l++] = k;
          mark[k] = i;
        }
      }
      cxadj[i+1] = l;
    }

    graph->nvtxs = cnvtxs;
    graph->nedges = l;
    graph->ncon = 1;

    idxset(graph->nedges, 1, graph->adjwgt);
    for (i=0; i<cnvtxs; i++)
      graph->adjwgtsum[i] = cxadj[i+1]-cxadj[i];

    graph->label = idxmalloc(cnvtxs, "CompressGraph: label");
    for (i=0; i<cnvtxs; i++)
      graph->label[i] = i;

  }

  GKfree((void**) &keys, (void**) &map, (void**) &mark, LTERM);
}



/*************************************************************************
* This function prunes all the vertices in a graph with degree greater
* than factor*average
**************************************************************************/
void PruneGraph(CtrlType *ctrl, GraphType *graph, int nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *iperm, float factor)
{
  int i, j, k, l, nlarge, pnvtxs, pnedges;
  idxtype *pxadj, *padjncy, *padjwgt, *pvwgt;
  idxtype *perm;

  perm = idxmalloc(nvtxs, "PruneGraph: perm");

  factor = factor*xadj[nvtxs]/nvtxs;

  pnvtxs = pnedges = nlarge = 0;
  for (i=0; i<nvtxs; i++) {
    if (xadj[i+1]-xadj[i] < factor) {
      perm[i] = pnvtxs;
      iperm[pnvtxs++] = i;
      pnedges += xadj[i+1]-xadj[i];
    }
    else {
      perm[i] = nvtxs - ++nlarge;
      iperm[nvtxs-nlarge] = i;
    }
  }

  /* printf("Pruned %d vertices\n", nlarge); */

  InitGraph(graph);

  if (nlarge == 0) { /* No prunning */
    graph->nvtxs = nvtxs;
    graph->nedges = xadj[nvtxs];
    graph->ncon = 1;
    graph->xadj = xadj;
    graph->adjncy = adjncy;

    graph->gdata = idxmalloc(3*nvtxs+graph->nedges, "CompressGraph: gdata");
    graph->vwgt    	= graph->gdata;
    graph->adjwgtsum    = graph->gdata+nvtxs;
    graph->cmap		= graph->gdata+2*nvtxs;
    graph->adjwgt	= graph->gdata+3*nvtxs;

    idxset(nvtxs, 1, graph->vwgt);
    idxset(graph->nedges, 1, graph->adjwgt);
    for (i=0; i<nvtxs; i++)
      graph->adjwgtsum[i] = xadj[i+1]-xadj[i];

    graph->label = idxmalloc(nvtxs, "CompressGraph: label");
    for (i=0; i<nvtxs; i++)
      graph->label[i] = i;
  }
  else { /* Prune the graph */
    /* Allocate memory for the compressed graph*/
    graph->gdata = idxmalloc(4*pnvtxs+1 + 2*pnedges, "PruneGraph: gdata");
    pxadj = graph->xadj		= graph->gdata;
    graph->vwgt         	= graph->gdata + pnvtxs+1;
    graph->adjwgtsum        	= graph->gdata + 2*pnvtxs+1;
    graph->cmap                 = graph->gdata + 3*pnvtxs+1;
    padjncy = graph->adjncy     = graph->gdata + 4*pnvtxs+1;
    graph->adjwgt            	= graph->gdata + 4*pnvtxs+1 + pnedges;

    pxadj[0] = pnedges = l = 0;
    for (i=0; i<nvtxs; i++) {
      if (xadj[i+1]-xadj[i] < factor) {
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = perm[adjncy[j]];
          if (k < pnvtxs)
            padjncy[pnedges++] = k;
        }
        pxadj[++l] = pnedges;
      }
    }

    graph->nvtxs = pnvtxs;
    graph->nedges = pnedges;
    graph->ncon = 1;

    idxset(pnvtxs, 1, graph->vwgt);
    idxset(pnedges, 1, graph->adjwgt);
    for (i=0; i<pnvtxs; i++)
      graph->adjwgtsum[i] = pxadj[i+1]-pxadj[i];

    graph->label = idxmalloc(pnvtxs, "CompressGraph: label");
    for (i=0; i<pnvtxs; i++)
      graph->label[i] = i;
  }

  free(perm);

}









