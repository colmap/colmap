/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * memory.c
 *
 * This file contains routines that deal with memory allocation
 *
 * Started 2/24/96
 * George
 *
 * $Id: memory.c,v 1.2 1998/11/27 18:16:18 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function allocates memory for the workspace
**************************************************************************/
void AllocateWorkSpace(CtrlType *ctrl, GraphType *graph, int nparts)
{
  ctrl->wspace.pmat = NULL;

  if (ctrl->optype == OP_KMETIS) {
    ctrl->wspace.edegrees = (EDegreeType *)GKmalloc(graph->nedges*sizeof(EDegreeType), "AllocateWorkSpace: edegrees");
    ctrl->wspace.vedegrees = NULL;
    ctrl->wspace.auxcore = (idxtype *)ctrl->wspace.edegrees;

    ctrl->wspace.pmat = idxmalloc(nparts*nparts, "AllocateWorkSpace: pmat");

    /* Memory requirements for different phases
          Coarsening
                    Matching: 4*nvtxs vectors
                 Contraction: 2*nvtxs vectors (from the above 4), 1*nparts, 1*Nedges
            Total = MAX(4*nvtxs, 2*nvtxs+nparts+nedges)

          Refinement
                Random Refinement/Balance: 5*nparts + 1*nvtxs + 2*nedges
                Greedy Refinement/Balance: 5*nparts + 2*nvtxs + 2*nedges + 1*PQueue(==Nvtxs)
            Total = 5*nparts + 3*nvtxs + 2*nedges

         Total = 5*nparts + 3*nvtxs + 2*nedges
    */
    ctrl->wspace.maxcore = 3*(graph->nvtxs+1) +                 /* Match/Refinement vectors */
                           5*(nparts+1) +                       /* Partition weights etc */
                           graph->nvtxs*(sizeof(ListNodeType)/sizeof(idxtype)) + /* Greedy k-way balance/refine */
                           20  /* padding for 64 bit machines */
                           ;
  }
  else if (ctrl->optype == OP_KVMETIS) {
    ctrl->wspace.edegrees = NULL;
    ctrl->wspace.vedegrees = (VEDegreeType *)GKmalloc(graph->nedges*sizeof(VEDegreeType), "AllocateWorkSpace: vedegrees");
    ctrl->wspace.auxcore = (idxtype *)ctrl->wspace.vedegrees;

    ctrl->wspace.pmat = idxmalloc(nparts*nparts, "AllocateWorkSpace: pmat");

    /* Memory requirements for different phases are identical to KMETIS */
    ctrl->wspace.maxcore = 3*(graph->nvtxs+1) +                 /* Match/Refinement vectors */
                           3*(nparts+1) +                       /* Partition weights etc */
                           graph->nvtxs*(sizeof(ListNodeType)/sizeof(idxtype)) + /* Greedy k-way balance/refine */
                           20  /* padding for 64 bit machines */
                           ;
  }
  else {
    ctrl->wspace.edegrees = (EDegreeType *)idxmalloc(graph->nedges, "AllocateWorkSpace: edegrees");
    ctrl->wspace.vedegrees = NULL;
    ctrl->wspace.auxcore = (idxtype *)ctrl->wspace.edegrees;

    ctrl->wspace.maxcore = 5*(graph->nvtxs+1) +                 /* Refinement vectors */
                           4*(nparts+1) +                       /* Partition weights etc */
                           2*graph->ncon*graph->nvtxs*(sizeof(ListNodeType)/sizeof(idxtype)) + /* 2-way refinement */
                           2*graph->ncon*(NEG_GAINSPAN+PLUS_GAINSPAN+1)*(sizeof(ListNodeType *)/sizeof(idxtype)) + /* 2-way refinement */
                           20  /* padding for 64 bit machines */
                           ;
  }

  ctrl->wspace.maxcore += HTLENGTH;
  ctrl->wspace.core = idxmalloc(ctrl->wspace.maxcore, "AllocateWorkSpace: maxcore");
  ctrl->wspace.ccore = 0;
}


/*************************************************************************
* This function allocates memory for the workspace
**************************************************************************/
void FreeWorkSpace(CtrlType *ctrl, GraphType *graph)
{
  GKfree((void**) &ctrl->wspace.edegrees, (void**) &ctrl->wspace.vedegrees, (void**) &ctrl->wspace.core, (void**) &ctrl->wspace.pmat, LTERM);
}

/*************************************************************************
* This function returns how may words are left in the workspace
**************************************************************************/
int WspaceAvail(CtrlType *ctrl)
{
  return ctrl->wspace.maxcore - ctrl->wspace.ccore;
}


/*************************************************************************
* This function allocate space from the core
**************************************************************************/
idxtype *idxwspacemalloc(CtrlType *ctrl, int n)
{
  n += n%2; /* This is a fix for 64 bit machines that require 8-byte pointer allignment */

  ctrl->wspace.ccore += n;
  ASSERT(ctrl->wspace.ccore <= ctrl->wspace.maxcore);
  return ctrl->wspace.core + ctrl->wspace.ccore - n;
}

/*************************************************************************
* This function frees space from the core
**************************************************************************/
void idxwspacefree(CtrlType *ctrl, int n)
{
  n += n%2; /* This is a fix for 64 bit machines that require 8-byte pointer allignment */

  ctrl->wspace.ccore -= n;
  ASSERT(ctrl->wspace.ccore >= 0);
}


/*************************************************************************
* This function allocate space from the core
**************************************************************************/
float *fwspacemalloc(CtrlType *ctrl, int n)
{
  n += n%2; /* This is a fix for 64 bit machines that require 8-byte pointer allignment */

  ctrl->wspace.ccore += n;
  ASSERT(ctrl->wspace.ccore <= ctrl->wspace.maxcore);
  return (float *) (ctrl->wspace.core + ctrl->wspace.ccore - n);
}

/*************************************************************************
* This function frees space from the core
**************************************************************************/
void fwspacefree(CtrlType *ctrl, int n)
{
  n += n%2; /* This is a fix for 64 bit machines that require 8-byte pointer allignment */

  ctrl->wspace.ccore -= n;
  ASSERT(ctrl->wspace.ccore >= 0);
}



/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
GraphType *CreateGraph(void)
{
  GraphType *graph;

  graph = (GraphType *)GKmalloc(sizeof(GraphType), "CreateCoarseGraph: graph");

  InitGraph(graph);

  return graph;
}


/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
void InitGraph(GraphType *graph)
{
  graph->gdata = graph->rdata = NULL;

  graph->nvtxs = graph->nedges = -1;
  graph->mincut = graph->minvol = -1;

  graph->xadj = graph->vwgt = graph->adjncy = graph->adjwgt = NULL;
  graph->adjwgtsum = NULL;
  graph->label = NULL;
  graph->cmap = NULL;

  graph->where = graph->pwgts = NULL;
  graph->id = graph->ed = NULL;
  graph->bndptr = graph->bndind = NULL;
  graph->rinfo = NULL;
  graph->vrinfo = NULL;
  graph->nrinfo = NULL;

  graph->ncon = -1;
  graph->nvwgt = NULL;
  graph->npwgts = NULL;

  graph->vsize = NULL;

  graph->coarser = graph->finer = NULL;

}

/*************************************************************************
* This function deallocates any memory stored in a graph
**************************************************************************/
void FreeGraph(GraphType *graph)
{

  GKfree((void**) &graph->gdata, (void**) &graph->nvwgt, (void**) &graph->rdata, (void**) &graph->npwgts, LTERM);
  free(graph);
}

