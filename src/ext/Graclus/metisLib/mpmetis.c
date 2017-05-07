/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mpmetis.c
 *
 * This file contains the top level routines for the multilevel recursive
 * bisection algorithm PMETIS.
 *
 * Started 7/24/97
 * George
 *
 * $Id: mpmetis.c,v 1.4 1998/11/30 14:50:44 karypis Exp $
 *
 */

#include "metis.h"



/*************************************************************************
* This function is the entry point for PWMETIS that accepts exact weights
* for the target partitions
**************************************************************************/
void METIS_mCPartGraphRecursive(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy,
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts,
       int *options, int *edgecut, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  SetUpGraph(&graph, OP_PMETIS, *nvtxs, *ncon, xadj, adjncy, vwgt, adjwgt, *wgtflag);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType  = McPMETIS_CTYPE;
    ctrl.IType  = McPMETIS_ITYPE;
    ctrl.RType  = McPMETIS_RTYPE;
    ctrl.dbglvl = McPMETIS_DBGLVL;
  }
  else {
    ctrl.CType  = options[OPTION_CTYPE];
    ctrl.IType  = options[OPTION_ITYPE];
    ctrl.RType  = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_PMETIS;
  ctrl.CoarsenTo = 100;

  ctrl.nmaxvwgt = 1.5/(1.0*ctrl.CoarsenTo);

  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MCMlevelRecursiveBisection(&ctrl, &graph, *nparts, part, 1.000, 0);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}



/*************************************************************************
* This function is the entry point for PWMETIS that accepts exact weights
* for the target partitions
**************************************************************************/
void METIS_mCHPartGraphRecursive(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy,
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts,
       float *ubvec, int *options, int *edgecut, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;
  float *myubvec;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  SetUpGraph(&graph, OP_PMETIS, *nvtxs, *ncon, xadj, adjncy, vwgt, adjwgt, *wgtflag);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = PMETIS_CTYPE;
    ctrl.IType = PMETIS_ITYPE;
    ctrl.RType = PMETIS_RTYPE;
    ctrl.dbglvl = PMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_PMETIS;
  ctrl.CoarsenTo = 100;

  ctrl.nmaxvwgt = 1.5/(1.0*ctrl.CoarsenTo);

  myubvec = fmalloc(*ncon, "PWMETIS: mytpwgts");
  scopy(*ncon, ubvec, myubvec);

  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MCHMlevelRecursiveBisection(&ctrl, &graph, *nparts, part, myubvec, 0);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);
  GKfree((void **) &myubvec, LTERM);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}



/*************************************************************************
* This function is the entry point for PWMETIS that accepts exact weights
* for the target partitions
**************************************************************************/
void METIS_mCPartGraphRecursiveInternal(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy,
       float *nvwgt, idxtype *adjwgt, int *nparts, int *options, int *edgecut, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;

  SetUpGraph2(&graph, *nvtxs, *ncon, xadj, adjncy, nvwgt, adjwgt);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = PMETIS_CTYPE;
    ctrl.IType = PMETIS_ITYPE;
    ctrl.RType = PMETIS_RTYPE;
    ctrl.dbglvl = PMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_PMETIS;
  ctrl.CoarsenTo = 100;

  ctrl.nmaxvwgt = 1.5/(1.0*ctrl.CoarsenTo);

  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MCMlevelRecursiveBisection(&ctrl, &graph, *nparts, part, 1.000, 0);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);

}


/*************************************************************************
* This function is the entry point for PWMETIS that accepts exact weights
* for the target partitions
**************************************************************************/
void METIS_mCHPartGraphRecursiveInternal(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy,
       float *nvwgt, idxtype *adjwgt, int *nparts, float *ubvec, int *options, int *edgecut,
       idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;
  float *myubvec;

  SetUpGraph2(&graph, *nvtxs, *ncon, xadj, adjncy, nvwgt, adjwgt);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = PMETIS_CTYPE;
    ctrl.IType = PMETIS_ITYPE;
    ctrl.RType = PMETIS_RTYPE;
    ctrl.dbglvl = PMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_PMETIS;
  ctrl.CoarsenTo = 100;

  ctrl.nmaxvwgt = 1.5/(1.0*ctrl.CoarsenTo);

  myubvec = fmalloc(*ncon, "PWMETIS: mytpwgts");
  scopy(*ncon, ubvec, myubvec);

  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MCHMlevelRecursiveBisection(&ctrl, &graph, *nparts, part, myubvec, 0);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);
  GKfree((void **) &myubvec, LTERM);

}




/*************************************************************************
* This function takes a graph and produces a bisection of it
**************************************************************************/
int MCMlevelRecursiveBisection(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *part,
       float ubfactor, int fpart)
{
  int i, j, nvtxs, ncon, cut;
  idxtype *label, *where;
  GraphType lgraph, rgraph;
  float tpwgts[2];

  nvtxs = graph->nvtxs;
  if (nvtxs == 0) {
    printf("\t***Cannot bisect a graph with 0 vertices!\n\t***You are trying to partition a graph into too many parts!\n");
    return 0;
  }

  /* Determine the weights of the partitions */
  tpwgts[0] = 1.0*(nparts>>1)/(1.0*nparts);
  tpwgts[1] = 1.0 - tpwgts[0];

  MCMlevelEdgeBisection(ctrl, graph, tpwgts, ubfactor);
  cut = graph->mincut;

  label = graph->label;
  where = graph->where;
  for (i=0; i<nvtxs; i++)
    part[label[i]] = where[i] + fpart;

  if (nparts > 2)
    SplitGraphPart(ctrl, graph, &lgraph, &rgraph);

  /* Free the memory of the top level graph */
  GKfree((void **) &graph->gdata, (void **) &graph->nvwgt, (void **) &graph->rdata, (void **) &graph->npwgts, (void **) &graph->label, LTERM);


  /* Do the recursive call */
  if (nparts > 3) {
    cut += MCMlevelRecursiveBisection(ctrl, &lgraph, nparts/2, part, ubfactor, fpart);
    cut += MCMlevelRecursiveBisection(ctrl, &rgraph, nparts-nparts/2, part, ubfactor, fpart+nparts/2);
  }
  else if (nparts == 3) {
    cut += MCMlevelRecursiveBisection(ctrl, &rgraph, nparts-nparts/2, part, ubfactor, fpart+nparts/2);
    GKfree((void **) &lgraph.gdata, (void **) &lgraph.nvwgt, (void **) &lgraph.label, LTERM);
  }

  return cut;

}



/*************************************************************************
* This function takes a graph and produces a bisection of it
**************************************************************************/
int MCHMlevelRecursiveBisection(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *part,
      float *ubvec, int fpart)
{
  int i, j, nvtxs, ncon, cut;
  idxtype *label, *where;
  GraphType lgraph, rgraph;
  float tpwgts[2], *npwgts, *lubvec, *rubvec;

  lubvec = rubvec = NULL;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  if (nvtxs == 0) {
    printf("\t***Cannot bisect a graph with 0 vertices!\n\t***You are trying to partition a graph into too many parts!\n");
    return 0;
  }

  /* Determine the weights of the partitions */
  tpwgts[0] = 1.0*(nparts>>1)/(1.0*nparts);
  tpwgts[1] = 1.0 - tpwgts[0];

  /* For now, relax at the coarsest level only */
  if (nparts == 2)
    MCHMlevelEdgeBisection(ctrl, graph, tpwgts, ubvec);
  else
    MCMlevelEdgeBisection(ctrl, graph, tpwgts, 1.000);
  cut = graph->mincut;

  label = graph->label;
  where = graph->where;
  for (i=0; i<nvtxs; i++)
    part[label[i]] = where[i] + fpart;

  if (nparts > 2) {
    /* Adjust the ubvecs before the split */
    npwgts = graph->npwgts;
    lubvec = fmalloc(ncon, "MCHMlevelRecursiveBisection");
    rubvec = fmalloc(ncon, "MCHMlevelRecursiveBisection");

    for (i=0; i<ncon; i++) {
      lubvec[i] = ubvec[i]*tpwgts[0]/npwgts[i];
      lubvec[i] = amax(lubvec[i], 1.01);

      rubvec[i] = ubvec[i]*tpwgts[1]/npwgts[ncon+i];
      rubvec[i] = amax(rubvec[i], 1.01);
    }

    SplitGraphPart(ctrl, graph, &lgraph, &rgraph);
  }

  /* Free the memory of the top level graph */
  GKfree((void **) &graph->gdata, (void **) &graph->nvwgt, (void **) &graph->rdata, (void **) &graph->npwgts, (void **) &graph->label, LTERM);


  /* Do the recursive call */
  if (nparts > 3) {
    cut += MCHMlevelRecursiveBisection(ctrl, &lgraph, nparts/2, part, lubvec, fpart);
    cut += MCHMlevelRecursiveBisection(ctrl, &rgraph, nparts-nparts/2, part, rubvec, fpart+nparts/2);
  }
  else if (nparts == 3) {
    cut += MCHMlevelRecursiveBisection(ctrl, &rgraph, nparts-nparts/2, part, rubvec, fpart+nparts/2);
    GKfree((void **) &lgraph.gdata, (void **) &lgraph.nvwgt, (void **) &lgraph.label, LTERM);
  }

  GKfree((void **) &lubvec, (void **) &rubvec, LTERM);

  return cut;

}




/*************************************************************************
* This function performs multilevel bisection
**************************************************************************/
void MCMlevelEdgeBisection(CtrlType *ctrl, GraphType *graph, float *tpwgts, float ubfactor)
{
  GraphType *cgraph;

  cgraph = MCCoarsen2Way(ctrl, graph);

  MocInit2WayPartition(ctrl, cgraph, tpwgts, ubfactor);

  MocRefine2Way(ctrl, graph, cgraph, tpwgts, ubfactor);

}



/*************************************************************************
* This function performs multilevel bisection
**************************************************************************/
void MCHMlevelEdgeBisection(CtrlType *ctrl, GraphType *graph, float *tpwgts, float *ubvec)
{
  int i;
  GraphType *cgraph;

/*
  for (i=0; i<graph->ncon; i++)
    printf("%.4f ", ubvec[i]);
  printf("\n");
*/

  cgraph = MCCoarsen2Way(ctrl, graph);

  MocInit2WayPartition2(ctrl, cgraph, tpwgts, ubvec);

  MocRefine2Way2(ctrl, graph, cgraph, tpwgts, ubvec);

}


