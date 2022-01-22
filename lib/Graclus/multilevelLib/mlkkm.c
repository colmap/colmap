/*
 * Copyright 2005,
 *
 * mlkkm.c
 *
 * This file contains the top level routines for the multilevel kernel k-means algorithm
 *
 *
 * Started 12/2004
 * Yuqiang Guan
 *
 * $Id: kmetis.c,v 1.1 1998/11/27 17:59:15 karypis Exp $
 *
 */

#include "metisLib/metis.h"

extern int spectral_initialization;

/*************************************************************************
* This function is the entry point for MLKKM
**************************************************************************/
void MLKKM_PartGraphKway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
                         idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *chainlength,
                         int *options, int *edgecut, idxtype *part, int levels)
{
  int i;
  float *tpwgts;

  tpwgts = fmalloc(*nparts, "MLKKM: tpwgts");
  for (i=0; i<*nparts; i++)
    tpwgts[i] = 1.0/(1.0*(*nparts));

  MLKKM_WPartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, chainlength,
                       tpwgts, options, edgecut, part, levels);

  free(tpwgts);
}


/*************************************************************************
* This function is the entry point for KWMETIS
**************************************************************************/
void MLKKM_WPartGraphKway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
                          idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *chainlength,
                          float *tpwgts, int *options, int *edgecut, idxtype *part, int levels)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  SetUpGraph(&graph, OP_KMETIS, *nvtxs, 1, xadj, adjncy, vwgt, adjwgt, *wgtflag);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = KMETIS_CTYPE;
    ctrl.IType = KMETIS_ITYPE;
    ctrl.RType = KMETIS_RTYPE;
    ctrl.dbglvl = KMETIS_DBGLVL;
    //ctrl.cutType = options[10];
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
    //ctrl.cutType = options[10];
  }
  ctrl.optype = OP_KMETIS;
  //ctrl.CoarsenTo = amax((*nvtxs)/(40*log2_metis(*nparts)), 5*(*nparts));
  ctrl.CoarsenTo = levels;
  //ctrl.CoarsenTo = amax(40*log2_metis(*nparts), 20*(*nparts));
  //printf("Coarsen To = %d\n", ctrl.CoarsenTo);
  ctrl.maxvwgt = floor(1.5*((graph.vwgt ? idxsum(*nvtxs, graph.vwgt) : (*nvtxs))/ctrl.CoarsenTo));
  ctrl.maxvwgt *= 100;
  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MLKKMPartitioning(&ctrl, &graph, *nparts, *chainlength, part, tpwgts, 1.03);
  /*
  graph.where = idxsmalloc(graph.nvtxs, 0, "Weighted_kernel_k_means: where");
  graph.where[0]=graph.where[1]=graph.where[2]=graph.where[3]=0;
  graph.where[4]=graph.where[5]=graph.where[6]=1;
  Weighted_kernel_k_means(&ctrl, &graph, *nparts, tpwgts, 1.03);
  */

  //idxcopy(graph.nvtxs, graph.where, part);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}

/************************************************************************
* This function takes a graph and produces a k-way partitioning of it
**************************************************************************/
/*void spectralInit(GraphType * graph, int nparts, int *numflag, int* options)
{
  int i, j, nvtxs, nedges;
  idxtype *adjwgt, *adjncy, *xadj, *where;
  spectral::Driver* d = new spectral::Driver();
  CtrlType ctrl;
  double * dense;
  idxtype *w;
  float *m_adjwgt;

  nvtxs = graph->nvtxs;
  nedges = graph->nedges;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;

  if (*numflag == 1)
    Change2CNumbering(nvtxs, xadj, adjncy);

  if (options[0] == 0) {  // Use the default parameters
    ctrl.CType = KMETIS_CTYPE;
    ctrl.IType = KMETIS_ITYPE;
    ctrl.RType = KMETIS_RTYPE;
    ctrl.dbglvl = KMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_KMETIS;
  //ctrl.CoarsenTo = amax((nvtxs)/(40*log2_metis(nparts), 20*nparts);
  ctrl.CoarsenTo = amax((nvtxs)/(40*log2_metis(nparts)), 5*nparts);
  ctrl.maxvwgt = (int) 1.5*(idxsum(nvtxs, graph->vwgt)/ctrl.CoarsenTo);
  w = idxsmalloc(nvtxs, 0, "pingpong: weight");
  Compute_Weights(&ctrl, graph, w);
  m_adjwgt = fmalloc(nedges, "pingpong: normalized matrix");

  transform_matrix_half(&ctrl, graph, w, m_adjwgt);
  dense = new double [nvtxs*nvtxs];
  sparse2dense(graph, dense, m_adjwgt);
  //printf("size: %d\n", nvtxs);
  AllocateWorkSpace(&ctrl, graph, nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  printf("Running spectral over %d nodes and %d clusters....\n", nvtxs, nparts);

  //d->execute2((int*)graph->xadj, (int*)graph->adjncy,(int*) graph->adjwgt, nvtxs, nedges, nparts);
  d->execute(dense, nvtxs ,nparts);
  d->copyClusterID(where, nvtxs);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, graph);
  //free(dense);
  free(w);
  free(m_adjwgt);

  if (*numflag == 1)
    Change2FNumbering(nvtxs, xadj, adjncy, where);
}
*/


/*************************************************************************
* This function takes a graph and produces a k-way partitioning of it
**************************************************************************/
int MLKKMPartitioning(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, idxtype *part, float *tpwgts, float ubfactor)
{
  int i, j, nvtxs, tvwgt, tpwgts2[2];
  GraphType *cgraph;
  int wgtflag=3, numflag=0, options[10], edgecut;
  float ncut;
  // idxtype *cptr, *cind;
  int numcomponents;
  char *mlwkkm_fname = "coarse.graph";

  // cptr = idxmalloc(graph->nvtxs, "MLKKMPartitioning: cptr");
  // cind = idxmalloc(graph->nvtxs, "MLKKMPartitioning: cind");
  //printf("Computing the number of connected components.\n");
  /*numcomponents = FindComponents(ctrl, graph, cptr, cind);

  printf("Number of connected components is %d.\n", numcomponents);
  */
  cgraph = Coarsen2Way(ctrl, graph);

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->InitPartTmr));
  AllocateKWayPartitionMemory(ctrl, cgraph, nparts);

  options[0] = 1;
  options[OPTION_CTYPE] = MATCH_SHEMKWAY;
  options[OPTION_ITYPE] = IPART_GGPKL;
  options[OPTION_RTYPE] = RTYPE_FM;
  options[OPTION_DBGLVL] = 0;

  if(spectral_initialization == 0)
  {
	  METIS_WPartGraphRecursive(&cgraph->nvtxs, cgraph->xadj, cgraph->adjncy, cgraph->vwgt,
                            cgraph->adjwgt, &wgtflag, &numflag, &nparts, tpwgts, options,
                            &edgecut, cgraph->where);
  }
//  else
//    spectralInit(cgraph, nparts, &numflag, options);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->InitPartTmr));
  IFSET(ctrl->dbglvl, DBG_IPART, printf("Initial %d-way partitioning cut: %d\n", nparts, edgecut));

  IFSET(ctrl->dbglvl, DBG_KWAYPINFO, ComputePartitionInfo(cgraph, nparts, cgraph->where));


  /* modification begins */
  /*
  for (int i=0; i<cgraph->nvtxs; i++)
    printf("%d ", cgraph->where[i]);
  printf("*\n");
  */


  //WriteCoarsestGraph(cgraph, mlwkkm_fname, &wgtflag);
  /*
  if (cutType == NCUT)
    strcat(mlwkkm_fname, ".iniNC");
  else
    strcat(mlwkkm_fname, ".iniRA");

  ReadCoarsestInit(cgraph, mlwkkm_fname, wgtflag);
  */

  /*for random initialization

  //AllocateKWayPartitionMemory(ctrl, graph, nparts);
  graph->where = imalloc(graph->nvtxs, "MLKKMPartitioning: where\n");
  printf("%d \n", graph->nvtxs);
  RandomInit(graph->nvtxs, nparts, graph->where);
  pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor);

  //for random initialization ends here
  */

  MLKKMRefine(ctrl, graph, cgraph, nparts, chain_length, tpwgts, ubfactor);

  /* modification ends */

  idxcopy(graph->nvtxs, graph->where, part);

  //ncut = ComputeNCut(graph, part, nparts);
  //printf("  %d-way Normalized-Cut: %7f\n", nparts, ncut);

  GKfree((void **) &graph->gdata, (void **) &graph->rdata, LTERM);

  return graph->mincut;

}

