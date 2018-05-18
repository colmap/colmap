/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * kvmetis.c
 *
 * This file contains the top level routines for the multilevel k-way partitioning
 * algorithm KMETIS.
 *
 * Started 7/28/97
 * George
 *
 * $Id: kvmetis.c,v 1.1 1998/11/27 17:59:16 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function is the entry point for KMETIS
**************************************************************************/
void METIS_PartGraphVKway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
                         idxtype *vsize, int *wgtflag, int *numflag, int *nparts,
                         int *options, int *volume, idxtype *part)
{
  int i;
  float *tpwgts;

  tpwgts = fmalloc(*nparts, "KMETIS: tpwgts");
  for (i=0; i<*nparts; i++)
    tpwgts[i] = 1.0/(1.0*(*nparts));

  METIS_WPartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts,
                       tpwgts, options, volume, part);

  free(tpwgts);
}


/*************************************************************************
* This function is the entry point for KWMETIS
**************************************************************************/
void METIS_WPartGraphVKway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
                          idxtype *vsize, int *wgtflag, int *numflag, int *nparts,
                          float *tpwgts, int *options, int *volume, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  VolSetUpGraph(&graph, OP_KVMETIS, *nvtxs, 1, xadj, adjncy, vwgt, vsize, *wgtflag);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = KVMETIS_CTYPE;
    ctrl.IType = KVMETIS_ITYPE;
    ctrl.RType = KVMETIS_RTYPE;
    ctrl.dbglvl = KVMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_KVMETIS;
  ctrl.CoarsenTo = amax((*nvtxs)/(40*log2_metis(*nparts)), 20*(*nparts));
  ctrl.maxvwgt = (int) 1.5*((graph.vwgt ? idxsum(*nvtxs, graph.vwgt) : (*nvtxs))/ctrl.CoarsenTo);

  InitRandom(-1);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *volume = MlevelVolKWayPartitioning(&ctrl, &graph, *nparts, part, tpwgts, 1.03);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}


/*************************************************************************
* This function takes a graph and produces a bisection of it
**************************************************************************/
int MlevelVolKWayPartitioning(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *part,
                              float *tpwgts, float ubfactor)
{
  int i, j, nvtxs, tvwgt, tpwgts2[2];
  GraphType *cgraph;
  int wgtflag=3, numflag=0, options[10], edgecut;

  cgraph = Coarsen2Way(ctrl, graph);

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->InitPartTmr));
  AllocateVolKWayPartitionMemory(ctrl, cgraph, nparts);

  options[0] = 1;
  options[OPTION_CTYPE] = MATCH_SHEMKWAY;
  options[OPTION_ITYPE] = IPART_GGPKL;
  options[OPTION_RTYPE] = RTYPE_FM;
  options[OPTION_DBGLVL] = 0;

  METIS_WPartGraphRecursive(&cgraph->nvtxs, cgraph->xadj, cgraph->adjncy, cgraph->vwgt,
                            cgraph->adjwgt, &wgtflag, &numflag, &nparts, tpwgts, options,
                            &edgecut, cgraph->where);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->InitPartTmr));
  IFSET(ctrl->dbglvl, DBG_IPART, printf("Initial %d-way partitioning cut: %d\n", nparts, edgecut));

  IFSET(ctrl->dbglvl, DBG_KWAYPINFO, ComputePartitionInfo(cgraph, nparts, cgraph->where));

  RefineVolKWay(ctrl, graph, cgraph, nparts, tpwgts, ubfactor);

  idxcopy(graph->nvtxs, graph->where, part);

  GKfree((void **) &graph->gdata, (void **) &graph->rdata, LTERM);

  return graph->minvol;

}

