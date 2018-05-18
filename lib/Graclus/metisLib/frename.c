/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * frename.c
 *
 * This file contains some renaming routines to deal with different Fortran compilers
 *
 * Started 9/15/97
 * George
 *
 * $Id: frename.c,v 1.1 1998/11/27 17:59:14 karypis Exp $
 *
 */

#include "metis.h"


void METIS_PARTGRAPHRECURSIVE(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphrecursive(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphrecursive_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphrecursive__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}


void METIS_WPARTGRAPHRECURSIVE(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphrecursive(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphrecursive_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphrecursive__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphRecursive(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}



void METIS_PARTGRAPHKWAY(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphkway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphkway_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_partgraphkway__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_PartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}



void METIS_WPARTGRAPHKWAY(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphkway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphkway_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}
void metis_wpartgraphkway__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  METIS_WPartGraphKway(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, tpwgts, options, edgecut, part);
}



void METIS_EDGEND(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_EdgeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_edgend(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_EdgeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_edgend_(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_EdgeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_edgend__(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_EdgeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}



void METIS_NODEND(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_nodend(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_nodend_(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}
void metis_nodend__(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeND(nvtxs, xadj, adjncy, numflag, options, perm, iperm);
}



void METIS_NODEWND(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeWND(nvtxs, xadj, adjncy, vwgt, numflag, options, perm, iperm);
}
void metis_nodewnd(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeWND(nvtxs, xadj, adjncy, vwgt, numflag, options, perm, iperm);
}
void metis_nodewnd_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeWND(nvtxs, xadj, adjncy, vwgt, numflag, options, perm, iperm);
}
void metis_nodewnd__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, int *numflag, int *options, idxtype *perm, idxtype *iperm)
{
  METIS_NodeWND(nvtxs, xadj, adjncy, vwgt, numflag, options, perm, iperm);
}



void METIS_PARTMESHNODAL(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshNodal(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshnodal(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshNodal(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshnodal_(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshNodal(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshnodal__(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshNodal(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}


void METIS_PARTMESHDUAL(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshDual(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshdual(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshDual(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshdual_(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshDual(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}
void metis_partmeshdual__(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  METIS_PartMeshDual(ne, nn, elmnts, etype, numflag, nparts, edgecut, epart, npart);
}


void METIS_MESHTONODAL(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToNodal(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtonodal(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToNodal(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtonodal_(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToNodal(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtonodal__(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToNodal(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}


void METIS_MESHTODUAL(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToDual(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtodual(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToDual(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtodual_(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToDual(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}
void metis_meshtodual__(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, idxtype *dxadj, idxtype *dadjncy)
{
  METIS_MeshToDual(ne, nn, elmnts, etype, numflag, dxadj, dadjncy);
}


void METIS_ESTIMATEMEMORY(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *optype, int *nbytes)
{
  METIS_EstimateMemory(nvtxs, xadj, adjncy, numflag, optype, nbytes);
}
void metis_estimatememory(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *optype, int *nbytes)
{
  METIS_EstimateMemory(nvtxs, xadj, adjncy, numflag, optype, nbytes);
}
void metis_estimatememory_(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *optype, int *nbytes)
{
  METIS_EstimateMemory(nvtxs, xadj, adjncy, numflag, optype, nbytes);
}
void metis_estimatememory__(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *optype, int *nbytes)
{
  METIS_EstimateMemory(nvtxs, xadj, adjncy, numflag, optype, nbytes);
}



void METIS_MCPARTGRAPHRECURSIVE(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphRecursive(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_mcpartgraphrecursive(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphRecursive(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_mcpartgraphrecursive_(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphRecursive(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}
void metis_mcpartgraphrecursive__(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphRecursive(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, part);
}


void METIS_MCPARTGRAPHKWAY(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *rubvec, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, rubvec, options, edgecut, part);
}
void metis_mcpartgraphkway(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *rubvec, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, rubvec, options, edgecut, part);
}
void metis_mcpartgraphkway_(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *rubvec, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, rubvec, options, edgecut, part);
}
void metis_mcpartgraphkway__(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, float *rubvec, int *options, int *edgecut, idxtype *part)
{
  METIS_mCPartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, rubvec, options, edgecut, part);
}


void METIS_PARTGRAPHVKWAY(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, int *options, int *volume, idxtype *part)
{
  METIS_PartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, options, volume, part);
}
void metis_partgraphvkway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, int *options, int *volume, idxtype *part)
{
  METIS_PartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, options, volume, part);
}
void metis_partgraphvkway_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, int *options, int *volume, idxtype *part)
{
  METIS_PartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, options, volume, part);
}
void metis_partgraphvkway__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, int *options, int *volume, idxtype *part)
{
  METIS_PartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, options, volume, part);
}

void METIS_WPARTGRAPHVKWAY(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *volume, idxtype *part)
{
  METIS_WPartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, tpwgts, options, volume, part);
}
void metis_wpartgraphvkway(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *volume, idxtype *part)
{
  METIS_WPartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, tpwgts, options, volume, part);
}
void metis_wpartgraphvkway_(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *volume, idxtype *part)
{
  METIS_WPartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, tpwgts, options, volume, part);
}
void metis_wpartgraphvkway__(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *vsize, int *wgtflag, int *numflag, int *nparts, float *tpwgts, int *options, int *volume, idxtype *part)
{
  METIS_WPartGraphVKway(nvtxs, xadj, adjncy, vwgt, vsize, wgtflag, numflag, nparts, tpwgts, options, volume, part);
}



