/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * estmem.c
 *
 * This file contains code for estimating the amount of memory required by
 * the various routines in METIS
 *
 * Started 11/4/97
 * George
 *
 * $Id: estmem.c,v 1.1 1998/11/27 17:59:13 karypis Exp $
 *
 */

#include "metis.h"

/*************************************************************************
* This function computes how much memory will be required by the various
* routines in METIS
**************************************************************************/
void METIS_EstimateMemory(int *nvtxs, idxtype *xadj, idxtype *adjncy, int *numflag, int *optype, int *nbytes)
{
  int i, j, k, nedges, nlevels;
  float vfraction, efraction, vmult, emult;
  int coresize, gdata, rdata;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  nedges = xadj[*nvtxs];

  InitRandom(-1);
  EstimateCFraction(*nvtxs, xadj, adjncy, &vfraction, &efraction);

  /* Estimate the amount of memory for coresize */
  if (*optype == 2)
    coresize = nedges;
  else
    coresize = 0;
  coresize += nedges + 11*(*nvtxs) + 4*1024 + 2*(NEG_GAINSPAN+PLUS_GAINSPAN+1)*(sizeof(ListNodeType *)/sizeof(idxtype));
  coresize += 2*(*nvtxs);  /* add some more fore other vectors */

  gdata = nedges;   /* Assume that the user does not pass weights */

  nlevels = (int)(log(100.0/(*nvtxs))/log(vfraction) + .5);
  vmult = 0.5 + (1.0 - pow(vfraction, nlevels))/(1.0 - vfraction);
  emult = 1.0 + (1.0 - pow(efraction, nlevels+1))/(1.0 - efraction);

  gdata += (int) vmult*4*(*nvtxs) + emult*2*nedges;
  if ((vmult-1.0)*4*(*nvtxs) + (emult-1.0)*2*nedges < 5*(*nvtxs))
    rdata = 0;
  else
    rdata = 5*(*nvtxs);

  *nbytes = sizeof(idxtype)*(coresize+gdata+rdata+(*nvtxs));

  if (*numflag == 1)
    Change2FNumbering2(*nvtxs, xadj, adjncy);
}


/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void EstimateCFraction(int nvtxs, idxtype *xadj, idxtype *adjncy, float *vfraction, float *efraction)
{
  int i, ii, j, cnvtxs, cnedges, maxidx;
  idxtype *match, *cmap, *perm;

  cmap = idxmalloc(nvtxs, "cmap");
  match = idxsmalloc(nvtxs, UNMATCHED, "match");
  perm = idxmalloc(nvtxs, "perm");
  RandomPermute(nvtxs, perm, 1);

  cnvtxs = 0;
  for (ii=0; ii<nvtxs; ii++) {
    i = perm[ii];

    if (match[i] == UNMATCHED) {  /* Unmatched */
      maxidx = i;

      /* Find a random matching, subject to maxvwgt constraints */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (match[adjncy[j]] == UNMATCHED) {
          maxidx = adjncy[j];
          break;
        }
      }

      cmap[i] = cmap[maxidx] = cnvtxs++;
      match[i] = maxidx;
      match[maxidx] = i;
    }
  }

  cnedges = ComputeCoarseGraphSize(nvtxs, xadj, adjncy, cnvtxs, cmap, match, perm);

  *vfraction = (1.0*cnvtxs)/(1.0*nvtxs);
  *efraction = (1.0*cnedges)/(1.0*xadj[nvtxs]);

  GKfree((void**) &cmap, (void**) &match, (void**) &perm, LTERM);
}




/*************************************************************************
* This function computes the size of the coarse graph
**************************************************************************/
int ComputeCoarseGraphSize(int nvtxs, idxtype *xadj, idxtype *adjncy, int cnvtxs, idxtype *cmap, idxtype *match, idxtype *perm)
{
  int i, j, k, istart, iend, nedges, cnedges, v, u;
  idxtype *htable;

  htable = idxsmalloc(cnvtxs, -1, "htable");

  cnvtxs = cnedges = 0;
  for (i=0; i<nvtxs; i++) {
    v = perm[i];
    if (cmap[v] != cnvtxs)
      continue;

    htable[cnvtxs] = cnvtxs;

    u = match[v];

    istart = xadj[v];
    iend = xadj[v+1];
    for (j=istart; j<iend; j++) {
      k = cmap[adjncy[j]];
      if (htable[k] != cnvtxs) {
        htable[k] = cnvtxs;
        cnedges++;
      }
    }

    if (v != u) {
      istart = xadj[u];
      iend = xadj[u+1];
      for (j=istart; j<iend; j++) {
        k = cmap[adjncy[j]];
        if (htable[k] != cnvtxs) {
          htable[k] = cnvtxs;
          cnedges++;
        }
      }
    }
    cnvtxs++;
  }

  GKfree((void**) &htable, LTERM);

  return cnedges;
}


