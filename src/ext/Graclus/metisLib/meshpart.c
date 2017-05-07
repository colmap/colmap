/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * meshpart.c
 *
 * This file contains routines for partitioning finite element meshes.
 *
 * Started 9/29/97
 * George
 *
 * $Id: meshpart.c,v 1.1 1998/11/27 17:59:21 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function partitions a finite element mesh by partitioning its nodal
* graph using KMETIS and then assigning elements in a load balanced fashion.
**************************************************************************/
void METIS_PartMeshNodal(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag,
                         int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  int i, j, k, me;
  idxtype *xadj, *adjncy, *pwgts;
  int options[10], pnumflag=0, wgtflag=0;
  int nnbrs, nbrind[200], nbrwgt[200], maxpwgt;
  int esize, esizes[] = {-1, 3, 4, 8, 4};

  esize = esizes[*etype];

  if (*numflag == 1)
    ChangeMesh2CNumbering((*ne)*esize, elmnts);

  xadj = idxmalloc(*nn+1, "METIS_MESHPARTNODAL: xadj");
  adjncy = idxmalloc(20*(*nn), "METIS_MESHPARTNODAL: adjncy");

  METIS_MeshToNodal(ne, nn, elmnts, etype, &pnumflag, xadj, adjncy);

  adjncy = (idxtype*) realloc(adjncy, xadj[*nn]*sizeof(idxtype));

  options[0] = 0;
  METIS_PartGraphKway(nn, xadj, adjncy, NULL, NULL, &wgtflag, &pnumflag, nparts, options, edgecut, npart);

  /* OK, now compute an element partition based on the nodal partition npart */
  idxset(*ne, -1, epart);
  pwgts = idxsmalloc(*nparts, 0, "METIS_MESHPARTNODAL: pwgts");
  for (i=0; i<*ne; i++) {
    me = npart[elmnts[i*esize]];
    for (j=1; j<esize; j++) {
      if (npart[elmnts[i*esize+j]] != me)
        break;
    }
    if (j == esize) {
      epart[i] = me;
      pwgts[me]++;
    }
  }

  maxpwgt = (int) 1.03*(*ne)/(*nparts);
  for (i=0; i<*ne; i++) {
    if (epart[i] == -1) { /* Assign the boundary element */
      nnbrs = 0;
      for (j=0; j<esize; j++) {
        me = npart[elmnts[i*esize+j]];
        for (k=0; k<nnbrs; k++) {
          if (nbrind[k] == me) {
            nbrwgt[k]++;
            break;
          }
        }
        if (k == nnbrs) {
          nbrind[nnbrs] = me;
          nbrwgt[nnbrs++] = 1;
        }
      }
      /* Try to assign it first to the domain with most things in common */
      j = iamax(nnbrs, nbrwgt);
      if (pwgts[nbrind[j]] < maxpwgt) {
        epart[i] = nbrind[j];
      }
      else {
        /* If that fails, assign it to a light domain */
        for (j=0; j<nnbrs; j++) {
          if (pwgts[nbrind[j]] < maxpwgt) {
            epart[i] = nbrind[j];
            break;
          }
        }
        if (j == nnbrs)
          epart[i] = nbrind[iamax(nnbrs, nbrwgt)];
      }
      pwgts[epart[i]]++;
    }
  }

  if (*numflag == 1)
    ChangeMesh2FNumbering2((*ne)*esize, elmnts, *ne, *nn, epart, npart);

  GKfree((void**) &xadj, (void**) &adjncy, (void**) &pwgts, LTERM);

}


/*************************************************************************
* This function partitions a finite element mesh by partitioning its dual
* graph using KMETIS and then assigning nodes in a load balanced fashion.
**************************************************************************/
void METIS_PartMeshDual(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag,
                        int *nparts, int *edgecut, idxtype *epart, idxtype *npart)
{
  int i, j, k, me;
  idxtype *xadj, *adjncy, *pwgts, *nptr, *nind;
  int options[10], pnumflag=0, wgtflag=0;
  int nnbrs, nbrind[200], nbrwgt[200], maxpwgt;
  int esize, esizes[] = {-1, 3, 4, 8, 4};

  esize = esizes[*etype];

  if (*numflag == 1)
    ChangeMesh2CNumbering((*ne)*esize, elmnts);

  xadj = idxmalloc(*ne+1, "METIS_MESHPARTNODAL: xadj");
  adjncy = idxmalloc(esize*(*ne), "METIS_MESHPARTNODAL: adjncy");

  METIS_MeshToDual(ne, nn, elmnts, etype, &pnumflag, xadj, adjncy);

  options[0] = 0;
  METIS_PartGraphKway(ne, xadj, adjncy, NULL, NULL, &wgtflag, &pnumflag, nparts, options, edgecut, epart);

  /* Construct the node-element list */
  nptr = idxsmalloc(*nn+1, 0, "METIS_MESHPARTDUAL: nptr");
  for (j=esize*(*ne), i=0; i<j; i++)
    nptr[elmnts[i]]++;
  MAKECSR(i, *nn, nptr);

  nind = idxmalloc(nptr[*nn], "METIS_MESHPARTDUAL: nind");
  for (k=i=0; i<(*ne); i++) {
    for (j=0; j<esize; j++, k++)
      nind[nptr[elmnts[k]]++] = i;
  }
  for (i=(*nn); i>0; i--)
    nptr[i] = nptr[i-1];
  nptr[0] = 0;


  /* OK, now compute a nodal partition based on the element partition npart */
  idxset(*nn, -1, npart);
  pwgts = idxsmalloc(*nparts, 0, "METIS_MESHPARTDUAL: pwgts");
  for (i=0; i<*nn; i++) {
    me = epart[nind[nptr[i]]];
    for (j=nptr[i]+1; j<nptr[i+1]; j++) {
      if (epart[nind[j]] != me)
        break;
    }
    if (j == nptr[i+1]) {
      npart[i] = me;
      pwgts[me]++;
    }
  }

  maxpwgt = (int) 1.03*(*nn)/(*nparts);
  for (i=0; i<*nn; i++) {
    if (npart[i] == -1) { /* Assign the boundary element */
      nnbrs = 0;
      for (j=nptr[i]; j<nptr[i+1]; j++) {
        me = epart[nind[j]];
        for (k=0; k<nnbrs; k++) {
          if (nbrind[k] == me) {
            nbrwgt[k]++;
            break;
          }
        }
        if (k == nnbrs) {
          nbrind[nnbrs] = me;
          nbrwgt[nnbrs++] = 1;
        }
      }
      /* Try to assign it first to the domain with most things in common */
      j = iamax(nnbrs, nbrwgt);
      if (pwgts[nbrind[j]] < maxpwgt) {
        npart[i] = nbrind[j];
      }
      else {
        /* If that fails, assign it to a light domain */
        npart[i] = nbrind[0];
        for (j=0; j<nnbrs; j++) {
          if (pwgts[nbrind[j]] < maxpwgt) {
            npart[i] = nbrind[j];
            break;
          }
        }
      }
      pwgts[npart[i]]++;
    }
  }

  if (*numflag == 1)
    ChangeMesh2FNumbering2((*ne)*esize, elmnts, *ne, *nn, epart, npart);

  GKfree((void**) &xadj, (void**) &adjncy, (void**) &pwgts, (void**) &nptr, (void**) &nind, LTERM);

}
