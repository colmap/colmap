/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mesh.c
 *
 * This file contains routines for converting 3D and 4D finite element
 * meshes into dual or nodal graphs
 *
 * Started 8/18/97
 * George
 *
 * $Id: mesh.c,v 1.1 1998/11/27 17:59:20 karypis Exp $
 *
 */

#include "metis.h"

/*****************************************************************************
* This function creates a graph corresponding to the dual of a finite element
* mesh. At this point the supported elements are triangles, tetrahedrons, and
* bricks.
******************************************************************************/
void METIS_MeshToDual(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag,
                      idxtype *dxadj, idxtype *dadjncy)
{
  int esizes[] = {-1, 3, 4, 8, 4};

  if (*numflag == 1)
    ChangeMesh2CNumbering((*ne)*esizes[*etype], elmnts);

  GENDUALMETIS(*ne, *nn, *etype, elmnts, dxadj, dadjncy);

  if (*numflag == 1)
    ChangeMesh2FNumbering((*ne)*esizes[*etype], elmnts, *ne, dxadj, dadjncy);
}


/*****************************************************************************
* This function creates a graph corresponding to the finite element mesh.
* At this point the supported elements are triangles, tetrahedrons.
******************************************************************************/
void METIS_MeshToNodal(int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag,
                       idxtype *dxadj, idxtype *dadjncy)
{
  int esizes[] = {-1, 3, 4, 8, 4};

  if (*numflag == 1)
    ChangeMesh2CNumbering((*ne)*esizes[*etype], elmnts);

  switch (*etype) {
    case 1:
      TRINODALMETIS(*ne, *nn, elmnts, dxadj, dadjncy);
      break;
    case 2:
      TETNODALMETIS(*ne, *nn, elmnts, dxadj, dadjncy);
      break;
    case 3:
      HEXNODALMETIS(*ne, *nn, elmnts, dxadj, dadjncy);
      break;
    case 4:
      QUADNODALMETIS(*ne, *nn, elmnts, dxadj, dadjncy);
      break;
  }

  if (*numflag == 1)
    ChangeMesh2FNumbering((*ne)*esizes[*etype], elmnts, *nn, dxadj, dadjncy);
}



/*****************************************************************************
* This function creates the dual of a finite element mesh
******************************************************************************/
void GENDUALMETIS(int nelmnts, int nvtxs, int etype, idxtype *elmnts, idxtype *dxadj, idxtype *dadjncy)
{
   int i, j, jj, k, kk, kkk, l, m, n, nedges, mask;
   idxtype *nptr, *nind;
   idxtype *mark, ind[200], wgt[200];
   int esize, esizes[] = {-1, 3, 4, 8, 4},
       mgcnum, mgcnums[] = {-1, 2, 3, 4, 2};

   mask = (1<<11)-1;
   mark = idxsmalloc(mask+1, -1, "GENDUALMETIS: mark");

   /* Get the element size and magic number for the particular element */
   esize = esizes[etype];
   mgcnum = mgcnums[etype];

   /* Construct the node-element list first */
   nptr = idxsmalloc(nvtxs+1, 0, "GENDUALMETIS: nptr");
   for (j=esize*nelmnts, i=0; i<j; i++)
     nptr[elmnts[i]]++;
   MAKECSR(i, nvtxs, nptr);

   nind = idxmalloc(nptr[nvtxs], "GENDUALMETIS: nind");
   for (k=i=0; i<nelmnts; i++) {
     for (j=0; j<esize; j++, k++)
       nind[nptr[elmnts[k]]++] = i;
   }
   for (i=nvtxs; i>0; i--)
     nptr[i] = nptr[i-1];
   nptr[0] = 0;

   for (i=0; i<nelmnts; i++)
     dxadj[i] = esize*i;

   for (i=0; i<nelmnts; i++) {
     for (m=j=0; j<esize; j++) {
       n = elmnts[esize*i+j];
       for (k=nptr[n+1]-1; k>=nptr[n]; k--) {
         if ((kk = nind[k]) <= i)
           break;

         kkk = kk&mask;
         if ((l = mark[kkk]) == -1) {
           ind[m] = kk;
           wgt[m] = 1;
           mark[kkk] = m++;
         }
         else if (ind[l] == kk) {
           wgt[l]++;
         }
         else {
           for (jj=0; jj<m; jj++) {
             if (ind[jj] == kk) {
               wgt[jj]++;
               break;
             }
           }
           if (jj == m) {
             ind[m] = kk;
             wgt[m++] = 1;
           }
         }
       }
     }
     for (j=0; j<m; j++) {
       if (wgt[j] == mgcnum) {
         k = ind[j];
         dadjncy[dxadj[i]++] = k;
         dadjncy[dxadj[k]++] = i;
       }
       mark[ind[j]&mask] = -1;
     }
   }

   /* Go and consolidate the dxadj and dadjncy */
   for (j=i=0; i<nelmnts; i++) {
     for (k=esize*i; k<dxadj[i]; k++, j++)
       dadjncy[j] = dadjncy[k];
     dxadj[i] = j;
   }
   for (i=nelmnts; i>0; i--)
     dxadj[i] = dxadj[i-1];
   dxadj[0] = 0;

   free(mark);
   free(nptr);
   free(nind);

}




/*****************************************************************************
* This function creates the nodal graph of a finite element mesh
******************************************************************************/
void TRINODALMETIS(int nelmnts, int nvtxs, idxtype *elmnts, idxtype *dxadj, idxtype *dadjncy)
{
   int i, j, jj, k, kk, kkk, l, m, n, nedges;
   idxtype *nptr, *nind;
   idxtype *mark;

   /* Construct the node-element list first */
   nptr = idxsmalloc(nvtxs+1, 0, "TRINODALMETIS: nptr");
   for (j=3*nelmnts, i=0; i<j; i++)
     nptr[elmnts[i]]++;
   MAKECSR(i, nvtxs, nptr);

   nind = idxmalloc(nptr[nvtxs], "TRINODALMETIS: nind");
   for (k=i=0; i<nelmnts; i++) {
     for (j=0; j<3; j++, k++)
       nind[nptr[elmnts[k]]++] = i;
   }
   for (i=nvtxs; i>0; i--)
     nptr[i] = nptr[i-1];
   nptr[0] = 0;


   mark = idxsmalloc(nvtxs, -1, "TRINODALMETIS: mark");

   nedges = dxadj[0] = 0;
   for (i=0; i<nvtxs; i++) {
     mark[i] = i;
     for (j=nptr[i]; j<nptr[i+1]; j++) {
       for (jj=3*nind[j], k=0; k<3; k++, jj++) {
         kk = elmnts[jj];
         if (mark[kk] != i) {
           mark[kk] = i;
           dadjncy[nedges++] = kk;
         }
       }
     }
     dxadj[i+1] = nedges;
   }

   free(mark);
   free(nptr);
   free(nind);

}


/*****************************************************************************
* This function creates the nodal graph of a finite element mesh
******************************************************************************/
void TETNODALMETIS(int nelmnts, int nvtxs, idxtype *elmnts, idxtype *dxadj, idxtype *dadjncy)
{
   int i, j, jj, k, kk, kkk, l, m, n, nedges;
   idxtype *nptr, *nind;
   idxtype *mark;

   /* Construct the node-element list first */
   nptr = idxsmalloc(nvtxs+1, 0, "TETNODALMETIS: nptr");
   for (j=4*nelmnts, i=0; i<j; i++)
     nptr[elmnts[i]]++;
   MAKECSR(i, nvtxs, nptr);

   nind = idxmalloc(nptr[nvtxs], "TETNODALMETIS: nind");
   for (k=i=0; i<nelmnts; i++) {
     for (j=0; j<4; j++, k++)
       nind[nptr[elmnts[k]]++] = i;
   }
   for (i=nvtxs; i>0; i--)
     nptr[i] = nptr[i-1];
   nptr[0] = 0;


   mark = idxsmalloc(nvtxs, -1, "TETNODALMETIS: mark");

   nedges = dxadj[0] = 0;
   for (i=0; i<nvtxs; i++) {
     mark[i] = i;
     for (j=nptr[i]; j<nptr[i+1]; j++) {
       for (jj=4*nind[j], k=0; k<4; k++, jj++) {
         kk = elmnts[jj];
         if (mark[kk] != i) {
           mark[kk] = i;
           dadjncy[nedges++] = kk;
         }
       }
     }
     dxadj[i+1] = nedges;
   }

   free(mark);
   free(nptr);
   free(nind);

}


/*****************************************************************************
* This function creates the nodal graph of a finite element mesh
******************************************************************************/
void HEXNODALMETIS(int nelmnts, int nvtxs, idxtype *elmnts, idxtype *dxadj, idxtype *dadjncy)
{
   int i, j, jj, k, kk, kkk, l, m, n, nedges;
   idxtype *nptr, *nind;
   idxtype *mark;
   int table[8][3] = {1, 3, 4,
                      0, 2, 5,
                      1, 3, 6,
                      0, 2, 7,
                      0, 5, 7,
                      1, 4, 6,
                      2, 5, 7,
                      3, 4, 6};

   /* Construct the node-element list first */
   nptr = idxsmalloc(nvtxs+1, 0, "HEXNODALMETIS: nptr");
   for (j=8*nelmnts, i=0; i<j; i++)
     nptr[elmnts[i]]++;
   MAKECSR(i, nvtxs, nptr);

   nind = idxmalloc(nptr[nvtxs], "HEXNODALMETIS: nind");
   for (k=i=0; i<nelmnts; i++) {
     for (j=0; j<8; j++, k++)
       nind[nptr[elmnts[k]]++] = i;
   }
   for (i=nvtxs; i>0; i--)
     nptr[i] = nptr[i-1];
   nptr[0] = 0;


   mark = idxsmalloc(nvtxs, -1, "HEXNODALMETIS: mark");

   nedges = dxadj[0] = 0;
   for (i=0; i<nvtxs; i++) {
     mark[i] = i;
     for (j=nptr[i]; j<nptr[i+1]; j++) {
       jj=8*nind[j];
       for (k=0; k<8; k++) {
         if (elmnts[jj+k] == i)
           break;
       }
       ASSERT(k != 8);

       /* You found the index, now go and put the 3 neighbors */
       kk = elmnts[jj+table[k][0]];
       if (mark[kk] != i) {
         mark[kk] = i;
         dadjncy[nedges++] = kk;
       }
       kk = elmnts[jj+table[k][1]];
       if (mark[kk] != i) {
         mark[kk] = i;
         dadjncy[nedges++] = kk;
       }
       kk = elmnts[jj+table[k][2]];
       if (mark[kk] != i) {
         mark[kk] = i;
         dadjncy[nedges++] = kk;
       }
     }
     dxadj[i+1] = nedges;
   }

   free(mark);
   free(nptr);
   free(nind);

}


/*****************************************************************************
* This function creates the nodal graph of a finite element mesh
******************************************************************************/
void QUADNODALMETIS(int nelmnts, int nvtxs, idxtype *elmnts, idxtype *dxadj, idxtype *dadjncy)
{
   int i, j, jj, k, kk, kkk, l, m, n, nedges;
   idxtype *nptr, *nind;
   idxtype *mark;
   int table[4][2] = {1, 3,
                      0, 2,
                      1, 3,
                      0, 2};

   /* Construct the node-element list first */
   nptr = idxsmalloc(nvtxs+1, 0, "QUADNODALMETIS: nptr");
   for (j=4*nelmnts, i=0; i<j; i++)
     nptr[elmnts[i]]++;
   MAKECSR(i, nvtxs, nptr);

   nind = idxmalloc(nptr[nvtxs], "QUADNODALMETIS: nind");
   for (k=i=0; i<nelmnts; i++) {
     for (j=0; j<4; j++, k++)
       nind[nptr[elmnts[k]]++] = i;
   }
   for (i=nvtxs; i>0; i--)
     nptr[i] = nptr[i-1];
   nptr[0] = 0;


   mark = idxsmalloc(nvtxs, -1, "QUADNODALMETIS: mark");

   nedges = dxadj[0] = 0;
   for (i=0; i<nvtxs; i++) {
     mark[i] = i;
     for (j=nptr[i]; j<nptr[i+1]; j++) {
       jj=4*nind[j];
       for (k=0; k<4; k++) {
         if (elmnts[jj+k] == i)
           break;
       }
       ASSERT(k != 4);

       /* You found the index, now go and put the 2 neighbors */
       kk = elmnts[jj+table[k][0]];
       if (mark[kk] != i) {
         mark[kk] = i;
         dadjncy[nedges++] = kk;
       }
       kk = elmnts[jj+table[k][1]];
       if (mark[kk] != i) {
         mark[kk] = i;
         dadjncy[nedges++] = kk;
       }
     }
     dxadj[i+1] = nedges;
   }

   free(mark);
   free(nptr);
   free(nind);

}
