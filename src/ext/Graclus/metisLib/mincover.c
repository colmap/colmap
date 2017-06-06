/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mincover.c
 *
 * This file implements the minimum cover algorithm
 *
 * Started 8/1/97
 * George
 *
 * $Id: mincover.c,v 1.1 1998/11/27 17:59:22 karypis Exp $
 */

#include "metis.h"

/*************************************************************************
* Constants used by mincover algorithm
**************************************************************************/
#define INCOL 10
#define INROW 20
#define VC 1
#define SC 2
#define HC 3
#define VR 4
#define SR 5
#define HR 6


/*************************************************************************
* This function returns the min-cover of a bipartite graph.
* The algorithm used is due to Hopcroft and Karp as modified by Duff etal
* adj: the adjacency list of the bipartite graph
*       asize: the number of vertices in the first part of the bipartite graph
* bsize-asize: the number of vertices in the second part
*        0..(asize-1) > A vertices
*        asize..bsize > B vertices
*
* Returns:
*  cover : the actual cover (array)
*  csize : the size of the cover
**************************************************************************/
void MinCover(idxtype *xadj, idxtype *adjncy, int asize, int bsize, idxtype *cover, int *csize)
{
  int i, j;
  idxtype *mate, *queue, *flag, *level, *lst;
  int fptr, rptr, lstptr;
  int row, maxlevel, col;

  mate = idxsmalloc(bsize, -1, "MinCover: mate");
  flag = idxmalloc(bsize, "MinCover: flag");
  level = idxmalloc(bsize, "MinCover: level");
  queue = idxmalloc(bsize, "MinCover: queue");
  lst = idxmalloc(bsize, "MinCover: lst");

  /* Get a cheap matching */
  for (i=0; i<asize; i++) {
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (mate[adjncy[j]] == -1) {
        mate[i] = adjncy[j];
        mate[adjncy[j]] = i;
        break;
      }
    }
  }

  /* Get into the main loop */
  while (1) {
    /* Initialization */
    fptr = rptr = 0;   /* Empty Queue */
    lstptr = 0;        /* Empty List */
    for (i=0; i<bsize; i++) {
      level[i] = -1;
      flag[i] = 0;
    }
    maxlevel = bsize;

    /* Insert free nodes into the queue */
    for (i=0; i<asize; i++)
      if (mate[i] == -1) {
        queue[rptr++] = i;
        level[i] = 0;
      }

    /* Perform the BFS */
    while (fptr != rptr) {
      row = queue[fptr++];
      if (level[row] < maxlevel) {
        flag[row] = 1;
        for (j=xadj[row]; j<xadj[row+1]; j++) {
          col = adjncy[j];
          if (!flag[col]) {  /* If this column has not been accessed yet */
            flag[col] = 1;
            if (mate[col] == -1) { /* Free column node was found */
              maxlevel = level[row];
              lst[lstptr++] = col;
            }
            else { /* This column node is matched */
              if (flag[mate[col]])
                printf("\nSomething wrong, flag[%d] is 1",mate[col]);
              queue[rptr++] = mate[col];
              level[mate[col]] = level[row] + 1;
            }
          }
        }
      }
    }

    if (lstptr == 0)
      break;   /* No free columns can be reached */

    /* Perform restricted DFS from the free column nodes */
    for (i=0; i<lstptr; i++)
      MinCover_Augment(xadj, adjncy, lst[i], mate, flag, level, maxlevel);
  }

  MinCover_Decompose(xadj, adjncy, asize, bsize, mate, cover, csize);

  GKfree((void**) &mate, (void**) &flag, (void**) &level, (void**) &queue, (void**) &lst, LTERM);

}


/*************************************************************************
* This function perfoms a restricted DFS and augments matchings
**************************************************************************/
int MinCover_Augment(idxtype *xadj, idxtype *adjncy, int col, idxtype *mate, idxtype *flag, idxtype *level, int maxlevel)
{
  int i;
  int row = -1;
  int status;

  flag[col] = 2;
  for (i=xadj[col]; i<xadj[col+1]; i++) {
    row = adjncy[i];

    if (flag[row] == 1) { /* First time through this row node */
      if (level[row] == maxlevel) {  /* (col, row) is an edge of the G^T */
        flag[row] = 2;  /* Mark this node as being visited */
        if (maxlevel != 0)
          status = MinCover_Augment(xadj, adjncy, mate[row], mate, flag, level, maxlevel-1);
        else
          status = 1;

        if (status) {
          mate[col] = row;
          mate[row] = col;
          return 1;
        }
      }
    }
  }

  return 0;
}



/*************************************************************************
* This function performs a coarse decomposition and determines the
* min-cover.
* REF: Pothen ACMTrans. on Amth Software
**************************************************************************/
void MinCover_Decompose(idxtype *xadj, idxtype *adjncy, int asize, int bsize, idxtype *mate, idxtype *cover, int *csize)
{
  int i, k;
  idxtype *where;
  int card[10];

  where = idxmalloc(bsize, "MinCover_Decompose: where");
  for (i=0; i<10; i++)
    card[i] = 0;

  for (i=0; i<asize; i++)
    where[i] = SC;
  for (; i<bsize; i++)
    where[i] = SR;

  for (i=0; i<asize; i++)
    if (mate[i] == -1)
      MinCover_ColDFS(xadj, adjncy, i, mate, where, INCOL);
  for (; i<bsize; i++)
    if (mate[i] == -1)
      MinCover_RowDFS(xadj, adjncy, i, mate, where, INROW);

  for (i=0; i<bsize; i++)
    card[where[i]]++;

  k = 0;
  if (abs(card[VC]+card[SC]-card[HR]) < abs(card[VC]-card[SR]-card[HR])) {  /* S = VC+SC+HR */
    /* printf("%d %d ",vc+sc, hr); */
    for (i=0; i<bsize; i++)
      if (where[i] == VC || where[i] == SC || where[i] == HR)
        cover[k++] = i;
  }
  else {  /* S = VC+SR+HR */
    /* printf("%d %d ",vc, hr+sr); */
    for (i=0; i<bsize; i++)
      if (where[i] == VC || where[i] == SR || where[i] == HR)
        cover[k++] = i;
  }

  *csize = k;
  free(where);

}


/*************************************************************************
* This function perfoms a dfs starting from an unmatched col node
* forming alternate paths
**************************************************************************/
void MinCover_ColDFS(idxtype *xadj, idxtype *adjncy, int root, idxtype *mate, idxtype *where, int flag)
{
  int i;

  if (flag == INCOL) {
    if (where[root] == HC)
      return;
    where[root] = HC;
    for (i=xadj[root]; i<xadj[root+1]; i++)
      MinCover_ColDFS(xadj, adjncy, adjncy[i], mate, where, INROW);
  }
  else {
    if (where[root] == HR)
      return;
    where[root] = HR;
    if (mate[root] != -1)
      MinCover_ColDFS(xadj, adjncy, mate[root], mate, where, INCOL);
  }

}

/*************************************************************************
* This function perfoms a dfs starting from an unmatched col node
* forming alternate paths
**************************************************************************/
void MinCover_RowDFS(idxtype *xadj, idxtype *adjncy, int root, idxtype *mate, idxtype *where, int flag)
{
  int i;

  if (flag == INROW) {
    if (where[root] == VR)
      return;
    where[root] = VR;
    for (i=xadj[root]; i<xadj[root+1]; i++)
      MinCover_RowDFS(xadj, adjncy, adjncy[i], mate, where, INCOL);
  }
  else {
    if (where[root] == VC)
      return;
    where[root] = VC;
    if (mate[root] != -1)
      MinCover_RowDFS(xadj, adjncy, mate[root], mate, where, INROW);
  }

}



