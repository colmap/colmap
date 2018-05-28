/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * bucketsort.c
 *
 * This file contains code that implement a variety of counting sorting
 * algorithms
 *
 * Started 7/25/97
 * George
 *
 * $Id: bucketsort.c,v 1.1 1998/11/27 17:59:11 karypis Exp $
 *
 */

#include "metis.h"



/*************************************************************************
* This function uses simple counting sort to return a permutation array
* corresponding to the sorted order. The keys are assumed to start from
* 0 and they are positive.  This sorting is used during matching.
**************************************************************************/
void BucketSortKeysInc(int n, int max, idxtype *keys, idxtype *tperm, idxtype *perm)
{
  int i, ii;
  idxtype *counts;

  counts = idxsmalloc(max+2, 0, "BucketSortKeysInc: counts");

  for (i=0; i<n; i++)
    counts[keys[i]]++;
  MAKECSR(i, max+1, counts);

  for (ii=0; ii<n; ii++) {
    i = tperm[ii];
    perm[counts[keys[i]]++] = i;
  }

  free(counts);
}

