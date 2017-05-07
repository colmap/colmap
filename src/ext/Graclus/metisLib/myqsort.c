/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * myqsort.c
 *
 * This file contains a fast idxtype increasing qsort algorithm.
 * Addopted from TeX
 *
 * Started 10/18/96
 * George
 *
 * $Id: myqsort.c,v 1.1 1998/11/27 17:59:27 karypis Exp $
 */

#include "metis.h"			/* only for type declarations */

#define		THRESH		1	/* threshold for insertion */
#define		MTHRESH		6	/* threshold for median */




static void siqst(idxtype *, idxtype *);
static void iiqst(int *, int *);
static void keyiqst(KeyValueType *, KeyValueType *);
static void keyvaliqst(KeyValueType *, KeyValueType *);


/*************************************************************************
* Entry point of idxtype increasing sort
**************************************************************************/
void iidxsort(int n, idxtype *base)
{
  register idxtype *i;
  register idxtype *j;
  register idxtype *lo;
  register idxtype *hi;
  register idxtype *min;
  register idxtype c;
  idxtype *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    siqst(base, max);
    hi = base + THRESH;
  }
  else
    hi = max;

  for (j = lo = base; lo++ < hi;) {
    if (*j > *lo)
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  for (min = base; (hi = min += 1) < max;) {
    while (*(--hi) > *min);
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }
}

static void siqst(idxtype *base, idxtype *max)
{
  register idxtype *i;
  register idxtype *j;
  register idxtype *jj;
  register idxtype *mid;
  register int ii;
  register idxtype c;
  idxtype *tmp;
  int lo;
  int hi;

  lo = max - base;		/* number of elements as idxtype */
  do {
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (*base > *mid ? base : mid);
      tmp = max - 1;
      if (*j > *tmp) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (*j < *tmp)
          j = tmp;
      }

      if (j != mid) {  /* SWAP */
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && *i <= *mid)
        i++;
      while (j > mid) {
        if (*mid <= *j) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid)
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    i = (j = mid) + 1;
    if ((lo = j - base) <= (hi = max - i)) {
      if (lo >= THRESH)
        siqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        siqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}





/*************************************************************************
* Entry point of int increasing sort
**************************************************************************/
void iintsort(int n, int *base)
{
  register int *i;
  register int *j;
  register int *lo;
  register int *hi;
  register int *min;
  register int c;
  int *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    iiqst(base, max);
    hi = base + THRESH;
  }
  else
    hi = max;

  for (j = lo = base; lo++ < hi;) {
    if (*j > *lo)
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  for (min = base; (hi = min += 1) < max;) {
    while (*(--hi) > *min);
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }
}


static void iiqst(int *base, int *max)
{
  register int *i;
  register int *j;
  register int *jj;
  register int *mid;
  register int ii;
  register int c;
  int *tmp;
  int lo;
  int hi;

  lo = max - base;		/* number of elements as ints */
  do {
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (*base > *mid ? base : mid);
      tmp = max - 1;
      if (*j > *tmp) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (*j < *tmp)
          j = tmp;
      }

      if (j != mid) {  /* SWAP */
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && *i <= *mid)
        i++;
      while (j > mid) {
        if (*mid <= *j) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid)
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    i = (j = mid) + 1;
    if ((lo = j - base) <= (hi = max - i)) {
      if (lo >= THRESH)
        iiqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        iiqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}





/*************************************************************************
* Entry point of KeyVal increasing sort, ONLY key part
**************************************************************************/
void ikeysort(int n, KeyValueType *base)
{
  register KeyValueType *i;
  register KeyValueType *j;
  register KeyValueType *lo;
  register KeyValueType *hi;
  register KeyValueType *min;
  register KeyValueType c;
  KeyValueType *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    keyiqst(base, max);
    hi = base + THRESH;
  }
  else
    hi = max;

  for (j = lo = base; lo++ < hi;) {
    if (j->key > lo->key)
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  for (min = base; (hi = min += 1) < max;) {
    while ((--hi)->key > min->key);
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }

  /* Sanity check */
  {
    int i;
    for (i=0; i<n-1; i++)
      if (base[i].key > base[i+1].key)
        printf("Something went wrong!\n");
  }
}


static void keyiqst(KeyValueType *base, KeyValueType *max)
{
  register KeyValueType *i;
  register KeyValueType *j;
  register KeyValueType *jj;
  register KeyValueType *mid;
  register KeyValueType c;
  KeyValueType *tmp;
  int lo;
  int hi;

  lo = (max - base)>>1;		/* number of elements as KeyValueType */
  do {
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (base->key > mid->key ? base : mid);
      tmp = max - 1;
      if (j->key > tmp->key) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (j->key < tmp->key)
          j = tmp;
      }

      if (j != mid) {  /* SWAP */
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && i->key <= mid->key)
        i++;
      while (j > mid) {
        if (mid->key <= j->key) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid)
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    i = (j = mid) + 1;
    if ((lo = (j - base)>>1) <= (hi = (max - i)>>1)) {
      if (lo >= THRESH)
        keyiqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        keyiqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}




/*************************************************************************
* Entry point of KeyVal increasing sort, BOTH key and val part
**************************************************************************/
void ikeyvalsort(int n, KeyValueType *base)
{
  register KeyValueType *i;
  register KeyValueType *j;
  register KeyValueType *lo;
  register KeyValueType *hi;
  register KeyValueType *min;
  register KeyValueType c;
  KeyValueType *max;

  if (n <= 1)
    return;

  max = base + n;

  if (n >= THRESH) {
    keyvaliqst(base, max);
    hi = base + THRESH;
  }
  else
    hi = max;

  for (j = lo = base; lo++ < hi;) {
    if ((j->key > lo->key) || (j->key == lo->key && j->val > lo->val))
      j = lo;
  }
  if (j != base) { /* swap j into place */
    c = *base;
    *base = *j;
    *j = c;
  }

  for (min = base; (hi = min += 1) < max;) {
    while ((--hi)->key > min->key || (hi->key == min->key && hi->val > min->val));
    if ((hi += 1) != min) {
      for (lo = min + 1; --lo >= min;) {
	c = *lo;
	for (i = j = lo; (j -= 1) >= hi; i = j)
	   *i = *j;
	*i = c;
      }
    }
  }
}


static void keyvaliqst(KeyValueType *base, KeyValueType *max)
{
  register KeyValueType *i;
  register KeyValueType *j;
  register KeyValueType *jj;
  register KeyValueType *mid;
  register KeyValueType c;
  KeyValueType *tmp;
  int lo;
  int hi;

  lo = (max - base)>>1;		/* number of elements as KeyValueType */
  do {
    mid = base + ((unsigned) lo>>1);
    if (lo >= MTHRESH) {
      j = (base->key > mid->key || (base->key == mid->key && base->val > mid->val) ? base : mid);
      tmp = max - 1;
      if (j->key > tmp->key || (j->key == tmp->key && j->val > tmp->val)) {
        j = (j == base ? mid : base); /* switch to first loser */
        if (j->key < tmp->key || (j->key == tmp->key && j->val < tmp->val))
          j = tmp;
      }

      if (j != mid) {  /* SWAP */
        c = *mid;
        *mid = *j;
        *j = c;
      }
    }

    /* Semi-standard quicksort partitioning/swapping */
    for (i = base, j = max - 1;;) {
      while (i < mid && (i->key < mid->key || (i->key == mid->key && i->val <= mid->val)))
        i++;
      while (j > mid) {
        if (mid->key < j->key || (mid->key == j->key && mid->val <= j->val)) {
          j--;
          continue;
        }
        tmp = i + 1;	/* value of i after swap */
        if (i == mid) 	/* j <-> mid, new mid is j */
          mid = jj = j;
        else 		/* i <-> j */
          jj = j--;
        goto swap;
      }

      if (i == mid)
	break;
      else {		/* i <-> mid, new mid is i */
        jj = mid;
        tmp = mid = i;	/* value of i after swap */
        j--;
      }
swap:
      c = *i;
      *i = *jj;
      *jj = c;
      i = tmp;
    }

    i = (j = mid) + 1;
    if ((lo = (j - base)>>1) <= (hi = (max - i)>>1)) {
      if (lo >= THRESH)
        keyvaliqst(base, j);
      base = i;
      lo = hi;
    }
    else {
      if (hi >= THRESH)
        keyvaliqst(i, max);
      max = j;
    }
  } while (lo >= THRESH);
}
