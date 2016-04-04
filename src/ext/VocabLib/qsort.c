/* 
 * Copyright 2011-2012 Noah Snavely, Cornell University
 * (snavely@cs.cornell.edu).  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY NOAH SNAVELY ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NOAH SNAVELY OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * 
 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of Noah Snavely.
 *
 */

/* qsort.c */
/* Contains a routine for sorting a list of floating point numbers and
 * returning the permutation that would map the original list to the
 * sorted list */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "qsort.h"

typedef enum {
    QSORT_ASCENDING,
    QSORT_DESCENDING
} qsort_order_t;

static qsort_order_t qsort_order = QSORT_DESCENDING;

/* Set whether we should sort in ascending or descending order */
void qsort_ascending() 
{
    qsort_order = QSORT_ASCENDING;
}

void qsort_descending() 
{
    qsort_order = QSORT_DESCENDING;
}

void qsort_perm_r(int n, double *arr, int *perm);

/* Sorts the array of doubles `arr' (of length n) and puts the
 * corresponding permutation in `perm' */
void qsort_perm(int n, double *arr, int *perm) {
    int i;

    /* Create the identity permutation */
    for (i = 0; i < n; i++) 
	perm[i] = i;
    
    qsort_perm_r(n, arr, perm);
}

#define SORT_ASCENDING
/* #define SORT_DESCENDING */

void qsort_perm_r(int n, double *arr, int *perm) {
    int pivot_idx;
    double *r, *l;
    double pivot;
    double *split;

    if (n <= 1)
	return;
    
    /* Pick the pivot */
    pivot_idx = n / 2;
    pivot = arr[pivot_idx];
    l = arr;
    r = arr + (n - 1);
    
    while (l < r) {
	if (qsort_order == QSORT_ASCENDING) {
	    if (*l >= pivot && *r <= pivot) {
		/* Swap */
		int lidx = (int) (l - arr);
		int ridx = (int) (r - arr);
		double tmp = *l;
		int tmpidx = perm[lidx];

		*l = *r;
		*r = tmp;

		perm[lidx] = perm[ridx];
		perm[ridx] = tmpidx;
		
		l++;
	    } else if (*l < pivot) {
		l++;
	    } else if (*r > pivot) {
		r--;
	    } else {
		printf("Execution should not reach this point\n");
                return;
	    }
	} else if (qsort_order == QSORT_DESCENDING) {
	    if (*l <= pivot && *r >= pivot) {
		/* Swap */
		int lidx = (int) (l - arr);
		int ridx = (int) (r - arr);
		double tmp = *l;
		int tmpidx = perm[lidx];

		*l = *r;
		*r = tmp;

		perm[lidx] = perm[ridx];
		perm[ridx] = tmpidx;
	    
		l++;
	    } else if (*l > pivot) {
		l++;
	    } else if (*r < pivot) {
		r--;
	    } else {
		printf("Execution should not reach this point\n");
                return;
	    }
	} else {
	    printf("[qsort_perm_r] Unknown qsort order\n");
	}
    }

    /* At this point l == r */
    split = l;

    /* Sort the left subarray */
    qsort_perm_r((int) (split - arr), arr, perm);

    /* Sort the right subarray */
    qsort_perm_r(n - (int) (split - arr), split, perm + (split - arr));
}

/* Find the median in a set of doubles */
double median(int n, double *arr) {
    return kth_element(n, n / 2, arr);
}

/* Find the median in a set of doubles (*/
double median_copy(int n, double *arr) {
    return kth_element_copy(n, n / 2, arr);
}

/* Find the kth element without changing the array */
double kth_element_copy(int n, int k, double *arr) {
    double *arr_copy = (double *)malloc(sizeof(double) * n);
    double kth_best;

    memcpy(arr_copy, arr, sizeof(double) * n);
    kth_best = kth_element(n, k, arr_copy);
    free(arr_copy);

    return kth_best;
}

static int partition(int n, double *arr) {
    int pivot_idx = n / 2;
    double pivot = arr[pivot_idx];
    double tmp;
    int i, store_index;

    /* Swap pivot and end of array */
    tmp = arr[n-1];
    arr[n-1] = pivot;
    arr[pivot_idx] = tmp;
    
    store_index = 0;
    for (i = 0; i < n-1; i++) {
	if (arr[i] < pivot) {
	    tmp = arr[store_index];
	    arr[store_index] = arr[i];
	    arr[i] = tmp;
	    store_index++;
	}
    }
    
    tmp = arr[store_index];
    arr[store_index] = arr[n-1];
    arr[n-1] = tmp;
    
    return store_index;
}
/* Find the kth element in an unordered list of doubles */
double kth_element(int n, int k, double *arr) {
    if (k >= n) {
	printf("[kth_element] Error: k should be < n\n");
	return 0.0;
    } else {
	int split = partition(n, arr);
	if (k == split)
	    return arr[split];
	else if (k < split)
	    return kth_element(split, k, arr);
	else
	    return kth_element(n - split - 1, k - split - 1, arr + split + 1);
    }
}

#if 0
/* Find the kth element in an unordered list of doubles */
double kth_element(int n, int k, double *arr) {
    if (k >= n) {
	printf("[kth_element] Error: k should be < n\n");
	return 0.0;
    } else {
	/* Pick a pivot */
	int pivot_idx = n / 2;
	double pivot = arr[pivot_idx];
	double *l = arr, *r = arr + n - 1, *split;
	int split_idx;

	while (l < r) {
	    if (*l >= pivot && *r <= pivot) {
		/* Swap */
		double tmp = *l;
		
		*l = *r;
		*r = tmp;

		l++;
		r--;
	    } else if (*l < pivot) {
		l++;
	    } else if (*r > pivot) {
		r--;
	    } else {
		printf("Execution should not reach this point\n");
	    }
	}

	/* At this point l == r */
	split = l;
	split_idx = (int)(l - arr);

	if (split_idx == k) {
	    return arr[split_idx];
	} else if (split_idx < k) {
	    return kth_element(n - split_idx, k - split_idx, split);
	} else {
	    return kth_element(split_idx, k, arr);
	}
    }
}
#endif

/* Permute the array `arr' given permutation `perm' */
void permute_dbl(int n, double *arr, int *perm) {
    double *tmparr = malloc(sizeof(double) * n);
    int i;

    for (i = 0; i < n; i++) 
	tmparr[i] = arr[perm[i]];
    
    memcpy(arr, tmparr, sizeof(double) * n);

    free(tmparr);
}

void permute(int n, int size, void *arr, int *perm) {
    void *tmparr = malloc(size * n);
    int i;

    for (i = 0; i < n; i++) 
	memcpy((char *) tmparr + i * size, (char *) arr + perm[i] * size, size);
    
    memcpy(arr, tmparr, size * n);

    free(tmparr);
}

/* Returns true if the given array is sorted */
int is_sorted(int n, double *arr) {
    int i;
    
    for (i = 0; i < n - 1; i++) {
	if (arr[i] > arr[i+1])
	    return 0;
    }
    
    return 1;
}

#if 0
int main() {
    int n = 2048;
    double *arr = malloc(sizeof(double) * n);
    int *perm = malloc(sizeof(int) * n);
    double *arr2 = malloc(sizeof(double) * n);
    int i;

    for (i = 0; i < n; i++)
	arr[i] = ((double) rand()) / RAND_MAX;

    memcpy(arr2, arr, sizeof(double) * n);

    qsort_perm(n, arr, perm);

    printf("%s\n", is_sorted(n, arr) ? "List is sorted" : "List is not sorted");
    permute(n, sizeof(double), arr2, perm);
    printf("%s\n", is_sorted(n, arr2) ? "List is sorted" : "List is not sorted");

    free(arr);
    free(arr2);
    free(perm);

    return 0;
}
#endif
