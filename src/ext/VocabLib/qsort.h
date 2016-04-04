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
 * implied, of Cornell University.
 *
 */

#ifndef __qsort_h__
#define __qsort_h__

#ifdef __cplusplus
extern "C" {
#endif

/* Set whether we should sort in ascending or descending order */
void qsort_ascending();
void qsort_descending();

/* Sorts the array of doubles `arr' (of length n) and puts the
 * corresponding permutation in `perm' */
void qsort_perm(int n, double *arr, int *perm);

/* Permute the array `arr' given permutation `perm' */
void permute_dbl(int n, double *arr, int *perm);
void permute(int n, int size, void *arr, int *perm);

/* Find the median in a set of doubles */
double median(int n, double *arr);
double median_copy(int n, double *arr);

/* Find the kth element in an unordered list of doubles (changes the
 * array) */
double kth_element(int n, int k, double *arr);
/* Same as above, doesn't change the array */
double kth_element_copy(int n, int k, double *arr);

#ifdef __cplusplus
}
#endif

#endif /* __qsort_h__ */
