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

/* kmeans.cpp */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans_kd.h"

/* Choose k numbers at random from 0 to n-1 */
static void choose(int n, int k, int *arr)
{
    int i;

    if (k > n) {
        printf("[choose] Error: k > n\n");
        return;
    }

    for (i = 0; i < k; i++) {
        while (1) {
            int idx = rand() % n;
            int j, redo = 0;

            for (j = 0; j < i; j++) {
                if (idx == arr[j]) {
                    redo = 1;
                    break;
                }
            }

            if (!redo) {
                arr[i] = idx;
                break;
            }
        }
    }
}

/* Copy 'dim' elements to array 'vec' from array 'v' */
static void fill_vector(double *vec, unsigned char *v, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
        vec[i] = (double) v[i];
}

/* Accumulate array 'v' (of dimension 'dim') into array 'acc' */
static void vec_accum(int dim, double *acc, unsigned char *v)
{
    int i;
    for (i = 0; i < dim; i++) {
        acc[i] += (double) v[i];
    }
}

/* Scale array 'v' (of length 'dim') by factor 'scale' */
static void vec_scale(int dim, double *v, double scale)
{
    int i;
    for (i = 0; i < dim; i++) {
        v[i] *= scale;
    }
}

/* Compute the difference of array 'a' and 'b' (of length 'dim'),
 * store in 'r' */
static void vec_diff(int dim, double *a, double *b, double *r)
{
    int i;
    for (i = 0; i < dim; i++) {
        r[i] = a[i] - b[i];
    }
}

/* Compute the squared length of an array 'v' (of length 'dim') */
static double vec_normsq(int dim, double *v)
{
    double norm = 0.0;
    int i;
    for (i = 0; i < dim; i++)
        norm += v[i] * v[i];

    return norm;
}

/* Function compute_means.
 * This function recomputes the means based on the current clustering
 * of the points.
 *
 * Inputs:
 *   n          : number of input descriptors
 *   dim        : dimension of each input descriptor
 *   k          : number of means
 *   v          : array of pointers to dim-dimensional descriptors
 *   clustering : current assignment of descriptors to means (should
 *                range between 0 and k-1)
 *
 * Output:
 *   means_out  : array of output means.  You need to fill this
 *                array.  The means should be concatenated into one
 *                long array of length k*dim.
 */
double compute_means(int n, int dim, int k, unsigned char **v,
                     unsigned int *clustering, double *means_out)
{
    int i;
    double max_change = 0.0;
    int *counts = (int *) malloc(sizeof(int) * k);

    for (i = 0; i < k * dim; i++)
        means_out[i] = 0.0;

    for (i = 0; i < k; i++)
        counts[i] = 0;

    for (i = 0; i < n; i++) {
        unsigned int cluster = clustering[i];
        vec_accum(dim, means_out + cluster * dim, v[i]);

        counts[cluster]++;
    }

    /* Normalize new means */
    for (i = 0; i < k; i++) {
        if (counts[i] == 0) {
            continue;
        }

        vec_scale(dim, means_out + i * dim, 1.0 / counts[i]);
    }

    free(counts);

    return max_change;
}

double compute_error(int n, int dim, int k, unsigned char **v,
                     double *means, unsigned int *clustering)
{
    int i, j;

    double error = 0;
    for (i = 0; i < n; i++) {
        unsigned int c = clustering[i];

        for (j = 0; j < dim; j++) {
            double d = means[c * dim + j] - v[i][j];
            error += d * d;
        }
    }

    return error;
}

/* Function compute_clustering.
 * This function recomputes the clustering based on the current means.
 *
 * Inputs:
 *   n          : number of input descriptors
 *   dim        : dimension of each input descriptor
 *   k          : number of means
 *   v          : array of pointers to dim-dimensional descriptors
 *   means      : current means, stored in a k*dim dimensional array
 *
 * Output:
 *   clustering : new assignment of descriptors to nearest means
 *                (should range between 0 and k-1)
 *   error_out  : total error of the new assignment
 *
 * Return value : return the number of points that changed assignment
 */
int compute_clustering(int n, int dim, int k, unsigned char **v,
                       double *means, unsigned int *clustering,
                       double &error_out)
{
    int i;
    double error = 0.0;

    int changed = 0;

    double *vec = (double *) malloc(sizeof(double) * dim);
    double *work = (double *) malloc(sizeof(double) * dim);

    for (i = 0; i < n; i++) {
        fill_vector(vec, v[i], dim);

        int j;
        double min_dist = DBL_MAX;
        unsigned int cluster = 0;

        for (j = 0; j < k; j++) {
            vec_diff(dim, vec, means + j * dim, work);
            double dist = vec_normsq(dim, work);

            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }

        error += min_dist;

        if (clustering[i] != cluster)
            changed++;

        clustering[i] = cluster;
    }

    free(vec);
    free(work);

    error_out = error;

    return changed;
}

/* Function kmeans.
 * Run kmeans clustering on a set of input descriptors.
 *
 * Inputs:
 *   n          : number of input descriptors
 *   dim        : dimension of each input descriptor
 *   k          : number of means to compute
 *   restarts   : number of random restarts to perform
 *   v          : array of pointers to dim-dimensional descriptors
 *
 * Output:
 *   means      : array of output means.  The means should be
 *                concatenated into one long array of length k*dim.
 *   clustering : assignment of descriptors to means (should
 *                range between 0 and k-1), stored as an array of
 *                length n.  clustering[i] contains the
 *                cluster ID for point i
 */
double kmeans(int n, int dim, int k, int restarts, unsigned char **v,
              double *means, unsigned int *clustering)
{
    int i;
    double min_error = DBL_MAX;

    double *means_curr, *means_new, *work;
    int *starts;
    unsigned int *clustering_curr;

    double changed_pct_threshold = 0.05; // 0.005;

    if (n <= k) {
        printf("[kmeans] Error: n <= k\n");
        return -1;
    }

    means_curr = (double *) malloc(sizeof(double) * dim * k);
    means_new = (double *) malloc(sizeof(double) * dim * k);
    clustering_curr = (unsigned int *) malloc(sizeof(unsigned int) * n);
    starts = (int *) malloc(sizeof(int) * k);
    work = (double *) malloc(sizeof(double) * dim);

    if (means_curr == NULL) {
        printf("[kmeans] Error allocating means_curr\n");
        exit(-1);
    }

    if (means_new == NULL) {
        printf("[kmeans] Error allocating means_new\n");
        exit(-1);
    }

    if (clustering_curr == NULL) {
        printf("[kmeans] Error allocating clustering_curr\n");
        exit(-1);
    }

    if (starts == NULL) {
        printf("[kmeans] Error allocating starts\n");
        exit(-1);
    }

    if (work == NULL) {
        printf("[kmeans] Error allocating work\n");
        exit(-1);
    }

    for (i = 0; i < restarts; i++) {
        int j;
        double max_change = 0.0;
        double error = 0.0;
        int round = 0;

        choose(n, k, starts);

        for (j = 0; j < k; j++) {
            fill_vector(means_curr + j * dim, v[starts[j]], dim);
        }

        /* Compute new assignments */
        int changed = 0;
        changed = compute_clustering_kd_tree(n, dim, k, v, means_curr,
                                             clustering_curr, error);

        double changed_pct = (double) changed / n;

        do {
            printf("Round %d: changed: %d\n", i, changed);
            fflush(stdout);

            /* Recompute means */
            max_change = compute_means(n, dim, k, v,
                                       clustering_curr, means_new);

            memcpy(means_curr, means_new, sizeof(double) * dim * k);

            /* Compute new assignments */
            changed = compute_clustering_kd_tree(n, dim, k, v, means_curr,
                                                 clustering_curr, error);

            changed_pct = (double) changed / n;

            round++;
        } while (changed_pct > changed_pct_threshold);

        max_change = compute_means(n, dim, k, v, clustering_curr, means_new);
        memcpy(means_curr, means_new, sizeof(double) * dim * k);

        if (error < min_error) {
            min_error = error;
            memcpy(means, means_curr, sizeof(double) * k * dim);
            memcpy(clustering, clustering_curr, sizeof(unsigned int) * n);
        }
    }


    free(means_curr);
    free(means_new);
    free(clustering_curr);
    free(starts);
    free(work);

    return compute_error(n, dim, k, v, means, clustering);
}
