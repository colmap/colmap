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

/* kmeans_kd.cpp */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "ext/ANNfloat/ANN.h"

using namespace ANNfloat;

static void fill_vector_float(float *vec, unsigned char *v, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
        vec[i] = (double) v[i];
}

int compute_clustering_kd_tree(int n, int dim, int k, unsigned char **v,
                               double *means, unsigned int *clustering,
                               double &error_out)
{
    double error = 0.0;

    /* Using a kd-tree */
    ANNpointArray pts = annAllocPts(k, dim);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            pts[i][j] = means[i * dim + j];
        }
    }

    ANNkd_tree *tree = new ANNkd_tree(pts, k, dim, 4);
    annMaxPtsVisit(512);

    int changed = 0;
    std::vector<float> vec(dim);

    for (int i = 0; i < n; i++) {
        int nn;
        float dist;
        fill_vector_float(vec.data(), v[i], dim);
        tree->annkPriSearch(vec.data(), 1, &nn, &dist, 0.0);

        error += (double) dist;

        if ((int) clustering[i] != nn) {
            changed++;
            clustering[i] = nn;
        }
    }

    error_out = error;

    delete tree;
    annDeallocPts(pts);

    return changed;
}
