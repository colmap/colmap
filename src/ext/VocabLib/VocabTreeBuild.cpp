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

/* VocabTreeBuild.cpp */
/* Routines for building a vocab tree */

#include "VocabTree.h"
#include "kmeans.h"
#include "util.h"

namespace VocabLib {

int VocabTreeLeaf::BuildRecurse(int n, int dim, int depth,
                                int depth_curr, int bf,
                                int restarts, unsigned char **v,
                                double *means, unsigned int *clustering)
{
    /* Nothing to do on the bottom level, everything was taken care of
     * above us */
    return 0;
}

int VocabTreeInteriorNode::BuildRecurse(int n, int dim, int depth,
                                        int depth_curr, int bf,
                                        int restarts, unsigned char **v,
                                        double *means,
                                        unsigned int *clustering)
{
    if (depth_curr > depth)
        return 0;

    if (depth_curr < 2) {
        for (int i = 0; i < depth_curr; i++)
            printf(" ");

        printf("[BuildRecurse] (level %d): Running k-means (%d features)\n",
               depth_curr, n);
        fflush(stdout);
    }

    /* Allocate the children for this node */
    m_children = new VocabTreeNode *[bf];

    /* Run k-means */
    double error = kmeans(n, dim, bf, restarts, v, means, clustering);

    double error_means = 0.0;
    for (int i = 0; i < bf; i++) {
        for (int j = 0; j < dim; j++) {
            double d = means[i * dim + j] - m_desc[j];
            error_means += d * d;
        }
    }

    if (depth_curr < 2) {
        for (int i = 0; i < depth_curr; i++)
            printf(" ");

        printf("[BuildRecurse] (level %d): "
               "%d features, error: %0.3f, %0.3f\n",
               depth_curr, n, error / n, error_means);
        fflush(stdout);
    }

    int *counts = new int[bf];

    for (int i = 0; i < bf; i++) {
        counts[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        counts[clustering[i]]++;
    }

    /* Create the children */
    for (int i = 0; i < bf; i++) {
        if (counts[i] > 0) {
            if (depth_curr == depth || counts[i] <= 2 * bf) {
                m_children[i] = new VocabTreeLeaf();
            } else {
                m_children[i] = new VocabTreeInteriorNode();
            }

            m_children[i]->m_desc = new unsigned char[dim];

            for (int j = 0; j < dim; j++) {
                m_children[i]->m_desc[j] = iround(means[i * dim + j]);
            }
        } else {
            m_children[i] = NULL;
        }
    }

    if (depth_curr < depth) {
        /* Reorder the pointers to the vectors */
        int idx = 0;
        for (int i = 0; i < bf; i++) {
            for (int j = 0; j < n; j++) {
                if ((int) clustering[j] == i) {
                    unsigned char *v_tmp = v[idx];
                    v[idx] = v[j];
                    v[j] = v_tmp;

                    unsigned int tmp = clustering[idx];
                    clustering[idx] = clustering[j];
                    clustering[j] = tmp;

                    idx++;
                }
            }
        }

        int off = 0;
        for (int i = 0; i < bf; i++) {
            if (m_children[i] != NULL) {
                m_children[i]->BuildRecurse(counts[i], dim, depth, depth_curr + 1,
                                            bf, restarts, v + off, means,
                                            clustering);
            }

            off += counts[i];
        }
    }

    delete [] counts;

    return 0;
}

int VocabTree::Build(int n, int dim, int depth, int bf, int restarts,
                     unsigned char **vp)
{
    printf("[VocabTree::Build] Building tree from %d features\n", n);
    printf("[VocabTree::Build]   with depth %d, branching factor %d\n",
           depth, bf);
    printf("[VocabTree::Build]   and restarts %d\n", restarts);
    fflush(stdout);

    m_depth = depth;
    m_dim = dim;
    m_branch_factor = bf;

    double *means = new double[bf * dim];
    unsigned int *clustering = new unsigned int[n];

    if (means == NULL) {
        printf("[VocabTree::Build] Error allocating means\n");
        exit(-1);
    }

    if (clustering == NULL) {
        printf("[VocabTree::Build] Error allocating clustering\n");
        exit(-1);
    }

    m_root = new VocabTreeInteriorNode();
    m_root->m_desc = new unsigned char[dim];
    for (int i = 0; i < dim; i++)
        m_root->m_desc[i] = 0;

    m_root->BuildRecurse(n, dim, depth, 0, bf, restarts,
                         vp, means, clustering);

    delete means;
    delete clustering;

    printf("[VocabTree::Build] Finished building tree.\n");
    fflush(stdout);

    return 0;
}

}
