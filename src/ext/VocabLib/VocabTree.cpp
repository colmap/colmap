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

/* VocabTree.cpp */
/* Build a vocabulary tree from a set of vectors */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "VocabTree.h"
#include "defines.h"
#include "qsort.h"
#include "util.h"

namespace VocabLib {

/* Useful utility function for computing the squared distance between
 * two vectors a and b of length dim */
static unsigned long vec_diff_normsq(int dim,
                                     unsigned char *a,
                                     unsigned char *b)
{
    int i;
    unsigned long normsq = 0;

    for (i = 0; i < dim; i++) {
        int d = (int) a[i] - (int) b[i];
        normsq += d * d;
    }

    return normsq;
}

void VocabTreeInteriorNode::Clear(int bf)
{
    if (m_children != NULL) {
        for (int i = 0; i < bf; i++) {
            m_children[i]->Clear(bf);
            delete m_children[i];
        }

        delete [] m_children;
    }

    if (m_desc != NULL)
        delete [] m_desc;
}

void VocabTreeLeaf::Clear(int bf)
{
    if (m_desc != NULL)
        delete [] m_desc;

    m_image_list.clear();
}

unsigned long VocabTreeInteriorNode::
    PushAndScoreFeature(unsigned char *v,
                        unsigned int index, int bf, int dim, bool add)
{
    unsigned long min_dist = ULONG_MAX;
    int best_idx = 0;

    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            unsigned long dist =
                vec_diff_normsq(dim, m_children[i]->m_desc, v);

            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
    }

    unsigned long r =
        m_children[best_idx]->PushAndScoreFeature(v, index, bf, dim, add);

    return r;
}

unsigned long VocabTreeLeaf::PushAndScoreFeature(unsigned char *v,
                                                 unsigned int index,
                                                 int bf, int dim,
                                                 bool add)
{
    m_score += m_weight;

    if (add) {
        /* Update the inverted file */
        AddFeatureToInvertedFile(index, bf, dim);
    }

    return m_id;
}

int VocabTreeLeaf::AddFeatureToInvertedFile(unsigned int index,
                                            int bf, int dim)
{
    /* Update the inverted file */
    int n = (int) m_image_list.size();

    if (n == 0) {
        m_image_list.push_back(ImageCount(index, (float) m_weight));
    } else {
        if (m_image_list[n-1].m_index == index) {
            m_image_list[n-1].m_count += m_weight;
        } else {
            m_image_list.
                push_back(ImageCount(index, (float) m_weight));
        }
    }

    return 0;
}

double VocabTreeInteriorNode::ComputeTFIDFWeights(int bf, double n)
{
    /* Compute TFIDF weights for all leaf nodes (visual words) */
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
             m_children[i]->ComputeTFIDFWeights(bf, n);
        }
    }

    return 0;
}

double VocabTreeLeaf::ComputeTFIDFWeights(int bf, double n)
{
    int len = (int) m_image_list.size();

    if (len > 0)
        m_weight = log((double) n / (double) len);
    else
        m_weight = 0.0;

    /* We'll pre-apply weights to the count values (TF scores) in the
     * inverted file.  We took care of this for you. */
    // printf("weight = %0.3f\n", m_weight);
    for (int i = 0; i < len; i++) {
        m_image_list[i].m_count *= m_weight;
    }

    return 0;
}

int VocabTreeInteriorNode::FillQueryVector(float *q, int bf,
                                           double mag_inv)
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->FillQueryVector(q, bf, mag_inv);
        }
    }

    return 0;
}

int VocabTreeLeaf::FillQueryVector(float *q, int bf, double mag_inv)
{
    q[m_id] = m_score * mag_inv;
    return 0;
}

int VocabTreeInteriorNode::ScoreQuery(float *q, int bf, DistanceType dtype,
                                      float *scores)
{
    /* Pass the scores to the children for updating */
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->ScoreQuery(q, bf, dtype, scores);
        }
    }

    return 0;
}

int VocabTreeLeaf::ScoreQuery(float *q, int bf, DistanceType dtype,
                              float *scores)
{
    /* Early exit */
    if (q[m_id] == 0.0) return 0;

    int n = (int) m_image_list.size();

    for (int i = 0; i < n; i++) {
        int img = m_image_list[i].m_index;

        switch (dtype) {
            case DistanceDot:
                scores[img] += q[m_id] * m_image_list[i].m_count;
                break;
            case DistanceMin:
                scores[img] += MIN(q[m_id], m_image_list[i].m_count);
                break;
        }
    }

    return 0;
}

double ComputeMagnitude(DistanceType dtype, double dim)
{
    switch (dtype) {
    case DistanceDot:
        return dim * dim;
    case DistanceMin:
        return dim;
    default:
        printf("[ComputeMagnitude] No case value found!\n");
        return 0.0;
    }
}

double VocabTreeInteriorNode::
    ComputeDatabaseVectorMagnitude(int bf, DistanceType dtype)
{
    double mag = 0.0;

    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            mag += m_children[i]->ComputeDatabaseVectorMagnitude(bf, dtype);
        }
    }

    return mag;
}

double VocabTreeLeaf::
    ComputeDatabaseVectorMagnitude(int bf, DistanceType dtype)
{
    double dim = m_score;
    double mag = ComputeMagnitude(dtype, dim);

    return mag;
}

int VocabTreeInteriorNode::
    ComputeDatabaseMagnitudes(int bf, DistanceType dtype, int start_index,
                              std::vector<float> &mags)
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->ComputeDatabaseMagnitudes(bf, dtype,
                                                     start_index, mags);
        }
    }

    return 0;
}

int VocabTreeLeaf::
    ComputeDatabaseMagnitudes(int bf, DistanceType dtype,
                              int start_index, std::vector<float> &mags)
{
    int len = (int) m_image_list.size();
    for (int i = 0; i < len; i++) {
        unsigned int index = m_image_list[i].m_index - start_index;
        double dim = m_image_list[i].m_count;
        assert(index < mags.size());
        mags[index] += ComputeMagnitude(dtype, dim);
    }

    return 0;
}

int VocabTreeInteriorNode::NormalizeDatabase(int bf, int start_index,
                                             std::vector<float> &mags)
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->NormalizeDatabase(bf, start_index, mags);
        }
    }

    return 0;
}

int VocabTreeLeaf::NormalizeDatabase(int bf, int start_index,
                                     std::vector<float> &mags)
{
    int len = (int) m_image_list.size();
    for (int i = 0; i < len; i++) {
        unsigned int index = m_image_list[i].m_index - start_index;
        assert(index < mags.size());
        m_image_list[i].m_count /= mags[index];
    }

    return 0;
}

int VocabTreeInteriorNode::Combine(VocabTreeNode *other, int bf)
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            assert(((VocabTreeInteriorNode *)other)->m_children[i] != NULL);
            m_children[i]->
                Combine(((VocabTreeInteriorNode *)other)->m_children[i], bf);
        }
    }

    return 0;
}

int VocabTreeLeaf::Combine(VocabTreeNode *other, int bf)
{
    std::vector<ImageCount> &other_list =
        ((VocabTreeLeaf *)other)->m_image_list;
    m_image_list.insert(m_image_list.end(),
                        other_list.begin(), other_list.end());

    return 0;
}

int VocabTreeInteriorNode::GetMaxDatabaseImageIndex(int bf) const
{
    int max_idx = 0;
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            max_idx =
                MAX(max_idx, m_children[i]->GetMaxDatabaseImageIndex(bf));
        }
    }

    return max_idx;
}

int VocabTreeLeaf::GetMaxDatabaseImageIndex(int bf) const
{
    int max_idx = 0;
    int len = (int) m_image_list.size();
    for (int i = 0; i < len; i++) {
        max_idx = MAX(max_idx, (int) m_image_list[i].m_index);
    }

    return max_idx;
}


/* Implementations of driver functions */
unsigned long VocabTree::PushAndScoreFeature(unsigned char *v,
                                             unsigned int index, bool add)
{
    qsort_descending();
    return m_root->PushAndScoreFeature(v, index, m_branch_factor, m_dim, add);
}

int VocabTree::ComputeTFIDFWeights(unsigned int num_db_images)
{
    if (m_root != NULL) {
        // double n = m_root->CountFeatures(m_branch_factor);
        // printf("[VocabTree::ComputeTFIDFWeights] Found %lf features\n", n);
        m_root->ComputeTFIDFWeights(m_branch_factor, num_db_images);
    }

    return 0;
}

int VocabTree::NormalizeDatabase(int start_index, int num_db_images)
{
    std::vector<float> mags;
    mags.resize(num_db_images);
    m_root->ComputeDatabaseMagnitudes(m_branch_factor, m_distance_type,
                                      start_index, mags);

//    for (int i = 0; i < num_db_images; i++) {
//        printf("[NormalizeDatabase] Vector %d has magnitude %0.3f\n",
//               start_index + i, mags[i]);
//    }

    return m_root->NormalizeDatabase(m_branch_factor, start_index, mags);
}


double VocabTree::AddImageToDatabase(int index, int n, unsigned char *v,
                                     unsigned long *ids)
{
    m_root->ClearScores(m_branch_factor);
    unsigned long off = 0;

    // printf("[AddImageToDatabase] Adding image with %d features...\n", n);
    // fflush(stdout);

    for (int i = 0; i < n; i++) {
        unsigned long id =
            m_root->PushAndScoreFeature(v+off, index, m_branch_factor, m_dim);

        if (ids != NULL)
            ids[i] = id;

        off += m_dim;
        // fflush(stdout);
    }

    double mag = m_root->ComputeDatabaseVectorMagnitude(m_branch_factor,
                                                        m_distance_type);

    m_database_images++;

    switch (m_distance_type) {
    case DistanceDot:
        return sqrt(mag);
    case DistanceMin:
        return mag;
    default:
        printf("[VocabTree::AddImageToDatabase] No case value found!\n");
        return 0.0;
    }
}

/* Returns the weighted magnitude of the query vector */
double VocabTree::ScoreQueryKeys(int n, bool normalize, unsigned char *v,
                                 float *scores)
{
    qsort_descending();

    /* Compute the query vector */
    m_root->ClearScores(m_branch_factor);
    unsigned long off = 0;
    for (int i = 0; i < n; i++) {
        m_root->PushAndScoreFeature(v + off, 0,
                                    m_branch_factor, m_dim, false);

        off += m_dim;
    }

    double mag = m_root->ComputeDatabaseVectorMagnitude(m_branch_factor,
                                                        m_distance_type);

    if (m_distance_type == DistanceDot)
        mag = sqrt(mag);

    /* Now, compute the normalized vector */
    int num_nodes = m_num_nodes;
    float *q = new float[num_nodes];

    if (normalize)
        m_root->FillQueryVector(q, m_branch_factor, 1.0 / mag);
    else
        m_root->FillQueryVector(q, m_branch_factor, 1.0);

    m_root->ScoreQuery(q, m_branch_factor, m_distance_type, scores);

    delete [] q;

    return mag;
}

unsigned long g_leaf_counter = 0;

void VocabTreeInteriorNode::PopulateLeaves(int bf, int dim,
                                           VocabTreeNode **leaves)
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->PopulateLeaves(bf, dim, leaves);
        }
    }
}

void VocabTreeLeaf::PopulateLeaves(int bf, int dim,
                                   VocabTreeNode **leaves)
{
    leaves[g_leaf_counter++] = this;
}

int VocabTree::Flatten()
{
    if (m_root == NULL)
        return -1;

    // ANNpointArray pts = annAllocPts(num_leaves, dim);

    // m_root->FillDescriptors(num_leaves, dim, pts[0]);

    int num_leaves = CountLeaves();

    VocabTreeFlatNode *new_root = new VocabTreeFlatNode;
    new_root->m_children = new VocabTreeNode *[num_leaves];

    for (int i = 0; i < num_leaves; i++) {
        new_root->m_children[i] = NULL;
    }

    g_leaf_counter = 0;
    m_root->PopulateLeaves(m_branch_factor, m_dim, new_root->m_children);
    new_root->BuildANNTree(num_leaves, m_dim);
    new_root->m_desc = new unsigned char[m_dim];
    memset(new_root->m_desc, 0, m_dim);
    new_root->m_id = 0;

    m_root = new_root;

    /* Reset the branch factor */
    m_branch_factor = num_leaves;

    return 0;
}

int VocabTree::Combine(const VocabTree &tree)
{
    return m_root->Combine(tree.m_root, m_branch_factor);
}

int VocabTree::GetMaxDatabaseImageIndex() const
{
    return m_root->GetMaxDatabaseImageIndex(m_branch_factor);
}

int VocabTree::Clear()
{
    if (m_root != NULL) {
        m_root->Clear(m_branch_factor);
        delete m_root;
    }

    return 0;
}

}
