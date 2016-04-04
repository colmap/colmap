/* VocabFlatNode.cpp */

#include "VocabTree.h"

#include "ext/ANNchar/ANN.h"

using namespace ANNchar;

namespace VocabLib {

#define NUM_NNS 1 // 3
#define SIGMA_SQ 6250.0
#define USE_SOFT_ASSIGNMENT 0

#if 0
int VocabTreeFlatNode::PushFeature(unsigned char *v, double weight,
                                   unsigned int index, int bf, int dim,
                                   double *work,
                                   std::vector<float> &parent_weights)
{
    // unsigned long min_dist = ULONG_MAX;
    // int best_idx = 0;

    int nn_idx[NUM_NNS];
    ANNdist distsq[NUM_NNS];

    // annMaxPtsVisit(256);
    annMaxPtsVisit(128);

    m_tree->annkPriSearch(v, NUM_NNS, nn_idx, distsq, 0.0);

#if 0
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
#endif

    unsigned long r;
    if (USE_SOFT_ASSIGNMENT) {
        double w_weights[NUM_NNS];

        double sum = 0.0;
        for (int i = 0; i < NUM_NNS; i++) {
            w_weights[i] = exp(-(double) distsq[i] / (2.0 * SIGMA_SQ));
            sum += w_weights[i];
        }

        for (int i = 0; i < NUM_NNS; i++) {
            w_weights[i] /= sum;
        }

        for (int i = 0; i < NUM_NNS; i++) {
            double w_weight = w_weights[i];
            r = m_children[nn_idx[i]]->PushFeature(v, w_weight * weight,
                                                   index,
                                                   bf, dim, work,
                                                   parent_weights);
        }
    } else {
        r = m_children[nn_idx[0]]->PushFeature(v, weight, index,
                                               bf, dim, work,
                                               parent_weights);
    }

    return r;
}
#endif

unsigned long VocabTreeFlatNode::
    PushAndScoreFeature(unsigned char *v, unsigned int index,
                        int bf, int dim, bool add)
{
    int nn_idx[NUM_NNS];
    ANNdist distsq[NUM_NNS];

    annMaxPtsVisit(256);
    // annMaxPtsVisit(128);

    m_tree->annkPriSearch(v, NUM_NNS, nn_idx, distsq, 0.0);

    unsigned long r;
    if (USE_SOFT_ASSIGNMENT) {
        double w_weights[NUM_NNS];

        double sum = 0.0;
        for (int i = 0; i < NUM_NNS; i++) {
            w_weights[i] = exp(-(double) distsq[i] / (2.0 * SIGMA_SQ));
            sum += w_weights[i];
        }

        for (int i = 0; i < NUM_NNS; i++) {
            w_weights[i] /= sum;
        }

        for (int i = 0; i < NUM_NNS; i++) {
            r = m_children[nn_idx[i]]->PushAndScoreFeature(v, index,
                                                           bf, dim, add);
        }
    } else {
        r = m_children[nn_idx[0]]->PushAndScoreFeature(v, index, bf, dim, add);
    }

    return r;
}

/* Create a search tree for the given set of keypoints */
void VocabTreeFlatNode::BuildANNTree(int num_leaves, int dim)
{
    // unsigned long mem_size = num_leaves * dim;
    // unsigned char *desc = new unsigned char[mem_size];

    /* Create a new array of points */
    ANNpointArray pts = annAllocPts(num_leaves, dim);

    unsigned long id = 0;
    FillDescriptors(num_leaves, dim, id, pts[0]);

    /* Create a search tree for k2 */
    m_tree = new ANNkd_tree(pts, num_leaves, dim, 16);
}

}
