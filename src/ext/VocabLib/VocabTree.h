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

/* VocabTree.h */
/* Vocabulary tree classes */

#ifndef __vocab_tree_h__
#define __vocab_tree_h__

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "ext/ANNchar/ANN.h"

namespace VocabLib {

/* Types of distances supported */
typedef enum {
    DistanceDot  = 0,
    DistanceMin = 1,
} DistanceType;

/* Sparse matrix types */
typedef std::pair<unsigned long,float> sp_entry;
typedef std::vector<sp_entry> sp_list;

/* ImageCount class used in the inverted file */
class ImageCount {
public:
    ImageCount() : m_index(0), m_count(0.0) { }
    ImageCount(unsigned int index, float count) :
        m_index(index), m_count(count) { }

    unsigned int m_index; /* Index of the database image this entry
                           * corresponds to */
    float m_count; /* (Weighted, normalized) count of how many times this
                    * feature appears */
};

/* Abstract class for a node of the vocabulary tree */
class VocabTreeNode {
public:
    /* Constructor */
    VocabTreeNode() : m_desc(NULL) { }
    /* Destructor */
    virtual ~VocabTreeNode() { }

    /* I/O routines */
    virtual int Read(FILE *f, int bf, int dim) = 0;
    virtual int WriteNode(FILE *f, int bf, int dim) const = 0;
    virtual int Write(FILE *f, int bf, int dim) const = 0;
    virtual int WriteFlat(FILE *f, int bf, int dim) const = 0;
    virtual int WriteASCII(FILE *f, int bf, int dim) const = 0;
    virtual void Clear(int bf) = 0;

    /* Recursively build the vocabulary tree via kmeans clustering */
    /* Inputs:
     *   n      : number of features to cluster
     *   dim    : dimensionality of each feature vector (e.g., 128)
     *   depth  : total depth of the tree to create
     *   depth_curr : current depth
     *   bf     : branching factor of the tree (children per node)
     *   restarts   : number of random restarts during clustering
     *   v      : list of arrays representing the features
     *
     *   means      : work array for storing means that get passed to kmeans
     *   clustering : work array for storing clustering in kmeans
     */
    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr,
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering) = 0;

    /* Push a feature down to a leaf of the tree, and accumulate it to
     * the score of that leaf.  Optionally, add the feature to the
     * inverted file.
     *
     * Inputs:
     *   v     : array containing the feature descriptor
     *   index : index of the image that this feature belongs to
     *   bf    : branch factor of the tree
     *   dim   : dimensionality of the tree
     *   add   : should this feature be added to the inverted file?
     */
    virtual unsigned long PushAndScoreFeature(unsigned char *v,
                                              unsigned int index,
                                              int bf, int dim,
                                              bool add = true) = 0;

    /* Update the counts in an inverted file associated with a visual
     * word
     *
     * Inputs:
     *   index : index of database image to which the feature belongs
     *   bf    : branching factor of the tree
     *   dim   : dimension of each feature
     */
    virtual int AddFeatureToInvertedFile(unsigned int index,
                                         int bf, int dim) = 0;

    /* Given a set of previously pushed features, fill a query array
     * of size num_nodes with the bag-of-words vector entries
     *
     * Inputs:
     *   bf      : branching factor of the tree
     *   mag_inv : inverse magnitude of the BoW vector
     *
     * Outputs:
     *   q       : array of size num_nodes storing the BoW vector on exit
     */
    virtual int FillQueryVector(float *q, int bf, double mag_inv)
        { return 0; }

    /* Given a query BoW vector, compute its similarity to all the
     * database vectors using the inverted file stored in the tree
     *
     * Inputs:
     *   q      : BoW query vector (length: num_nodes)
     *   bf     : branching factor of the tree
     *   dtype  : type of distance function to use
     *
     * Outputs:
     *   scores : at exit, array of score for each database image
     *            (similarity to the query vector)
     */
    virtual int ScoreQuery(float *q, int bf, DistanceType dtype,
                           float *scores)
        { return 0; }

    /* Compute TFIDF weights for each visual word
     *
     * Inputs:
     *   bf : branching factor of the tree
     *   n  : number of documents in the database
     */
    virtual double ComputeTFIDFWeights(int bf, double n)
        { return 0; }

    /* Fill a memory buffer with the descriptors scored in the leaves
     * of the tree */
    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const = 0;
    virtual int FillDatabaseVectors(std::vector<sp_list> &vectors,
                                    int start_index, int bf, int dim) const= 0;
    virtual void PopulateLeaves(int bf, int dim,
                                VocabTreeNode **leaves) = 0;

    /* Functions for normalizing the database vectors */
    virtual double ComputeDatabaseVectorMagnitude(int bf, DistanceType dtype)
        { return 0; }
    virtual int NormalizeDatabase(int bf, int start_index,
                                  std::vector<float> &mags)
        { return 0; }
    virtual int ComputeDatabaseMagnitudes(int bf, DistanceType dtype,
                                          int start_index,
                                          std::vector<float> &mags)
        { return 0; }

    /* Utility functions */
    virtual int PrintWeights(int depth_curr, int bf) const
        { return 0; }
    virtual unsigned long ComputeIDs(int bf, unsigned long id)
        { return 0; }
    virtual unsigned long CountNodes(int bf) const = 0;
    virtual unsigned long CountLeaves(int bf) const = 0;
    virtual double CountFeatures(int bf) = 0;
    virtual int ClearScores(int bf)
        { return 0; }
    virtual int ClearDatabase(int bf)
        { return 0; }
    virtual int SetInteriorNodeWeight(int bf, float weight)
        { return 0; }
    virtual int SetInteriorNodeWeight(int bf, int dist_from_leaves,
                                      float weight)
        { return 0; }
    virtual int SetConstantLeafWeights(int bf)
        { return 0; }

    virtual int Combine(VocabTreeNode *other, int bf)
        { return 0; }

    virtual int GetMaxDatabaseImageIndex(int bf) const
        { return 0; }

    /* Member variables */
    unsigned char *m_desc; /* Descriptor for this node */
    unsigned long m_id;    /* ID of this node */
};

/* Class representing an interior node of the vocab tree */
class VocabTreeInteriorNode : public VocabTreeNode {
public:
    VocabTreeInteriorNode() : VocabTreeNode(), m_children(NULL) { }
    virtual ~VocabTreeInteriorNode() { };

    virtual int Read(FILE *f, int bf, int dim);
    virtual int WriteNode(FILE *f, int bf, int dim) const;
    virtual int Write(FILE *f, int bf, int dim) const;
    virtual int WriteFlat(FILE *f, int bf, int dim) const;
    virtual int WriteASCII(FILE *f, int bf, int dim) const;
    virtual void Clear(int bf);

    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr,
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering);

    virtual unsigned long PushAndScoreFeature(unsigned char *v,
                                              unsigned int index,
                                              int bf, int dim,
                                              bool add = true);

    virtual int AddFeatureToInvertedFile(unsigned int index,
                                         int bf, int dim) { return 0; }

    virtual int ScoreQuery(float *q, int bf, DistanceType dtype,
                           float *scores);

    virtual double ComputeTFIDFWeights(int bf, double n);

    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const;
    virtual int FillDatabaseVectors(std::vector<sp_list> &vectors,
                                    int start_index, int bf, int dim) const;
    virtual void PopulateLeaves(int bf, int dim, VocabTreeNode **leaves);

    virtual int PrintWeights(int depth_curr, int bf) const;
    virtual unsigned long ComputeIDs(int bf, unsigned long id);
    virtual unsigned long CountNodes(int bf) const;
    virtual unsigned long CountLeaves(int bf) const;
    virtual double CountFeatures(int bf);
    virtual int ClearScores(int bf);
    virtual int NormalizeDatabase(int bf, int start_index,
                                  std::vector<float> &mags);
    virtual double ComputeDatabaseVectorMagnitude(int bf, DistanceType dtype);
    virtual int ComputeDatabaseMagnitudes(int bf, DistanceType dtype,
                                          int start_index,
                                          std::vector<float> &mags);

    virtual int ClearDatabase(int bf);
    virtual int SetConstantLeafWeights(int bf);
    virtual int FillQueryVector(float *q, int bf, double mag_inv);

    virtual int Combine(VocabTreeNode *other, int bf);
    virtual int GetMaxDatabaseImageIndex(int bf) const;

    /* Member variables */
    VocabTreeNode **m_children; /* Array of child nodes */
};

/* Class representing a leaf of the vocab tree.  Each leaf represents
 * a visual word */
class VocabTreeLeaf : public VocabTreeNode
{
public:
    VocabTreeLeaf() : VocabTreeNode(), m_score(0.0), m_weight(1.0) { }
    virtual ~VocabTreeLeaf() { };

    /* I/O functions */
    virtual int Read(FILE *f, int bf, int dim);
    virtual int WriteNode(FILE *f, int bf, int dim) const;
    virtual int Write(FILE *f, int bf, int dim) const;
    virtual int WriteFlat(FILE *f, int bf, int dim) const;
    virtual int WriteASCII(FILE *f, int bf, int dim) const;
    virtual void Clear(int bf);

    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr,
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering);

    virtual unsigned long PushAndScoreFeature(unsigned char *v,
                                              unsigned int index,
                                              int bf, int dim,
                                              bool add = true);

    virtual int ScoreQuery(float *q, int bf, DistanceType dtype,
                           float *scores);
    virtual int AddFeatureToInvertedFile(unsigned int index, int bf, int dim);
    virtual int FillQueryVector(float *q, int bf, double mag_inv);

    virtual double ComputeTFIDFWeights(int bf, double n);

    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const;
    virtual int FillDatabaseVectors(std::vector<sp_list> &vectors,
                                    int start_index, int bf, int dim) const;
    virtual void PopulateLeaves(int bf, int dim, VocabTreeNode **leaves);

    virtual double ComputeDatabaseVectorMagnitude(int bf, DistanceType dtype);
    virtual int ComputeDatabaseMagnitudes(int bf, DistanceType dtype,
                                          int start_index,
                                          std::vector<float> &mags);

    virtual int ClearScores(int bf);
    virtual int ClearDatabase(int bf);

    virtual int SetInteriorNodeWeight(int bf, float weight);
    virtual int SetInteriorNodeWeight(int bf, int dist_from_leaves,
                                      float weight);
    virtual int SetConstantLeafWeights(int bf);

    virtual int PrintWeights(int depth_curr, int bf) const;

    virtual unsigned long ComputeIDs(int bf, unsigned long id);
    virtual unsigned long CountNodes(int bf) const;
    virtual unsigned long CountLeaves(int bf) const;
    virtual double CountFeatures(int bf);

    virtual int NormalizeDatabase(int bf, int start_index,
                                  std::vector<float> &mags);

    virtual int Combine(VocabTreeNode *other, int bf);
    virtual int GetMaxDatabaseImageIndex(int bf) const;

    /* Member variables */
    float m_score;   /* Current, temporary score for the current image */
    float m_weight;  /* Weight for this visual word */
    std::vector<ImageCount> m_image_list;  /* Images that contain this word */
};


class VocabTreeFlatNode : public VocabTreeInteriorNode
{
public:
    VocabTreeFlatNode() : VocabTreeInteriorNode()
    { }

    virtual unsigned long PushAndScoreFeature(unsigned char *v,
                                              unsigned int index,
                                              int bf, int dim,
                                              bool add = true);

    void BuildANNTree(int num_leaves, int dim);

    ANNchar::ANNkd_tree *m_tree; /* For finding nearest neighbors */
};

class VocabTree {
public:
    VocabTree() : m_database_images(0), m_distance_type(DistanceMin) { }

    /* I/O routines */
    int Read(const char *filename);
    int WriteHeader(FILE *f) const;
    int Write(const char *filename) const;
    int WriteFlat(const char *filename) const;
    int WriteASCII(const char *filename) const;
    int WriteDatabaseVectors(const char *filename,
                             int start_index, int num_vectors) const;

    int Flatten(); /* Flatten the tree to a single level */

    /* Build the vocabulary tree using kmeans
     *
     * Inputs:
     *  n        : number of features to cluster
     *  dim      : dimensionality of each feature vector (e.g., 128)
     *  depth    : total depth of the tree to create
     *  bf       : desired branching factor of the tree (children per node)
     *  restarts : number of random restarts during clustering
     *  vp       : array of pointers to arrays representing the features
     */
    int Build(int n, int dim, int depth, int bf, int restarts,
              unsigned char **vp);

    /* Push a feature down to a leaf of the tree, and accumulate it to
     * the score of that leaf.  Optionally, add the feature to the
     * inverted file.  Recursively calls PushAndScoreFeature
     *
     * Inputs:
     *   v     : array containing the feature descriptor
     *   index : index of the image that this feature belongs to
     *   add   : should this feature be added to the inverted file?
     */
    unsigned long PushAndScoreFeature(unsigned char *v, unsigned int index,
                                      bool add);

    /* Add an image to the database.
     *
     * Inputs:
     *   index : identifier for the image
     *   n     : number of features in the database
     *   v     : array of feature descriptors, concatenated into one
     *         : big array of length n*dim
     *   ids   : optional output array of word ids each key mapped to
     */
    double AddImageToDatabase(int index, int n, unsigned char *v,
                              unsigned long *ids = NULL);

    /* Given a tree populated with database images, compute the TFIDF
     * weights */
    int ComputeTFIDFWeights(unsigned int num_db_images);

    /* Given a set of feature descriptors in a query image, compute
     * the similarity between the query image and all of the images in
     * the database.
     *
     * Inputs:
     *   n         : number of query feature descriptors
     *   normalize : normalize the database vectors?
     *   v         : array of query descriptors, concatenated into one
     *               big array of length n*dim
     *
     * Output:
     *   scores    : at exit, array of score for each database image
     *               (similarity to the query vector)
     *
     *   Returns the magnitude of the query vector
     */
    double ScoreQueryKeys(int n, bool normalize, unsigned char *v,
                          float *scores);

    /* Empty out the database */
    int ClearDatabase();
    /* Normalize the database */
    int NormalizeDatabase(int start_index, int num_db_images);
    /* Combine with another database */
    int Combine(const VocabTree &tree);
    int GetMaxDatabaseImageIndex() const;

    /* Utility functions */
    int PrintWeights();
    unsigned long CountNodes() const;
    unsigned long CountLeaves() const;
    int SetInteriorNodeWeight(float weight);
    int SetInteriorNodeWeight(int dist_from_leaves, float weight);
    int SetConstantLeafWeights();
    int SetDistanceType(DistanceType type);

    /* Destroy this tree */
    int Clear();

    /* Member variables */
    int m_database_images;         /* Number of images in the database */
    int m_branch_factor;           /* Branching factor for tree */
    int m_depth;                   /* Depth of the tree */
    int m_dim;                     /* Dimension of the descriptors */
    unsigned long m_num_nodes;     /* Number of nodes in the tree */
    DistanceType m_distance_type;  /* Type of the distance measure */
    VocabTreeNode *m_root;         /* Root of the tree */
};

}

#endif /* __vocab_tree_h__ */
