/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011       Andreas Muetzel (amuetzel@uni-koblenz.de). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_KDTREE_CUDA_3D_INDEX_H_
#define FLANN_KDTREE_CUDA_3D_INDEX_H_

#include <algorithm>
#include <map>
#include <cassert>
#include <cstring>
#include "FLANN/general.h"
#include "FLANN/algorithms/nn_index.h"
#include "FLANN/util/matrix.h"
#include "FLANN/util/result_set.h"
#include "FLANN/util/heap.h"
#include "FLANN/util/allocator.h"
#include "FLANN/util/random.h"
#include "FLANN/util/saving.h"
#include "FLANN/util/params.h"

namespace flann
{

struct KDTreeCuda3dIndexParams : public IndexParams
{
    KDTreeCuda3dIndexParams( int leaf_max_size = 64 )
    {
        (*this)["algorithm"] = FLANN_INDEX_KDTREE_CUDA;
        (*this)["leaf_max_size"] = leaf_max_size;
        (*this)["dim"] = 3;
    }
};

/**
 * Cuda KD Tree.
 * Tree is built with GPU assistance and search is performed on the GPU, too.
 *
 * Usually faster than the CPU search for data (and query) sets larger than 250000-300000 points, depending
 * on your CPU and GPU.
 */
template <typename Distance>
class KDTreeCuda3dIndex : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    typedef NNIndex<Distance> BaseClass;

    int visited_leafs;

    typedef bool needs_kdtree_distance;

    /**
     * KDTree constructor
     *
     * Params:
     *          inputData = dataset with the input features
     *          params = parameters passed to the kdtree algorithm
     */
    KDTreeCuda3dIndex(const Matrix<ElementType>& inputData, const IndexParams& params = KDTreeCuda3dIndexParams(),
                      Distance d = Distance() ) : BaseClass(params,d), dataset_(inputData), leaf_count_(0), visited_leafs(0), node_count_(0), current_node_count_(0)
    {
        size_ = dataset_.rows;
        dim_ = dataset_.cols;

        int dim_param = get_param(params,"dim",-1);
        if (dim_param>0) dim_ = dim_param;
        leaf_max_size_ = get_param(params,"leaf_max_size",10);
        assert( dim_ == 3 );
        gpu_helper_=0;
    }

    KDTreeCuda3dIndex(const KDTreeCuda3dIndex& other);
    KDTreeCuda3dIndex operator=(KDTreeCuda3dIndex other);

    /**
     * Standard destructor
     */
    ~KDTreeCuda3dIndex()
    {
        delete[] data_.ptr();
        clearGpuBuffers();
    }

    BaseClass* clone() const
    {
    	throw FLANNException("KDTreeCuda3dIndex cloning is not implemented");
    }

    /**
     * Builds the index
     */
    void buildIndex()
    {
        // Create a permutable array of indices to the input vectors.
        vind_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            vind_[i] = i;
        }

        leaf_count_=0;
        node_count_=0;
        //         computeBoundingBox(root_bbox_);
        //                              tree_.reserve(log2((double)size_/leaf_max_size_));
        //         divideTree(0, size_, root_bbox_,-1 );   // construct the tree

        delete[] data_.ptr();

        uploadTreeToGpu();
    }

    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_KDTREE_SINGLE;
    }


    void removePoint(size_t index)
    {
    	throw FLANNException( "removePoint not implemented for this index type!" );
    }

    ElementType* getPoint(size_t id)
    {
    	return dataset_[id];
    }

    void saveIndex(FILE* stream)
    {
        throw FLANNException( "Index saving not implemented!" );

    }


    void loadIndex(FILE* stream)
    {
        throw FLANNException( "Index loading not implemented!" );
    }

    size_t veclen() const
    {
        return dim_;
    }

    /**
     * Computes the inde memory usage
     * Returns: memory used by the index
     * TODO: return system or gpu RAM or both?
     */
    int usedMemory() const
    {
        //         return tree_.size()*sizeof(Node)+dataset_.rows*sizeof(int);  // pool memory and vind array memory
        return 0;
    }


    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const
    {
    	knnSearchGpu(queries,indices, dists, knn, params);
        return knn*queries.rows; // hack...
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                          std::vector< std::vector<int> >& indices,
                          std::vector<std::vector<DistanceType> >& dists,
                          size_t knn,
                          const SearchParams& params) const
    {
    	knnSearchGpu(queries,indices, dists, knn, params);
        return knn*queries.rows; // hack...
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    void knnSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const;

    int knnSearchGpu(const Matrix<ElementType>& queries,
                     std::vector< std::vector<int> >& indices,
                     std::vector<std::vector<DistanceType> >& dists,
                     size_t knn,
                     const SearchParams& params) const
    {
        flann::Matrix<int> ind( new int[knn*queries.rows], queries.rows,knn);
        flann::Matrix<DistanceType> dist( new DistanceType[knn*queries.rows], queries.rows,knn);
        knnSearchGpu(queries,ind,dist,knn,params);
        for( size_t i = 0; i<queries.rows; i++ ) {
            indices[i].resize(knn);
            dists[i].resize(knn);
            for( size_t j=0; j<knn; j++ ) {
                indices[i][j]=ind[i][j];
                dists[i][j]=dist[i][j];
            }
        }
        delete [] ind.ptr();
        delete [] dist.ptr();
        return knn*queries.rows; // hack...
    }

    int radiusSearch(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists,
                             float radius, const SearchParams& params) const
    {
    	return radiusSearchGpu(queries,indices, dists, radius, params);
    }

    int radiusSearch(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                             std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const
    {
    	return radiusSearchGpu(queries,indices, dists, radius, params);
    }

    int radiusSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists,
                        float radius, const SearchParams& params) const;

    int radiusSearchGpu(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                        std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const;

    /**
     * Not implemented, since it is only used by single-element searches.
     * (but is needed b/c it is abstract in the base class)
     */
    void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& searchParams) const
    {
    }

protected:
    void buildIndexImpl()
    {
        /* nothing to do here */
    }

    void freeIndex()
    {
        /* nothing to do here */
    }

private:

    void uploadTreeToGpu( );

    void clearGpuBuffers( );




private:

    struct GpuHelper;

    GpuHelper* gpu_helper_;

    const Matrix<ElementType> dataset_;

    int leaf_max_size_;

    int leaf_count_;
    int node_count_;
    //! used by convertTreeToGpuFormat
    int current_node_count_;


    /**
     *  Array of indices to vectors in the dataset.
     */
    std::vector<int> vind_;

    Matrix<ElementType> data_;

    size_t dim_;

    USING_BASECLASS_SYMBOLS
};   // class KDTreeCuda3dIndex


}

#endif //FLANN_KDTREE_SINGLE_INDEX_H_
