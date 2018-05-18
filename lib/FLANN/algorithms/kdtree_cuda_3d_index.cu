/**********************************************************************
 * Software License Agreement (BSD License)
 *
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

#include "kdtree_cuda_3d_index.h"
#include <FLANN/algorithms/dist.h>
#include <FLANN/util/cuda/result_set.h>
// #define THRUST_DEBUG 1
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <FLANN/util/cutil_math.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <FLANN/util/cuda/heap.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <FLANN/algorithms/kdtree_cuda_builder.h>
#include <vector_types.h>
namespace flann
{

namespace KdTreeCudaPrivate
{
template< typename GPUResultSet, typename Distance >
__device__
void searchNeighbors(const cuda::kd_tree_builder_detail::SplitInfo* splits,
                     const int* child1,
                     const int* parent,
                     const float4* aabbLow,
                     const float4* aabbHigh, const float4* elements, const float4& q, GPUResultSet& result, const Distance& distance = Distance() )
{

    bool backtrack=false;
    int lastNode=-1;
    int current=0;

    cuda::kd_tree_builder_detail::SplitInfo split;
    while(true) {
        if( current==-1 ) break;
        split = splits[current];

        float diff1;
        if( split.split_dim==0 ) diff1=q.x- split.split_val;
        else if( split.split_dim==1 ) diff1=q.y- split.split_val;
        else if( split.split_dim==2 ) diff1=q.z- split.split_val;

        // children are next to each other: leftChild+1 == rightChild
        int leftChild= child1[current];
        int bestChild=leftChild;
        int otherChild=leftChild;

        if ((diff1)<0) {
            otherChild++;
        }
        else {
            bestChild++;
        }

        if( !backtrack ) {
            /* If this is a leaf node, then do check and return. */
            if (leftChild==-1) {
                for (int i=split.left; i<split.right; ++i) {
                    float dist=distance.dist(elements[i],q);
                    result.insert(i,dist);
                }
                backtrack=true;
                lastNode=current;
                current=parent[current];
            }
            else { // go to closer child node
                lastNode=current;
                current=bestChild;
            }
        }
        else { // continue moving back up the tree or visit far node?
              // minimum possible distance between query point and a point inside the AABB
            float mindistsq=0;
            float4 aabbMin=aabbLow[otherChild];
            float4 aabbMax=aabbHigh[otherChild];

            if( q.x < aabbMin.x ) mindistsq+=distance.axisDist(q.x, aabbMin.x);
            else if( q.x > aabbMax.x ) mindistsq+=distance.axisDist(q.x, aabbMax.x);
            if( q.y < aabbMin.y ) mindistsq+=distance.axisDist(q.y, aabbMin.y);
            else if( q.y > aabbMax.y ) mindistsq+=distance.axisDist(q.y, aabbMax.y);
            if( q.z < aabbMin.z ) mindistsq+=distance.axisDist(q.z, aabbMin.z);
            else if( q.z > aabbMax.z ) mindistsq+=distance.axisDist(q.z, aabbMax.z);

            //  the far node was NOT the last node (== not visited yet) AND there could be a closer point in it
            if(( lastNode==bestChild) && (mindistsq <= result.worstDist() ) ) {
                lastNode=current;
                current=otherChild;
                backtrack=false;
            }
            else {
                lastNode=current;
                current=parent[current];
            }
        }

    }
}


template< typename GPUResultSet, typename Distance >
__global__
void nearestKernel(const cuda::kd_tree_builder_detail::SplitInfo* splits,
                   const int* child1,
                   const int* parent,
                   const float4* aabbMin,
                   const float4* aabbMax, const float4* elements, const float* query, int stride, int resultStride, int* resultIndex, float* resultDist, int querysize, GPUResultSet result, Distance dist = Distance())
{
    typedef float DistanceType;
    typedef float ElementType;
    //                  typedef DistanceType float;
    size_t tid = blockDim.x*blockIdx.x + threadIdx.x;

    if( tid >= querysize ) return;

    float4 q = make_float4(query[tid*stride],query[tid*stride+1],query[tid*stride+2],0);

    result.setResultLocation( resultDist, resultIndex, tid, resultStride );

    searchNeighbors(splits,child1,parent,aabbMin,aabbMax,elements, q, result, dist);

    result.finish();
}

}

//! contains some pointers that use cuda data types and that cannot be easily
//! forward-declared.
//! basically it contains all GPU buffers
template<typename Distance>
struct KDTreeCuda3dIndex<Distance>::GpuHelper
{
    thrust::device_vector< cuda::kd_tree_builder_detail::SplitInfo >* gpu_splits_;
    thrust::device_vector< int >* gpu_parent_;
    thrust::device_vector< int >* gpu_child1_;
    thrust::device_vector< float4 >* gpu_aabb_min_;
    thrust::device_vector< float4 >* gpu_aabb_max_;
    thrust::device_vector<float4>* gpu_points_;
    thrust::device_vector<int>* gpu_vind_;
    GpuHelper() :  gpu_splits_(0), gpu_parent_(0), gpu_child1_(0), gpu_aabb_min_(0), gpu_aabb_max_(0), gpu_points_(0), gpu_vind_(0){
    }
    ~GpuHelper()
    {
        delete gpu_splits_;
        gpu_splits_=0;
        delete gpu_parent_;
        gpu_parent_=0;
        delete gpu_child1_;
        gpu_child1_=0;
        delete gpu_aabb_max_;
        gpu_aabb_max_=0;
        delete gpu_aabb_min_;
        gpu_aabb_min_=0;
        delete gpu_vind_;
        gpu_vind_=0;

        delete gpu_points_;
        gpu_points_=0;
    }
};

//! thrust transform functor
//! transforms indices in the internal data set back to the original indices
struct map_indices
{
    const int* v_;

    map_indices(const int* v) : v_(v) {
    }

    __host__ __device__
    float operator() (const int&i) const
    {
        if( i>= 0 ) return v_[i];
        else return i;
    }
};

//! implementation of L2 distance for the CUDA kernels
struct CudaL2
{

    static float
    __host__ __device__
    axisDist( float a, float b )
    {
        return (a-b)*(a-b);
    }

    static float
    __host__ __device__
    dist( float4 a, float4 b )
    {
        float4 diff = a-b;
        return dot(diff,diff);
    }
};

//! implementation of L1 distance for the CUDA kernels
//! NOT TESTED!
struct CudaL1
{

    static float
    __host__ __device__
    axisDist( float a, float b )
    {
        return fabs(a-b);
    }

    static float
    __host__ __device__
    dist( float4 a, float4 b )
    {
        return fabs(a.x-b.x)+fabs (a.y-b.y)+( a.z-b.z)+(a.w-b.w);
    }
};

//! used to adapt CPU and GPU distance types.
//! specializations define the ::type as their corresponding GPU distance type
//! \see GpuDistance< L2<float> >, GpuDistance< L2_Simple<float> >
template< class Distance >
struct GpuDistance
{
};

template<>
struct GpuDistance< L2<float> >
{
    typedef CudaL2 type;
};

template<>
struct GpuDistance< L2_Simple<float> >
{
    typedef CudaL2 type;
};
template<>
struct GpuDistance< L1<float> >
{
    typedef CudaL1 type;
};


template< typename Distance >
void KDTreeCuda3dIndex<Distance>::knnSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const
{
    assert(indices.rows >= queries.rows);
    assert(dists.rows >= queries.rows);
    assert(int(indices.cols) >= knn);
    assert( dists.cols == indices.cols && dists.stride==indices.stride );

    int istride=queries.stride/sizeof(ElementType);
    int ostride=indices.stride/4;

    bool matrices_on_gpu = params.matrices_in_gpu_ram;

    int threadsPerBlock = 128;
    int blocksPerGrid=(queries.rows+threadsPerBlock-1)/threadsPerBlock;

    float epsError = 1+params.eps;
    bool sorted = params.sorted;
    bool use_heap = params.use_heap;

    typename GpuDistance<Distance>::type distance;
//       std::cout<<" search: "<<std::endl;
//       std::cout<<"  rows: "<<indices.rows<<" "<<dists.rows<<" "<<queries.rows<<std::endl;
//       std::cout<<"  cols: "<<indices.cols<<" "<<dists.cols<<" "<<queries.cols<<std::endl;
//       std::cout<<"  stride: "<<indices.stride<<" "<<dists.stride<<" "<<queries.stride<<std::endl;
//       std::cout<<"  stride2:"<<istride<<" "<<ostride<<std::endl;
//       std::cout<<"  knn:"<<knn<<"  matrices_on_gpu:"<<matrices_on_gpu<<std::endl;

    if( !matrices_on_gpu ) {
        thrust::device_vector<float> queriesDev(istride* queries.rows,0);
        thrust::copy( queries.ptr(), queries.ptr()+istride*queries.rows, queriesDev.begin() );
        thrust::device_vector<float> distsDev(queries.rows* ostride);
        thrust::device_vector<int> indicesDev(queries.rows* ostride);



        if( knn==1  ) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  ostride,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  thrust::raw_pointer_cast(&distsDev[0]),
                                                                                  queries.rows, flann::cuda::SingleResultSet<float>(epsError),distance);
            //                          KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_nodes_)[0])),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&queriesDev[0]),
            //                                                                                                                                                                                                                                                                                          queries.stride,
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&indicesDev[0]),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&distsDev[0]),
            //                                                                                                                                                                                                                                                                                          queries.rows, epsError);
            //
        }
        else {
            if( use_heap ) {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                      thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                      thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                      istride,
                                                                                      ostride,
                                                                                      thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                      thrust::raw_pointer_cast(&distsDev[0]),
                                                                                      queries.rows, flann::cuda::KnnResultSet<float, true>(knn,sorted,epsError)
                                                                                      , distance);
            }
            else {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                      thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                      thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                      istride,
                                                                                      ostride,
                                                                                      thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                      thrust::raw_pointer_cast(&distsDev[0]),
                                                                                      queries.rows, flann::cuda::KnnResultSet<float, false>(knn,sorted,epsError),
                                                                                      distance
                                                                                      );
            }
        }
        thrust::copy( distsDev.begin(), distsDev.end(), dists.ptr() );
        thrust::transform(indicesDev.begin(), indicesDev.end(), indicesDev.begin(), map_indices(thrust::raw_pointer_cast( &((*gpu_helper_->gpu_vind_))[0]) ));
        thrust::copy( indicesDev.begin(), indicesDev.end(), indices.ptr() );
    }
    else {
        thrust::device_ptr<float> qd = thrust::device_pointer_cast(queries.ptr());
        thrust::device_ptr<float> dd = thrust::device_pointer_cast(dists.ptr());
        thrust::device_ptr<int> id = thrust::device_pointer_cast(indices.ptr());



        if( knn==1  ) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  qd.get(),
                                                                                  istride,
                                                                                  ostride,
                                                                                  id.get(),
                                                                                  dd.get(),
                                                                                  queries.rows, flann::cuda::SingleResultSet<float>(epsError),distance);
            //                          KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_nodes_)[0])),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&queriesDev[0]),
            //                                                                                                                                                                                                                                                                                          queries.stride,
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&indicesDev[0]),
            //                                                                                                                                                                                                                                                                                          thrust::raw_pointer_cast(&distsDev[0]),
            //                                                                                                                                                                                                                                                                                          queries.rows, epsError);
            //
        }
        else {
            if( use_heap ) {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                      thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                      qd.get(),
                                                                                      istride,
                                                                                      ostride,
                                                                                      id.get(),
                                                                                      dd.get(),
                                                                                      queries.rows, flann::cuda::KnnResultSet<float, true>(knn,sorted,epsError)
                                                                                      , distance);
            }
            else {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                      thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                      thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                      qd.get(),
                                                                                      istride,
                                                                                      ostride,
                                                                                      id.get(),
                                                                                      dd.get(),
                                                                                      queries.rows, flann::cuda::KnnResultSet<float, false>(knn,sorted,epsError),
                                                                                      distance
                                                                                      );
            }
        }
        thrust::transform(id, id+knn*queries.rows, id, map_indices(thrust::raw_pointer_cast( &((*gpu_helper_->gpu_vind_))[0]) ));
    }
}


template< typename Distance>
int KDTreeCuda3dIndex<Distance >::radiusSearchGpu(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                                                  std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const
{
    //  assert(indices.roasdfws >= queries.rows);
    //  assert(dists.rows >= queries.rows);

    int max_neighbors = params.max_neighbors;
    bool sorted = params.sorted;
    bool use_heap = params.use_heap;
    if (indices.size() < queries.rows ) indices.resize(queries.rows);
    if (dists.size() < queries.rows ) dists.resize(queries.rows);

    int istride=queries.stride/sizeof(ElementType);

    thrust::device_vector<float> queriesDev(istride* queries.rows,0);
    thrust::copy( queries.ptr(), queries.ptr()+istride*queries.rows, queriesDev.begin() );
    thrust::device_vector<int> countsDev(queries.rows);

    typename GpuDistance<Distance>::type distance;

    int threadsPerBlock = 128;
    int blocksPerGrid=(queries.rows+threadsPerBlock-1)/threadsPerBlock;


    KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                          thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                          thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                          thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                          thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                          thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                          thrust::raw_pointer_cast(&queriesDev[0]),
                                                                          istride,
                                                                          1,
                                                                          thrust::raw_pointer_cast(&countsDev[0]),
                                                                          0,
                                                                          queries.rows, flann::cuda::CountingRadiusResultSet<float>(radius,max_neighbors),
                                                                          distance
                                                                          );

    thrust::host_vector<int> counts_host=countsDev;

    if( max_neighbors!=0 ) { // we'll need this later, but the exclusive_scan will change the array
        for( size_t i=0; i<queries.rows; i++ ) {
            int count = counts_host[i];
            if( count > 0 ) {
                indices[i].resize(count);
                dists[i].resize(count);
            }
            else {
                indices[i].clear();
                dists[i].clear();
            }

        }
    }

    int neighbors_last_elem = countsDev.back();
    thrust::exclusive_scan( countsDev.begin(), countsDev.end(), countsDev.begin() );

    size_t total_neighbors=neighbors_last_elem+countsDev.back();
    if( max_neighbors==0 ) return total_neighbors;

    thrust::device_vector<int> indicesDev(total_neighbors,-1);
    thrust::device_vector<float> distsDev(total_neighbors,std::numeric_limits<float>::infinity());

    if( max_neighbors<0 ) {
        KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                              thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                              thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                              thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                              thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                              thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                              thrust::raw_pointer_cast(&queriesDev[0]),
                                                                              istride,
                                                                              1,
                                                                              thrust::raw_pointer_cast(&indicesDev[0]),
                                                                              thrust::raw_pointer_cast(&distsDev[0]),
                                                                              queries.rows, flann::cuda::RadiusResultSet<float>(radius,thrust::raw_pointer_cast(&countsDev[0]),sorted), distance);
    }
    else {
        if( use_heap ) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  1,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  thrust::raw_pointer_cast(&distsDev[0]),
                                                                                  queries.rows, flann::cuda::RadiusKnnResultSet<float, true>(radius,max_neighbors, thrust::raw_pointer_cast(&countsDev[0]),sorted), distance);
        }
        else {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  1,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  thrust::raw_pointer_cast(&distsDev[0]),
                                                                                  queries.rows, flann::cuda::RadiusKnnResultSet<float, false>(radius,max_neighbors, thrust::raw_pointer_cast(&countsDev[0]),sorted), distance);
        }
    }
    thrust::transform(indicesDev.begin(), indicesDev.end(), indicesDev.begin(), map_indices(thrust::raw_pointer_cast( &((*gpu_helper_->gpu_vind_))[0]) ));
    thrust::host_vector<int> indices_temp = indicesDev;
    thrust::host_vector<float> dists_temp = distsDev;

    int buffer_index=0;
    for( size_t i=0; i<queries.rows; i++ ) {
        for( size_t j=0; j<counts_host[i]; j++ ) {
            dists[i][j]=dists_temp[buffer_index];
            indices[i][j]=indices_temp[buffer_index];
            ++buffer_index;
        }
    }

    return buffer_index;
}

//! used in the radius search to count the total number of neighbors
struct isNotMinusOne
{
    __host__ __device__
    bool operator() ( int i ){
        return i!=-1;
    }
};

template< typename Distance>
int KDTreeCuda3dIndex< Distance >::radiusSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, float radius, const SearchParams& params) const
{
	int max_neighbors = params.max_neighbors;
    assert(indices.rows >= queries.rows);
    assert(dists.rows >= queries.rows || max_neighbors==0 );
    assert(indices.stride==dists.stride  || max_neighbors==0 );
    assert( indices.cols==indices.stride/sizeof(int) );
    assert(dists.rows >= queries.rows || max_neighbors==0 );

    bool sorted = params.sorted;
    bool matrices_on_gpu = params.matrices_in_gpu_ram;
    float epsError = 1+params.eps;
    bool use_heap = params.use_heap;
    int istride=queries.stride/sizeof(ElementType);
    int ostride= indices.stride/4;


    if( max_neighbors<0 ) max_neighbors=indices.cols;

    if( !matrices_on_gpu ) {
        thrust::device_vector<float> queriesDev(istride* queries.rows,0);
        thrust::copy( queries.ptr(), queries.ptr()+istride*queries.rows, queriesDev.begin() );
        typename GpuDistance<Distance>::type distance;
        int threadsPerBlock = 128;
        int blocksPerGrid=(queries.rows+threadsPerBlock-1)/threadsPerBlock;
        if( max_neighbors== 0 ) {
            thrust::device_vector<int> indicesDev(queries.rows* ostride);
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  ostride,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  0,
                                                                                  queries.rows, flann::cuda::CountingRadiusResultSet<float>(radius,-1),
                                                                                  distance
                                                                                  );
            thrust::copy( indicesDev.begin(), indicesDev.end(), indices.ptr() );
            return thrust::reduce(indicesDev.begin(), indicesDev.end() );
        }


        thrust::device_vector<float> distsDev(queries.rows* max_neighbors);
        thrust::device_vector<int> indicesDev(queries.rows* max_neighbors);

        if( use_heap ) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  ostride,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  thrust::raw_pointer_cast(&distsDev[0]),
                                                                                  queries.rows, flann::cuda::KnnRadiusResultSet<float, true>(max_neighbors,sorted,epsError, radius), distance);
        }
        else {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  thrust::raw_pointer_cast(&queriesDev[0]),
                                                                                  istride,
                                                                                  ostride,
                                                                                  thrust::raw_pointer_cast(&indicesDev[0]),
                                                                                  thrust::raw_pointer_cast(&distsDev[0]),
                                                                                  queries.rows, flann::cuda::KnnRadiusResultSet<float, false>(max_neighbors,sorted,epsError, radius), distance);
        }

        thrust::copy( distsDev.begin(), distsDev.end(), dists.ptr() );
        thrust::transform(indicesDev.begin(), indicesDev.end(), indicesDev.begin(), map_indices(thrust::raw_pointer_cast( &((*gpu_helper_->gpu_vind_))[0]) ));
        thrust::copy( indicesDev.begin(), indicesDev.end(), indices.ptr() );

        return thrust::count_if(indicesDev.begin(), indicesDev.end(), isNotMinusOne() );
    }
    else {

        thrust::device_ptr<float> qd=thrust::device_pointer_cast(queries.ptr());
        thrust::device_ptr<float> dd=thrust::device_pointer_cast(dists.ptr());
        thrust::device_ptr<int> id=thrust::device_pointer_cast(indices.ptr());
        typename GpuDistance<Distance>::type distance;
        int threadsPerBlock = 128;
        int blocksPerGrid=(queries.rows+threadsPerBlock-1)/threadsPerBlock;

        if( max_neighbors== 0 ) {
            thrust::device_vector<int> indicesDev(queries.rows* indices.stride);
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  qd.get(),
                                                                                  istride,
                                                                                  ostride,
                                                                                  id.get(),
                                                                                  0,
                                                                                  queries.rows, flann::cuda::CountingRadiusResultSet<float>(radius,-1),
                                                                                  distance
                                                                                  );
            thrust::copy( indicesDev.begin(), indicesDev.end(), indices.ptr() );
            return thrust::reduce(indicesDev.begin(), indicesDev.end() );
        }

        if( use_heap ) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  qd.get(),
                                                                                  istride,
                                                                                  ostride,
                                                                                  id.get(),
                                                                                  dd.get(),
                                                                                  queries.rows, flann::cuda::KnnRadiusResultSet<float, true>(max_neighbors,sorted,epsError, radius), distance);
        }
        else {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock>>> (thrust::raw_pointer_cast(&((*gpu_helper_->gpu_splits_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                                                                                  thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                                                                                  thrust::raw_pointer_cast( &((*gpu_helper_->gpu_points_)[0]) ),
                                                                                  qd.get(),
                                                                                  istride,
                                                                                  ostride,
                                                                                  id.get(),
                                                                                  dd.get(),
                                                                                  queries.rows, flann::cuda::KnnRadiusResultSet<float, false>(max_neighbors,sorted,epsError, radius), distance);
        }

        thrust::transform(id, id+max_neighbors*queries.rows, id, map_indices(thrust::raw_pointer_cast( &((*gpu_helper_->gpu_vind_))[0]) ));

        return thrust::count_if(id, id+max_neighbors*queries.rows, isNotMinusOne() );
    }
}

template<typename Distance>
void KDTreeCuda3dIndex<Distance>::uploadTreeToGpu()
{
    // just make sure that no weird alignment stuff is going on...
    // shouldn't, but who knows
    // (I would make this a (boost) static assertion, but so far flann seems to avoid boost
    //  assert( sizeof( KdTreeCudaPrivate::GpuNode)==sizeof( Node ) );
    delete gpu_helper_;
    gpu_helper_ = new GpuHelper;
    gpu_helper_->gpu_points_=new thrust::device_vector<float4>(size_);
    thrust::device_vector<float4> tmp(size_);
    if( get_param(index_params_,"input_is_gpu_float4",false) ) {
		assert( dataset_.cols == 3 && dataset_.stride==4*sizeof(float));
        thrust::copy( thrust::device_pointer_cast((float4*)dataset_.ptr()),thrust::device_pointer_cast((float4*)(dataset_.ptr()))+size_,tmp.begin());

    }
    else {
        // k is limited to 4 -> use 128bit-alignment regardless of dimensionality
        // makes cpu search about 5% slower, but gpu can read a float4 w/ a single instruction
        // (vs a float2 and a float load for a float3 value)
        // pad data directly to avoid having to copy and re-format the data when
        // copying it to the GPU
        data_ = flann::Matrix<ElementType>(new ElementType[size_*4], size_, dim_,4*4);
        for (size_t i=0; i<size_; ++i) {
            for (size_t j=0; j<dim_; ++j) {
                data_[i][j] = dataset_[i][j];
            }
            for (size_t j=dim_; j<4; ++j) {
                data_[i][j] = 0;
            }
        }
        thrust::copy((float4*)data_.ptr(),(float4*)(data_.ptr())+size_,tmp.begin());
    }

    CudaKdTreeBuilder builder( tmp, leaf_max_size_ );
    builder.buildTree();

    gpu_helper_->gpu_splits_ = builder.splits_;
    gpu_helper_->gpu_aabb_min_ = builder.aabb_min_;
    gpu_helper_->gpu_aabb_max_ = builder.aabb_max_;
    gpu_helper_->gpu_child1_ = builder.child1_;
    gpu_helper_->gpu_parent_=builder.parent_;
    gpu_helper_->gpu_vind_=builder.index_x_;
    thrust::gather( builder.index_x_->begin(), builder.index_x_->end(), tmp.begin(), gpu_helper_->gpu_points_->begin());

    //  gpu_helper_->gpu_nodes_=new thrust::device_vector<KdTreeCudaPrivate::GpuNode>(node_count_);


    //  gpu_helper_->gpu_vind_=new thrust::device_vector<int>(size_);
    //  thrust::copy( (KdTreeCudaPrivate::GpuNode*)&(tree_[0]), ((KdTreeCudaPrivate::GpuNode*)&(tree_[0]))+tree_.size(),  gpu_helper_->gpu_nodes_->begin());

    //  thrust::copy(vind_.begin(),vind_.end(),gpu_helper_->gpu_vind_->begin());

    //  buildGpuTree();
}


template<typename Distance>
void KDTreeCuda3dIndex<Distance>::clearGpuBuffers()
{
    delete gpu_helper_;
    gpu_helper_=0;
}

// explicit instantiations for distance-independent functions
template
void KDTreeCuda3dIndex<flann::L2<float> >::uploadTreeToGpu();

template
void KDTreeCuda3dIndex<flann::L2<float> >::clearGpuBuffers();

template
struct KDTreeCuda3dIndex<flann::L2<float> >::GpuHelper;

template
void KDTreeCuda3dIndex<flann::L2<float> >::knnSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const;

template
int KDTreeCuda3dIndex< flann::L2<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, float radius, const SearchParams& params) const;
template
int KDTreeCuda3dIndex< flann::L2<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                                                           std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const;

// explicit instantiations for distance-independent functions
template
void KDTreeCuda3dIndex<flann::L2_Simple<float> >::uploadTreeToGpu();

template
void KDTreeCuda3dIndex<flann::L2_Simple<float> >::clearGpuBuffers();

template
struct KDTreeCuda3dIndex<flann::L2_Simple<float> >::GpuHelper;

template
void KDTreeCuda3dIndex<flann::L2_Simple<float> >::knnSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const;

template
int KDTreeCuda3dIndex< flann::L2_Simple<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, float radius, const SearchParams& params) const;
template
int KDTreeCuda3dIndex< flann::L2_Simple<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                                                                  std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const;


// explicit instantiations for distance-independent functions
template
void KDTreeCuda3dIndex<flann::L1<float> >::uploadTreeToGpu();

template
void KDTreeCuda3dIndex<flann::L1<float> >::clearGpuBuffers();

template
struct KDTreeCuda3dIndex<flann::L1<float> >::GpuHelper;

template
void KDTreeCuda3dIndex<flann::L1<float> >::knnSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, size_t knn, const SearchParams& params) const;

template
int KDTreeCuda3dIndex< flann::L1<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, Matrix<int>& indices, Matrix<DistanceType>& dists, float radius, const SearchParams& params) const;
template
int KDTreeCuda3dIndex< flann::L1<float> >::radiusSearchGpu(const Matrix<ElementType>& queries, std::vector< std::vector<int> >& indices,
                                                           std::vector<std::vector<DistanceType> >& dists, float radius, const SearchParams& params) const;
}
