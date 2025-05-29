// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/feature/matcher.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <flann/flann.hpp>

namespace colmap {
namespace {

class FaissFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  explicit FaissFeatureDescriptorIndex(int num_threads)
      : num_threads_(num_threads) {}

  void Build(const FeatureDescriptors& index_descriptors) override {
    if (index_descriptors.rows() == 0) {
      index_ = nullptr;
      return;
    }

    const Eigen::RowMajorMatrixXf index_descriptors_float =
        index_descriptors.cast<float>();

#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(num_threads_);
      omp_set_max_active_levels(1);

      if (index_descriptors.rows() >= 512) {
        const int num_centroids = 4 * std::sqrt(index_descriptors.rows());
        coarse_quantizer_ =
            std::make_unique<faiss::IndexFlatL2>(index_descriptors.cols());
        index_ = std::make_unique<faiss::IndexIVFFlat>(
            /*quantizer=*/coarse_quantizer_.get(),
            /*d=*/index_descriptors.cols(),
            /*nlist_=*/num_centroids);
        auto* index_impl = dynamic_cast<faiss::IndexIVFFlat*>(index_.get());
        // Avoid warnings during the training phase.
        index_impl->cp.min_points_per_centroid = 1;
        index_->train(index_descriptors.rows(), index_descriptors_float.data());
        index_->add(index_descriptors.rows(), index_descriptors_float.data());
      } else {
        index_ = std::make_unique<faiss::IndexFlatL2>(
            /*d=*/index_descriptors.cols());
        index_->add(index_descriptors.rows(), index_descriptors_float.data());
      }
    }
  }

  void Search(int num_neighbors,
              const FeatureDescriptors& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXi& l2_dists) const override {
    if (num_neighbors <= 0 || index_ == nullptr) {
      indices.resize(0, 0);
      l2_dists.resize(0, 0);
      return;
    }

    THROW_CHECK_EQ(query_descriptors.cols(), index_->d);
    const int64_t num_query_descriptors = query_descriptors.rows();
    if (num_query_descriptors == 0) {
      return;
    }

    const int64_t num_eff_neighbors =
        std::min<int64_t>(num_neighbors, index_->ntotal);

    indices.resize(num_query_descriptors, num_eff_neighbors);
    l2_dists.resize(num_query_descriptors, num_eff_neighbors);

    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        indices_long(num_query_descriptors, num_eff_neighbors);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        l2_dists_float(num_query_descriptors, num_eff_neighbors);
    const Eigen::RowMajorMatrixXf query_descriptors_float =
        query_descriptors.cast<float>();

#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(num_threads_);
      omp_set_max_active_levels(1);

      faiss::SearchParametersIVF search_params;
      search_params.nprobe = 8;
      index_->search(num_query_descriptors,
                     query_descriptors_float.data(),
                     num_eff_neighbors,
                     l2_dists_float.data(),
                     indices_long.data(),
                     &search_params);
    }

    // TODO(jsch): Change the output matrix types to avoid unnecessary
    // allocation and casting. This was optimized for the flann interface
    // before.
    for (int query_idx = 0; query_idx < num_query_descriptors; ++query_idx) {
      for (int k = 0; k < num_eff_neighbors; ++k) {
        indices(query_idx, k) = indices_long(query_idx, k);
        l2_dists(query_idx, k) =
            static_cast<int>(std::round(l2_dists_float(query_idx, k)));
      }
    }
  }

 private:
  const int num_threads_;
  std::unique_ptr<faiss::Index> index_;
  std::unique_ptr<faiss::IndexFlatL2> coarse_quantizer_;
};

// Silence clang-tidy warning:
// Call to virtual method 'KDTreeIndex::freeIndex' during destruction bypasses
// virtual dispatch.
// NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
class FlannFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  void Build(const FeatureDescriptors& index_descriptors) override {
    num_index_descriptors_ = index_descriptors.rows();
    if (num_index_descriptors_ == 0) {
      // Flann is not happy when the input has no descriptors.
      index_ = nullptr;
      return;
    }
    const flann::Matrix<uint8_t> descriptors_matrix(
        const_cast<uint8_t*>(index_descriptors.data()),
        num_index_descriptors_,
        index_descriptors.cols());
    index_ = std::make_unique<FlannIndexType>(
        descriptors_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
    index_->buildIndex();
  }

  void Search(int num_neighbors,
              const FeatureDescriptors& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXi& l2_dists) const override {
    if (num_neighbors <= 0 || index_ == nullptr) {
      indices.resize(0, 0);
      l2_dists.resize(0, 0);
      return;
    }

    THROW_CHECK_EQ(query_descriptors.cols(), index_->veclen());
    const int num_query_descriptors = query_descriptors.rows();
    if (num_query_descriptors == 0) {
      return;
    }

    const int num_eff_neighbors =
        std::min(num_neighbors, num_index_descriptors_);

    indices.resize(num_query_descriptors, num_eff_neighbors);
    l2_dists.resize(num_query_descriptors, num_eff_neighbors);
    const flann::Matrix<uint8_t> query_matrix(
        const_cast<uint8_t*>(query_descriptors.data()),
        num_query_descriptors,
        query_descriptors.cols());

    flann::Matrix<int> indices_matrix(
        indices.data(), num_query_descriptors, num_eff_neighbors);
    std::vector<float> l2_dist_vector(num_query_descriptors *
                                      num_eff_neighbors);
    flann::Matrix<float> l2_dist_matrix(
        l2_dist_vector.data(), num_query_descriptors, num_eff_neighbors);
    index_->knnSearch(query_matrix,
                      indices_matrix,
                      l2_dist_matrix,
                      num_eff_neighbors,
                      flann::SearchParams(kNumLeavesToVisit));

    for (int query_idx = 0; query_idx < num_query_descriptors; ++query_idx) {
      for (int k = 0; k < num_eff_neighbors; ++k) {
        l2_dists(query_idx, k) = static_cast<int>(
            std::round(l2_dist_vector[query_idx * num_eff_neighbors + k]));
      }
    }
  }

 private:
  // Tuned to produce similar results to brute-force matching. If speed is
  // important, the parameters can be reduced. The biggest speed improvement can
  // be gained by reducing the number of leaves.
  constexpr static int kNumTreesInForest = 4;
  constexpr static int kNumLeavesToVisit = 128;

  using FlannIndexType = flann::KDTreeIndex<flann::L2<uint8_t>>;
  std::unique_ptr<FlannIndexType> index_;
  int num_index_descriptors_ = 0;
};

}  // namespace

std::unique_ptr<FeatureDescriptorIndex> FeatureDescriptorIndex::Create(
    Type type, int num_threads) {
  switch (type) {
    case Type::FAISS:
      return std::make_unique<FaissFeatureDescriptorIndex>(num_threads);
    case Type::FLANN:
      return std::make_unique<FlannFeatureDescriptorIndex>();
    default:
      throw std::runtime_error("Feature descriptor index not implemented");
  }
  return nullptr;
}

}  // namespace colmap
