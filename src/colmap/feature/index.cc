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

#include <flann/flann.hpp>

namespace colmap {
namespace {

// Silence clang-tidy warning:
// Call to virtual method 'KDTreeIndex::freeIndex' during destruction bypasses
// virtual dispatch.
// NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
class FlannFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  void Build(const FeatureDescriptors& index_descriptors) override {
    THROW_CHECK_EQ(index_descriptors.cols(), 128);
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
    THROW_CHECK_NOTNULL(index_);
    THROW_CHECK_EQ(query_descriptors.cols(), 128);

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

std::unique_ptr<FeatureDescriptorIndex> FeatureDescriptorIndex::Create() {
  return std::make_unique<FlannFeatureDescriptorIndex>();
}

}  // namespace colmap
