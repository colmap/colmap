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
#include <omp.h>

namespace colmap {
namespace {

class FaissFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  explicit FaissFeatureDescriptorIndex(int num_threads)
      : num_threads_(num_threads) {}

  void Build(const FeatureDescriptorsFloat& index_descriptors) override {
    if (index_descriptors.rows() == 0) {
      index_ = nullptr;
      return;
    }

#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(num_threads_);
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif

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
        index_->train(index_descriptors.rows(), index_descriptors.data());
        index_->add(index_descriptors.rows(), index_descriptors.data());
      } else {
        index_ = std::make_unique<faiss::IndexFlatL2>(
            /*d=*/index_descriptors.cols());
        index_->add(index_descriptors.rows(), index_descriptors.data());
      }
    }
  }

  void Search(int num_neighbors,
              const FeatureDescriptorsFloat& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXf& l2_dists) const override {
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

    l2_dists.resize(num_query_descriptors, num_eff_neighbors);
    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        indices_long(num_query_descriptors, num_eff_neighbors);

#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(num_threads_);
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif

      faiss::SearchParametersIVF search_params;
      search_params.nprobe = 8;
      index_->search(num_query_descriptors,
                     query_descriptors.data(),
                     num_eff_neighbors,
                     l2_dists.data(),
                     indices_long.data(),
                     &search_params);
    }

    indices = indices_long.cast<int>();
  }

 private:
  const int num_threads_;
  std::unique_ptr<faiss::Index> index_;
  std::unique_ptr<faiss::IndexFlatL2> coarse_quantizer_;
};

}  // namespace

std::unique_ptr<FeatureDescriptorIndex> FeatureDescriptorIndex::Create(
    Type type, int num_threads) {
  switch (type) {
    case Type::FAISS:
      return std::make_unique<FaissFeatureDescriptorIndex>(num_threads);
    default:
      throw std::runtime_error("Feature descriptor index not implemented");
  }
  return nullptr;
}

}  // namespace colmap
