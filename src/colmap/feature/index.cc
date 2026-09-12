// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/index.h"

#include "colmap/util/logging.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace colmap {
namespace {

class FaissFeatureDescriptorIndex : public FeatureDescriptorIndex {
 public:
  explicit FaissFeatureDescriptorIndex(int num_threads)
      : num_threads_(num_threads) {}

  void Build(const FeatureDescriptorsFloat& index_descriptors) override {
    type_ = index_descriptors.type;
    if (index_descriptors.data.rows() == 0) {
      index_ = nullptr;
      return;
    }

#pragma omp parallel num_threads(1)
    {
#ifdef _OPENMP
      omp_set_num_threads(num_threads_);
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif
#endif

      if (index_descriptors.data.rows() >= 512) {
        const int num_centroids = 4 * std::sqrt(index_descriptors.data.rows());
        coarse_quantizer_ =
            std::make_unique<faiss::IndexFlatL2>(index_descriptors.data.cols());
        if (type_ == FeatureExtractorType::SIFT) {
          // SIFT descriptors are natively uint8, so QT_8bit_direct
          // quantization is lossless and faster than flat indexing.
          index_ = std::make_unique<faiss::IndexIVFScalarQuantizer>(
              /*quantizer=*/coarse_quantizer_.get(),
              /*d=*/index_descriptors.data.cols(),
              /*nlist=*/num_centroids,
              faiss::ScalarQuantizer::QT_8bit_direct,
              faiss::METRIC_L2,
              /*by_residual=*/false);
          auto* index_impl =
              static_cast<faiss::IndexIVFScalarQuantizer*>(index_.get());
          index_impl->cp.min_points_per_centroid = 1;
        } else {
          index_ = std::make_unique<faiss::IndexIVFFlat>(
              /*quantizer=*/coarse_quantizer_.get(),
              /*d=*/index_descriptors.data.cols(),
              /*nlist=*/num_centroids,
              faiss::METRIC_L2);
          auto* index_impl = static_cast<faiss::IndexIVFFlat*>(index_.get());
          index_impl->cp.min_points_per_centroid = 1;
        }
        index_->train(index_descriptors.data.rows(),
                      index_descriptors.data.data());
        index_->add(index_descriptors.data.rows(),
                    index_descriptors.data.data());
      } else {
        index_ = std::make_unique<faiss::IndexFlatL2>(
            /*d=*/index_descriptors.data.cols());
        index_->add(index_descriptors.data.rows(),
                    index_descriptors.data.data());
      }
    }
  }

  void Search(int num_neighbors,
              const FeatureDescriptorsFloat& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXf& l2_dists) const override {
    THROW_CHECK_EQ(query_descriptors.type, type_);

    if (num_neighbors <= 0 || index_ == nullptr) {
      indices.resize(0, 0);
      l2_dists.resize(0, 0);
      return;
    }

    THROW_CHECK_EQ(query_descriptors.data.cols(), index_->d);
    const int64_t num_query_descriptors = query_descriptors.data.rows();
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
#ifdef _OPENMP
      omp_set_num_threads(num_threads_);
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif
#endif

      faiss::SearchParametersIVF search_params;
      search_params.nprobe = 8;
      index_->search(num_query_descriptors,
                     query_descriptors.data.data(),
                     num_eff_neighbors,
                     l2_dists.data(),
                     indices_long.data(),
                     &search_params);
    }

    indices = indices_long.cast<int>();
  }

 private:
  const int num_threads_;
  FeatureExtractorType type_ = FeatureExtractorType::UNDEFINED;
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
