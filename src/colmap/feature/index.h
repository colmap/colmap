// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/util/types.h"

#include <memory>

namespace colmap {

class FeatureDescriptorIndex {
 public:
  enum class Type {
    DEFAULT = 1,
    FAISS = 1,
  };

  virtual ~FeatureDescriptorIndex() = default;

  static std::unique_ptr<FeatureDescriptorIndex> Create(
      Type type = Type::DEFAULT, int num_threads = 1);

  virtual void Build(const FeatureDescriptorsFloat& descriptors) = 0;

  virtual void Search(int num_neighbors,
                      const FeatureDescriptorsFloat& query_descriptors,
                      Eigen::RowMajorMatrixXi& indices,
                      Eigen::RowMajorMatrixXf& l2_dists) const = 0;
};

}  // namespace colmap
