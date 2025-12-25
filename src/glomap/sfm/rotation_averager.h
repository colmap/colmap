#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/estimators/global_rotation_averaging.h"

namespace glomap {

struct RotationAveragerOptions : public RotationEstimatorOptions {
  RotationAveragerOptions() = default;
  explicit RotationAveragerOptions(const RotationEstimatorOptions& options)
      : RotationEstimatorOptions(options) {}
  bool use_stratified = true;
};

bool SolveRotationAveraging(ViewGraph& view_graph,
                            colmap::Reconstruction& reconstruction,
                            std::vector<colmap::PosePrior>& pose_priors,
                            const RotationAveragerOptions& options);

}  // namespace glomap
