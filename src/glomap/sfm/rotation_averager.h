#pragma once

#include "colmap/scene/reconstruction.h"

#include "glomap/estimators/rotation_averaging.h"

namespace glomap {

bool SolveRotationAveraging(ViewGraph& view_graph,
                            colmap::Reconstruction& reconstruction,
                            std::vector<colmap::PosePrior>& pose_priors,
                            const RotationEstimatorOptions& options);

}  // namespace glomap
