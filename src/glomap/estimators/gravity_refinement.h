#pragma once

#include "glomap/estimators/optimization_base.h"
#include "glomap/scene/types_sfm.h"
#include "glomap/types.h"

#include <ceres/ceres.h>

namespace glomap {

struct GravityRefinerOptions : public OptimizationBaseOptions {
  // The minimal ratio that the gravity vector should be consistent with
  double max_outlier_ratio = 0.5;
  // The maximum allowed angle error in degree
  double max_gravity_error = 1.;
  // Only refine the gravity of the images with more than min_neighbors
  int min_num_neighbors = 7;

  GravityRefinerOptions() : OptimizationBaseOptions() {}

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::ArctanLoss>(
        1 - std::cos(colmap::DegToRad(max_gravity_error)));
  }
};

class GravityRefiner {
 public:
  explicit GravityRefiner(const GravityRefinerOptions& options)
      : options_(options) {}

  void RefineGravity(const ViewGraph& view_graph,
                     const std::unordered_map<frame_t, Frame>& frames,
                     const std::unordered_map<image_t, Image>& images,
                     std::vector<colmap::PosePrior>& pose_priors);

 private:
  void IdentifyErrorProneGravity(
      const ViewGraph& view_graph,
      const std::unordered_map<frame_t, Frame>& frames,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<image_t, colmap::PosePrior*>& image_to_pose_prior,
      std::unordered_set<frame_t>& error_prone_frames);
  GravityRefinerOptions options_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace glomap
