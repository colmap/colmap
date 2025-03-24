// Fixes the gauge by setting the rotation and translation of one camera as
// constant and one translational dimension of another camera. Notice that there
// are some degenerate cases where the current logic fails to properly fix the
// gauge (e.g., when the observations in the image are on a line, etc.).
void FixGaugeWithTwoCamsFromWorld(const BundleAdjustmentOptions& options,
                                  const BundleAdjustmentConfig& config,
                                  const std::set<image_t>& image_ids,
                                  Reconstruction& reconstruction,
                                  ceres::Problem& problem) {
  // Check if two camera poses are already constant. Then, we don't need to
  const size_t num_constant_images = std::count_if(
      image_ids.begin(),
      image_ids.end(),
      [&config, &reconstruction](image_t image_id) {
        Image& image = reconstruction.Image(image_id);
        if (image.HasTrivialFrame()) {
          return config.HasConstantFrameFromWorldPose(image.FrameId());
        } else {
          return config.HasConstantSensorFromRig(image.FramePtr()->RigId()) &&
                 config.HasConstantFrameFromWorldPose(image.FrameId());
        }
      });
  if (num_constant_images >= 2) {
    return;
  }

  // Check if one of the baseline dimensions is large enough and
  // choose it as the fixed coordinate. If there is no such pair of
  // frames, then the scale is not constrained well.

  Image* image1 = nullptr;
  Image* image2 = nullptr;
  int largest_baseline_dim = 0;

  auto has_constant_cam_from_world = [](const Image& image) {
    if (image.HasTrivialFrame()) {
      return config.HasConstantFrameFromWorldPose(image.FrameId());
    } else {
      return config.HasConstantSensorFromRig(image.FramePtr()->RigId()) &&
             config.HasConstantFrameFromWorldPose(image.FrameId());
    }
  };

  // First, try to find images whose poses are already constant.
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (image1 == nullptr && has_constant_cam_from_world(image)) {
      image1 = &image;
    } else if (image1 != nullptr && has_constant_cam_from_world(image)) {
      if (const std::optional<int> dim =
              FindLargestBaselineDimOrNull(*image1, image);
          dim.has_value()) {
        image2 = &image;
        largest_baseline_dim = *dim;
        break;
      }
    }
  }

  if (image1 != nullptr && image2 != nullptr) {
    return;
  }

  // Second, try to find image with trivial frames, so we only have to constrain
  // the frame_from_world pose instead of also fixing the sensor_from_rig poses.
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (image1 == nullptr && image.HasTrivialFrame()) {
      image1 = &image;
    } else if (image1 != nullptr && image.HasTrivialFrame()) {
      if (const std::optional<int> dim =
              FindLargestBaselineDimOrNull(*image1, image);
          dim.has_value()) {
        image2 = &image;
        largest_baseline_dim = *dim;
        break;
      }
    }
  }

  // Third, try to find ones with non-trivial frames as a fallback.
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (image1 == nullptr) {
      image1 = &image;
    } else if (const std::optional<int> dim =
                   FindLargestBaselineDimOrNull(*image1, image);
               dim.has_value()) {
      image2 = &image;
      largest_baseline_dim = *dim;
      break;
    }
  }

  // TODO(jsch): Once we support IMUs or other sensors, we have to
  // fix the Gauge differently, as we are not guaranteed to find two
  // images/cameras that are reference sensors in different frames.
  THROW_CHECK(image1 != nullptr && image2 != nullptr);

  if (image1->HasTrivialFrame() && image1->FrameId() == image2->FrameId()) {
    THROW_CHECK(!image2->HasTrivialFrame());
    SetSubsetManifold(3,
                      {static_cast<int>(largest_baseline_dim)},
                      &problem,
                      image2->FramePtr()
                          ->RigPtr()
                          ->SensorFromRig(image2->CameraPtr()->SensorId())
                          .translation.data());
  } else {
  }

  Rigid3d& frame1_from_world = image1->FramePtr()->FrameFromWorld();
  if (!config.HasConstantFrameFromWorldPose(image1->FrameId())) {
    problem.SetParameterBlockConstant(
        frame1_from_world.rotation.coeffs().data());
    problem.SetParameterBlockConstant(frame1_from_world.translation.data());
  }

  Rigid3d& frame2_from_world = image2->FramePtr()->FrameFromWorld();
  if (!config.HasConstantFrameFromWorldPose(image2->FrameId())) {
    SetSubsetManifold(3,
                      {static_cast<int>(largest_baseline_dim)},
                      &problem,
                      frame2_from_world.translation.data());
  }
}