#include "reconstruction_normalizer.h"

namespace glomap {

colmap::Sim3d NormalizeReconstruction(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    bool fixed_scale,
    double extent,
    double p0,
    double p1) {
  // Coordinates of image centers or point locations.
  std::vector<float> coords_x;
  std::vector<float> coords_y;
  std::vector<float> coords_z;

  coords_x.reserve(images.size());
  coords_y.reserve(images.size());
  coords_z.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    if (!image.IsRegistered()) continue;
    const Eigen::Vector3d proj_center = image.Center();
    coords_x.push_back(static_cast<float>(proj_center(0)));
    coords_y.push_back(static_cast<float>(proj_center(1)));
    coords_z.push_back(static_cast<float>(proj_center(2)));
  }

  // Determine robust bounding box and mean.
  std::sort(coords_x.begin(), coords_x.end());
  std::sort(coords_y.begin(), coords_y.end());
  std::sort(coords_z.begin(), coords_z.end());

  const size_t P0 = static_cast<size_t>(
      (coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0);
  const size_t P1 = static_cast<size_t>(
      (coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1);

  const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]);
  const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]);

  Eigen::Vector3d mean_coord(0, 0, 0);
  for (size_t i = P0; i <= P1; ++i) {
    mean_coord(0) += coords_x[i];
    mean_coord(1) += coords_y[i];
    mean_coord(2) += coords_z[i];
  }
  mean_coord /= P1 - P0 + 1;

  // Calculate scale and translation, such that
  // translation is applied before scaling.
  double scale = 1.;
  if (!fixed_scale) {
    const double old_extent = (bbox_max - bbox_min).norm();
    if (old_extent >= std::numeric_limits<double>::epsilon()) {
      scale = extent / old_extent;
    }
  }
  colmap::Sim3d tform(
      scale, Eigen::Quaterniond::Identity(), -scale * mean_coord);

  for (auto& [_, frame] : frames) {
    if (!frame.HasPose()) continue;
    Rigid3d& rig_from_world = frame.RigFromWorld();
    rig_from_world = TransformCameraWorld(tform, rig_from_world);
  }

  for (auto& [_, rig] : rigs) {
    for (auto& [sensor_id, sensor_from_rig_opt] : rig.NonRefSensors()) {
      if (sensor_from_rig_opt.has_value()) {
        Rigid3d sensor_from_rig = sensor_from_rig_opt.value();
        sensor_from_rig.translation *= scale;
        rig.SetSensorFromRig(sensor_id, sensor_from_rig);
      }
    }
  }

  for (auto& [_, track] : tracks) {
    track.xyz = tform * track.xyz;
  }

  return tform;
}

}  // namespace glomap
