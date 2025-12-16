#include "glomap/processors/track_filter.h"

namespace glomap {

int TrackFilter::FilterTracksByReprojection(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double max_reprojection_error,
    bool in_normalized_image) {
  int counter = 0;
  for (auto& [track_id, track] : tracks) {
    std::vector<colmap::TrackElement> observation_new;
    for (auto& observation : track.track.Elements()) {
      const Image& image = images.at(observation.image_id);
      Eigen::Vector3d pt_calc = image.CamFromWorld() * track.xyz;
      constexpr double kEps = 1e-12;
      if (pt_calc(2) < kEps) continue;

      double reprojection_error = max_reprojection_error;
      if (in_normalized_image) {
        const Eigen::Vector3d& feature_undist =
            image.features_undist.at(observation.point2D_idx);

        Eigen::Vector2d pt_reproj = pt_calc.head(2) / pt_calc(2);
        reprojection_error =
            (pt_reproj - feature_undist.head(2) / (feature_undist(2) + kEps))
                .norm();
      } else {
        const Eigen::Vector2d pt_dist = cameras.at(image.CameraId())
                                            .ImgFromCam(pt_calc)
                                            .value_or(Eigen::Vector2d::Zero());
        reprojection_error =
            (pt_dist - image.features.at(observation.point2D_idx)).norm();
      }

      // If the reprojection error is smaller than the threshold, then keep it
      if (reprojection_error < max_reprojection_error) {
        observation_new.emplace_back(observation.image_id,
                                     observation.point2D_idx);
      }
    }
    if (observation_new.size() != track.track.Length()) {
      counter++;
      track.track.SetElements(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by reprojection error";
  return counter;
}

int TrackFilter::FilterTracksByAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double max_angle_error) {
  int counter = 0;
  double thres = std::cos(colmap::DegToRad(max_angle_error));
  double thres_uncalib = std::cos(colmap::DegToRad(max_angle_error * 2));
  for (auto& [track_id, track] : tracks) {
    std::vector<colmap::TrackElement> observation_new;
    for (const auto& observation : track.track.Elements()) {
      const Image& image = images.at(observation.image_id);
      const Eigen::Vector3d& feature_undist =
          image.features_undist.at(observation.point2D_idx);
      const Eigen::Vector3d pt_calc =
          (image.CamFromWorld() * track.xyz).normalized();
      const double thres_cam =
          (cameras.at(image.CameraId()).has_prior_focal_length) ? thres
                                                                : thres_uncalib;

      if (pt_calc.dot(feature_undist) > thres_cam) {
        observation_new.emplace_back(observation.image_id,
                                     observation.point2D_idx);
      }
    }
    if (observation_new.size() != track.track.Length()) {
      counter++;
      track.track.SetElements(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by angle error";
  return counter;
}

int TrackFilter::FilterTrackTriangulationAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<point3D_t, Point3D>& tracks,
    double min_angle) {
  int counter = 0;
  double thres = std::cos(colmap::DegToRad(min_angle));
  for (auto& [track_id, track] : tracks) {
    std::vector<colmap::TrackElement> observation_new;
    std::vector<Eigen::Vector3d> pts_calc;
    pts_calc.reserve(track.track.Length());
    for (auto& observation : track.track.Elements()) {
      const Image& image = images.at(observation.image_id);
      Eigen::Vector3d pt_calc =
          (track.xyz - image.ProjectionCenter()).normalized();
      pts_calc.emplace_back(pt_calc);
    }
    bool status = false;
    for (int i = 0; i < track.track.Length(); i++) {
      for (int j = i + 1; j < track.track.Length(); j++) {
        if (pts_calc[i].dot(pts_calc[j]) < thres) {
          status = true;
          break;
        }
      }
    }

    // If the triangulation angle is too small, just remove it
    if (!status) {
      counter++;
      track.track.SetElements({});
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace glomap
