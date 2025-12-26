#include "glomap/processors/track_filter.h"

namespace glomap {

int TrackFilter::FilterObservationsWithLargeReprojectionError(
    colmap::Reconstruction& reconstruction,
    double max_reprojection_error,
    bool in_normalized_image) {
  int counter = 0;
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    std::vector<colmap::TrackElement> observation_new;
    for (const auto& observation : track.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      Eigen::Vector3d pt_calc = image.CamFromWorld() * track.xyz;
      constexpr double kEps = 1e-12;
      if (pt_calc(2) < kEps) continue;

      double reprojection_error = max_reprojection_error;
      if (in_normalized_image) {
        const std::optional<Eigen::Vector2d> cam_point =
            image.CameraPtr()->CamFromImg(
                image.Point2D(observation.point2D_idx).xy);
        Eigen::Vector2d pt_reproj = pt_calc.head(2) / pt_calc(2);
        if (cam_point.has_value()) {
          reprojection_error = (pt_reproj - *cam_point).norm();
        }
      } else {
        const std::optional<Eigen::Vector2d> img_point =
            image.CameraPtr()->ImgFromCam(pt_calc);
        if (img_point.has_value()) {
          reprojection_error =
              (*img_point - image.Point2D(observation.point2D_idx).xy).norm();
        }
      }

      // If the reprojection error is smaller than the threshold, then keep it
      if (reprojection_error < max_reprojection_error) {
        observation_new.emplace_back(observation.image_id,
                                     observation.point2D_idx);
      }
    }
    if (observation_new.size() != track.track.Length()) {
      counter++;
      reconstruction.Point3D(track_id).track.SetElements(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << reconstruction.NumPoints3D()
            << " tracks by reprojection error";
  return counter;
}

int TrackFilter::FilterObservationsWithLargeAngularError(
    colmap::Reconstruction& reconstruction, double max_angle_error) {
  int counter = 0;
  double thres = std::cos(colmap::DegToRad(max_angle_error));
  double thres_uncalib = std::cos(colmap::DegToRad(max_angle_error * 2));
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    std::vector<colmap::TrackElement> observation_new;
    for (const auto& observation : track.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      const std::optional<Eigen::Vector2d> cam_point =
          image.CameraPtr()->CamFromImg(
              image.Point2D(observation.point2D_idx).xy);
      const Eigen::Vector3d pt_calc =
          (image.CamFromWorld() * track.xyz).normalized();
      const double thres_cam =
          (image.CameraPtr()->has_prior_focal_length) ? thres : thres_uncalib;

      if (cam_point.has_value() &&
          pt_calc.dot(cam_point->homogeneous().normalized()) > thres_cam) {
        observation_new.emplace_back(observation.image_id,
                                     observation.point2D_idx);
      }
    }
    if (observation_new.size() != track.track.Length()) {
      counter++;
      reconstruction.Point3D(track_id).track.SetElements(observation_new);
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << reconstruction.NumPoints3D()
            << " tracks by angular reprojection error";
  return counter;
}

int TrackFilter::FilterTracksWithSmallTriangulationAngle(
    colmap::Reconstruction& reconstruction, double min_angle) {
  int counter = 0;
  double thres = std::cos(colmap::DegToRad(min_angle));
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    std::vector<Eigen::Vector3d> pts_calc;
    pts_calc.reserve(track.track.Length());
    for (const auto& observation : track.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      Eigen::Vector3d pt_calc =
          (track.xyz - image.ProjectionCenter()).normalized();
      pts_calc.emplace_back(pt_calc);
    }
    bool status = false;
    for (size_t i = 0; i < track.track.Length(); i++) {
      for (size_t j = i + 1; j < track.track.Length(); j++) {
        if (pts_calc[i].dot(pts_calc[j]) < thres) {
          status = true;
          break;
        }
      }
    }

    // If the triangulation angle is too small, just remove it
    if (!status) {
      counter++;
      reconstruction.Point3D(track_id).track.SetElements({});
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << reconstruction.NumPoints3D()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace glomap
