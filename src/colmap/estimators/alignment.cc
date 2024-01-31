// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/alignment.h"

#include "colmap/estimators/similarity_transform.h"
#include "colmap/geometry/pose.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/projection.h"
#include "colmap/util/logging.h"

#include <unordered_map>

namespace colmap {
namespace {

struct ReconstructionAlignmentEstimator {
  static const int kMinNumSamples = 3;

  typedef const Image* X_t;
  typedef const Image* Y_t;
  typedef Sim3d M_t;

  void SetMaxReprojError(const double max_reproj_error) {
    max_squared_reproj_error_ = max_reproj_error * max_reproj_error;
  }

  void SetReconstructions(const Reconstruction* src_reconstruction,
                          const Reconstruction* tgt_reconstruction) {
    THROW_CHECK_NOTNULL(src_reconstruction);
    THROW_CHECK_NOTNULL(tgt_reconstruction);
    src_reconstruction_ = src_reconstruction;
    tgt_reconstruction_ = tgt_reconstruction;
  }

  // Estimate 3D similarity transform from corresponding projection centers.
  void Estimate(const std::vector<X_t>& src_images,
                const std::vector<Y_t>& tgt_images,
                std::vector<M_t>* models) const {
    THROW_CHECK_GE(src_images.size(), 3);
    THROW_CHECK_GE(tgt_images.size(), 3);
    THROW_CHECK_EQ(src_images.size(), tgt_images.size());
    THROW_CHECK(models != nullptr);

    models->clear();

    std::vector<Eigen::Vector3d> proj_centers1(src_images.size());
    std::vector<Eigen::Vector3d> proj_centers2(tgt_images.size());
    for (size_t i = 0; i < src_images.size(); ++i) {
      THROW_CHECK_EQ(src_images[i]->ImageId(), tgt_images[i]->ImageId());
      proj_centers1[i] = src_images[i]->ProjectionCenter();
      proj_centers2[i] = tgt_images[i]->ProjectionCenter();
    }

    Sim3d tgt_from_src;
    if (!EstimateSim3d(proj_centers1, proj_centers2, tgt_from_src)) {
      return;
    }

    models->resize(1);
    (*models)[0] = tgt_from_src;
  }

  // For each image, determine the ratio of 3D points that correctly project
  // from one image to the other image and vice versa for the given
  // tgt_from_src. The residual is then defined as 1 minus this ratio, i.e., an
  // error threshold of 0.3 means that 70% of the points for that image must
  // reproject within the given maximum reprojection error threshold.
  void Residuals(const std::vector<X_t>& src_images,
                 const std::vector<Y_t>& tgt_images,
                 const M_t& tgt_from_src,
                 std::vector<double>* residuals) const {
    THROW_CHECK_EQ(src_images.size(), tgt_images.size());
    THROW_CHECK_NOTNULL(src_reconstruction_);
    THROW_CHECK_NOTNULL(tgt_reconstruction_);

    const Sim3d srcFromTgt = Inverse(tgt_from_src);

    residuals->resize(src_images.size());

    for (size_t i = 0; i < src_images.size(); ++i) {
      const auto& src_image = *src_images[i];
      const auto& tgt_image = *tgt_images[i];

      THROW_CHECK_EQ(src_image.ImageId(), tgt_image.ImageId());

      const auto& src_camera =
          src_reconstruction_->Camera(src_image.CameraId());
      const auto& tgt_camera =
          tgt_reconstruction_->Camera(tgt_image.CameraId());

      const Eigen::Matrix3x4d src_cam_from_world =
          src_image.CamFromWorld().ToMatrix();
      const Eigen::Matrix3x4d tgt_cam_from_world =
          tgt_image.CamFromWorld().ToMatrix();

      THROW_CHECK_EQ(src_image.NumPoints2D(), tgt_image.NumPoints2D());

      size_t num_inliers = 0;
      size_t num_common_points = 0;

      for (point2D_t point2D_idx = 0; point2D_idx < src_image.NumPoints2D();
           ++point2D_idx) {
        // Check if both images have a 3D point.

        const auto& src_point2D = src_image.Point2D(point2D_idx);
        if (!src_point2D.HasPoint3D()) {
          continue;
        }

        const auto& tgt_point2D = tgt_image.Point2D(point2D_idx);
        if (!tgt_point2D.HasPoint3D()) {
          continue;
        }

        num_common_points += 1;

        const Eigen::Vector3d src_point_in_tgt =
            tgt_from_src *
            src_reconstruction_->Point3D(src_point2D.point3D_id).xyz;
        if (CalculateSquaredReprojectionError(tgt_point2D.xy,
                                              src_point_in_tgt,
                                              tgt_cam_from_world,
                                              tgt_camera) >
            max_squared_reproj_error_) {
          continue;
        }

        const Eigen::Vector3d tgt_point_in_src =
            srcFromTgt *
            tgt_reconstruction_->Point3D(tgt_point2D.point3D_id).xyz;
        if (CalculateSquaredReprojectionError(src_point2D.xy,
                                              tgt_point_in_src,
                                              src_cam_from_world,
                                              src_camera) >
            max_squared_reproj_error_) {
          continue;
        }

        num_inliers += 1;
      }

      if (num_common_points == 0) {
        (*residuals)[i] = 1.0;
      } else {
        const double negative_inlier_ratio =
            1.0 - static_cast<double>(num_inliers) /
                      static_cast<double>(num_common_points);
        (*residuals)[i] = negative_inlier_ratio * negative_inlier_ratio;
      }
    }
  }

 private:
  double max_squared_reproj_error_ = 0.0;
  const Reconstruction* src_reconstruction_ = nullptr;
  const Reconstruction* tgt_reconstruction_ = nullptr;
};

}  // namespace

bool AlignReconstructionToLocations(
    const Reconstruction& src_reconstruction,
    const std::vector<std::string>& tgt_image_names,
    const std::vector<Eigen::Vector3d>& tgt_image_locations,
    const int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GE(min_common_images, 3);
  THROW_CHECK_EQ(tgt_image_names.size(), tgt_image_locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::unordered_set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < tgt_image_names.size(); ++i) {
    const class Image* src_image =
        src_reconstruction.FindImageWithName(tgt_image_names[i]);
    if (src_image == nullptr) {
      continue;
    }

    if (!src_reconstruction.IsImageRegistered(src_image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(src_image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(src_image->ImageId());
    src.push_back(src_image->ProjectionCenter());
    dst.push_back(tgt_image_locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  LORANSAC<SimilarityTransformEstimator<3, true>,
           SimilarityTransformEstimator<3, true>>
      ransac(ransac_options);

  const auto report = ransac.Estimate(src, dst);

  if (report.support.num_inliers < static_cast<size_t>(min_common_images)) {
    return false;
  }

  if (tgt_from_src != nullptr) {
    *tgt_from_src = Sim3d::FromMatrix(report.model);
  }

  return true;
}

bool AlignReconstructionsViaReprojections(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double min_inlier_observations,
    const double max_reproj_error,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GE(min_inlier_observations, 0.0);
  THROW_CHECK_LE(min_inlier_observations, 1.0);

  RANSACOptions ransac_options;
  ransac_options.max_error = 1.0 - min_inlier_observations;
  ransac_options.min_inlier_ratio = 0.2;

  LORANSAC<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator>
      ransac(ransac_options);
  ransac.estimator.SetMaxReprojError(max_reproj_error);
  ransac.estimator.SetReconstructions(&src_reconstruction, &tgt_reconstruction);
  ransac.local_estimator.SetMaxReprojError(max_reproj_error);
  ransac.local_estimator.SetReconstructions(&src_reconstruction,
                                            &tgt_reconstruction);

  const std::vector<std::pair<image_t, image_t>> common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);

  if (common_image_ids.size() < 3) {
    return false;
  }

  std::vector<const Image*> src_images(common_image_ids.size());
  std::vector<const Image*> tgt_images(common_image_ids.size());
  for (size_t i = 0; i < common_image_ids.size(); ++i) {
    src_images[i] = &src_reconstruction.Image(common_image_ids[i].first);
    tgt_images[i] = &tgt_reconstruction.Image(common_image_ids[i].second);
  }

  const auto report = ransac.Estimate(src_images, tgt_images);

  if (report.success) {
    *tgt_from_src = report.model;
  }

  return report.success;
}

bool AlignReconstructionsViaProjCenters(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double max_proj_center_error,
    Sim3d* tgt_from_src) {
  THROW_CHECK_GT(max_proj_center_error, 0);

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_proj_centers;
  for (const auto& image : tgt_reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      ref_image_names.push_back(image.second.Name());
      ref_proj_centers.push_back(image.second.ProjectionCenter());
    }
  }

  Sim3d tform;
  RANSACOptions ransac_options;
  ransac_options.max_error = max_proj_center_error;
  return AlignReconstructionToLocations(src_reconstruction,
                                        ref_image_names,
                                        ref_proj_centers,
                                        /*min_common_images=*/3,
                                        ransac_options,
                                        tgt_from_src);
}

std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgt_from_src) {
  const std::vector<std::pair<image_t, image_t>> common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);
  const int num_common_images = common_image_ids.size();
  std::vector<ImageAlignmentError> errors;
  errors.reserve(num_common_images);
  for (const auto& image_ids : common_image_ids) {
    const auto& src_image = src_reconstruction.Image(image_ids.first);
    const Rigid3d tgt_world_from_src_cam =
        Inverse(TransformCameraWorld(tgt_from_src, src_image.CamFromWorld()));
    const Rigid3d tgt_world_from_tgt_cam =
        Inverse(tgt_reconstruction.Image(image_ids.second).CamFromWorld());

    ImageAlignmentError error;
    error.image_name = src_image.Name();
    error.rotation_error_deg =
        RadToDeg(tgt_world_from_src_cam.rotation.angularDistance(
            tgt_world_from_tgt_cam.rotation));
    error.proj_center_error = (tgt_world_from_src_cam.translation -
                               tgt_world_from_tgt_cam.translation)
                                  .norm();
    errors.push_back(error);
  }
  return errors;
}

bool AlignReconstructionsViaPoints(const Reconstruction& src_reconstruction,
                                   const Reconstruction& tgt_reconstruction,
                                   const size_t min_common_observations,
                                   const double max_error,
                                   const double min_inlier_ratio,
                                   Sim3d* tgt_from_src) {
  THROW_CHECK_GT(min_common_observations, 0);
  THROW_CHECK_GT(max_error, 0.0);
  THROW_CHECK_GE(min_inlier_ratio, 0.0);
  THROW_CHECK_LE(min_inlier_ratio, 1.0);

  std::vector<Eigen::Vector3d> src_xyz;
  std::vector<Eigen::Vector3d> tgt_xyz;
  std::unordered_map<point3D_t, size_t> counts;
  // Associate 3D points using point2D_idx
  for (const auto& src_point3D : src_reconstruction.Points3D()) {
    counts.clear();
    // Count how often a 3D point in tgt is associated to this 3D point.
    for (const auto& track_el : src_point3D.second.track.Elements()) {
      if (!tgt_reconstruction.IsImageRegistered(track_el.image_id)) {
        continue;
      }
      const Point2D& tgt_point2D = tgt_reconstruction.Image(track_el.image_id)
                                       .Point2D(track_el.point2D_idx);
      if (tgt_point2D.HasPoint3D()) {
        if (counts.find(tgt_point2D.point3D_id) != counts.end()) {
          counts[tgt_point2D.point3D_id]++;
        } else {
          counts[tgt_point2D.point3D_id] = 0;
        }
      }
    }
    if (counts.empty()) {
      continue;
    }
    // The 3D point in tgt who is associated the most is selected
    auto best_p3D =
        std::max_element(counts.begin(),
                         counts.end(),
                         [](const std::pair<point3D_t, size_t>& p1,
                            const std::pair<point3D_t, size_t>& p2) {
                           return p1.second < p2.second;
                         });
    if (best_p3D->second >= min_common_observations) {
      src_xyz.push_back(src_point3D.second.xyz);
      tgt_xyz.push_back(tgt_reconstruction.Point3D(best_p3D->first).xyz);
    }
  }
  THROW_CHECK_EQ(src_xyz.size(), tgt_xyz.size());
  LOG(INFO) << "Found " << src_xyz.size() << " / "
            << src_reconstruction.NumPoints3D() << " valid correspondences.";

  RANSACOptions ransac_options;
  ransac_options.max_error = max_error;
  ransac_options.min_inlier_ratio = min_inlier_ratio;
  LORANSAC<SimilarityTransformEstimator<3, true>,
           SimilarityTransformEstimator<3, true>>
      ransac(ransac_options);
  const auto report = ransac.Estimate(src_xyz, tgt_xyz);
  if (report.success) {
    *tgt_from_src = Sim3d::FromMatrix(report.model);
  }
  return report.success;
}

bool MergeReconstructions(const double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction* tgt_reconstruction) {
  Sim3d tgt_from_src;
  if (!AlignReconstructionsViaReprojections(src_reconstruction,
                                            *tgt_reconstruction,
                                            /*min_inlier_observations=*/0.3,
                                            max_reproj_error,
                                            &tgt_from_src)) {
    return false;
  }

  // Find common and missing images in the two reconstructions.
  std::unordered_set<image_t> common_image_ids;
  common_image_ids.reserve(src_reconstruction.NumRegImages());
  std::unordered_set<image_t> missing_image_ids;
  missing_image_ids.reserve(src_reconstruction.NumRegImages());
  for (const auto& image_id : src_reconstruction.RegImageIds()) {
    if (tgt_reconstruction->ExistsImage(image_id)) {
      common_image_ids.insert(image_id);
    } else {
      missing_image_ids.insert(image_id);
    }
  }

  // Register the missing images in this src_reconstruction.
  for (const auto image_id : missing_image_ids) {
    auto src_image = src_reconstruction.Image(image_id);
    src_image.SetRegistered(false);
    src_image.CamFromWorld() =
        TransformCameraWorld(tgt_from_src, src_image.CamFromWorld());
    tgt_reconstruction->AddImage(src_image);
    tgt_reconstruction->RegisterImage(image_id);
    if (!tgt_reconstruction->ExistsCamera(src_image.CameraId())) {
      tgt_reconstruction->AddCamera(
          src_reconstruction.Camera(src_image.CameraId()));
    }
  }

  // Merge the two point clouds using the following two rules:
  //    - copy points to this src_reconstruction with non-conflicting tracks,
  //      i.e. points that do not have an already triangulated observation
  //      in this src_reconstruction.
  //    - merge tracks that are unambiguous, i.e. only merge points in the two
  //      reconstructions if they have a one-to-one mapping.
  // Note that in both cases no cheirality or reprojection test is performed.

  for (const auto& point3D : src_reconstruction.Points3D()) {
    Track new_track;
    Track old_track;
    std::unordered_set<point3D_t> old_point3D_ids;
    for (const auto& track_el : point3D.second.track.Elements()) {
      if (common_image_ids.count(track_el.image_id) > 0) {
        const auto& point2D = tgt_reconstruction->Image(track_el.image_id)
                                  .Point2D(track_el.point2D_idx);
        if (point2D.HasPoint3D()) {
          old_track.AddElement(track_el);
          old_point3D_ids.insert(point2D.point3D_id);
        } else {
          new_track.AddElement(track_el);
        }
      } else if (missing_image_ids.count(track_el.image_id) > 0) {
        tgt_reconstruction->Image(track_el.image_id)
            .ResetPoint3DForPoint2D(track_el.point2D_idx);
        new_track.AddElement(track_el);
      }
    }

    const bool create_new_point = new_track.Length() >= 2;
    const bool merge_new_and_old_point =
        (new_track.Length() + old_track.Length()) >= 2 &&
        old_point3D_ids.size() == 1;
    if (create_new_point || merge_new_and_old_point) {
      const Eigen::Vector3d xyz = tgt_from_src * point3D.second.xyz;
      const auto point3D_id =
          tgt_reconstruction->AddPoint3D(xyz, new_track, point3D.second.color);
      if (old_point3D_ids.size() == 1) {
        tgt_reconstruction->MergePoints3D(point3D_id, *old_point3D_ids.begin());
      }
    }
  }

  tgt_reconstruction->FilterAllPoints3D(max_reproj_error, /*min_tri_angle=*/0);

  return true;
}

}  // namespace colmap
