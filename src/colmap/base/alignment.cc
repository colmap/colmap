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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/base/alignment.h"

#include "colmap/estimators/similarity_transform.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/projection.h"
#include "colmap/optim/loransac.h"

namespace colmap {
namespace {

struct ReconstructionAlignmentEstimator {
  static const int kMinNumSamples = 3;

  typedef const Image* X_t;
  typedef const Image* Y_t;
  typedef Eigen::Matrix3x4d M_t;

  void SetMaxReprojError(const double max_reproj_error) {
    max_squared_reproj_error_ = max_reproj_error * max_reproj_error;
  }

  void SetReconstructions(const Reconstruction* src_reconstruction,
                          const Reconstruction* tgt_reconstruction) {
    CHECK_NOTNULL(src_reconstruction);
    CHECK_NOTNULL(tgt_reconstruction);
    src_reconstruction_ = src_reconstruction;
    tgt_reconstruction_ = tgt_reconstruction;
  }

  // Estimate 3D similarity transform from corresponding projection centers.
  std::vector<M_t> Estimate(const std::vector<X_t>& src_images,
                            const std::vector<Y_t>& tgt_images) const {
    CHECK_GE(src_images.size(), 3);
    CHECK_GE(tgt_images.size(), 3);
    CHECK_EQ(src_images.size(), tgt_images.size());

    std::vector<Eigen::Vector3d> proj_centers1(src_images.size());
    std::vector<Eigen::Vector3d> proj_centers2(tgt_images.size());
    for (size_t i = 0; i < src_images.size(); ++i) {
      CHECK_EQ(src_images[i]->ImageId(), tgt_images[i]->ImageId());
      proj_centers1[i] = src_images[i]->ProjectionCenter();
      proj_centers2[i] = tgt_images[i]->ProjectionCenter();
    }

    Sim3d tgtFromSrc;
    if (tgtFromSrc.Estimate(proj_centers1, proj_centers2)) {
      return {tgtFromSrc.Matrix()};
    }

    return {};
  }

  // For each image, determine the ratio of 3D points that correctly project
  // from one image to the other image and vice versa for the given tgtFromSrc.
  // The residual is then defined as 1 minus this ratio, i.e., an error
  // threshold of 0.3 means that 70% of the points for that image must reproject
  // within the given maximum reprojection error threshold.
  void Residuals(const std::vector<X_t>& src_images,
                 const std::vector<Y_t>& tgt_images,
                 const M_t& tgtFromSrc,
                 std::vector<double>* residuals) const {
    CHECK_EQ(src_images.size(), tgt_images.size());
    CHECK_NOTNULL(src_reconstruction_);
    CHECK_NOTNULL(tgt_reconstruction_);

    const Eigen::Matrix3x4d srcFromTgt =
        Sim3d(tgtFromSrc).Inverse().Matrix();

    residuals->resize(src_images.size());

    for (size_t i = 0; i < src_images.size(); ++i) {
      const auto& src_image = *src_images[i];
      const auto& tgt_image = *tgt_images[i];

      CHECK_EQ(src_image.ImageId(), tgt_image.ImageId());

      const auto& src_camera =
          src_reconstruction_->Camera(src_image.CameraId());
      const auto& tgt_camera =
          tgt_reconstruction_->Camera(tgt_image.CameraId());

      const Eigen::Matrix3x4d tgt_proj_matrix = src_image.ProjectionMatrix();
      const Eigen::Matrix3x4d src_proj_matrix = tgt_image.ProjectionMatrix();

      CHECK_EQ(src_image.NumPoints2D(), tgt_image.NumPoints2D());

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
            tgtFromSrc * src_reconstruction_->Point3D(src_point2D.Point3DId())
                             .XYZ()
                             .homogeneous();
        if (CalculateSquaredReprojectionError(tgt_point2D.XY(),
                                              src_point_in_tgt,
                                              src_proj_matrix,
                                              tgt_camera) >
            max_squared_reproj_error_) {
          continue;
        }

        const Eigen::Vector3d tgt_point_in_src =
            srcFromTgt * tgt_reconstruction_->Point3D(tgt_point2D.Point3DId())
                             .XYZ()
                             .homogeneous();
        if (CalculateSquaredReprojectionError(src_point2D.XY(),
                                              tgt_point_in_src,
                                              tgt_proj_matrix,
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
    const Reconstruction& reconstruction,
    const std::vector<std::string>& image_names,
    const std::vector<Eigen::Vector3d>& locations,
    const int min_common_images,
    const RANSACOptions& ransac_options,
    Sim3d* tform) {
  CHECK_GE(min_common_images, 3);
  CHECK_EQ(image_names.size(), locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::unordered_set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < image_names.size(); ++i) {
    const class Image* image = reconstruction.FindImageWithName(image_names[i]);
    if (image == nullptr) {
      continue;
    }

    if (!reconstruction.IsImageRegistered(image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(image->ImageId());
    src.push_back(image->ProjectionCenter());
    dst.push_back(locations[i]);
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

  if (tform != nullptr) {
    *tform = Sim3d(report.model);
  }

  return true;
}

bool AlignReconstructions(const Reconstruction& src_reconstruction,
                          const Reconstruction& tgt_reconstruction,
                          const double min_inlier_observations,
                          const double max_reproj_error,
                          Sim3d* tgtFromSrc) {
  CHECK_GE(min_inlier_observations, 0.0);
  CHECK_LE(min_inlier_observations, 1.0);

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

  const auto& common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);

  if (common_image_ids.size() < 3) {
    return false;
  }

  std::vector<const Image*> src_images(common_image_ids.size());
  std::vector<const Image*> tgt_images(common_image_ids.size());
  for (size_t i = 0; i < common_image_ids.size(); ++i) {
    src_images[i] = &src_reconstruction.Image(common_image_ids[i]);
    tgt_images[i] = &tgt_reconstruction.Image(common_image_ids[i]);
  }

  const auto report = ransac.Estimate(src_images, tgt_images);

  if (report.success) {
    *tgtFromSrc = Sim3d(report.model);
  }

  return report.success;
}

bool AlignReconstructions(const Reconstruction& src_reconstruction,
                          const Reconstruction& tgt_reconstruction,
                          const double max_proj_center_error,
                          Sim3d* tgtFromSrc) {
  CHECK_GT(max_proj_center_error, 0);

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
                                        tgtFromSrc);
}

std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgtFromSrc) {
  const std::vector<image_t> common_image_ids =
      src_reconstruction.FindCommonRegImageIds(tgt_reconstruction);
  const int num_common_images = common_image_ids.size();
  std::vector<ImageAlignmentError> errors;
  errors.reserve(num_common_images);
  for (const image_t image_id : common_image_ids) {
    const Image& src_image = src_reconstruction.Image(image_id);
    const Image& tgt_image = tgt_reconstruction.Image(image_id);

    Eigen::Vector4d src_qvec = src_image.Qvec();
    Eigen::Vector3d src_tvec = src_image.Tvec();
    tgtFromSrc.TransformPose(&src_qvec, &src_tvec);

    const Eigen::Vector4d normalized_qvec1 = NormalizeQuaternion(src_qvec);
    const Eigen::Quaterniond src_quat(normalized_qvec1(0),
                                      normalized_qvec1(1),
                                      normalized_qvec1(2),
                                      normalized_qvec1(3));
    const Eigen::Vector4d normalized_qvec2 =
        NormalizeQuaternion(tgt_image.Qvec());
    const Eigen::Quaterniond tgt_quat(normalized_qvec2(0),
                                      normalized_qvec2(1),
                                      normalized_qvec2(2),
                                      normalized_qvec2(3));

    ImageAlignmentError error;
    error.image_id = image_id;
    error.rotation_error_deg = RadToDeg(src_quat.angularDistance(tgt_quat));
    error.proj_center_error = (ProjectionCenterFromPose(src_qvec, src_tvec) -
                               tgt_image.ProjectionCenter())
                                  .norm();
    errors.push_back(error);
  }
  return errors;
}

bool MergeReconstructions(const double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction* tgt_reconstruction) {
  Sim3d tgtFromSrc;
  if (!AlignReconstructions(src_reconstruction,
                            *tgt_reconstruction,
                            /*min_inlier_observations=*/0.3,
                            max_reproj_error,
                            &tgtFromSrc)) {
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
    tgt_reconstruction->AddImage(src_image);
    tgt_reconstruction->RegisterImage(image_id);
    if (!tgt_reconstruction->ExistsCamera(src_image.CameraId())) {
      tgt_reconstruction->AddCamera(
          src_reconstruction.Camera(src_image.CameraId()));
    }
    auto& tgt_image = tgt_reconstruction->Image(image_id);
    tgtFromSrc.TransformPose(&tgt_image.Qvec(), &tgt_image.Tvec());
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
    for (const auto& track_el : point3D.second.Track().Elements()) {
      if (common_image_ids.count(track_el.image_id) > 0) {
        const auto& point2D = tgt_reconstruction->Image(track_el.image_id)
                                  .Point2D(track_el.point2D_idx);
        if (point2D.HasPoint3D()) {
          old_track.AddElement(track_el);
          old_point3D_ids.insert(point2D.Point3DId());
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
      const Eigen::Vector3d xyz = tgtFromSrc * point3D.second.XYZ();
      const auto point3D_id = tgt_reconstruction->AddPoint3D(
          xyz, new_track, point3D.second.Color());
      if (old_point3D_ids.size() == 1) {
        tgt_reconstruction->MergePoints3D(point3D_id, *old_point3D_ids.begin());
      }
    }
  }

  tgt_reconstruction->FilterAllPoints3D(max_reproj_error, /*min_tri_angle=*/0);

  return true;
}

}  // namespace colmap
