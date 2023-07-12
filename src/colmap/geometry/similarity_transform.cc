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

#include "colmap/geometry/similarity_transform.h"

#include "colmap/base/reconstruction.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/projection.h"
#include "colmap/optim/loransac.h"

#include <fstream>

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

    SimilarityTransform3 tgtFromSrc;
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
        SimilarityTransform3(tgtFromSrc).Inverse().Matrix();

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

SimilarityTransform3::SimilarityTransform3()
    : matrix_(Eigen::Matrix3x4d::Identity()) {}

SimilarityTransform3::SimilarityTransform3(const Eigen::Matrix3x4d& matrix)
    : matrix_(matrix) {}

SimilarityTransform3::SimilarityTransform3(const double scale,
                                           const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec) {
  matrix_ = ComposeProjectionMatrix(qvec, tvec);
  matrix_.leftCols<3>() *= scale;
}

SimilarityTransform3 SimilarityTransform3::Inverse() const {
  const double scale = Scale();
  Eigen::Matrix3x4d inverse;
  inverse.leftCols<3>() = matrix_.leftCols<3>().transpose() / (scale * scale);
  inverse.col(3) = inverse.leftCols<3>() * -matrix_.col(3);
  return SimilarityTransform3(inverse);
}

const Eigen::Matrix3x4d& SimilarityTransform3::Matrix() const {
  return matrix_;
}

double SimilarityTransform3::Scale() const { return matrix_.col(0).norm(); }

Eigen::Vector4d SimilarityTransform3::Rotation() const {
  return RotationMatrixToQuaternion(matrix_.leftCols<3>() / Scale());
}

Eigen::Vector3d SimilarityTransform3::Translation() const {
  return matrix_.col(3);
}

bool SimilarityTransform3::Estimate(const std::vector<Eigen::Vector3d>& src,
                                    const std::vector<Eigen::Vector3d>& tgt) {
  const auto results =
      SimilarityTransformEstimator<3, true>().Estimate(src, tgt);
  if (results.empty()) {
    return false;
  }
  CHECK_EQ(results.size(), 1);
  matrix_ = results[0];
  return true;
}

void SimilarityTransform3::TransformPose(Eigen::Vector4d* qvec,
                                         Eigen::Vector3d* tvec) const {
  Eigen::Matrix4d inverse;
  inverse.topRows<3>() = Inverse().Matrix();
  inverse.row(3) = Eigen::Vector4d(0, 0, 0, 1);
  const Eigen::Matrix3x4d transformed =
      ComposeProjectionMatrix(*qvec, *tvec) * inverse;
  const double transformed_scale = transformed.col(0).norm();
  *qvec =
      RotationMatrixToQuaternion(transformed.leftCols<3>() / transformed_scale);
  *tvec = transformed.col(3) / transformed_scale;
}

void SimilarityTransform3::ToFile(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.good()) << path;
  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);
  file << matrix_ << std::endl;
}

SimilarityTransform3 SimilarityTransform3::FromFile(const std::string& path) {
  std::ifstream file(path);
  CHECK(file.good()) << path;

  Eigen::Matrix3x4d matrix;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file >> matrix(i, j);
    }
  }

  return SimilarityTransform3(matrix);
}

bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double min_inlier_observations,
    const double max_reproj_error,
    SimilarityTransform3* tgtFromSrc) {
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
    *tgtFromSrc = SimilarityTransform3(report.model);
  }

  return report.success;
}

bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const double max_proj_center_error,
    SimilarityTransform3* tgtFromSrc) {
  CHECK_GT(max_proj_center_error, 0);

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_proj_centers;
  for (const auto& image : tgt_reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      ref_image_names.push_back(image.second.Name());
      ref_proj_centers.push_back(image.second.ProjectionCenter());
    }
  }

  SimilarityTransform3 tform;
  Reconstruction aligned_src_reconstruction = src_reconstruction;
  RANSACOptions ransac_options;
  ransac_options.max_error = max_proj_center_error;
  if (!aligned_src_reconstruction.AlignRobust(ref_image_names,
                                              ref_proj_centers,
                                              /*min_common_images=*/3,
                                              ransac_options,
                                              &tform)) {
    return false;
  }

  *tgtFromSrc = tform;
  return true;
}

std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const SimilarityTransform3& tgtFromSrc) {
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

}  // namespace colmap
