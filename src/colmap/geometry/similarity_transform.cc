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

  void SetReconstructions(const Reconstruction* reconstruction1,
                          const Reconstruction* reconstruction2) {
    CHECK_NOTNULL(reconstruction1);
    CHECK_NOTNULL(reconstruction2);
    reconstruction1_ = reconstruction1;
    reconstruction2_ = reconstruction2;
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

    SimilarityTransform3 tform12;
    tform12.Estimate(proj_centers1, proj_centers2);

    return {tform12.Matrix().topRows<3>()};
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
    CHECK_NOTNULL(reconstruction1_);
    CHECK_NOTNULL(reconstruction2_);

    const Eigen::Matrix3x4d srcFromTgt =
        SimilarityTransform3(tgtFromSrc).Inverse().Matrix().topRows<3>();

    residuals->resize(src_images.size());

    for (size_t i = 0; i < src_images.size(); ++i) {
      const auto& image1 = *src_images[i];
      const auto& image2 = *tgt_images[i];

      CHECK_EQ(image1.ImageId(), image2.ImageId());

      const auto& camera1 = reconstruction1_->Camera(image1.CameraId());
      const auto& camera2 = reconstruction2_->Camera(image2.CameraId());

      const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
      const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();

      CHECK_EQ(image1.NumPoints2D(), image2.NumPoints2D());

      size_t num_inliers = 0;
      size_t num_common_points = 0;

      for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
           ++point2D_idx) {
        // Check if both images have a 3D point.

        const auto& point2D1 = image1.Point2D(point2D_idx);
        if (!point2D1.HasPoint3D()) {
          continue;
        }

        const auto& point2D2 = image2.Point2D(point2D_idx);
        if (!point2D2.HasPoint3D()) {
          continue;
        }

        num_common_points += 1;

        // Reproject 3D point in image 1 to image 2.
        const Eigen::Vector3d xyz12 =
            tgtFromSrc *
            reconstruction1_->Point3D(point2D1.Point3DId()).XYZ().homogeneous();
        if (CalculateSquaredReprojectionError(
                point2D2.XY(), xyz12, proj_matrix2, camera2) >
            max_squared_reproj_error_) {
          continue;
        }

        // Reproject 3D point in image 2 to image 1.
        const Eigen::Vector3d xyz21 =
            srcFromTgt *
            reconstruction2_->Point3D(point2D2.Point3DId()).XYZ().homogeneous();
        if (CalculateSquaredReprojectionError(
                point2D1.XY(), xyz21, proj_matrix1, camera1) >
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
  const Reconstruction* reconstruction1_ = nullptr;
  const Reconstruction* reconstruction2_ = nullptr;
};

}  // namespace

SimilarityTransform3::SimilarityTransform3()
    : SimilarityTransform3(
          1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0)) {}

SimilarityTransform3::SimilarityTransform3(const Eigen::Matrix3x4d& matrix) {
  transform_.matrix().topLeftCorner<3, 4>() = matrix;
}

SimilarityTransform3::SimilarityTransform3(
    const Eigen::Transform<double, 3, Eigen::Affine>& transform)
    : transform_(transform) {}

SimilarityTransform3::SimilarityTransform3(const double scale,
                                           const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec) {
  Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity(4, 4);
  matrix.topLeftCorner<3, 4>() = ComposeProjectionMatrix(qvec, tvec);
  matrix.block<3, 3>(0, 0) *= scale;
  transform_.matrix() = matrix;
}

void SimilarityTransform3::Write(const std::string& path) {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;
  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);
  file << transform_.matrix() << std::endl;
}

SimilarityTransform3 SimilarityTransform3::Inverse() const {
  return SimilarityTransform3(transform_.inverse());
}

void SimilarityTransform3::TransformPoint(Eigen::Vector3d* xyz) const {
  *xyz = transform_ * *xyz;
}

void SimilarityTransform3::TransformPose(Eigen::Vector4d* qvec,
                                         Eigen::Vector3d* tvec) const {
  // Projection matrix P1 projects 3D object points to image plane and thus to
  // 2D image points in the source coordinate system:
  //    x' = P1 * X1
  // 3D object points can be transformed to the destination system by applying
  // the similarity transformation S:
  //    X2 = S * X1
  // To obtain the projection matrix P2 that transforms the object point in the
  // destination system to the 2D image points, which do not change:
  //    x' = P2 * X2 = P2 * S * X1 = P1 * S^-1 * S * X1 = P1 * I * X1
  // and thus:
  //    P2' = P1 * S^-1
  // Finally, undo the inverse scaling of the rotation matrix:
  //    P2 = s * P2'

  Eigen::Matrix4d src_matrix = Eigen::MatrixXd::Identity(4, 4);
  src_matrix.topLeftCorner<3, 4>() = ComposeProjectionMatrix(*qvec, *tvec);
  Eigen::Matrix4d dst_matrix =
      src_matrix.matrix() * transform_.inverse().matrix();
  dst_matrix *= Scale();

  *qvec = RotationMatrixToQuaternion(dst_matrix.block<3, 3>(0, 0));
  *tvec = dst_matrix.block<3, 1>(0, 3);
}

Eigen::Matrix4d SimilarityTransform3::Matrix() const {
  return transform_.matrix();
}

double SimilarityTransform3::Scale() const {
  return Matrix().block<1, 3>(0, 0).norm();
}

Eigen::Vector4d SimilarityTransform3::Rotation() const {
  return RotationMatrixToQuaternion(Matrix().block<3, 3>(0, 0) / Scale());
}

Eigen::Vector3d SimilarityTransform3::Translation() const {
  return Matrix().block<3, 1>(0, 3);
}

SimilarityTransform3 SimilarityTransform3::FromFile(const std::string& path) {
  std::ifstream file(path);
  CHECK(file.is_open()) << path;

  Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity(4, 4);
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file >> matrix(i, j);
    }
  }
  SimilarityTransform3 tform;
  tform.transform_.matrix() = matrix;
  return tform;
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
