// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#include "base/similarity_transform.h"

#include "base/pose.h"
#include "base/projection.h"
#include "base/reconstruction.h"
#include "estimators/similarity_transform.h"
#include "optim/loransac.h"

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
  std::vector<M_t> Estimate(const std::vector<X_t>& images1,
                            const std::vector<Y_t>& images2) const {
    CHECK_GE(images1.size(), 3);
    CHECK_GE(images2.size(), 3);
    CHECK_EQ(images1.size(), images2.size());

    std::vector<Eigen::Vector3d> proj_centers1(images1.size());
    std::vector<Eigen::Vector3d> proj_centers2(images2.size());
    for (size_t i = 0; i < images1.size(); ++i) {
      CHECK_EQ(images1[i]->ImageId(), images2[i]->ImageId());
      proj_centers1[i] = images1[i]->ProjectionCenter();
      proj_centers2[i] = images2[i]->ProjectionCenter();
    }

    SimilarityTransform3 tform12;
    tform12.Estimate(proj_centers1, proj_centers2);

    return {tform12.Matrix().topRows<3>()};
  }

  // For each image, determine the ratio of 3D points that correctly project
  // from one image to the other image and vice versa for the given alignment.
  // The residual is then defined as 1 minus this ratio, i.e., an error
  // threshold of 0.3 means that 70% of the points for that image must reproject
  // within the given maximum reprojection error threshold.
  void Residuals(const std::vector<X_t>& images1,
                 const std::vector<Y_t>& images2, const M_t& alignment12,
                 std::vector<double>* residuals) const {
    CHECK_EQ(images1.size(), images2.size());
    CHECK_NOTNULL(reconstruction1_);
    CHECK_NOTNULL(reconstruction2_);

    const Eigen::Matrix3x4d alignment21 =
        SimilarityTransform3(alignment12).Inverse().Matrix().topRows<3>();

    residuals->resize(images1.size());

    for (size_t i = 0; i < images1.size(); ++i) {
      const auto& image1 = *images1[i];
      const auto& image2 = *images2[i];

      CHECK_EQ(image1.ImageId(), image2.ImageId());
      CHECK_EQ(image1.CameraId(), image2.CameraId());

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
            alignment12 *
            reconstruction1_->Point3D(point2D1.Point3DId()).XYZ().homogeneous();
        if (CalculateSquaredReprojectionError(point2D2.XY(), xyz12,
                                              proj_matrix2, camera2) >
            max_squared_reproj_error_) {
          continue;
        }

        // Reproject 3D point in image 2 to image 1.
        const Eigen::Vector3d xyz21 =
            alignment21 *
            reconstruction2_->Point3D(point2D2.Point3DId()).XYZ().homogeneous();
        if (CalculateSquaredReprojectionError(point2D1.XY(), xyz21,
                                              proj_matrix1, camera1) >
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
    : SimilarityTransform3(1, ComposeIdentityQuaternion(),
                           Eigen::Vector3d(0, 0, 0)) {}

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
    const Reconstruction& ref_reconstruction,
    const double min_inlier_observations, const double max_reproj_error,
    Eigen::Matrix3x4d* alignment) {
  CHECK_GE(min_inlier_observations, 0.0);
  CHECK_LE(min_inlier_observations, 1.0);

  RANSACOptions ransac_options;
  ransac_options.max_error = 1.0 - min_inlier_observations;
  ransac_options.min_inlier_ratio = 0.2;

  LORANSAC<ReconstructionAlignmentEstimator, ReconstructionAlignmentEstimator>
      ransac(ransac_options);
  ransac.estimator.SetMaxReprojError(max_reproj_error);
  ransac.estimator.SetReconstructions(&src_reconstruction, &ref_reconstruction);
  ransac.local_estimator.SetMaxReprojError(max_reproj_error);
  ransac.local_estimator.SetReconstructions(&src_reconstruction,
                                            &ref_reconstruction);

  const auto& common_image_ids =
      src_reconstruction.FindCommonRegImageIds(ref_reconstruction);

  if (common_image_ids.size() < 3) {
    return false;
  }

  std::vector<const Image*> src_images(common_image_ids.size());
  std::vector<const Image*> ref_images(common_image_ids.size());
  for (size_t i = 0; i < common_image_ids.size(); ++i) {
    src_images[i] = &src_reconstruction.Image(common_image_ids[i]);
    ref_images[i] = &ref_reconstruction.Image(common_image_ids[i]);
  }

  const auto report = ransac.Estimate(src_images, ref_images);

  if (report.success) {
    *alignment = report.model;
  }

  return report.success;
}

}  // namespace colmap
