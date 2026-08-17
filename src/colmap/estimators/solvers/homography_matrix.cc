// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/solvers/homography_matrix.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cmath>
#include <limits>
#include <optional>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {
namespace {

bool HasCollinearTriplet(const std::vector<Eigen::Vector2d>& points) {
  constexpr double kMinNormalizedAreaSquared = 1e-24;
  const auto is_collinear = [&points](const size_t i,
                                      const size_t j,
                                      const size_t k) {
    const Eigen::Vector2d delta1 = points[j] - points[i];
    const Eigen::Vector2d delta2 = points[k] - points[i];
    const double scale_squared = delta1.squaredNorm() * delta2.squaredNorm();
    const double area = delta1.x() * delta2.y() - delta1.y() * delta2.x();
    return scale_squared == 0.0 ||
           area * area <= kMinNormalizedAreaSquared * scale_squared;
  };
  return is_collinear(0, 1, 2) || is_collinear(0, 1, 3) ||
         is_collinear(0, 2, 3) || is_collinear(1, 2, 3);
}

}  // namespace

void HomographyMatrixEstimator::Estimate(const std::vector<X_t>& points1,
                                         const std::vector<Y_t>& points2,
                                         std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), 4);
  THROW_CHECK(models != nullptr);

  models->clear();

  const size_t num_points = points1.size();
  // A minimal homography requires four points in general position (no three
  // collinear) in both images. See Hartley and Zisserman, Multiple View
  // Geometry in Computer Vision, 2nd ed., Sec. 4.1.3, pp. 91-92.
  if (num_points == 4 &&
      (HasCollinearTriplet(points1) || HasCollinearTriplet(points2))) {
    return;
  }

  // Setup constraint matrix.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * num_points, 9);
  for (size_t i = 0; i < num_points; ++i) {
    A.block<1, 3>(2 * i, 0) = points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i, 3).setZero();
    A.block<1, 3>(2 * i, 6) =
        -points2[i].x() * points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i + 1, 0).setZero();
    A.block<1, 3>(2 * i + 1, 3) = points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i + 1, 6) =
        -points2[i].y() * points1[i].transpose().homogeneous();
  }

  Eigen::Matrix3d H;
  if (num_points == 4) {
    const Eigen::Matrix<double, 9, 1> h = A.block<8, 8>(0, 0)
                                              .partialPivLu()
                                              .solve(-A.block<8, 1>(0, 8))
                                              .homogeneous();
    if (h.hasNaN()) {
      return;
    }
    H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();
  } else {
    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    if (svd.rank() < 8) {
      return;
    }
    const Eigen::VectorXd nullspace = svd.matrixV().col(8);
    H = Eigen::Map<const Eigen::Matrix3d>(nullspace.data()).transpose();
  }

  if (std::abs(H.determinant()) < 1e-8) {
    return;
  }

  models->resize(1);
  (*models)[0] = H;
}

void HomographyMatrixEstimator::Residuals(const std::vector<X_t>& points1,
                                          const std::vector<Y_t>& points2,
                                          const M_t& H,
                                          std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  // Note that this code might not be as nice as Eigen expressions,
  // but it is significantly faster in various tests.

  const double H_00 = H(0, 0);
  const double H_01 = H(0, 1);
  const double H_02 = H(0, 2);
  const double H_10 = H(1, 0);
  const double H_11 = H(1, 1);
  const double H_12 = H(1, 2);
  const double H_20 = H(2, 0);
  const double H_21 = H(2, 1);
  const double H_22 = H(2, 2);

  for (size_t i = 0; i < points1.size(); ++i) {
    const double s_0 = points1[i](0);
    const double s_1 = points1[i](1);
    const double d_0 = points2[i](0);
    const double d_1 = points2[i](1);

    const double pd_0 = H_00 * s_0 + H_01 * s_1 + H_02;
    const double pd_1 = H_10 * s_0 + H_11 * s_1 + H_12;
    const double pd_2 = H_20 * s_0 + H_21 * s_1 + H_22;

    const double inv_pd_2 = 1.0 / pd_2;
    const double dd_0 = d_0 - pd_0 * inv_pd_2;
    const double dd_1 = d_1 - pd_1 * inv_pd_2;

    (*residuals)[i] = dd_0 * dd_0 + dd_1 * dd_1;
  }
}

void HomographyMatrixRayEstimator::Estimate(const std::vector<X_t>& cam_rays1,
                                            const std::vector<Y_t>& cam_rays2,
                                            std::vector<M_t>* models) const {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), 4);
  THROW_CHECK(models != nullptr);

  models->clear();

  const size_t num_rays = cam_rays1.size();

  // Setup constraint matrix from x2 x (H x1) = 0. Of the three equations, the
  // rows of [x2]_x, only two are independent, and the weakest is always the one
  // omitting the largest component of x2. For a perspective camera z dominates
  // and this reduces to the pixel estimator's rows.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * num_rays, 9);
  for (size_t i = 0; i < num_rays; ++i) {
    const Eigen::Vector3d& ray1 = cam_rays1[i];
    const Eigen::Vector3d& ray2 = cam_rays2[i].ray;

    Eigen::Matrix<double, 3, 9> equations = Eigen::Matrix<double, 3, 9>::Zero();
    equations.block<1, 3>(0, 3) = -ray2.z() * ray1.transpose();
    equations.block<1, 3>(0, 6) = ray2.y() * ray1.transpose();
    equations.block<1, 3>(1, 0) = ray2.z() * ray1.transpose();
    equations.block<1, 3>(1, 6) = -ray2.x() * ray1.transpose();
    equations.block<1, 3>(2, 0) = -ray2.y() * ray1.transpose();
    equations.block<1, 3>(2, 3) = ray2.x() * ray1.transpose();

    // Equation j omits component j of x2, so the one to drop is the argmax.
    int dropped_equation_idx = 0;
    ray2.cwiseAbs().maxCoeff(&dropped_equation_idx);
    int num_kept = 0;
    for (int j = 0; j < 3; ++j) {
      if (j != dropped_equation_idx) {
        A.row(2 * i + num_kept++) = equations.row(j);
      }
    }
  }

  Eigen::Matrix3d H;
  if (num_rays == 4) {
    const Eigen::Matrix<double, 9, 1> h = A.block<8, 8>(0, 0)
                                              .partialPivLu()
                                              .solve(-A.block<8, 1>(0, 8))
                                              .homogeneous();
    if (h.hasNaN()) {
      return;
    }
    H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();
  } else {
    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    if (svd.rank() < 8) {
      return;
    }
    const Eigen::VectorXd nullspace = svd.matrixV().col(8);
    H = Eigen::Map<const Eigen::Matrix3d>(nullspace.data()).transpose();
  }

  if (std::abs(H.determinant()) < 1e-8) {
    return;
  }

  // H is defined up to scale, but the residual projects H x1 back into an image
  // that does not contain both a direction and its opposite, so the sign
  // matters. It is global, not per correspondence: a visible plane point has
  // positive depth, so x2 ~ lambda H x1 holds with lambda > 0 throughout and
  // the only freedom is that the solver may return -H. Resolving it per point
  // instead would score each against its nearer antipode, letting a 180 degree
  // error pass as a perfect inlier.
  int sign_votes = 0;
  for (size_t i = 0; i < num_rays; ++i) {
    sign_votes += (H * cam_rays1[i]).dot(cam_rays2[i].ray) > 0 ? 1 : -1;
  }
  if (sign_votes < 0) {
    H = -H;
  }

  models->resize(1);
  (*models)[0] = H;
}

void HomographyMatrixRayEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& H,
    std::vector<double>* residuals) const {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_NOTNULL(camera2_);

  residuals->resize(cam_rays1.size());

  // Azimuthal models wrap at the +-pi seam, where a raw pixel difference jumps
  // by about the image width. Wrap it into [-width/2, width/2), as
  // WrapEquirectangularHorizontalSeam does for the reprojection error, which
  // spells the same rounding as a floor since it must stay autodiff-safe.
  const bool is_periodic = camera2_->IsSpherical();
  const double width = static_cast<double>(camera2_->width);

  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    const std::optional<Eigen::Vector2d> img_point =
        camera2_->ImgFromCam(H * cam_rays1[i]);
    if (!img_point.has_value()) {
      // Transferred out of the camera's field, so there is nothing to score.
      (*residuals)[i] = std::numeric_limits<double>::max();
      continue;
    }
    Eigen::Vector2d error = *img_point - cam_rays2[i].img_point;
    if (is_periodic) {
      error.x() -= width * std::round(error.x() / width);
    }
    (*residuals)[i] = error.squaredNorm();
  }
}

}  // namespace colmap
