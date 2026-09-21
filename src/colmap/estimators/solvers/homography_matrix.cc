// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/homography_matrix.h"

#include "colmap/estimators/cost_functions/tiny_manifold.h"
#include "colmap/geometry/normalization.h"
#include "colmap/optim/tiny_solver.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <cmath>
#include <limits>
#include <optional>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {
namespace internal {

HomographyTransferCostFunction::HomographyTransferCostFunction(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2)
    : points1_(points1), points2_(points2) {}

int HomographyTransferCostFunction::NumResiduals() const {
  return 2 * static_cast<int>(points1_.size());
}

bool HomographyTransferCostFunction::operator()(const double* parameters,
                                                double* residuals,
                                                double* jacobian) const {
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> H(
      parameters);
  for (size_t i = 0; i < points1_.size(); ++i) {
    const Eigen::Vector3d transferred = H * points1_[i].homogeneous();
    const double inv_z = 1.0 / transferred.z();
    residuals[2 * i] = transferred.x() * inv_z - points2_[i].x();
    residuals[2 * i + 1] = transferred.y() * inv_z - points2_[i].y();
  }

  if (jacobian != nullptr) {
    // Analytic Jacobian, column-major 2Nx9. With q = points1[i].homogeneous(),
    // t = H q, u = tx/tz, v = ty/tz, and H flattened row-major, dt/dh is
    // block-diagonal in q', so du/dh = [inv_z q', 0, -u inv_z q'] and
    // dv/dh = [0, inv_z q', -v inv_z q'].
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 9>> J(
        jacobian, 2 * points1_.size(), 9);
    for (size_t i = 0; i < points1_.size(); ++i) {
      const Eigen::RowVector3d q = points1_[i].homogeneous().transpose();
      const Eigen::Vector3d transferred = H * q.transpose();
      const double inv_z = 1.0 / transferred.z();
      const double u = transferred.x() * inv_z;
      const double v = transferred.y() * inv_z;
      J.block<1, 3>(2 * i, 0) = inv_z * q;
      J.block<1, 3>(2 * i, 3).setZero();
      J.block<1, 3>(2 * i, 6) = -u * inv_z * q;
      J.block<1, 3>(2 * i + 1, 0).setZero();
      J.block<1, 3>(2 * i + 1, 3) = inv_z * q;
      J.block<1, 3>(2 * i + 1, 6) = -v * inv_z * q;
    }
  }
  return true;
}

}  // namespace internal

namespace {

bool HasCollinearTriplet(const std::vector<Eigen::Vector2d>& points) {
  const auto is_collinear = [&points](const size_t i,
                                      const size_t j,
                                      const size_t k) {
    constexpr double kMinNormalizedAreaSquared = 1e-24;
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

// Orientation pre-check for minimal samples, following PoseLib's
// homography_4pt (BSD-3-Clause, copyright Viktor Larsson): rejects 4-samples
// whose cyclic sidedness flips between views, which no orientation-preserving
// homography can map. This discards contaminated RANSAC samples before the
// solve and the full-data scoring. Only valid for det(H) > 0:
// orientation-reversing homographies, which require the camera to cross the
// observed plane, are systematically rejected.
bool PassCheiralityCheck(const std::vector<Eigen::Vector2d>& points1,
                         const std::vector<Eigen::Vector2d>& points2) {
  const Eigen::Vector3d x1[4] = {points1[0].homogeneous(),
                                 points1[1].homogeneous(),
                                 points1[2].homogeneous(),
                                 points1[3].homogeneous()};
  const Eigen::Vector3d x2[4] = {points2[0].homogeneous(),
                                 points2[1].homogeneous(),
                                 points2[2].homogeneous(),
                                 points2[3].homogeneous()};
  Eigen::Vector3d p = x1[0].cross(x1[1]);
  Eigen::Vector3d q = x2[0].cross(x2[1]);
  if (p.dot(x1[2]) * q.dot(x2[2]) < 0) return false;
  if (p.dot(x1[3]) * q.dot(x2[3]) < 0) return false;
  p = x1[2].cross(x1[3]);
  q = x2[2].cross(x2[3]);
  if (p.dot(x1[0]) * q.dot(x2[0]) < 0) return false;
  if (p.dot(x1[1]) * q.dot(x2[1]) < 0) return false;
  return true;
}

// Closed-form homography from 4 points via the similarity-kernel-similarity
// decomposition (Cai et al., "Fast and interpretable 2d homography
// decomposition: Similarity-kernel-similarity and affine-core-affine
// transformations", PAMI 2025), following PoseLib's homography_4pt
// implementation (BSD-3-Clause, copyright Viktor Larsson). A direct formula
// without any linear solves; agrees with DLT to machine precision. The output
// scale is arbitrary. Callers must reject collinear triplets first: the
// formula has no divisions, so degenerate inputs yield a zero matrix rather
// than NaN.
Eigen::Matrix3d SolveHomography4ptClosedForm(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  const Eigen::Vector3d xa[4] = {points1[0].homogeneous(),
                                 points1[1].homogeneous(),
                                 points1[2].homogeneous(),
                                 points1[3].homogeneous()};
  const Eigen::Vector3d xb[4] = {points2[0].homogeneous(),
                                 points2[1].homogeneous(),
                                 points2[2].homogeneous(),
                                 points2[3].homogeneous()};
  const double M1N1_x = xa[1].x() - xa[0].x();
  const double M1P1_x = xa[2].x() - xa[0].x();
  const double M1Q1_x = xa[3].x() - xa[0].x();
  const double M1N1_y = xa[1].y() - xa[0].y();
  const double M1P1_y = xa[2].y() - xa[0].y();
  const double M1Q1_y = xa[3].y() - xa[0].y();
  const double fA1 = M1N1_x * M1P1_y - M1N1_y * M1P1_x;
  const double Q3_x = M1P1_y * M1Q1_x - M1P1_x * M1Q1_y;
  const double Q3_y = M1N1_x * M1Q1_y - M1N1_y * M1Q1_x;
  const double M2N2_x = xb[1].x() - xb[0].x();
  const double M2P2_x = xb[2].x() - xb[0].x();
  const double M2Q2_x = xb[3].x() - xb[0].x();
  const double M2N2_y = xb[1].y() - xb[0].y();
  const double M2P2_y = xb[2].y() - xb[0].y();
  const double M2Q2_y = xb[3].y() - xb[0].y();
  const double fA2 = M2N2_x * M2P2_y - M2N2_y * M2P2_x;
  const double Q4_x = M2P2_y * M2Q2_x - M2P2_x * M2Q2_y;
  const double Q4_y = M2N2_x * M2Q2_y - M2N2_y * M2Q2_x;
  const double tt1 = fA1 - Q3_x - Q3_y;
  const double C11 = Q3_y * Q4_x * tt1;
  const double C22 = Q3_x * Q4_y * tt1;
  const double C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y);
  const double C31 = C11 - C33;
  const double C32 = C22 - C33;
  const double tt3 = xb[0].x() * C33;
  const double tt4 = xb[0].y() * C33;
  const double H1_11 = xb[1].x() * C11 - tt3;
  const double H1_12 = xb[2].x() * C22 - tt3;
  const double H1_21 = xb[1].y() * C11 - tt4;
  const double H1_22 = xb[2].y() * C22 - tt4;
  Eigen::Matrix<double, 9, 1> h;
  h[0] = H1_11 * M1P1_y - H1_12 * M1N1_y;
  h[1] = H1_12 * M1N1_x - H1_11 * M1P1_x;
  h[3] = H1_21 * M1P1_y - H1_22 * M1N1_y;
  h[4] = H1_22 * M1N1_x - H1_21 * M1P1_x;
  h[6] = C31 * M1P1_y - C32 * M1N1_y;
  h[7] = C32 * M1N1_x - C31 * M1P1_x;
  h[2] = tt3 * fA1 - h[0] * xa[0].x() - h[1] * xa[0].y();
  h[5] = tt4 * fA1 - h[3] * xa[0].x() - h[4] * xa[0].y();
  h[8] = C33 * fA1 - h[6] * xa[0].x() - h[7] * xa[0].y();
  return Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();
}

// Solve a 2Nx9 DLT system using LU for N == 4 and SVD otherwise.
// Returns false for rank-deficient, non-finite, or near-singular solutions.
bool SolveHomographyFromConstraintMatrix(
    const Eigen::Matrix<double, Eigen::Dynamic, 9>& A, Eigen::Matrix3d* H) {
  constexpr double kMinDeterminant = 1e-8;
  if (A.rows() == 8) {
    const Eigen::Matrix<double, 9, 1> h = A.block<8, 8>(0, 0)
                                              .partialPivLu()
                                              .solve(-A.block<8, 1>(0, 8))
                                              .homogeneous();
    if (h.hasNaN()) {
      return false;
    }
    *H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    if (svd.rank() < 8) {
      return false;
    }
    const Eigen::VectorXd nullspace = svd.matrixV().col(8);
    *H = Eigen::Map<const Eigen::Matrix3d>(nullspace.data()).transpose();
  }
  return std::abs(H->determinant()) >= kMinDeterminant;
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
  if (num_points == 4) {
    // Minimal samples take the closed-form path. A minimal homography
    // requires four points in general position (no three collinear) in both
    // images; see Hartley and Zisserman, Multiple View Geometry in Computer
    // Vision, 2nd ed., Sec. 4.1.3, pp. 91-92. Degeneracy is decided by the
    // scale-free collinearity test rather than a determinant threshold, which
    // would be scale-dependent on the arbitrary output scale.
    if (HasCollinearTriplet(points1) || HasCollinearTriplet(points2)) {
      return;
    }
    const Eigen::Matrix3d H = SolveHomography4ptClosedForm(points1, points2);
    if (!H.allFinite() || H.norm() == 0) {
      return;
    }
    models->resize(1);
    (*models)[0] = H.normalized();
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
  if (!SolveHomographyFromConstraintMatrix(A, &H)) {
    return;
  }

  models->resize(1);
  (*models)[0] = H;
}

bool HomographyMatrixEstimator::Refine(const std::vector<X_t>& points1,
                                       const std::vector<Y_t>& points2,
                                       M_t* H) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK_NOTNULL(H);

  // Normalize the points for better numerical stability, as in the
  // fundamental matrix refinement. Since both normalizations are similarities,
  // every residual is scaled by the same view-2 factor, so the minimizer is
  // unchanged while the normal equations are far better conditioned.
  std::vector<Eigen::Vector2d> normed_points1;
  std::vector<Eigen::Vector2d> normed_points2;
  Eigen::Matrix3d normed_from_orig1;
  Eigen::Matrix3d normed_from_orig2;
  CenterAndNormalizeImagePoints(points1, &normed_points1, &normed_from_orig1);
  CenterAndNormalizeImagePoints(points2, &normed_points2, &normed_from_orig2);

  // Map the initial model into the normalized frame, flatten row-major to
  // match the DLT storage order, and normalize to unit norm for the sphere
  // manifold.
  const Eigen::Matrix3d H_normed =
      normed_from_orig2 * (*H) * normed_from_orig1.inverse();
  Eigen::Matrix<double, 9, 1> h;
  for (int r = 0; r < 3; ++r) {
    h.segment<3>(3 * r) = H_normed.row(r).transpose();
  }
  h.normalize();

  // Plain least squares: the points are assumed to be the inlier set, so
  // robustness comes from the RANSAC inlier selection.
  const internal::HomographyTransferCostFunction f(normed_points1,
                                                   normed_points2);
  using Solver = TinySolver<decltype(f), SphereManifold<9>>;
  Solver solver;
  Solver::Options options;
  options.max_num_iterations = 25;
  solver.Solve(f, &h, options);

  if (!h.allFinite()) {
    return false;
  }

  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> H_ref(
      h.data());
  *H = normed_from_orig2.inverse() * H_ref * normed_from_orig1;
  return true;
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

void HomographyMatrixCheiralityEstimator::Estimate(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), kMinNumSamples);
  THROW_CHECK(models != nullptr);

  models->clear();
  if (points1.size() == kMinNumSamples &&
      !PassCheiralityCheck(points1, points2)) {
    return;
  }
  HomographyMatrixEstimator::Estimate(points1, points2, models);
}

bool HomographyMatrixCheiralityEstimator::Refine(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2, M_t* H) {
  return HomographyMatrixEstimator::Refine(points1, points2, H);
}

void HomographyMatrixCheiralityEstimator::Residuals(
    const std::vector<X_t>& points1,
    const std::vector<Y_t>& points2,
    const M_t& H,
    std::vector<double>* residuals) {
  HomographyMatrixEstimator::Residuals(points1, points2, H, residuals);
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
  if (!SolveHomographyFromConstraintMatrix(A, &H)) {
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
