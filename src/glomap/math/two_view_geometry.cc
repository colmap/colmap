#include "glomap/math/two_view_geometry.h"

namespace glomap {
// Code from PoseLib by Viktor Larsson
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_depth,
                     double max_depth) {
  // This code assumes that x1 and x2 are unit vectors
  const Eigen::Vector3d Rx1 = pose.rotation * x1;

  // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
  // [lambda1; lambda2] = [1 s-a; -a 1] * [b1; b2] / (1 - a*a)
  const double a = -Rx1.dot(x2);
  const double b1 = -Rx1.dot(pose.translation);
  const double b2 = x2.dot(pose.translation);

  // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;

  min_depth = min_depth * (1 - a * a);
  max_depth = max_depth * (1 - a * a);

  bool status = lambda1 > min_depth && lambda2 > min_depth;
  status = status && (lambda1 < max_depth) && (lambda2 < max_depth);
  return status;
}

// This code is from GC-RANSAC by Daniel Barath
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2) {
  double signum1 = F(0, 0) * pt2[0] + F(1, 0) * pt2[1] + F(2, 0);
  double signum2 = epipole(1) - epipole(2) * pt1[1];
  return signum1 * signum2;
}

void EssentialFromMotion(const Rigid3d& pose, Eigen::Matrix3d* E) {
  *E << 0.0, -pose.translation(2), pose.translation(1), pose.translation(2),
      0.0, -pose.translation(0), -pose.translation(1), pose.translation(0), 0.0;
  *E = (*E) * pose.rotation.toRotationMatrix();
}

// Get the essential matrix from relative pose
void FundamentalFromMotionAndCameras(const Camera& camera1,
                                     const Camera& camera2,
                                     const Rigid3d& pose,
                                     Eigen::Matrix3d* F) {
  Eigen::Matrix3d E;
  EssentialFromMotion(pose, &E);
  *F = camera2.GetK().transpose().inverse() * E * camera1.GetK().inverse();
}

double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector2d& x1,
                    const Eigen::Vector2d& x2) {
  Eigen::Vector3d Ex1 = E * x1.homogeneous();
  Eigen::Vector3d Etx2 = E.transpose() * x2.homogeneous();

  double C = Ex1.dot(x2.homogeneous());
  double Cx = Ex1.head(2).squaredNorm();
  double Cy = Etx2.head(2).squaredNorm();
  double r2 = C * C / (Cx + Cy);

  return r2;
}

double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector3d& x1,
                    const Eigen::Vector3d& x2) {
  Eigen::Vector3d Ex1 = E * x1 / (EPS + x1[2]);
  Eigen::Vector3d Etx2 = E.transpose() * x2 / (EPS + x2[2]);

  double C = Ex1.dot(x2);
  double Cx = Ex1.head(2).squaredNorm();
  double Cy = Etx2.head(2).squaredNorm();
  double r2 = C * C / (Cx + Cy);

  return r2;
}

double HomographyError(const Eigen::Matrix3d& H,
                       const Eigen::Vector2d& x1,
                       const Eigen::Vector2d& x2) {
  Eigen::Vector3d Hx1 = H * x1.homogeneous();
  Eigen::Vector2d Hx1_norm = Hx1.head(2) / (EPS + Hx1[2]);
  double r2 = (Hx1_norm - x2).squaredNorm();

  return r2;
}

}  // namespace glomap