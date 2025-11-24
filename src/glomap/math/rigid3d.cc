#include "glomap/math/rigid3d.h"

#include "glomap/scene/camera.h"

namespace glomap {

double CalcAngle(const Rigid3d& pose1, const Rigid3d& pose2) {
  return pose1.rotation.angularDistance(pose2.rotation) * 180 / EIGEN_PI;
}

double CalcTrans(const Rigid3d& pose1, const Rigid3d& pose2) {
  return (Inverse(pose1).translation - Inverse(pose2).translation).norm();
}

double CalcTransAngle(const Rigid3d& pose1, const Rigid3d& pose2) {
  double cos_r = (pose1.translation).dot(pose2.translation) /
                 (pose1.translation.norm() * pose2.translation.norm());
  cos_r = std::min(std::max(cos_r, -1.), 1.);
  return std::acos(cos_r) * 180 / EIGEN_PI;
}

double CalcAngle(const Eigen::Matrix3d& rotation1,
                 const Eigen::Matrix3d& rotation2) {
  double cos_r = ((rotation1.transpose() * rotation2).trace() - 1) / 2;
  cos_r = std::min(std::max(cos_r, -1.), 1.);
  return std::acos(cos_r) * 180 / EIGEN_PI;
}

double DegToRad(double degree) { return degree * EIGEN_PI / 180; }

double RadToDeg(double radian) { return radian * 180 / EIGEN_PI; }

Eigen::Vector3d Rigid3dToAngleAxis(const Rigid3d& pose) {
  Eigen::AngleAxis<double> aa(pose.rotation);
  Eigen::Vector3d aa_vec = aa.angle() * aa.axis();
  return aa_vec;
}

Eigen::Vector3d RotationToAngleAxis(const Eigen::Matrix3d& rot) {
  Eigen::AngleAxis<double> aa(rot);
  Eigen::Vector3d aa_vec = aa.angle() * aa.axis();
  return aa_vec;
}

Eigen::Matrix3d AngleAxisToRotation(const Eigen::Vector3d& aa_vec) {
  double aa_norm = aa_vec.norm();
  if (aa_norm > 1e-12) {
    return Eigen::AngleAxis<double>(aa_norm, aa_vec.normalized())
        .toRotationMatrix();
  } else {
    Eigen::Matrix3d R;
    R(0, 0) = 1;
    R(1, 0) = aa_vec[2];
    R(2, 0) = -aa_vec[1];
    R(0, 1) = -aa_vec[2];
    R(1, 1) = 1;
    R(2, 1) = aa_vec[0];
    R(0, 2) = aa_vec[1];
    R(1, 2) = -aa_vec[0];
    R(2, 2) = 1;
    return R;
  }
}

Eigen::Vector3d CenterFromPose(const Rigid3d& pose) {
  return pose.rotation.inverse() * -pose.translation;
}

}  // namespace glomap
