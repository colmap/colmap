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

#include "colmap/geometry/sim3.h"

#include "colmap/util/logging.h"

#include <fstream>

namespace colmap {

void Sim3d::ToFile(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK(file.good()) << path;
  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);
  file << scale << " " << rotation.w() << " " << rotation.x() << " "
       << rotation.y() << " " << rotation.z() << " " << translation.x() << " "
       << translation.y() << " " << translation.z() << '\n';
}

Sim3d Sim3d::FromFile(const std::string& path) {
  std::ifstream file(path);
  THROW_CHECK(file.good()) << path;
  Sim3d t;
  file >> t.scale;
  file >> t.rotation.w();
  file >> t.rotation.x();
  file >> t.rotation.y();
  file >> t.rotation.z();
  file >> t.translation(0);
  file >> t.translation(1);
  file >> t.translation(2);
  return t;
}

std::ostream& operator<<(std::ostream& stream, const Sim3d& tform) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "Sim3d(scale=" << tform.scale << ", rotation_xyzw=["
         << tform.rotation.coeffs().format(kVecFmt) << "], translation=["
         << tform.translation.format(kVecFmt) << "])";
  return stream;
}

Eigen::Matrix6d PropagateCovarianceForRigid3dUnderSim3(
    const Sim3d& new_from_old_world, const Eigen::Matrix6d& covariance) {
  const Sim3d old_from_new_world = Inverse(new_from_old_world);

  const Eigen::Matrix3d& R = old_from_new_world.rotation.toRotationMatrix();
  const Eigen::Vector3d& t = old_from_new_world.translation;
  const double s = old_from_new_world.scale;

  Eigen::Matrix3d t_hat;
  t_hat << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;

  Eigen::Matrix6d J = Eigen::Matrix6d::Zero();
  J.topLeftCorner<3, 3>() = R;
  J.topRightCorner<3, 3>() = Eigen::Matrix3d::Zero();
  J.bottomLeftCorner<3, 3>() = t_hat * R;
  J.bottomRightCorner<3, 3>() = s * R;

  return J * covariance * J.transpose();
}

Eigen::Matrix3d PropagateCovarianceForPositionUnderSim3(
    const Sim3d& new_from_old_world, const Eigen::Matrix3d& covariance) {
  const Eigen::Matrix3d R = new_from_old_world.rotation.toRotationMatrix();
  const double s = new_from_old_world.scale;
  return s * s * R * covariance * R.transpose();
}

Eigen::Matrix3d PropagateCovarianceForRotationUnderSim3(
    const Sim3d& new_from_old_world, const Eigen::Matrix3d& covariance) {
  const Eigen::Matrix3d R = new_from_old_world.rotation.toRotationMatrix();
  return R * covariance * R.transpose();
}

}  // namespace colmap
