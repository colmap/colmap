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

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Quaternion utilities for analytical Jacobian computation in cost functions.
//
// Convention: Eigen quaternion storage order (x, y, z, w), Hamilton product.
// A quaternion q = (x, y, z, w) represents the rotation matrix:
//   R(q) = (w^2 - ||v||^2) I + 2 v v^T + 2 w [v]_x
// where v = (x, y, z) is the vector part.
//
// The 4-vector representation used in matrices is [x, y, z, w] (Eigen order).
// Eigen::Quaterniond::coeffs() returns [x, y, z, w].

// Hamilton quaternion left-multiplication matrix (xyzw storage):
// QuaternionLeftMultMatrix(q) * p = q * p  (as 4-vectors).
inline Eigen::Matrix4d QuaternionLeftMultMatrix(const Eigen::Quaterniond& q) {
  Eigen::Matrix4d Q;
  const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
  // clang-format off
  Q << w, -z,  y, x,
       z,  w, -x, y,
      -y,  x,  w, z,
      -x, -y, -z, w;
  // clang-format on
  return Q;
}

// Hamilton quaternion right-multiplication matrix (xyzw storage):
// QuaternionRightMultMatrix(p) * q = q * p  (as 4-vectors).
inline Eigen::Matrix4d QuaternionRightMultMatrix(const Eigen::Quaterniond& q) {
  Eigen::Matrix4d Q;
  const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
  // clang-format off
  Q <<  w,  z, -y, x,
       -z,  w,  x, y,
        y, -x,  w, z,
       -x, -y, -z, w;
  // clang-format on
  return Q;
}

// Rotates the point and optionally computes the Jacobian of R(q) * p
// w.r.t. Eigen quaternion q (xyzw storage). J_out is a 3x4 row-major matrix.
// Pass nullptr for J_out to skip the Jacobian computation.
inline Eigen::Vector3d QuaternionRotatePointWithJac(const double* q,
                                                    const double* pt,
                                                    double* J_out) {
  const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const double px = pt[0], py = pt[1], pz = pt[2];

  const double qx_py = qx * py, qx_pz = qx * pz;
  const double qy_px = qy * px, qy_pz = qy * pz;
  const double qz_px = qz * px, qz_py = qz * py;

  // R(q) * p = p + 2*w*(v x p) + 2*(v x (v x p))
  const double v_x_p0 = qy_pz - qz_py;
  const double v_x_p1 = qz_px - qx_pz;
  const double v_x_p2 = qx_py - qy_px;

  const double v_x_v_x_p0 = qy * v_x_p2 - qz * v_x_p1;
  const double v_x_v_x_p1 = qz * v_x_p0 - qx * v_x_p2;
  const double v_x_v_x_p2 = qx * v_x_p1 - qy * v_x_p0;

  Eigen::Vector3d pt_out(px + 2.0 * (qw * v_x_p0 + v_x_v_x_p0),
                         py + 2.0 * (qw * v_x_p1 + v_x_v_x_p1),
                         pz + 2.0 * (qw * v_x_p2 + v_x_v_x_p2));

  if (J_out) {
    const double qx_px = qx * px;
    const double qy_py = qy * py;
    const double qz_pz = qz * pz;
    const double qw_px = qw * px;
    const double qw_py = qw * py;
    const double qw_pz = qw * pz;

    J_out[0] = 2.0 * (qy_py + qz_pz);
    J_out[1] = 2.0 * (-2.0 * qy_px + qx_py + qw_pz);
    J_out[2] = 2.0 * (-2.0 * qz_px - qw_py + qx_pz);
    J_out[3] = 2.0 * (-qz_py + qy_pz);

    J_out[4] = 2.0 * (qy_px - 2.0 * qx_py - qw_pz);
    J_out[5] = 2.0 * (qx_px + qz_pz);
    J_out[6] = 2.0 * (qw_px - 2.0 * qz_py + qy_pz);
    J_out[7] = 2.0 * (qz_px - qx_pz);

    J_out[8] = 2.0 * (qz_px + qw_py - 2.0 * qx_pz);
    J_out[9] = 2.0 * (-qw_px + qz_py - 2.0 * qy_pz);
    J_out[10] = 2.0 * (qx_px + qy_py);
    J_out[11] = 2.0 * (-qy_px + qx_py);
  }

  return pt_out;
}

}  // namespace colmap
