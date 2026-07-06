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

// Analytical image-projection Jacobian implementations (ImgFromCamWithJac) for
// the perspective camera models declared in models.h. These low-level kernels
// are split into a separate header to keep models.h manageable. models.h
// includes this header at its bottom, so including models.h alone still
// provides these definitions.

#include "colmap/sensor/models.h"

#include <limits>

namespace colmap {

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool SimplePinholeCameraModel::ImgFromCamWithJac(const double* params,
                                                 const double& u,
                                                 const double& v,
                                                 const double& w,
                                                 double* x,
                                                 double* y,
                                                 double* J_params,
                                                 double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f = params[0];
  const double c1 = params[1];
  const double c2 = params[2];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  *x = f * uu + c1;
  *y = f * vv + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
    // x = f * u / w + c1, y = f * v / w + c2
    const double f_inv_w = f * inv_w;
    J_uvw[0] = f_inv_w;
    J_uvw[1] = 0.0;
    J_uvw[2] = -f_inv_w * uu;
    J_uvw[3] = 0.0;
    J_uvw[4] = f_inv_w;
    J_uvw[5] = -f_inv_w * vv;
  }

  if (J_params) {
    // J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy)
    J_params[0] = uu;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = vv;
    J_params[4] = 0.0;
    J_params[5] = 1.0;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool PinholeCameraModel::ImgFromCamWithJac(const double* params,
                                           const double& u,
                                           const double& v,
                                           const double& w,
                                           double* x,
                                           double* y,
                                           double* J_params,
                                           double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f1 = params[0];
  const double f2 = params[1];
  const double c1 = params[2];
  const double c2 = params[3];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  *x = f1 * uu + c1;
  *y = f2 * vv + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
    // x = fx * u / w + cx, y = fy * v / w + cy
    J_uvw[0] = f1 * inv_w;
    J_uvw[1] = 0.0;
    J_uvw[2] = -f1 * inv_w * uu;
    J_uvw[3] = 0.0;
    J_uvw[4] = f2 * inv_w;
    J_uvw[5] = -f2 * inv_w * vv;
  }

  if (J_params) {
    // J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy)
    J_params[0] = uu;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = 0.0;
    J_params[5] = vv;
    J_params[6] = 0.0;
    J_params[7] = 1.0;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool SimpleRadialCameraModel::ImgFromCamWithJac(const double* params,
                                                const double& u,
                                                const double& v,
                                                const double& w,
                                                double* x,
                                                double* y,
                                                double* J_params,
                                                double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f = params[0];
  const double c1 = params[1];
  const double c2 = params[2];
  const double k = params[3];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double r2 = uu2 + vv2;
  const double k_r2 = k * r2;
  const double alpha = 1.0 + k_r2;
  const double xd = alpha * uu;
  const double yd = alpha * vv;

  *x = f * xd + c1;
  *y = f * yd + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
    //
    // x = f * alpha * uu + c1, y = f * alpha * vv + c2
    // where alpha = 1 + k * r2, r2 = uu^2 + vv^2, uu = u/w, vv = v/w
    //
    // Using chain rule:
    // dx/du = f/w * (alpha + 2*k*uu^2)
    // dx/dv = f/w * 2*k*uu*vv
    // dx/dw = -f*uu/w * (1 + 3*k*r2)
    // dy/du = f/w * 2*k*uu*vv
    // dy/dv = f/w * (alpha + 2*k*vv^2)
    // dy/dw = -f*vv/w * (1 + 3*k*r2)

    const double two_k = 2.0 * k;
    const double f_inv_w = f * inv_w;
    const double beta = 1.0 + 3.0 * k_r2;
    const double two_k_uu_vv = two_k * uu * vv;

    J_uvw[0] = f_inv_w * (alpha + two_k * uu2);
    J_uvw[1] = f_inv_w * two_k_uu_vv;
    J_uvw[2] = -f_inv_w * uu * beta;
    J_uvw[3] = f_inv_w * two_k_uu_vv;
    J_uvw[4] = f_inv_w * (alpha + two_k * vv2);
    J_uvw[5] = -f_inv_w * vv * beta;
  }

  if (J_params) {
    // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k)
    //
    // x = f * alpha * uu + cx, y = f * alpha * vv + cy
    //
    // dx/df = alpha * uu, dx/dcx = 1, dx/dcy = 0, dx/dk = f * uu * r2
    // dy/df = alpha * vv, dy/dcx = 0, dy/dcy = 1, dy/dk = f * vv * r2

    J_params[0] = xd;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = f * uu * r2;
    J_params[4] = yd;
    J_params[5] = 0.0;
    J_params[6] = 1.0;
    J_params[7] = f * vv * r2;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool RadialCameraModel::ImgFromCamWithJac(const double* params,
                                          const double& u,
                                          const double& v,
                                          const double& w,
                                          double* x,
                                          double* y,
                                          double* J_params,
                                          double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f = params[0];
  const double c1 = params[1];
  const double c2 = params[2];
  const double k1 = params[3];
  const double k2 = params[4];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double r2 = uu2 + vv2;
  const double r4 = r2 * r2;
  const double radial = k1 * r2 + k2 * r4;
  const double xd = uu * (1.0 + radial);
  const double yd = vv * (1.0 + radial);

  *x = f * xd + c1;
  *y = f * yd + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
    // With xd = uu * (1 + radial), yd = vv * (1 + radial),
    // radial = k1 * r2 + k2 * r2^2, r2 = uu^2 + vv^2, the distortion Jacobian
    // in normalized coordinates (uu, vv) is:
    //   d(xd)/d(uu) = 1 + radial + 2 * uu^2 * d_radial_d_r2
    //   d(xd)/d(vv) = 2 * uu * vv * d_radial_d_r2
    //   d(yd)/d(uu) = 2 * uu * vv * d_radial_d_r2
    //   d(yd)/d(vv) = 1 + radial + 2 * vv^2 * d_radial_d_r2
    // where d_radial_d_r2 = k1 + 2 * k2 * r2. The chain rule through
    // (uu, vv) = (u/w, v/w) yields the columns below.
    const double d_radial_d_r2 = k1 + 2.0 * k2 * r2;
    const double cross = 2.0 * uu * vv * d_radial_d_r2;
    const double a00 = f * (1.0 + radial + 2.0 * uu2 * d_radial_d_r2);
    const double a01 = f * cross;
    const double a10 = f * cross;
    const double a11 = f * (1.0 + radial + 2.0 * vv2 * d_radial_d_r2);

    J_uvw[0] = a00 * inv_w;
    J_uvw[1] = a01 * inv_w;
    J_uvw[2] = -(a00 * uu + a01 * vv) * inv_w;
    J_uvw[3] = a10 * inv_w;
    J_uvw[4] = a11 * inv_w;
    J_uvw[5] = -(a10 * uu + a11 * vv) * inv_w;
  }

  if (J_params) {
    // J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2)
    J_params[0] = xd;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = f * uu * r2;
    J_params[4] = f * uu * r4;
    J_params[5] = yd;
    J_params[6] = 0.0;
    J_params[7] = 1.0;
    J_params[8] = f * vv * r2;
    J_params[9] = f * vv * r4;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool OpenCVCameraModel::ImgFromCamWithJac(const double* params,
                                          const double& u,
                                          const double& v,
                                          const double& w,
                                          double* x,
                                          double* y,
                                          double* J_params,
                                          double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f1 = params[0];
  const double f2 = params[1];
  const double c1 = params[2];
  const double c2 = params[3];
  const double k1 = params[4];
  const double k2 = params[5];
  const double p1 = params[6];
  const double p2 = params[7];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double uv = uu * vv;
  const double r2 = uu2 + vv2;
  const double r4 = r2 * r2;
  const double radial = k1 * r2 + k2 * r4;

  const double du = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
  const double dv = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);
  const double xd = uu + du;
  const double yd = vv + dv;

  *x = f1 * xd + c1;
  *y = f2 * yd + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
    // Partial derivatives of the OpenCV distortion (radial + tangential) in
    // normalized coordinates (uu, vv), with d_radial_d_r2 = k1 + 2 * k2 * r2:
    //   d(du)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
    //   d(du)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
    //   d(dv)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
    //   d(dv)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
    // The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
    const double d_radial_d_r2 = k1 + 2.0 * k2 * r2;
    const double cross = 2.0 * uv * d_radial_d_r2;
    const double du_duu =
        radial + 2.0 * uu2 * d_radial_d_r2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
    const double du_dvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
    const double dv_duu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
    const double dv_dvv =
        radial + 2.0 * vv2 * d_radial_d_r2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

    const double a00 = f1 * (1.0 + du_duu);
    const double a01 = f1 * du_dvv;
    const double a10 = f2 * dv_duu;
    const double a11 = f2 * (1.0 + dv_dvv);

    J_uvw[0] = a00 * inv_w;
    J_uvw[1] = a01 * inv_w;
    J_uvw[2] = -(a00 * uu + a01 * vv) * inv_w;
    J_uvw[3] = a10 * inv_w;
    J_uvw[4] = a11 * inv_w;
    J_uvw[5] = -(a10 * uu + a11 * vv) * inv_w;
  }

  if (J_params) {
    // J_params is a 2x8 matrix (row-major):
    //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2)
    J_params[0] = xd;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * uu * r2;
    J_params[5] = f1 * uu * r4;
    J_params[6] = f1 * 2.0 * uv;
    J_params[7] = f1 * (r2 + 2.0 * uu2);
    J_params[8] = 0.0;
    J_params[9] = yd;
    J_params[10] = 0.0;
    J_params[11] = 1.0;
    J_params[12] = f2 * vv * r2;
    J_params[13] = f2 * vv * r4;
    J_params[14] = f2 * (r2 + 2.0 * vv2);
    J_params[15] = f2 * 2.0 * uv;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool FullOpenCVCameraModel::ImgFromCamWithJac(const double* params,
                                              const double& u,
                                              const double& v,
                                              const double& w,
                                              double* x,
                                              double* y,
                                              double* J_params,
                                              double* J_uvw) {
  if (w < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double f1 = params[0];
  const double f2 = params[1];
  const double c1 = params[2];
  const double c2 = params[3];
  const double k1 = params[4];
  const double k2 = params[5];
  const double p1 = params[6];
  const double p2 = params[7];
  const double k3 = params[8];
  const double k4 = params[9];
  const double k5 = params[10];
  const double k6 = params[11];

  const double inv_w = 1.0 / w;
  const double uu = u * inv_w;
  const double vv = v * inv_w;

  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double uv = uu * vv;
  const double r2 = uu2 + vv2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;

  // Rational radial term: radial = num / den.
  const double num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
  const double inv_den = 1.0 / den;
  const double radial = num * inv_den;

  const double xd = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
  const double yd = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);

  *x = f1 * xd + c1;
  *y = f2 * yd + c2;

  if (J_uvw) {
    // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
    // With xd = uu * radial + tangential_x, yd = vv * radial + tangential_y,
    // and radial = num / den, the derivative of the rational radial term is
    //   d(radial)/d(r2) = (num' * den - num * den') / den^2
    // with num' = k1 + 2*k2*r2 + 3*k3*r4, den' = k4 + 2*k5*r2 + 3*k6*r4.
    // The distortion Jacobian in normalized coordinates (uu, vv) is:
    //   d(xd)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
    //   d(xd)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
    //   d(yd)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
    //   d(yd)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
    // The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
    const double num_prime = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
    const double den_prime = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
    const double d_radial_d_r2 =
        (num_prime * den - num * den_prime) * inv_den * inv_den;
    const double cross = 2.0 * uv * d_radial_d_r2;
    const double xd_duu =
        radial + 2.0 * uu2 * d_radial_d_r2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
    const double xd_dvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
    const double yd_duu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
    const double yd_dvv =
        radial + 2.0 * vv2 * d_radial_d_r2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

    const double a00 = f1 * xd_duu;
    const double a01 = f1 * xd_dvv;
    const double a10 = f2 * yd_duu;
    const double a11 = f2 * yd_dvv;

    J_uvw[0] = a00 * inv_w;
    J_uvw[1] = a01 * inv_w;
    J_uvw[2] = -(a00 * uu + a01 * vv) * inv_w;
    J_uvw[3] = a10 * inv_w;
    J_uvw[4] = a11 * inv_w;
    J_uvw[5] = -(a10 * uu + a11 * vv) * inv_w;
  }

  if (J_params) {
    // J_params is a 2x12 matrix (row-major):
    //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
    // The numerator coefficients enter as d(radial)/d(k_i) = r^(2i) / den; the
    // denominator coefficients as d(radial)/d(k_j) = -num * r^(2j) / den^2.
    const double num_k1 = r2 * inv_den;
    const double num_k2 = r4 * inv_den;
    const double num_k3 = r6 * inv_den;
    const double neg_num_inv_den2 = -num * inv_den * inv_den;
    const double den_k4 = neg_num_inv_den2 * r2;
    const double den_k5 = neg_num_inv_den2 * r4;
    const double den_k6 = neg_num_inv_den2 * r6;

    J_params[0] = xd;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * uu * num_k1;
    J_params[5] = f1 * uu * num_k2;
    J_params[6] = f1 * 2.0 * uv;
    J_params[7] = f1 * (r2 + 2.0 * uu2);
    J_params[8] = f1 * uu * num_k3;
    J_params[9] = f1 * uu * den_k4;
    J_params[10] = f1 * uu * den_k5;
    J_params[11] = f1 * uu * den_k6;
    J_params[12] = 0.0;
    J_params[13] = yd;
    J_params[14] = 0.0;
    J_params[15] = 1.0;
    J_params[16] = f2 * vv * num_k1;
    J_params[17] = f2 * vv * num_k2;
    J_params[18] = f2 * (r2 + 2.0 * vv2);
    J_params[19] = f2 * 2.0 * uv;
    J_params[20] = f2 * vv * num_k3;
    J_params[21] = f2 * vv * den_k4;
    J_params[22] = f2 * vv * den_k5;
    J_params[23] = f2 * vv * den_k6;
  }

  return true;
}

}  // namespace colmap
