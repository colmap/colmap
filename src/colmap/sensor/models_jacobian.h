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

#include <cmath>
#include <limits>

namespace colmap {
namespace internal {

// Fisheye (equidistant) projection: maps normalized coordinates
// (a, b) = (u/w, v/w) to the projected fisheye coordinates (uu, vv). When
// J_fisheye is non-null it also receives the 2x2 Jacobian d(uu, vv) / d(a, b)
// in row-major order; pass null on the value-only path to skip that work.
// Mirrors BasePerspectiveFisheyeCameraModel::FisheyeFromNormal.
inline void FisheyeProjectionWithJac(
    double a, double b, double* uu, double* vv, double* J_fisheye) {
  const double r2 = a * a + b * b;
  const double r = std::sqrt(r2);
  if (r < std::numeric_limits<double>::epsilon()) {
    // Identity in the limit r -> 0 (theta / r -> 1).
    *uu = a;
    *vv = b;
    if (J_fisheye) {
      J_fisheye[0] = 1.0;
      J_fisheye[1] = 0.0;
      J_fisheye[2] = 0.0;
      J_fisheye[3] = 1.0;
    }
    return;
  }
  const double theta = std::atan(r);
  const double s = theta / r;
  *uu = s * a;
  *vv = s * b;
  if (J_fisheye) {
    // With s = atan(r) / r and r = sqrt(a^2 + b^2), the Jacobian of (uu, vv) =
    // (s * a, s * b) w.r.t. (a, b) is s * I + g * outer((a, b), (a, b)), where
    // g = (ds/dr) / r = (r / (1 + r^2) - atan(r)) / r^3.
    const double g = (r / (1.0 + r2) - theta) / (r2 * r);
    J_fisheye[0] = s + a * a * g;
    J_fisheye[1] = a * b * g;
    J_fisheye[2] = a * b * g;
    J_fisheye[3] = s + b * b * g;
  }
}

// Solves the one-parameter division model's projection scale r from the camera
// point (u, v, w): the depth is scaled by r = 2 / (w + sqrt(w^2 - 4*k*rho2)),
// with rho2 = u^2 + v^2. Returns false when the point is behind the model's
// projection surface (negative discriminant). When requested, also returns the
// derivatives of r w.r.t. (u, v, w, k).
inline bool DivisionScaleWithJac(double u,
                                 double v,
                                 double w,
                                 double k,
                                 double* r,
                                 double* dr_du,
                                 double* dr_dv,
                                 double* dr_dw,
                                 double* dr_dk) {
  const double rho2 = u * u + v * v;
  const double disc_sq = w * w - 4.0 * rho2 * k;
  if (disc_sq < 0.0) {
    return false;
  }
  const double disc = std::sqrt(disc_sq);
  *r = 2.0 / (w + disc);
  if (dr_du) {
    const double inv_disc = 1.0 / disc;
    const double r_sq = *r * *r;
    *dr_du = 2.0 * r_sq * k * u * inv_disc;
    *dr_dv = 2.0 * r_sq * k * v * inv_disc;
    *dr_dw = -0.5 * r_sq * (1.0 + w * inv_disc);
    *dr_dk = r_sq * rho2 * inv_disc;
  }
  return true;
}

// Multiplies two row-major 2x2 matrices: out = lhs * rhs.
inline void MatMul2x2(const double* lhs, const double* rhs, double* out) {
  out[0] = lhs[0] * rhs[0] + lhs[1] * rhs[2];
  out[1] = lhs[0] * rhs[1] + lhs[1] * rhs[3];
  out[2] = lhs[2] * rhs[0] + lhs[3] * rhs[2];
  out[3] = lhs[2] * rhs[1] + lhs[3] * rhs[3];
}

// Given J_ab (row-major 2x2) = d(x, y) / d(a, b) with a = u/w and b = v/w,
// computes the 2x3 Jacobian J_uvw = d(x, y) / d(u, v, w) via the chain rule
// through (a, b) = (u/w, v/w).
inline void UvwJacFromAbJac(
    const double* J_ab, double a, double b, double inv_w, double* J_uvw) {
  J_uvw[0] = J_ab[0] * inv_w;
  J_uvw[1] = J_ab[1] * inv_w;
  J_uvw[2] = -(J_ab[0] * a + J_ab[1] * b) * inv_w;
  J_uvw[3] = J_ab[2] * inv_w;
  J_uvw[4] = J_ab[3] * inv_w;
  J_uvw[5] = -(J_ab[2] * a + J_ab[3] * b) * inv_w;
}

}  // namespace internal

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

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool FOVCameraModel::ImgFromCamWithJac(const double* params,
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
  const double omega = params[4];

  const double inv_w = 1.0 / w;
  const double a = u * inv_w;
  const double b = v * inv_w;

  const double radius2 = a * a + b * b;
  const double omega2 = omega * omega;

  // Chosen to match FOVCameraModel::Distortion.
  constexpr double kEpsilon = 1e-4;

  // The distortion scales (a, b) by a radially symmetric factor. We compute the
  // factor and its partials factor_r = d(factor)/d(radius2) and factor_omega =
  // d(factor)/d(omega), matching whichever branch FOVCameraModel::Distortion
  // selects so that the analytic Jacobian agrees with autodiff everywhere.
  double factor = 0;
  double factor_r = 0;
  double factor_omega = 0;
  if (omega2 < kEpsilon) {
    factor = (omega2 * radius2) / 3.0 - omega2 / 12.0 + 1.0;
    factor_r = omega2 / 3.0;
    factor_omega = 2.0 * omega * radius2 / 3.0 - omega / 6.0;
  } else if (radius2 < kEpsilon) {
    const double t = std::tan(omega / 2.0);
    const double t2 = t * t;
    // Q = t * (4 * t^2 * radius2 - 3), factor = -2 * Q / (3 * omega).
    const double Q = t * (4.0 * t2 * radius2 - 3.0);
    factor = -2.0 * Q / (3.0 * omega);
    factor_r = -8.0 * t * t2 / (3.0 * omega);
    const double dt_domega = 0.5 * (1.0 + t2);
    const double Q_omega = dt_domega * (12.0 * t2 * radius2 - 3.0);
    factor_omega = -2.0 / (3.0 * omega2) * (Q_omega * omega - Q);
  } else {
    const double radius = std::sqrt(radius2);
    const double t = std::tan(omega / 2.0);
    const double arg = 2.0 * radius * t;
    const double atan_arg = std::atan(arg);
    const double denom_arg = 1.0 + arg * arg;
    // denom_arg divides both derivative numerators; hoist its reciprocal.
    const double inv_denom_arg = 1.0 / denom_arg;
    factor = atan_arg / (radius * omega);
    factor_r = (2.0 * t * radius * inv_denom_arg - atan_arg) /
               (2.0 * radius2 * radius * omega);
    factor_omega = (radius * omega * (1.0 + t * t) * inv_denom_arg - atan_arg) /
                   (radius * omega2);
  }

  const double du = a * factor;
  const double dv = b * factor;

  *x = f1 * du + c1;
  *y = f2 * dv + c2;

  if (J_uvw) {
    // d(du, dv) / d(a, b) with du = a * factor, dv = b * factor and factor a
    // function of radius2 = a^2 + b^2.
    const double cross = 2.0 * a * b * factor_r;
    const double da00 = factor + 2.0 * a * a * factor_r;
    const double da11 = factor + 2.0 * b * b * factor_r;
    const double J_ab[4] = {f1 * da00, f1 * cross, f2 * cross, f2 * da11};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, omega).
    J_params[0] = du;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * a * factor_omega;
    J_params[5] = 0.0;
    J_params[6] = dv;
    J_params[7] = 0.0;
    J_params[8] = 1.0;
    J_params[9] = f2 * b * factor_omega;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool SimpleRadialFisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  // Single-parameter radial distortion in fisheye coordinates.
  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double t2 = uu2 + vv2;
  const double radial = k * t2;
  const double uu_d = uu + uu * radial;
  const double vv_d = vv + vv * radial;

  *x = f * uu_d + c1;
  *y = f * vv_d + c2;

  if (J_uvw) {
    const double two_k = 2.0 * k;
    // I + d(distortion) / d(uu, vv).
    const double ipjd[4] = {1.0 + radial + two_k * uu2,
                            two_k * uu * vv,
                            two_k * uu * vv,
                            1.0 + radial + two_k * vv2};
    double m[4];
    internal::MatMul2x2(ipjd, J_fisheye, m);
    const double J_ab[4] = {f * m[0], f * m[1], f * m[2], f * m[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
    J_params[0] = uu_d;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = f * uu * t2;
    J_params[4] = vv_d;
    J_params[5] = 0.0;
    J_params[6] = 1.0;
    J_params[7] = f * vv * t2;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool RadialFisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double t2 = uu2 + vv2;
  const double t4 = t2 * t2;
  const double radial = k1 * t2 + k2 * t4;
  const double uu_d = uu + uu * radial;
  const double vv_d = vv + vv * radial;

  *x = f * uu_d + c1;
  *y = f * vv_d + c2;

  if (J_uvw) {
    const double d_radial = k1 + 2.0 * k2 * t2;
    const double cross = 2.0 * uu * vv * d_radial;
    const double ipjd[4] = {1.0 + radial + 2.0 * uu2 * d_radial,
                            cross,
                            cross,
                            1.0 + radial + 2.0 * vv2 * d_radial};
    double m[4];
    internal::MatMul2x2(ipjd, J_fisheye, m);
    const double J_ab[4] = {f * m[0], f * m[1], f * m[2], f * m[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2).
    J_params[0] = uu_d;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = f * uu * t2;
    J_params[4] = f * uu * t4;
    J_params[5] = vv_d;
    J_params[6] = 0.0;
    J_params[7] = 1.0;
    J_params[8] = f * vv * t2;
    J_params[9] = f * vv * t4;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool OpenCVFisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double k3 = params[6];
  const double k4 = params[7];

  const double inv_w = 1.0 / w;
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  // Radial distortion in the theta-scaled fisheye coordinates.
  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double t2 = uu2 + vv2;
  const double t4 = t2 * t2;
  const double t6 = t4 * t2;
  const double t8 = t4 * t4;
  const double radial = k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
  const double uu_d = uu + uu * radial;
  const double vv_d = vv + vv * radial;

  *x = f1 * uu_d + c1;
  *y = f2 * vv_d + c2;

  if (J_uvw) {
    const double d_radial = k1 + 2.0 * k2 * t2 + 3.0 * k3 * t4 + 4.0 * k4 * t6;
    const double cross = 2.0 * uu * vv * d_radial;
    const double ipjd[4] = {1.0 + radial + 2.0 * uu2 * d_radial,
                            cross,
                            cross,
                            1.0 + radial + 2.0 * vv2 * d_radial};
    double m[4];
    internal::MatMul2x2(ipjd, J_fisheye, m);
    const double J_ab[4] = {f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x8 matrix (row-major):
    //   d(x, y) / d(fx, fy, cx, cy, k1, k2, k3, k4)
    J_params[0] = uu_d;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * uu * t2;
    J_params[5] = f1 * uu * t4;
    J_params[6] = f1 * uu * t6;
    J_params[7] = f1 * uu * t8;
    J_params[8] = 0.0;
    J_params[9] = vv_d;
    J_params[10] = 0.0;
    J_params[11] = 1.0;
    J_params[12] = f2 * vv * t2;
    J_params[13] = f2 * vv * t4;
    J_params[14] = f2 * vv * t6;
    J_params[15] = f2 * vv * t8;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool ThinPrismFisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double sx1 = params[10];
  const double sy1 = params[11];

  const double inv_w = 1.0 / w;
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  // Radial + tangential + thin-prism distortion in fisheye coordinates.
  const double uu2 = uu * uu;
  const double vv2 = vv * vv;
  const double uv = uu * vv;
  const double r2 = uu2 + vv2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double r8 = r4 * r4;
  const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
  const double du =
      uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2) + sx1 * r2;
  const double dv =
      vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2) + sy1 * r2;
  const double uu_d = uu + du;
  const double vv_d = vv + dv;

  *x = f1 * uu_d + c1;
  *y = f2 * vv_d + c2;

  if (J_uvw) {
    const double d_radial = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
    const double cross = 2.0 * uv * d_radial;
    const double du_duu = radial + 2.0 * uu2 * d_radial + 2.0 * p1 * vv +
                          6.0 * p2 * uu + 2.0 * sx1 * uu;
    const double du_dvv =
        cross + 2.0 * p1 * uu + 2.0 * p2 * vv + 2.0 * sx1 * vv;
    const double dv_duu =
        cross + 2.0 * p2 * vv + 2.0 * p1 * uu + 2.0 * sy1 * uu;
    const double dv_dvv = radial + 2.0 * vv2 * d_radial + 2.0 * p2 * uu +
                          6.0 * p1 * vv + 2.0 * sy1 * vv;
    const double ipjd[4] = {1.0 + du_duu, du_dvv, dv_duu, 1.0 + dv_dvv};
    double m[4];
    internal::MatMul2x2(ipjd, J_fisheye, m);
    const double J_ab[4] = {f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x12 matrix (row-major):
    //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
    J_params[0] = uu_d;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * uu * r2;
    J_params[5] = f1 * uu * r4;
    J_params[6] = f1 * 2.0 * uv;
    J_params[7] = f1 * (r2 + 2.0 * uu2);
    J_params[8] = f1 * uu * r6;
    J_params[9] = f1 * uu * r8;
    J_params[10] = f1 * r2;
    J_params[11] = 0.0;
    J_params[12] = 0.0;
    J_params[13] = vv_d;
    J_params[14] = 0.0;
    J_params[15] = 1.0;
    J_params[16] = f2 * vv * r2;
    J_params[17] = f2 * vv * r4;
    J_params[18] = f2 * (r2 + 2.0 * vv2);
    J_params[19] = f2 * 2.0 * uv;
    J_params[20] = f2 * vv * r6;
    J_params[21] = f2 * vv * r8;
    J_params[22] = 0.0;
    J_params[23] = f2 * r2;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool RadTanThinPrismFisheyeModel::ImgFromCamWithJac(const double* params,
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
  const double* k = &params[4];  // k0..k5 (radial)
  const double p0 = params[10];
  const double p1 = params[11];
  const double s0 = params[12];
  const double s1 = params[13];
  const double s2 = params[14];
  const double s3 = params[15];

  const double inv_w = 1.0 / w;
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  // Radial distortion: (xr, yr) = th_radial * (uu, vv). Also accumulate its
  // derivative th_radial' w.r.t. theta2 and the powers theta2^(i+1) used by the
  // per-coefficient parameter Jacobians.
  const double theta2 = uu * uu + vv * vv;
  double th_radial = 1.0;
  double d_th_radial = 0.0;  // d(th_radial) / d(theta2)
  double theta_pow[6];       // theta2^(i+1)
  double power = 1.0;
  for (int i = 0; i < 6; ++i) {
    const double prev_power = power;  // theta2^i
    power *= theta2;                  // theta2^(i+1)
    theta_pow[i] = power;
    th_radial += k[i] * power;
    d_th_radial += static_cast<double>(i + 1) * k[i] * prev_power;
  }

  const double xr = th_radial * uu;
  const double yr = th_radial * vv;

  // Tangential + thin-prism distortion applied to (xr, yr).
  const double xr2 = xr * xr;
  const double yr2 = yr * yr;
  const double xyr = xr * yr;
  const double r2 = xr2 + yr2;
  const double r4 = r2 * r2;

  const double dx_tang = 2.0 * p1 * xyr + p0 * (r2 + 2.0 * xr2);
  const double dy_tang = 2.0 * p0 * xyr + p1 * (r2 + 2.0 * yr2);
  const double dx_tp = s0 * r2 + s1 * r4;
  const double dy_tp = s2 * r2 + s3 * r4;

  const double X = xr + dx_tang + dx_tp;
  const double Y = yr + dy_tang + dy_tp;

  *x = f1 * X + c1;
  *y = f2 * Y + c2;

  if (J_uvw || J_params) {
    // B = d(X, Y) / d(xr, yr) (tangential + thin-prism stage), used by both the
    // point/pose and the radial-coefficient Jacobians.
    const double B00 = 1.0 + 2.0 * p1 * yr + 6.0 * p0 * xr + 2.0 * s0 * xr +
                       4.0 * s1 * xr * r2;
    const double B01 =
        2.0 * p1 * xr + 2.0 * p0 * yr + 2.0 * s0 * yr + 4.0 * s1 * yr * r2;
    const double B10 =
        2.0 * p0 * yr + 2.0 * p1 * xr + 2.0 * s2 * xr + 4.0 * s3 * xr * r2;
    const double B11 = 1.0 + 2.0 * p0 * xr + 6.0 * p1 * yr + 2.0 * s2 * yr +
                       4.0 * s3 * yr * r2;

    if (J_uvw) {
      // A = d(xr, yr) / d(uu, vv) (radial stage).
      const double cross = 2.0 * uu * vv * d_th_radial;
      const double A[4] = {th_radial + 2.0 * uu * uu * d_th_radial,
                           cross,
                           cross,
                           th_radial + 2.0 * vv * vv * d_th_radial};
      const double B[4] = {B00, B01, B10, B11};
      double m2[4];
      internal::MatMul2x2(B, A, m2);  // d(X, Y) / d(uu, vv)
      double m[4];
      internal::MatMul2x2(m2, J_fisheye, m);  // d(X, Y) / d(a, b)
      const double J_ab[4] = {f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]};
      internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
    }

    if (J_params) {
      // J_params is a 2x16 matrix (row-major):
      //   d(x, y) / d(fx, fy, cx, cy, k0..k5, p0, p1, s0, s1, s2, s3)
      for (size_t j = 0; j < 2 * num_params; ++j) {
        J_params[j] = 0.0;
      }
      // Focal length and principal point.
      J_params[0] = X;         // dx/dfx
      J_params[2] = 1.0;       // dx/dcx
      J_params[16 + 1] = Y;    // dy/dfy
      J_params[16 + 3] = 1.0;  // dy/dcy

      // Radial coefficients k0..k5 (params 4..9): dxr/dk_i = uu * theta2^(i+1),
      // dyr/dk_i = vv * theta2^(i+1), propagated through the tangential/prism
      // stage B.
      for (int i = 0; i < 6; ++i) {
        const double dxr = uu * theta_pow[i];
        const double dyr = vv * theta_pow[i];
        const double dX = B00 * dxr + B01 * dyr;
        const double dY = B10 * dxr + B11 * dyr;
        J_params[4 + i] = f1 * dX;
        J_params[16 + 4 + i] = f2 * dY;
      }

      // Tangential coefficients p0, p1 (params 10, 11).
      J_params[10] = f1 * (r2 + 2.0 * xr2);       // dX/dp0
      J_params[11] = f1 * 2.0 * xyr;              // dX/dp1
      J_params[16 + 10] = f2 * 2.0 * xyr;         // dY/dp0
      J_params[16 + 11] = f2 * (r2 + 2.0 * yr2);  // dY/dp1

      // Thin-prism coefficients s0..s3 (params 12..15).
      J_params[12] = f1 * r2;       // dX/ds0
      J_params[13] = f1 * r4;       // dX/ds1
      J_params[16 + 14] = f2 * r2;  // dY/ds2
      J_params[16 + 15] = f2 * r4;  // dY/ds3
    }
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool SimpleFisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  *x = f * uu + c1;
  *y = f * vv + c2;

  if (J_uvw) {
    const double J_ab[4] = {
        f * J_fisheye[0], f * J_fisheye[1], f * J_fisheye[2], f * J_fisheye[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy).
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
bool FisheyeCameraModel::ImgFromCamWithJac(const double* params,
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
  const double a = u * inv_w;
  const double b = v * inv_w;

  double uu, vv, J_fisheye[4];
  internal::FisheyeProjectionWithJac(
      a, b, &uu, &vv, J_uvw ? J_fisheye : nullptr);

  *x = f1 * uu + c1;
  *y = f2 * vv + c2;

  if (J_uvw) {
    const double J_ab[4] = {f1 * J_fisheye[0],
                            f1 * J_fisheye[1],
                            f2 * J_fisheye[2],
                            f2 * J_fisheye[3]};
    internal::UvwJacFromAbJac(J_ab, a, b, inv_w, J_uvw);
  }

  if (J_params) {
    // J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy).
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
bool SimpleDivisionCameraModel::ImgFromCamWithJac(const double* params,
                                                  const double& u,
                                                  const double& v,
                                                  const double& w,
                                                  double* x,
                                                  double* y,
                                                  double* J_params,
                                                  double* J_uvw) {
  const double f = params[0];
  const double c1 = params[1];
  const double c2 = params[2];
  const double k = params[3];

  // The derivatives are only written when a Jacobian is requested; initialize
  // them so the value-only path does not trip -Wmaybe-uninitialized.
  double r;
  double dr_du = 0.0, dr_dv = 0.0, dr_dw = 0.0, dr_dk = 0.0;
  const bool with_jac = J_uvw || J_params;
  if (!internal::DivisionScaleWithJac(u,
                                      v,
                                      w,
                                      k,
                                      &r,
                                      with_jac ? &dr_du : nullptr,
                                      &dr_dv,
                                      &dr_dw,
                                      &dr_dk)) {
    return false;
  }

  *x = f * r * u + c1;
  *y = f * r * v + c2;

  if (J_uvw) {
    J_uvw[0] = f * (r + u * dr_du);
    J_uvw[1] = f * u * dr_dv;
    J_uvw[2] = f * u * dr_dw;
    J_uvw[3] = f * v * dr_du;
    J_uvw[4] = f * (r + v * dr_dv);
    J_uvw[5] = f * v * dr_dw;
  }

  if (J_params) {
    // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
    J_params[0] = r * u;
    J_params[1] = 1.0;
    J_params[2] = 0.0;
    J_params[3] = f * u * dr_dk;
    J_params[4] = r * v;
    J_params[5] = 0.0;
    J_params[6] = 1.0;
    J_params[7] = f * v * dr_dk;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool DivisionCameraModel::ImgFromCamWithJac(const double* params,
                                            const double& u,
                                            const double& v,
                                            const double& w,
                                            double* x,
                                            double* y,
                                            double* J_params,
                                            double* J_uvw) {
  const double f1 = params[0];
  const double f2 = params[1];
  const double c1 = params[2];
  const double c2 = params[3];
  const double k = params[4];

  // The derivatives are only written when a Jacobian is requested; initialize
  // them so the value-only path does not trip -Wmaybe-uninitialized.
  double r;
  double dr_du = 0.0, dr_dv = 0.0, dr_dw = 0.0, dr_dk = 0.0;
  const bool with_jac = J_uvw || J_params;
  if (!internal::DivisionScaleWithJac(u,
                                      v,
                                      w,
                                      k,
                                      &r,
                                      with_jac ? &dr_du : nullptr,
                                      &dr_dv,
                                      &dr_dw,
                                      &dr_dk)) {
    return false;
  }

  *x = f1 * r * u + c1;
  *y = f2 * r * v + c2;

  if (J_uvw) {
    J_uvw[0] = f1 * (r + u * dr_du);
    J_uvw[1] = f1 * u * dr_dv;
    J_uvw[2] = f1 * u * dr_dw;
    J_uvw[3] = f2 * v * dr_du;
    J_uvw[4] = f2 * (r + v * dr_dv);
    J_uvw[5] = f2 * v * dr_dw;
  }

  if (J_params) {
    // J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, k).
    J_params[0] = r * u;
    J_params[1] = 0.0;
    J_params[2] = 1.0;
    J_params[3] = 0.0;
    J_params[4] = f1 * u * dr_dk;
    J_params[5] = 0.0;
    J_params[6] = r * v;
    J_params[7] = 0.0;
    J_params[8] = 1.0;
    J_params[9] = f2 * v * dr_dk;
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool EUCMCameraModel::ImgFromCamWithJac(const double* params,
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
  const double alpha = params[4];
  const double beta = params[5];

  const double q = u * u + v * v;
  const double rho2 = beta * q + w * w;
  if (rho2 < 0.0) {
    return false;
  }
  const double rho = std::sqrt(rho2);
  const double den = alpha * rho + (1.0 - alpha) * w;
  if (den < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double xn = u / den;
  const double yn = v / den;

  *x = f1 * xn + c1;
  *y = f2 * yn + c2;

  if (J_uvw || J_params) {
    const double inv_rho = 1.0 / rho;
    const double inv_den = 1.0 / den;
    const double inv_den2 = inv_den * inv_den;
    // Derivatives of the denominator den = alpha*rho + (1-alpha)*w.
    const double dden_du = alpha * beta * u * inv_rho;
    const double dden_dv = alpha * beta * v * inv_rho;
    const double dden_dw = alpha * w * inv_rho + (1.0 - alpha);
    const double dden_dalpha = rho - w;
    const double dden_dbeta = alpha * q * 0.5 * inv_rho;

    if (J_uvw) {
      const double dxn_du = inv_den - u * dden_du * inv_den2;
      const double dxn_dv = -u * dden_dv * inv_den2;
      const double dxn_dw = -u * dden_dw * inv_den2;
      const double dyn_du = -v * dden_du * inv_den2;
      const double dyn_dv = inv_den - v * dden_dv * inv_den2;
      const double dyn_dw = -v * dden_dw * inv_den2;
      J_uvw[0] = f1 * dxn_du;
      J_uvw[1] = f1 * dxn_dv;
      J_uvw[2] = f1 * dxn_dw;
      J_uvw[3] = f2 * dyn_du;
      J_uvw[4] = f2 * dyn_dv;
      J_uvw[5] = f2 * dyn_dw;
    }

    if (J_params) {
      // J_params is a 2x6 matrix (row-major):
      //   d(x, y) / d(fx, fy, cx, cy, alpha, beta)
      const double dxn_dalpha = -u * dden_dalpha * inv_den2;
      const double dxn_dbeta = -u * dden_dbeta * inv_den2;
      const double dyn_dalpha = -v * dden_dalpha * inv_den2;
      const double dyn_dbeta = -v * dden_dbeta * inv_den2;
      J_params[0] = xn;
      J_params[1] = 0.0;
      J_params[2] = 1.0;
      J_params[3] = 0.0;
      J_params[4] = f1 * dxn_dalpha;
      J_params[5] = f1 * dxn_dbeta;
      J_params[6] = 0.0;
      J_params[7] = yn;
      J_params[8] = 0.0;
      J_params[9] = 1.0;
      J_params[10] = f2 * dyn_dalpha;
      J_params[11] = f2 * dyn_dbeta;
    }
  }

  return true;
}

template <bool Enable, typename std::enable_if<Enable, int>::type>
bool EquirectangularCameraModel::ImgFromCamWithJac(const double* params,
                                                   const double& u,
                                                   const double& v,
                                                   const double& w,
                                                   double* x,
                                                   double* y,
                                                   double* J_params,
                                                   double* J_uvw) {
  const double width = params[0];
  const double height = params[1];

  const double horizontal = std::sqrt(u * u + w * w);
  if (horizontal + std::abs(v) < std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double theta = std::atan2(u, w);
  const double phi = std::atan2(-v, horizontal);

  constexpr double kInv2Pi = 1.0 / (2.0 * EIGEN_PI);
  constexpr double kInvPi = 1.0 / EIGEN_PI;

  *x = (theta * kInv2Pi + 0.5) * width;
  *y = (0.5 - phi * kInvPi) * height;

  if (J_uvw) {
    const double R2 = horizontal * horizontal;  // horizontal^2
    const double N2 = R2 + v * v;               // full squared norm
    // Hoist the shared reciprocals: R2 and N2*horizontal each divide more than
    // one derivative, and without -ffast-math the compiler cannot factor the
    // repeated runtime division out on its own.
    const double inv_R2 = 1.0 / R2;
    const double inv_N2 = 1.0 / N2;
    const double inv_N2_horizontal = inv_N2 / horizontal;
    // theta = atan2(u, w).
    const double dtheta_du = w * inv_R2;
    const double dtheta_dw = -u * inv_R2;
    // phi = atan2(-v, horizontal), horizontal = sqrt(u^2 + w^2).
    const double dphi_du = u * v * inv_N2_horizontal;
    const double dphi_dv = -horizontal * inv_N2;
    const double dphi_dw = v * w * inv_N2_horizontal;

    J_uvw[0] = width * kInv2Pi * dtheta_du;
    J_uvw[1] = 0.0;
    J_uvw[2] = width * kInv2Pi * dtheta_dw;
    J_uvw[3] = -height * kInvPi * dphi_du;
    J_uvw[4] = -height * kInvPi * dphi_dv;
    J_uvw[5] = -height * kInvPi * dphi_dw;
  }

  if (J_params) {
    // J_params is a 2x2 matrix (row-major): d(x, y) / d(width, height).
    J_params[0] = theta * kInv2Pi + 0.5;
    J_params[1] = 0.0;
    J_params[2] = 0.0;
    J_params[3] = 0.5 - phi * kInvPi;
  }

  return true;
}

}  // namespace colmap
