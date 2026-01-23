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

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <array>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Function mapping 3D point in the camera frame to 2D point in the image.
// Returns null if the point projection is invalid (e.g., behind the camera).
using ImgFromCamFunc =
    std::function<std::optional<Eigen::Vector2d>(const Eigen::Vector3d&)>;

struct Point2DWithRay {
  // The 2D image point in pixels.
  Eigen::Vector2d image_point;
  // The normaled 3D ray direction in the camera frame.
  Eigen::Vector3d camera_ray;
};

class P3PEstimator {
 public:
  // The 2D image feature observations.
  typedef Point2DWithRay X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  explicit P3PEstimator(ImgFromCamFunc img_from_cam_func);

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world) const;

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D        Normalized 2D image points as Nx2 matrix.
  // @param points3D        3D world points as Nx3 matrix.
  // @param cam_from_world  3x4 projection matrix.
  // @param residuals       Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

 private:
  const ImgFromCamFunc img_from_cam_func_;
};

// Minimal solver for 6-DOF pose and focal length.
class P4PFEstimator {
 public:
  // The 2D image feature observations.
  // Expected to be normalized by the principal point.
  typedef Eigen::Vector2d X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  struct M_t {
    // The transformation from the world to the camera frame.
    Eigen::Matrix3x4d cam_from_world;
    // The focal length of the camera.
    double focal_length = 0.;
  };

  static const int kMinNumSamples = 4;

  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* models);

  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& model,
                        std::vector<double>* residuals);
};

// EPNP solver for the PNP (Perspective-N-Point) problem. The solver needs a
// minimum of 4 2D-3D correspondences.
//
// The algorithm is based on the following paper:
//
//    Lepetit, Vincent, Francesc Moreno-Noguer, and Pascal Fua.
//    "Epnp: An accurate o (n) solution to the pnp problem."
//    International journal of computer vision 81.2 (2009): 155-166.
//
// The implementation is based on their original open-source release, but is
// ported to Eigen and contains several improvements over the original code.
class EPNPEstimator {
 public:
  // The 2D image feature observations.
  typedef Point2DWithRay X_t;
  // The observed 3D features in the world frame.
  typedef Eigen::Vector3d Y_t;
  // The transformation from the world to the camera frame.
  typedef Eigen::Matrix3x4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 4;

  explicit EPNPEstimator(ImgFromCamFunc img_from_cam_func);

  // Estimate the most probable solution of the P3P problem from a set of
  // three 2D-3D point correspondences.
  //
  // @param points2D   Normalized 2D image points as 3x2 matrix.
  // @param points3D   3D world points as 3x3 matrix.
  //
  // @return           Most probable pose as length-1 vector of a 3x4 matrix.
  void Estimate(const std::vector<X_t>& points2D,
                const std::vector<Y_t>& points3D,
                std::vector<M_t>* cams_from_world);

  // Calculate the squared reprojection error given a set of 2D-3D point
  // correspondences and a projection matrix.
  //
  // @param points2D        Normalized 2D image points as Nx2 matrix.
  // @param points3D        3D world points as Nx3 matrix.
  // @param cam_from_world  3x4 projection matrix.
  // @param residuals       Output vector of residuals.
  void Residuals(const std::vector<X_t>& points2D,
                 const std::vector<Y_t>& points3D,
                 const M_t& cam_from_world,
                 std::vector<double>* residuals) const;

 private:
  bool ComputePose(const std::vector<X_t>& points2D,
                   const std::vector<Y_t>& points3D,
                   Eigen::Matrix3x4d* cam_from_world);

  void ChooseControlPoints();
  bool ComputeBarycentricCoordinates();

  Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM();
  Eigen::Matrix<double, 6, 10> ComputeL6x10(
      const Eigen::Matrix<double, 12, 12>& Ut);
  Eigen::Matrix<double, 6, 1> ComputeRho();

  void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);

  void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L_6x10,
                      const Eigen::Matrix<double, 6, 1>& rho,
                      Eigen::Vector4d* betas);

  double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                   const Eigen::Vector4d& betas,
                   Eigen::Matrix3d* R,
                   Eigen::Vector3d* t);

  void ComputeCcs(const Eigen::Vector4d& betas,
                  const Eigen::Matrix<double, 12, 12>& Ut);
  void ComputePcs();

  void SolveForSign();

  void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t);

  double ComputeTotalError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

  const ImgFromCamFunc img_from_cam_func_;
  const std::vector<X_t>* points2D_ = nullptr;
  const std::vector<Y_t>* points3D_ = nullptr;
  std::vector<Eigen::Vector3d> pcs_;
  std::vector<Eigen::Vector4d> alphas_;
  std::array<Eigen::Vector3d, 4> cws_;
  std::array<Eigen::Vector3d, 4> ccs_;
};

// Compute squared reprojection error in pixels.
void ComputeSquaredReprojectionError(
    const std::vector<Point2DWithRay>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const ImgFromCamFunc& img_from_cam_func,
    std::vector<double>* residuals);

}  // namespace colmap
