// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// TODO: correct and finish this
class PreintegratedImuMeasurement {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d omega_bias_;
  Eigen::Vector3d acc_bias_;
  Eigen::Vector3d delta_t_ij_;
  Eigen::Vector3d delta_v_ij_;
  Quaternion delta_R_ij_;
  double dt_sum_;
  // TODO: add covariance propagation

  // TODO: change this to support time-varying bias
  PreintegratedImuMeasurement(
      const Eigen::Vector3d& omega_bias,
      const Eigen::Vector3d& acc_bias);

  /// Add single measurements to be integrated
  void addMeasurement(const ImuMeasurement& m);

  /// Add many measurements to be integrated
  void addMeasurements(const ImuMeasurements& ms);

private:
  bool last_imu_measurement_set_;
  ImuMeasurement last_imu_measurement;
};

// TODO: implement this ceres cost functor
class PreintegratedImuMeasurementCostFunctor {};

}  // namespace colmap

