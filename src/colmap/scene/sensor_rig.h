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


#include <colmap/util/types.h>
#include <colmap/scene/camera.h>

#include <vector>

namespace colmap {

// TODO: remove this. tmp development space for general rig support
 
typedef int64_t timestamp_t;

typedef Vector6d InertialBias;

typedef Vector3d Velocity;

typedef uint64_t frame_t;

// TODO: design this
struct Gauge {
  double log_scale = 0.0;  // metric scale of the reconstruction
  Eigen::Vector3d gravity_direction;  // [0., 0., -1.] in the world coordinate frame
  double heading; // w.r.t East or North axis
  Eigen::Vector3d origin; // in ECEF or WGS84
};

struct GeneralReconstruction {
  Gauge gauge; // TODO: should be updated when calling Reconstruction::Normalize()

  // Calibrations
  std::unordered_map<sensor_t, CameraCalibration> cameras;
  std::unordered_map<sensor_t, InertialCalibration> imus;
  std::unordered_map<rig_t, RigCalibration> rigs;

  // Measurements
  std::unordered_map<data_t, Image> images;
  std::unordered_map<data_t, InertialData> inertial_data;
  
  // Poses estimated by the reconstruction process
  std::unordered_map<frame_t, Frame> frames;
  
  // 3D points
  std::unordered_map<point3D_t, Point3D> points3D;
};

enum SensorType {
  Camera = 0,
  IMU = 1,
  Location = 2 // includes GNSS, radios, compass, etc.             
};

typedef std::pair<SensorType, uint64_t> sensor_t; // (sensor_id, sensor type)


struct CameraCalibration: public Camera {
  sensor_t sensor_id;
};

struct InertialParameters {
  // Accelerometer noise density (sigma). [m/s^2*1/sqrt(Hz)]
  double acc_noise_density = 0.01883649;

  // Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
  double gyro_noise_density = 0.00073088444;

  // Accelerometer bias random walk (sigma). [m/s^3*1/sqrt(Hz)]
  double acc_bias_random_walk_sigma = 0.012589254;

  // Gyro bias random walk (sigma). [rad/s^2*1/sqrt(Hz)]
  double gyro_bias_random_walk_sigma = 0.00038765;

  // Accelerometer saturation. [m/s^2]
  double acc_saturation_max = 150;

  // Gyroscope saturation. [rad/s]
  double gyro_saturation_max = 7.8;

  // Norm of the Gravitational acceleration. [m/s^2]
  double gravity_magnitude = 9.81007;

  // Expected IMU Rate [1/s]
  double imu_rate = 20.0;
};

struct InertialCalibration {
  sensor_t sensor_id;
  InertialParameters parameters;
  InertialBias bias;
  // TODO: add rectification matrix?
};


}  // namespace colmap
