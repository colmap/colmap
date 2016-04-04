// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "base/camera_models"
#include <boost/test/unit_test.hpp>

#include "base/camera_models.h"

using namespace colmap;

template <typename CameraModel>
void TestWorldToImageToWorld(const std::vector<double> camera_params,
                             const double x0, const double y0) {
  double u, v, x, y;
  CameraModel::WorldToImage(camera_params.data(), x0, y0, &u, &v);
  CameraModel::ImageToWorld(camera_params.data(), u, v, &x, &y);
  BOOST_CHECK(std::abs(x - x0) < 1e-6);
  BOOST_CHECK(std::abs(y - y0) < 1e-6);
}

template <typename CameraModel>
void TestImageToWorldToImage(const std::vector<double> camera_params,
                             const double u0, const double v0) {
  double u, v, x, y;
  CameraModel::ImageToWorld(camera_params.data(), u0, v0, &x, &y);
  CameraModel::WorldToImage(camera_params.data(), x, y, &u, &v);
  BOOST_CHECK(std::abs(u - u0) < 1e-6);
  BOOST_CHECK(std::abs(v - v0) < 1e-6);
}

template <typename CameraModel>
void TestModel(const std::vector<double>& camera_params) {
  for (double x0 = -0.5; x0 <= 0.5; x0 += 0.1) {
    for (double y0 = -0.5; y0 <= 0.5; y0 += 0.1) {
      TestWorldToImageToWorld<CameraModel>(camera_params, x0, y0);
    }
  }

  for (double u0 = 0; u0 <= 800; u0 += 50) {
    for (double v0 = 0; v0 <= 800; v0 += 50) {
      TestImageToWorldToImage<CameraModel>(camera_params, u0, v0);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestSimplePinhole) {
  std::vector<double> params = {655.123, 386.123, 511.123};
  TestModel<SimplePinholeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestPinhole) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123};
  TestModel<PinholeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestSimpleRadial) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<SimpleRadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<SimpleRadialCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestRadial) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestModel<RadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<RadialCameraModel>(params);
  params[3] = 0.05;
  TestModel<RadialCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestOpenCV) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123,
                                -0.471,  0.223,   -0.001,  0.001};
  TestModel<OpenCVCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestOpenCVFisheye) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123,
                                -0.471,  0.223,   -0.001,  0.001};
  TestModel<OpenCVFisheyeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestFullOpenCV) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123,
                                -0.471,  0.223,   -0.001,  0.001,
                                0.001,   0.02,    -0.02,   0.001};
  TestModel<FullOpenCVCameraModel>(params);
}
