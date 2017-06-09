// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define TEST_NAME "base/camera_models"
#include "util/testing.h"

#include "base/camera_models.h"

using namespace colmap;

template <typename CameraModel>
void TestWorldToImageToWorld(const std::vector<double> params, const double u0,
                             const double v0) {
  double u, v, x, y, xx, yy;
  CameraModel::WorldToImage(params.data(), u0, v0, &x, &y);
  CameraModelWorldToImage(CameraModel::model_id, params, u0, v0, &xx, &yy);
  BOOST_CHECK_EQUAL(x, xx);
  BOOST_CHECK_EQUAL(y, yy);
  CameraModel::ImageToWorld(params.data(), x, y, &u, &v);
  BOOST_CHECK(std::abs(u - u0) < 1e-6);
  BOOST_CHECK(std::abs(v - v0) < 1e-6);
}

template <typename CameraModel>
void TestImageToWorldToImage(const std::vector<double> params, const double x0,
                             const double y0) {
  double u, v, x, y, uu, vv;
  CameraModel::ImageToWorld(params.data(), x0, y0, &u, &v);
  CameraModelImageToWorld(CameraModel::model_id, params, x0, y0, &uu, &vv);
  BOOST_CHECK_EQUAL(u, uu);
  BOOST_CHECK_EQUAL(v, vv);
  CameraModel::WorldToImage(params.data(), u, v, &x, &y);
  BOOST_CHECK(std::abs(x - x0) < 1e-6);
  BOOST_CHECK(std::abs(y - y0) < 1e-6);
}

template <typename CameraModel>
void TestModel(const std::vector<double>& params) {
  BOOST_CHECK(CameraModelVerifyParams(CameraModel::model_id, params));

  std::vector<double> default_params;
  CameraModelInitializeParams(CameraModel::model_id, 100, 100, 100,
                              &default_params);
  BOOST_CHECK(CameraModelVerifyParams(CameraModel::model_id, default_params));

  BOOST_CHECK_EQUAL(CameraModelParamsInfo(CameraModel::model_id),
                    CameraModel::params_info);
  BOOST_CHECK_EQUAL(&CameraModelFocalLengthIdxs(CameraModel::model_id),
                    &CameraModel::focal_length_idxs);
  BOOST_CHECK_EQUAL(&CameraModelPrincipalPointIdxs(CameraModel::model_id),
                    &CameraModel::principal_point_idxs);
  BOOST_CHECK_EQUAL(&CameraModelExtraParamsIdxs(CameraModel::model_id),
                    &CameraModel::extra_params_idxs);
  BOOST_CHECK_EQUAL(CameraModelNumParams(CameraModel::model_id),
                    CameraModel::num_params);

  BOOST_CHECK(!CameraModelHasBogusParams(CameraModel::model_id, default_params,
                                         100, 100, 0.1, 2.0, 1.0));
  BOOST_CHECK(CameraModelHasBogusParams(CameraModel::model_id, default_params,
                                        100, 100, 0.1, 0.5, 1.0));
  BOOST_CHECK(CameraModelHasBogusParams(CameraModel::model_id, default_params,
                                        100, 100, 1.5, 2.0, 1.0));
  if (CameraModel::extra_params_idxs.size() > 0) {
    BOOST_CHECK(CameraModelHasBogusParams(CameraModel::model_id, default_params,
                                          100, 100, 0.1, 2.0, -0.1));
  }

  BOOST_CHECK_EQUAL(
      CameraModelImageToWorldThreshold(CameraModel::model_id, params, 0), 0);
  BOOST_CHECK_GT(
      CameraModelImageToWorldThreshold(CameraModel::model_id, params, 1), 0);
  BOOST_CHECK_EQUAL(CameraModelImageToWorldThreshold(CameraModel::model_id,
                                                     default_params, 1),
                    1.0 / 100.0);

  BOOST_CHECK_EQUAL(
      CameraModelNameToId(CameraModelIdToName(CameraModel::model_id)),
      CameraModel::model_id);
  BOOST_CHECK_EQUAL(
      CameraModelIdToName(CameraModelNameToId(CameraModel::model_name)),
      CameraModel::model_name);

  for (double u = -0.5; u <= 0.5; u += 0.1) {
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      TestWorldToImageToWorld<CameraModel>(params, u, v);
    }
  }

  for (double x = 0; x <= 800; x += 50) {
    for (double y = 0; y <= 800; y += 50) {
      TestImageToWorldToImage<CameraModel>(params, x, y);
    }
  }

  const auto pp_idxs = CameraModel::principal_point_idxs;
  TestImageToWorldToImage<CameraModel>(params, params[pp_idxs.at(0)],
                                       params[pp_idxs.at(1)]);
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
  params[4] = 0.03;
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

BOOST_AUTO_TEST_CASE(TestFOV) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123, 0.9};
  TestModel<FOVCameraModel>(params);
  params[4] = 0;
  TestModel<FOVCameraModel>(params);
  params[4] = 1e-8;
  TestModel<FOVCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestSimpleRadialFisheye) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<SimpleRadialFisheyeCameraModel>(params);
  params[3] = 0.1;
  TestModel<SimpleRadialFisheyeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestRadialFisheye) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestModel<RadialFisheyeCameraModel>(params);
  params[3] = 0.1;
  TestModel<RadialFisheyeCameraModel>(params);
  params[3] = 0.05;
  TestModel<RadialFisheyeCameraModel>(params);
  params[4] = 0.03;
  TestModel<RadialFisheyeCameraModel>(params);
}

BOOST_AUTO_TEST_CASE(TestThinPrismFisheye) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123,
                                -0.471,  0.223,   -0.001,  0.001,
                                0.001,   0.02,    -0.02,   0.001};
  TestModel<ThinPrismFisheyeCameraModel>(params);
}
