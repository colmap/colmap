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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#define TEST_NAME "optim/bundle_adjustment"
#include "colmap/optim/bundle_adjustment.h"

#include "colmap/base/camera_models.h"
#include "colmap/base/correspondence_graph.h"
#include "colmap/base/projection.h"
#include "colmap/util/random.h"
#include "colmap/util/testing.h"

#define CheckVariableCamera(camera, orig_camera)          \
  {                                                       \
    const size_t focal_length_idx =                       \
        SimpleRadialCameraModel::focal_length_idxs[0];    \
    const size_t extra_param_idx =                        \
        SimpleRadialCameraModel::extra_params_idxs[0];    \
    BOOST_CHECK_NE(camera.Params(focal_length_idx),       \
                   orig_camera.Params(focal_length_idx)); \
    BOOST_CHECK_NE(camera.Params(extra_param_idx),        \
                   orig_camera.Params(extra_param_idx));  \
  }

#define CheckConstantCamera(camera, orig_camera)             \
  {                                                          \
    const size_t focal_length_idx =                          \
        SimpleRadialCameraModel::focal_length_idxs[0];       \
    const size_t extra_param_idx =                           \
        SimpleRadialCameraModel::extra_params_idxs[0];       \
    BOOST_CHECK_EQUAL(camera.Params(focal_length_idx),       \
                      orig_camera.Params(focal_length_idx)); \
    BOOST_CHECK_EQUAL(camera.Params(extra_param_idx),        \
                      orig_camera.Params(extra_param_idx));  \
  }

#define CheckVariableImage(image, orig_image)        \
  {                                                  \
    BOOST_CHECK_NE(image.Qvec(), orig_image.Qvec()); \
    BOOST_CHECK_NE(image.Tvec(), orig_image.Tvec()); \
  }

#define CheckConstantImage(image, orig_image)           \
  {                                                     \
    BOOST_CHECK_EQUAL(image.Qvec(), orig_image.Qvec()); \
    BOOST_CHECK_EQUAL(image.Tvec(), orig_image.Tvec()); \
  }

#define CheckConstantXImage(image, orig_image)            \
  {                                                       \
    CheckVariableImage(image, orig_image);                \
    BOOST_CHECK_EQUAL(image.Tvec(0), orig_image.Tvec(0)); \
  }

#define CheckConstantCameraRig(camera_rig, orig_camera_rig, camera_id) \
  {                                                                    \
    BOOST_CHECK_EQUAL(camera_rig.RelativeQvec(camera_id),              \
                      orig_camera_rig.RelativeQvec(camera_id));        \
    BOOST_CHECK_EQUAL(camera_rig.RelativeTvec(camera_id),              \
                      orig_camera_rig.RelativeTvec(camera_id));        \
  }

#define CheckVariableCameraRig(camera_rig, orig_camera_rig, camera_id) \
  {                                                                    \
    if (camera_rig.RefCameraId() == camera_id) {                       \
      CheckConstantCameraRig(camera_rig, orig_camera_rig, camera_id);  \
    } else {                                                           \
      BOOST_CHECK_NE(camera_rig.RelativeQvec(camera_id),               \
                     orig_camera_rig.RelativeQvec(camera_id));         \
      BOOST_CHECK_NE(camera_rig.RelativeTvec(camera_id),               \
                     orig_camera_rig.RelativeTvec(camera_id));         \
    }                                                                  \
  }

#define CheckVariablePoint(point, orig_point) \
  { BOOST_CHECK_NE(point.XYZ(), orig_point.XYZ()); }

#define CheckConstantPoint(point, orig_point) \
  { BOOST_CHECK_EQUAL(point.XYZ(), orig_point.XYZ()); }

using namespace colmap;

void GeneratePointCloud(const size_t num_points,
                        const Eigen::Vector3d& min,
                        const Eigen::Vector3d& max,
                        Reconstruction* reconstruction) {
  for (size_t i = 0; i < num_points; ++i) {
    Eigen::Vector3d xyz;
    xyz.x() = RandomReal(min.x(), max.x());
    xyz.y() = RandomReal(min.y(), max.y());
    xyz.z() = RandomReal(min.z(), max.z());
    reconstruction->AddPoint3D(xyz, Track());
  }
}

void GenerateReconstruction(const size_t num_images,
                            const size_t num_points,
                            Reconstruction* reconstruction,
                            CorrespondenceGraph* correspondence_graph) {
  SetPRNGSeed(0);

  GeneratePointCloud(num_points,
                     Eigen::Vector3d(-1, -1, -1),
                     Eigen::Vector3d(1, 1, 1),
                     reconstruction);

  const double kFocalLengthFactor = 1.2;
  const size_t kImageSize = 1000;

  for (size_t i = 0; i < num_images; ++i) {
    const camera_t camera_id = static_cast<camera_t>(i);
    const image_t image_id = static_cast<image_t>(i);

    Camera camera;
    camera.InitializeWithId(SimpleRadialCameraModel::model_id,
                            kFocalLengthFactor * kImageSize,
                            kImageSize,
                            kImageSize);
    camera.SetCameraId(camera_id);
    reconstruction->AddCamera(camera);

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(camera_id);
    image.SetName(std::to_string(i));
    image.Qvec() = ComposeIdentityQuaternion();
    image.Tvec() =
        Eigen::Vector3d(RandomReal(-1.0, 1.0), RandomReal(-1.0, 1.0), 10);
    image.SetRegistered(true);
    reconstruction->AddImage(image);

    const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();

    std::vector<Eigen::Vector2d> points2D;
    for (const auto& point3D : reconstruction->Points3D()) {
      BOOST_CHECK(HasPointPositiveDepth(proj_matrix, point3D.second.XYZ()));
      // Get exact projection of 3D point.
      Eigen::Vector2d point2D =
          ProjectPointToImage(point3D.second.XYZ(), proj_matrix, camera);
      // Add some uniform noise.
      point2D += Eigen::Vector2d(RandomReal(-2.0, 2.0), RandomReal(-2.0, 2.0));
      points2D.push_back(point2D);
    }

    correspondence_graph->AddImage(image_id, num_points);
    reconstruction->Image(image_id).SetPoints2D(points2D);
  }

  reconstruction->SetUp(correspondence_graph);

  for (size_t i = 0; i < num_images; ++i) {
    const image_t image_id = static_cast<image_t>(i);
    TrackElement track_el;
    track_el.image_id = image_id;
    track_el.point2D_idx = 0;
    for (const auto& point3D : reconstruction->Points3D()) {
      reconstruction->AddObservation(point3D.first, track_el);
      track_el.point2D_idx += 1;
    }
  }
}

BOOST_AUTO_TEST_CASE(TestConfigNumObservations) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(4, 100, &reconstruction, &correspondence_graph);

  BundleAdjustmentConfig config;

  config.AddImage(0);
  config.AddImage(1);
  BOOST_CHECK_EQUAL(config.NumResiduals(reconstruction), 400);

  config.AddVariablePoint(1);
  BOOST_CHECK_EQUAL(config.NumResiduals(reconstruction), 404);

  config.AddConstantPoint(2);
  BOOST_CHECK_EQUAL(config.NumResiduals(reconstruction), 408);

  config.AddImage(2);
  BOOST_CHECK_EQUAL(config.NumResiduals(reconstruction), 604);

  config.AddImage(3);
  BOOST_CHECK_EQUAL(config.NumResiduals(reconstruction), 800);
}

BOOST_AUTO_TEST_CASE(TestTwoView) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 309);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestTwoViewConstantCamera) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.SetConstantCamera(0);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 302);

  CheckConstantCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestPartiallyContainedTracks) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, 100, &reconstruction, &correspondence_graph);
  const auto variable_point3D_id =
      reconstruction.Image(2).Point2D(0).Point3DId();
  reconstruction.DeleteObservation(2, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 1 x 3 point parameters
  // 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 7);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantImage(reconstruction.Image(2), orig_reconstruction.Image(2));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == variable_point3D_id) {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestPartiallyContainedTracksForceToOptimizePoint) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, 100, &reconstruction, &correspondence_graph);
  const point3D_t variable_point3D_id =
      reconstruction.Image(2).Point2D(0).Point3DId();
  const point3D_t add_variable_point3D_id =
      reconstruction.Image(2).Point2D(1).Point3DId();
  const point3D_t add_constant_point3D_id =
      reconstruction.Image(2).Point2D(2).Point3DId();
  reconstruction.DeleteObservation(2, 0);

  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.AddVariablePoint(add_variable_point3D_id);
  config.AddConstantPoint(add_constant_point3D_id);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  // + 2 residuals in 3rd image for added variable 3D point
  // (added constant point does not add residuals since the image/camera
  // is also constant).
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 402);
  // 2 x 3 point parameters
  // 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 10);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckConstantImage(reconstruction.Image(2), orig_reconstruction.Image(2));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == variable_point3D_id ||
        point3D.first == add_variable_point3D_id) {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestConstantPoints) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  const point3D_t constant_point3D_id1 = 1;
  const point3D_t constant_point3D_id2 = 2;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantPose(1);
  config.AddConstantPoint(constant_point3D_id1);
  config.AddConstantPoint(constant_point3D_id2);

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 98 x 3 point parameters
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 298);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  for (const auto& point3D : reconstruction.Points3D()) {
    if (point3D.first == constant_point3D_id1 ||
        point3D.first == constant_point3D_id2) {
      CheckConstantPoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    } else {
      CheckVariablePoint(point3D.second,
                         orig_reconstruction.Point3D(point3D.first));
    }
  }
}

BOOST_AUTO_TEST_CASE(TestVariableImage) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(3, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  BundleAdjustmentOptions options;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 600);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 6 image parameters (pose of third image)
  // + 3 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 317);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCamera(reconstruction.Camera(2), orig_reconstruction.Camera(2));
  CheckVariableImage(reconstruction.Image(2), orig_reconstruction.Image(2));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestConstantFocalLength) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 307);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  BOOST_CHECK(camera0.Params(focal_length_idx) ==
              orig_camera0.Params(focal_length_idx));
  BOOST_CHECK(camera0.Params(extra_param_idx) !=
              orig_camera0.Params(extra_param_idx));

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  BOOST_CHECK(camera1.Params(focal_length_idx) ==
              orig_camera1.Params(focal_length_idx));
  BOOST_CHECK(camera1.Params(extra_param_idx) !=
              orig_camera1.Params(extra_param_idx));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestVariablePrincipalPoint) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  BundleAdjustmentOptions options;
  options.refine_principal_point = true;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 8 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 313);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t principal_point_idx_x =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t principal_point_idx_y =
      SimpleRadialCameraModel::principal_point_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  BOOST_CHECK(camera0.Params(focal_length_idx) !=
              orig_camera0.Params(focal_length_idx));
  BOOST_CHECK(camera0.Params(principal_point_idx_x) !=
              orig_camera0.Params(principal_point_idx_x));
  BOOST_CHECK(camera0.Params(principal_point_idx_y) !=
              orig_camera0.Params(principal_point_idx_y));
  BOOST_CHECK(camera0.Params(extra_param_idx) !=
              orig_camera0.Params(extra_param_idx));

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  BOOST_CHECK(camera1.Params(focal_length_idx) !=
              orig_camera1.Params(focal_length_idx));
  BOOST_CHECK(camera1.Params(principal_point_idx_x) !=
              orig_camera1.Params(principal_point_idx_x));
  BOOST_CHECK(camera1.Params(principal_point_idx_y) !=
              orig_camera1.Params(principal_point_idx_y));
  BOOST_CHECK(camera1.Params(extra_param_idx) !=
              orig_camera1.Params(extra_param_idx));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestConstantExtraParam) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.SetConstantPose(0);
  config.SetConstantTvec(1, {0});

  BundleAdjustmentOptions options;
  options.refine_extra_params = false;
  BundleAdjuster bundle_adjuster(options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 3 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 5 image parameters (pose of second image)
  // + 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 307);

  CheckConstantImage(reconstruction.Image(0), orig_reconstruction.Image(0));
  CheckConstantXImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  const size_t focal_length_idx = SimpleRadialCameraModel::focal_length_idxs[0];
  const size_t extra_param_idx = SimpleRadialCameraModel::extra_params_idxs[0];

  const auto& camera0 = reconstruction.Camera(0);
  const auto& orig_camera0 = orig_reconstruction.Camera(0);
  BOOST_CHECK(camera0.Params(focal_length_idx) !=
              orig_camera0.Params(focal_length_idx));
  BOOST_CHECK(camera0.Params(extra_param_idx) ==
              orig_camera0.Params(extra_param_idx));

  const auto& camera1 = reconstruction.Camera(1);
  const auto& orig_camera1 = orig_reconstruction.Camera(1);
  BOOST_CHECK(camera1.Params(focal_length_idx) !=
              orig_camera1.Params(focal_length_idx));
  BOOST_CHECK(camera1.Params(extra_param_idx) ==
              orig_camera1.Params(extra_param_idx));

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestRigTwoView) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(2, 100, &reconstruction, &correspondence_graph);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 400);
  // 100 x 3 point parameters
  // + 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 316);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestRigFourView) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(4, 100, &reconstruction, &correspondence_graph);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2, 3});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 322);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestConstantRigFourView) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(4, 100, &reconstruction, &correspondence_graph);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2, 3});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  rig_options.refine_relative_poses = false;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 316);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckConstantCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckConstantCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}

BOOST_AUTO_TEST_CASE(TestRigFourViewPartial) {
  Reconstruction reconstruction;
  CorrespondenceGraph correspondence_graph;
  GenerateReconstruction(4, 100, &reconstruction, &correspondence_graph);
  reconstruction.Image(2).SetCameraId(0);
  reconstruction.Image(3).SetCameraId(1);
  const auto orig_reconstruction = reconstruction;

  BundleAdjustmentConfig config;
  config.AddImage(0);
  config.AddImage(1);
  config.AddImage(2);
  config.AddImage(3);

  std::vector<CameraRig> camera_rigs;
  camera_rigs.emplace_back();
  camera_rigs[0].AddCamera(
      0, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddCamera(
      1, ComposeIdentityQuaternion(), Eigen::Vector3d(0, 0, 0));
  camera_rigs[0].AddSnapshot({0, 1});
  camera_rigs[0].AddSnapshot({2});
  camera_rigs[0].SetRefCameraId(0);
  const auto orig_camera_rigs = camera_rigs;

  BundleAdjustmentOptions options;
  RigBundleAdjuster::Options rig_options;
  RigBundleAdjuster bundle_adjuster(options, rig_options, config);
  BOOST_REQUIRE(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  const auto summary = bundle_adjuster.Summary();

  // 100 points, 2 images, 2 residuals per point per image
  BOOST_CHECK_EQUAL(summary.num_residuals_reduced, 800);
  // 100 x 3 point parameters
  // + 2 x 6 pose parameters for camera rig
  // + 1 x 6 relative pose parameters for camera rig
  // + 1 x 6 pose parameters for individual image
  // + 2 x 2 camera parameters
  BOOST_CHECK_EQUAL(summary.num_effective_parameters_reduced, 328);

  CheckVariableCamera(reconstruction.Camera(0), orig_reconstruction.Camera(0));
  CheckVariableImage(reconstruction.Image(0), orig_reconstruction.Image(0));

  CheckVariableCamera(reconstruction.Camera(1), orig_reconstruction.Camera(1));
  CheckVariableImage(reconstruction.Image(1), orig_reconstruction.Image(1));

  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 0);
  CheckVariableCameraRig(camera_rigs[0], orig_camera_rigs[0], 1);

  for (const auto& point3D : reconstruction.Points3D()) {
    CheckVariablePoint(point3D.second,
                       orig_reconstruction.Point3D(point3D.first));
  }
}
