// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/apriltag_alignment.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateMultiViewRigReconstruction(
    const std::vector<Eigen::Vector3d>& camera_positions,
    const Eigen::Vector3d& look_at,
    const CameraModelId model_id,
    const std::vector<double>& params,
    const size_t width,
    const size_t height) {
  Reconstruction reconstruction;
  Camera camera;
  camera.camera_id = 1;
  camera.model_id = model_id;
  camera.width = width;
  camera.height = height;
  camera.params = params;
  THROW_CHECK(camera.VerifyParams());
  reconstruction.AddCameraWithTrivialRig(std::move(camera));

  for (size_t i = 0; i < camera_positions.size(); ++i) {
    const image_t image_id = static_cast<image_t>(i + 1);
    const Eigen::Vector3d z_axis = (look_at - camera_positions[i]).normalized();
    const Eigen::Vector3d up_hint = std::abs(z_axis.y()) < 0.9
                                        ? Eigen::Vector3d::UnitY()
                                        : Eigen::Vector3d::UnitX();
    const Eigen::Vector3d x_axis = up_hint.cross(z_axis).normalized();
    const Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    Eigen::Matrix3d world_from_cam_rot;
    world_from_cam_rot << x_axis, y_axis, z_axis;
    const Rigid3d cam_from_world =
        Inverse(Rigid3d(Eigen::Quaterniond(world_from_cam_rot),
                        camera_positions[i]));

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image_" + std::to_string(image_id) + ".png");
    reconstruction.AddImageWithTrivialFrame(std::move(image), cam_from_world);
  }
  return reconstruction;
}

TEST(AprilTagAlignment, ParseDistanceAndPositionFiles) {
  const auto test_dir = CreateTestDir();
  const auto dist_path = test_dir / "distances.txt";
  {
    std::ofstream file(dist_path);
    file << "# Comment line\n\n"
         << "0 1 1.25 # inline comment\n"
         << "1 2 0.75\n";
  }
  const auto distances = ReadTagDistanceConstraints(dist_path);
  ASSERT_EQ(distances.size(), 2);
  EXPECT_EQ(distances[0].tag_id_a, 0);
  EXPECT_EQ(distances[0].tag_id_b, 1);
  EXPECT_DOUBLE_EQ(distances[0].target_distance, 1.25);
  EXPECT_EQ(distances[1].tag_id_a, 1);
  EXPECT_EQ(distances[1].tag_id_b, 2);
  EXPECT_DOUBLE_EQ(distances[1].target_distance, 0.75);

  const auto pos_path = test_dir / "positions.txt";
  {
    std::ofstream file(pos_path);
    file << "# tag_id x y z\n"
         << "10 1.5 -2.5 3.25\n"
         << "20 0 0 1\n";
  }
  const auto positions = ReadTagPositionConstraints(pos_path);
  ASSERT_EQ(positions.size(), 2);
  EXPECT_EQ(positions[0].tag_id, 10);
  EXPECT_TRUE(positions[0].position.isApprox(Eigen::Vector3d(1.5, -2.5, 3.25)));
  EXPECT_EQ(positions[1].tag_id, 20);
  EXPECT_TRUE(positions[1].position.isApprox(Eigen::Vector3d(0.0, 0.0, 1.0)));
}

TEST(AprilTagAlignment, TriangulateTagsDistortedAndOutlierPruning) {
  const std::vector<Eigen::Vector3d> cam_positions = {
      {-2.0, 0.0, 0.0},
      {-1.0, 0.5, 0.2},
      {0.0, -0.4, 0.1},
      {1.0, 0.3, -0.2},
      {2.0, 0.0, 0.0},
  };
  const Eigen::Vector3d tag0_xyz(0.2, -0.1, 4.0);
  const Eigen::Vector3d tag1_xyz(-0.3, 0.2, 4.5);

  const Reconstruction reconstruction = CreateMultiViewRigReconstruction(
      cam_positions,
      Eigen::Vector3d(0.0, 0.0, 4.25),
      OpenCVCameraModel::model_id,
      {800.0, 800.0, 500.0, 400.0, -0.15, 0.05, 0.001, -0.002},
      1000,
      800);

  std::vector<AprilTagImageObservation> observations;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    const Camera& camera = *image.CameraPtr();
    const auto proj0 = camera.ImgFromCam(image.CamFromWorld() * tag0_xyz);
    const auto proj1 = camera.ImgFromCam(image.CamFromWorld() * tag1_xyz);
    ASSERT_TRUE(proj0.has_value());
    ASSERT_TRUE(proj1.has_value());

    AprilTagImageObservation obs0;
    obs0.image_id = image_id;
    obs0.image_name = image.Name();
    obs0.tag_id = 0;
    // Inject one strong outlier in view 3 for tag 0.
    obs0.center_px = (image_id == 3) ? (*proj0 + Eigen::Vector2d(45.0, -30.0))
                                     : *proj0;
    observations.push_back(obs0);

    AprilTagImageObservation obs1;
    obs1.image_id = image_id;
    obs1.image_name = image.Name();
    obs1.tag_id = 1;
    obs1.center_px = *proj1;
    observations.push_back(obs1);
  }

  // Add a low-parallax tag observed by two nearly identical viewpoints.
  Reconstruction low_parallax_recon = CreateMultiViewRigReconstruction(
      {{0.0, 0.0, 0.0}, {0.001, 0.0, 0.0}},
      Eigen::Vector3d(0.0, 0.0, 10.0),
      SimplePinholeCameraModel::model_id,
      {800.0, 500.0, 400.0},
      1000,
      800);
  const Eigen::Vector3d far_tag(0.0, 0.0, 10.0);
  std::vector<AprilTagImageObservation> low_parallax_obs;
  for (const image_t id : low_parallax_recon.RegImageIds()) {
    const Image& img = low_parallax_recon.Image(id);
    AprilTagImageObservation obs;
    obs.image_id = id;
    obs.tag_id = 99;
    obs.center_px =
        img.CameraPtr()->ImgFromCam(img.CamFromWorld() * far_tag).value();
    low_parallax_obs.push_back(obs);
  }

  AprilTagTriangulationOptions tri_options;
  tri_options.min_views = 2;
  tri_options.min_parallax_angle_deg = 1.5;
  tri_options.max_reprojection_error_px = 2.0;

  std::vector<TriangulatedAprilTag> triangulated;
  std::vector<RejectedAprilTag> rejected;
  TriangulateAprilTags(
      reconstruction, observations, tri_options, &triangulated, &rejected);
  ASSERT_EQ(triangulated.size(), 2);
  EXPECT_TRUE(rejected.empty());
  EXPECT_EQ(triangulated[0].tag_id, 0);
  EXPECT_EQ(triangulated[0].num_observations, 5);
  EXPECT_EQ(triangulated[0].num_inliers, 4);
  EXPECT_LT((triangulated[0].xyz - tag0_xyz).norm(), 1e-6);
  EXPECT_EQ(triangulated[1].tag_id, 1);
  EXPECT_EQ(triangulated[1].num_inliers, 5);
  EXPECT_LT((triangulated[1].xyz - tag1_xyz).norm(), 1e-6);

  TriangulateAprilTags(low_parallax_recon,
                       low_parallax_obs,
                       tri_options,
                       &triangulated,
                       &rejected);
  EXPECT_TRUE(triangulated.empty());
  ASSERT_EQ(rejected.size(), 1);
  EXPECT_EQ(rejected[0].tag_id, 99);
  EXPECT_EQ(rejected[0].reason, "insufficient_parallax");
}

TEST(AprilTagAlignment, EstimateScaleExactAndWithOutliers) {
  std::vector<TriangulatedAprilTag> tags(4);
  tags[0].tag_id = 0;
  tags[0].xyz = Eigen::Vector3d(0.0, 0.0, 0.0);
  tags[1].tag_id = 1;
  tags[1].xyz = Eigen::Vector3d(1.0, 0.0, 0.0);
  tags[2].tag_id = 2;
  tags[2].xyz = Eigen::Vector3d(0.0, 2.0, 0.0);
  tags[3].tag_id = 3;
  tags[3].xyz = Eigen::Vector3d(1.0, 2.0, 0.0);

  constexpr double kTrueScale = 2.5;
  // Include 4 consistent constraints (scale 2.5) and 1 outlier (scale 8.0).
  const std::vector<TagDistanceConstraint> constraints = {
      {0, 1, 1.0 * kTrueScale},
      {0, 2, 2.0 * kTrueScale},
      {1, 3, 2.0 * kTrueScale},
      {0, 3, std::sqrt(5.0) * kTrueScale},
      {2, 3, 1.0 * 8.0},
  };

  AprilTagScaleAlignmentOptions options;
  options.min_inliers = 1;
  options.min_inlier_ratio = 0.5;
  options.max_relative_error = 0.02;

  Sim3d tform;
  AprilTagAlignmentErrorMetrics metrics;
  std::vector<TagDistanceResidual> residuals;
  ASSERT_TRUE(EstimateAprilTagScale(
      tags, constraints, options, &tform, &metrics, &residuals));

  EXPECT_NEAR(tform.scale(), kTrueScale, 1e-9);
  EXPECT_NEAR(tform.rotation().angularDistance(Eigen::Quaterniond::Identity()),
              0.0,
              1e-9);
  EXPECT_LT(tform.translation().norm(), 1e-9);
  EXPECT_EQ(metrics.num_constraints, 5);
  EXPECT_EQ(metrics.num_evaluable, 5);
  EXPECT_EQ(metrics.num_inliers, 4);
  EXPECT_NEAR(metrics.inlier_ratio, 0.8, 1e-9);
  EXPECT_LT(metrics.rmse, 1e-9);
  ASSERT_EQ(residuals.size(), 5);
  EXPECT_FALSE(residuals[4].inlier);

  // Verify single-constraint scale estimation works with default options.
  Sim3d single_tform;
  ASSERT_TRUE(EstimateAprilTagScale(
      tags, {{0, 1, 3.25}}, AprilTagScaleAlignmentOptions(), &single_tform));
  EXPECT_NEAR(single_tform.scale(), 3.25, 1e-9);
}

TEST(AprilTagAlignment, EstimatePositionAlignmentWithOutliersAndReportJson) {
  SetPRNGSeed(0);
  const Sim3d gt_tgt_from_src(
      1.75,
      Eigen::Quaterniond(Eigen::AngleAxisd(
          0.4, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())),
      Eigen::Vector3d(3.0, -1.5, 2.2));

  std::vector<TriangulatedAprilTag> tags;
  std::vector<TagPositionConstraint> constraints;
  const std::vector<Eigen::Vector3d> src_points = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {1.2, 0.8, -0.5},
      {-0.7, 1.1, 0.6},
  };
  for (size_t i = 0; i < src_points.size(); ++i) {
    TriangulatedAprilTag t;
    t.tag_id = static_cast<int>(i);
    t.xyz = src_points[i];
    t.num_observations = 4;
    t.num_inliers = 4;
    tags.push_back(t);

    TagPositionConstraint c;
    c.tag_id = static_cast<int>(i);
    c.position = gt_tgt_from_src * src_points[i];
    if (i == 5) {
      c.position += Eigen::Vector3d(2.0, -1.0, 1.5);  // outlier
    }
    constraints.push_back(c);
  }

  AprilTagPositionAlignmentOptions options;
  options.min_common_tags = 3;
  options.min_inlier_ratio = 0.5;
  options.ransac_options.max_error = 1e-3;

  Sim3d est_tform;
  AprilTagAlignmentErrorMetrics metrics;
  std::vector<TagPositionResidual> residuals;
  ASSERT_TRUE(EstimateAprilTagPositionAlignment(
      tags, constraints, options, &est_tform, &metrics, &residuals));

  EXPECT_NEAR(est_tform.scale(), gt_tgt_from_src.scale(), 1e-6);
  EXPECT_LT(est_tform.rotation().angularDistance(gt_tgt_from_src.rotation()),
            1e-6);
  EXPECT_LT((est_tform.translation() - gt_tgt_from_src.translation()).norm(),
            1e-6);
  EXPECT_EQ(metrics.num_inliers, 5);
  ASSERT_EQ(residuals.size(), 6);
  EXPECT_FALSE(residuals[5].inlier);

  AprilTagAlignmentReport report;
  report.success = true;
  report.mode = "positions";
  report.tgt_from_src = est_tform;
  report.triangulated_tags = tags;
  report.metrics = metrics;
  report.position_residuals = residuals;

  const auto report_path = CreateTestDir() / "report.json";
  WriteAprilTagAlignmentReportJson(report_path, report);
  boost::property_tree::ptree pt;
  EXPECT_NO_THROW(boost::property_tree::read_json(report_path.string(), pt));
  EXPECT_TRUE(pt.get<bool>("success"));
  EXPECT_EQ(pt.get<std::string>("mode"), "positions");
  EXPECT_NEAR(pt.get<double>("transform.scale"), gt_tgt_from_src.scale(), 1e-6);
}

}  // namespace
}  // namespace colmap
