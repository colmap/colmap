// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/exe/model.h"

#include "colmap/geometry/gps.h"
#include "colmap/geometry/sim3.h"
#include "colmap/math/math.h"
#include "colmap/scene/database.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

#include <Eigen/Geometry>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::vector<std::string> SplitCSVRow(const std::string& row) {
  std::vector<std::string> fields;
  std::stringstream stream(row);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

TEST(ModelAligner, PosePriorGeoreferenceReport) {
  const std::filesystem::path test_dir = CreateTestDir();
  const std::filesystem::path input_path = test_dir / "input";
  const std::filesystem::path output_path = test_dir / "output";
  const std::filesystem::path database_path = test_dir / "database.db";
  const std::filesystem::path report_path = test_dir / "georeference.json";
  const std::filesystem::path csv_path = test_dir / "residuals.csv";
  std::filesystem::create_directories(input_path);
  std::filesystem::create_directories(output_path);

  Reconstruction source;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 8;
  options.num_points3D = 80;
  SynthesizeDataset(options, &source);

  const std::vector<image_t> image_ids = source.RegImageIds();
  ASSERT_EQ(image_ids.size(), 8u);
  Eigen::Vector3d center_mean = Eigen::Vector3d::Zero();
  for (const image_t image_id : image_ids) {
    center_mean += source.Image(image_id).ProjectionCenter();
  }
  center_mean /= static_cast<double>(image_ids.size());

  const Eigen::Quaterniond enu_from_sfm_rotation =
      Eigen::AngleAxisd(DegToRad(27.0), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(DegToRad(-8.0), Eigen::Vector3d::UnitX());
  double unscaled_max_horizontal_baseline = 0.0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      const Eigen::Vector3d delta =
          enu_from_sfm_rotation *
          (source.Image(image_ids[i]).ProjectionCenter() -
           source.Image(image_ids[j]).ProjectionCenter());
      unscaled_max_horizontal_baseline =
          std::max(unscaled_max_horizontal_baseline, delta.head<2>().norm());
    }
  }
  ASSERT_GT(unscaled_max_horizontal_baseline, 0.0);
  const double scale = 2000.0 / unscaled_max_horizontal_baseline;
  const Eigen::Vector3d translation =
      Eigen::Vector3d(100.0, -50.0, 30.0) -
      scale * (enu_from_sfm_rotation * center_mean);
  const Sim3d nominal_enu_from_sfm(scale, enu_from_sfm_rotation, translation);

  Reconstruction target = source;
  target.Transform(nominal_enu_from_sfm);
  source.Write(input_path);

  const double reference_lat = 45.5;
  const double reference_lon = -73.6;
  const double reference_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  std::vector<std::pair<std::string, image_t>> sorted_images;
  for (const image_t image_id : image_ids) {
    sorted_images.emplace_back(source.Image(image_id).Name(), image_id);
  }
  std::sort(sorted_images.begin(), sorted_images.end());
  const image_t position_outlier_id = sorted_images.front().second;
  const image_t orientation_outlier_id = sorted_images.back().second;
  Eigen::Vector3d position_outlier_lla;

  auto database = Database::Open(database_path);
  for (const auto& [camera_id, camera] : source.Cameras()) {
    database->WriteCamera(camera, /*use_camera_id=*/true);
  }
  for (const auto& [image_id, source_image] : source.Images()) {
    Image database_image;
    database_image.SetImageId(image_id);
    database_image.SetName(source_image.Name());
    database_image.SetCameraId(source_image.CameraId());
    database->WriteImage(database_image, /*use_image_id=*/true);

    Eigen::Vector3d center_enu = target.Image(image_id).ProjectionCenter();
    if (image_id == position_outlier_id) {
      center_enu += Eigen::Vector3d(5000.0, 0.0, 0.0);
    }
    const Eigen::Vector3d lla = gps_transform.ENUToEllipsoid(
        {center_enu}, reference_lat, reference_lon, reference_alt)[0];
    if (image_id == position_outlier_id) {
      position_outlier_lla = lla;
    }

    const Eigen::Matrix3d shared_from_local =
        GPSTransform::ENUFromECEF(reference_lat, reference_lon) *
        GPSTransform::ECEFFromENU(lla.x(), lla.y());
    Eigen::Quaterniond sensor_from_shared =
        target.Image(image_id).CamFromWorld().rotation();
    if (image_id == orientation_outlier_id) {
      sensor_from_shared = Eigen::Quaterniond(Eigen::AngleAxisd(
                               DegToRad(90.0), Eigen::Vector3d::UnitY())) *
                           sensor_from_shared;
    }

    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, source_image.CameraId()), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = lla;
    prior.position_covariance = Eigen::Vector3d(0.25, 0.25, 1.0).asDiagonal();
    prior.rotation =
        (sensor_from_shared * Eigen::Quaterniond(shared_from_local.transpose()))
            .normalized();
    prior.rotation_covariance =
        Eigen::Vector3d::Constant(DegToRad(1.0) * DegToRad(1.0)).asDiagonal();
    // Sensor-frame down vector consistent with the (ENU-aligned) target
    // reconstruction's orientation, so the gravity-consistency diagnostic is
    // small and neither warning fires.
    prior.gravity = target.Image(image_id).CamFromWorld().rotation() *
                    Eigen::Vector3d(0.0, 0.0, -1.0);
    database->WritePosePrior(prior);
  }

  Image unregistered_image;
  const image_t unregistered_image_id =
      *std::max_element(image_ids.begin(), image_ids.end()) + 1;
  unregistered_image.SetImageId(unregistered_image_id);
  unregistered_image.SetName("unregistered,quoted.jpg");
  unregistered_image.SetCameraId(source.Cameras().begin()->first);
  database->WriteImage(unregistered_image, /*use_image_id=*/true);
  database.reset();

  std::vector<std::string> args{
      "model_aligner",
      "--input_path",
      input_path.string(),
      "--output_path",
      output_path.string(),
      "--database_path",
      database_path.string(),
      "--alignment_type",
      "enu",
      "--alignment_max_error",
      "10",
      "--min_common_images",
      "3",
      "--use_pose_prior_orientation",
      "1",
      "--orientation_max_error_deg",
      "10",
      "--alignment_random_seed",
      "12345",
      "--scene_id",
      "fixed-test-scene",
      "--georeference_json",
      report_path.string(),
      "--georeference_report_level",
      "full",
      "--camera_residuals_csv",
      csv_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EQ(RunModelAligner(static_cast<int>(argv.size()), argv.data()),
            EXIT_SUCCESS);

  boost::property_tree::ptree report;
  boost::property_tree::read_json(report_path.string(), report);
  EXPECT_EQ(report.get<std::string>("schema"), "colmap_scene_georeference");
  EXPECT_EQ(report.get<std::string>("scene_id"), "fixed-test-scene");
  EXPECT_FALSE(report.get<bool>("enu_origin.explicit"));
  EXPECT_TRUE(report.get<bool>("orientation_requested"));
  EXPECT_TRUE(report.get<bool>("orientation_engaged"));
  EXPECT_EQ(report.get<int>("support.num_position_inliers"), 7);
  EXPECT_EQ(report.get<int>("support.num_orientation_candidates"), 8);
  EXPECT_EQ(report.get<int>("support.num_orientation_inliers"), 7);
  EXPECT_EQ(report.get<int>("support.num_registered"), 8);
  EXPECT_NEAR(report.get<double>("metres_per_sfm_unit"), scale, 1e-4 * scale);
  // The 8-camera synthetic layout is normalized to a 2000 m baseline above,
  // but this diagnostic is measured over the 7 position-prior inliers only
  // (excluding position_outlier_id), so its exact value depends on which
  // point the PRNG happens to place at the layout's extremity. Use a
  // generous margin rather than a value tied to one PRNG implementation's
  // specific output.
  EXPECT_GT(report.get<double>("diagnostics.max_horizontal_baseline_m"),
            1000.0);
  EXPECT_LT(report.get<double>("diagnostics.position_3d_residual_m.max"), 0.1);
  EXPECT_LT(report.get<double>("diagnostics.orientation_residual_deg.max"),
            0.5);
  EXPECT_GT(report.get<double>("diagnostics.max_ellipsoid_tangent_departure_m"),
            0.0);
  EXPECT_GT(std::abs(report.get<double>("enu_origin.lat_deg") -
                     position_outlier_lla.x()),
            1e-4);
  EXPECT_EQ(report.get<int>("alignment_random_seed"), 12345);

  // Frame contract.
  const auto& frame_contract = report.get_child("frame_contract");
  EXPECT_EQ(frame_contract.get<int>("schema_version"), 2);
  EXPECT_EQ(frame_contract.get<std::string>("geometry_frame"), "ENU_LOCAL");
  EXPECT_TRUE(frame_contract.get<bool>("geometry_already_transformed"));
  EXPECT_EQ(frame_contract.get<std::string>("handedness"), "RIGHT");
  EXPECT_EQ(frame_contract.get<std::string>("up_axis"), "Z");
  EXPECT_EQ(frame_contract.get<std::string>("units"), "METRE");
  EXPECT_EQ(frame_contract.get<std::string>("crs.ellipsoid"), "WGS84");
  EXPECT_EQ(frame_contract.get<std::string>("crs.height_datum"), "ELLIPSOIDAL");
  EXPECT_NEAR(frame_contract.get<double>("crs.origin.lat_deg"),
              report.get<double>("enu_origin.lat_deg"),
              1e-9);
  const auto& target_entry = frame_contract.get_child("targets").front().second;
  EXPECT_EQ(target_entry.get<std::string>("name"), "GLTF_Y_UP");
  Eigen::Matrix3d target_from_geometry;
  {
    int row = 0;
    for (const auto& row_node :
         target_entry.get_child("matrix_row_major_target_from_geometry")) {
      int col = 0;
      for (const auto& value_node : row_node.second) {
        target_from_geometry(row, col) = value_node.second.get_value<double>();
        ++col;
      }
      ASSERT_EQ(col, 3);
      ++row;
    }
    ASSERT_EQ(row, 3);
  }
  EXPECT_NEAR(target_from_geometry.determinant(), 1.0, 1e-12);
  EXPECT_LT((target_from_geometry * Eigen::Vector3d(1.0, 0.0, 0.0) -
             Eigen::Vector3d(1.0, 0.0, 0.0))
                .norm(),
            1e-12);
  EXPECT_LT((target_from_geometry * Eigen::Vector3d(0.0, 0.0, 1.0) -
             Eigen::Vector3d(0.0, 1.0, 0.0))
                .norm(),
            1e-12);
  EXPECT_LT((target_from_geometry * Eigen::Vector3d(0.0, 1.0, 0.0) -
             Eigen::Vector3d(0.0, 0.0, -1.0))
                .norm(),
            1e-12);

  // Post-alignment diagnostics and warnings. The fixture's positions are
  // well-spread and gravity priors match the ENU-aligned target orientation,
  // so neither warning should fire.
  EXPECT_TRUE(std::isfinite(
      report.get<double>("diagnostics.horizontal_condition_ratio")));
  EXPECT_TRUE(std::isfinite(
      report.get<double>("diagnostics.gravity_consistency_angle_deg")));
  EXPECT_LT(report.get<double>("diagnostics.gravity_consistency_angle_deg"),
            3.0);
  // gravity_consistency_angle_deg is now defined as the median of the
  // per-image gravity_residual_deg stats, not a separate mean-vector angle.
  EXPECT_NEAR(report.get<double>("diagnostics.gravity_consistency_angle_deg"),
              report.get<double>("diagnostics.gravity_residual_deg.median"),
              1e-9);
  EXPECT_EQ(report.get<int>("diagnostics.gravity_residual_deg.num_support"), 8);
  EXPECT_GE(report.get<double>("diagnostics.gravity_residual_deg.max"),
            report.get<double>("diagnostics.gravity_residual_deg.median"));
  EXPECT_TRUE(std::isfinite(
      report.get<double>("diagnostics.gravity_residual_deg.mean")));
  const auto& warnings = report.get_child("warnings");
  EXPECT_EQ(warnings.get<double>("collinearity.threshold"), 0.1);
  EXPECT_FALSE(warnings.get<bool>("collinearity.fired"));
  EXPECT_TRUE(std::isfinite(warnings.get<double>("collinearity.value")));
  EXPECT_EQ(warnings.get<double>("gravity_disagreement.threshold"), 3.0);
  EXPECT_FALSE(warnings.get<bool>("gravity_disagreement.fired"));
  EXPECT_TRUE(
      std::isfinite(warnings.get<double>("gravity_disagreement.value")));

  // Position inlier ratio: 7/8 registered correspondences are inliers,
  // above the 0.8 policy threshold, so the warning does not fire.
  EXPECT_EQ(warnings.get<double>("position_inlier_ratio.threshold"), 0.8);
  EXPECT_FALSE(warnings.get<bool>("position_inlier_ratio.fired"));
  EXPECT_NEAR(
      warnings.get<double>("position_inlier_ratio.value"), 7.0 / 8.0, 1e-9);

  std::ifstream report_stream(report_path);
  const std::string report_text((std::istreambuf_iterator<char>(report_stream)),
                                std::istreambuf_iterator<char>());
  EXPECT_NE(report_text.find("\"metres_per_sfm_unit\":"), std::string::npos);
  EXPECT_EQ(report_text.find("\"metres_per_sfm_unit\":\""), std::string::npos);

  const std::vector<std::string> csv_lines = ReadTextFileLines(csv_path);
  ASSERT_EQ(csv_lines.size(), 10u);
  EXPECT_NE(csv_lines.back().find("\"unregistered,quoted.jpg\",0"),
            std::string::npos);
  for (const std::string& line : csv_lines) {
    const std::vector<std::string> fields = SplitCSVRow(line);
    if (line.rfind(source.Image(position_outlier_id).Name(), 0) == 0) {
      ASSERT_GE(fields.size(), 19u);
      EXPECT_EQ(fields[3], "0");
      EXPECT_FALSE(fields[4].empty());
      EXPECT_FALSE(fields[7].empty());
      EXPECT_FALSE(fields[10].empty());
    }
    if (line.rfind(source.Image(orientation_outlier_id).Name(), 0) == 0) {
      ASSERT_GE(fields.size(), 19u);
      EXPECT_EQ(fields[16], "1");
      EXPECT_EQ(fields[17], "0");
      EXPECT_GT(std::stod(fields[18]), 45.0);
    }
  }

  Reconstruction aligned;
  aligned.Read(output_path);
  const double source_distance =
      (source.Image(image_ids[0]).ProjectionCenter() -
       source.Image(image_ids[1]).ProjectionCenter())
          .norm();
  const double aligned_distance =
      (aligned.Image(image_ids[0]).ProjectionCenter() -
       aligned.Image(image_ids[1]).ProjectionCenter())
          .norm();
  EXPECT_NEAR(aligned_distance / source_distance, scale, 1e-4 * scale);

  const Sim3d ecef_from_parent(
      1.0, Eigen::Quaterniond::Identity(), Eigen::Vector3d(1.0, 2.0, 3.0));
  const Sim3d parent_from_child(1.0,
                                Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(5.0), Eigen::Vector3d::UnitZ())),
                                Eigen::Vector3d(4.0, 5.0, 6.0));
  const Sim3d ecef_from_unchanged_crop = ecef_from_parent;
  const Sim3d ecef_from_child = ecef_from_parent * parent_from_child;
  const Eigen::Vector3d probe(7.0, 8.0, 9.0);
  EXPECT_LT(
      (Inverse(ecef_from_unchanged_crop) * (ecef_from_unchanged_crop * probe) -
       probe)
          .norm(),
      1e-12);
  EXPECT_LT(
      (Inverse(ecef_from_child) * (ecef_from_child * probe) - probe).norm(),
      1e-12);

  // The same explicit --alignment_random_seed twice yields an identical
  // transform; a different seed still succeeds (equality not required).
  const auto run_and_read_enu_from_sfm =
      [&](const std::string& seed,
          const std::filesystem::path& out_dir,
          const std::filesystem::path& json_path) {
        std::filesystem::create_directories(out_dir);
        std::vector<std::string> seed_args{
            "model_aligner",
            "--input_path",
            input_path.string(),
            "--output_path",
            out_dir.string(),
            "--database_path",
            database_path.string(),
            "--alignment_type",
            "enu",
            "--alignment_max_error",
            "10",
            "--min_common_images",
            "3",
            "--alignment_random_seed",
            seed,
            "--georeference_json",
            json_path.string(),
        };
        std::vector<char*> seed_argv;
        seed_argv.reserve(seed_args.size());
        for (std::string& arg : seed_args) {
          seed_argv.push_back(arg.data());
        }
        EXPECT_EQ(RunModelAligner(static_cast<int>(seed_argv.size()),
                                  seed_argv.data()),
                  EXIT_SUCCESS);
        boost::property_tree::ptree seed_report;
        boost::property_tree::read_json(json_path.string(), seed_report);
        std::vector<double> values;
        values.push_back(
            seed_report.get<double>("transforms.enu_from_sfm.scale"));
        for (const auto& v :
             seed_report.get_child("transforms.enu_from_sfm.rotation_wxyz")) {
          values.push_back(v.second.get_value<double>());
        }
        for (const auto& v :
             seed_report.get_child("transforms.enu_from_sfm.translation_xyz")) {
          values.push_back(v.second.get_value<double>());
        }
        return values;
      };

  const std::vector<double> seed_a1 = run_and_read_enu_from_sfm(
      "777", test_dir / "seed_a1_out", test_dir / "seed_a1.json");
  const std::vector<double> seed_a2 = run_and_read_enu_from_sfm(
      "777", test_dir / "seed_a2_out", test_dir / "seed_a2.json");
  const std::vector<double> seed_b = run_and_read_enu_from_sfm(
      "778", test_dir / "seed_b_out", test_dir / "seed_b.json");
  ASSERT_EQ(seed_a1.size(), seed_a2.size());
  for (size_t i = 0; i < seed_a1.size(); ++i) {
    EXPECT_EQ(seed_a1[i], seed_a2[i]);
  }
  EXPECT_EQ(seed_b.size(), seed_a1.size());
}

TEST(ModelAligner, OutputCoordinateFrameGltfYUp) {
  const std::filesystem::path test_dir = CreateTestDir();
  const std::filesystem::path input_path = test_dir / "input";
  const std::filesystem::path enu_output_path = test_dir / "output_enu";
  const std::filesystem::path yup_output_path = test_dir / "output_yup";
  const std::filesystem::path database_path = test_dir / "database.db";
  const std::filesystem::path enu_report_path =
      test_dir / "georeference_enu.json";
  const std::filesystem::path yup_report_path =
      test_dir / "georeference_yup.json";
  std::filesystem::create_directories(input_path);
  std::filesystem::create_directories(enu_output_path);
  std::filesystem::create_directories(yup_output_path);

  Reconstruction source;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 6;
  options.num_points3D = 50;
  SynthesizeDataset(options, &source);
  source.Write(input_path);

  const double reference_lat = 45.5;
  const double reference_lon = -73.6;
  const double reference_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  auto database = Database::Open(database_path);
  for (const auto& [camera_id, camera] : source.Cameras()) {
    database->WriteCamera(camera, /*use_camera_id=*/true);
  }
  for (const auto& [image_id, source_image] : source.Images()) {
    Image database_image;
    database_image.SetImageId(image_id);
    database_image.SetName(source_image.Name());
    database_image.SetCameraId(source_image.CameraId());
    database->WriteImage(database_image, /*use_image_id=*/true);

    // The source reconstruction's own camera centers, reinterpreted as ENU
    // positions -- the specific frame doesn't matter for this test, only
    // that the aligned model ends up self-consistently in ENU.
    const Eigen::Vector3d center_enu =
        source.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d lla = gps_transform.ENUToEllipsoid(
        {center_enu}, reference_lat, reference_lon, reference_alt)[0];

    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, source_image.CameraId()), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = lla;
    prior.position_covariance = Eigen::Vector3d(1.0, 1.0, 2.0).asDiagonal();
    database->WritePosePrior(prior);
  }
  database.reset();

  const auto run_aligner = [&](const std::filesystem::path& output_path,
                               const std::filesystem::path& report_path,
                               const std::string& output_coordinate_frame) {
    std::vector<std::string> args{
        "model_aligner",
        "--input_path",
        input_path.string(),
        "--output_path",
        output_path.string(),
        "--database_path",
        database_path.string(),
        "--alignment_type",
        "enu",
        "--alignment_max_error",
        "10",
        "--min_common_images",
        "3",
        "--alignment_random_seed",
        "12345",
        "--georeference_json",
        report_path.string(),
        "--output_coordinate_frame",
        output_coordinate_frame,
    };
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
      argv.push_back(arg.data());
    }
    return RunModelAligner(static_cast<int>(argv.size()), argv.data());
  };

  ASSERT_EQ(run_aligner(enu_output_path, enu_report_path, "ENU_Z_UP"),
            EXIT_SUCCESS);
  ASSERT_EQ(run_aligner(yup_output_path, yup_report_path, "GLTF_Y_UP"),
            EXIT_SUCCESS);

  // frame_contract truthfully describes the frame actually written for the
  // Y-up run -- never claim ENU bytes while Y-up bytes were written.
  boost::property_tree::ptree yup_report;
  boost::property_tree::read_json(yup_report_path.string(), yup_report);
  const auto& yup_frame_contract = yup_report.get_child("frame_contract");
  EXPECT_EQ(yup_frame_contract.get<std::string>("geometry_frame"), "GLTF_Y_UP");
  EXPECT_EQ(yup_frame_contract.get<std::string>("up_axis"), "Y");
  const auto& yup_target =
      yup_frame_contract.get_child("targets").front().second;
  EXPECT_EQ(yup_target.get<std::string>("name"), "ENU_LOCAL");
  Eigen::Matrix3d enu_from_yup;
  {
    int row = 0;
    for (const auto& row_node :
         yup_target.get_child("matrix_row_major_target_from_geometry")) {
      int col = 0;
      for (const auto& value_node : row_node.second) {
        enu_from_yup(row, col) = value_node.second.get_value<double>();
        ++col;
      }
      ASSERT_EQ(col, 3);
      ++row;
    }
    ASSERT_EQ(row, 3);
  }
  // Basis mapping and determinant +1 (proper rotation, not a reflection).
  EXPECT_NEAR(enu_from_yup.determinant(), 1.0, 1e-12);
  EXPECT_LT((enu_from_yup * Eigen::Vector3d(1.0, 0.0, 0.0) -
             Eigen::Vector3d(1.0, 0.0, 0.0))
                .norm(),
            1e-12);
  EXPECT_LT((enu_from_yup * Eigen::Vector3d(0.0, 1.0, 0.0) -
             Eigen::Vector3d(0.0, 0.0, 1.0))
                .norm(),
            1e-12);
  EXPECT_LT((enu_from_yup * Eigen::Vector3d(0.0, 0.0, 1.0) -
             Eigen::Vector3d(0.0, -1.0, 0.0))
                .norm(),
            1e-12);

  // The ENU run's frame_contract is unchanged -- default behavior preserved
  // byte-for-byte when --output_coordinate_frame is left at its default.
  boost::property_tree::ptree enu_report;
  boost::property_tree::read_json(enu_report_path.string(), enu_report);
  const auto& enu_frame_contract = enu_report.get_child("frame_contract");
  EXPECT_EQ(enu_frame_contract.get<std::string>("geometry_frame"), "ENU_LOCAL");
  EXPECT_EQ(enu_frame_contract.get<std::string>("up_axis"), "Z");

  // Transform round trip: applying the reported enu_from_yup matrix to the
  // Y-up output's own camera centers must reproduce the ENU output's camera
  // centers, for the actual serialized reconstruction written by each run
  // (not just the in-memory Sim3 math).
  Reconstruction enu_aligned;
  enu_aligned.Read(enu_output_path);
  Reconstruction yup_aligned;
  yup_aligned.Read(yup_output_path);
  const std::vector<image_t> image_ids = source.RegImageIds();
  ASSERT_GE(image_ids.size(), 2u);
  for (const image_t image_id : image_ids) {
    const Eigen::Vector3d enu_center =
        enu_aligned.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d yup_center =
        yup_aligned.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d recovered_enu_center = enu_from_yup * yup_center;
    EXPECT_LT((recovered_enu_center - enu_center).norm(), 1e-6);
  }

  // Reprojection/rigidity invariance: a pure rotation applied consistently
  // to cameras and points preserves every pairwise camera-center distance
  // exactly (a shearing or non-rigid transform would not).
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      const double enu_distance =
          (enu_aligned.Image(image_ids[i]).ProjectionCenter() -
           enu_aligned.Image(image_ids[j]).ProjectionCenter())
              .norm();
      const double yup_distance =
          (yup_aligned.Image(image_ids[i]).ProjectionCenter() -
           yup_aligned.Image(image_ids[j]).ProjectionCenter())
              .norm();
      EXPECT_NEAR(enu_distance, yup_distance, 1e-6);
    }
  }

  // --output_coordinate_frame=GLTF_Y_UP requires an ENU-producing
  // alignment_type; 'plane' does not produce ENU, so this must fail rather
  // than silently rotate the geometry incorrectly.
  std::vector<std::string> bad_args{
      "model_aligner",
      "--input_path",
      input_path.string(),
      "--output_path",
      (test_dir / "bad_output").string(),
      "--alignment_type",
      "plane",
      "--alignment_max_error",
      "10",
      "--output_coordinate_frame",
      "GLTF_Y_UP",
  };
  std::vector<char*> bad_argv;
  bad_argv.reserve(bad_args.size());
  for (std::string& arg : bad_args) {
    bad_argv.push_back(arg.data());
  }
  EXPECT_EQ(RunModelAligner(static_cast<int>(bad_argv.size()), bad_argv.data()),
            EXIT_FAILURE);
}

TEST(ModelAligner, OutputCoordinateFrameLichtfeldColmap) {
  // Verify the LICHTFELD_COLMAP profile's basis mapping (raw COLMAP-data
  // East=+X, Up=-Y, North=+Z), that it is a proper rotation, and that the
  // georeference report's frame_contract.transforms sub-object (a full Sim3,
  // not just an informal rotation note) round-trips the actual serialized
  // reconstruction back to ENU.
  const std::filesystem::path test_dir = CreateTestDir();
  const std::filesystem::path input_path = test_dir / "input";
  const std::filesystem::path enu_output_path = test_dir / "output_enu";
  const std::filesystem::path lf_output_path = test_dir / "output_lf";
  const std::filesystem::path database_path = test_dir / "database.db";
  const std::filesystem::path lf_report_path =
      test_dir / "georeference_lf.json";
  std::filesystem::create_directories(input_path);
  std::filesystem::create_directories(enu_output_path);
  std::filesystem::create_directories(lf_output_path);

  Reconstruction source;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 6;
  options.num_points3D = 50;
  SynthesizeDataset(options, &source);
  source.Write(input_path);

  const double reference_lat = 45.5;
  const double reference_lon = -73.6;
  const double reference_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  auto database = Database::Open(database_path);
  for (const auto& [camera_id, camera] : source.Cameras()) {
    database->WriteCamera(camera, /*use_camera_id=*/true);
  }
  for (const auto& [image_id, source_image] : source.Images()) {
    Image database_image;
    database_image.SetImageId(image_id);
    database_image.SetName(source_image.Name());
    database_image.SetCameraId(source_image.CameraId());
    database->WriteImage(database_image, /*use_image_id=*/true);

    const Eigen::Vector3d center_enu =
        source.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d lla = gps_transform.ENUToEllipsoid(
        {center_enu}, reference_lat, reference_lon, reference_alt)[0];

    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, source_image.CameraId()), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = lla;
    prior.position_covariance = Eigen::Vector3d(1.0, 1.0, 2.0).asDiagonal();
    database->WritePosePrior(prior);
  }
  database.reset();

  const auto run_aligner = [&](const std::filesystem::path& output_path,
                               const std::filesystem::path& report_path,
                               const std::string& output_coordinate_frame) {
    std::vector<std::string> args{
        "model_aligner",
        "--input_path",
        input_path.string(),
        "--output_path",
        output_path.string(),
        "--database_path",
        database_path.string(),
        "--alignment_type",
        "enu",
        "--alignment_max_error",
        "10",
        "--min_common_images",
        "3",
        "--alignment_random_seed",
        "12345",
        "--georeference_json",
        report_path.string(),
        "--output_coordinate_frame",
        output_coordinate_frame,
    };
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
      argv.push_back(arg.data());
    }
    return RunModelAligner(static_cast<int>(argv.size()), argv.data());
  };

  ASSERT_EQ(
      run_aligner(
          enu_output_path, test_dir / "georeference_enu.json", "ENU_Z_UP"),
      EXIT_SUCCESS);
  ASSERT_EQ(run_aligner(lf_output_path, lf_report_path, "LICHTFELD_COLMAP"),
            EXIT_SUCCESS);

  boost::property_tree::ptree report;
  boost::property_tree::read_json(lf_report_path.string(), report);
  const auto& frame_contract = report.get_child("frame_contract");
  EXPECT_EQ(frame_contract.get<std::string>("geometry_frame"),
            "LICHTFELD_COLMAP");
  EXPECT_EQ(frame_contract.get<std::string>("up_axis"), "-Y");

  const auto& consumer_profile = frame_contract.get_child("consumer_profile");
  EXPECT_EQ(consumer_profile.get<std::string>("name"), "LICHTFELD_COLMAP");
  EXPECT_EQ(consumer_profile.get<int>("contract_version"), 1);
  EXPECT_EQ(consumer_profile.get<std::string>("boundary"),
            "DATA_TO_VISUALIZER_WORLD_AXES");
  EXPECT_FALSE(consumer_profile.get<std::string>("source_reference").empty());
  EXPECT_EQ(consumer_profile.get<std::string>("visualizer_up_axis"), "Y");

  const auto parse_sim3 = [](const boost::property_tree::ptree& node) {
    std::vector<double> rotation_wxyz;
    for (const auto& v : node.get_child("rotation_wxyz")) {
      rotation_wxyz.push_back(v.second.get_value<double>());
    }
    std::vector<double> translation_xyz;
    for (const auto& v : node.get_child("translation_xyz")) {
      translation_xyz.push_back(v.second.get_value<double>());
    }
    Sim3d s;
    s.scale() = node.get<double>("scale");
    s.rotation() = Eigen::Quaterniond(
        rotation_wxyz[0], rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3]);
    s.translation() = Eigen::Vector3d(
        translation_xyz[0], translation_xyz[1], translation_xyz[2]);
    return s;
  };

  const auto& transforms = frame_contract.get_child("transforms");
  const Sim3d geometry_from_enu =
      parse_sim3(transforms.get_child("geometry_from_enu"));
  const Sim3d enu_from_geometry =
      parse_sim3(transforms.get_child("enu_from_geometry"));

  // Basis mapping (raw COLMAP data): East->+X, North->+Z, Up->-Y. Determinant
  // +1 (proper rotation, handedness preserved).
  const Eigen::Matrix3d rotation =
      geometry_from_enu.rotation().toRotationMatrix();
  EXPECT_NEAR(rotation.determinant(), 1.0, 1e-9);
  EXPECT_LT((rotation * Eigen::Vector3d(1.0, 0.0, 0.0) -
             Eigen::Vector3d(1.0, 0.0, 0.0))
                .norm(),
            1e-9);
  EXPECT_LT((rotation * Eigen::Vector3d(0.0, 1.0, 0.0) -
             Eigen::Vector3d(0.0, 0.0, 1.0))
                .norm(),
            1e-9);
  EXPECT_LT((rotation * Eigen::Vector3d(0.0, 0.0, 1.0) -
             Eigen::Vector3d(0.0, -1.0, 0.0))
                .norm(),
            1e-9);

  // Full-Sim3 round trip on the actual serialized reconstructions (not just
  // in-memory math): transforms.enu_from_geometry applied to the
  // LICHTFELD_COLMAP output's camera centers must reproduce the ENU output's
  // camera centers.
  Reconstruction enu_aligned;
  enu_aligned.Read(enu_output_path);
  Reconstruction lf_aligned;
  lf_aligned.Read(lf_output_path);
  const std::vector<image_t> image_ids = source.RegImageIds();
  ASSERT_GE(image_ids.size(), 2u);
  for (const image_t image_id : image_ids) {
    const Eigen::Vector3d enu_center =
        enu_aligned.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d lf_center =
        lf_aligned.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d recovered_enu_center = enu_from_geometry * lf_center;
    EXPECT_LT((recovered_enu_center - enu_center).norm(), 1e-6);
  }
}

TEST(ModelAligner, GeoreferenceReportLevelControlsDiagnosticDetail) {
  const std::filesystem::path test_dir = CreateTestDir();
  const std::filesystem::path input_path = test_dir / "input";
  const std::filesystem::path output_path = test_dir / "output";
  const std::filesystem::path database_path = test_dir / "database.db";
  std::filesystem::create_directories(input_path);
  std::filesystem::create_directories(output_path);

  Reconstruction source;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 6;
  options.num_points3D = 50;
  SynthesizeDataset(options, &source);
  source.Write(input_path);

  const double reference_lat = 45.5;
  const double reference_lon = -73.6;
  const double reference_alt = 120.0;
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  auto database = Database::Open(database_path);
  for (const auto& [camera_id, camera] : source.Cameras()) {
    database->WriteCamera(camera, /*use_camera_id=*/true);
  }
  for (const auto& [image_id, source_image] : source.Images()) {
    Image database_image;
    database_image.SetImageId(image_id);
    database_image.SetName(source_image.Name());
    database_image.SetCameraId(source_image.CameraId());
    database->WriteImage(database_image, /*use_image_id=*/true);

    const Eigen::Vector3d center_enu =
        source.Image(image_id).ProjectionCenter();
    const Eigen::Vector3d lla = gps_transform.ENUToEllipsoid(
        {center_enu}, reference_lat, reference_lon, reference_alt)[0];

    PosePrior prior;
    prior.corr_data_id =
        data_t(sensor_t(SensorType::CAMERA, source_image.CameraId()), image_id);
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position = lla;
    prior.position_covariance = Eigen::Vector3d(1.0, 1.0, 2.0).asDiagonal();
    database->WritePosePrior(prior);
  }
  database.reset();

  const auto run_aligner =
      [&](const std::filesystem::path& output_path,
          const std::filesystem::path& report_path,
          const std::vector<std::string>& extra_args) {
        std::vector<std::string> args{
            "model_aligner",
            "--input_path",
            input_path.string(),
            "--output_path",
            output_path.string(),
            "--database_path",
            database_path.string(),
            "--alignment_type",
            "enu",
            "--alignment_max_error",
            "10",
            "--min_common_images",
            "3",
            "--alignment_random_seed",
            "12345",
            "--georeference_json",
            report_path.string(),
        };
        args.insert(args.end(), extra_args.begin(), extra_args.end());
        std::vector<char*> argv;
        argv.reserve(args.size());
        for (std::string& arg : args) {
          argv.push_back(arg.data());
        }
        return RunModelAligner(static_cast<int>(argv.size()), argv.data());
      };

  const std::filesystem::path summary_output = test_dir / "output_summary";
  const std::filesystem::path summary_report = test_dir / "summary.json";
  std::filesystem::create_directories(summary_output);
  ASSERT_EQ(run_aligner(summary_output, summary_report, {}), EXIT_SUCCESS);

  const std::filesystem::path full_output = test_dir / "output_full";
  const std::filesystem::path full_report = test_dir / "full.json";
  std::filesystem::create_directories(full_output);
  ASSERT_EQ(
      run_aligner(
          full_output, full_report, {"--georeference_report_level", "full"}),
      EXIT_SUCCESS);

  boost::property_tree::ptree summary;
  boost::property_tree::read_json(summary_report.string(), summary);
  boost::property_tree::ptree full;
  boost::property_tree::read_json(full_report.string(), full);

  EXPECT_EQ(summary.get<std::string>("report_level"), "summary");
  EXPECT_EQ(full.get<std::string>("report_level"), "full");

  // Both levels contain everything needed to georeference the geometry.
  for (const boost::property_tree::ptree* report : {&summary, &full}) {
    EXPECT_EQ(report->get<std::string>("schema"), "colmap_scene_georeference");
    EXPECT_NO_THROW(report->get_child("frame_contract"));
    EXPECT_NO_THROW(report->get_child("transforms"));
    EXPECT_NO_THROW(report->get_child("support"));
    EXPECT_NO_THROW(report->get_child("warnings.collinearity"));
    EXPECT_NO_THROW(report->get_child("warnings.gravity_disagreement"));
    EXPECT_NO_THROW(report->get_child("warnings.position_inlier_ratio"));
    EXPECT_NO_THROW(report->get_child("final_realignment_check"));
    // No gravity priors in this fixture, so the value itself is JSON null;
    // only its presence at both levels is being verified here.
    EXPECT_NO_THROW(
        report->get_child("diagnostics.gravity_consistency_angle_deg"));
  }

  // `full`-only detailed diagnostics are absent from `summary`.
  EXPECT_THROW(summary.get_child("diagnostics.position_3d_residual_m"),
              boost::property_tree::ptree_bad_path);
  EXPECT_THROW(summary.get_child("diagnostics.gravity_residual_deg"),
              boost::property_tree::ptree_bad_path);
  EXPECT_THROW(summary.get<double>("diagnostics.max_horizontal_baseline_m"),
              boost::property_tree::ptree_bad_path);
  EXPECT_THROW(summary.get<double>("diagnostics.horizontal_condition_ratio"),
              boost::property_tree::ptree_bad_path);

  // The same detailed diagnostics are present in `full`.
  EXPECT_NO_THROW(full.get_child("diagnostics.position_3d_residual_m"));
  EXPECT_NO_THROW(full.get_child("diagnostics.gravity_residual_deg"));
  EXPECT_TRUE(
      std::isfinite(full.get<double>("diagnostics.max_horizontal_baseline_m")));
  EXPECT_TRUE(std::isfinite(
      full.get<double>("diagnostics.horizontal_condition_ratio")));

  // The collinearity warning's own value/threshold/fired triple is present
  // at both levels regardless of whether the full diagnostics breakdown is
  // -- it is the "evaluated quality value" the report policy is judged
  // against, not a detailed breakdown.
  EXPECT_TRUE(
      std::isfinite(summary.get<double>("warnings.collinearity.value")));
}

}  // namespace
}  // namespace colmap
