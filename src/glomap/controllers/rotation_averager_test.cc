#include "glomap/controllers/rotation_averager.h"

#include "colmap/estimators/alignment.h"
#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include "glomap/controllers/global_mapper.h"
#include "glomap/estimators/gravity_refinement.h"
#include "glomap/io/colmap_io.h"
#include "glomap/math/rigid3d.h"
#include "glomap/types.h"

#include <gtest/gtest.h>

namespace glomap {
namespace {

Eigen::AngleAxisd CreateRandomRotation(std::mt19937& rng, double stddev) {
  const double theta = colmap::RandomUniformReal<double>(0, 2 * M_PI);
  const double phi = colmap::RandomUniformReal<double>(0, M_PI);
  const Eigen::Vector3d axis(std::cos(theta) * std::sin(phi),
                             std::sin(theta) * std::sin(phi),
                             std::cos(phi));
  std::normal_distribution<double> d{0, stddev};
  const double angle = d(rng);
  return Eigen::AngleAxisd(angle, axis);
}

void PrepareGravity(const colmap::Reconstruction& gt,
                    std::unordered_map<frame_t, Frame>& frames,
                    double gravity_noise_stddev = 0.0,
                    double outlier_ratio = 0.0) {
  std::mt19937 rng{std::random_device{}()};
  const Eigen::Vector3d kGravityInWorld = Eigen::Vector3d(0, 1, 0);
  for (auto& frame_id : gt.RegFrameIds()) {
    Eigen::Vector3d gravityInRig =
        gt.Frame(frame_id).RigFromWorld().rotation * kGravityInWorld;

    if (gravity_noise_stddev > 0.0) {
      gravityInRig =
          CreateRandomRotation(rng, colmap::DegToRad(gravity_noise_stddev)) *
          gravityInRig;
    }

    if (outlier_ratio > 0.0 &&
        colmap::RandomUniformReal<double>(0, 1) < outlier_ratio) {
      const Eigen::AngleAxisd q = CreateRandomRotation(rng, 1.);
      gravityInRig = (q.angle() * q.axis()).normalized();
    }

    frames[frame_id].gravity_info.SetGravity(gravityInRig);
    Rigid3d& cam_from_world = frames[frame_id].RigFromWorld();
    cam_from_world.rotation = frames[frame_id].gravity_info.GetRAlign();
  }
}

GlobalMapperOptions CreateMapperTestOptions() {
  GlobalMapperOptions options;
  options.skip_view_graph_calibration = false;
  options.skip_relative_pose_estimation = false;
  options.skip_rotation_averaging = true;
  options.skip_track_establishment = true;
  options.skip_global_positioning = true;
  options.skip_bundle_adjustment = true;
  options.skip_retriangulation = true;
  return options;
}

RotationAveragerOptions CreateRATestOptions(bool use_gravity = false) {
  RotationAveragerOptions options;
  options.skip_initialization = false;
  options.use_gravity = use_gravity;
  options.use_stratified = true;
  return options;
}

void ExpectEqualRotations(const colmap::Reconstruction& gt,
                          const colmap::Reconstruction& computed,
                          const double max_rotation_error_deg) {
  const double max_rotation_error_rad =
      colmap::DegToRad(max_rotation_error_deg);
  const std::vector<image_t> reg_image_ids = gt.RegImageIds();
  for (size_t i = 0; i < reg_image_ids.size(); i++) {
    const image_t image_id1 = reg_image_ids[i];
    for (size_t j = 0; j < i; j++) {
      const image_t image_id2 = reg_image_ids[j];

      const Eigen::Quaterniond cam2_from_cam1 =
          computed.Image(image_id2).CamFromWorld().rotation *
          computed.Image(image_id1).CamFromWorld().rotation.inverse();
      const Eigen::Quaterniond cam2_from_cam1_gt =
          gt.Image(image_id2).CamFromWorld().rotation *
          gt.Image(image_id1).CamFromWorld().rotation.inverse();

      EXPECT_LT(cam2_from_cam1.angularDistance(cam2_from_cam1_gt),
                max_rotation_error_rad);
    }
  }
}

void ExpectEqualGravity(
    const colmap::Reconstruction& gt,
    const std::unordered_map<image_t, Image>& images_computed,
    const double max_gravity_error_deg) {
  for (const auto& image_id : gt.RegImageIds()) {
    if (!images_computed.at(image_id).HasTrivialFrame()) {
      continue;  // Skip images that are not trivial frames
    }
    const Eigen::Vector3d gravity_gt =
        gt.Image(image_id).CamFromWorld().rotation * Eigen::Vector3d(0, 1, 0);
    const Eigen::Vector3d gravity_computed =
        images_computed.at(image_id).frame_ptr->gravity_info.GetGravity();

    double gravity_error_deg = CalcAngle(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_deg, max_gravity_error_deg);
  }
}

TEST(RotationEstimator, WithoutNoise) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  PrepareGravity(gt_reconstruction, frames);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(
        view_graph, rigs, frames, images, CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithoutNoiseWithNoneTrivialKnownRig) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  PrepareGravity(gt_reconstruction, frames);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  for (const bool use_gravity : {true, false}) {
    SolveRotationAveraging(
        view_graph, rigs, frames, images, CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithoutNoiseWithNoneTrivialUnknownRig) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 20.;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        rig.ResetSensorFromRig(sensor_id);
      }
    }
  }
  PrepareGravity(gt_reconstruction, frames);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  // For unknown rigs, it is not supported to use gravity.
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(
        view_graph, rigs, frames, images, CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1e-2);
  }
}

TEST(RotationEstimator, WithNoiseAndOutliers) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  PrepareGravity(gt_reconstruction, frames, /*gravity_noise_stddev=*/3e-1);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {false}) {
    SolveRotationAveraging(
        view_graph, rigs, frames, images, CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    ExpectEqualRotations(
        gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/3);
  }
}

TEST(RotationEstimator, WithNoiseAndOutliersWithNonTrivialKnownRigs) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 1;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);
  PrepareGravity(gt_reconstruction, frames, /*gravity_noise_stddev=*/3e-1);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  // TODO: The current 1-dof rotation averaging sometimes fails to pick the
  // right solution (e.g., 180 deg flipped).
  for (const bool use_gravity : {true, false}) {
    SolveRotationAveraging(
        view_graph, rigs, frames, images, CreateRATestOptions(use_gravity));

    colmap::Reconstruction reconstruction;
    ConvertGlomapToColmap(
        rigs, cameras, frames, images, tracks, reconstruction);
    if (use_gravity)
      ExpectEqualRotations(
          gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/1.5);
    else
      ExpectEqualRotations(
          gt_reconstruction, reconstruction, /*max_rotation_error_deg=*/2.);
  }
}

TEST(RotationEstimator, RefineGravity) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  PrepareGravity(gt_reconstruction,
                 frames,
                 /*gravity_noise_stddev=*/0.,
                 /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, frames, images);

  // Check whether the gravity does not have error after refinement
  ExpectEqualGravity(gt_reconstruction,
                     images,
                     /*max_gravity_error_deg=*/1e-2);
}

TEST(RotationEstimator, RefineGravityWithNontrivialRigs) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  PrepareGravity(gt_reconstruction,
                 frames,
                 /*gravity_noise_stddev=*/0.,
                 /*outlier_ratio=*/0.3);

  GlobalMapper global_mapper(CreateMapperTestOptions());
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, frames, images);

  // Check whether the gravity does not have error after refinement
  ExpectEqualGravity(gt_reconstruction,
                     images,
                     /*max_gravity_error_deg=*/1e-2);
}

}  // namespace
}  // namespace glomap
