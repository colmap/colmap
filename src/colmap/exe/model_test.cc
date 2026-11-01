// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/model.h"

#include "colmap/geometry/gps.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ModelAligner, SkipsGravityOnlyDatabasePosePriors) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto input_path = test_dir / "input";
  const auto output_path = test_dir / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 4;
  synthetic_options.num_points3D = 20;
  auto database = Database::Open(database_path);
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  // Shift ground-truth so image 1 sits at the ENU origin (0, 0, 0), write
  // WGS84 priors for the first 3 images, and write a gravity-only (UNDEFINED,
  // NaN position) prior for the 4th image.
  const std::vector<image_t> reg_image_ids = gt_reconstruction.RegImageIds();
  ASSERT_EQ(reg_image_ids.size(), 4);
  const Eigen::Vector3d origin_shift =
      -gt_reconstruction.Image(reg_image_ids[0]).ProjectionCenter();
  gt_reconstruction.Transform(
      Sim3d(1.0, Eigen::Quaterniond::Identity(), origin_shift));

  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  constexpr double kRefLat = 47.3769;
  constexpr double kRefLon = 8.5417;
  constexpr double kRefAlt = 408.0;
  for (size_t i = 0; i < 3; ++i) {
    const Image& image = gt_reconstruction.Image(reg_image_ids[i]);
    PosePrior prior;
    prior.corr_data_id = image.DataId();
    prior.coordinate_system = PosePrior::CoordinateSystem::WGS84;
    prior.position =
        gps_transform
            .ENUToEllipsoid(
                {image.ProjectionCenter()}, kRefLat, kRefLon, kRefAlt)
            .front();
    database->WritePosePrior(prior);
  }
  PosePrior gravity_only_prior;
  gravity_only_prior.corr_data_id =
      gt_reconstruction.Image(reg_image_ids[3]).DataId();
  gravity_only_prior.gravity = Eigen::Vector3d(0.0, 1.0, 0.0);
  database->WritePosePrior(gravity_only_prior);
  database.reset();

  Reconstruction input_reconstruction = gt_reconstruction;
  input_reconstruction.Transform(Sim3d(
      0.4,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(10.0, -5.0, 2.0)));
  input_reconstruction.Write(input_path);

  const std::string database_path_str = database_path.string();
  const std::string input_path_str = input_path.string();
  const std::string output_path_str = output_path.string();
  std::vector<std::string> args = {"model_aligner",
                                   "--database_path",
                                   database_path_str,
                                   "--input_path",
                                   input_path_str,
                                   "--output_path",
                                   output_path_str,
                                   "--alignment_type",
                                   "enu",
                                   "--alignment_max_error",
                                   "1e-3"};
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EQ(RunModelAligner(static_cast<int>(argv.size()), argv.data()),
            EXIT_SUCCESS);

  Reconstruction aligned_reconstruction;
  aligned_reconstruction.Read(output_path);
  EXPECT_THAT(aligned_reconstruction,
              ReconstructionNear(gt_reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-3,
                                 /*max_scale_error=*/1e-3,
                                 /*num_obs_tolerance=*/0.0,
                                 /*align=*/false));
}

}  // namespace
}  // namespace colmap
