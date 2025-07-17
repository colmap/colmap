#include "colmap/controllers/incremental_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"
#include <iostream>

using namespace colmap;

void ExpectEqualReconstructions(const Reconstruction& gt,
                                const Reconstruction& computed,
                                const double max_rotation_error_deg,
                                const double max_proj_center_error,
                                const double num_obs_tolerance,
                                const bool align = true) {
  THROW_CHECK_EQ(computed.NumCameras(), gt.NumCameras());
  THROW_CHECK_EQ(computed.NumImages(), gt.NumImages());
  THROW_CHECK_EQ(computed.NumRegImages(), gt.NumRegImages());
  THROW_CHECK_GE(computed.ComputeNumObservations(),
            (1 - num_obs_tolerance) * gt.ComputeNumObservations());

  Sim3d gt_from_computed;
  if (align) {
      AlignReconstructionsViaProjCenters(computed,
                                           gt,
                                           /*max_proj_center_error=*/0.1,
                                           &gt_from_computed);
  }

  const std::vector<ImageAlignmentError> errors =
      ComputeImageAlignmentError(computed, gt, gt_from_computed);
  THROW_CHECK_EQ(errors.size(), gt.NumImages());
  for (const auto& error : errors) {
    THROW_CHECK_LT(error.rotation_error_deg, max_rotation_error_deg);
    THROW_CHECK_LT(error.proj_center_error, max_proj_center_error);
  }
}


int main() {
  const std::string test_dir = "/tmp/colmap_test_data/IncrementalPipeline.PriorBasedSfMWithoutNoiseAndWithNonTrivialFrames/";
  const std::string database_path = test_dir + "database.db";
  Database database(database_path);
  Reconstruction gt_reconstruction;
  gt_reconstruction.Read(test_dir);

  // SyntheticDatasetOptions synthetic_dataset_options;
  // synthetic_dataset_options.num_rigs = 2;
  // synthetic_dataset_options.num_cameras_per_rig = 2;
  // synthetic_dataset_options.num_frames_per_rig = 7;
  // synthetic_dataset_options.num_points3D = 100;
  // synthetic_dataset_options.point2D_stddev = 0;
  // synthetic_dataset_options.camera_has_prior_focal_length = false;

  // synthetic_dataset_options.use_prior_position = true;
  // synthetic_dataset_options.prior_position_stddev = 0.0;
  // SynthesizeDataset(synthetic_dataset_options, &gt_reconstruction, &database);
  // gt_reconstruction.Write(test_dir);

  std::shared_ptr<IncrementalPipelineOptions> mapper_options =
      std::make_shared<IncrementalPipelineOptions>();

  mapper_options->use_prior_position = true;
  mapper_options->use_robust_loss_on_prior_position = true;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(mapper_options,
                             /*image_path=*/"",
                             database_path,
                             reconstruction_manager);
  mapper.Run();
  std::cout<<"bingo completed"<<std::endl;

  THROW_CHECK_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-1,
                             /*max_proj_center_error=*/1e-1,
                             /*num_obs_tolerance=*/0.02,
                             /*align=*/true);
}

