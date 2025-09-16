#include <colmap/estimators/bundle_adjustment.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/synthetic.h>

using namespace colmap;

int main() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  const Reconstruction orig_reconstruction = reconstruction;

  // delete observations from the two reference images
  THROW_CHECK(reconstruction.Image(1).HasTrivialFrame());
  THROW_CHECK(reconstruction.Image(3).HasTrivialFrame());
  for (point2D_t i = 0; i < reconstruction.Image(1).NumPoints2D(); ++i) {
      if (reconstruction.Image(1).Point2D(i).HasPoint3D()) {
          reconstruction.DeleteObservation(1, i);
      }
  }
  for (point2D_t i = 0; i < reconstruction.Image(3).NumPoints2D(); ++i) {
      if (reconstruction.Image(3).Point2D(i).HasPoint3D()) {
          reconstruction.DeleteObservation(3, i);
      }
  }

  BundleAdjustmentOptions options;
  BundleAdjustmentConfig config;
  config.AddImage(2);
  config.AddImage(4);

  auto ExpectValidSolve = [&options, &config, &reconstruction](
                              const int num_effective_parameters_reduced) {
    const auto summary1 =
        CreateDefaultBundleAdjuster(options, config, reconstruction)->Solve();
    THROW_CHECK_NE(summary1.termination_type, ceres::FAILURE);
    THROW_CHECK_EQ(summary1.num_effective_parameters_reduced,
              num_effective_parameters_reduced);
  };

  options.refine_rig_from_world = true;
  options.refine_sensor_from_rig = true; // this has no effect
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(316);

  options.refine_rig_from_world = false;
  config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  ExpectValidSolve(304);
  
  options.refine_sensor_from_rig = false;
  ExpectValidSolve(304);

  options.refine_rig_from_world = true;
  ExpectValidSolve(309);

  config.SetConstantRigFromWorldPose(1);
  ExpectValidSolve(309);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);

  config.SetConstantRigFromWorldPose(2);
  options.refine_rig_from_world = true;
  ExpectValidSolve(304);
  options.refine_rig_from_world = false;
  ExpectValidSolve(304);
}
