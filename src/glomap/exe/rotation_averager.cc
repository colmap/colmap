#include "glomap/sfm/rotation_averager.h"

#include "colmap/geometry/pose.h"
#include "colmap/scene/reconstruction_io_utils.h"
#include "colmap/util/file.h"
#include "colmap/util/timer.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/estimators/gravity_refinement.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/pose_io.h"

namespace glomap {
// -------------------------------------
// Running Global Rotation Averager
// -------------------------------------
int RunRotationAverager(int argc, char** argv) {
  std::string relpose_path;
  std::string output_path;
  std::string gravity_path = "";
  std::string weight_path = "";

  bool use_stratified = true;
  bool refine_gravity = false;
  bool use_weight = false;

  OptionManager options;
  options.AddRequiredOption("relpose_path", &relpose_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("gravity_path", &gravity_path);
  options.AddDefaultOption("weight_path", &weight_path);
  options.AddDefaultOption("use_stratified", &use_stratified);
  options.AddDefaultOption("refine_gravity", &refine_gravity);
  options.AddDefaultOption("use_weight", &use_weight);
  options.AddGravityRefinerOptions();
  options.Parse(argc, argv);

  if (!colmap::ExistsFile(relpose_path)) {
    LOG(ERROR) << "`relpose_path` is not a file";
    return EXIT_FAILURE;
  }

  if (gravity_path != "" && !colmap::ExistsFile(gravity_path)) {
    LOG(ERROR) << "`gravity_path` is not a file";
    return EXIT_FAILURE;
  }

  if (weight_path != "" && !colmap::ExistsFile(weight_path)) {
    LOG(ERROR) << "`weight_path` is not a file";
    return EXIT_FAILURE;
  }

  if (use_weight && weight_path == "") {
    LOG(ERROR) << "Weight path is required when use_weight is set to true";
    return EXIT_FAILURE;
  }

  RotationEstimatorOptions rotation_averager_options;
  rotation_averager_options.skip_initialization = true;
  rotation_averager_options.use_gravity = true;
  rotation_averager_options.use_stratified = use_stratified;
  rotation_averager_options.use_weight = use_weight;

  // Load the database
  ViewGraph view_graph;
  colmap::Reconstruction reconstruction;

  // Read relative poses and build view graph
  // First read into a temporary images map
  std::unordered_map<image_t, Image> temp_images;
  ReadRelPose(relpose_path, temp_images, view_graph);

  // Add cameras and images to reconstruction
  for (auto& [image_id, image] : temp_images) {
    image.SetCameraId(image.ImageId());

    // Add camera if it doesn't exist
    if (!reconstruction.ExistsCamera(image.CameraId())) {
      colmap::Camera camera;
      camera.camera_id = image.CameraId();
      reconstruction.AddCamera(std::move(camera));
    }

    reconstruction.AddImage(std::move(image));
  }

  // Create one rig per camera and frames for images
  colmap::CreateOneRigPerCamera(reconstruction);
  for (const auto& [image_id, image] : reconstruction.Images()) {
    colmap::CreateFrameForImage(image, Rigid3d(), reconstruction);
  }

  std::vector<colmap::PosePrior> pose_priors;
  if (gravity_path != "") {
    pose_priors = ReadGravity(gravity_path, reconstruction.Images());
    // Initialize frame rotations from gravity.
    // Currently rotation averaging only supports gravity prior on reference
    // sensors.
    for (const auto& pose_prior : pose_priors) {
      const auto& image = reconstruction.Image(pose_prior.pose_prior_id);
      if (!image.IsRefInFrame()) {
        continue;
      }
      Rigid3d& rig_from_world =
          reconstruction.Frame(image.FrameId()).RigFromWorld();
      rig_from_world.rotation = Eigen::Quaterniond(
          colmap::GravityAlignedRotation(pose_prior.gravity));
    }
  }

  if (use_weight) {
    ReadRelWeight(weight_path, reconstruction.Images(), view_graph);
  }

  int num_img = view_graph.KeepLargestConnectedComponents(reconstruction);
  LOG(INFO) << num_img << " / " << reconstruction.NumImages()
            << " are within the largest connected component";

  if (refine_gravity && gravity_path != "") {
    GravityRefiner grav_refiner(*options.gravity_refiner);
    grav_refiner.RefineGravity(view_graph, reconstruction, pose_priors);
  }

  colmap::Timer run_timer;
  run_timer.Start();
  if (!SolveRotationAveraging(
          view_graph, reconstruction, pose_priors, rotation_averager_options)) {
    LOG(ERROR) << "Failed to solve global rotation averaging";
    return EXIT_FAILURE;
  }
  run_timer.Pause();
  LOG(INFO) << "Global rotation averaging done in "
            << run_timer.ElapsedSeconds() << " seconds";

  // Write out the estimated rotation
  WriteGlobalRotation(output_path, reconstruction.Images());
  LOG(INFO) << "Global rotation averaging done" << '\n';

  return EXIT_SUCCESS;
}

}  // namespace glomap
