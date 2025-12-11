#include "glomap/sfm/rotation_averager.h"

#include "colmap/util/file.h"
#include "colmap/util/timer.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/estimators/gravity_refinement.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/pose_io.h"
#include "glomap/types.h"

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

  RotationAveragerOptions rotation_averager_options;
  rotation_averager_options.skip_initialization = true;
  rotation_averager_options.use_gravity = true;

  rotation_averager_options.use_stratified = use_stratified;
  rotation_averager_options.use_weight = use_weight;

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<image_t, Image> images;

  ReadRelPose(relpose_path, images, view_graph);

  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, colmap::Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;

  for (auto& [image_id, image] : images) {
    image.camera_id = image.image_id;
    cameras[image.camera_id] = colmap::Camera();
  }

  CreateOneRigPerCamera(cameras, rigs);

  // For frames that are not in any rig, add camera rigs
  // For images without frames, initialize trivial frames
  for (auto& [image_id, image] : images) {
    CreateFrameForImage(Rigid3d(), image, rigs, frames);
  }

  if (gravity_path != "") {
    ReadGravity(gravity_path, images);
  }

  if (use_weight) {
    ReadRelWeight(weight_path, images, view_graph);
  }

  int num_img = view_graph.KeepLargestConnectedComponents(frames, images);
  LOG(INFO) << num_img << " / " << images.size()
            << " are within the largest connected component";

  if (refine_gravity && gravity_path != "") {
    GravityRefiner grav_refiner(*options.gravity_refiner);
    grav_refiner.RefineGravity(view_graph, frames, images);
  }

  colmap::Timer run_timer;
  run_timer.Start();
  if (!SolveRotationAveraging(
          view_graph, rigs, frames, images, rotation_averager_options)) {
    LOG(ERROR) << "Failed to solve global rotation averaging";
    return EXIT_FAILURE;
  }
  run_timer.Pause();
  LOG(INFO) << "Global rotation averaging done in "
            << run_timer.ElapsedSeconds() << " seconds";

  // Write out the estimated rotation
  WriteGlobalRotation(output_path, images);
  LOG(INFO) << "Global rotation averaging done" << '\n';

  return EXIT_SUCCESS;
}

}  // namespace glomap
