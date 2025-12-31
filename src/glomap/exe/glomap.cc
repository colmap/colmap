// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "glomap/exe/glomap.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/geometry/pose.h"
#include "colmap/scene/reconstruction_io_utils.h"
#include "colmap/util/file.h"
#include "colmap/util/timer.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/estimators/gravity_refinement.h"
#include "glomap/estimators/rotation_averaging.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/pose_io.h"
#include "glomap/sfm/global_mapper.h"

namespace glomap {

// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunGlobalMapper(int argc, char** argv) {
  std::string output_path;
  std::string output_format = "bin";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperOptions();

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  auto database = colmap::Database::Open(*options.database_path);

  auto reconstruction = std::make_shared<colmap::Reconstruction>();

  GlobalMapper global_mapper(database);
  global_mapper.BeginReconstruction(reconstruction);

  if (global_mapper.ViewGraph()->Empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  options.mapper->image_path = *options.image_path;

  // Main solver
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;
  run_timer.Start();
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(*options.mapper, cluster_ids);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteReconstructionsByClusters(output_path,
                                 *reconstruction,
                                 cluster_ids,
                                 output_format,
                                 *options.image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Running Global Rotation Averager
// -------------------------------------
int RunRotationAverager(int argc, char** argv) {
  std::string relpose_path;
  std::string output_path;
  std::string gravity_path = "";

  bool use_stratified = true;
  bool refine_gravity = false;

  OptionManager options;
  options.AddRequiredOption("relpose_path", &relpose_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("gravity_path", &gravity_path);
  options.AddDefaultOption("use_stratified", &use_stratified);
  options.AddDefaultOption("refine_gravity", &refine_gravity);
  options.AddGravityRefinerOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!colmap::ExistsFile(relpose_path)) {
    LOG(ERROR) << "`relpose_path` is not a file";
    return EXIT_FAILURE;
  }

  if (gravity_path != "" && !colmap::ExistsFile(gravity_path)) {
    LOG(ERROR) << "`gravity_path` is not a file";
    return EXIT_FAILURE;
  }

  RotationEstimatorOptions rotation_averager_options;
  rotation_averager_options.skip_initialization = true;
  rotation_averager_options.use_gravity = true;
  rotation_averager_options.use_stratified = use_stratified;

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
          rotation_averager_options, view_graph, reconstruction, pose_priors)) {
    LOG(ERROR) << "Failed to solve global rotation averaging";
    return EXIT_FAILURE;
  }
  run_timer.Pause();
  LOG(INFO) << "Global rotation averaging done in "
            << run_timer.ElapsedSeconds() << " seconds";

  // Write out the estimated rotation
  WriteGlobalRotation(output_path, reconstruction.Images());
  LOG(INFO) << "Global rotation averaging done";

  return EXIT_SUCCESS;
}

}  // namespace glomap
