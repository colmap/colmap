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

#include "colmap/exe/sfm.h"

#include "colmap/controllers/automatic_reconstruction.h"
#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/hierarchical_mapper.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/exe/gui.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {

int RunAutomaticReconstructor(int argc, char** argv) {
  AutomaticReconstructionController::Options reconstruction_options;
  std::string data_type = "individual";
  std::string quality = "high";
  std::string mesher = "poisson";

  OptionManager options;
  options.AddRequiredOption("workspace_path",
                            &reconstruction_options.workspace_path);
  options.AddRequiredOption("image_path", &reconstruction_options.image_path);
  options.AddDefaultOption("mask_path", &reconstruction_options.mask_path);
  options.AddDefaultOption("vocab_tree_path",
                           &reconstruction_options.vocab_tree_path);
  options.AddDefaultOption(
      "data_type", &data_type, "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
  options.AddDefaultOption("camera_params",
                           &reconstruction_options.camera_params);
  options.AddDefaultOption("sparse", &reconstruction_options.sparse);
  options.AddDefaultOption("dense", &reconstruction_options.dense);
  options.AddDefaultOption("mesher", &mesher, "{poisson, delaunay}");
  options.AddDefaultOption("num_threads", &reconstruction_options.num_threads);
  options.AddDefaultOption("use_gpu", &reconstruction_options.use_gpu);
  options.AddDefaultOption("gpu_index", &reconstruction_options.gpu_index);
  options.Parse(argc, argv);

  StringToLower(&data_type);
  if (data_type == "individual") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INDIVIDUAL;
  } else if (data_type == "video") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::VIDEO;
  } else if (data_type == "internet") {
    reconstruction_options.data_type =
        AutomaticReconstructionController::DataType::INTERNET;
  } else {
    LOG(FATAL_THROW) << "Invalid data type provided";
  }

  StringToLower(&quality);
  if (quality == "low") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::LOW;
  } else if (quality == "medium") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::MEDIUM;
  } else if (quality == "high") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::HIGH;
  } else if (quality == "extreme") {
    reconstruction_options.quality =
        AutomaticReconstructionController::Quality::EXTREME;
  } else {
    LOG(FATAL_THROW) << "Invalid quality provided";
  }

  StringToLower(&mesher);
  if (mesher == "poisson") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::POISSON;
  } else if (mesher == "delaunay") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::DELAUNAY;
  } else {
    LOG(FATAL_THROW) << "Invalid mesher provided";
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  if (reconstruction_options.use_gpu && kUseOpenGL) {
    QApplication app(argc, argv);
    AutomaticReconstructionController controller(reconstruction_options,
                                                 reconstruction_manager);
    RunThreadWithOpenGLContext(&controller);
  } else {
    AutomaticReconstructionController controller(reconstruction_options,
                                                 reconstruction_manager);
    controller.Start();
    controller.Wait();
  }

  return EXIT_SUCCESS;
}

int RunBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  BundleAdjustmentController ba_controller(options, reconstruction);
  ba_controller.Run();

  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunColorExtractor(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  reconstruction.ExtractColorsForAllImages(*options.image_path);
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunMapper(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    const auto image_names = ReadTextFileLines(image_list_path);
    options.mapper->image_names =
        std::unordered_set<std::string>(image_names.begin(), image_names.end());
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      LOG(ERROR) << "`input_path` is not a directory.";
      return EXIT_FAILURE;
    }
    reconstruction_manager->Read(input_path);
  }

  // If fix_existing_images is enabled, we store the initial positions of
  // existing images in order to transform them back to the original coordinate
  // frame, as the reconstruction is normalized multiple times for numerical
  // stability.
  std::vector<Eigen::Vector3d> orig_fixed_image_positions;
  std::vector<image_t> fixed_image_ids;
  if (options.mapper->fix_existing_images &&
      reconstruction_manager->Size() > 0) {
    const auto& reconstruction = reconstruction_manager->Get(0);
    fixed_image_ids = reconstruction->RegImageIds();
    orig_fixed_image_positions.reserve(fixed_image_ids.size());
    for (const image_t image_id : fixed_image_ids) {
      orig_fixed_image_positions.push_back(
          reconstruction->Image(image_id).ProjectionCenter());
    }
  }

  IncrementalMapperController mapper(options.mapper,
                                     *options.image_path,
                                     *options.database_path,
                                     reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (input_path == "") {
    mapper.AddCallback(
        IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [&]() {
          // If the number of reconstructions has not changed, the last model
          // was discarded for some reason.
          if (reconstruction_manager->Size() > prev_num_reconstructions) {
            const std::string reconstruction_path = JoinPaths(
                output_path, std::to_string(prev_num_reconstructions));
            CreateDirIfNotExists(reconstruction_path);
            reconstruction_manager->Get(prev_num_reconstructions)
                ->Write(reconstruction_path);
            options.Write(JoinPaths(reconstruction_path, "project.ini"));
            prev_num_reconstructions = reconstruction_manager->Size();
          }
        });
  }

  mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (input_path != "") {
    const auto& reconstruction = reconstruction_manager->Get(0);

    // Transform the final reconstruction back to the original coordinate frame.
    if (options.mapper->fix_existing_images) {
      if (fixed_image_ids.size() < 3) {
        LOG(WARNING) << "Too few images to transform the reconstruction.";
      } else {
        std::vector<Eigen::Vector3d> new_fixed_image_positions;
        new_fixed_image_positions.reserve(fixed_image_ids.size());
        for (const image_t image_id : fixed_image_ids) {
          new_fixed_image_positions.push_back(
              reconstruction->Image(image_id).ProjectionCenter());
        }
        Sim3d orig_from_new;
        if (EstimateSim3d(new_fixed_image_positions,
                          orig_fixed_image_positions,
                          orig_from_new)) {
          reconstruction->Transform(orig_from_new);
        } else {
          LOG(WARNING) << "Failed to transform the reconstruction back "
                          "to the input coordinate frame.";
        }
      }
    }

    reconstruction->Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalMapperController::Options mapper_options;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &mapper_options.database_path);
  options.AddRequiredOption("image_path", &mapper_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_workers", &mapper_options.num_workers);
  options.AddDefaultOption("image_overlap",
                           &mapper_options.clustering_options.image_overlap);
  options.AddDefaultOption(
      "leaf_max_num_images",
      &mapper_options.clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory.";
    return EXIT_FAILURE;
  }

  mapper_options.incremental_options = *options.mapper;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalMapperController hierarchical_mapper(mapper_options,
                                                   reconstruction_manager);
  hierarchical_mapper.Run();

  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "failed to create sparse model";
    return EXIT_FAILURE;
  }

  reconstruction_manager->Write(output_path);
  options.Write(JoinPaths(output_path, "project.ini"));

  return EXIT_SUCCESS;
}

int RunPointFiltering(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  size_t min_track_len = 2;
  double max_reproj_error = 4.0;
  double min_tri_angle = 1.5;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_track_len", &min_track_len);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.AddDefaultOption("min_tri_angle", &min_tri_angle);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  size_t num_filtered =
      reconstruction.FilterAllPoints3D(max_reproj_error, min_tri_angle);

  for (const auto point3D_id : reconstruction.Point3DIds()) {
    const auto& point3D = reconstruction.Point3D(point3D_id);
    if (point3D.track.Length() < min_track_len) {
      num_filtered += point3D.track.Length();
      reconstruction.DeletePoint3D(point3D_id);
    }
  }

  LOG(INFO) << "Filtered observations: " << num_filtered;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunPointTriangulator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  bool clear_points = true;
  bool refine_intrinsics = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points",
      &clear_points,
      "Whether to clear all existing points and observations and recompute "
      "the image_ids based on matching filenames between the model and the "
      "database");
  options.AddDefaultOption("refine_intrinsics",
                           &refine_intrinsics,
                           "Whether to refine the intrinsics of the cameras "
                           "(fixing the principal point)");
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  PrintHeading1("Loading model");

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  RunPointTriangulatorImpl(reconstruction,
                           *options.database_path,
                           *options.image_path,
                           output_path,
                           *options.mapper,
                           clear_points,
                           refine_intrinsics);
  return EXIT_SUCCESS;
}

void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const IncrementalMapperOptions& options,
    const bool clear_points,
    const bool refine_intrinsics) {
  THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
      << "Need at least two images for triangulation";
  if (clear_points) {
    const Database database(database_path);
    reconstruction->DeleteAllPoints2DAndPoints3D();
    reconstruction->TranscribeImageIdsToDatabase(database);
  }

  auto options_tmp = std::make_shared<IncrementalMapperOptions>(options);
  options_tmp->fix_existing_images = true;
  options_tmp->ba_refine_focal_length = refine_intrinsics;
  options_tmp->ba_refine_principal_point = false;
  options_tmp->ba_refine_extra_params = refine_intrinsics;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      options_tmp, image_path, database_path, reconstruction_manager);
  mapper.TriangulateReconstruction(reconstruction);
  reconstruction->Write(output_path);
}

namespace {

// Read the configuration of the camera rigs from a JSON file. The input images
// of a camera rig must be named consistently to assign them to the appropriate
// camera rig and the respective snapshots.
//
// An example configuration of a single camera rig:
// [
//   {
//     "ref_camera_id": 1,
//     "cameras":
//     [
//       {
//           "camera_id": 1,
//           "image_prefix": "left1_image"
//           "cam_from_rig_rotation": [1, 0, 0, 0],
//           "cam_from_rig_translation": [0, 0, 0]
//       },
//       {
//           "camera_id": 2,
//           "image_prefix": "left2_image"
//           "cam_from_rig_rotation": [1, 0, 0, 0],
//           "cam_from_rig_translation": [0, 0, 1]
//       },
//       {
//           "camera_id": 3,
//           "image_prefix": "right1_image"
//           "cam_from_rig_rotation": [1, 0, 0, 0],
//           "cam_from_rig_translation": [0, 0, 2]
//       },
//       {
//           "camera_id": 4,
//           "image_prefix": "right2_image"
//           "cam_from_rig_rotation": [1, 0, 0, 0],
//           "cam_from_rig_translation": [0, 0, 3]
//       }
//     ]
//   }
// ]
//
// The "camera_id" and "image_prefix" fields are required, whereas the
// "cam_from_rig_rotation" and "cam_from_rig_translation" fields optionally
// specify the relative extrinsics of the camera rig in the form of a
// translation vector and a rotation quaternion (w, x, y, z). If the relative
// extrinsics are not provided then they are automatically inferred from the
// reconstruction.
//
// This file specifies the configuration for a single camera rig and that you
// could potentially define multiple camera rigs. The rig is composed of 4
// cameras: all images of the first camera must have "left1_image" as a name
// prefix, e.g., "left1_image_frame000.png" or "left1_image/frame000.png".
// Images with the same suffix ("_frame000.png" and "/frame000.png") are
// assigned to the same snapshot, i.e., they are assumed to be captured at the
// same time. Only snapshots with the reference image registered will be added
// to the bundle adjustment problem. The remaining images will be added with
// independent poses to the bundle adjustment problem. The above configuration
// could have the following input image file structure:
//
//    /path/to/images/...
//        left1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        left2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right1_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//        right2_image/...
//            frame000.png
//            frame001.png
//            frame002.png
//            ...
//
std::vector<CameraRig> ReadCameraRigConfig(const std::string& rig_config_path,
                                           const Reconstruction& reconstruction,
                                           bool estimate_rig_relative_poses) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

  std::vector<CameraRig> camera_rigs;
  for (const auto& rig_config : pt) {
    CameraRig camera_rig;

    std::vector<std::string> image_prefixes;
    for (const auto& camera : rig_config.second.get_child("cameras")) {
      const int camera_id = camera.second.get<int>("camera_id");
      image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));

      Rigid3d cam_from_rig;

      auto cam_from_rig_rotation_node =
          camera.second.get_child_optional("cam_from_rig_rotation");
      if (cam_from_rig_rotation_node) {
        int index = 0;
        Eigen::Vector4d cam_from_rig_wxyz;
        for (const auto& node : cam_from_rig_rotation_node.get()) {
          cam_from_rig_wxyz[index++] = node.second.get_value<double>();
        }
        cam_from_rig.rotation = Eigen::Quaterniond(cam_from_rig_wxyz(0),
                                                   cam_from_rig_wxyz(1),
                                                   cam_from_rig_wxyz(2),
                                                   cam_from_rig_wxyz(3));
      } else {
        estimate_rig_relative_poses = true;
      }

      auto cam_from_rig_translation_node =
          camera.second.get_child_optional("cam_from_rig_translation");
      if (cam_from_rig_translation_node) {
        int index = 0;
        for (const auto& node : cam_from_rig_translation_node.get()) {
          cam_from_rig.translation(index++) = node.second.get_value<double>();
        }
      } else {
        estimate_rig_relative_poses = true;
      }

      camera_rig.AddCamera(camera_id, cam_from_rig);
    }

    camera_rig.SetRefCameraId(rig_config.second.get<int>("ref_camera_id"));

    std::unordered_map<std::string, std::vector<image_t>> snapshots;
    for (const auto image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      for (const auto& image_prefix : image_prefixes) {
        if (StringContains(image.Name(), image_prefix)) {
          const std::string image_suffix =
              StringGetAfter(image.Name(), image_prefix);
          snapshots[image_suffix].push_back(image_id);
        }
      }
    }

    for (const auto& snapshot : snapshots) {
      bool has_ref_camera = false;
      for (const auto image_id : snapshot.second) {
        const auto& image = reconstruction.Image(image_id);
        if (image.CameraId() == camera_rig.RefCameraId()) {
          has_ref_camera = true;
          break;
        }
      }

      if (has_ref_camera) {
        camera_rig.AddSnapshot(snapshot.second);
      }
    }

    camera_rig.Check(reconstruction);
    if (estimate_rig_relative_poses) {
      PrintHeading2("Estimating relative rig poses");
      if (!camera_rig.ComputeCamsFromRigs(reconstruction)) {
        LOG(WARNING) << "Failed to estimate rig poses from reconstruction; "
                        "cannot use rig BA";
        return std::vector<CameraRig>();
      }
    }

    camera_rigs.push_back(camera_rig);
  }

  return camera_rigs;
}

}  // namespace

int RunRigBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string rig_config_path;
  bool estimate_rig_relative_poses = true;

  RigBundleAdjuster::Options rig_ba_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.AddDefaultOption("estimate_rig_relative_poses",
                           &estimate_rig_relative_poses);
  options.AddDefaultOption("RigBundleAdjustment.refine_relative_poses",
                           &rig_ba_options.refine_relative_poses);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Camera rig configuration");

  auto camera_rigs = ReadCameraRigConfig(
      rig_config_path, reconstruction, estimate_rig_relative_poses);

  BundleAdjustmentConfig config;
  for (size_t i = 0; i < camera_rigs.size(); ++i) {
    const auto& camera_rig = camera_rigs[i];
    PrintHeading2(StringPrintf("Camera Rig %d", i + 1));
    LOG(INFO) << StringPrintf("Cameras: %d", camera_rig.NumCameras());
    LOG(INFO) << StringPrintf("Snapshots: %d", camera_rig.NumSnapshots());

    // Add all registered images to the bundle adjustment configuration.
    for (const auto image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
  }

  PrintHeading1("Rig bundle adjustment");

  BundleAdjustmentOptions ba_options = *options.bundle_adjustment;
  RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
  THROW_CHECK(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
