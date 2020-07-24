// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "base/similarity_transform.h"
#include "controllers/automatic_reconstruction.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/hierarchical_mapper.h"
#include "estimators/coordinate_frame.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "feature/utils.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "retrieval/visual_index.h"
#include "ui/main_window.h"
#include "util/opengl_utils.h"
#include "util/version.h"

using namespace colmap;

#ifdef CUDA_ENABLED
const bool kUseOpenGL = false;
#else
const bool kUseOpenGL = true;
#endif

int RunGraphicalUserInterface(int argc, char** argv) {
  OptionManager options;

  std::string import_path;

  if (argc > 1) {
    options.AddDefaultOption("import_path", &import_path);
    options.AddAllOptions();
    options.Parse(argc, argv);
  }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 6, 0))
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

  Q_INIT_RESOURCE(resources);

  QApplication app(argc, argv);

  MainWindow main_window(options);
  main_window.show();

  if (!import_path.empty()) {
    main_window.ImportReconstruction(import_path);
  }

  return app.exec();
}

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
  options.AddDefaultOption("data_type", &data_type,
                           "{individual, video, internet}");
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.AddDefaultOption("camera_model",
                           &reconstruction_options.camera_model);
  options.AddDefaultOption("single_camera",
                           &reconstruction_options.single_camera);
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
    LOG(FATAL) << "Invalid data type provided";
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
    LOG(FATAL) << "Invalid quality provided";
  }

  StringToLower(&mesher);
  if (mesher == "poisson") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::POISSON;
  } else if (mesher == "delaunay") {
    reconstruction_options.mesher =
        AutomaticReconstructionController::Mesher::DELAUNAY;
  } else {
    LOG(FATAL) << "Invalid mesher provided";
  }

  ReconstructionManager reconstruction_manager;

  if (reconstruction_options.use_gpu && kUseOpenGL) {
    QApplication app(argc, argv);
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
    RunThreadWithOpenGLContext(&controller);
  } else {
    AutomaticReconstructionController controller(reconstruction_options,
                                                 &reconstruction_manager);
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
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  reconstruction.Write(output_path);

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

int RunDatabaseCreator(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.Parse(argc, argv);

  Database database(*options.database_path);

  return EXIT_SUCCESS;
}

int RunDatabaseMerger(int argc, char** argv) {
  std::string database_path1;
  std::string database_path2;
  std::string merged_database_path;

  OptionManager options;
  options.AddRequiredOption("database_path1", &database_path1);
  options.AddRequiredOption("database_path2", &database_path2);
  options.AddRequiredOption("merged_database_path", &merged_database_path);
  options.Parse(argc, argv);

  if (ExistsFile(merged_database_path)) {
    std::cout << "ERROR: Merged database file must not exist." << std::endl;
    return EXIT_FAILURE;
  }

  Database database1(database_path1);
  Database database2(database_path2);
  Database merged_database(merged_database_path);
  Database::Merge(database1, database2, &merged_database);

  return EXIT_SUCCESS;
}

int RunStereoFuser(int argc, char** argv) {
  std::string workspace_path;
  std::string input_type = "geometric";
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddDefaultOption("workspace_format", &workspace_format,
                           "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDefaultOption("input_type", &input_type,
                           "{photometric, geometric}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddStereoFusionOptions();
  options.Parse(argc, argv);

  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    std::cout << "ERROR: Invalid `workspace_format` - supported values are "
                 "'COLMAP' or 'PMVS'."
              << std::endl;
    return EXIT_FAILURE;
  }

  StringToLower(&input_type);
  if (input_type != "photometric" && input_type != "geometric") {
    std::cout << "ERROR: Invalid input type - supported values are "
                 "'photometric' and 'geometric'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::StereoFusion fuser(*options.stereo_fusion, workspace_path,
                          workspace_format, pmvs_option_name, input_type);

  fuser.Start();
  fuser.Wait();

  std::cout << "Writing output: " << output_path << std::endl;
  WriteBinaryPlyPoints(output_path, fuser.GetFusedPoints());
  mvs::WritePointsVisibility(output_path + ".vis",
                             fuser.GetFusedPointsVisibility());

  return EXIT_SUCCESS;
}

int RunPoissonMesher(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddPoissonMeshingOptions();
  options.Parse(argc, argv);

  CHECK(mvs::PoissonMeshing(*options.poisson_meshing, input_path, output_path));

  return EXIT_SUCCESS;
}

int RunProjectGenerator(int argc, char** argv) {
  std::string output_path;
  std::string quality = "high";

  OptionManager options;
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("quality", &quality, "{low, medium, high, extreme}");
  options.Parse(argc, argv);

  OptionManager output_options;
  output_options.AddAllOptions();

  StringToLower(&quality);
  if (quality == "low") {
    output_options.ModifyForLowQuality();
  } else if (quality == "medium") {
    output_options.ModifyForMediumQuality();
  } else if (quality == "high") {
    output_options.ModifyForHighQuality();
  } else if (quality == "extreme") {
    output_options.ModifyForExtremeQuality();
  } else {
    LOG(FATAL) << "Invalid quality provided";
  }

  output_options.Write(output_path);

  return EXIT_SUCCESS;
}

int RunDelaunayMesher(int argc, char** argv) {
#ifndef CGAL_ENABLED
  std::cerr << "ERROR: Delaunay meshing requires CGAL, which is not "
               "available on your system."
            << std::endl;
  return EXIT_FAILURE;
#else   // CGAL_ENABLED
  std::string input_path;
  std::string input_type = "dense";
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption(
      "input_path", &input_path,
      "Path to either the dense workspace folder or the sparse reconstruction");
  options.AddDefaultOption("input_type", &input_type, "{dense, sparse}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddDelaunayMeshingOptions();
  options.Parse(argc, argv);

  StringToLower(&input_type);
  if (input_type == "sparse") {
    mvs::SparseDelaunayMeshing(*options.delaunay_meshing, input_path,
                               output_path);
  } else if (input_type == "dense") {
    mvs::DenseDelaunayMeshing(*options.delaunay_meshing, input_path,
                              output_path);
  } else {
    std::cout << "ERROR: Invalid input type - "
                 "supported values are 'sparse' and 'dense'."
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif  // CGAL_ENABLED
}

int RunPatchMatchStereo(int argc, char** argv) {
#ifndef CUDA_ENABLED
  std::cerr << "ERROR: Dense stereo reconstruction requires CUDA, which is not "
               "available on your system."
            << std::endl;
  return EXIT_FAILURE;
#else   // CUDA_ENABLED
  std::string workspace_path;
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";

  OptionManager options;
  options.AddRequiredOption(
      "workspace_path", &workspace_path,
      "Path to the folder containing the undistorted images");
  options.AddDefaultOption("workspace_format", &workspace_format,
                           "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddPatchMatchStereoOptions();
  options.Parse(argc, argv);

  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    std::cout << "ERROR: Invalid `workspace_format` - supported values are "
                 "'COLMAP' or 'PMVS'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::PatchMatchController controller(*options.patch_match_stereo,
                                       workspace_path, workspace_format,
                                       pmvs_option_name);

  controller.Start();
  controller.Wait();

  return EXIT_SUCCESS;
#endif  // CUDA_ENABLED
}

int RunExhaustiveMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddExhaustiveMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  ExhaustiveFeatureMatcher feature_matcher(*options.exhaustive_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params) {
  if (!ExistsCameraModelWithName(camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
    return false;
  }

  const std::vector<double> camera_params = CSVToVector<double>(params);
  const int camera_model_id = CameraModelNameToId(camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return false;
  }
  return true;
}

int RunFeatureExtractor(int argc, char** argv) {
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddExtractionOptions();
  options.Parse(argc, argv);

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!ExistsCameraModelWithName(options.image_reader->camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
  }

  if (!VerifyCameraParams(options.image_reader->camera_model,
                          options.image_reader->camera_params)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.sift_extraction->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  SiftFeatureExtractor feature_extractor(reader_options,
                                         *options.sift_extraction);

  if (options.sift_extraction->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_extractor);
  } else {
    feature_extractor.Start();
    feature_extractor.Wait();
  }

  return EXIT_SUCCESS;
}

int RunFeatureImporter(int argc, char** argv) {
  std::string import_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("import_path", &import_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddExtractionOptions();
  options.Parse(argc, argv);

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!VerifyCameraParams(options.image_reader->camera_model,
                          options.image_reader->camera_params)) {
    return EXIT_FAILURE;
  }

  FeatureImporter feature_importer(reader_options, import_path);
  feature_importer.Start();
  feature_importer.Wait();

  return EXIT_SUCCESS;
}

// Read stereo image pair names from a text file. The text file is expected to
// have one image pair per line, e.g.:
//
//      image_name1.jpg image_name2.jpg
//      image_name3.jpg image_name4.jpg
//      image_name5.jpg image_name6.jpg
//      ...
//
std::vector<std::pair<image_t, image_t>> ReadStereoImagePairs(
    const std::string& path, const Reconstruction& reconstruction) {
  const std::vector<std::string> stereo_pair_lines = ReadTextFileLines(path);

  std::vector<std::pair<image_t, image_t>> stereo_pairs;
  stereo_pairs.reserve(stereo_pair_lines.size());

  for (const auto& line : stereo_pair_lines) {
    const std::vector<std::string> names = StringSplit(line, " ");
    CHECK_EQ(names.size(), 2);

    const Image* image1 = reconstruction.FindImageWithName(names[0]);
    const Image* image2 = reconstruction.FindImageWithName(names[1]);

    CHECK_NOTNULL(image1);
    CHECK_NOTNULL(image2);

    stereo_pairs.emplace_back(image1->ImageId(), image2->ImageId());
  }

  return stereo_pairs;
}

int RunImageDeleter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_ids_path;
  std::string image_names_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "image_ids_path", &image_ids_path,
      "Path to text file containing one image_id to delete per line");
  options.AddDefaultOption(
      "image_names_path", &image_names_path,
      "Path to text file containing one image name to delete per line");
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  if (!image_ids_path.empty()) {
    const auto image_ids = ReadTextFileLines(image_ids_path);

    for (const auto image_id_str : image_ids) {
      if (image_id_str.empty()) {
        continue;
      }

      const image_t image_id = std::stoi(image_id_str);
      if (reconstruction.ExistsImage(image_id)) {
        const auto& image = reconstruction.Image(image_id);
        std::cout
            << StringPrintf(
                   "Deleting image_id=%d, image_name=%s from reconstruction",
                   image.ImageId(), image.Name().c_str())
            << std::endl;
        reconstruction.DeRegisterImage(image_id);
      } else {
        std::cout << StringPrintf(
                         "WARNING: Skipping image_id=%s, because it does not "
                         "exist in the reconstruction",
                         image_id_str.c_str())
                  << std::endl;
      }
    }
  }

  if (!image_names_path.empty()) {
    const auto image_names = ReadTextFileLines(image_names_path);

    for (const auto image_name : image_names) {
      if (image_name.empty()) {
        continue;
      }

      const Image* image = reconstruction.FindImageWithName(image_name);
      if (image != nullptr) {
        std::cout
            << StringPrintf(
                   "Deleting image_id=%d, image_name=%s from reconstruction",
                   image->ImageId(), image->Name().c_str())
            << std::endl;
        reconstruction.DeRegisterImage(image->ImageId());
      } else {
        std::cout << StringPrintf(
                         "WARNING: Skipping image_name=%s, because it does not "
                         "exist in the reconstruction",
                         image_name.c_str())
                  << std::endl;
      }
    }
  }

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageFilterer(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 100.0;
  size_t min_num_observations = 10;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_focal_length_ratio", &min_focal_length_ratio);
  options.AddDefaultOption("max_focal_length_ratio", &max_focal_length_ratio);
  options.AddDefaultOption("max_extra_param", &max_extra_param);
  options.AddDefaultOption("min_num_observations", &min_num_observations);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const size_t num_reg_images = reconstruction.NumRegImages();

  reconstruction.FilterImages(min_focal_length_ratio, max_focal_length_ratio,
                              max_extra_param);

  std::vector<image_t> filtered_image_ids;
  for (const auto& image : reconstruction.Images()) {
    if (image.second.IsRegistered() &&
        image.second.NumPoints3D() < min_num_observations) {
      filtered_image_ids.push_back(image.first);
    }
  }

  for (const auto image_id : filtered_image_ids) {
    reconstruction.DeRegisterImage(image_id);
  }

  const size_t num_filtered_images =
      num_reg_images - reconstruction.NumRegImages();

  std::cout << StringPrintf("Filtered %d images from a total of %d images",
                            num_filtered_images, num_reg_images)
            << std::endl;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageRectifier(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string stereo_pairs_list;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("stereo_pairs_list", &stereo_pairs_list);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const auto stereo_pairs =
      ReadStereoImagePairs(stereo_pairs_list, reconstruction);

  StereoImageRectifier rectifier(undistort_camera_options, reconstruction,
                                 *options.image_path, output_path,
                                 stereo_pairs);
  rectifier.Start();
  rectifier.Wait();

  return EXIT_SUCCESS;
}

int RunImageRegistrator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Database database(*options.database_path);
    Timer timer;
    timer.Start();
    const size_t min_num_matches =
        static_cast<size_t>(options.mapper->min_num_matches);
    database_cache.Load(database, min_num_matches,
                        options.mapper->ignore_watermarks,
                        options.mapper->image_names);
    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  const auto mapper_options = options.mapper->Mapper();

  for (const auto& image : reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      continue;
    }

    PrintHeading1("Registering image #" + std::to_string(image.first) + " (" +
                  std::to_string(reconstruction.NumRegImages() + 1) + ")");

    std::cout << "  => Image sees " << image.second.NumVisiblePoints3D()
              << " / " << image.second.NumObservations() << " points"
              << std::endl;

    mapper.RegisterNextImage(mapper_options, image.first);
  }

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageUndistorter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string output_type = "COLMAP";

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("output_type", &output_type,
                           "{COLMAP, PMVS, CMP-MVS}");
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  options.Parse(argc, argv);

  CreateDirIfNotExists(output_path);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  std::unique_ptr<Thread> undistorter;
  if (output_type == "COLMAP") {
    undistorter.reset(new COLMAPUndistorter(undistort_camera_options,
                                            reconstruction, *options.image_path,
                                            output_path));
  } else if (output_type == "PMVS") {
    undistorter.reset(new PMVSUndistorter(undistort_camera_options,
                                          reconstruction, *options.image_path,
                                          output_path));
  } else if (output_type == "CMP-MVS") {
    undistorter.reset(new CMPMVSUndistorter(undistort_camera_options,
                                            reconstruction, *options.image_path,
                                            output_path));
  } else {
    std::cerr << "ERROR: Invalid `output_type` - supported values are "
                 "{'COLMAP', 'PMVS', 'CMP-MVS'}."
              << std::endl;
    return EXIT_FAILURE;
  }

  undistorter->Start();
  undistorter->Wait();

  return EXIT_SUCCESS;
}

int RunImageUndistorterStandalone(int argc, char** argv) {
  std::string input_file;
  std::string output_path;

  UndistortCameraOptions undistort_camera_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_file", &input_file);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  options.Parse(argc, argv);

  CreateDirIfNotExists(output_path);

  // Loads a text file containing the image names and camera information.
  // The format of the text file is
  //   image_name CAMERA_MODEL camera_params
  std::vector<std::pair<std::string, Camera>> image_names_and_cameras;

  {
    std::ifstream file(input_file);
    CHECK(file.is_open()) << input_file;

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty()) {
        continue;
      }

      std::string item;
      std::stringstream line_stream(line);

      // Loads the image name.
      std::string image_name;
      std::getline(line_stream, image_name, ' ');

      // Loads the camera and its parameters
      class Camera camera;

      std::getline(line_stream, item, ' ');
      if (!ExistsCameraModelWithName(item)) {
        std::cerr << "ERROR: Camera model " << item << " does not exist"
                  << std::endl;
        return EXIT_FAILURE;
      }
      camera.SetModelIdFromName(item);

      std::getline(line_stream, item, ' ');
      camera.SetWidth(std::stoll(item));

      std::getline(line_stream, item, ' ');
      camera.SetHeight(std::stoll(item));

      camera.Params().clear();
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        camera.Params().push_back(std::stold(item));
      }

      CHECK(camera.VerifyParams());

      image_names_and_cameras.emplace_back(image_name, camera);
    }
  }

  std::unique_ptr<Thread> undistorter;
  undistorter.reset(new PureImageUndistorter(undistort_camera_options,
                                             *options.image_path, output_path,
                                             image_names_and_cameras));

  undistorter->Start();
  undistorter->Wait();

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
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    const auto image_names = ReadTextFileLines(image_list_path);
    options.mapper->image_names =
        std::unordered_set<std::string>(image_names.begin(), image_names.end());
  }

  ReconstructionManager reconstruction_manager;
  if (input_path != "") {
    if (!ExistsDir(input_path)) {
      std::cerr << "ERROR: `input_path` is not a directory." << std::endl;
      return EXIT_FAILURE;
    }
    reconstruction_manager.Read(input_path);
  }

  IncrementalMapperController mapper(options.mapper.get(), *options.image_path,
                                     *options.database_path,
                                     &reconstruction_manager);

  // In case a new reconstruction is started, write results of individual sub-
  // models to as their reconstruction finishes instead of writing all results
  // after all reconstructions finished.
  size_t prev_num_reconstructions = 0;
  if (input_path == "") {
    mapper.AddCallback(
        IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [&]() {
          // If the number of reconstructions has not changed, the last model
          // was discarded for some reason.
          if (reconstruction_manager.Size() > prev_num_reconstructions) {
            const std::string reconstruction_path = JoinPaths(
                output_path, std::to_string(prev_num_reconstructions));
            const auto& reconstruction =
                reconstruction_manager.Get(prev_num_reconstructions);
            CreateDirIfNotExists(reconstruction_path);
            reconstruction.Write(reconstruction_path);
            options.Write(JoinPaths(reconstruction_path, "project.ini"));
            prev_num_reconstructions = reconstruction_manager.Size();
          }
        });
  }

  mapper.Start();
  mapper.Wait();

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  if (input_path != "" && reconstruction_manager.Size() > 0) {
    reconstruction_manager.Get(0).Write(output_path);
  }

  return EXIT_SUCCESS;
}

int RunHierarchicalMapper(int argc, char** argv) {
  HierarchicalMapperController::Options hierarchical_options;
  SceneClustering::Options clustering_options;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("database_path",
                            &hierarchical_options.database_path);
  options.AddRequiredOption("image_path", &hierarchical_options.image_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("num_workers", &hierarchical_options.num_workers);
  options.AddDefaultOption("image_overlap", &clustering_options.image_overlap);
  options.AddDefaultOption("leaf_max_num_images",
                           &clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  ReconstructionManager reconstruction_manager;

  HierarchicalMapperController hierarchical_mapper(
      hierarchical_options, clustering_options, *options.mapper,
      &reconstruction_manager);
  hierarchical_mapper.Start();
  hierarchical_mapper.Wait();

  reconstruction_manager.Write(output_path, &options);

  return EXIT_SUCCESS;
}

int RunMatchesImporter(int argc, char** argv) {
  std::string match_list_path;
  std::string match_type = "pairs";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("match_list_path", &match_list_path);
  options.AddDefaultOption("match_type", &match_type,
                           "{'pairs', 'raw', 'inliers'}");
  options.AddMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  std::unique_ptr<Thread> feature_matcher;
  if (match_type == "pairs") {
    ImagePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path;
    feature_matcher.reset(new ImagePairsFeatureMatcher(
        matcher_options, *options.sift_matching, *options.database_path));
  } else if (match_type == "raw" || match_type == "inliers") {
    FeaturePairsMatchingOptions matcher_options;
    matcher_options.match_list_path = match_list_path;
    matcher_options.verify_matches = match_type == "raw";
    feature_matcher.reset(new FeaturePairsFeatureMatcher(
        matcher_options, *options.sift_matching, *options.database_path));
  } else {
    std::cerr << "ERROR: Invalid `match_type`";
    return EXIT_FAILURE;
  }

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(feature_matcher.get());
  } else {
    feature_matcher->Start();
    feature_matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunModelAligner(int argc, char** argv) {
  std::string input_path;
  std::string ref_images_path;
  std::string output_path;
  int min_common_images = 3;
  bool robust_alignment = true;
  RANSACOptions ransac_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("ref_images_path", &ref_images_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_common_images", &min_common_images);
  options.AddDefaultOption("robust_alignment", &robust_alignment);
  options.AddDefaultOption("robust_alignment_max_error",
                           &ransac_options.max_error);
  options.Parse(argc, argv);

  if (robust_alignment && ransac_options.max_error <= 0) {
    std::cout << "ERROR: You must provide a maximum alignment error > 0"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> ref_image_names;
  std::vector<Eigen::Vector3d> ref_locations;
  std::vector<std::string> lines = ReadTextFileLines(ref_images_path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);
    std::string image_name = "";
    Eigen::Vector3d camera_position;
    line_parser >> image_name >> camera_position[0] >> camera_position[1] >>
        camera_position[2];
    ref_image_names.push_back(image_name);
    ref_locations.push_back(camera_position);
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading2("Aligning reconstruction");

  std::cout << StringPrintf(" => Using %d reference images",
                            ref_image_names.size())
            << std::endl;

  bool alignment_success;
  if (robust_alignment) {
    alignment_success = reconstruction.AlignRobust(
        ref_image_names, ref_locations, min_common_images, ransac_options);
  } else {
    alignment_success =
        reconstruction.Align(ref_image_names, ref_locations, min_common_images);
  }

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
    reconstruction.Write(output_path);

    std::vector<double> errors;
    errors.reserve(ref_image_names.size());

    for (size_t i = 0; i < ref_image_names.size(); ++i) {
      const Image* image = reconstruction.FindImageWithName(ref_image_names[i]);
      if (image != nullptr) {
        errors.push_back((image->ProjectionCenter() - ref_locations[i]).norm());
      }
    }

    std::cout << StringPrintf(" => Alignment error: %f (mean), %f (median)",
                              Mean(errors), Median(errors))
              << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
  }

  return EXIT_SUCCESS;
}

int RunModelAnalyzer(int argc, char** argv) {
  std::string path;

  OptionManager options;
  options.AddRequiredOption("path", &path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(path);

  std::cout << StringPrintf("Cameras: %d", reconstruction.NumCameras())
            << std::endl;
  std::cout << StringPrintf("Images: %d", reconstruction.NumImages())
            << std::endl;
  std::cout << StringPrintf("Registered images: %d",
                            reconstruction.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction.NumPoints3D())
            << std::endl;
  std::cout << StringPrintf("Observations: %d",
                            reconstruction.ComputeNumObservations())
            << std::endl;
  std::cout << StringPrintf("Mean track length: %f",
                            reconstruction.ComputeMeanTrackLength())
            << std::endl;
  std::cout << StringPrintf("Mean observations per image: %f",
                            reconstruction.ComputeMeanObservationsPerRegImage())
            << std::endl;
  std::cout << StringPrintf("Mean reprojection error: %fpx",
                            reconstruction.ComputeMeanReprojectionError())
            << std::endl;

  return EXIT_SUCCESS;
}

int RunModelConverter(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string output_type;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("output_type", &output_type,
                            "{BIN, TXT, NVM, Bundler, VRML, PLY}");
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  StringToLower(&output_type);
  if (output_type == "bin") {
    reconstruction.WriteBinary(output_path);
  } else if (output_type == "txt") {
    reconstruction.WriteText(output_path);
  } else if (output_type == "nvm") {
    reconstruction.ExportNVM(output_path);
  } else if (output_type == "bundler") {
    reconstruction.ExportBundler(output_path + ".bundle.out",
                                 output_path + ".list.txt");
  } else if (output_type == "ply") {
    reconstruction.ExportPLY(output_path);
  } else if (output_type == "vrml") {
    const auto base_path = output_path.substr(0, output_path.find_last_of("."));
    reconstruction.ExportVRML(base_path + ".images.wrl",
                              base_path + ".points3D.wrl", 1,
                              Eigen::Vector3d(1, 0, 0));
  } else {
    std::cerr << "ERROR: Invalid `output_type`" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int RunModelMerger(int argc, char** argv) {
  std::string input_path1;
  std::string input_path2;
  std::string output_path;
  double max_reproj_error = 64.0;

  OptionManager options;
  options.AddRequiredOption("input_path1", &input_path1);
  options.AddRequiredOption("input_path2", &input_path2);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("max_reproj_error", &max_reproj_error);
  options.Parse(argc, argv);

  Reconstruction reconstruction1;
  reconstruction1.Read(input_path1);
  PrintHeading2("Reconstruction 1");
  std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
            << std::endl;

  Reconstruction reconstruction2;
  reconstruction2.Read(input_path2);
  PrintHeading2("Reconstruction 2");
  std::cout << StringPrintf("Images: %d", reconstruction2.NumRegImages())
            << std::endl;
  std::cout << StringPrintf("Points: %d", reconstruction2.NumPoints3D())
            << std::endl;

  PrintHeading2("Merging reconstructions");
  if (reconstruction1.Merge(reconstruction2, max_reproj_error)) {
    std::cout << "=> Merge succeeded" << std::endl;
    PrintHeading2("Merged reconstruction");
    std::cout << StringPrintf("Images: %d", reconstruction1.NumRegImages())
              << std::endl;
    std::cout << StringPrintf("Points: %d", reconstruction1.NumPoints3D())
              << std::endl;
  } else {
    std::cout << "=> Merge failed" << std::endl;
  }

  reconstruction1.Write(output_path);

  return EXIT_SUCCESS;
}

int RunModelOrientationAligner(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string method = "MANHATTAN-WORLD";

  ManhattanWorldFrameEstimationOptions frame_estimation_options;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("method", &method,
                           "{MANHATTAN-WORLD, IMAGE-ORIENTATION}");
  options.AddDefaultOption("max_image_size",
                           &frame_estimation_options.max_image_size);
  options.Parse(argc, argv);

  StringToLower(&method);
  if (method != "manhattan-world" && method != "image-orientation") {
    std::cout << "ERROR: Invalid `method` - supported values are "
                 "'MANHATTAN-WORLD' or 'IMAGE-ORIENTATION'."
              << std::endl;
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Aligning Reconstruction");

  Eigen::Matrix3d tform;

  if (method == "manhattan-world") {
    const Eigen::Matrix3d frame = EstimateManhattanWorldFrame(
        frame_estimation_options, reconstruction, *options.image_path);

    if (frame.col(0).nonZeros() == 0) {
      std::cout << "Only aligning vertical axis" << std::endl;
      tform = RotationFromUnitVectors(frame.col(1), Eigen::Vector3d(0, 1, 0));
    } else if (frame.col(1).nonZeros() == 0) {
      tform = RotationFromUnitVectors(frame.col(0), Eigen::Vector3d(1, 0, 0));
      std::cout << "Only aligning horizontal axis" << std::endl;
    } else {
      tform = frame.transpose();
      std::cout << "Aligning horizontal and vertical axes" << std::endl;
    }
  } else if (method == "image-orientation") {
    const Eigen::Vector3d gravity_axis =
        EstimateGravityVectorFromImageOrientation(reconstruction);
    tform = RotationFromUnitVectors(gravity_axis, Eigen::Vector3d(0, 1, 0));
  } else {
    LOG(FATAL) << "Alignment method not supported";
  }

  std::cout << "Using the rotation matrix:" << std::endl;
  std::cout << tform << std::endl;

  reconstruction.Transform(SimilarityTransform3(
      1, RotationMatrixToQuaternion(tform), Eigen::Vector3d(0, 0, 0)));

  std::cout << "Writing aligned reconstruction..." << std::endl;
  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunSequentialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSequentialMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  SequentialFeatureMatcher feature_matcher(*options.sequential_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

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
    if (point3D.Track().Length() < min_track_len) {
      num_filtered += point3D.Track().Length();
      reconstruction.DeletePoint3D(point3D_id);
    }
  }

  std::cout << "Filtered observations: " << num_filtered << std::endl;

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunPointTriangulator(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  bool clear_points = false;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "clear_points", &clear_points,
      "Whether to clear all existing points and observations");
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(input_path)) {
    std::cerr << "ERROR: `input_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  const auto& mapper_options = *options.mapper;

  PrintHeading1("Loading model");

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Timer timer;
    timer.Start();

    Database database(*options.database_path);

    const size_t min_num_matches =
        static_cast<size_t>(mapper_options.min_num_matches);
    database_cache.Load(database, min_num_matches,
                        mapper_options.ignore_watermarks,
                        mapper_options.image_names);

    if (clear_points) {
      reconstruction.DeleteAllPoints2DAndPoints3D();
      reconstruction.TranscribeImageIdsToDatabase(database);
    }

    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  CHECK_GE(reconstruction.NumRegImages(), 2)
      << "Need at least two images for triangulation";

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  //////////////////////////////////////////////////////////////////////////////
  // Triangulation
  //////////////////////////////////////////////////////////////////////////////

  const auto tri_options = mapper_options.Triangulation();

  const auto& reg_image_ids = reconstruction.RegImageIds();

  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];

    const auto& image = reconstruction.Image(image_id);

    PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, i));

    const size_t num_existing_points3D = image.NumPoints3D();

    std::cout << "  => Image sees " << num_existing_points3D << " / "
              << image.NumObservations() << " points" << std::endl;

    mapper.TriangulateImage(tri_options, image_id);

    std::cout << "  => Triangulated "
              << (image.NumPoints3D() - num_existing_points3D) << " points"
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Retriangulation
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Retriangulation");

  CompleteAndMergeTracks(mapper_options, &mapper);

  //////////////////////////////////////////////////////////////////////////////
  // Bundle adjustment
  //////////////////////////////////////////////////////////////////////////////

  auto ba_options = mapper_options.GlobalBundleAdjustment();
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_extrinsics = false;

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
    // Avoid degeneracies in bundle adjustment.
    reconstruction.FilterObservationsWithNegativeDepth();

    const size_t num_observations = reconstruction.ComputeNumObservations();

    PrintHeading1("Bundle adjustment");
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    CHECK(bundle_adjuster.Solve(&reconstruction));

    size_t num_changed_observations = 0;
    num_changed_observations += CompleteAndMergeTracks(mapper_options, &mapper);
    num_changed_observations += FilterPoints(mapper_options, &mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < mapper_options.ba_global_max_refinement_change) {
      break;
    }
  }

  PrintHeading1("Extracting colors");
  reconstruction.ExtractColorsForAllImages(*options.image_path);

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

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
//       },
//       {
//           "camera_id": 2,
//           "image_prefix": "left2_image"
//       },
//       {
//           "camera_id": 3,
//           "image_prefix": "right1_image"
//       },
//       {
//           "camera_id": 4,
//           "image_prefix": "right2_image"
//       }
//     ]
//   }
// ]
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
// TODO: Provide an option to manually / explicitly set the relative extrinsics
// of the camera rig. At the moment, the relative extrinsics are automatically
// inferred from the reconstruction.
std::vector<CameraRig> ReadCameraRigConfig(
    const std::string& rig_config_path, const Reconstruction& reconstruction) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

  std::vector<CameraRig> camera_rigs;
  for (const auto& rig_config : pt) {
    CameraRig camera_rig;

    std::vector<std::string> image_prefixes;
    for (const auto& camera : rig_config.second.get_child("cameras")) {
      const int camera_id = camera.second.get<int>("camera_id");
      image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));
      camera_rig.AddCamera(camera_id, ComposeIdentityQuaternion(),
                           Eigen::Vector3d(0, 0, 0));
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
        }
      }

      if (has_ref_camera) {
        camera_rig.AddSnapshot(snapshot.second);
      }
    }

    camera_rig.Check(reconstruction);
    camera_rig.ComputeRelativePoses(reconstruction);

    camera_rigs.push_back(camera_rig);
  }

  return camera_rigs;
}

int RunRigBundleAdjuster(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string rig_config_path;

  RigBundleAdjuster::Options rig_ba_options;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.AddDefaultOption("RigBundleAdjustment.refine_relative_poses",
                           &rig_ba_options.refine_relative_poses);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  PrintHeading1("Camera rig configuration");

  auto camera_rigs = ReadCameraRigConfig(rig_config_path, reconstruction);

  BundleAdjustmentConfig config;
  for (size_t i = 0; i < camera_rigs.size(); ++i) {
    const auto& camera_rig = camera_rigs[i];
    PrintHeading2(StringPrintf("Camera Rig %d", i + 1));
    std::cout << StringPrintf("Cameras: %d", camera_rig.NumCameras())
              << std::endl;
    std::cout << StringPrintf("Snapshots: %d", camera_rig.NumSnapshots())
              << std::endl;

    // Add all registered images to the bundle adjustment configuration.
    for (const auto image_id : reconstruction.RegImageIds()) {
      config.AddImage(image_id);
    }
  }

  PrintHeading1("Rig bundle adjustment");

  BundleAdjustmentOptions ba_options = *options.bundle_adjustment;
  ba_options.solver_options.minimizer_progress_to_stdout = true;
  RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, config);
  CHECK(bundle_adjuster.Solve(&reconstruction, &camera_rigs));

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunSpatialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSpatialMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  SpatialFeatureMatcher feature_matcher(*options.spatial_matching,
                                        *options.sift_matching,
                                        *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

int RunTransitiveMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddTransitiveMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  TransitiveFeatureMatcher feature_matcher(*options.transitive_matching,
                                           *options.sift_matching,
                                           *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

// Loads descriptors for training from the database. Loads all descriptors from
// the database if max_num_images < 0, otherwise the descriptors of a random
// subset of images are selected.
FeatureDescriptors LoadRandomDatabaseDescriptors(
    const std::string& database_path, const int max_num_images) {
  Database database(database_path);
  DatabaseTransaction database_transaction(&database);

  const std::vector<Image> images = database.ReadAllImages();

  FeatureDescriptors descriptors;

  std::vector<size_t> image_idxs;
  size_t num_descriptors = 0;
  if (max_num_images < 0) {
    // All images in the database.
    image_idxs.resize(images.size());
    std::iota(image_idxs.begin(), image_idxs.end(), 0);
    num_descriptors = database.NumDescriptors();
  } else {
    // Random subset of images in the database.
    CHECK_LE(max_num_images, images.size());
    RandomSampler random_sampler(max_num_images);
    random_sampler.Initialize(images.size());
    image_idxs = random_sampler.Sample();
    for (const auto image_idx : image_idxs) {
      const auto& image = images.at(image_idx);
      num_descriptors += database.NumDescriptorsForImage(image.ImageId());
    }
  }

  descriptors.resize(num_descriptors, 128);

  size_t descriptor_row = 0;
  for (const auto image_idx : image_idxs) {
    const auto& image = images.at(image_idx);
    const FeatureDescriptors image_descriptors =
        database.ReadDescriptors(image.ImageId());
    descriptors.block(descriptor_row, 0, image_descriptors.rows(), 128) =
        image_descriptors;
    descriptor_row += image_descriptors.rows();
  }

  CHECK_EQ(descriptor_row, num_descriptors);

  return descriptors;
}

int RunVocabTreeBuilder(int argc, char** argv) {
  std::string vocab_tree_path;
  retrieval::VisualIndex<>::BuildOptions build_options;
  int max_num_images = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("vocab_tree_path", &vocab_tree_path);
  options.AddDefaultOption("num_visual_words", &build_options.num_visual_words);
  options.AddDefaultOption("num_checks", &build_options.num_checks);
  options.AddDefaultOption("branching", &build_options.branching);
  options.AddDefaultOption("num_iterations", &build_options.num_iterations);
  options.AddDefaultOption("max_num_images", &max_num_images);
  options.Parse(argc, argv);

  retrieval::VisualIndex<> visual_index;

  std::cout << "Loading descriptors..." << std::endl;
  const auto descriptors =
      LoadRandomDatabaseDescriptors(*options.database_path, max_num_images);
  std::cout << "  => Loaded a total of " << descriptors.rows() << " descriptors"
            << std::endl;

  std::cout << "Building index for visual words..." << std::endl;
  visual_index.Build(build_options, descriptors);
  std::cout << " => Quantized descriptor space using "
            << visual_index.NumVisualWords() << " visual words" << std::endl;

  std::cout << "Saving index to file..." << std::endl;
  visual_index.Write(vocab_tree_path);

  return EXIT_SUCCESS;
}

int RunVocabTreeMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddVocabTreeMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  VocabTreeFeatureMatcher feature_matcher(*options.vocab_tree_matching,
                                          *options.sift_matching,
                                          *options.database_path);

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(&feature_matcher);
  } else {
    feature_matcher.Start();
    feature_matcher.Wait();
  }

  return EXIT_SUCCESS;
}

std::vector<Image> ReadVocabTreeRetrievalImageList(const std::string& path,
                                                   Database* database) {
  std::vector<Image> images;
  if (path.empty()) {
    images.reserve(database->NumImages());
    for (const auto& image : database->ReadAllImages()) {
      images.push_back(image);
    }
  } else {
    DatabaseTransaction database_transaction(database);

    const auto image_names = ReadTextFileLines(path);
    images.reserve(image_names.size());
    for (const auto& image_name : image_names) {
      const auto image = database->ReadImageWithName(image_name);
      CHECK_NE(image.ImageId(), kInvalidImageId);
      images.push_back(image);
    }
  }
  return images;
}

int RunVocabTreeRetriever(int argc, char** argv) {
  std::string vocab_tree_path;
  std::string database_image_list_path;
  std::string query_image_list_path;
  std::string output_index_path;
  retrieval::VisualIndex<>::QueryOptions query_options;
  int max_num_features = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("vocab_tree_path", &vocab_tree_path);
  options.AddDefaultOption("database_image_list_path",
                           &database_image_list_path);
  options.AddDefaultOption("query_image_list_path", &query_image_list_path);
  options.AddDefaultOption("output_index_path", &output_index_path);
  options.AddDefaultOption("num_images", &query_options.max_num_images);
  options.AddDefaultOption("num_neighbors", &query_options.num_neighbors);
  options.AddDefaultOption("num_checks", &query_options.num_checks);
  options.AddDefaultOption("num_images_after_verification",
                           &query_options.num_images_after_verification);
  options.AddDefaultOption("max_num_features", &max_num_features);
  options.Parse(argc, argv);

  retrieval::VisualIndex<> visual_index;
  visual_index.Read(vocab_tree_path);

  Database database(*options.database_path);

  const auto database_images =
      ReadVocabTreeRetrievalImageList(database_image_list_path, &database);
  const auto query_images =
      (!query_image_list_path.empty() || output_index_path.empty())
          ? ReadVocabTreeRetrievalImageList(query_image_list_path, &database)
          : std::vector<Image>();

  //////////////////////////////////////////////////////////////////////////////
  // Perform image indexing
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < database_images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Indexing image [%d/%d]", i + 1,
                              database_images.size())
              << std::flush;

    if (visual_index.ImageIndexed(database_images[i].ImageId())) {
      std::cout << std::endl;
      continue;
    }

    auto keypoints = database.ReadKeypoints(database_images[i].ImageId());
    auto descriptors = database.ReadDescriptors(database_images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    visual_index.Add(retrieval::VisualIndex<>::IndexOptions(),
                     database_images[i].ImageId(), keypoints, descriptors);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  // Compute the TF-IDF weights, etc.
  visual_index.Prepare();

  // Optionally save the indexing data for the database images (as well as the
  // original vocabulary tree data) to speed up future indexing.
  if (!output_index_path.empty()) {
    visual_index.Write(output_index_path);
  }

  if (query_images.empty()) {
    return EXIT_SUCCESS;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Perform image queries
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_map<image_t, const Image*> image_id_to_image;
  image_id_to_image.reserve(database_images.size());
  for (const auto& image : database_images) {
    image_id_to_image.emplace(image.ImageId(), &image);
  }

  for (size_t i = 0; i < query_images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Querying for image %s [%d/%d]",
                              query_images[i].Name().c_str(), i + 1,
                              query_images.size())
              << std::flush;

    auto keypoints = database.ReadKeypoints(query_images[i].ImageId());
    auto descriptors = database.ReadDescriptors(query_images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    std::vector<retrieval::ImageScore> image_scores;
    visual_index.Query(query_options, keypoints, descriptors, &image_scores);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
    for (const auto& image_score : image_scores) {
      const auto& image = *image_id_to_image.at(image_score.image_id);
      std::cout << StringPrintf("  image_id=%d, image_name=%s, score=%f",
                                image_score.image_id, image.Name().c_str(),
                                image_score.score)
                << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

typedef std::function<int(int, char**)> command_func_t;

int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << StringPrintf(
                   "%s -- Structure-from-Motion and Multi-View Stereo\n"
                   "              (%s)",
                   GetVersionInfo().c_str(), GetBuildInfo().c_str())
            << std::endl
            << std::endl;

  std::cout << "Usage:" << std::endl;
  std::cout << "  colmap [command] [options]" << std::endl << std::endl;

  std::cout << "Documentation:" << std::endl;
  std::cout << "  https://colmap.github.io/" << std::endl << std::endl;

  std::cout << "Example usage:" << std::endl;
  std::cout << "  colmap help [ -h, --help ]" << std::endl;
  std::cout << "  colmap gui" << std::endl;
  std::cout << "  colmap gui -h [ --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor -h [ --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
               "--workspace_path WORKSPACE"
            << std::endl;
  std::cout << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE"
            << std::endl;
  std::cout << "  colmap exhaustive_matcher --database_path DATABASE"
            << std::endl;
  std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL"
            << std::endl;
  std::cout << "  ..." << std::endl << std::endl;

  std::cout << "Available commands:" << std::endl;
  std::cout << "  help" << std::endl;
  for (const auto& command : commands) {
    std::cout << "  " << command.first << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("gui", &RunGraphicalUserInterface);
  commands.emplace_back("automatic_reconstructor", &RunAutomaticReconstructor);
  commands.emplace_back("bundle_adjuster", &RunBundleAdjuster);
  commands.emplace_back("color_extractor", &RunColorExtractor);
  commands.emplace_back("database_creator", &RunDatabaseCreator);
  commands.emplace_back("database_merger", &RunDatabaseMerger);
  commands.emplace_back("delaunay_mesher", &RunDelaunayMesher);
  commands.emplace_back("exhaustive_matcher", &RunExhaustiveMatcher);
  commands.emplace_back("feature_extractor", &RunFeatureExtractor);
  commands.emplace_back("feature_importer", &RunFeatureImporter);
  commands.emplace_back("hierarchical_mapper", &RunHierarchicalMapper);
  commands.emplace_back("image_deleter", &RunImageDeleter);
  commands.emplace_back("image_filterer", &RunImageFilterer);
  commands.emplace_back("image_rectifier", &RunImageRectifier);
  commands.emplace_back("image_registrator", &RunImageRegistrator);
  commands.emplace_back("image_undistorter", &RunImageUndistorter);
  commands.emplace_back("image_undistorter_standalone",
                        &RunImageUndistorterStandalone);
  commands.emplace_back("mapper", &RunMapper);
  commands.emplace_back("matches_importer", &RunMatchesImporter);
  commands.emplace_back("model_aligner", &RunModelAligner);
  commands.emplace_back("model_analyzer", &RunModelAnalyzer);
  commands.emplace_back("model_converter", &RunModelConverter);
  commands.emplace_back("model_merger", &RunModelMerger);
  commands.emplace_back("model_orientation_aligner",
                        &RunModelOrientationAligner);
  commands.emplace_back("patch_match_stereo", &RunPatchMatchStereo);
  commands.emplace_back("point_filtering", &RunPointFiltering);
  commands.emplace_back("point_triangulator", &RunPointTriangulator);
  commands.emplace_back("poisson_mesher", &RunPoissonMesher);
  commands.emplace_back("project_generator", &RunProjectGenerator);
  commands.emplace_back("rig_bundle_adjuster", &RunRigBundleAdjuster);
  commands.emplace_back("sequential_matcher", &RunSequentialMatcher);
  commands.emplace_back("spatial_matcher", &RunSpatialMatcher);
  commands.emplace_back("stereo_fusion", &RunStereoFuser);
  commands.emplace_back("transitive_matcher", &RunTransitiveMatcher);
  commands.emplace_back("vocab_tree_builder", &RunVocabTreeBuilder);
  commands.emplace_back("vocab_tree_matcher", &RunVocabTreeMatcher);
  commands.emplace_back("vocab_tree_retriever", &RunVocabTreeRetriever);

  if (argc == 1) {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      std::cerr << StringPrintf(
                       "ERROR: Command `%s` not recognized. To list the "
                       "available commands, run `colmap help`.",
                       command.c_str())
                << std::endl;
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

  return ShowHelp(commands);
}
