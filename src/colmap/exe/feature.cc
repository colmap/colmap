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

#include "colmap/exe/feature.h"

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/exe/gui.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/threading.h"

namespace colmap {

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params) {
  if (!ExistsCameraModelWithName(camera_model)) {
    LOG(ERROR) << "Camera model does not exist";
    return false;
  }

  const std::vector<double> camera_params = CSVToVector<double>(params);
  const CameraModelId camera_model_id = CameraModelNameToId(camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    LOG(ERROR) << "Invalid camera parameters";
    return false;
  }
  return true;
}

void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions& options,
                                            CameraMode mode) {
  switch (mode) {
    case CameraMode::AUTO:
      options.single_camera = false;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = false;
      break;
    case CameraMode::SINGLE:
      options.single_camera = true;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = false;
      break;
    case CameraMode::PER_FOLDER:
      options.single_camera = false;
      options.single_camera_per_folder = true;
      options.single_camera_per_image = false;
      break;
    case CameraMode::PER_IMAGE:
      options.single_camera = false;
      options.single_camera_per_folder = false;
      options.single_camera_per_image = true;
      break;
  }
}

int RunFeatureExtractor(int argc, char** argv) {
  std::string image_list_path;
  int camera_mode = -1;
  std::string descriptor_normalization = "l1_root";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("camera_mode", &camera_mode);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("descriptor_normalization",
                           &descriptor_normalization,
                           "{'l1_root', 'l2'}");
  options.AddFeatureExtractionOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;
  reader_options.as_rgb = options.feature_extraction->RequiresRGB();

  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  StringToUpper(&descriptor_normalization);
  options.feature_extraction->sift->normalization =
      SiftExtractionOptions::NormalizationFromString(descriptor_normalization);

  if (!image_list_path.empty()) {
    reader_options.image_names = ReadTextFileLines(image_list_path);
    if (reader_options.image_names.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!ExistsCameraModelWithName(reader_options.camera_model)) {
    LOG(ERROR) << "Camera model does not exist";
  }

  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_extraction->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto feature_extractor = CreateFeatureExtractorController(
      *options.database_path, reader_options, *options.feature_extraction);

  if (options.feature_extraction->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(feature_extractor.get());
  } else {
    feature_extractor->Start();
    feature_extractor->Wait();
  }

  return EXIT_SUCCESS;
}

int RunFeatureImporter(int argc, char** argv) {
  std::string import_path;
  std::string image_list_path;
  int camera_mode = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("camera_mode", &camera_mode);
  options.AddRequiredOption("import_path", &import_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddFeatureExtractionOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;

  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  if (!image_list_path.empty()) {
    reader_options.image_names = ReadTextFileLines(image_list_path);
    if (reader_options.image_names.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;
  }

  auto feature_importer = CreateFeatureImporterController(
      *options.database_path, reader_options, import_path);
  feature_importer->Start();
  feature_importer->Wait();

  return EXIT_SUCCESS;
}

int RunExhaustiveMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddExhaustivePairingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto matcher = CreateExhaustiveFeatureMatcher(*options.exhaustive_pairing,
                                                *options.feature_matching,
                                                *options.two_view_geometry,
                                                *options.database_path);

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunMatchesImporter(int argc, char** argv) {
  std::string match_list_path;
  std::string match_type = "pairs";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("match_list_path", &match_list_path);
  options.AddDefaultOption(
      "match_type", &match_type, "{'pairs', 'raw', 'inliers'}");
  options.AddFeatureMatchingOptions();
  options.AddTwoViewGeometryOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  std::unique_ptr<Thread> matcher;
  if (match_type == "pairs") {
    ImportedPairingOptions pairing_options;
    pairing_options.match_list_path = match_list_path;
    matcher = CreateImagePairsFeatureMatcher(pairing_options,
                                             *options.feature_matching,
                                             *options.two_view_geometry,
                                             *options.database_path);
  } else if (match_type == "raw" || match_type == "inliers") {
    FeaturePairsMatchingOptions pairing_options;
    pairing_options.match_list_path = match_list_path;
    pairing_options.verify_matches = match_type == "raw";
    matcher = CreateFeaturePairsFeatureMatcher(pairing_options,
                                               *options.feature_matching,
                                               *options.two_view_geometry,
                                               *options.database_path);
  } else {
    LOG(ERROR) << "Invalid `match_type`";
    return EXIT_FAILURE;
  }

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunSequentialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSequentialPairingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto matcher = CreateSequentialFeatureMatcher(*options.sequential_pairing,
                                                *options.feature_matching,
                                                *options.two_view_geometry,
                                                *options.database_path);

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunSpatialMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddSpatialPairingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto matcher = CreateSpatialFeatureMatcher(*options.spatial_pairing,
                                             *options.feature_matching,
                                             *options.two_view_geometry,
                                             *options.database_path);

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunTransitiveMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddTransitivePairingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto matcher = CreateTransitiveFeatureMatcher(*options.transitive_pairing,
                                                *options.feature_matching,
                                                *options.two_view_geometry,
                                                *options.database_path);

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunVocabTreeMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddVocabTreePairingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (options.feature_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  auto matcher = CreateVocabTreeFeatureMatcher(*options.vocab_tree_pairing,
                                               *options.feature_matching,
                                               *options.two_view_geometry,
                                               *options.database_path);

  if (options.feature_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(matcher.get());
  } else {
    matcher->Start();
    matcher->Wait();
  }

  return EXIT_SUCCESS;
}

int RunGeometricVerifier(int argc, char** argv) {
  ExistingMatchedPairingOptions pairing_options;
  GeometricVerifierOptions verifier_options;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddTwoViewGeometryOptions();
  options.AddDefaultOption("batch_size", &pairing_options.batch_size);
  options.AddDefaultOption("num_threads", &verifier_options.num_threads);
  options.AddDefaultOption("rig_verification",
                           &verifier_options.rig_verification);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  auto verifier = CreateGeometricVerifier(verifier_options,
                                          pairing_options,
                                          *options.two_view_geometry,
                                          *options.database_path);
  verifier->Start();
  verifier->Wait();

  return EXIT_SUCCESS;
}

void RunGuidedGeometricVerifierImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const ExistingMatchedPairingOptions& pairing_options,
    const TwoViewGeometryOptions& geometry_options,
    const int num_threads) {
  // Set all relative poses from a given reconstruction.
  auto database = Database::Open(database_path);
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> two_view_geometries;
  two_view_geometries = database->ReadTwoViewGeometries();
  database->ClearTwoViewGeometries();
  for (const auto& [pair_id, two_view_geometry] : two_view_geometries) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!reconstruction->ExistsImage(image_id1) ||
        !reconstruction->ExistsImage(image_id2)) {
      database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      continue;
    }
    Image& image1 = reconstruction->Image(image_id1);
    Image& image2 = reconstruction->Image(image_id2);
    if (!image1.HasPose() || !image2.HasPose()) {
      database->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      continue;
    }
    const Rigid3d cam1_from_world = image1.CamFromWorld();
    const Rigid3d cam2_from_world = image2.CamFromWorld();

    TwoViewGeometry two_view_geometry_copy = two_view_geometry;
    two_view_geometry_copy.config =
        TwoViewGeometry::ConfigurationType::CALIBRATED;
    two_view_geometry_copy.cam2_from_cam1 =
        cam2_from_world * Inverse(cam1_from_world);
    two_view_geometry_copy.inlier_matches.clear();
    database->WriteTwoViewGeometry(
        image_id1, image_id2, two_view_geometry_copy);
  }

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = num_threads;
  // We do not need rig verification in this case.
  verifier_options.rig_verification = false;
  verifier_options.use_existing_relative_pose = true;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  verifier->Start();
  verifier->Wait();
}

int RunGuidedGeometricVerifier(int argc, char** argv) {
  std::string input_path;
  ExistingMatchedPairingOptions pairing_options;
  int num_threads = -1;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddDatabaseOptions();
  options.AddTwoViewGeometryOptions();
  options.AddDefaultOption("batch_size", &pairing_options.batch_size);
  options.AddDefaultOption("num_threads", &num_threads);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  RunGuidedGeometricVerifierImpl(reconstruction,
                                 *options.database_path,
                                 pairing_options,
                                 *options.two_view_geometry,
                                 num_threads);

  return EXIT_SUCCESS;
}

}  // namespace colmap
