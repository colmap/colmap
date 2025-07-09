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
#include "colmap/retrieval/visual_index.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

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
  options.AddExtractionOptions();
  options.Parse(argc, argv);

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;

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
  options.AddExtractionOptions();
  options.Parse(argc, argv);

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
  options.Parse(argc, argv);

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
  options.AddMatchingOptions();
  options.Parse(argc, argv);

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
  options.Parse(argc, argv);

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
  options.Parse(argc, argv);

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
  options.Parse(argc, argv);

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
  options.Parse(argc, argv);

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

}  // namespace colmap
