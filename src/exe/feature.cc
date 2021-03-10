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

#include "exe/feature.h"

#include <QApplication>

#include "base/camera_models.h"
#include "base/gps.h"
#include "base/image_reader.h"
#include "exe/gui.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "util/misc.h"
#include "util/opengl_utils.h"
#include "util/option_manager.h"

namespace colmap {
namespace {

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

bool UseSpatialMatcher(const std::string& database_path, double max_distance,
                       bool is_lla) {
  PrintHeading1("Checking for spatial matching");
  Database database(database_path);
  constexpr double limit = std::numeric_limits<double>::max();
  Eigen::Vector3d min_coord(limit, limit, limit);
  Eigen::Vector3d max_coord(-limit, -limit, -limit);
  GPSTransform gps(GPSTransform::WGS84);
  std::vector<Image> images = database.ReadAllImages();
  for (const Image& image : database.ReadAllImages()) {
    if (!image.HasTvecPrior()) {
      std::cout << "Found image with invalid location prior; using exhaustive "
                   "matching"
                << std::endl;
      return false;
    }
    Eigen::Vector3d xyz;
    if (is_lla) {
      xyz = gps.EllToXYZ({image.TvecPrior()}).front();
    } else {
      xyz = image.TvecPrior();
    }
    min_coord = xyz.cwiseMin(min_coord);
    max_coord = xyz.cwiseMax(max_coord);
  }

  const double image_distance = (max_coord - min_coord).norm();
  std::cout << StringPrintf("Max matching distance %f vs max image distance %f",
                            max_distance, image_distance)
            << std::endl;
  if (image_distance > max_distance) {
    std::cout << "All images have valid location priors and distance; using "
                 "spatial matching"
              << std::endl;
    return true;
  } else {
    std::cout << "Max matching distance exceeds max image distance; using "
                 "exhaustive matching"
              << std::endl;
    return false;
  }
}

bool UseImagePairsMatcher(const std::string& database_path,
                          const std::string& match_list_path) {
  PrintHeading1("Checking for fixed pair matching");
  if (!ExistsFile(match_list_path)) {
    std::cout << "Invalid matches file provided; trying spatial matching next"
              << std::endl;
    return false;
  }
  Database database(database_path);
  size_t num_images_read = 0;
  std::vector<std::string> match_image_pairs =
      ReadTextFileLines(match_list_path);
  std::unordered_set<std::string> match_images;

  for (const std::string image_name_pair : match_image_pairs) {
    std::vector<std::string> image_names = StringSplit(image_name_pair, " ");
    match_images.insert(image_names[0]);
    match_images.insert(image_names[1]);
  }

  for (const Image& image : database.ReadAllImages()) {
    if (match_images.count(image.Name()) == 0) {
      std::cout << "Image " << image.Name()
                << " exists in the DB, but was not found in the match file; "
                   "fixed image matching cannot be used, trying spatial "
                   "matching next."
                << std::endl;
      return false;
    }
    num_images_read++;
  }

  if (match_images.size() == num_images_read) {
    std::cout << "All images have valid pairings in the database; using "
                 "fixed matching."
              << std::endl;
  } else {
    std::cout << "WARN: Matching file contains more images than DB; "
                 "additional images will be ignored."
              << std::endl;
  }
  return true;
}

// This enum can be used as optional input for feature_extractor and
// feature_importer to ensure that the camera flags of ImageReader are set in an
// exclusive and unambigous way. The table below explains the corespondence of
// each setting with the flags
//
// -----------------------------------------------------------------------------------
// |            |                         ImageReaderOptions                         |
// | CameraMode | single_camera | single_camera_per_folder | single_camera_per_image |
// |------------|---------------|--------------------------|-------------------------|
// | AUTO       | false         | false                    | false                   |
// | SINGLE     | true          | false                    | false                   |
// | PER_FOLDER | false         | true                     | false                   |
// | PER_IMAGE  | false         | false                    | true                    |
// -----------------------------------------------------------------------------------
//
// Note: When using AUTO mode a camera model will be uniquely identified by the
// following 5 parameters from EXIF tags:
// 1. Camera Make
// 2. Camera Model
// 3. Focal Length
// 4. Image Width
// 5. Image Height
//
// If any of the tags is missing then a camera model is considered invalid and a
// new camera is created similar to the PER_IMAGE mode.
//
// If these considered fields are not sufficient to uniquely identify a camera
// then using the AUTO mode will lead to incorrect setup for the cameras, e.g.
// the same camera is used with same focal length but different principal point
// between captures. In these cases it is recommended to either use the
// PER_FOLDER or PER_IMAGE settings.
enum class CameraMode { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3 };

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

}  // namespace

int RunFeatureExtractor(int argc, char** argv) {
  std::string image_list_path;
  int camera_mode = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("camera_mode", &camera_mode);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddExtractionOptions();
  options.Parse(argc, argv);

  ImageReaderOptions reader_options = *options.image_reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!ExistsCameraModelWithName(reader_options.camera_model)) {
    std::cerr << "ERROR: Camera model does not exist" << std::endl;
  }

  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
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
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (camera_mode >= 0) {
    UpdateImageReaderOptionsFromCameraMode(reader_options,
                                           (CameraMode)camera_mode);
  }

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
    if (reader_options.image_list.empty()) {
      return EXIT_SUCCESS;
    }
  }

  if (!VerifyCameraParams(reader_options.camera_model,
                          reader_options.camera_params)) {
    return EXIT_FAILURE;
  }

  FeatureImporter feature_importer(reader_options, import_path);
  feature_importer.Start();
  feature_importer.Wait();

  return EXIT_SUCCESS;
}

int RunAutoMatcher(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddAutoMatchingOptions();
  options.Parse(argc, argv);

  std::unique_ptr<QApplication> app;
  if (options.sift_matching->use_gpu && kUseOpenGL) {
    app.reset(new QApplication(argc, argv));
  }

  std::shared_ptr<Thread> feature_matcher;
  if (!options.image_pairs_matching->match_list_path.empty() &&
      UseImagePairsMatcher(*options.database_path,
                           options.image_pairs_matching->match_list_path)) {
    feature_matcher.reset(new ImagePairsFeatureMatcher(
        *options.image_pairs_matching, *options.sift_matching,
        *options.database_path));
  } else if (UseSpatialMatcher(*options.database_path,
                               options.spatial_matching->max_distance,
                               options.spatial_matching->is_gps)) {
    feature_matcher.reset(new SpatialFeatureMatcher(*options.spatial_matching,
                                                    *options.sift_matching,
                                                    *options.database_path));
  } else {
    feature_matcher.reset(new ExhaustiveFeatureMatcher(
        *options.exhaustive_matching, *options.sift_matching,
        *options.database_path));
  }

  if (options.sift_matching->use_gpu && kUseOpenGL) {
    RunThreadWithOpenGLContext(feature_matcher.get());
  } else {
    feature_matcher->Start();
    feature_matcher->Wait();
  }

  return EXIT_SUCCESS;
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

}  // namespace colmap
