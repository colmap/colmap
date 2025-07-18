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
#include "colmap/estimators/generalized_pose.h"
#include "colmap/exe/gui.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/database_cache.h"
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

int RunRigVerifier(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddMatchingOptions();
  options.Parse(argc, argv);

  Database database(*options.database_path);
  const std::shared_ptr<DatabaseCache> database_cache =
      DatabaseCache::Create(database,
                            /*inlier_matches=*/false,
                            /*min_num_matches=*/1,
                            /*ignore_watermarks=*/true,
                            /*image_names=*/{});

  std::unordered_map<image_t, frame_t> image_to_frame_ids;
  for (const auto& [frame_id, frame] : database_cache->Frames()) {
    for (const data_t& data_id : frame.ImageIds()) {
      image_to_frame_ids[data_id.id] = frame_id;
    }
  }

  std::set<std::pair<frame_t, frame_t>> frame_pairs;
  for (const auto& [image_pair_id, _] : database.ReadNumMatches()) {
    const auto [image_id1, image_id2] =
        Database::PairIdToImagePair(image_pair_id);
    frame_t frame_id1 = image_to_frame_ids.at(image_id1);
    frame_t frame_id2 = image_to_frame_ids.at(image_id2);
    if (frame_id1 > frame_id2) {
      std::swap(frame_id1, frame_id2);
    }
    frame_pairs.insert(std::make_pair(frame_id1, frame_id2));
  }

  ThreadPool thread_pool(ThreadPool::kMaxNumThreads);
  std::mutex database_mutex;

  for (const auto& [frame_id1, frame_id2] : frame_pairs) {
    thread_pool.AddTask([&options,
                         &database,
                         &database_cache,
                         &database_mutex,
                         frame_id1 = frame_id1,
                         frame_id2 = frame_id2]() {
      std::vector<Eigen::Vector2d> points2D1;
      std::vector<Eigen::Vector2d> points2D2;
      std::vector<size_t> camera_idxs1;
      std::vector<size_t> camera_idxs2;
      std::vector<Rigid3d> cams_from_rig;
      std::vector<Camera> cameras;
      std::vector<std::tuple<image_t, point2D_t, image_t, point2D_t>> corrs;

      const Frame& frame1 = database_cache->Frame(frame_id1);
      const Frame& frame2 = database_cache->Frame(frame_id2);
      const Rig& rig1 = database_cache->Rig(frame1.RigId());
      const Rig& rig2 = database_cache->Rig(frame2.RigId());
      if (rig1.NumSensors() == 1 && rig2.NumSensors()) {
        return;
      }

      std::unordered_map<camera_t, size_t> camera_id_to_idx;
      auto maybe_add_camera = [&cameras, &cams_from_rig, &camera_id_to_idx](
                                  const Rig& rig, const Camera& camera) {
        const auto [it, inserted] =
            camera_id_to_idx.emplace(camera.camera_id, cameras.size());
        if (inserted) {
          cameras.push_back(camera);
          if (rig.IsRefSensor(camera.SensorId())) {
            cams_from_rig.push_back(Rigid3d());
          } else {
            cams_from_rig.push_back(rig.SensorFromRig(camera.SensorId()));
          }
        }
        return it->second;
      };

      for (const data_t& image_id1 : frame1.ImageIds()) {
        const Image& image1 = database_cache->Image(image_id1.id);
        const Camera& camera1 = database_cache->Camera(image1.CameraId());
        const size_t camera_idx1 = maybe_add_camera(rig1, camera1);

        for (const data_t& image_id2 : frame2.ImageIds()) {
          const Image& image2 = database_cache->Image(image_id2.id);
          const Camera& camera2 = database_cache->Camera(image2.CameraId());
          const size_t camera_idx2 = maybe_add_camera(rig2, camera2);

          const FeatureMatches matches = database_cache->CorrespondenceGraph()
                                             ->FindCorrespondencesBetweenImages(
                                                 image_id1.id, image_id2.id);
          for (const auto& match : matches) {
            points2D1.push_back(image1.Point2D(match.point2D_idx1).xy);
            points2D2.push_back(image2.Point2D(match.point2D_idx2).xy);
            camera_idxs1.push_back(camera_idx1);
            camera_idxs2.push_back(camera_idx2);
            corrs.emplace_back(image_id1.id,
                               match.point2D_idx1,
                               image_id2.id,
                               match.point2D_idx2);
          }
        }
      }

      if (points2D1.empty()) {
        return;
      }

      RANSACOptions ransac_options = options.two_view_geometry->ransac_options;
      double cam_max_error = 0;
      for (const Camera& camera : cameras) {
        cam_max_error += camera.CamFromImgThreshold(ransac_options.max_error);
      }
      ransac_options.max_error = cam_max_error / cameras.size();

      std::optional<Rigid3d> maybe_rig2_from_rig1;
      std::optional<Rigid3d> maybe_pano2_from_pano1;
      size_t num_inliers;
      std::vector<char> inlier_mask;
      if (!EstimateGeneralizedRelativePose(ransac_options,
                                           points2D1,
                                           points2D2,
                                           camera_idxs1,
                                           camera_idxs2,
                                           cams_from_rig,
                                           cameras,
                                           &maybe_rig2_from_rig1,
                                           &maybe_pano2_from_pano1,
                                           &num_inliers,
                                           &inlier_mask) ||
          num_inliers < options.two_view_geometry->min_num_inliers) {
        return;
      }

      std::unordered_map<image_pair_t, FeatureMatches> inlier_matches;
      inlier_matches.reserve(num_inliers);
      for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (!inlier_mask[i]) {
          continue;
        }
        const auto& [image_id1, point2D_idx1, image_id2, point2D_idx2] =
            corrs[i];
        if (Database::SwapImagePair(image_id1, image_id2)) {
          inlier_matches[Database::ImagePairToPairId(image_id1, image_id2)]
              .emplace_back(point2D_idx2, point2D_idx1);
        } else {
          inlier_matches[Database::ImagePairToPairId(image_id1, image_id2)]
              .emplace_back(point2D_idx1, point2D_idx2);
        }
      }

      std::lock_guard database_lock(database_mutex);
      for (const auto& [pair_id, pair_matches] : inlier_matches) {
        const auto& [image_id1, image_id2] =
            Database::PairIdToImagePair(pair_id);
        TwoViewGeometry two_view_geometry =
            database.ReadTwoViewGeometry(image_id1, image_id2);
        if (two_view_geometry.inlier_matches.size() >= pair_matches.size()) {
          continue;
        }

        two_view_geometry.inlier_matches = pair_matches;

        const Rigid3d rig2_from_rig1 = maybe_rig2_from_rig1.has_value()
                                           ? maybe_rig2_from_rig1.value()
                                           : maybe_pano2_from_pano1.value();

        const Image& image1 = database_cache->Image(image_id1);
        const Image& image2 = database_cache->Image(image_id2);

        const sensor_t camera_id1(SensorType::CAMERA, image1.CameraId());
        Rigid3d cam1_from_rig1;
        if (!rig1.IsRefSensor(camera_id1)) {
          cam1_from_rig1 = rig1.SensorFromRig(camera_id1);
        }

        const sensor_t camera_id2(SensorType::CAMERA, image2.CameraId());
        Rigid3d cam2_from_rig2;
        if (!rig2.IsRefSensor(camera_id2)) {
          cam2_from_rig2 = rig2.SensorFromRig(camera_id2);
        }

        two_view_geometry.cam2_from_cam1 =
            cam2_from_rig2 * rig2_from_rig1 * Inverse(cam1_from_rig1);
        two_view_geometry.config = TwoViewGeometry::CALIBRATED;

        database.DeleteInlierMatches(image_id1, image_id2);
        database.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      }
    });
  }

  thread_pool.Wait();

  return EXIT_SUCCESS;
}

}  // namespace colmap
