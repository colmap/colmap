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

#include "colmap/controllers/feature_matching.h"

#include "colmap/controllers/feature_matching_utils.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/utils.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>

namespace colmap {
namespace {

void RigVerification(const std::shared_ptr<Database> database,
                     const std::shared_ptr<FeatureMatcherCache> cache,
                     const TwoViewGeometryOptions& geometry_options,
                     const int num_threads) {
  std::unordered_map<rig_t, Rig> rigs;
  for (auto& rig : database->ReadAllRigs()) {
    rigs[rig.RigId()] = std::move(rig);
  }

  std::unordered_map<image_t, frame_t> image_to_frame_ids;
  for (const auto& frame : database->ReadAllFrames()) {
    for (const data_t& data_id : frame.ImageIds()) {
      image_to_frame_ids[data_id.id] = frame.FrameId();
    }
  }

  std::map<std::pair<frame_t, frame_t>, int> frame_pair_to_num_matches;
  for (const auto& [image_pair_id, num_matches] : database->ReadNumMatches()) {
    if (num_matches == 0) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(image_pair_id);
    frame_t frame_id1 = image_to_frame_ids.at(image_id1);
    frame_t frame_id2 = image_to_frame_ids.at(image_id2);
    if (frame_id1 > frame_id2) {
      std::swap(frame_id1, frame_id2);
    }
    frame_pair_to_num_matches[std::make_pair(frame_id1, frame_id2)] +=
        num_matches;
  }

  ThreadPool thread_pool(num_threads);
  for (const auto& [frame_pair, num_matches] : frame_pair_to_num_matches) {
    if (num_matches < geometry_options.min_num_inliers) {
      continue;
    }
    thread_pool.AddTask([&cache,
                         &rigs,
                         geometry_options,
                         frame_id1 = frame_pair.first,
                         frame_id2 = frame_pair.second]() {
      const Frame& frame1 = cache->GetFrame(frame_id1);
      const Frame& frame2 = cache->GetFrame(frame_id2);
      const Rig& rig1 = rigs.at(frame1.RigId());
      const Rig& rig2 = rigs.at(frame2.RigId());
      if (rig1.NumSensors() == 1 && rig2.NumSensors() == 1) {
        return;
      }

      std::unordered_map<image_t, Image> images;
      images.reserve(frame1.NumDataIds() + frame2.NumDataIds());
      std::unordered_map<camera_t, Camera> cameras;
      cameras.reserve(images.size());
      auto add_images_and_cameras = [&cache, &images, &cameras](
                                        const Frame& frame) {
        for (const data_t& data_id : frame.ImageIds()) {
          Image& image = images[data_id.id];
          image = cache->GetImage(data_id.id);
          image.SetPoints2D(
              FeatureKeypointsToPointsVector(*cache->GetKeypoints(data_id.id)));
          cameras[image.CameraId()] = cache->GetCamera(image.CameraId());
        }
      };
      add_images_and_cameras(frame1);
      add_images_and_cameras(frame2);

      std::vector<std::pair<std::pair<image_t, image_t>, FeatureMatches>>
          matches;
      matches.reserve(frame1.NumDataIds() * frame2.NumDataIds());
      for (const data_t& image_id1 : frame1.ImageIds()) {
        for (const data_t& image_id2 : frame2.ImageIds()) {
          if (!cache->ExistsMatches(image_id1.id, image_id2.id)) {
            continue;
          }
          matches.emplace_back(std::make_pair(image_id1.id, image_id2.id),
                               cache->GetMatches(image_id1.id, image_id2.id));
        }
      }

      for (const auto& [image_pair, two_view_geometry] :
           EstimateRigTwoViewGeometries(
               rig1, rig2, images, cameras, matches, geometry_options)) {
        const auto& [image_id1, image_id2] = image_pair;
        cache->DeleteInlierMatches(image_id1, image_id2);
        cache->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      }
    });
  }

  thread_pool.Wait();
}

class FeatureMatcherThread : public Thread {
 public:
  template <typename PairGeneratorType>
  static std::unique_ptr<Thread> Create(
      const typename PairGeneratorType::PairingOptions& pairing_options,
      const FeatureMatchingOptions& matching_options,
      const TwoViewGeometryOptions& geometry_options,
      const std::string& database_path) {
    auto database = Database::Open(database_path);
    auto cache = std::make_shared<FeatureMatcherCache>(
        pairing_options.CacheSize(), database);
    return std::make_unique<FeatureMatcherThread>(
        matching_options,
        geometry_options,
        database,
        cache,
        [pairing_options, cache]() {
          return std::make_unique<PairGeneratorType>(pairing_options, cache);
        });
  }

  using PairGeneratorFactory = std::function<std::unique_ptr<PairGenerator>()>;

  FeatureMatcherThread(const FeatureMatchingOptions& matching_options,
                       const TwoViewGeometryOptions& geometry_options,
                       std::shared_ptr<Database> database,
                       std::shared_ptr<FeatureMatcherCache> cache,
                       PairGeneratorFactory pair_generator_factory)
      : matching_options_(matching_options),
        geometry_options_(geometry_options),
        database_(std::move(database)),
        cache_(std::move(cache)),
        pair_generator_factory_(std::move(pair_generator_factory)),
        matcher_(matching_options, geometry_options, cache_) {
    THROW_CHECK(matching_options.Check());
    THROW_CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Feature matching & geometric verification");

    Timer run_timer;
    run_timer.Start();

    if (!matcher_.Setup()) {
      return;
    }

    std::unique_ptr<PairGenerator> pair_generator =
        THROW_CHECK_NOTNULL(pair_generator_factory_());

    while (!pair_generator->HasFinished()) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }
      Timer timer;
      timer.Start();
      const std::vector<std::pair<image_t, image_t>> image_pairs =
          pair_generator->Next();
      matcher_.Match(image_pairs);
      LOG(INFO) << StringPrintf("in %.3fs", timer.ElapsedSeconds());
    }

    run_timer.PrintMinutes();

    // Notice that we run rig verification after feature matching, because
    // feature matching operates on pairs of images instead of pairs of frames.
    // Rig verification operates on pairs of frames and we require all image
    // pairs between two frames to be matched before running rig verification.
    if (matching_options_.rig_verification) {
      run_timer.Restart();
      PrintHeading1("Rig verification");
      RigVerification(
          database_, cache_, geometry_options_, matching_options_.num_threads);
      run_timer.PrintMinutes();
    }
  }

  const FeatureMatchingOptions matching_options_;
  const TwoViewGeometryOptions geometry_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  const PairGeneratorFactory pair_generator_factory_;
  FeatureMatcherController matcher_;
};

class FeatureMatchesVerifierThread : public Thread {
 public:
  template <typename PairGeneratorType>
  static std::unique_ptr<Thread> Create(
      const typename PairGeneratorType::PairingOptions& pairing_options,
      const TwoViewGeometryOptions& geometry_options,
      const std::string& database_path,
      const VerifierOptions& options) {
    auto database = Database::Open(database_path);
    auto cache = std::make_shared<FeatureMatcherCache>(
        pairing_options.CacheSize(), database);
    return std::make_unique<FeatureMatchesVerifierThread>(
        geometry_options,
        database,
        cache,
        [pairing_options, cache]() {
          return std::make_unique<PairGeneratorType>(pairing_options, cache);
        },
        options);
  }

  using PairGeneratorFactory = std::function<std::unique_ptr<PairGenerator>()>;

  FeatureMatchesVerifierThread(const TwoViewGeometryOptions& geometry_options,
                               std::shared_ptr<Database> database,
                               std::shared_ptr<FeatureMatcherCache> cache,
                               PairGeneratorFactory pair_generator_factory,
                               const VerifierOptions& options)
      : geometry_options_(geometry_options),
        database_(std::move(database)),
        cache_(std::move(cache)),
        pair_generator_factory_(std::move(pair_generator_factory)),
        verifier_(geometry_options, cache_, options) {
    THROW_CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Geometric verification");

    Timer run_timer;
    run_timer.Start();

    if (!verifier_.Setup()) {
      return;
    }

    std::unique_ptr<PairGenerator> pair_generator =
        THROW_CHECK_NOTNULL(pair_generator_factory_());

    while (!pair_generator->HasFinished()) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }
      Timer timer;
      timer.Start();
      const std::vector<std::pair<image_t, image_t>> image_pairs =
          pair_generator->Next();
      verifier_.Verify(image_pairs);
      LOG(INFO) << StringPrintf("in %.3fs", timer.ElapsedSeconds());
    }

    if (verifier_.Options().rig_verification) {
      run_timer.Restart();
      PrintHeading1("Rig verification");
      RigVerification(database_,
                      cache_,
                      geometry_options_,
                      verifier_.Options().num_threads);
      run_timer.PrintMinutes();
    }

    run_timer.PrintMinutes();
  }

  const TwoViewGeometryOptions geometry_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
  const PairGeneratorFactory pair_generator_factory_;
  FeatureMatchesVerifierController verifier_;
};

}  // namespace

std::unique_ptr<Thread> CreateExhaustiveFeatureMatcher(
    const ExhaustivePairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<ExhaustivePairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateVocabTreeFeatureMatcher(
    const VocabTreePairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<VocabTreePairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateSequentialFeatureMatcher(
    const SequentialPairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<SequentialPairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateSpatialFeatureMatcher(
    const SpatialPairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<SpatialPairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateTransitiveFeatureMatcher(
    const TransitivePairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<TransitivePairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateImagePairsFeatureMatcher(
    const ImportedPairingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return FeatureMatcherThread::Create<ImportedPairGenerator>(
      pairing_options, matching_options, geometry_options, database_path);
}

namespace {

class FeaturePairsFeatureMatcher : public Thread {
 public:
  FeaturePairsFeatureMatcher(const FeaturePairsMatchingOptions& pairing_options,
                             const FeatureMatchingOptions& matching_options,
                             const TwoViewGeometryOptions& geometry_options,
                             const std::string& database_path)
      : options_(pairing_options),
        matching_options_(matching_options),
        geometry_options_(geometry_options),
        database_(Database::Open(database_path)),
        cache_(std::make_shared<FeatureMatcherCache>(/*cache_size=*/100,
                                                     database_)) {
    THROW_CHECK(pairing_options.Check());
    THROW_CHECK(matching_options.Check());
    THROW_CHECK(geometry_options.Check());
  }

 private:
  void Run() override {
    PrintHeading1("Importing matches");
    Timer run_timer;
    run_timer.Start();

    std::unordered_map<std::string, const Image*> image_name_to_image;
    image_name_to_image.reserve(cache_->GetImageIds().size());
    for (const auto image_id : cache_->GetImageIds()) {
      const auto& image = cache_->GetImage(image_id);
      image_name_to_image.emplace(image.Name(), &image);
    }

    std::ifstream file(options_.match_list_path);
    THROW_CHECK_FILE_OPEN(file, options_.match_list_path);

    std::string line;
    while (std::getline(file, line)) {
      if (IsStopped()) {
        run_timer.PrintMinutes();
        return;
      }

      StringTrim(&line);
      if (line.empty()) {
        continue;
      }

      std::istringstream line_stream(line);

      std::string image_name1, image_name2;
      try {
        line_stream >> image_name1 >> image_name2;
      } catch (...) {
        LOG(ERROR) << "Could not read image pair.";
        break;
      }

      LOG(INFO) << StringPrintf(
          "%s - %s", image_name1.c_str(), image_name2.c_str());

      if (image_name_to_image.count(image_name1) == 0) {
        LOG(INFO) << StringPrintf("SKIP: Image %s not found in database.",
                                  image_name1.c_str());
        break;
      }
      if (image_name_to_image.count(image_name2) == 0) {
        LOG(INFO) << StringPrintf("SKIP: Image %s not found in database.",
                                  image_name2.c_str());
        break;
      }

      const Image& image1 = *image_name_to_image[image_name1];
      const Image& image2 = *image_name_to_image[image_name2];

      bool skip_pair = false;
      if (database_->ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
        LOG(INFO) << "SKIP: Matches for image pair already exist in database.";
        skip_pair = true;
      }

      FeatureMatches matches;
      while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty()) {
          break;
        }

        std::istringstream line_stream(line);

        FeatureMatch match;
        try {
          line_stream >> match.point2D_idx1 >> match.point2D_idx2;
        } catch (...) {
          LOG(ERROR) << "Cannot read feature matches.";
          break;
        }

        matches.push_back(match);
      }

      if (skip_pair) {
        continue;
      }

      const Camera& camera1 = cache_->GetCamera(image1.CameraId());
      const Camera& camera2 = cache_->GetCamera(image2.CameraId());

      TwoViewGeometry two_view_geometry;
      if (options_.verify_matches) {
        database_->WriteMatches(image1.ImageId(), image2.ImageId(), matches);

        const std::shared_ptr<FeatureKeypoints> keypoints1 =
            cache_->GetKeypoints(image1.ImageId());
        const std::shared_ptr<FeatureKeypoints> keypoints2 =
            cache_->GetKeypoints(image2.ImageId());

        two_view_geometry =
            EstimateTwoViewGeometry(camera1,
                                    FeatureKeypointsToPointsVector(*keypoints1),
                                    camera2,
                                    FeatureKeypointsToPointsVector(*keypoints2),
                                    std::move(matches),
                                    geometry_options_);

      } else {
        if (camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
          two_view_geometry.config = TwoViewGeometry::CALIBRATED;
        } else {
          two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
        }
        two_view_geometry.inlier_matches = std::move(matches);
      }

      database_->WriteTwoViewGeometry(
          image1.ImageId(), image2.ImageId(), two_view_geometry);
    }

    run_timer.PrintMinutes();
  }

  const FeaturePairsMatchingOptions options_;
  const FeatureMatchingOptions matching_options_;
  const TwoViewGeometryOptions geometry_options_;
  const std::shared_ptr<Database> database_;
  const std::shared_ptr<FeatureMatcherCache> cache_;
};

}  // namespace

std::unique_ptr<Thread> CreateFeaturePairsFeatureMatcher(
    const FeaturePairsMatchingOptions& pairing_options,
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path) {
  return std::make_unique<FeaturePairsFeatureMatcher>(
      pairing_options, matching_options, geometry_options, database_path);
}

std::unique_ptr<Thread> CreateGeometricVerifier(
    const ExistingMatchedPairingOptions& pairing_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::string& database_path,
    const VerifierOptions& options) {
  return FeatureMatchesVerifierThread::Create<ExistingMatchedPairGenerator>(
      pairing_options, geometry_options, database_path, options);
}

}  // namespace colmap
