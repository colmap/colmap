// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/feature_matching.h"

#include <fstream>
#include <numeric>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "base/camera_models.h"
#include "base/database.h"
#include "base/gps.h"
#include "estimators/essential_matrix.h"
#include "estimators/two_view_geometry.h"
#include "optim/ransac.h"
#include "retrieval/visual_index.h"
#include "util/misc.h"

namespace colmap {
namespace {

void PrintElapsedTime(const Timer& timer) {
  std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
}

void IndexImagesInVisualIndex(const int num_threads, const int max_num_features,
                              const std::vector<image_t>& image_ids,
                              Thread* thread, Database* database,
                              FeatureMatcherCache* cache,
                              retrieval::VisualIndex* visual_index) {
  DatabaseTransaction database_transaction(database);

  retrieval::VisualIndex::IndexOptions index_options;
  index_options.num_threads = num_threads;

  for (size_t i = 0; i < image_ids.size(); ++i) {
    if (thread->IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Indexing image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    retrieval::VisualIndex::Desc descriptors =
        cache->GetDescriptors(image_ids[i]);
    if (descriptors.rows() > max_num_features) {
      const auto keypoints = cache->GetKeypoints(image_ids[i]);
      descriptors =
          ExtractTopScaleDescriptors(keypoints, descriptors, max_num_features);
    }

    visual_index->Add(index_options, image_ids[i], descriptors);

    PrintElapsedTime(timer);
  }

  // Compute the TF-IDF weights, etc.
  visual_index->Prepare();
}

void MatchNearestNeighborsInVisualIndex(
    const int num_threads, const int num_images, const int max_num_features,
    const std::vector<image_t>& image_ids, Thread* thread,
    FeatureMatcherCache* cache, retrieval::VisualIndex* visual_index,
    SiftFeatureMatcher* matcher) {
  // Start a thread pool to retrieve the nearest neighbors.
  ThreadPool retrieval_thread_pool(num_threads);
  JobQueue<std::vector<retrieval::ImageScore>> retrieval_queue(num_threads);

  // The retrieval thread kernel function. Note that the descriptors should be
  // extracted outside of this function sequentially to avoid any concurrent
  // access to the database causing race conditions.
  retrieval::VisualIndex::QueryOptions query_options;
  query_options.max_num_images = num_images;
  auto QueryFunc = [&](retrieval::VisualIndex::Desc descriptors) {
    std::vector<retrieval::ImageScore> image_scores;
    visual_index->Query(query_options, descriptors, &image_scores);
    retrieval_queue.Push(image_scores);
  };

  // Initially, make all retrieval threads busy and continue with the matching.
  size_t image_idx = 0;
  const size_t init_num_tasks =
      std::min(image_ids.size(), 2 * retrieval_thread_pool.NumThreads());
  for (; image_idx < init_num_tasks; ++image_idx) {
    const retrieval::VisualIndex::Desc& descriptors =
        cache->GetDescriptors(image_ids[image_idx]);
    retrieval_thread_pool.AddTask(QueryFunc, descriptors);
  }

  // Pop the finished retrieval results and enqueue them for feature matching.
  for (size_t i = 0; i < image_ids.size(); ++i) {
    if (thread->IsStopped()) {
      retrieval_queue.Stop();
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Matching image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    // Push the next image to the retrieval queue.
    if (image_idx < image_ids.size()) {
      retrieval::VisualIndex::Desc descriptors =
          cache->GetDescriptors(image_ids[image_idx]);
      if (descriptors.rows() > max_num_features) {
        const auto keypoints = cache->GetKeypoints(image_ids[image_idx]);
        descriptors = ExtractTopScaleDescriptors(keypoints, descriptors,
                                                 max_num_features);
      }
      retrieval_thread_pool.AddTask(QueryFunc, descriptors);
      image_idx += 1;
    }

    // Pop the next results from the retrieval queue.
    const auto image_scores = retrieval_queue.Pop();
    CHECK(image_scores.IsValid());

    // Compose the image pairs from the scores.
    std::vector<std::pair<image_t, image_t>> image_pairs;
    image_pairs.reserve(image_scores.Data().size());
    for (const auto image_score : image_scores.Data()) {
      image_pairs.emplace_back(image_ids[i], image_score.image_id);
    }

    matcher->MatchImagePairs(image_pairs);

    PrintElapsedTime(timer);
  }
}

Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter) {
  if (guided_filter != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(keypoints1->size(), descriptors1.rows());
    CHECK_EQ(keypoints2->size(), descriptors2.rows());
  }

  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
      descriptors1.rows(), descriptors2.rows());

  for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
      if (guided_filter != nullptr &&
          guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                        (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
        dists(i1, i2) = 0;
      } else {
        dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
      }
    }
  }

  return dists;
}

size_t FindBestMatchesOneWay(const Eigen::MatrixXi& dists,
                             const float max_ratio, const float max_distance,
                             std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (Eigen::MatrixXi::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const int dist = dists(i1, i2);
      if (dist > best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

void FindBestMatches(const Eigen::MatrixXi& dists, const float max_ratio,
                     const float max_distance, const bool cross_check,
                     FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 =
      FindBestMatchesOneWay(dists, max_ratio, max_distance, &matches12);

  if (cross_check) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWay(
        dists.transpose(), max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == i1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

}  // namespace

void SiftMatchOptions::Check() const {
  CHECK_GE(gpu_index, -1);
  CHECK_GT(max_ratio, 0.0);
  CHECK_GT(max_distance, 0.0);
  CHECK_GT(max_error, 0.0);
  CHECK_GT(max_num_trials, 0);
  CHECK_GE(min_inlier_ratio, 0);
  CHECK_LE(min_inlier_ratio, 1);
  CHECK_GE(min_num_inliers, 0);
}

FeatureMatcherCache::FeatureMatcherCache(const size_t cache_size,
                                         const Database* database)
    : database_(database) {
  CHECK_NOTNULL(database);

  const std::vector<Camera> cameras = database->ReadAllCameras();
  cameras_cache_.reserve(cameras.size());
  for (const auto& camera : cameras) {
    cameras_cache_.emplace(camera.CameraId(), camera);
  }

  const std::vector<Image> images = database->ReadAllImages();
  images_cache_.reserve(images.size());
  for (const auto& image : images) {
    images_cache_.emplace(image.ImageId(), image);
  }

  keypoints_cache_.reset(new LRUCache<image_t, FeatureKeypoints>(
      cache_size, [database](const image_t image_id) {
        return database->ReadKeypoints(image_id);
      }));

  descriptors_cache_.reset(new LRUCache<image_t, FeatureDescriptors>(
      cache_size, [database](const image_t image_id) {
        return database->ReadDescriptors(image_id);
      }));
}

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) const {
  return cameras_cache_.at(camera_id);
}

const Image& FeatureMatcherCache::GetImage(const image_t image_id) const {
  return images_cache_.at(image_id);
}

const FeatureKeypoints& FeatureMatcherCache::GetKeypoints(
    const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return keypoints_cache_->Get(image_id);
}

const FeatureDescriptors& FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return descriptors_cache_->Get(image_id);
}

FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}

std::vector<image_t> FeatureMatcherCache::GetImageIds() const {
  std::vector<image_t> image_ids;
  image_ids.reserve(images_cache_.size());
  for (const auto& image : images_cache_) {
    image_ids.push_back(image.first);
  }
  return image_ids;
}

SiftFeatureMatcher::SiftFeatureMatcher(const SiftMatchOptions& options,
                                       Database* database,
                                       FeatureMatcherCache* cache)
    : options_(options), database_(database), cache_(cache) {
  options_.Check();

  if (options_.use_gpu) {
// Create an OpenGL context.
#ifdef CUDA_ENABLED
    if (options_.gpu_index < 0) {
#endif
      opengl_context_.reset(new OpenGLContextManager());
#ifdef CUDA_ENABLED
    }
#endif

    ClearGPUData();
  }
}

bool SiftFeatureMatcher::Setup() {
  if (options_.use_gpu) {
#ifdef CUDA_ENABLED
    if (options_.gpu_index < 0) {
#endif
      CHECK(opengl_context_);
      opengl_context_->MakeCurrent();
#ifdef CUDA_ENABLED
    }
#endif

    sift_match_gpu_.reset(new SiftMatchGPU());
    if (!CreateSiftGPUMatcher(options_, sift_match_gpu_.get())) {
      opengl_context_.reset();
      sift_match_gpu_.reset();
      std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
      return false;
    }
  }

  thread_pool_.reset(new ThreadPool(options_.num_threads));

  return true;
}

void SiftFeatureMatcher::MatchImagePairs(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  CHECK_NOTNULL(database_);
  CHECK_NOTNULL(cache_);
  CHECK(thread_pool_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Determine image pairs to match
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::pair<bool, bool>> exists_mask;
  exists_mask.reserve(image_pairs.size());
  std::unordered_set<image_pair_t> pair_ids;
  pair_ids.reserve(image_pairs.size());

  bool exists_all = true;

  DatabaseTransaction database_transaction(database_);

  for (const auto image_pair : image_pairs) {
    // Avoid self-matches.
    if (image_pair.first == image_pair.second) {
      exists_mask.emplace_back(true, true);
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id =
        Database::ImagePairToPairId(image_pair.first, image_pair.second);
    if (pair_ids.count(pair_id) > 0) {
      exists_mask.emplace_back(true, true);
      continue;
    }

    pair_ids.insert(pair_id);

    const bool exists_matches =
        database_->ExistsMatches(image_pair.first, image_pair.second);
    const bool exists_inlier_matches =
        database_->ExistsInlierMatches(image_pair.first, image_pair.second);

    exists_all = exists_all && exists_matches && exists_inlier_matches;
    exists_mask.emplace_back(exists_matches, exists_inlier_matches);
  }

  if (exists_all) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Match the image pairs
  //////////////////////////////////////////////////////////////////////////////

  std::vector<MatchResult> match_results;
  std::vector<InlierMatchResult> inlier_match_results;

  if (options_.use_gpu) {
    MatchImagePairsGPU(image_pairs, exists_mask, &match_results,
                       &inlier_match_results);
  } else {
    MatchImagePairsCPU(image_pairs, exists_mask, &match_results,
                       &inlier_match_results);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  for (const auto& result : match_results) {
    database_->WriteMatches(result.image_id1, result.image_id2, result.matches);
  }

  for (const auto& result : inlier_match_results) {
    database_->WriteInlierMatches(result.image_id1, result.image_id2,
                                  result.two_view_geometry);
  }
}

void SiftFeatureMatcher::MatchImagePairsWithPreemptiveFilter(
    const size_t preemptive_num_features,
    const size_t preemptive_min_num_matches,
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  if (image_pairs.empty()) {
    return;
  }

  std::unordered_map<image_t, FeatureDescriptors> top_descriptors;

  image_t prev_image_id1 = kInvalidImageId;
  image_t prev_image_id2 = kInvalidImageId;

  std::unique_ptr<DatabaseTransaction> database_transaction(
      new DatabaseTransaction(database_));

  std::vector<std::pair<image_t, image_t>> filtered_image_pairs;
  for (const auto image_pair : image_pairs) {
    if (top_descriptors.count(image_pair.first) == 0) {
      top_descriptors.emplace(
          image_pair.first,
          ExtractTopScaleDescriptors(cache_->GetKeypoints(image_pair.first),
                                     cache_->GetDescriptors(image_pair.first),
                                     preemptive_num_features));
    }
    if (top_descriptors.count(image_pair.second) == 0) {
      top_descriptors.emplace(
          image_pair.second,
          ExtractTopScaleDescriptors(cache_->GetKeypoints(image_pair.second),
                                     cache_->GetDescriptors(image_pair.second),
                                     preemptive_num_features));
    }

    FeatureMatches preemptive_matches;
    if (options_.use_gpu) {
      const FeatureDescriptors* descriptors1_ptr = nullptr;
      if (image_pair.first != prev_image_id1) {
        descriptors1_ptr = &top_descriptors.at(image_pair.first);
        prev_image_id1 = image_pair.first;
      }

      const FeatureDescriptors* descriptors2_ptr = nullptr;
      if (image_pair.second != prev_image_id2) {
        descriptors2_ptr = &top_descriptors.at(image_pair.second);
        prev_image_id2 = image_pair.second;
      }
      MatchSiftFeaturesGPU(options_, descriptors1_ptr, descriptors2_ptr,
                           sift_match_gpu_.get(), &preemptive_matches);
    } else {
      MatchSiftFeaturesCPU(options_, top_descriptors.at(image_pair.first),
                           top_descriptors.at(image_pair.second),
                           &preemptive_matches);
    }

    if (preemptive_matches.size() >= preemptive_min_num_matches) {
      filtered_image_pairs.push_back(image_pair);
    }
  }

  database_transaction.reset();

  std::cout << StringPrintf(" P(%d/%d)", filtered_image_pairs.size(),
                            image_pairs.size())
            << std::flush;

  MatchImagePairs(filtered_image_pairs);
}

void SiftFeatureMatcher::MatchImagePairsCPU(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    const std::vector<std::pair<bool, bool>>& exists_mask,
    std::vector<MatchResult>* match_results,
    std::vector<InlierMatchResult>* inlier_match_results) {
  CHECK_EQ(image_pairs.size(), exists_mask.size());

  const size_t min_num_inliers = static_cast<size_t>(options_.min_num_inliers);

  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.min_num_inliers =
      static_cast<size_t>(options_.min_num_inliers);
  two_view_geometry_options.ransac_options.max_error = options_.max_error;
  two_view_geometry_options.ransac_options.confidence = options_.confidence;
  two_view_geometry_options.ransac_options.max_num_trials =
      static_cast<size_t>(options_.max_num_trials);
  two_view_geometry_options.ransac_options.min_inlier_ratio =
      options_.min_inlier_ratio;

  match_results->clear();
  match_results->reserve(image_pairs.size());
  inlier_match_results->clear();
  inlier_match_results->reserve(image_pairs.size());

  std::vector<std::future<void>> futures;
  futures.reserve(image_pairs.size());
  std::vector<FeatureMatches> match_results_raw;
  match_results_raw.reserve(image_pairs.size());
  std::vector<TwoViewGeometry> inlier_match_results_raw;
  inlier_match_results_raw.reserve(image_pairs.size());

  for (size_t i = 0; i < image_pairs.size(); ++i) {
    const auto exists = exists_mask[i];
    const auto image_pair = image_pairs[i];

    match_results_raw.emplace_back();
    inlier_match_results_raw.emplace_back();

    FeatureMatches* matches_ptr = &match_results_raw.back();
    TwoViewGeometry* inlier_matches_ptr = &inlier_match_results_raw.back();

    futures.push_back(thread_pool_->AddTask([this, &min_num_inliers,
                                             &two_view_geometry_options, exists,
                                             image_pair, matches_ptr,
                                             inlier_matches_ptr]() {
      if (exists.first && exists.second) {
        return;
      }

      // Feature matching

      if (exists.first) {
        *matches_ptr = cache_->GetMatches(image_pair.first, image_pair.second);
      } else {
        const FeatureDescriptors descriptors1 =
            cache_->GetDescriptors(image_pair.first);
        const FeatureDescriptors descriptors2 =
            cache_->GetDescriptors(image_pair.second);
        MatchSiftFeaturesCPU(options_, descriptors1, descriptors2, matches_ptr);
        if (matches_ptr->size() < min_num_inliers) {
          *matches_ptr = {};
        }
      }

      // Geometric verification.

      if (!exists.second && matches_ptr->size() >= min_num_inliers) {
        GeometricVerificationData data;
        data.camera1 =
            cache_->GetCamera(cache_->GetImage(image_pair.first).CameraId());
        data.camera2 =
            cache_->GetCamera(cache_->GetImage(image_pair.second).CameraId());
        data.keypoints1 = cache_->GetKeypoints(image_pair.first);
        data.keypoints2 = cache_->GetKeypoints(image_pair.second);
        data.matches = *matches_ptr;
        data.options = two_view_geometry_options;
        VerifyImagePair(data, options_, inlier_matches_ptr);
      }

      if (inlier_matches_ptr->inlier_matches.size() >= min_num_inliers &&
          options_.guided_matching) {
        const FeatureKeypoints keypoints1 =
            cache_->GetKeypoints(image_pair.first);
        const FeatureKeypoints keypoints2 =
            cache_->GetKeypoints(image_pair.second);
        const FeatureDescriptors descriptors1 =
            cache_->GetDescriptors(image_pair.first);
        const FeatureDescriptors descriptors2 =
            cache_->GetDescriptors(image_pair.second);
        MatchGuidedSiftFeaturesCPU(options_, keypoints1, keypoints2,
                                   descriptors1, descriptors2,
                                   inlier_matches_ptr);
        if (inlier_matches_ptr->inlier_matches.size() < min_num_inliers) {
          inlier_matches_ptr->inlier_matches = {};
        }
      }
    }));
  }

  CHECK_EQ(image_pairs.size(), futures.size());
  CHECK_EQ(image_pairs.size(), match_results_raw.size());
  CHECK_EQ(image_pairs.size(), inlier_match_results_raw.size());

  for (size_t i = 0; i < image_pairs.size(); ++i) {
    const auto exists = exists_mask[i];
    if (exists.first && exists.second) {
      continue;
    }

    const auto& image_pair = image_pairs[i];

    futures[i].get();

    if (!exists.first) {
      MatchResult match_result;
      match_result.image_id1 = image_pair.first;
      match_result.image_id2 = image_pair.second;
      match_result.matches = match_results_raw[i];
      match_results->push_back(match_result);
    }

    if (!exists.second) {
      InlierMatchResult inlier_match_result;
      inlier_match_result.image_id1 = image_pair.first;
      inlier_match_result.image_id2 = image_pair.second;
      inlier_match_result.two_view_geometry = inlier_match_results_raw[i];
      inlier_match_results->push_back(inlier_match_result);
    }
  }
}

void SiftFeatureMatcher::MatchImagePairsGPU(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    const std::vector<std::pair<bool, bool>>& exists_mask,
    std::vector<MatchResult>* match_results,
    std::vector<InlierMatchResult>* inlier_match_results) {
  CHECK_EQ(image_pairs.size(), exists_mask.size());
  CHECK(sift_match_gpu_);

  ClearGPUData();

  const size_t min_num_inliers = static_cast<size_t>(options_.min_num_inliers);

  match_results->clear();
  match_results->reserve(image_pairs.size());
  inlier_match_results->clear();
  inlier_match_results->reserve(image_pairs.size());

  std::vector<std::future<void>> verification_futures;
  verification_futures.reserve(image_pairs.size());
  std::vector<TwoViewGeometry> verification_results;
  verification_results.reserve(image_pairs.size());
  std::vector<std::pair<image_t, image_t>> verification_image_pairs;
  verification_image_pairs.reserve(image_pairs.size());

  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.min_num_inliers =
      static_cast<size_t>(options_.min_num_inliers);
  two_view_geometry_options.ransac_options.max_error = options_.max_error;
  two_view_geometry_options.ransac_options.confidence = options_.confidence;
  two_view_geometry_options.ransac_options.max_num_trials =
      static_cast<size_t>(options_.max_num_trials);
  two_view_geometry_options.ransac_options.min_inlier_ratio =
      options_.min_inlier_ratio;

  for (size_t i = 0; i < image_pairs.size(); ++i) {
    const auto exists = exists_mask[i];
    if (exists.first && exists.second) {
      continue;
    }

    const auto image_pair = image_pairs[i];
    const image_t image_id1 = image_pair.first;
    const image_t image_id2 = image_pair.second;

    ////////////////////////////////////////////////////////////////////////////
    // Feature matching
    ////////////////////////////////////////////////////////////////////////////

    MatchResult match_result;
    match_result.image_id1 = image_id1;
    match_result.image_id2 = image_id2;

    if (exists.first) {
      // Matches already computed previously. No need to re-compute or write
      // matches. We just need them for geometric verification.
      match_result.matches = cache_->GetMatches(image_id1, image_id2);
    } else {
      const FeatureDescriptors* descriptors1_ptr;
      GetGPUDescriptors(0, image_id1, &descriptors1_ptr);
      const FeatureDescriptors* descriptors2_ptr;
      GetGPUDescriptors(1, image_id2, &descriptors2_ptr);

      MatchSiftFeaturesGPU(options_, descriptors1_ptr, descriptors2_ptr,
                           sift_match_gpu_.get(), &match_result.matches);

      if (match_result.matches.size() < min_num_inliers) {
        match_result.matches = {};
      }

      match_results->push_back(match_result);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Geometric verification
    ////////////////////////////////////////////////////////////////////////////

    if (!exists.second) {
      if (match_result.matches.size() >= min_num_inliers) {
        GeometricVerificationData data;
        data.camera1 =
            cache_->GetCamera(cache_->GetImage(image_id1).CameraId());
        data.camera2 =
            cache_->GetCamera(cache_->GetImage(image_id2).CameraId());
        data.keypoints1 = cache_->GetKeypoints(image_id1);
        data.keypoints2 = cache_->GetKeypoints(image_id2);
        data.matches = match_result.matches;
        data.options = two_view_geometry_options;

        verification_image_pairs.push_back(image_pair);
        verification_results.emplace_back();

        std::function<void(GeometricVerificationData, const SiftMatchOptions&,
                           TwoViewGeometry*)>
            verifier_func = SiftFeatureMatcher::VerifyImagePair;
        verification_futures.push_back(thread_pool_->AddTask(
            verifier_func, data, options_, &verification_results.back()));
      } else {
        InlierMatchResult inlier_match_result;
        inlier_match_result.image_id1 = image_id1;
        inlier_match_result.image_id2 = image_id2;
        inlier_match_results->push_back(inlier_match_result);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Guided matching
  //////////////////////////////////////////////////////////////////////////////

  if (options_.guided_matching) {
    ClearGPUData();
  }

  CHECK_EQ(verification_image_pairs.size(), verification_futures.size());
  CHECK_EQ(verification_image_pairs.size(), verification_results.size());

  for (size_t i = 0; i < verification_results.size(); ++i) {
    const auto& image_pair = verification_image_pairs[i];
    verification_futures[i].get();
    InlierMatchResult inlier_match_result;
    inlier_match_result.image_id1 = image_pair.first;
    inlier_match_result.image_id2 = image_pair.second;
    inlier_match_result.two_view_geometry = verification_results[i];
    if (inlier_match_result.two_view_geometry.inlier_matches.size() >=
            min_num_inliers &&
        options_.guided_matching) {
      const FeatureDescriptors* descriptors1_ptr;
      GetGPUDescriptors(0, image_pair.first, &descriptors1_ptr);
      const FeatureKeypoints* keypoints1_ptr;
      GetGPUKeypoints(0, image_pair.first, descriptors1_ptr, &keypoints1_ptr);
      const FeatureDescriptors* descriptors2_ptr;
      GetGPUDescriptors(1, image_pair.second, &descriptors2_ptr);
      const FeatureKeypoints* keypoints2_ptr;
      GetGPUKeypoints(1, image_pair.second, descriptors2_ptr, &keypoints2_ptr);
      MatchGuidedSiftFeaturesGPU(options_, keypoints1_ptr, keypoints2_ptr,
                                 descriptors1_ptr, descriptors2_ptr,
                                 sift_match_gpu_.get(),
                                 &inlier_match_result.two_view_geometry);
      if (inlier_match_result.two_view_geometry.inlier_matches.size() <
          min_num_inliers) {
        inlier_match_result.two_view_geometry = TwoViewGeometry();
      }
    }
    inlier_match_results->push_back(inlier_match_result);
  }
}

void SiftFeatureMatcher::VerifyImagePair(const GeometricVerificationData data,
                                         const SiftMatchOptions& options,
                                         TwoViewGeometry* two_view_geometry) {
  *two_view_geometry = TwoViewGeometry();

  const auto points1 = FeatureKeypointsToPointsVector(data.keypoints1);
  const auto points2 = FeatureKeypointsToPointsVector(data.keypoints2);

  if (options.multiple_models) {
    two_view_geometry->EstimateMultiple(data.camera1, points1, data.camera2,
                                        points2, data.matches, data.options);
  } else {
    two_view_geometry->Estimate(data.camera1, points1, data.camera2, points2,
                                data.matches, data.options);
  }

  if (two_view_geometry->inlier_matches.size() <
      static_cast<size_t>(options.min_num_inliers)) {
    *two_view_geometry = TwoViewGeometry();
  }
}

void SiftFeatureMatcher::GetGPUKeypoints(
    const int index, const image_t image_id,
    const FeatureDescriptors* const descriptors_ptr,
    const FeatureKeypoints** keypoints_ptr) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  CHECK_EQ(image_id, prev_uploaded_image_ids_[index]);
  if (descriptors_ptr == nullptr) {
    *keypoints_ptr = nullptr;
  } else {
    prev_uploaded_keypoints_[index] = cache_->GetKeypoints(image_id);
    *keypoints_ptr = &prev_uploaded_keypoints_[index];
  }
}

void SiftFeatureMatcher::GetGPUDescriptors(
    const int index, const image_t image_id,
    const FeatureDescriptors** descriptors_ptr) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  if (prev_uploaded_image_ids_[index] == image_id) {
    *descriptors_ptr = nullptr;
  } else {
    prev_uploaded_descriptors_[index] = cache_->GetDescriptors(image_id);
    *descriptors_ptr = &prev_uploaded_descriptors_[index];
    prev_uploaded_image_ids_[index] = image_id;
  }
}

void SiftFeatureMatcher::ClearGPUData() {
  prev_uploaded_image_ids_[0] = kInvalidImageId;
  prev_uploaded_image_ids_[1] = kInvalidImageId;
}

void ExhaustiveFeatureMatcher::Options::Check() const {
  CHECK_GT(block_size, 1);
  CHECK_GT(preemptive_num_features, 0);
  CHECK_GE(preemptive_min_num_matches, 0);
  CHECK_LE(preemptive_min_num_matches, preemptive_num_features);
}

ExhaustiveFeatureMatcher::ExhaustiveFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(2 * options_.block_size, &database_),
      matcher_(match_options, &database_, &cache_) {
  options_.Check();
  match_options_.Check();
}

void ExhaustiveFeatureMatcher::Run() {
  PrintHeading1("Exhaustive feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  const std::vector<image_t> image_ids = cache_.GetImageIds();

  const size_t block_size = static_cast<size_t>(options_.block_size);
  const size_t num_blocks = static_cast<size_t>(
      std::ceil(static_cast<double>(image_ids.size()) / block_size));
  const size_t num_pairs_per_block = block_size * (block_size - 1) / 2;

  for (size_t start_idx1 = 0; start_idx1 < image_ids.size();
       start_idx1 += block_size) {
    const size_t end_idx1 =
        std::min(image_ids.size(), start_idx1 + block_size) - 1;
    for (size_t start_idx2 = 0; start_idx2 < image_ids.size();
         start_idx2 += block_size) {
      const size_t end_idx2 =
          std::min(image_ids.size(), start_idx2 + block_size) - 1;

      if (IsStopped()) {
        GetTimer().PrintMinutes();
        return;
      }

      Timer timer;
      timer.Start();

      std::cout << StringPrintf("Matching block [%d/%d, %d/%d]",
                                start_idx1 / block_size + 1, num_blocks,
                                start_idx2 / block_size + 1, num_blocks)
                << std::flush;

      std::vector<std::pair<image_t, image_t>> image_pairs;
      image_pairs.reserve(num_pairs_per_block);
      for (size_t idx1 = start_idx1; idx1 <= end_idx1; ++idx1) {
        for (size_t idx2 = start_idx2; idx2 <= end_idx2; ++idx2) {
          const size_t block_id1 = idx1 % block_size;
          const size_t block_id2 = idx2 % block_size;
          if ((idx1 > idx2 && block_id1 <= block_id2) ||
              (idx1 < idx2 &&
               block_id1 < block_id2)) {  // Avoid duplicate pairs
            image_pairs.emplace_back(image_ids[idx1], image_ids[idx2]);
          }
        }
      }

      if (options_.preemptive) {
        matcher_.MatchImagePairsWithPreemptiveFilter(
            options_.preemptive_num_features,
            options_.preemptive_min_num_matches, image_pairs);
      } else {
        matcher_.MatchImagePairs(image_pairs);
      }

      PrintElapsedTime(timer);
    }
  }

  GetTimer().PrintMinutes();
}

void SequentialFeatureMatcher::Options::Check() const {
  CHECK_GT(overlap, 0);
  CHECK_GT(loop_detection_period, 0);
  CHECK_GT(loop_detection_num_images, 0);
  CHECK_GT(loop_detection_max_num_features, 0);
  if (loop_detection) {
    CHECK(boost::filesystem::exists(vocab_tree_path));
  }
}

SequentialFeatureMatcher::SequentialFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(std::max(5 * options_.loop_detection_num_images,
                      5 * options_.overlap),
             &database_),
      matcher_(match_options, &database_, &cache_) {
  options_.Check();
  match_options_.Check();
}

void SequentialFeatureMatcher::Run() {
  PrintHeading1("Sequential feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  const std::vector<image_t> ordered_image_ids = GetOrderedImageIds();

  RunSequentialMatching(ordered_image_ids);
  if (options_.loop_detection) {
    RunLoopDetection(ordered_image_ids);
  }

  GetTimer().PrintMinutes();
}

std::vector<image_t> SequentialFeatureMatcher::GetOrderedImageIds() const {
  const std::vector<image_t> image_ids = cache_.GetImageIds();

  std::vector<Image> ordered_images;
  ordered_images.reserve(image_ids.size());
  for (const auto image_id : image_ids) {
    ordered_images.push_back(cache_.GetImage(image_id));
  }

  std::sort(ordered_images.begin(), ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  std::vector<image_t> ordered_image_ids;
  ordered_image_ids.reserve(image_ids.size());
  for (const auto& image : ordered_images) {
    ordered_image_ids.push_back(image.ImageId());
  }

  return ordered_image_ids;
}

void SequentialFeatureMatcher::RunSequentialMatching(
    const std::vector<image_t>& image_ids) {
  for (size_t i = 0; i < image_ids.size(); ++i) {
    if (IsStopped()) {
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Matching image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    const size_t max_image_idx =
        std::min(i + options_.overlap, image_ids.size());
    std::vector<std::pair<image_t, image_t>> image_pairs;
    for (size_t j = i; j < max_image_idx; ++j) {
      image_pairs.emplace_back(image_ids[i], image_ids[j]);
    }

    matcher_.MatchImagePairs(image_pairs);

    PrintElapsedTime(timer);
  }
}

void SequentialFeatureMatcher::RunLoopDetection(
    const std::vector<image_t>& image_ids) {
  // Read the pre-trained vocabulary tree from disk.
  retrieval::VisualIndex visual_index;
  visual_index.Read(options_.vocab_tree_path);

  // Index all images in the visual index.
  IndexImagesInVisualIndex(match_options_.num_threads,
                           options_.loop_detection_max_num_features, image_ids,
                           this, &database_, &cache_, &visual_index);

  if (IsStopped()) {
    return;
  }

  // Only perform loop detection for every n-th image.
  std::vector<image_t> match_image_ids;
  for (size_t i = 0; i < image_ids.size();
       i += options_.loop_detection_period) {
    match_image_ids.push_back(image_ids[i]);
  }
  MatchNearestNeighborsInVisualIndex(
      match_options_.num_threads, options_.loop_detection_num_images,
      options_.loop_detection_max_num_features, match_image_ids, this, &cache_,
      &visual_index, &matcher_);
}

void VocabTreeFeatureMatcher::Options::Check() const {
  CHECK_GT(num_images, 0);
  CHECK_GT(max_num_features, 0);
  CHECK(boost::filesystem::exists(vocab_tree_path));
}

VocabTreeFeatureMatcher::VocabTreeFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(5 * options_.num_images, &database_),
      matcher_(match_options, &database_, &cache_) {
  options_.Check();
  match_options_.Check();
}

void VocabTreeFeatureMatcher::Run() {
  PrintHeading1("Vocabulary tree feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  // Read the pre-trained vocabulary tree from disk.
  retrieval::VisualIndex visual_index;
  visual_index.Read(options_.vocab_tree_path);

  const std::vector<image_t> all_image_ids = cache_.GetImageIds();
  std::vector<image_t> image_ids;
  if (options_.match_list_path == "") {
    image_ids = cache_.GetImageIds();
  } else {
    // Map image names to image identifiers.
    std::unordered_map<std::string, image_t> image_name_to_image_id;
    image_name_to_image_id.reserve(all_image_ids.size());
    for (const auto image_id : all_image_ids) {
      const auto& image = cache_.GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }

    // Read the match list path.
    std::ifstream file(options_.match_list_path);
    CHECK(file.is_open());
    std::string line;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty() || line[0] == '#') {
        continue;
      }

      if (image_name_to_image_id.count(line) == 0) {
        std::cerr << "ERROR: Image " << line << " does not exist." << std::endl;
      } else {
        image_ids.push_back(image_name_to_image_id.at(line));
      }
    }
  }

  // Index all images in the visual index.
  IndexImagesInVisualIndex(match_options_.num_threads,
                           options_.max_num_features, all_image_ids, this,
                           &database_, &cache_, &visual_index);

  if (IsStopped()) {
    GetTimer().PrintMinutes();
    return;
  }

  // Match all images in the visual index.
  MatchNearestNeighborsInVisualIndex(match_options_.num_threads,
                                     options_.num_images,
                                     options_.max_num_features, image_ids, this,
                                     &cache_, &visual_index, &matcher_);

  GetTimer().PrintMinutes();
}

void SpatialFeatureMatcher::Options::Check() const {
  CHECK_GT(max_num_neighbors, 0);
  CHECK_GT(max_distance, 0.0);
}

SpatialFeatureMatcher::SpatialFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(5 * options_.max_num_neighbors, &database_),
      matcher_(match_options, &database_, &cache_) {
  options_.Check();
  match_options_.Check();
}

void SpatialFeatureMatcher::Run() {
  PrintHeading1("Spatial feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  const std::vector<image_t> image_ids = cache_.GetImageIds();

  //////////////////////////////////////////////////////////////////////////////
  // Spatial indexing
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;
  timer.Start();

  std::cout << "Indexing images..." << std::flush;

  GPSTransform gps_transform;

  size_t num_locations = 0;
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
      image_ids.size(), 3);

  std::vector<size_t> location_idxs;
  location_idxs.reserve(image_ids.size());

  std::vector<Eigen::Vector3d> ells(1);

  for (size_t i = 0; i < image_ids.size(); ++i) {
    const auto image_id = image_ids[i];
    const auto& image = cache_.GetImage(image_id);

    if ((image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         options_.ignore_z) ||
        (image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         image.TvecPrior(2) == 0 && !options_.ignore_z)) {
      continue;
    }

    location_idxs.push_back(i);

    if (options_.is_gps) {
      ells[0](0) = image.TvecPrior(0);
      ells[0](1) = image.TvecPrior(1);
      ells[0](2) = options_.ignore_z ? 0 : image.TvecPrior(2);

      const auto xyzs = gps_transform.EllToXYZ(ells);

      location_matrix(num_locations, 0) = static_cast<float>(xyzs[0](0));
      location_matrix(num_locations, 1) = static_cast<float>(xyzs[0](1));
      location_matrix(num_locations, 2) = static_cast<float>(xyzs[0](2));
    } else {
      location_matrix(num_locations, 0) =
          static_cast<float>(image.TvecPrior(0));
      location_matrix(num_locations, 1) =
          static_cast<float>(image.TvecPrior(1));
      location_matrix(num_locations, 2) =
          static_cast<float>(options_.ignore_z ? 0 : image.TvecPrior(2));
    }

    num_locations += 1;
  }

  PrintElapsedTime(timer);

  if (num_locations == 0) {
    std::cout << " => No images with location data." << std::endl;
    GetTimer().PrintMinutes();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Building spatial index
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Building search index..." << std::flush;

  flann::Matrix<float> locations(location_matrix.data(), num_locations,
                                 location_matrix.cols());

  flann::AutotunedIndexParams index_params;
  index_params["target_precision"] = 0.99f;
  flann::AutotunedIndex<flann::L2<float>> search_index(index_params);
  search_index.buildIndex(locations);

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Searching spatial index
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Searching for nearest neighbors..." << std::flush;

  const int knn = std::min<int>(options_.max_num_neighbors, num_locations);

  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      index_matrix(num_locations, knn);
  flann::Matrix<size_t> indices(index_matrix.data(), num_locations, knn);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix(num_locations, knn);
  flann::Matrix<float> distances(distance_matrix.data(), num_locations, knn);

  flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
  if (match_options_.num_threads == ThreadPool::kMaxNumThreads) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = match_options_.num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  search_index.knnSearch(locations, indices, distances, knn, search_params);

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Matching
  //////////////////////////////////////////////////////////////////////////////

  const float max_distance =
      static_cast<float>(options_.max_distance * options_.max_distance);

  for (size_t i = 0; i < num_locations; ++i) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    timer.Restart();

    std::cout << StringPrintf("Matching image [%d/%d]", i + 1, num_locations)
              << std::flush;

    std::vector<std::pair<image_t, image_t>> image_pairs;
    for (int j = 0; j < knn; ++j) {
      // Check if query equals result.
      if (index_matrix(i, j) == i) {
        continue;
      }

      // Since the nearest neighbors are sorted by distance, we can break.
      if (distance_matrix(i, j) > max_distance) {
        break;
      }

      const size_t idx = location_idxs[i];
      const image_t image_id = image_ids.at(idx);
      const size_t nn_idx = location_idxs.at(index_matrix(i, j));
      const image_t nn_image_id = image_ids.at(nn_idx);
      image_pairs.emplace_back(image_id, nn_image_id);
    }

    matcher_.MatchImagePairs(image_pairs);

    PrintElapsedTime(timer);
  }

  GetTimer().PrintMinutes();
}

void ImagePairsFeatureMatcher::Options::Check() const {
  CHECK_GT(block_size, 0);
  CHECK(boost::filesystem::exists(match_list_path));
}

ImagePairsFeatureMatcher::ImagePairsFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(options.block_size, &database_),
      matcher_(match_options, &database_, &cache_) {
  options_.Check();
  match_options_.Check();
}

void ImagePairsFeatureMatcher::Run() {
  PrintHeading1("Custom feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Reading image pairs list
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_map<std::string, image_t> image_name_to_image_id;
  image_name_to_image_id.reserve(cache_.GetImageIds().size());
  for (const auto image_id : cache_.GetImageIds()) {
    const auto& image = cache_.GetImage(image_id);
    image_name_to_image_id.emplace(image.Name(), image_id);
  }

  std::ifstream file(options_.match_list_path);
  CHECK(file.is_open());

  std::string line;
  std::vector<std::pair<image_t, image_t>> image_pairs;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    std::string image_name1;
    std::string image_name2;

    std::getline(line_stream, image_name1, ' ');
    StringTrim(&image_name1);
    std::getline(line_stream, image_name2, ' ');
    StringTrim(&image_name2);

    if (image_name_to_image_id.count(image_name1) == 0) {
      std::cerr << "ERROR: Image " << image_name1 << " does not exist."
                << std::endl;
      continue;
    }
    if (image_name_to_image_id.count(image_name2) == 0) {
      std::cerr << "ERROR: Image " << image_name2 << " does not exist."
                << std::endl;
      continue;
    }

    image_pairs.emplace_back(image_name_to_image_id.at(image_name1),
                             image_name_to_image_id.at(image_name2));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Feature matching
  //////////////////////////////////////////////////////////////////////////////

  const size_t num_match_blocks = image_pairs.size() / options_.block_size + 1;
  std::vector<std::pair<image_t, image_t>> block_image_pairs;
  block_image_pairs.reserve(options_.block_size);

  for (size_t i = 0; i < image_pairs.size(); i += options_.block_size) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Matching block [%d/%d]",
                              i / options_.block_size + 1, num_match_blocks)
              << std::flush;

    const size_t block_end = i + options_.block_size <= image_pairs.size()
                                 ? i + options_.block_size
                                 : image_pairs.size();
    std::vector<std::pair<image_t, image_t>> block_image_pairs;
    block_image_pairs.reserve(options_.block_size);
    for (size_t j = i; j < block_end; ++j) {
      block_image_pairs.push_back(image_pairs[j]);
    }

    matcher_.MatchImagePairs(block_image_pairs);

    PrintElapsedTime(timer);
  }

  GetTimer().PrintMinutes();
}

void FeaturePairsFeatureMatcher::Options::Check() const {
  CHECK(boost::filesystem::exists(match_list_path));
}

FeaturePairsFeatureMatcher::FeaturePairsFeatureMatcher(
    const Options& options, const SiftMatchOptions& match_options,
    const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(kCacheSize, &database_) {
  options_.Check();
  match_options_.Check();
}

void FeaturePairsFeatureMatcher::Run() {
  PrintHeading1("Importing matches");

  std::unordered_map<std::string, const Image*> image_name_to_image;
  image_name_to_image.reserve(cache_.GetImageIds().size());
  for (const auto image_id : cache_.GetImageIds()) {
    const auto& image = cache_.GetImage(image_id);
    image_name_to_image.emplace(image.Name(), &image);
  }

  std::ifstream file(options_.match_list_path);
  CHECK(file.is_open());

  DatabaseTransaction database_transaction(&database_);

  std::string line;
  while (std::getline(file, line)) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
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
      std::cerr << "ERROR: Could not read image pair." << std::endl;
      break;
    }

    std::cout << StringPrintf("%s - %s", image_name1.c_str(),
                              image_name2.c_str())
              << std::endl;

    if (image_name_to_image.count(image_name1) == 0) {
      std::cout << StringPrintf("SKIP: Image %s not found in database.",
                                image_name1.c_str())
                << std::endl;
      break;
    }
    if (image_name_to_image.count(image_name2) == 0) {
      std::cout << StringPrintf("SKIP: Image %s not found in database.",
                                image_name2.c_str())
                << std::endl;
      break;
    }

    const Image& image1 = *image_name_to_image[image_name1];
    const Image& image2 = *image_name_to_image[image_name2];

    bool skip_pair = false;
    if (database_.ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
      std::cout << "SKIP: Matches for image pair already exist in database."
                << std::endl;
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
        std::cerr << "ERROR: Cannot read feature matches." << std::endl;
        break;
      }

      matches.push_back(match);
    }

    if (skip_pair) {
      continue;
    }

    const Camera& camera1 = cache_.GetCamera(image1.CameraId());
    const Camera& camera2 = cache_.GetCamera(image2.CameraId());

    if (options_.verify_matches) {
      database_.WriteMatches(image1.ImageId(), image2.ImageId(), matches);

      const auto keypoints1 = cache_.GetKeypoints(image1.ImageId());
      const auto keypoints2 = cache_.GetKeypoints(image2.ImageId());

      TwoViewGeometry two_view_geometry;
      TwoViewGeometry::Options two_view_geometry_options;
      two_view_geometry_options.min_num_inliers =
          static_cast<size_t>(match_options_.min_num_inliers);
      two_view_geometry_options.ransac_options.max_error =
          match_options_.max_error;
      two_view_geometry_options.ransac_options.confidence =
          match_options_.confidence;
      two_view_geometry_options.ransac_options.max_num_trials =
          static_cast<size_t>(match_options_.max_num_trials);
      two_view_geometry_options.ransac_options.min_inlier_ratio =
          match_options_.min_inlier_ratio;

      two_view_geometry.Estimate(
          camera1, FeatureKeypointsToPointsVector(keypoints1), camera2,
          FeatureKeypointsToPointsVector(keypoints2), matches,
          two_view_geometry_options);

      database_.WriteInlierMatches(image1.ImageId(), image2.ImageId(),
                                   two_view_geometry);
    } else {
      TwoViewGeometry two_view_geometry;

      if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
        two_view_geometry.config = TwoViewGeometry::CALIBRATED;
      } else {
        two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
      }

      two_view_geometry.inlier_matches = matches;

      database_.WriteInlierMatches(image1.ImageId(), image2.ImageId(),
                                   two_view_geometry);
    }
  }

  GetTimer().PrintMinutes();
}

void MatchSiftFeaturesCPU(const SiftMatchOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {
  match_options.Check();
  CHECK_NOTNULL(matches);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      nullptr, nullptr, descriptors1, descriptors2, nullptr);

  FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                  match_options.cross_check, matches);
}

void MatchGuidedSiftFeaturesCPU(const SiftMatchOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry) {
  match_options.Check();
  CHECK_NOTNULL(two_view_geometry);

  const float max_residual = match_options.max_error * match_options.max_error;

  std::function<bool(float, float, float, float)> guided_filter;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector3f p2(x2, y2, 1.0f);
      const Eigen::Vector3f Fx1 = F * p1;
      const Eigen::Vector3f Ftx2 = F.transpose() * p2;
      const float x2tFx1 = p2.transpose() * Fx1;
      return x2tFx1 * x2tFx1 / (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) +
                                Ftx2(0) * Ftx2(0) + Ftx2(1) * Ftx2(1)) >
             max_residual;
    };
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector2f p2(x2, y2);
      return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
    };
  } else {
    two_view_geometry->inlier_matches.clear();
    return;
  }

  CHECK(guided_filter);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      &keypoints1, &keypoints2, descriptors1, descriptors2, guided_filter);

  FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                  match_options.cross_check,
                  &two_view_geometry->inlier_matches);
}

bool CreateSiftGPUMatcher(const SiftMatchOptions& match_options,
                          SiftMatchGPU* sift_match_gpu) {
  match_options.Check();
  CHECK_NOTNULL(sift_match_gpu);

  SiftGPU sift_gpu;
  sift_gpu.SetVerbose(0);

  *sift_match_gpu = SiftMatchGPU(match_options.max_num_matches);

#ifdef CUDA_ENABLED
  if (match_options.gpu_index >= 0) {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                match_options.gpu_index);
  } else {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
  }
#else  // CUDA_ENABLED
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif  // CUDA_ENABLED

  if (sift_match_gpu->VerifyContextGL() == 0) {
    return false;
  }

  return true;
}

void MatchSiftFeaturesGPU(const SiftMatchOptions& match_options,
                          const FeatureDescriptors* descriptors1,
                          const FeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
  match_options.Check();
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(matches);

  if (descriptors1 != nullptr) {
    CHECK_EQ(descriptors1->cols(), 128);
    sift_match_gpu->SetDescriptors(0, descriptors1->rows(),
                                   descriptors1->data());
  }

  if (descriptors2 != nullptr) {
    CHECK_EQ(descriptors2->cols(), 128);
    sift_match_gpu->SetDescriptors(1, descriptors2->rows(),
                                   descriptors2->data());
  }

  matches->resize(static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(matches->data()),
      static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio), match_options.cross_check);

  CHECK_LE(num_matches, matches->size());
  matches->resize(num_matches);
}

void MatchGuidedSiftFeaturesGPU(const SiftMatchOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const FeatureDescriptors* descriptors1,
                                const FeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry) {
  static_assert(sizeof(FeatureKeypoint) == 4 * sizeof(float),
                "Invalid feature keypoint data format");

  match_options.Check();
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(two_view_geometry);

  if (descriptors1 != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_EQ(descriptors1->rows(), keypoints1->size());
    CHECK_EQ(descriptors1->cols(), 128);
    const size_t kIndex = 0;
    sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(),
                                   descriptors1->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints1->data()), 2);
  }

  if (descriptors2 != nullptr) {
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(descriptors2->rows(), keypoints2->size());
    CHECK_EQ(descriptors2->cols(), 128);
    const size_t kIndex = 1;
    sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(),
                                   descriptors2->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints2->data()), 2);
  }

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F;
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
  float* F_ptr = nullptr;
  float* H_ptr = nullptr;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    F = two_view_geometry->F.cast<float>();
    F_ptr = F.data();
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    H = two_view_geometry->H.cast<float>();
    H_ptr = H.data();
  } else {
    two_view_geometry->inlier_matches.clear();
    return;
  }

  CHECK(F_ptr != nullptr || H_ptr != nullptr);

  two_view_geometry->inlier_matches.resize(
      static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(
          two_view_geometry->inlier_matches.data()),
      H_ptr, F_ptr, static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio),
      static_cast<float>(match_options.max_error * match_options.max_error),
      static_cast<float>(match_options.max_error * match_options.max_error),
      match_options.cross_check);

  CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
  two_view_geometry->inlier_matches.resize(num_matches);
}

}  // namespace colmap
