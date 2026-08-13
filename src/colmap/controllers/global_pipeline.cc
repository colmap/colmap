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

#include "colmap/controllers/global_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <deque>

namespace colmap {
namespace {

constexpr double kMinPriorFocalLengthRatio = 0.5;

bool HasInsufficientPriorFocalLengths(const DatabaseCache& database_cache) {
  const auto& cameras = database_cache.Cameras();
  if (cameras.empty()) {
    return false;
  }
  const size_t num_with_prior =
      std::count_if(cameras.begin(), cameras.end(), [](const auto& camera) {
        return camera.second.has_prior_focal_length;
      });
  return num_with_prior < kMinPriorFocalLengthRatio * cameras.size();
}

void WarnInsufficientPriorFocalLengths() {
  LOG(WARNING) << "Less than " << kMinPriorFocalLengthRatio * 100
               << "% of cameras have prior focal lengths. The "
                  "global mapper depends on reasonably good focal length "
                  "priors to perform well. Consider running "
                  "'colmap view_graph_calibrator' before 'colmap "
                  "global_mapper' or providing camera calibrations "
                  "manually.";
}

size_t NumFramesForImages(const Reconstruction& reconstruction,
                          const FlatHashSet<image_t>& image_ids) {
  FlatHashSet<frame_t> frame_ids;
  frame_ids.reserve(image_ids.size());
  for (const image_t image_id : image_ids) {
    frame_ids.insert(reconstruction.Image(image_id).FrameId());
  }
  return frame_ids.size();
}

}  // namespace

GlobalPipeline::GlobalPipeline(
    GlobalPipelineOptions options,
    std::shared_ptr<Database> database,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {
  THROW_CHECK_NOTNULL(database);
  THROW_CHECK_GE(options_.min_num_frames, 0);

  // Create database cache with relative poses for pose graph.
  DatabaseCache::Options database_cache_options;
  database_cache_options.min_num_matches = options_.min_num_matches;
  database_cache_options.ignore_watermarks = options_.ignore_watermarks;
  database_cache_options.image_names = {options_.image_names.begin(),
                                        options_.image_names.end()};
  database_cache_ = DatabaseCache::Create(*database, database_cache_options);
  if (options_.decompose_relative_pose) {
    MaybeDecomposeRelativePoses(database_cache_.get());
  }

  RegisterCallback(MODEL_UPDATE_CALLBACK);
}

GlobalPipeline::ReconstructionResult GlobalPipeline::ReconstructSingleComponent(
    const std::shared_ptr<const DatabaseCache>& database_cache,
    const GlobalMapperOptions& mapper_options) {
  auto reconstruction =
      reconstruction_manager_->Get(reconstruction_manager_->Add());

  GlobalMapper global_mapper(database_cache);
  global_mapper.BeginReconstruction(reconstruction);

  Timer run_timer;
  run_timer.Start();
  const bool success = global_mapper.Solve(mapper_options, [this]() {
    Callback(MODEL_UPDATE_CALLBACK);
    return CheckIfStopped();
  });
  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  // A stop requested through the callback is reported as success, so false
  // only denotes a genuine mapping failure. Return the failed reconstruction
  // temporarily so multi-component orchestration can skip its registered
  // frames while retrying the remainder; the caller removes it from the output
  // manager.
  if (!success) {
    LOG(ERROR) << "Global mapping failed";
    return {std::move(reconstruction), false};
  }

  // Align reconstruction to the original metric scales in rig extrinsics.
  AlignReconstructionToOrigRigScales(database_cache->Rigs(),
                                     reconstruction.get());

  return {std::move(reconstruction), true};
}

void GlobalPipeline::Run() {
  const bool has_insufficient_prior_focal_lengths =
      HasInsufficientPriorFocalLengths(*database_cache_);
  if (has_insufficient_prior_focal_lengths) {
    WarnInsufficientPriorFocalLengths();
  }

  // Prepare mapper options with top-level options.
  GlobalMapperOptions mapper_options = options_.mapper;
  mapper_options.image_path = options_.image_path;
  mapper_options.num_threads = options_.num_threads;
  mapper_options.random_seed = options_.random_seed;

  const size_t first_reconstruction_idx = reconstruction_manager_->Size();
  ReconstructionStats stats;
  if (options_.reconstruct_all_components) {
    stats = ReconstructMultiComponents(mapper_options);
  } else {
    const ReconstructionResult result =
        ReconstructSingleComponent(database_cache_, mapper_options);
    if (!result.success) {
      reconstruction_manager_->Delete(reconstruction_manager_->Size() - 1);
      ++stats.num_failed;
    } else if (static_cast<int>(result.reconstruction->NumRegFrames()) <
               options_.min_num_frames) {
      reconstruction_manager_->Delete(reconstruction_manager_->Size() - 1);
      ++stats.num_too_small;
    }
  }

  // Sort newly created reconstructions by registered frame count. Keep any
  // reconstructions that were already managed before this run untouched.
  std::vector<std::shared_ptr<Reconstruction>> reconstructions;
  reconstructions.reserve(reconstruction_manager_->Size() -
                          first_reconstruction_idx);
  for (size_t i = first_reconstruction_idx; i < reconstruction_manager_->Size();
       ++i) {
    reconstructions.push_back(reconstruction_manager_->Get(i));
  }
  std::sort(reconstructions.begin(),
            reconstructions.end(),
            [](const std::shared_ptr<Reconstruction>& lhs,
               const std::shared_ptr<Reconstruction>& rhs) {
              return lhs->NumRegFrames() > rhs->NumRegFrames();
            });
  for (size_t i = 0; i < reconstructions.size(); ++i) {
    reconstruction_manager_->Get(first_reconstruction_idx + i) =
        std::move(reconstructions[i]);
  }

  for (size_t i = first_reconstruction_idx; i < reconstruction_manager_->Size();
       ++i) {
    if (!options_.image_path.empty()) {
      LOG(INFO) << "Extracting colors ...";
      reconstruction_manager_->Get(i)->ExtractColorsForAllImages(
          options_.image_path, options_.num_threads);
    }
  }

  LOG(INFO) << "Kept "
            << reconstruction_manager_->Size() - first_reconstruction_idx
            << " reconstruction(s), discarded " << stats.num_too_small
            << " with fewer than " << options_.min_num_frames
            << " registered frames, and failed to reconstruct "
            << stats.num_failed;

  if (has_insufficient_prior_focal_lengths) {
    // Intentionally logging this warning before and after the reconstruction
    // to make sure it is not missed.
    WarnInsufficientPriorFocalLengths();
  }
}

GlobalPipeline::ReconstructionStats GlobalPipeline::ReconstructMultiComponents(
    const GlobalMapperOptions& mapper_options) {
  ReconstructionStats stats;

  // Build the base reconstruction and pose graph from the cache.
  Reconstruction base;
  base.Load(*database_cache_);
  PoseGraph pose_graph;
  pose_graph.Load(*database_cache_->CorrespondenceGraph());
  if (pose_graph.Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return stats;
  }

  // Partition the unfiltered input view graph. The full mapper may split these
  // components further when it filters relative rotations below.
  std::vector<FlatHashSet<image_t>> components =
      pose_graph.ComputeConnectedComponentImageIds(
          base, /*filter_unregistered=*/false);

  LOG(INFO) << "Found " << components.size() << " connected component(s)";

  // Run the full pipeline independently per input component. The mapper keeps
  // the largest connected component after rotation filtering. Remove its
  // registered frames, decompose the remainder, and enqueue every resulting
  // component. This uses the mapper's actual two-pass rotation averaging and
  // does not require a disposable preliminary solve.
  std::deque<FlatHashSet<image_t>> pending_components;
  for (auto& component : components) {
    pending_components.push_back(std::move(component));
  }
  size_t component_idx = 0;
  while (!pending_components.empty()) {
    if (CheckIfStopped()) {
      return stats;
    }

    FlatHashSet<image_t> image_ids = std::move(pending_components.front());
    pending_components.pop_front();
    ++component_idx;

    if (static_cast<int>(NumFramesForImages(base, image_ids)) <
        options_.min_num_frames) {
      ++stats.num_too_small;
      continue;
    }

    LOG_HEADING1(
        StringPrintf("Reconstructing component %d with %d images (%d pending)",
                     static_cast<int>(component_idx),
                     static_cast<int>(image_ids.size()),
                     static_cast<int>(pending_components.size())));

    DatabaseCache::Options cache_options;
    cache_options.image_names.reserve(image_ids.size());
    for (const image_t image_id : image_ids) {
      cache_options.image_names.insert(base.Image(image_id).Name());
    }
    const std::shared_ptr<DatabaseCache> component_cache =
        DatabaseCache::CreateFromCache(*database_cache_, cache_options);

    const ReconstructionResult result =
        ReconstructSingleComponent(component_cache, mapper_options);
    const std::shared_ptr<Reconstruction>& reconstruction =
        result.reconstruction;

    const std::vector<frame_t> reg_frame_ids = reconstruction->RegFrameIds();
    const FlatHashSet<frame_t> registered_frame_ids(reg_frame_ids.begin(),
                                                    reg_frame_ids.end());
    FlatHashSet<image_t> remaining_image_ids;
    remaining_image_ids.reserve(image_ids.size());
    for (const image_t image_id : image_ids) {
      if (registered_frame_ids.count(base.Image(image_id).FrameId()) == 0) {
        remaining_image_ids.insert(image_id);
      }
    }

    if (remaining_image_ids.size() == image_ids.size()) {
      LOG(ERROR) << "Global mapping made no registration progress";
      reconstruction_manager_->Delete(reconstruction_manager_->Size() - 1);
      ++stats.num_failed;
      continue;
    }

    if (!result.success) {
      reconstruction_manager_->Delete(reconstruction_manager_->Size() - 1);
      ++stats.num_failed;
    } else if (static_cast<int>(reconstruction->NumRegFrames()) <
               options_.min_num_frames) {
      reconstruction_manager_->Delete(reconstruction_manager_->Size() - 1);
      ++stats.num_too_small;
    }

    if (CheckIfStopped()) {
      return stats;
    }

    if (!remaining_image_ids.empty()) {
      PoseGraph remaining_pose_graph = pose_graph;
      remaining_pose_graph.InvalidatePairsOutsideActiveImageIds(
          remaining_image_ids);
      std::vector<FlatHashSet<image_t>> remaining_components =
          remaining_pose_graph.ComputeConnectedComponentImageIds(
              base, /*filter_unregistered=*/false);
      if (remaining_components.empty()) {
        ++stats.num_too_small;
      } else {
        for (auto& component : remaining_components) {
          pending_components.push_back(std::move(component));
        }
      }
    }
  }

  return stats;
}

}  // namespace colmap
