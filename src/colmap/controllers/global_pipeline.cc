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
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/sfm/global_mapper.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>

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

}  // namespace

GlobalPipeline::GlobalPipeline(
    GlobalPipelineOptions options,
    std::shared_ptr<Database> database,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      reconstruction_manager_(
          std::move(THROW_CHECK_NOTNULL(reconstruction_manager))) {
  THROW_CHECK_NOTNULL(database);

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

std::shared_ptr<Reconstruction> GlobalPipeline::RunSingleReconstruction(
    const std::shared_ptr<const DatabaseCache>& database_cache,
    const GlobalMapperOptions& mapper_options) {
  auto reconstruction = std::make_shared<Reconstruction>();

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

  // Note that a stop requested through the callback is reported as success, so
  // this only discards genuinely failed runs. The reconstruction is dropped
  // rather than written out, because the poses left behind by the stages that
  // did complete are not trustworthy either: the structure was filtered away
  // precisely because it disagreed with those poses.
  if (!success) {
    LOG(ERROR) << "Global mapping failed";
    return nullptr;
  }

  // Align reconstruction to the original metric scales in rig extrinsics.
  AlignReconstructionToOrigRigScales(database_cache->Rigs(),
                                     reconstruction.get());

  return reconstruction;
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

  std::vector<std::shared_ptr<Reconstruction>> reconstructions;
  if (options_.reconstruct_all_components) {
    RunMultiComponents(mapper_options, &reconstructions);
  } else {
    reconstructions.push_back(
        RunSingleReconstruction(database_cache_, mapper_options));
  }

  // Sort reconstructions by the number of registered frames (descending) and
  // discard those that failed (null) or are too small. Null entries sort last.
  std::sort(reconstructions.begin(),
            reconstructions.end(),
            [](const std::shared_ptr<Reconstruction>& lhs,
               const std::shared_ptr<Reconstruction>& rhs) {
              const size_t lhs_num = lhs ? lhs->NumRegFrames() : 0;
              const size_t rhs_num = rhs ? rhs->NumRegFrames() : 0;
              return lhs_num > rhs_num;
            });

  size_t num_discarded = 0;
  for (const auto& reconstruction : reconstructions) {
    if (reconstruction == nullptr ||
        static_cast<int>(reconstruction->NumRegFrames()) <
            options_.min_num_frames) {
      ++num_discarded;
      continue;
    }

    // Output the reconstruction.
    Reconstruction& output_reconstruction =
        *reconstruction_manager_->Get(reconstruction_manager_->Add());
    output_reconstruction = *reconstruction;
    if (!options_.image_path.empty()) {
      LOG(INFO) << "Extracting colors ...";
      output_reconstruction.ExtractColorsForAllImages(options_.image_path,
                                                      options_.num_threads);
    }
  }

  LOG(INFO) << "Kept " << reconstruction_manager_->Size()
            << " reconstruction(s), discarded " << num_discarded
            << " with fewer than " << options_.min_num_frames
            << " registered frames";

  if (has_insufficient_prior_focal_lengths) {
    // Intentionally logging this warning before and after the reconstruction
    // to make sure it is not missed.
    WarnInsufficientPriorFocalLengths();
  }
}

void GlobalPipeline::RunMultiComponents(
    const GlobalMapperOptions& mapper_options,
    std::vector<std::shared_ptr<Reconstruction>>* reconstructions) {
  // Build the base reconstruction, pose graph, and pose priors from the cache.
  auto base = std::make_shared<Reconstruction>();
  base->Load(*database_cache_);
  PoseGraph pose_graph;
  pose_graph.Load(*database_cache_->CorrespondenceGraph());
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  if (pose_graph.Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return;
  }

  // Partition the view graph into connected components via rotation averaging.
  ReconstructionManager component_manager;
  *component_manager.Get(component_manager.Add()) = *base;
  if (!RunRotationAveragingMultiComponents(mapper_options.RotationAveraging(),
                                           pose_graph,
                                           component_manager,
                                           pose_priors)) {
    LOG(ERROR) << "Failed to compute connected components";
    return;
  }

  LOG(INFO) << "Found " << component_manager.Size() << " connected component(s)";

  // Run the full pipeline independently per component, restricted to that
  // component's images.
  for (size_t i = 0; i < component_manager.Size(); ++i) {
    const auto component = component_manager.Get(i);

    FlatHashSet<std::string> image_names;
    for (const image_t image_id : component->RegImageIds()) {
      image_names.insert(component->Image(image_id).Name());
    }
    if (image_names.empty()) {
      continue;
    }

    LOG_HEADING1(StringPrintf("Reconstructing component %d / %d with %d images",
                              static_cast<int>(i + 1),
                              static_cast<int>(component_manager.Size()),
                              static_cast<int>(image_names.size())));

    DatabaseCache::Options cache_options;
    cache_options.min_num_matches = options_.min_num_matches;
    cache_options.ignore_watermarks = options_.ignore_watermarks;
    cache_options.image_names = std::move(image_names);
    std::shared_ptr<DatabaseCache> component_cache =
        DatabaseCache::CreateFromCache(*database_cache_, cache_options);

    reconstructions->push_back(
        RunSingleReconstruction(component_cache, mapper_options));
  }
}

}  // namespace colmap
