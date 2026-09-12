// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"

#include <filesystem>

namespace colmap {

void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const IncrementalPipelineOptions& options,
    bool clear_points,
    bool refine_intrinsics,
    std::function<bool()> check_if_stopped = {});

bool RunIncrementalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<IncrementalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager,
    std::function<void()> initial_image_pair_callback = {},
    std::function<void()> next_image_callback = {},
    std::function<bool()> check_if_stopped = {});

bool RunGlobalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<GlobalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager);

bool RunHierarchicalMapperImpl(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path,
    const std::shared_ptr<HierarchicalPipelineOptions>& mapper_options,
    std::shared_ptr<ReconstructionManager>& reconstruction_manager);

int RunAutomaticReconstructor(int argc, char** argv);
int RunBundleAdjuster(int argc, char** argv);
int RunColorExtractor(int argc, char** argv);
int RunMapper(int argc, char** argv);
int RunGlobalMapper(int argc, char** argv);
int RunHierarchicalMapper(int argc, char** argv);
int RunPosePriorMapper(int argc, char** argv);
int RunPointFiltering(int argc, char** argv);
int RunPointTriangulator(int argc, char** argv);
int RunRotationAverager(int argc, char** argv);
int RunViewGraphCalibrator(int argc, char** argv);

}  // namespace colmap
