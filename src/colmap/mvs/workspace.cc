// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/mvs/workspace.h"

#include "colmap/util/threading.h"

#include <numeric>

namespace colmap {
namespace mvs {

Workspace::Workspace(const Options& options) : options_(options) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.workspace_format);
  if (options_.max_image_size > 0) {
    for (auto& image : model_.images) {
      image.Downsize(options_.max_image_size, options_.max_image_size);
    }
  }

  depth_map_path_ = EnsureTrailingSlash(
      JoinPaths(options_.workspace_path, options_.stereo_folder, "depth_maps"));
  normal_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path, options_.stereo_folder, "normal_maps"));
}

std::string Workspace::GetFileName(const int image_idx) const {
  const auto& image_name = model_.GetImageName(image_idx);
  return StringPrintf(
      "%s.%s.bin", image_name.c_str(), options_.input_type.c_str());
}

void Workspace::Load(const std::vector<std::string>& image_names) {
  const size_t num_images = model_.images.size();
  bitmaps_.resize(num_images);
  depth_maps_.resize(num_images);
  normal_maps_.resize(num_images);

  auto LoadWorkspaceData = [&, this](const int image_idx) {
    const size_t width = model_.images.at(image_idx).GetWidth();
    const size_t height = model_.images.at(image_idx).GetHeight();

    // Read and rescale bitmap
    bitmaps_[image_idx] = std::make_unique<Bitmap>();
    bitmaps_[image_idx]->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      bitmaps_[image_idx]->Rescale((int)width, (int)height);
    }

    // Read and rescale depth map
    depth_maps_[image_idx] = std::make_unique<DepthMap>();
    depth_maps_[image_idx]->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      depth_maps_[image_idx]->Downsize(width, height);
    }

    // Read and rescale normal map
    normal_maps_[image_idx] = std::make_unique<NormalMap>();
    normal_maps_[image_idx]->Read(GetNormalMapPath(image_idx));
    if (options_.max_image_size > 0) {
      normal_maps_[image_idx]->Downsize(width, height);
    }
  };

  const int num_threads = GetEffectiveNumThreads(options_.num_threads);
  ThreadPool thread_pool(num_threads);
  Timer timer;
  timer.Start();

  LOG(INFO) << StringPrintf("Loading workspace data with %d threads...",
                            num_threads);
  for (size_t i = 0; i < image_names.size(); ++i) {
    const int image_idx = model_.GetImageIdx(image_names[i]);
    if (HasBitmap(image_idx) && HasDepthMap(image_idx)) {
      thread_pool.AddTask(LoadWorkspaceData, image_idx);
    } else {
      LOG(WARNING) << StringPrintf(
          "Ignoring image %s, because input does not exist.",
          image_names[i].c_str());
    }
  }
  thread_pool.Wait();
  timer.PrintMinutes();
}

const Bitmap& Workspace::GetBitmap(const int image_idx) {
  return *bitmaps_[image_idx];
}

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
  return *depth_maps_[image_idx];
}

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
  return *normal_maps_[image_idx];
}

std::string Workspace::GetBitmapPath(const int image_idx) const {
  return model_.images.at(image_idx).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_idx) const {
  return depth_map_path_ + GetFileName(image_idx);
}

std::string Workspace::GetNormalMapPath(const int image_idx) const {
  return normal_map_path_ + GetFileName(image_idx);
}

bool Workspace::HasBitmap(const int image_idx) const {
  return ExistsFile(GetBitmapPath(image_idx));
}

bool Workspace::HasDepthMap(const int image_idx) const {
  return ExistsFile(GetDepthMapPath(image_idx));
}

bool Workspace::HasNormalMap(const int image_idx) const {
  return ExistsFile(GetNormalMapPath(image_idx));
}

CachedWorkspace::CachedImage::CachedImage(CachedImage&& other) noexcept {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
}

CachedWorkspace::CachedImage& CachedWorkspace::CachedImage::operator=(
    CachedImage&& other) noexcept {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
  }
  return *this;
}

CachedWorkspace::CachedWorkspace(const Options& options)
    : Workspace(options),
      cache_((size_t)(1024.0 * 1024.0 * 1024.0 * options.cache_size),
             [](const int) { return CachedImage(); }) {}

const Bitmap& CachedWorkspace::GetBitmap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.bitmap) {
    cached_image.bitmap = std::make_unique<Bitmap>();
    cached_image.bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      cached_image.bitmap->Rescale(model_.images.at(image_idx).GetWidth(),
                                   model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.bitmap;
}

const DepthMap& CachedWorkspace::GetDepthMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.depth_map) {
    cached_image.depth_map = std::make_unique<DepthMap>();
    cached_image.depth_map->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.depth_map->Downsize(model_.images.at(image_idx).GetWidth(),
                                       model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.depth_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.depth_map;
}

const NormalMap& CachedWorkspace::GetNormalMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.normal_map) {
    cached_image.normal_map = std::make_unique<NormalMap>();
    cached_image.normal_map->Read(GetNormalMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.normal_map->Downsize(
          model_.images.at(image_idx).GetWidth(),
          model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.normal_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.normal_map;
}

void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name) {
  const std::string& workspace_path = workspace.GetOptions().workspace_path;
  const std::string& stereo_folder = workspace.GetOptions().stereo_folder;

  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, "depth_maps"));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, "normal_maps"));
  CreateDirIfNotExists(
      JoinPaths(workspace_path, stereo_folder, "consistency_graphs"));

  const auto option_lines =
      ReadTextFileLines(JoinPaths(workspace_path, option_name));
  for (const auto& line : option_lines) {
    if (!StringStartsWith(line, "timages")) {
      continue;
    }

    const auto elems = StringSplit(line, " ");
    int num_images = std::stoull(elems[1]);

    std::vector<int> image_idxs;
    if (num_images == -1) {
      THROW_CHECK_EQ(elems.size(), 4);
      const int range_lower = std::stoull(elems[2]);
      const int range_upper = std::stoull(elems[3]);
      THROW_CHECK_LT(range_lower, range_upper);
      num_images = range_upper - range_lower;
      image_idxs.resize(num_images);
      std::iota(image_idxs.begin(), image_idxs.end(), range_lower);
    } else {
      THROW_CHECK_EQ(num_images + 2, elems.size());
      image_idxs.reserve(num_images);
      for (size_t i = 2; i < elems.size(); ++i) {
        const int image_idx = std::stoull(elems[i]);
        image_idxs.push_back(image_idx);
      }
    }

    std::vector<std::string> image_names;
    image_names.reserve(num_images);
    for (const auto image_idx : image_idxs) {
      const std::string image_name =
          workspace.GetModel().GetImageName(image_idx);
      image_names.push_back(image_name);
    }

    const auto& overlapping_images =
        workspace.GetModel().GetMaxOverlappingImagesFromPMVS();

    const auto patch_match_path =
        JoinPaths(workspace_path, stereo_folder, "patch-match.cfg");
    const auto fusion_path =
        JoinPaths(workspace_path, stereo_folder, "fusion.cfg");
    std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
    std::ofstream fusion_file(fusion_path, std::ios::trunc);
    THROW_CHECK_FILE_OPEN(patch_match_file, patch_match_path);
    THROW_CHECK_FILE_OPEN(fusion_file, fusion_path);
    for (size_t i = 0; i < image_names.size(); ++i) {
      const auto& ref_image_name = image_names[i];
      patch_match_file << ref_image_name << "\n";
      if (overlapping_images.empty()) {
        patch_match_file << "__auto__, 20\n";
      } else {
        for (const int image_idx : overlapping_images[i]) {
          patch_match_file << workspace.GetModel().GetImageName(image_idx)
                           << ", ";
        }
        patch_match_file << "\n";
      }
      fusion_file << ref_image_name << "\n";
    }
  }
}

}  // namespace mvs
}  // namespace colmap
