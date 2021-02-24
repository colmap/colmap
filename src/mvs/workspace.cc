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

#include "mvs/workspace.h"

#include <numeric>

#include "util/misc.h"

namespace colmap {
namespace mvs {

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
  }
  return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options)
    : options_(options),
      cache_(1024 * 1024 * 1024 * options_.cache_size,
             [](const int) { return CachedImage(); }) {
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

void Workspace::ClearCache() { cache_.Clear(); }

const Workspace::Options& Workspace::GetOptions() const { return options_; }

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.bitmap) {
    cached_image.bitmap.reset(new Bitmap());
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

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
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

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
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

std::string Workspace::GetFileName(const int image_idx) const {
  const auto& image_name = model_.GetImageName(image_idx);
  return StringPrintf("%s.%s.bin", image_name.c_str(),
                      options_.input_type.c_str());
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
      CHECK_EQ(elems.size(), 4);
      const int range_lower = std::stoull(elems[2]);
      const int range_upper = std::stoull(elems[3]);
      CHECK_LT(range_lower, range_upper);
      num_images = range_upper - range_lower;
      image_idxs.resize(num_images);
      std::iota(image_idxs.begin(), image_idxs.end(), range_lower);
    } else {
      CHECK_EQ(num_images + 2, elems.size());
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
    CHECK(patch_match_file.is_open()) << patch_match_path;
    CHECK(fusion_file.is_open()) << fusion_path;
    for (size_t i = 0; i < image_names.size(); ++i) {
      const auto& ref_image_name = image_names[i];
      patch_match_file << ref_image_name << std::endl;
      if (overlapping_images.empty()) {
        patch_match_file << "__auto__, 20" << std::endl;
      } else {
        for (const int image_idx : overlapping_images[i]) {
          patch_match_file << workspace.GetModel().GetImageName(image_idx)
                           << ", ";
        }
        patch_match_file << std::endl;
      }
      fusion_file << ref_image_name << std::endl;
    }
  }
}

void NoCacheWorkspace::Load(const std::vector<std::string>& image_names) {
  const size_t num_images = model_.images.size();
  bitmaps_.resize(num_images);
  depth_maps_.resize(num_images);
  confidence_maps_.resize(num_images);
  normal_maps_.resize(num_images);

  std::cout << "Loading workspace data..." << std::endl;
#pragma omp parallel for
  for (int i = 0; i < image_names.size(); i) {
    const int image_idx = model_.GetImageIdx(image_names[i]);

    if (!HasBitmap(image_idx) || !HasDepthMap(image_idx)) {
      std::cout
          << StringPrintf(
                 "WARNING: Ignoring image %s, because input does not exist.",
                 image_names[i].c_str())
          << std::endl;
      continue;
    }

    const size_t width = model_.images.at(image_idx).GetWidth();
    const size_t height = model_.images.at(image_idx).GetHeight();

    // Read and rescale bitmap
    bitmaps_[image_idx].reset(new Bitmap());
    bitmaps_[image_idx]->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      bitmaps_[image_idx]->Rescale((int)width, (int)height);
    }

    // Read and rescale depth map
    depth_maps_[image_idx].reset(new DepthMap());
    depth_maps_[image_idx]->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      depth_maps_[image_idx]->Downsize(width, height);
    }

    // Read and rescale confidence map
    const std::string confidence_map_path = GetConfidenceMapPath(image_idx);
    if (HasConfidenceMap(image_idx)) {
      confidence_maps_[image_idx].reset(new ConfidenceMap());
      confidence_maps_[image_idx]->Read(confidence_map_path);
      if (options_.max_image_size > 0) {
        confidence_maps_[image_idx]->Downsize(width, height);
      }
    } else {
      // Assume depth confidence probability of 1.0 when the map is not given
      confidence_maps_[image_idx].reset(new ConfidenceMap(width, height));
      if (options_.save_calculated_maps) {
        confidence_maps_[image_idx]->Write(confidence_map_path);
      }
    }

    // Read and rescale normal map
    const std::string normal_map_path = GetNormalMapPath(image_idx);
    normal_maps_[image_idx].reset(new NormalMap());
    if (!options_.calculate_normals && HasNormalMap(image_idx)) {
      normal_maps_[image_idx]->Read(normal_map_path);
      if (options_.max_image_size > 0) {
        normal_maps_[image_idx]->Downsize(width, height);
      }
    } else {
      // Estimate normal map from depth when the map is not given
      normal_maps_[image_idx]->EstimateFromDepth(
          *depth_maps_[image_idx], model_.images.at(image_idx).GetInvK());
      if (options_.save_calculated_maps) {
        normal_maps_[image_idx]->Write(normal_map_path);
      }
    }
  }
}

}  // namespace mvs
}  // namespace colmap
