// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "mvs/workspace.h"

#include "util/misc.h"

namespace colmap {
namespace mvs {

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
  consistency_graph = std::move(other.consistency_graph);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
    consistency_graph = std::move(other.consistency_graph);
  }
  return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options)
    : options_(options),
      cache_(1024 * 1024 * 1024 * options_.cache_size, [&](const int image_id) {
        CachedImage cached_image;
        cached_image.num_bytes = GetNumBytes(image_id);
        return cached_image;
      }) {
  model_.Read(options_.workspace_path, options_.workspace_format);
}

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_id) {
  CHECK(options_.cache_bitmap);
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.bitmap) {
    cached_image.bitmap.reset(new Bitmap());
    cached_image.bitmap->Read(GetBitmapPath(image_id), true);
  }
  return *cached_image.bitmap;
}

const DepthMap& Workspace::GetDepthMap(const int image_id) {
  CHECK(options_.cache_depth_map);
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
    cached_image.depth_map->Read(GetDepthMapPath(image_id));
  }
  return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_id) {
  CHECK(options_.cache_normal_map);
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
    cached_image.normal_map->Read(GetNormalMapPath(image_id));
  }
  return *cached_image.normal_map;
}

const ConsistencyGraph& Workspace::GetConsistencyGraph(const int image_id) {
  CHECK(options_.cache_consistency_graph);
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.consistency_graph) {
    cached_image.consistency_graph.reset(new ConsistencyGraph());
    cached_image.consistency_graph->Read(GetConsistencyGraphPath(image_id));
  }
  return *cached_image.consistency_graph;
}

bool Workspace::HasImage(const int image_id) const {
  return (!options_.cache_bitmap || ExistsFile(GetBitmapPath(image_id))) &&
         (!options_.cache_depth_map || ExistsFile(GetDepthMapPath(image_id))) &&
         (!options_.cache_normal_map ||
          ExistsFile(GetNormalMapPath(image_id))) &&
         (!options_.cache_consistency_graph ||
          ExistsFile(GetConsistencyGraphPath(image_id)));
}

size_t Workspace::GetNumBytes(const int image_id) {
  const auto it = num_bytes_.find(image_id);
  if (it == num_bytes_.end()) {
    size_t num_bytes = 0;
    if (options_.cache_bitmap) {
      num_bytes += GetFileSize(GetBitmapPath(image_id));
    }
    if (options_.cache_depth_map) {
      num_bytes += GetFileSize(GetDepthMapPath(image_id));
    }
    if (options_.cache_normal_map) {
      num_bytes += GetFileSize(GetNormalMapPath(image_id));
    }
    if (options_.cache_consistency_graph) {
      num_bytes += GetFileSize(GetConsistencyGraphPath(image_id));
    }
    num_bytes_.emplace(image_id, num_bytes);
    return num_bytes;
  } else {
    return it->second;
  }
}

std::string Workspace::GetFileName(const int image_id) const {
  const auto& image_name = model_.GetImageName(image_id);
  return StringPrintf("%s.%s.bin", image_name.c_str(),
                      options_.input_type.c_str());
}

std::string Workspace::GetBitmapPath(const int image_id) const {
  return model_.images.at(image_id).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_id) const {
  return JoinPaths(options_.workspace_path, "stereo/depth_maps",
                   GetFileName(image_id));
}

std::string Workspace::GetNormalMapPath(const int image_id) const {
  return JoinPaths(options_.workspace_path, "stereo/normal_maps",
                   GetFileName(image_id));
}

std::string Workspace::GetConsistencyGraphPath(const int image_id) const {
  return JoinPaths(options_.workspace_path, "stereo/consistency_graphs",
                   GetFileName(image_id));
}

}  // namespace mvs
}  // namespace colmap
