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
             [](const int image_id) { return CachedImage(); }) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.workspace_format);
  if (options_.max_image_size > 0) {
    for (auto& image : model_.images) {
      image.Downsize(options_.max_image_size, options_.max_image_size);
    }
  }
}

void Workspace::ClearCache() { cache_.Clear(); }

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.bitmap) {
    cached_image.bitmap.reset(new Bitmap());
    cached_image.bitmap->Read(GetBitmapPath(image_id), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      cached_image.bitmap->Rescale(model_.images.at(image_id).GetWidth(),
                                   model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.bitmap;
}

const DepthMap& Workspace::GetDepthMap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
    cached_image.depth_map->Read(GetDepthMapPath(image_id));
    if (options_.max_image_size > 0) {
      cached_image.depth_map->Downsize(model_.images.at(image_id).GetWidth(),
                                       model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.depth_map->GetNumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
    cached_image.normal_map->Read(GetNormalMapPath(image_id));
    if (options_.max_image_size > 0) {
      cached_image.normal_map->Downsize(model_.images.at(image_id).GetWidth(),
                                        model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.normal_map->GetNumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.normal_map;
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

bool Workspace::HasBitmap(const int image_id) const {
  return ExistsFile(GetBitmapPath(image_id));
}

bool Workspace::HasDepthMap(const int image_id) const {
  return ExistsFile(GetDepthMapPath(image_id));
}

bool Workspace::HasNormalMap(const int image_id) const {
  return ExistsFile(GetNormalMapPath(image_id));
}

std::string Workspace::GetFileName(const int image_id) const {
  const auto& image_name = model_.GetImageName(image_id);
  return StringPrintf("%s.%s.bin", image_name.c_str(),
                      options_.input_type.c_str());
}

}  // namespace mvs
}  // namespace colmap
