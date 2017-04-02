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

Workspace::Workspace(const size_t cache_size, const std::string& workspace_path,
                     const std::string& workspace_format,
                     const std::string& input_type)
    : workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      input_type_(input_type) {
  model_.Read(workspace_path_, workspace_format_);

  bitmaps_.reset(new LRUCache<int, Bitmap>(cache_size, [&](const int image_id) {
    Bitmap bitmap;
    bitmap.Read(model_.images.at(image_id).GetPath(), true);
    return bitmap;
  }));

  depth_maps_.reset(
      new LRUCache<int, DepthMap>(cache_size, [&](const int image_id) {
        DepthMap depth_map;
        depth_map.Read(JoinPaths(workspace_path_, "stereo/depth_maps",
                                 GetFileName(image_id)));
        return depth_map;
      }));

  normal_maps_.reset(
      new LRUCache<int, NormalMap>(cache_size, [&](const int image_id) {
        NormalMap normal_map;
        normal_map.Read(JoinPaths(workspace_path_, "stereo/normal_maps",
                                  GetFileName(image_id)));
        return normal_map;
      }));

  consistency_graphs_.reset(
      new LRUCache<int, ConsistencyGraph>(cache_size, [&](const int image_id) {
        ConsistencyGraph consistency_graph;
        consistency_graph.Read(JoinPaths(workspace_path_,
                                         "stereo/consistency_graphs",
                                         GetFileName(image_id)));
        return consistency_graph;
      }));
}

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_id) {
  return bitmaps_->Get(image_id);
}

const DepthMap& Workspace::GetDepthMap(const int image_id) {
  return depth_maps_->Get(image_id);
}

const NormalMap& Workspace::GetNormalMap(const int image_id) {
  return normal_maps_->Get(image_id);
}

const ConsistencyGraph& Workspace::GetConsistencyGraph(const int image_id) {
  return consistency_graphs_->Get(image_id);
}

std::string Workspace::GetFileName(const int image_id) const {
  const auto& image_name = model_.GetImageName(image_id);
  return StringPrintf("%s.%s.bin", image_name.c_str(), input_type_.c_str());
}

}  // namespace mvs
}  // namespace colmap
