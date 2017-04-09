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

#ifndef COLMAP_SRC_MVS_WORKSPACE_H_
#define COLMAP_SRC_MVS_WORKSPACE_H_

#include "mvs/consistency_graph.h"
#include "mvs/depth_map.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "util/bitmap.h"
#include "util/cache.h"

namespace colmap {
namespace mvs {

class Workspace {
 public:
  Workspace(const size_t cache_size, const std::string& workspace_path,
            const std::string& workspace_format, const std::string& input_type);

  const Model& GetModel() const;
  const Bitmap& GetBitmap(const int image_id);
  const DepthMap& GetDepthMap(const int image_id);
  const NormalMap& GetNormalMap(const int image_id);
  const ConsistencyGraph& GetConsistencyGraph(const int image_id);

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasImage(const int image_id) const;

 private:
  std::string GetFileName(const int image_id) const;
  std::string GetBitmapPath(const int image_id) const;
  std::string GetDepthMapPath(const int image_id) const;
  std::string GetNormalMapPath(const int image_id) const;
  std::string GetConsistencyGraphPath(const int image_id) const;

  const std::string workspace_path_;
  const std::string workspace_format_;
  const std::string input_type_;

  Model model_;
  std::unique_ptr<LRUCache<int, Bitmap>> bitmaps_;
  std::unique_ptr<LRUCache<int, DepthMap>> depth_maps_;
  std::unique_ptr<LRUCache<int, NormalMap>> normal_maps_;
  std::unique_ptr<LRUCache<int, ConsistencyGraph>> consistency_graphs_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
