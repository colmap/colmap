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
  struct Options {
    // The maximum cache size in gigabytes.
    double cache_size = 32.0;

    // Which data to store in the cache.
    bool cache_bitmap = false;
    bool cache_depth_map = false;
    bool cache_normal_map = false;
    bool cache_consistency_graph = false;

    // Location and type of workspace.
    std::string workspace_path;
    std::string workspace_format;
    std::string input_type;
  };

  Workspace(const Options& options);

  const Model& GetModel() const;
  const Bitmap& GetBitmap(const int image_id);
  const DepthMap& GetDepthMap(const int image_id);
  const NormalMap& GetNormalMap(const int image_id);
  const ConsistencyGraph& GetConsistencyGraph(const int image_id);

  // Get paths to bitmap, depth map, normal map and consistency graph.
  std::string GetBitmapPath(const int image_id) const;
  std::string GetDepthMapPath(const int image_id) const;
  std::string GetNormalMapPath(const int image_id) const;
  std::string GetConsistencyGraphPath(const int image_id) const;

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasBitmap(const int image_id) const;
  bool HasDepthMap(const int image_id) const;
  bool HasNormalMap(const int image_id) const;
  bool HasConsistencyGraph(const int image_id) const;

 private:
  size_t GetNumBytes(const int image_id);
  std::string GetFileName(const int image_id) const;

  const Options options_;

  class CachedImage {
   public:
    CachedImage();
    CachedImage(CachedImage&& other);
    CachedImage& operator=(CachedImage&& other);
    size_t NumBytes() const;
    size_t num_bytes = 0;
    std::unique_ptr<Bitmap> bitmap;
    std::unique_ptr<DepthMap> depth_map;
    std::unique_ptr<NormalMap> normal_map;
    std::unique_ptr<ConsistencyGraph> consistency_graph;

   private:
    NON_COPYABLE(CachedImage)
  };

  Model model_;
  MemoryConstrainedLRUCache<int, CachedImage> cache_;
  std::unordered_map<int, size_t> num_bytes_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
