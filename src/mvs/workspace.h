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
// Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

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

    // Maximum image size in either dimension.
    int max_image_size = -1;

    // Whether to read image as RGB or gray scale.
    bool image_as_rgb = true;

    // Location and type of workspace.
    std::string workspace_path;
    std::string workspace_format;
    std::string input_type;
    std::string stereo_folder = "stereo";
  };

  Workspace(const Options& options);

  void ClearCache();

  const Options& GetOptions() const;

  const Model& GetModel() const;
  const Bitmap& GetBitmap(const int image_idx);
  const DepthMap& GetDepthMap(const int image_idx);
  const NormalMap& GetNormalMap(const int image_idx);

  // Get paths to bitmap, depth map, normal map and consistency graph.
  std::string GetBitmapPath(const int image_idx) const;
  std::string GetDepthMapPath(const int image_idx) const;
  std::string GetNormalMapPath(const int image_idx) const;

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasBitmap(const int image_idx) const;
  bool HasDepthMap(const int image_idx) const;
  bool HasNormalMap(const int image_idx) const;

 private:
  std::string GetFileName(const int image_idx) const;

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

   private:
    NON_COPYABLE(CachedImage)
  };

  Options options_;
  Model model_;
  MemoryConstrainedLRUCache<int, CachedImage> cache_;
  std::string depth_map_path_;
  std::string normal_map_path_;
};

// Import a PMVS workspace into the COLMAP workspace format. Only images in the
// provided option file name will be imported and used for reconstruction.
void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
