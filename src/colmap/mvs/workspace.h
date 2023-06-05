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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_MVS_WORKSPACE_H_
#define COLMAP_SRC_MVS_WORKSPACE_H_

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/util/bitmap.h"
#include "colmap/util/cache.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace mvs {

class Workspace {
 public:
  struct Options {
    // The maximum cache size in gigabytes.
    double cache_size = 32.0;

    // The number of threads to use when pre-loading workspace.
    int num_threads = -1;

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
  virtual ~Workspace() = default;

  // Do nothing when we use a cache. Data is loaded as needed.
  virtual void Load(const std::vector<std::string>& image_names);

  inline const Options& GetOptions() const { return options_; }

  inline const Model& GetModel() const { return model_; }

  virtual const Bitmap& GetBitmap(const int image_idx);
  virtual const DepthMap& GetDepthMap(const int image_idx);
  virtual const NormalMap& GetNormalMap(const int image_idx);

  // Get paths to bitmap, depth map, normal map and consistency graph.
  std::string GetBitmapPath(const int image_idx) const;
  std::string GetDepthMapPath(const int image_idx) const;
  std::string GetNormalMapPath(const int image_idx) const;

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasBitmap(const int image_idx) const;
  bool HasDepthMap(const int image_idx) const;
  bool HasNormalMap(const int image_idx) const;

 protected:
  std::string GetFileName(const int image_idx) const;

  Options options_;
  Model model_;

 private:
  std::string depth_map_path_;
  std::string normal_map_path_;
  std::vector<std::unique_ptr<Bitmap>> bitmaps_;
  std::vector<std::unique_ptr<DepthMap>> depth_maps_;
  std::vector<std::unique_ptr<NormalMap>> normal_maps_;
};

class CachedWorkspace : public Workspace {
 public:
  CachedWorkspace(const Options& options);

  void Load(const std::vector<std::string>& image_names) override {}

  inline void ClearCache() { cache_.Clear(); }

  const Bitmap& GetBitmap(const int image_idx) override;
  const DepthMap& GetDepthMap(const int image_idx) override;
  const NormalMap& GetNormalMap(const int image_idx) override;

 private:
  class CachedImage {
   public:
    CachedImage() {}
    CachedImage(CachedImage&& other);
    CachedImage& operator=(CachedImage&& other);
    inline size_t NumBytes() const { return num_bytes; }
    size_t num_bytes = 0;
    std::unique_ptr<Bitmap> bitmap;
    std::unique_ptr<DepthMap> depth_map;
    std::unique_ptr<NormalMap> normal_map;

   private:
    NON_COPYABLE(CachedImage)
  };

  MemoryConstrainedLRUCache<int, CachedImage> cache_;
};

// Import a PMVS workspace into the COLMAP workspace format. Only images in the
// provided option file name will be imported and used for reconstruction.
void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
