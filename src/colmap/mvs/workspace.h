// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cache.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <memory>
#include <mutex>

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
    std::filesystem::path workspace_path;
    std::string workspace_format;
    std::string input_type;
    std::string stereo_folder = "stereo";
  };

  explicit Workspace(const Options& options);
  virtual ~Workspace() = default;

  // Do nothing when we use a cache. Data is loaded as needed.
  virtual void Load(const std::vector<std::string>& image_names);

  inline const Options& GetOptions() const { return options_; }

  inline const Model& GetModel() const { return model_; }

  virtual const Bitmap& GetBitmap(int image_idx);
  virtual const DepthMap& GetDepthMap(int image_idx);
  virtual const NormalMap& GetNormalMap(int image_idx);

  // Get paths to bitmap, depth map, normal map and consistency graph.
  std::filesystem::path GetBitmapPath(int image_idx) const;
  std::filesystem::path GetDepthMapPath(int image_idx) const;
  std::filesystem::path GetNormalMapPath(int image_idx) const;

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasBitmap(int image_idx) const;
  bool HasDepthMap(int image_idx) const;
  bool HasNormalMap(int image_idx) const;

 protected:
  std::string GetFileName(int image_idx) const;

  Options options_;
  Model model_;

 private:
  std::filesystem::path depth_map_path_;
  std::filesystem::path normal_map_path_;
  std::vector<std::unique_ptr<Bitmap>> bitmaps_;
  std::vector<std::unique_ptr<DepthMap>> depth_maps_;
  std::vector<std::unique_ptr<NormalMap>> normal_maps_;
};

class CachedWorkspace : public Workspace {
 public:
  explicit CachedWorkspace(const Options& options);

  void Load(const std::vector<std::string>& image_names) override {}

  inline void ClearCache() { cache_.Clear(); }

  const Bitmap& GetBitmap(int image_idx) override;
  const DepthMap& GetDepthMap(int image_idx) override;
  const NormalMap& GetNormalMap(int image_idx) override;

 private:
  class CachedImage {
   public:
    CachedImage() {}
    CachedImage(CachedImage&& other) noexcept;
    CachedImage& operator=(CachedImage&& other) noexcept;
    inline size_t NumBytes() const { return num_bytes; }
    size_t num_bytes = 0;
    std::mutex mutex;
    std::unique_ptr<Bitmap> bitmap;
    std::unique_ptr<DepthMap> depth_map;
    std::unique_ptr<NormalMap> normal_map;

   private:
    NON_COPYABLE(CachedImage)
  };

  std::mutex cache_mutex_;
  MemoryConstrainedLRUCache<int, CachedImage> cache_;
};

// Import a PMVS workspace into the COLMAP workspace format. Only images in the
// provided option file name will be imported and used for reconstruction.
void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name);

}  // namespace mvs
}  // namespace colmap
