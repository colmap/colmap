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

#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {
namespace {

static constexpr int kNumImageTilesPerDim = 8;
static constexpr int kNumImageTiles =
    kNumImageTilesPerDim * kNumImageTilesPerDim;

std::unordered_map<image_t, std::vector<int>> ComputeImageTileIdxs(
    int num_tiles_per_dim, const Reconstruction& reconstruction) {
  std::unordered_map<image_t, std::vector<int>> image_tile_idxs;
  image_tile_idxs.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    const Camera& camera = reconstruction.Camera(image.CameraId());
    const point2D_t num_points2D = image.NumPoints2D();
    std::vector<int> tile_idxs(num_points2D);
    for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      const int tile_idx_x =
          Clamp<int>(num_tiles_per_dim * point2D.xy(0) / camera.width,
                     0,
                     num_tiles_per_dim - 1);
      const int tile_idx_y =
          Clamp<int>(num_tiles_per_dim * point2D.xy(1) / camera.height,
                     0,
                     num_tiles_per_dim - 1);
      tile_idxs[point2D_idx] = tile_idx_x * num_tiles_per_dim + tile_idx_y;
    }
    image_tile_idxs[image_id] = std::move(tile_idxs);
  }
  return image_tile_idxs;
}

double ComputeCoverageGain(
    const Point3D& point3D,
    const std::unordered_map<image_t, std::array<int, kNumImageTiles>>&
        num_selected_points3D_per_image_tile,
    const std::unordered_map<image_t, std::vector<int>>& image_tile_idxs) {
  double gain = 0;
  for (const auto& track_el : point3D.track.Elements()) {
    const int tile_idx =
        image_tile_idxs.at(track_el.image_id).at(track_el.point2D_idx);
    const int n = 1 + num_selected_points3D_per_image_tile.at(
                          track_el.image_id)[tile_idx];
    gain += 1. / std::sqrt(static_cast<double>(n)) -
            1. / std::sqrt(static_cast<double>(1 + n));
  }
  return gain;
}

}  // namespace

std::vector<point3D_t> FindRedundantPoints3D(
    double min_coverage_gain, const Reconstruction& reconstruction) {
  const size_t num_init_points3D = reconstruction.NumPoints3D();
  const size_t num_images = reconstruction.NumImages();

  const std::unordered_map<image_t, std::vector<int>> image_tile_idxs =
      ComputeImageTileIdxs(kNumImageTilesPerDim, reconstruction);
  std::unordered_map<image_t, std::array<int, kNumImageTiles>>
      num_selected_points3D_per_image_tile;
  num_selected_points3D_per_image_tile.reserve(num_images);
  for (const auto& image : reconstruction.Images()) {
    num_selected_points3D_per_image_tile[image.first].fill(0);
  }

  struct Point3DInfo {
    point3D_t point3D_id;
    const Point3D* point3D;
    double gain;
  };

  const auto has_left_smaller_gain = [](const Point3DInfo& left,
                                        const Point3DInfo& right) {
    return left.gain < right.gain;
  };

  std::vector<Point3DInfo> point3D_infos;
  point3D_infos.reserve(num_init_points3D);
  std::priority_queue<Point3DInfo,
                      std::vector<Point3DInfo>,
                      decltype(has_left_smaller_gain)>
      priority_queue(has_left_smaller_gain, std::move(point3D_infos));
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    priority_queue.push(
        {point3D_id,
         &point3D,
         ComputeCoverageGain(
             point3D, num_selected_points3D_per_image_tile, image_tile_idxs)});
  }

  std::unordered_set<point3D_t> selected_point3D_ids;
  selected_point3D_ids.reserve(num_init_points3D);
  while (!priority_queue.empty()) {
    auto point3D_info = priority_queue.top();
    priority_queue.pop();

    if (point3D_info.gain <= min_coverage_gain) {
      break;
    }

    // If another point has been selected that shares an image with the current
    // point, then the gain of the current point might have changed.
    const double updated_gain =
        ComputeCoverageGain(*point3D_info.point3D,
                            num_selected_points3D_per_image_tile,
                            image_tile_idxs);
    if (updated_gain < point3D_info.gain) {
      point3D_info.gain = updated_gain;
      priority_queue.push(point3D_info);
      continue;
    }

    for (const auto& track_el : point3D_info.point3D->track.Elements()) {
      const int tile_idx =
          image_tile_idxs.at(track_el.image_id).at(track_el.point2D_idx);
      num_selected_points3D_per_image_tile.at(track_el.image_id).at(tile_idx)++;
    }

    selected_point3D_ids.insert(point3D_info.point3D_id);
  }

  std::vector<point3D_t> redundant_point3D_ids;
  redundant_point3D_ids.reserve(num_init_points3D -
                                selected_point3D_ids.size());
  for (const auto& point3D : reconstruction.Points3D()) {
    if (selected_point3D_ids.count(point3D.first) == 0) {
      redundant_point3D_ids.push_back(point3D.first);
    }
  }

  return redundant_point3D_ids;
}

}  // namespace colmap
