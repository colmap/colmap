// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_MVS_FUSION_H_
#define COLMAP_SRC_MVS_FUSION_H_

#include <vector>

#include <Eigen/Core>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/mat.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "util/math.h"
#include "util/threading.h"

namespace colmap {
namespace mvs {

struct FusedPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
};

class StereoFusion : public Thread {
 public:
  struct Options {
    // Minimum number of fused pixels to produce a point.
    int min_num_pixels = 3;

    // Maximum number of pixels to fuse into a single point.
    int max_num_pixels = 1000;

    // Maximum depth in consistency graph traversal.
    int max_traversal_depth = 100;

    // Maximum relative difference between measured and projected pixel.
    double max_reproj_error = 2.0f;

    // Maximum relative difference between measured and projected depth.
    double max_depth_error = 0.01f;

    // Maximum difference between normals of pixels to be fused.
    double max_normal_error = 10.0f;

    // Check the options for validity.
    void Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  StereoFusion(const Options& options,
                        const std::string& workspace_path,
                        const std::string& workspace_format,
                        const std::string& input_type);

  const std::vector<FusedPoint>& GetFusedPoints() const;

 private:
  void Run();
  void Read();
  void Prepare();
  void FusePoint(const int image_id, const int row, const int col,
                 const size_t traversal_depth);

  struct ImageData {
    bool used = false;
    const Image* image = nullptr;
    const DepthMap* depth_map = nullptr;
    const NormalMap* normal_map = nullptr;
    Mat<char> visited_mask;
    Eigen::Matrix<float, 3, 4> P;
    Eigen::Matrix<float, 3, 4> inv_P;
    Eigen::Matrix3f inv_R;
  };

  // Consistency graph representation that efficiently maps each pixel in
  // an image to its consistent image identifies.
  class ConsistencyGraph {
   public:
    ConsistencyGraph();
    ConsistencyGraph(const std::vector<Image>& images,
                     const std::vector<std::vector<int>>* consistency_graph);

    void GetConsistentImageIds(const int image_id, const int row, const int col,
                               int* num_consistent,
                               const int** consistent_image_ids) const;

   private:
    const static int kNoConsistentImageIds = -1;
    const std::vector<std::vector<int>>* consistency_graph_;
    std::vector<Eigen::MatrixXi> image_maps_;
  };

  const Options options_;
  const std::string workspace_path_;
  const std::string workspace_format_;
  const std::string input_type_;
  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;

  Model model_;
  std::vector<char> used_image_mask_;
  ConsistencyGraph consistency_graph_;
  std::vector<ImageData> image_data_;
  std::vector<FusedPoint> fused_points_;

  Eigen::Vector4f fused_ref_point_;
  Eigen::Vector3f fused_ref_normal_;
  std::vector<float> fused_points_x_;
  std::vector<float> fused_points_y_;
  std::vector<float> fused_points_z_;
  Eigen::Vector3d fused_normal_sum_;
  BitmapColor<uint32_t> fused_color_sum_;
};

// Write the point cloud to PLY file.
void WritePlyText(const std::string& path,
                  const std::vector<FusedPoint>& points);
void WritePlyBinary(const std::string& path,
                    const std::vector<FusedPoint>& points);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_FUSION_H_
