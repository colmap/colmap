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

#include "mvs/fusion.h"

#include <Eigen/Geometry>

#include "util/logging.h"
#include "util/misc.h"

namespace colmap {
namespace mvs {
namespace internal {

float Median(std::vector<float>* elems) {
  CHECK(!elems->empty());
  const size_t mid_idx = elems->size() / 2;
  std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
  if (elems->size() % 2 == 0) {
    const float mid_element1 = (*elems)[mid_idx];
    const float mid_element2 =
        *std::max_element(elems->begin(), elems->begin() + mid_idx);
    return (mid_element1 + mid_element2) / 2.0f;
  } else {
    return (*elems)[mid_idx];
  }
}

}  // namespace internal

void StereoFusion::Options::Print() const {
#define PrintOption(option) std::cout << #option ": " << option << std::endl
  PrintHeading2("StereoFusion::Options");
  PrintOption(min_num_pixels);
  PrintOption(max_num_pixels);
  PrintOption(max_traversal_depth);
  PrintOption(max_reproj_error);
  PrintOption(max_depth_error);
  PrintOption(max_normal_error);
#undef PrintOption
}

void StereoFusion::Options::Check() const {
  CHECK_GE(min_num_pixels, 0);
  CHECK_LE(min_num_pixels, max_num_pixels);
  CHECK_GT(max_traversal_depth, 0);
  CHECK_GE(max_reproj_error, 0);
  CHECK_GE(max_depth_error, 0);
  CHECK_GE(max_normal_error, 0);
  CHECK_GT(cache_size, 0);
}

StereoFusion::StereoFusion(const Options& options,
                           const std::string& workspace_path,
                           const std::string& workspace_format,
                           const std::string& input_type)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      input_type_(input_type),
      max_squared_reproj_error_(options_.max_reproj_error *
                                options_.max_reproj_error),
      min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))) {}

const std::vector<FusedPoint>& StereoFusion::GetFusedPoints() const {
  return fused_points_;
}

void StereoFusion::Run() {
  fused_points_.clear();

  options_.Print();
  std::cout << std::endl;

  std::cout << "Reading workspace..." << std::endl;
  workspace_.reset(new Workspace(options_.cache_size, workspace_path_,
                                 workspace_format_, input_type_));

  if (IsStopped()) {
    GetTimer().PrintMinutes();
    return;
  }

  std::cout << "Reading configuration..." << std::endl;

  const auto& model = workspace_->GetModel();
  used_images_.resize(model.images.size());
  visited_masks_.resize(model.images.size());
  bitmap_scales_.resize(model.images.size());
  P_.resize(model.images.size());
  inv_P_.resize(model.images.size());
  inv_R_.resize(model.images.size());

  const auto image_names =
      ReadTextFileLines(JoinPaths(workspace_path_, "stereo/fusion.cfg"));
  for (const auto& image_name : image_names) {
    const int image_id = model.GetImageId(image_name);
    const auto& image = model.images.at(image_id);
    const auto& depth_map = workspace_->GetDepthMap(image_id);

    used_images_.at(image_id) = true;

    visited_masks_.at(image_id) =
        Mat<bool>(depth_map.GetWidth(), depth_map.GetHeight(), 1);
    visited_masks_.at(image_id).Fill(false);

    bitmap_scales_.at(image_id) = std::make_pair(
        static_cast<float>(depth_map.GetWidth()) / image.GetWidth(),
        static_cast<float>(depth_map.GetHeight()) / image.GetHeight());

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetK());
    K(0, 0) *= bitmap_scales_.at(image_id).first;
    K(0, 2) *= bitmap_scales_.at(image_id).first;
    K(1, 1) *= bitmap_scales_.at(image_id).second;
    K(1, 2) *= bitmap_scales_.at(image_id).second;

    ComposeProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                            P_.at(image_id).data());
    ComposeInverseProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                   inv_P_.at(image_id).data());
    inv_R_.at(image_id) =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetR())
            .transpose();
  }

  const size_t min_num_pixels = static_cast<size_t>(options_.min_num_pixels);

  for (size_t image_id = 0; image_id < model.images.size(); ++image_id) {
    if (IsStopped()) {
      break;
    }

    if (!used_images_.at(image_id)) {
      continue;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Fusing image [%d/%d]", image_id + 1,
                              model.images.size())
              << std::flush;

    const size_t width = workspace_->GetDepthMap(image_id).GetWidth();
    const size_t height = workspace_->GetDepthMap(image_id).GetHeight();
    const auto& visited_mask = visited_masks_.at(image_id);

    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        if (visited_mask.Get(row, col)) {
          continue;
        }

        fused_points_x_.clear();
        fused_points_y_.clear();
        fused_points_z_.clear();
        fused_normal_sum_.setZero();
        fused_color_sum_ = BitmapColor<uint32_t>(0, 0, 0);

        Fuse(image_id, row, col, 0);

        const size_t num_pixels = fused_points_x_.size();
        if (num_pixels >= min_num_pixels) {
          FusedPoint fused_point;

          fused_point.x = internal::Median(&fused_points_x_);
          fused_point.y = internal::Median(&fused_points_y_);
          fused_point.z = internal::Median(&fused_points_z_);

          const Eigen::Vector3d mean_normal = fused_normal_sum_.normalized();
          fused_point.nx = static_cast<float>(mean_normal(0));
          fused_point.ny = static_cast<float>(mean_normal(1));
          fused_point.nz = static_cast<float>(mean_normal(2));

          fused_point.r = TruncateCast<double, uint8_t>(
              std::round(static_cast<double>(fused_color_sum_.r) / num_pixels));
          fused_point.g = TruncateCast<double, uint8_t>(
              std::round(static_cast<double>(fused_color_sum_.g) / num_pixels));
          fused_point.b = TruncateCast<double, uint8_t>(
              std::round(static_cast<double>(fused_color_sum_.b) / num_pixels));

          fused_points_.push_back(fused_point);
        }
      }
    }

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  fused_points_.shrink_to_fit();

  std::cout << "Number of fused points: " << fused_points_.size() << std::endl;
  GetTimer().PrintMinutes();
}

void StereoFusion::Fuse(const int image_id, const int row, const int col,
                        const size_t traversal_depth) {
  if (!used_images_.at(image_id)) {
    return;
  }

  const auto& depth_map = workspace_->GetDepthMap(image_id);
  if (col < 0 || row < 0 || col >= static_cast<int>(depth_map.GetWidth()) ||
      row >= static_cast<int>(depth_map.GetHeight())) {
    return;
  }

  const float depth = depth_map.Get(row, col);

  // Pixels with negative depth are filtered.
  if (depth <= 0.0f) {
    return;
  }

  // Check if pixel already fused.
  auto& visited_mask = visited_masks_.at(image_id);
  if (visited_mask.Get(row, col)) {
    return;
  }

  // If the traversal depth is greater than zero, the initial reference pixel
  // has already been added and we need to check for consistency.
  if (traversal_depth > 0) {
    // Project reference point into current view.
    const Eigen::Vector3f proj = P_.at(image_id) * fused_ref_point_;

    // Depth error of reference depth with current depth.
    const float depth_error = std::abs((proj(2) - depth) / depth);
    if (depth_error > options_.max_depth_error) {
      return;
    }

    // Reprojection error reference point in the current view.
    const float col_diff = proj(0) / proj(2) - col;
    const float row_diff = proj(1) / proj(2) - row;
    const float squared_reproj_error =
        col_diff * col_diff + row_diff * row_diff;
    if (squared_reproj_error > max_squared_reproj_error_) {
      return;
    }
  }

  // Determine normal direction in global reference frame.
  const auto& normal_map = workspace_->GetNormalMap(image_id);
  const Eigen::Vector3f normal =
      inv_R_.at(image_id) * Eigen::Vector3f(normal_map.Get(row, col, 0),
                                            normal_map.Get(row, col, 1),
                                            normal_map.Get(row, col, 2));

  // Check for consistent normal direction with reference normal.
  if (traversal_depth > 0) {
    const float cos_normal_error = fused_ref_normal_.dot(normal);
    if (cos_normal_error < min_cos_normal_error_) {
      return;
    }
  }

  // Determine 3D location of current depth value.
  const Eigen::Vector3f xyz =
      inv_P_.at(image_id) *
      Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

  // Read the color of the pixel.
  BitmapColor<uint8_t> color;
  const auto& bitmap_scale = bitmap_scales_.at(image_id);
  workspace_->GetBitmap(image_id).InterpolateNearestNeighbor(
      col / bitmap_scale.first, row / bitmap_scale.second, &color);

  // Set the current pixel as visited.
  visited_mask.Set(row, col, true);

  // Accumulate statistics for fused point.
  fused_points_x_.push_back(xyz(0));
  fused_points_y_.push_back(xyz(1));
  fused_points_z_.push_back(xyz(2));
  fused_normal_sum_ += normal.cast<double>();
  fused_color_sum_.r += color.r;
  fused_color_sum_.g += color.g;
  fused_color_sum_.b += color.b;

  // Remember the first pixel as the reference.
  if (traversal_depth == 0) {
    fused_ref_point_ = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
    fused_ref_normal_ = normal;
  }

  const int next_traversal_depth = traversal_depth + 1;

  // Do not traverse the graph infinitely in one branch and limit the maximum
  // number of pixels fused in one point to avoid stack overflow.
  if (next_traversal_depth >= options_.max_traversal_depth ||
      fused_points_x_.size() >= static_cast<size_t>(options_.max_num_pixels)) {
    return;
  }

  // Traverse the consistency graph by projecting point into other views.
  int num_images = 0;
  const int* image_ids = nullptr;
  workspace_->GetConsistencyGraph(image_id).GetImageIds(row, col, &num_images,
                                                        &image_ids);

  // Copy the data, since the pointer from the graph might be invalidated due to
  // the recursive calls to Fuse and since the graph is cached.
  const std::vector<int> next_image_ids(image_ids, image_ids + num_images);

  for (const auto& next_image_id : next_image_ids) {
    const Eigen::Vector3f next_proj = P_.at(next_image_id) * xyz.homogeneous();
    const int next_col =
        static_cast<int>(std::round(next_proj(0) / next_proj(2)));
    const int next_row =
        static_cast<int>(std::round(next_proj(1) / next_proj(2)));
    Fuse(next_image_id, next_row, next_col, next_traversal_depth);
  }
}

void WritePlyText(const std::string& path,
                  const std::vector<FusedPoint>& points) {
  std::ofstream file(path);
  CHECK(file.is_open()) << path;

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << points.size() << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "property float nx" << std::endl;
  file << "property float ny" << std::endl;
  file << "property float nz" << std::endl;
  file << "property uchar red" << std::endl;
  file << "property uchar green" << std::endl;
  file << "property uchar blue" << std::endl;
  file << "end_header" << std::endl;

  for (const auto& point : points) {
    file << point.x << " " << point.y << " " << point.z << " " << point.nx
         << " " << point.ny << " " << point.nz << " "
         << static_cast<int>(point.r) << " " << static_cast<int>(point.g) << " "
         << static_cast<int>(point.b) << std::endl;
  }

  file.close();
}

void WritePlyBinary(const std::string& path,
                    const std::vector<FusedPoint>& points) {
  std::fstream text_file(path, std::ios_base::out);
  CHECK(text_file.is_open()) << path;

  text_file << "ply" << std::endl;
  if (IsBigEndian()) {
    text_file << "format binary_big_endian 1.0" << std::endl;
  } else {
    text_file << "format binary_little_endian 1.0" << std::endl;
  }
  text_file << "element vertex " << points.size() << std::endl;
  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;
  text_file << "property float nx" << std::endl;
  text_file << "property float ny" << std::endl;
  text_file << "property float nz" << std::endl;
  text_file << "property uchar red" << std::endl;
  text_file << "property uchar green" << std::endl;
  text_file << "property uchar blue" << std::endl;
  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(
      path, std::ios_base::out | std::ios_base::binary | std::ios_base::app);
  CHECK(binary_file.is_open()) << path;

  float xyz_normal_buffer[6];
  uint8_t rgb_buffer[3];
  for (const auto& point : points) {
    xyz_normal_buffer[0] = point.x;
    xyz_normal_buffer[1] = point.y;
    xyz_normal_buffer[2] = point.z;
    xyz_normal_buffer[3] = point.nx;
    xyz_normal_buffer[4] = point.ny;
    xyz_normal_buffer[5] = point.nz;
    binary_file.write(reinterpret_cast<const char*>(xyz_normal_buffer),
                      6 * sizeof(float));
    rgb_buffer[0] = point.r;
    rgb_buffer[1] = point.g;
    rgb_buffer[2] = point.b;
    binary_file.write(reinterpret_cast<const char*>(rgb_buffer),
                      3 * sizeof(uint8_t));
  }
  binary_file.close();
}

}  // namespace mvs
}  // namespace colmap
