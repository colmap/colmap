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

#ifndef COLMAP_SRC_MVS_MODEL_H_
#define COLMAP_SRC_MVS_MODEL_H_

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {
namespace mvs {

// Simple sparse model class.
struct Model {
  struct View {
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K = Eigen::Matrix3f::Identity();
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f T = Eigen::Vector3f::Zero();
    std::string path = "";
  };

  struct Point {
    Eigen::Vector3f X = Eigen::Vector3f::Zero();
    std::vector<int> track;
  };

  // Load sparse model from Middlebury, VisualSfM or PMVS file format.
  bool LoadFromCOLMAP(const std::string& folder_path);
  bool LoadFromMiddleBurry(const std::string& file_path);
  bool LoadFromNVM(const std::string& file_path);
  bool LoadFromPMVS(const std::string& folder_path);

  // Compute the robust minimum and maximum depths from the sparse point cloud.
  std::vector<std::pair<float, float>> ComputeDepthRanges() const;

  // Compute the number of shared points between all possible pairs of images.
  std::vector<std::map<int, int>> ComputeSharedPoints() const;

  std::vector<View> views;
  std::vector<Point> points;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_MODEL_H_
