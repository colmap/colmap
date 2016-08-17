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

#include "mvs/model.h"

#include "util/logging.h"
#include "util/misc.h"
#include "util/string.h"

namespace colmap {
namespace mvs {
namespace {

void QuaternionToRotationMatrix(const double q[4], double R[9]) {
  double qq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  double qw, qx, qy, qz;
  if (qq > 0) {
    qw = q[0] / qq;
    qx = q[1] / qq;
    qy = q[2] / qq;
    qz = q[3] / qq;
  } else {
    qw = 1;
    qx = qy = qz = 0;
  }

  R[0] = (qw * qw + qx * qx - qz * qz - qy * qy);
  R[1] = (2 * qx * qy - 2 * qz * qw);
  R[2] = (2 * qy * qw + 2 * qz * qx);
  R[3] = (2 * qx * qy + 2 * qw * qz);
  R[4] = (qy * qy + qw * qw - qz * qz - qx * qx);
  R[5] = (2 * qz * qy - 2 * qx * qw);
  R[6] = (2 * qx * qz - 2 * qy * qw);
  R[7] = (2 * qy * qz + 2 * qw * qx);
  R[8] = (qz * qz + qw * qw - qy * qy - qx * qx);
}

// T = -R * C
void CameraCenterToTranslation(const double R[9], const double C[3],
                               double T[3]) {
  T[0] = -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]);
  T[1] = -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]);
  T[2] = -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]);
}

}  // namespace

bool Model::LoadFromMiddleBurry(const std::string& file_name) {
  std::ifstream file(file_name);
  CHECK(file.is_open()) << file_name;

  int num_images;
  file >> num_images;

  views.resize(num_images);
  for (int image_id = 0; image_id < num_images; ++image_id) {
    auto& view = views[image_id];

    file >> view.path;

    for (size_t i = 0; i < 9; ++i) {
      file >> view.K(i);
    }

    for (size_t i = 0; i < 9; ++i) {
      file >> view.R(i);
    }

    for (size_t i = 0; i < 3; ++i) {
      file >> view.T(i);
    }
  }

  return true;
}

bool Model::LoadFromNVM(const std::string& file_name) {
  std::ifstream file(file_name);
  CHECK(file.is_open()) << file_name;

  std::string token;
  if (file.peek() == 'N') {
    file >> token;
    if (!strstr(token.c_str(), "NVM_V3")) {
      return false;
    }
  } else {
    return false;
  }

  int num_images = 0;
  int num_points = 0;
  file >> num_images;

  // Read the camera parameters.
  views.resize(num_images);
  for (int image_id = 0; image_id < num_images; ++image_id) {
    auto& view = views[image_id];

    file >> view.path;

    file >> view.K(0, 0);
    view.K(1, 1) = view.K(0, 0);

    double quat[4];
    for (size_t i = 0; i < 4; ++i) {
      file >> quat[i];
    }

    double C[3];
    file >> C[0] >> C[1] >> C[2];

    float k1, k2;
    file >> k1 >> k2;
    CHECK_EQ(k1, 0.0f);
    CHECK_EQ(k2, 0.0f);

    double R[9];
    QuaternionToRotationMatrix(quat, R);
    for (size_t i = 0; i < 9; ++i) {
      view.R(i) = static_cast<float>(R[i]);
    }

    double T[3];
    CameraCenterToTranslation(R, C, T);
    for (size_t i = 0; i < 3; ++i) {
      view.T(i) = static_cast<float>(T[i]);
    }
  }

  points.resize(num_points);
  for (int point_id = 0; point_id < num_points; ++point_id) {
    auto& point = points[point_id];

    file >> point.X(0) >> point.X(1) >> point.X(2);

    int color[3];
    file >> color[0] >> color[1] >> color[2];

    int track_len;
    file >> track_len;
    point.track.resize(track_len);

    for (int i = 0; i < track_len; ++i) {
      int feature_idx;
      float imx, imy;
      file >> point.track[i] >> feature_idx >> imx >> imy;
      CHECK_LT(point.track[i], views.size());
    }
  }

  return true;
}

bool Model::LoadFromPMVS(const std::string& folder_name) {
  const std::string base_path = EnsureTrailingSlash(folder_name);
  const std::string bundle_file_path = base_path + "bundle.rd.out";

  std::ifstream file(bundle_file_path);
  CHECK(file.is_open()) << bundle_file_path;

  // Header line.
  std::string header;
  std::getline(file, header);

  int num_images, num_points;
  file >> num_images >> num_points;

  views.resize(num_images);
  for (int image_id = 0; image_id < num_images; ++image_id) {
    auto& view = views[image_id];

    view.path =
        base_path + StringPrintf("visualize/%08d.jpg", image_id);

    file >> view.K(0, 0);
    view.K(1, 1) = view.K(0, 0);

    float k1, k2;
    file >> k1 >> k2;
    CHECK_EQ(k1, 0.0f);
    CHECK_EQ(k2, 0.0f);

    file >> view.R(0) >> view.R(1) >> view.R(2);
    file >> view.R(3) >> view.R(4) >> view.R(5);
    file >> view.R(6) >> view.R(7) >> view.R(8);
    for (size_t i = 3; i < 9; ++i) {
      view.R(i) = -view.R(i);
    }

    file >> view.T(0) >> view.T(1) >> view.T(2);
    view.T(1) = -view.T(1);
    view.T(2) = -view.T(2);
  }

  points.resize(num_points);
  for (int point_id = 0; point_id < num_points; ++point_id) {
    auto& point = points[point_id];

    file >> point.X(0) >> point.X(1) >> point.X(2);

    int color[3];
    file >> color[0] >> color[1] >> color[2];

    int track_len;
    file >> track_len;
    point.track.resize(track_len);

    for (int i = 0; i < track_len; ++i) {
      int feature_idx;
      float imx, imy;
      file >> point.track[i] >> feature_idx >> imx >> imy;
      CHECK_LT(point.track[i], views.size());
    }
  }

  return true;
}

std::vector<std::pair<float, float>> Model::ComputeDepthRanges() const {
  std::vector<std::vector<float>> depths(views.size());
  for (const auto& point : points) {
    for (const auto& image_id : point.track) {
      const auto& view = views.at(image_id);
      const float depth = view.R.row(2) * point.X + view.T(2);
      depths[image_id].push_back(depth);
    }
  }

  std::vector<std::pair<float, float>> depth_ranges(depths.size());
  for (size_t image_id = 0; image_id < depth_ranges.size(); ++image_id) {
    auto& depth_range = depth_ranges[image_id];

    auto& image_depths = depths[image_id];

    if (image_depths.empty()) {
      depth_range.first = -1.0f;
      depth_range.second = -1.0f;
      continue;
    }

    std::sort(image_depths.begin(), image_depths.end());

    const float kMinPercentile = 0.01f;
    const float kMaxPercentile = 0.99f;
    depth_range.first = image_depths[image_depths.size() * kMinPercentile];
    depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

    const float kStretchRatio = 0.25f;
    depth_range.first *= (1.0f - kStretchRatio);
    depth_range.second *= (1.0f + kStretchRatio);
  }

  return depth_ranges;
}

std::vector<std::map<int, int>> Model::ComputeSharedPoints() const {
  std::vector<std::map<int, int>> shared_points(views.size());
  for (const auto& point : points) {
    for (size_t i = 0; i < point.track.size(); ++i) {
      const int image_id1 = point.track[i];
      for (size_t j = 0; j < i; ++j) {
        const int image_id2 = point.track[j];
        if (image_id1 != image_id2) {
          shared_points.at(image_id1)[image_id2] += 1;
          shared_points.at(image_id2)[image_id1] += 1;
        }
      }
    }
  }
  return shared_points;
}

}  // namespace mvs
}  // namespace colmap
