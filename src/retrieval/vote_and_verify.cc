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

#include "retrieval/vote_and_verify.h"

#include <array>
#include <unordered_map>

#include "estimators/affine_transform.h"
#include "optim/ransac.h"
#include "util/logging.h"
#include "util/math.h"

namespace colmap {
namespace retrieval {
namespace {

// Affine transformation from left to right and from left to right image.
struct TwoWayTransform {
  TwoWayTransform()
      : A12(Eigen::Matrix2f::Zero()),
        t12(Eigen::Vector2f::Zero()),
        A21(Eigen::Matrix2f::Zero()),
        t21(Eigen::Vector2f::Zero()) {}
  TwoWayTransform(const FeatureGeometryTransform& tform) {
    const float sin_angle = std::sin(tform.angle);
    const float cos_angle = std::cos(tform.angle);

    Eigen::Matrix2f R;
    R << cos_angle, -sin_angle, sin_angle, cos_angle;

    A12 = tform.scale * R;
    t12 << tform.tx, tform.ty;
    A21 = R.transpose() / tform.scale;
    t21 = -A21 * t12;
  }

  Eigen::Matrix2f A12;
  Eigen::Vector2f t12;
  Eigen::Matrix2f A21;
  Eigen::Vector2f t21;
};

// Class representing a single bin in the voting space of a similarity
// transformation. Keeps track of the mean transformation described by the bin.
class VotingBin {
 public:
  void Vote(const FeatureGeometryTransform& tform) {
    num_votes_ += 1;
    sum_tform_.scale += tform.scale;
    sum_tform_.angle += tform.angle;
    sum_tform_.tx += tform.tx;
    sum_tform_.ty += tform.ty;
  }

  // Get the number of votes.
  size_t GetNumVotes() const { return num_votes_; }

  // Compute the mean transformation of the voting bin.
  FeatureGeometryTransform GetTransformation() const {
    const float inv_num_votes = 1.0f / static_cast<float>(num_votes_);
    FeatureGeometryTransform tform = sum_tform_;
    tform.scale *= inv_num_votes;
    tform.angle *= inv_num_votes;
    tform.tx *= inv_num_votes;
    tform.ty *= inv_num_votes;
    return tform;
  }

 private:
  size_t num_votes_ = 0;
  FeatureGeometryTransform sum_tform_;
};

// Compute the difference in scale between the two features when aligning them
// with the given transformation.
float ComputeScaleError(const FeatureGeometry& feature1,
                        const FeatureGeometry& feature2,
                        const TwoWayTransform& tform) {
  const float area_transformed = feature1.GetAreaUnderTransform(tform.A21);
  const float area_measured = feature2.GetArea();
  if (area_transformed > area_measured) {
    return area_transformed / area_measured;
  } else {
    return area_measured / area_transformed;
  }
}

// Compute the two-way transfer error between two features.
float ComputeTransferError(const FeatureGeometry& feature1,
                           const FeatureGeometry& feature2,
                           const TwoWayTransform& tform) {
  const Eigen::Vector2f xy1(feature1.x, feature1.y);
  const Eigen::Vector2f xy2(feature2.x, feature2.y);
  const float error1 = (xy2 - tform.A12 * xy1 - tform.t12).squaredNorm();
  const float error2 = (xy1 - tform.A21 * xy2 - tform.t21).squaredNorm();
  return error1 + error2;
}

// Compute inlier matches that satisfy the transfer, scale thresholds.
void ComputeInlier(const TwoWayTransform& tform,
                   const std::vector<FeatureGeometryMatch>& matches,
                   const float max_transfer_error, const float max_scale_error,
                   std::vector<std::pair<int, int>>* inlier_idxs) {
  CHECK_GT(max_transfer_error, 0);
  CHECK_GT(max_scale_error, 0);

  inlier_idxs->clear();
  for (size_t i = 0; i < matches.size(); ++i) {
    const auto& match = matches[i];
    for (size_t j = 0; j < match.geometries2.size(); ++j) {
      const auto& geometry2 = match.geometries2[j];
      if (ComputeScaleError(match.geometry1, geometry2, tform) <=
              max_scale_error &&
          ComputeTransferError(match.geometry1, geometry2, tform) <=
              max_transfer_error) {
        inlier_idxs->emplace_back(i, j);
      }
    }
  }
}

// Compute effective inlier count that satisfy the transfer, scale thresholds.
size_t ComputeEffectiveInlierCount(
    const TwoWayTransform& tform,
    const std::vector<FeatureGeometryMatch>& matches,
    const float max_transfer_error, const float max_scale_error,
    const int num_bins) {
  CHECK_GT(max_transfer_error, 0);
  CHECK_GT(max_scale_error, 0);
  CHECK_GT(num_bins, 0);

  std::vector<std::pair<float, float>> inlier_coords;
  inlier_coords.reserve(matches.size());

  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = 0;
  float max_y = 0;

  for (const auto& match : matches) {
    for (const auto& geometry2 : match.geometries2) {
      if (ComputeScaleError(match.geometry1, geometry2, tform) <=
              max_scale_error &&
          ComputeTransferError(match.geometry1, geometry2, tform) <=
              max_transfer_error) {
        inlier_coords.emplace_back(match.geometry1.x, match.geometry1.y);
        min_x = std::min(min_x, match.geometry1.x);
        min_y = std::min(min_y, match.geometry1.y);
        max_x = std::max(max_x, match.geometry1.x);
        max_y = std::max(min_y, match.geometry1.y);
        break;
      }
    }
  }

  if (inlier_coords.empty()) {
    return 0;
  }

  const float scale_x = num_bins / (max_x - min_x);
  const float scale_y = num_bins / (max_y - min_y);

  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> counter(num_bins,
                                                                 num_bins);
  counter.setZero();

  for (const auto& coord : inlier_coords) {
    const int c_x = (coord.first - min_x) * scale_x;
    const int c_y = (coord.first - min_y) * scale_y;
    counter(std::max(0, std::min(num_bins - 1, c_x)),
            std::max(0, std::min(num_bins - 1, c_y))) = 1;
  }

  return counter.sum();
}

}  // namespace

int VoteAndVerify(const VoteAndVerifyOptions& options,
                  const std::vector<FeatureGeometryMatch>& matches) {
  CHECK_GT(options.num_transformations, 0);
  CHECK_GT(options.num_trans_bins, 0);
  CHECK_EQ(options.num_trans_bins % 2, 0);
  CHECK_GT(options.num_scale_bins, 0);
  CHECK_EQ(options.num_scale_bins % 2, 0);
  CHECK_GT(options.num_angle_bins, 0);
  CHECK_EQ(options.num_angle_bins % 2, 0);
  CHECK_GT(options.max_image_size, 0);
  CHECK_GT(options.min_num_votes, 0);
  CHECK_GE(options.confidence, 0);
  CHECK_LE(options.confidence, 1);

  if (matches.size() < AffineTransformEstimator::kMinNumSamples) {
    return 0;
  }

  const float max_trans = options.max_image_size;
  const float kMaxScale = 10.0f;
  const float max_log_scale = std::log2(kMaxScale);

  //////////////////////////////////////////////////////////////////////////////
  // Fill the multi-resolution voting histogram.
  //////////////////////////////////////////////////////////////////////////////

  const int kNumLevels = 6;
  std::array<std::unordered_map<size_t, VotingBin>, kNumLevels> bins;
  std::unordered_map<size_t, Eigen::Vector4i> coords;

  for (const auto& match : matches) {
    for (const auto& geometry2 : match.geometries2) {
      const auto T =
          FeatureGeometry::TransformFromMatch(match.geometry1, geometry2);

      if (std::abs(T.tx) > max_trans || std::abs(T.ty) > max_trans) {
        continue;
      }

      const float log_scale = std::log2(T.scale);
      if (std::abs(log_scale) > max_log_scale) {
        continue;
      }

      const float x = (T.tx + max_trans) / (2.0f * max_trans);
      const float y = (T.ty + max_trans) / (2.0f * max_trans);
      const float s = (log_scale + max_log_scale) / (2.0f * max_log_scale);
      const float o = (T.angle + M_PI) / (2.0f * M_PI);

      int n_x = std::min(static_cast<int>(x * options.num_trans_bins),
                         static_cast<int>(options.num_trans_bins - 1));
      int n_y = std::min(static_cast<int>(y * options.num_trans_bins),
                         static_cast<int>(options.num_trans_bins - 1));
      int n_s = std::min(static_cast<int>(s * options.num_scale_bins),
                         static_cast<int>(options.num_scale_bins - 1));
      int n_a = std::min(static_cast<int>(o * options.num_angle_bins),
                         static_cast<int>(options.num_angle_bins - 1));

      for (int level = 0; level < kNumLevels; ++level) {
        const uint64_t index =
            n_a + options.num_angle_bins * n_s +
            options.num_scale_bins * options.num_angle_bins * n_x +
            options.num_scale_bins * options.num_angle_bins *
                options.num_trans_bins * n_y;

        if (level == 0) {
          coords[index] = Eigen::Vector4i(n_a, n_s, n_x, n_y);
        }

        bins[level][index].Vote(T);

        n_x >>= 1;
        n_y >>= 1;
        n_s >>= 1;
        n_a >>= 1;
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Compute the multi-resolution scores for all occupied bins.
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::pair<int, float>> bin_scores;
  for (const auto& bin : bins[0]) {
    if (bin.second.GetNumVotes() >=
        static_cast<size_t>(options.min_num_votes)) {
      const auto coord = coords.at(bin.first);
      int n_a = coord(0);
      int n_s = coord(1);
      int n_x = coord(2);
      int n_y = coord(3);
      float score = bin.second.GetNumVotes();
      for (int level = 1; level < kNumLevels; ++level) {
        n_x >>= 1;
        n_y >>= 1;
        n_s >>= 1;
        n_a >>= 1;
        const uint64_t index =
            n_a + options.num_angle_bins * n_s +
            options.num_scale_bins * options.num_angle_bins * n_x +
            options.num_scale_bins * options.num_angle_bins *
                options.num_trans_bins * n_y;
        score +=
            bins[level][index].GetNumVotes() / static_cast<float>(1 << level);
      }
      bin_scores.emplace_back(bin.first, score);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Extract the top transformations.
  //////////////////////////////////////////////////////////////////////////////

  const size_t num_transformations = std::min(
      static_cast<size_t>(options.num_transformations), bin_scores.size());

  std::partial_sort(bin_scores.begin(),
                    bin_scores.begin() + num_transformations, bin_scores.end(),
                    [](const std::pair<int, float>& score1,
                       const std::pair<int, float>& score2) {
                      return score1.second > score2.second;
                    });

  //////////////////////////////////////////////////////////////////////////////
  // Verify the top transformations.
  //////////////////////////////////////////////////////////////////////////////

  size_t max_num_trials = std::numeric_limits<size_t>::max();
  size_t best_num_inliers = 0;
  TwoWayTransform best_tform;

  std::vector<std::pair<int, int>> inlier_idxs;
  std::vector<Eigen::Vector2d> inlier_points1;
  std::vector<Eigen::Vector2d> inlier_points2;

  for (size_t i = 0; i < num_transformations && i < max_num_trials; ++i) {
    const auto& bin = bins[0].at(bin_scores.at(i).first);
    const auto tform = TwoWayTransform(bin.GetTransformation());
    ComputeInlier(tform, matches, options.max_transfer_error,
                  options.max_scale_error, &inlier_idxs);

    if (inlier_idxs.size() < best_num_inliers ||
        inlier_idxs.size() < AffineTransformEstimator::kMinNumSamples) {
      continue;
    }

    best_num_inliers = inlier_idxs.size();
    best_tform = tform;

    // Collect matching inlier points.
    inlier_points1.resize(inlier_idxs.size());
    inlier_points2.resize(inlier_idxs.size());
    for (size_t j = 0; j < inlier_idxs.size(); ++j) {
      const auto& inlier_idx = inlier_idxs[j];
      const auto& match = matches.at(inlier_idx.first);
      const auto& geometry1 = match.geometry1;
      const auto& geometry2 = match.geometries2.at(inlier_idx.second);
      inlier_points1[j] = Eigen::Vector2d(geometry1.x, geometry1.y);
      inlier_points2[j] = Eigen::Vector2d(geometry2.x, geometry2.y);
    }

    // Local optimization on matching inlier points.
    const Eigen::Matrix<double, 2, 3> A =
        AffineTransformEstimator::Estimate(inlier_points1, inlier_points2)[0];
    Eigen::Matrix3d A_homogeneous = Eigen::Matrix3d::Identity();
    A_homogeneous.topRows<2>() = A;
    const Eigen::Matrix<double, 2, 3> inv_A =
        A_homogeneous.inverse().topRows<2>();

    TwoWayTransform local_tform;
    local_tform.A12 = A.leftCols<2>().cast<float>();
    local_tform.t12 = A.rightCols<1>().cast<float>();
    local_tform.A21 = inv_A.leftCols<2>().cast<float>();
    local_tform.t21 = inv_A.rightCols<1>().cast<float>();

    ComputeInlier(tform, matches, options.max_transfer_error,
                  options.max_scale_error, &inlier_idxs);

    if (inlier_idxs.size() > best_num_inliers) {
      best_num_inliers = inlier_idxs.size();
      best_tform = local_tform;
    }

    max_num_trials = RANSAC<AffineTransformEstimator>::ComputeNumTrials(
        best_num_inliers, AffineTransformEstimator::kMinNumSamples,
        options.confidence);
  }

  if (best_num_inliers == 0) {
    return 0;
  }

  const int kNumBins = 64;
  return ComputeEffectiveInlierCount(best_tform, matches,
                                     options.max_transfer_error,
                                     options.max_scale_error, kNumBins);
}

}  // namespace retrieval
}  // namespace colmap
