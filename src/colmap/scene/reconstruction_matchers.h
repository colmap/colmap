// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/estimators/alignment.h"
#include "colmap/scene/reconstruction.h"

#include <limits>
#include <optional>

#include <gmock/gmock.h>

namespace colmap {

template <typename T>
class ReconstructionEqMatcher : public testing::MatcherInterface<T> {
 public:
  explicit ReconstructionEqMatcher(T rhs) : rhs_(std::forward<T>(rhs)) {}

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (lhs.Rigs() != rhs_.Rigs()) {
      *listener << " have different rigs";
      return false;
    }
    if (lhs.Cameras() != rhs_.Cameras()) {
      *listener << " have different cameras";
      return false;
    }
    if (lhs.Frames() != rhs_.Frames()) {
      *listener << " have different frames";
      return false;
    }
    if (lhs.Images() != rhs_.Images()) {
      *listener << " have different images";
      return false;
    }
    if (lhs.Points3D() != rhs_.Points3D()) {
      *listener << " have different points";
      return false;
    }
    return true;
  }

 private:
  const Reconstruction rhs_;
};

// Matcher to check for exact equality of two reconstructions.
template <typename T>
testing::PolymorphicMatcher<ReconstructionEqMatcher<T>> ReconstructionEq(
    T rhs) {
  return testing::MakePolymorphicMatcher(
      ReconstructionEqMatcher<T>(std::forward<T>(rhs)));
}

template <typename T>
class ReconstructionNearMatcher : public testing::MatcherInterface<T> {
 public:
  ReconstructionNearMatcher(T rhs,
                            double max_rotation_error_deg,
                            double max_proj_center_error,
                            std::optional<double> max_scale_error,
                            double num_obs_tolerance,
                            bool align = true)
      : rhs_(std::forward<T>(rhs)),
        max_rotation_error_deg_(max_rotation_error_deg),
        max_proj_center_error_(max_proj_center_error),
        max_scale_error_(max_scale_error),
        num_obs_tolerance_(num_obs_tolerance),
        align_(align) {
    CHECK_GE(max_rotation_error_deg, 0.0);
    CHECK_GE(max_proj_center_error, 0.0);
    if (max_scale_error.has_value()) {
      CHECK_GE(*max_scale_error, 0.0);
    }
    CHECK_GE(num_obs_tolerance, 0.0);
  }

  void DescribeTo(std::ostream* os) const override { *os << rhs_; }

  bool MatchAndExplain(T lhs,
                       testing::MatchResultListener* listener) const override {
    if (lhs.NumRigs() != rhs_.NumRigs()) {
      *listener << " have different number of rigs: " << lhs.NumRigs() << " vs "
                << rhs_.NumRigs();
      return false;
    }
    if (lhs.NumCameras() != rhs_.NumCameras()) {
      *listener << " have different number of cameras: " << lhs.NumCameras()
                << " vs " << rhs_.NumCameras();
      return false;
    }
    if (lhs.NumFrames() != rhs_.NumFrames()) {
      *listener << " have different number of frames: " << lhs.NumFrames()
                << " vs " << rhs_.NumFrames();
      return false;
    }
    if (lhs.NumImages() != rhs_.NumImages()) {
      *listener << " have different number of images: " << lhs.NumImages()
                << " vs " << rhs_.NumImages();
      return false;
    }
    if (lhs.NumRegImages() != rhs_.NumRegImages()) {
      *listener << " have different number of registered images: "
                << lhs.NumRegImages() << " vs " << rhs_.NumRegImages();
      return false;
    }
    const size_t lhs_num_obs = lhs.ComputeNumObservations();
    const size_t rhs_num_obs = rhs_.ComputeNumObservations();
    if ((lhs_num_obs != 0 || rhs_num_obs != 0) &&
        std::max(std::abs(1 - lhs_num_obs / static_cast<double>(rhs_num_obs)),
                 std::abs(1 - rhs_num_obs / static_cast<double>(lhs_num_obs))) >
            num_obs_tolerance_) {
      *listener << " have different number of observations: " << lhs_num_obs
                << " vs " << rhs_num_obs;
      return false;
    }

    Sim3d rhs_from_lhs;
    if (align_) {
      if (!AlignReconstructionsViaProjCenters(
              lhs,
              rhs_,
              /*max_proj_center_error=*/
              std::max(max_proj_center_error_,
                       std::numeric_limits<double>::epsilon()),
              &rhs_from_lhs)) {
        *listener << " failed to align";
        return false;
      }
      if (max_scale_error_.has_value() &&
          !(std::abs(rhs_from_lhs.scale - 1.0) < *max_scale_error_)) {
        *listener << " have different scale: " << rhs_from_lhs.scale;
        return false;
      }
    }

    const std::vector<ImageAlignmentError> errors =
        ComputeImageAlignmentError(lhs, rhs_, rhs_from_lhs);
    THROW_CHECK_EQ(errors.size(), rhs_.NumImages());
    for (const auto& error : errors) {
      if (error.rotation_error_deg > max_rotation_error_deg_) {
        *listener << "Image with name " << error.image_name
                  << " exceeds rotation error threshold: "
                  << error.rotation_error_deg << " vs "
                  << max_rotation_error_deg_;
        return false;
      }
      if (error.proj_center_error > max_proj_center_error_) {
        *listener << "Image with name " << error.image_name
                  << " exceeds projection center error threshold: "
                  << error.proj_center_error << " vs "
                  << max_proj_center_error_;
        return false;
      }
    }

    return true;
  }

 private:
  const Reconstruction rhs_;
  const double max_rotation_error_deg_;
  const double max_proj_center_error_;
  const std::optional<double> max_scale_error_;
  const double num_obs_tolerance_;
  const bool align_;
};

// Matcher to check for approximate equality of two reconstructions. Optionally
// aligns the two reconstruction worlds through common shared registered images.
template <typename T>
testing::PolymorphicMatcher<ReconstructionNearMatcher<T>> ReconstructionNear(
    T rhs,
    double max_rotation_error_deg = 1e-6,
    double max_proj_center_error = 1e-6,
    std::optional<double> max_scale_error = std::nullopt,
    double num_obs_tolerance = 0.0,
    bool align = true) {
  return testing::MakePolymorphicMatcher(
      ReconstructionNearMatcher<T>(std::forward<T>(rhs),
                                   max_rotation_error_deg,
                                   max_proj_center_error,
                                   max_scale_error,
                                   num_obs_tolerance,
                                   align));
}

}  // namespace colmap
