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

#include "mvs/patch_match.h"

#include <cmath>

#include "mvs/patch_match_cuda.h"

#define PrintOption(option) std::cout << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

PatchMatch::PatchMatch(const Options& options, const Problem& problem)
    : options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() {}

void PatchMatch::Options::Print() const {
  std::cout << "PatchMatch::Options" << std::endl;
  std::cout << "-------------------" << std::endl;
  PrintOption(depth_min);
  PrintOption(depth_max);
  PrintOption(window_radius);
  PrintOption(sigma_spatial);
  PrintOption(sigma_color);
  PrintOption(num_samples);
  PrintOption(ncc_sigma);
  PrintOption(min_triangulation_angle);
  PrintOption(incident_angle_sigma);
  PrintOption(num_iterations);
  PrintOption(geom_consistency);
  PrintOption(geom_consistency_regularizer);
  PrintOption(geom_consistency_max_cost);
  PrintOption(filter);
  PrintOption(filter_min_ncc);
  PrintOption(filter_min_triangulation_angle);
  PrintOption(filter_min_num_consistent);
  PrintOption(filter_geom_consistency_max_cost);
}

void PatchMatch::Problem::Print() const {
  std::cout << "PatchMatch::Problem" << std::endl;
  std::cout << "-------------------" << std::endl;
  PrintOption(ref_image_id);

  std::cout << "src_image_ids: ";
  if (!src_image_ids.empty()) {
    for (size_t i = 0; i < src_image_ids.size() - 1; ++i) {
      std::cout << src_image_ids[i] << " ";
    }
    std::cout << src_image_ids.back() << std::endl;
  } else {
    std::cout << std::endl;
  }
}

void PatchMatch::Check() const {
  CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency) {
    CHECK_NOTNULL(problem_.depth_maps);
    CHECK_NOTNULL(problem_.normal_maps);
    CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }

  CHECK_GT(problem_.src_image_ids.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_ids(problem_.src_image_ids.begin(),
                                 problem_.src_image_ids.end());
  unique_image_ids.insert(problem_.ref_image_id);
  CHECK_EQ(problem_.src_image_ids.size() + 1, unique_image_ids.size());

  // Check that input data is well-formed.
  for (const int image_id : unique_image_ids) {
    CHECK_GE(image_id, 0) << image_id;
    CHECK_LT(image_id, problem_.images->size()) << image_id;

    const Image& image = problem_.images->at(image_id);
    CHECK_GT(image.GetWidth(), 0) << image_id;
    CHECK_GT(image.GetHeight(), 0) << image_id;
    CHECK_EQ(image.GetChannels(), 1) << image_id;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_id;

    if (options_.geom_consistency) {
      CHECK_LT(image_id, problem_.depth_maps->size()) << image_id;
      const DepthMap& depth_map = problem_.depth_maps->at(image_id);
      CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_id;
      CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_id;
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_id);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_id);
    CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }

  CHECK_LT(options_.depth_min, options_.depth_max);
  CHECK_GT(options_.depth_min, 0.0f);
  CHECK_LE(options_.window_radius, kMaxWindowRadius);
  CHECK_GT(options_.sigma_spatial, 0.0f);
  CHECK_GT(options_.sigma_color, 0.0f);
  CHECK_GT(options_.window_radius, 0);
  CHECK_GT(options_.num_samples, 0);
  CHECK_GT(options_.ncc_sigma, 0.0f);
  CHECK_GE(options_.min_triangulation_angle, 0.0f);
  CHECK_LT(options_.min_triangulation_angle, DEG2RAD(180.0f));
  CHECK_GT(options_.incident_angle_sigma, 0.0f);
  CHECK_GT(options_.num_iterations, 0);
  CHECK_GE(options_.geom_consistency_regularizer, 0.0f);
  CHECK_GE(options_.geom_consistency_max_cost, 0.0f);
  CHECK_GE(options_.filter_min_ncc, -1.0f);
  CHECK_LE(options_.filter_min_ncc, 1.0f);
  CHECK_GE(options_.filter_min_triangulation_angle, 0.0f);
  CHECK_LE(options_.filter_min_triangulation_angle, DEG2RAD(180.0f));
  CHECK_GE(options_.filter_min_num_consistent, 0);
  CHECK_GE(options_.filter_geom_consistency_max_cost, 0.0f);
}

void PatchMatch::Run() {
  std::cout << "PatchMatch::Run" << std::endl;
  std::cout << "---------------" << std::endl;

  Check();

  patch_match_cuda_.reset(new PatchMatchCuda(options_, problem_));
  patch_match_cuda_->Run();
}

DepthMap PatchMatch::GetDepthMap() const {
  return patch_match_cuda_->GetDepthMap();
}

NormalMap PatchMatch::GetNormalMap() const {
  return patch_match_cuda_->GetNormalMap();
}

Mat<float> PatchMatch::GetSelProbMap() const {
  return patch_match_cuda_->GetSelProbMap();
}

std::vector<int> PatchMatch::GetConsistentImageIds() const {
  return patch_match_cuda_->GetConsistentImageIds();
}

}  // namespace mvs
}  // namespace colmap
