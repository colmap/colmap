// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#include "feature/sift.h"

#include <array>
#include <fstream>
#include <memory>

#include "FLANN/flann.hpp"
#if !defined(GUI_ENABLED) && !defined(CUDA_ENABLED)
#include "GL/glew.h"
#endif
#include "SiftGPU/SiftGPU.h"
#include "VLFeat/covdet.h"
#include "VLFeat/sift.h"
#include "feature/utils.h"
#include "util/cuda.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/opengl_utils.h"

namespace colmap {
namespace {

size_t FindBestMatchesOneWayBruteForce(const Eigen::MatrixXi& dists,
                                       const float max_ratio,
                                       const float max_distance,
                                       std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (Eigen::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const int dist = dists(i1, i2);
      if (dist > best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

void FindBestMatchesBruteForce(const Eigen::MatrixXi& dists,
                               const float max_ratio, const float max_distance,
                               const bool cross_check,
                               FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayBruteForce(
      dists, max_ratio, max_distance, &matches12);

  if (cross_check) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayBruteForce(
        dists.transpose(), max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

// Mutexes that ensure that only one thread extracts/matches on the same GPU
// at the same time, since SiftGPU internally uses static variables.
static std::map<int, std::unique_ptr<std::mutex>> sift_extraction_mutexes;
static std::map<int, std::unique_ptr<std::mutex>> sift_matching_mutexes;

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter) {
  if (guided_filter != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(keypoints1->size(), descriptors1.rows());
    CHECK_EQ(keypoints2->size(), descriptors2.rows());
  }

  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
      descriptors1.rows(), descriptors2.rows());

  for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
      if (guided_filter != nullptr &&
          guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                        (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
        dists(i1, i2) = 0;
      } else {
        dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
      }
    }
  }

  return dists;
}

void FindNearestNeighborsFLANN(
    const FeatureDescriptors& query, const FeatureDescriptors& database,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        indices,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        distances) {
  if (query.rows() == 0 || database.rows() == 0) {
    return;
  }

  const size_t kNumNearestNeighbors = 2;
  const size_t kNumTreesInForest = 4;

  const size_t num_nearest_neighbors =
      std::min(kNumNearestNeighbors, static_cast<size_t>(database.rows()));

  indices->resize(query.rows(), num_nearest_neighbors);
  distances->resize(query.rows(), num_nearest_neighbors);
  const flann::Matrix<uint8_t> query_matrix(const_cast<uint8_t*>(query.data()),
                                            query.rows(), 128);
  const flann::Matrix<uint8_t> database_matrix(
      const_cast<uint8_t*>(database.data()), database.rows(), 128);

  flann::Matrix<int> indices_matrix(indices->data(), query.rows(),
                                    num_nearest_neighbors);
  std::vector<float> distances_vector(query.rows() * num_nearest_neighbors);
  flann::Matrix<float> distances_matrix(distances_vector.data(), query.rows(),
                                        num_nearest_neighbors);
  flann::Index<flann::L2<uint8_t>> index(
      database_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
  index.buildIndex();
  index.knnSearch(query_matrix, indices_matrix, distances_matrix,
                  num_nearest_neighbors, flann::SearchParams(128));

  for (Eigen::Index query_index = 0; query_index < indices->rows();
       ++query_index) {
    for (Eigen::Index k = 0; k < indices->cols(); ++k) {
      const Eigen::Index database_index = indices->coeff(query_index, k);
      distances->coeffRef(query_index, k) =
          query.row(query_index)
              .cast<int>()
              .dot(database.row(database_index).cast<int>());
    }
  }
}

size_t FindBestMatchesOneWayFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances,
    const float max_ratio, const float max_distance,
    std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(indices.rows(), -1);

  for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (int n_idx = 0; n_idx < indices.cols(); ++n_idx) {
      const int d2_idx = indices(d1_idx, n_idx);
      const int dist = distances(d1_idx, n_idx);
      if (dist > best_dist) {
        best_i2 = d2_idx;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[d1_idx] = best_i2;
  }

  return num_matches;
}

void FindBestMatchesFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_2to1,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_2to1,
    const float max_ratio, const float max_distance, const bool cross_check,
    FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayFLANN(
      indices_1to2, distances_1to2, max_ratio, max_distance, &matches12);

  if (cross_check && indices_2to1.rows()) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayFLANN(
        indices_2to1, distances_2to1, max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void WarnIfMaxNumMatchesReachedGPU(const SiftMatchGPU& sift_match_gpu,
                                   const FeatureDescriptors& descriptors) {
  if (sift_match_gpu.GetMaxSift() < descriptors.rows()) {
    std::cout << StringPrintf(
                     "WARNING: Clamping features from %d to %d - consider "
                     "increasing the maximum number of matches.",
                     descriptors.rows(), sift_match_gpu.GetMaxSift())
              << std::endl;
  }
}

void WarnDarknessAdaptivityNotAvailable() {
  std::cout << "WARNING: Darkness adaptivity only available for GLSL SiftGPU."
            << std::endl;
}

}  // namespace

bool SiftExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  if (domain_size_pooling) {
    CHECK_OPTION_GT(dsp_min_scale, 0);
    CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
    CHECK_OPTION_GT(dsp_num_scales, 0);
  }
  return true;
}

bool SiftMatchingOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  CHECK_OPTION_GT(max_error, 0.0);
  CHECK_OPTION_GE(min_num_trials, 0);
  CHECK_OPTION_GT(max_num_trials, 0);
  CHECK_OPTION_LE(min_num_trials, max_num_trials);
  CHECK_OPTION_GE(min_inlier_ratio, 0);
  CHECK_OPTION_LE(min_inlier_ratio, 1);
  CHECK_OPTION_GE(min_num_inliers, 0);
  return true;
}

bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  CHECK(!options.estimate_affine_shape);
  CHECK(!options.domain_size_pooling);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(bitmap.Width(), bitmap.Height(), options.num_octaves,
                  options.octave_resolution, options.first_octave),
      &vl_sift_delete);
  if (!sift) {
    return false;
  }

  vl_sift_set_peak_thresh(sift.get(), options.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), options.edge_threshold);

  // Iterate through octaves.
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptors> level_descriptors;
  bool first_octave = true;
  while (true) {
    if (first_octave) {
      const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      if (vl_sift_process_first_octave(sift.get(), data_float.data())) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift.get())) {
        break;
      }
    }

    // Detect keypoints.
    vl_sift_detect(sift.get());

    // Extract detected keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift.get());
    const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level.
    size_t level_idx = 0;
    int prev_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level.
          level_keypoints.back().resize(level_idx);
          if (descriptors != nullptr) {
            level_descriptors.back().conservativeResize(level_idx, 128);
          }
        }

        // Add containers for new DOG level.
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(options.max_num_orientations *
                                     num_keypoints);
        if (descriptors != nullptr) {
          level_descriptors.emplace_back(
              options.max_num_orientations * num_keypoints, 128);
        }
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (options.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(
            sift.get(), angles, &vl_keypoints[i]);
      }

      // Note that this is different from SiftGPU, which selects the top
      // global maxima as orientations while this selects the first two
      // local maxima. It is not clear which procedure is better.
      const int num_used_orientations =
          std::min(num_orientations, options.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx] =
            FeatureKeypoint(vl_keypoints[i].x + 0.5f, vl_keypoints[i].y + 0.5f,
                            vl_keypoints[i].sigma, angles[o]);
        if (descriptors != nullptr) {
          Eigen::MatrixXf desc(1, 128);
          vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                           &vl_keypoints[i], angles[o]);
          if (options.normalization ==
              SiftExtractionOptions::Normalization::L2) {
            desc = L2NormalizeFeatureDescriptors(desc);
          } else if (options.normalization ==
                     SiftExtractionOptions::Normalization::L1_ROOT) {
            desc = L1RootNormalizeFeatureDescriptors(desc);
          } else {
            LOG(FATAL) << "Normalization type not supported";
          }

          level_descriptors.back().row(level_idx) =
              FeatureDescriptorsToUnsignedByte(desc);
        }

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave.
    level_keypoints.back().resize(level_idx);
    if (descriptors != nullptr) {
      level_descriptors.back().conservativeResize(level_idx, 128);
    }
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option.
  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = level_keypoints.size() - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += level_keypoints[i].size();
    if (num_features > options.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.
  {
    size_t k = 0;
    keypoints->resize(num_features_with_orientations);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        (*keypoints)[k] = level_keypoints[i][j];
        k += 1;
      }
    }
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    size_t k = 0;
    descriptors->resize(num_features_with_orientations, 128);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        descriptors->row(k) = level_descriptors[i].row(j);
        k += 1;
      }
    }
    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                                     const Bitmap& bitmap,
                                     FeatureKeypoints* keypoints,
                                     FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup covariant SIFT detector.
  std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
  if (!covdet) {
    return false;
  }

  const int kMaxOctaveResolution = 1000;
  CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

  vl_covdet_set_first_octave(covdet.get(), options.first_octave);
  vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
  vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
  vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

  {
    const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
    std::vector<float> data_float(data_uint8.size());
    for (size_t i = 0; i < data_uint8.size(); ++i) {
      data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
    vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
                        bitmap.Height());
  }

  vl_covdet_detect(covdet.get(), options.max_num_features);

  if (!options.upright) {
    if (options.estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }
  }

  const int num_features = vl_covdet_get_num_features(covdet.get());
  VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

  // Sort features according to detected octave and scale.
  std::sort(
      features, features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        } else {
          return feature1.o > feature2.o;
        }
      });

  const size_t max_num_features = static_cast<size_t>(options.max_num_features);

  // Copy detected keypoints and clamp when maximum number of features reached.
  int prev_octave_scale_idx = std::numeric_limits<int>::max();
  for (int i = 0; i < num_features; ++i) {
    FeatureKeypoint keypoint;
    keypoint.x = features[i].frame.x + 0.5;
    keypoint.y = features[i].frame.y + 0.5;
    keypoint.a11 = features[i].frame.a11;
    keypoint.a12 = features[i].frame.a12;
    keypoint.a21 = features[i].frame.a21;
    keypoint.a22 = features[i].frame.a22;
    keypoints->push_back(keypoint);

    const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
    CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

    if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints->size() >= max_num_features) {
      break;
    }

    prev_octave_scale_idx = octave_scale_idx;
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    descriptors->resize(keypoints->size(), 128);

    const size_t kPatchResolution = 15;
    const size_t kPatchSide = 2 * kPatchResolution + 1;
    const double kPatchRelativeExtent = 7.5;
    const double kPatchRelativeSmoothing = 1;
    const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
    const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

    std::vector<float> patch(kPatchSide * kPatchSide);
    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

    float dsp_min_scale = 1;
    float dsp_scale_step = 0;
    int dsp_num_scales = 1;
    if (options.domain_size_pooling) {
      dsp_min_scale = options.dsp_min_scale;
      dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
                       options.dsp_num_scales;
      dsp_num_scales = options.dsp_num_scales;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
        scaled_descriptors(dsp_num_scales, 128);

    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    if (!sift) {
      return false;
    }

    vl_sift_set_magnif(sift.get(), 3.0);

    for (size_t i = 0; i < keypoints->size(); ++i) {
      for (int s = 0; s < dsp_num_scales; ++s) {
        const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

        VlFrameOrientedEllipse scaled_frame = features[i].frame;
        scaled_frame.a11 *= dsp_scale;
        scaled_frame.a12 *= dsp_scale;
        scaled_frame.a21 *= dsp_scale;
        scaled_frame.a22 *= dsp_scale;

        vl_covdet_extract_patch_for_frame(
            covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
            kPatchRelativeSmoothing, scaled_frame);

        vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
                              2 * kPatchSide, patch.data(), kPatchSide,
                              kPatchSide, kPatchSide);

        vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
                                    scaled_descriptors.row(s).data(),
                                    kPatchSide, kPatchSide, kPatchResolution,
                                    kPatchResolution, kSigma, 0);
      }

      Eigen::Matrix<float, 1, 128> descriptor;
      if (options.domain_size_pooling) {
        descriptor = scaled_descriptors.colwise().mean();
      } else {
        descriptor = scaled_descriptors;
      }

      if (options.normalization == SiftExtractionOptions::Normalization::L2) {
        descriptor = L2NormalizeFeatureDescriptors(descriptor);
      } else if (options.normalization ==
                 SiftExtractionOptions::Normalization::L1_ROOT) {
        descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
      } else {
        LOG(FATAL) << "Normalization type not supported";
      }

      descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
    }

    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu) {
  CHECK(options.Check());
  CHECK_NOTNULL(sift_gpu);

  // SiftGPU uses many global static state variables and the initialization must
  // be thread-safe in order to work correctly. This is enforced here.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  std::vector<std::string> sift_gpu_args;

  sift_gpu_args.push_back("./sift_gpu");

#ifdef CUDA_ENABLED
  // Use CUDA version by default if darkness adaptivity is disabled.
  if (!options.darkness_adaptivity && gpu_indices[0] < 0) {
    gpu_indices[0] = 0;
  }

  if (gpu_indices[0] >= 0) {
    sift_gpu_args.push_back("-cuda");
    sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
  }
#endif  // CUDA_ENABLED

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  if (options.darkness_adaptivity) {
    if (gpu_indices[0] >= 0) {
      WarnDarknessAdaptivityNotAvailable();
    }
    sift_gpu_args.push_back("-da");
  }

  // No verbose logging.
  sift_gpu_args.push_back("-v");
  sift_gpu_args.push_back("0");

  // Fixed maximum image dimension.
  sift_gpu_args.push_back("-maxd");
  sift_gpu_args.push_back(std::to_string(options.max_image_size));

  // Keep the highest level features.
  sift_gpu_args.push_back("-tc2");
  sift_gpu_args.push_back(std::to_string(options.max_num_features));

  // First octave level.
  sift_gpu_args.push_back("-fo");
  sift_gpu_args.push_back(std::to_string(options.first_octave));

  // Number of octave levels.
  sift_gpu_args.push_back("-d");
  sift_gpu_args.push_back(std::to_string(options.octave_resolution));

  // Peak threshold.
  sift_gpu_args.push_back("-t");
  sift_gpu_args.push_back(std::to_string(options.peak_threshold));

  // Edge threshold.
  sift_gpu_args.push_back("-e");
  sift_gpu_args.push_back(std::to_string(options.edge_threshold));

  if (options.upright) {
    // Fix the orientation to 0 for upright features.
    sift_gpu_args.push_back("-ofix");
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back("1");
  } else {
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back(std::to_string(options.max_num_orientations));
  }

  std::vector<const char*> sift_gpu_args_cstr;
  sift_gpu_args_cstr.reserve(sift_gpu_args.size());
  for (const auto& arg : sift_gpu_args) {
    sift_gpu_args_cstr.push_back(arg.c_str());
  }

  sift_gpu->ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

  sift_gpu->gpu_index = gpu_indices[0];
  if (sift_extraction_mutexes.count(gpu_indices[0]) == 0) {
    sift_extraction_mutexes.emplace(
        gpu_indices[0], std::unique_ptr<std::mutex>(new std::mutex()));
  }

  return sift_gpu->VerifyContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
}

bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

  CHECK(!options.estimate_affine_shape);
  CHECK(!options.domain_size_pooling);

  std::unique_lock<std::mutex> lock(
      *sift_extraction_mutexes[sift_gpu->gpu_index]);

  // Note, that this produces slightly different results than using SiftGPU
  // directly for RGB->GRAY conversion, since it uses different weights.
  const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
  const int code =
      sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
                        bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

  const int kSuccessCode = 1;
  if (code != kSuccessCode) {
    return false;
  }

  const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());

  std::vector<SiftKeypoint> keypoints_data(num_features);

  // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_float(num_features, 128);

  // Download the extracted keypoints and descriptors.
  sift_gpu->GetFeatureVector(keypoints_data.data(), descriptors_float.data());

  keypoints->resize(num_features);
  for (size_t i = 0; i < num_features; ++i) {
    (*keypoints)[i] = FeatureKeypoint(keypoints_data[i].x, keypoints_data[i].y,
                                      keypoints_data[i].s, keypoints_data[i].o);
  }

  // Save and normalize the descriptors.
  if (options.normalization == SiftExtractionOptions::Normalization::L2) {
    descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (options.normalization ==
             SiftExtractionOptions::Normalization::L1_ROOT) {
    descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
  } else {
    LOG(FATAL) << "Normalization type not supported";
  }

  *descriptors = FeatureDescriptorsToUnsignedByte(descriptors_float);

  return true;
}

void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors) {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);

  std::ifstream file(path.c_str());
  CHECK(file.is_open()) << path;

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const point2D_t num_features = std::stoul(item);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const size_t dim = std::stoul(item);

  CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

  keypoints->resize(num_features);
  descriptors->resize(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float x = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float y = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float scale = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float orientation = std::stold(item);

    (*keypoints)[i] = FeatureKeypoint(x, y, scale, orientation);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream >> std::ws, item, ' ');
      const float value = std::stod(item);
      CHECK_GE(value, 0);
      CHECK_LE(value, 255);
      (*descriptors)(i, j) = TruncateCast<float, uint8_t>(value);
    }
  }
}

void MatchSiftFeaturesCPUBruteForce(const SiftMatchingOptions& match_options,
                                    const FeatureDescriptors& descriptors1,
                                    const FeatureDescriptors& descriptors2,
                                    FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  const Eigen::MatrixXi distances = ComputeSiftDistanceMatrix(
      nullptr, nullptr, descriptors1, descriptors2, nullptr);

  FindBestMatchesBruteForce(distances, match_options.max_ratio,
                            match_options.max_distance,
                            match_options.cross_check, matches);
}

void MatchSiftFeaturesCPUFLANN(const SiftMatchingOptions& match_options,
                               const FeatureDescriptors& descriptors1,
                               const FeatureDescriptors& descriptors2,
                               FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_2to1;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_2to1;

  FindNearestNeighborsFLANN(descriptors1, descriptors2, &indices_1to2,
                            &distances_1to2);
  if (match_options.cross_check) {
    FindNearestNeighborsFLANN(descriptors2, descriptors1, &indices_2to1,
                              &distances_2to1);
  }

  FindBestMatchesFLANN(indices_1to2, distances_1to2, indices_2to1,
                       distances_2to1, match_options.max_ratio,
                       match_options.max_distance, match_options.cross_check,
                       matches);
}

void MatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {
  MatchSiftFeaturesCPUFLANN(match_options, descriptors1, descriptors2, matches);
}

void MatchGuidedSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(two_view_geometry);

  const float max_residual = match_options.max_error * match_options.max_error;

  const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
  const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();

  std::function<bool(float, float, float, float)> guided_filter;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector3f p2(x2, y2, 1.0f);
      const Eigen::Vector3f Fx1 = F * p1;
      const Eigen::Vector3f Ftx2 = F.transpose() * p2;
      const float x2tFx1 = p2.transpose() * Fx1;
      return x2tFx1 * x2tFx1 /
                 (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) +
                  Ftx2(1) * Ftx2(1)) >
             max_residual;
    };
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector2f p2(x2, y2);
      return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
    };
  } else {
    return;
  }

  CHECK(guided_filter);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      &keypoints1, &keypoints2, descriptors1, descriptors2, guided_filter);

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_1to2(dists.rows(), dists.cols());
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_2to1(dists.cols(), dists.rows());
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_1to2 = dists;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_2to1 = dists.transpose();

  for (int i = 0; i < indices_1to2.rows(); ++i) {
    indices_1to2.row(i) = Eigen::VectorXi::LinSpaced(indices_1to2.cols(), 0,
                                                     indices_1to2.cols() - 1);
  }
  for (int i = 0; i < indices_2to1.rows(); ++i) {
    indices_2to1.row(i) = Eigen::VectorXi::LinSpaced(indices_2to1.cols(), 0,
                                                     indices_2to1.cols() - 1);
  }

  FindBestMatchesFLANN(indices_1to2, distances_1to2, indices_2to1,
                       distances_2to1, match_options.max_ratio,
                       match_options.max_distance, match_options.cross_check,
                       &two_view_geometry->inlier_matches);
}

bool CreateSiftGPUMatcher(const SiftMatchingOptions& match_options,
                          SiftMatchGPU* sift_match_gpu) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);

  // SiftGPU uses many global static state variables and the initialization must
  // be thread-safe in order to work correctly. This is enforced here.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  const std::vector<int> gpu_indices =
      CSVToVector<int>(match_options.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  SiftGPU sift_gpu;
  sift_gpu.SetVerbose(0);

  *sift_match_gpu = SiftMatchGPU(match_options.max_num_matches);

#ifdef CUDA_ENABLED
  if (gpu_indices[0] >= 0) {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                gpu_indices[0]);
  } else {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
  }
#else   // CUDA_ENABLED
  sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif  // CUDA_ENABLED

  if (sift_match_gpu->VerifyContextGL() == 0) {
    return false;
  }

  if (!sift_match_gpu->Allocate(match_options.max_num_matches,
                                match_options.cross_check)) {
    std::cout << StringPrintf(
                     "ERROR: Not enough GPU memory to match %d features. "
                     "Reduce the maximum number of matches.",
                     match_options.max_num_matches)
              << std::endl;
    return false;
  }

#ifndef CUDA_ENABLED
  if (sift_match_gpu->GetMaxSift() < match_options.max_num_matches) {
    std::cout << StringPrintf(
                     "WARNING: OpenGL version of SiftGPU only supports a "
                     "maximum of %d matches - consider changing to CUDA-based "
                     "feature matching to avoid this limitation.",
                     sift_match_gpu->GetMaxSift())
              << std::endl;
  }
#endif  // CUDA_ENABLED

  sift_match_gpu->gpu_index = gpu_indices[0];
  if (sift_matching_mutexes.count(gpu_indices[0]) == 0) {
    sift_matching_mutexes.emplace(
        gpu_indices[0], std::unique_ptr<std::mutex>(new std::mutex()));
  }

  return true;
}

void MatchSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors* descriptors1,
                          const FeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(matches);

  std::unique_lock<std::mutex> lock(
      *sift_matching_mutexes[sift_match_gpu->gpu_index]);

  if (descriptors1 != nullptr) {
    CHECK_EQ(descriptors1->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
    sift_match_gpu->SetDescriptors(0, descriptors1->rows(),
                                   descriptors1->data());
  }

  if (descriptors2 != nullptr) {
    CHECK_EQ(descriptors2->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
    sift_match_gpu->SetDescriptors(1, descriptors2->rows(),
                                   descriptors2->data());
  }

  matches->resize(static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(matches->data()),
      static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio), match_options.cross_check);

  if (num_matches < 0) {
    std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                 "insufficient GPU memory. Consider reducing the maximum "
                 "number of features and/or matches."
              << std::endl;
    matches->clear();
  } else {
    CHECK_LE(num_matches, matches->size());
    matches->resize(num_matches);
  }
}

void MatchGuidedSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const FeatureDescriptors* descriptors1,
                                const FeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry) {
  static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                "Invalid keypoint format");
  static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                "Invalid keypoint format");
  static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                "Invalid keypoint format");

  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(two_view_geometry);

  std::unique_lock<std::mutex> lock(
      *sift_matching_mutexes[sift_match_gpu->gpu_index]);

  const size_t kFeatureShapeNumElems = 4;

  if (descriptors1 != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_EQ(descriptors1->rows(), keypoints1->size());
    CHECK_EQ(descriptors1->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
    const size_t kIndex = 0;
    sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(),
                                   descriptors1->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints1->data()),
        kFeatureShapeNumElems);
  }

  if (descriptors2 != nullptr) {
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(descriptors2->rows(), keypoints2->size());
    CHECK_EQ(descriptors2->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
    const size_t kIndex = 1;
    sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(),
                                   descriptors2->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints2->data()),
        kFeatureShapeNumElems);
  }

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F;
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
  float* F_ptr = nullptr;
  float* H_ptr = nullptr;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    F = two_view_geometry->F.cast<float>();
    F_ptr = F.data();
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    H = two_view_geometry->H.cast<float>();
    H_ptr = H.data();
  } else {
    return;
  }

  CHECK(F_ptr != nullptr || H_ptr != nullptr);

  two_view_geometry->inlier_matches.resize(
      static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(
          two_view_geometry->inlier_matches.data()),
      H_ptr, F_ptr, static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio),
      static_cast<float>(match_options.max_error * match_options.max_error),
      static_cast<float>(match_options.max_error * match_options.max_error),
      match_options.cross_check);

  if (num_matches < 0) {
    std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                 "insufficient GPU memory. Consider reducing the maximum "
                 "number of features."
              << std::endl;
    two_view_geometry->inlier_matches.clear();
  } else {
    CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
    two_view_geometry->inlier_matches.resize(num_matches);
  }
}

}  //  namespace colmap
