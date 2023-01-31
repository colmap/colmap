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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_FEATURE_SIFT_H_
#define COLMAP_SRC_FEATURE_SIFT_H_

#include "estimators/two_view_geometry.h"
#include "feature/types.h"
#include "util/bitmap.h"

class SiftGPU;
class SiftMatchGPU;

namespace colmap {

struct SiftExtractionOptions {
  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 3200;

  // Maximum number of features to detect, keeping larger-scale features.
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level.
  int first_octave = -1;

  // Number of octaves.
  int num_octaves = 4;

  // Number of levels per octave.
  int octave_resolution = 3;

  // Peak threshold for detection.
  double peak_threshold = 0.02 / octave_resolution;

  // Edge threshold for detection.
  double edge_threshold = 10.0;

  // Estimate affine shape of SIFT features in the form of oriented ellipses as
  // opposed to original SIFT which estimates oriented disks.
  bool estimate_affine_shape = false;

  // Maximum number of orientations per keypoint if not estimate_affine_shape.
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features.
  bool upright = false;

  // Whether to adapt the feature detection depending on the image darkness.
  // Note that this feature is only available in the OpenGL SiftGPU version.
  bool darkness_adaptivity = false;

  // Domain-size pooling parameters. Domain-size pooling computes an average
  // SIFT descriptor across multiple scales around the detected scale. This was
  // proposed in "Domain-Size Pooling in Local Descriptors and Network
  // Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
  // outperform other SIFT variants and learned descriptors in "Comparative
  // Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
  // Hardmeier, Sattler, Pollefeys, CVPR 2016.
  bool domain_size_pooling = false;
  double dsp_min_scale = 1.0 / 6.0;
  double dsp_max_scale = 3.0;
  int dsp_num_scales = 10;

  enum class Normalization {
    // L1-normalizes each descriptor followed by element-wise square rooting.
    // This normalization is usually better than standard L2-normalization.
    // See "Three things everyone should know to improve object retrieval",
    // Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
    L1_ROOT,
    // Each vector is L2-normalized.
    L2,
  };
  Normalization normalization = Normalization::L1_ROOT;

  bool Check() const;
};

struct SiftMatchingOptions {
  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
  bool use_gpu = true;

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum distance ratio between first and second best match.
  double max_ratio = 0.8;

  // Maximum distance to best match.
  double max_distance = 0.7;

  // Whether to enable cross checking in matching.
  bool cross_check = true;

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Maximum epipolar error in pixels for geometric verification.
  double max_error = 4.0;

  // Confidence threshold for geometric verification.
  double confidence = 0.999;

  // Minimum/maximum number of RANSAC iterations. Note that this option
  // overrules the min_inlier_ratio option.
  int min_num_trials = 100;
  int max_num_trials = 10000;

  // A priori assumed minimum inlier ratio, which determines the maximum
  // number of iterations.
  double min_inlier_ratio = 0.25;

  // Minimum number of inliers for an image pair to be considered as
  // geometrically verified.
  int min_num_inliers = 15;

  // Whether to attempt to estimate multiple geometric models per image pair.
  bool multiple_models = false;

  // Whether to perform guided matching, if geometric verification succeeds.
  bool guided_matching = false;

  // Force Homography use for Two-view Geometry (can help for planar scenes)
  bool planar_scene = false;

  // Whether to estimate the relative pose between the two images and save them
  // to the DB.
  bool compute_relative_pose = false;

  bool Check() const;
};

// Extract SIFT features for the given image on the CPU. Only extract
// descriptors if the given input is not NULL.
bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors);
bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                                     const Bitmap& bitmap,
                                     FeatureKeypoints* keypoints,
                                     FeatureDescriptors* descriptors);

// Create a SiftGPU feature extractor. The same SiftGPU instance can be used to
// extract features for multiple images. Note a OpenGL context must be made
// current in the thread of the caller. If the gpu_index is not -1, the CUDA
// version of SiftGPU is used, which produces slightly different results
// than the OpenGL implementation.
bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu);

// Extract SIFT features for the given image on the GPU.
// SiftGPU must already be initialized using `CreateSiftGPU`.
bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors);

// Load keypoints and descriptors from text file in the following format:
//
//    LINE_0:            NUM_FEATURES DIM
//    LINE_1:            X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    LINE_I:            ...
//    LINE_NUM_FEATURES: X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//
// where the first line specifies the number of features and the descriptor
// dimensionality followed by one line per feature: X, Y, SCALE, ORIENTATION are
// of type float and D_J represent the descriptor in the range [0, 255].
//
// For example:
//
//    2 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//
void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors);

// Match the given SIFT features on the CPU.
void MatchSiftFeaturesCPUBruteForce(const SiftMatchingOptions& match_options,
                                    const FeatureDescriptors& descriptors1,
                                    const FeatureDescriptors& descriptors2,
                                    FeatureMatches* matches);
void MatchSiftFeaturesCPUFLANN(const SiftMatchingOptions& match_options,
                               const FeatureDescriptors& descriptors1,
                               const FeatureDescriptors& descriptors2,
                               FeatureMatches* matches);
void MatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches);
void MatchGuidedSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry);

// Create a SiftGPU feature matcher. Note that if CUDA is not available or the
// gpu_index is -1, the OpenGLContextManager must be created in the main thread
// of the Qt application before calling this function. The same SiftMatchGPU
// instance can be used to match features between multiple image pairs.
bool CreateSiftGPUMatcher(const SiftMatchingOptions& match_options,
                          SiftMatchGPU* sift_match_gpu);

// Match the given SIFT features on the GPU. If either of the descriptors is
// NULL, the keypoints/descriptors will not be uploaded and the previously
// uploaded descriptors will be reused for the matching.
void MatchSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors* descriptors1,
                          const FeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches);
void MatchGuidedSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const FeatureDescriptors* descriptors1,
                                const FeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry);

}  // namespace colmap

#endif  // COLMAP_SRC_FEATURE_SIFT_H_
