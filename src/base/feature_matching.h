// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#ifndef COLMAP_SRC_BASE_FEATURE_MATCHING_H_
#define COLMAP_SRC_BASE_FEATURE_MATCHING_H_

#include <string>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>

#include <QMutex>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QThread>

#include "base/database.h"
#include "ext/SiftGPU/SiftGPU.h"
#include "ext/VocabLib/VocabTree.h"
#include "util/bitmap.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

// Abstract feature matching base class. It implements basic feature matching
// and geometric verification for image pairs. The job of the child classes is
// only to select the image pairs to be matched by implementing the DoMatching
// method and calling the MatchImagePairs for batches of image pairs. This class
// automatically caches keypoints and descriptors between consecutive calls to
// MatchImagePairs to reduce disk access.
class FeatureMatcher : public QThread {
 public:
  struct Options {
    // Number of threads for geometric verification.
    int num_threads = ThreadPool::kMaxNumThreads;

    // Index of the GPU used for feature matching.
    int gpu_index = -1;

    // Maximum distance ratio between first and second best match.
    double max_ratio = 0.8;

    // Maximum distance to best match.
    double max_distance = 0.7;

    // Whether to enable cross checking in matching.
    bool cross_check = true;

    // Maximum number of matches.
    int max_num_matches = 8192;

    // Maximum epipolar error in pixels for geometric verification.
    double max_error = 4.0;

    // Confidence threshold for geometric verification.
    double confidence = 0.999;

    // Maximum number of RANSAC iterations. Note that this option overrules
    // the min_inlier_ratio option.
    int max_num_trials = 10000;

    // A priori assumed minimum inlier ratio, which determines the maximum
    // number of iterations.
    double min_inlier_ratio = 0.25;

    // Minimum number of inliers for an image pair to be considered as
    // geometrically verified.
    int min_num_inliers = 15;

    // Whether to attempt to estimate multiple models per image pair.
    bool multiple_models = false;

    void Check() const;
  };

  FeatureMatcher(const Options& options, const std::string& database_path);
  ~FeatureMatcher();

  void run();
  virtual void Stop();

 protected:
  struct GeometricVerificationData {
    const Camera* camera1;
    const Camera* camera2;
    const FeatureKeypoints* keypoints1;
    const FeatureKeypoints* keypoints2;
    const FeatureMatches* matches;
    TwoViewGeometry::Options* options;
  };

  // To be implemented by the matching class.
  virtual void DoMatching() = 0;

  void SetupWorkers();
  void SetupData();
  bool IsStopped();
  void PrintElapsedTime(const Timer& timer);

  const FeatureKeypoints& CacheKeypoints(const image_t image_id);
  const FeatureDescriptors& CacheDescriptors(const image_t image_id);
  void CleanCache(const std::unordered_set<image_t>& keep_image_ids);

  void MatchImagePairs(
      const std::vector<std::pair<image_t, image_t>>& image_pairs);
  static TwoViewGeometry VerifyImagePair(const GeometricVerificationData data,
                                         const bool multiple_models);

  Timer total_timer_;

  bool stop_;
  QMutex stop_mutex_;

  Options options_;
  Database database_;
  std::string database_path_;

  QThread* parent_thread_;
  QOpenGLContext* context_;
  QOffscreenSurface* surface_;

  SiftGPU* sift_gpu_;
  SiftMatchGPU* sift_match_gpu_;
  ThreadPool* verifier_thread_pool_;

  std::unordered_map<camera_t, Camera> cameras_;
  std::unordered_map<image_t, Image> images_;
  std::unordered_map<image_t, FeatureKeypoints> keypoints_cache_;
  std::unordered_map<image_t, FeatureDescriptors> descriptors_cache_;
};

// Exhaustively match images by processing each block in the exhaustive match
// matrix as one batch:
//
// +----+----+-----------------> images[i]
// |#000|0000|
// |1#00|1000| <- Above the main diagonal, the block diagonal is not matched
// |11#0|1100|                                                             ^
// |111#|1110|                                                             |
// +----+----+                                                             |
// |1000|#000|\                                                            |
// |1100|1#00| \ One block                                                 |
// |1110|11#0| / of image pairs                                            |
// |1111|111#|/                                                            |
// +----+----+                                                             |
// |  ^                                                                    |
// |  |                                                                    |
// | Below the main diagonal, the block diagonal is matched <--------------+
// |
// v
// images[i]
//
// Pairs will only be matched if 1, to avoid duplicate pairs. Pairs with #
// are on the main diagonal and denote pairs of the same image.
class ExhaustiveFeatureMatcher : public FeatureMatcher {
 public:
  struct ExhaustiveOptions {
    // Block size, i.e. number of images to simultaneously load into memory.
    int block_size = 100;

    // Whether to enable preemptive matching as described in
    // "Towards Linear-time Incremental Structure from Motion",
    // Chanchang Wu, 3DV 2013.
    bool preemptive = false;

    // Number of features to use for preemptive filtering.
    int preemptive_num_features = 100;

    // Minimum number of successful matches for image pair to pass filtering.
    int preemptive_min_num_matches = 4;

    void Check() const;
  };

  ExhaustiveFeatureMatcher(const Options& options,
                           const ExhaustiveOptions& exhaustive_options,
                           const std::string& database_path);

 private:
  virtual void DoMatching();

  std::vector<std::pair<image_t, image_t>> PreemptivelyFilterImagePairs(
      const std::vector<std::pair<image_t, image_t>>& image_pairs);

  ExhaustiveOptions exhaustive_options_;
};

// Sequentially match images within neighborhood:
//
// +-------------------------------+-----------------------> images[i]
//                      ^          |           ^
//                      |   Current image[i]   |
//                      |          |           |
//                      +----------+-----------+
//                                 |
//                            Match against
//                    images[i-overlap, i+overlap],
//
// Sequential order is determined based on the image names in ascending order.
//
// Invoke loop detection if `(i mod loop_detection_period) == 0`, retrieve
// most similar `loop_detection_num_images` images from vocabulary tree,
// and perform matching and verification.
class SequentialFeatureMatcher : public FeatureMatcher {
 public:
  struct SequentialOptions {
    // Number of overlapping image pairs.
    int overlap = 5;

    // Whether to enable vocabulary tree based loop detection.
    bool loop_detection = false;

    // Loop detection is invoked every `loop_detection_period` images.
    int loop_detection_period = 10;

    // The number of images to retrieve in loop detection. This number should
    // be significantly bigger than the sequential matching overlap.
    int loop_detection_num_images = 30;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

    void Check() const;
  };

  SequentialFeatureMatcher(const Options& options,
                           const SequentialOptions& sequential_options,
                           const std::string& database_path);

 private:
  virtual void DoMatching();

  SequentialOptions sequential_options_;
};

// Match each image against nearest neighbor in vocabulary tree.
class VocabTreeFeatureMatcher : public FeatureMatcher {
 public:
  struct VocabTreeOptions {
    // Number of images to retrieve for each query image.
    int num_images = 100;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

    void Check() const;
  };

  VocabTreeFeatureMatcher(const Options& options,
                          const VocabTreeOptions& vocab_tree_options,
                          const std::string& database_path);

 private:
  virtual void DoMatching();

  VocabTreeOptions vocab_tree_options_;
};

// Match images against spatial nearest neighbors using prior location
// information, e.g. provided manually or extracted from EXIF.
class SpatialFeatureMatcher : public FeatureMatcher {
 public:
  struct SpatialOptions {
    // Whether the location priors in the database are GPS coordinates in
    // the form of longitude and latitude coordinates in degrees.
    bool is_gps = true;

    // Whether to ignore the Z-component of the location prior.
    bool ignore_z = true;

    // The maximum number of nearest neighbors to match.
    int max_num_neighbors = 50;

    // The maximum distance between the query and nearest neighbor. For GPS
    // coordinates the unit is Euclidean distance in meters.
    double max_distance = 100;

    void Check() const;
  };

  SpatialFeatureMatcher(const Options& options,
                        const SpatialOptions& spatial_options,
                        const std::string& database_path);

 private:
  virtual void DoMatching();

  SpatialOptions spatial_options_;
};

// Match images manually specified in a list of image pairs.
//
// Read matches file with the following format:
//
//    image_name1 image_name2
//    image_name1 image_name3
//    image_name2 image_name3
//    ...
//
class ImagePairsFeatureMatcher : public FeatureMatcher {
 public:
  ImagePairsFeatureMatcher(const Options& options,
                           const std::string& database_path,
                           const std::string& match_list_path);

 private:
  virtual void DoMatching();

  std::vector<std::pair<image_t, image_t>> ReadImagePairsList();

  std::string match_list_path_;
};

// Import feature matches from a text file.
//
// Read matches file with the following format:
//
//      image_name1 image_name2
//      0 1
//      1 2
//      2 3
//      <empty line>
//      image_name1 image_name3
//      0 1
//      1 2
//      2 3
//      ...
//
// Note that this class does not inherit from the FeatureMatcher class as it
// does not use any SiftGPU functionality.
class FeaturePairsFeatureMatcher : public QThread {
 public:
  FeaturePairsFeatureMatcher(const FeatureMatcher::Options& options,
                             const bool compute_inliers,
                             const std::string& database_path,
                             const std::string& match_list_path);

  void run();
  void Stop();

 private:
  bool stop_;

  QMutex mutex_;

  std::string database_path_;
  std::string match_list_path_;
  FeatureMatcher::Options options_;
  bool compute_inliers_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_FEATURE_MATCHING_H_
