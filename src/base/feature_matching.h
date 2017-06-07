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

#ifndef COLMAP_SRC_BASE_FEATURE_MATCHING_H_
#define COLMAP_SRC_BASE_FEATURE_MATCHING_H_

#include <array>
#include <string>
#include <unordered_set>
#include <vector>

#include "base/database.h"
#include "util/alignment.h"
#include "util/cache.h"
#include "util/opengl_utils.h"
#include "util/threading.h"
#include "util/timer.h"

class SiftMatchGPU;

namespace colmap {

struct SiftMatchingOptions {
  // Number of threads for feature matching and geometric verification.
  int num_threads = ThreadPool::kMaxNumThreads;

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
  int min_num_trials = 30;
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

  bool Check() const;
};

namespace internal {

struct ImagePairData {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
};

struct MatchData : public ImagePairData {
  FeatureMatches matches;
};

struct InlierMatchData : public MatchData {
  TwoViewGeometry two_view_geometry;
};

}  // namespace internal

// Cache for feature matching to minimize database access during matching.
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(const size_t cache_size, const Database* database);

  void Setup();

  const Camera& GetCamera(const camera_t camera_id) const;
  const Image& GetImage(const image_t image_id) const;
  const FeatureKeypoints& GetKeypoints(const image_t image_id);
  const FeatureDescriptors& GetDescriptors(const image_t image_id);
  FeatureMatches GetMatches(const image_t image_id1, const image_t image_id2);
  std::vector<image_t> GetImageIds() const;

  bool ExistsMatches(const image_t image_id1, const image_t image_id2);
  bool ExistsInlierMatches(const image_t image_id1, const image_t image_id2);

  void WriteMatches(const image_t image_id1, const image_t image_id2,
                    const FeatureMatches& matches);
  void WriteInlierMatches(const image_t image_id1, const image_t image_id2,
                          const TwoViewGeometry& two_view_geometry);

  void DeleteMatches(const image_t image_id1, const image_t image_id2);
  void DeleteInlierMatches(const image_t image_id1, const image_t image_id2);

 private:
  const size_t cache_size_;
  const Database* database_;
  std::mutex database_mutex_;
  EIGEN_STL_UMAP(camera_t, Camera) cameras_cache_;
  EIGEN_STL_UMAP(image_t, Image) images_cache_;
  std::unique_ptr<LRUCache<image_t, FeatureKeypoints>> keypoints_cache_;
  std::unique_ptr<LRUCache<image_t, FeatureDescriptors>> descriptors_cache_;
};

class FeatureMatcherThread : public Thread {
 public:
  FeatureMatcherThread(const SiftMatchingOptions& options,
                       FeatureMatcherCache* cache);

  void SetMaxNumMatches(const int max_num_matches);

  virtual bool IsValid();

 protected:
  virtual void SetValid();
  virtual void SetInvalid();

  SiftMatchingOptions options_;
  FeatureMatcherCache* cache_;

 private:
  std::mutex mutex_;
  std::condition_variable is_setup_;
  std::atomic<bool> is_valid_;
};

class SiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::ImagePairData Input;
  typedef internal::MatchData Output;

  SiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                        FeatureMatcherCache* cache,
                        JobQueue<Input>* input_queue,
                        JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class SiftGPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::ImagePairData Input;
  typedef internal::MatchData Output;

  SiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                        FeatureMatcherCache* cache,
                        JobQueue<Input>* input_queue,
                        JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  void GetDescriptorData(const int index, const image_t image_id,
                         const FeatureDescriptors** descriptors_ptr);

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  // The previously uploaded images to the GPU.
  std::array<image_t, 2> prev_uploaded_image_ids_;
  std::array<FeatureDescriptors, 2> prev_uploaded_descriptors_;
};

class GuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::InlierMatchData Input;
  typedef internal::InlierMatchData Output;

  GuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class GuidedSiftGPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::InlierMatchData Input;
  typedef internal::InlierMatchData Output;

  GuidedSiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  void GetFeatureData(const int index, const image_t image_id,
                      const FeatureKeypoints** keypoints_ptr,
                      const FeatureDescriptors** descriptors_ptr);

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  // The previously uploaded images to the GPU.
  std::array<image_t, 2> prev_uploaded_image_ids_;
  std::array<FeatureKeypoints, 2> prev_uploaded_keypoints_;
  std::array<FeatureDescriptors, 2> prev_uploaded_descriptors_;
};

class TwoViewGeometryVerifier : public Thread {
 public:
  typedef internal::MatchData Input;
  typedef internal::InlierMatchData Output;

  TwoViewGeometryVerifier(const SiftMatchingOptions& options,
                          FeatureMatcherCache* cache,
                          JobQueue<Input>* input_queue,
                          JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  const SiftMatchingOptions options_;
  TwoViewGeometry::Options two_view_geometry_options_;
  FeatureMatcherCache* cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

// Multi-threaded and multi-GPU SIFT feature matcher, which writes the computed
// results to the database and skips already matched image pairs. To improve
// performance of the matching by taking advantage of caching and database
// transactions, pass multiple images to the `Match` function. Note that the
// database should be in an active transaction while calling `Match`.
class SiftFeatureMatcher {
 public:
  SiftFeatureMatcher(const SiftMatchingOptions& options, Database* database,
                     FeatureMatcherCache* cache);

  ~SiftFeatureMatcher();

  // Setup the matchers and return if successful.
  bool Setup();

  // Match one batch of multiple image pairs.
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  SiftMatchingOptions options_;
  Database* database_;
  FeatureMatcherCache* cache_;

  bool is_setup_;

  std::vector<std::unique_ptr<FeatureMatcherThread>> matchers_;
  std::vector<std::unique_ptr<FeatureMatcherThread>> guided_matchers_;
  std::vector<std::unique_ptr<Thread>> verifiers_;
  std::unique_ptr<ThreadPool> thread_pool_;

  JobQueue<internal::ImagePairData> matcher_queue_;
  JobQueue<internal::MatchData> verifier_queue_;
  JobQueue<internal::InlierMatchData> guided_matcher_queue_;
  JobQueue<internal::InlierMatchData> output_queue_;
};

// Exhaustively match images by processing each block in the exhaustive match
// matrix in one batch:
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
class ExhaustiveFeatureMatcher : public Thread {
 public:
  struct Options {
    // Block size, i.e. number of images to simultaneously load into memory.
    int block_size = 50;

    bool Check() const;
  };

  ExhaustiveFeatureMatcher(const Options& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Sequentially match images within neighborhood:
//
// +-------------------------------+-----------------------> images[i]
//                      ^          |           ^
//                      |   Current image[i]   |
//                      |          |           |
//                      +----------+-----------+
//                                 |
//                        Match image_i against
//                    image_[i - 2^o, i + 2^o] with o = 1 .. overlap
//
// Sequential order is determined based on the image names in ascending order.
//
// Invoke loop detection if `(i mod loop_detection_period) == 0`, retrieve
// most similar `loop_detection_num_images` images from vocabulary tree,
// and perform matching and verification.
class SequentialFeatureMatcher : public Thread {
 public:
  struct Options {
    // Number of overlapping image pairs.
    int overlap = 5;

    // Whether to enable vocabulary tree based loop detection.
    bool loop_detection = false;

    // Loop detection is invoked every `loop_detection_period` images.
    int loop_detection_period = 10;

    // The number of images to retrieve in loop detection. This number should
    // be significantly bigger than the sequential matching overlap.
    int loop_detection_num_images = 30;

    // Number of images to spatially verify for re-ranking.
    int loop_detection_num_verifications = 0;

    // The maximum number of features to use for indexing an image. If an
    // image has more features, only the largest-scale features will be indexed.
    int loop_detection_max_num_features = -1;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

    bool Check() const;
  };

  SequentialFeatureMatcher(const Options& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  std::vector<image_t> GetOrderedImageIds() const;
  void RunSequentialMatching(const std::vector<image_t>& image_ids);
  void RunLoopDetection(const std::vector<image_t>& image_ids);

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match each image against its nearest neighbors using a vocabulary tree.
class VocabTreeFeatureMatcher : public Thread {
 public:
  struct Options {
    // Number of images to retrieve for each query image.
    int num_images = 100;

    // Number of images to spatially verify for re-ranking.
    int num_verifications = 0;

    // The maximum number of features to use for indexing an image. If an
    // image has more features, only the largest-scale features will be indexed.
    int max_num_features = -1;

    // Path to the vocabulary tree.
    std::string vocab_tree_path = "";

    // Optional path to file with specific image names to match.
    std::string match_list_path = "";

    bool Check() const;
  };

  VocabTreeFeatureMatcher(const Options& options,
                          const SiftMatchingOptions& match_options,
                          const std::string& database_path);

 private:
  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match images against spatial nearest neighbors using prior location
// information, e.g. provided manually or extracted from EXIF.
class SpatialFeatureMatcher : public Thread {
 public:
  struct Options {
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

    bool Check() const;
  };

  SpatialFeatureMatcher(const Options& options,
                        const SiftMatchingOptions& match_options,
                        const std::string& database_path);

 private:
  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match transitive image pairs in a database with existing feature matches.
// This matcher transitively closes loops. For example, if image pairs A-B and
// B-C match but A-C has not been matched, then this matcher attempts to match
// A-C. This procedure is performed for multiple iterations.
class TransitiveFeatureMatcher : public Thread {
 public:
  struct Options {
    // The maximum number of image pairs to process in one batch.
    int batch_size = 1000;

    // The number of transitive closure iterations.
    int num_iterations = 3;

    bool Check() const;
  };

  TransitiveFeatureMatcher(const Options& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
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
class ImagePairsFeatureMatcher : public Thread {
 public:
  struct Options {
    // Number of image pairs to match in one batch.
    int block_size = 100;

    // Path to the file with the matches.
    std::string match_list_path = "";

    bool Check() const;
  };

  ImagePairsFeatureMatcher(const Options& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
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
class FeaturePairsFeatureMatcher : public Thread {
 public:
  struct Options {
    // Whether to geometrically verify the given matches.
    bool verify_matches = true;

    // Path to the file with the matches.
    std::string match_list_path = "";

    bool Check() const;
  };

  FeaturePairsFeatureMatcher(const Options& options,
                             const SiftMatchingOptions& match_options,
                             const std::string& database_path);

 private:
  const static size_t kCacheSize = 100;

  void Run() override;

  const Options options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
};

// Match the given SIFT features on the CPU.
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

#endif  // COLMAP_SRC_BASE_FEATURE_MATCHING_H_
