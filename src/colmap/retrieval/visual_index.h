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

#pragma once

#include "colmap/feature/types.h"
#include "colmap/math/math.h"
#include "colmap/retrieval/inverted_file.h"
#include "colmap/retrieval/inverted_index.h"
#include "colmap/retrieval/vote_and_verify.h"
#include "colmap/util/endian.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <boost/heap/fibonacci_heap.hpp>
#include <flann/flann.hpp>

namespace colmap {
namespace retrieval {

// Visual index for image retrieval using a vocabulary tree with Hamming
// embedding, based on the papers:
//
//    Schönberger, Price, Sattler, Pollefeys, Frahm. "A Vote-and-Verify Strategy
//    for Fast Spatial Verification in Image Retrieval". ACCV 2016.
//
//    Arandjelovic, Zisserman: Scalable descriptor
//    distinctiveness for location recognition. ACCV 2014.
template <typename kDescType = uint8_t,
          int kDescDim = 128,
          int kEmbeddingDim = 64>
class VisualIndex {
 public:
  static const int kMaxNumThreads = -1;
  typedef InvertedIndex<kDescType, kDescDim, kEmbeddingDim> InvertedIndexType;
  typedef FeatureKeypoints GeomType;
  typedef typename InvertedIndexType::DescType DescType;
  typedef typename InvertedIndexType::EntryType EntryType;

  struct IndexOptions {
    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 1;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  struct QueryOptions {
    // The maximum number of most similar images to retrieve.
    int max_num_images = -1;

    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 5;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // Whether to perform spatial verification after image retrieval.
    int num_images_after_verification = 0;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  struct BuildOptions {
    // The desired number of visual words, i.e. the number of leaf node
    // clusters. Note that the actual number of visual words might be less.
    int num_visual_words = 256 * 256;

    // The branching factor of the hierarchical k-means tree.
    int branching = 256;

    // The number of iterations for the clustering.
    int num_iterations = 11;

    // The target precision of the visual word search index.
    double target_precision = 0.95;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  VisualIndex();
  ~VisualIndex();

  size_t NumVisualWords() const;

  // Add image to the visual index.
  void Add(const IndexOptions& options,
           int image_id,
           const GeomType& geometries,
           const DescType& descriptors);

  // Check if an image has been indexed.
  bool ImageIndexed(int image_id) const;

  // Query for most similar images in the visual index.
  void Query(const QueryOptions& options,
             const DescType& descriptors,
             std::vector<ImageScore>* image_scores) const;

  // Query for most similar images in the visual index.
  void Query(const QueryOptions& options,
             const GeomType& geometries,
             const DescType& descriptors,
             std::vector<ImageScore>* image_scores) const;

  // Prepare the index after adding images and before querying.
  void Prepare();

  // Build a visual index from a set of training descriptors by quantizing the
  // descriptor space into visual words and compute their Hamming embedding.
  void Build(const BuildOptions& options, const DescType& descriptors);

  // Read and write the visual index. This can be done for an index with and
  // without indexed images.
  void Read(const std::string& path);
  void Write(const std::string& path);

 private:
  // Quantize the descriptor space into visual words.
  void Quantize(const BuildOptions& options, const DescType& descriptors);

  // Query for nearest neighbor images and return nearest neighbor visual word
  // identifiers for each descriptor.
  void QueryAndFindWordIds(const QueryOptions& options,
                           const DescType& descriptors,
                           std::vector<ImageScore>* image_scores,
                           Eigen::MatrixXi* word_ids) const;

  // Find the nearest neighbor visual words for the given descriptors.
  Eigen::MatrixXi FindWordIds(const DescType& descriptors,
                              int num_neighbors,
                              int num_checks,
                              int num_threads) const;

  // The search structure on the quantized descriptor space.
  flann::AutotunedIndex<flann::L2<kDescType>> visual_word_index_;

  // The centroids of the visual words.
  flann::Matrix<kDescType> visual_words_;

  // The inverted index of the database.
  InvertedIndexType inverted_index_;

  // Identifiers of all indexed images.
  std::unordered_set<int> image_ids_;

  // Whether the index is prepared.
  bool prepared_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename kDescType, int kDescDim, int kEmbeddingDim>
VisualIndex<kDescType, kDescDim, kEmbeddingDim>::VisualIndex()
    : prepared_(false) {}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
VisualIndex<kDescType, kDescDim, kEmbeddingDim>::~VisualIndex() {
  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
size_t VisualIndex<kDescType, kDescDim, kEmbeddingDim>::NumVisualWords() const {
  return visual_words_.rows;
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Add(
    const IndexOptions& options,
    const int image_id,
    const GeomType& geometries,
    const DescType& descriptors) {
  CHECK_EQ(geometries.size(), descriptors.rows());

  // If the image is already indexed, do nothing.
  if (ImageIndexed(image_id)) {
    return;
  }

  image_ids_.insert(image_id);

  prepared_ = false;

  if (descriptors.rows() == 0) {
    return;
  }

  const Eigen::MatrixXi word_ids = FindWordIds(descriptors,
                                               options.num_neighbors,
                                               options.num_checks,
                                               options.num_threads);

  for (typename DescType::Index i = 0; i < descriptors.rows(); ++i) {
    const auto& descriptor = descriptors.row(i);

    typename InvertedIndexType::GeomType geometry;
    geometry.x = geometries[i].x;
    geometry.y = geometries[i].y;
    geometry.scale = geometries[i].ComputeScale();
    geometry.orientation = geometries[i].ComputeOrientation();

    for (int n = 0; n < options.num_neighbors; ++n) {
      const int word_id = word_ids(i, n);
      if (word_id != InvertedIndexType::kInvalidWordId) {
        inverted_index_.AddEntry(image_id, word_id, i, descriptor, geometry);
      }
    }
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
bool VisualIndex<kDescType, kDescDim, kEmbeddingDim>::ImageIndexed(
    const int image_id) const {
  return image_ids_.count(image_id) != 0;
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Query(
    const QueryOptions& options,
    const DescType& descriptors,
    std::vector<ImageScore>* image_scores) const {
  const GeomType geometries;
  Query(options, geometries, descriptors, image_scores);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Query(
    const QueryOptions& options,
    const GeomType& geometries,
    const DescType& descriptors,
    std::vector<ImageScore>* image_scores) const {
  Eigen::MatrixXi word_ids;
  QueryAndFindWordIds(options, descriptors, image_scores, &word_ids);

  if (options.num_images_after_verification <= 0) {
    return;
  }

  CHECK_EQ(descriptors.rows(), geometries.size());

  // Extract top-ranked images to verify.
  std::unordered_set<int> image_ids;
  for (const auto& image_score : *image_scores) {
    image_ids.insert(image_score.image_id);
  }

  // Find matches for top-ranked images
  typedef std::vector<
      std::pair<float, std::pair<const EntryType*, const EntryType*>>>
      OrderedMatchListType;

  // Reference our matches (with their lowest distance) for both
  // {query feature => db feature} and vice versa.
  std::unordered_map<int, std::unordered_map<int, OrderedMatchListType>>
      query_to_db_matches;
  std::unordered_map<int, std::unordered_map<int, OrderedMatchListType>>
      db_to_query_matches;

  std::vector<const EntryType*> word_matches;

  std::vector<EntryType> query_entries;  // Convert query features, too.
  query_entries.reserve(descriptors.rows());

  // NOTE: Currently, we are redundantly computing the feature weighting.
  const HammingDistWeightFunctor<kEmbeddingDim> hamming_dist_weight_functor;

  for (typename DescType::Index i = 0; i < descriptors.rows(); ++i) {
    const auto& descriptor = descriptors.row(i);

    EntryType query_entry;
    query_entry.feature_idx = i;
    query_entry.geometry.x = geometries[i].x;
    query_entry.geometry.y = geometries[i].y;
    query_entry.geometry.scale = geometries[i].ComputeScale();
    query_entry.geometry.orientation = geometries[i].ComputeOrientation();
    query_entries.push_back(query_entry);

    // For each db feature, keep track of the lowest distance (if db features
    // are mapped to more than one visual word).
    std::unordered_map<
        int,
        std::unordered_map<int, std::pair<float, const EntryType*>>>
        image_matches;

    for (int j = 0; j < word_ids.cols(); ++j) {
      const int word_id = word_ids(i, j);

      if (word_id != InvertedIndexType::kInvalidWordId) {
        inverted_index_.ConvertToBinaryDescriptor(
            word_id, descriptor, &query_entries[i].descriptor);

        const auto idf_weight = inverted_index_.GetIDFWeight(word_id);
        const auto squared_idf_weight = idf_weight * idf_weight;

        inverted_index_.FindMatches(word_id, image_ids, &word_matches);

        for (const auto& match : word_matches) {
          const size_t hamming_dist =
              (query_entries[i].descriptor ^ match->descriptor).count();

          if (hamming_dist <= hamming_dist_weight_functor.kMaxHammingDistance) {
            const float dist =
                hamming_dist_weight_functor(hamming_dist) * squared_idf_weight;

            auto& feature_matches = image_matches[match->image_id];
            const auto feature_match = feature_matches.find(match->feature_idx);

            if (feature_match == feature_matches.end() ||
                feature_match->first < dist) {
              feature_matches[match->feature_idx] = std::make_pair(dist, match);
            }
          }
        }
      }
    }

    // Finally, cross-reference the query and db feature matches.
    for (const auto& feature_matches : image_matches) {
      const auto image_id = feature_matches.first;

      for (const auto& feature_match : feature_matches.second) {
        const auto feature_idx = feature_match.first;
        const auto dist = feature_match.second.first;
        const auto db_match = feature_match.second.second;

        const auto entry_pair = std::make_pair(&query_entries[i], db_match);

        query_to_db_matches[image_id][i].emplace_back(dist, entry_pair);
        db_to_query_matches[image_id][feature_idx].emplace_back(dist,
                                                                entry_pair);
      }
    }
  }

  // Verify top-ranked images using the found matches.
  for (auto& image_score : *image_scores) {
    auto& query_matches = query_to_db_matches[image_score.image_id];
    auto& db_matches = db_to_query_matches[image_score.image_id];

    // No matches found.
    if (query_matches.empty()) {
      continue;
    }

    // Enforce 1-to-1 matching: Build Fibonacci heaps for the query and database
    // features, ordered by the minimum number of matches per feature. We'll
    // select these matches one at a time. For convenience, we'll also pre-sort
    // the matched feature lists by matching score.

    typedef boost::heap::fibonacci_heap<std::pair<int, int>> FibonacciHeapType;
    FibonacciHeapType query_heap;
    FibonacciHeapType db_heap;
    std::unordered_map<int, typename FibonacciHeapType::handle_type>
        query_heap_handles;
    std::unordered_map<int, typename FibonacciHeapType::handle_type>
        db_heap_handles;

    for (auto& match_data : query_matches) {
      std::sort(
          match_data.second.begin(),
          match_data.second.end(),
          std::greater<
              std::pair<float,
                        std::pair<const EntryType*, const EntryType*>>>());

      query_heap_handles[match_data.first] = query_heap.push(std::make_pair(
          -static_cast<int>(match_data.second.size()), match_data.first));
    }

    for (auto& match_data : db_matches) {
      std::sort(
          match_data.second.begin(),
          match_data.second.end(),
          std::greater<
              std::pair<float,
                        std::pair<const EntryType*, const EntryType*>>>());

      db_heap_handles[match_data.first] = db_heap.push(std::make_pair(
          -static_cast<int>(match_data.second.size()), match_data.first));
    }

    // Keep tabs on what features have been already matched.
    std::vector<FeatureGeometryMatch> matches;

    auto db_top = db_heap.top();  // (-num_available_matches, feature_idx)
    auto query_top = query_heap.top();

    while (!db_heap.empty() && !query_heap.empty()) {
      // Take the query or database feature with the smallest number of
      // available matches.
      const bool use_query =
          (query_top.first >= db_top.first) && !query_heap.empty();

      // Find the best matching feature that hasn't already been matched.
      auto& heap1 = (use_query) ? query_heap : db_heap;
      auto& heap2 = (use_query) ? db_heap : query_heap;
      auto& handles1 = (use_query) ? query_heap_handles : db_heap_handles;
      auto& handles2 = (use_query) ? db_heap_handles : query_heap_handles;
      auto& matches1 = (use_query) ? query_matches : db_matches;
      auto& matches2 = (use_query) ? db_matches : query_matches;

      const auto idx1 = heap1.top().second;
      heap1.pop();

      // Entries that have been matched (or processed and subsequently ignored)
      // get their handles removed.
      if (handles1.count(idx1) > 0) {
        handles1.erase(idx1);

        bool match_found = false;

        // The matches have been ordered by Hamming distance, already --
        // select the lowest available match.
        for (auto& entry2 : matches1[idx1]) {
          const auto idx2 = (use_query) ? entry2.second.second->feature_idx
                                        : entry2.second.first->feature_idx;

          if (handles2.count(idx2) > 0) {
            if (!match_found) {
              match_found = true;
              FeatureGeometryMatch match;
              match.geometry1 = entry2.second.first->geometry;
              match.geometry2 = entry2.second.second->geometry;
              matches.push_back(match);

              handles2.erase(idx2);

              // Remove this feature from consideration for all other features
              // that matched to it.
              for (auto& entry1 : matches2[idx2]) {
                const auto other_idx1 = (use_query)
                                            ? entry1.second.first->feature_idx
                                            : entry1.second.second->feature_idx;
                if (handles1.count(other_idx1) > 0) {
                  (*handles1[other_idx1]).first += 1;
                  heap1.increase(handles1[other_idx1]);
                }
              }
            } else {
              (*handles2[idx2]).first += 1;
              heap2.increase(handles2[idx2]);
            }
          }
        }
      }

      if (!query_heap.empty()) {
        query_top = query_heap.top();
      }

      if (!db_heap.empty()) {
        db_top = db_heap.top();
      }
    }

    // Finally, run verification for the current image.
    VoteAndVerifyOptions vote_and_verify_options;
    image_score.score += VoteAndVerify(vote_and_verify_options, matches);
  }

  // Re-rank the images using the spatial verification scores.

  const size_t num_images = std::min<size_t>(
      image_scores->size(), options.num_images_after_verification);

  auto SortFunc = [](const ImageScore& score1, const ImageScore& score2) {
    return score1.score > score2.score;
  };

  if (num_images == image_scores->size()) {
    std::sort(image_scores->begin(), image_scores->end(), SortFunc);
  } else {
    std::partial_sort(image_scores->begin(),
                      image_scores->begin() + num_images,
                      image_scores->end(),
                      SortFunc);
    image_scores->resize(num_images);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Prepare() {
  inverted_index_.Finalize();
  prepared_ = true;
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Build(
    const BuildOptions& options, const DescType& descriptors) {
  // Quantize the descriptor space into visual words.
  Quantize(options, descriptors);

  // Build the search index on the visual words.
  flann::AutotunedIndexParams index_params;
  index_params["target_precision"] =
      static_cast<float>(options.target_precision);
  visual_word_index_ =
      flann::AutotunedIndex<flann::L2<kDescType>>(index_params);
  visual_word_index_.buildIndex(visual_words_);

  // Initialize a new inverted index.
  inverted_index_ = InvertedIndexType();
  inverted_index_.Initialize(NumVisualWords());

  // Generate descriptor projection matrix.
  inverted_index_.GenerateHammingEmbeddingProjection();

  // Learn the Hamming embedding.
  const int kNumNeighbors = 1;
  const Eigen::MatrixXi word_ids = FindWordIds(
      descriptors, kNumNeighbors, options.num_checks, options.num_threads);
  inverted_index_.ComputeHammingEmbedding(descriptors, word_ids);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Read(
    const std::string& path) {
  long int file_offset = 0;

  // Read the visual words.

  {
    if (visual_words_.ptr() != nullptr) {
      delete[] visual_words_.ptr();
    }

    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    const uint64_t rows = ReadBinaryLittleEndian<uint64_t>(&file);
    const uint64_t cols = ReadBinaryLittleEndian<uint64_t>(&file);
    kDescType* visual_words_data = new kDescType[rows * cols];
    for (size_t i = 0; i < rows * cols; ++i) {
      visual_words_data[i] = ReadBinaryLittleEndian<kDescType>(&file);
    }
    visual_words_ = flann::Matrix<kDescType>(visual_words_data, rows, cols);
    file_offset = file.tellg();
  }

  // Read the visual words search index.

  visual_word_index_ =
      flann::AutotunedIndex<flann::L2<kDescType>>(visual_words_);

  {
    FILE* fin = fopen(path.c_str(), "rb");
    CHECK_NOTNULL(fin);
    fseek(fin, file_offset, SEEK_SET);
    visual_word_index_.loadIndex(fin);
    file_offset = ftell(fin);
    fclose(fin);
  }

  // Read the inverted index.

  {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    file.seekg(file_offset, std::ios::beg);
    inverted_index_.Read(&file);
  }

  image_ids_.clear();
  inverted_index_.GetImageIds(&image_ids_);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Write(
    const std::string& path) {
  // Write the visual words.

  {
    CHECK_NOTNULL(visual_words_.ptr());
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    WriteBinaryLittleEndian<uint64_t>(&file, visual_words_.rows);
    WriteBinaryLittleEndian<uint64_t>(&file, visual_words_.cols);
    for (size_t i = 0; i < visual_words_.rows * visual_words_.cols; ++i) {
      WriteBinaryLittleEndian<kDescType>(&file, visual_words_.ptr()[i]);
    }
  }

  // Write the visual words search index.

  {
    FILE* fout = fopen(path.c_str(), "ab");
    CHECK_NOTNULL(fout);
    visual_word_index_.saveIndex(fout);
    fclose(fout);
  }

  // Write the inverted index.

  {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    CHECK(file.is_open()) << path;
    inverted_index_.Write(&file);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::Quantize(
    const BuildOptions& options, const DescType& descriptors) {
  static_assert(DescType::IsRowMajor, "Descriptors must be row-major.");

  CHECK_GE(options.num_visual_words, options.branching);
  CHECK_GE(descriptors.rows(), options.num_visual_words);

  const flann::Matrix<kDescType> descriptor_matrix(
      const_cast<kDescType*>(descriptors.data()),
      descriptors.rows(),
      descriptors.cols());

  std::vector<typename flann::L2<kDescType>::ResultType> centers_data(
      options.num_visual_words * descriptors.cols());
  flann::Matrix<typename flann::L2<kDescType>::ResultType> centers(
      centers_data.data(), options.num_visual_words, descriptors.cols());

  flann::KMeansIndexParams index_params;
  index_params["branching"] = options.branching;
  index_params["iterations"] = options.num_iterations;
  index_params["centers_init"] = flann::FLANN_CENTERS_KMEANSPP;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  const int num_centers = flann::hierarchicalClustering<flann::L2<kDescType>>(
      descriptor_matrix, centers, index_params);

  CHECK_LE(num_centers, options.num_visual_words);

  const size_t visual_word_data_size = num_centers * descriptors.cols();
  kDescType* visual_words_data = new kDescType[visual_word_data_size];
  for (size_t i = 0; i < visual_word_data_size; ++i) {
    if (std::is_integral<kDescType>::value) {
      visual_words_data[i] = std::round(centers_data[i]);
    } else {
      visual_words_data[i] = centers_data[i];
    }
  }

  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }

  visual_words_ = flann::Matrix<kDescType>(
      visual_words_data, num_centers, descriptors.cols());
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void VisualIndex<kDescType, kDescDim, kEmbeddingDim>::QueryAndFindWordIds(
    const QueryOptions& options,
    const DescType& descriptors,
    std::vector<ImageScore>* image_scores,
    Eigen::MatrixXi* word_ids) const {
  CHECK(prepared_);

  if (descriptors.rows() == 0) {
    image_scores->clear();
    return;
  }

  *word_ids = FindWordIds(descriptors,
                          options.num_neighbors,
                          options.num_checks,
                          options.num_threads);
  inverted_index_.Query(descriptors, *word_ids, image_scores);

  auto SortFunc = [](const ImageScore& score1, const ImageScore& score2) {
    return score1.score > score2.score;
  };

  size_t num_images = image_scores->size();
  if (options.max_num_images >= 0) {
    num_images = std::min<size_t>(image_scores->size(), options.max_num_images);
  }

  if (num_images == image_scores->size()) {
    std::sort(image_scores->begin(), image_scores->end(), SortFunc);
  } else {
    std::partial_sort(image_scores->begin(),
                      image_scores->begin() + num_images,
                      image_scores->end(),
                      SortFunc);
    image_scores->resize(num_images);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
Eigen::MatrixXi VisualIndex<kDescType, kDescDim, kEmbeddingDim>::FindWordIds(
    const DescType& descriptors,
    const int num_neighbors,
    const int num_checks,
    const int num_threads) const {
  static_assert(DescType::IsRowMajor, "Descriptors must be row-major");

  CHECK_GT(descriptors.rows(), 0);
  CHECK_GT(num_neighbors, 0);

  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      word_ids(descriptors.rows(), num_neighbors);
  word_ids.setConstant(InvertedIndexType::kInvalidWordId);
  flann::Matrix<size_t> indices(
      word_ids.data(), descriptors.rows(), num_neighbors);

  Eigen::Matrix<typename flann::L2<kDescType>::ResultType,
                Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::RowMajor>
      distance_matrix(descriptors.rows(), num_neighbors);
  flann::Matrix<typename flann::L2<kDescType>::ResultType> distances(
      distance_matrix.data(), descriptors.rows(), num_neighbors);

  const flann::Matrix<kDescType> query(
      const_cast<kDescType*>(descriptors.data()),
      descriptors.rows(),
      descriptors.cols());

  flann::SearchParams search_params(num_checks);
  if (num_threads < 0) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  visual_word_index_.knnSearch(
      query, indices, distances, num_neighbors, search_params);

  return word_ids.cast<int>();
}

}  // namespace retrieval
}  // namespace colmap
