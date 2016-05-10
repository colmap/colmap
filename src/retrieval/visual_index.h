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

#ifndef COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_
#define COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_

#include <Eigen/Core>

#include "base/feature.h"
#include "ext/FLANN/flann.hpp"
#include "retrieval/inverted_file.h"
#include "retrieval/inverted_index.h"

namespace colmap {
namespace retrieval {

// Visual index for image retrieval using a vocabulary tree with Hamming
// embedding, based on the paper:
//
//    Arandjelovic, Zisserman: Scalable descriptor
//    distinctiveness for location recognition. ACCV 2014.
class VisualIndex {
 public:
  const static int kProjDescDim = 64;
  static const int kMaxNumThreads = -1;
  typedef InvertedIndex<kProjDescDim> InvertedIndexType;
  typedef InvertedIndexType::Desc Desc;

  struct IndexOptions {
    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 1;

    // The number of checks in the nearest neighbor search.
    int num_checks = flann::FLANN_CHECKS_AUTOTUNED;

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
    int num_checks = flann::FLANN_CHECKS_AUTOTUNED;

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
    int num_checks = flann::FLANN_CHECKS_AUTOTUNED;

    // The number of threads used in the index.
    int num_threads = kMaxNumThreads;
  };

  VisualIndex();
  ~VisualIndex();

  size_t NumVisualWords() const;

  // Add image to the visual index.
  void Add(const IndexOptions& options, const int image_id, Desc& descriptors);

  // Query for most similar images in the visual index.
  void Query(const QueryOptions& options, Desc& descriptors,
             std::vector<ImageScore>* image_scores) const;

  // Prepare the index after adding images and before querying.
  void Prepare();

  // Build a visual index from a set of training descriptors by quantizing the
  // descriptor space into visual words and compute their Hamming embedding.
  void Build(const BuildOptions& options, Desc& descriptors);

  // Read and write the visual index. This can be done for an index with and
  // without indexed images.
  void Read(const std::string& path);
  void Write(const std::string& path);

 private:
  // Quantize the descriptor space into visual words.
  void Quantize(const BuildOptions& options, Desc& descriptors);

  // Find the nearest neighbor visual words for the given descriptors.
  Eigen::MatrixXi FindWordIds(Desc& descriptors, const int num_neighbors,
                              const int num_checks,
                              const int num_threads) const;

  // The search structure on the quantized descriptor space.
  flann::AutotunedIndex<flann::L2<uint8_t>> visual_word_index_;

  // The centroids of the visual words.
  flann::Matrix<uint8_t> visual_words_;

  // The inverted index of the database.
  InvertedIndexType inverted_index_;

  // Identifiers of all indexed images.
  std::unordered_set<int> image_ids_;

  // Whether the index is prepared.
  bool prepared_;
};

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_VISUAL_INDEX_H_
