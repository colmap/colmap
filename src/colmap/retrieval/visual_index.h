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

#include "colmap/feature/types.h"
#include "colmap/retrieval/utils.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>

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
class VisualIndex {
 public:
  using Descriptors = Eigen::RowMajorMatrixXf;
  using Geometries = FeatureKeypoints;

  struct IndexOptions {
    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 1;

    // The number of checks in the nearest neighbor search.
    int num_checks = 64;

    // Number of threads to use.
    int num_threads = -1;
  };

  struct QueryOptions {
    // The maximum number of most similar images to retrieve.
    int max_num_images = -1;

    // The number of nearest neighbor visual words that each feature descriptor
    // is assigned to.
    int num_neighbors = 5;

    // Whether to perform spatial verification after image retrieval.
    int num_images_after_verification = 0;

    // The number of checks in the nearest neighbor search.
    int num_checks = 64;

    // Number of threads to use.
    int num_threads = -1;
  };

  struct BuildOptions {
    // The desired number of visual words, i.e. the number of leaf node
    // clusters. Note that the actual number of visual words might be less.
    int num_visual_words = 256 * 256;

    // The number of iterations for the clustering.
    int num_iterations = 100;

    // Redo clustering multiple times and keep the clusters with the best
    // training objective.
    int num_rounds = 3;

    // The number of checks in the nearest neighbor search.
    int num_checks = 256;

    // Number of threads to use.
    int num_threads = -1;
  };

  // Create visual index with specific input feature descriptor dimension and
  // Hamming embedding dimension per descriptor in the inverted index.
  static std::unique_ptr<VisualIndex> Create(int desc_dim = 128,
                                             int embedding_dim = 64);

  virtual ~VisualIndex() = default;

  // Total number of visual words.
  virtual size_t NumVisualWords() const = 0;
  virtual size_t NumImages() const = 0;
  virtual int DescDim() const = 0;
  virtual int EmbeddingDim() const = 0;

  // Add image to the visual index.
  virtual void Add(const IndexOptions& options,
                   int image_id,
                   const Geometries& geometries,
                   const Descriptors& descriptors) = 0;

  // Check if an image has been indexed.
  virtual bool IsImageIndexed(int image_id) const = 0;

  // Query for most similar images in the visual index.
  virtual void Query(const QueryOptions& options,
                     const Descriptors& descriptors,
                     std::vector<ImageScore>* image_scores) const = 0;

  // Query for most similar images in the visual index.
  virtual void Query(const QueryOptions& options,
                     const Geometries& geometries,
                     const Descriptors& descriptors,
                     std::vector<ImageScore>* image_scores) const = 0;

  // Prepare the index after adding images and before querying.
  virtual void Prepare() = 0;

  // Build a visual index from a set of training descriptors by quantizing the
  // descriptor space into visual words and compute their Hamming embedding.
  virtual void Build(const BuildOptions& options,
                     const Descriptors& descriptors) = 0;

  // Read and write the visual index. This can be done for an index with and
  // without indexed images.
  static std::unique_ptr<VisualIndex> Read(const std::string& vocab_tree_path);
  virtual void Write(const std::string& path) const = 0;

 protected:
  virtual void ReadFromFaiss(const std::string& path, long offset) = 0;
};

}  // namespace retrieval
}  // namespace colmap
