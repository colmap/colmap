// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/retrieval/utils.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace colmap {
namespace retrieval {

struct GlobalDescriptorQueryOptions {
  // Maximum number of most similar images to retrieve.
  int max_num_images = 100;

  // Optional filter for restricting the retrieved image identifiers. The
  // filter is applied before selecting the most similar `max_num_images`.
  std::function<bool(image_t)> image_id_filter;
};

// Global descriptor index for image retrieval using pre-computed per-image
// global descriptors (e.g. from MixVPR, NetVLAD, etc.).
//
// Unlike the VisualIndex (vocabulary tree + Hamming embedding) which operates
// on thousands of local feature descriptors per image, this index stores a
// single compact descriptor per image and performs nearest-neighbor search
// using cosine similarity (dot product on L2-normalized vectors via FAISS).
//
// The pipeline is:
//   1. Add(image_id, descriptor) – store a pre-computed L2-normalized
//   descriptor
//   2. Prepare() – build a FAISS IndexFlatIP over all stored descriptors
//   3. Query()  – look up the descriptor for a given image_id and return
//                 the top-k nearest neighbors by cosine similarity
//
// Descriptors are assumed to be L2-normalized. The caller is responsible for
// computing descriptors (e.g. via an ONNX model) before calling Add().
class GlobalDescriptorIndex {
 public:
  using QueryOptions = GlobalDescriptorQueryOptions;

  // Construct an index for descriptors of the given dimensionality.
  explicit GlobalDescriptorIndex(int descriptor_dim = 4096);

  // Number of currently indexed images.
  size_t NumImages() const;

  // Add a pre-computed L2-normalized descriptor for the given image.
  // The descriptor must have size descriptor_dim_.
  // Images can be added in any order; Prepare() finalizes the index.
  void Add(image_t image_id, const std::vector<float>& descriptor);

  // Prepare the index for querying after all images have been added.
  // Builds the FAISS index over the stored descriptors.
  void Prepare();

  // Return true if the index has been prepared and is ready for queries.
  bool IsPrepared() const;

  // Query for the most similar images to the given query image.
  // The query image must have been previously added via Add().
  // Results are returned in descending order of similarity (cosine score).
  void Query(const QueryOptions& options,
             image_t query_image_id,
             std::vector<ImageScore>* image_scores) const;

  // Serialize the index to a binary file.
  void Write(const std::filesystem::path& path) const;

  // Deserialize the index from a binary file.
  // After reading, the index is ready for Query() — no Prepare() needed.
  void Read(const std::filesystem::path& path);

 private:
  void BuildFaissIndex();

  int descriptor_dim_;
  std::vector<image_t> image_ids_;
  // Descriptors stored as row-major matrix (num_images × descriptor_dim).
  // Each row is L2-normalized.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_;
  // Mapping from image_id to row index in descriptors_.
  std::unordered_map<image_t, size_t> image_id_to_idx_;
  // FAISS index for fast nearest-neighbor search (owned via opaque pointer).
  struct FaissIndexDeleter {
    void operator()(void* ptr) const;
  };
  std::unique_ptr<void, FaissIndexDeleter> faiss_index_;
  bool prepared_ = false;
};

}  // namespace retrieval
}  // namespace colmap
