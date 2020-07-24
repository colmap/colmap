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

#ifndef COLMAP_SRC_RETRIEVAL_INVERTED_FILE_H_
#define COLMAP_SRC_RETRIEVAL_INVERTED_FILE_H_

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "retrieval/geometry.h"
#include "retrieval/inverted_file_entry.h"
#include "retrieval/utils.h"
#include "util/alignment.h"
#include "util/logging.h"
#include "util/math.h"

namespace colmap {
namespace retrieval {

// Implements an inverted file, including the ability to compute image scores
// and matches. The template parameter is the length of the binary vectors
// in the Hamming Embedding.
// This class is based on an original implementation by Torsten Sattler.
template <int kEmbeddingDim>
class InvertedFile {
 public:
  typedef Eigen::VectorXf DescType;
  typedef FeatureGeometry GeomType;
  typedef InvertedFileEntry<kEmbeddingDim> EntryType;

  enum Status {
    UNUSABLE = 0x00,
    HAS_EMBEDDING = 0x01,
    ENTRIES_SORTED = 0x02,
    USABLE = 0x03,
  };

  InvertedFile();

  // The number of added entries.
  size_t NumEntries() const;

  // Return all entries in the file.
  const std::vector<EntryType>& GetEntries() const;

  // Whether the Hamming embedding was computed for this file.
  bool HasHammingEmbedding() const;

  // Whether the entries in this file are sorted.
  bool EntriesSorted() const;

  // Whether this file is usable for scoring, i.e. the entries are sorted and
  // the Hamming embedding has been computed.
  bool IsUsable() const;

  // Adds an inverted file entry given a projected descriptor and its image
  // information stored in an inverted file entry. In particular, this function
  // generates the binary descriptor for the inverted file entry and then stores
  // the entry in the inverted file.
  void AddEntry(const int image_id, typename DescType::Index feature_idx,
                const DescType& descriptor, const GeomType& geometry);

  // Sorts the inverted file entries in ascending order of image ids. This is
  // required for efficient scoring and must be called before ScoreFeature.
  void SortEntries();

  // Clear all entries in this file.
  void ClearEntries();

  // Reset all computed weights/thresholds and clear all entries.
  void Reset();

  // Given a projected descriptor, returns the corresponding binary string.
  void ConvertToBinaryDescriptor(
      const DescType& descriptor,
      std::bitset<kEmbeddingDim>* binary_descriptor) const;

  // Compute the idf-weight for this inverted file.
  void ComputeIDFWeight(const int num_total_images);

  // Return the idf-weight of this inverted file.
  float IDFWeight() const;

  // Given a set of descriptors, learns the thresholds required for the Hamming
  // embedding. Each row in descriptors represents a single descriptor projected
  // into the kEmbeddingDim dimensional space used for Hamming embedding.
  void ComputeHammingEmbedding(
      const Eigen::Matrix<float, Eigen::Dynamic, kEmbeddingDim>& descriptors);

  // Given a query feature, performs inverted file scoring.
  void ScoreFeature(const DescType& descriptor,
                    std::vector<ImageScore>* image_scores) const;

  // Get the identifiers of all indexed images in this file.
  void GetImageIds(std::unordered_set<int>* ids) const;

  // For each image in the inverted file, computes the self-similarity of each
  // image in the file (the part caused by this word) and adds the weight to the
  // entry corresponding to that image. This function is useful to determine the
  // normalization factor for each image that is used during retrieval.
  void ComputeImageSelfSimilarities(
      std::unordered_map<int, double>* self_similarities) const;

  // Read/write the inverted file from/to a binary file.
  void Read(std::ifstream* ifs);
  void Write(std::ofstream* ofs) const;

 private:
  // Whether the inverted file is initialized.
  uint8_t status_;

  // The inverse document frequency weight of this inverted file.
  float idf_weight_;

  // The entries of the inverted file system.
  std::vector<EntryType> entries_;

  // The thresholds used for Hamming embedding.
  DescType thresholds_;

  // The functor to derive a voting weight from a Hamming distance.
  static const HammingDistWeightFunctor<kEmbeddingDim>
      hamming_dist_weight_functor_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kEmbeddingDim>
const HammingDistWeightFunctor<kEmbeddingDim>
    InvertedFile<kEmbeddingDim>::hamming_dist_weight_functor_;

template <int kEmbeddingDim>
InvertedFile<kEmbeddingDim>::InvertedFile()
    : status_(UNUSABLE), idf_weight_(0.0f) {
  static_assert(kEmbeddingDim % 8 == 0,
                "Dimensionality of projected space needs to"
                " be a multiple of 8.");
  static_assert(kEmbeddingDim > 0,
                "Dimensionality of projected space needs to be > 0.");

  thresholds_.resize(kEmbeddingDim);
  thresholds_.setZero();
}

template <int kEmbeddingDim>
size_t InvertedFile<kEmbeddingDim>::NumEntries() const {
  return entries_.size();
}

template <int kEmbeddingDim>
const std::vector<typename InvertedFile<kEmbeddingDim>::EntryType>&
InvertedFile<kEmbeddingDim>::GetEntries() const {
  return entries_;
}

template <int kEmbeddingDim>
bool InvertedFile<kEmbeddingDim>::HasHammingEmbedding() const {
  return status_ & HAS_EMBEDDING;
}

template <int kEmbeddingDim>
bool InvertedFile<kEmbeddingDim>::EntriesSorted() const {
  return status_ & ENTRIES_SORTED;
}

template <int kEmbeddingDim>
bool InvertedFile<kEmbeddingDim>::IsUsable() const {
  return status_ & USABLE;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::AddEntry(const int image_id,
                                           typename DescType::Index feature_idx,
                                           const DescType& descriptor,
                                           const GeomType& geometry) {
  CHECK_GE(image_id, 0);
  CHECK_EQ(descriptor.size(), kEmbeddingDim);
  EntryType entry;
  entry.image_id = image_id;
  entry.feature_idx = feature_idx;
  entry.geometry = geometry;
  ConvertToBinaryDescriptor(descriptor, &entry.descriptor);
  entries_.push_back(entry);
  status_ &= ~ENTRIES_SORTED;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::SortEntries() {
  std::sort(entries_.begin(), entries_.end(),
            [](const EntryType& entry1, const EntryType& entry2) {
              return entry1.image_id < entry2.image_id;
            });
  status_ |= ENTRIES_SORTED;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ClearEntries() {
  entries_.clear();
  status_ &= ~ENTRIES_SORTED;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::Reset() {
  status_ = UNUSABLE;
  idf_weight_ = 0.0f;
  entries_.clear();
  thresholds_.setZero();
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ConvertToBinaryDescriptor(
    const DescType& descriptor,
    std::bitset<kEmbeddingDim>* binary_descriptor) const {
  CHECK_EQ(descriptor.size(), kEmbeddingDim);
  for (int i = 0; i < kEmbeddingDim; ++i) {
    (*binary_descriptor)[i] = descriptor[i] > thresholds_[i];
  }
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ComputeIDFWeight(const int num_total_images) {
  if (entries_.empty()) {
    return;
  }

  std::unordered_set<int> image_ids;
  GetImageIds(&image_ids);

  idf_weight_ = std::log(static_cast<double>(num_total_images) /
                         static_cast<double>(image_ids.size()));
}

template <int kEmbeddingDim>
float InvertedFile<kEmbeddingDim>::IDFWeight() const {
  return idf_weight_;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ComputeHammingEmbedding(
    const Eigen::Matrix<float, Eigen::Dynamic, kEmbeddingDim>& descriptors) {
  const int num_descriptors = static_cast<int>(descriptors.rows());
  if (num_descriptors < 2) {
    return;
  }

  std::vector<float> elements(num_descriptors);
  for (int n = 0; n < kEmbeddingDim; ++n) {
    for (int i = 0; i < num_descriptors; ++i) {
      elements[i] = descriptors(i, n);
    }
    thresholds_[n] = Median(elements);
  }

  status_ |= HAS_EMBEDDING;
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ScoreFeature(
    const DescType& descriptor, std::vector<ImageScore>* image_scores) const {
  CHECK_EQ(descriptor.size(), kEmbeddingDim);

  image_scores->clear();

  if (!IsUsable()) {
    return;
  }

  if (entries_.size() == 0) {
    return;
  }

  const float squared_idf_weight = idf_weight_ * idf_weight_;

  std::bitset<kEmbeddingDim> bin_descriptor;
  ConvertToBinaryDescriptor(descriptor, &bin_descriptor);

  ImageScore image_score;
  image_score.image_id = entries_.front().image_id;
  image_score.score = 0.0f;
  int num_image_votes = 0;

  // Note that this assumes that the entries are sorted using SortEntries
  // according to their image identifiers.
  for (const auto& entry : entries_) {
    if (image_score.image_id < entry.image_id) {
      if (num_image_votes > 0) {
        // Finalizes the voting since we now know how many features from
        // the database image match the current image feature. This is
        // required to perform burstiness normalization (cf. Eqn. 2 in
        // Arandjelovic, Zisserman: Scalable descriptor
        // distinctiveness for location recognition. ACCV 2014).
        // Notice that the weight from the descriptor matching is already
        // accumulated in image_score.score, i.e., we only need
        // to apply the burstiness weighting.
        image_score.score /= std::sqrt(static_cast<float>(num_image_votes));
        image_score.score *= squared_idf_weight;
        image_scores->push_back(image_score);
      }

      image_score.image_id = entry.image_id;
      image_score.score = 0.0f;
      num_image_votes = 0;
    }

    const size_t hamming_dist = (bin_descriptor ^ entry.descriptor).count();

    if (hamming_dist <= hamming_dist_weight_functor_.kMaxHammingDistance) {
      image_score.score += hamming_dist_weight_functor_(hamming_dist);
      num_image_votes += 1;
    }
  }

  // Add the voting for the largest image_id in the entries.
  if (num_image_votes > 0) {
    image_score.score /= std::sqrt(static_cast<float>(num_image_votes));
    image_score.score *= squared_idf_weight;
    image_scores->push_back(image_score);
  }
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::GetImageIds(
    std::unordered_set<int>* ids) const {
  for (const EntryType& entry : entries_) {
    ids->insert(entry.image_id);
  }
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::ComputeImageSelfSimilarities(
    std::unordered_map<int, double>* self_similarities) const {
  const double squared_idf_weight = idf_weight_ * idf_weight_;
  for (const auto& entry : entries_) {
    (*self_similarities)[entry.image_id] += squared_idf_weight;
  }
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::Read(std::ifstream* ifs) {
  CHECK(ifs->is_open());

  ifs->read(reinterpret_cast<char*>(&status_), sizeof(uint8_t));
  ifs->read(reinterpret_cast<char*>(&idf_weight_), sizeof(float));

  for (int i = 0; i < kEmbeddingDim; ++i) {
    ifs->read(reinterpret_cast<char*>(&thresholds_[i]), sizeof(float));
  }

  uint32_t num_entries = 0;
  ifs->read(reinterpret_cast<char*>(&num_entries), sizeof(uint32_t));
  entries_.resize(num_entries);

  for (uint32_t i = 0; i < num_entries; ++i) {
    entries_[i].Read(ifs);
  }
}

template <int kEmbeddingDim>
void InvertedFile<kEmbeddingDim>::Write(std::ofstream* ofs) const {
  CHECK(ofs->is_open());

  ofs->write(reinterpret_cast<const char*>(&status_), sizeof(uint8_t));
  ofs->write(reinterpret_cast<const char*>(&idf_weight_), sizeof(float));

  for (int i = 0; i < kEmbeddingDim; ++i) {
    ofs->write(reinterpret_cast<const char*>(&thresholds_[i]), sizeof(float));
  }

  const uint32_t num_entries = static_cast<uint32_t>(entries_.size());
  ofs->write(reinterpret_cast<const char*>(&num_entries), sizeof(uint32_t));

  for (uint32_t i = 0; i < num_entries; ++i) {
    entries_[i].Write(ofs);
  }
}

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_INVERTED_FILE_H_
