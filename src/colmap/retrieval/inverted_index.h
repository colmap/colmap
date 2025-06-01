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

#include "colmap/math/random.h"
#include "colmap/retrieval/inverted_file.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace colmap {
namespace retrieval {

// Implements an inverted index system. The template parameter is the length of
// the binary vectors in the Hamming Embedding.
// This class is based on an original implementation by Torsten Sattler.
template <typename kDescType, int kDescDim, int kEmbeddingDim>
class InvertedIndex {
 public:
  const static int64_t kInvalidWordId;
  typedef Eigen::Matrix<kDescType, Eigen::Dynamic, kDescDim, Eigen::RowMajor>
      DescType;
  typedef typename InvertedFile<kEmbeddingDim>::EntryType EntryType;
  typedef typename InvertedFile<kEmbeddingDim>::GeomType GeomType;
  typedef Eigen::Matrix<float, Eigen::Dynamic, kDescDim> ProjMatrixType;
  typedef Eigen::VectorXf ProjDescType;
  using WordIds =
      Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  InvertedIndex();

  // The number of visual words in the index.
  int NumVisualWords() const;

  // Initializes the inverted index with num_words empty inverted files.
  void Initialize(int num_words);

  // Finalizes the inverted index by sorting each inverted file such that all
  // entries are in ascending order of image ids.
  void Finalize();

  // Generate projection matrix for Hamming embedding.
  void GenerateHammingEmbeddingProjection();

  // Compute Hamming embedding thresholds from a set of descriptors with
  // given visual word identifies.
  void ComputeHammingEmbedding(const DescType& descriptors,
                               const WordIds& word_ids);

  // Add single entry to the index.
  void AddEntry(int image_id,
                int64_t word_id,
                typename DescType::Index feature_idx,
                const DescType& descriptor,
                const GeomType& geometry);

  // Clear all index entries.
  void ClearEntries();

  // Query the inverted file and return a list of sorted images.
  void Query(const DescType& descriptors,
             const WordIds& word_ids,
             std::vector<ImageScore>* image_scores) const;

  void ConvertToBinaryDescriptor(
      int64_t word_id,
      const DescType& descriptor,
      std::bitset<kEmbeddingDim>* binary_descriptor) const;

  float GetIDFWeight(int64_t word_id) const;

  void FindMatches(int64_t word_id,
                   const std::unordered_set<int>& image_ids,
                   std::vector<const EntryType*>* matches) const;

  // Compute the self-similarity for the image.
  float ComputeSelfSimilarity(const WordIds& word_ids) const;

  // Get the identifiers of all indexed images.
  void GetImageIds(std::unordered_set<int>* image_ids) const;

  // Read/write the inverted index from/to a binary file.
  void Read(std::istream* in);
  void Write(std::ostream* out) const;

 private:
  void ComputeWeightsAndNormalizationConstants();

  // The individual inverted indices.
  std::vector<InvertedFile<kEmbeddingDim>,
              Eigen::aligned_allocator<InvertedFile<kEmbeddingDim>>>
      inverted_files_;

  // For each image in the database, a normalization factor to be used to
  // normalize the votes.
  std::unordered_map<int, float> normalization_constants_;

  // The projection matrix used to project SIFT descriptors.
  ProjMatrixType proj_matrix_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename kDescType, int kDescDim, int kEmbeddingDim>
const int64_t
    InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::kInvalidWordId = -1;

template <typename kDescType, int kDescDim, int kEmbeddingDim>
InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::InvertedIndex() {
  proj_matrix_.resize(kEmbeddingDim, kDescDim);
  proj_matrix_.setIdentity();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
int InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::NumVisualWords() const {
  return static_cast<int>(inverted_files_.size());
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Initialize(
    const int num_words) {
  THROW_CHECK_GT(num_words, 0);
  inverted_files_.resize(num_words);
  for (auto& inverted_file : inverted_files_) {
    inverted_file.Reset();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Finalize() {
  THROW_CHECK_GT(NumVisualWords(), 0);

  for (auto& inverted_file : inverted_files_) {
    inverted_file.SortEntries();
  }

  ComputeWeightsAndNormalizationConstants();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::
    GenerateHammingEmbeddingProjection() {
  Eigen::MatrixXf random_matrix(kDescDim, kDescDim);
  for (Eigen::MatrixXf::Index i = 0; i < random_matrix.size(); ++i) {
    random_matrix(i) = RandomGaussian(0.0f, 1.0f);
  }
  const Eigen::MatrixXf Q = random_matrix.colPivHouseholderQr().matrixQ();
  proj_matrix_ = Q.topRows<kEmbeddingDim>();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ComputeHammingEmbedding(
    const DescType& descriptors, const WordIds& word_ids) {
  THROW_CHECK_EQ(descriptors.rows(), word_ids.rows());
  THROW_CHECK_EQ(descriptors.cols(), kDescDim);
  THROW_CHECK_EQ(word_ids.cols(), 1);

  // Skip every inverted file with less than kMinEntries entries.
  const size_t kMinEntries = 5;

  // Determines for each word the corresponding descriptors.
  std::vector<std::vector<int>> indices_per_word(NumVisualWords());
  for (Eigen::MatrixXi::Index i = 0; i < word_ids.rows(); ++i) {
    const int64_t word_id = word_ids(i);
    if (word_id != kInvalidWordId) {
      indices_per_word.at(word_ids(i)).push_back(i);
    }
  }

  // For each word, learn the Hamming embedding threshold and the local
  // descriptor space densities.
  for (int i = 0; i < NumVisualWords(); ++i) {
    const auto& indices = indices_per_word[i];
    if (indices.size() < kMinEntries) {
      continue;
    }

    Eigen::Matrix<float, Eigen::Dynamic, kEmbeddingDim> proj_desc(
        indices.size(), kEmbeddingDim);
    for (size_t j = 0; j < indices.size(); ++j) {
      proj_desc.row(j) =
          proj_matrix_ *
          descriptors.row(indices[j]).transpose().template cast<float>();
    }

    inverted_files_[i].ComputeHammingEmbedding(proj_desc);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::AddEntry(
    const int image_id,
    const int64_t word_id,
    typename DescType::Index feature_idx,
    const DescType& descriptor,
    const GeomType& geometry) {
  THROW_CHECK_EQ(descriptor.size(), kDescDim);
  const ProjDescType proj_desc =
      proj_matrix_ * descriptor.transpose().template cast<float>();
  inverted_files_.at(word_id).AddEntry(
      image_id, feature_idx, proj_desc, geometry);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ClearEntries() {
  for (auto& inverted_file : inverted_files_) {
    inverted_file.ClearEntries();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Query(
    const DescType& descriptors,
    const WordIds& word_ids,
    std::vector<ImageScore>* image_scores) const {
  THROW_CHECK_EQ(descriptors.cols(), kDescDim);

  image_scores->clear();

  // Computes the self-similarity score for the query image.
  const float self_similarity = ComputeSelfSimilarity(word_ids);
  float normalization_weight = 1.0f;
  if (self_similarity > 0.0f) {
    normalization_weight = 1.0f / std::sqrt(self_similarity);
  }

  std::unordered_map<int, int> score_map;
  std::vector<ImageScore> inverted_file_scores;

  for (typename DescType::Index i = 0; i < descriptors.rows(); ++i) {
    const ProjDescType proj_descriptor =
        proj_matrix_ * descriptors.row(i).transpose().template cast<float>();
    for (Eigen::MatrixXi::Index n = 0; n < word_ids.cols(); ++n) {
      const int64_t word_id = word_ids(i, n);
      if (word_id == kInvalidWordId) {
        continue;
      }

      inverted_files_.at(word_id).ScoreFeature(proj_descriptor,
                                               &inverted_file_scores);

      for (const ImageScore& score : inverted_file_scores) {
        const auto score_map_it = score_map.find(score.image_id);
        if (score_map_it == score_map.end()) {
          // Image not found in another inverted file.
          score_map.emplace(score.image_id,
                            static_cast<int>(image_scores->size()));
          image_scores->push_back(score);
        } else {
          // Image already found in another inverted file, so accumulate.
          (*image_scores).at(score_map_it->second).score += score.score;
        }
      }
    }
  }

  // Normalization.
  for (ImageScore& score : *image_scores) {
    score.score *=
        normalization_weight * normalization_constants_.at(score.image_id);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::
    ConvertToBinaryDescriptor(
        const int64_t word_id,
        const DescType& descriptor,
        std::bitset<kEmbeddingDim>* binary_descriptor) const {
  const ProjDescType proj_desc =
      proj_matrix_ * descriptor.transpose().template cast<float>();
  inverted_files_.at(word_id).ConvertToBinaryDescriptor(proj_desc,
                                                        binary_descriptor);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
float InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::GetIDFWeight(
    const int64_t word_id) const {
  return inverted_files_.at(word_id).IDFWeight();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::FindMatches(
    const int64_t word_id,
    const std::unordered_set<int>& image_ids,
    std::vector<const EntryType*>* matches) const {
  matches->clear();
  const auto& entries = inverted_files_.at(word_id).GetEntries();
  for (const auto& entry : entries) {
    if (image_ids.count(entry.image_id)) {
      matches->emplace_back(&entry);
    }
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
float InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ComputeSelfSimilarity(
    const WordIds& word_ids) const {
  double self_similarity = 0.0;
  for (Eigen::MatrixXi::Index i = 0; i < word_ids.size(); ++i) {
    const int64_t word_id = word_ids(i);
    if (word_id != kInvalidWordId) {
      const auto& inverted_file = inverted_files_.at(word_id);
      self_similarity += inverted_file.IDFWeight() * inverted_file.IDFWeight();
    }
  }
  return static_cast<float>(self_similarity);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::GetImageIds(
    std::unordered_set<int>* image_ids) const {
  for (const auto& inverted_file : inverted_files_) {
    inverted_file.GetImageIds(image_ids);
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Read(std::istream* in) {
  THROW_CHECK(in->good());

  const int32_t num_words = ReadBinaryLittleEndian<int>(in);
  THROW_CHECK_GT(num_words, 0);

  Initialize(num_words);

  const int in_embedding_dim = ReadBinaryLittleEndian<int>(in);
  THROW_CHECK_EQ(in_embedding_dim, kEmbeddingDim)
      << "The length of the binary strings should be " << kEmbeddingDim
      << " but is " << in_embedding_dim << ". The indices are not compatible!";

  for (int i = 0; i < kEmbeddingDim; ++i) {
    for (int j = 0; j < kDescDim; ++j) {
      in->read(reinterpret_cast<char*>(&proj_matrix_(i, j)), sizeof(float));
    }
  }

  for (auto& inverted_file : inverted_files_) {
    inverted_file.Read(in);
  }

  const int num_images = ReadBinaryLittleEndian<int>(in);
  THROW_CHECK_GE(num_images, 0);

  normalization_constants_.clear();
  normalization_constants_.reserve(num_images);
  for (int32_t i = 0; i < num_images; ++i) {
    const int image_id = ReadBinaryLittleEndian<int>(in);
    float value;
    in->read(reinterpret_cast<char*>(&value), sizeof(float));
    normalization_constants_[image_id] = value;
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Write(
    std::ostream* out) const {
  THROW_CHECK(out->good());

  const int num_words = static_cast<int32_t>(NumVisualWords());
  THROW_CHECK_GT(num_words, 0);
  WriteBinaryLittleEndian<int>(out, num_words);

  const int32_t embedding_dim = static_cast<int32_t>(kEmbeddingDim);
  WriteBinaryLittleEndian<int>(out, embedding_dim);

  for (int i = 0; i < kEmbeddingDim; ++i) {
    for (int j = 0; j < kDescDim; ++j) {
      out->write(reinterpret_cast<const char*>(&proj_matrix_(i, j)),
                 sizeof(float));
    }
  }

  for (const auto& inverted_file : inverted_files_) {
    inverted_file.Write(out);
  }

  const int32_t num_images = normalization_constants_.size();
  WriteBinaryLittleEndian<int>(out, num_images);

  for (const auto& constant : normalization_constants_) {
    WriteBinaryLittleEndian<int>(out, constant.first);
    out->write(reinterpret_cast<const char*>(&constant.second), sizeof(float));
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::
    ComputeWeightsAndNormalizationConstants() {
  std::unordered_set<int> image_ids;
  GetImageIds(&image_ids);

  for (auto& inverted_file : inverted_files_) {
    inverted_file.ComputeIDFWeight(image_ids.size());
  }

  std::unordered_map<int, double> self_similarities(image_ids.size());
  for (const auto& inverted_file : inverted_files_) {
    inverted_file.ComputeImageSelfSimilarities(&self_similarities);
  }

  normalization_constants_.clear();
  normalization_constants_.reserve(image_ids.size());
  for (const auto& self_similarity : self_similarities) {
    if (self_similarity.second > 0.0) {
      normalization_constants_[self_similarity.first] =
          static_cast<float>(1.0 / std::sqrt(self_similarity.second));
    } else {
      normalization_constants_[self_similarity.first] = 0.0f;
    }
  }
}

}  // namespace retrieval
}  // namespace colmap
