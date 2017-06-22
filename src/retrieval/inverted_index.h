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

#ifndef COLMAP_SRC_RETRIEVAL_INVERTED_INDEX_H_
#define COLMAP_SRC_RETRIEVAL_INVERTED_INDEX_H_

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "retrieval/inverted_file.h"
#include "util/alignment.h"
#include "util/random.h"

namespace colmap {
namespace retrieval {

// Implements an inverted index system. The template parameter is the length of
// the binary vectors in the Hamming Embedding.
// This class is based on an original implementation by Torsten Sattler.
template <typename kDescType, int kDescDim, int kEmbeddingDim>
class InvertedIndex {
 public:
  const static int kInvalidWordId;
  typedef Eigen::Matrix<kDescType, Eigen::Dynamic, kDescDim, Eigen::RowMajor>
      DescType;
  typedef typename InvertedFile<kEmbeddingDim>::GeomType GeomType;
  typedef Eigen::Matrix<float, Eigen::Dynamic, kDescDim> ProjMatrixType;
  typedef Eigen::VectorXf ProjDescType;

  InvertedIndex();

  // The number of visual words in the index.
  int NumVisualWords() const;

  // Initializes the inverted index with num_words empty inverted files.
  void Initialize(const int num_words);

  // Finalizes the inverted index by sorting each inverted file such that all
  // entries are in ascending order of image ids.
  void Finalize();

  // Generate projection matrix for Hamming embedding.
  void GenerateHammingEmbeddingProjection();

  // Compute Hamming embedding thresholds from a set of descriptors with
  // given visual word identifies.
  void ComputeHammingEmbedding(const DescType& descriptors,
                               const Eigen::VectorXi& word_ids);

  // Add single entry to the index.
  void AddEntry(const int image_id, const int word_id,
                const DescType& descriptor, const GeomType& geometry);

  // Clear all index entries.
  void ClearEntries();

  // Query the inverted file and return a list of sorted images.
  void Query(const DescType& descriptors, const Eigen::MatrixXi& word_ids,
             std::vector<ImageScore>* image_scores) const;

  void FindMatches(const int word_id, const std::unordered_set<int>& image_ids,
                   std::vector<std::pair<int, GeomType>>* matches) const;

  // Compute the self-similarity for the image.
  float ComputeSelfSimilarity(const Eigen::MatrixXi& word_ids) const;

  // Get the identifiers of all indexed images.
  void GetImageIds(std::unordered_set<int>* image_ids) const;

  // Read/write the inverted index from/to a binary file.
  void Read(std::ifstream* ifs);
  void Write(std::ofstream* ofs) const;

 private:
  void ComputeWeightsAndNormalizationConstants();

  // The individual inverted indices.
  std::vector<InvertedFile<kEmbeddingDim>,
              Eigen::aligned_allocator<InvertedFile<kEmbeddingDim>>>
      inverted_files_;

  // For each image in the database, a normalization factor to be used to
  // normalize the votes.
  std::vector<float> normalization_constants_;

  // The projection matrix used to project SIFT descriptors.
  ProjMatrixType proj_matrix_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename kDescType, int kDescDim, int kEmbeddingDim>
const int InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::kInvalidWordId =
    std::numeric_limits<int>::max();

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
  CHECK_GT(num_words, 0);
  inverted_files_.resize(num_words);
  for (auto& inverted_file : inverted_files_) {
    inverted_file.Reset();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Finalize() {
  CHECK_GT(NumVisualWords(), 0);

  for (auto& inverted_file : inverted_files_) {
    inverted_file.SortEntries();
  }

  ComputeWeightsAndNormalizationConstants();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim,
                   kEmbeddingDim>::GenerateHammingEmbeddingProjection() {
  Eigen::MatrixXf random_matrix(kDescDim, kDescDim);
  for (Eigen::MatrixXf::Index i = 0; i < random_matrix.size(); ++i) {
    random_matrix(i) = RandomGaussian(0.0f, 1.0f);
  }
  const Eigen::MatrixXf Q = random_matrix.colPivHouseholderQr().matrixQ();
  proj_matrix_ = Q.topRows<kEmbeddingDim>();
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ComputeHammingEmbedding(
    const DescType& descriptors, const Eigen::VectorXi& word_ids) {
  CHECK_EQ(descriptors.rows(), word_ids.rows());
  CHECK_EQ(descriptors.cols(), kDescDim);

  // Skip every inverted file with less than kMinEntries entries.
  const size_t kMinEntries = 5;

  // Determines for each word the corresponding descriptors.
  std::vector<std::vector<int>> indices_per_word(NumVisualWords());
  for (Eigen::MatrixXi::Index i = 0; i < word_ids.rows(); ++i) {
    indices_per_word.at(word_ids(i)).push_back(i);
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
    const int image_id, const int word_id, const DescType& descriptor,
    const GeomType& geometry) {
  CHECK_EQ(descriptor.size(), kDescDim);
  const ProjDescType proj_desc =
      proj_matrix_ * descriptor.transpose().template cast<float>();
  inverted_files_.at(word_id).AddEntry(image_id, proj_desc, geometry);
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ClearEntries() {
  for (auto& inverted_file : inverted_files_) {
    inverted_file.ClearEntries();
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Query(
    const DescType& descriptors, const Eigen::MatrixXi& word_ids,
    std::vector<ImageScore>* image_scores) const {
  CHECK_EQ(descriptors.cols(), kDescDim);

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
      const int word_id = word_ids(i, n);
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
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::FindMatches(
    const int word_id, const std::unordered_set<int>& image_ids,
    std::vector<std::pair<int, GeomType>>* matches) const {
  matches->clear();
  const auto& entries = inverted_files_.at(word_id).GetEntries();
  for (const auto& entry : entries) {
    if (image_ids.count(entry.image_id)) {
      matches->emplace_back(entry.image_id, entry.geometry);
    }
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
float InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::ComputeSelfSimilarity(
    const Eigen::MatrixXi& word_ids) const {
  double self_similarity = 0.0;
  for (Eigen::MatrixXi::Index i = 0; i < word_ids.size(); ++i) {
    const int word_id = word_ids(i);
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
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Read(
    std::ifstream* ifs) {
  CHECK(ifs->is_open());

  int32_t num_words = 0;
  ifs->read(reinterpret_cast<char*>(&num_words), sizeof(int32_t));
  CHECK_GT(num_words, 0);

  Initialize(num_words);

  int32_t N_t = 0;
  ifs->read(reinterpret_cast<char*>(&N_t), sizeof(int32_t));
  CHECK_EQ(N_t, kEmbeddingDim)
      << "The length of the binary strings should be " << kEmbeddingDim
      << " but is " << N_t << ". The indices are not compatible!";

  for (int i = 0; i < kEmbeddingDim; ++i) {
    for (int j = 0; j < kDescDim; ++j) {
      ifs->read(reinterpret_cast<char*>(&proj_matrix_(i, j)), sizeof(float));
    }
  }

  for (auto& inverted_file : inverted_files_) {
    inverted_file.Read(ifs);
  }

  int32_t num_images = 0;
  ifs->read(reinterpret_cast<char*>(&num_images), sizeof(int32_t));
  CHECK_GE(num_images, 0);

  normalization_constants_.resize(num_images);
  for (int32_t image_id = 0; image_id < num_images; ++image_id) {
    ifs->read(reinterpret_cast<char*>(&normalization_constants_[image_id]),
              sizeof(float));
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim, kEmbeddingDim>::Write(
    std::ofstream* ofs) const {
  CHECK(ofs->is_open());

  int32_t num_words = static_cast<int32_t>(NumVisualWords());
  ofs->write(reinterpret_cast<const char*>(&num_words), sizeof(int32_t));
  CHECK_GT(num_words, 0);

  const int32_t N_t = static_cast<int32_t>(kEmbeddingDim);
  ofs->write(reinterpret_cast<const char*>(&N_t), sizeof(int32_t));

  for (int i = 0; i < kEmbeddingDim; ++i) {
    for (int j = 0; j < kDescDim; ++j) {
      ofs->write(reinterpret_cast<const char*>(&proj_matrix_(i, j)),
                 sizeof(float));
    }
  }

  for (const auto& inverted_file : inverted_files_) {
    inverted_file.Write(ofs);
  }

  const int32_t num_images = normalization_constants_.size();
  ofs->write(reinterpret_cast<const char*>(&num_images), sizeof(int32_t));

  for (int32_t image_id = 0; image_id < num_images; ++image_id) {
    ofs->write(
        reinterpret_cast<const char*>(&normalization_constants_[image_id]),
        sizeof(float));
  }
}

template <typename kDescType, int kDescDim, int kEmbeddingDim>
void InvertedIndex<kDescType, kDescDim,
                   kEmbeddingDim>::ComputeWeightsAndNormalizationConstants() {
  std::unordered_set<int> image_ids;
  GetImageIds(&image_ids);

  for (auto& inverted_file : inverted_files_) {
    inverted_file.ComputeIDFWeight(image_ids.size());
  }

  const int max_image_id =
      *std::max_element(image_ids.begin(), image_ids.end());

  std::vector<double> self_similarities(max_image_id + 1, 0.0);
  for (const auto& inverted_file : inverted_files_) {
    inverted_file.ComputeImageSelfSimilarities(&self_similarities);
  }

  normalization_constants_.resize(max_image_id + 1);
  for (int image_id = 0; image_id <= max_image_id; ++image_id) {
    const double self_similarity = self_similarities.at(image_id);
    if (self_similarity > 0.0) {
      normalization_constants_.at(image_id) =
          static_cast<float>(1.0 / std::sqrt(self_similarity));
    } else {
      normalization_constants_.at(image_id) = 0.0f;
    }
  }
}

}  // namespace retrieval
}  // namespace colmap

#endif  // COLMAP_SRC_RETRIEVAL_INVERTED_INDEX_H_
