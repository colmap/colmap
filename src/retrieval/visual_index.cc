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

#include "retrieval/visual_index.h"

#include "util/logging.h"
#include "util/math.h"

namespace colmap {
namespace retrieval {

VisualIndex::VisualIndex() : prepared_(false) {}

VisualIndex::~VisualIndex() {
  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }
}

size_t VisualIndex::NumVisualWords() const { return visual_words_.rows; }

void VisualIndex::Add(const IndexOptions& options, const int image_id,
                      Desc& descriptors) {
  CHECK(image_ids_.count(image_id) == 0);
  image_ids_.insert(image_id);

  prepared_ = false;

  if (descriptors.rows() == 0) {
    return;
  }

  const Eigen::MatrixXi word_ids =
      FindWordIds(descriptors, options.num_neighbors, options.num_checks,
                  options.num_threads);

  for (Desc::Index i = 0; i < descriptors.rows(); ++i) {
    for (int n = 0; n < options.num_neighbors; ++n) {
      const int word_id = word_ids(i, n);
      if (word_id != InvertedIndexType::kInvalidWordId) {
        inverted_index_.AddEntry(image_id, word_id, descriptors.row(i));
      }
    }
  }
}

void VisualIndex::Query(const QueryOptions& options, Desc& descriptors,
                        std::vector<ImageScore>* image_scores) const {
  CHECK(prepared_);

  if (descriptors.rows() == 0) {
    image_scores->clear();
    return;
  }

  const Eigen::MatrixXi word_ids =
      FindWordIds(descriptors, options.num_neighbors, options.num_checks,
                  options.num_threads);
  inverted_index_.Query(descriptors, word_ids, image_scores);

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
    std::partial_sort(image_scores->begin(), image_scores->begin() + num_images,
                      image_scores->end(), SortFunc);
    image_scores->resize(num_images);
  }
}

void VisualIndex::Prepare() {
  inverted_index_.Finalize();
  prepared_ = true;
}

void VisualIndex::Build(const BuildOptions& options, Desc& descriptors) {
  // Quantize the descriptor space into visual words.
  Quantize(options, descriptors);

  // Build the search index on the visual words.
  flann::AutotunedIndexParams index_params;
  index_params["target_precision"] =
      static_cast<float>(options.target_precision);
  visual_word_index_ = flann::AutotunedIndex<flann::L2<uint8_t>>(index_params);
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

void VisualIndex::Read(const std::string& path) {
  long int file_offset = 0;

  // Read the visual words.

  {
    if (visual_words_.ptr() != nullptr) {
      delete[] visual_words_.ptr();
    }

    std::ifstream file(path, std::ios::binary);
    uint64_t rows;
    file.read(reinterpret_cast<char*>(&rows), sizeof(uint64_t));
    uint64_t cols;
    file.read(reinterpret_cast<char*>(&cols), sizeof(uint64_t));
    uint8_t* visual_words_data = new uint8_t[rows * cols];
    file.read(reinterpret_cast<char*>(visual_words_data), rows * cols);
    visual_words_ = flann::Matrix<uint8_t>(visual_words_data, rows, cols);
    file_offset = file.tellg();
  }

  // Read the visual words search index.

  visual_word_index_ = flann::AutotunedIndex<flann::L2<uint8_t>>(visual_words_);

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
    CHECK(file);
    file.seekg(file_offset, std::ios::beg);
    inverted_index_.Read(&file);
  }
}

void VisualIndex::Write(const std::string& path) {
  // Write the visual words.

  {
    CHECK_NOTNULL(visual_words_.ptr());
    std::ofstream file(path, std::ios::binary);
    const uint64_t rows = static_cast<uint64_t>(visual_words_.rows);
    file.write(reinterpret_cast<const char*>(&rows), sizeof(uint64_t));
    const uint64_t cols = static_cast<uint64_t>(visual_words_.cols);
    file.write(reinterpret_cast<const char*>(&cols), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(visual_words_.ptr()),
               visual_words_.rows * visual_words_.cols);
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
    CHECK(file);
    inverted_index_.Write(&file);
  }
}

void VisualIndex::Quantize(const BuildOptions& options, Desc& descriptors) {
  static_assert(Desc::IsRowMajor, "Descriptors must be row-major.");

  CHECK_GE(options.num_visual_words, options.branching);
  CHECK_GE(descriptors.rows(), options.num_visual_words);

  const flann::Matrix<uint8_t> descriptor_matrix(
      descriptors.data(), descriptors.rows(), descriptors.cols());

  std::vector<float> centers_data(options.num_visual_words *
                                  descriptors.cols());
  flann::Matrix<float> centers(centers_data.data(), options.num_visual_words,
                               descriptors.cols());

  flann::KMeansIndexParams index_params;
  index_params["branching"] = options.branching;
  index_params["iterations"] = options.num_iterations;
  index_params["centers_init"] = flann::FLANN_CENTERS_KMEANSPP;
  const int num_centers = flann::hierarchicalClustering<flann::L2<uint8_t>>(
      descriptor_matrix, centers, index_params);

  CHECK_LE(num_centers, options.num_visual_words);

  const size_t visual_word_data_size = num_centers * descriptors.cols();
  uint8_t* visual_words_data = new uint8_t[visual_word_data_size];
  for (size_t i = 0; i < visual_word_data_size; ++i) {
    visual_words_data[i] = Clip(std::round(centers_data[i]), 0.0f, 255.0f);
  }

  if (visual_words_.ptr() != nullptr) {
    delete[] visual_words_.ptr();
  }

  visual_words_ = flann::Matrix<uint8_t>(visual_words_data, num_centers,
                                         descriptors.cols());
}

Eigen::MatrixXi VisualIndex::FindWordIds(Desc& descriptors,
                                         const int num_neighbors,
                                         const int num_checks,
                                         const int num_threads) const {
  static_assert(Desc::IsRowMajor, "Descriptors must be row-major.");

  CHECK_GT(descriptors.rows(), 0);
  CHECK_GT(num_neighbors, 0);

  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      word_ids(descriptors.rows(), num_neighbors);
  word_ids.setConstant(InvertedIndexType::kInvalidWordId);
  flann::Matrix<size_t> indices(word_ids.data(), descriptors.rows(),
                                num_neighbors);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix(descriptors.rows(), num_neighbors);
  flann::Matrix<float> distances(distance_matrix.data(), descriptors.rows(),
                                 num_neighbors);

  flann::Matrix<uint8_t> query(descriptors.data(), descriptors.rows(),
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

  visual_word_index_.knnSearch(query, indices, distances, num_neighbors,
                               search_params);

  return word_ids.cast<int>();
}

}  // namespace retrieval
}  // namespace colmap
