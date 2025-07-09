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

#include "colmap/retrieval/visual_index.h"

#include "colmap/math/math.h"
#include "colmap/retrieval/inverted_file.h"
#include "colmap/retrieval/inverted_index.h"
#include "colmap/retrieval/vote_and_verify.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <Eigen/Core>
#include <boost/heap/fibonacci_heap.hpp>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <omp.h>

namespace colmap {
namespace retrieval {
namespace {

std::unique_ptr<faiss::IndexIVF> BuildFaissIndex(
    const VisualIndex::BuildOptions& options,
    const Eigen::RowMajorMatrixXf& visual_words) {
  const int64_t num_centroids = std::min<int64_t>(
      visual_words.rows(), 2 * std::sqrt(visual_words.rows()));
  const int64_t spectral_hash_dim =
      std::min<int64_t>(visual_words.rows(), visual_words.cols() / 2);
  std::ostringstream index_type;
  index_type << "IVF" << num_centroids << ",ITQ" << spectral_hash_dim << ",SH";
  VLOG(2) << "Training " << index_type.str()
          << " search index for visual words";
  const auto index_factory_verbose = faiss::index_factory_verbose;
  faiss::index_factory_verbose = VLOG_IS_ON(3);
  auto index = std::unique_ptr<faiss::IndexIVF>(dynamic_cast<faiss::IndexIVF*>(
      faiss::index_factory(visual_words.cols(), index_type.str().c_str())));
  faiss::index_factory_verbose = index_factory_verbose;

#pragma omp parallel num_threads(1)
  {
    omp_set_num_threads(GetEffectiveNumThreads(options.num_threads));
#ifdef _MSC_VER
    omp_set_nested(1);
#else
    omp_set_max_active_levels(1);
#endif

    index->train(visual_words.rows(), visual_words.data());
    index->add(visual_words.rows(), visual_words.data());
  }

  return index;
}

template <int kDescDim = 128, int kEmbeddingDim = 64>
class FaissVisualIndex : public VisualIndex {
 public:
  typedef InvertedIndex<float, kDescDim, kEmbeddingDim> InvertedIndexType;
  typedef typename InvertedIndexType::EntryType EntryType;

  FaissVisualIndex() : prepared_(false) {}

  size_t NumVisualWords() const override {
    return (index_ == nullptr) ? 0 : index_->ntotal;
  }

  size_t NumImages() const override { return image_ids_.size(); }

  virtual int DescDim() const override { return kDescDim; }

  virtual int EmbeddingDim() const override { return kEmbeddingDim; }

  void Add(const IndexOptions& options,
           int image_id,
           const Geometries& geometries,
           const Descriptors& descriptors) override {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);
    THROW_CHECK_EQ(geometries.size(), descriptors.rows());

    // If the image is already indexed, do nothing.
    if (IsImageIndexed(image_id)) {
      return;
    }

    image_ids_.insert(image_id);

    prepared_ = false;

    if (descriptors.rows() == 0) {
      return;
    }

    const WordIds word_ids = FindWordIds(descriptors,
                                         options.num_neighbors,
                                         options.num_checks,
                                         options.num_threads);

    for (typename Descriptors::Index i = 0; i < descriptors.rows(); ++i) {
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

  bool IsImageIndexed(int image_id) const override {
    return image_ids_.count(image_id) != 0;
  }

  void Query(const QueryOptions& options,
             const Descriptors& descriptors,
             std::vector<ImageScore>* image_scores) const override {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);
    const Geometries geometries;
    Query(options, geometries, descriptors, image_scores);
  }

  void Query(const QueryOptions& options,
             const Geometries& geometries,
             const Descriptors& descriptors,
             std::vector<ImageScore>* image_scores) const override {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);

    WordIds word_ids;
    QueryAndFindWordIds(options, descriptors, image_scores, &word_ids);

    if (options.num_images_after_verification <= 0) {
      return;
    }

    THROW_CHECK_EQ(descriptors.rows(), geometries.size());

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

    for (typename Descriptors::Index i = 0; i < descriptors.rows(); ++i) {
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

            if (hamming_dist <=
                hamming_dist_weight_functor.kMaxHammingDistance) {
              const float dist = hamming_dist_weight_functor(hamming_dist) *
                                 squared_idf_weight;

              auto& feature_matches = image_matches[match->image_id];
              const auto feature_match =
                  feature_matches.find(match->feature_idx);

              if (feature_match == feature_matches.end() ||
                  feature_match->first < dist) {
                feature_matches[match->feature_idx] =
                    std::make_pair(dist, match);
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

    for (auto& image_score : *image_scores) {
      auto& query_matches = query_to_db_matches[image_score.image_id];
      auto& db_matches = db_to_query_matches[image_score.image_id];

      // No matches found.
      if (query_matches.empty()) {
        continue;
      }

      // Enforce 1-to-1 matching: Build Fibonacci heaps for the query and
      // database features, ordered by the minimum number of matches per
      // feature. We'll select these matches one at a time. For convenience,
      // we'll also pre-sort the matched feature lists by matching score.

      typedef boost::heap::fibonacci_heap<std::pair<int, int>>
          FibonacciHeapType;
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

        // Entries that have been matched (or processed and subsequently
        // ignored) get their handles removed.
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
                  const auto other_idx1 =
                      (use_query) ? entry1.second.first->feature_idx
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

    // NOLINTNEXTLINE(bugprone-branch-clone)
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

  void Prepare() override {
    inverted_index_.Finalize();
    prepared_ = true;
  }

  void Build(const BuildOptions& options,
             const Descriptors& descriptors) override {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);

    const Eigen::RowMajorMatrixXf visual_words = Quantize(options, descriptors);
    THROW_CHECK_EQ(visual_words.cols(), kDescDim);

    index_ = BuildFaissIndex(options, visual_words);

    // Initialize a new inverted index.
    inverted_index_ = InvertedIndexType();
    inverted_index_.Initialize(index_->ntotal);
    VLOG(2) << "Initialized inverted index";

    // Generate descriptor projection matrix.
    inverted_index_.GenerateHammingEmbeddingProjection();
    VLOG(2) << "Generated hamming embedding";

    // Learn the Hamming embedding.
    const int kNumNeighbors = 1;
    const WordIds word_ids = FindWordIds(
        descriptors, kNumNeighbors, options.num_checks, options.num_threads);
    inverted_index_.ComputeHammingEmbedding(descriptors, word_ids);
    VLOG(2) << "Computed hamming embeddings";
  }

  void ReadFromFaiss(const std::string& path, long offset) override {
    FILE* fin = nullptr;
#ifdef _MSC_VER
    THROW_CHECK_EQ(fopen_s(&fin, path.c_str(), "rb"), 0);
#else
    fin = fopen(path.c_str(), "rb");
#endif
    THROW_CHECK_NOTNULL(fin);
    fseek(fin, offset, SEEK_SET);
    index_ = std::unique_ptr<faiss::IndexIVF>(dynamic_cast<faiss::IndexIVF*>(
        THROW_CHECK_NOTNULL(faiss::read_index(fin))));
    offset = ftell(fin);
    fclose(fin);

    // Read the inverted index.
    std::ifstream file(path, std::ios::binary);
    THROW_CHECK_FILE_OPEN(file, path);
    file.seekg(offset, std::ios::beg);
    inverted_index_.Read(&file);
    image_ids_.clear();
    inverted_index_.GetImageIds(&image_ids_);
  }

  void Write(const std::string& path) const override {
    THROW_CHECK_NOTNULL(index_);

    // Write index metadata header.

    {
      std::ofstream file(path, std::ios::binary);
      THROW_CHECK_FILE_OPEN(file, path);
      const int kFileVersion = 1;
      file.write(reinterpret_cast<const char*>(&kFileVersion), sizeof(int));
      const int desc_dim = kDescDim;
      file.write(reinterpret_cast<const char*>(&desc_dim), sizeof(int));
      const int embedding_dim = kEmbeddingDim;
      file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(int));
    }

    // Write the visual words search index.

    {
      FILE* fout = nullptr;
#ifdef _MSC_VER
      THROW_CHECK_EQ(fopen_s(&fout, path.c_str(), "ab"), 0);
#else
      fout = fopen(path.c_str(), "ab");
#endif
      THROW_CHECK_NOTNULL(fout);
      faiss::write_index(index_.get(), fout);
      fclose(fout);
    }

    // Write the inverted index.

    {
      std::ofstream file(path, std::ios::binary | std::ios::app);
      THROW_CHECK_FILE_OPEN(file, path);
      inverted_index_.Write(&file);
    }
  }

 private:
  using WordIds =
      Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // Quantize the descriptor space into visual words.
  Eigen::RowMajorMatrixXf Quantize(const BuildOptions& options,
                                   const Descriptors& descriptors) const {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);

    VLOG(2) << "Clustering " << descriptors.rows()
            << " descriptors using kmeans";

    faiss::Clustering clustering(kDescDim, options.num_visual_words);
    clustering.niter = options.num_iterations;
    clustering.nredo = options.num_rounds;
    clustering.verbose = VLOG_IS_ON(3);

    faiss::IndexFlatL2 index(kDescDim);
    clustering.train(descriptors.rows(), descriptors.data(), index);

    VLOG(2) << "Quantized into " << options.num_visual_words
            << " visual words with error "
            << clustering.iteration_stats.back().obj;

    return Eigen::Map<const Eigen::RowMajorMatrixXf>(
        clustering.centroids.data(), options.num_visual_words, kDescDim);
  }

  // Query for nearest neighbor images and return nearest neighbor visual word
  // identifiers for each descriptor.
  void QueryAndFindWordIds(const QueryOptions& options,
                           const Descriptors& descriptors,
                           std::vector<ImageScore>* image_scores,
                           WordIds* word_ids) const {
    THROW_CHECK(prepared_);
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);

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

    const size_t num_images =
        (options.max_num_images >= 0)
            ? std::min<size_t>(image_scores->size(), options.max_num_images)
            : image_scores->size();

    // NOLINTNEXTLINE(bugprone-branch-clone)
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

  // Find the nearest neighbor visual words for the given descriptors.
  WordIds FindWordIds(const Descriptors& descriptors,
                      int num_neighbors,
                      int num_checks,
                      int num_threads) const {
    THROW_CHECK_EQ(descriptors.cols(), kDescDim);
    THROW_CHECK_GT(descriptors.rows(), 0);
    THROW_CHECK_GT(num_neighbors, 0);
    THROW_CHECK_NOTNULL(index_);

    WordIds word_ids(descriptors.rows(), num_neighbors);
    Eigen::RowMajorMatrixXf distances(descriptors.rows(), num_neighbors);

    faiss::IVFSearchParameters search_params;
    search_params.nprobe = num_checks;

#pragma omp parallel num_threads(1)
    {
      omp_set_num_threads(GetEffectiveNumThreads(num_threads));
#ifdef _MSC_VER
      omp_set_nested(1);
#else
      omp_set_max_active_levels(1);
#endif

      index_->search(descriptors.rows(),
                     descriptors.data(),
                     num_neighbors,
                     distances.data(),
                     word_ids.data(),
                     &search_params);
    }

    return word_ids;
  }

  // The search structure on the quantized descriptor space.
  std::unique_ptr<faiss::IndexIVF> index_;
  std::unique_ptr<faiss::Index> quantizer_;

  // The inverted index of the database.
  InvertedIndexType inverted_index_;

  // Identifiers of all indexed images.
  std::unordered_set<int> image_ids_;

  // Whether the index is prepared.
  bool prepared_;
};

}  // namespace

std::unique_ptr<VisualIndex> VisualIndex::Create(int desc_dim,
                                                 int embedding_dim) {
  if (desc_dim == 128 && embedding_dim == 64) {
    return std::make_unique<FaissVisualIndex<128, 64>>();
  } else if (desc_dim == 32 && embedding_dim == 8) {
    return std::make_unique<FaissVisualIndex<32, 8>>();
  } else {
    std::ostringstream error;
    error << "Visual index with descriptor dimension " << desc_dim
          << " and embedding dimension " << embedding_dim
          << " not implemented.";
    throw std::runtime_error(error.str());
  }
}

std::unique_ptr<VisualIndex> VisualIndex::Read(
    const std::string& vocab_tree_path) {
  const std::string resolved_path = MaybeDownloadAndCacheFile(vocab_tree_path);

  std::ifstream file(resolved_path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, resolved_path);

  int file_version = 0;
  file.read(reinterpret_cast<char*>(&file_version), sizeof(int));

  // For a legacy flann-based index, the first int in the file defined the
  // number of visual words. We now encode a file version in the first int.
  // We detect legacy indices based on a file version mismatch, which works
  // as long as we do not increment the file version beyond the count of
  // visual words in a legacy file (which we do not expect to happen).
  THROW_CHECK_EQ(file_version, 1)
      << "Failed to read faiss index. This may be caused by reading a legacy "
         "flann-based index, because COLMAP switched from flann to faiss in "
         "May 2025. If you want to upgrade your existing flann-based index "
         "to faiss, you can check out COLMAP commit "
         "c7a58462b813e406c304a9dafb475b87036924cf and apply the "
         "vocab_tree_upgrader command.";

  int desc_dim = 0;
  file.read(reinterpret_cast<char*>(&desc_dim), sizeof(int));
  int embedding_dim = 0;
  file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int));

  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  visual_index->ReadFromFaiss(resolved_path, file.tellg());

  return visual_index;
}

}  // namespace retrieval
}  // namespace colmap
