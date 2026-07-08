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

#include "colmap/retrieval/global_descriptor_index.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <fstream>
#include <unordered_map>

#include <faiss/IndexFlat.h>

namespace colmap {
namespace retrieval {

namespace {

// Magic number for file format identification: "GDSK"
constexpr uint32_t kFileMagic = 0x4744534B;
constexpr uint32_t kFileVersion = 1;

}  // namespace

// FAISS index deleter — defined here to keep faiss headers out of the .h file.
void GlobalDescriptorIndex::FaissIndexDeleter::operator()(void* ptr) const {
  delete static_cast<faiss::IndexFlatIP*>(ptr);
}

GlobalDescriptorIndex::GlobalDescriptorIndex(const int descriptor_dim)
    : descriptor_dim_(descriptor_dim) {
  THROW_CHECK_GT(descriptor_dim_, 0);
  // Reserve space for descriptors — the matrix is grown row-by-row.
  descriptors_.resize(0, descriptor_dim_);
}

size_t GlobalDescriptorIndex::NumImages() const { return image_ids_.size(); }

void GlobalDescriptorIndex::Add(const image_t image_id,
                                const std::vector<float>& descriptor) {
  THROW_CHECK_EQ(static_cast<int>(descriptor.size()), descriptor_dim_)
      << "Descriptor dimension mismatch: expected " << descriptor_dim_
      << " but got " << descriptor.size();
  THROW_CHECK(!prepared_)
      << "Cannot add images after Prepare() has been called.";

  // Check for duplicate image_id.
  if (image_id_to_idx_.count(image_id) > 0) {
    LOG(WARNING) << "Image " << image_id
                 << " already indexed, skipping duplicate.";
    return;
  }

  const size_t row = static_cast<size_t>(descriptors_.rows());

  // Grow the descriptor matrix by one row.
  descriptors_.conservativeResize(row + 1, descriptor_dim_);
  Eigen::Map<Eigen::RowVectorXf> row_map(
      const_cast<float*>(descriptors_.row(row).data()), descriptor_dim_);
  for (int i = 0; i < descriptor_dim_; ++i) {
    row_map(i) = descriptor[i];
  }

  image_ids_.push_back(image_id);
  image_id_to_idx_[image_id] = row;
}

void GlobalDescriptorIndex::Prepare() {
  THROW_CHECK(!prepared_) << "Prepare() already called.";
  THROW_CHECK_GT(image_ids_.size(), 0) << "No images added to the index.";

  // Build FAISS IndexFlatIP for cosine similarity search.
  // Since descriptors are L2-normalized, inner product = cosine similarity.
  auto* index = new faiss::IndexFlatIP(descriptor_dim_);
  index->add(static_cast<faiss::idx_t>(image_ids_.size()),
             descriptors_.data());

  faiss_index_.reset(index);
  prepared_ = true;

  LOG(INFO) << "Built global descriptor index with " << image_ids_.size()
            << " images, descriptor dimension = " << descriptor_dim_;
}

bool GlobalDescriptorIndex::IsPrepared() const { return prepared_; }

void GlobalDescriptorIndex::Query(
    const QueryOptions& options,
    const image_t query_image_id,
    std::vector<ImageScore>* image_scores) const {
  THROW_CHECK(prepared_)
      << "Index not prepared. Call Prepare() before Query().";
  THROW_CHECK_NOTNULL(image_scores);
  image_scores->clear();

  // Look up the query descriptor.
  const auto it = image_id_to_idx_.find(query_image_id);
  THROW_CHECK(it != image_id_to_idx_.end())
      << "Query image " << query_image_id << " not found in index.";

  const size_t query_idx = it->second;

  // Retrieve one more than requested to account for potential self-match.
  const int k = std::min(static_cast<int>(options.max_num_images) + 1,
                         static_cast<int>(image_ids_.size()));

  // FAISS search: the single query vector.
  const auto* index = static_cast<faiss::IndexFlatIP*>(faiss_index_.get());
  std::vector<faiss::idx_t> labels(k);
  std::vector<float> distances(k);

  index->search(1,
                descriptors_.row(query_idx).data(),
                k,
                distances.data(),
                labels.data());

  // Convert results, skipping self-matches.
  image_scores->reserve(options.max_num_images);
  for (int i = 0; i < k; ++i) {
    const int idx = static_cast<int>(labels[i]);
    // Skip self-match (distance 1.0 for identical L2-normalized vectors).
    if (static_cast<size_t>(idx) == query_idx) {
      continue;
    }
    if (idx < 0 || static_cast<size_t>(idx) >= image_ids_.size()) {
      continue;
    }
    ImageScore score;
    score.image_id = image_ids_[idx];
    score.score = distances[i];  // Cosine similarity (inner product)
    image_scores->push_back(score);

    if (static_cast<int>(image_scores->size()) >= options.max_num_images) {
      break;
    }
  }
}

void GlobalDescriptorIndex::Write(const std::filesystem::path& path) const {
  THROW_CHECK_GT(image_ids_.size(), 0) << "No images to write.";

  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  // Header: magic, version, num_images, descriptor_dim.
  WriteBinaryLittleEndian<uint32_t>(&file, kFileMagic);
  WriteBinaryLittleEndian<uint32_t>(&file, kFileVersion);
  WriteBinaryLittleEndian<uint32_t>(
      &file, static_cast<uint32_t>(image_ids_.size()));
  WriteBinaryLittleEndian<uint32_t>(
      &file, static_cast<uint32_t>(descriptor_dim_));

  // Image IDs.
  for (const image_t id : image_ids_) {
    WriteBinaryLittleEndian<int64_t>(&file, static_cast<int64_t>(id));
  }

  // Descriptors (row-major float32).
  for (int i = 0; i < descriptors_.rows(); ++i) {
    for (int j = 0; j < descriptors_.cols(); ++j) {
      WriteBinaryLittleEndian<float>(&file, descriptors_(i, j));
    }
  }

  LOG(INFO) << "Saved global descriptor index (" << image_ids_.size()
            << " images, dim=" << descriptor_dim_ << ") to " << path;
}

void GlobalDescriptorIndex::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  // Header.
  const uint32_t magic = ReadBinaryLittleEndian<uint32_t>(&file);
  THROW_CHECK_EQ(magic, kFileMagic)
      << "Invalid global descriptor index file: bad magic number.";
  const uint32_t version = ReadBinaryLittleEndian<uint32_t>(&file);
  THROW_CHECK_EQ(version, kFileVersion)
      << "Unsupported global descriptor index file version: " << version;
  const uint32_t num_images = ReadBinaryLittleEndian<uint32_t>(&file);
  const uint32_t desc_dim = ReadBinaryLittleEndian<uint32_t>(&file);

  descriptor_dim_ = static_cast<int>(desc_dim);

  // Image IDs.
  image_ids_.clear();
  image_ids_.reserve(num_images);
  for (uint32_t i = 0; i < num_images; ++i) {
    const int64_t id = ReadBinaryLittleEndian<int64_t>(&file);
    image_ids_.push_back(static_cast<image_t>(id));
  }

  // Descriptors.
  descriptors_.resize(static_cast<int>(num_images), descriptor_dim_);
  for (int i = 0; i < descriptors_.rows(); ++i) {
    for (int j = 0; j < descriptors_.cols(); ++j) {
      descriptors_(i, j) = ReadBinaryLittleEndian<float>(&file);
    }
  }

  // Rebuild mapping.
  image_id_to_idx_.clear();
  for (size_t i = 0; i < image_ids_.size(); ++i) {
    image_id_to_idx_[image_ids_[i]] = i;
  }

  // Build FAISS index.
  auto* index = new faiss::IndexFlatIP(descriptor_dim_);
  index->add(static_cast<faiss::idx_t>(image_ids_.size()),
             descriptors_.data());
  faiss_index_.reset(index);
  prepared_ = true;

  LOG(INFO) << "Loaded global descriptor index (" << image_ids_.size()
            << " images, dim=" << descriptor_dim_ << ") from " << path;
}

}  // namespace retrieval
}  // namespace colmap
