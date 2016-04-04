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

#include "base/vocabulary_tree.h"

#include "util/logging.h"
#include "util/timer.h"

namespace colmap {

VocabularyTree::VocabularyTree() : prepared_(false) {}

VocabularyTree::VocabularyTree(const VocabLib::VocabTree& vocab_tree)
    : prepared_(false), vocab_tree_(vocab_tree) {}

void VocabularyTree::Clear() {
  vocab_tree_.ClearDatabase();
  prepared_ = false;
}

void VocabularyTree::Prepare() {
  prepared_ = true;

  vocab_tree_.SetDistanceType(VocabLib::DistanceMin);
  vocab_tree_.SetInteriorNodeWeight(0.0);
  vocab_tree_.SetConstantLeafWeights();

  vocab_tree_.ComputeTFIDFWeights(static_cast<int>(objects_.size()));
  vocab_tree_.NormalizeDatabase(0, static_cast<int>(objects_.size()));
}

void VocabularyTree::Index(const int object_id,
                           const FeatureDescriptors& descriptors) {
  prepared_ = false;
  objects_.push_back(object_id);
  vocab_tree_.AddImageToDatabase(static_cast<int>(objects_.size() - 1),
                                 static_cast<int>(descriptors.rows()),
                                 const_cast<uint8_t*>(descriptors.data()));
}

std::vector<std::pair<int, float>> VocabularyTree::Retrieve(
    FeatureDescriptors& descriptors, const size_t max_num_objects) {
  CHECK(prepared_) << "Vocabulary tree must be prepared before retrieval";

  scores_.resize(objects_.size(), 0);
  vocab_tree_.ScoreQueryKeys(static_cast<int>(descriptors.rows()), true,
                             descriptors.data(), scores_.data());

  std::vector<std::pair<int, float>> valid_scores;
  for (size_t i = 0; i < objects_.size(); ++i) {
    const auto score = scores_[i];
    if (score > 0) {
      valid_scores.emplace_back(objects_[i], score);
    }
  }

  const size_t num_eff_objects = std::min(valid_scores.size(), max_num_objects);

  std::partial_sort(
      valid_scores.begin(), valid_scores.begin() + num_eff_objects,
      valid_scores.end(),
      [](const std::pair<int, float> i1, const std::pair<int, float> i2) {
        return i1.second > i2.second;
      });

  std::vector<std::pair<int, float>> retrievals;
  retrievals.reserve(num_eff_objects);

  for (size_t i = 0; i < num_eff_objects; ++i) {
    retrievals.emplace_back(valid_scores[i].first, valid_scores[i].second);
  }

  return retrievals;
}

size_t VocabularyTree::FindVisualWord(const FeatureDescriptors& descriptor) {
  CHECK(prepared_) << "Vocabulary tree must be prepared before retrieval";
  CHECK_EQ(descriptor.rows(), 1);
  return vocab_tree_.PushAndScoreFeature(
      const_cast<uint8_t*>(descriptor.data()), 0, false);
}

VocabularyTree VocabularyTree::Build(const Database& database, const int depth,
                                     const int branching_factor,
                                     const int restarts) {
  Timer timer;
  timer.Start();

  database.BeginTransaction();

  const std::vector<Image> images = database.ReadAllImages();

  // Temporary storage for all descriptors in database.
  std::vector<FeatureDescriptors> descriptors;
  descriptors.reserve(images.size());

  // Pointers to rows (individual features) in descriptors.
  std::vector<uint8_t*> features;
  features.reserve(database.NumDescriptors());

  const int kDescriptorDim = 128;

  int num_features = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    const Image& image = images[i];
    std::cout << "Loading features for image " << image.Name() << " [" << i
              << "/" << images.size() << "]" << std::endl;

    const FeatureDescriptors image_descriptors =
        database.ReadDescriptors(image.ImageId());
    if (image_descriptors.rows() == 0) {
      continue;
    }

    // Make sure that all descriptors have the same dimensionality.
    CHECK_EQ(image_descriptors.cols(), kDescriptorDim);

    descriptors.push_back(image_descriptors);
    FeatureDescriptors& descs = descriptors.back();

    for (FeatureDescriptors::Index j = 0; j < descs.rows(); ++j) {
      // FeatureDescriptors is row-major.
      features.push_back(&descs(j, 0));
    }

    num_features += descs.rows();
  }

  database.EndTransaction();

  VocabLib::VocabTree vocab_tree;
  vocab_tree.Build(num_features, kDescriptorDim, depth, branching_factor,
                   restarts, features.data());

  timer.PrintMinutes();

  return VocabularyTree(vocab_tree);
}

void VocabularyTree::Read(const std::string& path) {
  vocab_tree_.Read(path.c_str());

  vocab_tree_.SetDistanceType(VocabLib::DistanceMin);
  vocab_tree_.SetInteriorNodeWeight(0.0);
  vocab_tree_.SetConstantLeafWeights();

  Clear();
}

void VocabularyTree::Write(const std::string& path) const {
  vocab_tree_.Write(path.c_str());
}

}  //  namespace colmap
