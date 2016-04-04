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

#ifndef COLMAP_SRC_BASE_VOCABULARY_TREE_H_
#define COLMAP_SRC_BASE_VOCABULARY_TREE_H_

#include "base/database.h"
#include "ext/VocabLib/VocabTree.h"

namespace colmap {

// TF-IDF weighted vocabulary tree class.
class VocabularyTree {
 public:
  VocabularyTree();
  VocabularyTree(const VocabLib::VocabTree& vocab_tree);

  // Clear all indexed objects in the database.
  void Clear();

  // Normalize descriptors and compute TF-IDF weights.
  void Prepare();

  // Index object in vocabulary tree.
  //
  // @param id                  Unique identifier, returned by `retrieve`.
  // @param descriptors         Descriptors of indexed object.
  void Index(const int object_id, const FeatureDescriptors& descriptors);

  // Retrieve nearest neighbors for given query object.
  //
  // @param descriptors         Descriptors of query object.
  // @param max_num_objects     Maximum number of nearest neighbors to return.
  //
  // @return                    List of ordered nearest neihbor objects in
  //                            descending order according to similarity.
  std::vector<std::pair<int, float>> Retrieve(FeatureDescriptors& descriptors,
                                              const size_t max_num_objects);

  // Find identifier of visual word for given descriptor vector.
  //
  // @param descriptor          Single feature descriptor.
  //
  // @return                    Identifier of visual word.
  size_t FindVisualWord(const FeatureDescriptors& descriptor);

  // Build vocabulary tree from all images in given database.
  //
  // @param database            Database in which the images are stored.
  // @param depth               Depth (number of levels with out root) of tree.
  // @param branching_factor    Branching factor (leaves per node) of tree.
  // @param restarts            Number of random restarts for k-means.
  //
  // @return                    Trained vocabulary tree.
  static VocabularyTree Build(const Database& database, const int depth = 5,
                              const int branching_factor = 10,
                              const int restarts = 5);

  // Read and write vocabulary tree from disk.
  //
  // @param                     Path to binary file on disk.
  void Read(const std::string& path);
  void Write(const std::string& path) const;

 private:
  bool prepared_;
  VocabLib::VocabTree vocab_tree_;
  std::vector<int> objects_;
  std::vector<float> scores_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_VOCABULARY_TREE_H_
