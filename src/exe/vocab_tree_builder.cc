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

#include "base/database.h"
#include "optim/random_sampler.h"
#include "retrieval/visual_index.h"
#include "util/logging.h"
#include "util/option_manager.h"

using namespace colmap;

// Loads descriptors for training from the database. Loads all descriptors from
// the database if max_num_images < 0, otherwise the descriptors of a random
// subset of images are selected.
FeatureDescriptors LoadDescriptors(const std::string& database_path,
                                   const int max_num_images) {
  Database database(database_path);
  DatabaseTransaction database_transaction(&database);

  const std::vector<Image> images = database.ReadAllImages();

  FeatureDescriptors descriptors;

  std::vector<size_t> image_ids;
  size_t num_descriptors = 0;
  if (max_num_images < 0) {
    // All images in the database.
    image_ids.resize(images.size());
    std::iota(image_ids.begin(), image_ids.end(), 0);
    num_descriptors = database.NumDescriptors();
  } else {
    // Random subset of images in the database.
    CHECK_LE(max_num_images, images.size());
    RandomSampler random_sampler(max_num_images);
    random_sampler.Initialize(images.size());
    image_ids = random_sampler.Sample();
    for (const auto image_id : image_ids) {
      const auto& image = images.at(image_id);
      num_descriptors += database.NumDescriptorsForImage(image.ImageId());
    }
  }

  descriptors.resize(num_descriptors, 128);

  size_t descriptor_row = 0;
  for (const auto image_id : image_ids) {
    const auto& image = images.at(image_id);
    const FeatureDescriptors image_descriptors =
        database.ReadDescriptors(image.ImageId());
    descriptors.block(descriptor_row, 0, image_descriptors.rows(), 128) =
        image_descriptors;
    descriptor_row += image_descriptors.rows();
  }

  CHECK_EQ(descriptor_row, num_descriptors);

  return descriptors;
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string vocab_tree_path;
  retrieval::VisualIndex<>::BuildOptions build_options;
  int max_num_images = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("vocab_tree_path", &vocab_tree_path);
  options.AddDefaultOption("num_visual_words", &build_options.num_visual_words);
  options.AddDefaultOption("branching", &build_options.branching);
  options.AddDefaultOption("num_iterations", &build_options.num_iterations);
  options.AddDefaultOption("max_num_images", &max_num_images);
  options.Parse(argc, argv);

  retrieval::VisualIndex<> visual_index;

  std::cout << "Loading descriptors..." << std::endl;
  const auto descriptors =
      LoadDescriptors(*options.database_path, max_num_images);
  std::cout << "  => Loaded a total of " << descriptors.rows() << " descriptors"
            << std::endl;

  std::cout << "Building index for visual words..." << std::endl;
  visual_index.Build(build_options, descriptors);
  std::cout << " => Quantized descriptor space using "
            << visual_index.NumVisualWords() << " visual words" << std::endl;

  std::cout << "Saving index to file..." << std::endl;
  visual_index.Write(vocab_tree_path);

  return EXIT_SUCCESS;
}
