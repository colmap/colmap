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

#include "base/feature.h"
#include "retrieval/visual_index.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

std::vector<Image> ReadImageList(const std::string& path, Database* database) {
  std::vector<Image> images;
  if (path.empty()) {
    images.reserve(database->NumImages());
    for (const auto& image : database->ReadAllImages()) {
      images.push_back(image);
    }
  } else {
    DatabaseTransaction database_transaction(database);
    const auto image_names = ReadTextFileLines(path);
    images.reserve(image_names.size());
    for (const auto& image_name : image_names) {
      const auto image = database->ReadImageWithName(image_name);
      CHECK_NE(image.ImageId(), kInvalidImageId);
      images.push_back(image);
    }
  }
  return images;
}

void IndexImagesInVisualIndex(const int max_num_features,
                              const std::vector<Image>& images,
                              Database* database,
                              retrieval::VisualIndex<>* visual_index) {
  DatabaseTransaction database_transaction(database);

  for (size_t i = 0; i < images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Indexing image [%d/%d]", i + 1, images.size())
              << std::flush;

    auto keypoints = database->ReadKeypoints(images[i].ImageId());
    auto descriptors = database->ReadDescriptors(images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    visual_index->Add(retrieval::VisualIndex<>::IndexOptions(),
                      images[i].ImageId(), keypoints, descriptors);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  // Compute the TF-IDF weights, etc.
  visual_index->Prepare();
}

void QueryImagesInVisualIndex(const int max_num_features,
                              const std::vector<Image>& database_images,
                              const std::vector<Image>& query_images,
                              const int num_images, const int num_verifications,
                              Database* database,
                              retrieval::VisualIndex<>* visual_index) {
  DatabaseTransaction database_transaction(database);

  retrieval::VisualIndex<>::QueryOptions query_options;
  query_options.max_num_images = num_images;
  query_options.max_num_verifications = num_verifications;

  std::unordered_map<image_t, const Image*> image_id_to_image;
  image_id_to_image.reserve(database_images.size());
  for (const auto& image : database_images) {
    image_id_to_image.emplace(image.ImageId(), &image);
  }

  for (size_t i = 0; i < query_images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Querying for image %s [%d/%d]",
                              query_images[i].Name().c_str(), i + 1,
                              query_images.size())
              << std::flush;

    auto keypoints = database->ReadKeypoints(query_images[i].ImageId());
    auto descriptors = database->ReadDescriptors(query_images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    std::vector<retrieval::ImageScore> image_scores;
    visual_index->QueryWithVerification(query_options, keypoints, descriptors,
                                        &image_scores);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
    for (const auto& image_score : image_scores) {
      const auto& image = *image_id_to_image.at(image_score.image_id);
      std::cout << StringPrintf("  image_id=%d, image_name=%s, score=%f",
                                image_score.image_id, image.Name().c_str(),
                                image_score.score)
                << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string vocab_tree_path;
  std::string database_image_list_path;
  std::string query_image_list_path;
  int num_images = -1;
  int num_verifications = 0;
  int max_num_features = -1;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("vocab_tree_path", &vocab_tree_path);
  options.AddDefaultOption("database_image_list_path",
                           &database_image_list_path);
  options.AddDefaultOption("query_image_list_path", &query_image_list_path);
  options.AddDefaultOption("num_images", &num_images);
  options.AddDefaultOption("num_verifications", &num_verifications);
  options.AddDefaultOption("max_num_features", &max_num_features);
  options.Parse(argc, argv);

  retrieval::VisualIndex<> visual_index;
  visual_index.Read(vocab_tree_path);

  Database database(*options.database_path);

  const auto database_images =
      ReadImageList(database_image_list_path, &database);
  const auto query_images = ReadImageList(query_image_list_path, &database);

  IndexImagesInVisualIndex(max_num_features, database_images, &database,
                           &visual_index);
  QueryImagesInVisualIndex(max_num_features, database_images, query_images,
                           num_images, num_verifications, &database,
                           &visual_index);

  return EXIT_SUCCESS;
}
