// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include <iostream>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "mvs/cuda_utils.h"
#include "mvs/image.h"
#include "mvs/model.h"
#include "mvs/patch_match.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "util/threading.h"

using namespace colmap;
using namespace colmap::mvs;

struct Config {
  PatchMatch::Options options;
  PatchMatch::Problem problem;
  std::string output_prefix;
};

struct Input {
  int image_id = -1;
  std::string depth_map_path;
  std::string normal_map_path;
};

bool ReadConfigs(const PatchMatch::Options default_options,
                 const std::string& config_path, std::vector<Config>* configs,
                 std::vector<mvs::Image>* images,
                 std::vector<DepthMap>* depth_maps,
                 std::vector<NormalMap>* normal_maps) {
  std::string input_path;
  std::string input_type;

  int gpu_id = -1;
  int image_max_size = -1;
  float image_scale_factor = -1.0f;
  std::vector<Input> inputs;

  std::cout << "Reading configuration..." << std::endl;
  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path.c_str(), pt);

    input_path = pt.get<std::string>("input_path");
    input_type = pt.get<std::string>("input_type");

    boost::optional<int> gpu_id_optional = pt.get_optional<int>("gpu_id");
    if (gpu_id_optional) {
      gpu_id = gpu_id_optional.get();
      CHECK_GE(gpu_id, 0);
    }

    boost::optional<int> image_max_size_optional =
        pt.get_optional<int>("image_max_size");
    if (image_max_size_optional) {
      image_max_size = image_max_size_optional.get();
      CHECK_GT(image_max_size, 0);
    }

    boost::optional<float> image_scale_factor_optional =
        pt.get_optional<float>("image_scale_factor");
    if (image_scale_factor_optional) {
      image_scale_factor = image_scale_factor_optional.get();
      CHECK_GT(image_scale_factor, 0.0f);
    }

    const auto input_list = pt.get_child_optional("input_list");
    if (input_list) {
      for (const auto& input_elem : input_list.get()) {
        Input input;
        input.image_id = input_elem.second.get<int>("image_id");
        const auto depth_map_path =
            input_elem.second.get_optional<std::string>("depth_map_path");
        if (depth_map_path) {
          input.depth_map_path = depth_map_path.get();
        }
        const auto normal_map_path =
            input_elem.second.get_optional<std::string>("normal_map_path");
        if (normal_map_path) {
          input.normal_map_path = normal_map_path.get();
        }
        inputs.push_back(input);
      }
    }

    boost::property_tree::ptree output_list = pt.get_child("output_list");
    for (const auto& output : output_list) {
      Config config;

      config.problem.ref_image_id = output.second.get<int>("ref_image_id");
      for (const auto& image_id : output.second.get_child("src_image_ids")) {
        config.problem.src_image_ids.push_back(
            image_id.second.get_value<int>());
      }
      if (config.problem.src_image_ids.empty()) {
        std::cerr << "ERROR: Need at least one source image" << std::endl;
        return false;
      }

      config.output_prefix = output.second.get<std::string>("output_prefix");

      configs->push_back(config);
    }
  } catch (std::exception const& exc) {
    std::cerr << "ERROR: Problem with configuration file - " << exc.what()
              << std::endl;
    return false;
  }

  std::cout << "Selecting GPU... " << std::flush;
  SetBestCudaDevice(gpu_id);

  std::cout << "Reading model..." << std::endl;
  Model model;
  bool extract_principal_point = false;
  if (input_type == "NVM") {
    extract_principal_point = true;
    if (!model.LoadFromNVM(input_path)) {
      return false;
    }
  } else if (input_type == "PMVS") {
    extract_principal_point = true;
    if (!model.LoadFromPMVS(input_path)) {
      return false;
    }
  } else if (input_type == "Middlebury") {
    extract_principal_point = false;
    if (!model.LoadFromMiddleBurry(input_path)) {
      return false;
    }
  } else {
    std::cerr << "ERROR: Unknown input type" << std::endl;
    return false;
  }

  // Set source images.
  std::cout << "Setting up problem..." << std::endl;
  std::vector<std::map<int, int>> shared_points;
  std::set<int> used_image_ids;
  for (auto& config : *configs) {
    used_image_ids.insert(config.problem.ref_image_id);

    if (config.problem.src_image_ids.size() == 1 &&
        config.problem.src_image_ids[0] == -1) {
      // Use all images as source images.

      config.problem.src_image_ids.clear();
      config.problem.src_image_ids.reserve(model.views.size() - 1);
      for (size_t image_id = 0; image_id < model.views.size(); ++image_id) {
        if (static_cast<int>(image_id) != config.problem.ref_image_id) {
          config.problem.src_image_ids.push_back(image_id);
          used_image_ids.insert(image_id);
        }
      }
    } else if (config.problem.src_image_ids.size() == 2 &&
               config.problem.src_image_ids[0] == -2) {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected.

      CHECK_NE(input_type, "Middlebury");

      if (shared_points.empty()) {
        shared_points = model.ComputeSharedPoints();
      }

      const size_t max_num_src_images = config.problem.src_image_ids[1];

      config.problem.src_image_ids.clear();

      const auto& overlapping_images =
          shared_points.at(config.problem.ref_image_id);

      if (max_num_src_images >= overlapping_images.size()) {
        config.problem.src_image_ids.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
          config.problem.src_image_ids.push_back(image.first);
          used_image_ids.insert(image.first);
        }
      } else {
        std::vector<std::pair<int, int>> src_images;
        src_images.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
          src_images.emplace_back(image.first, image.second);
        }

        std::partial_sort(
            src_images.begin(), src_images.begin() + max_num_src_images,
            src_images.end(), [](const std::pair<int, int> image1,
                                 const std::pair<int, int> image2) {
              return image1.second > image2.second;
            });

        config.problem.src_image_ids.reserve(max_num_src_images);
        for (size_t i = 0; i < max_num_src_images; ++i) {
          config.problem.src_image_ids.push_back(src_images[i].first);
          used_image_ids.insert(src_images[i].first);
        }
      }
    } else if (config.problem.src_image_ids.size() == 2 &&
               config.problem.src_image_ids[0] == -3) {
      // Use maximum number of images as source images, ranked by the viewing
      // direction distance between the reference image and all other images.

      const size_t max_num_src_images = config.problem.src_image_ids[1];

      config.problem.src_image_ids.clear();

      const float* ref_viewing_direction =
          images->at(config.problem.ref_image_id).GetViewingDirection();

      std::vector<std::pair<int, float>> src_images;
      src_images.reserve(model.views.size());
      for (size_t image_id = 0; image_id < model.views.size(); ++image_id) {
        if (static_cast<int>(image_id) != config.problem.ref_image_id) {
          const float* src_viewing_direction =
              images->at(image_id).GetViewingDirection();
          const float viewing_direction_dist =
              ref_viewing_direction[0] * src_viewing_direction[0] +
              ref_viewing_direction[1] * src_viewing_direction[1] +
              ref_viewing_direction[2] * src_viewing_direction[2];
          src_images.emplace_back(image_id, viewing_direction_dist);
        }
      }

      std::partial_sort(
          src_images.begin(), src_images.begin() + max_num_src_images,
          src_images.end(), [](const std::pair<int, float> image1,
                               const std::pair<int, float> image2) {
            return image1.second > image2.second;
          });

      config.problem.src_image_ids.reserve(max_num_src_images);
      for (size_t i = 0; i < max_num_src_images; ++i) {
        config.problem.src_image_ids.push_back(src_images[i].first);
        used_image_ids.insert(src_images[i].first);
      }
    } else {
      used_image_ids.insert(config.problem.src_image_ids.begin(),
                            config.problem.src_image_ids.end());
    }
  }

  std::cout << "Reading images..." << std::endl;
  images->resize(model.views.size());
  for (const auto image_id : used_image_ids) {
    const auto& view = model.views.at(image_id);
    auto& image = images->at(image_id);
    image.Load(view.K.data(), view.R.data(), view.T.data(), view.path, false,
               extract_principal_point);
  }

  std::cout << "Computing depth ranges..." << std::endl;
  if (input_type == "NVM" || input_type == "PMVS") {
    const auto& depth_ranges = model.ComputeDepthRanges();
    for (auto& config : *configs) {
      config.options.depth_min =
          depth_ranges.at(config.problem.ref_image_id).first;
      config.options.depth_max =
          depth_ranges.at(config.problem.ref_image_id).second;
    }
  }

  // Downsample images if necessary.
  std::cout << "Resizing data..." << std::endl;
  CHECK(!(image_max_size != -1 && image_scale_factor != -1.0f))
      << "ERROR: Cannot both set `image_max_size` and `image_scale_factor`";
  {
    ThreadPool thread_pool;
    for (size_t image_id = 0; image_id < images->size(); ++image_id) {
      thread_pool.AddTask([&, image_id]() {
        if (image_max_size != -1) {
          images->at(image_id).Downsize(image_max_size, image_max_size);
        } else if (image_scale_factor != -1.0f) {
          images->at(image_id).Rescale(image_scale_factor);
        }
      });
    }
    thread_pool.Wait();
  }

  // Read input data.
  std::cout << "Reading input data..." << std::endl;
  depth_maps->resize(images->size());
  normal_maps->resize(images->size());
  for (const auto& input : inputs) {
    CHECK_GE(input.image_id, 0);
    CHECK_LT(input.image_id, images->size());
    CHECK_LT(input.image_id, images->size());
    if (!input.depth_map_path.empty()) {
      (*depth_maps)[input.image_id].Read(input.depth_map_path);
      if (image_max_size != -1) {
        (*depth_maps)[input.image_id].Downsize(image_max_size, image_max_size);
      } else if (image_scale_factor != -1.0f) {
        (*depth_maps)[input.image_id].Rescale(image_scale_factor);
      }
    }
    if (!input.normal_map_path.empty()) {
      (*normal_maps)[input.image_id].Read(input.normal_map_path);
      if (image_max_size != -1) {
        (*normal_maps)[input.image_id].Downsize(image_max_size, image_max_size);
      } else if (image_scale_factor != -1.0f) {
        (*normal_maps)[input.image_id].Rescale(image_scale_factor);
      }
    }
  }

  return true;
}

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string config_path;

  OptionManager options;
  options.AddDenseMapperOptions();
  options.AddRequiredOption("config_path", &config_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  std::vector<Config> configs;
  std::vector<mvs::Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  if (!ReadConfigs(options.dense_mapper_options->patch_match, config_path,
                   &configs, &images, &depth_maps, &normal_maps)) {
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < configs.size(); ++i) {
    auto config = configs[i];
    config.problem.images = &images;
    config.problem.depth_maps = &depth_maps;
    config.problem.normal_maps = &normal_maps;

    PrintHeading1(
        StringPrintf("Processing view %d / %d", i + 1, configs.size()));

    config.options.Print();
    std::cout << std::endl;
    config.problem.Print();
    std::cout << std::endl;

    PatchMatch patch_match(config.options, config.problem);
    patch_match.Run();

    if (config.problem.src_image_ids.empty()) {
      std::cout << "WARNING: No source images defined - skipping image."
                << std::endl;
      continue;
    }

    std::cout << std::endl
              << "Writing output: " << config.output_prefix << std::endl;
    patch_match.GetDepthMap().Write(config.output_prefix + ".depth_map.txt");
    patch_match.GetNormalMap().Write(config.output_prefix + ".normal_map.txt");
    WriteBinaryBlob(config.output_prefix + ".consistency_graph.bin",
                    patch_match.GetConsistentImageIds());
  }

  return EXIT_SUCCESS;
}
