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

#include "mvs/image.h"
#include "mvs/model.h"
#include "mvs/patch_match.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "util/threading.h"

using namespace colmap;
using namespace colmap::mvs;

struct Input {
  int image_id = -1;
  std::string depth_map_path;
  std::string normal_map_path;
};

struct Output {
  PatchMatch::Options options;
  PatchMatch::Problem problem;
  std::string output_prefix;
};

void ReadConfig(const PatchMatch::Options default_options,
                const std::string& config_path, std::vector<Input>* inputs,
                std::vector<Output>* outputs, int* image_max_size,
                float* image_scale_factor) {
  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path.c_str(), pt);

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
        inputs->push_back(input);
      }
    }

    boost::property_tree::ptree output_list = pt.get_child("output_list");
    for (const auto& output_list_el : output_list) {
      Output output;

      output.options = default_options;

      output.problem.ref_image_id =
          output_list_el.second.get<int>("ref_image_id");
      for (const auto& image_id :
           output_list_el.second.get_child("src_image_ids")) {
        output.problem.src_image_ids.push_back(
            image_id.second.get_value<int>());
      }
      CHECK(!output.problem.src_image_ids.empty())
          << "ERROR: Need at least one source image.";

      output.output_prefix =
          output_list_el.second.get<std::string>("output_prefix");

      outputs->push_back(output);
    }
  } catch (std::exception const& exc) {
    LOG(ERROR) << "ERROR: Problem with configuration file - " << exc.what();
  }
}

void ReadModel(const std::string& input_path, const std::string& input_type,
               Model* model, bool* extract_principal_point) {
  if (input_type == "COLMAP") {
    *extract_principal_point = false;
    CHECK(model->LoadFromCOLMAP(input_path));
  } else if (input_type == "NVM") {
    *extract_principal_point = true;
    CHECK(model->LoadFromNVM(input_path));
  } else if (input_type == "PMVS") {
    *extract_principal_point = true;
    CHECK(model->LoadFromPMVS(input_path));
  } else if (input_type == "Middlebury") {
    *extract_principal_point = false;
    CHECK(model->LoadFromMiddleBurry(input_path));
  } else {
    LOG(FATAL) << "Unknown input type.";
  }
}

void SetupProblem(const std::string& input_type, const Model& model,
                  std::set<int>* used_image_ids, std::vector<Output>* outputs) {
  std::vector<std::map<int, int>> shared_points;

  for (auto& output : *outputs) {
    used_image_ids->insert(output.problem.ref_image_id);

    if (output.problem.src_image_ids.size() == 1 &&
        output.problem.src_image_ids[0] == -1) {
      // Use all images as source images.

      output.problem.src_image_ids.clear();
      output.problem.src_image_ids.reserve(model.views.size() - 1);
      for (size_t image_id = 0; image_id < model.views.size(); ++image_id) {
        if (static_cast<int>(image_id) != output.problem.ref_image_id) {
          output.problem.src_image_ids.push_back(image_id);
          used_image_ids->insert(image_id);
        }
      }
    } else if (output.problem.src_image_ids.size() == 2 &&
               output.problem.src_image_ids[0] == -2) {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected.

      CHECK_NE(input_type, "Middlebury");

      if (shared_points.empty()) {
        shared_points = model.ComputeSharedPoints();
      }

      const size_t max_num_src_images = output.problem.src_image_ids[1];

      output.problem.src_image_ids.clear();

      const auto& overlapping_images =
          shared_points.at(output.problem.ref_image_id);

      if (max_num_src_images >= overlapping_images.size()) {
        output.problem.src_image_ids.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
          output.problem.src_image_ids.push_back(image.first);
          used_image_ids->insert(image.first);
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

        output.problem.src_image_ids.reserve(max_num_src_images);
        for (size_t i = 0; i < max_num_src_images; ++i) {
          output.problem.src_image_ids.push_back(src_images[i].first);
          used_image_ids->insert(src_images[i].first);
        }
      }
    }
  }
}

bool ReadInputsAndOutputs(
    const PatchMatch::Options default_options, const std::string& input_path,
    const std::string& input_type, const std::string& config_path,
    std::vector<mvs::Image>* images, std::vector<DepthMap>* depth_maps,
    std::vector<NormalMap>* normal_maps, std::vector<Output>* outputs) {
  std::cout << "Reading configuration..." << std::endl;
  std::vector<Input> inputs;
  int image_max_size;
  float image_scale_factor;
  ReadConfig(default_options, config_path, &inputs, outputs, &image_max_size,
             &image_scale_factor);

  std::cout << "Reading model..." << std::endl;
  Model model;
  bool extract_principal_point;
  ReadModel(input_path, input_type, &model, &extract_principal_point);

  std::cout << "Setting up problem..." << std::endl;
  std::set<int> used_image_ids;
  SetupProblem(input_type, model, &used_image_ids, outputs);

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
    for (auto& output : *outputs) {
      output.options.depth_min =
          depth_ranges.at(output.problem.ref_image_id).first;
      output.options.depth_max =
          depth_ranges.at(output.problem.ref_image_id).second;
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

  std::string input_path;
  std::string input_type;
  std::string config_path;

  OptionManager options;
  options.AddDenseMapperOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("input_type", &input_type);
  options.AddRequiredOption("config_path", &config_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (input_type != "COLMAP" && input_type != "NVM" && input_type != "PMVS" &&
      input_type != "Middlebury") {
    std::cout << "ERROR: Invalid `input_type` - supported values are "
                 "{'COLMAP', 'NVM', PMVS', 'Middlebury'}."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<mvs::Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  std::vector<Output> outputs;
  if (!ReadInputsAndOutputs(options.dense_mapper_options->patch_match,
                            input_path, input_type, config_path, &images,
                            &depth_maps, &normal_maps, &outputs)) {
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    PrintHeading1(
        StringPrintf("Processing output %d / %d", i + 1, outputs.size()));

    auto& output = outputs[i];
    output.problem.images = &images;
    output.problem.depth_maps = &depth_maps;
    output.problem.normal_maps = &normal_maps;

    output.options.Print();
    output.problem.Print();

    PatchMatch patch_match(output.options, output.problem);
    patch_match.Run();

    if (output.problem.src_image_ids.empty()) {
      std::cout << "WARNING: No source images defined - skipping image."
                << std::endl;
      continue;
    }

    std::cout << std::endl
              << "Writing output: " << output.output_prefix << std::endl;
    patch_match.GetDepthMap().Write(output.output_prefix + ".depth_map.txt");
    patch_match.GetNormalMap().Write(output.output_prefix + ".normal_map.txt");
    WriteBinaryBlob(output.output_prefix + ".consistency_graph.bin",
                    patch_match.GetConsistentImageIds());
  }

  return EXIT_SUCCESS;
}
