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

#include "mvs/depth_map.h"
#include "mvs/fusion.h"
#include "mvs/image.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "util/math.h"
#include "util/misc.h"

using namespace colmap;
using namespace colmap::mvs;

struct Input {
  std::string depth_map_path;
  std::string normal_map_path;
  std::string consistency_graph_path;
};

bool ReadInput(const std::string& path, std::vector<uint8_t>* used_image_mask,
               std::vector<Image>* images, std::vector<DepthMap>* depth_maps,
               std::vector<NormalMap>* normal_maps,
               std::vector<std::vector<int>>* consistency_graph,
               std::string* output_path, StereoFusionOptions* options) {
  std::string input_path;
  std::string input_type;

  std::unordered_map<int, Input> inputs;
  std::set<int> used_image_ids;

  std::cout << "Reading configuration..." << std::endl;
  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(path.c_str(), pt);

    input_path = pt.get<std::string>("input_path");
    input_type = pt.get<std::string>("input_type");
    *output_path = pt.get<std::string>("output_path");

    boost::optional<int> min_num_pixels =
        pt.get_optional<int>("min_num_pixels");
    if (min_num_pixels) {
      options->min_num_pixels = min_num_pixels.get();
    }

    boost::optional<int> max_num_pixels =
        pt.get_optional<int>("max_num_pixels");
    if (max_num_pixels) {
      options->max_num_pixels = max_num_pixels.get();
    }

    boost::optional<int> max_traversal_depth =
        pt.get_optional<int>("max_traversal_depth");
    if (max_traversal_depth) {
      options->max_traversal_depth = max_traversal_depth.get();
    }

    boost::optional<float> max_reproj_error =
        pt.get_optional<float>("max_reproj_error");
    if (max_reproj_error) {
      options->max_reproj_error = max_reproj_error.get();
    }

    boost::optional<float> max_depth_error =
        pt.get_optional<float>("max_depth_error");
    if (max_depth_error) {
      options->max_depth_error = max_depth_error.get();
    }

    boost::optional<float> max_normal_error =
        pt.get_optional<float>("max_normal_error");
    if (max_normal_error) {
      options->max_normal_error = DegToRad(max_normal_error.get());
    }

    for (const auto& input_elem : pt.get_child("input_list")) {
      Input input;
      const int image_id = input_elem.second.get<int>("image_id");
      used_image_ids.insert(image_id);
      input.depth_map_path =
          input_elem.second.get<std::string>("depth_map_path");
      input.normal_map_path =
          input_elem.second.get<std::string>("normal_map_path");
      const auto consistency_graph_path =
          input_elem.second.get_optional<std::string>("consistency_graph_path");
      if (consistency_graph_path) {
        input.consistency_graph_path = consistency_graph_path.get();
      }
      inputs.emplace(image_id, input);
    }
  } catch (std::exception const& exc) {
    std::cerr << "Error: Problem with configuration file - " << exc.what()
              << std::endl;
    return false;
  }

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
    std::cerr << "Error: Unknown input type" << std::endl;
    return false;
  }

  std::cout << "Reading images, depth maps, and normal maps..." << std::endl;
  images->resize(model.views.size());
  used_image_mask->resize(images->size(), false);
  depth_maps->resize(images->size());
  normal_maps->resize(images->size());
  consistency_graph->resize(images->size());
  for (const auto& elem : inputs) {
    const int image_id = elem.first;
    const auto& input = elem.second;
    const auto& view = model.views.at(image_id);

    auto& image = images->at(image_id);
    image.Load(view.K.data(), view.R.data(), view.T.data(), view.path, true,
               extract_principal_point);

    used_image_mask->at(image_id) = true;

    auto& depth_map = depth_maps->at(image_id);
    auto& normal_map = normal_maps->at(image_id);
    depth_map.Read(input.depth_map_path);
    normal_map.Read(input.normal_map_path);

    CHECK_EQ(depth_map.GetWidth(), normal_map.GetWidth());
    CHECK_EQ(depth_map.GetHeight(), normal_map.GetHeight());

    if (depth_map.GetWidth() != image.GetWidth() ||
        depth_map.GetHeight() != image.GetHeight()) {
      image.Rescale(
          depth_map.GetWidth() / static_cast<float>(image.GetWidth()),
          depth_map.GetHeight() / static_cast<float>(image.GetHeight()));
      CHECK_EQ(image.GetWidth(), depth_map.GetWidth());
      CHECK_EQ(image.GetHeight(), depth_map.GetHeight());
    }

    if (!input.consistency_graph_path.empty()) {
      ReadBinaryBlob<int>(input.consistency_graph_path,
                          &consistency_graph->at(image_id));
    }
  }

  return true;
}

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  if (argc < 2) {
    std::cout << "Error: Configuration file not specified, "
                 "call as "
              << argv[0] << " <json-config-file-path>" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<uint8_t> used_image_mask;
  std::vector<Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  std::vector<std::vector<int>> consistency_graph;
  std::string output_path;
  StereoFusionOptions options;
  if (!ReadInput(argv[1], &used_image_mask, &images, &depth_maps, &normal_maps,
                 &consistency_graph, &output_path, &options)) {
    exit(EXIT_FAILURE);
  }

  std::cout << std::endl;
  options.Print();
  std::cout << std::endl;

  const auto points = StereoFusion(options, used_image_mask, images, depth_maps,
                                   normal_maps, consistency_graph);

  std::cout << "Number of fused points " << points.size() << std::endl;

  std::cout << "Writing output" << std::endl;
  WritePlyBinary(output_path, points);

  return EXIT_SUCCESS;
}
