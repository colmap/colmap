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
#include "util/option_manager.h"

using namespace colmap;

bool ReadConfig(const std::string& workspace_path,
                const std::string& workspace_format,
                const std::string& input_type, mvs::Model* model,
                std::vector<uint8_t>* used_image_mask) {
  std::cout << "Reading model..." << std::endl;
  model->Read(workspace_path, workspace_format);
  used_image_mask->resize(model->images.size(), false);

  std::cout << "Reading configuration..." << std::endl;

  std::ifstream file(JoinPaths(workspace_path, "dense/fusion.cfg"));
  CHECK(file.is_open());

  std::string line;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    const std::string image_name = line;
    const int image_id = model->GetImageId(image_name);

    used_image_mask->at(image_id) = true;

    auto& image = model->images.at(image_id);
    auto& depth_map = model->depth_maps.at(image_id);
    auto& normal_map = model->normal_maps.at(image_id);

    const std::string file_name =
        StringPrintf("%s.%s.bin", image_name.c_str(), input_type.c_str());
    depth_map.Read(JoinPaths(workspace_path, "dense/depth_maps", file_name));
    normal_map.Read(JoinPaths(workspace_path, "dense/normal_maps", file_name));
    ReadBinaryBlob<int>(
        JoinPaths(workspace_path, "dense/consistency_graphs", file_name),
        &model->consistency_graph.at(image_id));

    const bool kReadImageAsRGB = true;
    image.Read(kReadImageAsRGB);

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
  }

  return true;
}

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string workspace_path;
  std::string input_type;
  std::string workspace_format;
  std::string output_path;
  std::string config_path;

  OptionManager options;
  options.AddDenseMapperOptions();
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddRequiredOption("workspace_format", &workspace_format);
  options.AddRequiredOption("input_type", &input_type);
  options.AddRequiredOption("output_path", &output_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (input_type != "photometric" && input_type != "geometric") {
    std::cout << "ERROR: Invalid input type - supported values are "
                 "'photometric' and 'geometric'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::Model model;
  std::vector<uint8_t> used_image_mask;
  if (!ReadConfig(workspace_path, workspace_format, input_type, &model,
                  &used_image_mask)) {
    return EXIT_FAILURE;
  }

  std::cout << std::endl;
  options.dense_mapper_options->fusion.Print();
  std::cout << std::endl;

  const auto points = mvs::StereoFusion(
      options.dense_mapper_options->fusion, used_image_mask, model.images,
      model.depth_maps, model.normal_maps, model.consistency_graph);

  std::cout << "Number of fused points: " << points.size() << std::endl;

  std::cout << "Writing output: " << output_path << std::endl;
  WritePlyBinary(output_path, points);

  return EXIT_SUCCESS;
}
