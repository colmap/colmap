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

void ReadConfig(const DenseMapperOptions& options,
                const std::string& workspace_path,
                const std::string& workspace_format, mvs::Model* model,
                std::vector<mvs::PatchMatch::Problem>* problems) {
  std::cout << "Reading model..." << std::endl;
  model->Read(workspace_path, workspace_format);

  std::cout << "Reading configuration..." << std::endl;

  std::ifstream file(JoinPaths(workspace_path, "dense/patch-match.cfg"));
  CHECK(file.is_open());

  std::set<int> used_image_ids;
  std::vector<std::map<int, int>> shared_points;

  std::string line;
  std::string ref_image_name;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = line;
      continue;
    }

    mvs::PatchMatch::Problem problem;

    problem.images = &model->images;
    problem.depth_maps = &model->depth_maps;
    problem.normal_maps = &model->normal_maps;

    problem.ref_image_id = model->GetImageId(ref_image_name);

    const std::vector<std::string> src_image_names =
        CSVToVector<std::string>(line);

    if (src_image_names.size() == 1 && src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_ids.clear();
      problem.src_image_ids.reserve(model->images.size() - 1);
      for (size_t image_id = 0; image_id < model->images.size(); ++image_id) {
        if (static_cast<int>(image_id) != problem.ref_image_id) {
          problem.src_image_ids.push_back(image_id);
          used_image_ids.insert(image_id);
        }
      }
    } else if (src_image_names.size() == 2 &&
               src_image_names[0] == "__auto__") {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected.

      if (shared_points.empty()) {
        shared_points = model->ComputeSharedPoints();
      }

      const size_t max_num_src_images =
          boost::lexical_cast<int>(src_image_names[1]);

      const auto& overlapping_images = shared_points.at(problem.ref_image_id);

      if (max_num_src_images >= overlapping_images.size()) {
        problem.src_image_ids.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
          problem.src_image_ids.push_back(image.first);
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

        problem.src_image_ids.reserve(max_num_src_images);
        for (size_t i = 0; i < max_num_src_images; ++i) {
          problem.src_image_ids.push_back(src_images[i].first);
          used_image_ids.insert(src_images[i].first);
        }
      }
    } else {
      problem.src_image_ids.reserve(src_image_names.size());
      for (const auto& src_image_name : src_image_names) {
        problem.src_image_ids.push_back(model->GetImageId(src_image_name));
      }
    }

    CHECK(!problem.src_image_ids.empty()) << "Need at least one source image";

    problems->push_back(problem);

    ref_image_name.clear();
  }

  std::cout << "Reading inputs..." << std::endl;
  for (const auto image_id : used_image_ids) {
    auto& image = model->images.at(image_id);
    const bool kReadImageAsRGB = false;
    image.Read(kReadImageAsRGB);
    if (options.patch_match.geom_consistency) {
      const std::string file_name =
          model->GetImageName(image_id) + ".photometric.bin";
      auto& depth_map = model->depth_maps.at(image_id);
      depth_map.Read(JoinPaths(workspace_path, "dense/depth_maps", file_name));
      auto& normal_map = model->normal_maps.at(image_id);
      normal_map.Read(
          JoinPaths(workspace_path, "dense/normal_maps", file_name));
    }
  }

  std::cout << "Resampling model..." << std::endl;
  CHECK(!(options.image_max_size > 0 && options.image_scale_factor > 0))
      << "ERROR: Cannot both set `image_max_size` and `image_scale_factor`";
  ThreadPool resample_thread_pool;
  for (size_t image_id = 0; image_id < model->images.size(); ++image_id) {
    resample_thread_pool.AddTask([&, image_id]() {
      if (options.image_max_size > 0) {
        model->images.at(image_id).Downsize(options.image_max_size,
                                            options.image_max_size);
        model->depth_maps.at(image_id).Downsize(options.image_max_size,
                                                options.image_max_size);
        model->normal_maps.at(image_id).Downsize(options.image_max_size,
                                                 options.image_max_size);
      } else if (options.image_scale_factor > 0) {
        model->images.at(image_id).Rescale(options.image_scale_factor);
        model->depth_maps.at(image_id).Rescale(options.image_scale_factor);
        model->normal_maps.at(image_id).Rescale(options.image_scale_factor);
      }
    });
  }
  resample_thread_pool.Wait();
}

int main(int argc, char* argv[]) {
  InitializeGlog(argv);

  std::string workspace_path;
  std::string workspace_format;
  std::string output_path;
  std::string config_path;

  OptionManager options;
  options.AddDenseMapperOptions();
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddRequiredOption("workspace_format", &workspace_format);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (workspace_format != "COLMAP" && workspace_format != "PMVS") {
    std::cout << "ERROR: Invalid `workspace_format` - supported values are "
                 "'COLMAP' or 'PMVS'."
              << std::endl;
    return EXIT_FAILURE;
  }

  mvs::Model model;
  std::vector<mvs::PatchMatch::Problem> problems;
  ReadConfig(*options.dense_mapper_options, workspace_path, workspace_format,
             &model, &problems);

  const auto depth_ranges = model.ComputeDepthRanges();

  for (size_t i = 0; i < problems.size(); ++i) {
    PrintHeading1(
        StringPrintf("Processing view %d / %d", i + 1, problems.size()));

    const auto& problem = problems[i];
    problem.Print();

    mvs::PatchMatch::Options patch_match_options =
        options.dense_mapper_options->patch_match;
    patch_match_options.depth_min = depth_ranges.at(problem.ref_image_id).first;
    patch_match_options.depth_max =
        depth_ranges.at(problem.ref_image_id).second;
    patch_match_options.Print();

    mvs::PatchMatch patch_match(patch_match_options, problem);
    patch_match.Run();

    std::string output_suffix;
    if (patch_match_options.geom_consistency) {
      output_suffix = "geometric";
    } else {
      output_suffix = "photometric";
    }

    const std::string image_name = model.GetImageName(problem.ref_image_id);
    const std::string file_name =
        StringPrintf("%s.%s.bin", image_name.c_str(), output_suffix.c_str());
    std::cout << std::endl << "Writing output: " << file_name << std::endl;

    patch_match.GetDepthMap().Write(
        JoinPaths(workspace_path, "dense/depth_maps", file_name));
    patch_match.GetNormalMap().Write(
        JoinPaths(workspace_path, "dense/normal_maps", file_name));
    WriteBinaryBlob(
        JoinPaths(workspace_path, "dense/consistency_graphs", file_name),
        patch_match.GetConsistentImageIds());
  }

  return EXIT_SUCCESS;
}
