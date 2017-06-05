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

#include <QApplication>

#include "base/camera_models.h"
#include "base/feature_extraction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  bool use_gpu = true;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("use_gpu", &use_gpu);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddExtractionOptions();
  options.Parse(argc, argv);

  ImageReader::Options reader_options = *options.image_reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
  }

  const std::vector<double> camera_params =
      CSVToVector<double>(options.image_reader->camera_params);
  const int camera_model_id =
      CameraModelNameToId(options.image_reader->camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  
  const std::vector<int> gpu_indices =
      CSVToVector<int>(options.sift_gpu_extraction->index);
  if (use_gpu && gpu_indices.size() == 1 && gpu_indices[0] < 0) {
    app.reset(new QApplication(argc, argv));
  }

  std::vector<std::unique_ptr<Thread>> feature_extractors;
  if (use_gpu) {
    feature_extractors.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      feature_extractors.emplace_back(
          new SiftGPUFeatureExtractor(reader_options, *options.sift_extraction,
                                    *options.sift_gpu_extraction));
    }
  } else {
    feature_extractors.emplace_back(
        new SiftCPUFeatureExtractor(reader_options, *options.sift_extraction,
                                    *options.sift_cpu_extraction));
  }

  for (int i = 0; i < gpu_indices.size(); i++) {
    if (use_gpu && gpu_indices[i] < 0) {
      RunThreadWithOpenGLContext(feature_extractors[i].get());
    } else {
      feature_extractors[i]->Start();
      feature_extractors[i]->Wait();
    }
  }

  return EXIT_SUCCESS;
}
