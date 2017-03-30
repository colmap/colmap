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
  options.AddExtractionOptions();
  options.AddDefaultOption("use_gpu", use_gpu, &use_gpu);
  options.AddDefaultOption("image_list_path", image_list_path,
                           &image_list_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  ImageReader::Options reader_options = options.extraction_options->reader;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (!image_list_path.empty()) {
    reader_options.image_list = ReadTextFileLines(image_list_path);
  }

  const std::vector<double> camera_params =
      CSVToVector<double>(options.extraction_options->reader.camera_params);
  const int camera_model_id =
      CameraModelNameToId(options.extraction_options->reader.camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr<QApplication> app;
  if (use_gpu && options.extraction_options->gpu.index < 0) {
    app.reset(new QApplication(argc, argv));
  }

  std::unique_ptr<Thread> feature_extractor;
  if (use_gpu) {
    feature_extractor.reset(new SiftGPUFeatureExtractor(
        reader_options, options.extraction_options->sift,
        options.extraction_options->gpu));
  } else {
    feature_extractor.reset(new SiftCPUFeatureExtractor(
        reader_options, options.extraction_options->sift,
        options.extraction_options->cpu));
  }

  if (use_gpu && options.extraction_options->gpu.index < 0) {
    RunThreadWithOpenGLContext(app.get(), feature_extractor.get());
  } else {
    feature_extractor->Start();
    feature_extractor->Wait();
  }

  return EXIT_SUCCESS;
}
