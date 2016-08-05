// COLMAP - Structure-from-Motion.
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

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddExtractionOptions();
  options.AddDefaultOption("use_gpu", use_gpu, &use_gpu);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  const std::vector<double> camera_params = CSVToVector<double>(
      options.extraction_options->reader_options.camera_params);
  const int camera_model_id = CameraModelNameToId(
      options.extraction_options->reader_options.camera_model);

  if (camera_params.size() > 0 &&
      !CameraModelVerifyParams(camera_model_id, camera_params)) {
    std::cerr << "ERROR: Invalid camera parameters" << std::endl;
    return EXIT_FAILURE;
  }

  ImageReader::Options reader_options =
      options.extraction_options->reader_options;
  reader_options.database_path = *options.database_path;
  reader_options.image_path = *options.image_path;

  if (use_gpu) {
    QApplication app(argc, argv);
    SiftGPUFeatureExtractor feature_extractor(
        reader_options, options.extraction_options->sift_options);

    std::thread thread([&app, &feature_extractor]() {
      feature_extractor.Start();
      feature_extractor.Wait();
      app.exit();
    });

    app.exec();
    thread.join();
  } else {
    SiftCPUFeatureExtractor feature_extractor(
        reader_options, options.extraction_options->sift_options,
        options.extraction_options->cpu_options);
    feature_extractor.Start();
    feature_extractor.Wait();
  }

  return EXIT_SUCCESS;
}
