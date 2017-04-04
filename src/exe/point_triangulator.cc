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

#include <boost/filesystem.hpp>

#include "sfm/controllers.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  if (!boost::filesystem::is_directory(import_path)) {
    std::cerr << "ERROR: `import_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!boost::filesystem::is_directory(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory" << std::endl;
    return EXIT_FAILURE;
  }

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Database database(*options.database_path);
    Timer timer;
    timer.Start();
    const size_t min_num_matches =
        static_cast<size_t>(options.mapper_options->min_num_matches);
    database_cache.Load(database, min_num_matches,
                        options.mapper_options->ignore_watermarks,
                        options.mapper_options->image_names);
    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  CHECK_GE(reconstruction.NumRegImages(), 2)
      << "Need at least two images for triangulation";

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&reconstruction);

  //////////////////////////////////////////////////////////////////////////////
  // Triangulation
  //////////////////////////////////////////////////////////////////////////////

  const IncrementalTriangulator::Options inc_tri_options =
      options.mapper_options->TriangulationOptions();

  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);

    PrintHeading1("Triangulating image #" + std::to_string(image_id) + " (" +
                  std::to_string(reconstruction.NumRegImages() + 1) + ")");

    const size_t num_visible_points3D = image.NumVisiblePoints3D();

    std::cout << "  => Image sees " << num_visible_points3D << " / "
              << image.NumObservations() << " points" << std::endl;

    mapper.TriangulateImage(inc_tri_options, image_id);

    std::cout << "  => Triangulated "
              << (image.NumVisiblePoints3D() - num_visible_points3D)
              << " points" << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Bundle adjustment
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Bundle adjustment");

  // Avoid degeneracies in bundle adjustment.
  reconstruction.FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    ba_config.AddImage(image_id);
    ba_config.SetConstantPose(image_id);
    ba_config.SetConstantCamera(reconstruction.Image(image_id).CameraId());
  }

  // Run bundle adjustment.
  const auto ba_options =
      options.mapper_options->GlobalBundleAdjustmentOptions();
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  CHECK(bundle_adjuster.Solve(&reconstruction));

  // Filter outlier observations.
  const size_t num_filtered_observations =
      mapper.FilterPoints(options.mapper_options->IncrementalMapperOptions());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);

  reconstruction.Write(export_path);

  return EXIT_SUCCESS;
}
