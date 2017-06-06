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

#include "controllers/hierarchical_mapper.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  HierarchicalMapperController::Options hierarchical_options;
  SceneClustering::Options clustering_options;
  std::string export_path;

  OptionManager options;
  options.AddRequiredOption("database_path",
                            &hierarchical_options.database_path);
  options.AddRequiredOption("image_path", &hierarchical_options.image_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddDefaultOption("num_workers", &hierarchical_options.num_workers);
  options.AddDefaultOption("image_overlap", &clustering_options.image_overlap);
  options.AddDefaultOption("leaf_max_num_images",
                           &clustering_options.leaf_max_num_images);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  ReconstructionManager reconstruction_manager;

  HierarchicalMapperController hierarchical_mapper(
      hierarchical_options, clustering_options, *options.mapper,
      &reconstruction_manager);
  hierarchical_mapper.Start();
  hierarchical_mapper.Wait();

  reconstruction_manager.Write(export_path, &options);

  return EXIT_SUCCESS;
}
