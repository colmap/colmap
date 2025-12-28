#include "glomap/sfm/global_mapper.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/util/file.h"
#include "colmap/util/timer.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/io/colmap_io.h"

namespace glomap {
// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunGlobalMapper(int argc, char** argv) {
  std::string output_path;

  std::string constraint_type = "ONLY_POINTS";
  std::string output_format = "bin";

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("constraint_type",
                           &constraint_type,
                           "{ONLY_POINTS, ONLY_CAMERAS, "
                           "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperOptions();

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (constraint_type == "ONLY_POINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else {
    LOG(ERROR) << "Invalid constriant type";
    return EXIT_FAILURE;
  }

  // Check whether output_format is valid
  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  auto database = colmap::Database::Open(*options.database_path);

  colmap::Reconstruction reconstruction;
  ViewGraph view_graph;
  InitializeGlomapFromDatabase(*database, reconstruction, view_graph);

  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();

  if (view_graph.Empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }

  options.mapper->image_path = *options.image_path;

  GlobalMapper global_mapper(*options.mapper);

  // Main solver
  LOG(INFO) << "Loaded database";
  colmap::Timer run_timer;
  run_timer.Start();
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(
      database.get(), view_graph, reconstruction, pose_priors, cluster_ids);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteReconstructionsByClusters(output_path,
                                 reconstruction,
                                 cluster_ids,
                                 output_format,
                                 *options.image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap
