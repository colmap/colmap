#include <string>

#include "colmap/controllers/automatic_reconstruction.h"
#include "colmap/controllers/bundle_adjustment.h"
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/estimators/similarity_transform.h"
#include "colmap/exe/gui.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

using namespace colmap;

void RunPointTriangulatorImpl(
    const std::shared_ptr<Reconstruction>& reconstruction,
    const std::string& database_path,
    const std::string& image_path,
    const std::string& output_path,
    const IncrementalPipelineOptions& options,
    const bool clear_points,
    const bool refine_intrinsics) {
  THROW_CHECK_GE(reconstruction->NumRegImages(), 2)
      << "Need at least two images for triangulation";
  if (clear_points) {
    reconstruction->DeleteAllPoints2DAndPoints3D();
    reconstruction->TranscribeImageIdsToDatabase(
        *OpenSqliteDatabase(database_path));
  }

  auto options_tmp = std::make_shared<IncrementalPipelineOptions>(options);
  options_tmp->fix_existing_frames = true;
  options_tmp->ba_refine_focal_length = refine_intrinsics;
  options_tmp->ba_refine_principal_point = false;
  options_tmp->ba_refine_extra_params = refine_intrinsics;

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalPipeline mapper(
      options_tmp, image_path, database_path, reconstruction_manager);
  mapper.TriangulateReconstruction(reconstruction);
  reconstruction->Write(output_path);
}

int main() {
    const std::string base_dir = "/local/home/shaoliu/data/testing/lamaria/dat";
    const std::string database_path = base_dir + "/database.db";
    const std::string recon_path = base_dir + "/keyframe_recon";
    const std::string image_path = base_dir + "/keyframes";
    const std::string output_path = base_dir + "/triangulated";

    IncrementalPipelineOptions options;
    std::shared_ptr<Reconstruction> recon = std::make_shared<Reconstruction>();
    recon->Read(recon_path);
    RunPointTriangulatorImpl(recon, database_path, image_path, output_path, options, true, false);
}
