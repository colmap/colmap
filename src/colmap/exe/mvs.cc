// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/exe/mvs.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/misc.h"

namespace colmap {

int RunDelaunayMesher(int argc, char** argv) {
#if !defined(COLMAP_CGAL_ENABLED)
  LOG(ERROR) << "Delaunay meshing requires CGAL, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CGAL_ENABLED
  std::string input_path;
  std::string input_type = "dense";
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption(
      "input_path",
      &input_path,
      "Path to either the dense workspace folder or the sparse reconstruction");
  options.AddDefaultOption("input_type", &input_type, "{dense, sparse}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddDelaunayMeshingOptions();
  options.Parse(argc, argv);

  StringToLower(&input_type);
  if (input_type == "sparse") {
    mvs::SparseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else if (input_type == "dense") {
    mvs::DenseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else {
    LOG(ERROR) << "Invalid input type - "
                  "supported values are 'sparse' and 'dense'.";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif  // COLMAP_CGAL_ENABLED
}

int RunPatchMatchStereo(int argc, char** argv) {
#if !defined(COLMAP_CUDA_ENABLED)
  LOG(ERROR) << "Dense stereo reconstruction requires CUDA, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CUDA_ENABLED
  std::string workspace_path;
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";
  std::string config_path;

  OptionManager options;
  options.AddRequiredOption(
      "workspace_path",
      &workspace_path,
      "Path to the folder containing the undistorted images");
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDefaultOption("config_path", &config_path);
  options.AddPatchMatchStereoOptions();
  options.Parse(argc, argv);

  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    LOG(ERROR) << "Invalid `workspace_format` - supported values are "
                  "'COLMAP' or 'PMVS'.";
    return EXIT_FAILURE;
  }

  mvs::PatchMatchController controller(*options.patch_match_stereo,
                                       workspace_path,
                                       workspace_format,
                                       pmvs_option_name,
                                       config_path);

  controller.Start();
  controller.Wait();

  return EXIT_SUCCESS;
#endif  // COLMAP_CUDA_ENABLED
}

int RunPoissonMesher(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddPoissonMeshingOptions();
  options.Parse(argc, argv);

  CHECK(mvs::PoissonMeshing(*options.poisson_meshing, input_path, output_path));

  return EXIT_SUCCESS;
}

int RunStereoFuser(int argc, char** argv) {
  std::string workspace_path;
  std::string input_type = "geometric";
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";
  std::string output_type = "PLY";
  std::string output_path;
  std::string bbox_path;

  OptionManager options;
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDefaultOption(
      "input_type", &input_type, "{photometric, geometric}");
  options.AddDefaultOption("output_type", &output_type, "{BIN, TXT, PLY}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("bbox_path", &bbox_path);
  options.AddStereoFusionOptions();
  options.Parse(argc, argv);

  StringToLower(&workspace_format);
  if (workspace_format != "colmap" && workspace_format != "pmvs") {
    LOG(ERROR) << "Invalid `workspace_format` - supported values are "
                  "'COLMAP' or 'PMVS'.";
    return EXIT_FAILURE;
  }

  StringToLower(&input_type);
  if (input_type != "photometric" && input_type != "geometric") {
    LOG(ERROR) << "Invalid input type - supported values are "
                  "'photometric' and 'geometric'.";
    return EXIT_FAILURE;
  }

  if (!bbox_path.empty()) {
    std::ifstream file(bbox_path);
    if (file.is_open()) {
      auto& min_bound = options.stereo_fusion->bounding_box.first;
      auto& max_bound = options.stereo_fusion->bounding_box.second;
      file >> min_bound(0) >> min_bound(1) >> min_bound(2);
      file >> max_bound(0) >> max_bound(1) >> max_bound(2);
    } else {
      LOG(WARNING) << "Invalid bounds path: \"" << bbox_path
                   << "\" - continuing without bounds check";
    }
  }

  mvs::StereoFusion fuser(*options.stereo_fusion,
                          workspace_path,
                          workspace_format,
                          pmvs_option_name,
                          input_type);

  fuser.Start();
  fuser.Wait();

  Reconstruction reconstruction;

  // read data from sparse reconstruction
  if (workspace_format == "colmap") {
    reconstruction.Read(JoinPaths(workspace_path, "sparse"));
  }

  // overwrite sparse point cloud with dense point cloud from fuser
  reconstruction.ImportPLY(fuser.GetFusedPoints());

  LOG(INFO) << "Writing output: " << output_path;

  // write output
  StringToLower(&output_type);
  if (output_type == "bin") {
    reconstruction.WriteBinary(output_path);
  } else if (output_type == "txt") {
    reconstruction.WriteText(output_path);
  } else if (output_type == "ply") {
    WriteBinaryPlyPoints(output_path, fuser.GetFusedPoints());
    mvs::WritePointsVisibility(output_path + ".vis",
                               fuser.GetFusedPointsVisibility());
  } else {
    LOG(ERROR) << "Invalid `output_type`";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
