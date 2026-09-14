// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/mvs_estimator.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/threading.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

class Workspace;

// PatchMatch stereo estimator, which requires a CUDA or HIP enabled build. The
// actual implementation in PatchMatchCuda is hidden from any boost or Eigen
// code, since NVCC/MSVC cannot compile complex C++ code.
class PatchMatch : public MVSEstimator {
 public:
  using Options = PatchMatchOptions;

  explicit PatchMatch(Options options);

  Capabilities GetCapabilities() const override;
  Result Estimate(const Problem& problem, Pass pass) override;

 private:
  Options options_;
};

// This thread processes all problems in a workspace. A workspace has the
// following file structure, if the workspace format is "COLMAP":
//
//    images/*
//    sparse/{cameras.txt, images.txt, points3D.txt}
//    stereo/
//      depth_maps/*
//      normal_maps/*
//      consistency_graphs/*
//      patch-match.cfg
//
// The `patch-match.cfg` file specifies the images to be processed as:
//
//    image_name1.jpg
//    __all__
//    image_name2.jpg
//    __auto__, 20
//    image_name3.jpg
//    image_name1.jpg, image_name2.jpg
//
// Two consecutive lines specify the images used to compute one patch match
// problem. The first line specifies the reference image and the second line the
// source images. Image names are relative to the `images` directory. In this
// example, the first reference image uses all other images as source images,
// the second reference image uses the 20 most connected images as source
// images, and the third reference image uses the first and second as source
// images. Note that all specified images must be reconstructed in the COLMAP
// reconstruction provided in the `sparse` folder.

class PatchMatchController : public BaseController {
 public:
  PatchMatchController(const PatchMatchOptions& options,
                       const std::filesystem::path& workspace_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name,
                       const std::filesystem::path& config_path = "");
  void Run();

 private:
  void ReadWorkspace();
  void ReadProblems();
  void ReadGpuIndices();
  void ProcessProblem(MVSEstimator::Pass pass, size_t problem_idx);

  const PatchMatchOptions options_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::filesystem::path config_path_;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ThreadPool> io_thread_pool_;
  std::mutex workspace_mutex_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<PatchMatch::Problem> problems_;
  std::vector<int> gpu_indices_;
  std::vector<std::pair<float, float>> depth_ranges_;
};

}  // namespace mvs
}  // namespace colmap
