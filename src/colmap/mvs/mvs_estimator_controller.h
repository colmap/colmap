// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/mvs_estimator.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/threading.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

class Workspace;

class MVSEstimatorController : public BaseController {
 public:
  MVSEstimatorController(const MVSEstimator::Options& options,
                         const std::filesystem::path& workspace_path,
                         const std::string& workspace_format,
                         const std::string& pmvs_option_name,
                         const std::filesystem::path& config_path = "");
  ~MVSEstimatorController();

  void Run();
  std::string OutputType(MVSEstimator::Pass pass) const;

 private:
  void ReadWorkspace();
  void ReadProblems();
  void ReadDeviceIndices();
  void ProcessProblem(MVSEstimator::Pass pass, size_t problem_idx);

  const MVSEstimator::Options options_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::filesystem::path config_path_;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex workspace_mutex_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<MVSEstimator::Problem> problems_;
  std::vector<std::unique_ptr<MVSEstimator>> estimators_;
  std::vector<int> device_indices_;
  std::vector<std::pair<float, float>> depth_ranges_;
};

}  // namespace mvs
}  // namespace colmap
