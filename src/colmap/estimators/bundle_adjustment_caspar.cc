#include "colmap/estimators/bundle_adjustment_caspar.h"

#include <memory>
#include <utility>

#include "generated/solver.h"
#include "generated/solver_params.h"
#include <ceres/solver.h>

namespace colmap {
namespace {

class CasparBundleAdjuster : public BundleAdjuster {
 public:
  CasparBundleAdjuster(BundleAdjustmentOptions options,
                       BundleAdjustmentConfig config,
                       Reconstruction& reconstruction)
      : BundleAdjuster(std::move(options), std::move(config)) {
    // Get params from BundjeAdjustmentConfig
    reconstruction.Cameras();
  }

  ceres::Solver::Summary Solve()
      override {  // Do we need to convert to a CERES summary?
    ceres::Solver::Summary summary;
    // We should check for number of residuals do exit early if problem has size
    // zero
    caspar::GraphSolver solver(solver_params, 100, 100);
    solver_params.diag_init = 1.0;
    solver_params.solver_iter_max = 100;
    solver_params.pcg_iter_max = 10;
    solver_params.pcg_rel_error_exit = 1e-2;

    solver.set_params(solver_params);
    auto result =
        solver.solve(true);  // Should pipe printing progress somewhere

    printf("Caspar solver returned: %f", result);
    return summary;
  }

  void AddImagetoProblem() { int i = 0; }

  std::shared_ptr<ceres::Problem>& Problem() override { return problem_; }

 private:
  caspar::SolverParams solver_params;
  std::shared_ptr<ceres::Problem> problem_;
};

}  // namespace

std::unique_ptr<BundleAdjuster> CreateDefaultCasparBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction) {
  return std::make_unique<CasparBundleAdjuster>(
      std::move(options), std::move(config), reconstruction);
}

}  // namespace colmap