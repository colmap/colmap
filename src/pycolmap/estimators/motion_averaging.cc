#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/gravity_refinement.h"
#include "colmap/estimators/rotation_averaging.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindGlobalPositioner(py::module& m) {
  auto PyGlobalPositionerOptions =
      py::classh<GlobalPositionerOptions>(m, "GlobalPositionerOptions")
          .def(py::init<>())
          .def_readwrite("generate_random_positions",
                         &GlobalPositionerOptions::generate_random_positions,
                         "Whether to initialize camera positions randomly.")
          .def_readwrite("generate_random_points",
                         &GlobalPositionerOptions::generate_random_points,
                         "Whether to initialize 3D point positions randomly.")
          .def_readwrite("generate_scales",
                         &GlobalPositionerOptions::generate_scales,
                         "Whether to initialize scales to constant 1 or derive "
                         "from positions.")
          .def_readwrite("optimize_positions",
                         &GlobalPositionerOptions::optimize_positions,
                         "Whether to optimize camera positions.")
          .def_readwrite("optimize_points",
                         &GlobalPositionerOptions::optimize_points,
                         "Whether to optimize 3D point positions.")
          .def_readwrite("optimize_scales",
                         &GlobalPositionerOptions::optimize_scales,
                         "Whether to optimize scales.")
          .def_readwrite("use_gpu",
                         &GlobalPositionerOptions::use_gpu,
                         "Whether to use GPU for optimization.")
          .def_readwrite("gpu_index",
                         &GlobalPositionerOptions::gpu_index,
                         "GPU device index (-1 for auto).")
          .def_readwrite("min_num_images_gpu_solver",
                         &GlobalPositionerOptions::min_num_images_gpu_solver,
                         "Minimum number of images to use GPU solver.")
          .def_readwrite("min_num_view_per_track",
                         &GlobalPositionerOptions::min_num_view_per_track,
                         "Minimum number of views per track.")
          .def_readwrite("random_seed",
                         &GlobalPositionerOptions::random_seed,
                         "PRNG seed for random initialization. -1 for "
                         "non-deterministic.")
          .def_readwrite("loss_function_scale",
                         &GlobalPositionerOptions::loss_function_scale,
                         "Scaling factor for the loss function.")
          .def_readwrite("use_parameter_block_ordering",
                         &GlobalPositionerOptions::use_parameter_block_ordering,
                         "Whether to use custom parameter block ordering.");
  MakeDataclass(PyGlobalPositionerOptions);

  m.def(
      "run_global_positioning",
      [](const GlobalPositionerOptions& options,
         const PoseGraph& pose_graph,
         Reconstruction& reconstruction) {
        py::gil_scoped_release release;
        bool success =
            RunGlobalPositioning(options, pose_graph, reconstruction);
        return success;
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "Solve global positioning using point-to-camera constraints. Returns "
      "True if optimization succeeded.");
}

void BindGravityRefiner(py::module& m) {
  auto PyGravityRefinerOptions =
      py::classh<GravityRefinerOptions>(m, "GravityRefinerOptions")
          .def(py::init<>())
          .def_readwrite(
              "max_outlier_ratio",
              &GravityRefinerOptions::max_outlier_ratio,
              "Maximum ratio that gravity should be consistent with.")
          .def_readwrite("max_gravity_error",
                         &GravityRefinerOptions::max_gravity_error,
                         "Maximum allowed angle error in degrees.")
          .def_readwrite("min_num_neighbors",
                         &GravityRefinerOptions::min_num_neighbors,
                         "Minimum neighbors required for refinement.");
  MakeDataclass(PyGravityRefinerOptions);

  m.def(
      "run_gravity_refinement",
      [](const GravityRefinerOptions& options,
         const PoseGraph& pose_graph,
         const Reconstruction& reconstruction,
         std::vector<PosePrior>& pose_priors) {
        py::gil_scoped_release release;
        RunGravityRefinement(options, pose_graph, reconstruction, pose_priors);
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a,
      "Refine gravity stored in pose priors using relative rotations from the "
      "pose graph. Modifies pose_priors in-place.");
}

void BindRotationEstimator(py::module& m) {
  using WeightType = RotationEstimatorOptions::WeightType;
  auto PyWeightType = py::enum_<WeightType>(m, "RotationWeightType")
                          .value("GEMAN_MCCLURE", WeightType::GEMAN_MCCLURE)
                          .value("HALF_NORM", WeightType::HALF_NORM);
  AddStringToEnumConstructor(PyWeightType);

  auto PyRotationEstimatorOptions =
      py::classh<RotationEstimatorOptions>(m, "RotationEstimatorOptions")
          .def(py::init<>())
          .def_readwrite("random_seed",
                         &RotationEstimatorOptions::random_seed,
                         "PRNG seed. -1 for non-deterministic, >=0 for "
                         "deterministic.")
          .def_readwrite("max_num_l1_iterations",
                         &RotationEstimatorOptions::max_num_l1_iterations,
                         "Maximum number of L1 minimization iterations.")
          .def_readwrite(
              "l1_step_convergence_threshold",
              &RotationEstimatorOptions::l1_step_convergence_threshold,
              "Average step size threshold to terminate L1 minimization.")
          .def_readwrite("max_num_irls_iterations",
                         &RotationEstimatorOptions::max_num_irls_iterations,
                         "Number of IRLS iterations to perform.")
          .def_readwrite(
              "irls_step_convergence_threshold",
              &RotationEstimatorOptions::irls_step_convergence_threshold,
              "Average step size threshold to terminate IRLS.")
          .def_readwrite("gravity_dir",
                         &RotationEstimatorOptions::gravity_dir,
                         "Gravity direction vector.")
          .def_readwrite(
              "irls_loss_parameter_sigma",
              &RotationEstimatorOptions::irls_loss_parameter_sigma,
              "Point where Huber-like cost switches from L1 to L2 (degrees).")
          .def_readwrite("weight_type",
                         &RotationEstimatorOptions::weight_type,
                         "Weight type for IRLS: GEMAN_MCCLURE or HALF_NORM.")
          .def_readwrite("skip_initialization",
                         &RotationEstimatorOptions::skip_initialization,
                         "Skip maximum spanning tree initialization.")
          .def_readwrite("use_gravity",
                         &RotationEstimatorOptions::use_gravity,
                         "Use gravity priors for rotation averaging.")
          .def_readwrite("use_stratified",
                         &RotationEstimatorOptions::use_stratified,
                         "Use stratified solving for mixed gravity systems.")
          .def_readwrite(
              "filter_unregistered",
              &RotationEstimatorOptions::filter_unregistered,
              "Only consider frames with existing poses for connected "
              "components.")
          .def_readwrite(
              "max_rotation_error_deg",
              &RotationEstimatorOptions::max_rotation_error_deg,
              "Filter pairs with rotation error exceeding this threshold "
              "(degrees).");
  MakeDataclass(PyRotationEstimatorOptions);

  m.def(
      "run_rotation_averaging",
      [](const RotationEstimatorOptions& options,
         PoseGraph& pose_graph,
         Reconstruction& reconstruction,
         const std::vector<PosePrior>& pose_priors) {
        py::gil_scoped_release release;
        bool success = RunRotationAveraging(
            options, pose_graph, reconstruction, pose_priors);
        return success;
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a,
      "High-level rotation averaging solver that handles rig expansion. "
      "Returns True if rotation averaging succeeded.");
}

void BindMotionAveraging(py::module& m) {
  BindGravityRefiner(m);
  BindRotationEstimator(m);
  BindGlobalPositioner(m);
}
