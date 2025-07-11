#include "pycolmap/helpers.h"

#include <ceres/ceres.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

// Some important ceres bindings for using crucial pycolmap features (e.g.
// bundle adjustment) without pyceres.
//
// Branched from
// https://github.com/cvg/pyceres/blob/main/_pyceres/core/types.h
// https://github.com/cvg/pyceres/blob/main/_pyceres/core/solver.h
void BindCeresTypes(py::module& m) {
  auto ownt = py::enum_<ceres::Ownership>(m, "Ownership", py::module_local())
                  .value("DO_NOT_TAKE_OWNERSHIP",
                         ceres::Ownership::DO_NOT_TAKE_OWNERSHIP)
                  .value("TAKE_OWNERSHIP", ceres::Ownership::TAKE_OWNERSHIP)
                  .export_values();
  AddStringToEnumConstructor(ownt);

  auto mt =
      py::enum_<ceres::MinimizerType>(m, "MinimizerType", py::module_local())
          .value("LINE_SEARCH", ceres::MinimizerType::LINE_SEARCH)
          .value("TRUST_REGION", ceres::MinimizerType::TRUST_REGION);
  AddStringToEnumConstructor(mt);

  auto linesearcht =
      py::enum_<ceres::LineSearchType>(m, "LineSearchType", py::module_local())
          .value("ARMIJO", ceres::LineSearchType::ARMIJO)
          .value("WOLFE", ceres::LineSearchType::WOLFE);
  AddStringToEnumConstructor(linesearcht);

  auto lsdt =
      py::enum_<ceres::LineSearchDirectionType>(
          m, "LineSearchDirectionType", py::module_local())
          .value("BFGS", ceres::LineSearchDirectionType::BFGS)
          .value("LBFGS", ceres::LineSearchDirectionType::LBFGS)
          .value("NONLINEAR_CONJUGATE_GRADIENT",
                 ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT)
          .value("STEEPEST_DESCENT",
                 ceres::LineSearchDirectionType::STEEPEST_DESCENT);
  AddStringToEnumConstructor(lsdt);

  auto lsit =
      py::enum_<ceres::LineSearchInterpolationType>(
          m, "LineSearchInterpolationType", py::module_local())
          .value("BISECTION", ceres::LineSearchInterpolationType::BISECTION)
          .value("CUBIC", ceres::LineSearchInterpolationType::CUBIC)
          .value("QUADRATIC", ceres::LineSearchInterpolationType::QUADRATIC);
  AddStringToEnumConstructor(lsit);

  auto ncgt =
      py::enum_<ceres::NonlinearConjugateGradientType>(
          m, "NonlinearConjugateGradientType", py::module_local())
          .value("FLETCHER_REEVES",
                 ceres::NonlinearConjugateGradientType::FLETCHER_REEVES)
          .value("HESTENES_STIEFEL",
                 ceres::NonlinearConjugateGradientType::HESTENES_STIEFEL)
          .value("POLAK_RIBIERE",
                 ceres::NonlinearConjugateGradientType::POLAK_RIBIERE);
  AddStringToEnumConstructor(ncgt);

  auto linsolt =
      py::enum_<ceres::LinearSolverType>(
          m, "LinearSolverType", py::module_local())
          .value("DENSE_NORMAL_CHOLESKY",
                 ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY)
          .value("DENSE_QR", ceres::LinearSolverType::DENSE_QR)
          .value("SPARSE_NORMAL_CHOLESKY",
                 ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY)
          .value("DENSE_SCHUR", ceres::LinearSolverType::DENSE_SCHUR)
          .value("SPARSE_SCHUR", ceres::LinearSolverType::SPARSE_SCHUR)
          .value("ITERATIVE_SCHUR", ceres::LinearSolverType::ITERATIVE_SCHUR)
          .value("CGNR", ceres::LinearSolverType::CGNR);
  AddStringToEnumConstructor(linsolt);

  auto dogt =
      py::enum_<ceres::DoglegType>(m, "DoglegType", py::module_local())
          .value("TRADITIONAL_DOGLEG", ceres::DoglegType::TRADITIONAL_DOGLEG)
          .value("SUBSPACE_DOGLEG", ceres::DoglegType::SUBSPACE_DOGLEG);
  AddStringToEnumConstructor(dogt);

  auto trst = py::enum_<ceres::TrustRegionStrategyType>(
                  m, "TrustRegionStrategyType", py::module_local())
                  .value("LEVENBERG_MARQUARDT",
                         ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT)
                  .value("DOGLEG", ceres::TrustRegionStrategyType::DOGLEG);
  AddStringToEnumConstructor(trst);

  auto prt =
      py::enum_<ceres::PreconditionerType>(
          m, "PreconditionerType", py::module_local())
          .value("IDENTITY", ceres::PreconditionerType::IDENTITY)
          .value("JACOBI", ceres::PreconditionerType::JACOBI)
          .value("SCHUR_JACOBI", ceres::PreconditionerType::SCHUR_JACOBI)
          .value("CLUSTER_JACOBI", ceres::PreconditionerType::CLUSTER_JACOBI)
          .value("CLUSTER_TRIDIAGONAL",
                 ceres::PreconditionerType::CLUSTER_TRIDIAGONAL);
  AddStringToEnumConstructor(prt);

  auto vct = py::enum_<ceres::VisibilityClusteringType>(
                 m, "VisibilityClusteringType", py::module_local())
                 .value("CANONICAL_VIEWS",
                        ceres::VisibilityClusteringType::CANONICAL_VIEWS)
                 .value("SINGLE_LINKAGE",
                        ceres::VisibilityClusteringType::SINGLE_LINKAGE);
  AddStringToEnumConstructor(vct);

  auto dlalt =
      py::enum_<ceres::DenseLinearAlgebraLibraryType>(
          m, "DenseLinearAlgebraLibraryType", py::module_local())
          .value("EIGEN", ceres::DenseLinearAlgebraLibraryType::EIGEN)
          .value("LAPACK", ceres::DenseLinearAlgebraLibraryType::LAPACK)
          .value("CUDA", ceres::DenseLinearAlgebraLibraryType::CUDA);
  AddStringToEnumConstructor(dlalt);

  auto slalt =
      py::enum_<ceres::SparseLinearAlgebraLibraryType>(
          m, "SparseLinearAlgebraLibraryType", py::module_local())
          .value("SUITE_SPARSE",
                 ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE)
          .value("EIGEN_SPARSE",
                 ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE)
          .value("ACCELERATE_SPARSE",
                 ceres::SparseLinearAlgebraLibraryType::ACCELERATE_SPARSE)
          .value("NO_SPARSE", ceres::SparseLinearAlgebraLibraryType::NO_SPARSE);
  AddStringToEnumConstructor(slalt);

  auto logt =
      py::enum_<ceres::LoggingType>(m, "LoggingType", py::module_local())
          .value("SILENT", ceres::LoggingType::SILENT)
          .value("PER_MINIMIZER_ITERATION",
                 ceres::LoggingType::PER_MINIMIZER_ITERATION);
  AddStringToEnumConstructor(logt);

  auto cbrt =
      py::enum_<ceres::CallbackReturnType>(
          m, "CallbackReturnType", py::module_local())
          .value("SOLVER_CONTINUE", ceres::CallbackReturnType::SOLVER_CONTINUE)
          .value("SOLVER_ABORT", ceres::CallbackReturnType::SOLVER_ABORT)
          .value("SOLVER_TERMINATE_SUCCESSFULLY",
                 ceres::CallbackReturnType::SOLVER_TERMINATE_SUCCESSFULLY);
  AddStringToEnumConstructor(cbrt);

  auto dft =
      py::enum_<ceres::DumpFormatType>(m, "DumpFormatType", py::module_local())
          .value("CONSOLE", ceres::DumpFormatType::CONSOLE)
          .value("TEXTFILE", ceres::DumpFormatType::TEXTFILE);
  AddStringToEnumConstructor(dft);

  auto termt =
      py::enum_<ceres::TerminationType>(
          m, "TerminationType", py::module_local())
          .value("CONVERGENCE", ceres::TerminationType::CONVERGENCE)
          .value("NO_CONVERGENCE", ceres::TerminationType::NO_CONVERGENCE)
          .value("FAILURE", ceres::TerminationType::FAILURE)
          .value("USER_SUCCESS", ceres::TerminationType::USER_SUCCESS)
          .value("USER_FAILURE", ceres::TerminationType::USER_FAILURE);
  AddStringToEnumConstructor(termt);
}

void BindCeresSolver(py::module& m) {
  using Options = ceres::Solver::Options;
  py::class_<Options, std::shared_ptr<Options>> PyOptions(
      m, "SolverOptions", py::module_local());
  PyOptions.def(py::init<>())
      .def(py::init<const Options&>())
      .def("IsValid", &Options::IsValid)
      .def_readwrite("minimizer_type", &Options::minimizer_type)
      .def_readwrite("line_search_direction_type",
                     &Options::line_search_direction_type)
      .def_readwrite("line_search_type", &Options::line_search_type)
      .def_readwrite("nonlinear_conjugate_gradient_type",
                     &Options::nonlinear_conjugate_gradient_type)
      .def_readwrite("max_lbfgs_rank", &Options::max_lbfgs_rank)
      .def_readwrite("use_approximate_eigenvalue_bfgs_scaling",
                     &Options::use_approximate_eigenvalue_bfgs_scaling)
      .def_readwrite("line_search_interpolation_type",
                     &Options::line_search_interpolation_type)
      .def_readwrite("min_line_search_step_size",
                     &Options::min_line_search_step_size)
      .def_readwrite("line_search_sufficient_function_decrease",
                     &Options::line_search_sufficient_function_decrease)
      .def_readwrite("max_line_search_step_contraction",
                     &Options::max_line_search_step_contraction)
      .def_readwrite("min_line_search_step_contraction",
                     &Options::min_line_search_step_contraction)
      .def_readwrite("max_num_line_search_step_size_iterations",
                     &Options::max_num_line_search_step_size_iterations)
      .def_readwrite("max_num_line_search_direction_restarts",
                     &Options::max_num_line_search_direction_restarts)
      .def_readwrite("line_search_sufficient_curvature_decrease",
                     &Options::line_search_sufficient_curvature_decrease)
      .def_readwrite("max_line_search_step_expansion",
                     &Options::max_line_search_step_expansion)
      .def_readwrite("trust_region_strategy_type",
                     &Options::trust_region_strategy_type)
      .def_readwrite("dogleg_type", &Options::dogleg_type)
      .def_readwrite("use_nonmonotonic_steps", &Options::use_nonmonotonic_steps)
      .def_readwrite("max_consecutive_nonmonotonic_steps",
                     &Options::max_consecutive_nonmonotonic_steps)
      .def_readwrite("max_num_iterations", &Options::max_num_iterations)
      .def_readwrite("max_solver_time_in_seconds",
                     &Options::max_solver_time_in_seconds)
      .def_property(
          "num_threads",
          [](const Options& self) { return self.num_threads; },
          [](Options& self, int n_threads) {
            int effective_n_threads = colmap::GetEffectiveNumThreads(n_threads);
            self.num_threads = effective_n_threads;
#if CERES_VERSION_MAJOR < 2
            self.num_linear_solver_threads = effective_n_threads;
#endif  // CERES_VERSION_MAJOR
          })
      .def_readwrite("initial_trust_region_radius",
                     &Options::initial_trust_region_radius)
      .def_readwrite("max_trust_region_radius",
                     &Options::max_trust_region_radius)
      .def_readwrite("min_trust_region_radius",
                     &Options::min_trust_region_radius)
      .def_readwrite("min_relative_decrease", &Options::min_relative_decrease)
      .def_readwrite("min_lm_diagonal", &Options::min_lm_diagonal)
      .def_readwrite("max_lm_diagonal", &Options::max_lm_diagonal)
      .def_readwrite("max_num_consecutive_invalid_steps",
                     &Options::max_num_consecutive_invalid_steps)
      .def_readwrite("function_tolerance", &Options::function_tolerance)
      .def_readwrite("gradient_tolerance", &Options::gradient_tolerance)
      .def_readwrite("parameter_tolerance", &Options::parameter_tolerance)
      .def_readwrite("linear_solver_type", &Options::linear_solver_type)
      .def_readwrite("preconditioner_type", &Options::preconditioner_type)
      .def_readwrite("visibility_clustering_type",
                     &Options::visibility_clustering_type)
      .def_readwrite("dense_linear_algebra_library_type",
                     &Options::dense_linear_algebra_library_type)
      .def_readwrite("sparse_linear_algebra_library_type",
                     &Options::sparse_linear_algebra_library_type)
      .def_readwrite("use_explicit_schur_complement",
                     &Options::use_explicit_schur_complement)
      .def_readwrite("dynamic_sparsity", &Options::dynamic_sparsity)
      .def_readwrite("use_inner_iterations", &Options::use_inner_iterations)
      .def_readwrite("inner_iteration_tolerance",
                     &Options::inner_iteration_tolerance)
      .def_readwrite("min_linear_solver_iterations",
                     &Options::min_linear_solver_iterations)
      .def_readwrite("max_linear_solver_iterations",
                     &Options::max_linear_solver_iterations)
      .def_readwrite("eta", &Options::eta)
      .def_readwrite("jacobi_scaling", &Options::jacobi_scaling)
      .def_readwrite("logging_type", &Options::logging_type)
      .def_readwrite("minimizer_progress_to_stdout",
                     &Options::minimizer_progress_to_stdout)
      .def_readwrite("trust_region_problem_dump_directory",
                     &Options::trust_region_problem_dump_directory)
      .def_readwrite("trust_region_problem_dump_format_type",
                     &Options::trust_region_problem_dump_format_type)
      .def_readwrite("check_gradients", &Options::check_gradients)
      .def_readwrite("gradient_check_relative_precision",
                     &Options::gradient_check_relative_precision)
      .def_readwrite(
          "gradient_check_numeric_derivative_relative_step_size",
          &Options::gradient_check_numeric_derivative_relative_step_size)
      .def_readwrite("update_state_every_iteration",
                     &Options::update_state_every_iteration);
  MakeDataclass(PyOptions);

  using Summary = ceres::Solver::Summary;
  py::class_<Summary, std::shared_ptr<Summary>> PySummary(
      m, "SolverSummary", py::module_local());
  PySummary.def(py::init<>())
      .def(py::init<const Summary&>())
      .def("BriefReport", &Summary::BriefReport)
      .def("FullReport", &Summary::FullReport)
      .def("IsSolutionUsable", &Summary::IsSolutionUsable)
      .def_readwrite("minimizer_type", &Summary::minimizer_type)
      .def_readwrite("termination_type", &Summary::termination_type)
      .def_readwrite("message", &Summary::message)
      .def_readwrite("initial_cost", &Summary::initial_cost)
      .def_readwrite("final_cost", &Summary::final_cost)
      .def_readwrite("fixed_cost", &Summary::fixed_cost)
      .def_readwrite("num_successful_steps", &Summary::num_successful_steps)
      .def_readwrite("num_unsuccessful_steps", &Summary::num_unsuccessful_steps)
      .def_readwrite("num_inner_iteration_steps",
                     &Summary::num_inner_iteration_steps)
      .def_readwrite("num_line_search_steps", &Summary::num_line_search_steps)
      .def_readwrite("preprocessor_time_in_seconds",
                     &Summary::preprocessor_time_in_seconds)
      .def_readwrite("minimizer_time_in_seconds",
                     &Summary::minimizer_time_in_seconds)
      .def_readwrite("postprocessor_time_in_seconds",
                     &Summary::postprocessor_time_in_seconds)
      .def_readwrite("total_time_in_seconds", &Summary::total_time_in_seconds)
      .def_readwrite("linear_solver_time_in_seconds",
                     &Summary::linear_solver_time_in_seconds)
      .def_readwrite("num_linear_solves", &Summary::num_linear_solves)
      .def_readwrite("residual_evaluation_time_in_seconds",
                     &Summary::residual_evaluation_time_in_seconds)
      .def_readwrite("num_residual_evaluations",
                     &Summary::num_residual_evaluations)
      .def_readwrite("jacobian_evaluation_time_in_seconds",
                     &Summary::jacobian_evaluation_time_in_seconds)
      .def_readwrite("num_jacobian_evaluations",
                     &Summary::num_jacobian_evaluations)
      .def_readwrite("inner_iteration_time_in_seconds",
                     &Summary::inner_iteration_time_in_seconds)
      .def_readwrite("line_search_cost_evaluation_time_in_seconds",
                     &Summary::line_search_cost_evaluation_time_in_seconds)
      .def_readwrite("line_search_gradient_evaluation_time_in_seconds",
                     &Summary::line_search_gradient_evaluation_time_in_seconds)
      .def_readwrite(
          "line_search_polynomial_minimization_time_in_seconds",
          &Summary::line_search_polynomial_minimization_time_in_seconds)
      .def_readwrite("line_search_total_time_in_seconds",
                     &Summary::line_search_total_time_in_seconds)
      .def_readwrite("num_parameter_blocks", &Summary::num_parameter_blocks)
      .def_readwrite("num_parameters", &Summary::num_parameters)
      .def_readwrite("num_effective_parameters",
                     &Summary::num_effective_parameters)
      .def_readwrite("num_residual_blocks", &Summary::num_residual_blocks)
      .def_readwrite("num_residuals", &Summary::num_residuals)
      .def_readwrite("num_parameter_blocks_reduced",
                     &Summary::num_parameter_blocks_reduced)
      .def_readwrite("num_parameters_reduced", &Summary::num_parameters_reduced)
      .def_readwrite("num_effective_parameters_reduced",
                     &Summary::num_effective_parameters_reduced)
      .def_readwrite("num_residual_blocks_reduced",
                     &Summary::num_residual_blocks_reduced)
      .def_readwrite("num_residuals_reduced", &Summary::num_residuals_reduced)
      .def_readwrite("is_constrained", &Summary::is_constrained)
      .def_readwrite("num_threads_given", &Summary::num_threads_given)
      .def_readwrite("num_threads_used", &Summary::num_threads_used)
#if CERES_VERSION_MAJOR < 2
      .def_readwrite("num_linear_solver_threads_given",
                     &Summary::num_linear_solver_threads_given)
      .def_readwrite("num_linear_solver_threads_used",
                     &Summary::num_linear_solver_threads_used)
#endif
      .def_readwrite("linear_solver_type_given",
                     &Summary::linear_solver_type_given)
      .def_readwrite("linear_solver_type_used",
                     &Summary::linear_solver_type_used)
      .def_readwrite("schur_structure_given", &Summary::schur_structure_given)
      .def_readwrite("schur_structure_used", &Summary::schur_structure_used)
      .def_readwrite("inner_iterations_given", &Summary::inner_iterations_given)
      .def_readwrite("inner_iterations_used", &Summary::inner_iterations_used)
      .def_readwrite("preconditioner_type_given",
                     &Summary::preconditioner_type_given)
      .def_readwrite("preconditioner_type_used",
                     &Summary::preconditioner_type_used)
      .def_readwrite("visibility_clustering_type",
                     &Summary::visibility_clustering_type)
      .def_readwrite("trust_region_strategy_type",
                     &Summary::trust_region_strategy_type)
      .def_readwrite("dogleg_type", &Summary::dogleg_type)
      .def_readwrite("dense_linear_algebra_library_type",
                     &Summary::dense_linear_algebra_library_type)
      .def_readwrite("sparse_linear_algebra_library_type",
                     &Summary::sparse_linear_algebra_library_type)
      .def_readwrite("line_search_direction_type",
                     &Summary::line_search_direction_type)
      .def_readwrite("line_search_type", &Summary::line_search_type)
      .def_readwrite("line_search_interpolation_type",
                     &Summary::line_search_interpolation_type)
      .def_readwrite("nonlinear_conjugate_gradient_type",
                     &Summary::nonlinear_conjugate_gradient_type)
      .def_readwrite("max_lbfgs_rank", &Summary::max_lbfgs_rank);
  MakeDataclass(PySummary);

  using IterSummary = ceres::IterationSummary;
  py::class_<IterSummary, std::shared_ptr<IterSummary>> PyIterSummary(
      m, "IterationSummary", py::module_local());
  PyIterSummary.def(py::init<>())
      .def(py::init<const IterSummary&>())
      .def_readonly("iteration", &IterSummary::iteration)
      .def_readonly("step_is_valid", &IterSummary::step_is_valid)
      .def_readonly("step_is_nonmonotonic", &IterSummary::step_is_nonmonotonic)
      .def_readonly("step_is_successful", &IterSummary::step_is_successful)
      .def_readonly("cost", &IterSummary::cost)
      .def_readonly("cost_change", &IterSummary::cost_change)
      .def_readonly("gradient_max_norm", &IterSummary::gradient_max_norm)
      .def_readonly("gradient_norm", &IterSummary::gradient_norm)
      .def_readonly("step_norm", &IterSummary::step_norm)
      .def_readonly("relative_decrease", &IterSummary::relative_decrease)
      .def_readonly("trust_region_radius", &IterSummary::trust_region_radius)
      .def_readonly("eta", &IterSummary::eta)
      .def_readonly("step_size", &IterSummary::step_size)
      .def_readonly("line_search_function_evaluations",
                    &IterSummary::line_search_function_evaluations)
      .def_readonly("line_search_gradient_evaluations",
                    &IterSummary::line_search_gradient_evaluations)
      .def_readonly("line_search_iterations",
                    &IterSummary::line_search_iterations)
      .def_readonly("linear_solver_iterations",
                    &IterSummary::linear_solver_iterations)
      .def_readonly("iteration_time_in_seconds",
                    &IterSummary::iteration_time_in_seconds)
      .def_readonly("step_solver_time_in_seconds",
                    &IterSummary::step_solver_time_in_seconds)
      .def_readonly("cumulative_time_in_seconds",
                    &IterSummary::cumulative_time_in_seconds);
}

void BindCeres(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("pyceres");

  BindCeresTypes(m);
  BindCeresSolver(m);
}
