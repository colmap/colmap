#include "colmap/estimators/bundle_adjustment.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindBundleAdjuster(py::module& m) {
  IsPyceresAvailable();  // Try to import pyceres to populate the docstrings.

  using BACfg = BundleAdjustmentConfig;
  py::class_<BACfg> PyBundleAdjustmentConfig(m, "BundleAdjustmentConfig");
  PyBundleAdjustmentConfig.def(py::init<>())
      .def("num_images", &BACfg::NumImages)
      .def("num_points", &BACfg::NumPoints)
      .def("num_constant_cam_intrinsics", &BACfg::NumConstantCamIntrinsics)
      .def("num_constant_cam_poses", &BACfg::NumConstantCamPoses)
      .def("num_constant_cam_positions", &BACfg::NumConstantCamPositions)
      .def("num_variable_points", &BACfg::NumVariablePoints)
      .def("num_constant_points", &BACfg::NumConstantPoints)
      .def("num_residuals", &BACfg::NumResiduals, "reconstruction"_a)
      .def("add_image", &BACfg::AddImage, "image_id"_a)
      .def("has_image", &BACfg::HasImage, "image_id"_a)
      .def("remove_image", &BACfg::RemoveImage, "image_id"_a)
      .def("set_constant_cam_intrinsics",
           &BACfg::SetConstantCamIntrinsics,
           "camera_id"_a)
      .def("set_variable_cam_intrinsics",
           &BACfg::SetVariableCamIntrinsics,
           "camera_id"_a)
      .def("has_constant_cam_intrinsics",
           &BACfg::HasConstantCamIntrinsics,
           "camera_id"_a)
      .def("set_constant_cam_pose", &BACfg::SetConstantCamPose, "image_id"_a)
      .def("set_variable_cam_pose", &BACfg::SetVariableCamPose, "image_id"_a)
      .def("has_constant_cam_pose", &BACfg::HasConstantCamPose, "image_id"_a)
      .def("set_constant_cam_positions",
           &BACfg::SetConstantCamPositions,
           "image_id"_a,
           "idxs"_a)
      .def("remove_variable_cam_positions",
           &BACfg::RemoveConstantCamPositions,
           "image_id"_a)
      .def("has_constant_cam_positions",
           &BACfg::HasConstantCamPositions,
           "image_id"_a)
      .def("add_variable_point", &BACfg::AddVariablePoint, "point3D_id"_a)
      .def("add_constant_point", &BACfg::AddConstantPoint, "point3D_id"_a)
      .def("has_point", &BACfg::HasPoint, "point3D_id"_a)
      .def("has_variable_point", &BACfg::HasVariablePoint, "point3D_id"_a)
      .def("has_constant_point", &BACfg::HasConstantPoint, "point3D_id"_a)
      .def("remove_variable_point", &BACfg::RemoveVariablePoint, "point3D_id"_a)
      .def("remove_constant_point", &BACfg::RemoveConstantPoint, "point3D_id"_a)
      .def_property_readonly("constant_intrinsics", &BACfg::ConstantIntrinsics)
      .def_property_readonly("image_ids", &BACfg::Images)
      .def_property_readonly("variable_point3D_ids", &BACfg::VariablePoints)
      .def_property_readonly("constant_point3D_ids", &BACfg::ConstantPoints)
      .def_property_readonly("constant_cam_poses", &BACfg::ConstantCamPoses)
      .def(
          "constant_cam_positions", &BACfg::ConstantCamPositions, "image_id"_a);
  MakeDataclass(PyBundleAdjustmentConfig);

  using BAOpts = BundleAdjustmentOptions;
  auto PyBALossFunctionType =
      py::enum_<BAOpts::LossFunctionType>(m, "LossFunctionType")
          .value("TRIVIAL", BAOpts::LossFunctionType::TRIVIAL)
          .value("SOFT_L1", BAOpts::LossFunctionType::SOFT_L1)
          .value("CAUCHY", BAOpts::LossFunctionType::CAUCHY);
  AddStringToEnumConstructor(PyBALossFunctionType);

  auto PyBundleAdjustmentOptions =
      py::class_<BAOpts>(m, "BundleAdjustmentOptions")
          .def(py::init<>())
          .def("create_loss_function", &BAOpts::CreateLossFunction)
          .def("create_solver_options",
               &BAOpts::CreateSolverOptions,
               "config"_a,
               "problem"_a)
          .def_readwrite("loss_function_type",
                         &BAOpts::loss_function_type,
                         "Loss function types: Trivial (non-robust) and Cauchy "
                         "(robust) loss.")
          .def_readwrite("loss_function_scale",
                         &BAOpts::loss_function_scale,
                         "Scaling factor determines residual at which "
                         "robustification takes place.")
          .def_readwrite("refine_focal_length",
                         &BAOpts::refine_focal_length,
                         "Whether to refine the focal length parameter group.")
          .def_readwrite(
              "refine_principal_point",
              &BAOpts::refine_principal_point,
              "Whether to refine the principal point parameter group.")
          .def_readwrite("refine_extra_params",
                         &BAOpts::refine_extra_params,
                         "Whether to refine the extra parameter group.")
          .def_readwrite("refine_extrinsics",
                         &BAOpts::refine_extrinsics,
                         "Whether to refine the extrinsic parameter group.")
          .def_readwrite("print_summary",
                         &BAOpts::print_summary,
                         "Whether to print a final summary.")
          .def_readwrite("use_gpu",
                         &BAOpts::use_gpu,
                         "Whether to use Ceres' CUDA linear algebra library, "
                         "if available.")
          .def_readwrite("gpu_index",
                         &BAOpts::gpu_index,
                         "Which GPU to use for solving the problem.")
          .def_readwrite(
              "min_num_residuals_for_cpu_multi_threading",
              &BAOpts::min_num_residuals_for_cpu_multi_threading,
              "Minimum number of residuals to enable multi-threading. Note "
              "that single-threaded is typically better for small bundle "
              "adjustment problems due to the overhead of threading.")
          .def_readwrite("min_num_images_gpu_solver",
                         &BAOpts::min_num_images_gpu_solver,
                         "Minimum number of images to use the GPU solver.")
          .def_readwrite("max_num_images_direct_dense_cpu_solver",
                         &BAOpts::max_num_images_direct_dense_cpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_sparse_cpu_solver",
                         &BAOpts::max_num_images_direct_sparse_cpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_dense_gpu_solver",
                         &BAOpts::max_num_images_direct_dense_gpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("max_num_images_direct_sparse_gpu_solver",
                         &BAOpts::max_num_images_direct_sparse_gpu_solver,
                         "Threshold to switch between direct, sparse, and "
                         "iterative solvers.")
          .def_readwrite("solver_options",
                         &BAOpts::solver_options,
                         "Options for the Ceres solver. Using this member "
                         "requires having PyCeres installed.");
  MakeDataclass(PyBundleAdjustmentOptions);

  using PosePriorBAOpts = PosePriorBundleAdjustmentOptions;
  auto PyPosePriorBundleAdjustmentOptions =
      py::class_<PosePriorBAOpts>(m, "PosePriorBundleAdjustmentOptions")
          .def(py::init<>())
          .def_readwrite("use_robust_loss_on_prior_position",
                         &PosePriorBAOpts::use_robust_loss_on_prior_position,
                         "Whether to use a robust loss on prior locations.")
          .def_readwrite("prior_position_loss_scale",
                         &PosePriorBAOpts::prior_position_loss_scale,
                         "Threshold on the residual for the robust loss (chi2 "
                         "for 3DOF at 95% = 7.815).")
          .def_readwrite("ransac_max_error",
                         &PosePriorBAOpts::ransac_max_error,
                         "Maximum RANSAC error for Sim3 alignment.");
  MakeDataclass(PyPosePriorBundleAdjustmentOptions);

  class PyBundleAdjuster : public BundleAdjuster {
   public:
    PyBundleAdjuster(BundleAdjustmentOptions options,
                     BundleAdjustmentConfig config)
        : BundleAdjuster(std::move(options), std::move(config)) {}

    ceres::Solver::Summary Solve() override {
      PYBIND11_OVERRIDE_PURE(ceres::Solver::Summary, BundleAdjuster, Solve);
    }

    std::shared_ptr<ceres::Problem>& Problem() override {
      PYBIND11_OVERRIDE_PURE(
          std::shared_ptr<ceres::Problem>&, BundleAdjuster, Problem);
    }
  };

  py::class_<BundleAdjuster, PyBundleAdjuster>(m, "BundleAdjuster")
      .def(py::init<BundleAdjustmentOptions, BundleAdjustmentConfig>(),
           "options"_a,
           "config"_a)
      .def("solve", &BundleAdjuster::Solve)
      .def_property_readonly("problem", &BundleAdjuster::Problem)
      .def_property_readonly("options", &BundleAdjuster::Options)
      .def_property_readonly("config", &BundleAdjuster::Config);

  m.def("create_default_bundle_adjuster",
        CreateDefaultBundleAdjuster,
        "options"_a,
        "config"_a,
        "reconstruction"_a);

  m.def("create_pose_prior_bundle_adjuster",
        CreatePosePriorBundleAdjuster,
        "options"_a,
        "prior_options"_a,
        "config"_a,
        "pose_priors"_a,
        "reconstruction"_a);
}
