// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/two_view_pose_salvage.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTwoViewPoseSalvageEstimator(py::module& m) {
  auto PyTwoViewPoseSalvageOptions =
      py::classh<TwoViewPoseSalvageOptions>(m, "TwoViewPoseSalvageOptions")
          .def(py::init<>())
          .def_readwrite("num_threads", &TwoViewPoseSalvageOptions::num_threads)
          .def_readwrite("random_seed", &TwoViewPoseSalvageOptions::random_seed)
          .def_readwrite("max_epipolar_error_px",
                         &TwoViewPoseSalvageOptions::max_epipolar_error_px)
          .def_readwrite("min_num_inliers",
                         &TwoViewPoseSalvageOptions::min_num_inliers)
          .def_readwrite("min_tri_angle_deg",
                         &TwoViewPoseSalvageOptions::min_tri_angle_deg)
          .def_readwrite("max_num_trials",
                         &TwoViewPoseSalvageOptions::max_num_trials)
          .def_readwrite("min_num_trials",
                         &TwoViewPoseSalvageOptions::min_num_trials)
          .def_readwrite("confidence", &TwoViewPoseSalvageOptions::confidence)
          .def_readwrite("sample_rotation_sigma_deg",
                         &TwoViewPoseSalvageOptions::sample_rotation_sigma_deg)
          .def_readwrite("gate_significance",
                         &TwoViewPoseSalvageOptions::gate_significance)
          .def_readwrite("posterior_significance",
                         &TwoViewPoseSalvageOptions::posterior_significance)
          .def_readwrite("covariance_options",
                         &TwoViewPoseSalvageOptions::covariance_options);
  MakeDataclass(PyTwoViewPoseSalvageOptions);

  py::classh<SalvagedTwoViewPose>(m, "SalvagedTwoViewPose")
      .def(py::init<>())
      .def_readwrite("geometry", &SalvagedTwoViewPose::geometry)
      .def_readwrite("cam2_from_cam1_rotation_cov",
                     &SalvagedTwoViewPose::cam2_from_cam1_rotation_cov)
      .def_readwrite("posterior_statistic",
                     &SalvagedTwoViewPose::posterior_statistic)
      .def_readwrite("posterior_p_value",
                     &SalvagedTwoViewPose::posterior_p_value);

  m.def(
      "salvage_two_view_pose",
      [](const Camera& camera1,
         const std::vector<Eigen::Vector2d>& points1,
         const Camera& camera2,
         const std::vector<Eigen::Vector2d>& points2,
         const FeatureMatchesMatrix& matches_mat,
         const Eigen::Quaterniond& prior_cam2_from_cam1_rotation,
         const Eigen::Matrix3d& prior_cam2_from_cam1_rotation_cov,
         const TwoViewPoseSalvageOptions& options,
         const double variance_factor) {
        py::gil_scoped_release release;
        const FeatureMatches matches = MatchesFromMatrix(matches_mat);
        return SalvageTwoViewPose(camera1,
                                  points1,
                                  camera2,
                                  points2,
                                  matches,
                                  prior_cam2_from_cam1_rotation,
                                  prior_cam2_from_cam1_rotation_cov,
                                  options,
                                  variance_factor);
      },
      "camera1"_a,
      "points1"_a,
      "camera2"_a,
      "points2"_a,
      "matches"_a,
      "prior_cam2_from_cam1_rotation"_a,
      "prior_cam2_from_cam1_rotation_cov"_a,
      py::arg_v(
          "options", TwoViewPoseSalvageOptions(), "TwoViewPoseSalvageOptions()"),
      "variance_factor"_a = 1.0,
      "Attempt to salvage a calibrated two-view relative pose using a "
      "leave-one-out rotation prior and its covariance.");

  m.def("salvage_two_view_poses",
        &SalvageTwoViewPoses,
        "options"_a,
        "rotation_options"_a,
        "database_cache"_a,
        "reconstruction"_a,
        "pose_graph"_a,
        "correspondence_graph"_a,
        py::call_guard<py::gil_scoped_release>(),
        "Attempt to salvage invalid pose graph edges and unverified candidate "
        "match pairs between registered images.");
}
