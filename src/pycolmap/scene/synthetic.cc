#include "colmap/scene/synthetic.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindSynthetic(py::module& m) {
  auto PyBaseSyntheticOptions =
      py::classh<BaseSyntheticOptions>(m, "BaseSyntheticOptions")
          .def(py::init<>())
          .def_readwrite("num_rigs", &BaseSyntheticOptions::num_rigs)
          .def_readwrite("num_cameras_per_rig",
                         &BaseSyntheticOptions::num_cameras_per_rig)
          .def_readwrite("num_frames_per_rig",
                         &BaseSyntheticOptions::num_frames_per_rig)
          .def_readwrite(
              "sensor_from_rig_translation_stddev",
              &BaseSyntheticOptions::sensor_from_rig_translation_stddev)
          .def_readwrite(
              "sensor_from_rig_rotation_stddev",
              &BaseSyntheticOptions::sensor_from_rig_rotation_stddev,
              "Random rotation in degrees around the z-axis of the sensor.")
          .def_readwrite("camera_width", &BaseSyntheticOptions::camera_width)
          .def_readwrite("camera_height", &BaseSyntheticOptions::camera_height)
          .def_readwrite("camera_model_id",
                         &BaseSyntheticOptions::camera_model_id)
          .def_readwrite("camera_params", &BaseSyntheticOptions::camera_params)
          .def_readwrite("match_config", &BaseSyntheticOptions::match_config)
          .def_readwrite("match_sparsity",
                         &BaseSyntheticOptions::match_sparsity,
                         "Sparsity parameter for SPARSE match config [0,1].")
          .def_readwrite("prior_position",
                         &BaseSyntheticOptions::prior_position)
          .def_readwrite("prior_gravity", &BaseSyntheticOptions::prior_gravity)
          .def_readwrite(
              "prior_position_coordinate_system",
              &BaseSyntheticOptions::prior_position_coordinate_system)
          .def_readwrite("prior_gravity_in_world",
                         &BaseSyntheticOptions::prior_gravity_in_world,
                         "Prior gravity direction in world coordinates.");
  MakeDataclass(PyBaseSyntheticOptions);

  auto PySyntheticMatchConfig =
      py::enum_<BaseSyntheticOptions::MatchConfig>(
          m, "SyntheticDatasetMatchConfig")
          .value("EXHAUSTIVE", BaseSyntheticOptions::MatchConfig::EXHAUSTIVE)
          .value("CHAINED", BaseSyntheticOptions::MatchConfig::CHAINED)
          .value("SPARSE", BaseSyntheticOptions::MatchConfig::SPARSE);
  AddStringToEnumConstructor(PySyntheticMatchConfig);

  auto PySyntheticDatasetOptions =
      py::classh<SyntheticDatasetOptions, BaseSyntheticOptions>(
          m, "SyntheticDatasetOptions")
          .def(py::init<>())
          .def_readwrite("feature_type",
                         &SyntheticDatasetOptions::feature_type,
                         "The type of feature descriptors to synthesize.")
          .def_readwrite("num_points3D", &SyntheticDatasetOptions::num_points3D)
          .def_readwrite("track_length",
                         &SyntheticDatasetOptions::track_length,
                         "Target track length per 3D point. -1 = dense "
                         "visibility (default), >= 2 = pruned observations.")
          .def_readwrite(
              "camera_has_prior_focal_length",
              &SyntheticDatasetOptions::camera_has_prior_focal_length)
          .def_readwrite("num_points2D_without_point3D",
                         &SyntheticDatasetOptions::num_points2D_without_point3D)
          .def_readwrite("inlier_match_ratio",
                         &SyntheticDatasetOptions::inlier_match_ratio)
          .def_readwrite(
              "two_view_geometry_has_relative_pose",
              &SyntheticDatasetOptions::two_view_geometry_has_relative_pose,
              "Whether to include decomposed relative poses in two-view "
              "geometries.");
  MakeDataclass(PySyntheticDatasetOptions);

  m.def(
      "synthesize_dataset",
      [](const SyntheticDatasetOptions& options, Database* database = nullptr) {
        Reconstruction reconstruction;
        SynthesizeDataset(options, &reconstruction, database);
        return reconstruction;
      },
      "options"_a,
      "database"_a = py::none());

  auto PySyntheticNoiseOptions =
      py::classh<SyntheticNoiseOptions>(m, "SyntheticNoiseOptions")
          .def(py::init<>())
          .def_readwrite(
              "rig_from_world_translation_stddev",
              &SyntheticNoiseOptions::rig_from_world_translation_stddev)
          .def_readwrite(
              "rig_from_world_rotation_stddev",
              &SyntheticNoiseOptions::rig_from_world_rotation_stddev,
              "Random rotation in degrees around the z-axis of the rig.")
          .def_readwrite("point3D_stddev",
                         &SyntheticNoiseOptions::point3D_stddev)
          .def_readwrite("point2D_stddev",
                         &SyntheticNoiseOptions::point2D_stddev)
          .def_readwrite("prior_position_stddev",
                         &SyntheticNoiseOptions::prior_position_stddev)
          .def_readwrite("prior_gravity_stddev",
                         &SyntheticNoiseOptions::prior_gravity_stddev);
  MakeDataclass(PySyntheticNoiseOptions);

  m.def("synthesize_noise",
        &SynthesizeNoise,
        "options"_a,
        "reconstruction"_a,
        "database"_a = py::none());

  auto PySyntheticImageOptions =
      py::classh<SyntheticImageOptions>(m, "SyntheticImageOptions")
          .def(py::init<>())
          .def_readwrite("feature_peak_radius",
                         &SyntheticImageOptions::feature_peak_radius)
          .def_readwrite(
              "feature_patch_radius",
              &SyntheticImageOptions::feature_patch_radius,
              "Random rotation in degrees around the z-axis of the rig.")
          .def_readwrite("feature_patch_max_brightness",
                         &SyntheticImageOptions::feature_patch_max_brightness);
  MakeDataclass(PySyntheticImageOptions);

  m.def("synthesize_images",
        &SynthesizeImages,
        "options"_a,
        "reconstruction"_a,
        "image_path"_a);

  auto PySyntheticPoseGraphData =
      py::classh<SyntheticPoseGraphData>(m, "SyntheticPoseGraphData")
          .def(py::init<>())
          .def_readwrite("reconstruction",
                         &SyntheticPoseGraphData::reconstruction)
          .def_readwrite("pose_graph", &SyntheticPoseGraphData::pose_graph)
          .def_readwrite("pose_priors", &SyntheticPoseGraphData::pose_priors);
  MakeDataclass(PySyntheticPoseGraphData);

  m.attr("SyntheticPoseGraphOptions") = m.attr("BaseSyntheticOptions");

  m.def("synthesize_pose_graph", &SynthesizePoseGraph, "options"_a);

  auto PySyntheticPoseGraphNoiseOptions =
      py::classh<SyntheticPoseGraphNoiseOptions>(
          m, "SyntheticPoseGraphNoiseOptions")
          .def(py::init<>())
          .def_readwrite(
              "rel_rotation_noise_deg",
              &SyntheticPoseGraphNoiseOptions::rel_rotation_noise_deg)
          .def_readwrite(
              "rel_translation_noise_deg",
              &SyntheticPoseGraphNoiseOptions::rel_translation_noise_deg)
          .def_readwrite("prior_position_stddev",
                         &SyntheticPoseGraphNoiseOptions::prior_position_stddev)
          .def_readwrite("prior_gravity_stddev",
                         &SyntheticPoseGraphNoiseOptions::prior_gravity_stddev);
  MakeDataclass(PySyntheticPoseGraphNoiseOptions);

  m.def("synthesize_pose_graph_noise",
        &SynthesizePoseGraphNoise,
        "options"_a,
        "data"_a);
}
