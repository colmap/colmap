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
  auto PySyntheticMatchConfig =
      py::enum_<SyntheticDatasetOptions::MatchConfig>(
          m, "SyntheticDatasetMatchConfig")
          .value("EXHAUSTIVE", SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE)
          .value("CHAINED", SyntheticDatasetOptions::MatchConfig::CHAINED);
  AddStringToEnumConstructor(PySyntheticMatchConfig);

  auto PySyntheticDatasetOptions =
      py::class_<SyntheticDatasetOptions>(m, "SyntheticDatasetOptions")
          .def(py::init<>())
          .def_readwrite("num_rigs", &SyntheticDatasetOptions::num_rigs)
          .def_readwrite("num_cameras_per_rig",
                         &SyntheticDatasetOptions::num_cameras_per_rig)
          .def_readwrite("num_frames_per_rig",
                         &SyntheticDatasetOptions::num_frames_per_rig)
          .def_readwrite("num_points3D", &SyntheticDatasetOptions::num_points3D)
          .def_readwrite(
              "sensor_from_rig_translation_stddev",
              &SyntheticDatasetOptions::sensor_from_rig_translation_stddev)
          .def_readwrite("camera_width", &SyntheticDatasetOptions::camera_width)
          .def_readwrite("camera_height",
                         &SyntheticDatasetOptions::camera_height)
          .def_readwrite("camera_model_id",
                         &SyntheticDatasetOptions::camera_model_id)
          .def_readwrite("camera_params",
                         &SyntheticDatasetOptions::camera_params)
          .def_readwrite("num_points2D_without_point3D",
                         &SyntheticDatasetOptions::num_points2D_without_point3D)
          .def_readwrite("point2D_stddev",
                         &SyntheticDatasetOptions::point2D_stddev)
          .def_readwrite("match_config", &SyntheticDatasetOptions::match_config)
          .def_readwrite("use_prior_position",
                         &SyntheticDatasetOptions::use_prior_position)
          .def_readwrite("use_geographic_coords_prior",
                         &SyntheticDatasetOptions::use_geographic_coords_prior)
          .def_readwrite("prior_position_stddev",
                         &SyntheticDatasetOptions::prior_position_stddev);
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
}
