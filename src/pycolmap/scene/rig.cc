#include "colmap/scene/rig.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
namespace py = pybind11;
using namespace pybind11::literals;

void BindSceneRig(py::module& m) {
  py::class_ext_<RigConfig::RigCamera, std::shared_ptr<RigConfig::RigCamera>>
      PyRigConfigCamera(m, "RigConfigCamera");
  PyRigConfigCamera.def(py::init<>())
      .def_readwrite("ref_sensor", &RigConfig::RigCamera::ref_sensor)
      .def_readwrite("image_prefix", &RigConfig::RigCamera::image_prefix)
      .def_readwrite("cam_from_rig", &RigConfig::RigCamera::cam_from_rig)
      .def_readwrite("camera", &RigConfig::RigCamera::camera);
  MakeDataclass(PyRigConfigCamera);

  py::class_ext_<RigConfig, std::shared_ptr<RigConfig>> PyRigConfig(
      m, "RigConfig");
  PyRigConfig.def(py::init<>()).def_readwrite("cameras", &RigConfig::cameras);
  MakeDataclass(PyRigConfig);

  m.def("read_rig_config",
        &ReadRigConfig,
        "path"_a,
        "Read the rig configuration from a .json file.");
  m.def("apply_rig_config",
        &ApplyRigConfig,
        "configs"_a,
        "database"_a,
        "reconstruction"_a = py::none(),
        "Applies the given rig configuration to the database and optionally "
        "derives camera rig extrinsics and intrinsics from the reconstruction, "
        "if not defined in the config. If the reconstruction is provided, it "
        "is also updated with the provided config and any previous rigs/frames "
        "are cleared and overwritten.");
}
