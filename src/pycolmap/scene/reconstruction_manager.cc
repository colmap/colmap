#include "colmap/scene/reconstruction_manager.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindReconstructionManager(py::module& m) {
  py::class_<ReconstructionManager, std::shared_ptr<ReconstructionManager>>(
      m, "ReconstructionManager")
      .def(py::init<>())
      .def("size", &ReconstructionManager::Size)
      .def("get",
           py::overload_cast<size_t>(&ReconstructionManager::Get),
           "idx"_a)
      .def("add", &ReconstructionManager::Add)
      .def("delete", &ReconstructionManager::Delete, "idx"_a)
      .def("clear", &ReconstructionManager::Clear)
      .def("read", &ReconstructionManager::Read, "path"_a)
      .def("write", &ReconstructionManager::Write, "path"_a);
}
