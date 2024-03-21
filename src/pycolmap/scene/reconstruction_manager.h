#pragma once

#include "colmap/scene/reconstruction_manager.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindReconstructionManager(py::module& m) {
  py::class_<ReconstructionManager, std::shared_ptr<ReconstructionManager>>(m,
                                                              "ReconstructionManager")
      .def(py::init<>())
      .def("Size", &ReconstructionManager::Size)
      .def("Get", [](ReconstructionManager& self, size_t idx) {
          std::shared_ptr<const Reconstruction> rec = self.Get(idx);
          return rec;
          }, "idx"_a)
      .def("Add", &ReconstructionManager::Add)
      .def("Delete", &ReconstructionManager::Delete, "idx"_a)
      .def("Clear", &ReconstructionManager::Clear)
      .def("Read", &ReconstructionManager::Read, "path"_a)
      .def("Write", &ReconstructionManager::Write, "path"_a);
}
