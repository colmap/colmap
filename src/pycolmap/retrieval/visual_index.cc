#include "colmap/retrieval/visual_index.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using VisualIndex = retrieval::VisualIndex<>;
using ImageScore = retrieval::ImageScore;
using namespace pybind11::literals;
namespace py = pybind11;

void BindVisualIndex(py::module& m) {
  auto PyImageScore = py::class_<ImageScore>(m, "ImageScore")
                          .def(py::init<>())
                          .def_readonly("image_id", &ImageScore::image_id)
                          .def_readonly("score", &ImageScore::score);
  MakeDataclass(PyImageScore);

  py::class_<VisualIndex, std::shared_ptr<VisualIndex>> PyVisualIndex(
      m, "VisualIndex");

  auto PyIndexOptions =
      py::class_<VisualIndex::IndexOptions>(PyVisualIndex, "IndexOptions")
          .def(py::init<>())
          .def_readwrite("num_neighbors",
                         &VisualIndex::IndexOptions::num_neighbors)
          .def_readwrite("num_checks", &VisualIndex::IndexOptions::num_checks)
          .def_readwrite("num_threads",
                         &VisualIndex::IndexOptions::num_threads);
  MakeDataclass(PyIndexOptions);

  auto PyQueryOptions =
      py::class_<VisualIndex::QueryOptions>(PyVisualIndex, "QueryOptions")
          .def(py::init<>())
          .def_readwrite("max_num_images",
                         &VisualIndex::QueryOptions::max_num_images)
          .def_readwrite("num_neighbors",
                         &VisualIndex::QueryOptions::num_neighbors)
          .def_readwrite("num_checks", &VisualIndex::QueryOptions::num_checks)
          .def_readwrite(
              "num_images_after_verification",
              &VisualIndex::QueryOptions::num_images_after_verification)
          .def_readwrite("num_threads",
                         &VisualIndex::QueryOptions::num_threads);
  MakeDataclass(PyQueryOptions);

  auto PyBuildOptions =
      py::class_<VisualIndex::BuildOptions>(PyVisualIndex, "BuildOptions")
          .def(py::init<>())
          .def_readwrite("num_visual_words",
                         &VisualIndex::BuildOptions::num_visual_words)
          .def_readwrite("branching", &VisualIndex::BuildOptions::branching)
          .def_readwrite("num_iterations",
                         &VisualIndex::BuildOptions::num_iterations)
          .def_readwrite("target_precision",
                         &VisualIndex::BuildOptions::target_precision)
          .def_readwrite("num_checks", &VisualIndex::BuildOptions::num_checks)
          .def_readwrite("num_threads",
                         &VisualIndex::BuildOptions::num_threads);
  MakeDataclass(PyBuildOptions);

  PyVisualIndex.def(py::init<>())
      .def("add", &VisualIndex::Add)
      .def("is_image_indexed", &VisualIndex::IsImageIndexed)
      .def("num_visual_words", &VisualIndex::NumVisualWords)
      .def("query",
           static_cast<void (VisualIndex::*)(
               const typename VisualIndex::QueryOptions&,
               const typename VisualIndex::DescType&,
               std::vector<ImageScore>*) const>(&VisualIndex::Query),
           py::call_guard<py::gil_scoped_release>())
      .def("query",
           static_cast<void (VisualIndex::*)(
               const typename VisualIndex::QueryOptions&,
               const typename VisualIndex::GeomType&,
               const typename VisualIndex::DescType&,
               std::vector<ImageScore>*) const>(&VisualIndex::Query),
           py::call_guard<py::gil_scoped_release>())
      .def("prepare",
           &VisualIndex::Prepare,
           py::call_guard<py::gil_scoped_release>())
      .def("build",
           &VisualIndex::Build,
           py::call_guard<py::gil_scoped_release>())
      .def("read", &VisualIndex::Read, py::call_guard<py::gil_scoped_release>())
      .def("write",
           &VisualIndex::Write,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const VisualIndex& self) {
        std::ostringstream ss;
        ss << "VisualIndex(num_visual_words=" << self.NumVisualWords() << ")";
        return ss.str();
      });
}
