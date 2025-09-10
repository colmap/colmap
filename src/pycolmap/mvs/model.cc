#include "colmap/mvs/model.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindMVSModel(py::module& m) {
    py::classh<mvs::Model> PyMVSModel(m, "MVSModel");
    PyMVSModel.def(py::init<>())
    .def("read",
         &mvs::Model::Read,
         "path"_a,
         "format"_a,
         "Read the model from the given path in the specified format.")
    .def("read_from_colmap",
         &mvs::Model::ReadFromCOLMAP,
         "path"_a,
         "sparse_path"_a = "sparse",
         "images_path"_a = "images",
         "Read the model from a COLMAP reconstruction.")
    .def("read_from_pmvs",
         &mvs::Model::ReadFromPMVS,
         "path"_a,
         "Read the model from PMVS output.")
    .def("get_image_idx",
         &mvs::Model::GetImageIdx,
         "name"_a,
         "Get the image index for the given image name.")
    .def("get_image_name",
         &mvs::Model::GetImageName,
         "image_idx"_a,
         "Get the image name for the given image index.")
    .def("get_max_overlapping_images",
         &mvs::Model::GetMaxOverlappingImages,
         "num_images"_a,
         "min_triangulation_angle"_a,
         "Determine maximally overlapping images for each image, "
         "sorted by number of shared points subject to a minimum "
         "triangulation angle.")
    .def("get_max_overlapping_images_from_pmvs",
         &mvs::Model::GetMaxOverlappingImagesFromPMVS,
         py::return_value_policy::reference_internal,
         "Get overlapping images defined in the PMVS vis.dat file.")
    .def("compute_depth_ranges",
         &mvs::Model::ComputeDepthRanges,
         "Compute robust minimum and maximum depths from the sparse point cloud.")
    .def("compute_shared_points",
         &mvs::Model::ComputeSharedPoints,
         "Compute the number of shared points between all overlapping images.")
    .def("compute_triangulation_angles",
         &mvs::Model::ComputeTriangulationAngles,
         "percentile"_a = 50.0f,
         "Compute the median triangulation angles between all overlapping images.")
    .def("__repr__",
         [](const mvs::Model& self) {
             std::ostringstream ss;
             ss << "MVSModel(num_images=" << self.images.size()
                << ", num_points=" << self.points.size() << ")";
             return ss.str();
         });
}
