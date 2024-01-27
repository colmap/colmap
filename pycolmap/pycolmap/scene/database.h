#pragma once

#include "colmap/scene/database.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabase(py::module& m) {
  py::class_<Database> PyDatabase(m, "Database");
  PyDatabase.def(py::init<const std::string&>(), "path"_a)
      .def("open", &Database::Open, "path"_a)
      .def("close", &Database::Close)
      .def_property_readonly("num_cameras", &Database::NumCameras)
      .def_property_readonly("num_images", &Database::NumImages)
      .def_property_readonly("num_keypoints", &Database::NumKeypoints)
      .def_property_readonly("num_keypoints_for_image",
                             &Database::NumKeypointsForImage)
      .def_property_readonly("num_descriptors", &Database::NumDescriptors)
      .def_property_readonly("num_descriptors_for_image",
                             &Database::NumDescriptorsForImage)
      .def_property_readonly("num_matches", &Database::NumMatches)
      .def_property_readonly("num_inlier_matches", &Database::NumInlierMatches)
      .def_property_readonly("num_matched_image_pairs",
                             &Database::NumMatchedImagePairs)
      .def_property_readonly("num_verified_image_pairs",
                             &Database::NumVerifiedImagePairs)
      .def("image_pair_to_pair_id", &Database::ImagePairToPairId)
      .def("pair_id_to_image_pair", &Database::PairIdToImagePair)
      .def("read_camera", &Database::ReadCamera)
      .def("read_all_cameras", &Database::ReadAllCameras)
      .def("read_image", &Database::ReadImage)
      .def("read_image_with_name", &Database::ReadImageWithName)
      .def("read_all_images", &Database::ReadAllImages)
      .def("read_two_view_geometry", &Database::ReadTwoViewGeometry)
      .def("write_camera", &Database::WriteCamera)
      .def("write_image", &Database::WriteImage);

  py::class_<DatabaseTransaction>(m, "DatabaseTransaction")
      .def(py::init<Database*>(), "database"_a);
}
