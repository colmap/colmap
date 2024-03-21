#pragma once

#include "colmap/scene/database_cache.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabaseCache(py::module& m) {
  py::class_<DatabaseCache> PyDatabaseCache(m, "DatabaseCache");
  PyDatabaseCache.def(py::init<>())
    .def_static("Create", &DatabaseCache::Create, "database"_a, "min_num_matches"_a, "ignore_watermarks"_a, "image_names"_a)
    .def("NumCameras", &DatabaseCache::NumCameras)
    .def("NumImages", &DatabaseCache::NumImages)
    .def("Camera", [](DatabaseCache& self, camera_t camera_id) {
          const struct Camera& camera = self.Camera(camera_id);
          return camera;
        }, "camera_id"_a)
    .def("Image", [](DatabaseCache& self, image_t image_id) {
          const struct Image& image = self.Image(image_id);
          return image;
        }, "image_id"_a)
    .def("ExistsCamera", &DatabaseCache::ExistsCamera, "camera_id"_a)
    .def("ExistsImage", &DatabaseCache::ExistsImage, "image_id"_a)
    .def("Cameras", &DatabaseCache::Cameras)
    .def("Images", &DatabaseCache::Images)
    .def("CorrespondenceGraph", &DatabaseCache::CorrespondenceGraph)
    .def("FindImageWithName", &DatabaseCache::FindImageWithName);
}
