#include "colmap/scene/database_cache.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabaseCache(py::module& m) {
  py::classh<DatabaseCache> PyDatabaseCache(m, "DatabaseCache");
  PyDatabaseCache.def(py::init<>())
      .def_static("create",
                  py::overload_cast<const Database&,
                                    size_t,
                                    bool,
                                    const std::unordered_set<std::string>&>(
                      &DatabaseCache::Create),
                  "database"_a,
                  "min_num_matches"_a = 0,
                  "ignore_watermarks"_a = false,
                  "image_names"_a = std::unordered_set<std::string>{})
      .def_static("create_from_cache",
                  py::overload_cast<const DatabaseCache&,
                                    size_t,
                                    const std::unordered_set<std::string>&>(
                      &DatabaseCache::Create),
                  "database_cache"_a,
                  "min_num_matches"_a,
                  "image_names"_a)
      .def("load",
           &DatabaseCache::Load,
           "database"_a,
           "min_num_matches"_a = 0,
           "ignore_watermarks"_a = false,
           "image_names"_a = std::unordered_set<std::string>{})
      .def("add_rig", &DatabaseCache::AddRig)
      .def("add_camera", &DatabaseCache::AddCamera)
      .def("add_frame", &DatabaseCache::AddFrame)
      .def("add_image", &DatabaseCache::AddImage)
      .def("num_rigs", &DatabaseCache::NumRigs)
      .def("num_cameras", &DatabaseCache::NumCameras)
      .def("num_frames", &DatabaseCache::NumFrames)
      .def("num_images", &DatabaseCache::NumImages)
      .def("exists_rig", &DatabaseCache::ExistsRig, "rig_id"_a)
      .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
      .def("exists_frame", &DatabaseCache::ExistsFrame, "frame_id"_a)
      .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
      .def_property_readonly("rigs", &DatabaseCache::Rigs)
      .def_property_readonly("cameras", &DatabaseCache::Cameras)
      .def_property_readonly("frames", &DatabaseCache::Frames)
      .def_property_readonly("images", &DatabaseCache::Images)
      .def_property_readonly("correspondence_graph",
                             &DatabaseCache::CorrespondenceGraph)
      .def("find_image_with_name", &DatabaseCache::FindImageWithName, "name"_a);
}
