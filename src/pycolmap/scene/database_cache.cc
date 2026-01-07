#include "colmap/scene/database_cache.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabaseCache(py::module& m) {
  py::class_<DatabaseCache::Options>(m, "DatabaseCacheOptions")
      .def(py::init<>())
      .def_readwrite("min_num_matches",
                     &DatabaseCache::Options::min_num_matches)
      .def_readwrite("ignore_watermarks",
                     &DatabaseCache::Options::ignore_watermarks)
      .def_readwrite("image_names", &DatabaseCache::Options::image_names)
      .def_readwrite("load_relative_pose",
                     &DatabaseCache::Options::load_relative_pose);

  py::classh<DatabaseCache> PyDatabaseCache(m, "DatabaseCache");
  PyDatabaseCache.def(py::init<>())
      .def_static(
          "create",
          [](const Database& database,
             size_t min_num_matches,
             bool ignore_watermarks,
             const std::unordered_set<std::string>& image_names,
             bool load_relative_pose) {
            DatabaseCache::Options options;
            options.min_num_matches = min_num_matches;
            options.ignore_watermarks = ignore_watermarks;
            options.image_names = image_names;
            options.load_relative_pose = load_relative_pose;
            return DatabaseCache::Create(database, options);
          },
          "database"_a,
          "min_num_matches"_a = 0,
          "ignore_watermarks"_a = false,
          "image_names"_a = std::unordered_set<std::string>{},
          "load_relative_pose"_a = false)
      .def_static(
          "create_from_cache",
          [](const DatabaseCache& database_cache,
             size_t min_num_matches,
             const std::unordered_set<std::string>& image_names,
             bool load_relative_pose) {
            DatabaseCache::Options options;
            options.min_num_matches = min_num_matches;
            options.image_names = image_names;
            options.load_relative_pose = load_relative_pose;
            return DatabaseCache::CreateFromCache(database_cache, options);
          },
          "database_cache"_a,
          "min_num_matches"_a = 0,
          "image_names"_a = std::unordered_set<std::string>{},
          "load_relative_pose"_a = false)
      .def(
          "load",
          [](DatabaseCache& self,
             const Database& database,
             size_t min_num_matches,
             bool ignore_watermarks,
             const std::unordered_set<std::string>& image_names,
             bool load_relative_pose) {
            DatabaseCache::Options options;
            options.min_num_matches = min_num_matches;
            options.ignore_watermarks = ignore_watermarks;
            options.image_names = image_names;
            options.load_relative_pose = load_relative_pose;
            self.Load(database, options);
          },
          "database"_a,
          "min_num_matches"_a = 0,
          "ignore_watermarks"_a = false,
          "image_names"_a = std::unordered_set<std::string>{},
          "load_relative_pose"_a = false)
      .def("add_rig", &DatabaseCache::AddRig)
      .def("add_camera", &DatabaseCache::AddCamera)
      .def("add_frame", &DatabaseCache::AddFrame)
      .def("add_image", &DatabaseCache::AddImage)
      .def("num_rigs", &DatabaseCache::NumRigs)
      .def("num_cameras", &DatabaseCache::NumCameras)
      .def("num_frames", &DatabaseCache::NumFrames)
      .def("num_images", &DatabaseCache::NumImages)
      .def("num_relative_poses", &DatabaseCache::NumRelativePoses)
      .def("exists_rig", &DatabaseCache::ExistsRig, "rig_id"_a)
      .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
      .def("exists_frame", &DatabaseCache::ExistsFrame, "frame_id"_a)
      .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
      .def("exists_relative_pose",
           &DatabaseCache::ExistsRelativePose,
           "image_id1"_a,
           "image_id2"_a)
      .def("relative_pose",
           &DatabaseCache::RelativePose,
           "image_id1"_a,
           "image_id2"_a)
      .def_property_readonly("rigs", &DatabaseCache::Rigs)
      .def_property_readonly("cameras", &DatabaseCache::Cameras)
      .def_property_readonly("frames", &DatabaseCache::Frames)
      .def_property_readonly("images", &DatabaseCache::Images)
      .def_property_readonly("relative_poses", &DatabaseCache::RelativePoses)
      .def_property_readonly("correspondence_graph",
                             &DatabaseCache::CorrespondenceGraph)
      .def("find_image_with_name", &DatabaseCache::FindImageWithName, "name"_a);
}
