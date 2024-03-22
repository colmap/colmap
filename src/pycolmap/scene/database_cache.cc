#include "colmap/scene/database_cache.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabaseCache(py::module& m) {
  py::class_<DatabaseCache, std::shared_ptr<DatabaseCache>> PyDatabaseCache(
      m, "DatabaseCache");
  PyDatabaseCache.def(py::init<>())
      .def_static("create",
                  &DatabaseCache::Create,
                  "database"_a,
                  "min_num_matches"_a,
                  "ignore_watermarks"_a,
                  "image_names"_a)
      .def("num_cameras", &DatabaseCache::NumCameras)
      .def("num_images", &DatabaseCache::NumImages)
      .def(
          "camera",
          [](DatabaseCache& self, camera_t camera_id) {
            const struct Camera& camera = self.Camera(camera_id);
            return camera;
          },
          "camera_id"_a)
      .def(
          "image",
          [](DatabaseCache& self, image_t image_id) {
            const struct Image& image = self.Image(image_id);
            return image;
          },
          "image_id"_a)
      .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
      .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
      .def("cameras", &DatabaseCache::Cameras)
      .def("images", &DatabaseCache::Images)
      .def("correspondence_graph", &DatabaseCache::CorrespondenceGraph)
      .def("find_image_with_name", &DatabaseCache::FindImageWithName);
}
