#include "colmap/scene/database_cache.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindDatabaseCache(py::module& m) {
  using Opts = DatabaseCache::Options;
  auto PyOpts = py::classh<Opts>(m, "DatabaseCacheOptions");
  PyOpts.def(py::init<>())
      .def_readwrite("min_num_matches",
                     &Opts::min_num_matches,
                     "Only load image pairs with a minimum number of matches.")
      .def_readwrite("ignore_watermarks",
                     &Opts::ignore_watermarks,
                     "Whether to ignore watermark image pairs.")
      .def_readwrite("image_names",
                     &Opts::image_names,
                     "Only load the data for a subset of the images. "
                     "All images are used if empty.")
      .def_readwrite(
          "convert_pose_priors_to_enu",
          &Opts::convert_pose_priors_to_enu,
          "Whether to convert pose priors to ENU coordinate system.");

  MakeDataclass(PyOpts);

  py::classh<DatabaseCache> PyDatabaseCache(m, "DatabaseCache");
  PyDatabaseCache.def(py::init<>())
      .def_static("create", &DatabaseCache::Create, "database"_a, "options"_a)
      .def_static("create_from_cache",
                  &DatabaseCache::CreateFromCache,
                  "database_cache"_a,
                  "options"_a)
      .def("load", &DatabaseCache::Load, "database"_a, "options"_a)
      .def("add_rig", &DatabaseCache::AddRig)
      .def("add_camera", &DatabaseCache::AddCamera)
      .def("add_frame", &DatabaseCache::AddFrame)
      .def("add_image", &DatabaseCache::AddImage)
      .def("add_pose_prior", &DatabaseCache::AddPosePrior)
      .def("num_rigs", &DatabaseCache::NumRigs)
      .def("num_cameras", &DatabaseCache::NumCameras)
      .def("num_frames", &DatabaseCache::NumFrames)
      .def("num_images", &DatabaseCache::NumImages)
      .def("num_pose_priors", &DatabaseCache::NumPosePriors)
      .def("exists_rig", &DatabaseCache::ExistsRig, "rig_id"_a)
      .def("exists_camera", &DatabaseCache::ExistsCamera, "camera_id"_a)
      .def("exists_frame", &DatabaseCache::ExistsFrame, "frame_id"_a)
      .def("exists_image", &DatabaseCache::ExistsImage, "image_id"_a)
      .def("rig",
           py::overload_cast<rig_t>(&DatabaseCache::Rig),
           py::return_value_policy::reference_internal,
           "rig_id"_a)
      .def("camera",
           py::overload_cast<camera_t>(&DatabaseCache::Camera),
           py::return_value_policy::reference_internal,
           "camera_id"_a)
      .def("frame",
           py::overload_cast<frame_t>(&DatabaseCache::Frame),
           py::return_value_policy::reference_internal,
           "frame_id"_a)
      .def("image",
           py::overload_cast<image_t>(&DatabaseCache::Image),
           py::return_value_policy::reference_internal,
           "image_id"_a)
      .def_property_readonly("rigs", &DatabaseCache::Rigs)
      .def_property_readonly("cameras", &DatabaseCache::Cameras)
      .def_property_readonly("frames", &DatabaseCache::Frames)
      .def_property_readonly("images", &DatabaseCache::Images)
      .def_property_readonly("pose_priors", &DatabaseCache::PosePriors)
      .def_property_readonly(
          "correspondence_graph",
          static_cast<std::shared_ptr<const class CorrespondenceGraph> (
              DatabaseCache::*)() const>(&DatabaseCache::CorrespondenceGraph))
      .def("find_image_with_name", &DatabaseCache::FindImageWithName, "name"_a);
}
