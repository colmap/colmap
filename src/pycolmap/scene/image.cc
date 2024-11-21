#include "colmap/scene/image.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/point2d.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>
#include <optional>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename T>
std::shared_ptr<Image> MakeImage(const std::string& name,
                                 const std::vector<T>& points2D,
                                 const std::optional<Rigid3d>& cam_from_world,
                                 size_t camera_id,
                                 image_t image_id) {
  auto image = std::make_shared<Image>();
  image->SetName(name);
  image->SetPoints2D(points2D);
  image->SetCamFromWorld(cam_from_world);
  if (camera_id != kInvalidCameraId) {
    image->SetCameraId(camera_id);
  }
  image->SetImageId(image_id);
  return image;
}

void BindImage(py::module& m) {
  py::class_<Image, std::shared_ptr<Image>> PyImage(m, "Image");
  PyImage.def(py::init<>())
      .def(py::init(&MakeImage<Point2D>),
           "name"_a = "",
           py::arg_v("points2D", Point2DVector(), "ListPoint2D()"),
           "cam_from_world"_a = py::none(),
           "camera_id"_a = kInvalidCameraId,
           "id"_a = kInvalidImageId)
      .def(py::init(&MakeImage<Eigen::Vector2d>),
           "name"_a = "",
           "keypoints"_a = std::vector<Eigen::Vector2d>(),
           "cam_from_world"_a = Rigid3d(),
           "camera_id"_a = kInvalidCameraId,
           "id"_a = kInvalidImageId)
      .def_property("image_id",
                    &Image::ImageId,
                    &Image::SetImageId,
                    "Unique identifier of the image.")
      .def_property("camera_id",
                    &Image::CameraId,
                    &Image::SetCameraId,
                    "Unique identifier of the camera.")
      .def_property(
          "camera",
          [](Image& self) -> py::typing::Optional<Camera> {
            if (self.HasCameraPtr()) {
              return py::cast(*self.CameraPtr());
            } else {
              return py::none();
            }
          },
          &Image::SetCameraPtr,
          "The address of the camera")
      .def_property("name",
                    py::overload_cast<>(&Image::Name),
                    &Image::SetName,
                    "Name of the image.")
      .def_property(
          "cam_from_world",
          py::overload_cast<>(&Image::MaybeCamFromWorld),
          py::overload_cast<const std::optional<Rigid3d>&>(
              &Image::SetCamFromWorld),
          "The pose of the image, defined as the transformation from world to "
          "camera space. None if the image is not registered.")
      .def_property_readonly(
          "has_pose", &Image::HasPose, "Whether the image has a valid pose.")
      .def("reset_pose", &Image::ResetPose, "Invalidate the pose of the image.")
      .def_property(
          "points2D",
          py::overload_cast<>(&Image::Points2D),
          py::overload_cast<const Point2DVector&>(&Image::SetPoints2D),
          "Array of Points2D (=keypoints).")
      .def("point2D",
           py::overload_cast<camera_t>(&Image::Point2D),
           "point2D_idx"_a,
           "Direct accessor for a point2D.")
      .def(
          "set_point3D_for_point2D",
          &Image::SetPoint3DForPoint2D,
          "point2D_Idx"_a,
          "point3D_id"_a,
          "Set the point as triangulated, i.e. it is part of a 3D point track.")
      .def("reset_point3D_for_point2D",
           &Image::ResetPoint3DForPoint2D,
           "point2D_idx"_a,
           "Set the point as not triangulated, i.e. it is not part of a 3D "
           "point track")
      .def("has_point3D",
           &Image::HasPoint3D,
           "point3D_id"_a,
           "Check whether one of the image points is part of a 3D point track.")
      .def("projection_center",
           &Image::ProjectionCenter,
           "Extract the projection center in world space.")
      .def("viewing_direction",
           &Image::ViewingDirection,
           "Extract the viewing direction of the image.")
      .def(
          "project_point",
          [](const Image& self, const Eigen::Vector3d& point3D)
              -> py::typing::Optional<Eigen::Vector2d> {
            auto result = self.ProjectPoint(point3D);
            if (result.first) {
              return py::cast(result.second);
            } else {
              return py::none();
            }
          },
          "Project 3D point onto the image")
      .def("has_camera_id",
           &Image::HasCameraId,
           "Check whether identifier of camera has been set.")
      .def("has_camera_ptr",
           &Image::HasCameraPtr,
           "Check whether the camera pointer has been set.")
      .def("reset_camera_ptr",
           &Image::ResetCameraPtr,
           "Make the camera pointer a nullptr.")
      .def("num_points2D",
           &Image::NumPoints2D,
           "Get the number of image points (keypoints).")
      .def_property_readonly(
          "num_points3D",
          &Image::NumPoints3D,
          "Get the number of triangulations, i.e. the number of points that\n"
          "are part of a 3D point track.")
      .def(
          "get_observation_point2D_idxs",
          [](const Image& self) {
            std::vector<point2D_t> point2D_idxs;
            for (point2D_t point2D_idx = 0; point2D_idx < self.NumPoints2D();
                 ++point2D_idx) {
              if (self.Point2D(point2D_idx).HasPoint3D()) {
                point2D_idxs.push_back(point2D_idx);
              }
            }
            return point2D_idxs;
          },
          "Get the indices of 2D points that observe a 3D point.")
      .def(
          "get_observation_points2D",
          [](const Image& self) {
            Point2DVector points2D;
            for (const auto& point2D : self.Points2D()) {
              if (point2D.HasPoint3D()) {
                points2D.push_back(point2D);
              }
            }
            return points2D;
          },
          "Get the 2D points that observe a 3D point.");
  MakeDataclass(PyImage);

  py::bind_map<ImageMap>(m, "MapImageIdToImage");
}
