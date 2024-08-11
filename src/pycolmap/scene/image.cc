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
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

std::string PrintBaseImage(const BaseImage& image) {
  std::stringstream ss;
  ss << "BaseImage(image_id="
     << (image.ImageId() != kInvalidImageId ? std::to_string(image.ImageId())
                                            : "Invalid")
     << ", camera_id="
     << (image.HasCamera() ? std::to_string(image.CameraId()) : "Invalid")
     << ", name=\"" << image.Name() << "\""
     << ", triangulated=" << image.NumPoints3D() << "/" << image.NumPoints2D()
     << ")";
  return ss.str();
}

std::string PrintImage(const Image& image) {
  std::stringstream ss;
  ss << "Image(image_id="
     << (image.ImageId() != kInvalidImageId ? std::to_string(image.ImageId())
                                            : "Invalid")
     << ", camera=Camera(camera_id="
     << (image.HasCamera() ? std::to_string(image.CameraId()) : "Invalid")
     << "), name=\"" << image.Name() << "\""
     << ", triangulated=" << image.NumPoints3D() << "/" << image.NumPoints2D()
     << ")";
  return ss.str();
}

template <typename T>
std::shared_ptr<BaseImage> MakeBaseImage(const std::string& name,
                                         const std::vector<T>& points2D,
                                         const Rigid3d& cam_from_world,
                                         size_t camera_id,
                                         image_t image_id) {
  auto image = std::make_shared<BaseImage>();
  image->SetName(name);
  image->SetPoints2D(points2D);
  image->CamFromWorld() = cam_from_world;
  if (camera_id != kInvalidCameraId) {
    image->SetCameraId(camera_id);
  }
  image->SetImageId(image_id);
  return image;
}

template <typename T>
std::shared_ptr<Image> MakeImage(Camera* camera,
                                 const std::string& name,
                                 const std::vector<T>& points2D,
                                 const Rigid3d& cam_from_world,
                                 image_t image_id) {
  auto image = std::make_shared<Image>(camera);
  image->SetName(name);
  image->SetPoints2D(points2D);
  image->CamFromWorld() = cam_from_world;
  image->SetImageId(image_id);
  return image;
}

void BindImage(py::module& m) {
  py::class_<BaseImage, std::shared_ptr<BaseImage>> PyBaseImage(m, "BaseImage");
  PyBaseImage.def(py::init<>())
      .def(py::init(&MakeBaseImage<Point2D>),
           "name"_a = "",
           "points2D"_a = Point2DVector(),
           "cam_from_world"_a = Rigid3d(),
           "camera_id"_a = kInvalidCameraId,
           "id"_a = kInvalidImageId)
      .def(py::init(&MakeBaseImage<Eigen::Vector2d>),
           "name"_a = "",
           "keypoints"_a = std::vector<Eigen::Vector2d>(),
           "cam_from_world"_a = Rigid3d(),
           "camera_id"_a = kInvalidCameraId,
           "id"_a = kInvalidImageId)
      .def_property("image_id",
                    &BaseImage::ImageId,
                    &BaseImage::SetImageId,
                    "Unique identifier of image.")
      .def_property("camera_id",
                    &BaseImage::CameraId,
                    &BaseImage::SetCameraId,
                    "Unique identifier of the camera.")
      .def_property("name",
                    py::overload_cast<>(&BaseImage::Name),
                    &BaseImage::SetName,
                    "Name of the image.")
      .def_property(
          "cam_from_world",
          py::overload_cast<>(&BaseImage::CamFromWorld),
          [](Image& self, const Rigid3d& cam_from_world) {
            self.CamFromWorld() = cam_from_world;
          },
          "The pose of the image, defined as the transformation from world to "
          "camera space.")
      .def_property(
          "points2D",
          py::overload_cast<>(&BaseImage::Points2D),
          py::overload_cast<const Point2DVector&>(&BaseImage::SetPoints2D),
          "Array of Points2D (=keypoints).")
      .def("point2D", py::overload_cast<camera_t>(&BaseImage::Point2D))
      .def(
          "set_point3D_for_point2D",
          &BaseImage::SetPoint3DForPoint2D,
          "point2D_Idx"_a,
          "point3D_id"_a,
          "Set the point as triangulated, i.e. it is part of a 3D point track.")
      .def("reset_point3D_for_point2D",
           &BaseImage::ResetPoint3DForPoint2D,
           "Set the point as not triangulated, i.e. it is not part of a 3D "
           "point track")
      .def("has_point3D",
           &BaseImage::HasPoint3D,
           "Check whether one of the image points is part of the 3D point "
           "track.")
      .def("projection_center",
           &BaseImage::ProjectionCenter,
           "Extract the projection center in world space.")
      .def("viewing_direction",
           &BaseImage::ViewingDirection,
           "Extract the viewing direction of the image.")
      .def("has_camera",
           &BaseImage::HasCamera,
           "Check whether identifier of camera has been set.")
      .def_property("registered",
                    &BaseImage::IsRegistered,
                    &BaseImage::SetRegistered,
                    "Whether image is registered in the reconstruction.")
      .def("num_points2D",
           &BaseImage::NumPoints2D,
           "Get the number of image points (keypoints).")
      .def_property_readonly(
          "num_points3D",
          &BaseImage::NumPoints3D,
          "Get the number of triangulations, i.e. the number of points that\n"
          "are part of a 3D point track.")
      .def("get_valid_point2D_ids",
           [](const Image& self) {
             std::vector<point2D_t> valid_point2D_ids;

             for (point2D_t point2D_idx = 0; point2D_idx < self.NumPoints2D();
                  ++point2D_idx) {
               if (self.Point2D(point2D_idx).HasPoint3D()) {
                 valid_point2D_ids.push_back(point2D_idx);
               }
             }

             return valid_point2D_ids;
           })
      .def("get_valid_points2D",
           [](const Image& self) {
             Point2DVector valid_points2D;

             for (point2D_t point2D_idx = 0; point2D_idx < self.NumPoints2D();
                  ++point2D_idx) {
               if (self.Point2D(point2D_idx).HasPoint3D()) {
                 valid_points2D.push_back(self.Point2D(point2D_idx));
               }
             }

             return valid_points2D;
           })
      .def("__repr__", &PrintBaseImage);
  MakeDataclass(PyBaseImage);

  py::class_<Image, BaseImage, std::shared_ptr<Image>> PyImage(m, "Image");
  PyImage.def(py::init<const BaseImage&, Camera*>())
      .def(py::init(&MakeImage<Point2D>),
           "camera"_a,
           "name"_a = "",
           "points2D"_a = Point2DVector(),
           "cam_from_world"_a = Rigid3d(),
           "id"_a = kInvalidImageId)
      .def(py::init(&MakeImage<Eigen::Vector2d>),
           "camera"_a,
           "name"_a = "",
           "keypoints"_a = std::vector<Eigen::Vector2d>(),
           "cam_from_world"_a = Rigid3d(),
           "id"_a = kInvalidImageId)
      .def("cast_to_base", [](Image& self) -> BaseImage { return self; })
      .def_property(
          "camera_id",
          &Image::CameraId,
          [](Image& self, camera_t camera_id) {
            LOG(FATAL_THROW)
                << "Error! The ``camera_id`` property can no longer be "
                   "directly set in the Image class since it now comes from "
                   "``image.camera``. Update the ``camera`` property instead.";
          },
          "Disable the setter of the camera id.")
      .def_property("camera",
                    &Image::Camera,
                    &Image::SetCamera,
                    "The address of the camera")
      .def(
          "project_point",
          [](const Image& self, const Eigen::Vector3d& point3D) -> py::object {
            auto res = self.ProjectPoint(point3D);
            if (res.first)
              return py::cast(res.second);
            else
              return py::none();
          },
          "Project 3D point onto the image")
      .def("__repr__", &PrintImage);

  py::bind_map<ImageMap>(m, "MapImageIdToImage")
      .def("__repr__", [](const ImageMap& self) {
        std::stringstream ss;
        ss << "{";
        bool is_first = true;
        for (const auto& pair : self) {
          if (!is_first) {
            ss << ",\n ";
          }
          is_first = false;
          ss << pair.first << ": " << PrintImage(pair.second);
        }
        ss << "}";
        return ss.str();
      });
}
