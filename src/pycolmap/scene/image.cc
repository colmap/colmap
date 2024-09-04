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

std::string PrintImage(const Image& image) {
  std::stringstream ss;
  ss << "Image(image_id="
     << (image.ImageId() != kInvalidImageId ? std::to_string(image.ImageId())
                                            : "Invalid")
     << ", camera_id="
     << (image.HasCamera() ? std::to_string(image.CameraId()) : "Invalid")
     << ", name=\"" << image.Name() << "\""
     << ", triangulated=" << image.NumPoints3D() << "/" << image.NumPoints2D()
     << ")";
  return ss.str();
}

template <typename T>
std::shared_ptr<Image> MakeImage(const std::string& name,
                                 const std::vector<T>& points2D,
                                 const Rigid3d& cam_from_world,
                                 size_t camera_id,
                                 image_t image_id) {
  auto image = std::make_shared<Image>();
  image->SetName(name);
  image->SetPoints2D(points2D);
  image->CamFromWorld() = cam_from_world;
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
           "cam_from_world"_a = Rigid3d(),
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
                    "Unique identifier of image.")
      .def_property("camera_id",
                    &Image::CameraId,
                    &Image::SetCameraId,
                    "Unique identifier of the camera.")
      .def_property("name",
                    py::overload_cast<>(&Image::Name),
                    &Image::SetName,
                    "Name of the image.")
      .def_property(
          "cam_from_world",
          py::overload_cast<>(&Image::CamFromWorld),
          [](Image& self, const Rigid3d& cam_from_world) {
            self.CamFromWorld() = cam_from_world;
          },
          "The pose of the image, defined as the transformation from world to "
          "camera space.")
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
      .def("has_camera",
           &Image::HasCamera,
           "Check whether identifier of camera has been set.")
      .def_property("registered",
                    &Image::IsRegistered,
                    &Image::SetRegistered,
                    "Whether image is registered in the reconstruction.")
      .def("num_points2D",
           &Image::NumPoints2D,
           "Get the number of image points (keypoints).")
      .def_property_readonly(
          "num_points3D",
          &Image::NumPoints3D,
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
      .def("__repr__", &PrintImage);
  MakeDataclass(PyImage);

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
