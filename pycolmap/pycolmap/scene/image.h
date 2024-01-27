#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/image.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/log_exceptions.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

using ImageMap = std::unordered_map<image_t, Image>;
PYBIND11_MAKE_OPAQUE(ImageMap);

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

  py::class_<Image, std::shared_ptr<Image>> PyImage(m, "Image");
  PyImage.def(py::init<>())
      .def(py::init(&MakeImage<Point2D>),
           "name"_a = "",
           "points2D"_a = std::vector<Point2D>(),
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
      .def_property(
          "camera_id",
          &Image::CameraId,
          [](Image& self, const camera_t camera_id) {
            THROW_CHECK_NE(camera_id, kInvalidCameraId);
            self.SetCameraId(camera_id);
          },
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
          "cam_from_world_prior",
          py::overload_cast<>(&Image::CamFromWorldPrior),
          [](Image& self, const Rigid3d& cam_from_world) {
            self.CamFromWorldPrior() = cam_from_world;
          },
          "The pose prior of the image, e.g. extracted from EXIF tags.")
      .def_property(
          "points2D",
          py::overload_cast<>(&Image::Points2D),
          [](Image& self, const std::vector<struct Point2D>& points2D) {
            THROW_CUSTOM_CHECK(!points2D.empty(), std::invalid_argument);
            self.SetPoints2D(points2D);
          },
          "Array of Points2D (=keypoints).")
      .def(
          "set_point3D_for_point2D",
          [](Image& self,
             const point2D_t point2D_idx,
             const point3D_t point3D_id) {
            THROW_CHECK_NE(point3D_id, kInvalidPoint3DId);
            self.SetPoint3DForPoint2D(point2D_idx, point3D_id);
          },
          "Set the point as triangulated, i.e. it is part of a 3D point track.")
      .def("reset_point3D_for_point2D",
           &Image::ResetPoint3DForPoint2D,
           "Set the point as not triangulated, i.e. it is not part of a 3D "
           "point track")
      .def("is_point3D_visible",
           &Image::IsPoint3DVisible,
           "Check whether an image point has a correspondence to an image "
           "point in\n"
           "another image that has a 3D point.")
      .def("has_point3D",
           &Image::HasPoint3D,
           "Check whether one of the image points is part of the 3D point "
           "track.")
      .def("increment_correspondence_has_point3D",
           &Image::IncrementCorrespondenceHasPoint3D,
           "Indicate that another image has a point that is triangulated and "
           "has\n"
           "a correspondence to this image point. Note that this must only be "
           "called\n"
           "after calling `SetUp`.")
      .def("decrement_correspondence_has_point3D",
           &Image::DecrementCorrespondenceHasPoint3D,
           "Indicate that another image has a point that is not triangulated "
           "any more\n"
           "and has a correspondence to this image point. This assumes that\n"
           "`IncrementCorrespondenceHasPoint3D` was called for the same image "
           "point\n"
           "and correspondence before. Note that this must only be called\n"
           "after calling `SetUp`.")
      .def("projection_center",
           &Image::ProjectionCenter,
           "Extract the projection center in world space.")
      .def("viewing_direction",
           &Image::ViewingDirection,
           "Extract the viewing direction of the image.")
      .def(
          "set_up",
          [](Image& self, const struct Camera& camera) {
            THROW_CHECK_EQ(self.CameraId(), camera.camera_id);
            self.SetUp(camera);
          },
          "Setup the image and necessary internal data structures before being "
          "used in reconstruction.")
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
      .def_property(
          "num_observations",
          &Image::NumObservations,
          &Image::SetNumObservations,
          "Number of observations, i.e. the number of image points that\n"
          "have at least one correspondence to another image.")
      .def_property("num_correspondences",
                    &Image::NumCorrespondences,
                    &Image::SetNumCorrespondences,
                    "Number of correspondences for all image points.")
      .def("num_visible_points3D",
           &Image::NumVisiblePoints3D,
           "Get the number of observations that see a triangulated point, i.e. "
           "the\n"
           "number of image points that have at least one correspondence to a\n"
           "triangulated point in another image.")
      .def("point3D_visibility_score",
           &Image::Point3DVisibilityScore,
           "Get the score of triangulated observations. In contrast to\n"
           "`NumVisiblePoints3D`, this score also captures the distribution\n"
           "of triangulated observations in the image. This is useful to "
           "select\n"
           "the next best image in incremental reconstruction, because a more\n"
           "uniform distribution of observations results in more robust "
           "registration.")
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
             std::vector<Point2D> valid_points2D;

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
}
