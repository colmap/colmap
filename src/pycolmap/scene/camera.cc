#include "colmap/scene/camera.h"

#include "colmap/scene/point2d.h"
#include "colmap/sensor/models.h"
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

void BindCamera(py::module& m) {
  py::enum_<CameraModelId> PyCameraModelId(m, "CameraModelId");
  PyCameraModelId.value("INVALID", CameraModelId::kInvalid);
#define CAMERA_MODEL_CASE(CameraModel) \
  PyCameraModelId.value(CameraModel::model_name.c_str(), CameraModel::model_id);

  CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  AddStringToEnumConstructor(PyCameraModelId);
  py::implicitly_convertible<int, CameraModelId>();

  py::class_<Camera, std::shared_ptr<Camera>> PyCamera(m, "Camera");
  PyCamera.def(py::init<>())
      .def_static("create",
                  &Camera::CreateFromModelId,
                  "camera_id"_a,
                  "model"_a,
                  "focal_length"_a,
                  "width"_a,
                  "height"_a)
      .def_readwrite(
          "camera_id", &Camera::camera_id, "Unique identifier of the camera.")
      .def_readwrite("model", &Camera::model_id, "Camera model.")
      .def_readwrite("width", &Camera::width, "Width of camera sensor.")
      .def_readwrite("height", &Camera::height, "Height of camera sensor.")
      .def("mean_focal_length", &Camera::MeanFocalLength)
      .def_property(
          "focal_length", &Camera::FocalLength, &Camera::SetFocalLength)
      .def_property(
          "focal_length_x", &Camera::FocalLengthX, &Camera::SetFocalLengthX)
      .def_property(
          "focal_length_y", &Camera::FocalLengthY, &Camera::SetFocalLengthY)
      .def_readwrite("has_prior_focal_length", &Camera::has_prior_focal_length)
      .def_property("principal_point_x",
                    &Camera::PrincipalPointX,
                    &Camera::SetPrincipalPointX)
      .def_property("principal_point_y",
                    &Camera::PrincipalPointY,
                    &Camera::SetPrincipalPointY)
      .def("focal_length_idxs",
           &Camera::FocalLengthIdxs,
           "Indices of focal length parameters in params property.")
      .def("principal_point_idxs",
           &Camera::PrincipalPointIdxs,
           "Indices of principal point parameters in params property.")
      .def("extra_params_idxs",
           &Camera::ExtraParamsIdxs,
           "Indices of extra parameters in params property.")
      .def("calibration_matrix",
           &Camera::CalibrationMatrix,
           "Compute calibration matrix from params.")
      .def_property_readonly(
          "params_info",
          &Camera::ParamsInfo,
          "Get human-readable information about the parameter vector "
          "ordering.")
      .def_property(
          "params",
          [](Camera& self) {
            // Return a view (via a numpy array) instead of a copy.
            return Eigen::Map<Eigen::VectorXd>(self.params.data(),
                                               self.params.size());
          },
          [](Camera& self, const std::vector<double>& params) {
            self.params = params;
          },
          "Camera parameters.")
      .def("params_to_string",
           &Camera::ParamsToString,
           "Concatenate parameters as comma-separated list.")
      .def("set_params_from_string",
           &Camera::SetParamsFromString,
           "params"_a,
           "Set camera parameters from comma-separated list.")
      .def("verify_params",
           &Camera::VerifyParams,
           "Check whether parameters are valid, i.e. the parameter vector has"
           " the correct dimensions that match the specified camera model.")
      .def("has_bogus_params",
           &Camera::HasBogusParams,
           "min_focal_length_ratio"_a,
           "max_focal_length_ratio"_a,
           "max_extra_param"_a,
           "Check whether camera has bogus parameters.")
      .def("cam_from_img",
           &Camera::CamFromImg,
           "image_point"_a,
           "Project point in image plane to world / infinity.")
      .def(
          "cam_from_img",
          [](const Camera& self,
             const py::EigenDRef<const Eigen::MatrixX2d>& image_points) {
            std::vector<Eigen::Vector2d> world_points(image_points.rows());
            for (size_t idx = 0; idx < image_points.rows(); ++idx) {
              world_points[idx] = self.CamFromImg(image_points.row(idx));
            }
            return world_points;
          },
          "image_points"_a,
          "Project list of points in image plane to world / infinity.")
      .def(
          "cam_from_img",
          [](const Camera& self, const Point2DVector& image_points) {
            std::vector<Eigen::Vector2d> world_points(image_points.size());
            for (size_t idx = 0; idx < image_points.size(); ++idx) {
              world_points[idx] = self.CamFromImg(image_points[idx].xy);
            }
            return world_points;
          },
          "image_points"_a,
          "Project list of points in image plane to world / infinity.")
      .def("cam_from_img_threshold",
           &Camera::CamFromImgThreshold,
           "threshold"_a,
           "Convert pixel threshold in image plane to world space.")
      .def("img_from_cam",
           &Camera::ImgFromCam,
           "cam_point"_a,
           "Project point from world / infinity to image plane.")
      .def(
          "img_from_cam",
          [](const Camera& self,
             const py::EigenDRef<const Eigen::MatrixX2d>& world_points) {
            std::vector<Eigen::Vector2d> image_points(world_points.rows());
            for (size_t idx = 0; idx < world_points.rows(); ++idx) {
              image_points[idx] = self.ImgFromCam(world_points.row(idx));
            }
            return image_points;
          },
          "cam_points"_a,
          "Project list of points from world / infinity to image plane.")
      .def(
          "img_from_cam",
          [](const Camera& self,
             const py::EigenDRef<const Eigen::MatrixX3d>& world_points) {
            return py::cast(self).attr("img_from_cam")(
                world_points.rowwise().hnormalized());
          },
          "cam_points"_a,
          "Project list of points from world / infinity to image plane.")
      .def(
          "img_from_cam",
          [](const Camera& self, const Point2DVector& world_points) {
            std::vector<Eigen::Vector2d> image_points(world_points.size());
            for (size_t idx = 0; idx < world_points.size(); ++idx) {
              image_points[idx] = self.ImgFromCam(world_points[idx].xy);
            }
            return image_points;
          },
          "cam_points"_a,
          "Project list of points from world / infinity to image plane.")
      .def("rescale",
           py::overload_cast<size_t, size_t>(&Camera::Rescale),
           "new_width"_a,
           "new_height"_a,
           "Rescale the camera dimensions and accordingly the "
           "focal length and the principal point.")
      .def("rescale",
           py::overload_cast<double>(&Camera::Rescale),
           "scale"_a,
           "Rescale the camera dimensions and accordingly the "
           "focal length and the principal point.");
  MakeDataclass(PyCamera,
                {"camera_id",
                 "model",
                 "width",
                 "height",
                 "params",
                 "has_prior_focal_length"});

  py::bind_map<CameraMap>(m, "MapCameraIdToCamera");
}
