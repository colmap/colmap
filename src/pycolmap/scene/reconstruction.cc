#include "colmap/scene/reconstruction.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
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

void BindReconstruction(py::module& m) {
  py::class_<Reconstruction, std::shared_ptr<Reconstruction>>(m,
                                                              "Reconstruction")
      .def(py::init<>())
      .def(py::init<const Reconstruction&>(), "reconstruction"_a)
      .def(py::init([](const std::string& path) {
             auto reconstruction = std::make_shared<Reconstruction>();
             reconstruction->Read(path);
             return reconstruction;
           }),
           "path"_a)
      .def("read",
           &Reconstruction::Read,
           "path"_a,
           "Read reconstruction in COLMAP format. Prefer binary.")
      .def("write",
           &Reconstruction::Write,
           "output_dir"_a,
           "Write reconstruction in COLMAP binary format.")
      .def("read_text", &Reconstruction::ReadText, "path"_a)
      .def("read_binary", &Reconstruction::ReadBinary, "path"_a)
      .def("write_text", &Reconstruction::WriteText, "path"_a)
      .def("write_binary", &Reconstruction::WriteBinary, "path"_a)
      .def("num_cameras", &Reconstruction::NumCameras)
      .def("num_images", &Reconstruction::NumImages)
      .def("num_reg_images", &Reconstruction::NumRegImages)
      .def("num_points3D", &Reconstruction::NumPoints3D)
      .def_property_readonly("images",
                             &Reconstruction::Images,
                             py::return_value_policy::reference_internal)
      .def("image",
           py::overload_cast<image_t>(&Reconstruction::Image),
           "image_id"_a,
           "Direct accessor for an image.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("cameras",
                             &Reconstruction::Cameras,
                             py::return_value_policy::reference_internal)
      .def("camera",
           py::overload_cast<camera_t>(&Reconstruction::Camera),
           "camera_id"_a,
           "Direct accessor for a camera.",
           py::return_value_policy::reference_internal)
      .def_property_readonly("points3D",
                             &Reconstruction::Points3D,
                             py::return_value_policy::reference_internal)
      .def("point3D",
           py::overload_cast<point3D_t>(&Reconstruction::Point3D),
           "point3D_id"_a,
           "Direct accessor for a Point3D.",
           py::return_value_policy::reference_internal)
      .def("point3D_ids", &Reconstruction::Point3DIds)
      .def("reg_image_ids", &Reconstruction::RegImageIds)
      .def("exists_camera", &Reconstruction::ExistsCamera, "camera_id"_a)
      .def("exists_image", &Reconstruction::ExistsImage, "image_id"_a)
      .def("exists_point3D", &Reconstruction::ExistsPoint3D, "point3D_id"_a)
      .def("tear_down", &Reconstruction::TearDown)
      .def("add_camera",
           &Reconstruction::AddCamera,
           "camera"_a,
           "Add new camera. There is only one camera per image, while multiple "
           "images might be taken by the same camera.")
      .def(
          "add_image",
          &Reconstruction::AddImage,
          "image"_a,
          "Add new image. Its camera must have been added before. If its "
          "camera object is unset, it will be automatically populated from the "
          "added cameras.")
      .def("add_point3D",
           py::overload_cast<const Eigen::Vector3d&,
                             Track,
                             const Eigen::Vector3ub&>(
               &Reconstruction::AddPoint3D),
           "Add new 3D object, and return its unique ID.",
           "xyz"_a,
           "track"_a,
           "color"_a = Eigen::Vector3ub::Zero())
      .def("add_observation",
           &Reconstruction::AddObservation,
           "point3D_id"_a,
           "track_element"_a,
           "Add observation to existing 3D point.")
      .def(
          "merge_points3D",
          &Reconstruction::MergePoints3D,
          "point3D_id1"_a,
          "point3D_id2"_a,
          "Merge two 3D points and return new identifier of new 3D point."
          "The location of the merged 3D point is a weighted average of the "
          "two original 3D point's locations according to their track lengths.")
      .def("delete_point3D",
           &Reconstruction::DeletePoint3D,
           "point3D_id"_a,
           "Delete a 3D point, and all its references in the observed images.")
      .def("delete_observation",
           &Reconstruction::DeleteObservation,
           "image_id"_a,
           "point2D_idx"_a,
           "Delete one observation from an image and the corresponding 3D "
           "point. Note that this deletes the entire 3D point, if the track "
           "has two elements prior to calling this method.")
      .def("register_image",
           &Reconstruction::RegisterImage,
           "image_id"_a,
           "Register an existing image.")
      .def("deregister_image",
           &Reconstruction::DeRegisterImage,
           "image_id"_a,
           "De-register an existing image, and all its references.")
      .def("is_image_registered",
           &Reconstruction::IsImageRegistered,
           "image_id"_a,
           "Check if image is registered.")
      .def("normalize",
           &Reconstruction::Normalize,
           "fixed_scale"_a = false,
           "extent"_a = 10.0,
           "min_percentile"_a = 0.1,
           "max_percentile"_a = 0.9,
           "use_images"_a = true,
           "Normalize scene by scaling and translation to avoid degenerate"
           "visualization after bundle adjustment and to improve numerical"
           "stability of algorithms.\n\n"
           "Translates scene such that the mean of the camera centers or point"
           "locations are at the origin of the coordinate system.\n\n Scales "
           "scene such that the minimum and maximum camera centers (or points) "
           "are  at the given `extent`, whereas `min_percentile` and  "
           "`max_percentile` determine the minimum  and maximum percentiles of "
           "the camera centers (or points) considered.")
      .def("transform",
           &Reconstruction::Transform,
           "new_from_old_world"_a,
           "Apply the 3D similarity transformation to all images and points.")
      .def("compute_centroid",
           &Reconstruction::ComputeCentroid,
           "min_percentile"_a = 0.0,
           "max_percentile"_a = 1.0,
           "use_images"_a = false)
      .def("compute_bounding_box",
           &Reconstruction::ComputeBoundingBox,
           "min_percentile"_a = 0.0,
           "max_percentile"_a = 1.0,
           "use_images"_a = false)
      .def("crop", &Reconstruction::Crop, "bbox"_a)
      .def("find_image_with_name",
           &Reconstruction::FindImageWithName,
           py::return_value_policy::reference_internal,
           "name"_a,
           "Find image with matching name. Returns None if no match is found.")
      .def("find_common_reg_image_ids",
           &Reconstruction::FindCommonRegImageIds,
           "other"_a,
           "Find images that are both present in this and the given "
           "reconstruction.")
      .def("update_point_3d_errors", &Reconstruction::UpdatePoint3DErrors)
      .def("compute_num_observations", &Reconstruction::ComputeNumObservations)
      .def("compute_mean_track_length", &Reconstruction::ComputeMeanTrackLength)
      .def("compute_mean_observations_per_reg_image",
           &Reconstruction::ComputeMeanObservationsPerRegImage)
      .def("compute_mean_reprojection_error",
           &Reconstruction::ComputeMeanReprojectionError)
      .def("import_PLY",
           py::overload_cast<const std::string&>(&Reconstruction::ImportPLY),
           "path"_a,
           "Import from PLY format. Note that these import functions are"
           "only intended for visualization of data and usable for "
           "reconstruction.")
      .def("export_PLY",
           &ExportPLY,
           "output_path"_a,
           "Export 3D points to PLY format (.ply).")
      .def("extract_colors_for_image",
           &Reconstruction::ExtractColorsForImage,
           "image_id"_a,
           "path"_a,
           "Extract colors for 3D points of given image. Colors will be "
           "extracted only for 3D points which are completely black. "
           "Return True if the image could be read at the given path.")
      .def("extract_colors_for_all_images",
           &Reconstruction::ExtractColorsForAllImages,
           "Extract colors for all 3D points by computing the mean color of "
           "all images.",
           "path"_a)
      .def("create_image_dirs",
           &Reconstruction::CreateImageDirs,
           "path"_a,
           "Create all image sub-directories in the given path.")
      .def(
          "check",
          [](Reconstruction& self) {
            for (auto& p3D_p : self.Points3D()) {
              const Point3D& p3D = p3D_p.second;
              const point3D_t p3Did = p3D_p.first;
              for (auto& track_el : p3D.track.Elements()) {
                image_t image_id = track_el.image_id;
                point2D_t point2D_idx = track_el.point2D_idx;
                THROW_CHECK(self.ExistsImage(image_id)) << image_id;
                THROW_CHECK(self.IsImageRegistered(image_id)) << image_id;
                const Image& image = self.Image(image_id);
                THROW_CHECK(image.HasPose());
                THROW_CHECK_EQ(image.Point2D(point2D_idx).point3D_id, p3Did);
              }
            }
            for (auto& image_id : self.RegImageIds()) {
              THROW_CHECK(self.Image(image_id).HasCameraId()) << image_id;
              camera_t camera_id = self.Image(image_id).CameraId();
              THROW_CHECK(self.ExistsCamera(camera_id)) << camera_id;
            }
          },
          "Check if current reconstruction is well formed.")
      .def("__copy__",
           [](const Reconstruction& self) { return Reconstruction(self); })
      .def("__deepcopy__",
           [](const Reconstruction& self, const py::dict&) {
             return Reconstruction(self);
           })
      .def("__repr__", &CreateRepresentation<Reconstruction>)
      .def("summary", [](const Reconstruction& self) {
        std::ostringstream ss;
        ss << "Reconstruction:"
           << "\n\tnum_cameras = " << self.NumCameras()
           << "\n\tnum_images = " << self.NumImages()
           << "\n\tnum_reg_images = " << self.NumRegImages()
           << "\n\tnum_points3D = " << self.NumPoints3D()
           << "\n\tnum_observations = " << self.ComputeNumObservations()
           << "\n\tmean_track_length = " << self.ComputeMeanTrackLength()
           << "\n\tmean_observations_per_image = "
           << self.ComputeMeanObservationsPerRegImage()
           << "\n\tmean_reprojection_error = "
           << self.ComputeMeanReprojectionError();
        return ss.str();
      });
}
