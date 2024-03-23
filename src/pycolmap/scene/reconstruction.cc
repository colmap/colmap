#include "colmap/scene/reconstruction.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"
#include "colmap/util/types.h"

#include "pycolmap/pybind11_extension.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

bool ExistsReconstructionText(const std::string& path) {
  return (ExistsFile(JoinPaths(path, "cameras.txt")) &&
          ExistsFile(JoinPaths(path, "images.txt")) &&
          ExistsFile(JoinPaths(path, "points3D.txt")));
}

bool ExistsReconstructionBinary(const std::string& path) {
  return (ExistsFile(JoinPaths(path, "cameras.bin")) &&
          ExistsFile(JoinPaths(path, "images.bin")) &&
          ExistsFile(JoinPaths(path, "points3D.bin")));
}

bool ExistsReconstruction(const std::string& path) {
  return (ExistsReconstructionText(path) || ExistsReconstructionBinary(path));
}

void BindReconstruction(py::module& m) {
  py::class_ext_<Reconstruction::ImagePairStat,
                 std::shared_ptr<Reconstruction::ImagePairStat>>(
      m, "ImagePairStat")
      .def(py::init<>())
      .def_readwrite("num_tri_corrs",
                     &Reconstruction::ImagePairStat::num_tri_corrs)
      .def_readwrite("num_total_corrs",
                     &Reconstruction::ImagePairStat::num_total_corrs);

  py::class_<Reconstruction, std::shared_ptr<Reconstruction>>(m,
                                                              "Reconstruction")
      .def(py::init<>())
      .def(py::init([](const std::string& path) {
             auto reconstruction = std::make_shared<Reconstruction>();
             reconstruction->Read(path);
             return reconstruction;
           }),
           "sfm_dir"_a)
      .def("read",
           &Reconstruction::Read,
           "sfm_dir"_a,
           "Read reconstruction in COLMAP format. Prefer binary.")
      .def("write",
           &Reconstruction::Write,
           "output_dir"_a,
           "Write reconstruction in COLMAP binary format.")
      .def("read_text", &Reconstruction::ReadText)
      .def("read_binary", &Reconstruction::ReadBinary)
      .def("write_text", &Reconstruction::WriteText)
      .def("write_binary", &Reconstruction::WriteBinary)
      .def("num_images", &Reconstruction::NumImages)
      .def("num_cameras", &Reconstruction::NumCameras)
      .def("num_reg_images", &Reconstruction::NumRegImages)
      .def("num_points3D", &Reconstruction::NumPoints3D)
      .def("num_image_pairs", &Reconstruction::NumImagePairs)
      .def_property_readonly("images",
                             &Reconstruction::Images,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("image_pairs", &Reconstruction::ImagePairs)
      .def_property_readonly("cameras",
                             &Reconstruction::Cameras,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("points3D",
                             &Reconstruction::Points3D,
                             py::return_value_policy::reference_internal)
      .def("point3D_ids", &Reconstruction::Point3DIds)
      .def("reg_image_ids", &Reconstruction::RegImageIds)
      .def("exists_camera", &Reconstruction::ExistsCamera)
      .def("exists_image", &Reconstruction::ExistsImage)
      .def("exists_point3D", &Reconstruction::ExistsPoint3D)
      .def("exists_image_pair", &Reconstruction::ExistsImagePair)
      .def("image_pair",
           py::overload_cast<image_t, image_t>(&Reconstruction::ImagePair))
      .def("image_pair",
           py::overload_cast<image_pair_t>(&Reconstruction::ImagePair))
      .def("set_up", &Reconstruction::SetUp, "correspondence_graph"_a)
      .def("tear_down", &Reconstruction::TearDown)
      .def("add_camera",
           &Reconstruction::AddCamera,
           "camera"_a,
           "Add new camera. There is only one camera per image, while multiple "
           "images\n"
           "might be taken by the same camera.")
      .def(
          "add_image", &Reconstruction::AddImage, "image"_a, "Add a new image.")
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
           "Add observation to existing 3D point.")
      .def("merge_points3D",
           &Reconstruction::MergePoints3D,
           "Merge two 3D points and return new identifier of new 3D point.\n"
           "The location of the merged 3D point is a weighted average of the "
           "two\n"
           "original 3D point's locations according to their track lengths.")
      .def("delete_point3D",
           &Reconstruction::DeletePoint3D,
           "Delete a 3D point, and all its references in the observed images.")
      .def("delete_observation",
           &Reconstruction::DeleteObservation,
           "Delete one observation from an image and the corresponding 3D "
           "point.\n"
           "Note that this deletes the entire 3D point, if the track has two "
           "elements\n"
           "prior to calling this method.")
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
           "Check if image is registered.")
      .def(
          "normalize",
          &Reconstruction::Normalize,
          "extent"_a = 10.0,
          "p0"_a = 0.1,
          "p1"_a = 0.9,
          "use_images"_a = true,
          "Normalize scene by scaling and translation to avoid degenerate\n"
          "visualization after bundle adjustment and to improve numerical\n"
          "stability of algorithms.\n\n"
          "Translates scene such that the mean of the camera centers or point\n"
          "locations are at the origin of the coordinate system.\n\n"
          "Scales scene such that the minimum and maximum camera centers are "
          "at the\n"
          "given `extent`, whereas `p0` and `p1` determine the minimum and\n"
          "maximum percentiles of the camera centers considered.")
      .def("transform",
           &Reconstruction::Transform,
           "Apply the 3D similarity transformation to all images and points.")
      .def("compute_bounding_box",
           &Reconstruction::ComputeBoundingBox,
           "p0"_a = 0.0,
           "p1"_a = 1.0)
      .def("crop", &Reconstruction::Crop)
      .def("find_image_with_name",
           &Reconstruction::FindImageWithName,
           py::return_value_policy::reference_internal,
           "Find image with matching name. Returns None if no match is found.")
      .def("find_common_reg_image_ids",
           &Reconstruction::FindCommonRegImageIds,
           "Find images that are both present in this and the given "
           "reconstruction.")
      .def(
          "filter_points3D",
          &Reconstruction::FilterPoints3D,
          "Filter 3D points with large reprojection error, negative depth, or\n"
          "insufficient triangulation angle.\n\n"
          "@param max_reproj_error    The maximum reprojection error.\n"
          "@param min_tri_angle       The minimum triangulation angle.\n"
          "@param point3D_ids         The points to be filtered.\n\n"
          "@return                    The number of filtered observations.")
      .def(
          "filter_points3D_in_images",
          &Reconstruction::FilterPoints3DInImages,
          "Filter 3D points with large reprojection error, negative depth, or\n"
          "insufficient triangulation angle.\n\n"
          "@param max_reproj_error    The maximum reprojection error.\n"
          "@param min_tri_angle       The minimum triangulation angle.\n"
          "@param image_ids           The the image ids in which the points3D "
          "are filtered.\n\n"
          "@return                    The number of filtered observations.")
      .def(
          "filter_all_points3D",
          &Reconstruction::FilterAllPoints3D,
          "Filter 3D points with large reprojection error, negative depth, or\n"
          "insufficient triangulation angle.\n\n"
          "@param max_reproj_error    The maximum reprojection error.\n"
          "@param min_tri_angle       The minimum triangulation angle.\n\n"
          "@return                    The number of filtered observations.")
      .def("filter_observations_with_negative_depth",
           &Reconstruction::FilterObservationsWithNegativeDepth,
           "Filter observations that have negative depth.\n\n"
           "@return    The number of filtered observations.")
      .def("filter_images",
           &Reconstruction::FilterImages,
           "Filter images without observations or bogus camera parameters.\n\n"
           "@return    The identifiers of the filtered images.")
      .def("compute_num_observations", &Reconstruction::ComputeNumObservations)
      .def("compute_mean_track_length", &Reconstruction::ComputeMeanTrackLength)
      .def("compute_mean_observations_per_reg_image",
           &Reconstruction::ComputeMeanObservationsPerRegImage)
      .def("compute_mean_reprojection_error",
           &Reconstruction::ComputeMeanReprojectionError)
      // .def("convert_to_PLY", &Reconstruction::ConvertToPLY)
      .def("import_PLY",
           py::overload_cast<const std::string&>(&Reconstruction::ImportPLY),
           "Import from PLY format. Note that these import functions are\n"
           "only intended for visualization of data and usable for "
           "reconstruction.")
      .def("export_PLY",
           &ExportPLY,
           "output_path"_a,
           "Export 3D points to PLY format (.ply).")
      .def("extract_colors_for_image",
           &Reconstruction::ExtractColorsForImage,
           "Extract colors for 3D points of given image. Colors will be "
           "extracted\n"
           "only for 3D points which are completely black.\n\n"
           "@param image_id      Identifier of the image for which to extract "
           "colors.\n"
           "@param path          Absolute or relative path to root folder of "
           "image.\n"
           "                     The image path is determined by concatenating "
           "the\n"
           "                     root path and the name of the image.\n\n"
           "@return              True if image could be read at given path.")
      .def("extract_colors_for_all_images",
           &Reconstruction::ExtractColorsForAllImages,
           "Extract colors for all 3D points by computing the mean color of "
           "all images.\n\n"
           "@param path          Absolute or relative path to root folder of "
           "image.\n"
           "                     The image path is determined by concatenating "
           "the\n"
           "                     root path and the name of the image.")
      .def("create_image_dirs",
           &Reconstruction::CreateImageDirs,
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
                THROW_CHECK(image.IsRegistered());
                THROW_CHECK_EQ(image.Point2D(point2D_idx).point3D_id, p3Did);
              }
            }
            for (auto& image_id : self.RegImageIds()) {
              THROW_CHECK(self.Image(image_id).HasCamera()) << image_id;
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
      .def("__repr__",
           [](const Reconstruction& self) {
             std::stringstream ss;
             ss << "Reconstruction(num_reg_images=" << self.NumRegImages()
                << ", num_cameras=" << self.NumCameras()
                << ", num_points3D=" << self.NumPoints3D()
                << ", num_observations=" << self.ComputeNumObservations()
                << ")";
             return ss.str();
           })
      .def("summary", [](const Reconstruction& self) {
        std::stringstream ss;
        ss << "Reconstruction:"
           << "\n\tnum_reg_images = " << self.NumRegImages()
           << "\n\tnum_cameras = " << self.NumCameras()
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
