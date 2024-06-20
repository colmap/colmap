#include "colmap/scene/database.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

class DatabaseTransactionWrapper {
 public:
  explicit DatabaseTransactionWrapper(Database* database)
      : database_(database) {}

  void enter() {
    transaction_ = std::make_unique<DatabaseTransaction>(database_);
  }

  void exit(const py::args&) { transaction_.reset(); }

 private:
  Database* database_;
  std::unique_ptr<DatabaseTransaction> transaction_;
};

void BindDatabase(py::module& m) {
  py::class_<Database, std::shared_ptr<Database>> PyDatabase(m, "Database");
  PyDatabase.def(py::init<>())
      .def(py::init<const std::string&>(), "path"_a)
      .def("open", &Database::Open, "path"_a)
      .def("close", &Database::Close)
      .def_property_readonly("num_cameras", &Database::NumCameras)
      .def_property_readonly("num_images", &Database::NumImages)
      .def_property_readonly("num_keypoints", &Database::NumKeypoints)
      .def("num_keypoints_for_image",
           &Database::NumKeypointsForImage,
           "image_id"_a)
      .def_property_readonly("num_descriptors", &Database::NumDescriptors)
      .def("num_descriptors_for_image",
           &Database::NumDescriptorsForImage,
           "image_id"_a)
      .def_property_readonly("num_matches", &Database::NumMatches)
      .def_property_readonly("num_inlier_matches", &Database::NumInlierMatches)
      .def_property_readonly("num_matched_image_pairs",
                             &Database::NumMatchedImagePairs)
      .def_property_readonly("num_verified_image_pairs",
                             &Database::NumVerifiedImagePairs)
      .def_static("image_pair_to_pair_id",
                  &Database::ImagePairToPairId,
                  "image_id1"_a,
                  "image_id2"_a)
      .def_static(
          "pair_id_to_image_pair", &Database::PairIdToImagePair, "pair_id"_a)
      .def_static("swap_image_pair",
                  &Database::SwapImagePair,
                  "image_id1"_a,
                  "image_id2"_a)
      .def("read_camera", &Database::ReadCamera, "camera_id"_a)
      .def("read_all_cameras", &Database::ReadAllCameras)
      .def("read_image", &Database::ReadImage, "image_id"_a)
      .def("read_image", &Database::ReadImageWithName, "name"_a)
      .def("read_all_images", &Database::ReadAllImages)
      .def("read_keypoints", &Database::ReadKeypointsBlob, "image_id"_a)
      .def("read_descriptors", &Database::ReadDescriptors, "image_id"_a)
      .def("read_matches",
           &Database::ReadMatchesBlob,
           "image_id1"_a,
           "image_id2"_a)
      // TODO: ReadAllMatches
      .def("read_two_view_geometry",
           &Database::ReadTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a)
      .def("read_two_view_geometries",
           [](const Database& self) {
             std::vector<image_pair_t> image_pair_ids;
             std::vector<TwoViewGeometry> two_view_geometries;
             self.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
             return std::make_pair(image_pair_ids, two_view_geometries);
           })
      .def("read_two_view_geometry_num_inliers",
           [](const Database& self) {
             std::vector<std::pair<image_t, image_t>> image_pair_ids;
             std::vector<int> num_inliers;
             self.ReadTwoViewGeometryNumInliers(&image_pair_ids, &num_inliers);
             return std::make_pair(image_pair_ids, num_inliers);
           })
      .def("write_camera",
           &Database::WriteCamera,
           "camera"_a,
           "use_camera_id"_a = false)
      .def("write_image",
           &Database::WriteImage,
           "image"_a,
           "use_image_id"_a = false)
      .def("write_keypoints",
           py::overload_cast<image_t, const FeatureKeypointsBlob&>(
               &Database::WriteKeypoints, py::const_),
           "image_id"_a,
           "keypoints"_a)
      .def("write_descriptors",
           &Database::WriteDescriptors,
           "image_id"_a,
           "descriptors"_a)
      .def("write_matches",
           py::overload_cast<image_t, image_t, const FeatureMatchesBlob&>(
               &Database::WriteMatches, py::const_),
           "image_id1"_a,
           "image_id2"_a,
           "matches"_a)
      .def("write_two_view_geometry",
           &Database::WriteTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a,
           "two_view_geometry"_a)
      .def("update_camera", &Database::UpdateCamera, "camera"_a)
      .def("update_image", &Database::UpdateImage, "image"_a)
      .def("delete_matches",
           &Database::DeleteMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("delete_inlier_matches",
           &Database::DeleteInlierMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("clear_all_tables", &Database::ClearAllTables)
      .def("clear_cameras", &Database::ClearCameras)
      .def("clear_images", &Database::ClearImages)
      .def("clear_descriptors", &Database::ClearDescriptors)
      .def("clear_keypoints", &Database::ClearKeypoints)
      .def("clear_matches", &Database::ClearMatches)
      .def("clear_two_view_geometries", &Database::ClearTwoViewGeometries)
      .def_static("merge",
                  &Database::Merge,
                  "database1"_a,
                  "database2"_a,
                  "merged_database"_a);

  py::class_<DatabaseTransactionWrapper>(m, "DatabaseTransaction")
      .def(py::init<Database*>(), "database"_a)
      .def("__enter__", &DatabaseTransactionWrapper::enter)
      .def("__exit__", &DatabaseTransactionWrapper::exit);
}
