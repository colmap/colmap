#include "colmap/scene/database.h"

#include "pycolmap/pybind11_extension.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

class DatabaseTransactionWrapper {
 public:
  explicit DatabaseTransactionWrapper(Database* database)
      : database_(database) {}

  void Enter() {
    transaction_ = std::make_unique<DatabaseTransaction>(database_);
  }

  void Exit(const py::args&) { transaction_.reset(); }

 private:
  Database* database_;
  std::unique_ptr<DatabaseTransaction> transaction_;
};

}  // namespace

void BindDatabase(py::module& m) {
  py::class_<Database, std::shared_ptr<Database>> PyDatabase(m, "Database");
  PyDatabase.def(py::init<>())
      .def(py::init<const std::string&>(), "path"_a)
      .def("open", &Database::Open, "path"_a)
      .def("close", &Database::Close)
      .def("__enter__", [](Database& self) { return &self; })
      .def("__exit__", [](Database& self, const py::args&) { self.Close(); })
      .def("exists_camera", &Database::ExistsCamera, "camera_id"_a)
      .def("exists_image", &Database::ExistsImage, "image_id"_a)
      .def("exists_image", &Database::ExistsImageWithName, "name"_a)
      .def("exists_pose_prior", &Database::ExistsPosePrior, "image_id"_a)
      .def("exists_keypoints", &Database::ExistsKeypoints, "image_id"_a)
      .def("exists_descriptors", &Database::ExistsDescriptors, "image_id"_a)
      .def("exists_matches",
           &Database::ExistsMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("exists_inlier_matches",
           &Database::ExistsInlierMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def_property_readonly("num_cameras", &Database::NumCameras)
      .def_property_readonly("num_images", &Database::NumImages)
      .def_property_readonly("num_pose_priors", &Database::NumPosePriors)
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
      .def("read_rig", &Database::ReadRig, "rig_id"_a)
      .def("read_rig_with_sensor", &Database::ReadRigWithSensor, "sensor_id"_a)
      .def("read_all_rigs", &Database::ReadAllRigs)
      .def("read_camera", &Database::ReadCamera, "camera_id"_a)
      .def("read_all_cameras", &Database::ReadAllCameras)
      .def("read_frame", &Database::ReadFrame, "frame_id"_a)
      .def("read_all_frames", &Database::ReadAllFrames)
      .def("read_image", &Database::ReadImage, "image_id"_a)
      .def("read_image_with_name", &Database::ReadImageWithName, "name"_a)
      .def("read_all_images", &Database::ReadAllImages)
      .def("read_pose_prior", &Database::ReadPosePrior, "image_id"_a)
      .def("read_keypoints", &Database::ReadKeypointsBlob, "image_id"_a)
      .def("read_descriptors", &Database::ReadDescriptors, "image_id"_a)
      .def("read_matches",
           &Database::ReadMatchesBlob,
           "image_id1"_a,
           "image_id2"_a)
      .def("read_all_matches",
           [](const Database& self) {
             std::vector<std::pair<image_pair_t, FeatureMatchesBlob>>
                 pair_ids_and_matches = self.ReadAllMatchesBlob();
             std::vector<image_pair_t> all_pair_ids;
             all_pair_ids.reserve(pair_ids_and_matches.size());
             std::vector<FeatureMatchesBlob> all_matches;
             all_matches.reserve(pair_ids_and_matches.size());
             for (auto& [pair_id, matches] : pair_ids_and_matches) {
               all_pair_ids.push_back(pair_id);
               all_matches.push_back(std::move(matches));
             }
             return std::make_pair(std::move(all_pair_ids),
                                   std::move(all_matches));
           })
      .def("read_two_view_geometry",
           &Database::ReadTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a)
      .def("read_two_view_geometries",
           [](const Database& self) {
             std::vector<std::pair<image_pair_t, TwoViewGeometry>>
                 pair_ids_and_two_view_geometries =
                     self.ReadTwoViewGeometries();
             std::vector<image_pair_t> all_pair_ids;
             all_pair_ids.reserve(pair_ids_and_two_view_geometries.size());
             std::vector<TwoViewGeometry> all_two_view_geometries;
             all_two_view_geometries.reserve(
                 pair_ids_and_two_view_geometries.size());
             for (auto& [pair_id, two_view_geometry] :
                  pair_ids_and_two_view_geometries) {
               all_pair_ids.push_back(pair_id);
               all_two_view_geometries.push_back(two_view_geometry);
             }
             return std::make_pair(std::move(all_pair_ids),
                                   std::move(all_two_view_geometries));
           })
      .def(
          "read_two_view_geometry_num_inliers",
          [](const Database& self) {
            std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_inliers =
                self.ReadTwoViewGeometryNumInliers();
            std::vector<image_pair_t> all_pair_ids;
            all_pair_ids.reserve(pair_ids_and_num_inliers.size());
            std::vector<int> all_num_inliers;
            all_num_inliers.reserve(pair_ids_and_num_inliers.size());
            for (auto& [pair_id, num_inliers] : pair_ids_and_num_inliers) {
              all_pair_ids.push_back(pair_id);
              all_num_inliers.push_back(num_inliers);
            }
            return std::make_pair(std::move(all_pair_ids),
                                  std::move(all_num_inliers));
          })
      .def("write_camera",
           &Database::WriteCamera,
           "camera"_a,
           "use_camera_id"_a = false)
      .def("write_image",
           &Database::WriteImage,
           "image"_a,
           "use_image_id"_a = false)
      .def("write_pose_prior",
           &Database::WritePosePrior,
           "image_id"_a,
           "pose_prior"_a)
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
      .def("clear_pose_priors", &Database::ClearPosePriors)
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
      .def("__enter__", &DatabaseTransactionWrapper::Enter)
      .def("__exit__", &DatabaseTransactionWrapper::Exit);
}
