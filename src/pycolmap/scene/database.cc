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

class PyDatabaseImpl : public Database, py::trampoline_self_life_support {
 public:
  void Close() override { PYBIND11_OVERRIDE_PURE(void, Database, Close); }

  std::shared_ptr<Database> Clone() const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Database>, Database, Clone);
  }

  bool ExistsRig(rig_t rig_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsRig, rig_id);
  }

  bool ExistsCamera(camera_t camera_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsCamera, camera_id);
  }

  bool ExistsFrame(frame_t frame_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsFrame, frame_id);
  }

  bool ExistsImage(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsImage, image_id);
  }

  bool ExistsImageWithName(const std::string& name) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsImageWithName, name);
  }

  bool ExistsPosePrior(pose_prior_t pose_prior_id,
                       bool is_deprecated_image_prior = true) const override {
    PYBIND11_OVERRIDE_PURE(bool,
                           Database,
                           ExistsPosePrior,
                           pose_prior_id,
                           is_deprecated_image_prior);
  }

  bool ExistsKeypoints(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsKeypoints, image_id);
  }

  bool ExistsDescriptors(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsDescriptors, image_id);
  }

  bool ExistsMatches(image_t image_id1, image_t image_id2) const override {
    PYBIND11_OVERRIDE_PURE(bool, Database, ExistsMatches, image_id1, image_id2);
  }

  bool ExistsTwoViewGeometry(image_t image_id1,
                             image_t image_id2) const override {
    PYBIND11_OVERRIDE_PURE(
        bool, Database, ExistsTwoViewGeometry, image_id1, image_id2);
  }

  size_t NumRigs() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumRigs);
  }

  size_t NumCameras() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumCameras);
  }

  size_t NumFrames() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumFrames);
  }

  size_t NumImages() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumImages);
  }

  size_t NumPosePriors() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumPosePriors);
  }

  size_t NumKeypoints() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumKeypoints);
  }

  size_t MaxNumKeypoints() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, MaxNumKeypoints);
  }

  size_t NumKeypointsForImage(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumKeypointsForImage);
  }

  size_t NumDescriptors() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumDescriptors);
  }

  size_t MaxNumDescriptors() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, MaxNumDescriptors);
  }

  size_t NumDescriptorsForImage(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumDescriptorsForImage, image_id);
  }

  size_t NumMatches() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumMatches);
  }

  size_t NumInlierMatches() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumInlierMatches);
  }

  size_t NumMatchedImagePairs() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumMatchedImagePairs);
  }

  size_t NumVerifiedImagePairs() const override {
    PYBIND11_OVERRIDE_PURE(size_t, Database, NumVerifiedImagePairs);
  }

  Rig ReadRig(rig_t rig_id) const override {
    PYBIND11_OVERRIDE_PURE(Rig, Database, ReadRig, rig_id);
  }

  std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const override {
    PYBIND11_OVERRIDE_PURE(
        std::optional<Rig>, Database, ReadRigWithSensor, sensor_id);
  }

  std::vector<Rig> ReadAllRigs() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<Rig>, Database, ReadAllRigs);
  }

  Camera ReadCamera(camera_t camera_id) const override {
    PYBIND11_OVERRIDE_PURE(Camera, Database, ReadCamera, camera_id);
  }

  std::vector<Camera> ReadAllCameras() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<Camera>, Database, ReadAllCameras);
  }

  Frame ReadFrame(frame_t frame_id) const override {
    PYBIND11_OVERRIDE_PURE(Frame, Database, ReadFrame, frame_id);
  }

  std::vector<Frame> ReadAllFrames() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<Frame>, Database, ReadAllFrames);
  }

  Image ReadImage(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(Image, Database, ReadImage, image_id);
  }

  std::optional<Image> ReadImageWithName(
      const std::string& name) const override {
    PYBIND11_OVERRIDE_PURE(
        std::optional<Image>, Database, ReadImageWithName, name);
  }

  std::vector<Image> ReadAllImages() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<Image>, Database, ReadAllImages);
  }

  PosePrior ReadPosePrior(
      pose_prior_t pose_prior_id,
      bool is_deprecated_image_prior = true) const override {
    PYBIND11_OVERRIDE_PURE(PosePrior,
                           Database,
                           ReadPosePrior,
                           pose_prior_id,
                           is_deprecated_image_prior);
  }

  std::vector<PosePrior> ReadAllPosePriors() const override {
    PYBIND11_OVERRIDE_PURE(std::vector<PosePrior>, Database, ReadAllPosePriors);
  }

  FeatureKeypointsBlob ReadKeypointsBlob(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(
        FeatureKeypointsBlob, Database, ReadKeypointsBlob, image_id);
  }

  FeatureKeypoints ReadKeypoints(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(FeatureKeypoints, Database, ReadKeypoints, image_id);
  }

  FeatureDescriptors ReadDescriptors(image_t image_id) const override {
    PYBIND11_OVERRIDE_PURE(
        FeatureDescriptors, Database, ReadDescriptors, image_id);
  }

  FeatureMatchesBlob ReadMatchesBlob(image_t image_id1,
                                     image_t image_id2) const override {
    PYBIND11_OVERRIDE_PURE(
        FeatureMatchesBlob, Database, ReadMatchesBlob, image_id1, image_id2);
  }

  FeatureMatches ReadMatches(image_t image_id1,
                             image_t image_id2) const override {
    PYBIND11_OVERRIDE_PURE(
        FeatureMatches, Database, ReadMatches, image_id1, image_id2);
  }

  std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> ReadAllMatchesBlob()
      const override {
    using ReturnType = std::vector<std::pair<image_pair_t, FeatureMatchesBlob>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, Database, ReadAllMatchesBlob);
  }
  std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches()
      const override {
    using ReturnType = std::vector<std::pair<image_pair_t, FeatureMatches>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, Database, ReadAllMatches);
  }
  std::vector<std::pair<image_pair_t, int>> ReadNumMatches() const override {
    using ReturnType = std::vector<std::pair<image_pair_t, int>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, Database, ReadNumMatches);
  }

  TwoViewGeometry ReadTwoViewGeometry(image_t image_id1,
                                      image_t image_id2) const override {
    PYBIND11_OVERRIDE_PURE(
        TwoViewGeometry, Database, ReadTwoViewGeometry, image_id1, image_id2);
  }

  std::vector<std::pair<image_pair_t, TwoViewGeometry>> ReadTwoViewGeometries()
      const override {
    using ReturnType = std::vector<std::pair<image_pair_t, TwoViewGeometry>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, Database, ReadTwoViewGeometries);
  }

  std::vector<std::pair<image_pair_t, int>> ReadTwoViewGeometryNumInliers()
      const override {
    using ReturnType = std::vector<std::pair<image_pair_t, int>>;
    PYBIND11_OVERRIDE_PURE(ReturnType, Database, ReadTwoViewGeometryNumInliers);
  }

  rig_t WriteRig(const Rig& rig, bool use_rig_id = false) override {
    PYBIND11_OVERRIDE_PURE(rig_t, Database, WriteRig, rig, use_rig_id);
  }

  camera_t WriteCamera(const Camera& camera,
                       bool use_camera_id = false) override {
    PYBIND11_OVERRIDE_PURE(
        camera_t, Database, WriteCamera, camera, use_camera_id);
  }

  frame_t WriteFrame(const Frame& frame, bool use_frame_id = false) override {
    PYBIND11_OVERRIDE_PURE(frame_t, Database, WriteFrame, frame, use_frame_id);
  }

  image_t WriteImage(const Image& image, bool use_image_id = false) override {
    PYBIND11_OVERRIDE_PURE(image_t, Database, WriteImage, image, use_image_id);
  }

  pose_prior_t WritePosePrior(const PosePrior& pose_prior,
                              bool use_pose_prior_id = false) override {
    PYBIND11_OVERRIDE_PURE(
        pose_prior_t, Database, WritePosePrior, pose_prior, use_pose_prior_id);
  }

  void WriteKeypoints(image_t image_id,
                      const FeatureKeypoints& keypoints) override {
    PYBIND11_OVERRIDE_PURE(void, Database, WriteKeypoints, image_id, keypoints);
  }

  void WriteKeypoints(image_t image_id,
                      const FeatureKeypointsBlob& blob) override {
    PYBIND11_OVERRIDE_PURE(void, Database, WriteKeypoints, image_id, blob);
  }

  void WriteDescriptors(image_t image_id,
                        const FeatureDescriptors& descriptors) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, WriteDescriptors, image_id, descriptors);
  }

  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatches& matches) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, WriteMatches, image_id1, image_id2, matches);
  }

  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatchesBlob& blob) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, WriteMatches, image_id1, image_id2, blob);
  }

  void WriteTwoViewGeometry(image_t image_id1,
                            image_t image_id2,
                            const TwoViewGeometry& two_view_geometry) override {
    PYBIND11_OVERRIDE_PURE(void,
                           Database,
                           WriteTwoViewGeometry,
                           image_id1,
                           image_id2,
                           two_view_geometry);
  }

  void UpdateRig(const Rig& rig) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdateRig, rig);
  }

  void UpdateCamera(const Camera& camera) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdateCamera, camera);
  }

  void UpdateFrame(const Frame& frame) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdateFrame, frame);
  }

  void UpdateImage(const Image& image) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdateImage, image);
  }

  void UpdatePosePrior(const PosePrior& pose_prior) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdatePosePrior, pose_prior);
  }

  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypoints& keypoints) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, UpdateKeypoints, image_id, keypoints);
  }

  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypointsBlob& blob) override {
    PYBIND11_OVERRIDE_PURE(void, Database, UpdateKeypoints, image_id, blob);
  }

  void UpdateTwoViewGeometry(
      image_t image_id1,
      image_t image_id2,
      const TwoViewGeometry& two_view_geometry) override {
    PYBIND11_OVERRIDE_PURE(void,
                           Database,
                           UpdateTwoViewGeometry,
                           image_id1,
                           image_id2,
                           two_view_geometry);
  }

  void DeleteMatches(image_t image_id1, image_t image_id2) override {
    PYBIND11_OVERRIDE_PURE(void, Database, DeleteMatches, image_id1, image_id2);
  }

  void DeleteTwoViewGeometry(image_t image_id1, image_t image_id2) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, DeleteTwoViewGeometry, image_id1, image_id2);
  }

  void DeleteInlierMatches(image_t image_id1, image_t image_id2) override {
    PYBIND11_OVERRIDE_PURE(
        void, Database, DeleteInlierMatches, image_id1, image_id2);
  }

  void ClearAllTables() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearAllTables);
  }

  void ClearRigs() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearRigs);
  }

  void ClearCameras() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearCameras);
  }

  void ClearFrames() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearFrames);
  }

  void ClearImages() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearImages);
  }

  void ClearPosePriors() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearPosePriors);
  }

  void ClearDescriptors() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearDescriptors);
  }

  void ClearKeypoints() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearKeypoints);
  }

  void ClearMatches() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearMatches);
  }

  void ClearTwoViewGeometries() override {
    PYBIND11_OVERRIDE_PURE(void, Database, ClearTwoViewGeometries);
  }

  void BeginTransaction() const override {
    PYBIND11_OVERRIDE_PURE(void, Database, BeginTransaction);
  }

  void EndTransaction() const override {
    PYBIND11_OVERRIDE_PURE(void, Database, EndTransaction);
  }
};

}  // namespace

void BindDatabase(py::module& m) {
  py::classh<Database, PyDatabaseImpl> PyDatabase(m, "Database");
  PyDatabase.def(py::init<>())
      .def_static("open", &Database::Open, "path"_a)
      .def("close", &Database::Close)
      .def("clone", &Database::Clone)
      .def("__enter__", [](Database& self) { return &self; })
      .def("__exit__", [](Database& self, const py::args&) { self.Close(); })
      .def("exists_rig", &Database::ExistsRig, "rig_id"_a)
      .def("exists_camera", &Database::ExistsCamera, "camera_id"_a)
      .def("exists_frame", &Database::ExistsFrame, "frame_id"_a)
      .def("exists_image", &Database::ExistsImage, "image_id"_a)
      .def("exists_image", &Database::ExistsImageWithName, "name"_a)
      .def("exists_pose_prior",
           &Database::ExistsPosePrior,
           "pose_prior_id"_a,
           "is_deprecated_image_prior"_a = true)
      .def("exists_keypoints", &Database::ExistsKeypoints, "image_id"_a)
      .def("exists_descriptors", &Database::ExistsDescriptors, "image_id"_a)
      .def("exists_matches",
           &Database::ExistsMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("exists_two_view_geometry",
           &Database::ExistsTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a)
      .def("num_rigs", &Database::NumRigs)
      .def("num_cameras", &Database::NumCameras)
      .def("num_frames", &Database::NumFrames)
      .def("num_images", &Database::NumImages)
      .def("num_pose_priors", &Database::NumPosePriors)
      .def("num_keypoints", &Database::NumKeypoints)
      .def("num_keypoints_for_image",
           &Database::NumKeypointsForImage,
           "image_id"_a)
      .def("num_descriptors", &Database::NumDescriptors)
      .def("num_descriptors_for_image",
           &Database::NumDescriptorsForImage,
           "image_id"_a)
      .def("num_matches", &Database::NumMatches)
      .def("num_inlier_matches", &Database::NumInlierMatches)
      .def("num_matched_image_pairs", &Database::NumMatchedImagePairs)
      .def("num_verified_image_pairs", &Database::NumVerifiedImagePairs)
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
      .def("read_pose_prior",
           &Database::ReadPosePrior,
           "pose_prior_id"_a,
           "is_deprecated_image_prior"_a = true)
      .def("read_all_pose_priors", &Database::ReadAllPosePriors)
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
      .def(
          "read_num_matches",
          [](const Database& self) {
            std::vector<std::pair<image_pair_t, int>> pair_ids_and_num_matches =
                self.ReadNumMatches();
            std::vector<image_pair_t> all_pair_ids;
            all_pair_ids.reserve(pair_ids_and_num_matches.size());
            std::vector<int> all_num_matches;
            all_num_matches.reserve(pair_ids_and_num_matches.size());
            for (auto& [pair_id, num_matches] : pair_ids_and_num_matches) {
              all_pair_ids.push_back(pair_id);
              all_num_matches.push_back(num_matches);
            }
            return std::make_pair(std::move(all_pair_ids),
                                  std::move(all_num_matches));
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
      .def("write_rig", &Database::WriteRig, "rig"_a, "use_rig_id"_a = false)
      .def("write_camera",
           &Database::WriteCamera,
           "camera"_a,
           "use_camera_id"_a = false)
      .def("write_frame",
           &Database::WriteFrame,
           "frame"_a,
           "use_frame_id"_a = false)
      .def("write_image",
           &Database::WriteImage,
           "image"_a,
           "use_image_id"_a = false)
      .def("write_pose_prior",
           &Database::WritePosePrior,
           "pose_prior"_a,
           "use_pose_prior_id"_a = false)
      .def("write_keypoints",
           py::overload_cast<image_t, const FeatureKeypointsBlob&>(
               &Database::WriteKeypoints),
           "image_id"_a,
           "keypoints"_a)
      .def("write_descriptors",
           &Database::WriteDescriptors,
           "image_id"_a,
           "descriptors"_a)
      .def("write_matches",
           py::overload_cast<image_t, image_t, const FeatureMatchesBlob&>(
               &Database::WriteMatches),
           "image_id1"_a,
           "image_id2"_a,
           "matches"_a)
      .def("write_two_view_geometry",
           &Database::WriteTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a,
           "two_view_geometry"_a)
      .def("update_rig", &Database::UpdateRig, "rig"_a)
      .def("update_camera", &Database::UpdateCamera, "camera"_a)
      .def("update_frame", &Database::UpdateFrame, "frame"_a)
      .def("update_image", &Database::UpdateImage, "image"_a)
      .def("update_pose_prior", &Database::UpdatePosePrior, "pose_prior"_a)
      .def("update_keypoints",
           py::overload_cast<image_t, const FeatureKeypointsBlob&>(
               &Database::UpdateKeypoints),
           "image_id"_a,
           "keypoints"_a)
      .def("update_two_view_geometry",
           &Database::UpdateTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a,
           "two_view_geometry"_a)
      .def("delete_matches",
           &Database::DeleteMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("delete_two_view_geometry",
           &Database::DeleteTwoViewGeometry,
           "image_id1"_a,
           "image_id2"_a)
      .def("delete_inlier_matches",
           &Database::DeleteInlierMatches,
           "image_id1"_a,
           "image_id2"_a)
      .def("clear_all_tables", &Database::ClearAllTables)
      .def("clear_rigs", &Database::ClearRigs)
      .def("clear_cameras", &Database::ClearCameras)
      .def("clear_frames", &Database::ClearFrames)
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

  py::classh<DatabaseTransactionWrapper>(m, "DatabaseTransaction")
      .def(py::init<Database*>(), "database"_a)
      .def("__enter__", &DatabaseTransactionWrapper::Enter)
      .def("__exit__", &DatabaseTransactionWrapper::Exit);
}
