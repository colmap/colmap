// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/scene/database.h"

#include "colmap/scene/database_sqlite.h"

namespace colmap {

std::vector<Database::Factory> Database::factories_ = {&OpenSqliteDatabase};

void Database::Register(Factory factory) {
  factories_.push_back(std::move(factory));
}

Database::~Database() = default;

std::shared_ptr<Database> Database::Open(const std::string& path) {
  for (auto it = factories_.rbegin(); it != factories_.rend(); ++it) {
    try {
      return (*it)(path);
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "Failed to open database with registered factory, because: "
          << e.what() << ". Trying next registered factory.";
    }
  }
  throw std::runtime_error("No registered database factory succeeded.");
}

void Database::Merge(const Database& database1,
                     const Database& database2,
                     Database* merged_database) {
  // Merge the cameras.

  std::unordered_map<camera_t, camera_t> new_camera_ids1;
  for (const auto& camera : database1.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids1.emplace(camera.camera_id, new_camera_id);
  }

  std::unordered_map<camera_t, camera_t> new_camera_ids2;
  for (const auto& camera : database2.ReadAllCameras()) {
    const camera_t new_camera_id = merged_database->WriteCamera(camera);
    new_camera_ids2.emplace(camera.camera_id, new_camera_id);
  }

  // Merge the rigs.

  auto update_rig =
      [](const Rig& rig,
         const std::unordered_map<camera_t, camera_t>& new_camera_ids) {
        if (rig.NumSensors() == 0) {
          return rig;
        }
        Rig updated_rig;
        updated_rig.SetRigId(rig.RigId());
        sensor_t ref_sensor_id = rig.RefSensorId();
        if (ref_sensor_id.type == SensorType::CAMERA) {
          ref_sensor_id.id = new_camera_ids.at(ref_sensor_id.id);
        }
        updated_rig.AddRefSensor(ref_sensor_id);
        for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
          sensor_t updated_sensor_id = sensor_id;
          if (sensor_id.type == SensorType::CAMERA) {
            updated_sensor_id.id = new_camera_ids.at(sensor_id.id);
          }
          updated_rig.AddSensor(updated_sensor_id, sensor_from_rig);
        }
        return updated_rig;
      };

  std::unordered_map<rig_t, rig_t> new_rig_ids1;
  for (auto& rig : database1.ReadAllRigs()) {
    const rig_t new_rig_id =
        merged_database->WriteRig(update_rig(rig, new_camera_ids1));
    new_rig_ids1.emplace(rig.RigId(), new_rig_id);
  }

  std::unordered_map<rig_t, rig_t> new_rig_ids2;
  for (auto& rig : database2.ReadAllRigs()) {
    const rig_t new_rig_id =
        merged_database->WriteRig(update_rig(rig, new_camera_ids2));
    new_rig_ids2.emplace(rig.RigId(), new_rig_id);
  }

  // Merge the images.

  std::unordered_map<image_t, image_t> new_image_ids1;
  for (auto& image : database1.ReadAllImages()) {
    image.SetCameraId(new_camera_ids1.at(image.CameraId()));
    image.SetFrameId(kInvalidFrameId);
    THROW_CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids1.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database1.ReadKeypoints(image.ImageId());
    const auto descriptors = database1.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    merged_database->WriteDescriptors(new_image_id, descriptors);
    if (database1.ExistsPosePrior(image.ImageId())) {
      merged_database->WritePosePrior(new_image_id,
                                      database1.ReadPosePrior(image.ImageId()));
    }
  }

  std::unordered_map<image_t, image_t> new_image_ids2;
  for (auto& image : database2.ReadAllImages()) {
    image.SetCameraId(new_camera_ids2.at(image.CameraId()));
    image.SetFrameId(kInvalidFrameId);
    THROW_CHECK(!merged_database->ExistsImageWithName(image.Name()))
        << "The two databases must not contain images with the same name, but "
           "there are images with name "
        << image.Name() << " in both databases";
    const image_t new_image_id = merged_database->WriteImage(image);
    new_image_ids2.emplace(image.ImageId(), new_image_id);
    const auto keypoints = database2.ReadKeypoints(image.ImageId());
    const auto descriptors = database2.ReadDescriptors(image.ImageId());
    merged_database->WriteKeypoints(new_image_id, keypoints);
    merged_database->WriteDescriptors(new_image_id, descriptors);
    if (database2.ExistsPosePrior(image.ImageId())) {
      merged_database->WritePosePrior(new_image_id,
                                      database2.ReadPosePrior(image.ImageId()));
    }
  }

  // Merge the frames.

  auto update_frame =
      [](const Frame& frame,
         const std::unordered_map<camera_t, camera_t>& new_camera_ids,
         const std::unordered_map<image_t, image_t>& new_image_ids) {
        Frame updated_frame;
        updated_frame.SetFrameId(frame.FrameId());
        updated_frame.SetRigId(frame.RigId());
        for (data_t data_id : frame.DataIds()) {
          if (data_id.sensor_id.type == SensorType::CAMERA) {
            data_id.id = new_image_ids.at(data_id.id);
            data_id.sensor_id.id = new_camera_ids.at(data_id.sensor_id.id);
            updated_frame.AddDataId(data_id);
          } else {
            std::ostringstream error;
            error << "Data type not supported: " << data_id.sensor_id.type;
            throw std::runtime_error(error.str());
          }
        }
        return updated_frame;
      };

  for (Frame& frame : database1.ReadAllFrames()) {
    merged_database->WriteFrame(
        update_frame(frame, new_camera_ids1, new_image_ids1));
  }

  for (Frame& frame : database2.ReadAllFrames()) {
    merged_database->WriteFrame(
        update_frame(frame, new_camera_ids2, new_image_ids2));
  }

  // Merge the matches.

  for (const auto& matches : database1.ReadAllMatches()) {
    const auto image_pair = PairIdToImagePair(matches.first);

    const image_t new_image_id1 = new_image_ids1.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids1.at(image_pair.second);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  for (const auto& matches : database2.ReadAllMatches()) {
    const auto image_pair = PairIdToImagePair(matches.first);

    const image_t new_image_id1 = new_image_ids2.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids2.at(image_pair.second);

    merged_database->WriteMatches(new_image_id1, new_image_id2, matches.second);
  }

  // Merge the two-view geometries.

  for (const auto& [pair_id, two_view_geometry] :
       database1.ReadTwoViewGeometries()) {
    const auto image_pair = PairIdToImagePair(pair_id);

    const image_t new_image_id1 = new_image_ids1.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids1.at(image_pair.second);

    merged_database->WriteTwoViewGeometry(
        new_image_id1, new_image_id2, two_view_geometry);
  }

  for (const auto& [pair_id, two_view_geometry] :
       database2.ReadTwoViewGeometries()) {
    const auto image_pair = PairIdToImagePair(pair_id);

    const image_t new_image_id1 = new_image_ids2.at(image_pair.first);
    const image_t new_image_id2 = new_image_ids2.at(image_pair.second);

    merged_database->WriteTwoViewGeometry(
        new_image_id1, new_image_id2, two_view_geometry);
  }
}

DatabaseTransaction::DatabaseTransaction(Database* database)
    : database_(database), database_lock_(database->transaction_mutex_) {
  THROW_CHECK_NOTNULL(database_);
  database_->BeginTransaction();
}

DatabaseTransaction::~DatabaseTransaction() { database_->EndTransaction(); }

}  // namespace colmap
