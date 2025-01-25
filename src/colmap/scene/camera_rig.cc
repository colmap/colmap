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

#include "colmap/scene/camera_rig.h"

#include "colmap/util/misc.h"

namespace colmap {

CameraRig::CameraRig() {}

size_t CameraRig::NumCameras() const { return cams_from_rigs_.size(); }

size_t CameraRig::NumSnapshots() const { return snapshots_.size(); }

bool CameraRig::HasCamera(const camera_t camera_id) const {
  return cams_from_rigs_.count(camera_id);
}

camera_t CameraRig::RefCameraId() const { return ref_camera_id_; }

void CameraRig::SetRefCameraId(const camera_t camera_id) {
  THROW_CHECK(HasCamera(camera_id));
  ref_camera_id_ = camera_id;
}

std::vector<camera_t> CameraRig::GetCameraIds() const {
  std::vector<camera_t> rig_camera_ids;
  rig_camera_ids.reserve(cams_from_rigs_.size());
  for (const auto& rig_camera : cams_from_rigs_) {
    rig_camera_ids.push_back(rig_camera.first);
  }
  return rig_camera_ids;
}

const std::vector<std::vector<image_t>>& CameraRig::Snapshots() const {
  return snapshots_;
}

void CameraRig::AddCamera(const camera_t camera_id,
                          const Rigid3d& cam_from_rig) {
  THROW_CHECK(!HasCamera(camera_id));
  THROW_CHECK_EQ(NumSnapshots(), 0);
  cams_from_rigs_.emplace(camera_id, cam_from_rig);
}

void CameraRig::AddSnapshot(const std::vector<image_t>& image_ids) {
  THROW_CHECK(!image_ids.empty());
  THROW_CHECK_LE(image_ids.size(), NumCameras());
  THROW_CHECK(!VectorContainsDuplicateValues(image_ids));
  snapshots_.push_back(image_ids);
}

void CameraRig::Check(const Reconstruction& reconstruction) const {
  THROW_CHECK(HasCamera(ref_camera_id_));

  for (const auto& rig_camera : cams_from_rigs_) {
    THROW_CHECK(reconstruction.ExistsCamera(rig_camera.first));
  }

  std::unordered_set<image_t> all_image_ids;
  for (const auto& snapshot : snapshots_) {
    THROW_CHECK(!snapshot.empty());
    THROW_CHECK_LE(snapshot.size(), NumCameras());
    bool has_ref_camera = false;
    for (const auto image_id : snapshot) {
      THROW_CHECK(reconstruction.ExistsImage(image_id));
      THROW_CHECK_EQ(all_image_ids.count(image_id), 0);
      all_image_ids.insert(image_id);
      const auto& image = reconstruction.Image(image_id);
      THROW_CHECK(HasCamera(image.CameraId()));
      if (image.CameraId() == ref_camera_id_) {
        has_ref_camera = true;
      }
    }
    THROW_CHECK(has_ref_camera);
  }
}

const Rigid3d& CameraRig::CamFromRig(const camera_t camera_id) const {
  return cams_from_rigs_.at(camera_id);
}

Rigid3d& CameraRig::CamFromRig(const camera_t camera_id) {
  return cams_from_rigs_.at(camera_id);
}

double CameraRig::ComputeRigFromWorldScale(
    const Reconstruction& reconstruction) const {
  THROW_CHECK_GT(NumSnapshots(), 0);
  const size_t num_cameras = NumCameras();
  THROW_CHECK_GT(num_cameras, 0);

  double rig_from_world_scale = 0;
  size_t num_dists = 0;
  std::vector<Eigen::Vector3d> proj_centers_in_rig(num_cameras);
  std::vector<Eigen::Vector3d> proj_centers_in_world(num_cameras);
  for (const auto& snapshot : snapshots_) {
    for (size_t i = 0; i < num_cameras; ++i) {
      const auto& image = reconstruction.Image(snapshot[i]);
      proj_centers_in_rig[i] =
          Inverse(CamFromRig(image.CameraId())).translation;
      proj_centers_in_world[i] = image.ProjectionCenter();
    }

    // Accumulate the relative scale for all pairs of camera distances.
    for (size_t i = 0; i < num_cameras; ++i) {
      for (size_t j = 0; j < i; ++j) {
        const double rig_dist =
            (proj_centers_in_rig[i] - proj_centers_in_rig[j]).norm();
        const double world_dist =
            (proj_centers_in_world[i] - proj_centers_in_world[j]).norm();
        const double kMinDist = 1e-6;
        if (rig_dist > kMinDist && world_dist > kMinDist) {
          rig_from_world_scale += rig_dist / world_dist;
          num_dists += 1;
        }
      }
    }
  }

  if (num_dists == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return rig_from_world_scale / num_dists;
}

bool CameraRig::ComputeCamsFromRigs(const Reconstruction& reconstruction) {
  THROW_CHECK_GT(NumSnapshots(), 0);
  THROW_CHECK_NE(ref_camera_id_, kInvalidCameraId);

  for (auto& cam_from_rig : cams_from_rigs_) {
    cam_from_rig.second.translation = Eigen::Vector3d::Zero();
  }

  std::unordered_map<camera_t, std::vector<Eigen::Quaterniond>>
      cam_from_ref_cam_rotations;
  for (const auto& snapshot : snapshots_) {
    // Find the image of the reference camera in the current snapshot.
    const Image* ref_image = nullptr;
    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() == ref_camera_id_) {
        ref_image = &image;
        break;
      }
    }

    const Rigid3d world_from_ref_cam =
        Inverse(THROW_CHECK_NOTNULL(ref_image)->CamFromWorld());

    // Compute the relative poses from all cameras in the current snapshot to
    // the reference camera.
    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() != ref_camera_id_) {
        const Rigid3d cam_from_ref_cam =
            image.CamFromWorld() * world_from_ref_cam;
        cam_from_ref_cam_rotations[image.CameraId()].push_back(
            cam_from_ref_cam.rotation);
        CamFromRig(image.CameraId()).translation +=
            cam_from_ref_cam.translation;
      }
    }
  }

  cams_from_rigs_.at(ref_camera_id_) = Rigid3d();

  // Compute the average relative poses.
  for (auto& cam_from_rig : cams_from_rigs_) {
    if (cam_from_rig.first != ref_camera_id_) {
      if (cam_from_ref_cam_rotations.count(cam_from_rig.first) == 0) {
        LOG(INFO) << "Need at least one snapshot with an image of camera "
                  << cam_from_rig.first << " and the reference camera "
                  << ref_camera_id_
                  << " to compute its relative pose in the camera rig";
        return false;
      }
      const std::vector<Eigen::Quaterniond>& cam_from_rig_rotations =
          cam_from_ref_cam_rotations.at(cam_from_rig.first);
      const std::vector<double> weights(cam_from_rig_rotations.size(), 1.0);
      cam_from_rig.second.rotation =
          AverageQuaternions(cam_from_rig_rotations, weights);
      cam_from_rig.second.translation /= cam_from_rig_rotations.size();
    }
  }
  return true;
}

Rigid3d CameraRig::ComputeRigFromWorld(
    const size_t snapshot_idx, const Reconstruction& reconstruction) const {
  const auto& snapshot = snapshots_.at(snapshot_idx);

  std::vector<Eigen::Quaterniond> rig_from_world_rotations;
  rig_from_world_rotations.reserve(snapshot.size());
  Eigen::Vector3d rig_from_world_translations = Eigen::Vector3d::Zero();
  for (const auto image_id : snapshot) {
    const auto& image = reconstruction.Image(image_id);
    const Rigid3d rig_from_world =
        Inverse(CamFromRig(image.CameraId())) * image.CamFromWorld();
    rig_from_world_rotations.push_back(rig_from_world.rotation);
    rig_from_world_translations += rig_from_world.translation;
  }

  const std::vector<double> rotation_weights(snapshot.size(), 1);
  return Rigid3d(AverageQuaternions(rig_from_world_rotations, rotation_weights),
                 rig_from_world_translations /= snapshot.size());
}

}  // namespace colmap
