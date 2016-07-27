// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/camera_rig.h"

#include "base/pose.h"
#include "util/logging.h"

namespace colmap {

CameraRig::CameraRig() {}

size_t CameraRig::NumCameras() const { return rig_cameras_.size(); }

size_t CameraRig::NumSnapshots() const { return snapshots_.size(); }

bool CameraRig::HasCamera(const camera_t camera_id) const {
  return rig_cameras_.count(camera_id);
}

camera_t CameraRig::RefCameraId() const { return ref_camera_id_; }

void CameraRig::SetRefCameraId(const camera_t camera_id) {
  CHECK(HasCamera(camera_id));
  ref_camera_id_ = camera_id;
}

std::vector<camera_t> CameraRig::GetCameraIds() const {
  std::vector<camera_t> rig_camera_ids;
  rig_camera_ids.reserve(NumCameras());
  for (const auto& rig_camera : rig_cameras_) {
    rig_camera_ids.push_back(rig_camera.first);
  }
  return rig_camera_ids;
}

const std::vector<std::vector<image_t>>& CameraRig::Snapshots() const {
  return snapshots_;
}

void CameraRig::AddCamera(const camera_t camera_id,
                          const Eigen::Vector4d rel_qvec,
                          const Eigen::Vector3d& rel_tvec) {
  CHECK(!HasCamera(camera_id));
  CHECK_EQ(NumSnapshots(), 0);
  RigCamera rig_camera;
  rig_camera.rel_qvec = rel_qvec;
  rig_camera.rel_tvec = rel_tvec;
  rig_cameras_.emplace(camera_id, rig_camera);
}

void CameraRig::AddSnapshot(const std::vector<image_t>& image_ids) {
  CHECK(!image_ids.empty());
  CHECK_EQ(image_ids.size(), NumCameras());
  snapshots_.push_back(image_ids);
}

void CameraRig::Check(const Reconstruction& reconstruction) const {
  CHECK(HasCamera(ref_camera_id_));

  for (const auto& rig_camera : rig_cameras_) {
    CHECK(reconstruction.ExistsCamera(rig_camera.first));
  }

  std::unordered_set<image_t> all_image_ids;
  for (const auto& snapshot : snapshots_) {
    CHECK_EQ(snapshot.size(), NumCameras());
    for (const auto image_id : snapshot) {
      CHECK(reconstruction.ExistsImage(image_id));
      CHECK_EQ(all_image_ids.count(image_id), 0);
      all_image_ids.insert(image_id);
      const auto& image = reconstruction.Image(image_id);
      CHECK(HasCamera(image.CameraId()));
    }
  }
}

Eigen::Vector4d& CameraRig::RelativeQvec(const camera_t camera_id) {
  return rig_cameras_.at(camera_id).rel_qvec;
}

const Eigen::Vector4d& CameraRig::RelativeQvec(const camera_t camera_id) const {
  return rig_cameras_.at(camera_id).rel_qvec;
}

Eigen::Vector3d& CameraRig::RelativeTvec(const camera_t camera_id) {
  return rig_cameras_.at(camera_id).rel_tvec;
}

const Eigen::Vector3d& CameraRig::RelativeTvec(const camera_t camera_id) const {
  return rig_cameras_.at(camera_id).rel_tvec;
}

double CameraRig::ComputeScale(const Reconstruction& reconstruction) const {
  CHECK_GT(NumSnapshots(), 0);
  CHECK_GT(NumCameras(), 0);
  double scaling_factor = 0;
  size_t num_dists = 0;
  std::vector<Eigen::Vector3d> rel_proj_centers(NumCameras());
  std::vector<Eigen::Vector3d> abs_proj_centers(NumCameras());
  for (const auto& snapshot : snapshots_) {
    // Compute the projection relative and absolute projection centers.
    for (size_t i = 0; i < NumCameras(); ++i) {
      const auto& image = reconstruction.Image(snapshot[i]);
      rel_proj_centers[i] = ProjectionCenterFromParameters(
          RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()));
      abs_proj_centers[i] = image.ProjectionCenter();
    }

    // Accumulate the scaling factor for all pairs of camera distances.
    for (size_t i = 0; i < NumCameras(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        const double rel_dist =
            (rel_proj_centers[i] - rel_proj_centers[j]).norm();
        const double abs_dist =
            (abs_proj_centers[i] - abs_proj_centers[j]).norm();
        const double kMinDist = 1e-6;
        if (rel_dist > kMinDist && abs_dist > kMinDist) {
          scaling_factor += rel_dist / abs_dist;
          num_dists += 1;
        }
      }
    }
  }

  if (num_dists == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return scaling_factor / num_dists;
}

void CameraRig::ComputeRelativePoses(const Reconstruction& reconstruction) {
  CHECK_GT(NumSnapshots(), 0);
  CHECK_NE(ref_camera_id_, kInvalidCameraId);

  for (auto& rig_camera : rig_cameras_) {
    rig_camera.second.rel_tvec = Eigen::Vector3d::Zero();
  }

  std::unordered_map<camera_t, std::vector<Eigen::Vector4d>> rel_qvecs;

  for (const auto& snapshot : snapshots_) {
    const Image* ref_image = nullptr;
    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() == ref_camera_id_) {
        ref_image = &image;
      }
    }

    CHECK_NOTNULL(ref_image);

    for (const auto image_id : snapshot) {
      const auto& image = reconstruction.Image(image_id);
      if (image.CameraId() != ref_camera_id_) {
        Eigen::Vector4d rel_qvec;
        Eigen::Vector3d rel_tvec;
        ComputeRelativePose(ref_image->Qvec(), ref_image->Tvec(), image.Qvec(),
                            image.Tvec(), &rel_qvec, &rel_tvec);

        rel_qvecs[image.CameraId()].push_back(rel_qvec);
        RelativeTvec(image.CameraId()) += rel_tvec;
      }
    }
  }

  RelativeQvec(ref_camera_id_) = ComposeIdentityQuaternion();
  RelativeTvec(ref_camera_id_) = Eigen::Vector3d(0, 0, 0);

  for (auto& rig_camera : rig_cameras_) {
    if (rig_camera.first != ref_camera_id_) {
      const std::vector<double> rel_qvec_weights(NumSnapshots(), 1.0);
      rig_camera.second.rel_qvec =
          AverageQuaternions(rel_qvecs.at(rig_camera.first), rel_qvec_weights);
      rig_camera.second.rel_tvec /= NumSnapshots();
    }
  }
}

void CameraRig::ComputeAbsolutePose(const size_t snapshot_idx,
                                    const Reconstruction& reconstruction,
                                    Eigen::Vector4d* abs_qvec,
                                    Eigen::Vector3d* abs_tvec) const {
  const auto& snapshot = snapshots_.at(snapshot_idx);

  std::vector<Eigen::Vector4d> abs_qvecs;
  *abs_tvec = Eigen::Vector3d::Zero();

  for (const auto image_id : snapshot) {
    const auto& image = reconstruction.Image(image_id);
    Eigen::Vector4d inv_rel_qvec;
    Eigen::Vector3d inv_rel_tvec;
    InvertPose(RelativeQvec(image.CameraId()), RelativeTvec(image.CameraId()),
               &inv_rel_qvec, &inv_rel_tvec);

    const Eigen::Vector4d qvec =
        ConcatenateQuaternions(image.Qvec(), inv_rel_qvec);
    const Eigen::Vector3d tvec = QuaternionRotatePoint(
        inv_rel_qvec, image.Tvec() - RelativeTvec(image.CameraId()));

    abs_qvecs.push_back(qvec);
    *abs_tvec += tvec;
  }

  const std::vector<double> abs_qvec_weights(snapshot.size(), 1);
  *abs_qvec = AverageQuaternions(abs_qvecs, abs_qvec_weights);
  *abs_tvec /= snapshot.size();
}

}  // namespace colmap
