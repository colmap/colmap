// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_BASE_CAMERA_RIG_H_
#define COLMAP_SRC_BASE_CAMERA_RIG_H_

#include <unordered_map>
#include <vector>

#include "base/camera.h"
#include "base/pose.h"
#include "base/reconstruction.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// This class holds information about the relative configuration of camera rigs.
// Camera rigs are composed of multiple cameras with a rigid relative extrinsic
// configuration over multiple snapshots. Snapshots are defined as the
// collection of images captured simultaneously by all cameras in the rig.
class CameraRig {
 public:
  CameraRig();

  // The number of cameras in the rig.
  size_t NumCameras() const;

  // The number of snapshots captured by this rig.
  size_t NumSnapshots() const;

  // Check whether the given camera is part of the rig.
  bool HasCamera(const camera_t camera_id) const;

  // Access the reference camera.
  camera_t RefCameraId() const;
  void SetRefCameraId(const camera_t camera_id);

  // Get the identifiers of the cameras in the rig.
  std::vector<camera_t> GetCameraIds() const;

  // Get the snapshots of the camera rig.
  const std::vector<std::vector<image_t>>& Snapshots() const;

  // Add a new camera to the rig. The relative pose may contain dummy values and
  // can then be computed automatically from a given reconstruction using the
  // method `ComputeRelativePoses`.
  void AddCamera(const camera_t camera_id, const Eigen::Vector4d rel_qvec,
                 const Eigen::Vector3d& rel_tvec);

  // Add the images of a single snapshot to rig. A snapshot consists of the
  // captured images of all cameras of the rig. All images of a snapshot share
  // the same global camera rig pose, i.e. all images in the camera rig are
  // captured simultaneously.
  void AddSnapshot(const std::vector<image_t>& image_ids);

  // Check whether the camera rig setup is valid.
  void Check(const Reconstruction& reconstruction) const;

  // Get the relative poses of the cameras in the rig.
  Eigen::Vector4d& RelativeQvec(const camera_t camera_id);
  const Eigen::Vector4d& RelativeQvec(const camera_t camera_id) const;
  Eigen::Vector3d& RelativeTvec(const camera_t camera_id);
  const Eigen::Vector3d& RelativeTvec(const camera_t camera_id) const;

  // Compute the scaling factor from the reconstruction to the camera rig
  // dimensions by averaging over the distances between the projection centers.
  // Note that this assumes that there is at least one camera pair in the rig
  // with non-zero baseline, otherwise the function returns NaN.
  double ComputeScale(const Reconstruction& reconstruction) const;

  // Compute the relative poses in the rig from the reconstruction by averaging
  // the relative poses over all snapshots. The pose of the reference camera
  // will be the identity transformation. This assumes that the camera rig has
  // snapshots that are registered in the reconstruction.
  void ComputeRelativePoses(const Reconstruction& reconstruction);

  // Compute the absolute camera pose of the rig. The absolute camera pose of
  // the rig is computed as the average of all relative camera poses in the rig
  // and their corresponding image poses in the reconstruction.
  void ComputeAbsolutePose(const size_t snapshot_idx,
                           const Reconstruction& reconstruction,
                           Eigen::Vector4d* abs_qvec,
                           Eigen::Vector3d* abs_tvec) const;

 private:
  struct RigCamera {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector4d rel_qvec = ComposeIdentityQuaternion();
    Eigen::Vector3d rel_tvec = Eigen::Vector3d(0, 0, 0);
  };

  camera_t ref_camera_id_ = kInvalidCameraId;
  EIGEN_STL_UMAP(camera_t, RigCamera) rig_cameras_;
  std::vector<std::vector<image_t>> snapshots_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_CAMERA_RIG_H_
