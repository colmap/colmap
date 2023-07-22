// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/geometry/pose.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <vector>

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
  bool HasCamera(camera_t camera_id) const;

  // Access the reference camera.
  camera_t RefCameraId() const;
  void SetRefCameraId(camera_t camera_id);

  // Get the identifiers of the cameras in the rig.
  std::vector<camera_t> GetCameraIds() const;

  // Get the snapshots of the camera rig.
  const std::vector<std::vector<image_t>>& Snapshots() const;

  // Add a new camera to the rig. The relative pose may contain invalid values
  // and can then be computed automatically from a given reconstruction using
  // the method `ComputeCamsFromRigs`.
  void AddCamera(camera_t camera_id, const Rigid3d& cam_from_rig);

  // Add the images of a single snapshot to rig. A snapshot consists of the
  // captured images of all cameras of the rig. All images of a snapshot share
  // the same global camera rig pose, i.e. all images in the camera rig are
  // captured simultaneously.
  void AddSnapshot(const std::vector<image_t>& image_ids);

  // Check whether the camera rig setup is valid.
  void Check(const Reconstruction& reconstruction) const;

  // Get the relative poses of the cameras in the rig.
  const Rigid3d& CamFromRig(camera_t camera_id) const;
  Rigid3d& CamFromRig(camera_t camera_id);

  // Compute the scaling factor from the world scale of the reconstruction to
  // the camera rig scale by averaging over the distances between the projection
  // centers. Note that this assumes that there is at least one camera pair in
  // the rig with non-zero baseline, otherwise the function returns NaN.
  double ComputeRigFromWorldScale(const Reconstruction& reconstruction) const;

  // Compute the camera rig poses from the reconstruction by averaging
  // the relative poses over all snapshots. The pose of the reference camera
  // will have the identity transformation. This assumes that the camera rig has
  // snapshots that are registered in the reconstruction.
  bool ComputeCamsFromRigs(const Reconstruction& reconstruction);

  // Compute the pose of the camera rig. The rig pose is computed as the average
  // of all relative camera poses in the rig and their corresponding image poses
  // in the reconstruction.
  Rigid3d ComputeRigFromWorld(size_t snapshot_idx,
                              const Reconstruction& reconstruction) const;

 private:
  camera_t ref_camera_id_ = kInvalidCameraId;
  std::unordered_map<camera_t, Rigid3d> cams_from_rigs_;
  std::vector<std::vector<image_t>> snapshots_;
};

}  // namespace colmap
