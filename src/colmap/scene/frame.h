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

#pragma once

#include "colmap/geometry/pose.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <vector>

namespace colmap {

typedef uint64_t frame_t;
typedef uint32_t rig_t;

enum class SensorType {
  Camera = 0,
  IMU = 1,
  Location = 2, // include GNSS, radios, compass, etc.
};

typedef data_t std::pair<SensorType, uint64_t>;

class RigCalibration {
 public:
  inline rig_t RigId() const {
    return rig_id_;
  }

  inline data_t RefDataId() const {
    return ref_data_id_;
  }

  inline bool HasData(data_t data_id) const {
    if (data_id == ref_data_id_)
      return true;
    else
      return sensor_from_rig_.find(data_id) != sensor_from_rig_.end();
  }

  inline bool IsReference(data_t data_id) const {
    return data_id == ref_data_id_;
  }

  inline size_t CountData() const {
    return sensor_from_rig_.size() + 1;
  }

  inline Rigi3d& GetSensorFromRig(data_t data_id) {
    THROW_CHECK(!IsReference(data_id));
    auto it = sensor_from_rig_.find(data_id);
    if (it == sensor_from_rig_.end())
      LOG(FATAL_THROW) << StringPrintf("Data id (%d) not found in the rig calibration", data_id);
    return sensor_from_rig_.at(data_id);
  }

  inline const Rigid3d GetSensorFromRig(data_t data_id) const {
    if (IsReference(data_id))
      return Rigid3d(); // return identity
    else {
      auto it = sensor_from_rig_.find(data_id);
      if (it == sensor_from_rig_.end())
        LOG(FATAL_THROW) << StringPrintf("Data id (%d) not found in the rig calibration", data_id);
      return sensor_from_rig_.at(data_id);
    }
  }

 private:
  rig_t rig_id_;
  data_t ref_data_id_;
  std::unordered_map<data_t, Rigid3d> sensor_from_rig_;
};

struct Frame {
 public:
  inline frame_t FrameId() const { return frame_id_; }
  inline std::unordered_set<data_t>& DataIds() { return data_ids_; }
  inline const std::unordered_set<data_t>& DataIds() const { return data_ids_; }
  inline std::shared_ptr<RigCalibration> RigCalibration() const {
    return rig_calibration_;
  }

  inline bool HasData(data_t data_id) const {
    return data_ids_.find(data_id) != data_ids_.end();
  }

  inline bool HasRigCalibration() const {
    if (rig_calibration_ == nullptr)
      return false;
    else
      return rig_calibration_.CountData() > 1;
  }

  inline const Rigid3d& FrameFromWorld() const { return frame_from_world_; }
  inline Rigid3d& FrameFromWorld() { return frame_from_world_; }

  inline const Rigid3d& SensorFromWorld() const {
    THROW_CHECK(!HasRigCalibration());
    return FrameFromWorld();
  }
  inline Rigid3d SensorFromWorld() const {
    THROW_CHECK(!HasRigCalibration());
    return FrameFromWorld();
  }

  inline Rigid3d SensorFromWorld(data_t data_id) const {
    THROW_CHECK(HasData(data_id));
    if (!HasRigCalibration()) {
      return SensorFromWorld();
    }
    return rig_calibration_.GetSensorFromRig(data_id) * frame_from_world_;
  }

 private:
  frame_t frame_id_;
  std::unordered_set<data_t> data_ids_;
  Rigid3d frame_from_world_;
  std::shared_ptr<RigCalibration> rig_calibration_ = nullptr; // nullptr if no rig
};

}  // namespace colmap
