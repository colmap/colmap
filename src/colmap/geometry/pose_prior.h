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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <optional>
#include <ostream>

#include <Eigen/Core>

namespace colmap {

// Represents a hierarchical coordinate system context.
//
// This structure models a coordinate system hierarchy, where each coordinate
// system is defined relative to its parent frame. It allows defining and
// managing relationships between global, regional, and local coordinate
// systems, such as WGS84, ECEF, ENU, UTM, body frame and model frame.
// e.g. WGS84/GRS80 -> UTM/ENU -> OBJECT_1 -> OBJECT_2, MODEL_1 -> MODEL_2
struct CoordinateSystemContext {
 public:
  MAKE_ENUM_CLASS(CoordinateSystem,
                  -1,
                  UNDEFINED,  // = -1
                  WGS84,      // = 0
                  ECEF,       // = 1
                  GRS80,      // = 2
                  ENU,        // = 3
                  UTM,        // = 4
                  OBJECT,     // = 5, body frame
                  MODEL       // = 6, non-metric frame
  );

  MAKE_ENUM(CoordinateSystemLevel,
            -1,
            UNDEFINED,
            // 0, Global geocentric frame, e.g. WGS84/GRS80/ECEF
            GLOBAL,
            // 1, Regional geodetic frame, based on global frame, e.g. ENU/UTM
            REGIONAL,
            // 2, Local object frame, based on regional frame, e.g. body frame
            LOCAL,
            // 3, Virtual model frame, non-metric and used for simulation
            VIRTUAL);

  // Directly represents the pose of a frame, including orientation and position
  // in parent frame. Not use Rigid3d since WGS84/GRS80 is spherical.
  struct FramePose {
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    // Returns the corresponding Rigid3d transformation from parent.
    // Assumes the pose position is in Cartesian coordinates.
    Rigid3d ToRigid3d() const;

    // Returns a FramePose representing the identity transformation:
    // Position (0, 0, 0) and rotation as the identity matrix.
    static FramePose Identity();
  };

  // Represents a node in a hierarchical coordinate system.
  // Each FrameNode has a parent frame (if any), and an pose describing the
  // position and orientation of the current coordinate system relative to its
  // parent frame.
  struct FrameNode {
    CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;
    // Pose of current coordinate system in parent frame, defines the position
    // and orientaion in parent frame, it's null for global and virtual
    // coordinate system.
    FramePose frame_pose = FramePose::Identity();

    // Parent frame for hierarchical coordinate systems, it's null for global
    // and virtual coordinate system.
    std::unique_ptr<FrameNode> parent_frame = nullptr;

    // Unique identifier for frame, mainly used for object/model frame.
    std::uint32_t frame_id = 0;

    explicit FrameNode(CoordinateSystem system,
                       const FramePose& pose = FramePose::Identity(),
                       FrameNode* parent = nullptr,
                       std::uint32_t id = 0);
    FrameNode(CoordinateSystem system, std::uint32_t id);
    FrameNode(const FrameNode& other);

    CoordinateSystemLevel Level() const;

    inline bool IsCartesian() const;
    inline bool IsMetric() const;

    Eigen::Vector3d ConvertFromParent(
        const Eigen::Vector3d& coordinate_in_parent);
    Eigen::Vector3d ConvertToParent(
        const Eigen::Vector3d& coordinate_in_current);

    // Two frame node equals if they have the same coordinate system name and
    // same id.
    inline bool operator==(const FrameNode& other) const;
    inline bool operator!=(const FrameNode& other) const;
  };

  CoordinateSystemContext() {};
  CoordinateSystemContext(const CoordinateSystemContext& other);

  // This constructor initializes a coordinate system context from a list of
  // frames.
  //
  // e.g. Construct a context: object <- utm <- WGS84
  // ```cpp
  // std::vector<std::pair<FrameNode, FramePose> > frame_chain;
  // frame_chain.emplace_back(FrameNode{CoordinateSystem::OBJECT},
  //                          FramePose::Identity());
  // frame_chain.emplace_back(FrameNode{CoordinateSystem::UTM},
  //                          object_in_utm);
  // frame_chain.emplace_back(FrameNode{CoordinateSystem::WGS84},
  //                          utm_in_wgs84);
  // CoordinateSystemContext context_1(frame_chain);
  // ```
  explicit CoordinateSystemContext(
      const std::vector<std::pair<FrameNode, FramePose> >& frame_chain,
      bool reverse = false);

  ~CoordinateSystemContext();

  // Appends a new parent frame node to the current coordinate system hierarchy
  void Append(const FrameNode& frame, const FramePose& current_base_in_appened);
  // Prepends a new parent frame node to the current coordinate system
  // hierarchy.
  void Prepend(const FrameNode& frame, const FramePose& prepended_in_current);

  // Retrieves the last frame in the hierarchical coordinate system.
  std::optional<FrameNode> MaybeBaseFrame() const;
  // Retrieves the child of base frame (the second last frame in the
  // hierarchical coordinate system).
  std::optional<FrameNode> MaybePenultimateFrame() const;

  // Serializes the entire coordinate system hierarchy into an Eigen matrix.
  // The resulting matrix has the following structure:
  // - Each row: Corresponds to one frame node in the hierarchy.
  // - Column 1: Coordinate system enum.
  // - Columns 2 to 4: Position (x, y, z).
  // - Columns 5 to 8: Orientation (x, y, z, w).
  // - Column 9: Coordinate frame id.
  Eigen::MatrixXd ToMatrix() const;
  // Serializes the entire coordinate system hierarchy into a string, every 9
  // elements separated by ',' represent one frame node (one row of the
  // serialized matrix).
  std::string ToString() const;

  Eigen::Vector3d ConvertFromSpecifiedParent(
      const Eigen::Vector3d& coordinate_in_specified,
      const FrameNode& specified_frame) const;
  Eigen::Vector3d ConvertToSpecifiedParent(
      const Eigen::Vector3d& coordinate_in_current,
      const FrameNode& specified_frame) const;

  Eigen::Vector3d ConvertFromBase(
      const Eigen::Vector3d& coordinate_in_base) const;
  Eigen::Vector3d ConvertToBase(
      const Eigen::Vector3d& coordinate_in_current) const;

  inline std::optional<CoordinateSystemLevel> MaybeCurrentLevel() const;
  inline std::optional<CoordinateSystem> MaybeCurrentCoordinateSystem() const;

  std::optional<Eigen::Vector3d> MaybeCurrentPosition() const;
  std::optional<Eigen::Quaterniond> MaybeCurrentOrientation() const;

  // Checks if the CoordinateSystemContext is complete.
  //
  // A complete coordinate system means that the hierarchy eventually traces
  // back to a parent coordinate system with a `GLOBAL` level. This ensures that
  // the current context is well-defined and has a global reference.
  bool IsComplete() const;

  inline bool IsCurrentCartesian() const;
  inline bool IsCurrentMetric() const;

  bool operator==(const CoordinateSystemContext& other) const;
  bool operator!=(const CoordinateSystemContext& other) const;

  // Serves as the ultimate reference frame in the hierarchy.
  std::unique_ptr<FrameNode> current_frame = nullptr;

 private:
  // A helper function to traverse through the coordinate system hierarchy
  // (frame chain). This function allows the user to specify a custom operation
  // to perform on each frame, the traverse stops and returns false when the
  // operation returns false. Note: ensure the operation to be safe.
  template <typename Functor>
  bool Traverse(Functor operation);
  template <typename Functor>
  bool Traverse(Functor operation) const;
};

// Class responsible for converting coordinates between two coordinate system
// context.
class CoordinateSystemConverter {
 public:
  using FrameNode = CoordinateSystemContext::FrameNode;

  CoordinateSystemConverter(const CoordinateSystemContext& source,
                            const CoordinateSystemContext& target);

  // Conversion is possible if the source and target coordinate system context
  // share a common parent frame or are both based on the global frame.
  inline bool IsConvertable() const;

  // Converts a coordinate from the source coordinate system to the target
  // coordinate system.
  Eigen::Vector3d Convert(const Eigen::Vector3d& coordinate) const;

  // Calls `Convert` function.
  inline Eigen::Vector3d operator()(const Eigen::Vector3d& coordinate) const;

  // Returns a converter reversing the conversion direction: target -> source
  inline CoordinateSystemConverter Reverse() const;

 private:
  // Attempts to find the first common parent coordinate system between the
  // source and target.
  std::optional<FrameNode> MaybeFirstCommonParent() const;

  const CoordinateSystemContext& source_;
  const CoordinateSystemContext& target_;
};

std::ostream& operator<<(std::ostream& stream,
                         const CoordinateSystemContext& context);

struct PosePrior {
 public:
  using CoordinateSystem = CoordinateSystemContext::CoordinateSystem;
  Eigen::Vector3d position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d position_covariance =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  CoordinateSystem coordinate_system = CoordinateSystem::UNDEFINED;

  PosePrior() = default;
  explicit PosePrior(const Eigen::Vector3d& position) : position(position) {}
  PosePrior(const Eigen::Vector3d& position, const CoordinateSystem system)
      : position(position), coordinate_system(system) {}
  PosePrior(const Eigen::Vector3d& position, const Eigen::Matrix3d& covariance)
      : position(position), position_covariance(covariance) {}
  PosePrior(const Eigen::Vector3d& position,
            const Eigen::Matrix3d& covariance,
            const CoordinateSystem system)
      : position(position),
        position_covariance(covariance),
        coordinate_system(system) {}

  inline bool IsValid() const { return position.allFinite(); }
  inline bool IsCovarianceValid() const {
    return position_covariance.allFinite();
  }

  inline bool operator==(const PosePrior& other) const;
  inline bool operator!=(const PosePrior& other) const;
};

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior);

bool CoordinateSystemContext::FrameNode::IsCartesian() const {
  return coordinate_system != CoordinateSystem::WGS84 &&
         coordinate_system != CoordinateSystem::GRS80 &&
         coordinate_system != CoordinateSystem::UNDEFINED;
}

bool CoordinateSystemContext::FrameNode::IsMetric() const {
  return coordinate_system != CoordinateSystem::MODEL &&
         coordinate_system != CoordinateSystem::UNDEFINED;
}

bool CoordinateSystemContext::FrameNode::operator==(
    const FrameNode& other) const {
  return coordinate_system == other.coordinate_system &&
         frame_id == other.frame_id;
}

bool CoordinateSystemContext::FrameNode::operator!=(
    const FrameNode& other) const {
  return !(*this == other);
}

std::optional<CoordinateSystemContext::CoordinateSystem>
CoordinateSystemContext::MaybeCurrentCoordinateSystem() const {
  return current_frame == nullptr
             ? std::nullopt
             : std::make_optional(current_frame->coordinate_system);
}

template <typename Functor>
bool CoordinateSystemContext::Traverse(Functor operation) {
  FrameNode* current = current_frame.get();

  while (current != nullptr) {
    FrameNode* parent = current->parent_frame.get();
    if (!operation(current)) {
      return false;
    }
    current = parent;
  }
  return true;
}

template <typename Functor>
bool CoordinateSystemContext::Traverse(Functor operation) const {
  FrameNode* current = current_frame.get();

  while (current != nullptr) {
    FrameNode* parent = current->parent_frame.get();
    if (!operation(current)) {
      return false;
    }
    current = parent;
  }
  return true;
}

std::optional<CoordinateSystemContext::CoordinateSystemLevel>
CoordinateSystemContext::MaybeCurrentLevel() const {
  return current_frame == nullptr ? std::nullopt
                                  : std::make_optional(current_frame->Level());
}

bool CoordinateSystemContext::IsCurrentCartesian() const {
  return current_frame->IsCartesian();
}

bool CoordinateSystemContext::IsCurrentMetric() const {
  return current_frame->IsMetric();
}

bool CoordinateSystemConverter::IsConvertable() const {
  return MaybeFirstCommonParent().has_value() ||
         (source_.IsComplete() && target_.IsComplete());
}

Eigen::Vector3d CoordinateSystemConverter::operator()(
    const Eigen::Vector3d& coordinate) const {
  return Convert(coordinate);
}

CoordinateSystemConverter CoordinateSystemConverter::Reverse() const {
  return CoordinateSystemConverter(target_, source_);
}

bool PosePrior::operator==(const PosePrior& other) const {
  return coordinate_system == other.coordinate_system &&
         position == other.position &&
         position_covariance == other.position_covariance;
}

bool PosePrior::operator!=(const PosePrior& other) const {
  return !(*this == other);
}

}  // namespace colmap
