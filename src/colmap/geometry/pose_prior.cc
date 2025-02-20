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

#include "colmap/geometry/pose_prior.h"

#include "colmap/geometry/gps.h"

#include <iomanip>
#include <stack>

namespace colmap {
namespace {
int MeridianToZone(double meridian) {
  return static_cast<int>(std::floor((meridian + 180) / 6)) + 1;
}

GPSTransform GetGPSTransform(CoordinateSystemContext::CoordinateSystem system) {
  GPSTransform gps_transform;
  switch (system) {
    case CoordinateSystemContext::CoordinateSystem::WGS84:
      gps_transform = GPSTransform(GPSTransform::WGS84);
      break;
    case CoordinateSystemContext::CoordinateSystem::GRS80:
      gps_transform = GPSTransform(GPSTransform::GRS80);
      break;
    default:
      LOG(ERROR) << "Unsupported coordinate system.";
  }
  return gps_transform;
}
}  // namespace

Rigid3d CoordinateSystemContext::FramePose::ToRigid3d() const {
  Eigen::Quaterniond rotation = orientation.inverse();
  Eigen::Vector3d translation = rotation * -position;
  return Rigid3d(rotation, translation);
}

CoordinateSystemContext::FramePose
CoordinateSystemContext::FramePose::Identity() {
  return CoordinateSystemContext::FramePose{Eigen::Quaterniond::Identity(),
                                            Eigen::Vector3d::Zero()};
}

CoordinateSystemContext::FrameNode::FrameNode(CoordinateSystem system,
                                              const FramePose& pose,
                                              FrameNode* parent,
                                              std::uint32_t id)
    : coordinate_system(system), frame_pose(pose), frame_id(id) {
  if (parent != nullptr) {
    parent_frame = std::make_unique<FrameNode>(*parent);
  } else {
    parent_frame = nullptr;
  }
}

CoordinateSystemContext::FrameNode::FrameNode(CoordinateSystem system,
                                              std::uint32_t id)
    : coordinate_system(system), frame_id(id) {}

CoordinateSystemContext::FrameNode::FrameNode(
    const CoordinateSystemContext::FrameNode& other)
    : coordinate_system(other.coordinate_system),
      frame_pose(other.frame_pose),
      frame_id(other.frame_id) {
  if (other.parent_frame != nullptr) {
    parent_frame = std::make_unique<FrameNode>(*(other.parent_frame));
  } else {
    parent_frame = nullptr;
  }
}

CoordinateSystemContext::CoordinateSystemContext(
    const CoordinateSystemContext& other) {
  std::stack<FrameNode> frame_stack;
  other.Traverse([&](FrameNode* frame) {
    frame_stack.push(FrameNode(*frame));
    return true;
  });
  while (!frame_stack.empty()) {
    FrameNode pretended = frame_stack.top();
    this->Prepend(pretended, pretended.frame_pose);
    frame_stack.pop();
  }
}

CoordinateSystemContext::CoordinateSystemContext(
    const std::vector<std::pair<FrameNode, FramePose> >& frame_chain,
    bool reverse) {
  if (frame_chain.empty()) {
    return;
  }

  if (!reverse) {
    for (auto iter = frame_chain.begin(); iter != frame_chain.end(); ++iter) {
      Append(iter->first, iter->second);
    }
  } else {
    for (auto iter = frame_chain.rbegin(); iter != frame_chain.rend(); ++iter) {
      Append(iter->first, iter->second);
    }
  }
}

CoordinateSystemContext::~CoordinateSystemContext() {
  // Avoid default destructor destroy list nodes recursively, which would
  // cause stack overflow for sufficiently large lists.
  while (current_frame != nullptr) {
    std::unique_ptr<FrameNode> next_frame =
        std::move(current_frame->parent_frame);
    current_frame = std::move(next_frame);
  }
}

void CoordinateSystemContext::Append(const FrameNode& frame,
                                     const FramePose& current_base_in_appened) {
  auto new_frame = std::make_unique<FrameNode>(frame);
  if (current_frame == nullptr) {
    current_frame = std::move(new_frame);
    current_frame->frame_pose = current_base_in_appened;
  } else {
    FrameNode* temp = current_frame.get();
    while (temp->parent_frame != nullptr) {
      temp = temp->parent_frame.get();
    }
    // Parent's level need to be lower or equal.
    THROW_CHECK_LE(new_frame->Level(), temp->Level());

    temp->parent_frame = std::move(new_frame);

    temp->frame_pose = current_base_in_appened;
  }
}

void CoordinateSystemContext::Prepend(const FrameNode& frame,
                                      const FramePose& prepended_in_current) {
  auto new_frame = std::make_unique<FrameNode>(frame);
  if (current_frame == nullptr) {
    current_frame = std::move(new_frame);
    current_frame->frame_pose = prepended_in_current;
  } else {
    // Parent's level need to be lower or equal.
    THROW_CHECK_LE(current_frame->Level(), new_frame->Level());

    std::unique_ptr<FrameNode> second_frame = std::move(current_frame);
    current_frame = std::move(new_frame);

    current_frame->frame_pose = prepended_in_current;
    current_frame->parent_frame = std::move(second_frame);
  }
}

std::optional<CoordinateSystemContext::FrameNode>
CoordinateSystemContext::MaybeBaseFrame() const {
  FrameNode* current = current_frame.get();
  while (current != nullptr && current->parent_frame != nullptr) {
    current = current->parent_frame.get();
  }
  return current == nullptr ? std::nullopt
                            : std::make_optional<FrameNode>(*current);
}

std::optional<CoordinateSystemContext::FrameNode>
CoordinateSystemContext::MaybePenultimateFrame() const {
  if (current_frame == nullptr || current_frame->parent_frame == nullptr) {
    return std::nullopt;
  }

  FrameNode* current = current_frame.get();
  while (current->parent_frame != nullptr &&
         current->parent_frame->parent_frame != nullptr) {
    current = current->parent_frame.get();
  }
  return std::make_optional<FrameNode>(*current);
}

CoordinateSystemContext::CoordinateSystemLevel
CoordinateSystemContext::FrameNode::Level() const {
  CoordinateSystemLevel level;
  switch (coordinate_system) {
    case CoordinateSystem::WGS84:
    case CoordinateSystem::ECEF:
    case CoordinateSystem::GRS80:
      level = CoordinateSystemLevel::GLOBAL;
      break;
    case CoordinateSystem::ENU:
    case CoordinateSystem::UTM:
      level = CoordinateSystemLevel::REGIONAL;
      break;
    case CoordinateSystem::OBJECT:
      level = CoordinateSystemLevel::LOCAL;
      break;
    case CoordinateSystem::MODEL:
      level = CoordinateSystemLevel::VIRTUAL;
      break;
    default:
      level = CoordinateSystemLevel::UNDEFINED;
  }

  return level;
}

Eigen::Vector3d CoordinateSystemContext::FrameNode::ConvertFromParent(
    const Eigen::Vector3d& coordinate_in_parent) {
  if (parent_frame == nullptr) {
    return coordinate_in_parent;
  }
  // We ensure that spherical is lower level and don't have parent.
  if (parent_frame->IsCartesian()) {
    return frame_pose.ToRigid3d() * coordinate_in_parent;
  } else {
    GPSTransform gps_transform =
        GetGPSTransform(parent_frame->coordinate_system);

    switch (coordinate_system) {
      case CoordinateSystem::ECEF:
        return gps_transform.EllToXYZ(coordinate_in_parent);
      case CoordinateSystem::ENU: {
        const Eigen::Vector3d& origin = frame_pose.position;
        return gps_transform.EllToENU(
            coordinate_in_parent, origin[0], origin[1], origin[2]);
      }
      case CoordinateSystem::UTM: {
        const Eigen::Vector3d& origin = frame_pose.position;
        auto [converted_coordinate, zone] =
            gps_transform.EllToUTM(coordinate_in_parent);

        // TODO: Support for UTM zone conversion.
        THROW_CHECK_EQ(std::abs(zone), MeridianToZone(origin[1]));

        return converted_coordinate;
      }
      default:
        LOG(ERROR) << "Unsupported coordinate system.";
        return Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
    }
  }
}

Eigen::Vector3d CoordinateSystemContext::FrameNode::ConvertToParent(
    const Eigen::Vector3d& coordinate_in_current) {
  if (parent_frame == nullptr) {
    return coordinate_in_current;
  }
  // We ensure that spherical is lower level and don't have parent.
  if (parent_frame->IsCartesian()) {
    return Inverse(frame_pose.ToRigid3d()) * coordinate_in_current;
  } else {
    GPSTransform gps_transform =
        GetGPSTransform(parent_frame->coordinate_system);

    switch (coordinate_system) {
      case CoordinateSystem::ECEF:
        return gps_transform.XYZToEll(coordinate_in_current);
      case CoordinateSystem::ENU: {
        const Eigen::Vector3d& origin = frame_pose.position;
        return gps_transform.ENUToEll(
            coordinate_in_current, origin[0], origin[1], origin[2]);
      }
      case CoordinateSystem::UTM: {
        const Eigen::Vector3d& origin = frame_pose.position;
        return gps_transform.UTMToEll(coordinate_in_current,
                                      MeridianToZone(origin[1]));
      }
      default:
        LOG(ERROR) << "Unsupported coordinate system.";
        return Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
    }
  }
}

bool CoordinateSystemContext::IsComplete() const {
  const FrameNode* current = current_frame.get();

  while (current != nullptr) {
    if (current->Level() == CoordinateSystemLevel::GLOBAL) {
      return true;
    }

    current = current->parent_frame.get();
  }

  // If no GLOBAL-level frame is found, the system is incomplete.
  return false;
}

Eigen::MatrixXd CoordinateSystemContext::ToMatrix() const {
  // Count the number of nodes to determine the number of rows
  int count = 0;
  Traverse([&](const FrameNode*) {
    ++count;
    return true;
  });

  // Create an Eigen matrix with 8 columns: coordinate_system(1),
  // position (3), orientation (4), frame_id(1)
  Eigen::MatrixXd matrix(count, 8);

  int row = 0;
  Traverse([&](const FrameNode* node) {
    matrix(row, 0) = static_cast<int>(node->coordinate_system);
    matrix.row(row).segment<3>(1) = node->frame_pose.position;
    matrix.row(row).segment<4>(4) = node->frame_pose.orientation.coeffs();
    matrix(row, 5) = static_cast<int>(node->frame_id);
    ++row;
    return true;
  });

  return matrix;
}

std::string CoordinateSystemContext::ToString() const {
  std::ostringstream oss;

  Traverse([&](const FrameNode* node) {
    oss << static_cast<int>(node->coordinate_system) << ",";

    int p = 4;
    bool is_base_frame = node->parent_frame == nullptr;
    if (!is_base_frame) {
      p = node->parent_frame->IsCartesian() ? 4 : 8;
    }

    oss << std::fixed << std::setprecision(p) << node->frame_pose.position.x()
        << "," << node->frame_pose.position.y() << ","
        << node->frame_pose.position.z() << ",";

    oss << std::fixed << std::setprecision(4)
        << node->frame_pose.orientation.x() << ","
        << node->frame_pose.orientation.y() << ","
        << node->frame_pose.orientation.z() << ","
        << node->frame_pose.orientation.w() << ",";

    oss << static_cast<int>(node->frame_id);

    if (!is_base_frame) {
      oss << ",";
    }
    return true;
  });

  return oss.str();
}
Eigen::Vector3d CoordinateSystemContext::ConvertFromSpecifiedParent(
    const Eigen::Vector3d& coordinate_in_specified,
    const FrameNode& specified_frame) const {
  // We need to travese inversely.
  std::stack<FrameNode> frame_stack;
  bool is_specified_in_context = !Traverse([&](FrameNode* frame) {
    if (frame->parent_frame == nullptr) {
      return true;
    }
    if (*(frame->parent_frame) == specified_frame) {
      frame_stack.push(FrameNode(*frame));
      return false;
    } else {
      frame_stack.push(FrameNode(*frame));
      return true;
    }
  });

  THROW_CHECK(is_specified_in_context);

  Eigen::Vector3d coordinate_in_current = coordinate_in_specified;

  // Actually the base is the penultimate's parent.
  while (!frame_stack.empty()) {
    FrameNode frame = frame_stack.top();
    coordinate_in_current = frame.ConvertFromParent(coordinate_in_current);

    frame_stack.pop();
  }
  return coordinate_in_current;
}

Eigen::Vector3d CoordinateSystemContext::ConvertToSpecifiedParent(
    const Eigen::Vector3d& coordinate_in_current,
    const FrameNode& specified_frame) const {
  Eigen::Vector3d coordinate_in_specified = coordinate_in_current;
  bool is_specified_in_context = !Traverse([&](FrameNode* frame) {
    if (*frame == specified_frame) {
      return false;
    }
    if (frame->parent_frame == nullptr) {
      return true;
    }
    coordinate_in_specified = frame->ConvertToParent(coordinate_in_specified);

    return true;
  });

  THROW_CHECK(is_specified_in_context);

  return coordinate_in_specified;
}

Eigen::Vector3d CoordinateSystemContext::ConvertFromBase(
    const Eigen::Vector3d& coordinate_in_base) const {
  // We need to travese inversely.
  std::stack<FrameNode> frame_stack;
  Traverse([&](FrameNode* frame) {
    if (frame->parent_frame == nullptr) {
      return true;
    }
    frame_stack.push(FrameNode(*frame));
    return true;
  });

  Eigen::Vector3d coordinate_in_current = coordinate_in_base;

  // Actually the base is the penultimate's parent.
  while (!frame_stack.empty()) {
    FrameNode frame = frame_stack.top();
    coordinate_in_current = frame.ConvertFromParent(coordinate_in_current);

    frame_stack.pop();
  }
  return coordinate_in_current;
}

Eigen::Vector3d CoordinateSystemContext::ConvertToBase(
    const Eigen::Vector3d& coordinate_in_current) const {
  Eigen::Vector3d coordinate_in_base = coordinate_in_current;
  Traverse([&](FrameNode* frame) {
    if (frame->parent_frame == nullptr) {
      return true;
    }
    coordinate_in_base = frame->ConvertToParent(coordinate_in_base);

    return true;
  });

  return coordinate_in_base;
}

bool CoordinateSystemContext::operator==(
    const CoordinateSystemContext& other) const {
  FrameNode* this_frame = current_frame.get();
  FrameNode* other_frame = other.current_frame.get();
  while (this_frame != nullptr && other_frame != nullptr) {
    if (*this_frame != *other_frame) {
      return false;
    }
    this_frame = this_frame->parent_frame.get();
    other_frame = other_frame->parent_frame.get();
  }
  return true;
}

bool CoordinateSystemContext::operator!=(
    const CoordinateSystemContext& other) const {
  return !(*this == other);
}

using CoordinateSystemLevel = CoordinateSystemContext::CoordinateSystemLevel;
using CoordinateSystem = CoordinateSystemContext::CoordinateSystem;
using FrameNode = CoordinateSystemContext::FrameNode;

CoordinateSystemConverter::CoordinateSystemConverter(
    const CoordinateSystemContext& source,
    const CoordinateSystemContext& target)
    : source_(source), target_(target) {}

Eigen::Vector3d CoordinateSystemConverter::Convert(
    const Eigen::Vector3d& coordinate) const {
  THROW_CHECK(IsConvertable())
      << "Unsupported conversion between "
      << CoordinateSystemContext::CoordinateSystemToString(
             source_.MaybeCurrentCoordinateSystem().value_or(
                 CoordinateSystem::UNDEFINED))
      << " and "
      << CoordinateSystemContext::CoordinateSystemToString(
             target_.MaybeCurrentCoordinateSystem().value_or(
                 CoordinateSystem::UNDEFINED));

  Eigen::Vector3d converted_coordinate = coordinate;

  if (source_ == target_) {
    return converted_coordinate;
  }

  auto common_parent = MaybeFirstCommonParent();

  if (!common_parent.has_value()) {
    // If the coordinate systems are convertible and don't share common parent,
    // then both are based on a global frame

    converted_coordinate = source_.ConvertToSpecifiedParent(
        coordinate, source_.MaybeBaseFrame().value());

    converted_coordinate = target_.ConvertFromSpecifiedParent(
        converted_coordinate, target_.MaybeBaseFrame().value());

  } else {
    // If a common parent exists, convert using the common parent frame.
    converted_coordinate =
        source_.ConvertToSpecifiedParent(coordinate, common_parent.value());

    converted_coordinate = target_.ConvertFromSpecifiedParent(
        converted_coordinate, common_parent.value());
  }

  return converted_coordinate;
}

std::optional<FrameNode> CoordinateSystemConverter::MaybeFirstCommonParent()
    const {
  FrameNode* source_current = source_.current_frame.get();
  FrameNode* target_current = target_.current_frame.get();
  while (source_current != nullptr) {
    while (target_current != nullptr) {
      if (*source_current == *target_current) {
        return std::make_optional<FrameNode>(*source_current);
      } else {
        target_current = target_current->parent_frame.get();
      }
    }
    target_current = target_.current_frame.get();
    source_current = source_current->parent_frame.get();
  }
  return std::nullopt;
}

std::optional<Eigen::Vector3d> CoordinateSystemContext::MaybeCurrentPosition()
    const {
  if (current_frame->frame_pose.position != FramePose::Identity().position) {
    return current_frame->frame_pose.position;
  } else {
    return std::nullopt;
  }
}

std::optional<Eigen::Quaterniond>
CoordinateSystemContext::MaybeCurrentOrientation() const {
  if (current_frame->frame_pose.orientation !=
      FramePose::Identity().orientation) {
    return current_frame->frame_pose.orientation;
  } else {
    return std::nullopt;
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const CoordinateSystemContext& context) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "CoordinateSystemContext(";

  FrameNode* current = context.current_frame.get();
  while (current != nullptr) {
    FrameNode* parent = current->parent_frame.get();
    stream << "coordinate_system="
           << CoordinateSystemContext::CoordinateSystemToString(
                  current->coordinate_system)
           << ", position=[" << current->frame_pose.position.format(kVecFmt)
           << "], orientation=["
           << current->frame_pose.orientation.coeffs().format(kVecFmt)
           << "], id=" << current->frame_id;
    if (parent != nullptr) {
      stream << '\n';
    }
    current = parent;
  }

  stream << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const PosePrior& prior) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "PosePrior(position=[" << prior.position.format(kVecFmt)
         << "], position_covariance=["
         << prior.position_covariance.format(kVecFmt) << "], coordinate_system="
         << CoordinateSystemContext::CoordinateSystemToString(
                prior.coordinate_system)
         << ")";
  return stream;
}

}  // namespace colmap
