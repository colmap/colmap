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
#include "colmap/math/math.h"
#include "colmap/util/eigen_matchers.h"

#include <random>

#include <gtest/gtest.h>

namespace colmap {
namespace {
using FramePose = CoordinateSystemContext::FramePose;
using FrameNode = CoordinateSystemContext::FrameNode;
using CoordinateSystem = CoordinateSystemContext::CoordinateSystem;

static std::random_device rd;
static std::mt19937 gen(rd());

int RandomInt(int min = 1, int max = 100) {
  std::uniform_int_distribution<int> dis(min, max);
  return dis(gen);
}

Eigen::Vector3d RandomPosition(double min, double max) {
  std::uniform_real_distribution<> dis(min, max);
  return Eigen::Vector3d(dis(gen), dis(gen), dis(gen));
}

FramePose RandomFramePose() {
  std::uniform_real_distribution<> pos_dis(-89.0, 89.0);
  std::uniform_real_distribution<> angle_dis(0.0, 2 * M_PI);

  Eigen::Vector3d random_position(pos_dis(gen), pos_dis(gen), pos_dis(gen));
  Eigen::Vector3d axis(pos_dis(gen), pos_dis(gen), pos_dis(gen));
  axis.normalize();

  double angle = angle_dis(gen);

  Eigen::Quaterniond random_orientation(Eigen::AngleAxisd(angle, axis));

  return FramePose{random_orientation, random_position};
}

TEST(PosePrior, FrameNodeCoordinateConversionSpherical) {
  // 1. Test for ENU conversion
  FramePose pose_a_in_b = RandomFramePose();

  CoordinateSystem coordinate_system =
      RandomInt() % 2 ? CoordinateSystem::WGS84 : CoordinateSystem::GRS80;

  FrameNode frame_b{coordinate_system, FramePose::Identity(), nullptr, 0};
  FrameNode frame_a{CoordinateSystem::ENU, pose_a_in_b, &frame_b, 0};

  Eigen::Vector3d coordinate_in_a = RandomPosition(-100, 100);
  Eigen::Vector3d coordinate_in_b_random = RandomPosition(-89, 89);

  // Convert coordinate from frame_a -> frame_b -> frame_a
  Eigen::Vector3d recovered_in_a =
      frame_a.ConvertFromParent(frame_a.ConvertToParent(coordinate_in_a));
  EXPECT_THAT(coordinate_in_a, EigenMatrixNear(recovered_in_a, 1e-8));

  // Convert coordinate from frame_b -> frame_a -> frame_b
  Eigen::Vector3d recovered_in_b = frame_a.ConvertToParent(
      frame_a.ConvertFromParent(coordinate_in_b_random));
  EXPECT_THAT(coordinate_in_b_random, EigenMatrixNear(recovered_in_b, 1e-8));

  // 2. Test for UTM conversion
  Eigen::Vector3d coordinate_in_c = Eigen::Vector3d(
      1.91125018424899e5 + 5.0e5, 5.335909515367108e6, 561.1851);
  Eigen::Vector3d coordinate_in_b_fixed =
      Eigen::Vector3d(48 + 8.0 / 60 + 51.70361 / 3600,
                      11 + 34.0 / 60 + 10.51777 / 3600,
                      561.1851);

  FramePose pose_c_in_b{Eigen::Quaterniond::Identity(), {0, 9, 0}};  // Zone 32

  FrameNode frame_c{CoordinateSystem::UTM, pose_c_in_b, &frame_b, 0};

  // Convert UTM coordinate in frame_c -> frame_b -> frame_c
  Eigen::Vector3d recovered_in_c =
      frame_c.ConvertFromParent(frame_c.ConvertToParent(coordinate_in_c));
  EXPECT_THAT(coordinate_in_c, EigenMatrixNear(recovered_in_c, 1e-8));

  // Convert coordinate from frame_b -> frame_c -> frame_b
  recovered_in_b =
      frame_c.ConvertToParent(frame_c.ConvertFromParent(coordinate_in_b_fixed));
  EXPECT_THAT(coordinate_in_b_fixed, EigenMatrixNear(recovered_in_b, 1e-8));
}

TEST(PosePrior, FrameNodeCoordinateConversionCartesian) {
  FramePose pose_a_in_b = RandomFramePose();

  CoordinateSystem model_or_object =
      RandomInt() % 2 ? CoordinateSystem::MODEL : CoordinateSystem::OBJECT;
  FrameNode frame_b{model_or_object, FramePose::Identity(), nullptr, 0};
  FrameNode frame_a{model_or_object, pose_a_in_b, &frame_b, 0};

  Eigen::Vector3d coordinate_in_a = RandomPosition(-100, 100);
  Eigen::Vector3d coordinate_in_b = RandomPosition(-100, 100);

  Eigen::Vector3d recovered_in_a =
      frame_a.ConvertFromParent(frame_a.ConvertToParent(coordinate_in_a));
  EXPECT_THAT(coordinate_in_a, EigenMatrixNear(recovered_in_a, 1e-8));

  Eigen::Vector3d recovered_in_b =
      frame_a.ConvertFromParent(frame_a.ConvertToParent(coordinate_in_b));
  EXPECT_THAT(coordinate_in_b, EigenMatrixNear(recovered_in_b, 1e-8));
}

TEST(PosePrior, CoordinateSystemContextStructure) {
  CoordinateSystemContext complete_context;
  CoordinateSystemContext incomplete_context;

  CoordinateSystem wgs84_or_grs80 =
      RandomInt() % 2 ? CoordinateSystem::WGS84 : CoordinateSystem::GRS80;
  CoordinateSystem enu_or_utm =
      RandomInt() % 2 ? CoordinateSystem::ENU : CoordinateSystem::UTM;

  FramePose pose_a_in_b = RandomFramePose();
  FramePose pose_b_in_c = RandomFramePose();
  FramePose pose_c_in_d = RandomFramePose();

  FramePose pose_e_in_f = RandomFramePose();

  FrameNode frame_d{wgs84_or_grs80, FramePose::Identity()};
  FrameNode frame_c{
      enu_or_utm,
      pose_c_in_d,
  };
  FrameNode frame_b{CoordinateSystem::OBJECT, pose_b_in_c};
  FrameNode frame_a{CoordinateSystem::OBJECT, pose_a_in_b};

  FrameNode frame_f{CoordinateSystem::MODEL, FramePose::Identity()};
  FrameNode frame_e{CoordinateSystem::MODEL, pose_e_in_f};

  // WGS84/GRS80 -> UTM/ENU -> OBJECT_1 -> OBJECT_2
  complete_context.Append(frame_c, FramePose::Identity());
  complete_context.Append(frame_d, pose_c_in_d);
  complete_context.Prepend(frame_b, pose_b_in_c);
  complete_context.Prepend(frame_a, pose_a_in_b);

  // MODEL_1 -> MODEL_2
  incomplete_context.Append(frame_f, FramePose::Identity());
  incomplete_context.Prepend(frame_e, pose_e_in_f);

  EXPECT_TRUE(complete_context.IsComplete());
  EXPECT_FALSE(incomplete_context.IsComplete());

  EXPECT_EQ(complete_context.MaybeBaseFrame().value(), frame_d);
  EXPECT_EQ(complete_context.MaybePenultimateFrame().value(), frame_c);
  EXPECT_EQ(incomplete_context.MaybeBaseFrame().value(), frame_f);
  EXPECT_EQ(incomplete_context.MaybePenultimateFrame().value(), frame_e);
}

TEST(PosePrior, CoordinateSystemConverterGlobal) {
  Eigen::Vector3d utm_in_wgs84(0, 9, 0);

  Eigen::Vector3d enu_in_wgs84(48 + 8.0 / 60 + 51.70361 / 3600,
                               11 + 34.0 / 60 + 10.51777 / 3600,
                               561.1851);

  // frame_chain_a: object -> utm -> WGS84
  FramePose object_in_utm = RandomFramePose();
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_a;
  frame_chain_a.emplace_back(FrameNode{CoordinateSystem::OBJECT},
                             FramePose::Identity());
  frame_chain_a.emplace_back(FrameNode{CoordinateSystem::UTM}, object_in_utm);
  frame_chain_a.emplace_back(
      FrameNode{CoordinateSystem::WGS84},
      FramePose{Eigen::Quaterniond::Identity(), utm_in_wgs84});
  CoordinateSystemContext context_a(frame_chain_a);
  context_a.current_frame->frame_id = 0;

  // frame_chain_b: object -> enu -> WGS84
  FramePose object_in_enu = RandomFramePose();
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_b;
  frame_chain_b.emplace_back(FrameNode{CoordinateSystem::OBJECT},
                             FramePose::Identity());
  frame_chain_b.emplace_back(FrameNode{CoordinateSystem::ENU}, object_in_enu);
  frame_chain_b.emplace_back(
      FrameNode{CoordinateSystem::WGS84},
      FramePose{Eigen::Quaterniond::Identity(), enu_in_wgs84});
  CoordinateSystemContext context_b(frame_chain_b);
  context_b.current_frame->frame_id = 1;

  // We test a same point converted from context a to context b and back.
  Eigen::Vector3d point_in_wgs84(48 + 8.0 / 60 + 52.40575 / 3600,
                                 11 + 34.0 / 60 + 11.77179 / 3600,
                                 561.1509);

  GPSTransform gps_transform(GPSTransform::WGS84);
  Eigen::Vector3d point_in_enu = gps_transform.EllToENU(
      point_in_wgs84, enu_in_wgs84[0], enu_in_wgs84[1], enu_in_wgs84[2]);

  // 1.91150201163177e5 + 5.0e5, 5.335932057413140e6, 561.1509
  Eigen::Vector3d point_in_utm = gps_transform.EllToUTM(point_in_wgs84).first;

  Eigen::Vector3d point_in_a = object_in_utm.ToRigid3d() * point_in_utm;
  Eigen::Vector3d point_in_b = object_in_enu.ToRigid3d() * point_in_enu;

  CoordinateSystemConverter converter_a_to_b(context_a, context_b);

  Eigen::Vector3d point_in_a_to_b = converter_a_to_b.Convert(point_in_a);
  EXPECT_THAT(point_in_a_to_b, EigenMatrixNear(point_in_b, 1e-4));  // 0.1mm

  Eigen::Vector3d point_in_b_to_a = converter_a_to_b.Reverse()(point_in_b);
  EXPECT_THAT(point_in_b_to_a, EigenMatrixNear(point_in_a, 1e-4));
}

TEST(PosePrior, CoordinateSystemConverterCommonParent) {
  Eigen::Vector3d utm_in_wgs84(0, 9, 0);

  Eigen::Vector3d enu_in_wgs84(48 + 8.0 / 60 + 51.70361 / 3600,
                               11 + 34.0 / 60 + 10.51777 / 3600,
                               561.1851);

  // We test a same point converted from context a to context b and back.
  Eigen::Vector3d point_in_wgs84(48 + 8.0 / 60 + 52.40575 / 3600,
                                 11 + 34.0 / 60 + 11.77179 / 3600,
                                 561.1509);

  GPSTransform gps_transform(GPSTransform::WGS84);
  Eigen::Vector3d point_in_enu = gps_transform.EllToENU(
      point_in_wgs84, enu_in_wgs84[0], enu_in_wgs84[1], enu_in_wgs84[2]);

  // 1.91150201163177e5 + 5.0e5, 5.335932057413140e6, 561.1509
  Eigen::Vector3d point_in_utm = gps_transform.EllToUTM(point_in_wgs84).first;

  FramePose object_0_in_object_utm = RandomFramePose();
  FramePose object_0_in_object_enu =
      FramePose::Identity();  // Set object0 same as the enu
  FramePose object_1_in_object_2 = RandomFramePose();
  FramePose object_2_in_object_0 = RandomFramePose();
  FramePose object_3_in_object_4 = RandomFramePose();
  FramePose object_4_in_object_0 = RandomFramePose();

  // frame_chain_a: object_1 <- object_2 <- object_0 <- enu
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_a;
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::OBJECT, 1),
                             FramePose::Identity());
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::OBJECT, 2),
                             object_1_in_object_2);
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::OBJECT, 0),
                             object_2_in_object_0);
  frame_chain_a.emplace_back(FrameNode{CoordinateSystem::ENU},
                             object_0_in_object_enu);
  CoordinateSystemContext context_a(frame_chain_a);

  // frame_chain_b: object_3 <- object_4 <- object_0 <- utm
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_b;
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::OBJECT, 3),
                             FramePose::Identity());
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::OBJECT, 4),
                             object_3_in_object_4);
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::OBJECT, 0),
                             object_4_in_object_0);
  frame_chain_b.emplace_back(FrameNode{CoordinateSystem::UTM},
                             object_0_in_object_utm);
  CoordinateSystemContext context_b(frame_chain_b);

  CoordinateSystemConverter converter_a_to_b(context_a, context_b);

  Eigen::Vector3d point_in_object_0 =
      object_0_in_object_enu.ToRigid3d() * point_in_enu;
  Eigen::Vector3d point_in_object_1 = object_1_in_object_2.ToRigid3d() *
                                      object_2_in_object_0.ToRigid3d() *
                                      point_in_object_0;
  Eigen::Vector3d point_in_object_3 = object_3_in_object_4.ToRigid3d() *
                                      object_4_in_object_0.ToRigid3d() *
                                      point_in_object_0;

  Eigen::Vector3d point_in_a_to_b = converter_a_to_b.Convert(point_in_object_1);
  EXPECT_THAT(point_in_a_to_b, EigenMatrixNear(point_in_object_3, 1e-8));

  Eigen::Vector3d point_in_b_to_a =
      converter_a_to_b.Reverse()(point_in_object_3);
  EXPECT_THAT(point_in_b_to_a, EigenMatrixNear(point_in_object_1, 1e-8));
}

TEST(PosePrior, CoordinateSystemConverterCommonParentVirtual) {
  FramePose model_4_in_model_3 = RandomFramePose();
  FramePose model_3_in_model_2 = RandomFramePose();
  FramePose model_2_in_model_0 = RandomFramePose();
  FramePose model_6_in_model_5 = RandomFramePose();
  FramePose model_5_in_model_2 = RandomFramePose();
  FramePose model_2_in_model_1 = RandomFramePose();

  // frame chain a: model_0 -> model_2 -> model_3 -> model_4
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_a;
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::MODEL, 4),
                             FramePose::Identity());
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::MODEL, 3),
                             model_4_in_model_3);
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::MODEL, 2),
                             model_3_in_model_2);
  frame_chain_a.emplace_back(FrameNode(CoordinateSystem::MODEL, 0),
                             model_2_in_model_0);
  CoordinateSystemContext context_a(frame_chain_a);

  // frame chain b: model_1 -> model_2 -> model_5 -> model_6
  std::vector<std::pair<FrameNode, FramePose>> frame_chain_b;
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::MODEL, 6),
                             FramePose::Identity());
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::MODEL, 5),
                             model_6_in_model_5);
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::MODEL, 2),
                             model_5_in_model_2);
  frame_chain_b.emplace_back(FrameNode(CoordinateSystem::MODEL, 1),
                             model_2_in_model_1);
  CoordinateSystemContext context_b(frame_chain_b);

  CoordinateSystemConverter converter_a_to_b(context_a, context_b);

  Eigen::Vector3d point_in_model_2 = RandomPosition(-100, 100);
  Eigen::Vector3d point_in_model_4 = model_4_in_model_3.ToRigid3d() *
                                     model_3_in_model_2.ToRigid3d() *
                                     point_in_model_2;
  Eigen::Vector3d point_in_model_6 = model_6_in_model_5.ToRigid3d() *
                                     model_5_in_model_2.ToRigid3d() *
                                     point_in_model_2;

  Eigen::Vector3d point_in_a_to_b = converter_a_to_b.Convert(point_in_model_4);
  EXPECT_THAT(point_in_a_to_b, EigenMatrixNear(point_in_model_6, 1e-8));

  Eigen::Vector3d point_in_b_to_a =
      converter_a_to_b.Reverse()(point_in_model_6);
  EXPECT_THAT(point_in_b_to_a, EigenMatrixNear(point_in_model_4, 1e-8));
}

TEST(PosePrior, Equals) {
  PosePrior prior;
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::ECEF;
  PosePrior other = prior;
  EXPECT_EQ(prior, other);
  prior.position.x() = 1;
  EXPECT_NE(prior, other);
  other.position.x() = 1;
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::ECEF;
  std::ostringstream stream;
  stream << prior;
  EXPECT_EQ(stream.str(),
            "PosePrior(position=[0, 0, 0], position_covariance=[1, 0, 0, 0, 1, "
            "0, 0, 0, 1], coordinate_system=ECEF)");
}

}  // namespace
}  // namespace colmap
