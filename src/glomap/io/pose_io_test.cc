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

#include "glomap/io/pose_io.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/testing.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace glomap {
namespace {

TEST(PoseIO, RelativePosesRoundtrip) {
  const std::string test_dir = colmap::CreateTestDir();

  std::unordered_map<image_t, std::string> image_names = {
      {0, "image0.jpg"}, {1, "image1.jpg"}, {2, "image2.jpg"}};

  std::unordered_map<image_pair_t, Rigid3d> poses;
  poses[colmap::ImagePairToPairId(0, 1)] =
      Rigid3d(Eigen::Quaterniond(1, 0, 0, 0), Eigen::Vector3d(1.0, 2.0, 3.0));
  poses[colmap::ImagePairToPairId(0, 2)] =
      Rigid3d(Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5).normalized(),
              Eigen::Vector3d(-1.0, 0.0, 1.0));
  poses[colmap::ImagePairToPairId(1, 2)] =
      Rigid3d(Eigen::Quaterniond(0.7, 0.1, 0.2, 0.3).normalized(),
              Eigen::Vector3d(0.5, 0.5, 0.5));

  std::string file_path =
      (std::filesystem::path(test_dir) / "relative_poses.txt").string();
  WriteRelativePoses(file_path, image_names, poses);
  auto read_poses = ReadRelativePoses(file_path, image_names);

  ASSERT_EQ(read_poses.size(), poses.size());
  for (const auto& [pair_id, pose] : poses) {
    ASSERT_TRUE(read_poses.count(pair_id));
    EXPECT_THAT(read_poses.at(pair_id), colmap::Rigid3dNear(pose, 1e-6, 1e-6));
  }
}

TEST(PoseIO, ImagePairWeightsRoundtrip) {
  const std::string test_dir = colmap::CreateTestDir();

  std::unordered_map<image_t, std::string> image_names = {
      {0, "image0.jpg"}, {1, "image1.jpg"}, {2, "image2.jpg"}};

  std::unordered_map<image_pair_t, double> weights;
  weights[colmap::ImagePairToPairId(0, 1)] =
      colmap::RandomUniformReal(0.0, 1.0);
  weights[colmap::ImagePairToPairId(0, 2)] =
      colmap::RandomUniformReal(0.0, 1.0);
  weights[colmap::ImagePairToPairId(1, 2)] =
      colmap::RandomUniformReal(0.0, 1.0);

  std::string file_path =
      (std::filesystem::path(test_dir) / "weights.txt").string();
  WriteImagePairWeights(file_path, image_names, weights);
  auto read_weights = ReadImagePairWeights(file_path, image_names);

  ASSERT_EQ(read_weights.size(), weights.size());
  for (const auto& [pair_id, weight] : weights) {
    ASSERT_TRUE(read_weights.count(pair_id));
    EXPECT_NEAR(weight, read_weights.at(pair_id), 1e-6);
  }
}

TEST(PoseIO, GravityPriorsRoundtrip) {
  const std::string test_dir = colmap::CreateTestDir();

  std::unordered_map<image_t, std::string> image_names = {
      {0, "image0.jpg"}, {1, "image1.jpg"}, {2, "image2.jpg"}};

  std::vector<colmap::PosePrior> priors;
  {
    colmap::PosePrior prior;
    prior.pose_prior_id = 0;
    prior.gravity = Eigen::Vector3d(0.0, 1.0, 0.0);
    priors.push_back(prior);
  }
  {
    colmap::PosePrior prior;
    prior.pose_prior_id = 1;
    prior.gravity = Eigen::Vector3d(0.1, 0.9, 0.0).normalized();
    priors.push_back(prior);
  }
  {
    colmap::PosePrior prior;
    prior.pose_prior_id = 2;
    prior.gravity = Eigen::Vector3d(-0.1, 0.95, 0.05).normalized();
    priors.push_back(prior);
  }

  std::string file_path =
      (std::filesystem::path(test_dir) / "gravity.txt").string();
  WriteGravityPriors(file_path, image_names, priors);
  auto read_priors = ReadGravityPriors(file_path, image_names);

  ASSERT_EQ(read_priors.size(), priors.size());
  for (size_t i = 0; i < priors.size(); ++i) {
    bool found = false;
    for (const auto& read_prior : read_priors) {
      if (read_prior.pose_prior_id == priors[i].pose_prior_id) {
        EXPECT_THAT(read_prior.gravity,
                    colmap::EigenMatrixNear(priors[i].gravity, 1e-6));
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(PoseIO, RotationsRoundtrip) {
  const std::string test_dir = colmap::CreateTestDir();

  std::unordered_map<image_t, std::string> image_names = {
      {0, "image0.jpg"}, {1, "image1.jpg"}, {2, "image2.jpg"}};

  std::unordered_map<image_t, Eigen::Quaterniond> rotations;
  rotations[0] = Eigen::Quaterniond(1, 0, 0, 0);
  rotations[1] = Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5).normalized();
  rotations[2] = Eigen::Quaterniond(0.7, 0.1, 0.2, 0.3).normalized();

  std::string file_path =
      (std::filesystem::path(test_dir) / "rotations.txt").string();
  WriteRotations(file_path, image_names, rotations);
  auto read_rotations = ReadRotations(file_path, image_names);

  ASSERT_EQ(read_rotations.size(), rotations.size());
  for (const auto& [id, q] : rotations) {
    ASSERT_TRUE(read_rotations.count(id));
    EXPECT_THAT(read_rotations.at(id).coeffs(),
                colmap::EigenMatrixNear(q.coeffs(), 1e-6));
  }
}

TEST(PoseIO, ReadImageNames) {
  const std::string test_dir = colmap::CreateTestDir();

  std::string file_path =
      (std::filesystem::path(test_dir) / "pairs.txt").string();
  {
    std::ofstream file(file_path);
    file << "a.jpg b.jpg 1 0 0 0 1 2 3\n";
    file << "b.jpg c.jpg 1 0 0 0 0 0 1\n";
    file << "a.jpg c.jpg 1 0 0 0 0 1 0\n";
  }

  auto image_names = ReadImageNames(file_path);

  EXPECT_EQ(image_names.size(), 3);
  std::set<std::string> names;
  for (const auto& [id, name] : image_names) {
    names.insert(name);
  }
  EXPECT_TRUE(names.count("a.jpg"));
  EXPECT_TRUE(names.count("b.jpg"));
  EXPECT_TRUE(names.count("c.jpg"));
}

}  // namespace
}  // namespace glomap
