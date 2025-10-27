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

#include "colmap/scene/reconstruction_matchers.h"

#include "colmap/scene/synthetic.h"

namespace colmap {
namespace {

struct TestClass {
  virtual ~TestClass() = default;
  virtual void TestMethod(const Reconstruction&) const {}
};

struct MockTestClass : public TestClass {
  MOCK_METHOD(void, TestMethod, (const Reconstruction&), (const, override));
};

TEST(Reconstruction, Near) {
  Reconstruction reconstruction1;
  Reconstruction reconstruction2;
  EXPECT_THAT(reconstruction1,
              ReconstructionNear(reconstruction2,
                                 /*max_rotation_error_deg=*/0,
                                 /*max_proj_center_error=*/0,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0,
                                 /*align=*/false));
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2,
                                              /*max_rotation_error_deg=*/0,
                                              /*max_proj_center_error=*/0,
                                              /*max_scale_error=*/std::nullopt,
                                              /*num_obs_tolerance=*/0,
                                              /*align=*/true)));

  SyntheticDatasetOptions synthetic_dataset_options;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction1);

  reconstruction2 = reconstruction1;
  EXPECT_THAT(reconstruction1, ReconstructionNear(reconstruction2));

  reconstruction2 = reconstruction1;
  reconstruction2.Frame(1).RigFromWorld().translation.x() += 0.1;
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  reconstruction2 = reconstruction1;
  reconstruction2.Frame(1).RigFromWorld().rotation *=
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()));
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  reconstruction2 = reconstruction1;
  reconstruction2.DeleteObservation(1, 0);
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(ReconstructionNear(reconstruction1))).Times(1);
  EXPECT_CALL(mock, TestMethod(ReconstructionNear(reconstruction2))).Times(1);
  mock.TestMethod(reconstruction1);
  mock.TestMethod(reconstruction2);
}

}  // namespace
}  // namespace colmap
