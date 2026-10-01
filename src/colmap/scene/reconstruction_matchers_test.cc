// SPDX-License-Identifier: BSD-3-Clause

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

TEST(Reconstruction, Eq) {
  Reconstruction reconstruction1;
  Reconstruction reconstruction2;
  EXPECT_THAT(reconstruction1, ReconstructionEq(reconstruction2));

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction1);

  reconstruction2 = reconstruction1;
  EXPECT_THAT(reconstruction1, ReconstructionEq(reconstruction2));

  reconstruction2 = reconstruction1;
  reconstruction2.Frame(1).RigFromWorld().translation().x() += 0.1;
  EXPECT_THAT(reconstruction1, testing::Not(ReconstructionEq(reconstruction2)));

  reconstruction2 = reconstruction1;
  reconstruction2.DeleteObservation(1, 0);
  EXPECT_THAT(reconstruction1, testing::Not(ReconstructionEq(reconstruction2)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(ReconstructionEq(reconstruction1))).Times(1);
  EXPECT_CALL(mock, TestMethod(ReconstructionEq(reconstruction2))).Times(2);
  mock.TestMethod(reconstruction1);
  mock.TestMethod(reconstruction2);
  mock.TestMethod(reconstruction2);
}

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
  synthetic_dataset_options.num_points2D_without_point3D = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction1);

  reconstruction2 = reconstruction1;
  EXPECT_THAT(reconstruction1, ReconstructionNear(reconstruction2));

  reconstruction2 = reconstruction1;
  reconstruction2.Frame(1).RigFromWorld().translation().x() += 0.1;
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  reconstruction2 = reconstruction1;
  reconstruction2.Frame(1).RigFromWorld().rotation() *=
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()));
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  reconstruction2 = reconstruction1;
  reconstruction2.DeleteObservation(1, 0);
  EXPECT_THAT(reconstruction1,
              testing::Not(ReconstructionNear(reconstruction2)));

  testing::StrictMock<MockTestClass> mock;
  EXPECT_CALL(mock, TestMethod(ReconstructionNear(reconstruction1))).Times(1);
  EXPECT_CALL(mock, TestMethod(ReconstructionNear(reconstruction2))).Times(2);
  mock.TestMethod(reconstruction1);
  mock.TestMethod(reconstruction2);
  mock.TestMethod(reconstruction2);
}

}  // namespace
}  // namespace colmap
