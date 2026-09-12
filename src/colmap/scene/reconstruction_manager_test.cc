// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/reconstruction_manager.h"

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ReconstructionManager, Empty) {
  ReconstructionManager reconstruction_manager;
  EXPECT_EQ(reconstruction_manager.Size(), 0);
}

TEST(ReconstructionManager, AddGet) {
  ReconstructionManager reconstruction_manager;
  EXPECT_EQ(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    const size_t idx = reconstruction_manager.Add();
    EXPECT_EQ(reconstruction_manager.Size(), i + 1);
    EXPECT_EQ(idx, i);
    EXPECT_EQ(reconstruction_manager.Get(idx)->NumCameras(), 0);
    EXPECT_EQ(reconstruction_manager.Get(idx)->NumImages(), 0);
    EXPECT_EQ(reconstruction_manager.Get(idx)->NumPoints3D(), 0);
  }
}

TEST(ReconstructionManager, Delete) {
  ReconstructionManager reconstruction_manager;
  EXPECT_EQ(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  EXPECT_EQ(reconstruction_manager.Size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Delete(0);
    EXPECT_EQ(reconstruction_manager.Size(), 9 - i);
  }
}

TEST(ReconstructionManager, Clear) {
  ReconstructionManager reconstruction_manager;
  EXPECT_EQ(reconstruction_manager.Size(), 0);
  for (size_t i = 0; i < 10; ++i) {
    reconstruction_manager.Add();
  }

  EXPECT_EQ(reconstruction_manager.Size(), 10);
  reconstruction_manager.Clear();
  EXPECT_EQ(reconstruction_manager.Size(), 0);
}

}  // namespace
}  // namespace colmap
