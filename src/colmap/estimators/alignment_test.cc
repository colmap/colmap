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

#include "colmap/estimators/alignment.h"

#include "colmap/geometry/sim3.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {

Sim3d TestSim3d() {
  return Sim3d(RandomUniformReal<double>(0.5, 2),
               Eigen::Quaterniond::UnitRandom(),
               Eigen::Vector3d::Random());
}

void ExpectEqualSim3d(const Sim3d& gt_tgt_from_src, const Sim3d& tgt_from_src) {
  EXPECT_NEAR(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
            1e-6);
}

Reconstruction GenerateReconstructionForAlignment() {
  // const std::string database_path = CreateTestDir() + "/database.db";
  // Database database(database_path);
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = 2;
  synthetic_dataset_options.num_images = 20;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  return reconstruction;
}

TEST(Alignment, AlignReconstructionsViaReprojections) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  CHECK(AlignReconstructionsViaReprojections(src_reconstruction,
                                             tgt_reconstruction,
                                             /*min_inlier_observations=*/0.9,
                                             /*max_reproj_error=*/2,
                                             &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionsViaProjCenters) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  CHECK(AlignReconstructionsViaProjCenters(src_reconstruction,
                                           tgt_reconstruction,
                                           /*max_proj_center_error=*/0.1,
                                           &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

TEST(Alignment, AlignReconstructionsViaPoints) {
  Reconstruction src_reconstruction = GenerateReconstructionForAlignment();
  Reconstruction tgt_reconstruction = src_reconstruction;

  Sim3d gt_tgt_from_src = TestSim3d();
  tgt_reconstruction.Transform(gt_tgt_from_src);

  Sim3d tgt_from_src;
  CHECK(AlignReconstructionsViaPoints(src_reconstruction,
                                      tgt_reconstruction,
                                      /*min_common_observations=*/3,
                                      /*max_error=*/0.01,
                                      /*min_inlier_ratio=*/0.9,
                                      &tgt_from_src));
  ExpectEqualSim3d(gt_tgt_from_src, tgt_from_src);
}

}  // namespace colmap
