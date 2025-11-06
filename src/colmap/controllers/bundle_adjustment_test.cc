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

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/math/random.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BundleAdjustmentController, EmptyReconstruction) {
  SetPRNGSeed(1);

  auto reconstruction = std::make_shared<Reconstruction>();

  BundleAdjustmentController controller(OptionManager(), reconstruction);
  EXPECT_NO_THROW(controller.Run());

  EXPECT_EQ(reconstruction->NumRegImages(), 0);
  EXPECT_EQ(reconstruction->NumPoints3D(), 0);
}

TEST(BundleAdjustmentController, Reconstruction) {
  SetPRNGSeed(1);

  auto reconstruction = std::make_shared<Reconstruction>();

  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 2;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 30;
  SynthesizeDataset(synthetic_options, reconstruction.get());
  SyntheticNoiseOptions noise_options;
  noise_options.rig_from_world_translation_stddev = 0.01;
  noise_options.rig_from_world_rotation_stddev = 0.5;
  noise_options.point3D_stddev = 0.01;
  noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(noise_options, reconstruction.get());

  const double initial_error = reconstruction->ComputeMeanReprojectionError();

  BundleAdjustmentController controller(OptionManager(), reconstruction);
  controller.Run();

  const double final_error = reconstruction->ComputeMeanReprojectionError();

  // Bundle adjustment should reduce the mean reprojection error
  EXPECT_LT(final_error, initial_error);

  // The reconstruction should still have the same structure
  EXPECT_EQ(reconstruction->NumRegImages(), 6);
  EXPECT_EQ(reconstruction->NumPoints3D(), 30);
}

}  // namespace
}  // namespace colmap
