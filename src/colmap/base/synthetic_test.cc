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

#include "colmap/base/synthetic.h"

#include "colmap/controllers/incremental_mapper.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(SynthesizeDataset, Nominal) {
  Database database("database.db");
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  SynthesizeDataset(options, &reconstruction, &database);

  for (const auto& camera : reconstruction.Cameras()) {
    EXPECT_EQ(camera.second.ParamsToString(),
              database.ReadCamera(camera.first).ParamsToString());
  }

  for (const auto& image : reconstruction.Images()) {
    EXPECT_EQ(image.second.Name(), database.ReadImage(image.first).Name());
    EXPECT_EQ(image.second.NumPoints2D(),
              database.ReadKeypoints(image.first).size());
  }

  EXPECT_GT(database.NumInlierMatches(), 0);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  IncrementalMapperController mapper(
      std::make_shared<IncrementalMapperOptions>(),
      /*image_path=*/"",
      "database.db",
      reconstruction_manager);

  mapper.Start();
  mapper.Wait();
}

}  // namespace colmap
