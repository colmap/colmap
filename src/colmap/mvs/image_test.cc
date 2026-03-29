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

#include "colmap/mvs/image.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Image, DefaultConstructor) {
  Image image;
  EXPECT_EQ(image.GetWidth(), 0);
  EXPECT_EQ(image.GetHeight(), 0);
  EXPECT_TRUE(image.GetPath().empty());
}

TEST(Image, ParameterizedConstructor) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);

  EXPECT_EQ(image.GetWidth(), 100);
  EXPECT_EQ(image.GetHeight(), 100);
  EXPECT_EQ(image.GetPath(), "test.jpg");

  const float* image_K = image.GetK();
  const float* image_R = image.GetR();
  const float* image_T = image.GetT();

  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(image_K[i], K[i]);
    EXPECT_EQ(image_R[i], R[i]);
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(image_T[i], T[i]);
  }
}

TEST(Image, Rescale) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Rescale(0.5);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 50);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled
  EXPECT_FLOAT_EQ(image_K[4], 50.0f);  // fy scaled
  EXPECT_FLOAT_EQ(image_K[5], 25.0f);  // cy scaled
}

TEST(Image, RescaleNonUniform) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Rescale(0.5, 0.25);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 25);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled by factor_x
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled by factor_x
  EXPECT_FLOAT_EQ(image_K[4], 25.0f);  // fy scaled by factor_y
  EXPECT_FLOAT_EQ(image_K[5], 12.5f);  // cy scaled by factor_y
}

TEST(Image, Downsize) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Downsize(50, 50);

  EXPECT_EQ(image.GetWidth(), 50);
  EXPECT_EQ(image.GetHeight(), 50);

  const float* image_K = image.GetK();
  EXPECT_FLOAT_EQ(image_K[0], 50.0f);  // fx scaled
  EXPECT_FLOAT_EQ(image_K[2], 25.0f);  // cx scaled
  EXPECT_FLOAT_EQ(image_K[4], 50.0f);  // fy scaled
  EXPECT_FLOAT_EQ(image_K[5], 25.0f);  // cy scaled
}

TEST(Image, DownsizeNoChange) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  image.Downsize(200, 200);

  EXPECT_EQ(image.GetWidth(), 100);
  EXPECT_EQ(image.GetHeight(), 100);
}

TEST(Image, GetViewingDirection) {
  const float K[9] = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};

  Image image("test.jpg", 100, 100, K, R, T);
  const float* viewing_dir = image.GetViewingDirection();

  EXPECT_EQ(viewing_dir[0], R[6]);
  EXPECT_EQ(viewing_dir[1], R[7]);
  EXPECT_EQ(viewing_dir[2], R[8]);
}

TEST(ComputeProjectionCenter, Identity) {
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {1, 2, 3};
  float C[3];
  ComputeProjectionCenter(R, T, C);

  EXPECT_FLOAT_EQ(C[0], -1.0f);
  EXPECT_FLOAT_EQ(C[1], -2.0f);
  EXPECT_FLOAT_EQ(C[2], -3.0f);
}

TEST(ComposeProjectionMatrix, Identity) {
  const float K[9] = {2, 0, 0, 0, 2, 0, 0, 0, 1};
  const float R[9] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
  const float T[3] = {1, 2, 3};
  float P[12];
  ComposeProjectionMatrix(K, R, T, P);

  const float expected[12] = {0, 2, 0, 2, 2, 0, 0, 4, 0, 0, 1, 3};
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(P[i], expected[i]) << i;
  }
}

TEST(RotatePose, Identity) {
  const float RR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float T[3] = {1, 2, 3};

  RotatePose(RR, R, T);

  const float expected_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float expected_T[3] = {1, 2, 3};

  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(R[i], expected_R[i]);
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(T[i], expected_T[i]);
  }
}

TEST(RotatePose, Rotation90DegreesZ) {
  // 90 degree rotation around Z axis
  const float RR[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float T[3] = {1, 0, 0};

  RotatePose(RR, R, T);

  const float expected_R[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  const float expected_T[3] = {0, 1, 0};

  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ(R[i], expected_R[i]) << i;
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(T[i], expected_T[i]) << i;
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
