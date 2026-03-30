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

#include "colmap/sensor/imu.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

const Eigen::Vector3d kZero = Eigen::Vector3d::Zero();

ImuMeasurement M(timestamp_t t) { return ImuMeasurement(t, kZero, kZero); }

TEST(ImuMeasurements, Nominal) {
  // Default state.
  ImuMeasurements ms;
  EXPECT_TRUE(ms.Empty());
  EXPECT_EQ(ms.Size(), 0);

  // Insert out of order — should be sorted.
  ms.Insert(M(30));
  ms.Insert(M(10));
  ms.Insert(M(20));
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 20);
  EXPECT_EQ(ms[2].timestamp, 30);
  EXPECT_EQ(ms.front().timestamp, 10);
  EXPECT_EQ(ms.back().timestamp, 30);

  // Insert in order — fast path.
  ms.Clear();
  ms.Insert(M(10));
  ms.Insert(M(20));
  ms.Insert(M(30));
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[2].timestamp, 30);

  // Insert from unsorted vector.
  ImuMeasurements ms2;
  ms2.Insert(std::vector<ImuMeasurement>{M(30), M(10), M(20)});
  ASSERT_EQ(ms2.Size(), 3);
  EXPECT_EQ(ms2[0].timestamp, 10);
  EXPECT_EQ(ms2[2].timestamp, 30);

  // Insert from another ImuMeasurements (merge).
  ImuMeasurements ms3;
  ms3.Insert(M(15));
  ms3.Insert(M(25));
  ms2.Insert(ms3);
  ASSERT_EQ(ms2.Size(), 5);
  EXPECT_EQ(ms2[0].timestamp, 10);
  EXPECT_EQ(ms2[1].timestamp, 15);
  EXPECT_EQ(ms2[2].timestamp, 20);
  EXPECT_EQ(ms2[3].timestamp, 25);
  EXPECT_EQ(ms2[4].timestamp, 30);

  // InsertSorted — append fast path.
  ImuMeasurements ms4;
  ms4.Insert(M(10));
  ms4.InsertSorted({M(20), M(30)});
  ASSERT_EQ(ms4.Size(), 3);
  EXPECT_EQ(ms4[0].timestamp, 10);
  EXPECT_EQ(ms4[2].timestamp, 30);

  // InsertSorted — merge path.
  ms4.InsertSorted({M(15), M(25)});
  ASSERT_EQ(ms4.Size(), 5);
  EXPECT_EQ(ms4[1].timestamp, 15);
  EXPECT_EQ(ms4[3].timestamp, 25);

  // Remove.
  ms4.Remove(M(15));
  ASSERT_EQ(ms4.Size(), 4);
  EXPECT_EQ(ms4[1].timestamp, 20);
  EXPECT_ANY_THROW(ms4.Remove(M(99)));

  // Copy constructor.
  ImuMeasurements ms5(ms4);
  ASSERT_EQ(ms5.Size(), 4);
  EXPECT_EQ(ms5[0].timestamp, 10);
  EXPECT_EQ(ms5[3].timestamp, 30);

  // Construct from vector.
  ImuMeasurements ms6(std::vector<ImuMeasurement>{M(30), M(10)});
  ASSERT_EQ(ms6.Size(), 2);
  EXPECT_EQ(ms6[0].timestamp, 10);
  EXPECT_EQ(ms6[1].timestamp, 30);

  // Iterator.
  std::vector<timestamp_t> timestamps;
  for (const auto& m : ms6) {
    timestamps.push_back(m.timestamp);
  }
  EXPECT_EQ(timestamps, std::vector<timestamp_t>({10, 30}));

  // Duplicate timestamps throw.
  ImuMeasurements ms7;
  ms7.Insert(M(10));
  EXPECT_ANY_THROW(ms7.Insert(M(10)));
  // Duplicate via append fast path.
  ImuMeasurements ms8;
  ms8.Insert(M(10));
  ms8.Insert(M(20));
  EXPECT_ANY_THROW(ms8.Insert(M(20)));
  // Duplicate via unsorted vector.
  EXPECT_ANY_THROW(ImuMeasurements(std::vector<ImuMeasurement>{M(10), M(10)}));
  // Duplicate via InsertSorted.
  ImuMeasurements ms9;
  ms9.Insert(M(10));
  EXPECT_ANY_THROW(ms9.InsertSorted({M(10), M(20)}));
  // Duplicate within sorted input.
  ImuMeasurements ms10;
  ms10.Insert(M(5));
  EXPECT_ANY_THROW(ms10.InsertSorted({M(10), M(10)}));
}

TEST(ImuMeasurements, ExtractMeasurementsContainEdge) {
  // Build measurements at [100, 200, 300, 400, 500].
  ImuMeasurements ms;
  for (timestamp_t t = 100; t <= 500; t += 100) {
    ms.Insert(M(t));
  }

  // Nominal interior with exact matches: [200, 400] -> [200, 300, 400].
  // upper_bound(200)->300, it1-1->200; lower_bound(400)->400, it2+1->500.
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(200, 400);
    ASSERT_EQ(r.Size(), 3);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
    EXPECT_EQ(r[2].timestamp, 400);
  }

  // Full range: [100, 500] -> all five.
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(100, 500);
    ASSERT_EQ(r.Size(), 5);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[4].timestamp, 500);
  }

  // t1 between samples: [150, 400] -> includes 100 (sample before t1).
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(150, 400);
    ASSERT_EQ(r.Size(), 4);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[3].timestamp, 400);
  }

  // t2 between samples: [200, 350] -> includes 400 (sample after t2).
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(200, 350);
    ASSERT_EQ(r.Size(), 3);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
    EXPECT_EQ(r[2].timestamp, 400);
  }

  // Both between same adjacent pair: [210, 290] -> [200, 300].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(210, 290);
    ASSERT_EQ(r.Size(), 2);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
  }

  // Minimal adjacent exact matches: [200, 300] -> [200, 300].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(200, 300);
    ASSERT_EQ(r.Size(), 2);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
  }

  // Both between samples spanning multiple: [150, 350] -> [100, 200, 300, 400].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsContainEdge(150, 350);
    ASSERT_EQ(r.Size(), 4);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[3].timestamp, 400);
  }

  // Error cases.
  ImuMeasurements empty;
  EXPECT_ANY_THROW(empty.ExtractMeasurementsContainEdge(100, 200));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsContainEdge(300, 200));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsContainEdge(200, 200));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsContainEdge(50, 300));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsContainEdge(200, 600));
}

}  // namespace
}  // namespace colmap
