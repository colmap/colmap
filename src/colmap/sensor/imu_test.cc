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

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

const Eigen::Vector3d kZero = Eigen::Vector3d::Zero();

ImuMeasurement CreateMeasurement(timestamp_t t) {
  return ImuMeasurement(t, kZero, kZero);
}

TEST(ImuCalibration, Default) {
  const ImuCalibration calib;
  EXPECT_EQ(calib.gyro_rectification, Eigen::Matrix3d::Identity());
  EXPECT_EQ(calib.accel_rectification, Eigen::Matrix3d::Identity());
  EXPECT_GT(calib.imu_rate, 0.0);
  EXPECT_GT(calib.gravity_magnitude, 0.0);
}

TEST(ImuCalibration, Print) {
  ImuCalibration calib;
  calib.gyro_noise_density = 1;
  calib.accel_noise_density = 2;
  calib.gravity_magnitude = 3;
  std::ostringstream stream;
  stream << calib;
  EXPECT_EQ(stream.str(),
            "ImuCalibration(gyro_noise_density=1, accel_noise_density=2, "
            "gravity_magnitude=3)");
}

TEST(ImuMeasurement, Default) {
  const ImuMeasurement measurement;
  EXPECT_EQ(measurement.timestamp, kInvalidTimestamp);
  EXPECT_EQ(measurement.gyro, Eigen::Vector3d::Zero());
  EXPECT_EQ(measurement.accel, Eigen::Vector3d::Zero());
}

TEST(ImuMeasurement, Construct) {
  const ImuMeasurement measurement(
      100, Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  EXPECT_EQ(measurement.timestamp, 100);
  EXPECT_EQ(measurement.gyro, Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(measurement.accel, Eigen::Vector3d(4, 5, 6));
}

TEST(ImuMeasurement, Print) {
  const ImuMeasurement measurement(
      100, Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6));
  std::ostringstream stream;
  stream << measurement;
  EXPECT_EQ(stream.str(), "ImuMeasurement(t=100, gyro=[1 2 3], accel=[4 5 6])");
}

TEST(ImuMeasurements, Default) {
  ImuMeasurements ms;
  EXPECT_TRUE(ms.Empty());
  EXPECT_EQ(ms.Size(), 0);
}

TEST(ImuMeasurements, InsertSingle) {
  // Insert out of order — should be sorted.
  ImuMeasurements ms;
  ms.Insert(CreateMeasurement(30));
  ms.Insert(CreateMeasurement(10));
  ms.Insert(CreateMeasurement(20));
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 20);
  EXPECT_EQ(ms[2].timestamp, 30);
  EXPECT_EQ(ms.front().timestamp, 10);
  EXPECT_EQ(ms.back().timestamp, 30);

  // Insert in order — append fast path.
  ms.Clear();
  ms.Insert(CreateMeasurement(10));
  ms.Insert(CreateMeasurement(20));
  ms.Insert(CreateMeasurement(30));
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[2].timestamp, 30);
}

TEST(ImuMeasurements, InsertVector) {
  ImuMeasurements ms;
  ms.Insert(std::vector<ImuMeasurement>{
      CreateMeasurement(30), CreateMeasurement(10), CreateMeasurement(20)});
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 20);
  EXPECT_EQ(ms[2].timestamp, 30);
}

TEST(ImuMeasurements, InsertMeasurements) {
  ImuMeasurements ms;
  ms.Insert(std::vector<ImuMeasurement>{
      CreateMeasurement(10), CreateMeasurement(20), CreateMeasurement(30)});
  ImuMeasurements other;
  other.Insert(CreateMeasurement(15));
  other.Insert(CreateMeasurement(25));
  ms.Insert(other);
  ASSERT_EQ(ms.Size(), 5);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 15);
  EXPECT_EQ(ms[2].timestamp, 20);
  EXPECT_EQ(ms[3].timestamp, 25);
  EXPECT_EQ(ms[4].timestamp, 30);
}

TEST(ImuMeasurements, InsertSorted) {
  ImuMeasurements ms;
  ms.Insert(CreateMeasurement(10));
  // Append fast path.
  ms.InsertSorted({CreateMeasurement(20), CreateMeasurement(30)});
  ASSERT_EQ(ms.Size(), 3);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[2].timestamp, 30);
  // Merge path.
  ms.InsertSorted({CreateMeasurement(15), CreateMeasurement(25)});
  ASSERT_EQ(ms.Size(), 5);
  EXPECT_EQ(ms[1].timestamp, 15);
  EXPECT_EQ(ms[3].timestamp, 25);
}

TEST(ImuMeasurements, Remove) {
  ImuMeasurements ms;
  ms.Insert(std::vector<ImuMeasurement>{
      CreateMeasurement(10), CreateMeasurement(20), CreateMeasurement(30)});
  ms.Remove(CreateMeasurement(20));
  ASSERT_EQ(ms.Size(), 2);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 30);
  EXPECT_ANY_THROW(ms.Remove(CreateMeasurement(99)));
}

TEST(ImuMeasurements, Copy) {
  ImuMeasurements ms;
  ms.Insert(std::vector<ImuMeasurement>{CreateMeasurement(10),
                                        CreateMeasurement(20)});
  const ImuMeasurements copy(ms);
  ASSERT_EQ(copy.Size(), 2);
  EXPECT_EQ(copy[0].timestamp, 10);
  EXPECT_EQ(copy[1].timestamp, 20);
}

TEST(ImuMeasurements, ConstructFromVector) {
  const ImuMeasurements ms(std::vector<ImuMeasurement>{CreateMeasurement(30),
                                                       CreateMeasurement(10)});
  ASSERT_EQ(ms.Size(), 2);
  EXPECT_EQ(ms[0].timestamp, 10);
  EXPECT_EQ(ms[1].timestamp, 30);
}

TEST(ImuMeasurements, Iterator) {
  const ImuMeasurements ms(std::vector<ImuMeasurement>{CreateMeasurement(10),
                                                       CreateMeasurement(30)});
  std::vector<timestamp_t> timestamps;
  for (const auto& m : ms) {
    timestamps.push_back(m.timestamp);
  }
  EXPECT_EQ(timestamps, std::vector<timestamp_t>({10, 30}));
}

TEST(ImuMeasurements, DuplicateThrows) {
  // Duplicate via single insert.
  ImuMeasurements ms1;
  ms1.Insert(CreateMeasurement(10));
  EXPECT_ANY_THROW(ms1.Insert(CreateMeasurement(10)));

  // Duplicate via append fast path.
  ImuMeasurements ms2;
  ms2.Insert(CreateMeasurement(10));
  ms2.Insert(CreateMeasurement(20));
  EXPECT_ANY_THROW(ms2.Insert(CreateMeasurement(20)));

  // Duplicate via unsorted vector constructor.
  EXPECT_ANY_THROW(ImuMeasurements(std::vector<ImuMeasurement>{
      CreateMeasurement(10), CreateMeasurement(10)}));

  // Duplicate via InsertSorted (overlap with existing).
  ImuMeasurements ms3;
  ms3.Insert(CreateMeasurement(10));
  EXPECT_ANY_THROW(
      ms3.InsertSorted({CreateMeasurement(10), CreateMeasurement(20)}));

  // Duplicate within sorted input.
  ImuMeasurements ms4;
  ms4.Insert(CreateMeasurement(5));
  EXPECT_ANY_THROW(
      ms4.InsertSorted({CreateMeasurement(10), CreateMeasurement(10)}));
}

TEST(ImuMeasurements, ExtractMeasurementsInRange) {
  // Build measurements at [100, 200, 300, 400, 500].
  ImuMeasurements ms;
  for (timestamp_t t = 100; t <= 500; t += 100) {
    ms.Insert(CreateMeasurement(t));
  }

  // Nominal interior with exact matches: [200, 400] -> [200, 300, 400].
  // upper_bound(200)->300, it1-1->200; lower_bound(400)->400, it2+1->500.
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(200, 400);
    ASSERT_EQ(r.Size(), 3);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
    EXPECT_EQ(r[2].timestamp, 400);
  }

  // Full range: [100, 500] -> all five.
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(100, 500);
    ASSERT_EQ(r.Size(), 5);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[4].timestamp, 500);
  }

  // t1 between samples: [150, 400] -> includes 100 (sample before t1).
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(150, 400);
    ASSERT_EQ(r.Size(), 4);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[3].timestamp, 400);
  }

  // t2 between samples: [200, 350] -> includes 400 (sample after t2).
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(200, 350);
    ASSERT_EQ(r.Size(), 3);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
    EXPECT_EQ(r[2].timestamp, 400);
  }

  // Both between same adjacent pair: [210, 290] -> [200, 300].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(210, 290);
    ASSERT_EQ(r.Size(), 2);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
  }

  // Minimal adjacent exact matches: [200, 300] -> [200, 300].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(200, 300);
    ASSERT_EQ(r.Size(), 2);
    EXPECT_EQ(r[0].timestamp, 200);
    EXPECT_EQ(r[1].timestamp, 300);
  }

  // Both between samples spanning multiple: [150, 350] -> [100, 200, 300, 400].
  {
    const ImuMeasurements r = ms.ExtractMeasurementsInRange(150, 350);
    ASSERT_EQ(r.Size(), 4);
    EXPECT_EQ(r[0].timestamp, 100);
    EXPECT_EQ(r[3].timestamp, 400);
  }

  // Error cases.
  ImuMeasurements empty;
  EXPECT_ANY_THROW(empty.ExtractMeasurementsInRange(100, 200));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsInRange(300, 200));
  EXPECT_ANY_THROW(ms.ExtractMeasurementsInRange(200, 200));
  // Ranges that extend beyond the available samples cannot bracket the edge
  // and return empty rather than throwing.
  EXPECT_TRUE(ms.ExtractMeasurementsInRange(50, 300).Empty());
  EXPECT_TRUE(ms.ExtractMeasurementsInRange(200, 600).Empty());
}

}  // namespace
}  // namespace colmap
