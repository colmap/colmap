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

#include "colmap/sfm/incremental_triangulator.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void DeleteAllPoints3D(Reconstruction& reconstruction) {
  std::vector<point3D_t> point3D_ids_to_delete;
  for (const auto point3D_id : reconstruction.Point3DIds()) {
    point3D_ids_to_delete.push_back(point3D_id);
  }
  for (const auto point3D_id : point3D_ids_to_delete) {
    reconstruction.DeletePoint3D(point3D_id);
  }
  EXPECT_EQ(reconstruction.NumPoints3D(), 0);
}

void DeleteOneObservationFromEachTrack(Reconstruction& reconstruction) {
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    ASSERT_GT(point3D.track.Length(), 0);
    reconstruction.DeleteObservation(point3D.track.Element(0).image_id,
                                     point3D.track.Element(0).point2D_idx);
  }
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            reconstruction.NumPoints3D() * (reconstruction.NumRegImages() - 1));
}

void SplitPoint3D(Reconstruction& reconstruction, point3D_t point3D_id) {
  auto& point3D = reconstruction.Point3D(point3D_id);
  ASSERT_GE(point3D.track.Length(), 4);
  Track split_track;
  for (size_t i = point3D.track.Length() / 2; i < point3D.track.Length(); ++i) {
    const auto& track_el = point3D.track.Element(i);
    split_track.AddElement(track_el);
    reconstruction.Image(track_el.image_id)
        .ResetPoint3DForPoint2D(track_el.point2D_idx);
  }
  point3D.track.Elements().resize(point3D.track.Length() / 2);
  const point3D_t new_point3D_id =
      reconstruction.AddPoint3D(point3D.xyz, split_track);
  for (const auto& track_el : split_track.Elements()) {
    reconstruction.Image(track_el.image_id)
        .SetPoint3DForPoint2D(track_el.point2D_idx, new_point3D_id);
  }
}

TEST(IncrementalTriangulator, Print) {
  Reconstruction reconstruction;
  IncrementalTriangulator triangulator(std::make_shared<CorrespondenceGraph>(),
                                       reconstruction);
  std::ostringstream stream;
  stream << triangulator;
  EXPECT_EQ(
      stream.str(),
      "IncrementalTriangulator(reconstruction=Reconstruction(num_rigs=0, "
      "num_cameras=0, num_frames=0, num_reg_frames=0, num_images=0, "
      "num_points3D=0), correspondence_graph=CorrespondenceGraph(num_images=0, "
      "num_image_pairs=0))");
}

TEST(IncrementalTriangulator, ModifiedPoints3D) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 50;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  EXPECT_THAT(triangulator.GetModifiedPoints3D(), testing::IsEmpty());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  triangulator.AddModifiedPoint3D(point3D_id1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().count(point3D_id1), 1);

  triangulator.AddModifiedPoint3D(point3D_id2);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 2);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().count(point3D_id2), 1);

  triangulator.AddModifiedPoint3D(point3D_id1);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 2);

  triangulator.ClearModifiedPoints3D();
  EXPECT_TRUE(triangulator.GetModifiedPoints3D().empty());
}

TEST(IncrementalTriangulator, ModifiedPoints3DRemovesNonExistent) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 10;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  auto point3D_ids = reconstruction.Point3DIds();
  ASSERT_GE(point3D_ids.size(), 1);
  const point3D_t point3D_id = *point3D_ids.begin();

  triangulator.AddModifiedPoint3D(point3D_id);
  EXPECT_EQ(triangulator.GetModifiedPoints3D().size(), 1);
  reconstruction.DeletePoint3D(point3D_id);
  EXPECT_TRUE(triangulator.GetModifiedPoints3D().empty());
}

TEST(IncrementalTriangulator, TriangulateImage) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteAllPoints3D(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);
  size_t total_tris = 0;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    total_tris += triangulator.TriangulateImage(
        IncrementalTriangulator::Options(), image_id);
  }

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(total_tris,
            synthetic_options.num_points3D * reconstruction.NumRegImages());
}

TEST(IncrementalTriangulator, CompleteImage) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  triangulator.CompleteImage(IncrementalTriangulator::Options(),
                             reconstruction.RegImageIds().at(0));
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(
      reconstruction.ComputeNumObservations(),
      synthetic_options.num_points3D * (reconstruction.NumRegImages() - 1));
}

TEST(IncrementalTriangulator, CompleteTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_completions =
      triangulator.CompleteTracks(IncrementalTriangulator::Options(),
                                  {reconstruction.Points3D().begin()->first});
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(num_completions, 1);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            synthetic_options.num_points3D * reconstruction.NumRegImages() -
                synthetic_options.num_points3D + 1);
}

TEST(IncrementalTriangulator, CompleteAllTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  DeleteOneObservationFromEachTrack(reconstruction);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_completions =
      triangulator.CompleteAllTracks(IncrementalTriangulator::Options());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(num_completions, synthetic_options.num_points3D);
  EXPECT_EQ(reconstruction.ComputeNumObservations(),
            synthetic_options.num_points3D * reconstruction.NumRegImages());
}

TEST(IncrementalTriangulator, MergeTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 5;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  SplitPoint3D(reconstruction, point3D_id1);
  SplitPoint3D(reconstruction, point3D_id2);

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 2);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_merged = triangulator.MergeTracks(
      IncrementalTriangulator::Options(), {point3D_id1});
  EXPECT_EQ(num_merged, reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 1);
}

TEST(IncrementalTriangulator, MergeAllTracks) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 10;
  synthetic_options.num_points3D = 5;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  auto points3D_it = reconstruction.Points3D().begin();
  const point3D_t point3D_id1 = (points3D_it++)->first;
  const point3D_t point3D_id2 = (points3D_it++)->first;

  SplitPoint3D(reconstruction, point3D_id1);
  SplitPoint3D(reconstruction, point3D_id2);

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D + 2);

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_merged =
      triangulator.MergeAllTracks(IncrementalTriangulator::Options());
  EXPECT_EQ(num_merged, 2 * reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
}

TEST(IncrementalTriangulator, Retriangulate) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  DeleteAllPoints3D(reconstruction);

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);

  const size_t num_tris =
      triangulator.Retriangulate(IncrementalTriangulator::Options());
  EXPECT_EQ(num_tris,
            synthetic_options.num_points3D * reconstruction.NumRegImages());
  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
}

}  // namespace
}  // namespace colmap
