// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/incremental_triangulator.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sfm/covariance_cache.h"

#include <limits>
#include <map>
#include <random>

#include <Eigen/Dense>
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

void SetExactPoseCovariances(const Reconstruction& reconstruction,
                             MapperCovarianceCache* covariance_cache) {
  for (const image_t image_id : reconstruction.RegImageIds()) {
    covariance_cache->SetPoseCov(image_id, Eigen::Matrix6d::Zero());
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

TEST(IncrementalTriangulator, LegacyFallbackMatchesCovariantCounts) {
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
  IncrementalTriangulator::Options options;
  options.use_covariance = false;
  size_t total_tris = 0;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    total_tris += triangulator.TriangulateImage(options, image_id);
  }

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  EXPECT_EQ(total_tris,
            synthetic_options.num_points3D * reconstruction.NumRegImages());
}

TEST(IncrementalTriangulator, NoCovarianceCacheFallsBackToLegacy) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());
  DeleteAllPoints3D(reconstruction);

  const auto triangulate = [&](bool use_covariance) {
    Reconstruction triangulated_reconstruction = reconstruction;
    // Without a cache, poses are unknown rather than exact.
    IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                         triangulated_reconstruction);
    IncrementalTriangulator::Options options;
    options.use_covariance = use_covariance;
    options.random_seed = 0;
    for (const image_t image_id : triangulated_reconstruction.RegImageIds()) {
      triangulator.TriangulateImage(options, image_id);
    }
    std::map<point3D_t, Eigen::Vector3d> points3D;
    for (const auto& [point3D_id, point3D] :
         triangulated_reconstruction.Points3D()) {
      points3D.emplace(point3D_id, point3D.xyz);
    }
    return points3D;
  };

  const std::map<point3D_t, Eigen::Vector3d> legacy_points3D =
      triangulate(/*use_covariance=*/false);
  EXPECT_EQ(legacy_points3D.size(), synthetic_options.num_points3D);
  EXPECT_EQ(triangulate(/*use_covariance=*/true), legacy_points3D);
}

TEST(IncrementalTriangulator, GetPointCovUnknownPosesAsExact) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);
  const point3D_t point3D_id = *reconstruction.Point3DIds().begin();

  // Without a cache, pose covariances are unknown.
  EXPECT_FALSE(triangulator.GetPointCov(point3D_id).has_value());
  const std::optional<Eigen::Matrix3d> exact_pose_cov =
      triangulator.GetPointCov(point3D_id, /*unknown_poses_as_exact=*/true);
  ASSERT_TRUE(exact_pose_cov.has_value());
  EXPECT_GT(exact_pose_cov->determinant(), 0.0);

  // Missing pose covariances in the cache are unknown as well and results
  // conditioned on exact poses are not cached.
  MapperCovarianceCache covariance_cache;
  triangulator.SetCovarianceCache(&covariance_cache);
  EXPECT_FALSE(triangulator.GetPointCov(point3D_id).has_value());
  EXPECT_EQ(
      triangulator.GetPointCov(point3D_id, /*unknown_poses_as_exact=*/true),
      exact_pose_cov);
  EXPECT_FALSE(covariance_cache.PointCov(point3D_id).has_value());

  // With known pose covariances, the result is cached.
  SetExactPoseCovariances(reconstruction, &covariance_cache);
  EXPECT_EQ(triangulator.GetPointCov(point3D_id), exact_pose_cov);
  EXPECT_EQ(covariance_cache.PointCov(point3D_id), exact_pose_cov);
}

TEST(IncrementalTriangulator, MissingPoseCovarianceFallsBackToLegacy) {
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
  MapperCovarianceCache covariance_cache;
  triangulator.SetCovarianceCache(&covariance_cache);
  for (const image_t image_id : reconstruction.RegImageIds()) {
    triangulator.TriangulateImage(IncrementalTriangulator::Options(), image_id);
  }

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    EXPECT_FALSE(covariance_cache.PointCov(point3D_id).has_value());
  }
}

TEST(IncrementalTriangulator, CovariantStoresPointCovariances) {
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
  MapperCovarianceCache covariance_cache;
  SetExactPoseCovariances(reconstruction, &covariance_cache);
  triangulator.SetCovarianceCache(&covariance_cache);
  for (const image_t image_id : reconstruction.RegImageIds()) {
    triangulator.TriangulateImage(IncrementalTriangulator::Options(), image_id);
  }

  EXPECT_EQ(reconstruction.NumPoints3D(), synthetic_options.num_points3D);
  // Continued points had their covariance dropped; completing all tracks
  // re-estimates them on demand.
  triangulator.CompleteAllTracks(IncrementalTriangulator::Options());
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    const auto cov = covariance_cache.PointCov(point3D_id);
    ASSERT_TRUE(cov.has_value());
    EXPECT_GT(cov->determinant(), 0.0);
  }
}

TEST(IncrementalTriangulator, CovariantCompleteRefreshesCovariances) {
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
  MapperCovarianceCache covariance_cache;
  SetExactPoseCovariances(reconstruction, &covariance_cache);
  triangulator.SetCovarianceCache(&covariance_cache);

  const size_t num_completions =
      triangulator.CompleteAllTracks(IncrementalTriangulator::Options());
  EXPECT_EQ(num_completions, synthetic_options.num_points3D);
  // Completion drops the stale covariances of grown tracks; they are
  // re-estimated on demand at next use.
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    EXPECT_FALSE(covariance_cache.PointCov(point3D_id).has_value());
  }
}

TEST(IncrementalTriangulator, CovariantMergeStoresFreshCovariance) {
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

  SplitPoint3D(reconstruction, point3D_id1);

  const std::vector<point3D_t> ids_before(reconstruction.Point3DIds().begin(),
                                          reconstruction.Point3DIds().end());
  const point3D_t max_id_before =
      *std::max_element(ids_before.begin(), ids_before.end());

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);
  MapperCovarianceCache covariance_cache;
  SetExactPoseCovariances(reconstruction, &covariance_cache);
  triangulator.SetCovarianceCache(&covariance_cache);

  const size_t num_merged = triangulator.MergeTracks(
      IncrementalTriangulator::Options(), {point3D_id1});
  EXPECT_EQ(num_merged, reconstruction.NumRegImages());

  // The merged point has a new id with a fresh covariance; deleted halves
  // leave no stale entries behind.
  point3D_t merged_id = kInvalidPoint3DId;
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    if (point3D_id > max_id_before) {
      merged_id = point3D_id;
    }
  }
  ASSERT_NE(merged_id, kInvalidPoint3DId);
  const auto merged_cov = covariance_cache.PointCov(merged_id);
  ASSERT_TRUE(merged_cov.has_value());
  EXPECT_GT(merged_cov->determinant(), 0.0);
  for (const point3D_t point3D_id : ids_before) {
    if (!reconstruction.ExistsPoint3D(point3D_id)) {
      EXPECT_FALSE(covariance_cache.PointCov(point3D_id).has_value());
    }
  }
}

TEST(IncrementalTriangulator, CovariantUsesPoseCovariance) {
  const auto triangulate_with_pose_cov =
      [](const std::optional<Eigen::Matrix6d>& pose_cov) {
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
        MapperCovarianceCache covariance_cache;
        triangulator.SetCovarianceCache(&covariance_cache);
        for (const image_t image_id : reconstruction.RegImageIds()) {
          covariance_cache.SetPoseCov(
              image_id, pose_cov.value_or(Eigen::Matrix6d::Zero()));
        }
        for (const image_t image_id : reconstruction.RegImageIds()) {
          triangulator.TriangulateImage(IncrementalTriangulator::Options(),
                                        image_id);
        }
        // Re-estimate dropped covariances, so all points are covered.
        triangulator.CompleteAllTracks(IncrementalTriangulator::Options());

        double mean_trace = 0.0;
        for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
          const auto cov = covariance_cache.PointCov(point3D_id);
          if (cov.has_value()) {
            mean_trace += cov->trace();
          }
        }
        mean_trace /= reconstruction.NumPoints3D();
        return std::make_pair(reconstruction.NumPoints3D(), mean_trace);
      };

  const auto [num_points_exact, mean_trace_exact] =
      triangulate_with_pose_cov(std::nullopt);
  EXPECT_EQ(num_points_exact, 20);

  Eigen::Matrix6d pose_cov = Eigen::Matrix6d::Zero();
  pose_cov.topLeftCorner<3, 3>() = 1e-8 * Eigen::Matrix3d::Identity();
  pose_cov.bottomRightCorner<3, 3>() = 1e-6 * Eigen::Matrix3d::Identity();
  const auto [num_points_uncertain, mean_trace_uncertain] =
      triangulate_with_pose_cov(pose_cov);
  // Same tracks are triangulated, but pose uncertainty inflates the point
  // covariances.
  EXPECT_EQ(num_points_uncertain, num_points_exact);
  EXPECT_GT(mean_trace_uncertain, mean_trace_exact);
}

// Completes tracks whose first observation was removed and offset by the
// given pixels, with exact pose covariances and identity measurement
// covariances.
size_t CompleteOffsetObservations(const IncrementalTriangulator::Options& options,
                                  const double offset,
                                  const double measurement_variance_scale) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  auto cache = DatabaseCache::Create(*database, DatabaseCache::Options());

  std::vector<TrackElement> removed_track_els;
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    removed_track_els.push_back(point3D.track.Element(0));
  }
  DeleteOneObservationFromEachTrack(reconstruction);
  for (const TrackElement& track_el : removed_track_els) {
    reconstruction.Image(track_el.image_id)
        .Point2D(track_el.point2D_idx)
        .xy.x() += offset;
  }

  IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                       reconstruction);
  MapperCovarianceCache covariance_cache;
  SetExactPoseCovariances(reconstruction, &covariance_cache);
  covariance_cache.SetMeasurementVarianceScale(measurement_variance_scale);
  triangulator.SetCovarianceCache(&covariance_cache);
  return triangulator.CompleteAllTracks(options);
}

TEST(IncrementalTriangulator, CovariantGatesUseMeasurementVarianceScale) {
  // The offset of 2 sigma of the modeled noise is within the 99% gate, but it
  // is 20 sigma of the calibrated noise.
  IncrementalTriangulator::Options options;
  options.measurement_noise_dof = std::numeric_limits<double>::infinity();
  options.covariance_legacy_gates = false;
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/2, /*scale=*/1),
            20);
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/2, /*scale=*/0.01),
            0);
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/0.5, /*scale=*/0.01),
            0);
  // Heavy-tailed noise widens the gate from 3 to 7 sigma.
  options.measurement_noise_dof = 3;
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/0.5, /*scale=*/0.01),
            20);
}

TEST(IncrementalTriangulator, CovariantLegacyGates) {
  IncrementalTriangulator::Options options;
  options.complete_max_reproj_error = 1;
  options.covariance_legacy_gates = false;
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/2, /*scale=*/1),
            20);
  options.covariance_legacy_gates = true;
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/2, /*scale=*/1), 0);
  EXPECT_EQ(CompleteOffsetObservations(options, /*offset=*/0.5, /*scale=*/1),
            20);
}

TEST(IncrementalTriangulator, CovariantLegacyFallback) {
  // Returns the number of triangulated points and of cached point covariances.
  const auto triangulate = [](const IncrementalTriangulator::Options& options,
                              const double noise_stddev) {
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
    std::mt19937 generator(42);
    std::normal_distribution<double> distribution(0, noise_stddev);
    for (const image_t image_id : reconstruction.RegImageIds()) {
      for (Point2D& point2D : reconstruction.Image(image_id).Points2D()) {
        point2D.xy += Eigen::Vector2d(distribution(generator),
                                      distribution(generator));
      }
    }

    IncrementalTriangulator triangulator(cache->CorrespondenceGraph(),
                                         reconstruction);
    MapperCovarianceCache covariance_cache;
    SetExactPoseCovariances(reconstruction, &covariance_cache);
    // The noise is far outside the gate of the calibrated noise, even for the
    // epipolar residual components of two-view triangulations.
    covariance_cache.SetMeasurementVarianceScale(1e-10);
    triangulator.SetCovarianceCache(&covariance_cache);
    for (const image_t image_id : reconstruction.RegImageIds()) {
      triangulator.TriangulateImage(options, image_id);
    }

    size_t num_point_covs = 0;
    for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
      num_point_covs += covariance_cache.PointCov(point3D_id).has_value();
    }
    return std::make_pair(reconstruction.NumPoints3D(), num_point_covs);
  };

  using NumPointsAndCovs = std::pair<size_t, size_t>;
  IncrementalTriangulator::Options options;
  options.covariance_legacy_fallback = false;
  EXPECT_EQ(triangulate(options, /*noise_stddev=*/0.5), NumPointsAndCovs(0, 0));
  options.covariance_legacy_fallback = true;
  EXPECT_EQ(triangulate(options, /*noise_stddev=*/0.5),
            NumPointsAndCovs(20, 0));
  // No fallback for depth uncertainty failures.
  options.max_relative_depth_uncertainty = 1e-9;
  EXPECT_EQ(triangulate(options, /*noise_stddev=*/0), NumPointsAndCovs(0, 0));
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
