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

#include "colmap/sfm/incremental_triangulator.h"

#include "colmap/estimators/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace {

bool TriangulateTrack(
    const EstimateTriangulationOptions& options,
    const std::vector<IncrementalTriangulator::CorrData>& corrs_data,
    std::vector<char>& inlier_mask,
    Eigen::Vector3d& xyz) {
  std::vector<Eigen::Vector2d> points;
  points.resize(corrs_data.size());
  std::vector<Rigid3d const*> cams_from_world;
  cams_from_world.resize(corrs_data.size());
  std::vector<Camera const*> cameras;
  cameras.resize(corrs_data.size());
  for (size_t i = 0; i < corrs_data.size(); ++i) {
    const auto& corr_data = corrs_data[i];
    points[i] = corr_data.point2D->xy;
    cams_from_world[i] = &corr_data.image->CamFromWorld();
    cameras[i] = corr_data.camera;
  }

  // Enforce exhaustive sampling for small track lengths.
  EstimateTriangulationOptions options_(options);
  const size_t kExhaustiveSamplingThreshold = 15;
  if (points.size() <= kExhaustiveSamplingThreshold) {
    options_.ransac_options.min_num_trials = NChooseK(points.size(), 2);
  }

  return EstimateTriangulation(
      options_, points, cams_from_world, cameras, &inlier_mask, &xyz);
}

}  // namespace

bool IncrementalTriangulator::Options::Check() const {
  CHECK_OPTION_GE(max_transitivity, 0);
  CHECK_OPTION_GT(create_max_angle_error, 0);
  CHECK_OPTION_GT(continue_max_angle_error, 0);
  CHECK_OPTION_GT(merge_max_reproj_error, 0);
  CHECK_OPTION_GT(complete_max_reproj_error, 0);
  CHECK_OPTION_GE(complete_max_transitivity, 0);
  CHECK_OPTION_GT(re_max_angle_error, 0);
  CHECK_OPTION_GE(re_min_ratio, 0);
  CHECK_OPTION_LE(re_min_ratio, 1);
  CHECK_OPTION_GE(re_max_trials, 0);
  CHECK_OPTION_GT(min_angle, 0);
  return true;
}

IncrementalTriangulator::IncrementalTriangulator(
    std::shared_ptr<const CorrespondenceGraph> correspondence_graph,
    Reconstruction& reconstruction,
    std::shared_ptr<ObservationManager> obs_manager)
    : correspondence_graph_(std::move(correspondence_graph)),
      reconstruction_(reconstruction),
      obs_manager_(std::move(obs_manager)) {
  if (!obs_manager_) {
    obs_manager_ = std::make_shared<ObservationManager>(reconstruction_,
                                                        correspondence_graph_);
  }
}

size_t IncrementalTriangulator::TriangulateImage(const Options& options,
                                                 const image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  const Image& image = reconstruction_.Image(image_id);
  if (!image.HasPose()) {
    return num_tris;
  }

  if (HasCameraBogusParams(options, *image.CameraPtr())) {
    return num_tris;
  }

  // Correspondence data for reference observation in given image. We iterate
  // over all observations of the image and each observation once becomes
  // the reference correspondence.
  CorrData ref_corr_data;
  ref_corr_data.image_id = image_id;
  ref_corr_data.image = &image;
  ref_corr_data.camera = image.CameraPtr();

  // Container for correspondences from reference observation to other images.
  std::vector<CorrData> corrs_data;

  // Try to triangulate all image observations.
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const size_t num_triangulated =
        Find(options,
             image_id,
             point2D_idx,
             static_cast<size_t>(options.max_transitivity),
             &corrs_data);
    if (corrs_data.empty()) {
      continue;
    }

    const Point2D& point2D = image.Point2D(point2D_idx);
    ref_corr_data.point2D_idx = point2D_idx;
    ref_corr_data.point2D = &point2D;

    if (num_triangulated == 0) {
      corrs_data.push_back(ref_corr_data);
      num_tris += Create(options, corrs_data);
    } else {
      // Continue correspondences to existing 3D points.
      num_tris += Continue(options, ref_corr_data, corrs_data);
      // Create points from correspondences that are not continued.
      corrs_data.push_back(ref_corr_data);
      num_tris += Create(options, corrs_data);
    }
  }

  return num_tris;
}

size_t IncrementalTriangulator::CompleteImage(const Options& options,
                                              const image_t image_id) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  const Image& image = reconstruction_.Image(image_id);
  if (!image.HasPose()) {
    return num_tris;
  }

  const Camera& camera = *image.CameraPtr();
  if (HasCameraBogusParams(options, camera)) {
    return num_tris;
  }

  // Setup estimation options.
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_angle);
  tri_options.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  tri_options.ransac_options.max_error = options.complete_max_reproj_error;

  // Correspondence data for reference observation in given image. We iterate
  // over all observations of the image and each observation once becomes
  // the reference correspondence.
  CorrData ref_corr_data;
  ref_corr_data.image_id = image_id;
  ref_corr_data.image = &image;
  ref_corr_data.camera = &camera;

  // Container for correspondences from reference observation to other images.
  std::vector<CorrData> corrs_data;

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    if (point2D.HasPoint3D()) {
      // Complete existing track.
      num_tris += Complete(options, point2D.point3D_id);
      continue;
    }

    if (options.ignore_two_view_tracks &&
        correspondence_graph_->IsTwoViewObservation(image_id, point2D_idx)) {
      continue;
    }

    const size_t num_triangulated =
        Find(options,
             image_id,
             point2D_idx,
             static_cast<size_t>(options.max_transitivity),
             &corrs_data);
    if (num_triangulated || corrs_data.empty()) {
      continue;
    }

    ref_corr_data.point2D = &point2D;
    ref_corr_data.point2D_idx = point2D_idx;
    corrs_data.push_back(ref_corr_data);

    // Estimate triangulation.
    Eigen::Vector3d xyz;
    std::vector<char> inlier_mask;
    if (!TriangulateTrack(tri_options, corrs_data, inlier_mask, xyz)) {
      continue;
    }

    // Add inliers to estimated track.
    Track track;
    track.Reserve(corrs_data.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        const CorrData& corr_data = corrs_data[i];
        track.AddElement(corr_data.image_id, corr_data.point2D_idx);
        num_tris += 1;
      }
    }

    const point3D_t point3D_id = obs_manager_->AddPoint3D(xyz, track);
    modified_point3D_ids_.insert(point3D_id);
  }

  return num_tris;
}

size_t IncrementalTriangulator::CompleteTracks(
    const Options& options, const std::unordered_set<point3D_t>& point3D_ids) {
  THROW_CHECK(options.Check());

  size_t num_completed = 0;

  ClearCaches();

  for (const point3D_t point3D_id : point3D_ids) {
    num_completed += Complete(options, point3D_id);
  }

  return num_completed;
}

size_t IncrementalTriangulator::CompleteAllTracks(const Options& options) {
  THROW_CHECK(options.Check());

  size_t num_completed = 0;

  ClearCaches();

  for (const point3D_t point3D_id : reconstruction_.Point3DIds()) {
    num_completed += Complete(options, point3D_id);
  }

  return num_completed;
}

size_t IncrementalTriangulator::MergeTracks(
    const Options& options, const std::unordered_set<point3D_t>& point3D_ids) {
  THROW_CHECK(options.Check());

  size_t num_merged = 0;

  ClearCaches();

  for (const point3D_t point3D_id : point3D_ids) {
    num_merged += Merge(options, point3D_id);
  }

  return num_merged;
}

size_t IncrementalTriangulator::MergeAllTracks(const Options& options) {
  THROW_CHECK(options.Check());

  size_t num_merged = 0;

  ClearCaches();

  for (const point3D_t point3D_id : reconstruction_.Point3DIds()) {
    num_merged += Merge(options, point3D_id);
  }

  return num_merged;
}

size_t IncrementalTriangulator::Retriangulate(const Options& options) {
  THROW_CHECK(options.Check());

  size_t num_tris = 0;

  ClearCaches();

  Options re_options = options;
  re_options.continue_max_angle_error = options.re_max_angle_error;

  for (const auto& image_pair : obs_manager_->ImagePairs()) {
    // Only perform retriangulation for under-reconstructed image pairs.
    const double tri_ratio =
        static_cast<double>(image_pair.second.num_tri_corrs) /
        static_cast<double>(image_pair.second.num_total_corrs);
    if (tri_ratio >= options.re_min_ratio) {
      continue;
    }

    // Check if images are registered yet.

    image_t image_id1;
    image_t image_id2;
    std::tie(image_id1, image_id2) =
        Database::PairIdToImagePair(image_pair.first);

    const Image& image1 = reconstruction_.Image(image_id1);
    if (!image1.HasPose()) {
      continue;
    }

    const Image& image2 = reconstruction_.Image(image_id2);
    if (!image2.HasPose()) {
      continue;
    }

    // Only perform retriangulation for a maximum number of trials.

    int& num_re_trials = re_num_trials_[image_pair.first];
    if (num_re_trials >= options.re_max_trials) {
      continue;
    }
    num_re_trials += 1;

    const Camera& camera1 = *image1.CameraPtr();
    const Camera& camera2 = *image2.CameraPtr();
    if (HasCameraBogusParams(options, camera1) ||
        HasCameraBogusParams(options, camera2)) {
      continue;
    }

    // Find correspondences and perform retriangulation.

    const FeatureMatches& corrs =
        correspondence_graph_->FindCorrespondencesBetweenImages(image_id1,
                                                                image_id2);

    for (const auto& corr : corrs) {
      const Point2D& point2D1 = image1.Point2D(corr.point2D_idx1);
      const Point2D& point2D2 = image2.Point2D(corr.point2D_idx2);

      // Two cases are possible here: both points belong to the same 3D point
      // or to different 3D points. In the former case, there is nothing
      // to do. In the latter case, we do not attempt retriangulation,
      // as retriangulated correspondences are very likely bogus and
      // would therefore destroy both 3D points if merged.
      if (point2D1.HasPoint3D() && point2D2.HasPoint3D()) {
        continue;
      }

      CorrData corr_data1;
      corr_data1.image_id = image_id1;
      corr_data1.point2D_idx = corr.point2D_idx1;
      corr_data1.image = &image1;
      corr_data1.camera = &camera1;
      corr_data1.point2D = &point2D1;

      CorrData corr_data2;
      corr_data2.image_id = image_id2;
      corr_data2.point2D_idx = corr.point2D_idx2;
      corr_data2.image = &image2;
      corr_data2.camera = &camera2;
      corr_data2.point2D = &point2D2;

      if (point2D1.HasPoint3D() && !point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data1 = {corr_data1};
        num_tris += Continue(re_options, corr_data2, corrs_data1);
      } else if (!point2D1.HasPoint3D() && point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data2 = {corr_data2};
        num_tris += Continue(re_options, corr_data1, corrs_data2);
      } else if (!point2D1.HasPoint3D() && !point2D2.HasPoint3D()) {
        const std::vector<CorrData> corrs_data = {corr_data1, corr_data2};
        // Do not use larger triangulation threshold as this causes
        // significant drift when creating points (options vs. re_options).
        num_tris += Create(options, corrs_data);
      }
      // Else both points have a 3D point, but we do not want to
      // merge points in retriangulation.
    }
  }

  return num_tris;
}

void IncrementalTriangulator::AddModifiedPoint3D(const point3D_t point3D_id) {
  modified_point3D_ids_.insert(point3D_id);
}

const std::unordered_set<point3D_t>&
IncrementalTriangulator::GetModifiedPoints3D() {
  // First remove any missing 3D points from the set.
  for (auto it = modified_point3D_ids_.begin();
       it != modified_point3D_ids_.end();) {
    if (reconstruction_.ExistsPoint3D(*it)) {
      ++it;
    } else {
      modified_point3D_ids_.erase(it++);
    }
  }
  return modified_point3D_ids_;
}

void IncrementalTriangulator::ClearModifiedPoints3D() {
  modified_point3D_ids_.clear();
}

void IncrementalTriangulator::ClearCaches() {
  camera_has_bogus_params_.clear();
  merge_trials_.clear();
  found_corrs_.clear();
}

size_t IncrementalTriangulator::Find(const Options& options,
                                     const image_t image_id,
                                     const point2D_t point2D_idx,
                                     const size_t transitivity,
                                     std::vector<CorrData>* corrs_data) {
  correspondence_graph_->ExtractTransitiveCorrespondences(
      image_id, point2D_idx, transitivity, &found_corrs_);

  corrs_data->clear();
  corrs_data->reserve(found_corrs_.size());

  size_t num_triangulated = 0;

  for (const auto& corr : found_corrs_) {
    const Image& corr_image = reconstruction_.Image(corr.image_id);
    if (!corr_image.HasPose()) {
      continue;
    }

    const Camera& corr_camera = *corr_image.CameraPtr();
    if (HasCameraBogusParams(options, corr_camera)) {
      continue;
    }

    CorrData corr_data;
    corr_data.image_id = corr.image_id;
    corr_data.point2D_idx = corr.point2D_idx;
    corr_data.image = &corr_image;
    corr_data.camera = &corr_camera;
    corr_data.point2D = &corr_image.Point2D(corr.point2D_idx);

    corrs_data->push_back(corr_data);

    if (corr_data.point2D->HasPoint3D()) {
      num_triangulated += 1;
    }
  }

  return num_triangulated;
}

size_t IncrementalTriangulator::Create(
    const Options& options, const std::vector<CorrData>& corrs_data) {
  // Extract correspondences without an existing triangulated observation.
  std::vector<CorrData> create_corrs_data;
  create_corrs_data.reserve(corrs_data.size());
  for (const CorrData& corr_data : corrs_data) {
    if (!corr_data.point2D->HasPoint3D()) {
      create_corrs_data.push_back(corr_data);
    }
  }

  if (create_corrs_data.size() < 2) {
    // Need at least two observations for triangulation.
    return 0;
  } else if (options.ignore_two_view_tracks && create_corrs_data.size() == 2) {
    const CorrData& corr_data1 = create_corrs_data[0];
    if (correspondence_graph_->IsTwoViewObservation(corr_data1.image_id,
                                                    corr_data1.point2D_idx)) {
      return 0;
    }
  }

  // Setup estimation options.
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_angle);
  tri_options.residual_type =
      TriangulationEstimator::ResidualType::ANGULAR_ERROR;
  tri_options.ransac_options.max_error =
      DegToRad(options.create_max_angle_error);

  // Estimate triangulation.
  Eigen::Vector3d xyz;
  std::vector<char> inlier_mask;
  if (!TriangulateTrack(tri_options, create_corrs_data, inlier_mask, xyz)) {
    return 0;
  }

  // Add inliers to estimated track.
  Track track;
  track.Reserve(create_corrs_data.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const CorrData& corr_data = create_corrs_data[i];
      track.AddElement(corr_data.image_id, corr_data.point2D_idx);
    }
  }

  // Add estimated point to reconstruction.
  const size_t track_length = track.Length();
  const point3D_t point3D_id = obs_manager_->AddPoint3D(xyz, track);
  modified_point3D_ids_.insert(point3D_id);

  const size_t kMinRecursiveTrackLength = 3;
  if (create_corrs_data.size() - track_length >= kMinRecursiveTrackLength) {
    return track_length + Create(options, create_corrs_data);
  }

  return track_length;
}

size_t IncrementalTriangulator::Continue(
    const Options& options,
    const CorrData& ref_corr_data,
    const std::vector<CorrData>& corrs_data) {
  // No need to continue, if the reference observation is triangulated.
  if (ref_corr_data.point2D->HasPoint3D()) {
    return 0;
  }

  double best_angle_error = std::numeric_limits<double>::max();
  size_t best_idx = std::numeric_limits<size_t>::max();

  for (size_t idx = 0; idx < corrs_data.size(); ++idx) {
    const CorrData& corr_data = corrs_data[idx];
    if (!corr_data.point2D->HasPoint3D()) {
      continue;
    }

    const Point3D& point3D =
        reconstruction_.Point3D(corr_data.point2D->point3D_id);

    const double angle_error =
        CalculateAngularError(ref_corr_data.point2D->xy,
                              point3D.xyz,
                              ref_corr_data.image->CamFromWorld(),
                              *ref_corr_data.camera);
    if (angle_error < best_angle_error) {
      best_angle_error = angle_error;
      best_idx = idx;
    }
  }

  const double max_angle_error = DegToRad(options.continue_max_angle_error);
  if (best_angle_error <= max_angle_error &&
      best_idx != std::numeric_limits<size_t>::max()) {
    const CorrData& corr_data = corrs_data[best_idx];
    const TrackElement track_el(ref_corr_data.image_id,
                                ref_corr_data.point2D_idx);
    obs_manager_->AddObservation(corr_data.point2D->point3D_id, track_el);
    modified_point3D_ids_.insert(corr_data.point2D->point3D_id);
    return 1;
  }

  return 0;
}

size_t IncrementalTriangulator::Merge(const Options& options,
                                      const point3D_t point3D_id) {
  if (!reconstruction_.ExistsPoint3D(point3D_id)) {
    return 0;
  }

  const double max_squared_reproj_error =
      options.merge_max_reproj_error * options.merge_max_reproj_error;

  const auto& point3D = reconstruction_.Point3D(point3D_id);

  for (const auto& track_el : point3D.track.Elements()) {
    const auto corr_range = correspondence_graph_->FindCorrespondences(
        track_el.image_id, track_el.point2D_idx);
    for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
      const auto& image = reconstruction_.Image(corr->image_id);
      if (!image.HasPose()) {
        continue;
      }

      const Point2D& corr_point2D = image.Point2D(corr->point2D_idx);
      if (!corr_point2D.HasPoint3D() || corr_point2D.point3D_id == point3D_id ||
          merge_trials_[point3D_id].count(corr_point2D.point3D_id) > 0) {
        continue;
      }

      // Try to merge the two 3D points.

      const Point3D& corr_point3D =
          reconstruction_.Point3D(corr_point2D.point3D_id);

      merge_trials_[point3D_id].insert(corr_point2D.point3D_id);
      merge_trials_[corr_point2D.point3D_id].insert(point3D_id);

      // Weighted average of point locations, depending on track length.
      const Eigen::Vector3d merged_xyz =
          (point3D.track.Length() * point3D.xyz +
           corr_point3D.track.Length() * corr_point3D.xyz) /
          (point3D.track.Length() + corr_point3D.track.Length());

      // Count number of inlier track elements of the merged track.
      bool merge_success = true;
      for (const Track* track : {&point3D.track, &corr_point3D.track}) {
        for (const auto test_track_el : track->Elements()) {
          const Image& test_image =
              reconstruction_.Image(test_track_el.image_id);
          const Camera& test_camera = *test_image.CameraPtr();
          const Point2D& test_point2D =
              test_image.Point2D(test_track_el.point2D_idx);
          if (CalculateSquaredReprojectionError(test_point2D.xy,
                                                merged_xyz,
                                                test_image.CamFromWorld(),
                                                test_camera) >
              max_squared_reproj_error) {
            merge_success = false;
            break;
          }
        }
        if (!merge_success) {
          break;
        }
      }

      // Only accept merge if all track elements are inliers.
      if (merge_success) {
        const size_t num_merged =
            point3D.track.Length() + corr_point3D.track.Length();

        const point3D_t merged_point3D_id =
            obs_manager_->MergePoints3D(point3D_id, corr_point2D.point3D_id);

        modified_point3D_ids_.erase(point3D_id);
        modified_point3D_ids_.erase(corr_point2D.point3D_id);
        modified_point3D_ids_.insert(merged_point3D_id);

        // Merge merged 3D point and return, as the original points are
        // deleted.
        const size_t num_merged_recursive = Merge(options, merged_point3D_id);
        if (num_merged_recursive > 0) {
          return num_merged_recursive;
        } else {
          return num_merged;
        }
      }
    }
  }

  return 0;
}

size_t IncrementalTriangulator::Complete(const Options& options,
                                         const point3D_t point3D_id) {
  size_t num_completed = 0;

  if (!reconstruction_.ExistsPoint3D(point3D_id)) {
    return num_completed;
  }

  const double max_squared_reproj_error =
      options.complete_max_reproj_error * options.complete_max_reproj_error;

  const Point3D& point3D = reconstruction_.Point3D(point3D_id);

  std::vector<TrackElement> queue = point3D.track.Elements();

  const int max_transitivity = options.complete_max_transitivity;
  for (int transitivity = 1; transitivity <= max_transitivity; ++transitivity) {
    while (!queue.empty()) {
      const TrackElement queue_elem = queue.back();
      queue.pop_back();

      const auto corr_range = correspondence_graph_->FindCorrespondences(
          queue_elem.image_id, queue_elem.point2D_idx);
      for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
        const Image& image = reconstruction_.Image(corr->image_id);
        if (!image.HasPose()) {
          continue;
        }

        const Point2D& point2D = image.Point2D(corr->point2D_idx);
        if (point2D.HasPoint3D()) {
          continue;
        }

        const Camera& camera = *image.CameraPtr();
        if (HasCameraBogusParams(options, camera)) {
          continue;
        }

        if (CalculateSquaredReprojectionError(
                point2D.xy, point3D.xyz, image.CamFromWorld(), camera) >
            max_squared_reproj_error) {
          continue;
        }

        // Success, add observation to point track.
        obs_manager_->AddObservation(
            point3D_id, TrackElement(corr->image_id, corr->point2D_idx));
        modified_point3D_ids_.insert(point3D_id);

        // Recursively complete track for this new correspondence.
        if (transitivity < max_transitivity) {
          queue.emplace_back(corr->image_id, corr->point2D_idx);
        }

        num_completed += 1;
      }
    }

    if (queue.empty()) {
      break;
    }
  }

  return num_completed;
}

bool IncrementalTriangulator::HasCameraBogusParams(const Options& options,
                                                   const Camera& camera) {
  const auto it = camera_has_bogus_params_.find(camera.camera_id);
  if (it == camera_has_bogus_params_.end()) {
    const bool has_bogus_params =
        camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param);
    camera_has_bogus_params_.emplace(camera.camera_id, has_bogus_params);
    return has_bogus_params;
  } else {
    return it->second;
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const IncrementalTriangulator& triangulator) {
  stream << "IncrementalTriangulator(reconstruction="
         << triangulator.reconstruction_ << ", correspondence_graph=";
  if (triangulator.correspondence_graph_ == nullptr) {
    stream << "null";
  } else {
    stream << *triangulator.correspondence_graph_;
  }
  stream << ")";
  return stream;
}

}  // namespace colmap
