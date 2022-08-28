// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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

#include "sfm/incremental_mapper.h"

#include <array>
#include <fstream>

#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/pose.h"
#include "util/bitmap.h"
#include "util/misc.h"

namespace colmap {
namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }

  image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D()) /
         static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
  return static_cast<float>(image.Point3DVisibilityScore());
}

}  // namespace

bool IncrementalMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  return true;
}

IncrementalMapper::IncrementalMapper(const DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
  CHECK(reconstruction_ == nullptr);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  reconstruction_->SetUp(&database_cache_->CorrespondenceGraph());
  triangulator_.reset(new IncrementalTriangulator(
      &database_cache_->CorrespondenceGraph(), reconstruction));

  num_shared_reg_images_ = 0;
  num_reg_images_per_camera_.clear();
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                  reconstruction->RegImageIds().end());

  prev_init_image_pair_id_ = kInvalidImagePairId;
  prev_init_two_view_geometry_ = TwoViewGeometry();

  filtered_images_.clear();
  num_reg_trials_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}

bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t* image_id1,
                                             image_t* image_id2) {
  CHECK(options.Check());

  std::vector<image_t> image_ids1;
  if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
    // Only *image_id1 provided.
    if (!database_cache_->ExistsImage(*image_id1)) {
      return false;
    }
    image_ids1.push_back(*image_id1);
  } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
    // Only *image_id2 provided.
    if (!database_cache_->ExistsImage(*image_id2)) {
      return false;
    }
    image_ids1.push_back(*image_id2);
  } else {
    // No initial seed image provided.
    image_ids1 = FindFirstInitialImage(options);
  }

  // Try to find good initial pair.
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    *image_id1 = image_ids1[i1];

    const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, *image_id1);

    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      *image_id2 = image_ids2[i2];

      const image_pair_t pair_id =
          Database::ImagePairToPairId(*image_id1, *image_id2);

      // Try every pair only once.
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      init_image_pairs_.insert(pair_id);

      if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
        return true;
      }
    }
  }

  // No suitable pair found in entire dataset.
  *image_id1 = kInvalidImageId;
  *image_id2 = kInvalidImageId;

  return false;
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

bool IncrementalMapper::RegisterInitialImagePair(const Options& options,
                                                 const image_t image_id1,
                                                 const image_t image_id2) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  CHECK(options.Check());

  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // Estimate two-view geometry
  //////////////////////////////////////////////////////////////////////////////

  if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
    return false;
  }

  image1.Qvec() = ComposeIdentityQuaternion();
  image1.Tvec() = Eigen::Vector3d(0, 0, 0);
  image2.Qvec() = prev_init_two_view_geometry_.qvec;
  image2.Tvec() = prev_init_two_view_geometry_.tvec;

  const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
  const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches& corrs =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // Add 3D point tracks.
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  for (const auto& corr : corrs) {
    const Eigen::Vector2d point1_N =
        camera1.ImageToWorld(image1.Point2D(corr.point2D_idx1).XY());
    const Eigen::Vector2d point2_N =
        camera2.ImageToWorld(image2.Point2D(corr.point2D_idx2).XY());
    const Eigen::Vector3d& xyz =
        TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);
    const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
    if (tri_angle >= min_tri_angle_rad &&
        HasPointPositiveDepth(proj_matrix1, xyz) &&
        HasPointPositiveDepth(proj_matrix2, xyz)) {
      track.Element(0).point2D_idx = corr.point2D_idx1;
      track.Element(1).point2D_idx = corr.point2D_idx2;
      reconstruction_->AddPoint3D(xyz, track);
    }
  }

  return true;
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (image.NumVisiblePoints3D() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  const int kCorrTransitivity = 1;

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);
    const CorrespondenceGraph& correspondence_graph =
        database_cache_->CorrespondenceGraph();
    const std::vector<CorrespondenceGraph::Correspondence> corrs =
        correspondence_graph.FindTransitiveCorrespondences(
            image_id, point2D_idx, kCorrTransitivity);

    std::unordered_set<point3D_t> point3D_ids;

    for (const auto corr : corrs) {
      const Image& corr_image = reconstruction_->Image(corr.image_id);
      if (!corr_image.IsRegistered()) {
        continue;
      }

      const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (point3D_ids.count(corr_point2D.Point3DId()) > 0) {
        continue;
      }

      const Camera& corr_camera =
          reconstruction_->Camera(corr_image.CameraId());

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
      point3D_ids.insert(corr_point2D.Point3DId());
      tri_points2D.push_back(point2D.XY());
      tri_points3D.push_back(point3D.XYZ());
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)

  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 100;
  abs_pose_options.ransac_options.max_num_trials = 10000;
  abs_pose_options.ransac_options.confidence = 0.99999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (num_reg_images_per_camera_[image.CameraId()] > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-estimage.
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
      abs_pose_refinement_options.refine_extra_params = false;
    }
  } else {
    // Camera not refined before. Note that the camera parameters might have
    // been changed before but the image was filtered, so we explicitly reset
    // the camera parameters and try to re-estimate them.
    camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  size_t num_inliers;
  std::vector<char> inlier_mask;

  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image.Qvec(), &image.Tvec(), &camera, &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                          tri_points2D, tri_points3D, &image.Qvec(),
                          &image.Tvec(), &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        const TrackElement track_el(image_id, point2D_idx);
        reconstruction_->AddObservation(point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  LocalBundleAdjustmentReport report;

  // Find images that have most 3D points with given image in common.
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  // Do the bundle adjustment only if there is any connected images.
  if (local_bundle.size() > 0) {
    BundleAdjustmentConfig ba_config;
    ba_config.AddImage(image_id);
    for (const image_t local_image_id : local_bundle) {
      ba_config.AddImage(local_image_id);
    }

    // Fix the existing images, if option specified.
    if (options.fix_existing_images) {
      for (const image_t local_image_id : local_bundle) {
        if (existing_image_ids_.count(local_image_id)) {
          ba_config.SetConstantPose(local_image_id);
        }
      }
    }

    // Determine which cameras to fix, when not all the registered images
    // are within the current local bundle.
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }

    for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
      if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
        ba_config.SetConstantCamera(camera_id_and_num_images_pair.first);
      }
    }

    // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
    if (local_bundle.size() == 1) {
      ba_config.SetConstantPose(local_bundle[0]);
      ba_config.SetConstantTvec(image_id, {0});
    } else if (local_bundle.size() > 1) {
      const image_t image_id1 = local_bundle[local_bundle.size() - 1];
      const image_t image_id2 = local_bundle[local_bundle.size() - 2];
      ba_config.SetConstantPose(image_id1);
      if (!options.fix_existing_images ||
          !existing_image_ids_.count(image_id2)) {
        ba_config.SetConstantTvec(image_id2, {0});
      }
    }

    // Make sure, we refine all new and short-track 3D points, no matter if
    // they are fully contained in the local image set or not. Do not include
    // long track 3D points as they are usually already very stable and adding
    // to them to bundle adjustment and track merging/completion would slow
    // down the local bundle adjustment significantly.
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      const size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // Merge refined tracks with other existing points.
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // Complete tracks that may have failed to triangulate before refinement
    // of camera pose and calibration in bundle-adjustment. This may avoid
    // that some points are filtered and it helps for subsequent image
    // registrations.
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // Filter both the modified images and all changed 3D points to make sure
  // there are no outlier points in the model. This results in duplicate work as
  // many of the provided 3D points may also be contained in the adjusted
  // images, but the filtering is not a bottleneck at this point.
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  return report;
}

bool IncrementalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Fix the existing images, if option specified.
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // Fix 7-DOFs of the bundle adjustment problem.
  ba_config.SetConstantPose(reg_image_ids[0]);
  if (!options.fix_existing_images ||
      !existing_image_ids_.count(reg_image_ids[1])) {
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  return true;
}

bool IncrementalMapper::AdjustParallelGlobalBundle(
    const BundleAdjustmentOptions& ba_options,
    const ParallelBundleAdjuster::Options& parallel_ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2)
      << "At least two images must be registered for global bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  ParallelBundleAdjuster bundle_adjuster(parallel_ba_options, ba_options,
                                         ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  return true;
}

size_t IncrementalMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
  CHECK_NOTNULL(reconstruction_);
  return *reconstruction_;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper::FindFirstInitialImage(
    const Options& options) const {
  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    image_t num_correspondences;
  };

  const size_t init_max_reg_trials =
      static_cast<size_t>(options.init_max_reg_trials);

  // Collect information of all not yet registered images with
  // correspondences.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto& image : reconstruction_->Images()) {
    // Only images with correspondences can be registered.
    if (image.second.NumCorrespondences() == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (init_num_reg_trials_.count(image.first) &&
        init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (num_registrations_.count(image.first) > 0 &&
        num_registrations_.at(image.first) > 0) {
      continue;
    }

    const class Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
    ImageInfo image_info;
    image_info.image_id = image.first;
    image_info.prior_focal_length = camera.HasPriorFocalLength();
    image_info.num_correspondences = image.second.NumCorrespondences();
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(
    const Options& options, const image_t image_id1) const {
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction_->Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id1, point2D_idx)) {
      if (num_registrations_.count(corr.image_id) == 0 ||
          num_registrations_.at(corr.image_id) == 0) {
        num_correspondences[corr.image_id] += 1;
      }
    }
  }

  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    point2D_t num_correspondences;
  };

  const size_t init_min_num_inliers =
      static_cast<size_t>(options.init_min_num_inliers);

  // Compose image information in a compact form for sorting.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto elem : num_correspondences) {
    if (elem.second >= init_min_num_inliers) {
      const class Image& image = reconstruction_->Image(elem.first);
      const class Camera& camera = reconstruction_->Camera(image.CameraId());
      ImageInfo image_info;
      image_info.image_id = elem.first;
      image_info.prior_focal_length = camera.HasPriorFocalLength();
      image_info.num_correspondences = elem.second;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  CHECK(options.Check());

  const Image& image = reconstruction_->Image(image_id);
  CHECK(image.IsRegistered());

  // Extract all images that have at least one 3D point with the query image
  // in common, and simultaneously count the number of common 3D points.

  std::unordered_map<image_t, size_t> shared_observations;

  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(image.NumPoints3D());

  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      point3D_ids.insert(point2D.Point3DId());
      const Point3D& point3D = reconstruction_->Point3D(point2D.Point3DId());
      for (const TrackElement& track_el : point3D.Track().Elements()) {
        if (track_el.image_id != image_id) {
          shared_observations[track_el.image_id] += 1;
        }
      }
    }
  }

  // Sort overlapping images according to number of shared observations.

  std::vector<std::pair<image_t, size_t>> overlapping_images(
      shared_observations.begin(), shared_observations.end());
  std::sort(overlapping_images.begin(), overlapping_images.end(),
            [](const std::pair<image_t, size_t>& image1,
               const std::pair<image_t, size_t>& image2) {
              return image1.second > image2.second;
            });

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.

  const size_t num_images =
      static_cast<size_t>(options.local_ba_num_images - 1);
  const size_t num_eff_images = std::min(num_images, overlapping_images.size());

  // Extract most connected images and ensure sufficient triangulation angle.

  std::vector<image_t> local_bundle_image_ids;
  local_bundle_image_ids.reserve(num_eff_images);

  // If the number of overlapping images equals the number of desired images in
  // the local bundle, then simply copy over the image identifiers.
  if (overlapping_images.size() == num_eff_images) {
    for (const auto& overlapping_image : overlapping_images) {
      local_bundle_image_ids.push_back(overlapping_image.first);
    }
    return local_bundle_image_ids;
  }

  // In the following iteration, we start with the most overlapping images and
  // check whether it has sufficient triangulation angle. If none of the
  // overlapping images has sufficient triangulation angle, we relax the
  // triangulation angle threshold and start from the most overlapping image
  // again. In the end, if we still haven't found enough images, we simply use
  // the most overlapping images.

  const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

  // The selection thresholds (minimum triangulation angle, minimum number of
  // shared observations), which are successively relaxed.
  const std::array<std::pair<double, double>, 8> selection_thresholds = {{
      std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
  }};

  const Eigen::Vector3d proj_center = image.ProjectionCenter();
  std::vector<Eigen::Vector3d> shared_points3D;
  shared_points3D.reserve(image.NumPoints3D());
  std::vector<double> tri_angles(overlapping_images.size(), -1.0);
  std::vector<char> used_overlapping_images(overlapping_images.size(), false);

  for (const auto& selection_threshold : selection_thresholds) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Check if the image has sufficient overlap. Since the images are ordered
      // based on the overlap, we can just skip the remaining ones.
      if (overlapping_images[overlapping_image_idx].second <
          selection_threshold.second) {
        break;
      }

      // Check if the image is already in the local bundle.
      if (used_overlapping_images[overlapping_image_idx]) {
        continue;
      }

      const auto& overlapping_image = reconstruction_->Image(
          overlapping_images[overlapping_image_idx].first);
      const Eigen::Vector3d overlapping_proj_center =
          overlapping_image.ProjectionCenter();

      // In the first iteration, compute the triangulation angle. In later
      // iterations, reuse the previously computed value.
      double& tri_angle = tri_angles[overlapping_image_idx];
      if (tri_angle < 0.0) {
        // Collect the commonly observed 3D points.
        shared_points3D.clear();
        for (const Point2D& point2D : image.Points2D()) {
          if (point2D.HasPoint3D() && point3D_ids.count(point2D.Point3DId())) {
            shared_points3D.push_back(
                reconstruction_->Point3D(point2D.Point3DId()).XYZ());
          }
        }

        // Calculate the triangulation angle at a certain percentile.
        const double kTriangulationAnglePercentile = 75;
        tri_angle = Percentile(
            CalculateTriangulationAngles(proj_center, overlapping_proj_center,
                                         shared_points3D),
            kTriangulationAnglePercentile);
      }

      // Check that the image has sufficient triangulation angle.
      if (tri_angle >= selection_threshold.first) {
        local_bundle_image_ids.push_back(overlapping_image.ImageId());
        used_overlapping_images[overlapping_image_idx] = true;
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }

    // Check if we already collected enough images.
    if (local_bundle_image_ids.size() >= num_eff_images) {
      break;
    }
  }

  // In case there are not enough images with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping images.

  if (local_bundle_image_ids.size() < num_eff_images) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_images[overlapping_image_idx]) {
        local_bundle_image_ids.push_back(
            overlapping_images[overlapping_image_idx].first);
        used_overlapping_images[overlapping_image_idx] = true;

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }
  }

  return local_bundle_image_ids;
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_.at(image.CameraId());
  CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const Options& options, const image_t image_id1, const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches matches =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.XY());
  }

  TwoViewGeometry two_view_geometry;
  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry.EstimateCalibrated(camera1, points1, camera2, points2,
                                       matches, two_view_geometry_options);

  if (!two_view_geometry.EstimateRelativePose(camera1, points1, camera2,
                                              points2)) {
    return false;
  }

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.tvec.z()) < options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_ = two_view_geometry;
    return true;
  }

  return false;
}

}  // namespace colmap
