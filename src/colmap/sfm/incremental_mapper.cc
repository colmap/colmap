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

#include "colmap/sfm/incremental_mapper.h"

#include "colmap/estimators/pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sfm/incremental_mapper_impl.h"
#include "colmap/util/misc.h"

#include <array>
#include <fstream>

namespace colmap {

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

IncrementalMapper::IncrementalMapper(
    std::shared_ptr<const DatabaseCache> database_cache)
    : database_cache_(std::move(database_cache)),
      reconstruction_(nullptr),
      obs_manager_(nullptr),
      triangulator_(nullptr) {}

void IncrementalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK(reconstruction_ == nullptr);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  obs_manager_ = std::make_shared<class ObservationManager>(
      *reconstruction_, database_cache_->CorrespondenceGraph());
  triangulator_ = std::make_shared<IncrementalTriangulator>(
      database_cache_->CorrespondenceGraph(), *reconstruction_, obs_manager_);

  reg_stats_.num_shared_reg_images = 0;
  reg_stats_.num_reg_images_per_camera.clear();
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                  reconstruction->RegImageIds().end());

  filtered_images_.clear();
  reg_stats_.num_reg_trials.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  THROW_CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  triangulator_.reset();
  obs_manager_.reset();
  reconstruction_->TearDown();
  reconstruction_ = nullptr;
}

bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             TwoViewGeometry& two_view_geometry,
                                             image_t& image_id1,
                                             image_t& image_id2) {
  return IncrementalMapperImpl::FindInitialImagePair(
      options,
      *database_cache_,
      *reconstruction_,
      reg_stats_.init_num_reg_trials,
      reg_stats_.num_registrations,
      reg_stats_.init_image_pairs,
      two_view_geometry,
      image_id1,
      image_id2);
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  return IncrementalMapperImpl::FindNextImages(
      options, *obs_manager_, filtered_images_, reg_stats_.num_reg_trials);
}

void IncrementalMapper::RegisterInitialImagePair(
    const Options& options,
    const TwoViewGeometry& two_view_geometry,
    const image_t image_id1,
    const image_t image_id2) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK_EQ(reconstruction_->NumRegImages(), 0);

  THROW_CHECK(options.Check());

  reg_stats_.init_num_reg_trials[image_id1] += 1;
  reg_stats_.init_num_reg_trials[image_id2] += 1;
  reg_stats_.num_reg_trials[image_id1] += 1;
  reg_stats_.num_reg_trials[image_id2] += 1;

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  reg_stats_.init_image_pairs.insert(pair_id);

  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = *image1.CameraPtr();

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = *image2.CameraPtr();

  //////////////////////////////////////////////////////////////////////////////
  // Estimate two-view geometry
  //////////////////////////////////////////////////////////////////////////////

  image1.SetCamFromWorld(Rigid3d());
  image2.SetCamFromWorld(two_view_geometry.cam2_from_cam1);

  const Eigen::Matrix3x4d cam_from_world1 = image1.CamFromWorld().ToMatrix();
  const Eigen::Matrix3x4d cam_from_world2 = image2.CamFromWorld().ToMatrix();
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  const FeatureMatches& corrs =
      database_cache_->CorrespondenceGraph()->FindCorrespondencesBetweenImages(
          image_id1, image_id2);

  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // Add 3D point tracks.
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  for (const auto& corr : corrs) {
    const std::optional<Eigen::Vector2d> cam_point1 =
        camera1.CamFromImg(image1.Point2D(corr.point2D_idx1).xy);
    const std::optional<Eigen::Vector2d> cam_point2 =
        camera2.CamFromImg(image2.Point2D(corr.point2D_idx2).xy);
    if (!cam_point1 || !cam_point2) {
      continue;
    }
    Eigen::Vector3d xyz;
    if (TriangulatePoint(
            cam_from_world1, cam_from_world2, *cam_point1, *cam_point2, &xyz) &&
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz) >=
            min_tri_angle_rad &&
        HasPointPositiveDepth(cam_from_world1, xyz) &&
        HasPointPositiveDepth(cam_from_world2, xyz)) {
      track.Element(0).point2D_idx = corr.point2D_idx1;
      track.Element(1).point2D_idx = corr.point2D_idx2;
      obs_manager_->AddPoint3D(xyz, track);
    }
  }
}

bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK_GE(reconstruction_->NumRegImages(), 2);

  THROW_CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = *image.CameraPtr();

  THROW_CHECK(!image.HasPose()) << "Image cannot be registered multiple times";

  reg_stats_.num_reg_trials[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (obs_manager_->NumVisiblePoints3D(image_id) <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  const std::shared_ptr<const CorrespondenceGraph> correspondence_graph =
      database_cache_->CorrespondenceGraph();

  std::unordered_set<point3D_t> corr_point3D_ids;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);

    corr_point3D_ids.clear();
    const auto corr_range =
        correspondence_graph->FindCorrespondences(image_id, point2D_idx);
    for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
      const Image& corr_image = reconstruction_->Image(corr->image_id);
      if (!corr_image.HasPose()) {
        continue;
      }

      const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (corr_point3D_ids.count(corr_point2D.point3D_id) > 0) {
        continue;
      }

      const Camera& corr_camera = *corr_image.CameraPtr();

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.point3D_id);

      tri_corrs.emplace_back(point2D_idx, corr_point2D.point3D_id);
      corr_point3D_ids.insert(corr_point2D.point3D_id);
      tri_points2D.push_back(point2D.xy);
      tri_points3D.push_back(point3D.xyz);
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
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (reg_stats_.num_reg_images_per_camera[image.CameraId()] > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-estimage.
      camera.params = database_cache_->Camera(image.CameraId()).params;
      abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
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
    camera.params = database_cache_->Camera(image.CameraId()).params;
    abs_pose_options.estimate_focal_length = !camera.has_prior_focal_length;
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

  Rigid3d cam_from_world;
  if (!EstimateAbsolutePose(abs_pose_options,
                            tri_points2D,
                            tri_points3D,
                            &cam_from_world,
                            &camera,
                            &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  if (!RefineAbsolutePose(abs_pose_refinement_options,
                          inlier_mask,
                          tri_points2D,
                          tri_points3D,
                          &cam_from_world,
                          &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  image.SetCamFromWorld(cam_from_world);
  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        const TrackElement track_el(image_id, point2D_idx);
        obs_manager_->AddObservation(point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  THROW_CHECK_NOTNULL(reconstruction_);
  VLOG(1) << "=> Continued observations: "
          << reconstruction_->Image(image_id).NumPoints3D();
  const size_t num_tris =
      triangulator_->TriangulateImage(tri_options, image_id);
  VLOG(1) << "=> Added observations: " << num_tris;
  return num_tris;
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

size_t IncrementalMapper::CompleteAndMergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  const size_t num_completed_observations = CompleteTracks(tri_options);
  VLOG(1) << "=> Completed observations: " << num_completed_observations;
  const size_t num_merged_observations = MergeTracks(tri_options);
  VLOG(1) << "=> Merged observations: " << num_merged_observations;
  return num_completed_observations + num_merged_observations;
}

IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());

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
          ba_config.SetConstantCamPose(local_image_id);
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

    for (const auto& [camera_id, num_images] : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          reg_stats_.num_reg_images_per_camera.at(camera_id);
      if (num_images < num_reg_images_for_camera) {
        ba_config.SetConstantCamIntrinsics(camera_id);
      }
    }

    // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
    if (local_bundle.size() == 1) {
      ba_config.SetConstantCamPose(local_bundle[0]);
      ba_config.SetConstantCamPositions(image_id, {0});
    } else if (local_bundle.size() > 1) {
      const image_t image_id1 = local_bundle[local_bundle.size() - 1];
      const image_t image_id2 = local_bundle[local_bundle.size() - 2];
      ba_config.SetConstantCamPose(image_id1);
      if (!options.fix_existing_images ||
          !existing_image_ids_.count(image_id2)) {
        ba_config.SetConstantCamPositions(image_id2, {0});
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
      if (!point3D.HasError() || point3D.track.Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    std::unique_ptr<BundleAdjuster> bundle_adjuster =
        CreateDefaultBundleAdjuster(
            ba_options, std::move(ba_config), *reconstruction_);
    const ceres::Solver::Summary summary = bundle_adjuster->Solve();

    report.num_adjusted_observations = summary.num_residuals / 2;

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
  report.num_filtered_observations =
      obs_manager_->FilterPoints3DInImages(options.filter_max_reproj_error,
                                           options.filter_min_tri_angle,
                                           filter_image_ids);
  report.num_filtered_observations +=
      obs_manager_->FilterPoints3D(options.filter_max_reproj_error,
                                   options.filter_min_tri_angle,
                                   point3D_ids);

  return report;
}

bool IncrementalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);

  const std::set<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  THROW_CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                             "registered for global "
                                             "bundle-adjustment";

  BundleAdjustmentOptions custom_ba_options = ba_options;
  // Use stricter convergence criteria for first registered images.
  const size_t kMinNumRegImagesForFastBA = 10;
  if (reg_image_ids.size() < kMinNumRegImagesForFastBA) {
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  // Avoid degeneracies in bundle adjustment.
  obs_manager_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Fix the existing images, if option specified.
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantCamPose(image_id);
      }
    }
  }

  // Only use prior pose if at least 3 images have been registered.
  const bool use_prior_position =
      options.use_prior_position && reg_image_ids.size() > 2;

  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  if (!use_prior_position) {
    // Fix 7-DOFs of the bundle adjustment problem.
    auto reg_image_ids_it = reg_image_ids.begin();
    ba_config.SetConstantCamPose(*(reg_image_ids_it++));  // 1st image
    if (!options.fix_existing_images ||
        !existing_image_ids_.count(*reg_image_ids_it)) {
      ba_config.SetConstantCamPositions(*reg_image_ids_it, {0});  // 2nd image
    }

    bundle_adjuster = CreateDefaultBundleAdjuster(
        std::move(custom_ba_options), std::move(ba_config), *reconstruction_);
  } else {
    PosePriorBundleAdjustmentOptions prior_options;
    prior_options.use_robust_loss_on_prior_position =
        options.use_robust_loss_on_prior_position;
    prior_options.prior_position_loss_scale = options.prior_position_loss_scale;
    bundle_adjuster =
        CreatePosePriorBundleAdjuster(std::move(custom_ba_options),
                                      prior_options,
                                      std::move(ba_config),
                                      database_cache_->PosePriors(),
                                      *reconstruction_);
  }

  return bundle_adjuster->Solve().termination_type != ceres::FAILURE;
}

void IncrementalMapper::IterativeLocalRefinement(
    const int max_num_refinements,
    const double max_refinement_change,
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  BundleAdjustmentOptions custom_ba_options = ba_options;
  for (int i = 0; i < max_num_refinements; ++i) {
    const auto report = AdjustLocalBundle(options,
                                          custom_ba_options,
                                          tri_options,
                                          image_id,
                                          GetModifiedPoints3D());
    VLOG(1) << "=> Merged observations: " << report.num_merged_observations;
    VLOG(1) << "=> Completed observations: "
            << report.num_completed_observations;
    VLOG(1) << "=> Filtered observations: " << report.num_filtered_observations;
    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    custom_ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  ClearModifiedPoints3D();
}

void IncrementalMapper::IterativeGlobalRefinement(
    const int max_num_refinements,
    const double max_refinement_change,
    const Options& options,
    const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options,
    const bool normalize_reconstruction) {
  CompleteAndMergeTracks(tri_options);
  VLOG(1) << "=> Retriangulated observations: " << Retriangulate(tri_options);
  for (int i = 0; i < max_num_refinements; ++i) {
    const size_t num_observations = reconstruction_->ComputeNumObservations();
    AdjustGlobalBundle(options, ba_options);
    if (normalize_reconstruction && !options.use_prior_position) {
      // Normalize scene for numerical stability and
      // to avoid large scale changes in the viewer.
      reconstruction_->Normalize();
    }
    size_t num_changed_observations = CompleteAndMergeTracks(tri_options);
    num_changed_observations += FilterPoints(options);
    const double changed =
        num_observations == 0
            ? 0
            : static_cast<double>(num_changed_observations) / num_observations;
    VLOG(1) << StringPrintf("=> Changed observations: %.6f", changed);
    if (changed < max_refinement_change) {
      break;
    }
  }
  ClearModifiedPoints3D();
}

size_t IncrementalMapper::FilterImages(const Options& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids =
      obs_manager_->FilterImages(options.min_focal_length_ratio,
                                 options.max_focal_length_ratio,
                                 options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  const size_t num_filtered_images = image_ids.size();
  VLOG(1) << "=> Filtered images: " << num_filtered_images;
  return num_filtered_images;
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  THROW_CHECK_NOTNULL(obs_manager_);
  THROW_CHECK(options.Check());
  const size_t num_filtered_observations = obs_manager_->FilterAllPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle);
  VLOG(1) << "=> Filtered observations: " << num_filtered_observations;
  return num_filtered_observations;
}

std::shared_ptr<class Reconstruction> IncrementalMapper::Reconstruction()
    const {
  return reconstruction_;
}

class ObservationManager& IncrementalMapper::ObservationManager() const {
  THROW_CHECK_NOTNULL(obs_manager_);
  return *obs_manager_;
}

IncrementalTriangulator& IncrementalMapper::Triangulator() const {
  THROW_CHECK_NOTNULL(triangulator_);
  return *triangulator_;
}

const std::unordered_set<image_t>& IncrementalMapper::FilteredImages() const {
  return filtered_images_;
}

const std::unordered_set<image_t>& IncrementalMapper::ExistingImageIds() const {
  return existing_image_ids_;
}

void IncrementalMapper::ResetInitializationStats() {
  reg_stats_.init_image_pairs.clear();
  reg_stats_.init_num_reg_trials.clear();
}

const std::unordered_map<camera_t, size_t>&
IncrementalMapper::NumRegImagesPerCamera() const {
  return reg_stats_.num_reg_images_per_camera;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return reg_stats_.num_total_reg_images;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return reg_stats_.num_shared_reg_images;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  return IncrementalMapperImpl::FindLocalBundle(
      options, image_id, *reconstruction_);
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      reg_stats_.num_reg_images_per_camera[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = reg_stats_.num_registrations[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    reg_stats_.num_total_reg_images += 1;
  } else if (num_regs_for_image > 1) {
    reg_stats_.num_shared_reg_images += 1;
  }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      reg_stats_.num_reg_images_per_camera.at(image.CameraId());
  THROW_CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = reg_stats_.num_registrations[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    reg_stats_.num_total_reg_images -= 1;
  } else if (num_regs_for_image > 0) {
    reg_stats_.num_shared_reg_images -= 1;
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const IncrementalMapper::Options& options,
    const image_t image_id1,
    const image_t image_id2,
    TwoViewGeometry& two_view_geometry) {
  return IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
      options, *database_cache_, image_id1, image_id2, two_view_geometry);
}

}  // namespace colmap
