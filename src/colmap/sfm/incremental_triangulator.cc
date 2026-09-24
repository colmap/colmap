// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/incremental_triangulator.h"

#include "colmap/estimators/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace colmap {
namespace {

// Squared Mahalanobis distance of an observation to a 3D point under the
// joint measurement + propagated point and pose covariance. Returns nullopt if
// the observation cannot be scored (projection failure, non-finite inputs,
// non-positive-definite joint covariance).
std::optional<double> WhitenedReprojectionError(
    const Eigen::Vector2d& img_point,
    const Eigen::Matrix2d& img_cov,
    const Eigen::Vector3d& xyz,
    const Eigen::Matrix3d& xyz_cov,
    const Rigid3d& cam_from_world,
    const Eigen::Matrix6d& pose_cov,
    const Camera& camera) {
  CovariantTriangulationEstimator::PointData point_data;
  point_data.img_point = img_point;
  point_data.img_cov = img_cov;
  if (!camera.IsPerspective()) {
    // Only spherical cameras need the observed bearing for the hemisphere
    // test.
    const std::optional<Eigen::Vector3d> cam_ray =
        camera.CamRayFromImg(img_point);
    if (!cam_ray.has_value()) {
      return std::nullopt;
    }
    point_data.cam_ray = *cam_ray;
  }
  CovariantTriangulationEstimator::PoseData pose_data;
  pose_data.cam_from_world = cam_from_world.ToMatrix();
  pose_data.camera = &camera;
  pose_data.pose_cov = pose_cov;
  return CovariantSquaredReprojectionError(
      point_data, pose_data, xyz, xyz_cov);
}

// Extract the observations of a track for triangulation.
void ExtractTrackObservations(
    const std::vector<IncrementalTriangulator::CorrData>& corrs_data,
    std::vector<Eigen::Vector2d>* points,
    std::vector<Rigid3d>* cams_from_world,
    std::vector<Camera const*>* cameras) {
  points->resize(corrs_data.size());
  cams_from_world->resize(corrs_data.size());
  cameras->resize(corrs_data.size());
  for (size_t i = 0; i < corrs_data.size(); ++i) {
    const auto& corr_data = corrs_data[i];
    (*points)[i] = corr_data.point2D->xy;
    (*cams_from_world)[i] = corr_data.image->CamFromWorld();
    (*cameras)[i] = corr_data.camera;
  }
}

// Enforce exhaustive sampling for small track lengths.
void EnforceExhaustiveSampling(const size_t num_observations,
                               RANSACOptions* ransac_options) {
  constexpr size_t kExhaustiveSamplingThreshold = 15;
  if (num_observations <= kExhaustiveSamplingThreshold) {
    ransac_options->min_num_trials = NChooseK(num_observations, 2);
  }
}

bool TriangulateTrack(
    const EstimateTriangulationOptions& options,
    const std::vector<IncrementalTriangulator::CorrData>& corrs_data,
    std::vector<char>& inlier_mask,
    Eigen::Vector3d& xyz) {
  std::vector<Eigen::Vector2d> points;
  std::vector<Rigid3d> cams_from_world;
  std::vector<Camera const*> cameras;
  ExtractTrackObservations(corrs_data, &points, &cams_from_world, &cameras);

  EstimateTriangulationOptions options_(options);
  EnforceExhaustiveSampling(points.size(), &options_.ransac_options);

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
  CHECK_OPTION_IN(re_min_ratio, 0, 1);
  CHECK_OPTION_GE(re_max_trials, 0);
  CHECK_OPTION_GT(min_angle, 0);
  CHECK_OPTION_GT(inlier_chi2_threshold, 0);
  CHECK_OPTION_GT(re_chi2_threshold, 0);
  CHECK_OPTION_GT(max_relative_depth_uncertainty, 0);
  CHECK_OPTION_GT(measurement_noise_dof, 0);
  CHECK_OPTION_GE(random_seed, -1);
  return true;
}

void IncrementalTriangulator::SetCovarianceCache(
    MapperCovarianceCache* covariance_cache) {
  covariance_cache_ = covariance_cache;
}

std::optional<Eigen::Matrix6d> IncrementalTriangulator::GetPoseCov(
    const image_t image_id) const {
  if (covariance_cache_ == nullptr) {
    return std::nullopt;
  }
  return covariance_cache_->PoseCov(image_id);
}

bool IncrementalTriangulator::HasPoseCovariances(
    const std::vector<CorrData>& corrs_data) const {
  return std::all_of(
      corrs_data.begin(), corrs_data.end(), [this](const auto& c) {
        return GetPoseCov(c.image_id).has_value();
      });
}

bool IncrementalTriangulator::HasPoseCovariances(const Track& track) const {
  return std::all_of(track.Elements().begin(),
                     track.Elements().end(),
                     [this](const TrackElement& el) {
                       return GetPoseCov(el.image_id).has_value();
                     });
}

double IncrementalTriangulator::CovariantChiSquareThreshold(
    const Options& options, const double chi2_threshold) const {
  const double measurement_variance_scale =
      covariance_cache_ == nullptr
          ? 1.0
          : covariance_cache_->MeasurementVarianceScale();
  return measurement_variance_scale *
         StudentTTwoDofThreshold(chi2_threshold, options.measurement_noise_dof);
}

IncrementalTriangulator::CovariantTriangulationStatus
IncrementalTriangulator::TriangulateTrackCovariant(
    const Options& options,
    const std::vector<CorrData>& corrs_data,
    std::vector<char>& inlier_mask,
    Eigen::Vector3d& xyz,
    Eigen::Matrix3d& xyz_cov) {
  std::vector<Eigen::Vector2d> points;
  std::vector<Rigid3d> cams_from_world;
  std::vector<Camera const*> cameras;
  ExtractTrackObservations(corrs_data, &points, &cams_from_world, &cameras);
  std::vector<Eigen::Matrix2d> points2D_cov(corrs_data.size());
  std::vector<Eigen::Matrix6d> pose_covs(corrs_data.size());
  for (size_t i = 0; i < corrs_data.size(); ++i) {
    points2D_cov[i] = corrs_data[i].point2D->cov.cast<double>();
    const std::optional<Eigen::Matrix6d> pose_cov =
        GetPoseCov(corrs_data[i].image_id);
    if (!pose_cov.has_value()) {
      return CovariantTriangulationStatus::FAILURE;
    }
    pose_covs[i] = *pose_cov;
  }

  // The relative depth uncertainty is gated below to distinguish its failures.
  CovariantTriangulationOptions cov_options;
  cov_options.inlier_chi2_threshold =
      CovariantChiSquareThreshold(options, options.inlier_chi2_threshold);
  cov_options.max_relative_depth_uncertainty =
      std::numeric_limits<double>::infinity();
  cov_options.ransac_options.random_seed = options.random_seed;
  EnforceExhaustiveSampling(points.size(), &cov_options.ransac_options);

  if (!EstimateCovariantTriangulation(cov_options,
                                      points,
                                      points2D_cov,
                                      cams_from_world,
                                      pose_covs,
                                      cameras,
                                      &inlier_mask,
                                      &xyz,
                                      &xyz_cov)) {
    return CovariantTriangulationStatus::FAILURE;
  }

  // The point covariance scales with the measurement variances, so the
  // relative depth uncertainty scales with their standard deviation.
  const double measurement_std_scale =
      covariance_cache_ == nullptr
          ? 1.0
          : std::sqrt(covariance_cache_->MeasurementVarianceScale());
  std::vector<Rigid3d> inlier_cams_from_world;
  inlier_cams_from_world.reserve(cams_from_world.size());
  for (size_t i = 0; i < cams_from_world.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_cams_from_world.push_back(cams_from_world[i]);
    }
  }
  if (!(measurement_std_scale *
            MaxRelativeDepthUncertainty(xyz, xyz_cov, inlier_cams_from_world) <=
        options.max_relative_depth_uncertainty)) {
    return CovariantTriangulationStatus::UNCERTAIN_DEPTH;
  }

  return CovariantTriangulationStatus::SUCCESS;
}

std::optional<Eigen::Matrix3d> IncrementalTriangulator::GetPointCov(
    const point3D_t point3D_id, const bool unknown_poses_as_exact) {
  if (!reconstruction_.ExistsPoint3D(point3D_id)) {
    return std::nullopt;
  }
  const Point3D& point3D = reconstruction_.Point3D(point3D_id);
  const bool has_pose_covs =
      covariance_cache_ != nullptr && HasPoseCovariances(point3D.track);
  if (has_pose_covs) {
    if (const auto cov = covariance_cache_->PointCov(point3D_id)) {
      return cov;
    }
  } else if (!unknown_poses_as_exact) {
    return std::nullopt;
  }
  std::vector<CovariantTriangulationEstimator::PointData> point_data;
  std::vector<CovariantTriangulationEstimator::PoseData> pose_data;
  point_data.reserve(point3D.track.Length());
  pose_data.reserve(point3D.track.Length());
  for (const auto& track_el : point3D.track.Elements()) {
    const Image& image = reconstruction_.Image(track_el.image_id);
    if (!image.HasPose()) {
      continue;
    }
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);
    CovariantTriangulationEstimator::PointData point_datum;
    point_datum.img_point = point2D.xy;
    point_datum.cam_ray = image.CameraPtr()
                              ->CamRayFromImg(point2D.xy)
                              .value_or(Eigen::Vector3d::UnitZ());
    point_datum.img_cov = point2D.cov.cast<double>();
    point_data.push_back(point_datum);
    CovariantTriangulationEstimator::PoseData pose_datum;
    pose_datum.cam_from_world = image.CamFromWorld().ToMatrix();
    pose_datum.camera = image.CameraPtr();
    pose_datum.pose_cov =
        GetPoseCov(track_el.image_id).value_or(Eigen::Matrix6d::Zero());
    pose_data.push_back(pose_datum);
  }
  const std::optional<Eigen::Matrix3d> cov =
      CovariantTriangulationEstimator::PointCovariance(
          point_data, pose_data, point3D.xyz);
  // Only cache covariances that account for the uncertainty of all poses.
  if (cov.has_value() && has_pose_covs) {
    covariance_cache_->SetPointCov(point3D_id, *cov);
  }
  return cov;
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
  tri_options.ransac_options.random_seed = options.random_seed;

  // Correspondence data for reference observation in given image. We iterate
  // over all observations of the image and each observation once becomes
  // the reference correspondence.
  // (The covariant path ignores tri_options; see below.)
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
    Eigen::Matrix3d xyz_cov;
    std::vector<char> inlier_mask;
    bool use_covariant_estimator =
        options.use_covariance && HasPoseCovariances(corrs_data);
    if (use_covariant_estimator) {
      const CovariantTriangulationStatus status = TriangulateTrackCovariant(
          options, corrs_data, inlier_mask, xyz, xyz_cov);
      if (status == CovariantTriangulationStatus::UNCERTAIN_DEPTH ||
          (status == CovariantTriangulationStatus::FAILURE &&
           !options.covariance_legacy_fallback)) {
        continue;
      }
      use_covariant_estimator =
          status == CovariantTriangulationStatus::SUCCESS;
    }
    if (!use_covariant_estimator &&
        !TriangulateTrack(tri_options, corrs_data, inlier_mask, xyz)) {
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
    if (use_covariant_estimator && covariance_cache_ != nullptr) {
      covariance_cache_->SetPointCov(point3D_id, xyz_cov);
    }
    modified_point3D_ids_.insert(point3D_id);
  }

  return num_tris;
}

size_t IncrementalTriangulator::CompleteTracks(
    const Options& options, const FlatHashSet<point3D_t>& point3D_ids) {
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
    const Options& options, const FlatHashSet<point3D_t>& point3D_ids) {
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
  re_options.inlier_chi2_threshold = options.re_chi2_threshold;

  FeatureMatches matches;
  for (const auto& image_pair : obs_manager_->ImagePairs()) {
    // Only perform retriangulation for under-reconstructed image pairs.
    const double tri_ratio =
        static_cast<double>(image_pair.second.num_tri_corrs) /
        static_cast<double>(image_pair.second.num_total_corrs);
    if (tri_ratio >= options.re_min_ratio) {
      continue;
    }

    // Check if images are registered yet.

    const auto [image_id1, image_id2] = PairIdToImagePair(image_pair.first);

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

    correspondence_graph_->ExtractMatchesBetweenImages(
        image_id1, image_id2, matches);

    for (const auto& match : matches) {
      const Point2D& point2D1 = image1.Point2D(match.point2D_idx1);
      const Point2D& point2D2 = image2.Point2D(match.point2D_idx2);

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
      corr_data1.point2D_idx = match.point2D_idx1;
      corr_data1.image = &image1;
      corr_data1.camera = &camera1;
      corr_data1.point2D = &point2D1;

      CorrData corr_data2;
      corr_data2.image_id = image_id2;
      corr_data2.point2D_idx = match.point2D_idx2;
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
  if (covariance_cache_ != nullptr) {
    covariance_cache_->ErasePointCov(point3D_id);
  }
  modified_point3D_ids_.insert(point3D_id);
}

const FlatHashSet<point3D_t>& IncrementalTriangulator::GetModifiedPoints3D() {
  // First remove any missing 3D points from the set. Collect the ids to remove
  // and erase them by key rather than via an iterator loop:
  // modified_point3D_ids_ is a flat (open-addressing) set whose erase can
  // invalidate other iterators, so an iterator-based erase loop would be
  // unsafe. Erase-by-key is safe.
  std::vector<point3D_t> missing_point3D_ids;
  for (const point3D_t point3D_id : modified_point3D_ids_) {
    if (!reconstruction_.ExistsPoint3D(point3D_id)) {
      missing_point3D_ids.push_back(point3D_id);
    }
  }
  for (const point3D_t point3D_id : missing_point3D_ids) {
    modified_point3D_ids_.erase(point3D_id);
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

  // Estimate triangulation.
  Eigen::Vector3d xyz;
  Eigen::Matrix3d xyz_cov;
  std::vector<char> inlier_mask;
  bool use_covariant_estimator =
      options.use_covariance && HasPoseCovariances(create_corrs_data);
  if (use_covariant_estimator) {
    const CovariantTriangulationStatus status = TriangulateTrackCovariant(
        options, create_corrs_data, inlier_mask, xyz, xyz_cov);
    if (status == CovariantTriangulationStatus::UNCERTAIN_DEPTH ||
        (status == CovariantTriangulationStatus::FAILURE &&
         !options.covariance_legacy_fallback)) {
      return 0;
    }
    use_covariant_estimator = status == CovariantTriangulationStatus::SUCCESS;
  }
  if (!use_covariant_estimator) {
    // Setup estimation options.
    EstimateTriangulationOptions tri_options;
    tri_options.min_tri_angle = DegToRad(options.min_angle);
    tri_options.residual_type =
        TriangulationEstimator::ResidualType::ANGULAR_ERROR;
    tri_options.ransac_options.max_error =
        DegToRad(options.create_max_angle_error);
    tri_options.ransac_options.random_seed = options.random_seed;
    if (!TriangulateTrack(tri_options, create_corrs_data, inlier_mask, xyz)) {
      return 0;
    }
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
  if (use_covariant_estimator && covariance_cache_ != nullptr) {
    covariance_cache_->SetPointCov(point3D_id, xyz_cov);
  }
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

  const std::optional<Eigen::Matrix6d> ref_pose_cov =
      options.use_covariance ? GetPoseCov(ref_corr_data.image_id)
                             : std::nullopt;
  bool can_use_covariance = ref_pose_cov.has_value();
  std::vector<std::optional<Eigen::Matrix3d>> point_covs(corrs_data.size());
  if (can_use_covariance) {
    for (size_t idx = 0; idx < corrs_data.size(); ++idx) {
      const CorrData& corr_data = corrs_data[idx];
      if (!corr_data.point2D->HasPoint3D()) {
        continue;
      }
      point_covs[idx] = GetPointCov(corr_data.point2D->point3D_id);
      if (!point_covs[idx].has_value()) {
        can_use_covariance = false;
        break;
      }
    }
  }

  if (can_use_covariance) {
    double best_mahalanobis_dist_sqr = std::numeric_limits<double>::max();
    size_t best_idx = std::numeric_limits<size_t>::max();

    for (size_t idx = 0; idx < corrs_data.size(); ++idx) {
      const CorrData& corr_data = corrs_data[idx];
      if (!corr_data.point2D->HasPoint3D()) {
        continue;
      }

      const point3D_t point3D_id = corr_data.point2D->point3D_id;

      const Point3D& point3D = reconstruction_.Point3D(point3D_id);
      const std::optional<double> mahalanobis_dist_sqr =
          WhitenedReprojectionError(ref_corr_data.point2D->xy,
                                    ref_corr_data.point2D->cov.cast<double>(),
                                    point3D.xyz,
                                    *point_covs[idx],
                                    ref_corr_data.image->CamFromWorld(),
                                    *ref_pose_cov,
                                    *ref_corr_data.camera);
      if (mahalanobis_dist_sqr.has_value() &&
          *mahalanobis_dist_sqr < best_mahalanobis_dist_sqr) {
        best_mahalanobis_dist_sqr = *mahalanobis_dist_sqr;
        best_idx = idx;
      }
    }

    if (best_mahalanobis_dist_sqr <=
            CovariantChiSquareThreshold(options,
                                        options.inlier_chi2_threshold) &&
        best_idx != std::numeric_limits<size_t>::max() &&
        (!options.covariance_legacy_gates ||
         CalculateAngularReprojectionError(
             ref_corr_data.point2D->xy,
             reconstruction_.Point3D(corrs_data[best_idx].point2D->point3D_id)
                 .xyz,
             ref_corr_data.image->CamFromWorld(),
             *ref_corr_data.camera) <=
             DegToRad(options.continue_max_angle_error))) {
      const CorrData& corr_data = corrs_data[best_idx];
      const TrackElement track_el(ref_corr_data.image_id,
                                  ref_corr_data.point2D_idx);
      obs_manager_->AddObservation(corr_data.point2D->point3D_id, track_el);
      if (covariance_cache_ != nullptr) {
        // The track changed; drop the stale covariance.
        covariance_cache_->ErasePointCov(corr_data.point2D->point3D_id);
      }
      modified_point3D_ids_.insert(corr_data.point2D->point3D_id);
      return 1;
    }

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
        CalculateAngularReprojectionError(ref_corr_data.point2D->xy,
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
    if (covariance_cache_ != nullptr) {
      covariance_cache_->ErasePointCov(corr_data.point2D->point3D_id);
    }
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
      if (!corr_point2D.HasPoint3D() || corr_point2D.point3D_id == point3D_id) {
        continue;
      }

      // Canonical (min, max) pair so this merge is keyed identically
      // regardless of which side of the pair we are visiting from.
      const std::pair<point3D_t, point3D_t> merge_trial_key =
          point3D_id < corr_point2D.point3D_id
              ? std::pair{point3D_id, corr_point2D.point3D_id}
              : std::pair{corr_point2D.point3D_id, point3D_id};
      if (!merge_trials_.insert(merge_trial_key).second) {
        continue;
      }

      // Try to merge the two 3D points.

      const Point3D& corr_point3D =
          reconstruction_.Point3D(corr_point2D.point3D_id);

      // Count number of inlier track elements of the merged track.
      bool merge_success = true;
      Eigen::Vector3d info_merged_xyz;
      Eigen::Matrix3d info_merged_cov;
      const std::optional<Eigen::Matrix3d> cov1 =
          options.use_covariance ? GetPointCov(point3D_id) : std::nullopt;
      const std::optional<Eigen::Matrix3d> cov2 =
          options.use_covariance ? GetPointCov(corr_point2D.point3D_id)
                                 : std::nullopt;
      const bool use_covariant_merge = cov1.has_value() && cov2.has_value() &&
                                       HasPoseCovariances(point3D.track) &&
                                       HasPoseCovariances(corr_point3D.track);
      if (use_covariant_merge) {
        const double chi2_threshold =
            CovariantChiSquareThreshold(options, options.inlier_chi2_threshold);
        // Information-weighted average of the point locations.
        Eigen::LDLT<Eigen::Matrix3d> ldlt1(*cov1);
        Eigen::LDLT<Eigen::Matrix3d> ldlt2(*cov2);
        if (ldlt1.info() != Eigen::Success || ldlt2.info() != Eigen::Success ||
            (ldlt1.vectorD().array() <= 0).any() ||
            (ldlt2.vectorD().array() <= 0).any()) {
          continue;
        }
        const Eigen::Matrix3d info1 = ldlt1.solve(Eigen::Matrix3d::Identity());
        const Eigen::Matrix3d info2 = ldlt2.solve(Eigen::Matrix3d::Identity());
        Eigen::LDLT<Eigen::Matrix3d> ldlt_merged(info1 + info2);
        if (ldlt_merged.info() != Eigen::Success ||
            (ldlt_merged.vectorD().array() <= 0).any()) {
          continue;
        }
        info_merged_cov = ldlt_merged.solve(Eigen::Matrix3d::Identity());
        info_merged_xyz =
            info_merged_cov * (info1 * point3D.xyz + info2 * corr_point3D.xyz);
        if (!info_merged_xyz.allFinite() || !info_merged_cov.allFinite()) {
          continue;
        }
        for (const Track* track : {&point3D.track, &corr_point3D.track}) {
          for (const auto& test_track_el : track->Elements()) {
            const Image& test_image =
                reconstruction_.Image(test_track_el.image_id);
            const Camera& test_camera = *test_image.CameraPtr();
            const Point2D& test_point2D =
                test_image.Point2D(test_track_el.point2D_idx);
            const std::optional<Eigen::Matrix6d> pose_cov =
                GetPoseCov(test_track_el.image_id);
            THROW_CHECK(pose_cov.has_value());
            const std::optional<double> mahalanobis_dist_sqr =
                WhitenedReprojectionError(test_point2D.xy,
                                          test_point2D.cov.cast<double>(),
                                          info_merged_xyz,
                                          info_merged_cov,
                                          test_image.CamFromWorld(),
                                          *pose_cov,
                                          test_camera);
            if (!mahalanobis_dist_sqr.has_value() ||
                *mahalanobis_dist_sqr > chi2_threshold ||
                (options.covariance_legacy_gates &&
                 CalculateSquaredReprojectionError(test_point2D.xy,
                                                   info_merged_xyz,
                                                   test_image.CamFromWorld(),
                                                   test_camera) >
                     max_squared_reproj_error)) {
              merge_success = false;
              break;
            }
          }
          if (!merge_success) {
            break;
          }
        }
      } else {
        // Weighted average of point locations, depending on track length.
        const Eigen::Vector3d merged_xyz =
            (point3D.track.Length() * point3D.xyz +
             corr_point3D.track.Length() * corr_point3D.xyz) /
            (point3D.track.Length() + corr_point3D.track.Length());

        for (const Track* track : {&point3D.track, &corr_point3D.track}) {
          for (const auto& test_track_el : track->Elements()) {
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
      }

      // Only accept merge if all track elements are inliers.
      if (merge_success) {
        const size_t num_merged =
            point3D.track.Length() + corr_point3D.track.Length();

        // Capture before merging: afterwards corr_point2D.point3D_id reads
        // the merged id.
        const point3D_t corr_point3D_id_before_merge = corr_point2D.point3D_id;
        const point3D_t merged_point3D_id =
            obs_manager_->MergePoints3D(point3D_id, corr_point2D.point3D_id);

        if (covariance_cache_ != nullptr) {
          covariance_cache_->ErasePointCov(point3D_id);
          covariance_cache_->ErasePointCov(corr_point3D_id_before_merge);
        }

        if (use_covariant_merge) {
          // Store the validated information-weighted estimate (MergePoints3D
          // uses track-length weighting).
          reconstruction_.Point3D(merged_point3D_id).xyz = info_merged_xyz;
          // Recompute over the merged track (also stores it).
          GetPointCov(merged_point3D_id);
        }

        modified_point3D_ids_.erase(point3D_id);
        modified_point3D_ids_.erase(corr_point3D_id_before_merge);
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

  // Covariance over the pre-completion track, used for gating throughout the
  // completion below. It goes stale as observations are added (the true
  // covariance only shrinks), i.e. gating loosens conservatively; the entry
  // is dropped at the end if the track changed.
  std::optional<Eigen::Matrix3d> point_cov;
  if (options.use_covariance) {
    point_cov = GetPointCov(point3D_id);
  }
  const double chi2_threshold =
      CovariantChiSquareThreshold(options, options.inlier_chi2_threshold);

  // Reuse member-held BFS scratch buffers across Complete() invocations to
  // avoid per-call heap allocations.
  complete_curr_queue_ = point3D.track.Elements();
  complete_next_queue_.clear();
  complete_visited_.clear();

  // Seed visited with the existing track members so the BFS never tries to
  // re-add them.
  for (const TrackElement& el : complete_curr_queue_) {
    complete_visited_.insert(std::make_pair(el.image_id, el.point2D_idx));
  }

  const int max_transitivity = options.complete_max_transitivity;
  for (int transitivity = 1; transitivity <= max_transitivity; ++transitivity) {
    while (!complete_curr_queue_.empty()) {
      const TrackElement queue_elem = complete_curr_queue_.back();
      complete_curr_queue_.pop_back();

      const auto corr_range = correspondence_graph_->FindCorrespondences(
          queue_elem.image_id, queue_elem.point2D_idx);
      for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
        // Two queue entries at the same transitivity level can share
        // correspondences. Dedupe before the (more expensive) reprojection
        // check below so we don't redo it for each parent.
        if (!complete_visited_
                 .insert(std::make_pair(corr->image_id, corr->point2D_idx))
                 .second) {
          continue;
        }

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

        const std::optional<Eigen::Matrix6d> pose_cov =
            options.use_covariance ? GetPoseCov(corr->image_id) : std::nullopt;
        if (point_cov.has_value() && pose_cov.has_value()) {
          const std::optional<double> mahalanobis_dist_sqr =
              WhitenedReprojectionError(point2D.xy,
                                        point2D.cov.cast<double>(),
                                        point3D.xyz,
                                        *point_cov,
                                        image.CamFromWorld(),
                                        *pose_cov,
                                        camera);
          if (!mahalanobis_dist_sqr.has_value() ||
              *mahalanobis_dist_sqr > chi2_threshold ||
              (options.covariance_legacy_gates &&
               CalculateSquaredReprojectionError(point2D.xy,
                                                 point3D.xyz,
                                                 image.CamFromWorld(),
                                                 camera) >
                   max_squared_reproj_error)) {
            continue;
          }
        } else if (CalculateSquaredReprojectionError(
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
          complete_next_queue_.emplace_back(corr->image_id, corr->point2D_idx);
        }

        num_completed += 1;
      }
    }

    if (complete_next_queue_.empty()) {
      break;
    }

    std::swap(complete_curr_queue_, complete_next_queue_);
  }

  if (num_completed > 0 && covariance_cache_ != nullptr) {
    // The track changed; drop the stale covariance.
    covariance_cache_->ErasePointCov(point3D_id);
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
