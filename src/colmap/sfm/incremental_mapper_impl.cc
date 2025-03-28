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

#include "colmap/sfm/incremental_mapper_impl.h"

#include "colmap/estimators/pose.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/util/misc.h"

#include <array>
#include <fstream>

namespace colmap {
namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(),
            image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }
}

float RankNextImageMaxVisiblePointsNum(
    const image_t image_id, const class ObservationManager& obs_manager) {
  return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id));
}

float RankNextImageMaxVisiblePointsRatio(
    const image_t image_id, const class ObservationManager& obs_manager) {
  return static_cast<float>(obs_manager.NumVisiblePoints3D(image_id)) /
         static_cast<float>(obs_manager.NumObservations(image_id));
}

float RankNextImageMinUncertainty(const image_t image_id,
                                  const class ObservationManager& obs_manager) {
  return static_cast<float>(obs_manager.Point3DVisibilityScore(image_id));
}

}  // namespace

std::vector<image_t> IncrementalMapperImpl::FindFirstInitialImage(
    const IncrementalMapper::Options& options,
    const CorrespondenceGraph& correspondence_graph,
    const Reconstruction& reconstruction,
    const std::unordered_map<image_t, size_t>& init_num_reg_trials,
    const std::unordered_map<image_t, size_t>& num_registrations) {
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
  image_infos.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.HasTrivialFrame()) {  // Remove. Temp hack.
      continue;
    }
    // Only images with correspondences can be registered.
    if (correspondence_graph.NumCorrespondencesForImage(image_id) == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (const auto init_num_reg_trials_it = init_num_reg_trials.find(image_id);
        init_num_reg_trials_it != init_num_reg_trials.end() &&
        init_num_reg_trials_it->second >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (const auto num_registrations_it = num_registrations.find(image_id);
        num_registrations_it != num_registrations.end() &&
        num_registrations_it->second > 0) {
      continue;
    }

    const Camera& camera = *image.CameraPtr();
    ImageInfo image_info;
    image_info.image_id = image_id;
    image_info.prior_focal_length = camera.has_prior_focal_length;
    image_info.num_correspondences =
        correspondence_graph.NumCorrespondencesForImage(image_id);
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(),
      image_infos.end(),
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

std::vector<image_t> IncrementalMapperImpl::FindSecondInitialImage(
    const IncrementalMapper::Options& options,
    image_t image_id1,
    const CorrespondenceGraph& correspondence_graph,
    const Reconstruction& reconstruction,
    const std::unordered_map<image_t, size_t>& num_registrations) {
  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction.Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    const auto corr_range =
        correspondence_graph.FindCorrespondences(image_id1, point2D_idx);
    for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
      if (const auto num_registrations_it =
              num_registrations.find(corr->image_id);
          num_registrations_it == num_registrations.end() ||
          num_registrations_it->second == 0) {
        num_correspondences[corr->image_id] += 1;
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
  image_infos.reserve(num_correspondences.size());
  for (const auto& [image_id, num_corrs] : num_correspondences) {
    if (num_corrs >= init_min_num_inliers) {
      const Image& image = reconstruction.Image(image_id);
      if (!image.HasTrivialFrame()) {  // Remove. Temp hack.
        continue;
      }
      const Camera& camera = *image.CameraPtr();
      ImageInfo image_info;
      image_info.image_id = image_id;
      image_info.prior_focal_length = camera.has_prior_focal_length;
      image_info.num_correspondences = num_corrs;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(),
      image_infos.end(),
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

bool IncrementalMapperImpl::FindInitialImagePair(
    const IncrementalMapper::Options& options,
    const DatabaseCache& database_cache,
    const Reconstruction& reconstruction,
    const std::unordered_map<image_t, size_t>& init_num_reg_trials,
    const std::unordered_map<image_t, size_t>& num_registrations,
    std::unordered_set<image_pair_t>& init_image_pairs,
    TwoViewGeometry& two_view_geometry,
    image_t& image_id1,
    image_t& image_id2) {
  THROW_CHECK(options.Check());

  std::vector<image_t> image_ids1;
  if (image_id1 != kInvalidImageId && image_id2 == kInvalidImageId) {
    // Only image_id1 provided.
    if (!database_cache.ExistsImage(image_id1)) {
      return false;
    }
    image_ids1.push_back(image_id1);
  } else if (image_id1 == kInvalidImageId && image_id2 != kInvalidImageId) {
    // Only image_id2 provided.
    if (!database_cache.ExistsImage(image_id2)) {
      return false;
    }
    image_ids1.push_back(image_id2);
  } else {
    // No initial seed image provided.
    image_ids1 = IncrementalMapperImpl::FindFirstInitialImage(
        options,
        *database_cache.CorrespondenceGraph(),
        reconstruction,
        init_num_reg_trials,
        num_registrations);
  }

  // Try to find good initial pair.
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    image_id1 = image_ids1[i1];

    const std::vector<image_t> image_ids2 =
        IncrementalMapperImpl::FindSecondInitialImage(
            options,
            image_id1,
            *database_cache.CorrespondenceGraph(),
            reconstruction,
            num_registrations);

    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      image_id2 = image_ids2[i2];

      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id1, image_id2);

      // Try every pair only once.
      if (!init_image_pairs.emplace(pair_id).second) {
        continue;
      }

      if (IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
              options,
              database_cache,
              image_id1,
              image_id2,
              two_view_geometry)) {
        return true;
      }
    }
  }

  // No suitable pair found in entire dataset.
  image_id1 = kInvalidImageId;
  image_id2 = kInvalidImageId;

  return false;
}

std::vector<image_t> IncrementalMapperImpl::FindNextImages(
    const IncrementalMapper::Options& options,
    const ObservationManager& obs_manager,
    const std::unordered_set<image_t>& filtered_images,
    std::unordered_map<image_t, size_t>& num_reg_trials) {
  THROW_CHECK(options.Check());
  const Reconstruction& reconstruction = obs_manager.Reconstruction();

  std::function<float(image_t, const class ObservationManager&)>
      rank_image_func;
  switch (options.image_selection_method) {
    case IncrementalMapper::Options::ImageSelectionMethod::
        MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case IncrementalMapper::Options::ImageSelectionMethod::
        MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case IncrementalMapper::Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    // Skip images that are already registered.
    if (image.HasPose()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (obs_manager.NumVisiblePoints3D(image_id) <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t image_num_reg_trials = num_reg_trials[image_id];
    if (image_num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image_id, obs_manager);
    if (filtered_images.count(image_id) == 0 && image_num_reg_trials == 0) {
      image_ranks.emplace_back(image_id, rank);
    } else {
      other_image_ranks.emplace_back(image_id, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(std::move(image_ranks), &ranked_images_ids);
  SortAndAppendNextImages(std::move(other_image_ranks), &ranked_images_ids);

  return ranked_images_ids;
}

std::vector<frame_t> IncrementalMapperImpl::FindLocalBundle(
    const IncrementalMapper::Options& options,
    frame_t frame_id,
    const Reconstruction& reconstruction) {
  THROW_CHECK(options.Check());

  const Frame& frame = reconstruction.Frame(frame_id);
  THROW_CHECK(frame.HasPose());

  // Extract all images (not part of the query frame) that have at least one 3D
  // point with the query frame in common, and simultaneously count the number
  // of common 3D points.
  std::unordered_map<frame_t, size_t> shared_observations;
  std::unordered_set<point3D_t> point3D_ids;
  for (const data_t& data_id : frame.ImageIds()) {
    const image_t image_id = data_id.id;
    const Image& image = reconstruction.Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids.insert(point2D.point3D_id);
        const Point3D& point3D = reconstruction.Point3D(point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          const Image& other_image = reconstruction.Image(track_el.image_id);
          if (other_image.FrameId() != frame_id) {
            ++shared_observations[other_image.FrameId()];
          }
        }
      }
    }
  }

  std::vector<std::pair<frame_t, size_t>> overlapping_frames(
      shared_observations.begin(), shared_observations.end());
  const size_t num_overlapping_frames = overlapping_frames.size();

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.
  const size_t num_eff_frames =
      std::min(static_cast<size_t>(options.local_ba_num_frames - 1),
               overlapping_frames.size());

  // Extract most connected images and ensure sufficient triangulation angle.

  std::vector<frame_t> local_bundle;
  local_bundle.reserve(num_eff_frames);

  // If the number of overlapping images equals the number of desired images in
  // the local bundle, then simply copy over the image identifiers.
  if (num_overlapping_frames == num_eff_frames) {
    for (const auto& [frame_id, _] : overlapping_frames) {
      local_bundle.push_back(frame_id);
    }
    return local_bundle;
  }

  // Sort overlapping images according to number of shared observations.
  std::sort(overlapping_frames.begin(),
            overlapping_frames.end(),
            [](const std::pair<frame_t, size_t>& left,
               const std::pair<frame_t, size_t>& right) {
              return left.second > right.second;
            });

  // In the following iteration, we start with the most overlapping images and
  // check whether it has sufficient triangulation angle. If none of the
  // overlapping images has sufficient triangulation angle, we relax the
  // triangulation angle threshold and start from the most overlapping image
  // again. In the end, if we still haven't found enough images, we simply use
  // the most overlapping images.

  const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

  // The selection thresholds (minimum triangulation angle, minimum number of
  // shared observations), which are successively relaxed.
  const size_t num_points3D = point3D_ids.size();
  const std::array<std::pair<double, double>, 8> selection_thresholds = {{
      std::make_pair(min_tri_angle_rad / 1.0, 0.6 * num_points3D),
      std::make_pair(min_tri_angle_rad / 1.5, 0.6 * num_points3D),
      std::make_pair(min_tri_angle_rad / 2.0, 0.5 * num_points3D),
      std::make_pair(min_tri_angle_rad / 2.5, 0.4 * num_points3D),
      std::make_pair(min_tri_angle_rad / 3.0, 0.3 * num_points3D),
      std::make_pair(min_tri_angle_rad / 4.0, 0.2 * num_points3D),
      std::make_pair(min_tri_angle_rad / 5.0, 0.1 * num_points3D),
      std::make_pair(min_tri_angle_rad / 6.0, 0.1 * num_points3D),
  }};

  std::vector<Eigen::Vector3d> shared_points3D;
  shared_points3D.reserve(num_points3D);
  std::unordered_map<image_pair_t, double> tri_angles;
  std::vector<char> used_overlapping_frames(num_overlapping_frames, false);

  for (const auto& [min_tri_angle_rad, min_num_shared_obs] :
       selection_thresholds) {
    for (size_t overlapping_frame_idx = 0;
         overlapping_frame_idx < num_overlapping_frames;
         ++overlapping_frame_idx) {
      const auto& [overlapping_frame_id, num_shared_obs] =
          overlapping_frames[overlapping_frame_idx];
      // Check if the frame has sufficient overlap. Since the frames are ordered
      // based on the overlap, we can just skip the remaining ones.
      if (num_shared_obs < min_num_shared_obs) {
        break;
      }

      // Check if the frame is already in the local bundle.
      char& used_overlapping_frame =
          used_overlapping_frames[overlapping_frame_idx];
      if (used_overlapping_frame) {
        continue;
      }

      const Frame& overlapping_frame =
          reconstruction.Frame(overlapping_frame_id);
      for (const data_t& data_id : frame.ImageIds()) {
        const Image& image = reconstruction.Image(data_id.id);
        const Eigen::Vector3d proj_center = image.ProjectionCenter();
        for (const data_t& overlapping_data_id : overlapping_frame.ImageIds()) {
          const Image& overlapping_image =
              reconstruction.Image(overlapping_data_id.id);
          const Eigen::Vector3d overlapping_proj_center =
              overlapping_image.ProjectionCenter();

          const image_pair_t image_pair_id = Database::ImagePairToPairId(
              image.ImageId(), overlapping_image.ImageId());

          // In the first iteration, compute the triangulation angle. In later
          // iterations, reuse the previously computed value.
          auto tri_angle_rad_it = tri_angles.find(image_pair_id);
          if (tri_angle_rad_it == tri_angles.end()) {
            // Collect the commonly observed 3D points.
            shared_points3D.clear();
            for (const Point2D& point2D : overlapping_image.Points2D()) {
              if (point2D.HasPoint3D() &&
                  point3D_ids.count(point2D.point3D_id)) {
                shared_points3D.push_back(
                    reconstruction.Point3D(point2D.point3D_id).xyz);
              }
            }

            // Calculate the triangulation angle at a certain percentile.
            constexpr double kTriangulationAnglePercentile = 75;
            tri_angle_rad_it = tri_angles.emplace_hint(
                tri_angle_rad_it,
                image_pair_id,
                Percentile(
                    CalculateTriangulationAngles(
                        proj_center, overlapping_proj_center, shared_points3D),
                    kTriangulationAnglePercentile));
          }

          // Check that the image has sufficient triangulation angle.
          if (tri_angle_rad_it->second >= min_tri_angle_rad) {
            local_bundle.push_back(overlapping_frame_id);
            used_overlapping_frame = true;
            break;
          }
        }

        if (used_overlapping_frame) {
          break;
        }
      }

      // Check if we already collected enough frames.
      if (local_bundle.size() >= num_eff_frames) {
        break;
      }
    }

    // Check if we already collected enough frames.
    if (local_bundle.size() >= num_eff_frames) {
      break;
    }
  }

  // In case there are not enough frames with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping frames.
  if (local_bundle.size() < num_eff_frames) {
    for (size_t overlapping_frame_idx = 0;
         overlapping_frame_idx < overlapping_frames.size();
         ++overlapping_frame_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_frames[overlapping_frame_idx]) {
        local_bundle.push_back(overlapping_frames[overlapping_frame_idx].first);
        used_overlapping_frames[overlapping_frame_idx] = true;

        // Check if we already collected enough frames.
        if (local_bundle.size() >= num_eff_frames) {
          break;
        }
      }
    }
  }

  return local_bundle;
}

bool IncrementalMapperImpl::EstimateInitialTwoViewGeometry(
    const IncrementalMapper::Options& options,
    const DatabaseCache& database_cache,
    const image_t image_id1,
    const image_t image_id2,
    TwoViewGeometry& two_view_geometry) {
  const Image& image1 = database_cache.Image(image_id1);
  const Camera& camera1 = database_cache.Camera(image1.CameraId());

  const Image& image2 = database_cache.Image(image_id2);
  const Camera& camera2 = database_cache.Camera(image2.CameraId());

  const FeatureMatches matches =
      database_cache.CorrespondenceGraph()->FindCorrespondencesBetweenImages(
          image_id1, image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.xy);
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.xy);
  }

  TwoViewGeometryOptions two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry = EstimateCalibratedTwoViewGeometry(
      camera1, points1, camera2, points2, matches, two_view_geometry_options);

  if (!EstimateTwoViewGeometryPose(
          camera1, points1, camera2, points2, &two_view_geometry)) {
    return false;
  }

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.cam2_from_cam1.translation.z()) <
          options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    return true;
  }

  return false;
}

}  // namespace colmap
