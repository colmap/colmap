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

#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"
#include "colmap/util/types.h"

namespace colmap {

bool MergeAndFilterReconstructions(double max_reproj_error,
                                   const Reconstruction& src_reconstruction,
                                   Reconstruction& tgt_reconstruction);

class ObservationManager {
 public:
  struct ImagePairStat {
    // The number of triangulated correspondences between two images.
    size_t num_tri_corrs = 0;
    // The number of total correspondences/matches between two images.
    size_t num_total_corrs = 0;
  };

  explicit ObservationManager(Reconstruction& reconstruction,
                              std::shared_ptr<const CorrespondenceGraph>
                                  correspondence_graph = nullptr);

  inline const std::unordered_map<image_pair_t, ImagePairStat>& ImagePairs()
      const;

  // Add new 3D object, and return its unique ID.
  point3D_t AddPoint3D(
      const Eigen::Vector3d& xyz,
      const Track& track,
      const Eigen::Vector3ub& color = Eigen::Vector3ub::Zero());

  // Add observation to existing 3D point.
  void AddObservation(point3D_t point3D_id, const TrackElement& track_el);

  // Delete a 3D point, and all its references in the observed images.
  void DeletePoint3D(point3D_t point3D_id);

  // Delete one observation from an image and the corresponding 3D point.
  // Note that this deletes the entire 3D point, if the track has two elements
  // prior to calling this method.
  void DeleteObservation(image_t image_id, point2D_t point2D_idx);

  point3D_t MergePoints3D(point3D_t point3D_id1, point3D_t point3D_id2);

  // Filter 3D points with large reprojection error, negative depth, or
  // insufficient triangulation angle.
  //
  // @param max_reproj_error    The maximum reprojection error.
  // @param min_tri_angle       The minimum triangulation angle.
  // @param point3D_ids         The points to be filtered.
  //
  // @return                    The number of filtered observations.
  size_t FilterPoints3D(double max_reproj_error,
                        double min_tri_angle,
                        const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DInImages(double max_reproj_error,
                                double min_tri_angle,
                                const std::unordered_set<image_t>& image_ids);
  size_t FilterAllPoints3D(double max_reproj_error, double min_tri_angle);

  // Filter observations that have negative depth.
  //
  // @return    The number of filtered observations.
  size_t FilterObservationsWithNegativeDepth();

  size_t FilterPoints3DWithSmallTriangulationAngle(
      double min_tri_angle, const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DWithLargeReprojectionError(
      double max_reproj_error,
      const std::unordered_set<point3D_t>& point3D_ids);

  // Filter images without observations or bogus camera parameters.
  //
  // @return    The identifiers of the filtered images.
  std::vector<image_t> FilterImages(double min_focal_length_ratio,
                                    double max_focal_length_ratio,
                                    double max_extra_param);

  // De-register an existing image, and all its references.
  void DeRegisterImage(image_t image_id);

  // Get the number of observations, i.e. the number of image points that
  // have at least one correspondence to another image.
  inline point2D_t NumObservations(image_t image_id) const;

  // Get the number of correspondences for all image points.
  inline point2D_t NumCorrespondences(image_t image_id) const;

  // Get the number of observations that see a triangulated point, i.e. the
  // number of image points that have at least one correspondence to a
  // triangulated point in another image.
  inline point2D_t NumVisiblePoints3D(image_t image_id) const;

  // Get the score of triangulated observations. In contrast to
  // `NumVisiblePoints3D`, this score also captures the distribution
  // of triangulated observations in the image. This is useful to select
  // the next best image in incremental reconstruction, because a more
  // uniform distribution of observations results in more robust registration.
  inline size_t Point3DVisibilityScore(image_t image_id) const;

  // The number of levels in the 3D point multi-resolution visibility pyramid.
  static const int kNumPoint3DVisibilityPyramidLevels;

  // Indicate that another image has a point that is triangulated and has
  // a correspondence to this image point.
  void IncrementCorrespondenceHasPoint3D(image_t image_id,
                                         point2D_t point2D_idx);

  // Indicate that another image has a point that is not triangulated any more
  // and has a correspondence to this image point. This assumes that
  // `IncrementCorrespondenceHasPoint3D` was called for the same image point
  // and correspondence before.
  void DecrementCorrespondenceHasPoint3D(image_t image_id,
                                         point2D_t point2D_idx);

 private:
  void SetObservationAsTriangulated(image_t image_id,
                                    point2D_t point2D_idx,
                                    bool is_continued_point3D);
  void ResetTriObservations(image_t image_id,
                            point2D_t point2D_idx,
                            bool is_deleted_point3D);

  struct ImageStat {
    // The number of image points that have at least one correspondence to
    // another image.
    point2D_t num_observations;

    // The sum of correspondences per image point.
    point2D_t num_correspondences;

    // The number of 2D points, which have at least one corresponding 2D point
    // in another image that is part of a 3D point track, i.e. the sum of
    // `points2D` where `num_tris > 0`.
    point2D_t num_visible_points3D;

    // Per image point, the number of correspondences that have a 3D point.
    std::vector<point2D_t> num_correspondences_have_point3D;

    // Data structure to compute the distribution of triangulated
    // correspondences in the image.
    VisibilityPyramid point3D_visibility_pyramid;
  };

  Reconstruction& reconstruction_;
  const std::shared_ptr<const CorrespondenceGraph> correspondence_graph_;
  std::unordered_map<image_pair_t, ImagePairStat> image_pair_stats_;
  std::unordered_map<image_t, ImageStat> image_stats_;
};

const std::unordered_map<image_pair_t, ObservationManager::ImagePairStat>&
ObservationManager::ImagePairs() const {
  return image_pair_stats_;
}

point2D_t ObservationManager::NumObservations(const image_t image_id) const {
  return image_stats_.at(image_id).num_observations;
}

point2D_t ObservationManager::NumCorrespondences(const image_t image_id) const {
  return image_stats_.at(image_id).num_correspondences;
}

point2D_t ObservationManager::NumVisiblePoints3D(const image_t image_id) const {
  return image_stats_.at(image_id).num_visible_points3D;
}

size_t ObservationManager::Point3DVisibilityScore(
    const image_t image_id) const {
  return image_stats_.at(image_id).point3D_visibility_pyramid.Score();
}

}  // namespace colmap
