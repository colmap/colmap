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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"

#include <memory>

namespace colmap {

// Class that triangulates points during the incremental reconstruction.
// It holds the state and provides all functionality for triangulation.
class IncrementalTriangulator {
 public:
  struct Options {
    // Maximum transitivity to search for correspondences.
    int max_transitivity = 1;

    // Maximum angular error to create new triangulations.
    double create_max_angle_error = 2.0;

    // Maximum angular error to continue existing triangulations.
    double continue_max_angle_error = 2.0;

    // Maximum reprojection error in pixels to merge triangulations.
    double merge_max_reproj_error = 4.0;

    // Maximum reprojection error to complete an existing triangulation.
    double complete_max_reproj_error = 4.0;

    // Maximum transitivity for track completion.
    int complete_max_transitivity = 5;

    // Maximum angular error to re-triangulate under-reconstructed image pairs.
    double re_max_angle_error = 5.0;

    // Minimum ratio of common triangulations between an image pair over the
    // number of correspondences between that image pair to be considered
    // as under-reconstructed.
    double re_min_ratio = 0.2;

    // Maximum number of trials to re-triangulate an image pair.
    int re_max_trials = 1;

    // Minimum pairwise triangulation angle for a stable triangulation.
    double min_angle = 1.5;

    // Whether to ignore two-view tracks.
    bool ignore_two_view_tracks = true;

    // Thresholds for bogus camera parameters. Images with bogus camera
    // parameters are ignored in triangulation.
    double min_focal_length_ratio = 0.1;
    double max_focal_length_ratio = 10.0;
    double max_extra_param = 1.0;

    bool Check() const;
  };

  // Create new incremental triangulator. Note that both the correspondence
  // graph and the reconstruction objects must live as long as the triangulator.
  IncrementalTriangulator(
      std::shared_ptr<const CorrespondenceGraph> correspondence_graph,
      std::shared_ptr<Reconstruction> reconstruction);

  // Triangulate observations of image.
  //
  // Triangulation includes creation of new points, continuation of existing
  // points, and merging of separate points if given image bridges tracks.
  //
  // Note that the given image must be registered and its pose must be set
  // in the associated reconstruction.
  size_t TriangulateImage(const Options& options, image_t image_id);

  // Complete triangulations for image. Tries to create new tracks for not
  // yet triangulated observations and tries to complete existing tracks.
  // Returns the number of completed observations.
  size_t CompleteImage(const Options& options, image_t image_id);

  // Complete tracks for specific 3D points.
  //
  // Completion tries to recursively add observations to a track that might
  // have failed to triangulate before due to inaccurate poses, etc.
  // Returns the number of completed observations.
  size_t CompleteTracks(const Options& options,
                        const std::unordered_set<point3D_t>& point3D_ids);

  // Complete tracks of all 3D points.
  // Returns the number of completed observations.
  size_t CompleteAllTracks(const Options& options);

  // Merge tracks of for specific 3D points.
  // Returns the number of merged observations.
  size_t MergeTracks(const Options& options,
                     const std::unordered_set<point3D_t>& point3D_ids);

  // Merge tracks of all 3D points.
  // Returns the number of merged observations.
  size_t MergeAllTracks(const Options& options);

  // Perform retriangulation for under-reconstructed image pairs. Under-
  // reconstruction usually occurs in the case of a drifting reconstruction.
  //
  // Image pairs are under-reconstructed if less than `Options::tri_re_min_ratio
  // > tri_ratio`, where `tri_ratio` is the number of triangulated matches over
  // inlier matches between the image pair.
  size_t Retriangulate(const Options& options);

  // Indicate that a 3D point has been modified.
  void AddModifiedPoint3D(point3D_t point3D_id);

  // Get changed 3D points, since the last call to `ClearModifiedPoints3D`.
  const std::unordered_set<point3D_t>& GetModifiedPoints3D();

  // Clear the collection of changed 3D points.
  void ClearModifiedPoints3D();

  // Data for a correspondence / element of a track, used to store all
  // relevant data for triangulation, in order to avoid duplicate lookup
  // in the underlying unordered_map's in the Reconstruction
  struct CorrData {
    image_t image_id;
    point2D_t point2D_idx;
    const Image* image;
    const Camera* camera;
    const Point2D* point2D;
  };

 private:
  // Clear cache of bogus camera parameters and merge trials.
  void ClearCaches();

  // Find (transitive) correspondences to other images.
  size_t Find(const Options& options,
              image_t image_id,
              point2D_t point2D_idx,
              size_t transitivity,
              std::vector<CorrData>* corrs_data);

  // Try to create a new 3D point from the given correspondences.
  size_t Create(const Options& options,
                const std::vector<CorrData>& corrs_data);

  // Try to continue the 3D point with the given correspondences.
  size_t Continue(const Options& options,
                  const CorrData& ref_corr_data,
                  const std::vector<CorrData>& corrs_data);

  // Try to merge 3D point with any of its corresponding 3D points.
  size_t Merge(const Options& options, point3D_t point3D_id);

  // Try to transitively complete the track of a 3D point.
  size_t Complete(const Options& options, point3D_t point3D_id);

  // Check if camera has bogus parameters and cache the result.
  bool HasCameraBogusParams(const Options& options, const Camera& camera);

  // Database cache for the reconstruction. Used to retrieve correspondence
  // information for triangulation.
  const std::shared_ptr<const CorrespondenceGraph> correspondence_graph_;

  // Reconstruction of the model. Modified when triangulating new points.
  std::shared_ptr<Reconstruction> reconstruction_;

  // Cache for cameras with bogus parameters.
  std::unordered_map<camera_t, bool> camera_has_bogus_params_;

  // Cache for tried track merges to avoid duplicate merge trials.
  std::unordered_map<point3D_t, std::unordered_set<point3D_t>> merge_trials_;

  // Cache for found correspondences in the graph.
  std::vector<CorrespondenceGraph::Correspondence> found_corrs_;

  // Number of trials to retriangulate image pair.
  std::unordered_map<image_pair_t, int> re_num_trials_;

  // Changed 3D points, i.e. if a 3D point is modified (created, continued,
  // deleted, merged, etc.). Cleared once `ModifiedPoints3D` is called.
  std::unordered_set<point3D_t> modified_point3D_ids_;
};

}  // namespace colmap
