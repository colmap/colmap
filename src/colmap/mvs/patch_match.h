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

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/normal_map.h"

#include <iostream>
#include <memory>
#include <vector>
#ifndef __CUDACC__
#include "colmap/util/threading.h"
#endif

namespace colmap {
namespace mvs {

// Maximum possible window radius for the photometric consistency cost. This
// value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
// arises from the shared memory implementation.
const static size_t kMaxPatchMatchWindowRadius = 32;

class ConsistencyGraph;
class PatchMatchCuda;
class Workspace;

struct PatchMatchOptions {
  // Maximum image size in either dimension.
  int max_image_size = -1;

  // Index of the GPU used for patch match. For multi-GPU usage,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Depth range in which to randomly sample depth hypotheses.
  double depth_min = -1.0f;
  double depth_max = -1.0f;

  // Half window size to compute NCC photo-consistency cost.
  int window_radius = 5;

  // Number of pixels to skip when computing NCC. For a value of 1, every
  // pixel is used to compute the NCC. For larger values, only every n-th row
  // and column is used and the computation speed thereby increases roughly by
  // a factor of window_step^2. Note that not all combinations of window sizes
  // and steps produce nice results, especially if the step is greather than 2.
  int window_step = 1;

  // Parameters for bilaterally weighted NCC.
  double sigma_spatial = -1;
  double sigma_color = 0.2f;

  // Number of random samples to draw in Monte Carlo sampling.
  int num_samples = 15;

  // Spread of the NCC likelihood function.
  double ncc_sigma = 0.6f;

  // Minimum triangulation angle in degrees.
  double min_triangulation_angle = 1.0f;

  // Spread of the incident angle likelihood function.
  double incident_angle_sigma = 0.9f;

  // Number of coordinate descent iterations. Each iteration consists
  // of four sweeps from left to right, top to bottom, and vice versa.
  int num_iterations = 5;

  // Whether to add a regularized geometric consistency term to the cost
  // function. If true, the `depth_maps` and `normal_maps` must not be null.
  bool geom_consistency = true;

  // The relative weight of the geometric consistency term w.r.t. to
  // the photo-consistency term.
  double geom_consistency_regularizer = 0.3f;

  // Maximum geometric consistency cost in terms of the forward-backward
  // reprojection error in pixels.
  double geom_consistency_max_cost = 3.0f;

  // Whether to enable filtering.
  bool filter = true;

  // Minimum NCC coefficient for pixel to be photo-consistent.
  double filter_min_ncc = 0.1f;

  // Minimum triangulation angle to be stable.
  double filter_min_triangulation_angle = 3.0f;

  // Minimum number of source images have to be consistent
  // for pixel not to be filtered.
  int filter_min_num_consistent = 2;

  // Maximum forward-backward reprojection error for pixel
  // to be geometrically consistent.
  double filter_geom_consistency_max_cost = 1.0f;

  // Cache size in gigabytes for patch match, which keeps the bitmaps, depth
  // maps, and normal maps of this number of images in memory. A higher value
  // leads to less disk access and faster computation, while a lower value
  // leads to reduced memory usage. Note that a single image can consume a lot
  // of memory, if the consistency graph is dense.
  double cache_size = 32.0;

  // Whether to tolerate missing images/maps in the problem setup
  bool allow_missing_files = false;

  // Whether to write the consistency graph.
  bool write_consistency_graph = false;

  void Print() const;
  bool Check() const {
    if (depth_min != -1.0f || depth_max != -1.0f) {
      CHECK_OPTION_LE(depth_min, depth_max);
      CHECK_OPTION_GE(depth_min, 0.0f);
    }
    CHECK_OPTION_LE(window_radius,
                    static_cast<int>(kMaxPatchMatchWindowRadius));
    CHECK_OPTION_GT(sigma_color, 0.0f);
    CHECK_OPTION_GT(window_radius, 0);
    CHECK_OPTION_GT(window_step, 0);
    CHECK_OPTION_LE(window_step, 2);
    CHECK_OPTION_GT(num_samples, 0);
    CHECK_OPTION_GT(ncc_sigma, 0.0f);
    CHECK_OPTION_GE(min_triangulation_angle, 0.0f);
    CHECK_OPTION_LT(min_triangulation_angle, 180.0f);
    CHECK_OPTION_GT(incident_angle_sigma, 0.0f);
    CHECK_OPTION_GT(num_iterations, 0);
    CHECK_OPTION_GE(geom_consistency_regularizer, 0.0f);
    CHECK_OPTION_GE(geom_consistency_max_cost, 0.0f);
    CHECK_OPTION_GE(filter_min_ncc, -1.0f);
    CHECK_OPTION_LE(filter_min_ncc, 1.0f);
    CHECK_OPTION_GE(filter_min_triangulation_angle, 0.0f);
    CHECK_OPTION_LE(filter_min_triangulation_angle, 180.0f);
    CHECK_OPTION_GE(filter_min_num_consistent, 0);
    CHECK_OPTION_GE(filter_geom_consistency_max_cost, 0.0f);
    CHECK_OPTION_GT(cache_size, 0);
    return true;
  }
};

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
 public:
  struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const PatchMatchOptions& options, const Problem& problem);
  ~PatchMatch();

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Get the computed values after running the algorithm.
  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  ConsistencyGraph GetConsistencyGraph() const;
  Mat<float> GetSelProbMap() const;

 private:
  const PatchMatchOptions options_;
  const Problem problem_;
  std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
};

// This thread processes all problems in a workspace. A workspace has the
// following file structure, if the workspace format is "COLMAP":
//
//    images/*
//    sparse/{cameras.txt, images.txt, points3D.txt}
//    stereo/
//      depth_maps/*
//      normal_maps/*
//      consistency_graphs/*
//      patch-match.cfg
//
// The `patch-match.cfg` file specifies the images to be processed as:
//
//    image_name1.jpg
//    __all__
//    image_name2.jpg
//    __auto__, 20
//    image_name3.jpg
//    image_name1.jpg, image_name2.jpg
//
// Two consecutive lines specify the images used to compute one patch match
// problem. The first line specifies the reference image and the second line the
// source images. Image names are relative to the `images` directory. In this
// example, the first reference image uses all other images as source images,
// the second reference image uses the 20 most connected images as source
// images, and the third reference image uses the first and second as source
// images. Note that all specified images must be reconstructed in the COLMAP
// reconstruction provided in the `sparse` folder.

#ifndef __CUDACC__

class PatchMatchController : public Thread {
 public:
  PatchMatchController(const PatchMatchOptions& options,
                       const std::string& workspace_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name,
                       const std::string& config_path = "");

 private:
  void Run();
  void ReadWorkspace();
  void ReadProblems();
  void ReadGpuIndices();
  void ProcessProblem(const PatchMatchOptions& options, size_t problem_idx);

  const PatchMatchOptions options_;
  const std::string workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::string config_path_;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex workspace_mutex_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<PatchMatch::Problem> problems_;
  std::vector<int> gpu_indices_;
  std::vector<std::pair<float, float>> depth_ranges_;
};

#endif

}  // namespace mvs
}  // namespace colmap
