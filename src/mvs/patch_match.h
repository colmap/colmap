// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_MVS_PATCH_MATCH_H_
#define COLMAP_SRC_MVS_PATCH_MATCH_H_

#include <iostream>
#include <memory>
#include <vector>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#ifndef __CUDACC__
#include "util/threading.h"
#endif

namespace colmap {
namespace mvs {

class PatchMatchCuda;

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
 public:
  // Maximum possible window radius for the photometric consistency cost. This
  // value is equal to THREADS_PER_BLOCK in patch_match_cuda.cu and the limit
  // arises from the shared memory implementation.
  const static size_t kMaxWindowRadius = 32;

  struct Options {
    // Index of the GPU used for patch match.
    int gpu_index = -1;

    // Depth range in which to randomly sample depth hypotheses.
    double depth_min = 0.0f;
    double depth_max = 1.0f;

    // Half window size to compute NCC photo-consistency cost.
    int window_radius = 5;

    // Parameters for bilaterally weighted NCC.
    double sigma_spatial = window_radius;
    double sigma_color = 0.2f;

    // Number of random samples to draw in Monte Carlo sampling.
    int num_samples = 15;

    // Spread of the NCC likelihood function.
    double ncc_sigma = 0.6f;

    // Minimum triangulation angle in degrees.
    double min_triangulation_angle = 0.5f;

    // Spread of the incident angle likelihood function.
    double incident_angle_sigma = 0.9f;

    // Number of coordinate descent iterations. Each iteration consists
    // of four sweeps from left to right, top to bottom, and vice versa.
    int num_iterations = 5;

    // Whether to add a regularized geometric consistency term to the cost
    // function. If true, the `depth_maps` and `normal_maps` must not be null.
    bool geom_consistency = false;

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

    // Print the options to stdout.
    void Print() const;
  };

  struct Problem {
    // Index of the reference image.
    int ref_image_id = -1;

    // Indices of the source images.
    std::vector<int> src_image_ids;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const Options& options, const Problem& problem);
  ~PatchMatch();

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Get the computed values after running the algorithm.
  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;

  // Get a list of geometrically consistent images, in the following format:
  //
  //    r_1, c_1, N_1, i_11, i_12, ..., i_1N_1,
  //    r_2, c_2, N_2, i_21, i_22, ..., i_2N_2, ...
  //
  // where r, c are the row and column image coordinates of the pixel,
  // N is the number of consistent images, followed by the N image identifiers.
  // Note that only pixels are listed which are not filtered and that the
  // consistency graph is only filled if filtering is enabled.
  std::vector<int> GetConsistentImageIds() const;

 private:
  const Options options_;
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
//    patch-match.cfg
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
  PatchMatchController(const PatchMatch::Options& options,
                       const std::string& workspace_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name,
                       const int max_image_size);

 private:
  void Run();

  const PatchMatch::Options options_;
  const std::string workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const int max_image_size_;
};

#endif

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_PATCH_MATCH_H_
