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

#include "colmap/scene/reconstruction.h"

#include <ceres/ceres.h>

namespace colmap {

// Problem partitioner for bundle adjustment (or extended) problem, useful for
// covariance estimation. The ceres problem is partitioned into three blocks:
// pose blocks, point blocks (can be used for Schur elimination), and other
// variable blocks. One can also get parameter and residual blocks for a
// subproblem of the original problem with a subset of the original pose blocks.
class ProblemPartitioner {
 public:
  ProblemPartitioner();
  ProblemPartitioner(ceres::Problem* problem, Reconstruction* reconstruction);
  ProblemPartitioner(ceres::Problem* problem,
                     const std::vector<const double*>& pose_blocks,
                     const std::vector<const double*>& point_blocks);
  // Manually set pose blocks that are interested while keeping the point blocks
  // unchanged. Needed for the cases where the poses does not fully come from
  // reconstruction, e.g., the rig setup.
  void SetPoseBlocks(const std::vector<const double*>& pose_blocks);

  void GetBlocks(std::vector<const double*>* pose_blocks,
                 std::vector<const double*>* other_variables_blocks,
                 std::vector<const double*>* point_blocks) const;

  // Get parameter blocks and residual blocks for a subproblem with a subset of
  // the original pose blocks. The subproblem include all constraints that
  // connects with the subset pose blocks without passing the complementary set
  // w.r.t. the full pose blocks. This is particularly useful for covariance
  // estimation for very large-scale bundle adjustment problem, e.g., > 10k
  // images.
  void GetBlocksForSubproblem(
      const std::vector<const double*>& subset_pose_blocks,
      std::vector<const double*>* subproblem_other_variables_blocks,
      std::vector<const double*>* subproblem_point_blocks,
      std::vector<ceres::ResidualBlockId>* residual_block_ids) const;

 private:
  struct BipartiteGraph {
    BipartiteGraph();
    explicit BipartiteGraph(ceres::Problem* problem);

    void AddEdge(double* param_block, ceres::ResidualBlockId residual_block_id);

    std::vector<ceres::ResidualBlockId> GetResidualBlocks(
        double* param_block) const;

    std::vector<double*> GetParameterBlocks(
        ceres::ResidualBlockId residual_block_id) const;

    std::unordered_map<double*, std::vector<ceres::ResidualBlockId>>
        param_to_residual;
    std::unordered_map<ceres::ResidualBlockId, std::vector<double*>>
        residual_to_param;
  };

  // The address of the problem
  ceres::Problem* problem_;

  // The bipartite between param blocks and residuals
  std::unique_ptr<BipartiteGraph> graph_;

  // All the parameter blocks
  std::unordered_set<double*> pose_blocks_;
  std::unordered_set<double*> other_variables_blocks_;
  std::unordered_set<double*> point_blocks_;
  void SetUpOtherVariablesBlocks();
};

}  // namespace colmap
