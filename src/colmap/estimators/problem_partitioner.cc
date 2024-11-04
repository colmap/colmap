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

#include "colmap/estimators/problem_partitioner.h"

#include <queue>

namespace colmap {

ProblemPartitioner::BipartiteGraph::BipartiteGraph() {}

ProblemPartitioner::BipartiteGraph::BipartiteGraph(ceres::Problem* problem) {
  std::vector<ceres::ResidualBlockId> residual_block_ids;
  problem->GetResidualBlocks(&residual_block_ids);
  for (const auto& residual_block_id : residual_block_ids) {
    std::vector<double*> param_blocks;
    problem->GetParameterBlocksForResidualBlock(residual_block_id,
                                                &param_blocks);

    for (auto& param_block : param_blocks) {
      AddEdge(param_block, residual_block_id);
    }
  }
}

void ProblemPartitioner::BipartiteGraph::AddEdge(
    double* param_block, ceres::ResidualBlockId residual_block_id) {
  param_to_residual[param_block].push_back(residual_block_id);
  residual_to_param[residual_block_id].push_back(param_block);
}

std::vector<ceres::ResidualBlockId>
ProblemPartitioner::BipartiteGraph::GetResidualBlocks(
    double* param_block) const {
  return param_to_residual.at(param_block);
}

std::vector<double*> ProblemPartitioner::BipartiteGraph::GetParameterBlocks(
    ceres::ResidualBlockId residual_block_id) const {
  return residual_to_param.at(residual_block_id);
}

ProblemPartitioner::ProblemPartitioner() {}

ProblemPartitioner::ProblemPartitioner(ceres::Problem* problem,
                                       Reconstruction* reconstruction) {
  problem_ = problem;
  graph_ = std::make_unique<BipartiteGraph>(problem);

  // Parse parameter blocks for poses
  for (const auto& [image_id, image] : reconstruction->Images()) {
    const double* qvec = image.CamFromWorld().rotation.coeffs().data();
    if (problem_->HasParameterBlock(qvec) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(qvec))) {
      pose_blocks_.insert(const_cast<double*>(qvec));
    }
    const double* tvec = image.CamFromWorld().translation.data();
    if (problem_->HasParameterBlock(tvec) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(tvec))) {
      pose_blocks_.insert(const_cast<double*>(tvec));
    }
  }

  // Parse parameter blocks for 3D points
  for (const auto& [point3D_id, point3D] : reconstruction->Points3D()) {
    const double* point3D_ptr = point3D.xyz.data();
    if (problem_->HasParameterBlock(point3D_ptr) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(point3D_ptr))) {
      point_blocks_.insert(const_cast<double*>(point3D_ptr));
    }
  }
  // Parse parameter blocks for other variables
  SetUpOtherVariablesBlocks();
}

ProblemPartitioner::ProblemPartitioner(
    ceres::Problem* problem,
    const std::vector<const double*>& pose_blocks,
    const std::vector<const double*>& point_blocks) {
  problem_ = problem;
  graph_ = std::make_unique<BipartiteGraph>(problem);

  // Set parameter blocks for poses
  for (auto it = pose_blocks.begin(); it != pose_blocks.end(); ++it) {
    if (problem->HasParameterBlock(*it) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(*it))) {
      pose_blocks_.insert(const_cast<double*>(*it));
    }
  }

  // Set parameter blocks for 3D Points
  for (auto it = point_blocks.begin(); it != point_blocks.end(); ++it) {
    if (problem->HasParameterBlock(*it) &&
        !problem->IsParameterBlockConstant(const_cast<double*>(*it))) {
      point_blocks_.insert(const_cast<double*>(*it));
    }
  }

  // Parse parameter blocks for other variables
  SetUpOtherVariablesBlocks();
}

void ProblemPartitioner::SetPoseBlocks(
    const std::vector<const double*>& pose_blocks) {
  // Parse parameter blocks for poses
  pose_blocks_.clear();
  for (const double* block : pose_blocks) {
    THROW_CHECK(
        problem_->HasParameterBlock(block) &&
        !problem_->IsParameterBlockConstant(const_cast<double*>(block)));
    pose_blocks_.insert(const_cast<double*>(block));
  }

  // Parse parameter blocks for other variables
  SetUpOtherVariablesBlocks();
}

void ProblemPartitioner::SetUpOtherVariablesBlocks() {
  // Parse parameter blocks for other variables
  other_variables_blocks_.clear();
  std::vector<double*> all_parameter_blocks;
  problem_->GetParameterBlocks(&all_parameter_blocks);
  for (double* block : all_parameter_blocks) {
    if (problem_->IsParameterBlockConstant(block)) continue;
    // check if the current parameter block is in either the pose or point
    if (pose_blocks_.find(block) != pose_blocks_.end()) {
      continue;
    }
    if (point_blocks_.find(block) != point_blocks_.end()) {
      continue;
    }
    other_variables_blocks_.insert(block);
  }
}

void ProblemPartitioner::GetBlocks(
    std::vector<const double*>* pose_blocks,
    std::vector<const double*>* other_variables_blocks,
    std::vector<const double*>* point_blocks) const {
  pose_blocks->clear();
  for (double* block : pose_blocks_) {
    pose_blocks->push_back(block);
  }
  other_variables_blocks->clear();
  for (double* block : other_variables_blocks_) {
    other_variables_blocks->push_back(block);
  }
  point_blocks->clear();
  for (double* block : point_blocks_) {
    point_blocks->push_back(block);
  }
}

void ProblemPartitioner::GetBlocksForSubproblem(
    const std::vector<const double*>& subproblem_pose_blocks,
    std::vector<const double*>* subproblem_other_variables_blocks,
    std::vector<const double*>* subproblem_point_blocks,
    std::unordered_set<ceres::ResidualBlockId>* residual_block_ids) const {
  // Reset
  subproblem_other_variables_blocks->clear();
  subproblem_point_blocks->clear();
  residual_block_ids->clear();

  // Construct irrelevant pose blocks
  std::unordered_set<double*> relevant_pose_blocks;
  for (const auto& param : subproblem_pose_blocks) {
    THROW_CHECK(problem_->HasParameterBlock(param));
    THROW_CHECK(
        !problem_->IsParameterBlockConstant(const_cast<double*>(param)));
    relevant_pose_blocks.insert(const_cast<double*>(param));
  }
  std::unordered_set<double*> irrelevant_pose_blocks;
  for (double* param : pose_blocks_) {
    if (relevant_pose_blocks.find(param) != relevant_pose_blocks.end())
      continue;
    irrelevant_pose_blocks.insert(param);
  }

  // Customizd BFS: Traverse all residuals from the relevant pose blocks but
  // stop at irrelevant pose blocks
  std::queue<double*> bfs_queue;
  std::unordered_set<double*> param_visited;
  for (const auto& param : subproblem_pose_blocks) {
    bfs_queue.push(const_cast<double*>(param));
    param_visited.insert(const_cast<double*>(param));
  }
  while (!bfs_queue.empty()) {
    double* current_param = bfs_queue.front();
    bfs_queue.pop();
    for (auto& residual_block_id : graph_->GetResidualBlocks(current_param)) {
      // check if the residual block exists
      if (residual_block_ids->find(residual_block_id) !=
          residual_block_ids->end())
        continue;
      // skip if the residual block links to irrelevant poses
      bool flag = true;
      for (auto& param_block : graph_->GetParameterBlocks(residual_block_id)) {
        if (irrelevant_pose_blocks.find(param_block) !=
            irrelevant_pose_blocks.end()) {
          flag = false;
          break;
        }
      }
      if (!flag) continue;
      // insert residual blocks
      residual_block_ids->insert(residual_block_id);
      for (auto& param_block : graph_->GetParameterBlocks(residual_block_id)) {
        // check if visited
        if (param_visited.find(param_block) != param_visited.end()) continue;
        // non-pose parameter update bfs and subproblem parameter blocks
        bfs_queue.push(param_block);
        param_visited.insert(param_block);
        if (point_blocks_.find(param_block) != point_blocks_.end()) {
          subproblem_point_blocks->push_back(param_block);
        } else {
          subproblem_other_variables_blocks->push_back(param_block);
        }
      }
    }
  }
}

}  // namespace colmap
