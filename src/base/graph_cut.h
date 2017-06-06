// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_BASE_GRAPH_CUT_H_
#define COLMAP_SRC_BASE_GRAPH_CUT_H_

#include <unordered_map>
#include <vector>

namespace colmap {

// Compute the min-cut of a undirected graph using the Stoer Wagner algorithm.
void ComputeMinGraphCut(const std::vector<std::pair<int, int>>& edges,
                        const std::vector<int>& weights, int* cut_weight,
                        std::vector<char>* cut_labels);

// Compute the normalized min-cut of an undirected graph using Graclus.
// Partitions the graph into clusters and returns the cluster labels per vertex.
std::unordered_map<int, int> ComputeNormalizedMinGraphCut(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& weights, const int num_parts);

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_GRAPH_CUT_H_
