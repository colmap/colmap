#include "glomap/math/tree.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace glomap {
namespace {

typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::undirectedS,
                              boost::no_property,
                              boost::property<boost::edge_weight_t, double>>
    weighted_graph;
typedef boost::property_map<weighted_graph, boost::edge_weight_t>::type
    weight_map;
typedef boost::graph_traits<weighted_graph>::edge_descriptor edge_desc;
typedef boost::graph_traits<weighted_graph>::vertex_descriptor vertex_desc;

int BreadthFirstSearch(const std::vector<std::vector<int>>& adjacency_list,
                       int root,
                       std::vector<int>& parents) {
  const int num_vertices = adjacency_list.size();

  // Create a vector to store the visited status of each vertex
  std::vector<char> visited(num_vertices, false);

  // Create a vector to store the parent vertex for each vertex
  parents.clear();
  parents.resize(num_vertices, -1);
  parents[root] = root;

  // Create a queue for BreadthFirstSearch traversal
  std::queue<int> queue;

  // Mark the start vertex as visited and enqueue it
  visited[root] = true;
  queue.push(root);

  int counter = 0;
  while (!queue.empty()) {
    const int current_vertex = queue.front();
    queue.pop();

    // Process the current vertex
    // Traverse the adjacent vertices
    for (const int neighbor : adjacency_list[current_vertex]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        parents[neighbor] = current_vertex;
        queue.push(neighbor);
        ++counter;
      }
    }
  }

  return counter;
}

}  // namespace

image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type) {
  std::unordered_map<image_t, int> image_id_to_idx;
  image_id_to_idx.reserve(images.size());
  std::unordered_map<int, image_t> image_idx_to_id;
  image_idx_to_id.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    if (!image.HasPose()) {
      continue;
    }
    const int image_idx = image_id_to_idx.size();
    image_idx_to_id[image_idx] = image_id;
    image_id_to_idx[image_id] = image_idx;
  }

  double max_weight = 0;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) {
      continue;
    }
    if (type == WeightType::INLIER_RATIO) {
      max_weight = std::max(max_weight, image_pair.weight);
    } else {
      max_weight =
          std::max(max_weight, static_cast<double>(image_pair.inliers.size()));
    }
  }

  // establish graph
  weighted_graph G(image_id_to_idx.size());
  weight_map weights_boost = boost::get(boost::edge_weight, G);

  edge_desc e;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) {
      continue;
    }

    const Image& image1 = images.at(image_pair.image_id1);
    const Image& image2 = images.at(image_pair.image_id2);

    if (!image1.HasPose() || !image2.HasPose()) {
      continue;
    }

    int idx1 = image_id_to_idx[image_pair.image_id1];
    int idx2 = image_id_to_idx[image_pair.image_id2];

    // Set the weight to be negative, then the result would be a maximum
    // spanning tree
    e = boost::add_edge(idx1, idx2, G).first;
    switch (type) {
      case WeightType::INLIER_NUM:
        weights_boost[e] = max_weight - image_pair.inliers.size();
        break;
      case WeightType::INLIER_RATIO:
        weights_boost[e] = max_weight - image_pair.weight;
        break;
    }
  }

  std::vector<edge_desc>
      mst;  // vector to store MST edges (not a property map!)
  boost::kruskal_minimum_spanning_tree(G, std::back_inserter(mst));

  std::vector<std::vector<int>> edges_list(image_id_to_idx.size());
  for (const auto& edge : mst) {
    const auto source = boost::source(edge, G);
    const auto target = boost::target(edge, G);
    edges_list[source].push_back(target);
    edges_list[target].push_back(source);
  }

  std::vector<int> parents_idx;
  BreadthFirstSearch(edges_list, 0, parents_idx);

  // change the index back to image id
  parents.clear();
  for (int i = 0; i < parents_idx.size(); i++) {
    parents[image_idx_to_id[i]] = image_idx_to_id[parents_idx[i]];
  }

  return image_idx_to_id[0];
}

}  // namespace glomap
