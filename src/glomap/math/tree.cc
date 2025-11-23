#include "tree.h"

// BGL includes
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

}  // namespace

// Function to perform breadth-first search (BFS) on a graph represented by an
// adjacency list
int BFS(const std::vector<std::vector<int>>& graph,
        int root,
        std::vector<int>& parents,
        std::vector<std::pair<int, int>> banned_edges) {
  int num_vertices = graph.size();

  // Create a vector to store the visited status of each vertex
  std::vector<bool> visited(num_vertices, false);

  // Create a vector to store the parent vertex for each vertex
  parents.clear();
  parents.resize(num_vertices, -1);
  parents[root] = root;

  // Create a queue for BFS traversal
  std::queue<int> q;

  // Mark the start vertex as visited and enqueue it
  visited[root] = true;
  q.push(root);

  int counter = 0;
  while (!q.empty()) {
    int current_vertex = q.front();
    q.pop();

    // Process the current vertex
    // Traverse the adjacent vertices
    for (int neighbor : graph[current_vertex]) {
      if (std::find(banned_edges.begin(),
                    banned_edges.end(),
                    std::make_pair(current_vertex, neighbor)) !=
          banned_edges.end())
        continue;
      if (std::find(banned_edges.begin(),
                    banned_edges.end(),
                    std::make_pair(neighbor, current_vertex)) !=
          banned_edges.end())
        continue;

      if (!visited[neighbor]) {
        visited[neighbor] = true;
        parents[neighbor] = current_vertex;
        q.push(neighbor);
        counter++;
      }
    }
  }

  return counter;
}

image_t MaximumSpanningTree(const ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type) {
  std::unordered_map<image_t, int> image_id_to_idx;
  image_id_to_idx.reserve(images.size());
  std::unordered_map<int, image_t> idx_to_image_id;
  idx_to_image_id.reserve(images.size());
  for (auto& [image_id, image] : images) {
    if (image.IsRegistered() == false) continue;
    idx_to_image_id[image_id_to_idx.size()] = image_id;
    image_id_to_idx[image_id] = image_id_to_idx.size();
  }

  double max_weight = 0;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;
    if (type == INLIER_RATIO)
      max_weight = std::max(max_weight, image_pair.weight);
    else
      max_weight =
          std::max(max_weight, static_cast<double>(image_pair.inliers.size()));
  }

  // establish graph
  weighted_graph G(image_id_to_idx.size());
  weight_map weights_boost = boost::get(boost::edge_weight, G);

  edge_desc e;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    const Image& image1 = images.at(image_pair.image_id1);
    const Image& image2 = images.at(image_pair.image_id2);

    if (image1.IsRegistered() == false || image2.IsRegistered() == false) {
      continue;
    }

    int idx1 = image_id_to_idx[image_pair.image_id1];
    int idx2 = image_id_to_idx[image_pair.image_id2];

    // Set the weight to be negative, then the result would be a maximum
    // spanning tree
    e = boost::add_edge(idx1, idx2, G).first;
    if (type == INLIER_NUM)
      weights_boost[e] = max_weight - image_pair.inliers.size();
    else if (type == INLIER_RATIO)
      weights_boost[e] = max_weight - image_pair.weight;
    else
      weights_boost[e] = max_weight - image_pair.inliers.size();
  }

  std::vector<edge_desc>
      mst;  // vector to store MST edges (not a property map!)
  boost::kruskal_minimum_spanning_tree(G, std::back_inserter(mst));

  std::vector<std::vector<int>> edges_list(image_id_to_idx.size());
  for (const auto& edge : mst) {
    auto source = boost::source(edge, G);
    auto target = boost::target(edge, G);
    edges_list[source].push_back(target);
    edges_list[target].push_back(source);
  }

  std::vector<int> parents_idx;
  BFS(edges_list, 0, parents_idx);

  // change the index back to image id
  parents.clear();
  for (int i = 0; i < parents_idx.size(); i++) {
    parents[idx_to_image_id[i]] = idx_to_image_id[parents_idx[i]];
  }

  return idx_to_image_id[0];
}

};  // namespace glomap
