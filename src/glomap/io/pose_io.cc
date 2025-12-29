#include "glomap/io/pose_io.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/types.h"

#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace glomap {
namespace {

std::unordered_map<std::string, image_t> InvertImageNames(
    const std::unordered_map<image_t, std::string>& image_names) {
  std::unordered_map<std::string, image_t> result;
  for (const auto& [id, name] : image_names) {
    result[name] = id;
  }
  return result;
}

}  // namespace

std::unordered_map<image_t, std::string> ReadImageNames(
    const std::string& file_path) {
  std::unordered_map<image_t, std::string> image_names;
  std::unordered_map<std::string, image_t> name_to_id;
  image_t next_image_id = 0;

  std::ifstream file(file_path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream line_stream(line);

    std::string name1, name2;
    line_stream >> name1 >> name2;

    for (const auto& name : {name1, name2}) {
      if (name_to_id.find(name) == name_to_id.end()) {
        image_t id = next_image_id++;
        name_to_id[name] = id;
        image_names[id] = name;
      }
    }
  }
  return image_names;
}

std::unordered_map<image_pair_t, Rigid3d> ReadRelativePoses(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names) {
  const auto name_to_id = InvertImageNames(image_names);

  std::unordered_map<image_pair_t, Rigid3d> result;
  std::ifstream file(file_path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream line_stream(line);

    std::string name1, name2;
    line_stream >> name1 >> name2;

    auto it1 = name_to_id.find(name1);
    auto it2 = name_to_id.find(name2);
    if (it1 == name_to_id.end() || it2 == name_to_id.end()) continue;

    const image_t id1 = it1->second;
    const image_t id2 = it2->second;

    Rigid3d pose_rel;
    double qw, qx, qy, qz;
    line_stream >> qw >> qx >> qy >> qz;
    pose_rel.rotation = Eigen::Quaterniond(qw, qx, qy, qz);

    line_stream >> pose_rel.translation[0] >> pose_rel.translation[1] >>
        pose_rel.translation[2];

    const image_pair_t pair_id = colmap::ImagePairToPairId(id1, id2);
    const bool swapped = (id1 > id2);
    result[pair_id] = swapped ? Inverse(pose_rel) : pose_rel;
  }
  return result;
}

void WriteRelativePoses(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_pair_t, Rigid3d>& relative_poses) {
  std::ofstream file(file_path);

  std::map<std::string, image_pair_t> sorted_pairs;
  for (const auto& [pair_id, pose] : relative_poses) {
    const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
    auto it1 = image_names.find(id1);
    auto it2 = image_names.find(id2);
    if (it1 == image_names.end() || it2 == image_names.end()) continue;
    sorted_pairs[it1->second + " " + it2->second] = pair_id;
  }

  for (const auto& [name_pair, pair_id] : sorted_pairs) {
    const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
    const Rigid3d& pose = relative_poses.at(pair_id);
    const auto& q = pose.rotation;
    const auto& t = pose.translation;
    file << image_names.at(id1) << " " << image_names.at(id2) << " " << q.w()
         << " " << q.x() << " " << q.y() << " " << q.z() << " " << t[0] << " "
         << t[1] << " " << t[2] << "\n";
  }
}

std::unordered_map<image_pair_t, double> ReadImagePairWeights(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names) {
  const auto name_to_id = InvertImageNames(image_names);

  std::unordered_map<image_pair_t, double> result;
  std::ifstream file(file_path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream line_stream(line);

    std::string name1, name2;
    double weight;
    line_stream >> name1 >> name2 >> weight;

    auto it1 = name_to_id.find(name1);
    auto it2 = name_to_id.find(name2);
    if (it1 == name_to_id.end() || it2 == name_to_id.end()) continue;

    const image_t id1 = it1->second;
    const image_t id2 = it2->second;

    const image_pair_t pair_id = colmap::ImagePairToPairId(id1, id2);
    result[pair_id] = weight;
  }
  return result;
}

void WriteImagePairWeights(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_pair_t, double>& weights) {
  std::ofstream file(file_path);

  std::map<std::string, image_pair_t> sorted_pairs;
  for (const auto& [pair_id, weight] : weights) {
    const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
    auto it1 = image_names.find(id1);
    auto it2 = image_names.find(id2);
    if (it1 == image_names.end() || it2 == image_names.end()) continue;
    sorted_pairs[it1->second + " " + it2->second] = pair_id;
  }

  for (const auto& [name_pair, pair_id] : sorted_pairs) {
    const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
    file << image_names.at(id1) << " " << image_names.at(id2) << " "
         << weights.at(pair_id) << "\n";
  }
}

std::vector<colmap::PosePrior> ReadGravityPriors(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names) {
  const auto name_to_id = InvertImageNames(image_names);

  std::vector<colmap::PosePrior> pose_priors;
  std::ifstream file(file_path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream line_stream(line);

    std::string name;
    Eigen::Vector3d gravity;
    line_stream >> name >> gravity[0] >> gravity[1] >> gravity[2];

    auto it = name_to_id.find(name);
    if (it == name_to_id.end()) continue;

    auto& pose_prior = pose_priors.emplace_back();
    pose_prior.pose_prior_id = it->second;
    pose_prior.corr_data_id = colmap::data_t(
        colmap::sensor_t(colmap::SensorType::CAMERA, it->second), it->second);
    pose_prior.gravity = gravity;
  }
  return pose_priors;
}

void WriteGravityPriors(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::vector<colmap::PosePrior>& pose_priors) {
  std::ofstream file(file_path);

  std::map<std::string, size_t> sorted_priors;
  for (size_t i = 0; i < pose_priors.size(); ++i) {
    auto it = image_names.find(pose_priors[i].pose_prior_id);
    if (it == image_names.end()) continue;
    sorted_priors[it->second] = i;
  }

  for (const auto& [name, idx] : sorted_priors) {
    const auto& g = pose_priors[idx].gravity;
    file << name << " " << g[0] << " " << g[1] << " " << g[2] << "\n";
  }
}

std::unordered_map<image_t, Eigen::Quaterniond> ReadRotations(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names) {
  const auto name_to_id = InvertImageNames(image_names);

  std::unordered_map<image_t, Eigen::Quaterniond> result;
  std::ifstream file(file_path);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream line_stream(line);

    std::string name;
    double qw, qx, qy, qz;
    line_stream >> name >> qw >> qx >> qy >> qz;

    auto it = name_to_id.find(name);
    if (it == name_to_id.end()) continue;

    result[it->second] = Eigen::Quaterniond(qw, qx, qy, qz);
  }
  return result;
}

void WriteRotations(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_t, Eigen::Quaterniond>& rotations) {
  std::ofstream file(file_path);

  std::set<image_t> sorted_ids;
  for (const auto& [id, q] : rotations) {
    sorted_ids.insert(id);
  }

  for (const image_t id : sorted_ids) {
    auto name_it = image_names.find(id);
    if (name_it == image_names.end()) continue;

    const auto& q = rotations.at(id);
    file << name_it->second << " " << q.w() << " " << q.x() << " " << q.y()
         << " " << q.z() << "\n";
  }
}

}  // namespace glomap
