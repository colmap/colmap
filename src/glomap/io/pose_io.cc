#include "pose_io.h"

#include <fstream>
#include <map>
#include <set>

namespace glomap {
void ReadRelPose(const std::string& file_path,
                 std::unordered_map<image_t, Image>& images,
                 ViewGraph& view_graph) {
  std::unordered_map<std::string, image_t> name_idx;
  image_t max_image_id = 0;
  for (const auto& [image_id, image] : images) {
    name_idx[image.file_name] = image_id;

    max_image_id = std::max(max_image_id, image_id);
  }

  // Mark every edge in te view graph as invalid
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    image_pair.is_valid = false;
  }

  std::ifstream file(file_path);

  // Read in data
  std::string line;
  std::string item;

  size_t counter = 0;

  // Required data structures
  // IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
  while (std::getline(file, line)) {
    std::stringstream line_stream(line);

    std::string file1, file2;
    std::getline(line_stream, item, ' ');
    file1 = item;
    std::getline(line_stream, item, ' ');
    file2 = item;

    if (name_idx.find(file1) == name_idx.end()) {
      max_image_id += 1;
      images.insert(
          std::make_pair(max_image_id, Image(max_image_id, -1, file1)));
      name_idx[file1] = max_image_id;
    }
    if (name_idx.find(file2) == name_idx.end()) {
      max_image_id += 1;
      images.insert(
          std::make_pair(max_image_id, Image(max_image_id, -1, file2)));
      name_idx[file2] = max_image_id;
    }

    image_t index1 = name_idx[file1];
    image_t index2 = name_idx[file2];

    const image_pair_t pair_id = colmap::ImagePairToPairId(index1, index2);

    // rotation
    Rigid3d pose_rel;
    for (int i = 0; i < 4; i++) {
      std::getline(line_stream, item, ' ');
      pose_rel.rotation.coeffs()[(i + 3) % 4] = std::stod(item);
    }

    for (int i = 0; i < 3; i++) {
      std::getline(line_stream, item, ' ');
      pose_rel.translation[i] = std::stod(item);
    }

    if (view_graph.image_pairs.find(pair_id) == view_graph.image_pairs.end()) {
      view_graph.image_pairs.insert(
          std::make_pair(pair_id, ImagePair(index1, index2, pose_rel)));
    } else {
      view_graph.image_pairs[pair_id].cam2_from_cam1 = pose_rel;
      view_graph.image_pairs[pair_id].is_valid = true;
      view_graph.image_pairs[pair_id].config =
          colmap::TwoViewGeometry::CALIBRATED;
    }
    counter++;
  }
  LOG(INFO) << counter << " relative poses are loaded";
}

void ReadRelWeight(const std::string& file_path,
                   const std::unordered_map<image_t, Image>& images,
                   ViewGraph& view_graph) {
  std::unordered_map<std::string, image_t> name_idx;
  for (const auto& [image_id, image] : images) {
    name_idx[image.file_name] = image_id;
  }

  std::ifstream file(file_path);

  // Read in data
  std::string line;
  std::string item;

  size_t counter = 0;

  // Required data structures
  // IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
  while (std::getline(file, line)) {
    std::stringstream line_stream(line);

    std::string file1, file2;
    std::getline(line_stream, item, ' ');
    file1 = item;
    std::getline(line_stream, item, ' ');
    file2 = item;

    if (name_idx.find(file1) == name_idx.end() ||
        name_idx.find(file2) == name_idx.end())
      continue;

    image_t index1 = name_idx[file1];
    image_t index2 = name_idx[file2];

    image_pair_t pair_id = colmap::ImagePairToPairId(index1, index2);

    if (view_graph.image_pairs.find(pair_id) == view_graph.image_pairs.end())
      continue;

    std::getline(line_stream, item, ' ');
    view_graph.image_pairs[pair_id].weight = std::stod(item);
    counter++;
  }
  LOG(INFO) << counter << " weights are used are loaded";
}

// TODO: now, we only store 1 single gravity per rig.
// for ease of implementation, we only store from the image with trivial frame
void ReadGravity(const std::string& gravity_path,
                 std::unordered_map<image_t, Image>& images) {
  std::unordered_map<std::string, image_t> name_idx;
  for (const auto& [image_id, image] : images) {
    name_idx[image.file_name] = image_id;
  }

  std::ifstream file(gravity_path);

  // Read in the file list
  std::string line, item;
  Eigen::Vector3d gravity;
  int counter = 0;
  while (std::getline(file, line)) {
    std::stringstream line_stream(line);

    // file_name
    std::string name;
    std::getline(line_stream, name, ' ');

    // Gravity
    for (int i = 0; i < 3; i++) {
      std::getline(line_stream, item, ' ');
      gravity[i] = std::stod(item);
    }

    // Check whether the image present
    auto ite = name_idx.find(name);
    if (ite != name_idx.end()) {
      counter++;
      if (images[ite->second].IsReferenceInFrame()) {
        images[ite->second].frame_ptr->gravity_info.SetGravity(gravity);
        Rigid3d& cam_from_world = images[ite->second].frame_ptr->RigFromWorld();
        // Set the rotation from the camera to the world
        // Make sure the initialization is aligned with the gravity
        cam_from_world.rotation = Eigen::Quaterniond(
            images[ite->second].frame_ptr->gravity_info.GetRAlign());
      }
    }
  }
  LOG(INFO) << counter << " images are loaded with gravity";
}

void WriteGlobalRotation(const std::string& file_path,
                         const std::unordered_map<image_t, Image>& images) {
  std::ofstream file(file_path);
  std::set<image_t> existing_images;
  for (const auto& [image_id, image] : images) {
    if (image.IsRegistered()) {
      existing_images.insert(image_id);
    }
  }
  for (const auto& image_id : existing_images) {
    const auto& image = images.at(image_id);
    if (!image.IsRegistered()) continue;
    file << image.file_name;
    Rigid3d cam_from_world = image.CamFromWorld();
    for (int i = 0; i < 4; i++) {
      file << " " << cam_from_world.rotation.coeffs()[(i + 3) % 4];
    }
    file << "\n";
  }
}

void WriteRelPose(const std::string& file_path,
                  const std::unordered_map<image_t, Image>& images,
                  const ViewGraph& view_graph) {
  std::ofstream file(file_path);

  // Sort the image pairs by image name
  std::map<std::string, image_pair_t> name_pair;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid) {
      const auto& image1 = images.at(image_pair.image_id1);
      const auto& image2 = images.at(image_pair.image_id2);
      name_pair[image1.file_name + " " + image2.file_name] = pair_id;
    }
  }

  // Write the image pairs
  for (const auto& [name, pair_id] : name_pair) {
    const auto image_pair = view_graph.image_pairs.at(pair_id);
    if (!image_pair.is_valid) continue;
    file << images.at(image_pair.image_id1).file_name << " "
         << images.at(image_pair.image_id2).file_name;
    for (int i = 0; i < 4; i++) {
      file << " " << image_pair.cam2_from_cam1.rotation.coeffs()[(i + 3) % 4];
    }
    for (int i = 0; i < 3; i++) {
      file << " " << image_pair.cam2_from_cam1.translation[i];
    }
    file << "\n";
  }

  LOG(INFO) << name_pair.size() << " relpose are written";
}
}  // namespace glomap
