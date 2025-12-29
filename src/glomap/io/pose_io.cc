#include "glomap/io/pose_io.h"

#include "colmap/geometry/pose.h"

#include <fstream>
#include <map>
#include <set>

namespace glomap {
namespace {

std::unordered_map<std::string, image_t> ExtractImageNameToId(
    const std::unordered_map<image_t, Image>& images) {
  std::unordered_map<std::string, image_t> image_name_to_id;
  for (const auto& [image_id, image] : images) {
    image_name_to_id[image.Name()] = image_id;
  }
  return image_name_to_id;
}

}  // namespace

void ReadRelPose(const std::string& file_path,
                 std::unordered_map<image_t, Image>& images,
                 ViewGraph& view_graph) {
  std::unordered_map<std::string, image_t> image_name_to_id =
      ExtractImageNameToId(images);

  image_t max_image_id = 0;
  camera_t max_camera_id = 0;
  for (const auto& [image_id, image] : images) {
    max_image_id = std::max(max_image_id, image_id);
    max_camera_id = std::max(max_camera_id, image.CameraId());
  }

  // Mark every edge in the view graph as invalid
  for (const auto& [pair_id, image_pair] : view_graph.ImagePairs()) {
    view_graph.SetInvalidImagePair(pair_id);
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

    if (image_name_to_id.find(file1) == image_name_to_id.end()) {
      max_image_id += 1;
      max_camera_id += 1;
      Image image1;
      image1.SetImageId(max_image_id);
      image1.SetCameraId(max_camera_id);
      image1.SetName(file1);
      images.insert(std::make_pair(max_image_id, std::move(image1)));
      image_name_to_id[file1] = max_image_id;
    }
    if (image_name_to_id.find(file2) == image_name_to_id.end()) {
      max_image_id += 1;
      max_camera_id += 1;
      Image image2;
      image2.SetImageId(max_image_id);
      image2.SetCameraId(max_camera_id);
      image2.SetName(file2);
      images.insert(std::make_pair(max_image_id, std::move(image2)));
      image_name_to_id[file2] = max_image_id;
    }

    const image_t index1 = image_name_to_id[file1];
    const image_t index2 = image_name_to_id[file2];

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

    if (!view_graph.HasImagePair(index1, index2)) {
      ImagePair image_pair(pose_rel);
      image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
      view_graph.AddImagePair(index1, index2, std::move(image_pair));
    } else {
      auto [image_pair, swapped] = view_graph.ImagePair(index1, index2);
      image_pair.cam2_from_cam1 = swapped ? Inverse(pose_rel) : pose_rel;
      image_pair.config = colmap::TwoViewGeometry::CALIBRATED;
      view_graph.SetValidImagePair(colmap::ImagePairToPairId(index1, index2));
    }
    counter++;
  }
  LOG(INFO) << counter << " relative poses are loaded";
}

// TODO: now, we only store 1 single gravity per rig.
// for ease of implementation, we only store from the image with trivial frame
std::vector<colmap::PosePrior> ReadGravity(
    const std::string& gravity_path,
    const std::unordered_map<image_t, Image>& images) {
  const std::unordered_map<std::string, image_t> image_name_to_id =
      ExtractImageNameToId(images);

  std::vector<colmap::PosePrior> pose_priors;

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
    auto ite = image_name_to_id.find(name);
    if (ite != image_name_to_id.end()) {
      const auto& image = images.at(ite->second);
      if (image.IsRefInFrame()) {
        counter++;
        auto& pose_prior = pose_priors.emplace_back();
        pose_prior.pose_prior_id = ite->second;
        pose_prior.corr_data_id = image.DataId();
        pose_prior.gravity = gravity;
      } else {
        LOG(INFO) << "Ignoring gravity of image " << name
                  << " because it is not from the reference sensor";
      }
    }
  }

  LOG(INFO) << counter << " images are loaded with gravity";

  return pose_priors;
}

void WriteGlobalRotation(const std::string& file_path,
                         const std::unordered_map<image_t, Image>& images) {
  std::ofstream file(file_path);
  std::set<image_t> existing_images;
  for (const auto& [image_id, image] : images) {
    if (image.HasPose()) {
      existing_images.insert(image_id);
    }
  }
  for (const auto& image_id : existing_images) {
    const auto& image = images.at(image_id);
    if (!image.HasPose()) continue;
    file << image.Name();
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
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const auto& image1 = images.at(image_id1);
    const auto& image2 = images.at(image_id2);
    name_pair[image1.Name() + " " + image2.Name()] = pair_id;
  }

  // Write the image pairs
  for (const auto& [name, pair_id] : name_pair) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const ImagePair& image_pair =
        view_graph.ImagePair(image_id1, image_id2).first;
    file << images.at(image_id1).Name() << " " << images.at(image_id2).Name();
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
