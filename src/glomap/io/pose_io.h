#pragma once

#include "colmap/geometry/pose_prior.h"

#include "glomap/scene/types.h"

#include <unordered_map>

#include <Eigen/Geometry>

namespace glomap {

// I/O functions for the rotation averaging CLI.

// Format: IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
std::unordered_map<image_t, std::string> ReadImageNames(
    const std::string& file_path);
std::unordered_map<image_pair_t, Rigid3d> ReadRelativePoses(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names);
void WriteRelativePoses(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_pair_t, Rigid3d>& relative_poses);

// Format: IMAGE_NAME_1 IMAGE_NAME_2 WEIGHT
std::unordered_map<image_pair_t, double> ReadImagePairWeights(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names);
void WriteImagePairWeights(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_pair_t, double>& weights);

// Format: IMAGE_NAME GX GY GZ
// Gravity is the direction of [0,1,0] in the image frame.
std::vector<colmap::PosePrior> ReadGravityPriors(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names);
void WriteGravityPriors(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::vector<colmap::PosePrior>& pose_priors);

// Format: IMAGE_NAME QW QX QY QZ
std::unordered_map<image_t, Eigen::Quaterniond> ReadRotations(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names);
void WriteRotations(
    const std::string& file_path,
    const std::unordered_map<image_t, std::string>& image_names,
    const std::unordered_map<image_t, Eigen::Quaterniond>& rotations);

}  // namespace glomap
