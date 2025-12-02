#pragma once

#include "glomap/scene/types_sfm.h"

#include <unordered_map>

namespace glomap {
// Required data structures
// IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
void ReadRelPose(const std::string& file_path,
                 std::unordered_map<image_t, Image>& images,
                 ViewGraph& view_graph);

// Required data structures
// IMAGE_NAME_1 IMAGE_NAME_2 weight
void ReadRelWeight(const std::string& file_path,
                   const std::unordered_map<image_t, Image>& images,
                   ViewGraph& view_graph);

// Require the gravity in the format:
// IMAGE_NAME GX GY GZ
// Gravity should be the direction of [0,1,0] in the image frame
// image.cam_from_world * [0,1,0]^T = g
void ReadGravity(const std::string& gravity_path,
                 std::unordered_map<image_t, Image>& images);

// Output would be of the format:
// IMAGE_NAME QW QX QY QZ
void WriteGlobalRotation(const std::string& file_path,
                         const std::unordered_map<image_t, Image>& images);

// Output would be of the format:
// IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ
void WriteRelPose(const std::string& file_path,
                  const std::unordered_map<image_t, Image>& images,
                  const ViewGraph& view_graph);
}  // namespace glomap