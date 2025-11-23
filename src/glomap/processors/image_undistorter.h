#pragma once

#include "glomap/scene/types_sfm.h"

namespace glomap {

void UndistortImages(std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points = true);

}  // namespace glomap
