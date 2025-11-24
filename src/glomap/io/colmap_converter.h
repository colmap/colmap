#pragma once
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/scene/reconstruction.h"

#include "glomap/scene/types_sfm.h"

namespace glomap {

void ConvertGlomapToColmapImage(const Image& image,
                                colmap::Image& image_colmap,
                                bool keep_points = false);

void ConvertGlomapToColmap(
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks,
    colmap::Reconstruction& reconstruction,
    int cluster_id = -1,
    bool include_image_points = false);

void ConvertColmapToGlomap(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks);

void ConvertColmapPoints3DToGlomapTracks(
    const colmap::Reconstruction& reconstruction,
    std::unordered_map<track_t, Track>& tracks);

void ConvertDatabaseToGlomap(
    const colmap::Database& database,
    ViewGraph& view_graph,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images);

void CreateOneRigPerCamera(
    const std::unordered_map<camera_t, colmap::Camera>& cameras,
    std::unordered_map<rig_t, Rig>& rigs);

void CreateFrameForImage(const Rigid3d& cam_from_world,
                         Image& image,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<frame_t, Frame>& frames,
                         rig_t rig_id = -1,
                         frame_t frame_id = -1);

}  // namespace glomap
