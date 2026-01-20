// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/image/undistortion.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/file.h"

namespace colmap {

// Undistort images and export undistorted cameras, as required by the
// mvs::PatchMatchController class.
class COLMAPUndistorter : public BaseController {
 public:
  struct Options {
    // The copy type to use when copying already undistorted images to the
    // output directory. This can be used to speed up the undistortion process
    // when a majority of the images are already undistorted and choosing
    // COPY_HARD_LINK or COPY_SYMLINK to avoid duplicating the images.
    FileCopyType copy_type = FileCopyType::COPY;

    // How many images to use as patch match source images when generating the
    // patch match config file.
    int num_patch_match_src_images = 20;

    // List of images to undistort. If empty, all images are undistorted.
    std::vector<image_t> image_ids;

    // JPEG quality setting in the range [0, 100]. A value of -1 uses the
    // default (quality 100). Lower values produce smaller file sizes.
    int jpeg_quality = -1;
  };

  COLMAPUndistorter(Options options,
                    const UndistortCameraOptions& camera_options,
                    const Reconstruction& reconstruction,
                    const std::filesystem::path& image_path,
                    const std::filesystem::path& output_path);

  void Run();

 private:
  bool Undistort(image_t image_id) const;
  void WritePatchMatchConfig(const std::vector<std::string>& image_names) const;
  void WriteFusionConfig(const std::vector<std::string>& image_names) const;
  void WriteScript(bool geometric) const;

  const Options options_;
  const UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  const std::filesystem::path image_path_;
  const std::filesystem::path output_path_;
};

// Undistort images and prepare data for CMVS/PMVS.
class PMVSUndistorter : public BaseController {
 public:
  PMVSUndistorter(const UndistortCameraOptions& camera_options,
                  const Reconstruction& reconstruction,
                  const std::filesystem::path& image_path,
                  const std::filesystem::path& output_path);

  void Run();

 private:
  bool Undistort(size_t reg_image_idx) const;
  void WriteVisibilityData() const;
  void WriteOptionFile() const;
  void WritePMVSScript() const;
  void WriteCMVSPMVSScript() const;
  void WriteCOLMAPScript(bool geometric) const;
  void WriteCMVSCOLMAPScript(bool geometric) const;

  const UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  const std::filesystem::path image_path_;
  const std::filesystem::path output_path_;
};

// Undistort images and prepare data for CMP-MVS.
class CMPMVSUndistorter : public BaseController {
 public:
  CMPMVSUndistorter(const UndistortCameraOptions& options,
                    const Reconstruction& reconstruction,
                    const std::filesystem::path& image_path,
                    const std::filesystem::path& output_path);

  void Run();

 private:
  bool Undistort(size_t reg_image_idx) const;

  const UndistortCameraOptions options_;
  const std::filesystem::path image_path_;
  const std::filesystem::path output_path_;
  const Reconstruction& reconstruction_;
};

// Undistort images and export undistorted cameras without the need for a
// reconstruction. Instead, the image names and camera model information are
// read from a text file.
class StandaloneImageUndistorter : public BaseController {
 public:
  struct Options {
    // The images and cameras to undistort.
    std::vector<std::pair<std::string, Camera>> image_names_and_cameras;

    // JPEG quality setting in the range [0, 100]. A value of -1 uses the
    // default (quality 100). Lower values produce smaller file sizes.
    int jpeg_quality = -1;
  };

  StandaloneImageUndistorter(Options options,
                             const UndistortCameraOptions& camera_options,
                             const std::filesystem::path& image_path,
                             const std::filesystem::path& output_path);

  void Run();

 private:
  bool Undistort(size_t image_idx) const;

  const Options options_;
  const UndistortCameraOptions camera_options_;
  const std::filesystem::path image_path_;
  const std::filesystem::path output_path_;
};

// Rectify stereo image pairs.
class StereoImageRectifier : public BaseController {
 public:
  struct Options {
    // The stereo image pairs to rectify.
    std::vector<std::pair<image_t, image_t>> stereo_pairs;

    // JPEG quality setting in the range [0, 100]. A value of -1 uses the
    // default (quality 100). Lower values produce smaller file sizes.
    int jpeg_quality = -1;
  };

  StereoImageRectifier(Options options,
                       const UndistortCameraOptions& camera_options,
                       const Reconstruction& reconstruction,
                       const std::filesystem::path& image_path,
                       const std::filesystem::path& output_path);

  void Run();

 private:
  void Rectify(image_t image_id1, image_t image_id2) const;

  const Options options_;
  const UndistortCameraOptions camera_options_;
  const Reconstruction& reconstruction_;
  const std::filesystem::path image_path_;
  const std::filesystem::path output_path_;
};

}  // namespace colmap
