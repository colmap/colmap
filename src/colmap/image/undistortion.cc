// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/image/undistortion.h"

#include "colmap/geometry/pose.h"
#include "colmap/image/warp.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <fstream>

namespace colmap {
namespace {

template <typename Derived>
void WriteMatrix(const Eigen::MatrixBase<Derived>& matrix,
                 std::ofstream* file) {
  typedef typename Eigen::MatrixBase<Derived>::Index index_t;
  for (index_t r = 0; r < matrix.rows(); ++r) {
    for (index_t c = 0; c < matrix.cols() - 1; ++c) {
      *file << matrix(r, c) << " ";
    }
    *file << matrix(r, matrix.cols() - 1) << std::endl;
  }
}

// Write projection matrix P = K * [R t] to file and prepend given header.
void WriteProjectionMatrix(const std::string& path,
                           const Camera& camera,
                           const Image& image,
                           const std::string& header) {
  THROW_CHECK(camera.model_id == PinholeCameraModel::model_id);

  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  Eigen::Matrix3d calib_matrix = Eigen::Matrix3d::Identity();
  calib_matrix(0, 0) = camera.FocalLengthX();
  calib_matrix(1, 1) = camera.FocalLengthY();
  calib_matrix(0, 2) = camera.PrincipalPointX();
  calib_matrix(1, 2) = camera.PrincipalPointY();

  const Eigen::Matrix3x4d img_from_world =
      calib_matrix * image.CamFromWorld().ToMatrix();

  if (!header.empty()) {
    file << header << std::endl;
  }

  WriteMatrix(img_from_world, &file);
}

void WriteCOLMAPCommands(const bool geometric,
                         const std::string& workspace_path,
                         const std::string& workspace_format,
                         const std::string& pmvs_option_name,
                         const std::string& output_prefix,
                         const std::string& indent,
                         std::ofstream* file) {
  if (geometric) {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\"
          << std::endl;
    *file << indent << "  --workspace_path " << workspace_path << " \\"
          << std::endl;
    *file << indent << "  --workspace_format " << workspace_format << " \\"
          << std::endl;
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\"
            << std::endl;
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\"
          << std::endl;
    *file << indent << "  --PatchMatchStereo.geom_consistency true"
          << std::endl;
  } else {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\"
          << std::endl;
    *file << indent << "  --workspace_path " << workspace_path << " \\"
          << std::endl;
    *file << indent << "  --workspace_format " << workspace_format << " \\"
          << std::endl;
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\"
            << std::endl;
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\"
          << std::endl;
    *file << indent << "  --PatchMatchStereo.geom_consistency false"
          << std::endl;
  }

  *file << indent << "$COLMAP_EXE_PATH/colmap stereo_fusion \\" << std::endl;
  *file << indent << "  --workspace_path " << workspace_path << " \\"
        << std::endl;
  *file << indent << "  --workspace_format " << workspace_format << " \\"
        << std::endl;
  if (workspace_format == "PMVS") {
    *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\"
          << std::endl;
  }
  if (geometric) {
    *file << indent << "  --input_type geometric \\" << std::endl;
  } else {
    *file << indent << "  --input_type photometric \\" << std::endl;
  }
  *file << indent << "  --output_path "
        << JoinPaths(workspace_path, output_prefix + "fused.ply") << std::endl;

  *file << indent << "$COLMAP_EXE_PATH/colmap poisson_mesher \\" << std::endl;
  *file << indent << "  --input_path "
        << JoinPaths(workspace_path, output_prefix + "fused.ply") << " \\"
        << std::endl;
  *file << indent << "  --output_path "
        << JoinPaths(workspace_path, output_prefix + "meshed-poisson.ply")
        << std::endl;

  *file << indent << "$COLMAP_EXE_PATH/colmap delaunay_mesher \\" << std::endl;
  *file << indent << "  --input_path "
        << JoinPaths(workspace_path, output_prefix) << " \\" << std::endl;
  *file << indent << "  --input_type dense \\" << std::endl;
  *file << indent << "  --output_path "
        << JoinPaths(workspace_path, output_prefix + "meshed-delaunay.ply")
        << std::endl;
}

}  // namespace

COLMAPUndistorter::COLMAPUndistorter(const UndistortCameraOptions& options,
                                     const Reconstruction& reconstruction,
                                     const std::string& image_path,
                                     const std::string& output_path,
                                     const int num_patch_match_src_images,
                                     const CopyType copy_type,
                                     const std::vector<image_t>& image_ids)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      copy_type_(copy_type),
      num_patch_match_src_images_(num_patch_match_src_images),
      reconstruction_(reconstruction),
      image_ids_(image_ids) {}

void COLMAPUndistorter::Run() {
  PrintHeading1("Image undistortion");
  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(JoinPaths(output_path_, "images"));
  CreateDirIfNotExists(JoinPaths(output_path_, "sparse"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo/depth_maps"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo/normal_maps"));
  CreateDirIfNotExists(JoinPaths(output_path_, "stereo/consistency_graphs"));
  reconstruction_.CreateImageDirs(JoinPaths(output_path_, "images"));
  reconstruction_.CreateImageDirs(JoinPaths(output_path_, "stereo/depth_maps"));
  reconstruction_.CreateImageDirs(
      JoinPaths(output_path_, "stereo/normal_maps"));
  reconstruction_.CreateImageDirs(
      JoinPaths(output_path_, "stereo/consistency_graphs"));

  ThreadPool thread_pool;
  std::vector<std::future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  if (image_ids_.empty()) {
    for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
      const image_t image_id = reconstruction_.RegImageIds().at(i);
      futures.push_back(
          thread_pool.AddTask(&COLMAPUndistorter::Undistort, this, image_id));
    }
  } else {
    for (const image_t image_id : image_ids_) {
      futures.push_back(
          thread_pool.AddTask(&COLMAPUndistorter::Undistort, this, image_id));
    }
  }

  // Only use the image names for the successfully undistorted images
  // when writing the MVS config files
  image_names_.clear();
  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    if (futures[i].get()) {
      if (image_ids_.empty()) {
        const image_t image_id = reconstruction_.RegImageIds().at(i);
        image_names_.push_back(reconstruction_.Image(image_id).Name());
      } else {
        image_names_.push_back(reconstruction_.Image(image_ids_[i]).Name());
      }
    }
  }

  LOG(INFO) << "Writing reconstruction...";
  Reconstruction undistorted_reconstruction = reconstruction_;
  UndistortReconstruction(options_, &undistorted_reconstruction);
  undistorted_reconstruction.Write(JoinPaths(output_path_, "sparse"));

  LOG(INFO) << "Writing configuration...";
  WritePatchMatchConfig();
  WriteFusionConfig();

  LOG(INFO) << "Writing scripts...";
  WriteScript(false);
  WriteScript(true);

  run_timer.PrintMinutes();
}

bool COLMAPUndistorter::Undistort(const image_t image_id) const {
  const Image& image = reconstruction_.Image(image_id);

  Bitmap distorted_bitmap;
  Bitmap undistorted_bitmap;
  const Camera& camera = reconstruction_.Camera(image.CameraId());
  Camera undistorted_camera;

  const std::string input_image_path = JoinPaths(image_path_, image.Name());
  const std::string output_image_path =
      JoinPaths(output_path_, "images", image.Name());

  // Check if the image is already undistorted and copy from source if no
  // scaling is needed
  if (camera.IsUndistorted() && options_.max_image_size < 0 &&
      ExistsFile(input_image_path)) {
    LOG(INFO) << "Undistorted image found; copying to location: "
              << output_image_path;
    FileCopy(input_image_path, output_image_path, copy_type_);
    return true;
  }

  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  UndistortImage(options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);
  return undistorted_bitmap.Write(output_image_path);
}

void COLMAPUndistorter::WritePatchMatchConfig() const {
  const auto path = JoinPaths(output_path_, "stereo/patch-match.cfg");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  for (const auto& image_name : image_names_) {
    file << image_name << std::endl;
    file << "__auto__, " << num_patch_match_src_images_ << std::endl;
  }
}

void COLMAPUndistorter::WriteFusionConfig() const {
  const auto path = JoinPaths(output_path_, "stereo/fusion.cfg");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  for (const auto& image_name : image_names_) {
    file << image_name << std::endl;
  }
}

void COLMAPUndistorter::WriteScript(const bool geometric) const {
  const std::string path = JoinPaths(
      output_path_,
      geometric ? "run-colmap-geometric.sh" : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to " << std::endl
       << "# the directory containing the COLMAP executables." << std::endl;
  WriteCOLMAPCommands(geometric, ".", "COLMAP", "option-all", "", "", &file);
}

PMVSUndistorter::PMVSUndistorter(const UndistortCameraOptions& options,
                                 const Reconstruction& reconstruction,
                                 const std::string& image_path,
                                 const std::string& output_path)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      reconstruction_(reconstruction) {}

void PMVSUndistorter::Run() {
  Timer run_timer;
  run_timer.Start();
  PrintHeading1("Image undistortion (CMVS/PMVS)");

  CreateDirIfNotExists(JoinPaths(output_path_, "pmvs"));
  CreateDirIfNotExists(JoinPaths(output_path_, "pmvs/txt"));
  CreateDirIfNotExists(JoinPaths(output_path_, "pmvs/visualize"));
  CreateDirIfNotExists(JoinPaths(output_path_, "pmvs/models"));

  ThreadPool thread_pool;
  std::vector<std::future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&PMVSUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      thread_pool.Stop();
      LOG(WARNING) << "Stopped the undistortion process. Image point "
                      "locations and camera parameters for not yet processed "
                      "images in the Bundler output file is probably wrong.";
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    futures[i].get();
  }

  LOG(INFO) << "Writing bundle file...";
  Reconstruction undistorted_reconstruction = reconstruction_;
  UndistortReconstruction(options_, &undistorted_reconstruction);
  const std::string bundle_path = JoinPaths(output_path_, "pmvs/bundle.rd.out");
  ExportBundler(
      undistorted_reconstruction, bundle_path, bundle_path + ".list.txt");

  LOG(INFO) << "Writing visibility file...";
  WriteVisibilityData();

  LOG(INFO) << "Writing option file...";
  WriteOptionFile();

  LOG(INFO) << "Writing scripts...";
  WritePMVSScript();
  WriteCMVSPMVSScript();
  WriteCOLMAPScript(false);
  WriteCOLMAPScript(true);
  WriteCMVSCOLMAPScript(false);
  WriteCMVSCOLMAPScript(true);

  run_timer.PrintMinutes();
}

bool PMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const std::string output_image_path = JoinPaths(
      output_path_, StringPrintf("pmvs/visualize/%08d.jpg", reg_image_idx));
  const std::string proj_matrix_path =
      JoinPaths(output_path_, StringPrintf("pmvs/txt/%08d.txt", reg_image_idx));

  const image_t image_id = reconstruction_.RegImageIds().at(reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  Bitmap distorted_bitmap;
  const std::string input_image_path = JoinPaths(image_path_, image.Name());
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
  return undistorted_bitmap.Write(output_image_path);
}

void PMVSUndistorter::WriteVisibilityData() const {
  const auto path = JoinPaths(output_path_, "pmvs/vis.dat");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "VISDATA" << std::endl;
  file << reconstruction_.NumRegImages() << std::endl;

  const std::vector<image_t>& reg_image_ids = reconstruction_.RegImageIds();

  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];
    const Image& image = reconstruction_.Image(image_id);
    std::unordered_set<image_t> visible_image_ids;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          if (track_el.image_id != image_id) {
            visible_image_ids.insert(track_el.image_id);
          }
        }
      }
    }

    std::vector<image_t> sorted_visible_image_ids(visible_image_ids.begin(),
                                                  visible_image_ids.end());
    std::sort(sorted_visible_image_ids.begin(), sorted_visible_image_ids.end());

    file << i << " " << visible_image_ids.size();
    for (const image_t visible_image_id : sorted_visible_image_ids) {
      file << " " << visible_image_id;
    }
    file << std::endl;
  }
}

void PMVSUndistorter::WritePMVSScript() const {
  const auto path = JoinPaths(output_path_, "run-pmvs.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to " << std::endl
       << "# the directory containing the CMVS-PMVS executables." << std::endl;
  file << "$PMVS_EXE_PATH/pmvs2 pmvs/ option-all" << std::endl;
}

void PMVSUndistorter::WriteCMVSPMVSScript() const {
  const auto path = JoinPaths(output_path_, "run-cmvs-pmvs.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to " << std::endl
       << "# the directory containing the CMVS-PMVS executables." << std::endl;
  file << "$PMVS_EXE_PATH/cmvs pmvs/" << std::endl;
  file << "$PMVS_EXE_PATH/genOption pmvs/" << std::endl;
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name"
       << std::endl;
  file << "do" << std::endl;
  file << "    option_name=$(basename \"$file_name\")" << std::endl;
  file << "    if [ \"$option_name\" = \"option-all\" ]; then" << std::endl;
  file << "        continue" << std::endl;
  file << "    fi" << std::endl;
  file << "    $PMVS_EXE_PATH/pmvs2 pmvs/ $option_name" << std::endl;
  file << "done" << std::endl;
}

void PMVSUndistorter::WriteCOLMAPScript(const bool geometric) const {
  const std::string path = JoinPaths(
      output_path_,
      geometric ? "run-colmap-geometric.sh" : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to " << std::endl
       << "# the directory containing the COLMAP executables." << std::endl;
  WriteCOLMAPCommands(
      geometric, "pmvs", "PMVS", "option-all", "option-all-", "", &file);
}

void PMVSUndistorter::WriteCMVSCOLMAPScript(const bool geometric) const {
  const std::string path =
      JoinPaths(output_path_,
                geometric ? "run-cmvs-colmap-geometric.sh"
                          : "run-cmvs-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to " << std::endl
       << "# the directory containing the CMVS-PMVS executables" << std::endl;
  file << "# and you must set $COLMAP_EXE_PATH to " << std::endl
       << "# the directory containing the COLMAP executables." << std::endl;
  file << "$PMVS_EXE_PATH/cmvs pmvs/" << std::endl;
  file << "$PMVS_EXE_PATH/genOption pmvs/" << std::endl;
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name"
       << std::endl;
  file << "do" << std::endl;
  file << "    workspace_path=$(dirname \"$file_name\")" << std::endl;
  file << "    option_name=$(basename \"$file_name\")" << std::endl;
  file << "    if [ \"$option_name\" = \"option-all\" ]; then" << std::endl;
  file << "        continue" << std::endl;
  file << "    fi" << std::endl;
  file << "    rm -rf \"$workspace_path/stereo\"" << std::endl;
  WriteCOLMAPCommands(geometric,
                      "pmvs",
                      "PMVS",
                      "$option_name",
                      "$option_name-",
                      "    ",
                      &file);
  file << "done" << std::endl;
}

void PMVSUndistorter::WriteOptionFile() const {
  const auto path = JoinPaths(output_path_, "pmvs/option-all");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# Generated by COLMAP - all images, no clustering." << std::endl;

  file << "level 1" << std::endl;
  file << "csize 2" << std::endl;
  file << "threshold 0.7" << std::endl;
  file << "wsize 7" << std::endl;
  file << "minImageNum 3" << std::endl;
  file << "CPU " << std::thread::hardware_concurrency() << std::endl;
  file << "setEdge 0" << std::endl;
  file << "useBound 0" << std::endl;
  file << "useVisData 1" << std::endl;
  file << "sequence -1" << std::endl;
  file << "maxAngle 10" << std::endl;
  file << "quad 2.0" << std::endl;

  file << "timages " << reconstruction_.NumRegImages();
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    file << " " << i;
  }
  file << std::endl;

  file << "oimages 0" << std::endl;
}

CMPMVSUndistorter::CMPMVSUndistorter(const UndistortCameraOptions& options,
                                     const Reconstruction& reconstruction,
                                     const std::string& image_path,
                                     const std::string& output_path)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      reconstruction_(reconstruction) {}

void CMPMVSUndistorter::Run() {
  Timer run_timer;
  run_timer.Start();
  PrintHeading1("Image undistortion (CMP-MVS)");

  ThreadPool thread_pool;
  std::vector<std::future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&CMPMVSUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    futures[i].get();
  }

  run_timer.PrintMinutes();
}

bool CMPMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const std::string output_image_path =
      JoinPaths(output_path_, StringPrintf("%05d.jpg", reg_image_idx + 1));
  const std::string proj_matrix_path =
      JoinPaths(output_path_, StringPrintf("%05d_P.txt", reg_image_idx + 1));

  const image_t image_id = reconstruction_.RegImageIds().at(reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  Bitmap distorted_bitmap;
  const std::string input_image_path = JoinPaths(image_path_, image.Name());
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
  return undistorted_bitmap.Write(output_image_path);
}

PureImageUndistorter::PureImageUndistorter(
    const UndistortCameraOptions& options,
    const std::string& image_path,
    const std::string& output_path,
    const std::vector<std::pair<std::string, Camera>>& image_names_and_cameras)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      image_names_and_cameras_(image_names_and_cameras) {}

void PureImageUndistorter::Run() {
  Timer run_timer;
  run_timer.Start();
  PrintHeading1("Image undistortion");

  CreateDirIfNotExists(output_path_);

  ThreadPool thread_pool;
  std::vector<std::future<bool>> futures;
  size_t num_images = image_names_and_cameras_.size();
  futures.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    futures.push_back(
        thread_pool.AddTask(&PureImageUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    futures[i].get();
  }

  run_timer.PrintMinutes();
}

bool PureImageUndistorter::Undistort(const size_t image_idx) const {
  const std::string& image_name = image_names_and_cameras_[image_idx].first;
  const Camera& camera = image_names_and_cameras_[image_idx].second;

  const std::string output_image_path = JoinPaths(output_path_, image_name);

  Bitmap distorted_bitmap;
  const std::string input_image_path = JoinPaths(image_path_, image_name);
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  return undistorted_bitmap.Write(output_image_path);
}

StereoImageRectifier::StereoImageRectifier(
    const UndistortCameraOptions& options,
    const Reconstruction& reconstruction,
    const std::string& image_path,
    const std::string& output_path,
    const std::vector<std::pair<image_t, image_t>>& stereo_pairs)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      stereo_pairs_(stereo_pairs),
      reconstruction_(reconstruction) {}

void StereoImageRectifier::Run() {
  PrintHeading1("Stereo rectification");
  Timer run_timer;
  run_timer.Start();

  ThreadPool thread_pool;
  std::vector<std::future<void>> futures;
  futures.reserve(stereo_pairs_.size());
  for (const auto& stereo_pair : stereo_pairs_) {
    futures.push_back(thread_pool.AddTask(&StereoImageRectifier::Rectify,
                                          this,
                                          stereo_pair.first,
                                          stereo_pair.second));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      break;
    }

    LOG(INFO) << StringPrintf(
        "Rectifying image pair [%d/%d]", i + 1, futures.size());

    futures[i].get();
  }

  run_timer.PrintMinutes();
}

void StereoImageRectifier::Rectify(const image_t image_id1,
                                   const image_t image_id2) const {
  const Image& image1 = reconstruction_.Image(image_id1);
  const Image& image2 = reconstruction_.Image(image_id2);
  const Camera& camera1 = reconstruction_.Camera(image1.CameraId());
  const Camera& camera2 = reconstruction_.Camera(image2.CameraId());

  const std::string image_name1 = StringReplace(image1.Name(), "/", "-");
  const std::string image_name2 = StringReplace(image2.Name(), "/", "-");

  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image_name1.c_str(), image_name2.c_str());

  CreateDirIfNotExists(JoinPaths(output_path_, stereo_pair_name));

  const std::string output_image1_path =
      JoinPaths(output_path_, stereo_pair_name, image_name1);
  const std::string output_image2_path =
      JoinPaths(output_path_, stereo_pair_name, image_name2);

  Bitmap distorted_bitmap1;
  const std::string input_image1_path = JoinPaths(image_path_, image1.Name());
  if (!distorted_bitmap1.Read(input_image1_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image1_path;
    return;
  }

  Bitmap distorted_bitmap2;
  const std::string input_image2_path = JoinPaths(image_path_, image2.Name());
  if (!distorted_bitmap2.Read(input_image2_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image2_path;
    return;
  }

  const Rigid3d cam2_from_cam1 =
      image2.CamFromWorld() * Inverse(image1.CamFromWorld());

  Bitmap undistorted_bitmap1;
  Bitmap undistorted_bitmap2;
  Camera undistorted_camera;
  Eigen::Matrix4d Q;
  RectifyAndUndistortStereoImages(options_,
                                  distorted_bitmap1,
                                  distorted_bitmap2,
                                  camera1,
                                  camera2,
                                  cam2_from_cam1,
                                  &undistorted_bitmap1,
                                  &undistorted_bitmap2,
                                  &undistorted_camera,
                                  &Q);

  undistorted_bitmap1.Write(output_image1_path);
  undistorted_bitmap2.Write(output_image2_path);

  const auto Q_path = JoinPaths(output_path_, stereo_pair_name, "Q.txt");
  std::ofstream Q_file(Q_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(Q_file, Q_path);
  WriteMatrix(Q, &Q_file);
}

Camera UndistortCamera(const UndistortCameraOptions& options,
                       const Camera& camera) {
  THROW_CHECK_GE(options.blank_pixels, 0);
  THROW_CHECK_LE(options.blank_pixels, 1);
  THROW_CHECK_GT(options.min_scale, 0.0);
  THROW_CHECK_LE(options.min_scale, options.max_scale);
  THROW_CHECK_NE(options.max_image_size, 0);
  THROW_CHECK_GE(options.roi_min_x, 0.0);
  THROW_CHECK_GE(options.roi_min_y, 0.0);
  THROW_CHECK_LE(options.roi_max_x, 1.0);
  THROW_CHECK_LE(options.roi_max_y, 1.0);
  THROW_CHECK_LT(options.roi_min_x, options.roi_max_x);
  THROW_CHECK_LT(options.roi_min_y, options.roi_max_y);

  Camera undistorted_camera;
  undistorted_camera.model_id = PinholeCameraModel::model_id;
  undistorted_camera.width = camera.width;
  undistorted_camera.height = camera.height;
  undistorted_camera.params.resize(PinholeCameraModel::num_params, 0);

  // Copy focal length parameters.
  const span<const size_t> focal_length_idxs = camera.FocalLengthIdxs();
  THROW_CHECK_LE(focal_length_idxs.size(), 2)
      << "Not more than two focal length parameters supported.";
  undistorted_camera.SetFocalLengthX(camera.FocalLengthX());
  undistorted_camera.SetFocalLengthY(camera.FocalLengthY());

  // Copy principal point parameters.
  undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX());
  undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY());

  // Modify undistorted camera parameters based on ROI if enabled
  size_t roi_min_x = 0;
  size_t roi_min_y = 0;
  size_t roi_max_x = camera.width;
  size_t roi_max_y = camera.height;

  const bool roi_enabled = options.roi_min_x > 0.0 || options.roi_min_y > 0.0 ||
                           options.roi_max_x < 1.0 || options.roi_max_y < 1.0;

  if (roi_enabled) {
    roi_min_x = static_cast<size_t>(
        std::round(options.roi_min_x * static_cast<double>(camera.width)));
    roi_min_y = static_cast<size_t>(
        std::round(options.roi_min_y * static_cast<double>(camera.height)));
    roi_max_x = static_cast<size_t>(
        std::round(options.roi_max_x * static_cast<double>(camera.width)));
    roi_max_y = static_cast<size_t>(
        std::round(options.roi_max_y * static_cast<double>(camera.height)));

    // Make sure that the roi is valid.
    roi_min_x = std::min(roi_min_x, camera.width - 1);
    roi_min_y = std::min(roi_min_y, camera.height - 1);
    roi_max_x = std::max(roi_max_x, roi_min_x + 1);
    roi_max_y = std::max(roi_max_y, roi_min_y + 1);

    undistorted_camera.width = roi_max_x - roi_min_x;
    undistorted_camera.height = roi_max_y - roi_min_y;

    undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX() -
                                          static_cast<double>(roi_min_x));
    undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY() -
                                          static_cast<double>(roi_min_y));
  }

  // Scale the image such the the boundary of the undistorted image.
  if (roi_enabled || (camera.model_id != SimplePinholeCameraModel::model_id &&
                      camera.model_id != PinholeCameraModel::model_id)) {
    // Determine min/max coordinates along top / bottom image border.

    double left_min_x = std::numeric_limits<double>::max();
    double left_max_x = std::numeric_limits<double>::lowest();
    double right_min_x = std::numeric_limits<double>::max();
    double right_max_x = std::numeric_limits<double>::lowest();

    for (size_t y = roi_min_y; y < roi_max_y; ++y) {
      // Left border.
      const Eigen::Vector2d point1_in_cam =
          camera.CamFromImg(Eigen::Vector2d(0.5, y + 0.5));
      const Eigen::Vector2d undistorted_point1 =
          undistorted_camera.ImgFromCam(point1_in_cam);
      left_min_x = std::min(left_min_x, undistorted_point1(0));
      left_max_x = std::max(left_max_x, undistorted_point1(0));
      // Right border.
      const Eigen::Vector2d point2_in_cam =
          camera.CamFromImg(Eigen::Vector2d(camera.width - 0.5, y + 0.5));
      const Eigen::Vector2d undistorted_point2 =
          undistorted_camera.ImgFromCam(point2_in_cam);
      right_min_x = std::min(right_min_x, undistorted_point2(0));
      right_max_x = std::max(right_max_x, undistorted_point2(0));
    }

    // Determine min, max coordinates along left / right image border.

    double top_min_y = std::numeric_limits<double>::max();
    double top_max_y = std::numeric_limits<double>::lowest();
    double bottom_min_y = std::numeric_limits<double>::max();
    double bottom_max_y = std::numeric_limits<double>::lowest();

    for (size_t x = roi_min_x; x < roi_max_x; ++x) {
      // Top border.
      const Eigen::Vector2d point1_in_cam =
          camera.CamFromImg(Eigen::Vector2d(x + 0.5, 0.5));
      const Eigen::Vector2d undistorted_point1 =
          undistorted_camera.ImgFromCam(point1_in_cam);
      top_min_y = std::min(top_min_y, undistorted_point1(1));
      top_max_y = std::max(top_max_y, undistorted_point1(1));
      // Bottom border.
      const Eigen::Vector2d point2_in_cam =
          camera.CamFromImg(Eigen::Vector2d(x + 0.5, camera.height - 0.5));
      const Eigen::Vector2d undistorted_point2 =
          undistorted_camera.ImgFromCam(point2_in_cam);
      bottom_min_y = std::min(bottom_min_y, undistorted_point2(1));
      bottom_max_y = std::max(bottom_max_y, undistorted_point2(1));
    }

    const double cx = undistorted_camera.PrincipalPointX();
    const double cy = undistorted_camera.PrincipalPointY();

    // Scale such that undistorted image contains all pixels of distorted image.
    const double min_scale_x =
        std::min(cx / (cx - left_min_x),
                 (undistorted_camera.width - 0.5 - cx) / (right_max_x - cx));
    const double min_scale_y =
        std::min(cy / (cy - top_min_y),
                 (undistorted_camera.height - 0.5 - cy) / (bottom_max_y - cy));

    // Scale such that there are no blank pixels in undistorted image.
    const double max_scale_x =
        std::max(cx / (cx - left_max_x),
                 (undistorted_camera.width - 0.5 - cx) / (right_min_x - cx));
    const double max_scale_y =
        std::max(cy / (cy - top_max_y),
                 (undistorted_camera.height - 0.5 - cy) / (bottom_min_y - cy));

    // Interpolate scale according to blank_pixels.
    double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
                            max_scale_x * (1.0 - options.blank_pixels));
    double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
                            max_scale_y * (1.0 - options.blank_pixels));

    // Clip the scaling factors.
    scale_x = Clamp(scale_x, options.min_scale, options.max_scale);
    scale_y = Clamp(scale_y, options.min_scale, options.max_scale);

    // Scale undistorted camera dimensions.
    const size_t orig_undistorted_camera_width = undistorted_camera.width;
    const size_t orig_undistorted_camera_height = undistorted_camera.height;
    undistorted_camera.width =
        static_cast<size_t>(std::max(1.0, scale_x * undistorted_camera.width));
    undistorted_camera.height =
        static_cast<size_t>(std::max(1.0, scale_y * undistorted_camera.height));

    // Scale the principal point according to the new dimensions of the camera.
    undistorted_camera.SetPrincipalPointX(
        undistorted_camera.PrincipalPointX() *
        static_cast<double>(undistorted_camera.width) /
        static_cast<double>(orig_undistorted_camera_width));
    undistorted_camera.SetPrincipalPointY(
        undistorted_camera.PrincipalPointY() *
        static_cast<double>(undistorted_camera.height) /
        static_cast<double>(orig_undistorted_camera_height));
  }

  if (options.max_image_size > 0) {
    const double max_image_scale_x =
        options.max_image_size / static_cast<double>(undistorted_camera.width);
    const double max_image_scale_y =
        options.max_image_size / static_cast<double>(undistorted_camera.height);
    const double max_image_scale =
        std::min(max_image_scale_x, max_image_scale_y);
    if (max_image_scale < 1.0) {
      undistorted_camera.Rescale(max_image_scale);
    }
  }

  return undistorted_camera;
}

void UndistortImage(const UndistortCameraOptions& options,
                    const Bitmap& distorted_bitmap,
                    const Camera& distorted_camera,
                    Bitmap* undistorted_bitmap,
                    Camera* undistorted_camera) {
  THROW_CHECK_EQ(distorted_camera.width, distorted_bitmap.Width());
  THROW_CHECK_EQ(distorted_camera.height, distorted_bitmap.Height());

  *undistorted_camera = UndistortCamera(options, distorted_camera);

  WarpImageBetweenCameras(distorted_camera,
                          *undistorted_camera,
                          distorted_bitmap,
                          undistorted_bitmap);

  distorted_bitmap.CloneMetadata(undistorted_bitmap);
}

void UndistortReconstruction(const UndistortCameraOptions& options,
                             Reconstruction* reconstruction) {
  const auto distorted_cameras = reconstruction->Cameras();
  for (const auto& camera : distorted_cameras) {
    if (camera.second.IsUndistorted()) {
      continue;
    }
    reconstruction->Camera(camera.first) =
        UndistortCamera(options, camera.second);
  }

  for (const auto& distorted_image : reconstruction->Images()) {
    auto& image = reconstruction->Image(distorted_image.first);
    const auto& distorted_camera = distorted_cameras.at(image.CameraId());
    const auto& undistorted_camera = reconstruction->Camera(image.CameraId());
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      auto& point2D = image.Point2D(point2D_idx);
      point2D.xy = undistorted_camera.ImgFromCam(
          distorted_camera.CamFromImg(point2D.xy));
    }
  }
}

void RectifyStereoCameras(const Camera& camera1,
                          const Camera& camera2,
                          const Rigid3d& cam2_from_cam1,
                          Eigen::Matrix3d* H1,
                          Eigen::Matrix3d* H2,
                          Eigen::Matrix4d* Q) {
  THROW_CHECK(camera1.model_id == SimplePinholeCameraModel::model_id ||
              camera1.model_id == PinholeCameraModel::model_id);
  THROW_CHECK(camera2.model_id == SimplePinholeCameraModel::model_id ||
              camera2.model_id == PinholeCameraModel::model_id);

  // Compute the average rotation between the first and the second camera.
  Eigen::AngleAxisd half_cam2_from_cam1(cam2_from_cam1.rotation);
  half_cam2_from_cam1.angle() *= -0.5;

  Eigen::Matrix3d R2 = half_cam2_from_cam1.toRotationMatrix();
  Eigen::Matrix3d R1 = R2.transpose();

  // Determine the translation, such that it coincides with the X-axis.
  Eigen::Vector3d t = R2 * cam2_from_cam1.translation;

  Eigen::Vector3d x_unit_vector(1, 0, 0);
  if (t.transpose() * x_unit_vector < 0) {
    x_unit_vector *= -1;
  }

  const Eigen::Vector3d rotation_axis = t.cross(x_unit_vector);

  Eigen::Matrix3d R_x;
  if (rotation_axis.norm() < std::numeric_limits<double>::epsilon()) {
    R_x = Eigen::Matrix3d::Identity();
  } else {
    const double angle = std::acos(std::abs(t.transpose() * x_unit_vector) /
                                   (t.norm() * x_unit_vector.norm()));
    R_x = Eigen::AngleAxisd(angle, rotation_axis.normalized());
  }

  // Apply the X-axis correction.
  R1 = R_x * R1;
  R2 = R_x * R2;
  t = R_x * t;

  // Determine the intrinsic calibration matrix.
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = std::min(camera1.MeanFocalLength(), camera2.MeanFocalLength());
  K(1, 1) = K(0, 0);
  K(0, 2) = camera1.PrincipalPointX();
  K(1, 2) = (camera1.PrincipalPointY() + camera2.PrincipalPointY()) / 2;

  // Compose the homographies.
  *H1 = K * R1 * camera1.CalibrationMatrix().inverse();
  *H2 = K * R2 * camera2.CalibrationMatrix().inverse();

  // Determine the inverse projection matrix that transforms disparity values
  // to 3D world coordinates: [x, y, disparity, 1] * Q = [X, Y, Z, 1] * w.
  *Q = Eigen::Matrix4d::Identity();
  (*Q)(3, 0) = -K(1, 2);
  (*Q)(3, 1) = -K(0, 2);
  (*Q)(3, 2) = K(0, 0);
  (*Q)(2, 3) = -1 / t(0);
  (*Q)(3, 3) = 0;
}

void RectifyAndUndistortStereoImages(const UndistortCameraOptions& options,
                                     const Bitmap& distorted_image1,
                                     const Bitmap& distorted_image2,
                                     const Camera& distorted_camera1,
                                     const Camera& distorted_camera2,
                                     const Rigid3d& cam2_from_cam1,
                                     Bitmap* undistorted_image1,
                                     Bitmap* undistorted_image2,
                                     Camera* undistorted_camera,
                                     Eigen::Matrix4d* Q) {
  THROW_CHECK_EQ(distorted_camera1.width, distorted_image1.Width());
  THROW_CHECK_EQ(distorted_camera1.height, distorted_image1.Height());
  THROW_CHECK_EQ(distorted_camera2.width, distorted_image2.Width());
  THROW_CHECK_EQ(distorted_camera2.height, distorted_image2.Height());

  *undistorted_camera = UndistortCamera(options, distorted_camera1);
  undistorted_image1->Allocate(static_cast<int>(undistorted_camera->width),
                               static_cast<int>(undistorted_camera->height),
                               distorted_image1.IsRGB());
  distorted_image1.CloneMetadata(undistorted_image1);

  undistorted_image2->Allocate(static_cast<int>(undistorted_camera->width),
                               static_cast<int>(undistorted_camera->height),
                               distorted_image2.IsRGB());
  distorted_image2.CloneMetadata(undistorted_image2);

  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  RectifyStereoCameras(
      *undistorted_camera, *undistorted_camera, cam2_from_cam1, &H1, &H2, Q);

  WarpImageWithHomographyBetweenCameras(H1.inverse(),
                                        distorted_camera1,
                                        *undistorted_camera,
                                        distorted_image1,
                                        undistorted_image1);
  WarpImageWithHomographyBetweenCameras(H2.inverse(),
                                        distorted_camera2,
                                        *undistorted_camera,
                                        distorted_image2,
                                        undistorted_image2);
}

}  // namespace colmap
