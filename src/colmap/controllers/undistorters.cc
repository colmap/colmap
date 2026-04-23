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

#include "colmap/controllers/undistorters.h"

#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <fstream>
#include <thread>

namespace colmap {
namespace {

void MaybeSetJpegQuality(const std::filesystem::path& path,
                         Bitmap& bitmap,
                         int jpeg_quality) {
  if ((HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg")) &&
      jpeg_quality > 0) {
    bitmap.SetMetaData("Compression", "jpeg:" + std::to_string(jpeg_quality));
  }
}

template <typename Derived>
void WriteMatrix(const Eigen::MatrixBase<Derived>& matrix,
                 std::ofstream* file) {
  using index_t = typename Eigen::MatrixBase<Derived>::Index;
  for (index_t r = 0; r < matrix.rows(); ++r) {
    for (index_t c = 0; c < matrix.cols() - 1; ++c) {
      *file << matrix(r, c) << " ";
    }
    *file << matrix(r, matrix.cols() - 1) << '\n';
  }
}

// Write projection matrix P = K * [R t] to file and prepend given header.
void WriteProjectionMatrix(const std::filesystem::path& path,
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
    file << header << '\n';
  }

  WriteMatrix(img_from_world, &file);
}

void WriteCOLMAPCommands(const bool geometric,
                         const std::filesystem::path& workspace_path,
                         const std::string& workspace_format,
                         const std::string& pmvs_option_name,
                         const std::string& output_prefix,
                         const std::string& indent,
                         std::ofstream* file) {
  if (geometric) {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n";
    *file << indent << "  --workspace_path " << workspace_path << " \\\n";
    *file << indent << "  --workspace_format " << workspace_format << " \\\n";
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\\n";
    *file << indent << "  --PatchMatchStereo.geom_consistency true\n";
  } else {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n";
    *file << indent << "  --workspace_path " << workspace_path << " \\\n";
    *file << indent << "  --workspace_format " << workspace_format << " \\\n";
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\\n";
    *file << indent << "  --PatchMatchStereo.geom_consistency false\n";
  }

  *file << indent << "$COLMAP_EXE_PATH/colmap stereo_fusion \\\n";
  *file << indent << "  --workspace_path " << workspace_path << " \\\n";
  *file << indent << "  --workspace_format " << workspace_format << " \\\n";
  if (workspace_format == "PMVS") {
    *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
  }
  if (geometric) {
    *file << indent << "  --input_type geometric \\\n";
  } else {
    *file << indent << "  --input_type photometric \\\n";
  }
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "fused.ply") << " \\\n";

  *file << indent << "$COLMAP_EXE_PATH/colmap poisson_mesher \\\n";
  *file << indent << "  --input_path "
        << workspace_path / (output_prefix + "fused.ply") << " \\\n";
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "meshed-poisson.ply") << " \\\n";

  *file << indent << "$COLMAP_EXE_PATH/colmap delaunay_mesher \\\n";
  *file << indent << "  --input_path " << workspace_path / output_prefix
        << " \\\n";
  *file << indent << "  --input_type dense \\\n";
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "meshed-delaunay.ply") << " \\\n";
}

}  // namespace

COLMAPUndistorter::COLMAPUndistorter(
    Options options,
    const UndistortCameraOptions& camera_options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.num_patch_match_src_images, 1);
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void COLMAPUndistorter::Run() {
  LOG_HEADING1("Image undistortion");

  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(output_path_ / "images");
  CreateDirIfNotExists(output_path_ / "sparse");
  CreateDirIfNotExists(output_path_ / "stereo");
  CreateDirIfNotExists(output_path_ / "stereo" / "depth_maps");
  CreateDirIfNotExists(output_path_ / "stereo" / "normal_maps");
  CreateDirIfNotExists(output_path_ / "stereo" / "consistency_graphs");
  reconstruction_.CreateImageDirs(output_path_ / "images");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" / "depth_maps");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" / "normal_maps");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" /
                                  "consistency_graphs");

  const std::vector<image_t> image_ids = options_.image_ids.empty()
                                             ? reconstruction_.RegImageIds()
                                             : options_.image_ids;
  const size_t num_images = image_ids.size();

  ThreadPool thread_pool;
  std::vector<std::shared_future<bool>> futures;
  futures.reserve(num_images);
  for (const image_t image_id : image_ids) {
    futures.push_back(
        thread_pool.AddTask(&COLMAPUndistorter::Undistort, this, image_id));
  }

  // Only use the image names for the successfully undistorted images
  // when writing the MVS config files
  std::vector<std::string> image_names;
  image_names.reserve(num_images);
  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    if (futures[i].get()) {
      image_names.push_back(reconstruction_.Image(image_ids[i]).Name());
    }
  }

  LOG(INFO) << "Writing reconstruction...";
  Reconstruction undistorted_reconstruction = reconstruction_;
  UndistortReconstruction(camera_options_, &undistorted_reconstruction);
  undistorted_reconstruction.Write(output_path_ / "sparse");

  LOG(INFO) << "Writing configuration...";
  WritePatchMatchConfig(image_names);
  WriteFusionConfig(image_names);

  LOG(INFO) << "Writing scripts...";
  WriteScript(/*geometric=*/false);
  WriteScript(/*geometric=*/true);

  run_timer.PrintMinutes();
}

bool COLMAPUndistorter::Undistort(const image_t image_id) const {
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  const auto input_image_path = image_path_ / image.Name();
  const auto output_image_path = output_path_ / "images" / image.Name();

  // Check if the image is already undistorted and copy from source if no
  // scaling is needed
  if (camera.IsUndistorted() && camera_options_.max_image_size < 0 &&
      ExistsFile(input_image_path)) {
    LOG(INFO) << "Copying already distorted image to location: "
              << output_image_path;
    FileCopy(input_image_path, output_image_path, options_.copy_type);
    return true;
  }

  Bitmap distorted_bitmap;
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path: " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);

  return undistorted_bitmap.Write(output_image_path);
}

void COLMAPUndistorter::WritePatchMatchConfig(
    const std::vector<std::string>& image_names) const {
  const auto path = output_path_ / "stereo" / "patch-match.cfg";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  for (const auto& image_name : image_names) {
    file << image_name << '\n';

    file << "__auto__, " << options_.num_patch_match_src_images << '\n';
  }
}

void COLMAPUndistorter::WriteFusionConfig(
    const std::vector<std::string>& image_names) const {
  const auto path = output_path_ / "stereo" / "fusion.cfg";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  for (const auto& image_name : image_names) {
    file << image_name << '\n';
  }
}

void COLMAPUndistorter::WriteScript(const bool geometric) const {
  const auto path = output_path_ / (geometric ? "run-colmap-geometric.sh"
                                              : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  WriteCOLMAPCommands(geometric, ".", "COLMAP", "option-all", "", "", &file);
}

PMVSUndistorter::PMVSUndistorter(const UndistortCameraOptions& camera_options,
                                 const Reconstruction& reconstruction,
                                 const std::filesystem::path& image_path,
                                 const std::filesystem::path& output_path)
    : camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {}

void PMVSUndistorter::Run() {
  LOG_HEADING1("Image undistortion (CMVS/PMVS)");

  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(output_path_ / "pmvs");
  CreateDirIfNotExists(output_path_ / "pmvs" / "txt");
  CreateDirIfNotExists(output_path_ / "pmvs" / "visualize");
  CreateDirIfNotExists(output_path_ / "pmvs" / "models");

  ThreadPool thread_pool;
  std::vector<std::shared_future<bool>> futures;
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
  UndistortReconstruction(camera_options_, &undistorted_reconstruction);
  const auto bundle_path = output_path_ / "pmvs" / "bundle.rd.out";
  ExportBundler(undistorted_reconstruction,
                bundle_path,
                AddFileExtension(bundle_path, ".list.txt"));

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
  const auto output_image_path =
      output_path_ / StringPrintf("pmvs/visualize/%08d.jpg", reg_image_idx);
  const auto proj_matrix_path =
      output_path_ / StringPrintf("pmvs/txt/%08d.txt", reg_image_idx);

  const image_t image_id =
      *std::next(reconstruction_.RegImageIds().begin(), reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  Bitmap distorted_bitmap;
  const auto input_image_path = image_path_ / image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
  return undistorted_bitmap.Write(output_image_path);
}

void PMVSUndistorter::WriteVisibilityData() const {
  const auto path = output_path_ / "pmvs" / "vis.dat";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "VISDATA\n";
  file << reconstruction_.NumRegImages() << '\n';

  size_t image_idx = 0;
  for (const image_t image_id : reconstruction_.RegImageIds()) {
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

    file << image_idx++ << " " << visible_image_ids.size();
    for (const image_t visible_image_id : sorted_visible_image_ids) {
      file << " " << visible_image_id;
    }
    file << '\n';
  }
}

void PMVSUndistorter::WritePMVSScript() const {
  const auto path = output_path_ / "run-pmvs.sh";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables.\n";
  file << "$PMVS_EXE_PATH/pmvs2 pmvs/ option-all\n";
}

void PMVSUndistorter::WriteCMVSPMVSScript() const {
  const auto path = output_path_ / "run-cmvs-pmvs.sh";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables.\n";
  file << "$PMVS_EXE_PATH/cmvs pmvs/\n";
  file << "$PMVS_EXE_PATH/genOption pmvs/\n";
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name\n";
  file << "do\n";
  file << "    option_name=$(basename \"$file_name\")\n";
  file << "    if [ \"$option_name\" = \"option-all\" ]; then\n";
  file << "        continue\n";
  file << "    fi\n";
  file << "    $PMVS_EXE_PATH/pmvs2 pmvs/ $option_name\n";
  file << "done\n";
}

void PMVSUndistorter::WriteCOLMAPScript(const bool geometric) const {
  const auto path = output_path_ / (geometric ? "run-colmap-geometric.sh"
                                              : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  WriteCOLMAPCommands(
      geometric, "pmvs", "PMVS", "option-all", "option-all-", "", &file);
}

void PMVSUndistorter::WriteCMVSCOLMAPScript(const bool geometric) const {
  const auto path =
      output_path_ / (geometric ? "run-cmvs-colmap-geometric.sh"
                                : "run-cmvs-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables\n";
  file << "# and you must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  file << "$PMVS_EXE_PATH/cmvs pmvs/\n";
  file << "$PMVS_EXE_PATH/genOption pmvs/\n";
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name\n";
  file << "do\n";
  file << "    workspace_path=$(dirname \"$file_name\")\n";
  file << "    option_name=$(basename \"$file_name\")\n";
  file << "    if [ \"$option_name\" = \"option-all\" ]; then\n";
  file << "        continue\n";
  file << "    fi\n";
  file << "    rm -rf \"$workspace_path/stereo\"\n";
  WriteCOLMAPCommands(geometric,
                      "pmvs",
                      "PMVS",
                      "$option_name",
                      "$option_name-",
                      "    ",
                      &file);
  file << "done\n";
}

void PMVSUndistorter::WriteOptionFile() const {
  const auto path = output_path_ / "pmvs" / "option-all";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# Generated by COLMAP - all images, no clustering.\n";

  file << "level 1\n";
  file << "csize 2\n";
  file << "threshold 0.7\n";
  file << "wsize 7\n";
  file << "minImageNum 3\n";
  file << "CPU " << std::thread::hardware_concurrency() << '\n';
  file << "setEdge 0\n";
  file << "useBound 0\n";
  file << "useVisData 1\n";
  file << "sequence -1\n";
  file << "maxAngle 10\n";
  file << "quad 2.0\n";

  file << "timages " << reconstruction_.NumRegImages();
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    file << " " << i;
  }
  file << '\n';

  file << "oimages 0\n";
}

CMPMVSUndistorter::CMPMVSUndistorter(const UndistortCameraOptions& options,
                                     const Reconstruction& reconstruction,
                                     const std::filesystem::path& image_path,
                                     const std::filesystem::path& output_path)
    : options_(options),
      image_path_(image_path),
      output_path_(output_path),
      reconstruction_(reconstruction) {}

void CMPMVSUndistorter::Run() {
  LOG_HEADING1("Image undistortion (CMP-MVS)");

  Timer run_timer;
  run_timer.Start();

  ThreadPool thread_pool;
  std::vector<std::shared_future<bool>> futures;
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
  const auto output_image_path =
      output_path_ / StringPrintf("%05d.jpg", reg_image_idx + 1);
  const auto proj_matrix_path =
      output_path_ / StringPrintf("%05d_P.txt", reg_image_idx + 1);

  const image_t image_id =
      *std::next(reconstruction_.RegImageIds().begin(), reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  Bitmap distorted_bitmap;
  const auto input_image_path = image_path_ / image.Name();
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

StandaloneImageUndistorter::StandaloneImageUndistorter(
    Options options,
    const UndistortCameraOptions& camera_options,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void StandaloneImageUndistorter::Run() {
  LOG_HEADING1("Image undistortion");

  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(output_path_);

  ThreadPool thread_pool;
  std::vector<std::shared_future<bool>> futures;
  const size_t num_images = options_.image_names_and_cameras.size();
  futures.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    futures.push_back(
        thread_pool.AddTask(&StandaloneImageUndistorter::Undistort, this, i));
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

bool StandaloneImageUndistorter::Undistort(const size_t image_idx) const {
  const auto& [image_name, camera] =
      options_.image_names_and_cameras[image_idx];

  const auto output_image_path = output_path_ / image_name;

  Bitmap distorted_bitmap;
  const auto input_image_path = image_path_ / image_name;
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);

  return undistorted_bitmap.Write(output_image_path);
}

StereoImageRectifier::StereoImageRectifier(
    Options options,
    const UndistortCameraOptions& camera_options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void StereoImageRectifier::Run() {
  LOG_HEADING1("Stereo rectification");

  Timer run_timer;
  run_timer.Start();

  ThreadPool thread_pool;
  std::vector<std::shared_future<void>> futures;
  futures.reserve(options_.stereo_pairs.size());
  for (const auto& stereo_pair : options_.stereo_pairs) {
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

  CreateDirIfNotExists(output_path_ / stereo_pair_name);

  const auto output_image_path1 = output_path_ / stereo_pair_name / image_name1;
  const auto output_image_path2 = output_path_ / stereo_pair_name / image_name2;

  Bitmap distorted_bitmap1;
  const auto input_image1_path = image_path_ / image1.Name();
  if (!distorted_bitmap1.Read(input_image1_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image1_path;
    return;
  }

  Bitmap distorted_bitmap2;
  const auto input_image2_path = image_path_ / image2.Name();
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
  RectifyAndUndistortStereoImages(camera_options_,
                                  distorted_bitmap1,
                                  distorted_bitmap2,
                                  camera1,
                                  camera2,
                                  cam2_from_cam1,
                                  &undistorted_bitmap1,
                                  &undistorted_bitmap2,
                                  &undistorted_camera,
                                  &Q);

  MaybeSetJpegQuality(
      output_image_path1, undistorted_bitmap1, options_.jpeg_quality);
  MaybeSetJpegQuality(
      output_image_path2, undistorted_bitmap2, options_.jpeg_quality);

  undistorted_bitmap1.Write(output_image_path1);
  undistorted_bitmap2.Write(output_image_path2);

  const auto Q_path = output_path_ / stereo_pair_name / "Q.txt";
  std::ofstream Q_file(Q_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(Q_file, Q_path);
  WriteMatrix(Q, &Q_file);
}

}  // namespace colmap
