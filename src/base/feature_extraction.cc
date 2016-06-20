// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/feature_extraction.h"

#include <fstream>

#include "base/camera_models.h"
#include "base/feature.h"
#include "ext/SiftGPU/SiftGPU.h"
#include "ext/VLFeat/sift.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {
namespace {

bool LoadFeaturesFromTextFile(const std::string& path, Database* database,
                              Image* image) {
  std::ifstream file(path.c_str());

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream, item, ' ');
  boost::trim(item);
  const point2D_t num_features = boost::lexical_cast<point2D_t>(item);

  std::getline(header_line_stream, item, ' ');
  boost::trim(item);
  const size_t dim = boost::lexical_cast<size_t>(item);

  FeatureKeypoints keypoints(num_features);
  FeatureDescriptors descriptors(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    // X
    std::getline(feature_line_stream, item, ' ');
    boost::trim(item);
    keypoints[i].x = boost::lexical_cast<float>(item);

    // Y
    std::getline(feature_line_stream, item, ' ');
    boost::trim(item);
    keypoints[i].y = boost::lexical_cast<float>(item);

    // Scale
    std::getline(feature_line_stream, item, ' ');
    boost::trim(item);
    keypoints[i].scale = boost::lexical_cast<float>(item);

    // Orientation
    std::getline(feature_line_stream, item, ' ');
    boost::trim(item);
    keypoints[i].orientation = boost::lexical_cast<float>(item);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream, item, ' ');
      boost::trim(item);
      const int value = boost::lexical_cast<int>(item);
      if (value < 0) {
        descriptors(i, j) = 0;
      } else if (value > 255) {
        descriptors(i, j) = 255;
      } else {
        descriptors(i, j) = static_cast<uint8_t>(value);
      }
    }
  }

  std::cout << "  Features:       " << num_features << std::endl;

  file.close();

  database->BeginTransaction();

  if (image->ImageId() == kInvalidImageId) {
    image->SetImageId(database->WriteImage(*image));
  }

  if (!database->ExistsKeypoints(image->ImageId())) {
    database->WriteKeypoints(image->ImageId(), keypoints);
  }

  if (!database->ExistsDescriptors(image->ImageId())) {
    database->WriteDescriptors(image->ImageId(), descriptors);
  }

  database->EndTransaction();

  return true;
}

void ScaleBitmap(const Camera& camera, const int max_image_size,
                 double* scale_x, double* scale_y, Bitmap* bitmap) {
  if (static_cast<int>(camera.Width()) > max_image_size ||
      static_cast<int>(camera.Height()) > max_image_size) {
    // Fit the down-sampled version exactly into the max dimensions
    const double scale = static_cast<double>(max_image_size) /
                         std::max(camera.Width(), camera.Height());
    const int new_width = static_cast<int>(camera.Width() * scale);
    const int new_height = static_cast<int>(camera.Height() * scale);

    std::cout << StringPrintf(
                     "  WARNING: Image exceeds maximum dimensions "
                     "- resizing to %dx%d.",
                     new_width, new_height)
              << std::endl;

    // These scales differ from `scale`, if we round one of the dimensions.
    // But we want to exactly scale the keypoint locations below.
    *scale_x = static_cast<double>(new_width) / camera.Width();
    *scale_y = static_cast<double>(new_height) / camera.Height();

    bitmap->Rescale(new_width, new_height);
  } else {
    *scale_x = 1.0;
    *scale_y = 1.0;
  }
}

// Recursively collect list of files in sorted order.
std::vector<std::string> GetRecursiveFileList(const std::string& path) {
  namespace fs = boost::filesystem;
  std::vector<std::string> file_list;
  for (auto it = fs::recursive_directory_iterator(path);
       it != fs::recursive_directory_iterator(); ++it) {
    if (fs::is_regular_file(*it)) {
      const fs::path file_path = *it;
      file_list.push_back(file_path.string());
    }
  }
  std::sort(file_list.begin(), file_list.end());
  return file_list;
}

}  // namespace

void SIFTOptions::Check() const {
  CHECK_GT(max_image_size, 0);
  CHECK_GT(max_num_features, 0);
  CHECK_GT(octave_resolution, 0);
  CHECK_GT(peak_threshold, 0.0);
  CHECK_GT(edge_threshold, 0.0);
  CHECK_GT(max_num_orientations, 0);
}

void FeatureExtractor::Options::Check() const {
  CHECK_GT(default_focal_length_factor, 0.0);
  const int model_id = CameraModelNameToId(camera_model);
  CHECK_NE(model_id, -1);
  if (!camera_params.empty()) {
    CHECK(
        CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
  }
}

FeatureExtractor::FeatureExtractor(const Options& options,
                                   const std::string& database_path,
                                   const std::string& image_path)
    : stop_(false),
      options_(options),
      database_path_(database_path),
      image_path_(image_path) {
  options_.Check();
  // Ensure trailing slash, so that we can build the correct image name
  image_path_ = StringReplace(image_path_, "\\", "/");
  image_path_ = EnsureTrailingSlash(image_path_);
}

void FeatureExtractor::run() {
  last_camera_.SetModelIdFromName(options_.camera_model);
  last_camera_id_ = kInvalidCameraId;
  if (!options_.camera_params.empty() &&
      !last_camera_.SetParamsFromString(options_.camera_params)) {
    std::cerr << "  ERROR: Invalid camera parameters." << std::endl;
    return;
  }

  Timer total_timer;
  total_timer.Start();

  database_.Open(database_path_);
  DoExtraction();
  database_.Close();

  total_timer.PrintMinutes();
}

void FeatureExtractor::Stop() {
  QMutexLocker locker(&mutex_);
  stop_ = true;
}

bool FeatureExtractor::ReadImage(const std::string& image_path, Image* image,
                                 Bitmap* bitmap) {
  image->SetName(image_path);
  image->SetName(StringReplace(image->Name(), "\\", "/"));
  image->SetName(StringReplace(image->Name(), image_path_, ""));

  std::cout << "  Name:           " << image->Name() << std::endl;

  const bool exists_image = database_.ExistsImageName(image->Name());

  if (exists_image) {
    database_.BeginTransaction();
    *image = database_.ReadImageFromName(image->Name());
    const bool exists_keypoints = database_.ExistsKeypoints(image->ImageId());
    const bool exists_descriptors =
        database_.ExistsDescriptors(image->ImageId());
    database_.EndTransaction();

    if (exists_keypoints && exists_descriptors) {
      std::cout << "  SKIP: Image already processed." << std::endl;
      return false;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Read image
  //////////////////////////////////////////////////////////////////////////////

  if (!bitmap->Read(image_path, false)) {
    std::cout << "  SKIP: Cannot read image at path " << image_path
              << std::endl;
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check for well-formed data.
  //////////////////////////////////////////////////////////////////////////////

  if (exists_image) {
    const Camera camera = database_.ReadCamera(image->CameraId());

    if (options_.single_camera && last_camera_id_ != kInvalidCameraId &&
        (camera.Width() != last_camera_.Width() ||
         camera.Height() != last_camera_.Height())) {
      std::cerr << "  ERROR: Single camera specified, but images have "
                   "different dimensions."
                << std::endl;
      return false;
    }

    if (static_cast<size_t>(bitmap->Width()) != camera.Width() ||
        static_cast<size_t>(bitmap->Height()) != camera.Height()) {
      std::cerr << "  ERROR: Image previously processed, but current version "
                   "has different dimensions."
                << std::endl;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Extract image dimensions
  //////////////////////////////////////////////////////////////////////////////

  if (options_.single_camera && last_camera_id_ != kInvalidCameraId &&
      (last_camera_.Width() != static_cast<size_t>(bitmap->Width()) ||
       last_camera_.Height() != static_cast<size_t>(bitmap->Height()))) {
    std::cerr << "  ERROR: Single camera specified, but images have "
                 "different dimensions"
              << std::endl;
    return false;
  }

  last_camera_.SetWidth(static_cast<size_t>(bitmap->Width()));
  last_camera_.SetHeight(static_cast<size_t>(bitmap->Height()));

  std::cout << "  Width:          " << last_camera_.Width() << "px"
            << std::endl;
  std::cout << "  Height:         " << last_camera_.Height() << "px"
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract camera model and focal length
  //////////////////////////////////////////////////////////////////////////////

  if (!options_.single_camera || last_camera_id_ == kInvalidCameraId) {
    if (options_.camera_params.empty()) {
      // Extract focal length.
      double focal_length = 0.0;
      if (bitmap->ExifFocalLength(&focal_length)) {
        last_camera_.SetPriorFocalLength(true);
        std::cout << "  Focal length:   " << focal_length << "px (EXIF)"
                  << std::endl;
      } else {
        focal_length = options_.default_focal_length_factor *
                       std::max(bitmap->Width(), bitmap->Height());
        last_camera_.SetPriorFocalLength(false);
        std::cout << "  Focal length:   " << focal_length << "px" << std::endl;
      }

      last_camera_.InitializeWithId(last_camera_.ModelId(), focal_length,
                                    last_camera_.Width(),
                                    last_camera_.Height());
    }

    if (!last_camera_.VerifyParams()) {
      std::cerr << "  ERROR: Invalid camera parameters." << std::endl;
      return false;
    }

    last_camera_id_ = database_.WriteCamera(last_camera_);
  }

  image->SetCameraId(last_camera_id_);

  std::cout << "  Camera ID:      " << last_camera_id_ << std::endl;
  std::cout << "  Camera Model:   " << last_camera_.ModelName() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract GPS data
  //////////////////////////////////////////////////////////////////////////////

  if (bitmap->ExifLatitude(&image->TvecPrior(0)) &&
      bitmap->ExifLongitude(&image->TvecPrior(1)) &&
      bitmap->ExifAltitude(&image->TvecPrior(2))) {
    std::cout << StringPrintf("  EXIF GPS:       LAT=%.3f, LON=%.3f, ALT=%.3f",
                              image->TvecPrior(0), image->TvecPrior(1),
                              image->TvecPrior(2))
              << std::endl;
  } else {
    image->TvecPrior(0) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(1) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(2) = std::numeric_limits<double>::quiet_NaN();
  }

  return true;
}

SiftCPUFeatureExtractor::SiftCPUFeatureExtractor(
    const Options& options, const SIFTOptions& sift_options,
    const CPUOptions& cpu_options, const std::string& database_path,
    const std::string& image_path)
    : FeatureExtractor(options, database_path, image_path),
      sift_options_(sift_options),
      cpu_options_(cpu_options) {
  sift_options_.Check();
  cpu_options_.Check();
}

void SiftCPUFeatureExtractor::CPUOptions::Check() const {
  CHECK_GT(batch_size_factor, 0);
}

void SiftCPUFeatureExtractor::DoExtraction() {
  PrintHeading1("Feature extraction (CPU)");

  //////////////////////////////////////////////////////////////////////////////
  // Extract features
  //////////////////////////////////////////////////////////////////////////////

  ThreadPool thread_pool(cpu_options_.num_threads);

  const size_t batch_size = static_cast<size_t>(cpu_options_.batch_size_factor *
                                                thread_pool.NumThreads());

  const std::vector<std::string> file_list = GetRecursiveFileList(image_path_);

  size_t file_idx = 0;
  while (file_idx < file_list.size()) {
    {
      QMutexLocker locker(&mutex_);
      if (stop_) {
        return;
      }
    }

    PrintHeading2("Preparing batch");

    std::vector<size_t> file_idxs;
    std::vector<Image> images;
    std::vector<std::future<ExtractionResult>> futures;
    for (; futures.size() < batch_size && file_idx < file_list.size();
         ++file_idx) {
      std::cout << "Preparing file [" << file_idx + 1 << "/" << file_list.size()
                << "]" << std::endl;

      const std::string image_path = file_list[file_idx];

      Image image;
      std::shared_ptr<Bitmap> bitmap = std::make_shared<Bitmap>();
      if (!ReadImage(image_path, &image, bitmap.get())) {
        continue;
      }

      file_idxs.push_back(file_idx);
      images.push_back(image);
      futures.push_back(
          thread_pool.AddTask(SiftCPUFeatureExtractor::DoExtractionKernel,
                              last_camera_, image, bitmap, sift_options_));
    }

    PrintHeading2("Processing batch");

    for (size_t i = 0; i < futures.size(); ++i) {
      Image& image = images[i];
      const ExtractionResult result = futures[i].get();

      // Save the features to the database.

      database_.BeginTransaction();

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database_.WriteImage(image));
      }

      if (!database_.ExistsKeypoints(image.ImageId())) {
        database_.WriteKeypoints(image.ImageId(), result.keypoints);
      }

      if (!database_.ExistsDescriptors(image.ImageId())) {
        database_.WriteDescriptors(image.ImageId(), result.descriptors);
      }

      std::cout << "  Features:       " << result.keypoints.size() << " ["
                << file_idxs[i] << "/" << file_list.size() << "]" << std::endl;

      database_.EndTransaction();
    }
  }
}

SiftCPUFeatureExtractor::ExtractionResult
SiftCPUFeatureExtractor::DoExtractionKernel(
    const Camera& camera, const Image& image,
    const std::shared_ptr<Bitmap>& bitmap, const SIFTOptions& sift_options) {
  ExtractionResult result;

  //////////////////////////////////////////////////////////////////////////////
  // Read image
  //////////////////////////////////////////////////////////////////////////////

  Bitmap scaled_bitmap = bitmap->Clone();
  double scale_x;
  double scale_y;
  ScaleBitmap(camera, sift_options.max_image_size, &scale_x, &scale_y,
              &scaled_bitmap);

  //////////////////////////////////////////////////////////////////////////////
  // Extract features
  //////////////////////////////////////////////////////////////////////////////

  const float inv_scale_x = static_cast<float>(1.0 / scale_x);
  const float inv_scale_y = static_cast<float>(1.0 / scale_y);
  const float inv_scale_xy = (inv_scale_x + inv_scale_y) / 2.0f;

  // Setup SIFT.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(scaled_bitmap.Width(), scaled_bitmap.Height(),
                  sift_options.num_octaves, sift_options.octave_resolution,
                  sift_options.first_octave),
      &vl_sift_delete);
  vl_sift_set_peak_thresh(sift.get(), sift_options.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), sift_options.edge_threshold);

  // Iterate through octaves.
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptors> level_descriptors;
  bool first_octave = true;
  while (true) {
    if (first_octave) {
      const std::vector<uint8_t> data_uint8 =
          scaled_bitmap.ConvertToRowMajorArray();
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      if (vl_sift_process_first_octave(sift.get(), data_float.data())) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift.get())) {
        break;
      }
    }

    // Detect keypoints.
    vl_sift_detect(sift.get());

    // Extract detected keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift.get());
    const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level.
    size_t level_idx = 0;
    int prev_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level.
          level_keypoints.back().resize(level_idx);
          level_descriptors.back().conservativeResize(level_idx, 128);
        }

        // Add containers for new DOG level.
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(sift_options.max_num_orientations *
                                     num_keypoints);
        level_descriptors.emplace_back(
            sift_options.max_num_orientations * num_keypoints, 128);
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (sift_options.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(
            sift.get(), angles, &vl_keypoints[i]);
      }

      // Note that this is different from SiftGPU, which selects the top
      // global maxima as orientations while this selects the first two
      // local maxima. It is not clear which procedure is better.
      const int num_used_orientations =
          std::min(num_orientations, sift_options.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx].x = vl_keypoints[i].x + 0.5f;
        level_keypoints.back()[level_idx].y = vl_keypoints[i].y + 0.5f;
        level_keypoints.back()[level_idx].scale = vl_keypoints[i].sigma;
        level_keypoints.back()[level_idx].orientation = angles[o];

        if (scale_x != 1.0 || scale_y != 1.0) {
          level_keypoints.back()[level_idx].x *= inv_scale_x;
          level_keypoints.back()[level_idx].y *= inv_scale_y;
          level_keypoints.back()[level_idx].scale *= inv_scale_xy;
        }

        Eigen::MatrixXf desc(1, 128);
        vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                         &vl_keypoints[i], angles[o]);
        if (sift_options.normalization == SIFTOptions::Normalization::L2) {
          desc = L2NormalizeFeatureDescriptors(desc);
        } else if (sift_options.normalization ==
                   SIFTOptions::Normalization::L1_ROOT) {
          desc = L1RootNormalizeFeatureDescriptors(desc);
        }
        level_descriptors.back().row(level_idx) =
            FeatureDescriptorsToUnsignedByte(desc);

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave.
    level_keypoints.back().resize(level_idx);
    level_descriptors.back().conservativeResize(level_idx, 128);
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option.

  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = level_keypoints.size() - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += level_keypoints[i].size();
    if (num_features > sift_options.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.

  size_t k = 0;
  result.keypoints = FeatureKeypoints(num_features_with_orientations);
  result.descriptors = FeatureDescriptors(num_features_with_orientations, 128);
  for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
    for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
      result.keypoints[k] = level_keypoints[i][j];
      result.descriptors.row(k) = level_descriptors[i].row(j);
      k += 1;
    }
  }

  return result;
}

SiftGPUFeatureExtractor::SiftGPUFeatureExtractor(
    const Options& options, const SIFTOptions& sift_options,
    const std::string& database_path, const std::string& image_path)
    : FeatureExtractor(options, database_path, image_path),
      sift_options_(sift_options),
      parent_thread_(QThread::currentThread()) {
  sift_options_.Check();
  surface_ = new QOffscreenSurface();
  surface_->create();
  context_ = new QOpenGLContext();
  context_->create();
  context_->makeCurrent(surface_);
  context_->doneCurrent();
  context_->moveToThread(this);
}

SiftGPUFeatureExtractor::~SiftGPUFeatureExtractor() {
  delete context_;
  surface_->deleteLater();
}

void SiftGPUFeatureExtractor::TearDown() {
  context_->doneCurrent();
  context_->moveToThread(parent_thread_);
}

void SiftGPUFeatureExtractor::DoExtraction() {
  PrintHeading1("Feature extraction (GPU)");

  context_->makeCurrent(surface_);

  //////////////////////////////////////////////////////////////////////////////
  // Set up SiftGPU
  //////////////////////////////////////////////////////////////////////////////

  std::vector<std::string> sift_gpu_args;

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  sift_gpu_args.push_back("-da");

  // No verbose logging.
  sift_gpu_args.push_back("-v");
  sift_gpu_args.push_back("0");

  // Fixed maximum image dimension.
  sift_gpu_args.push_back("-maxd");
  sift_gpu_args.push_back(std::to_string(sift_options_.max_image_size));

  // Keep the highest level features.
  sift_gpu_args.push_back("-tc2");
  sift_gpu_args.push_back(std::to_string(sift_options_.max_num_features));

  // First octave level.
  sift_gpu_args.push_back("-fo");
  sift_gpu_args.push_back(std::to_string(sift_options_.first_octave));

  // Number of octave levels.
  sift_gpu_args.push_back("-d");
  sift_gpu_args.push_back(std::to_string(sift_options_.octave_resolution));

  // Peak threshold.
  sift_gpu_args.push_back("-t");
  sift_gpu_args.push_back(std::to_string(sift_options_.peak_threshold));

  // Edge threshold.
  sift_gpu_args.push_back("-e");
  sift_gpu_args.push_back(std::to_string(sift_options_.edge_threshold));

  if (sift_options_.upright) {
    // Fix the orientation to 0 for upright features.
    sift_gpu_args.push_back("-ofix");
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back("1");
  } else {
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back(std::to_string(sift_options_.max_num_orientations));
  }

  std::vector<const char*> sift_gpu_args_cstr;
  sift_gpu_args_cstr.reserve(sift_gpu_args.size());
  for (const auto& arg : sift_gpu_args) {
    sift_gpu_args_cstr.push_back(arg.c_str());
  }

  SiftGPU sift_gpu;
  sift_gpu.ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

  if (sift_gpu.VerifyContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
    std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
    TearDown();
    return;
  }

  const int max_image_size = sift_gpu.GetMaxDimension();

  //////////////////////////////////////////////////////////////////////////////
  // Extract features
  //////////////////////////////////////////////////////////////////////////////

  const std::vector<std::string> file_list = GetRecursiveFileList(image_path_);

  for (size_t file_idx = 0; file_idx < file_list.size(); ++file_idx) {
    {
      QMutexLocker locker(&mutex_);
      if (stop_) {
        TearDown();
        return;
      }
    }

    std::cout << "Processing file [" << file_idx + 1 << "/" << file_list.size()
              << "]" << std::endl;

    const std::string image_path = file_list[file_idx];

    ////////////////////////////////////////////////////////////////////////////
    // Read image
    ////////////////////////////////////////////////////////////////////////////

    Image image;
    Bitmap bitmap;
    if (!ReadImage(image_path, &image, &bitmap)) {
      continue;
    }

    double scale_x;
    double scale_y;
    ScaleBitmap(last_camera_, max_image_size, &scale_x, &scale_y, &bitmap);

    ////////////////////////////////////////////////////////////////////////////
    // Extract features
    ////////////////////////////////////////////////////////////////////////////

    // Note, that this produces slightly different results than using SiftGPU
    // directly for RGB->GRAY conversion, since it uses different weights.
    const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
    const int code = sift_gpu.RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
                                      bitmap_raw_bits.data(), GL_LUMINANCE,
                                      GL_UNSIGNED_BYTE);

    const int kSuccessCode = 1;
    if (code == kSuccessCode) {
      database_.BeginTransaction();

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database_.WriteImage(image));
      }

      const size_t num_features = static_cast<size_t>(sift_gpu.GetFeatureNum());
      std::vector<SiftGPU::SiftKeypoint> sift_gpu_keypoints(num_features);

      // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          descriptors(num_features, 128);

      sift_gpu.GetFeatureVector(sift_gpu_keypoints.data(), descriptors.data());

      if (!database_.ExistsKeypoints(image.ImageId())) {
        // Assume that coordinates start at (0, 0) for the upper left corner
        // of the image, i.e. the upper left pixel center has the coordinate
        // (0.5, 0.5). This is the default in SiftGPU.
        if (scale_x != 1.0 || scale_y != 1.0) {
          const float inv_scale_x = static_cast<float>(1.0 / scale_x);
          const float inv_scale_y = static_cast<float>(1.0 / scale_y);
          const float inv_scale_xy = (inv_scale_x + inv_scale_y) / 2.0f;
          for (size_t i = 0; i < sift_gpu_keypoints.size(); ++i) {
            sift_gpu_keypoints[i].x *= inv_scale_x;
            sift_gpu_keypoints[i].y *= inv_scale_y;
            sift_gpu_keypoints[i].s *= inv_scale_xy;
          }
        }

        FeatureKeypoints keypoints(num_features);
        for (size_t i = 0; i < sift_gpu_keypoints.size(); ++i) {
          keypoints[i].x = sift_gpu_keypoints[i].x;
          keypoints[i].y = sift_gpu_keypoints[i].y;
          keypoints[i].scale = sift_gpu_keypoints[i].s;
          keypoints[i].orientation = sift_gpu_keypoints[i].o;
        }

        database_.WriteKeypoints(image.ImageId(), keypoints);
      }

      if (!database_.ExistsDescriptors(image.ImageId())) {
        if (sift_options_.normalization == SIFTOptions::Normalization::L2) {
          descriptors = L2NormalizeFeatureDescriptors(descriptors);
        } else if (sift_options_.normalization ==
                   SIFTOptions::Normalization::L1_ROOT) {
          descriptors = L1RootNormalizeFeatureDescriptors(descriptors);
        }
        const FeatureDescriptors descriptors_byte =
            FeatureDescriptorsToUnsignedByte(descriptors);
        database_.WriteDescriptors(image.ImageId(), descriptors_byte);
      }

      std::cout << "  Features:       " << num_features << std::endl;

      database_.EndTransaction();
    } else {
      std::cerr << "  ERROR: Could not extract features." << std::endl;
    }
  }

  TearDown();
}

FeatureImporter::FeatureImporter(const Options& options,
                                 const std::string& database_path,
                                 const std::string& image_path,
                                 const std::string& import_path)
    : FeatureExtractor(options, database_path, image_path),
      import_path_(EnsureTrailingSlash(import_path)) {}

void FeatureImporter::DoExtraction() {
  PrintHeading1("Feature import");

  if (!boost::filesystem::exists(import_path_)) {
    std::cerr << "  ERROR: Path does not exist." << std::endl;
    return;
  }

  last_camera_.SetModelIdFromName(options_.camera_model);

  const std::vector<std::string> file_list = GetRecursiveFileList(image_path_);

  for (size_t file_idx = 0; file_idx < file_list.size(); ++file_idx) {
    {
      QMutexLocker locker(&mutex_);
      if (stop_) {
        return;
      }
    }

    std::cout << "Processing file [" << file_idx + 1 << "/" << file_list.size()
              << "]" << std::endl;

    const std::string image_path = file_list[file_idx];

    // Load image data and possibly save camera to database.
    Bitmap bitmap;
    Image image;
    if (!ReadImage(image_path, &image, &bitmap)) {
      continue;
    }

    const std::string path = import_path_ + image.Name() + ".txt";

    if (boost::filesystem::exists(path)) {
      if (!LoadFeaturesFromTextFile(path, &database_, &image)) {
        std::cout << "  SKIP: Image already processed." << std::endl;
      }
    } else {
      std::cout << "  SKIP: No features found at " << path << std::endl;
    }
  }
}

}  // namespace colmap
