// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include <boost/filesystem.hpp>

#include "base/camera_models.h"
#include "base/feature.h"
#include "ext/VLFeat/sift.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/timer.h"

namespace colmap {
namespace {

void ScaleBitmap(const int max_image_size, double* scale_x, double* scale_y,
                 Bitmap* bitmap) {
  if (static_cast<int>(bitmap->Width()) > max_image_size ||
      static_cast<int>(bitmap->Height()) > max_image_size) {
    // Fit the down-sampled version exactly into the max dimensions
    const double scale = static_cast<double>(max_image_size) /
                         std::max(bitmap->Width(), bitmap->Height());
    const int new_width = static_cast<int>(bitmap->Width() * scale);
    const int new_height = static_cast<int>(bitmap->Height() * scale);

    std::cout << StringPrintf(
                     "  WARNING: Image exceeds maximum dimensions "
                     "- resizing to %dx%d.",
                     new_width, new_height)
              << std::endl;

    // These scales differ from `scale`, if we round one of the dimensions.
    // But we want to exactly scale the keypoint locations below.
    *scale_x = static_cast<double>(new_width) / bitmap->Width();
    *scale_y = static_cast<double>(new_height) / bitmap->Height();

    bitmap->Rescale(new_width, new_height);
  } else {
    *scale_x = 1.0;
    *scale_y = 1.0;
  }
}

}  // namespace

void SiftOptions::Check() const {
  CHECK_GT(max_image_size, 0);
  CHECK_GT(max_num_features, 0);
  CHECK_GT(octave_resolution, 0);
  CHECK_GT(peak_threshold, 0.0);
  CHECK_GT(edge_threshold, 0.0);
  CHECK_GT(max_num_orientations, 0);
}

void ImageReader::Options::Check() const {
  CHECK_GT(default_focal_length_factor, 0.0);
  const int model_id = CameraModelNameToId(camera_model);
  CHECK_NE(model_id, -1);
  if (!camera_params.empty()) {
    CHECK(
        CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
  }
}

ImageReader::ImageReader(const Options& options)
    : options_(options), image_index_(0) {
  options_.Check();

  // Ensure trailing slash, so that we can build the correct image name.
  options_.image_path =
      EnsureTrailingSlash(StringReplace(options_.image_path, "\\", "/"));

  // Get a list of all files in the image path, sorted by image name.
  if (options_.image_list.empty()) {
    options_.image_list = GetRecursiveFileList(options_.image_path);
    std::sort(options_.image_list.begin(), options_.image_list.end());
  } else {
    for (auto& image_name : options_.image_list) {
      image_name = JoinPaths(options_.image_path, image_name);
    }
  }

  // Set the manually specified camera parameters.
  prev_camera_.SetCameraId(kInvalidCameraId);
  prev_camera_.SetModelIdFromName(options_.camera_model);
  if (!options_.camera_params.empty()) {
    prev_camera_.SetParamsFromString(options_.camera_params);
  }
}

bool ImageReader::Next(Image* image, Bitmap* bitmap) {
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(bitmap);

  image_index_ += 1;
  if (image_index_ > options_.image_list.size()) {
    return false;
  }

  const std::string image_path = options_.image_list.at(image_index_ - 1);

  Database database(options_.database_path);

  //////////////////////////////////////////////////////////////////////////////
  // Set the image name.
  //////////////////////////////////////////////////////////////////////////////

  image->SetName(image_path);
  image->SetName(StringReplace(image->Name(), "\\", "/"));
  image->SetName(StringReplace(image->Name(), options_.image_path, ""));

  std::cout << "  Name:           " << image->Name() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Check if image already read.
  //////////////////////////////////////////////////////////////////////////////

  const bool exists_image = database.ExistsImageWithName(image->Name());

  if (exists_image) {
    const DatabaseTransaction database_transaction(&database);
    *image = database.ReadImageWithName(image->Name());
    const bool exists_keypoints = database.ExistsKeypoints(image->ImageId());
    const bool exists_descriptors =
        database.ExistsDescriptors(image->ImageId());

    if (exists_keypoints && exists_descriptors) {
      std::cout << "  SKIP: Features already extracted." << std::endl;
      return false;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Read image.
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
    const Camera camera = database.ReadCamera(image->CameraId());

    if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
        (camera.Width() != prev_camera_.Width() ||
         camera.Height() != prev_camera_.Height())) {
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
  // Extract image dimensions.
  //////////////////////////////////////////////////////////////////////////////

  if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
      (prev_camera_.Width() != static_cast<size_t>(bitmap->Width()) ||
       prev_camera_.Height() != static_cast<size_t>(bitmap->Height()))) {
    std::cerr << "  ERROR: Single camera specified, but images have "
                 "different dimensions."
              << std::endl;
    return false;
  }

  prev_camera_.SetWidth(static_cast<size_t>(bitmap->Width()));
  prev_camera_.SetHeight(static_cast<size_t>(bitmap->Height()));

  std::cout << "  Width:          " << prev_camera_.Width() << "px"
            << std::endl;
  std::cout << "  Height:         " << prev_camera_.Height() << "px"
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract camera model and focal length
  //////////////////////////////////////////////////////////////////////////////

  if (!options_.single_camera || prev_camera_.CameraId() == kInvalidCameraId) {
    if (options_.camera_params.empty()) {
      // Extract focal length.
      double focal_length = 0.0;
      if (bitmap->ExifFocalLength(&focal_length)) {
        prev_camera_.SetPriorFocalLength(true);
        std::cout << "  Focal length:   " << focal_length << "px (EXIF)"
                  << std::endl;
      } else {
        focal_length = options_.default_focal_length_factor *
                       std::max(bitmap->Width(), bitmap->Height());
        prev_camera_.SetPriorFocalLength(false);
        std::cout << "  Focal length:   " << focal_length << "px" << std::endl;
      }

      prev_camera_.InitializeWithId(prev_camera_.ModelId(), focal_length,
                                    prev_camera_.Width(),
                                    prev_camera_.Height());
    }

    if (!prev_camera_.VerifyParams()) {
      std::cerr << "  ERROR: Invalid camera parameters." << std::endl;
      return false;
    }

    prev_camera_.SetCameraId(database.WriteCamera(prev_camera_));
  }

  image->SetCameraId(prev_camera_.CameraId());

  std::cout << "  Camera ID:      " << prev_camera_.CameraId() << std::endl;
  std::cout << "  Camera Model:   " << prev_camera_.ModelName() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract GPS data.
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

size_t ImageReader::NextIndex() const { return image_index_; }

size_t ImageReader::NumImages() const { return options_.image_list.size(); }

SiftCPUFeatureExtractor::SiftCPUFeatureExtractor(
    const ImageReader::Options& reader_options, const SiftOptions& sift_options,
    const Options& cpu_options)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      cpu_options_(cpu_options) {
  sift_options_.Check();
  cpu_options_.Check();
}

void SiftCPUFeatureExtractor::Options::Check() const {
  CHECK_GT(batch_size_factor, 0);
}

void SiftCPUFeatureExtractor::Run() {
  PrintHeading1("Feature extraction (CPU)");

  ImageReader image_reader(reader_options_);
  Database database(reader_options_.database_path);

  ThreadPool thread_pool(cpu_options_.num_threads);

  const size_t batch_size = static_cast<size_t>(cpu_options_.batch_size_factor *
                                                thread_pool.NumThreads());

  struct ExtractionResult {
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
  };

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    PrintHeading2("Preparing batch");

    std::vector<size_t> image_idxs;
    std::vector<Image> images;
    std::vector<std::future<ExtractionResult>> futures;
    while (futures.size() < batch_size &&
           image_reader.NextIndex() < image_reader.NumImages()) {
      if (IsStopped()) {
        break;
      }

      std::cout << StringPrintf("Preparing file [%d/%d]",
                                image_reader.NextIndex() + 1,
                                image_reader.NumImages())
                << std::endl;

      Image image;
      std::shared_ptr<Bitmap> bitmap = std::make_shared<Bitmap>();
      if (!image_reader.Next(&image, bitmap.get())) {
        continue;
      }

      image_idxs.push_back(image_reader.NextIndex());
      images.push_back(image);
      futures.push_back(thread_pool.AddTask([this, bitmap]() {
        ExtractionResult result;
        if (!ExtractSiftFeaturesCPU(sift_options_, *bitmap, &result.keypoints,
                                    &result.descriptors)) {
          std::cerr << "  ERROR: Could not extract features." << std::endl;
        }
        return result;
      }));
    }

    if (futures.empty()) {
      break;
    }

    PrintHeading2("Processing batch");

    DatabaseTransaction database_transaction(&database);

    for (size_t i = 0; i < futures.size(); ++i) {
      Image& image = images[i];
      const ExtractionResult result = futures[i].get();

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database.WriteImage(image));
      }

      if (!database.ExistsKeypoints(image.ImageId())) {
        database.WriteKeypoints(image.ImageId(), result.keypoints);
      }

      if (!database.ExistsDescriptors(image.ImageId())) {
        database.WriteDescriptors(image.ImageId(), result.descriptors);
      }

      std::cout << StringPrintf("  Features:       %d [%d/%d]",
                                result.keypoints.size(), image_idxs[i],
                                image_reader.NumImages())
                << std::endl;
    }
  }

  GetTimer().PrintMinutes();
}

void SiftGPUFeatureExtractor::Options::Check() const { CHECK_GE(index, -1); }

SiftGPUFeatureExtractor::SiftGPUFeatureExtractor(
    const ImageReader::Options& reader_options, const SiftOptions& sift_options,
    const Options& gpu_options)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      gpu_options_(gpu_options) {
  sift_options_.Check();
  gpu_options_.Check();

  // Create an OpenGL context.
  if (gpu_options_.index < 0) {
    opengl_context_.reset(new OpenGLContextManager());
  }
}

void SiftGPUFeatureExtractor::Run() {
  PrintHeading1("Feature extraction (GPU)");

  if (gpu_options_.index < 0) {
    CHECK(opengl_context_);
    opengl_context_->MakeCurrent();
  }

  SiftGPU sift_gpu;
  if (!CreateSiftGPUExtractor(sift_options_, gpu_options_.index, &sift_gpu)) {
    std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
    return;
  }

  ImageReader image_reader(reader_options_);
  Database database(reader_options_.database_path);

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Processing file [%d/%d]",
                              image_reader.NextIndex() + 1,
                              image_reader.NumImages())
              << std::endl;

    Image image;
    Bitmap bitmap;
    if (!image_reader.Next(&image, &bitmap)) {
      continue;
    }

    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    if (!ExtractSiftFeaturesGPU(sift_options_, bitmap, &sift_gpu, &keypoints,
                                &descriptors)) {
      std::cerr << "  ERROR: Could not extract features." << std::endl;
      continue;
    }

    DatabaseTransaction database_transaction(&database);

    if (image.ImageId() == kInvalidImageId) {
      image.SetImageId(database.WriteImage(image));
    }

    if (!database.ExistsKeypoints(image.ImageId())) {
      database.WriteKeypoints(image.ImageId(), keypoints);
    }

    if (!database.ExistsDescriptors(image.ImageId())) {
      database.WriteDescriptors(image.ImageId(), descriptors);
    }

    std::cout << "  Features:       " << keypoints.size() << std::endl;
  }

  GetTimer().PrintMinutes();
}

FeatureImporter::FeatureImporter(const ImageReader::Options& reader_options,
                                 const std::string& import_path)
    : reader_options_(reader_options), import_path_(import_path) {}

void FeatureImporter::Run() {
  PrintHeading1("Feature import");

  if (!boost::filesystem::exists(import_path_)) {
    std::cerr << "  ERROR: Path does not exist." << std::endl;
    return;
  }

  ImageReader image_reader(reader_options_);
  Database database(reader_options_.database_path);

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Processing file [%d/%d]",
                              image_reader.NextIndex() + 1,
                              image_reader.NumImages())
              << std::endl;

    // Load image data and possibly save camera to database.
    Bitmap bitmap;
    Image image;
    if (!image_reader.Next(&image, &bitmap)) {
      continue;
    }

    const std::string path = JoinPaths(import_path_, image.Name() + ".txt");

    if (boost::filesystem::exists(path)) {
      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

      std::cout << "  Features:       " << keypoints.size() << std::endl;

      DatabaseTransaction database_transaction(&database);

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database.WriteImage(image));
      }

      if (!database.ExistsKeypoints(image.ImageId())) {
        database.WriteKeypoints(image.ImageId(), keypoints);
      }

      if (!database.ExistsDescriptors(image.ImageId())) {
        database.WriteDescriptors(image.ImageId(), descriptors);
      }

      std::cout << "  SKIP: Image already processed." << std::endl;
    } else {
      std::cout << "  SKIP: No features found at " << path << std::endl;
    }
  }

  GetTimer().PrintMinutes();
}

bool ExtractSiftFeaturesCPU(const SiftOptions& options, const Bitmap& bitmap,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  options.Check();
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);

  Bitmap scaled_bitmap = bitmap.Clone();
  double scale_x;
  double scale_y;
  ScaleBitmap(options.max_image_size, &scale_x, &scale_y, &scaled_bitmap);

  //////////////////////////////////////////////////////////////////////////////
  // Extract features
  //////////////////////////////////////////////////////////////////////////////

  const float inv_scale_x = static_cast<float>(1.0 / scale_x);
  const float inv_scale_y = static_cast<float>(1.0 / scale_y);
  const float inv_scale_xy = (inv_scale_x + inv_scale_y) / 2.0f;

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(scaled_bitmap.Width(), scaled_bitmap.Height(),
                  options.num_octaves, options.octave_resolution,
                  options.first_octave),
      &vl_sift_delete);
  if (!sift) {
    return false;
  }

  vl_sift_set_peak_thresh(sift.get(), options.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), options.edge_threshold);

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
        level_keypoints.emplace_back(options.max_num_orientations *
                                     num_keypoints);
        level_descriptors.emplace_back(
            options.max_num_orientations * num_keypoints, 128);
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (options.upright) {
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
          std::min(num_orientations, options.max_num_orientations);

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
        if (options.normalization == SiftOptions::Normalization::L2) {
          desc = L2NormalizeFeatureDescriptors(desc);
        } else if (options.normalization ==
                   SiftOptions::Normalization::L1_ROOT) {
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
    if (num_features > options.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.
  size_t k = 0;
  keypoints->resize(num_features_with_orientations);
  descriptors->resize(num_features_with_orientations, 128);
  for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
    for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
      (*keypoints)[k] = level_keypoints[i][j];
      descriptors->row(k) = level_descriptors[i].row(j);
      k += 1;
    }
  }

  return true;
}

bool CreateSiftGPUExtractor(const SiftOptions& options, const int gpu_index,
                            SiftGPU* sift_gpu) {
  options.Check();
  CHECK_GE(gpu_index, -1);
  CHECK_NOTNULL(sift_gpu);

  std::vector<std::string> sift_gpu_args;

  sift_gpu_args.push_back("./binary");

  if (gpu_index >= 0) {
    sift_gpu_args.push_back("-cuda");
    sift_gpu_args.push_back(std::to_string(gpu_index));
  }

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  if (options.darkness_adaptivity) {
    if (gpu_index >= 0) {
      std::cout << "WARNING: Darkness adaptivity only available for GLSL "
                   "but CUDA version selected."
                << std::endl;
    }
    sift_gpu_args.push_back("-da");
  }

  // No verbose logging.
  sift_gpu_args.push_back("-v");
  sift_gpu_args.push_back("0");

  // Fixed maximum image dimension.
  sift_gpu_args.push_back("-maxd");
  sift_gpu_args.push_back(std::to_string(options.max_image_size));

  // Keep the highest level features.
  sift_gpu_args.push_back("-tc2");
  sift_gpu_args.push_back(std::to_string(options.max_num_features));

  // First octave level.
  sift_gpu_args.push_back("-fo");
  sift_gpu_args.push_back(std::to_string(options.first_octave));

  // Number of octave levels.
  sift_gpu_args.push_back("-d");
  sift_gpu_args.push_back(std::to_string(options.octave_resolution));

  // Peak threshold.
  sift_gpu_args.push_back("-t");
  sift_gpu_args.push_back(std::to_string(options.peak_threshold));

  // Edge threshold.
  sift_gpu_args.push_back("-e");
  sift_gpu_args.push_back(std::to_string(options.edge_threshold));

  if (options.upright) {
    // Fix the orientation to 0 for upright features.
    sift_gpu_args.push_back("-ofix");
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back("1");
  } else {
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back(std::to_string(options.max_num_orientations));
  }

  std::vector<const char*> sift_gpu_args_cstr;
  sift_gpu_args_cstr.reserve(sift_gpu_args.size());
  for (const auto& arg : sift_gpu_args) {
    sift_gpu_args_cstr.push_back(arg.c_str());
  }

  sift_gpu->ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

  return sift_gpu->VerifyContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
}

bool ExtractSiftFeaturesGPU(const SiftOptions& options, const Bitmap& bitmap,
                            SiftGPU* sift_gpu, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  options.Check();
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

  Bitmap scaled_bitmap = bitmap.Clone();

  double scale_x;
  double scale_y;
  ScaleBitmap(sift_gpu->GetMaxDimension(), &scale_x, &scale_y, &scaled_bitmap);

  ////////////////////////////////////////////////////////////////////////////
  // Extract features
  ////////////////////////////////////////////////////////////////////////////

  // Note, that this produces slightly different results than using SiftGPU
  // directly for RGB->GRAY conversion, since it uses different weights.
  const std::vector<uint8_t> bitmap_raw_bits = scaled_bitmap.ConvertToRawBits();
  const int code =
      sift_gpu->RunSIFT(scaled_bitmap.ScanWidth(), scaled_bitmap.Height(),
                        bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

  const int kSuccessCode = 1;
  if (code != kSuccessCode) {
    return false;
  }

  const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());
  std::vector<SiftGPU::SiftKeypoint> sift_gpu_keypoints(num_features);

  // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_float(num_features, 128);

  // Download the extracted keypoints and descriptors.
  sift_gpu->GetFeatureVector(sift_gpu_keypoints.data(),
                             descriptors_float.data());

  // Save keypoints and scale locations if original bitmap was down-sampled.
  keypoints->resize(num_features);
  if (scale_x != 1.0 || scale_y != 1.0) {
    const float inv_scale_x = static_cast<float>(1.0f / scale_x);
    const float inv_scale_y = static_cast<float>(1.0f / scale_y);
    const float inv_scale_xy = (inv_scale_x + inv_scale_y) / 2.0f;
    for (size_t i = 0; i < sift_gpu_keypoints.size(); ++i) {
      (*keypoints)[i].x = inv_scale_x * sift_gpu_keypoints[i].x;
      (*keypoints)[i].y = inv_scale_y * sift_gpu_keypoints[i].y;
      (*keypoints)[i].scale = inv_scale_xy * sift_gpu_keypoints[i].s;
      (*keypoints)[i].orientation = sift_gpu_keypoints[i].o;
    }
  } else {
    for (size_t i = 0; i < sift_gpu_keypoints.size(); ++i) {
      (*keypoints)[i].x = sift_gpu_keypoints[i].x;
      (*keypoints)[i].y = sift_gpu_keypoints[i].y;
      (*keypoints)[i].scale = sift_gpu_keypoints[i].s;
      (*keypoints)[i].orientation = sift_gpu_keypoints[i].o;
    }
  }

  // Save and normalize the descriptors.
  if (options.normalization == SiftOptions::Normalization::L2) {
    descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (options.normalization == SiftOptions::Normalization::L1_ROOT) {
    descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
  }
  *descriptors = FeatureDescriptorsToUnsignedByte(descriptors_float);

  return true;
}

void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors) {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);

  std::ifstream file(path.c_str());
  CHECK(file.is_open());

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const point2D_t num_features = boost::lexical_cast<point2D_t>(item);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const size_t dim = boost::lexical_cast<size_t>(item);

  CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

  keypoints->resize(num_features);
  descriptors->resize(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    // X
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].x = boost::lexical_cast<float>(item);

    // Y
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].y = boost::lexical_cast<float>(item);

    // Scale
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].scale = boost::lexical_cast<float>(item);

    // Orientation
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].orientation = boost::lexical_cast<float>(item);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream >> std::ws, item, ' ');
      const double value = boost::lexical_cast<double>(item);
      CHECK_GE(value, 0);
      CHECK_LE(value, 255);
      (*descriptors)(i, j) = TruncateCast<double, uint8_t>(value);
    }
  }
}

}  // namespace colmap
