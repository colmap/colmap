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

#include <array>
#include <fstream>
#include <memory>

#include "ext/SiftGPU/SiftGPU.h"
#include "ext/VLFeat/sift.h"
#include "util/cuda.h"
#include "util/misc.h"

namespace colmap {
namespace {

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

void ScaleKeypoints(const Bitmap& bitmap, const Camera& camera,
                    FeatureKeypoints* keypoints) {
  if (static_cast<size_t>(bitmap.Width()) != camera.Width() ||
      static_cast<size_t>(bitmap.Height()) != camera.Height()) {
    const float scale_x = static_cast<float>(camera.Width()) / bitmap.Width();
    const float scale_y = static_cast<float>(camera.Height()) / bitmap.Height();
    const float scale_xy = 0.5f * (scale_x + scale_y);
    for (auto& keypoint : *keypoints) {
      keypoint.x *= scale_x;
      keypoint.y *= scale_x;
      keypoint.scale *= scale_xy;
    }
  }
}

int RemoveKeypointsByAlpha(const Bitmap& bitmap,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  size_t new_descriptor = 0;
  size_t d = 0;
  auto new_keypoint = keypoints->begin();
  for ( auto i = keypoints->begin(); i < keypoints->end(); ) {
    const double x = static_cast<double>( i->x );
    const double y = static_cast<double>( i->y );
    BitmapColor<uint8_t> color;
    bitmap.InterpolateAlphaNearestNeighbor(x, y, &color);
    // TODO: use the case where the keypoint is masked: color.r < 255
    //       more efficient?
    if (color.r == 255) {
      *new_keypoint = std::move(*i);
      // TODO: is row().swap() better?
      descriptors->row(new_descriptor) = descriptors->row(d);
      ++new_keypoint;
      ++new_descriptor;
    }
    ++i; ++d;
  }
  if (new_descriptor < keypoints->size() ) {
    keypoints->resize(new_descriptor);
    descriptors->conservativeResize(new_descriptor, descriptors->cols());
  }
  return static_cast<int>( new_descriptor );
}


}  // namespace

bool SiftExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  return true;
}

SiftFeatureExtractor::SiftFeatureExtractor(
   const ImageReader::Options& reader_options,
   const SiftExtractionOptions& sift_options)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      database_(reader_options_.database_path),
      image_reader_(reader_options_, &database_)
{

  CHECK(reader_options_.Check());
  CHECK(sift_options_.Check());

  const int num_threads = GetEffectiveNumThreads(sift_options_.num_threads);
  CHECK_GT(num_threads, 0);

  // Make sure that we only have limited number of objects in the queue to avoid
  // excess in memory usage since images and features take lots of memory.
  const int kQueueSize = 1;
  resizer_queue_.reset(new JobQueue<internal::ImageData>(kQueueSize));
  extractor_queue_.reset(new JobQueue<internal::ImageData>(kQueueSize));
  writer_queue_.reset(new JobQueue<internal::ImageData>(kQueueSize));

  if (sift_options_.max_image_size > 0) {
    for (int i = 0; i < num_threads; ++i) {
      resizers_.emplace_back(new internal::ImageResizerThread(
          sift_options_.max_image_size, resizer_queue_.get(),
          extractor_queue_.get()));
    }
  }

  if (sift_options_.use_gpu) {
    std::vector<int> gpu_indices = CSVToVector<int>(sift_options_.gpu_index);
    CHECK_GT(gpu_indices.size(), 0);

#ifdef CUDA_ENABLED
    if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
      const int num_cuda_devices = GetNumCudaDevices();
      CHECK_GT(num_cuda_devices, 0);
      gpu_indices.resize(num_cuda_devices);
      std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
    }
#endif  // CUDA_ENABLED

    auto sift_gpu_options = sift_options_;
    for (const auto& gpu_index : gpu_indices) {
      sift_gpu_options.gpu_index = std::to_string(gpu_index);
      extractors_.emplace_back(new internal::SiftGPUFeatureExtractorThread(
          sift_gpu_options, extractor_queue_.get(), writer_queue_.get()));
    }
  } else {
    for (int i = 0; i < num_threads; ++i) {
      extractors_.emplace_back(new internal::SiftCPUFeatureExtractorThread(
          sift_options_, extractor_queue_.get(), writer_queue_.get()));
    }
  }

  writer_.reset(new internal::FeatureWriterThread(
      image_reader_.NumImages(), &database_, writer_queue_.get()));
}

void SiftFeatureExtractor::Run() {
  PrintHeading1("Feature extraction");

  for (auto& resizer : resizers_) {
    resizer->Start();
  }

  for (auto& extractor : extractors_) {
    extractor->Start();
  }

  writer_->Start();

  for (auto& extractor : extractors_) {
    if (!extractor->CheckValidSetup()) {
      return;
    }
  }

  while (image_reader_.NextIndex() < image_reader_.NumImages()) {
    if (IsStopped()) {
      resizer_queue_->Stop();
      extractor_queue_->Stop();
      writer_queue_->Stop();
      resizer_queue_->Clear();
      extractor_queue_->Clear();
      writer_queue_->Clear();
      break;
    }

    internal::ImageData image_data;
    image_data.status = image_reader_.Next(
        &image_data.camera, &image_data.image, &image_data.bitmap);

    if (image_data.status != ImageReader::Status::SUCCESS) {
      image_data.bitmap.Deallocate();
    }

    if (sift_options_.max_image_size > 0) {
      CHECK(resizer_queue_->Push(image_data));
    } else {
      CHECK(extractor_queue_->Push(image_data));
    }
  }

  resizer_queue_->Wait();
  resizer_queue_->Stop();
  for (auto& resizer : resizers_) {
    resizer->Wait();
  }

  extractor_queue_->Wait();
  extractor_queue_->Stop();
  for (auto& extractor : extractors_) {
    extractor->Wait();
  }

  writer_queue_->Wait();
  writer_queue_->Stop();
  writer_->Wait();

  GetTimer().PrintMinutes();
}

FeatureImporter::FeatureImporter(const ImageReader::Options& reader_options,
                                 const std::string& import_path)
    : reader_options_(reader_options), import_path_(import_path) {}

void FeatureImporter::Run() {
  PrintHeading1("Feature import");

  if (!ExistsDir(import_path_)) {
    std::cerr << "  ERROR: Import directory does not exist." << std::endl;
    return;
  }

  Database database(reader_options_.database_path);
  ImageReader image_reader(reader_options_, &database);

  while (image_reader.NextIndex() < image_reader.NumImages()) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Processing file [%d/%d]",
                              image_reader.NextIndex() + 1,
                              image_reader.NumImages())
              << std::endl;

    // Load image data and possibly save camera to database.
    Camera camera;
    Image image;
    Bitmap bitmap;
    if (image_reader.Next(&camera, &image, &bitmap) !=
        ImageReader::Status::SUCCESS) {
      continue;
    }

    const std::string path = JoinPaths(import_path_, image.Name() + ".txt");

    if (ExistsFile(path)) {
      FeatureKeypoints keypoints;
      FeatureDescriptors descriptors;
      LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

      std::cout << "  Features:       " << keypoints.size() << std::endl;

      DatabaseTransaction database_transaction(&database);

      if (image.ImageId() == kInvalidImageId) {
        image.SetImageId(database.WriteImage(image));
      }

      if (!database.ExistsKeypoints(image.ImageId())) {
        // possible place to filter keypoints
        database.WriteKeypoints(image.ImageId(), keypoints);
      }

      if (!database.ExistsDescriptors(image.ImageId())) {
        database.WriteDescriptors(image.ImageId(), descriptors);
      }
    } else {
      std::cout << "  SKIP: No features found at " << path << std::endl;
    }
  }

  GetTimer().PrintMinutes();
}

bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);

  //////////////////////////////////////////////////////////////////////////////
  // Extract features
  //////////////////////////////////////////////////////////////////////////////

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(bitmap.Width(), bitmap.Height(), options.num_octaves,
                  options.octave_resolution, options.first_octave),
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
      const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
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

        Eigen::MatrixXf desc(1, 128);
        vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                         &vl_keypoints[i], angles[o]);
        if (options.normalization == SiftExtractionOptions::Normalization::L2) {
          desc = L2NormalizeFeatureDescriptors(desc);
        } else if (options.normalization ==
                   SiftExtractionOptions::Normalization::L1_ROOT) {
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

  *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);

  return true;
}

bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu) {
  CHECK(options.Check());
  CHECK_NOTNULL(sift_gpu);

  // SiftGPU uses many global static state variables and the initialization must
  // be thread-safe in order to work correctly. This is enforced here.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  std::vector<std::string> sift_gpu_args;

  sift_gpu_args.push_back("./sift_gpu");

#ifdef CUDA_ENABLED
  // Use CUDA version by default if darkness adaptivity is disabled.
  if (!options.darkness_adaptivity && gpu_indices[0] < 0) {
    gpu_indices[0] = 0;
  }

  if (gpu_indices[0] >= 0) {
    sift_gpu_args.push_back("-cuda");
    sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
  }
#endif  // CUDA_ENABLED

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  if (options.darkness_adaptivity) {
    if (gpu_indices[0] >= 0) {
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

bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

  // Make sure the SiftGPU keypoint is equivalent to ours.
  static_assert(
      offsetof(SiftGPU::SiftKeypoint, x) == offsetof(FeatureKeypoint, x),
      "Invalid keypoint format");
  static_assert(
      offsetof(SiftGPU::SiftKeypoint, y) == offsetof(FeatureKeypoint, y),
      "Invalid keypoint format");
  static_assert(
      offsetof(SiftGPU::SiftKeypoint, s) == offsetof(FeatureKeypoint, scale),
      "Invalid keypoint format");
  static_assert(offsetof(SiftGPU::SiftKeypoint, o) ==
                    offsetof(FeatureKeypoint, orientation),
                "Invalid keypoint format");

  // Note, that this produces slightly different results than using SiftGPU
  // directly for RGB->GRAY conversion, since it uses different weights.
  const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
  const int code =
      sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
                        bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

  const int kSuccessCode = 1;
  if (code != kSuccessCode) {
    return false;
  }

  const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());
  keypoints->resize(num_features);

  // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_float(num_features, 128);

  // Download the extracted keypoints and descriptors.
  sift_gpu->GetFeatureVector(
      reinterpret_cast<SiftGPU::SiftKeypoint*>(keypoints->data()),
      descriptors_float.data());

  // Save and normalize the descriptors.
  if (options.normalization == SiftExtractionOptions::Normalization::L2) {
    descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (options.normalization ==
             SiftExtractionOptions::Normalization::L1_ROOT) {
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
  CHECK(file.is_open()) << path;

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const point2D_t num_features = std::stoi(item);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const size_t dim = std::stoi(item);

  CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

  keypoints->resize(num_features);
  descriptors->resize(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    // X
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].x = std::stof(item);

    // Y
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].y = std::stof(item);

    // Scale
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].scale = std::stof(item);

    // Orientation
    std::getline(feature_line_stream >> std::ws, item, ' ');
    (*keypoints)[i].orientation = std::stof(item);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream >> std::ws, item, ' ');
      const float value = std::stof(item);
      CHECK_GE(value, 0);
      CHECK_LE(value, 255);
      (*descriptors)(i, j) = TruncateCast<float, uint8_t>(value);
    }
  }
}

namespace internal {

ImageResizerThread::ImageResizerThread(const int max_image_size,
                                       JobQueue<ImageData>* input_queue,
                                       JobQueue<ImageData>* output_queue)
    : max_image_size_(max_image_size),
      input_queue_(input_queue),
      output_queue_(output_queue) {}

void ImageResizerThread::Run() {
  while (true) {
    if (IsStopped()) {
      break;
    }

    const auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto image_data = input_job.Data();

      if (image_data.status == ImageReader::Status::SUCCESS) {
        if (static_cast<int>(image_data.bitmap.Width()) > max_image_size_ ||
            static_cast<int>(image_data.bitmap.Height()) > max_image_size_) {
          // Fit the down-sampled version exactly into the max dimensions.
          const double scale =
              static_cast<double>(max_image_size_) /
              std::max(image_data.bitmap.Width(), image_data.bitmap.Height());
          const int new_width =
              static_cast<int>(image_data.bitmap.Width() * scale);
          const int new_height =
              static_cast<int>(image_data.bitmap.Height() * scale);

          image_data.bitmap.Rescale(new_width, new_height);
        }
      }

      output_queue_->Push(image_data);
    } else {
      break;
    }
  }
}

SiftCPUFeatureExtractorThread::SiftCPUFeatureExtractorThread(
    const SiftExtractionOptions& sift_options, JobQueue<ImageData>* input_queue,
    JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(sift_options_.Check());
}

void SiftCPUFeatureExtractorThread::Run() {
  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    const auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto image_data = input_job.Data();

      if (image_data.status == ImageReader::Status::SUCCESS) {
        if (ExtractSiftFeaturesCPU(sift_options_, image_data.bitmap,
                                   &image_data.keypoints,
                                   &image_data.descriptors)) {
          ScaleKeypoints(image_data.bitmap, image_data.camera,
                         &image_data.keypoints);
        } else {
          image_data.status = ImageReader::Status::FAILURE;
        }
      }
      std::cout << StringPrintf("  Image size:      %d",
                                image_data.image.Points2D().size())
                    << std::endl;
      int n_keypoints = RemoveKeypointsByAlpha(image_data.bitmap,
                             &image_data.keypoints,
                             &image_data.descriptors);
      std::cout << StringPrintf("  Keypoints after Alpha Filtering:      %d",
                                n_keypoints)
                    << std::endl;
      if (image_data.bitmap.HasAlpha() && sift_options_.use_alpha ) {
        int n_keypoints = RemoveKeypointsByAlpha(image_data.bitmap,
                               &image_data.keypoints,
                               &image_data.descriptors);
        std::cout << StringPrintf("  Keypoints after Alpha Filtering:      %d",
                                  n_keypoints)
                      << std::endl;
        std::cout << "Has Alpha" << std::endl;
      }

      image_data.bitmap.Deallocate();
      output_queue_->Push(image_data);
    } else {
      break;
    }
  }
}

SiftGPUFeatureExtractorThread::SiftGPUFeatureExtractorThread(
    const SiftExtractionOptions& sift_options, JobQueue<ImageData>* input_queue,
    JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(sift_options_.Check());

#ifndef CUDA_ENABLED
  opengl_context_.reset(new OpenGLContextManager());
#endif
}

void SiftGPUFeatureExtractorThread::Run() {
#ifndef CUDA_ENABLED
  CHECK(opengl_context_);
  opengl_context_->MakeCurrent();
#endif

  SiftGPU sift_gpu;
  if (!CreateSiftGPUExtractor(sift_options_, &sift_gpu)) {
    std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
    SignalInvalidSetup();
    return;
  }

  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    const auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto image_data = input_job.Data();

      if (image_data.status == ImageReader::Status::SUCCESS) {
        if (ExtractSiftFeaturesGPU(sift_options_, image_data.bitmap, &sift_gpu,
                                   &image_data.keypoints,
                                   &image_data.descriptors)) {
          ScaleKeypoints(image_data.bitmap, image_data.camera,
                         &image_data.keypoints);
        } else {
          image_data.status = ImageReader::Status::FAILURE;
        }
      }
      if (image_data.bitmap.HasAlpha() && sift_options_.use_alpha) {
        int n_keypoints = RemoveKeypointsByAlpha(image_data.bitmap,
                               &image_data.keypoints,
                               &image_data.descriptors);
        std::cout << StringPrintf("  Keypoints after Alpha Filtering:      %d",
                                  n_keypoints)
                      << std::endl;
        std::cout << "Has alpha" << std::endl;
      }
      image_data.bitmap.Deallocate();

      output_queue_->Push(image_data);
    } else {
      break;
    }
  }
}

FeatureWriterThread::FeatureWriterThread(const size_t num_images,
                                         Database* database,
                                         JobQueue<ImageData>* input_queue)
    : num_images_(num_images), database_(database), input_queue_(input_queue) {}

void FeatureWriterThread::Run() {
  size_t image_index = 0;
  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& image_data = input_job.Data();

      image_index += 1;

      std::cout << StringPrintf("Processed file [%d/%d]", image_index,
                                num_images_)
                << std::endl;

      std::cout << StringPrintf("  Name:            %s",
                                image_data.image.Name().c_str())
                << std::endl;

      if (image_data.status == ImageReader::Status::IMAGE_EXISTS) {
        std::cout << "  SKIP: Features for image already extracted."
                  << std::endl;
      } else if (image_data.status == ImageReader::Status::BITMAP_ERROR) {
        std::cout << "  ERROR: Failed to read image file format." << std::endl;
      } else if (image_data.status ==
                 ImageReader::Status::CAMERA_SINGLE_ERROR) {
        std::cout << "  ERROR: Single camera specified, "
                     "but images have different dimensions."
                  << std::endl;
      } else if (image_data.status == ImageReader::Status::CAMERA_DIM_ERROR) {
        std::cout << "  ERROR: Image previously processed, but current file "
                     "has different image dimensions."
                  << std::endl;
      } else if (image_data.status == ImageReader::Status::CAMERA_PARAM_ERROR) {
        std::cout << "  ERROR: Camera has invalid parameters." << std::endl;
      } else if (image_data.status == ImageReader::Status::FAILURE) {
        std::cout << "  ERROR: Failed to extract features." << std::endl;
      }

      if (image_data.status != ImageReader::Status::SUCCESS) {
        continue;
      }

      std::cout << StringPrintf("  Dimensions:      %d x %d",
                                image_data.camera.Width(),
                                image_data.camera.Height())
                << std::endl;
      std::cout << StringPrintf("  Camera:          %d (%s)",
                                image_data.camera.CameraId(),
                                image_data.camera.ModelName().c_str())
                << std::endl;
      std::cout << StringPrintf("  Focal Length:    %.2fpx (%s)",
                                image_data.camera.MeanFocalLength(),
                                image_data.camera.HasPriorFocalLength()
                                    ? "EXIF"
                                    : "Default")
                << std::endl;
      if (image_data.image.HasTvecPrior()) {
        std::cout
            << StringPrintf(
                   "  GPS:             LAT=%.3f, LON=%.3f, ALT=%.3f (EXIF)",
                   image_data.image.TvecPrior(0), image_data.image.TvecPrior(1),
                   image_data.image.TvecPrior(2))
            << std::endl;
      }
      std::cout << StringPrintf("  Features:        %d",
                                image_data.keypoints.size())
                << std::endl;

      image_data.PrintKeypoints();

      DatabaseTransaction database_transaction(database_);

      if (image_data.image.ImageId() == kInvalidImageId) {
        image_data.image.SetImageId(database_->WriteImage(image_data.image));
      }

// keypoints should be filtered by alpha at this point
// keypoints and descriptors are defined in feature.h
// descriptors is a keypoints.size() x 128 matrix of uint8_t
      if (!database_->ExistsKeypoints(image_data.image.ImageId())) {
        database_->WriteKeypoints(image_data.image.ImageId(),
                                  image_data.keypoints);
      }

      if (!database_->ExistsDescriptors(image_data.image.ImageId())) {
        database_->WriteDescriptors(image_data.image.ImageId(),
                                    image_data.descriptors);
      }
    } else {
      break;
    }
  }
}

}  // namespace internal
}  // namespace colmap
