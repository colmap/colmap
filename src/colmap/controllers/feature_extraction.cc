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

#include "colmap/controllers/feature_extraction.h"

#include "colmap/feature/sift.h"
#include "colmap/geometry/gps.h"
#include "colmap/scene/database.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"
#include "colmap/util/timer.h"

#include <numeric>

namespace colmap {
namespace {

void ScaleKeypoints(const Bitmap& bitmap,
                    const Camera& camera,
                    FeatureKeypoints* keypoints) {
  if (static_cast<size_t>(bitmap.Width()) != camera.width ||
      static_cast<size_t>(bitmap.Height()) != camera.height) {
    const float scale_x = static_cast<float>(camera.width) / bitmap.Width();
    const float scale_y = static_cast<float>(camera.height) / bitmap.Height();
    for (auto& keypoint : *keypoints) {
      keypoint.Rescale(scale_x, scale_y);
    }
  }
}

void MaskKeypoints(const Bitmap& mask,
                   FeatureKeypoints* keypoints,
                   FeatureDescriptors* descriptors) {
  size_t out_index = 0;
  BitmapColor<uint8_t> color;
  for (size_t i = 0; i < keypoints->size(); ++i) {
    if (!mask.GetPixel(static_cast<int>(keypoints->at(i).x),
                       static_cast<int>(keypoints->at(i).y),
                       &color) ||
        color.r == 0) {
      // Delete this keypoint by not copying it to the output.
    } else {
      // Retain this keypoint by copying it to the output index (in case this
      // index differs from its current position).
      if (out_index != i) {
        keypoints->at(out_index) = keypoints->at(i);
        for (int col = 0; col < descriptors->cols(); ++col) {
          (*descriptors)(out_index, col) = (*descriptors)(i, col);
        }
      }
      out_index += 1;
    }
  }

  keypoints->resize(out_index);
  descriptors->conservativeResize(out_index, descriptors->cols());
}

enum class FeatureExtractionStatus {
  FAILURE,
  SUCCESS,
  IMAGE_EXISTS,
  BITMAP_ERROR,
};

std::string FeatureExtractionStatusToString(FeatureExtractionStatus status) {
  switch (status) {
    case FeatureExtractionStatus::SUCCESS:
      return "SUCCESS";
    case FeatureExtractionStatus::FAILURE:
      return "FAILURE: Failed to process the image.";
    case FeatureExtractionStatus::IMAGE_EXISTS:
      return "IMAGE_EXISTS: Features for image were already extracted.";
    case FeatureExtractionStatus::BITMAP_ERROR:
      return "BITMAP_ERROR: Failed to read the image file format.";
    default:
      return "Unknown";
  }
}

struct ImageData {
  FeatureExtractionStatus status = FeatureExtractionStatus::SUCCESS;

  Camera camera;
  Image image;
  Bitmap bitmap;
  Bitmap mask;

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

class ImageResizerThread : public Thread {
 public:
  ImageResizerThread(int max_image_size,
                     JobQueue<ImageData>* input_queue,
                     JobQueue<ImageData>* output_queue)
      : max_image_size_(max_image_size),
        input_queue_(input_queue),
        output_queue_(output_queue) {}

 private:
  void Run() override {
    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& image_data = input_job.Data();

        if (image_data.status == FeatureExtractionStatus::SUCCESS) {
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

        output_queue_->Push(std::move(image_data));
      } else {
        break;
      }
    }
  }

  const int max_image_size_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

class FeatureExtractorThread : public Thread {
 public:
  FeatureExtractorThread(const FeatureExtractionOptions& extraction_options,
                         const std::shared_ptr<Bitmap>& camera_mask,
                         JobQueue<ImageData>* input_queue,
                         JobQueue<ImageData>* output_queue)
      : extraction_options_(extraction_options),
        camera_mask_(camera_mask),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    THROW_CHECK(extraction_options_.Check());

#if !defined(COLMAP_CUDA_ENABLED)
    if (extraction_options_.use_gpu) {
      opengl_context_ = std::make_unique<OpenGLContextManager>();
    }
#endif
  }

 private:
  void Run() override {
    if (extraction_options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
      THROW_CHECK_NOTNULL(opengl_context_);
      THROW_CHECK(opengl_context_->MakeCurrent());
#endif
    }

    std::unique_ptr<FeatureExtractor> extractor =
        FeatureExtractor::Create(extraction_options_);
    if (extractor == nullptr) {
      LOG(ERROR) << "Failed to create feature extractor.";
      SignalInvalidSetup();
      return;
    }

    SignalValidSetup();

    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& image_data = input_job.Data();

        if (image_data.status == FeatureExtractionStatus::SUCCESS) {
          if (extractor->Extract(image_data.bitmap,
                                 &image_data.keypoints,
                                 &image_data.descriptors)) {
            ScaleKeypoints(
                image_data.bitmap, image_data.camera, &image_data.keypoints);
            if (camera_mask_) {
              MaskKeypoints(*camera_mask_,
                            &image_data.keypoints,
                            &image_data.descriptors);
            }
            if (image_data.mask.Data()) {
              MaskKeypoints(image_data.mask,
                            &image_data.keypoints,
                            &image_data.descriptors);
            }
          } else {
            image_data.status = FeatureExtractionStatus::FAILURE;
          }
        }

        image_data.bitmap.Deallocate();

        output_queue_->Push(std::move(image_data));
      } else {
        break;
      }
    }
  }

  const FeatureExtractionOptions extraction_options_;
  std::shared_ptr<Bitmap> camera_mask_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

class FeatureWriterThread : public Thread {
 public:
  FeatureWriterThread(FeatureExtractorType extractor_type,
                      Database* database, JobQueue<ImageData>* input_queue)
      : extractor_type_str_(FeatureExtractorTypeToString(extractor_type)),
        database_(database), input_queue_(input_queue) {}

 private:
  void Run() override {
    size_t image_index = 0;
    size_t num_images = database_->NumImages();
    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& image_data = input_job.Data();

        image_index += 1;

        LOG(INFO) << StringPrintf(
            "Processed file [%d/%d]", image_index, num_images);
        LOG(INFO) << StringPrintf("  Name:            %s",
                                  image_data.image.Name().c_str());
        if (image_data.status != FeatureExtractionStatus::SUCCESS) {
          LOG(ERROR) << image_data.image.Name() << " "
                     << FeatureExtractionStatusToString(image_data.status);
          continue;
        }
        LOG(INFO) << "  Features:        " << image_data.keypoints.size()
                  << " (" << extractor_type_str_ << ")";
        if (image_data.mask.Data()) {
          LOG(INFO) << "  Mask:            Yes";
        }

        DatabaseTransaction database_transaction(database_);

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

  const std::string extractor_type_str_;
  Database* database_;
  JobQueue<ImageData>* input_queue_;
};

// Feature extraction class to extract features for all images in a directory.
class FeatureExtractorController : public Thread {
 public:
  FeatureExtractorController(const std::string& database_path,
                             const std::string& image_root_path,
                             const FeatureExtractionOptions& extraction_options)
      : image_root_path_(image_root_path),
        extraction_options_(extraction_options),
        database_(database_path) {
    THROW_CHECK(extraction_options_.Check());

    std::shared_ptr<Bitmap> camera_mask;
    const std::string& camera_mask_path = extraction_options_.camera_mask_path;
    if (!camera_mask_path.empty()) {
      if (ExistsFile(camera_mask_path)) {
        camera_mask = std::make_shared<Bitmap>();
        if (!camera_mask->Read(camera_mask_path,
                               /*as_rgb=*/false)) {
          LOG(ERROR) << "Failed to read invalid mask file at: "
                     << camera_mask_path
                     << ". No mask is going to be used.";
          camera_mask.reset();
        }
      } else {
        LOG(ERROR) << "Mask at " << camera_mask_path
                   << " does not exist.";
      }
    }

    const int num_threads =
        GetEffectiveNumThreads(extraction_options_.num_threads);
    THROW_CHECK_GT(num_threads, 0);

    // Make sure that we only have limited number of objects in the queue to
    // avoid excess in memory usage since images and features take lots of
    // memory.
    const int kQueueSize = 1;
    resizer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
    extractor_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
    writer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);

    const int max_image_size = extraction_options_.MaxImageSize();
    if (max_image_size > 0) {
      for (int i = 0; i < num_threads; ++i) {
        resizers_.emplace_back(std::make_unique<ImageResizerThread>(
            max_image_size, resizer_queue_.get(), extractor_queue_.get()));
      }
    }

    if (!extraction_options_.sift->domain_size_pooling &&
        !extraction_options_.sift->estimate_affine_shape &&
        extraction_options_.use_gpu) {
      std::vector<int> gpu_indices =
          CSVToVector<int>(extraction_options_.gpu_index);
      THROW_CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED)
      if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        THROW_CHECK_GT(num_cuda_devices, 0);
        gpu_indices.resize(num_cuda_devices);
        std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
      }
#endif  // COLMAP_CUDA_ENABLED

      auto sift_gpu_options = extraction_options_;
      for (const auto& gpu_index : gpu_indices) {
        sift_gpu_options.gpu_index = std::to_string(gpu_index);
        extractors_.emplace_back(
            std::make_unique<FeatureExtractorThread>(sift_gpu_options,
                                                     camera_mask,
                                                     extractor_queue_.get(),
                                                     writer_queue_.get()));
      }
    } else {
      const static FeatureExtractionOptions kDefaultExtractionOptions;
      if (extraction_options_.num_threads == -1 &&
          max_image_size == kDefaultExtractionOptions.MaxImageSize() &&
          extraction_options_.sift->first_octave ==
              kDefaultExtractionOptions.sift->first_octave) {
        LOG(WARNING)
            << "Your current options use the maximum number of "
               "threads on the machine to extract features. Extracting SIFT "
               "features on the CPU can consume a lot of RAM per thread for "
               "large images. Consider reducing the maximum image size and/or "
               "the first octave or manually limit the number of extraction "
               "threads. Ignore this warning, if your machine has sufficient "
               "memory for the current settings.";
      }

      auto custom_extraction_options = extraction_options_;
      custom_extraction_options.use_gpu = false;
      for (int i = 0; i < num_threads; ++i) {
        extractors_.emplace_back(
            std::make_unique<FeatureExtractorThread>(custom_extraction_options,
                                                     camera_mask,
                                                     extractor_queue_.get(),
                                                     writer_queue_.get()));
      }
    }

    writer_ = std::make_unique<FeatureWriterThread>(extraction_options_.type,
                                                    &database_,
                                                    writer_queue_.get());
  }

 private:
  ImageData ReadImageData(const Image& image) {
    ImageData image_data;
    image_data.image = image;
    image_data.camera = database_.ReadCamera(image.CameraId());

    {
      DatabaseTransaction database_transaction(&database_);
      const bool exists_keypoints = database_.ExistsKeypoints(image.ImageId());
      const bool exists_descriptors =
          database_.ExistsDescriptors(image.ImageId());
      if (exists_keypoints && exists_descriptors) {
        image_data.status = FeatureExtractionStatus::IMAGE_EXISTS;
        return image_data;
      }
    }

    // Construct full image path and load bitmap
    const std::string image_path = JoinPaths(image_root_path_, image.Name());
    if (!image_data.bitmap.Read(image_path, false)) {
      LOG(ERROR) << image.Name() << " Failed to read the image file format.";
      image_data.status = FeatureExtractionStatus::FAILURE;
      return image_data;
    }

    // Load mask if path is provided
    if (!extraction_options_.mask_path.empty()) {
      std::string mask_path =
          JoinPaths(extraction_options_.mask_path, image.Name() + ".png");
      bool exists_mask = true;
      if (!ExistsFile(mask_path)) {
        exists_mask = false;
        // Try without .png extension if original image is .png
        if (HasFileExtension(image.Name(), ".png")) {
          std::string alt_mask_path =
              JoinPaths(extraction_options_.mask_path, image.Name());
          if (ExistsFile(alt_mask_path)) {
            mask_path = std::move(alt_mask_path);
            exists_mask = true;
          }
        }
      }
      if (exists_mask) {
        if (!image_data.mask.Read(mask_path, false)) {
          LOG(ERROR) << image.Name() << " Failed to read the mask file";
          image_data.mask.Deallocate();  // Skip the mask but not the image!
        }
      } else {
        LOG(WARNING) << "Mask for " << image.Name() << " not found at "
                     << mask_path;
      }
    }
    return image_data;
  }

  void Run() override {
    PrintHeading1("Feature extraction");
    Timer run_timer;
    run_timer.Start();

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

    const bool should_resize = extraction_options_.MaxImageSize() > 0;

    for (const auto& image : database_.ReadAllImages()) {
      if (IsStopped()) {
        resizer_queue_->Stop();
        extractor_queue_->Stop();
        resizer_queue_->Clear();
        extractor_queue_->Clear();
        break;
      }
      ImageData image_data = ReadImageData(image);
      if (should_resize) {
        THROW_CHECK(resizer_queue_->Push(std::move(image_data)));
      } else {
        THROW_CHECK(extractor_queue_->Push(std::move(image_data)));
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

    run_timer.PrintMinutes();
  }

  const std::string image_root_path_;
  const FeatureExtractionOptions extraction_options_;

  Database database_;

  std::vector<std::unique_ptr<Thread>> resizers_;
  std::vector<std::unique_ptr<Thread>> extractors_;
  std::unique_ptr<Thread> writer_;

  std::unique_ptr<JobQueue<ImageData>> resizer_queue_;
  std::unique_ptr<JobQueue<ImageData>> extractor_queue_;
  std::unique_ptr<JobQueue<ImageData>> writer_queue_;
};

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
// Currently hard-coded to support SIFT features.
class FeatureImporterController : public Thread {
 public:
  FeatureImporterController(const std::string& database_path,
                            const std::string& import_path)
      : database_path_(database_path), import_path_(import_path) {}

 private:
  void Run() override {
    PrintHeading1("Feature import");
    Timer run_timer;
    run_timer.Start();

    if (!ExistsDir(import_path_)) {
      LOG(ERROR) << "Import directory does not exist.";
      return;
    }

    Database database(database_path_);

    const std::vector<Image> images = database.ReadAllImages();
    size_t image_index = 0;
    for (const auto& image : images) {
      if (IsStopped()) {
        break;
      }

      image_index += 1;

      LOG(INFO) << StringPrintf(
          "Processing file [%d/%d]", image_index, images.size());

      const std::string path = JoinPaths(import_path_, image.Name() + ".txt");

      if (ExistsFile(path)) {
        FeatureKeypoints keypoints;
        FeatureDescriptors descriptors;
        LoadSiftFeaturesFromTextFile(path, &keypoints, &descriptors);

        LOG(INFO) << "Features:       " << keypoints.size()
                  << "(Imported SIFT)";

        DatabaseTransaction database_transaction(&database);

        if (!database.ExistsKeypoints(image.ImageId())) {
          database.WriteKeypoints(image.ImageId(), keypoints);
        }

        if (!database.ExistsDescriptors(image.ImageId())) {
          database.WriteDescriptors(image.ImageId(), descriptors);
        }
      } else {
        LOG(INFO) << "SKIP: No features found at " << path;
      }
    }

    run_timer.PrintMinutes();
  }

  const std::string database_path_;
  const std::string import_path_;
};

}  // namespace

std::unique_ptr<Thread> CreateFeatureExtractorController(
    const std::string& database_path,
    const std::string& image_root_path,
    const FeatureExtractionOptions& extraction_options) {
  return std::make_unique<FeatureExtractorController>(database_path,
                                                      image_root_path,
                                                      extraction_options);
}

std::unique_ptr<Thread> CreateFeatureImporterController(
    const std::string& database_path, const std::string& import_path) {
  return std::make_unique<FeatureImporterController>(database_path,
                                                     import_path);
}

}  // namespace colmap
