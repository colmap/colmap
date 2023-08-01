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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/controllers/feature_extraction.h"

#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

#include <numeric>

namespace colmap {
namespace {

void ScaleKeypoints(const Bitmap& bitmap,
                    const Camera& camera,
                    FeatureKeypoints* keypoints) {
  if (static_cast<size_t>(bitmap.Width()) != camera.Width() ||
      static_cast<size_t>(bitmap.Height()) != camera.Height()) {
    const float scale_x = static_cast<float>(camera.Width()) / bitmap.Width();
    const float scale_y = static_cast<float>(camera.Height()) / bitmap.Height();
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

struct ImageData {
  ImageReader::Status status = ImageReader::Status::FAILURE;

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

class SiftFeatureExtractorThread : public Thread {
 public:
  SiftFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                             const std::shared_ptr<Bitmap>& camera_mask,
                             JobQueue<ImageData>* input_queue,
                             JobQueue<ImageData>* output_queue)
      : sift_options_(sift_options),
        camera_mask_(camera_mask),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    CHECK(sift_options_.Check());

#if !defined(COLMAP_CUDA_ENABLED)
    if (sift_options_.use_gpu) {
      opengl_context_ = std::make_unique<OpenGLContextManager>();
    }
#endif
  }

 private:
  void Run() override {
    if (sift_options_.use_gpu) {
#if !defined(COLMAP_CUDA_ENABLED)
      CHECK(opengl_context_);
      CHECK(opengl_context_->MakeCurrent());
#endif
    }

    std::unique_ptr<FeatureExtractor> extractor =
        CreateSiftFeatureExtractor(sift_options_);
    if (extractor == nullptr) {
      std::cerr << "ERROR: Failed to create feature extractor." << std::endl;
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

        if (image_data.status == ImageReader::Status::SUCCESS) {
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
            image_data.status = ImageReader::Status::FAILURE;
          }
        }

        image_data.bitmap.Deallocate();

        output_queue_->Push(std::move(image_data));
      } else {
        break;
      }
    }
  }

  const SiftExtractionOptions sift_options_;
  std::shared_ptr<Bitmap> camera_mask_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

class FeatureWriterThread : public Thread {
 public:
  FeatureWriterThread(size_t num_images,
                      Database* database,
                      JobQueue<ImageData>* input_queue)
      : num_images_(num_images),
        database_(database),
        input_queue_(input_queue) {}

 private:
  void Run() override {
    size_t image_index = 0;
    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& image_data = input_job.Data();

        image_index += 1;

        std::cout << StringPrintf(
                         "Processed file [%d/%d]", image_index, num_images_)
                  << std::endl;

        std::cout << StringPrintf("  Name:            %s",
                                  image_data.image.Name().c_str())
                  << std::endl;

        if (image_data.status == ImageReader::Status::IMAGE_EXISTS) {
          std::cout << "  SKIP: Features for image already extracted."
                    << std::endl;
        } else if (image_data.status == ImageReader::Status::BITMAP_ERROR) {
          std::cout << "  ERROR: Failed to read image file format."
                    << std::endl;
        } else if (image_data.status ==
                   ImageReader::Status::CAMERA_SINGLE_DIM_ERROR) {
          std::cout << "  ERROR: Single camera specified, "
                       "but images have different dimensions."
                    << std::endl;
        } else if (image_data.status ==
                   ImageReader::Status::CAMERA_EXIST_DIM_ERROR) {
          std::cout << "  ERROR: Image previously processed, but current image "
                       "has different dimensions."
                    << std::endl;
        } else if (image_data.status ==
                   ImageReader::Status::CAMERA_PARAM_ERROR) {
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
        std::cout << StringPrintf("  Camera:          #%d - %s",
                                  image_data.camera.CameraId(),
                                  image_data.camera.ModelName().c_str())
                  << std::endl;
        std::cout << StringPrintf("  Focal Length:    %.2fpx",
                                  image_data.camera.MeanFocalLength());
        if (image_data.camera.HasPriorFocalLength()) {
          std::cout << " (Prior)" << std::endl;
        } else {
          std::cout << std::endl;
        }
        const Eigen::Vector3d& translation_prior =
            image_data.image.CamFromWorldPrior().translation;
        if (translation_prior.array().isFinite().any()) {
          std::cout << StringPrintf(
                           "  GPS:             LAT=%.3f, LON=%.3f, ALT=%.3f",
                           translation_prior.x(),
                           translation_prior.y(),
                           translation_prior.z())
                    << std::endl;
        }
        std::cout << StringPrintf("  Features:        %d",
                                  image_data.keypoints.size())
                  << std::endl;

        DatabaseTransaction database_transaction(database_);

        if (image_data.image.ImageId() == kInvalidImageId) {
          image_data.image.SetImageId(database_->WriteImage(image_data.image));
        }

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

  const size_t num_images_;
  Database* database_;
  JobQueue<ImageData>* input_queue_;
};

// Feature extraction class to extract features for all images in a directory.
class FeatureExtractorController : public Thread {
 public:
  FeatureExtractorController(const ImageReaderOptions& reader_options,
                             const SiftExtractionOptions& sift_options)
      : reader_options_(reader_options),
        sift_options_(sift_options),
        database_(reader_options_.database_path),
        image_reader_(reader_options_, &database_) {
    CHECK(reader_options_.Check());
    CHECK(sift_options_.Check());

    std::shared_ptr<Bitmap> camera_mask;
    if (!reader_options_.camera_mask_path.empty()) {
      camera_mask = std::make_shared<Bitmap>();
      if (!camera_mask->Read(reader_options_.camera_mask_path,
                             /*as_rgb*/ false)) {
        std::cerr << "  ERROR: Cannot read camera mask file: "
                  << reader_options_.camera_mask_path
                  << ". No mask is going to be used." << std::endl;
        camera_mask.reset();
      }
    }

    const int num_threads = GetEffectiveNumThreads(sift_options_.num_threads);
    CHECK_GT(num_threads, 0);

    // Make sure that we only have limited number of objects in the queue to
    // avoid excess in memory usage since images and features take lots of
    // memory.
    const int kQueueSize = 1;
    resizer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
    extractor_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);
    writer_queue_ = std::make_unique<JobQueue<ImageData>>(kQueueSize);

    if (sift_options_.max_image_size > 0) {
      for (int i = 0; i < num_threads; ++i) {
        resizers_.emplace_back(
            std::make_unique<ImageResizerThread>(sift_options_.max_image_size,
                                                 resizer_queue_.get(),
                                                 extractor_queue_.get()));
      }
    }

    if (!sift_options_.domain_size_pooling &&
        !sift_options_.estimate_affine_shape && sift_options_.use_gpu) {
      std::vector<int> gpu_indices = CSVToVector<int>(sift_options_.gpu_index);
      CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED)
      if (gpu_indices.size() == 1 && gpu_indices[0] == -1) {
        const int num_cuda_devices = GetNumCudaDevices();
        CHECK_GT(num_cuda_devices, 0);
        gpu_indices.resize(num_cuda_devices);
        std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
      }
#endif  // COLMAP_CUDA_ENABLED

      auto sift_gpu_options = sift_options_;
      for (const auto& gpu_index : gpu_indices) {
        sift_gpu_options.gpu_index = std::to_string(gpu_index);
        extractors_.emplace_back(
            std::make_unique<SiftFeatureExtractorThread>(sift_gpu_options,
                                                         camera_mask,
                                                         extractor_queue_.get(),
                                                         writer_queue_.get()));
      }
    } else {
      if (sift_options_.num_threads == -1 &&
          sift_options_.max_image_size ==
              SiftExtractionOptions().max_image_size &&
          sift_options_.first_octave == SiftExtractionOptions().first_octave) {
        std::cout
            << "WARNING: Your current options use the maximum number of "
               "threads on the machine to extract features. Extracting SIFT "
               "features on the CPU can consume a lot of RAM per thread for "
               "large images. Consider reducing the maximum image size and/or "
               "the first octave or manually limit the number of extraction "
               "threads. Ignore this warning, if your machine has sufficient "
               "memory for the current settings."
            << std::endl;
      }

      auto custom_sift_options = sift_options_;
      custom_sift_options.use_gpu = false;
      for (int i = 0; i < num_threads; ++i) {
        extractors_.emplace_back(
            std::make_unique<SiftFeatureExtractorThread>(custom_sift_options,
                                                         camera_mask,
                                                         extractor_queue_.get(),
                                                         writer_queue_.get()));
      }
    }

    writer_ = std::make_unique<FeatureWriterThread>(
        image_reader_.NumImages(), &database_, writer_queue_.get());
  }

 private:
  void Run() override {
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
        resizer_queue_->Clear();
        extractor_queue_->Clear();
        break;
      }

      ImageData image_data;
      image_data.status = image_reader_.Next(&image_data.camera,
                                             &image_data.image,
                                             &image_data.bitmap,
                                             &image_data.mask);

      if (image_data.status != ImageReader::Status::SUCCESS) {
        image_data.bitmap.Deallocate();
      }

      if (sift_options_.max_image_size > 0) {
        CHECK(resizer_queue_->Push(std::move(image_data)));
      } else {
        CHECK(extractor_queue_->Push(std::move(image_data)));
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

  const ImageReaderOptions reader_options_;
  const SiftExtractionOptions sift_options_;

  Database database_;
  ImageReader image_reader_;

  std::vector<std::unique_ptr<Thread>> resizers_;
  std::vector<std::unique_ptr<Thread>> extractors_;
  std::unique_ptr<Thread> writer_;

  std::unique_ptr<JobQueue<ImageData>> resizer_queue_;
  std::unique_ptr<JobQueue<ImageData>> extractor_queue_;
  std::unique_ptr<JobQueue<ImageData>> writer_queue_;
};

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
class FeatureImporterController : public Thread {
 public:
  FeatureImporterController(const ImageReaderOptions& reader_options,
                            const std::string& import_path)
      : reader_options_(reader_options), import_path_(import_path) {}

 private:
  void Run() override {
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
      if (image_reader.Next(&camera, &image, &bitmap, nullptr) !=
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

  const ImageReaderOptions reader_options_;
  const std::string import_path_;
};

}  // namespace

std::unique_ptr<Thread> CreateFeatureExtractorController(
    const ImageReaderOptions& reader_options,
    const SiftExtractionOptions& sift_options) {
  return std::make_unique<FeatureExtractorController>(reader_options,
                                                      sift_options);
}

std::unique_ptr<Thread> CreateFeatureImporterController(
    const ImageReaderOptions& reader_options, const std::string& import_path) {
  return std::make_unique<FeatureImporterController>(reader_options,
                                                     import_path);
}

}  // namespace colmap
