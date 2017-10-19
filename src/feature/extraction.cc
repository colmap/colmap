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

#include "feature/extraction.h"

#include "ext/SiftGPU/SiftGPU.h"
#include "feature/sift.h"
#include "util/cuda.h"
#include "util/misc.h"

namespace colmap {
namespace {

void ScaleKeypoints(const Bitmap& bitmap, const Camera& camera,
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

}  // namespace

SiftFeatureExtractor::SiftFeatureExtractor(
    const ImageReaderOptions& reader_options,
    const SiftExtractionOptions& sift_options)
    : reader_options_(reader_options),
      sift_options_(sift_options),
      database_(reader_options_.database_path),
      image_reader_(reader_options_, &database_) {
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

  if (!sift_options_.domain_size_pooling &&
      !sift_options_.estimate_affine_shape && sift_options_.use_gpu) {
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
      extractors_.emplace_back(new internal::SiftFeatureExtractorThread(
          sift_gpu_options, extractor_queue_.get(), writer_queue_.get()));
    }
  } else {
    auto custom_sift_options = sift_options_;
    custom_sift_options.use_gpu = false;
    for (int i = 0; i < num_threads; ++i) {
      extractors_.emplace_back(new internal::SiftFeatureExtractorThread(
          custom_sift_options, extractor_queue_.get(), writer_queue_.get()));
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
      resizer_queue_->Clear();
      extractor_queue_->Clear();
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

FeatureImporter::FeatureImporter(const ImageReaderOptions& reader_options,
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

SiftFeatureExtractorThread::SiftFeatureExtractorThread(
    const SiftExtractionOptions& sift_options, JobQueue<ImageData>* input_queue,
    JobQueue<ImageData>* output_queue)
    : sift_options_(sift_options),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(sift_options_.Check());

#ifndef CUDA_ENABLED
  if (sift_options_.use_gpu) {
    opengl_context_.reset(new OpenGLContextManager());
  }
#endif
}

void SiftFeatureExtractorThread::Run() {
  std::unique_ptr<SiftGPU> sift_gpu;
  if (sift_options_.use_gpu) {
#ifndef CUDA_ENABLED
    CHECK(opengl_context_);
    opengl_context_->MakeCurrent();
#endif

    sift_gpu.reset(new SiftGPU);
    if (!CreateSiftGPUExtractor(sift_options_, sift_gpu.get())) {
      std::cerr << "ERROR: SiftGPU not fully supported." << std::endl;
      SignalInvalidSetup();
      return;
    }
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
        bool success = false;
        if (sift_options_.estimate_affine_shape ||
            sift_options_.domain_size_pooling) {
          success = ExtractCovariantSiftFeaturesCPU(
              sift_options_, image_data.bitmap, &image_data.keypoints,
              &image_data.descriptors);
        } else if (sift_options_.use_gpu) {
          success = ExtractSiftFeaturesGPU(
              sift_options_, image_data.bitmap, sift_gpu.get(),
              &image_data.keypoints, &image_data.descriptors);
        } else {
          success = ExtractSiftFeaturesCPU(sift_options_, image_data.bitmap,
                                           &image_data.keypoints,
                                           &image_data.descriptors);
        }
        if (success) {
          ScaleKeypoints(image_data.bitmap, image_data.camera,
                         &image_data.keypoints);
        } else {
          image_data.status = ImageReader::Status::FAILURE;
        }
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

}  // namespace internal
}  // namespace colmap
