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

#ifndef COLMAP_SRC_BASE_FEATURE_EXTRACTION_H_
#define COLMAP_SRC_BASE_FEATURE_EXTRACTION_H_

#include "base/database.h"
#include "base/image_reader.h"
#include "util/bitmap.h"
#include "util/opengl_utils.h"
#include "util/threading.h"

class SiftGPU;

namespace colmap {

struct SiftExtractionOptions {
  // Number of threads for feature extraction.
  int num_threads = ThreadPool::kMaxNumThreads;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 3200;

  // Maximum number of features to detect, keeping larger-scale features.
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level.
  int first_octave = -1;

  // Number of octaves.
  int num_octaves = 4;

  // Number of levels per octave.
  int octave_resolution = 3;

  // Peak threshold for detection.
  double peak_threshold = 0.02 / octave_resolution;

  // Edge threshold for detection.
  double edge_threshold = 10.0;

  // Maximum number of orientations per keypoint.
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features.
  bool upright = false;

  // Whether to adapt the feature detection depending on the image darkness.
  // Note that this feature is only available in the OpenGL SiftGPU version.
  bool darkness_adaptivity = false;

  enum class Normalization {
    // L1-normalizes each descriptor followed by element-wise square rooting.
    // This normalization is usually better than standard L2-normalization.
    // See "Three things everyone should know to improve object retrieval",
    // Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
    L1_ROOT,
    // Each vector is L2-normalized.
    L2,
  };
  Normalization normalization = Normalization::L1_ROOT;

  bool Check() const;
};

namespace internal {

struct ImageData;

}  // namespace internal

// Feature extraction class to extract features for all images in a directory.
class SiftFeatureExtractor : public Thread {
 public:
  SiftFeatureExtractor(const ImageReader::Options& reader_options,
                       const SiftExtractionOptions& sift_options);

 private:
  void Run();

  const ImageReader::Options reader_options_;
  const SiftExtractionOptions sift_options_;

  Database database_;
  ImageReader image_reader_;

  std::vector<std::unique_ptr<Thread>> resizers_;
  std::vector<std::unique_ptr<Thread>> extractors_;
  std::unique_ptr<Thread> writer_;

  std::unique_ptr<JobQueue<internal::ImageData>> resizer_queue_;
  std::unique_ptr<JobQueue<internal::ImageData>> extractor_queue_;
  std::unique_ptr<JobQueue<internal::ImageData>> writer_queue_;
};

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
class FeatureImporter : public Thread {
 public:
  FeatureImporter(const ImageReader::Options& reader_options,
                  const std::string& import_path);

 private:
  void Run();

  const ImageReader::Options reader_options_;
  const std::string import_path_;
};

// Extract SIFT features for the given image on the CPU.
bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& sift_options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors);

// Create a SiftGPU feature extractor. The same SiftGPU instance can be used to
// extract features for multiple images. Note a OpenGL context must be made
// current in the thread of the caller. If the gpu_index is not -1, the CUDA
// version of SiftGPU is used, which produces slightly different results
// than the OpenGL implementation.
bool CreateSiftGPUExtractor(const SiftExtractionOptions& sift_options,
                            SiftGPU* sift_gpu);

// Extract SIFT features for the given image on the GPU.
// SiftGPU must already be initialized using `CreateSiftGPU`.
bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& sift_options,
                            const Bitmap& bitmap, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors);

// Load keypoints and descriptors from text file in the following format:
//
//    LINE_0:            NUM_FEATURES DIM
//    LINE_1:            X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    LINE_I:            ...
//    LINE_NUM_FEATURES: X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//
// where the first line specifies the number of features and the descriptor
// dimensionality followed by one line per feature: X, Y, SCALE, ORIENTATION are
// of type float and D_J represent the descriptor in the range [0, 255].
//
// For example:
//
//    2 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//
void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

struct ImageData {
  ImageReader::Status status = ImageReader::Status::FAILURE;

  Camera camera;
  Image image;
  Bitmap bitmap;

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
};

class ImageResizerThread : public Thread {
 public:
  ImageResizerThread(const int max_image_size, JobQueue<ImageData>* input_queue,
                     JobQueue<ImageData>* output_queue);

 private:
  void Run();

  const int max_image_size_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

// Extract DoG SIFT features using the CPU.
class SiftCPUFeatureExtractorThread : public Thread {
 public:
  SiftCPUFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                                JobQueue<ImageData>* input_queue,
                                JobQueue<ImageData>* output_queue);

 private:
  void Run();

  const SiftExtractionOptions sift_options_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

// Extract DoG SIFT features using the GPU.
class SiftGPUFeatureExtractorThread : public Thread {
 public:
  SiftGPUFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                                JobQueue<ImageData>* input_queue,
                                JobQueue<ImageData>* output_queue);

 private:
  void Run();

  const SiftExtractionOptions sift_options_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  JobQueue<ImageData>* input_queue_;
  JobQueue<ImageData>* output_queue_;
};

class FeatureWriterThread : public Thread {
 public:
  FeatureWriterThread(const size_t num_images, Database* database,
                      JobQueue<ImageData>* input_queue);

 private:
  void Run();

  const size_t num_images_;
  Database* database_;
  JobQueue<ImageData>* input_queue_;
};

}  // namespace internal

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_FEATURE_EXTRACTION_H_
