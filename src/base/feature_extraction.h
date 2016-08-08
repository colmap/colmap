// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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
#include "ext/SiftGPU/SiftGPU.h"
#include "util/bitmap.h"
#include "util/opengl_utils.h"
#include "util/threading.h"

namespace colmap {

struct SiftOptions {
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

  void Check() const;
};

class ImageReader {
 public:
  struct Options {
    // Path to database in which to store the extracted data.
    std::string database_path = "";

    // Root path to folder which contains the image.
    std::string image_path = "";

    // Name of the camera model.
    std::string camera_model = "SIMPLE_RADIAL";

    // Whether to use the same camera for all images.
    bool single_camera = false;

    // Specification of manual camera parameters. If empty, camera parameters
    // will be extracted from EXIF, i.e. principal point and focal length.
    std::string camera_params = "";

    // If camera parameters are not specified manually and the image does not
    // have focal length EXIF information, the focal length is set to the
    // value `default_focal_length_factor * max(width, height)`.
    double default_focal_length_factor = 1.2;

    void Check() const;
  };

  ImageReader(const Options& options);

  bool Next(Image* image, Bitmap* bitmap);
  size_t NextIndex() const;
  size_t NumImages() const;

 private:
  // Image reader options.
  Options options_;
  // List of images in the folder.
  std::vector<std::string> image_list_;
  // Index of previously processed image.
  size_t image_index_;
  // Previously processed camera.
  Camera prev_camera_;
};

// Extract DoG SIFT features using the CPU.
class SiftCPUFeatureExtractor : public Thread {
 public:
  struct Options {
    // Number of images to process in one batch,
    // defined as a factor of the number of threads.
    int batch_size_factor = 3;

    // Number of threads for parallel feature extraction.
    int num_threads = -1;

    void Check() const;
  };

  SiftCPUFeatureExtractor(const ImageReader::Options& reader_options,
                          const SiftOptions& sift_options,
                          const Options& cpu_options);

 private:
  void Run();

  ImageReader::Options reader_options_;
  SiftOptions sift_options_;
  Options cpu_options_;
};

// Extract DoG SIFT features using the GPU.
class SiftGPUFeatureExtractor : public Thread {
 public:
  SiftGPUFeatureExtractor(const ImageReader::Options& reader_options,
                          const SiftOptions& sift_options);

 private:
  void Run();

  ImageReader::Options reader_options_;
  SiftOptions sift_options_;
  OpenGLContextManager opengl_context_;
};

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
class FeatureImporter : public Thread {
 public:
  FeatureImporter(const ImageReader::Options& reader_options,
                  const std::string& import_path);

 private:
  void Run();

  ImageReader::Options reader_options_;
  std::string import_path_;
};

// Extract SIFT features for the given image on the CPU.
bool ExtractSiftFeaturesCPU(const SiftOptions& sift_options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors);

// Create a SiftGPU feature extractor. Note that the OpenGLContextManager must
// be created in the main thread of the Qt application. The same SiftGPU
// instance can be used to extract features for  multiple images.
bool CreateSiftGPU(const SiftOptions& sift_options,
                   OpenGLContextManager* opengl_context, SiftGPU* sift_gpu);

// Extract SIFT features for the given image on the GPU.
// SiftGPU must already be initialized using `CreateSiftGPU`.
bool ExtractSiftFeaturesGPU(const SiftOptions& sift_options,
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

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_FEATURE_EXTRACTION_H_
