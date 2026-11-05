// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift.h"

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#ifdef COLMAP_CUSIFT_ENABLED
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <cmath>

#ifdef COLMAP_CUSIFT_ENABLED
#include <array>
#include <string>
#include <vector>

// No <cuda_runtime.h> or cudautils.h here: every CUDA type stays inside
// colmap_cusift (see cusift_device.h); device selection is COLMAP's.
#include "cudaImage.h"
#include "cudaSift.h"
#include "cusift_device.h"
#endif  // COLMAP_CUSIFT_ENABLED

namespace colmap {
namespace {

#ifdef COLMAP_CUSIFT_ENABLED
constexpr int kSiftDescriptorDim = 128;
#endif  // COLMAP_CUSIFT_ENABLED

// cusift's compile-time levels-per-octave (NUM_SCALES in cudaSiftD.h).
constexpr int kCusiftOctaveResolution = 3;

#ifdef COLMAP_CUSIFT_ENABLED

// Uploads a grey bitmap as the [0, 255] float image cusift's kernels take.
// cusift scales by 1/255 in its first kernel, where COLMAP's VLFeat path
// divides by 255 on the host; the two can differ in the last bit, which the
// VLFeat parity tests tolerate. `host_buffer` backs the image's host side and
// must outlive `image`.
void UploadGreyBitmap(const Bitmap& bitmap,
                      std::vector<float>* host_buffer,
                      CudaImage* image) {
  THROW_CHECK(bitmap.IsGrey());
  const std::vector<uint8_t>& data_uint8 = bitmap.RowMajorData();
  host_buffer->resize(data_uint8.size());
  std::transform(data_uint8.begin(),
                 data_uint8.end(),
                 host_buffer->begin(),
                 [](const uint8_t v) { return static_cast<float>(v); });
  image->Allocate(bitmap.Width(),
                  bitmap.Height(),
                  iAlignUp(bitmap.Width(), 128),
                  /*devMem=*/nullptr,
                  /*hostMem=*/host_buffer->data());
  image->Download();
}

// The VLFeat CPU path's final step: cusift returns what
// vl_sift_calc_keypoint_descriptor does.
FeatureDescriptorsData NormalizeAndQuantize(
    const SiftExtractionOptions::Normalization normalization,
    FeatureDescriptorsFloatData* descriptors_float) {
  if (normalization == SiftExtractionOptions::Normalization::L2) {
    L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (normalization == SiftExtractionOptions::Normalization::L1_ROOT) {
    L1RootNormalizeFeatureDescriptors(descriptors_float);
  } else {
    LOG(FATAL_THROW) << "Normalization type not supported";
  }
  return FeatureDescriptorsToUnsignedByte(*descriptors_float);
}

class CusiftFeatureExtractor : public FeatureExtractor {
 public:
  explicit CusiftFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options) {
    // Includes CheckCusiftSiftOptions.
    THROW_CHECK(options_.Check())
        << "invalid options for the cusift extractor; the reason is logged "
           "above";
  }

  // Frees the scratch, then returns the memory cusift's pool holds but no
  // longer uses to the driver, so GPU stages that run after extraction can
  // have it. COLMAP destroys the extractor on the thread that extracted,
  // which is bound to the extractor's GPU.
  ~CusiftFeatureExtractor() override {
    scratch_.Release();
    cusift::ReleaseUnusedMemory();
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    BindCusiftDevice(options.gpu_index);
    return std::make_unique<CusiftFeatureExtractor>(options);
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);

    std::vector<float> host_buffer;
    CudaImage image;
    UploadGreyBitmap(bitmap, &host_buffer, &image);

    // Absolute coordinates without the +0.5 offset, orientation in VLFeat's
    // convention, descriptors normalized and in UBC layout.
    x_.clear();
    y_.clear();
    sigma_.clear();
    orientation_.clear();
    descriptors_.clear();
    ExtractSiftFullVLFeat(
        scratch_,
        image,
        CusiftEffectiveNumOctaves(*THROW_CHECK_NOTNULL(options_.cusift),
                                  bitmap.Width(),
                                  bitmap.Height(),
                                  options_.sift->first_octave),
        static_cast<float>(options_.sift->peak_threshold),
        static_cast<float>(options_.sift->edge_threshold),
        options_.sift->max_num_orientations,
        options_.sift->first_octave,
        options_.sift->max_num_features,
        x_,
        y_,
        sigma_,
        orientation_,
        descriptors_);

    const size_t num_features = x_.size();

    // +0.5 pixel-centre offset and VLFeat-convention orientation, as the
    // VLFeat CPU backend stores them. SiftGPU stores a mirrored orientation
    // (SiftGPUFeatureExtractor); cusift deliberately follows VLFeat.
    keypoints->resize(num_features);
    for (size_t i = 0; i < num_features; ++i) {
      (*keypoints)[i] = FeatureKeypoint(
          x_[i] + 0.5f, y_[i] + 0.5f, sigma_[i], orientation_[i]);
    }

    if (descriptors != nullptr) {
      FeatureDescriptorsFloatData descriptors_float(num_features,
                                                    kSiftDescriptorDim);
      for (size_t i = 0; i < num_features; ++i) {
        descriptors_float.row(i) = Eigen::Map<const Eigen::RowVectorXf>(
            descriptors_[i].data(), kSiftDescriptorDim);
      }

      // Already in UBC order: no TransformVLFeatToUBCFeatureDescriptors.
      descriptors->data = NormalizeAndQuantize(options_.sift->normalization,
                                               &descriptors_float);
      descriptors->type = FeatureExtractorType::SIFT;
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  // Kernel I/O buffers reused across images; freed in the destructor.
  SiftScratch scratch_;

  // Reused across images to avoid reallocating per call.
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> sigma_;
  std::vector<float> orientation_;
  std::vector<std::array<float, kSiftDescriptorDim>> descriptors_;
};

#endif  // COLMAP_CUSIFT_ENABLED

}  // namespace

int CusiftNumOctaves(const int width,
                     const int height,
                     const int first_octave) {
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  // SiftGPU upsamples by 2^-first_octave (or downsamples by 2^first_octave)
  // before counting, so each step of first_octave shifts the count by one.
  const int min_side = std::min(width, height);
  const int num_octaves =
      static_cast<int>(std::floor(std::log2(static_cast<double>(min_side)))) -
      first_octave - 3;
  return std::max(num_octaves, 1);
}

int CusiftEffectiveNumOctaves(const CusiftExtractionOptions& options,
                              const int width,
                              const int height,
                              const int first_octave) {
  return options.num_octaves > 0
             ? options.num_octaves
             : CusiftNumOctaves(width, height, first_octave);
}

bool CusiftExtractionOptions::Check() const {
  CHECK_OPTION(num_octaves == -1 || num_octaves >= 1);
  return true;
}

bool CheckCusiftSiftOptions(const SiftExtractionOptions& sift) {
  const auto reject = [](const char* reason) {
    LOG(ERROR) << "cusift " << reason;
    return false;
  };
  if (sift.estimate_affine_shape) {
    return reject("does not implement affine shape estimation");
  }
  if (sift.domain_size_pooling) {
    return reject("does not implement domain-size pooling");
  }
  if (sift.upright) {
    return reject("does not implement upright features");
  }
  if (sift.darkness_adaptivity) {
    return reject("does not implement darkness adaptivity");
  }
  if (sift.force_covariant_extractor) {
    return reject("is not a covariant extractor");
  }
  // VLFeat's o_min. cusift implements the 2x-upsampled extra octave (-1),
  // which is COLMAP's default, and the plain first octave (0); nothing else.
  if (sift.first_octave != 0 && sift.first_octave != -1) {
    LOG(ERROR) << "cusift supports first_octave 0 or -1, got "
               << sift.first_octave;
    return false;
  }
  if (sift.octave_resolution != kCusiftOctaveResolution) {
    LOG(ERROR) << "cusift's levels per octave are fixed at "
               << kCusiftOctaveResolution << ", got " << sift.octave_resolution;
    return false;
  }
  return true;
}

void BindCusiftDevice(const std::string& gpu_index) {
#ifdef COLMAP_CUSIFT_ENABLED
  const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1) << "cusift can only run on one GPU";
  SetBestCudaDevice(gpu_indices[0]);
  cusift::InitMemoryPool();
#else
  (void)gpu_index;
  LOG(FATAL_THROW) << "COLMAP was built without cusift (CUSIFT_ENABLED, which "
                      "needs CUDA >= 11.2)";
#endif  // COLMAP_CUSIFT_ENABLED
}

std::unique_ptr<FeatureExtractor> CreateCusiftFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_CUSIFT_ENABLED
  LOG(INFO) << "Creating cusift GPU feature extractor";
  return CusiftFeatureExtractor::Create(options);
#else
  LOG(ERROR) << "COLMAP was built without cusift (CUSIFT_ENABLED, which needs "
                "CUDA >= 11.2)";
  return nullptr;
#endif  // COLMAP_CUSIFT_ENABLED
}

}  // namespace colmap
