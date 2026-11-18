// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/cusift.h"

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#ifdef COLMAP_CUSIFT_ENABLED
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <cmath>

#ifdef COLMAP_CUSIFT_ENABLED
#include <array>
#include <exception>
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

// cusift's compile-time levels-per-octave (NUM_SCALES in cudaSiftD.h).
constexpr int kCusiftOctaveResolution = 3;

// floor(log2(min side)) of the first octave: the image is 2x-upsampled for
// first_octave -1 and taken as is for 0, so each step of first_octave shifts
// the count by one.
int Log2MinSideAfterFirstOctave(const int width,
                                const int height,
                                const int first_octave) {
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  const int min_side = std::min(width, height);
  return static_cast<int>(
             std::floor(std::log2(static_cast<double>(min_side)))) -
         first_octave;
}

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
  image->Upload();
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
  // `device` is the GPU the calling thread is bound to; Extract binds its
  // calling thread to it again (SiftScratch, the pool and the constant bank
  // are per device).
  CusiftFeatureExtractor(const FeatureExtractionOptions& options,
                         const int device)
      : options_(options), device_(device) {
    // options_.Check() dispatches on options.type, so options typed SIFT
    // would skip cusift's own checks; they run here whatever the type.
    THROW_CHECK(options_.Check() &&
                THROW_CHECK_NOTNULL(options_.cusift)->Check() &&
                CheckCusiftSiftOptions(*THROW_CHECK_NOTNULL(options_.sift)))
        << "invalid options for the cusift extractor; the reason is logged "
           "above";
  }

  // Frees the scratch, then returns the memory cusift's pool holds but no
  // longer uses to the driver, so GPU stages that run after extraction can
  // have it. Both act on the extractor's device, whatever the calling thread's.
  ~CusiftFeatureExtractor() override {
    scratch_.Release();
    cusift::ReleaseUnusedMemoryOnDevice(device_);
  }

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options) {
    BindCusiftDevice(options.gpu_index);
    return std::make_unique<CusiftFeatureExtractor>(options,
                                                    cusift::CurrentDevice());
  }

  // Any exception fails this image only, as with SiftGPU: the controller skips
  // it and goes on. An exception would end the worker thread, and the
  // extraction controller would then wait for it forever. Running out of GPU
  // (or pinned host) memory also returns the pool memory it no longer uses.
  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    try {
      SetBestCudaDevice(device_);
      return ExtractOrThrow(bitmap, keypoints, descriptors);
    } catch (const cusift::OutOfMemoryError& error) {
      LOG(WARNING) << "cusift ran out of memory on a " << bitmap.Width()
                   << " x " << bitmap.Height()
                   << " image; skipping it: " << error.what();
      cusift::ReleaseUnusedMemory();
      return false;
    } catch (const std::exception& error) {
      LOG(ERROR) << "cusift failed on a " << bitmap.Width() << " x "
                 << bitmap.Height() << " image; skipping it: " << error.what();
      return false;
    }
  }

 private:
  bool ExtractOrThrow(const Bitmap& bitmap,
                      FeatureKeypoints* keypoints,
                      FeatureDescriptors* descriptors) {
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
        options_.sift->peak_threshold,
        options_.sift->edge_threshold,
        options_.sift->max_num_orientations,
        options_.sift->first_octave,
        options_.sift->max_num_features,
        /*computeDescriptors=*/descriptors != nullptr,
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

  const FeatureExtractionOptions options_;
  const int device_;
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
  return std::max(Log2MinSideAfterFirstOctave(width, height, first_octave) - 3,
                  1);
}

int CusiftEffectiveNumOctaves(const CusiftExtractionOptions& options,
                              const int width,
                              const int height,
                              const int first_octave) {
  const int num_octaves = options.num_octaves > 0
                              ? options.num_octaves
                              : CusiftNumOctaves(width, height, first_octave);
  // Each octave halves the previous one, so after this many the pyramid would
  // drop below one pixel. VLFeat's later octaves are simply empty.
  const int max_num_octaves =
      Log2MinSideAfterFirstOctave(width, height, first_octave) + 1;
  return std::min(num_octaves, max_num_octaves);
}

void ClearUnsupportedCusiftSiftOptions(SiftExtractionOptions* sift) {
  THROW_CHECK_NOTNULL(sift);
  const SiftExtractionOptions defaults;
  sift->estimate_affine_shape = false;
  sift->domain_size_pooling = false;
  sift->upright = false;
  sift->darkness_adaptivity = false;
  sift->force_covariant_extractor = false;
  if (sift->first_octave != 0 && sift->first_octave != -1) {
    sift->first_octave = defaults.first_octave;
  }
  sift->octave_resolution = kCusiftOctaveResolution;
  sift->num_octaves = defaults.num_octaves;
}

bool CusiftExtractionOptions::Check() const {
  CHECK_OPTION(num_octaves == -1 || num_octaves >= 1);
  return true;
}

std::string CusiftUnsupportedSiftOption(const SiftExtractionOptions& sift) {
  if (sift.estimate_affine_shape) {
    return "sift.estimate_affine_shape: cusift does not implement affine shape "
           "estimation";
  }
  if (sift.domain_size_pooling) {
    return "sift.domain_size_pooling: cusift does not implement domain-size "
           "pooling";
  }
  if (sift.upright) {
    return "sift.upright: cusift does not implement upright features";
  }
  if (sift.darkness_adaptivity) {
    return "sift.darkness_adaptivity: cusift does not implement darkness "
           "adaptivity";
  }
  if (sift.force_covariant_extractor) {
    return "sift.force_covariant_extractor: cusift is not a covariant "
           "extractor";
  }
  // VLFeat's o_min. cusift implements the 2x-upsampled extra octave (-1),
  // which is COLMAP's default, and the plain first octave (0); nothing else.
  if (sift.first_octave != 0 && sift.first_octave != -1) {
    return StringPrintf("sift.first_octave: cusift supports 0 or -1, got %d",
                        sift.first_octave);
  }
  if (sift.octave_resolution != kCusiftOctaveResolution) {
    return StringPrintf(
        "sift.octave_resolution: cusift's levels per octave are fixed at %d, "
        "got %d",
        kCusiftOctaveResolution,
        sift.octave_resolution);
  }
  // cusift's octave count is CusiftExtractionOptions::num_octaves; a changed
  // SiftExtraction.num_octaves would otherwise be ignored without notice.
  const int default_num_octaves = SiftExtractionOptions().num_octaves;
  if (sift.num_octaves != default_num_octaves) {
    return StringPrintf(
        "sift.num_octaves: cusift ignores it (default %d, got %d); set "
        "cusift.num_octaves instead",
        default_num_octaves,
        sift.num_octaves);
  }
  return "";
}

bool CheckCusiftSiftOptions(const SiftExtractionOptions& sift) {
  const std::string unsupported = CusiftUnsupportedSiftOption(sift);
  if (!unsupported.empty()) {
    LOG(ERROR) << unsupported;
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
