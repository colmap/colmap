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

#include "colmap/feature/cusift.h"

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <cmath>

#ifdef COLMAP_CUDA_ENABLED
#include <array>
#include <string>
#include <vector>

// Deliberately no <cuda_runtime.h> and no cudautils.h here. Every CUDA type and
// CUDA API reference stays inside colmap_cusift (see cusift_device.h), so
// colmap_feature does not acquire a CUDA runtime dependency of its own.
#include "cudaImage.h"
#include "cudaSift.h"
#include "cusift_device.h"
#endif  // COLMAP_CUDA_ENABLED

namespace colmap {
namespace {

#ifdef COLMAP_CUDA_ENABLED
constexpr int kSiftDescriptorDim = 128;
#endif  // COLMAP_CUDA_ENABLED

// cusift's compile-time levels-per-octave (NUM_SCALES in cudaSiftD.h).
constexpr int kCusiftOctaveResolution = 3;

#ifdef COLMAP_CUDA_ENABLED

// Binds the calling thread to the single GPU named by `gpu_index`
// (cudaSetDevice is per-thread, so each extraction thread does this for
// itself) and initializes cusift's memory pool on it.
void BindCusiftDevice(const std::string& gpu_index) {
  const std::vector<int> gpu_indices = CSVToVector<int>(gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1) << "cusift can only run on one GPU";
  THROW_CHECK(cusift::InitDevice(gpu_indices[0]))
      << "cusift found no usable CUDA device";
}

// Uploads a grey bitmap as the [0, 255] float image cusift's kernels take.
// cusift scales by 1/255 in its first kernel, where COLMAP's VLFeat path
// divides by 255 on the host. Same uint8 source bytes either way, which is
// what the grayscale parity gate covers; the multiply-vs-divide rounding is
// inside the measured parity envelope, because every cusift-vs-VLFeat gate was
// run through this same kernel. `host_buffer` backs the image's host side and
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
                  /*withHost=*/false,
                  /*devMem=*/nullptr,
                  /*hostMem=*/host_buffer->data());
  image->Download();
}

// COLMAP's own normalization and uint8 quantization, deliberately: cusift
// hands back exactly what vl_sift_calc_keypoint_descriptor does, so this is
// the same final step the VLFeat CPU path applies. Shared by the extractor and
// the descriptor-isolation path, so both write the same representation.
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
    // Includes CheckCusiftSiftOptions: the options cusift cannot serve are
    // rejected loudly, since a silent no-op would produce a database that
    // looks fine and reconstructs worse.
    THROW_CHECK(options_.Check())
        << "invalid options for the cusift extractor; the reason is logged "
           "above";
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

    // Absolute image coordinates, sigma in image units, orientation in radians
    // in VLFeat's convention, descriptors L2 -> clip 0.2 -> L2 normalized and
    // already in UBC layout. No +0.5 applied, and no VLFeat-to-UBC permutation
    // to undo - see the conventions notes below.
    x_.clear();
    y_.clear();
    sigma_.clear();
    orientation_.clear();
    descriptors_.clear();
    ExtractSiftFullVLFeat(
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

    // The +0.5 pixel-centre offset is applied here and only here: cusift's
    // kernels work in VLFeat's pixel frame throughout, and this is the single
    // boundary where features leave it for COLMAP. Matches the VLFeat CPU
    // backend, which adds it at the same point.
    //
    // Orientation is stored in VLFeat-convention radians, also matching that
    // backend, and is deliberately NOT negated to SiftGPU's sign. COLMAP's two
    // SIFT backends disagree here -- the VLFeat CPU path stores VLFeat theta
    // while SiftGPU's raw mirrored theta is written uncorrected
    // (sift.cc:716-719) -- and cusift follows the VLFeat reference it is gated
    // against. Consumers of stored orientation (LightGlue via
    // onnx_matchers.cc:596, vocab-tree geometry via retrieval/visual_index.cc)
    // therefore see VLFeat convention from cusift, as they do from the CPU
    // backend. Do not "fix" this to match SiftGPU.
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

      // No TransformVLFeatToUBCFeatureDescriptors call: cusift's descriptor
      // kernel writes the UBC order natively. Adding the permutation here
      // would silently scramble every descriptor.
      descriptors->data = NormalizeAndQuantize(options_.sift->normalization,
                                               &descriptors_float);
      // These are SIFT descriptors in COLMAP's usual layout, so they are
      // tagged as such. That is the whole point: the existing SIFT matchers,
      // index, and databases consume them with no changes, and cusift and
      // SiftGPU features remain interoperable.
      descriptors->type = FeatureExtractorType::SIFT;
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;

  // Reused across images to avoid reallocating per call.
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> sigma_;
  std::vector<float> orientation_;
  std::vector<std::array<float, kSiftDescriptorDim>> descriptors_;
};

#endif  // COLMAP_CUDA_ENABLED

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

std::unique_ptr<FeatureExtractor> CreateCusiftFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_CUDA_ENABLED
  LOG(INFO) << "Creating cusift GPU feature extractor";
  return CusiftFeatureExtractor::Create(options);
#else
  LOG(ERROR) << "cusift feature extraction requires CUDA support";
  return nullptr;
#endif  // COLMAP_CUDA_ENABLED
}

#ifdef COLMAP_CUDA_ENABLED
namespace {

// Unclamped replica of the sigma -> octave inversion inside cusift's
// AssignOctaveLevel (itself a replica of vl_sift_keypoint_init, sift.c:2161).
// cusift clamps the result to >= 0; we need the true value in order to reject
// octave -1 rather than let it be clamped onto the wrong pyramid level.
int TrueVLFeatOctave(const float abs_sigma, const int octave_resolution) {
  const float sigmak = std::pow(2.0f, 1.0f / octave_resolution);
  const float sigma0 = 1.6f * sigmak;
  const double phi =
      std::log2((static_cast<double>(abs_sigma) + 1e-10) / sigma0);
  constexpr int kSMin = -1;
  return static_cast<int>(
      std::floor(phi - (static_cast<double>(kSMin) + 0.5) / octave_resolution));
}

}  // namespace
#endif  // COLMAP_CUDA_ENABLED

CusiftDescriptorOnlyResult ExtractCusiftDescriptorsForKeypoints(
    const Bitmap& bitmap,
    const FeatureKeypoints& keypoints,
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_CUDA_ENABLED
  THROW_CHECK(bitmap.IsGrey());
  THROW_CHECK_NOTNULL(options.sift);
  THROW_CHECK_EQ(options.sift->octave_resolution, kCusiftOctaveResolution);
  BindCusiftDevice(options.gpu_index);

  CusiftDescriptorOnlyResult result;
  if (keypoints.empty()) {
    result.descriptors.type = FeatureExtractorType::SIFT;
    return result;
  }

  // The isolation path is non-upsampled only (first octave 0), with as many
  // octaves as the extractor would build for this image. Keypoints outside
  // [0, num_octaves) are rejected up front: AssignOctaveLevel would clamp them
  // onto the nearest pyramid level and describe them at the wrong scale.
  const int num_octaves =
      CusiftEffectiveNumOctaves(*THROW_CHECK_NOTNULL(options.cusift),
                                bitmap.Width(),
                                bitmap.Height(),
                                /*first_octave=*/0);
  std::vector<size_t> accepted;
  accepted.reserve(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const int octave = TrueVLFeatOctave(keypoints[i].ComputeScale(),
                                        options.sift->octave_resolution);
    if (octave < 0) {
      ++result.num_rejected_octave_minus1;
    } else if (octave >= num_octaves) {
      ++result.num_rejected_octave_above_pyramid;
    } else {
      accepted.push_back(i);
    }
  }
  if (accepted.empty()) {
    result.descriptors.type = FeatureExtractorType::SIFT;
    return result;
  }

  std::vector<float> host_buffer;
  CudaImage image;
  UploadGreyBitmap(bitmap, &host_buffer, &image);

  SiftData sift_data;
  InitSiftData(sift_data, static_cast<int>(accepted.size()), true, true);
  sift_data.numPts = static_cast<int>(accepted.size());
  for (size_t k = 0; k < accepted.size(); ++k) {
    const FeatureKeypoint& kp = keypoints[accepted[k]];
    SiftPoint& p = sift_data.h_data[k];
    // Undo COLMAP's +0.5 pixel-centre offset: the kernel domain is VLFeat's
    // frame, which is where the extractor applies the offset on the way out.
    p.xpos = kp.x - 0.5f;
    p.ypos = kp.y - 0.5f;
    p.scale = kp.ComputeScale();
    // SiftPoint carries orientation in DEGREES; ComputeOrientation is radians.
    p.orientation = static_cast<float>(RadToDeg(kp.ComputeOrientation()));
  }

  ExtractSiftDescriptorsOnly(sift_data, image, num_octaves, /*initBlur=*/1.0);

  // A descriptor window that fell outside the image comes back all-zero.
  // Detect that rather than guessing a support radius, and drop it so it cannot
  // zero-fill the caller's statistics.
  std::vector<size_t> kept;
  kept.reserve(accepted.size());
  for (size_t k = 0; k < accepted.size(); ++k) {
    const float* d = sift_data.h_data[k].data;
    double norm = 0.0;
    for (int j = 0; j < kSiftDescriptorDim; ++j) {
      norm += static_cast<double>(d[j]) * d[j];
    }
    if (norm > 1e-12) {
      kept.push_back(k);
    } else {
      ++result.num_rejected_degenerate;
    }
  }

  FeatureDescriptorsFloatData desc_float(kept.size(), kSiftDescriptorDim);
  result.keypoint_indices.resize(kept.size());
  for (size_t r = 0; r < kept.size(); ++r) {
    result.keypoint_indices[r] = accepted[kept[r]];
    desc_float.row(r) = Eigen::Map<const Eigen::RowVectorXf>(
        sift_data.h_data[kept[r]].data, kSiftDescriptorDim);
  }
  FreeSiftData(sift_data);

  // Same normalization and quantization the extractor applies, so the rows meet
  // a reference database's rows in the same representation. Anything else would
  // measure RootSIFT-vs-L2 or float-vs-quantized instead of the kernel.
  result.descriptors.data =
      NormalizeAndQuantize(options.sift->normalization, &desc_float);
  result.descriptors.type = FeatureExtractorType::SIFT;
  return result;
#else
  (void)bitmap;
  (void)keypoints;
  (void)options;
  LOG(FATAL_THROW) << "cusift descriptor extraction requires CUDA support";
  return {};
#endif  // COLMAP_CUDA_ENABLED
}

}  // namespace colmap
