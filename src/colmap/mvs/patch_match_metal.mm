// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/mvs/image.h"
#include "colmap/mvs/patch_match_metal_source.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace colmap {
namespace mvs {
namespace {

// Number of threads per threadgroup. Must be equal to THREADS_PER_BLOCK in
// patch_match_metal.metal.
constexpr int kThreadsPerBlock = 32;

// Same limit as for the templated window sizes in the CUDA implementation.
constexpr int kMaxWindowRadius = 20;

// Number of floats per source image in the poses buffer:
// K(4), R(9), T(3), C(3), P(12), P^-1(12).
constexpr int kNumPoseParams = 4 + 9 + 3 + 3 + 12 + 12;

// Limits of 2D array textures on Apple GPUs.
constexpr size_t kMaxTextureSize = 16384;
constexpr size_t kMaxNumSrcImages = 2048;

// Keep in sync with Params in patch_match_metal.metal.
struct Params {
  int width;
  int height;
  int num_src_images;
  int num_samples;
  float ref_K[4];
  float ref_inv_K[4];
  float perturbation;
  float spatial_norm;
  float color_norm;
  float cos_min_triangulation_angle;
  float inv_incident_angle_sigma_sq;
  float inv_ncc_sigma_sq;
  float ncc_norm;
  float prev_sel_prob_weight;
  float geom_consistency_regularizer;
  float geom_consistency_max_cost;
  float filter_min_ncc;
  float cos_filter_min_triangulation_angle;
  int filter_min_num_consistent;
  float filter_geom_consistency_max_cost;
  float depth_min;
  float depth_max;
  uint32_t seed;
  int src_width;
  int src_height;
};
static_assert(sizeof(Params) == 31 * sizeof(float), "Params layout");

// Values of the function constants of the kernels.
struct Specialization {
  int window_radius = 0;
  int window_step = 0;
  bool geom_consistency_term = false;
  bool filter_photo_consistency = false;
  bool filter_geom_consistency = false;
};

struct PipelineCache {
  std::mutex mutex;
  std::map<uint64_t, id<MTLLibrary>> libraries;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
};

// The shaders are compiled once per process and device, and the pipelines are
// specialized once per set of function constants, as the problems of a
// workspace are processed one after another with the same options. The cache
// is intentionally leaked to avoid releasing Metal objects during static
// destruction.
PipelineCache& GetPipelineCache() {
  static PipelineCache* cache = new PipelineCache();
  return *cache;
}

std::string GetErrorMessage(NSError* error) {
  return error == nil ? "unknown error"
                      : std::string(error.localizedDescription.UTF8String);
}

// Must be called with the cache locked.
id<MTLLibrary> GetLibrary(id<MTLDevice> device, PipelineCache& cache) {
  auto it = cache.libraries.find(device.registryID);
  if (it != cache.libraries.end()) {
    return it->second;
  }

  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  if (@available(macOS 11.0, *)) {
    options.languageVersion = MTLLanguageVersion2_3;
  }
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString
                               stringWithUTF8String:kPatchMatchMetalSource]
                   options:options
                     error:&error];
  THROW_CHECK(library != nil)
      << "Failed to compile the Metal patch match shaders: "
      << GetErrorMessage(error);
  cache.libraries[device.registryID] = library;
  return library;
}

id<MTLComputePipelineState> GetPipeline(
    id<MTLDevice> device,
    const std::string& function_name,
    const Specialization& spec = Specialization()) {
  PipelineCache& cache = GetPipelineCache();
  std::lock_guard<std::mutex> lock(cache.mutex);

  const std::string key = StringPrintf("%llu/%s/%d/%d/%d/%d/%d",
                                       device.registryID,
                                       function_name.c_str(),
                                       spec.window_radius,
                                       spec.window_step,
                                       spec.geom_consistency_term,
                                       spec.filter_photo_consistency,
                                       spec.filter_geom_consistency);
  auto it = cache.pipelines.find(key);
  if (it != cache.pipelines.end()) {
    return it->second;
  }

  id<MTLLibrary> library = GetLibrary(device, cache);

  MTLFunctionConstantValues* constants =
      [[MTLFunctionConstantValues alloc] init];
  int window_radius = spec.window_radius;
  int window_step = spec.window_step;
  bool geom_consistency_term = spec.geom_consistency_term;
  bool filter_photo_consistency = spec.filter_photo_consistency;
  bool filter_geom_consistency = spec.filter_geom_consistency;
  [constants setConstantValue:&window_radius type:MTLDataTypeInt atIndex:0];
  [constants setConstantValue:&window_step type:MTLDataTypeInt atIndex:1];
  [constants setConstantValue:&geom_consistency_term
                         type:MTLDataTypeBool
                      atIndex:2];
  [constants setConstantValue:&filter_photo_consistency
                         type:MTLDataTypeBool
                      atIndex:3];
  [constants setConstantValue:&filter_geom_consistency
                         type:MTLDataTypeBool
                      atIndex:4];

  NSError* error = nil;
  id<MTLFunction> function =
      [library newFunctionWithName:@(function_name.c_str())
                    constantValues:constants
                             error:&error];
  THROW_CHECK(function != nil)
      << "Failed to create the Metal function " << function_name << ": "
      << GetErrorMessage(error);

  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  THROW_CHECK(pipeline != nil)
      << "Failed to create the Metal pipeline " << function_name << ": "
      << GetErrorMessage(error);
  THROW_CHECK_GE(pipeline.maxTotalThreadsPerThreadgroup, kThreadsPerBlock)
      << "The Metal device cannot run " << function_name << " with "
      << kThreadsPerBlock << " threads per threadgroup";

  cache.pipelines[key] = pipeline;
  return pipeline;
}

id<MTLDevice> GetDevice(const int gpu_index) {
  if (gpu_index < 0) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    THROW_CHECK(device != nil) << "No Metal devices available";
    return device;
  }
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
  THROW_CHECK_LT(gpu_index, static_cast<int>(devices.count))
      << "Invalid Metal GPU selected";
  return devices[gpu_index];
}

// A map that is rotated by out-of-place kernels between two sweeps.
struct PingPongBuffer {
  id<MTLBuffer> buffers[2] = {nil, nil};
  int current = 0;

  id<MTLBuffer> Current() const { return buffers[current]; }
  id<MTLBuffer> Other() const { return buffers[1 - current]; }
  void Swap() { current = 1 - current; }
};

}  // namespace

int GetNumMetalDevices() {
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    return static_cast<int>(devices.count);
  }
}

class PatchMatchMetal::Impl {
 public:
  Impl(const PatchMatchOptions& options, const PatchMatch::Problem& problem);

  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  id<MTLBuffer> NewBuffer(size_t num_bytes) const;

  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();

  void SetParams(id<MTLComputeCommandEncoder> encoder,
                 int index,
                 float perturbation,
                 float prev_sel_prob_weight,
                 uint32_t seed);
  void Dispatch2D(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  size_t width,
                  size_t height,
                  size_t depth = 1) const;
  void DispatchColumns(id<MTLComputeCommandEncoder> encoder,
                       id<MTLComputePipelineState> pipeline) const;
  void CommitAndWait(id<MTLCommandBuffer> command_buffer,
                     const std::string& label) const;

  // Rotate all maps by 90 degrees in counter-clockwise direction.
  void EncodeRotate(id<MTLComputeCommandEncoder> encoder, bool rotate_mask);

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;

  // Original (not rotated) dimension of reference image.
  size_t ref_width_ = 0;
  size_t ref_height_ = 0;

  // Dimension of the currently rotated maps.
  size_t width_ = 0;
  size_t height_ = 0;

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // rotations mod 4.
  int rotation_in_half_pi_ = 0;

  // Size of the layers of the source image array.
  size_t src_width_ = 0;
  size_t src_height_ = 0;

  // Reference and source image input data.
  id<MTLTexture> src_images_texture_ = nil;
  id<MTLBuffer> src_depth_maps_ = nil;

  // Relative poses from rotated versions of reference image to source images
  // for each rotation, see PatchMatchCuda.
  id<MTLBuffer> poses_[4] = {nil, nil, nil, nil};

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to
  // rotation_in_half_pi_.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];

  // Data for reference image.
  PingPongBuffer ref_image_;
  PingPongBuffer ref_sum_image_;
  PingPongBuffer ref_squared_sum_image_;
  PingPongBuffer depth_map_;
  PingPongBuffer normal_map_;
  PingPongBuffer cost_map_;
  // The selection probability map of the last sweep is rotated into the
  // previous selection probability map.
  id<MTLBuffer> sel_prob_map_ = nil;
  id<MTLBuffer> prev_sel_prob_map_ = nil;
  PingPongBuffer consistency_mask_;

  // Workspace memory for each thread of the sweep, which is too large to be
  // held in thread local memory.
  id<MTLBuffer> global_workspace_ = nil;

  Params params_;
  Specialization spec_;
  bool has_run_ = false;

  id<MTLComputePipelineState> rotate_float_pipeline_ = nil;
  id<MTLComputePipelineState> rotate_uint8_pipeline_ = nil;
  id<MTLComputePipelineState> rotate_normal_pipeline_ = nil;
};

PatchMatchMetal::Impl::Impl(const PatchMatchOptions& options,
                            const PatchMatch::Problem& problem)
    : options_(options), problem_(problem) {
  THROW_CHECK_GE(options_.window_radius, 1);
  THROW_CHECK_LE(options_.window_radius, kMaxWindowRadius)
      << "Window radius " << options_.window_radius
      << " is not supported by the Metal patch match implementation";
  THROW_CHECK(options_.window_step == 1 || options_.window_step == 2)
      << "Window step " << options_.window_step << " is not supported";
  THROW_CHECK_GT(problem_.src_image_idxs.size(), 0);
  THROW_CHECK_LE(problem_.src_image_idxs.size(), kMaxNumSrcImages)
      << "Too many source images for Metal";

  device_ = GetDevice(std::stoi(options_.gpu_index));
  queue_ = [device_ newCommandQueue];
  THROW_CHECK(queue_ != nil);

  const size_t local_ref_image_bytes =
      (2 * options_.window_radius + 1) * 3 * kThreadsPerBlock * sizeof(float);
  THROW_CHECK_LE(local_ref_image_bytes, device_.maxThreadgroupMemoryLength)
      << "The window radius is too large for the threadgroup memory of "
      << device_.name.UTF8String;

  spec_.window_radius = options_.window_radius;
  spec_.window_step = options_.window_step;

  InitRefImage();
  InitSourceImages();
  InitTransforms();
  InitWorkspaceMemory();

  // Constant parameters for all kernels.
  std::memset(&params_, 0, sizeof(params_));
  params_.num_src_images = problem_.src_image_idxs.size();
  params_.num_samples = options_.num_samples;
  params_.spatial_norm =
      1.0f / (2.0f * options_.sigma_spatial * options_.sigma_spatial);
  params_.color_norm =
      1.0f / (2.0f * options_.sigma_color * options_.sigma_color);
  const double kDeg2Rad = 0.0174532925199432;
  params_.cos_min_triangulation_angle =
      std::cos(options_.min_triangulation_angle * kDeg2Rad);
  params_.inv_incident_angle_sigma_sq =
      -0.5f / (options_.incident_angle_sigma * options_.incident_angle_sigma);
  params_.inv_ncc_sigma_sq = -0.5f / (options_.ncc_sigma * options_.ncc_sigma);
  // Normalization of the likelihood function, see
  // LikelihoodComputer::ComputeNCCCostNormFactor in patch_match_cuda.cu.
  // Computed on the host, since MSL has no erf.
  params_.ncc_norm =
      2.0f / (std::sqrt(2.0f * M_PI) * options_.ncc_sigma *
              std::erf(2.0f / (options_.ncc_sigma * 1.414213562f)));
  params_.geom_consistency_regularizer = options_.geom_consistency_regularizer;
  params_.geom_consistency_max_cost = options_.geom_consistency_max_cost;
  params_.filter_min_ncc = options_.filter_min_ncc;
  params_.cos_filter_min_triangulation_angle =
      std::cos(options_.filter_min_triangulation_angle * kDeg2Rad);
  params_.filter_min_num_consistent = options_.filter_min_num_consistent;
  params_.filter_geom_consistency_max_cost =
      options_.filter_geom_consistency_max_cost;
  params_.depth_min = options_.depth_min;
  params_.depth_max = options_.depth_max;
  params_.src_width = src_width_;
  params_.src_height = src_height_;

  rotate_float_pipeline_ = GetPipeline(device_, "RotateFloat");
  rotate_uint8_pipeline_ = GetPipeline(device_, "RotateUint8");
  rotate_normal_pipeline_ = GetPipeline(device_, "RotateNormalMap");
}

id<MTLBuffer> PatchMatchMetal::Impl::NewBuffer(const size_t num_bytes) const {
  // The contents of new buffers are zero.
  id<MTLBuffer> buffer =
      [device_ newBufferWithLength:std::max<size_t>(num_bytes, 4)
                           options:MTLResourceStorageModeShared];
  THROW_CHECK(buffer != nil)
      << "Failed to allocate " << num_bytes / (1024.0 * 1024.0)
      << " MB of Metal memory. Try reducing the image size or the number of "
         "source images.";
  return buffer;
}

void PatchMatchMetal::Impl::InitRefImage() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  ref_width_ = ref_image.GetWidth();
  ref_height_ = ref_image.GetHeight();
  width_ = ref_width_;
  height_ = ref_height_;

  const size_t num_pixels = ref_width_ * ref_height_;
  for (int i = 0; i < 2; ++i) {
    ref_image_.buffers[i] = NewBuffer(num_pixels);
    ref_sum_image_.buffers[i] = NewBuffer(num_pixels * sizeof(float));
    ref_squared_sum_image_.buffers[i] = NewBuffer(num_pixels * sizeof(float));
  }
  std::memcpy(ref_image_.Current().contents,
              ref_image.GetBitmap().RowMajorData().data(),
              num_pixels);
}

void PatchMatchMetal::Impl::InitSourceImages() {
  // Determine maximum image size.
  for (const auto image_idx : problem_.src_image_idxs) {
    const Image& image = problem_.images->at(image_idx);
    src_width_ = std::max(src_width_, image.GetWidth());
    src_height_ = std::max(src_height_, image.GetHeight());
  }
  THROW_CHECK_LE(src_width_, kMaxTextureSize)
      << "Source images are too large for Metal. Reduce max_image_size.";
  THROW_CHECK_LE(src_height_, kMaxTextureSize)
      << "Source images are too large for Metal. Reduce max_image_size.";

  const size_t num_src_images = problem_.src_image_idxs.size();

  // Upload source images to device. The layers have a row stride of
  // src_width_, see PatchMatchCuda::InitSourceImages.
  MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
  descriptor.textureType = MTLTextureType2DArray;
  descriptor.pixelFormat = MTLPixelFormatR8Unorm;
  descriptor.width = src_width_;
  descriptor.height = src_height_;
  descriptor.arrayLength = num_src_images;
  descriptor.usage = MTLTextureUsageShaderRead;
  descriptor.storageMode = MTLStorageModeShared;
  src_images_texture_ = [device_ newTextureWithDescriptor:descriptor];
  THROW_CHECK(src_images_texture_ != nil)
      << "Failed to allocate the Metal source image texture";

  const uint8_t kDefaultValue = 0;
  std::vector<uint8_t> src_images_host_data(src_width_ * src_height_,
                                            kDefaultValue);
  for (size_t i = 0; i < num_src_images; ++i) {
    const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
    const uint8_t* src = image.GetBitmap().RowMajorData().data();
    std::fill(src_images_host_data.begin(),
              src_images_host_data.end(),
              kDefaultValue);
    uint8_t* dest = src_images_host_data.data();
    for (size_t r = 0; r < image.GetHeight(); ++r) {
      std::memcpy(dest, src + r * image.GetWidth(), image.GetWidth());
      dest += src_width_;
    }
    [src_images_texture_
        replaceRegion:MTLRegionMake2D(0, 0, src_width_, src_height_)
          mipmapLevel:0
                slice:i
            withBytes:src_images_host_data.data()
          bytesPerRow:src_width_
        bytesPerImage:0];
  }

  // Upload source depth maps to device.
  if (options_.geom_consistency) {
    src_depth_maps_ =
        NewBuffer(src_width_ * src_height_ * num_src_images * sizeof(float));
    float* dest_base = static_cast<float*>(src_depth_maps_.contents);
    for (size_t i = 0; i < num_src_images; ++i) {
      const DepthMap& depth_map =
          problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest = dest_base + src_width_ * src_height_ * i;
      for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
        std::memcpy(dest,
                    depth_map.GetPtr() + r * depth_map.GetWidth(),
                    depth_map.GetWidth() * sizeof(float));
        dest += src_width_;
      }
    }
  } else {
    src_depth_maps_ = NewBuffer(sizeof(float));
  }
}

void PatchMatchMetal::Impl::InitTransforms() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions (counter-clockwise) of calibration matrix.
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < 4; ++i) {
    ref_K_host_[i][0] = ref_image.GetK()[0];
    ref_K_host_[i][1] = ref_image.GetK()[2] - 0.5f;
    ref_K_host_[i][2] = ref_image.GetK()[4];
    ref_K_host_[i][3] = ref_image.GetK()[5] - 0.5f;
  }

  // Rotated by 90 degrees.
  std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
  std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
  ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

  // Rotated by 180 degrees.
  ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
  ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

  // Rotated by 270 degrees.
  std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
  std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
  ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

  // Extract 1/fx, -cx/fx, fy, -cy/fy.
  for (size_t i = 0; i < 4; ++i) {
    ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
    ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
    ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
    ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
  }

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions of camera poses.
  //////////////////////////////////////////////////////////////////////////////

  float rotated_R[9];
  std::memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

  float rotated_T[3];
  std::memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

  // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
  const float R_z90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};

  for (size_t i = 0; i < 4; ++i) {
    poses_[i] = NewBuffer(kNumPoseParams * problem_.src_image_idxs.size() *
                          sizeof(float));
    float* poses_host_data = static_cast<float*>(poses_[i].contents);
    int offset = 0;
    for (const auto image_idx : problem_.src_image_idxs) {
      const Image& image = problem_.images->at(image_idx);

      float src_K[9];
      std::memcpy(src_K, image.GetK(), 9 * sizeof(float));
      src_K[2] -= 0.5f;
      src_K[5] -= 0.5f;

      const float K[4] = {src_K[0], src_K[2], src_K[4], src_K[5]};
      std::memcpy(poses_host_data + offset, K, 4 * sizeof(float));
      offset += 4;

      float rel_R[9];
      float rel_T[3];
      ComputeRelativePose(
          rotated_R, rotated_T, image.GetR(), image.GetT(), rel_R, rel_T);
      std::memcpy(poses_host_data + offset, rel_R, 9 * sizeof(float));
      offset += 9;
      std::memcpy(poses_host_data + offset, rel_T, 3 * sizeof(float));
      offset += 3;

      float C[3];
      ComputeProjectionCenter(rel_R, rel_T, C);
      std::memcpy(poses_host_data + offset, C, 3 * sizeof(float));
      offset += 3;

      float P[12];
      ComposeProjectionMatrix(src_K, rel_R, rel_T, P);
      std::memcpy(poses_host_data + offset, P, 12 * sizeof(float));
      offset += 12;

      float inv_P[12];
      ComposeInverseProjectionMatrix(src_K, rel_R, rel_T, inv_P);
      std::memcpy(poses_host_data + offset, inv_P, 12 * sizeof(float));
      offset += 12;
    }

    RotatePose(R_z90, rotated_R, rotated_T);
  }
}

void PatchMatchMetal::Impl::InitWorkspaceMemory() {
  const size_t num_pixels = ref_width_ * ref_height_;
  const size_t num_src_images = problem_.src_image_idxs.size();

  for (int i = 0; i < 2; ++i) {
    depth_map_.buffers[i] = NewBuffer(num_pixels * sizeof(float));
    normal_map_.buffers[i] = NewBuffer(3 * num_pixels * sizeof(float));
    cost_map_.buffers[i] =
        NewBuffer(num_pixels * num_src_images * sizeof(float));
  }
  if (options_.geom_consistency) {
    const DepthMap& init_depth_map =
        problem_.depth_maps->at(problem_.ref_image_idx);
    std::memcpy(depth_map_.Current().contents,
                init_depth_map.GetPtr(),
                num_pixels * sizeof(float));
    const NormalMap& init_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    std::memcpy(normal_map_.Current().contents,
                init_normal_map.GetPtr(),
                3 * num_pixels * sizeof(float));
  }

  // Note that it is not necessary to keep the selection probability map in
  // memory for all pixels. Theoretically, it is possible to incorporate
  // the temporary selection probabilities in the global_workspace_.
  // However, it is useful to keep the probabilities for the entire image
  // in memory, so that it can be exported.
  sel_prob_map_ = NewBuffer(num_pixels * num_src_images * sizeof(float));
  prev_sel_prob_map_ = NewBuffer(num_pixels * num_src_images * sizeof(float));
  float* prev_sel_prob_data = static_cast<float*>(prev_sel_prob_map_.contents);
  std::fill(prev_sel_prob_data,
            prev_sel_prob_data + num_pixels * num_src_images,
            0.5f);

  const size_t ref_max_dim = std::max(ref_width_, ref_height_);
  global_workspace_ =
      NewBuffer(ref_max_dim * num_src_images * 2 * sizeof(float));

  // The consistency mask is only used (and rotated) in the last sweep, if
  // filtering is enabled. Otherwise, only a dummy is bound to the kernels.
  for (int i = 0; i < 2; ++i) {
    consistency_mask_.buffers[i] =
        NewBuffer(options_.filter ? num_pixels * num_src_images : 1);
  }
}

void PatchMatchMetal::Impl::SetParams(id<MTLComputeCommandEncoder> encoder,
                                      const int index,
                                      const float perturbation,
                                      const float prev_sel_prob_weight,
                                      const uint32_t seed) {
  params_.width = width_;
  params_.height = height_;
  std::memcpy(
      params_.ref_K, ref_K_host_[rotation_in_half_pi_], 4 * sizeof(float));
  std::memcpy(params_.ref_inv_K,
              ref_inv_K_host_[rotation_in_half_pi_],
              4 * sizeof(float));
  params_.perturbation = perturbation;
  params_.prev_sel_prob_weight = prev_sel_prob_weight;
  params_.seed = seed;
  [encoder setBytes:&params_ length:sizeof(params_) atIndex:index];
}

void PatchMatchMetal::Impl::Dispatch2D(id<MTLComputeCommandEncoder> encoder,
                                       id<MTLComputePipelineState> pipeline,
                                       const size_t width,
                                       const size_t height,
                                       const size_t depth) const {
  // One thread per element. The kernels check the bounds.
  const MTLSize block_size =
      MTLSizeMake(kThreadsPerBlock, depth > 1 ? 1 : 8, 1);
  const MTLSize grid_size =
      MTLSizeMake((width + block_size.width - 1) / block_size.width,
                  (height + block_size.height - 1) / block_size.height,
                  depth);
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:block_size];
}

void PatchMatchMetal::Impl::DispatchColumns(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline) const {
  // One thread per column, see ComputeCudaConfig in patch_match_cuda.cu.
  const size_t local_ref_image_bytes =
      (2 * options_.window_radius + 1) * 3 * kThreadsPerBlock * sizeof(float);
  [encoder setThreadgroupMemoryLength:local_ref_image_bytes atIndex:0];
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreadgroups:MTLSizeMake((width_ + kThreadsPerBlock - 1) /
                                                kThreadsPerBlock,
                                            1,
                                            1)
          threadsPerThreadgroup:MTLSizeMake(kThreadsPerBlock, 1, 1)];
}

void PatchMatchMetal::Impl::CommitAndWait(id<MTLCommandBuffer> command_buffer,
                                          const std::string& label) const {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  THROW_CHECK(command_buffer.status == MTLCommandBufferStatusCompleted)
      << "Metal command buffer failed (" << label
      << "): " << GetErrorMessage(command_buffer.error);
}

void PatchMatchMetal::Impl::EncodeRotate(id<MTLComputeCommandEncoder> encoder,
                                         const bool rotate_mask) {
  // Padded to int4, as the kernels read an int3.
  const int dims_2d[4] = {
      static_cast<int>(width_), static_cast<int>(height_), 1, 0};
  const int dims_src[4] = {static_cast<int>(width_),
                           static_cast<int>(height_),
                           static_cast<int>(problem_.src_image_idxs.size()),
                           0};

  const auto rotate = [&](id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> input,
                          id<MTLBuffer> output,
                          const int* dims) {
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:dims length:4 * sizeof(int) atIndex:2];
    Dispatch2D(encoder, pipeline, dims[0], dims[1], dims[2]);
  };

  // Rotate depth map.
  rotate(rotate_float_pipeline_,
         depth_map_.Current(),
         depth_map_.Other(),
         dims_2d);
  depth_map_.Swap();

  // Rotate normal map (the normals are rotated as well).
  rotate(rotate_normal_pipeline_,
         normal_map_.Current(),
         normal_map_.Other(),
         dims_2d);
  normal_map_.Swap();

  // Rotate reference image.
  rotate(rotate_uint8_pipeline_,
         ref_image_.Current(),
         ref_image_.Other(),
         dims_2d);
  ref_image_.Swap();
  rotate(rotate_float_pipeline_,
         ref_sum_image_.Current(),
         ref_sum_image_.Other(),
         dims_2d);
  ref_sum_image_.Swap();
  rotate(rotate_float_pipeline_,
         ref_squared_sum_image_.Current(),
         ref_squared_sum_image_.Other(),
         dims_2d);
  ref_squared_sum_image_.Swap();

  // Rotate selection probability map. The result is the previous selection
  // probability map of the next sweep.
  rotate(rotate_float_pipeline_, sel_prob_map_, prev_sel_prob_map_, dims_src);

  // Rotate cost map.
  rotate(
      rotate_float_pipeline_, cost_map_.Current(), cost_map_.Other(), dims_src);
  cost_map_.Swap();

  // Rotate selected image map.
  if (rotate_mask) {
    rotate(rotate_uint8_pipeline_,
           consistency_mask_.Current(),
           consistency_mask_.Other(),
           dims_src);
    consistency_mask_.Swap();
  }

  // Rotate calibration.
  rotation_in_half_pi_ = (rotation_in_half_pi_ + 1) % 4;
  std::swap(width_, height_);
}

void PatchMatchMetal::Impl::Run() {
  THROW_CHECK(!has_run_) << "Run() can only be called once";
  has_run_ = true;

  Timer total_timer;
  total_timer.Start();
  Timer timer;
  timer.Start();

  const bool filter_photo = options_.filter;
  const bool filter_geom = options_.filter && options_.geom_consistency;

  Specialization init_spec = spec_;
  const auto filter_ref_image_pipeline =
      GetPipeline(device_, "FilterRefImage", init_spec);
  const auto init_depth_normal_pipeline =
      GetPipeline(device_, "InitDepthAndNormalMap");
  const auto initial_cost_pipeline =
      GetPipeline(device_, "ComputeInitialCost", init_spec);

  // Sweeps before the last one do not filter.
  Specialization sweep_spec = spec_;
  sweep_spec.geom_consistency_term = options_.geom_consistency;
  Specialization last_sweep_spec = sweep_spec;
  last_sweep_spec.filter_photo_consistency = filter_photo;
  last_sweep_spec.filter_geom_consistency = filter_geom;
  const auto sweep_pipeline =
      GetPipeline(device_, "SweepFromTopToBottom", sweep_spec);
  const auto last_sweep_pipeline =
      GetPipeline(device_, "SweepFromTopToBottom", last_sweep_spec);

  // Initialize the reference image statistics, the random depth and normal
  // maps (unless they are given), and the initial cost.
  {
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];

    [encoder setBuffer:ref_image_.Current() offset:0 atIndex:0];
    [encoder setBuffer:ref_sum_image_.Current() offset:0 atIndex:1];
    [encoder setBuffer:ref_squared_sum_image_.Current() offset:0 atIndex:2];
    SetParams(encoder, 3, 0.0f, 0.0f, 0);
    Dispatch2D(encoder, filter_ref_image_pipeline, width_, height_);

    if (!options_.geom_consistency) {
      [encoder setBuffer:depth_map_.Current() offset:0 atIndex:0];
      [encoder setBuffer:normal_map_.Current() offset:0 atIndex:1];
      SetParams(encoder, 2, 0.0f, 0.0f, 12345);
      Dispatch2D(encoder, init_depth_normal_pipeline, width_, height_);
    }

    [encoder setBuffer:cost_map_.Current() offset:0 atIndex:0];
    [encoder setBuffer:depth_map_.Current() offset:0 atIndex:1];
    [encoder setBuffer:normal_map_.Current() offset:0 atIndex:2];
    [encoder setBuffer:ref_image_.Current() offset:0 atIndex:3];
    [encoder setBuffer:ref_sum_image_.Current() offset:0 atIndex:4];
    [encoder setBuffer:ref_squared_sum_image_.Current() offset:0 atIndex:5];
    [encoder setTexture:src_images_texture_ atIndex:0];
    [encoder setBuffer:poses_[0] offset:0 atIndex:6];
    SetParams(encoder, 7, 0.0f, 0.0f, 0);
    DispatchColumns(encoder, initial_cost_pipeline);
    [encoder endEncoding];
    CommitAndWait(command_buffer, "initialization");
  }

  LOG(INFO) << StringPrintf("Initialization: %.4fs", timer.ElapsedSeconds());

  const float total_num_steps = options_.num_iterations * 4;

  for (int iter = 0; iter < options_.num_iterations; ++iter) {
    Timer iter_timer;
    iter_timer.Start();

    for (int sweep = 0; sweep < 4; ++sweep) {
      Timer sweep_timer;
      sweep_timer.Start();

      // Expenentially reduce amount of perturbation during the optimization.
      const float perturbation = 1.0f / std::pow(2.0f, iter + sweep / 4.0f);

      // Linearly increase the influence of previous selection probabilities.
      const float prev_sel_prob_weight =
          static_cast<float>(iter * 4 + sweep) / total_num_steps;

      const bool last_sweep = iter == options_.num_iterations - 1 && sweep == 3;

      // One command buffer per sweep, to keep the GPU time of a single command
      // buffer short for large images.
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];

      [encoder setBuffer:global_workspace_ offset:0 atIndex:0];
      [encoder setBuffer:cost_map_.Current() offset:0 atIndex:1];
      [encoder setBuffer:depth_map_.Current() offset:0 atIndex:2];
      [encoder setBuffer:normal_map_.Current() offset:0 atIndex:3];
      [encoder setBuffer:consistency_mask_.Current() offset:0 atIndex:4];
      [encoder setBuffer:sel_prob_map_ offset:0 atIndex:5];
      [encoder setBuffer:prev_sel_prob_map_ offset:0 atIndex:6];
      [encoder setBuffer:ref_image_.Current() offset:0 atIndex:7];
      [encoder setBuffer:ref_sum_image_.Current() offset:0 atIndex:8];
      [encoder setBuffer:ref_squared_sum_image_.Current() offset:0 atIndex:9];
      [encoder setTexture:src_images_texture_ atIndex:0];
      [encoder setBuffer:src_depth_maps_ offset:0 atIndex:10];
      [encoder setBuffer:poses_[rotation_in_half_pi_] offset:0 atIndex:11];
      SetParams(encoder,
                12,
                perturbation,
                prev_sel_prob_weight,
                1000 + iter * 4 + sweep);
      DispatchColumns(encoder,
                      last_sweep ? last_sweep_pipeline : sweep_pipeline);

      EncodeRotate(encoder, last_sweep && options_.filter);
      [encoder endEncoding];
      CommitAndWait(command_buffer, StringPrintf("sweep %d", sweep + 1));

      LOG(INFO) << StringPrintf(
          " Sweep %d: %.4fs", sweep + 1, sweep_timer.ElapsedSeconds());
    }

    LOG(INFO) << StringPrintf(
        "Iteration %d: %.4fs", iter + 1, iter_timer.ElapsedSeconds());
  }

  LOG(INFO) << StringPrintf("Total: %.4fs", total_timer.ElapsedSeconds());
}

DepthMap PatchMatchMetal::Impl::GetDepthMap() const {
  Mat<float> mat(ref_width_, ref_height_, 1);
  std::memcpy(mat.GetPtr(), depth_map_.Current().contents, mat.GetNumBytes());
  return DepthMap(mat, options_.depth_min, options_.depth_max);
}

NormalMap PatchMatchMetal::Impl::GetNormalMap() const {
  Mat<float> mat(ref_width_, ref_height_, 3);
  std::memcpy(mat.GetPtr(), normal_map_.Current().contents, mat.GetNumBytes());
  return NormalMap(mat);
}

Mat<float> PatchMatchMetal::Impl::GetSelProbMap() const {
  Mat<float> mat(ref_width_, ref_height_, problem_.src_image_idxs.size());
  std::memcpy(mat.GetPtr(), prev_sel_prob_map_.contents, mat.GetNumBytes());
  return mat;
}

std::vector<int> PatchMatchMetal::Impl::GetConsistentImageIdxs() const {
  std::vector<int> consistent_image_idxs;
  if (!options_.filter) {
    return consistent_image_idxs;
  }

  const size_t num_src_images = problem_.src_image_idxs.size();
  const size_t num_pixels = ref_width_ * ref_height_;
  const uint8_t* mask =
      static_cast<const uint8_t*>(consistency_mask_.Current().contents);
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(num_src_images);
  for (size_t r = 0; r < ref_height_; ++r) {
    for (size_t c = 0; c < ref_width_; ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < num_src_images; ++d) {
        if (mask[d * num_pixels + r * ref_width_ + c]) {
          pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
        }
      }
      if (pixel_consistent_image_idxs.size() > 0) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

PatchMatchMetal::PatchMatchMetal(const PatchMatchOptions& options,
                                 const PatchMatch::Problem& problem) {
  @autoreleasepool {
    impl_ = std::make_unique<Impl>(options, problem);
  }
}

PatchMatchMetal::~PatchMatchMetal() {
  @autoreleasepool {
    impl_.reset();
  }
}

void PatchMatchMetal::Run() {
  @autoreleasepool {
    impl_->Run();
  }
}

DepthMap PatchMatchMetal::GetDepthMap() const { return impl_->GetDepthMap(); }

NormalMap PatchMatchMetal::GetNormalMap() const {
  return impl_->GetNormalMap();
}

Mat<float> PatchMatchMetal::GetSelProbMap() const {
  return impl_->GetSelProbMap();
}

std::vector<int> PatchMatchMetal::GetConsistentImageIdxs() const {
  return impl_->GetConsistentImageIdxs();
}

}  // namespace mvs
}  // namespace colmap
