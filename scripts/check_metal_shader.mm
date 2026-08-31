// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in COPYING.txt
// are met.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct alignas(16) MetalParams {
  uint32_t width, height, source_width, source_height;
  uint32_t num_sources, window_radius, window_step, num_samples;
  uint32_t sweep_direction, iteration, geom_consistency, filter;
  uint32_t filter_min_num_consistent, initialize_from_input, seed, pose_stride;
  float depth_min, depth_max, sigma_spatial, sigma_color;
  float ncc_sigma, geom_consistency_regularizer, geom_consistency_max_cost, filter_min_ncc;
  float filter_geom_consistency_max_cost, perturbation, ref_fx, ref_cx;
  float ref_fy, ref_cy, ref_inv_fx, ref_neg_cx_inv_fx;
  float ref_inv_fy, ref_neg_cy_inv_fy, filter_cos_min_triangulation_angle, reserved1;
  float cos_min_triangulation_angle, inv_incident_angle_sigma_square;
  float ncc_norm_factor, prev_sel_prob_weight;
};

static_assert(sizeof(MetalParams) == 160);

template <typename T>
id<MTLBuffer> MakeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
  return [device newBufferWithBytes:values.data()
                             length:values.size() * sizeof(T)
                            options:MTLResourceStorageModeShared];
}

bool Run(id<MTLCommandQueue> queue,
         id<MTLComputePipelineState> pipeline,
         id<MTLTexture> source_texture,
         id<MTLBuffer> spatial_weight_buffer,
         id<MTLBuffer> color_weight_buffer,
         const std::vector<id<MTLBuffer>>& buffers,
         const MetalParams& params) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:source_texture atIndex:0];
  [encoder setBuffer:spatial_weight_buffer offset:0 atIndex:11];
  [encoder setBuffer:color_weight_buffer offset:0 atIndex:12];
  for (size_t index = 0; index < buffers.size(); ++index) {
    if (index == 4) {
      [encoder setBytes:&params length:sizeof(params) atIndex:index];
    } else {
      [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    }
  }
  [encoder dispatchThreads:MTLSizeMake(params.width, params.height, 1)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::cerr << [[command.error localizedDescription] UTF8String] << "\n";
    return false;
  }
  return true;
}

bool RunSweep(id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> pipeline,
              id<MTLTexture> source_texture,
              const std::vector<id<MTLBuffer>>& buffers,
              const MetalParams& params) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setTexture:source_texture atIndex:0];
  for (size_t index = 0; index < buffers.size(); ++index) {
    if (index == 4) {
      [encoder setBytes:&params length:sizeof(params) atIndex:index];
    } else {
      [encoder setBuffer:buffers[index] offset:0 atIndex:index];
    }
  }
  const NSUInteger simd_width = pipeline.threadExecutionWidth;
  if (simd_width != 32) {
    std::cerr << "Expected a 32-lane Metal SIMD group, got " << simd_width << "\n";
    return false;
  }
  const NSUInteger simdgroups = std::max<NSUInteger>(
      1, std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / simd_width));
  [encoder setThreadgroupMemoryLength:simdgroups * 5 * params.num_samples * sizeof(float)
                              atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(params.width * simd_width, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(simdgroups * simd_width, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::cerr << [[command.error localizedDescription] UTF8String] << "\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "Usage: check_metal_shader patch_match_metal.mm\n";
      return 2;
    }

    std::ifstream file(argv[1]);
    std::ostringstream stream;
    stream << file.rdbuf();
    const std::string translation_unit = stream.str();
    const std::string begin_marker = "R\"METAL(";
    const std::string end_marker = ")METAL\"";
    const size_t begin = translation_unit.find(begin_marker);
    const size_t end = translation_unit.find(end_marker, begin);
    if (!file || begin == std::string::npos || end == std::string::npos) {
      std::cerr << "Could not extract the embedded Metal source\n";
      return 2;
    }
    const std::string shader_source =
        translation_unit.substr(begin + begin_marker.size(), end - begin - begin_marker.size());

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::cerr << "No Metal device is available\n";
      return 1;
    }
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:shader_source.c_str()]
                             options:nil
                               error:&error];
    if (library == nil) {
      std::cerr << [[error localizedDescription] UTF8String] << "\n";
      return 1;
    }

    const NSArray<NSString*>* functions = @[
      @"patch_match_initialize_fidelity",
      @"patch_match_column_sweep_simd",
      @"patch_match_finalize_fidelity"
    ];
    id<MTLComputePipelineState> pipelines[3] = {nil, nil, nil};
    size_t pipeline_index = 0;
    for (NSString* name in functions) {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function == nil) {
        std::cerr << "Missing kernel " << [name UTF8String] << "\n";
        return 1;
      }
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
      if (pipeline == nil) {
        std::cerr << [[error localizedDescription] UTF8String] << "\n";
        return 1;
      }
      pipelines[pipeline_index++] = pipeline;
    }

    constexpr uint32_t width = 8;
    constexpr uint32_t height = 6;
    const size_t pixels = width * height;
    std::vector<uint8_t> reference(pixels);
    std::vector<uint8_t> source(pixels);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        reference[y * width + x] = static_cast<uint8_t>(17 * x + 11 * y);
        source[y * width + x] = reference[y * width + x];
      }
    }
    std::vector<float> source_depth(1, 0.0f);
    std::vector<float> poses(45, 0.0f);
    poses[0] = 8.0f;
    poses[1] = 3.5f;
    poses[2] = 8.0f;
    poses[3] = 2.5f;
    poses[4] = poses[8] = poses[12] = 1.0f;
    poses[13] = 0.1f;
    poses[16] = -0.1f;
    poses[43] = width;
    poses[44] = height;

    MetalParams params{};
    params.width = params.source_width = width;
    params.height = params.source_height = height;
    params.num_sources = 1;
    params.window_radius = 1;
    params.window_step = params.num_samples = 1;
    params.seed = 12345;
    params.pose_stride = 45;
    params.depth_min = 1.0f;
    params.depth_max = 4.0f;
    params.sigma_spatial = 1.0f;
    params.sigma_color = 0.2f;
    params.ncc_sigma = 0.6f;
    params.geom_consistency_max_cost = 3.0f;
    params.filter_min_ncc = -1.0f;
    params.filter_geom_consistency_max_cost = 1.0f;
    params.perturbation = 0.5f;
    params.ref_fx = params.ref_fy = 8.0f;
    params.ref_cx = 3.5f;
    params.ref_cy = 2.5f;
    params.ref_inv_fx = params.ref_inv_fy = 0.125f;
    params.ref_neg_cx_inv_fx = -params.ref_cx / params.ref_fx;
    params.ref_neg_cy_inv_fy = -params.ref_cy / params.ref_fy;
    params.filter_cos_min_triangulation_angle = std::cos(3.0f * 3.14159265358979323846f / 180.0f);
    params.cos_min_triangulation_angle = std::cos(1.0f * 3.14159265358979323846f / 180.0f);
    params.inv_incident_angle_sigma_square = -0.5f / (0.9f * 0.9f);
    params.ncc_norm_factor = 2.0f / (std::sqrt(2.0f * 3.14159265358979323846f) * params.ncc_sigma *
                                     std::erf(2.0f / (params.ncc_sigma * 1.414213562f)));

    const int radius = params.window_radius;
    const int spatial_stride = 2 * radius + 1;
    const float inv_spatial = 0.5f / std::max(1e-6f, params.sigma_spatial * params.sigma_spatial);
    std::vector<float> spatial_weights(spatial_stride * spatial_stride);
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        spatial_weights[(dy + radius) * spatial_stride + dx + radius] =
            std::exp(-(dx * dx + dy * dy) * inv_spatial);
      }
    }
    const float inv_color = 0.5f / std::max(1e-6f, params.sigma_color * params.sigma_color);
    std::vector<float> color_weights(256);
    for (size_t delta = 0; delta < color_weights.size(); ++delta) {
      const float normalized_delta = static_cast<float>(delta) / 255.0f;
      color_weights[delta] = std::exp(-normalized_delta * normalized_delta * inv_color);
    }

    id<MTLBuffer> reference_buffer = MakeBuffer(device, reference);
    MTLTextureDescriptor* source_descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    source_descriptor.textureType = MTLTextureType2DArray;
    source_descriptor.arrayLength = 1;
    source_descriptor.storageMode = MTLStorageModeShared;
    source_descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> source_texture = [device newTextureWithDescriptor:source_descriptor];
    [source_texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                      mipmapLevel:0
                            slice:0
                        withBytes:source.data()
                      bytesPerRow:width
                    bytesPerImage:pixels];
    id<MTLBuffer> source_depth_buffer = MakeBuffer(device, source_depth);
    id<MTLBuffer> pose_buffer = MakeBuffer(device, poses);
    id<MTLBuffer> spatial_weight_buffer = MakeBuffer(device, spatial_weights);
    id<MTLBuffer> color_weight_buffer = MakeBuffer(device, color_weights);
    id<MTLBuffer> depth_a = MakeBuffer(device, std::vector<float>(pixels, 0.0f));
    id<MTLBuffer> normal_a = MakeBuffer(device, std::vector<float>(pixels * 3, 0.0f));
    id<MTLBuffer> cost_a = MakeBuffer(device, std::vector<float>(pixels, 0.0f));
    id<MTLBuffer> probability = MakeBuffer(device, std::vector<float>(pixels, 0.5f));
    id<MTLBuffer> probability_current = MakeBuffer(device, std::vector<float>(pixels, 0.5f));
    std::vector<uint32_t> random_states(pixels);
    for (size_t index = 0; index < pixels; ++index) {
      random_states[index] = static_cast<uint32_t>(index) ^ params.seed;
    }
    id<MTLBuffer> random_state = MakeBuffer(device, random_states);
    id<MTLBuffer> workspace =
        MakeBuffer(device, std::vector<float>(2 * std::max(width, height), 0.5f));
    id<MTLBuffer> mask = MakeBuffer(device, std::vector<uint8_t>(pixels, 0));

    if (!Run([device newCommandQueue],
             pipelines[0],
             source_texture,
             spatial_weight_buffer,
             color_weight_buffer,
             {
               reference_buffer, nil, source_depth_buffer, pose_buffer, nil, depth_a, normal_a,
                   cost_a, nil, nil, random_state, spatial_weight_buffer, color_weight_buffer
             },
             params) ||
        !RunSweep([device newCommandQueue],
                  pipelines[1],
                  source_texture,
                  {
                    reference_buffer, nil, source_depth_buffer, pose_buffer, nil, depth_a, normal_a,
                        cost_a, probability, probability_current, random_state,
                        spatial_weight_buffer, color_weight_buffer, workspace
                  },
                  params) ||
        !Run([device newCommandQueue],
             pipelines[2],
             source_texture,
             spatial_weight_buffer,
             color_weight_buffer,
             {
               nil, nil, source_depth_buffer, pose_buffer, nil, depth_a, normal_a,
                   probability_current, mask
             },
             params)) {
      return 1;
    }

    const float* depths = static_cast<const float*>([depth_a contents]);
    const float* normals = static_cast<const float*>([normal_a contents]);
    const float* probabilities = static_cast<const float*>([probability_current contents]);
    for (size_t index = 0; index < pixels; ++index) {
      const float normal_length = std::sqrt(normals[index * 3] * normals[index * 3] +
                                            normals[index * 3 + 1] * normals[index * 3 + 1] +
                                            normals[index * 3 + 2] * normals[index * 3 + 2]);
      if (!std::isfinite(depths[index]) || !std::isfinite(normal_length) || normal_length < 0.99f ||
          normal_length > 1.01f || !std::isfinite(probabilities[index]) ||
          probabilities[index] < 0.0f || probabilities[index] > 1.0f) {
        std::cerr << "Invalid result at synthetic pixel " << index << ": depth=" << depths[index]
                  << ", normal_length=" << normal_length << ", probability=" << probabilities[index]
                  << "\n";
        return 1;
      }
    }

    std::cout << "Compiled and executed all Metal PatchMatch kernels on "
              << [[device name] UTF8String] << "\n";
    return 0;
  }
}
