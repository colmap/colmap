// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in COPYING.txt
// are met.

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/mvs/image.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace colmap {
namespace mvs {
namespace {

constexpr uint32_t kPoseStride = 45;

// Keep this layout synchronized with Params in kPatchMatchMetalSource.
struct alignas(16) MetalParams {
  uint32_t width;
  uint32_t height;
  uint32_t source_width;
  uint32_t source_height;

  uint32_t num_sources;
  uint32_t window_radius;
  uint32_t window_step;
  uint32_t num_samples;

  uint32_t sweep_direction;
  uint32_t iteration;
  uint32_t geom_consistency;
  uint32_t filter;

  uint32_t filter_min_num_consistent;
  uint32_t initialize_from_input;
  uint32_t seed;
  uint32_t pose_stride;

  float depth_min;
  float depth_max;
  float sigma_spatial;
  float sigma_color;

  float ncc_sigma;
  float geom_consistency_regularizer;
  float geom_consistency_max_cost;
  float filter_min_ncc;

  float filter_geom_consistency_max_cost;
  float perturbation;
  float ref_fx;
  float ref_cx;

  float ref_fy;
  float ref_cy;
  float ref_inv_fx;
  float ref_neg_cx_inv_fx;

  float ref_inv_fy;
  float ref_neg_cy_inv_fy;
  float filter_cos_min_triangulation_angle;
  float reserved1;

  float cos_min_triangulation_angle;
  float inv_incident_angle_sigma_square;
  float ncc_norm_factor;
  float prev_sel_prob_weight;
};

static_assert(sizeof(MetalParams) == 160, "MetalParams must match the Metal shader layout");

const char* kPatchMatchMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint width;
  uint height;
  uint source_width;
  uint source_height;

  uint num_sources;
  uint window_radius;
  uint window_step;
  uint num_samples;

  uint sweep_direction;
  uint iteration;
  uint geom_consistency;
  uint filter;

  uint filter_min_num_consistent;
  uint initialize_from_input;
  uint seed;
  uint pose_stride;

  float depth_min;
  float depth_max;
  float sigma_spatial;
  float sigma_color;

  float ncc_sigma;
  float geom_consistency_regularizer;
  float geom_consistency_max_cost;
  float filter_min_ncc;

  float filter_geom_consistency_max_cost;
  float perturbation;
  float ref_fx;
  float ref_cx;

  float ref_fy;
  float ref_cy;
  float ref_inv_fx;
  float ref_neg_cx_inv_fx;

  float ref_inv_fy;
  float ref_neg_cy_inv_fy;
  float filter_cos_min_triangulation_angle;
  float reserved1;

  float cos_min_triangulation_angle;
  float inv_incident_angle_sigma_square;
  float ncc_norm_factor;
  float prev_sel_prob_weight;
};

inline uint hash_u32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

inline float rand01(thread uint& state) {
  state = hash_u32(state + 0x9e3779b9u);
  return float(state & 0x00ffffffu) / float(0x01000000u);
}

inline uchar ref_byte(device const uchar* image,
                      int x,
                      int y,
                      constant Params& p) {
  if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height)) {
    return 0;
  }
  return image[uint(y) * p.width + uint(x)];
}

inline float source_bilinear(texture2d_array<float, access::sample> images,
                             uint source,
                             float x,
                             float y) {
  constexpr sampler linear_sampler(coord::pixel,
                                   address::clamp_to_zero,
                                   filter::linear);
  return images.sample(linear_sampler, float2(x + 0.5f, y + 0.5f), source).r;
}

inline float3 viewing_ray(uint x, uint y, constant Params& p) {
  return float3(p.ref_inv_fx * float(x) + p.ref_neg_cx_inv_fx,
                p.ref_inv_fy * float(y) + p.ref_neg_cy_inv_fy,
                1.0f);
}

inline float propagated_depth(uint x,
                              uint y,
                              uint nx,
                              uint ny,
                              float neighbor_depth,
                              float3 neighbor_normal,
                              constant Params& p) {
  const float3 neighbor_point = neighbor_depth * viewing_ray(nx, ny, p);
  const float denom = dot(neighbor_normal, viewing_ray(x, y, p));
  if (abs(denom) < 1e-6f) {
    return neighbor_depth;
  }
  return dot(neighbor_normal, neighbor_point) / denom;
}

inline void compose_homography(uint source,
                               uint x,
                               uint y,
                               float depth,
                               float3 normal,
                               device const float* poses,
                               constant Params& p,
                               thread float H[9]) {
  const uint b = source * p.pose_stride;
  const float fx = poses[b + 0];
  const float cx = poses[b + 1];
  const float fy = poses[b + 2];
  const float cy = poses[b + 3];
  device const float* R = poses + b + 4;
  device const float* T = poses + b + 13;
  const float dist = depth * dot(normal, viewing_ray(x, y, p));
  const float inv_dist = 1.0f / select(dist, copysign(1e-6f, dist),
                                       abs(dist) < 1e-6f);
  const float3 q = normal * inv_dist;

  H[0] = p.ref_inv_fx * (fx * (R[0] + q.x * T[0]) +
                             cx * (R[6] + q.x * T[2]));
  H[1] = p.ref_inv_fy * (fx * (R[1] + q.y * T[0]) +
                             cx * (R[7] + q.y * T[2]));
  H[2] = fx * (R[2] + q.z * T[0]) + cx * (R[8] + q.z * T[2]) +
         p.ref_neg_cx_inv_fx *
             (fx * (R[0] + q.x * T[0]) + cx * (R[6] + q.x * T[2])) +
         p.ref_neg_cy_inv_fy *
             (fx * (R[1] + q.y * T[0]) + cx * (R[7] + q.y * T[2]));
  H[3] = p.ref_inv_fx * (fy * (R[3] + q.x * T[1]) +
                             cy * (R[6] + q.x * T[2]));
  H[4] = p.ref_inv_fy * (fy * (R[4] + q.y * T[1]) +
                             cy * (R[7] + q.y * T[2]));
  H[5] = fy * (R[5] + q.z * T[1]) + cy * (R[8] + q.z * T[2]) +
         p.ref_neg_cx_inv_fx *
             (fy * (R[3] + q.x * T[1]) + cy * (R[6] + q.x * T[2])) +
         p.ref_neg_cy_inv_fy *
             (fy * (R[4] + q.y * T[1]) + cy * (R[7] + q.y * T[2]));
  H[6] = p.ref_inv_fx * (R[6] + q.x * T[2]);
  H[7] = p.ref_inv_fy * (R[7] + q.y * T[2]);
  H[8] = R[8] + p.ref_neg_cx_inv_fx * (R[6] + q.x * T[2]) +
         p.ref_neg_cy_inv_fy * (R[7] + q.y * T[2]) + q.z * T[2];
}

inline float photo_cost(device const uchar* reference,
                        texture2d_array<float, access::sample> sources,
                        device const float* poses,
                        constant float* spatial_weights,
                        constant float* color_weights,
                        uint source,
                        uint x,
                        uint y,
                        float depth,
                        float3 normal,
                        constant Params& p) {
  float H[9];
  compose_homography(source, x, y, depth, normal, poses, p, H);
  const uchar center_byte = ref_byte(reference, int(x), int(y), p);
  float ref_sum = 0.0f;
  float ref_sq_sum = 0.0f;
  float src_sum = 0.0f;
  float src_sq_sum = 0.0f;
  float cross_sum = 0.0f;
  float weight_sum = 0.0f;
  const int radius = int(p.window_radius);
  const int step = max(1, int(p.window_step));
  const uint spatial_stride = 2 * p.window_radius + 1;

  for (int dy = -radius; dy <= radius; dy += step) {
    for (int dx = -radius; dx <= radius; dx += step) {
      const int rx = int(x) + dx;
      const int ry = int(y) + dy;
      const uchar ref_byte_value = ref_byte(reference, rx, ry, p);
      const float ref = float(ref_byte_value) / 255.0f;
      const uint spatial_index =
          uint(dy + radius) * spatial_stride + uint(dx + radius);
      const uint color_index = uint(abs(int(center_byte) - int(ref_byte_value)));
      const float weight = spatial_weights[spatial_index] * color_weights[color_index];
      const float hx = H[0] * float(rx) + H[1] * float(ry) + H[2];
      const float hy = H[3] * float(rx) + H[4] * float(ry) + H[5];
      const float hz = H[6] * float(rx) + H[7] * float(ry) + H[8];
      const float src = abs(hz) < 1e-8f
                            ? 0.0f
                            : source_bilinear(sources, source, hx / hz, hy / hz);
      ref_sum += weight * ref;
      ref_sq_sum += weight * ref * ref;
      src_sum += weight * src;
      src_sq_sum += weight * src * src;
      cross_sum += weight * ref * src;
      weight_sum += weight;
    }
  }
  if (weight_sum <= 1e-8f) {
    return 2.0f;
  }
  const float inv_weight = 1.0f / weight_sum;
  ref_sum *= inv_weight;
  ref_sq_sum *= inv_weight;
  src_sum *= inv_weight;
  src_sq_sum *= inv_weight;
  cross_sum *= inv_weight;
  const float ref_var = ref_sq_sum - ref_sum * ref_sum;
  const float src_var = src_sq_sum - src_sum * src_sum;
  if (ref_var < 1e-5f || src_var < 1e-5f) {
    return 2.0f;
  }
  const float ncc = (cross_sum - ref_sum * src_sum) /
                    sqrt(max(1e-10f, ref_var * src_var));
  return clamp(1.0f - ncc, 0.0f, 2.0f);
}

inline float source_depth_value(device const float* depths,
                                uint source,
                                int x,
                                int y,
                                constant Params& p,
                                device const float* poses) {
  const uint b = source * p.pose_stride;
  const int width = int(poses[b + 43]);
  const int height = int(poses[b + 44]);
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return 0.0f;
  }
  const uint stride = p.source_width * p.source_height;
  return depths[source * stride + uint(y) * p.source_width + uint(x)];
}

inline float geom_cost(device const float* source_depths,
                       device const float* poses,
                       uint source,
                       uint x,
                       uint y,
                       float depth,
                       constant Params& p) {
  const uint b = source * p.pose_stride;
  device const float* P = poses + b + 19;
  device const float* invP = poses + b + 31;
  const float3 point = depth * viewing_ray(x, y, p);
  const float z = P[8] * point.x + P[9] * point.y + P[10] * point.z + P[11];
  if (abs(z) < 1e-8f) {
    return p.geom_consistency_max_cost;
  }
  const float sx = (P[0] * point.x + P[1] * point.y + P[2] * point.z + P[3]) / z;
  const float sy = (P[4] * point.x + P[5] * point.y + P[6] * point.z + P[7]) / z;
  const float source_depth = source_depth_value(
      source_depths, source, int(round(sx)), int(round(sy)), p, poses);
  if (source_depth <= 0.0f) {
    return p.geom_consistency_max_cost;
  }
  const float px = sx * source_depth;
  const float py = sy * source_depth;
  const float3 backward = float3(
      invP[0] * px + invP[1] * py + invP[2] * source_depth + invP[3],
      invP[4] * px + invP[5] * py + invP[6] * source_depth + invP[7],
      invP[8] * px + invP[9] * py + invP[10] * source_depth + invP[11]);
  if (abs(backward.z) < 1e-8f) {
    return p.geom_consistency_max_cost;
  }
  const float bx = p.ref_fx * backward.x / backward.z + p.ref_cx;
  const float by = p.ref_fy * backward.y / backward.z + p.ref_cy;
  return min(p.geom_consistency_max_cost,
             length(float2(float(x) - bx, float(y) - by)));
}

inline bool has_stable_view(device const float* poses,
                            uint source,
                            uint x,
                            uint y,
                            float depth,
                            float3 normal,
                            constant Params& p) {
  const uint b = source * p.pose_stride;
  const float3 point = depth * viewing_ray(x, y, p);
  const float3 source_ray = float3(poses[b + 16],
                                   poses[b + 17],
                                   poses[b + 18]) - point;
  const float point_norm = length(point);
  const float source_norm = length(source_ray);
  if (point_norm < 1e-8f || source_norm < 1e-8f) {
    return false;
  }
  const float cos_incident = dot(source_ray, normal) / source_norm;
  const float cos_triangulation =
      -dot(source_ray, point) / (source_norm * point_norm);
  return cos_incident > 0.0f &&
         cos_triangulation <= p.filter_cos_min_triangulation_angle;
}

inline float aggregate_cost(device const uchar* reference,
                            texture2d_array<float, access::sample> sources,
                            device const float* source_depths,
                            device const float* poses,
                            constant float* spatial_weights,
                            constant float* color_weights,
                            uint x,
                            uint y,
                            float depth,
                            float3 normal,
                            constant Params& p) {
  if (!isfinite(depth) || depth <= p.depth_min || depth >= p.depth_max) {
    return 1e6f;
  }
  float best[8];
  for (uint i = 0; i < 8; ++i) {
    best[i] = 1e6f;
  }
  const uint keep = max(1u, min(8u, min(p.num_samples, p.num_sources)));
  for (uint source = 0; source < p.num_sources; ++source) {
    float cost = photo_cost(reference,
                            sources,
                            poses,
                            spatial_weights,
                            color_weights,
                            source,
                            x,
                            y,
                            depth,
                            normal,
                            p);
    if (p.geom_consistency != 0) {
      cost += p.geom_consistency_regularizer *
              geom_cost(source_depths, poses, source, x, y, depth, p);
    }
    if (cost < best[keep - 1]) {
      best[keep - 1] = cost;
      for (uint j = keep - 1; j > 0 && best[j] < best[j - 1]; --j) {
        const float temp = best[j - 1];
        best[j - 1] = best[j];
        best[j] = temp;
      }
    }
  }
  float total = 0.0f;
  for (uint i = 0; i < keep; ++i) {
    total += best[i];
  }
  return total / float(keep);
}

inline float3 random_normal(uint x,
                            uint y,
                            thread uint& state,
                            constant Params& p) {
  const float z = 2.0f * rand01(state) - 1.0f;
  const float angle = 6.28318530718f * rand01(state);
  const float radius = sqrt(max(0.0f, 1.0f - z * z));
  float3 normal = float3(radius * cos(angle), radius * sin(angle), z);
  if (dot(normal, viewing_ray(x, y, p)) > 0.0f) {
    normal = -normal;
  }
  return normal;
}

kernel void patch_match_initialize(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device float* cost [[buffer(7)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  const uint index = gid.y * p.width + gid.x;
  if (p.initialize_from_input == 0) {
    uint state = hash_u32(index ^ p.seed);
    depth[index] = mix(p.depth_min, p.depth_max, rand01(state));
    const float3 n = random_normal(gid.x, gid.y, state, p);
    normal[index * 3 + 0] = n.x;
    normal[index * 3 + 1] = n.y;
    normal[index * 3 + 2] = n.z;
  }
  const float3 n = float3(normal[index * 3 + 0],
                          normal[index * 3 + 1],
                          normal[index * 3 + 2]);
  cost[index] = aggregate_cost(reference,
                               sources,
                               source_depths,
                               poses,
                               spatial_weights,
                               color_weights,
                               gid.x,
                               gid.y,
                               depth[index],
                               n,
                               p);
}

kernel void patch_match_initialize_fidelity(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device float* source_cost [[buffer(7)]],
    device uint* random_state [[buffer(10)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  const uint index = gid.y * p.width + gid.x;
  uint state = random_state[index];
  if (p.initialize_from_input == 0) {
    depth[index] = mix(p.depth_min, p.depth_max, rand01(state));
    const float3 n = random_normal(gid.x, gid.y, state, p);
    normal[index * 3 + 0] = n.x;
    normal[index * 3 + 1] = n.y;
    normal[index * 3 + 2] = n.z;
  }
  random_state[index] = state;
  const float3 n = float3(normal[index * 3 + 0],
                          normal[index * 3 + 1],
                          normal[index * 3 + 2]);
  for (uint source = 0; source < p.num_sources; ++source) {
    source_cost[index * p.num_sources + source] =
        photo_cost(reference,
                   sources,
                   poses,
                   spatial_weights,
                   color_weights,
                   source,
                   gid.x,
                   gid.y,
                   depth[index],
                   n,
                   p);
  }
}

inline void consider_candidate(device const uchar* reference,
                               texture2d_array<float, access::sample> sources,
                               device const float* source_depths,
                               device const float* poses,
                               constant float* spatial_weights,
                               constant float* color_weights,
                               uint x,
                               uint y,
                               float candidate_depth,
                               float3 candidate_normal,
                               constant Params& p,
                               thread float& best_depth,
                               thread float3& best_normal,
                               thread float& best_cost) {
  const float candidate_cost = aggregate_cost(reference,
                                              sources,
                                              source_depths,
                                              poses,
                                              spatial_weights,
                                              color_weights,
                                              x,
                                              y,
                                              candidate_depth,
                                              candidate_normal,
                                              p);
  if (candidate_cost < best_cost) {
    best_cost = candidate_cost;
    best_depth = candidate_depth;
    best_normal = candidate_normal;
  }
}

inline uint2 physical_coord(uint col,
                            uint row,
                            uint direction,
                            constant Params& p) {
  if (direction == 0) {
    return uint2(col, row);
  } else if (direction == 1) {
    return uint2(p.width - 1 - row, col);
  } else if (direction == 2) {
    return uint2(p.width - 1 - col, p.height - 1 - row);
  }
  return uint2(row, p.height - 1 - col);
}

inline float3 vector_to_local(float3 value, uint direction) {
  if (direction == 0) {
    return value;
  } else if (direction == 1) {
    return float3(value.y, -value.x, value.z);
  } else if (direction == 2) {
    return float3(-value.x, -value.y, value.z);
  }
  return float3(-value.y, value.x, value.z);
}

inline float3 vector_from_local(float3 value, uint direction) {
  if (direction == 0) {
    return value;
  } else if (direction == 1) {
    return float3(-value.y, value.x, value.z);
  } else if (direction == 2) {
    return float3(-value.x, -value.y, value.z);
  }
  return float3(value.y, -value.x, value.z);
}

inline float local_inv_fy(uint direction, constant Params& p) {
  return (direction & 1u) == 0 ? p.ref_inv_fy : p.ref_inv_fx;
}

inline float local_neg_cy_inv_fy(uint direction, constant Params& p) {
  if (direction == 0) {
    return p.ref_neg_cy_inv_fy;
  } else if (direction == 1) {
    return -(float(p.width - 1) - p.ref_cx) * p.ref_inv_fx;
  } else if (direction == 2) {
    return -(float(p.height - 1) - p.ref_cy) * p.ref_inv_fy;
  }
  return p.ref_neg_cx_inv_fx;
}

inline float propagate_local_depth(float depth,
                                   float3 local_normal,
                                   float row1,
                                   float row2,
                                   uint direction,
                                   constant Params& p) {
  const float inv_fy = local_inv_fy(direction, p);
  const float neg_cy_inv_fy = local_neg_cy_inv_fy(direction, p);
  const float x1 = depth * (inv_fy * row1 + neg_cy_inv_fy);
  const float y1 = depth;
  const float x2 = x1 + local_normal.z;
  const float y2 = y1 - local_normal.y;
  const float x4 = inv_fy * row2 + neg_cy_inv_fy;
  const float denom = x2 - x1 + x4 * (y1 - y2);
  if (abs(denom) < 1e-5f) {
    return depth;
  }
  return (y1 * x2 - x1 * y2) / denom;
}

inline float3 perturb_normal_local(float3 normal,
                                   float3 view_ray,
                                   float perturbation,
                                   thread uint& state) {
  float trial_perturbation = perturbation;
  for (uint trial = 0; trial < 4; ++trial) {
    const float a1 = (rand01(state) - 0.5f) * trial_perturbation;
    const float a2 = (rand01(state) - 0.5f) * trial_perturbation;
    const float a3 = (rand01(state) - 0.5f) * trial_perturbation;
    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);
    const float3 candidate =
        float3(cos_a2 * cos_a3 * normal.x - cos_a2 * sin_a3 * normal.y +
                   sin_a2 * normal.z,
               (cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2) * normal.x +
                   (cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3) * normal.y -
                   cos_a2 * sin_a1 * normal.z,
               (sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2) * normal.x +
                   (cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3) * normal.y +
                   cos_a1 * cos_a2 * normal.z);
    if (dot(candidate, view_ray) < 0.0f) {
      return normalize(candidate);
    }
    trial_perturbation *= 0.5f;
  }
  return normal;
}

inline float ncc_probability(float cost, constant Params& p) {
  const float inv_sigma_square =
      -0.5f / max(1e-8f, p.ncc_sigma * p.ncc_sigma);
  return exp(cost * cost * inv_sigma_square) * p.ncc_norm_factor;
}

inline float forward_message(float cost, float previous, constant Params& p) {
  constexpr float uniform_probability = 0.5f;
  constexpr float no_change_probability = 0.99999f;
  constexpr float change_probability = 1.0f - no_change_probability;
  const float emission = ncc_probability(cost, p);
  const float z0 =
      (previous * change_probability +
       (1.0f - previous) * no_change_probability) * uniform_probability;
  const float z1 =
      (previous * no_change_probability +
       (1.0f - previous) * change_probability) * emission;
  return z1 / max(1e-12f, z0 + z1);
}

inline float backward_message(float cost, float previous, constant Params& p) {
  constexpr float uniform_probability = 0.5f;
  constexpr float no_change_probability = 0.99999f;
  constexpr float change_probability = 1.0f - no_change_probability;
  const float emission = ncc_probability(cost, p);
  const float z0 = previous * emission * change_probability +
                   (1.0f - previous) * uniform_probability *
                       no_change_probability;
  const float z1 = previous * emission * no_change_probability +
                   (1.0f - previous) * uniform_probability * change_probability;
  return z1 / max(1e-12f, z0 + z1);
}

inline float selection_probability(float alpha,
                                   float beta,
                                   float previous,
                                   constant Params& p) {
  const float z0 = (1.0f - alpha) * (1.0f - beta);
  const float z1 = alpha * beta;
  const float current = z1 / max(1e-12f, z0 + z1);
  return p.prev_sel_prob_weight * previous +
         (1.0f - p.prev_sel_prob_weight) * current;
}

inline void viewing_angles(device const float* poses,
                           uint source,
                           float3 point,
                           float3 normal,
                           constant Params& p,
                           thread float& cos_triangulation,
                           thread float& cos_incident) {
  const uint b = source * p.pose_stride;
  const float3 source_ray =
      float3(poses[b + 16], poses[b + 17], poses[b + 18]) - point;
  const float point_norm = length(point);
  const float source_norm = length(source_ray);
  if (point_norm < 1e-8f || source_norm < 1e-8f) {
    cos_triangulation = 1.0f;
    cos_incident = 0.0f;
    return;
  }
  cos_incident = dot(source_ray, normal) / source_norm;
  cos_triangulation = -dot(source_ray, point) / (source_norm * point_norm);
}

inline float triangulation_probability(float cosine, constant Params& p) {
  if (cosine <= p.cos_min_triangulation_angle) {
    return 1.0f;
  }
  const float scaled =
      1.0f - (1.0f - cosine) / (1.0f - p.cos_min_triangulation_angle);
  return clamp(1.0f - scaled * scaled, 0.0f, 1.0f);
}

inline float incident_probability(float cosine, constant Params& p) {
  const float value = 1.0f - max(0.0f, cosine);
  return exp(value * value * p.inv_incident_angle_sigma_square);
}

inline float2 warp_point(thread const float H[9], float x, float y) {
  const float z = H[6] * x + H[7] * y + H[8];
  const float inv_z = 1.0f / z;
  return float2(inv_z * (H[0] * x + H[1] * y + H[2]),
                inv_z * (H[3] * x + H[4] * y + H[5]));
}

inline float resolution_probability(device const float* poses,
                                    uint source,
                                    uint x,
                                    uint y,
                                    float depth,
                                    float3 normal,
                                    constant Params& p) {
  float H[9];
  compose_homography(source, x, y, depth, normal, poses, p, H);
  const float radius = float(p.window_radius);
  const float2 p1 = warp_point(H, float(x) - radius, float(y) - radius);
  const float2 p2 = warp_point(H, float(x) - radius, float(y) + radius);
  const float2 p3 = warp_point(H, float(x) + radius, float(y) + radius);
  const float2 p4 = warp_point(H, float(x) + radius, float(y) - radius);
  const float source_area =
      abs(0.5f * (p1.x * p2.y - p2.x * p1.y - p1.x * p4.y +
                  p2.x * p3.y - p3.x * p2.y + p4.x * p1.y +
                  p3.x * p4.y - p4.x * p3.y));
  const float window_size = float(2 * p.window_radius + 1);
  const float reference_area = window_size * window_size;
  if (!isfinite(source_area) || source_area <= 1e-12f) {
    return 0.0f;
  }
  return source_area < reference_area ? source_area / reference_area
                                      : reference_area / source_area;
}

kernel void patch_match_column_sweep(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device float* source_cost [[buffer(7)]],
    device const float* previous_probability [[buffer(8)]],
    device float* probability [[buffer(9)]],
    device uint* random_state [[buffer(10)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    device float* workspace [[buffer(13)]],
    uint col [[thread_position_in_grid]]) {
  const uint logical_width =
      (p.sweep_direction & 1u) == 0 ? p.width : p.height;
  const uint logical_height =
      (p.sweep_direction & 1u) == 0 ? p.height : p.width;
  if (col >= logical_width) {
    return;
  }
  const uint max_dimension = max(p.width, p.height);
  const uint workspace_stride = max_dimension * p.num_sources;
  const uint column_workspace = col * p.num_sources;
  constexpr float uniform_probability = 0.5f;

  for (uint source = 0; source < p.num_sources; ++source) {
    float beta = uniform_probability;
    for (int row = int(logical_height) - 1; row >= 0; --row) {
      const uint2 xy = physical_coord(col, uint(row), p.sweep_direction, p);
      const uint index = xy.y * p.width + xy.x;
      const uint source_index = index * p.num_sources + source;
      beta = backward_message(source_cost[source_index], beta, p);
      probability[source_index] = beta;
    }
    workspace[column_workspace + source] = uniform_probability;
  }

  const uint2 boundary_xy = physical_coord(col, 0, p.sweep_direction, p);
  const uint boundary_index = boundary_xy.y * p.width + boundary_xy.x;
  uint state = random_state[boundary_index];
  float previous_depth = depth[boundary_index];
  float3 previous_normal =
      float3(normal[boundary_index * 3 + 0],
             normal[boundary_index * 3 + 1],
             normal[boundary_index * 3 + 2]);

  for (uint row = 0; row < logical_height; ++row) {
    const uint2 xy = physical_coord(col, row, p.sweep_direction, p);
    const uint index = xy.y * p.width + xy.x;
    const float3 previous_local_normal =
        vector_to_local(previous_normal, p.sweep_direction);
    previous_depth = propagate_local_depth(previous_depth,
                                           previous_local_normal,
                                           float(row) - 1.0f,
                                           float(row),
                                           p.sweep_direction,
                                           p);

    const float current_depth = depth[index];
    const float3 current_normal =
        float3(normal[index * 3 + 0],
               normal[index * 3 + 1],
               normal[index * 3 + 2]);
    const float random_depth =
        mix((1.0f - p.perturbation) * current_depth,
            (1.0f + p.perturbation) * current_depth,
            rand01(state));
    const float3 current_local_normal =
        vector_to_local(current_normal, p.sweep_direction);
    const float3 local_view_ray =
        vector_to_local(viewing_ray(xy.x, xy.y, p), p.sweep_direction);
    const float3 random_local_normal =
        perturb_normal_local(current_local_normal,
                             local_view_ray,
                             p.perturbation * 3.14159265358979323846f,
                             state);
    const float3 random_normal =
        vector_from_local(random_local_normal, p.sweep_direction);
    const float3 point = current_depth * viewing_ray(xy.x, xy.y, p);

    float probability_sum = 0.0f;
    for (uint source = 0; source < p.num_sources; ++source) {
      const uint source_index = index * p.num_sources + source;
      const float cost = source_cost[source_index];
      const float alpha =
          forward_message(cost, workspace[column_workspace + source], p);
      const float beta = probability[source_index];
      const float selected_probability =
          selection_probability(alpha, beta, previous_probability[source_index], p);
      float cos_triangulation;
      float cos_incident;
      viewing_angles(poses,
                     source,
                     point,
                     current_normal,
                     p,
                     cos_triangulation,
                     cos_incident);
      const float prior = triangulation_probability(cos_triangulation, p) *
                          incident_probability(cos_incident, p) *
                          resolution_probability(poses,
                                                 source,
                                                 xy.x,
                                                 xy.y,
                                                 current_depth,
                                                 current_normal,
                                                 p);
      probability_sum += selected_probability * prior;
      workspace[workspace_stride + column_workspace + source] = probability_sum;
    }
    if (!isfinite(probability_sum) || probability_sum <= 1e-12f) {
      for (uint source = 0; source < p.num_sources; ++source) {
        workspace[workspace_stride + column_workspace + source] =
            float(source + 1) / float(p.num_sources);
      }
    } else {
      const float inverse_sum = 1.0f / probability_sum;
      for (uint source = 0; source < p.num_sources; ++source) {
        workspace[workspace_stride + column_workspace + source] *= inverse_sum;
      }
    }

    float candidate_depth[5] = {
        current_depth, previous_depth, random_depth, current_depth, random_depth};
    float3 candidate_normal[5] = {current_normal,
                                  previous_normal,
                                  random_normal,
                                  random_normal,
                                  current_normal};
    float candidate_cost[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint sample = 0; sample < p.num_samples; ++sample) {
      const float draw = rand01(state) - 1.1920928955078125e-7f;
      uint selected_source = p.num_sources - 1;
      for (uint source = 0; source < p.num_sources; ++source) {
        if (workspace[workspace_stride + column_workspace + source] > draw) {
          selected_source = source;
          break;
        }
      }
      const uint selected_index = index * p.num_sources + selected_source;
      candidate_cost[0] += source_cost[selected_index];
      if (p.geom_consistency != 0) {
        candidate_cost[0] +=
            p.geom_consistency_regularizer *
            geom_cost(source_depths,
                      poses,
                      selected_source,
                      xy.x,
                      xy.y,
                      candidate_depth[0],
                      p);
      }
      for (uint candidate = 1; candidate < 5; ++candidate) {
        candidate_cost[candidate] +=
            photo_cost(reference,
                       sources,
                       poses,
                       spatial_weights,
                       color_weights,
                       selected_source,
                       xy.x,
                       xy.y,
                       candidate_depth[candidate],
                       candidate_normal[candidate],
                       p);
        if (p.geom_consistency != 0) {
          candidate_cost[candidate] +=
              p.geom_consistency_regularizer *
              geom_cost(source_depths,
                        poses,
                        selected_source,
                        xy.x,
                        xy.y,
                        candidate_depth[candidate],
                        p);
        }
      }
    }

    uint best = 0;
    for (uint candidate = 1; candidate < 5; ++candidate) {
      if (candidate_cost[candidate] <= candidate_cost[best]) {
        best = candidate;
      }
    }
    const float best_depth = candidate_depth[best];
    const float3 best_normal = candidate_normal[best];
    depth[index] = best_depth;
    normal[index * 3 + 0] = best_normal.x;
    normal[index * 3 + 1] = best_normal.y;
    normal[index * 3 + 2] = best_normal.z;

    for (uint source = 0; source < p.num_sources; ++source) {
      const uint source_index = index * p.num_sources + source;
      float cost = source_cost[source_index];
      if (best != 0) {
        cost = photo_cost(reference,
                          sources,
                          poses,
                          spatial_weights,
                          color_weights,
                          source,
                          xy.x,
                          xy.y,
                          best_depth,
                          best_normal,
                          p);
        source_cost[source_index] = cost;
      }
      const float alpha =
          forward_message(cost, workspace[column_workspace + source], p);
      const float prob = selection_probability(
          alpha, probability[source_index], previous_probability[source_index], p);
      workspace[column_workspace + source] = alpha;
      probability[source_index] = prob;
    }
    previous_depth = best_depth;
    previous_normal = best_normal;
  }
  random_state[boundary_index] = state;
}

kernel void patch_match_column_sweep_simd(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device float* source_cost [[buffer(7)]],
    device const float* previous_probability [[buffer(8)]],
    device float* probability [[buffer(9)]],
    device uint* random_state [[buffer(10)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    device float* workspace [[buffer(13)]],
    threadgroup float* candidate_scratch [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint threadgroup_lane [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  const uint col = gid / simd_width;
  const uint logical_width =
      (p.sweep_direction & 1u) == 0 ? p.width : p.height;
  const uint logical_height =
      (p.sweep_direction & 1u) == 0 ? p.height : p.width;
  if (col >= logical_width) {
    return;
  }
  const uint max_dimension = max(p.width, p.height);
  const uint workspace_stride = max_dimension * p.num_sources;
  const uint column_workspace = col * p.num_sources;
  const uint simdgroup_index = threadgroup_lane / simd_width;
  threadgroup float* selected_sources =
      candidate_scratch + simdgroup_index * 5 * p.num_samples;
  threadgroup float* sampled_costs = selected_sources + p.num_samples;
  constexpr float uniform_probability = 0.5f;

  for (uint source = lane; source < p.num_sources; source += simd_width) {
    float beta = uniform_probability;
    for (int row = int(logical_height) - 1; row >= 0; --row) {
      const uint2 xy = physical_coord(col, uint(row), p.sweep_direction, p);
      const uint index = xy.y * p.width + xy.x;
      const uint source_index = index * p.num_sources + source;
      beta = backward_message(source_cost[source_index], beta, p);
      probability[source_index] = beta;
    }
    workspace[column_workspace + source] = uniform_probability;
  }
  simdgroup_barrier(mem_flags::mem_device);

  const uint2 boundary_xy = physical_coord(col, 0, p.sweep_direction, p);
  const uint boundary_index = boundary_xy.y * p.width + boundary_xy.x;
  uint state = 0;
  float previous_depth = 0.0f;
  float3 previous_normal = float3(0.0f);
  if (lane == 0) {
    state = random_state[boundary_index];
    previous_depth = depth[boundary_index];
    previous_normal =
        float3(normal[boundary_index * 3 + 0],
               normal[boundary_index * 3 + 1],
               normal[boundary_index * 3 + 2]);
  }
  previous_depth = simd_broadcast(previous_depth, 0);
  previous_normal.x = simd_broadcast(previous_normal.x, 0);
  previous_normal.y = simd_broadcast(previous_normal.y, 0);
  previous_normal.z = simd_broadcast(previous_normal.z, 0);

  for (uint row = 0; row < logical_height; ++row) {
    const uint2 xy = physical_coord(col, row, p.sweep_direction, p);
    const uint index = xy.y * p.width + xy.x;
    float current_depth = 0.0f;
    float3 current_normal = float3(0.0f);
    float random_depth = 0.0f;
    float3 random_normal = float3(0.0f);
    if (lane == 0) {
      const float3 previous_local_normal =
          vector_to_local(previous_normal, p.sweep_direction);
      previous_depth = propagate_local_depth(previous_depth,
                                             previous_local_normal,
                                             float(row) - 1.0f,
                                             float(row),
                                             p.sweep_direction,
                                             p);
      current_depth = depth[index];
      current_normal = float3(normal[index * 3 + 0],
                              normal[index * 3 + 1],
                              normal[index * 3 + 2]);
      random_depth = mix((1.0f - p.perturbation) * current_depth,
                         (1.0f + p.perturbation) * current_depth,
                         rand01(state));
      const float3 current_local_normal =
          vector_to_local(current_normal, p.sweep_direction);
      const float3 local_view_ray =
          vector_to_local(viewing_ray(xy.x, xy.y, p), p.sweep_direction);
      const float3 random_local_normal =
          perturb_normal_local(current_local_normal,
                               local_view_ray,
                               p.perturbation * 3.14159265358979323846f,
                               state);
      random_normal = vector_from_local(random_local_normal, p.sweep_direction);
    }
    previous_depth = simd_broadcast(previous_depth, 0);
    previous_normal.x = simd_broadcast(previous_normal.x, 0);
    previous_normal.y = simd_broadcast(previous_normal.y, 0);
    previous_normal.z = simd_broadcast(previous_normal.z, 0);
    current_depth = simd_broadcast(current_depth, 0);
    current_normal.x = simd_broadcast(current_normal.x, 0);
    current_normal.y = simd_broadcast(current_normal.y, 0);
    current_normal.z = simd_broadcast(current_normal.z, 0);
    random_depth = simd_broadcast(random_depth, 0);
    random_normal.x = simd_broadcast(random_normal.x, 0);
    random_normal.y = simd_broadcast(random_normal.y, 0);
    random_normal.z = simd_broadcast(random_normal.z, 0);
    const float3 point = current_depth * viewing_ray(xy.x, xy.y, p);

    for (uint source = lane; source < p.num_sources; source += simd_width) {
      const uint source_index = index * p.num_sources + source;
      const float cost = source_cost[source_index];
      const float alpha =
          forward_message(cost, workspace[column_workspace + source], p);
      const float selected_probability =
          selection_probability(alpha,
                                probability[source_index],
                                previous_probability[source_index],
                                p);
      float cos_triangulation;
      float cos_incident;
      viewing_angles(poses,
                     source,
                     point,
                     current_normal,
                     p,
                     cos_triangulation,
                     cos_incident);
      workspace[workspace_stride + column_workspace + source] =
          selected_probability * triangulation_probability(cos_triangulation, p) *
          incident_probability(cos_incident, p) *
          resolution_probability(poses,
                                 source,
                                 xy.x,
                                 xy.y,
                                 current_depth,
                                 current_normal,
                                 p);
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
      float probability_sum = 0.0f;
      for (uint source = 0; source < p.num_sources; ++source) {
        probability_sum += workspace[workspace_stride + column_workspace + source];
      }
      if (!isfinite(probability_sum) || probability_sum <= 1e-12f) {
        for (uint source = 0; source < p.num_sources; ++source) {
          workspace[workspace_stride + column_workspace + source] =
              float(source + 1) / float(p.num_sources);
        }
      } else {
        const float inverse_sum = 1.0f / probability_sum;
        float cumulative_probability = 0.0f;
        for (uint source = 0; source < p.num_sources; ++source) {
          cumulative_probability +=
              workspace[workspace_stride + column_workspace + source] * inverse_sum;
          workspace[workspace_stride + column_workspace + source] =
              cumulative_probability;
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_device);

    float candidate_depth[5] = {
        current_depth, previous_depth, random_depth, current_depth, random_depth};
    float3 candidate_normal[5] = {current_normal,
                                  previous_normal,
                                  random_normal,
                                  random_normal,
                                  current_normal};
    float candidate_cost[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (lane == 0) {
      for (uint sample = 0; sample < p.num_samples; ++sample) {
        const float draw = rand01(state) - 1.1920928955078125e-7f;
        uint selected_source = p.num_sources;
        for (uint source = 0; source < p.num_sources; ++source) {
          if (workspace[workspace_stride + column_workspace + source] > draw) {
            selected_source = source;
            break;
          }
        }
        selected_sources[sample] = float(selected_source);
        if (selected_source < p.num_sources) {
          float cached_cost =
              source_cost[index * p.num_sources + selected_source];
          if (p.geom_consistency != 0) {
            cached_cost += p.geom_consistency_regularizer *
                           geom_cost(source_depths,
                                     poses,
                                     selected_source,
                                     xy.x,
                                     xy.y,
                                     candidate_depth[0],
                                     p);
          }
          candidate_cost[0] += cached_cost;
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint sampled_cost_count = 4 * p.num_samples;
    for (uint task = lane; task < sampled_cost_count; task += simd_width) {
      const uint candidate = task / p.num_samples + 1;
      const uint sample = task % p.num_samples;
      const uint selected_source = uint(selected_sources[sample]);
      float cost = 0.0f;
      if (selected_source < p.num_sources) {
        cost = photo_cost(reference,
                          sources,
                          poses,
                          spatial_weights,
                          color_weights,
                          selected_source,
                          xy.x,
                          xy.y,
                          candidate_depth[candidate],
                          candidate_normal[candidate],
                          p);
        if (p.geom_consistency != 0) {
          cost += p.geom_consistency_regularizer *
                  geom_cost(source_depths,
                            poses,
                            selected_source,
                            xy.x,
                            xy.y,
                            candidate_depth[candidate],
                            p);
        }
      }
      sampled_costs[task] = cost;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      for (uint candidate = 1; candidate < 5; ++candidate) {
        for (uint sample = 0; sample < p.num_samples; ++sample) {
          candidate_cost[candidate] +=
              sampled_costs[(candidate - 1) * p.num_samples + sample];
        }
      }
    }

    uint best = 0;
    if (lane == 0) {
      for (uint candidate = 1; candidate < 5; ++candidate) {
        if (candidate_cost[candidate] <= candidate_cost[best]) {
          best = candidate;
        }
      }
    }
    best = simd_broadcast(best, 0);
    const float best_depth = candidate_depth[best];
    const float3 best_normal = candidate_normal[best];
    if (lane == 0) {
      depth[index] = best_depth;
      normal[index * 3 + 0] = best_normal.x;
      normal[index * 3 + 1] = best_normal.y;
      normal[index * 3 + 2] = best_normal.z;
    }

    for (uint source = lane; source < p.num_sources; source += simd_width) {
      const uint source_index = index * p.num_sources + source;
      float cost = source_cost[source_index];
      if (best != 0) {
        cost = photo_cost(reference,
                          sources,
                          poses,
                          spatial_weights,
                          color_weights,
                          source,
                          xy.x,
                          xy.y,
                          best_depth,
                          best_normal,
                          p);
        source_cost[source_index] = cost;
      }
      const float alpha =
          forward_message(cost, workspace[column_workspace + source], p);
      probability[source_index] =
          selection_probability(alpha,
                                probability[source_index],
                                previous_probability[source_index],
                                p);
      workspace[column_workspace + source] = alpha;
    }
    simdgroup_barrier(mem_flags::mem_device);
    previous_depth = best_depth;
    previous_normal = best_normal;
  }
  if (lane == 0) {
    random_state[boundary_index] = state;
  }
}

kernel void patch_match_sweep(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device const float* depth_in [[buffer(5)]],
    device const float* normal_in [[buffer(6)]],
    device const float* cost_in [[buffer(7)]],
    device float* depth_out [[buffer(8)]],
    device float* normal_out [[buffer(9)]],
    device float* cost_out [[buffer(10)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  const uint index = gid.y * p.width + gid.x;
  float best_depth = depth_in[index];
  float3 best_normal = float3(normal_in[index * 3 + 0],
                              normal_in[index * 3 + 1],
                              normal_in[index * 3 + 2]);
  float best_cost = cost_in[index];
  const int sx = (p.sweep_direction == 0 || p.sweep_direction == 3) ? -1 : 1;
  const int sy = (p.sweep_direction == 0 || p.sweep_direction == 2) ? -1 : 1;
  const int nx[2] = {int(gid.x) + sx, int(gid.x)};
  const int ny[2] = {int(gid.y), int(gid.y) + sy};
  for (uint candidate = 0; candidate < 2; ++candidate) {
    if (nx[candidate] < 0 || ny[candidate] < 0 ||
        nx[candidate] >= int(p.width) || ny[candidate] >= int(p.height)) {
      continue;
    }
    const uint neighbor = uint(ny[candidate]) * p.width + uint(nx[candidate]);
    const float3 neighbor_normal =
        float3(normal_in[neighbor * 3 + 0],
               normal_in[neighbor * 3 + 1],
               normal_in[neighbor * 3 + 2]);
    const float candidate_depth = propagated_depth(gid.x,
                                                   gid.y,
                                                   uint(nx[candidate]),
                                                   uint(ny[candidate]),
                                                   depth_in[neighbor],
                                                   neighbor_normal,
                                                   p);
    consider_candidate(reference,
                       sources,
                       source_depths,
                       poses,
                       spatial_weights,
                       color_weights,
                       gid.x,
                       gid.y,
                       candidate_depth,
                       neighbor_normal,
                       p,
                       best_depth,
                       best_normal,
                       best_cost);
  }

  uint state = hash_u32(index ^ p.seed ^ (p.iteration * 0x85ebca6bu) ^
                        (p.sweep_direction * 0xc2b2ae35u));
  const float depth_scale = max(1e-5f, best_depth * p.perturbation);
  const float random_depth = clamp(best_depth +
                                       (2.0f * rand01(state) - 1.0f) * depth_scale,
                                   p.depth_min,
                                   p.depth_max);
  float3 random_normal = normalize(
      best_normal +
      p.perturbation *
          float3(2.0f * rand01(state) - 1.0f,
                 2.0f * rand01(state) - 1.0f,
                 2.0f * rand01(state) - 1.0f));
  if (dot(random_normal, viewing_ray(gid.x, gid.y, p)) > 0.0f) {
    random_normal = -random_normal;
  }
  consider_candidate(reference,
                     sources,
                     source_depths,
                     poses,
                     spatial_weights,
                     color_weights,
                     gid.x,
                     gid.y,
                     random_depth,
                     random_normal,
                     p,
                     best_depth,
                     best_normal,
                     best_cost);

  depth_out[index] = best_depth;
  normal_out[index * 3 + 0] = best_normal.x;
  normal_out[index * 3 + 1] = best_normal.y;
  normal_out[index * 3 + 2] = best_normal.z;
  cost_out[index] = best_cost;
}

kernel void patch_match_finalize(
    device const uchar* reference [[buffer(0)]],
    texture2d_array<float, access::sample> sources [[texture(0)]],
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device float* selection_probability [[buffer(7)]],
    device uchar* consistency_mask [[buffer(8)]],
    constant float* spatial_weights [[buffer(11)]],
    constant float* color_weights [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  const uint index = gid.y * p.width + gid.x;
  const float3 n = float3(normal[index * 3 + 0],
                          normal[index * 3 + 1],
                          normal[index * 3 + 2]);
  uint consistent = 0;
  const float sigma2 = max(1e-6f, 2.0f * p.ncc_sigma * p.ncc_sigma);
  for (uint source = 0; source < p.num_sources; ++source) {
    const float cost = photo_cost(reference,
                                  sources,
                                  poses,
                                  spatial_weights,
                                  color_weights,
                                  source,
                                  gid.x,
                                  gid.y,
                                  depth[index],
                                  n,
                                  p);
    const float probability = exp(-(cost * cost) / sigma2);
    selection_probability[index * p.num_sources + source] = probability;
    bool is_consistent =
        has_stable_view(poses, source, gid.x, gid.y, depth[index], n, p) &&
        (1.0f - cost) >= p.filter_min_ncc;
    if (is_consistent && p.geom_consistency != 0) {
      is_consistent =
          geom_cost(source_depths,
                    poses,
                    source,
                    gid.x,
                    gid.y,
                    depth[index],
                    p) <= p.filter_geom_consistency_max_cost;
    }
    consistency_mask[index * p.num_sources + source] = is_consistent ? 1 : 0;
    consistent += is_consistent ? 1 : 0;
  }
  if (p.filter != 0 && consistent < p.filter_min_num_consistent) {
    depth[index] = 0.0f;
    normal[index * 3 + 0] = 0.0f;
    normal[index * 3 + 1] = 0.0f;
    normal[index * 3 + 2] = 0.0f;
    for (uint source = 0; source < p.num_sources; ++source) {
      consistency_mask[index * p.num_sources + source] = 0;
    }
  }
}

kernel void patch_match_finalize_fidelity(
    device const float* source_depths [[buffer(2)]],
    device const float* poses [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    device float* depth [[buffer(5)]],
    device float* normal [[buffer(6)]],
    device const float* selection_probability [[buffer(7)]],
    device uchar* consistency_mask [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  const uint index = gid.y * p.width + gid.x;
  const float3 n = float3(normal[index * 3 + 0],
                          normal[index * 3 + 1],
                          normal[index * 3 + 2]);
  const float minimum_probability = ncc_probability(1.0f - p.filter_min_ncc, p);
  uint consistent = 0;
  for (uint source = 0; source < p.num_sources; ++source) {
    bool is_consistent =
        has_stable_view(poses, source, gid.x, gid.y, depth[index], n, p) &&
        selection_probability[index * p.num_sources + source] >= minimum_probability;
    if (is_consistent && p.geom_consistency != 0) {
      is_consistent =
          geom_cost(source_depths,
                    poses,
                    source,
                    gid.x,
                    gid.y,
                    depth[index],
                    p) <= p.filter_geom_consistency_max_cost;
    }
    consistency_mask[index * p.num_sources + source] = is_consistent ? 1 : 0;
    consistent += is_consistent ? 1 : 0;
  }
  if (p.filter != 0 && consistent < p.filter_min_num_consistent) {
    depth[index] = 0.0f;
    normal[index * 3 + 0] = 0.0f;
    normal[index * 3 + 1] = 0.0f;
    normal[index * 3 + 2] = 0.0f;
    for (uint source = 0; source < p.num_sources; ++source) {
      consistency_mask[index * p.num_sources + source] = 0;
    }
  }
}
)METAL";

template <typename T>
id<MTLBuffer> MakeBuffer(id<MTLDevice> device, const std::vector<T>& values, const char* label) {
  const size_t length = std::max<size_t>(sizeof(T), values.size() * sizeof(T));
  id<MTLBuffer> buffer = nil;
  if (values.empty()) {
    buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    std::memset([buffer contents], 0, length);
  } else {
    buffer = [device newBufferWithBytes:values.data()
                                 length:length
                                options:MTLResourceStorageModeShared];
  }
  THROW_CHECK(buffer != nil) << "Failed to allocate Metal buffer " << label;
  buffer.label = [NSString stringWithUTF8String:label];
  return buffer;
}

id<MTLComputePipelineState> MakePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         const char* name) {
  NSString* function_name = [NSString stringWithUTF8String:name];
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  THROW_CHECK(function != nil) << "Missing Metal kernel " << name;
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
  THROW_CHECK(pipeline != nil) << "Failed to create Metal pipeline " << name << ": "
                               << (error == nil ? "unknown error"
                                                : [[error localizedDescription] UTF8String]);
  return pipeline;
}

}  // namespace

class PatchMatchMetal::Impl {
 public:
  struct CommandTiming {
    double wall = 0.0;
    double gpu = 0.0;

    CommandTiming& operator+=(const CommandTiming& other) {
      wall += other.wall;
      gpu += other.gpu;
      return *this;
    }
  };

  friend CommandTiming operator+(CommandTiming lhs, const CommandTiming& rhs) {
    lhs += rhs;
    return lhs;
  }

  Impl(const PatchMatchOptions& options, const PatchMatch::Problem& problem)
      : options_(options), problem_(problem) {
    InitDevice();
    InitInputs();
    InitBuffers();
  }

  void Run() {
    Timer timer;
    timer.Start();

    const CommandTiming initialize_timing = DispatchInitialize();
    CommandTiming sweep_timing;
    const uint32_t num_steps = std::max(1, options_.num_iterations * 4);
    for (uint32_t iteration = 0; iteration < static_cast<uint32_t>(options_.num_iterations);
         ++iteration) {
      for (uint32_t direction = 0; direction < 4; ++direction) {
        params_.iteration = iteration * 4 + direction;
        params_.sweep_direction = direction;
        params_.perturbation =
            1.0f / std::pow(2.0f, static_cast<float>(iteration) + direction / 4.0f);
        params_.prev_sel_prob_weight =
            static_cast<float>(params_.iteration) / static_cast<float>(num_steps);
        sweep_timing += DispatchSweep();
        std::swap(selection_probability_buffers_[0], selection_probability_buffers_[1]);
      }
    }
    params_.iteration = num_steps;
    const CommandTiming finalize_timing = DispatchFinalize();
    DownloadOutputs();

    const CommandTiming total_timing = initialize_timing + sweep_timing + finalize_timing;
    LOG(INFO) << "Metal command wall timing: initialize=" << initialize_timing.wall * 1000.0
              << " ms, sweeps=" << sweep_timing.wall * 1000.0 << " ms ("
              << sweep_timing.wall * 1000.0 / num_steps << " ms average x " << num_steps
              << "), finalize=" << finalize_timing.wall * 1000.0
              << " ms, total=" << total_timing.wall * 1000.0 << " ms";
    LOG(INFO) << "Metal reported GPU timing: initialize=" << initialize_timing.gpu * 1000.0
              << " ms, sweeps=" << sweep_timing.gpu * 1000.0
              << " ms, finalize=" << finalize_timing.gpu * 1000.0
              << " ms, total=" << total_timing.gpu * 1000.0 << " ms";
    LOG(INFO) << "Metal PatchMatch completed on " << [[device_ name] UTF8String] << " in "
              << timer.ElapsedSeconds() << " seconds";
  }

  DepthMap GetDepthMap() const {
    Mat<float> mat(width_, height_, 1);
    std::memcpy(mat.GetPtr(), depth_.data(), depth_.size() * sizeof(float));
    return DepthMap(mat, options_.depth_min, options_.depth_max);
  }

  NormalMap GetNormalMap() const {
    Mat<float> mat(width_, height_, 3);
    const size_t pixels = width_ * height_;
    for (size_t index = 0; index < pixels; ++index) {
      const size_t row = index / width_;
      const size_t col = index % width_;
      for (size_t channel = 0; channel < 3; ++channel) {
        mat.Set(row, col, channel, normal_[index * 3 + channel]);
      }
    }
    return NormalMap(mat);
  }

  Mat<float> GetSelProbMap() const {
    Mat<float> mat(width_, height_, num_sources_);
    const size_t pixels = width_ * height_;
    for (size_t index = 0; index < pixels; ++index) {
      const size_t row = index / width_;
      const size_t col = index % width_;
      for (size_t source = 0; source < num_sources_; ++source) {
        mat.Set(row, col, source, selection_probability_[index * num_sources_ + source]);
      }
    }
    return mat;
  }

  std::vector<int> GetConsistentImageIdxs() const {
    std::vector<int> result;
    std::vector<int> pixel_sources;
    pixel_sources.reserve(num_sources_);
    for (size_t row = 0; row < height_; ++row) {
      for (size_t col = 0; col < width_; ++col) {
        pixel_sources.clear();
        const size_t index = row * width_ + col;
        for (size_t source = 0; source < num_sources_; ++source) {
          if (consistency_mask_[index * num_sources_ + source] != 0) {
            pixel_sources.push_back(problem_.src_image_idxs[source]);
          }
        }
        if (!pixel_sources.empty()) {
          result.push_back(static_cast<int>(col));
          result.push_back(static_cast<int>(row));
          result.push_back(static_cast<int>(pixel_sources.size()));
          result.insert(result.end(), pixel_sources.begin(), pixel_sources.end());
        }
      }
    }
    return result;
  }

 private:
  void InitDevice() {
    device_ = MTLCreateSystemDefaultDevice();
    THROW_CHECK(device_ != nil) << "Metal is enabled but no system Metal device is available";
    command_queue_ = [device_ newCommandQueue];
    THROW_CHECK(command_queue_ != nil) << "Failed to create Metal command queue";

    MTLCompileOptions* compile_options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
      compile_options.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      compile_options.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kPatchMatchMetalSource];
    library_ = [device_ newLibraryWithSource:source options:compile_options error:&error];
    THROW_CHECK(library_ != nil) << "Failed to compile Metal PatchMatch kernels: "
                                 << (error == nil ? "unknown error"
                                                  : [[error localizedDescription] UTF8String]);
    initialize_pipeline_ = MakePipeline(device_, library_, "patch_match_initialize_fidelity");
    sweep_pipeline_ = MakePipeline(device_, library_, "patch_match_column_sweep_simd");
    finalize_pipeline_ = MakePipeline(device_, library_, "patch_match_finalize_fidelity");
  }

  void InitInputs() {
    const Image& reference = problem_.images->at(problem_.ref_image_idx);
    width_ = reference.GetWidth();
    height_ = reference.GetHeight();
    num_sources_ = problem_.src_image_idxs.size();
    THROW_CHECK_GT(width_, 0);
    THROW_CHECK_GT(height_, 0);
    THROW_CHECK_GT(num_sources_, 0);

    reference_image_ = reference.GetBitmap().RowMajorData();
    source_width_ = 0;
    source_height_ = 0;
    for (const int image_idx : problem_.src_image_idxs) {
      const Image& image = problem_.images->at(image_idx);
      source_width_ = std::max(source_width_, image.GetWidth());
      source_height_ = std::max(source_height_, image.GetHeight());
    }

    const size_t layer_stride = source_width_ * source_height_;
    source_images_.assign(layer_stride * num_sources_, 0);
    source_depths_.assign(options_.geom_consistency ? layer_stride * num_sources_ : 1, 0.0f);
    poses_.assign(kPoseStride * num_sources_, 0.0f);

    for (size_t source = 0; source < num_sources_; ++source) {
      const int image_idx = problem_.src_image_idxs[source];
      const Image& image = problem_.images->at(image_idx);
      const auto& bitmap = image.GetBitmap().RowMajorData();
      for (size_t row = 0; row < image.GetHeight(); ++row) {
        std::memcpy(source_images_.data() + source * layer_stride + row * source_width_,
                    bitmap.data() + row * image.GetWidth(),
                    image.GetWidth());
      }
      if (options_.geom_consistency) {
        const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
        for (size_t row = 0; row < depth_map.GetHeight(); ++row) {
          std::memcpy(source_depths_.data() + source * layer_stride + row * source_width_,
                      depth_map.GetPtr() + row * depth_map.GetWidth(),
                      depth_map.GetWidth() * sizeof(float));
        }
      }

      float* pose = poses_.data() + source * kPoseStride;
      const float K[4] = {image.GetK()[0], image.GetK()[2], image.GetK()[4], image.GetK()[5]};
      std::memcpy(pose, K, sizeof(K));
      float relative_R[9];
      float relative_T[3];
      ComputeRelativePose(
          reference.GetR(), reference.GetT(), image.GetR(), image.GetT(), relative_R, relative_T);
      std::memcpy(pose + 4, relative_R, sizeof(relative_R));
      std::memcpy(pose + 13, relative_T, sizeof(relative_T));
      ComputeProjectionCenter(relative_R, relative_T, pose + 16);
      ComposeProjectionMatrix(image.GetK(), relative_R, relative_T, pose + 19);
      ComposeInverseProjectionMatrix(image.GetK(), relative_R, relative_T, pose + 31);
      pose[43] = static_cast<float>(image.GetWidth());
      pose[44] = static_cast<float>(image.GetHeight());
    }

    params_ = {};
    params_.width = static_cast<uint32_t>(width_);
    params_.height = static_cast<uint32_t>(height_);
    params_.source_width = static_cast<uint32_t>(source_width_);
    params_.source_height = static_cast<uint32_t>(source_height_);
    params_.num_sources = static_cast<uint32_t>(num_sources_);
    params_.window_radius = static_cast<uint32_t>(options_.window_radius);
    params_.window_step = static_cast<uint32_t>(options_.window_step);
    params_.num_samples = static_cast<uint32_t>(std::max(1, options_.num_samples));
    params_.geom_consistency = options_.geom_consistency ? 1 : 0;
    params_.filter = options_.filter ? 1 : 0;
    params_.filter_min_num_consistent = static_cast<uint32_t>(options_.filter_min_num_consistent);
    params_.initialize_from_input = options_.geom_consistency ? 1 : 0;
    params_.seed = 0x6d2b79f5u;
    params_.pose_stride = kPoseStride;
    params_.depth_min = options_.depth_min;
    params_.depth_max = options_.depth_max;
    params_.sigma_spatial = options_.sigma_spatial;
    params_.sigma_color = options_.sigma_color;
    params_.ncc_sigma = options_.ncc_sigma;
    params_.geom_consistency_regularizer = options_.geom_consistency_regularizer;
    params_.geom_consistency_max_cost = options_.geom_consistency_max_cost;
    params_.filter_min_ncc = options_.filter_min_ncc;
    params_.filter_geom_consistency_max_cost = options_.filter_geom_consistency_max_cost;
    params_.ref_fx = reference.GetK()[0];
    params_.ref_cx = reference.GetK()[2];
    params_.ref_fy = reference.GetK()[4];
    params_.ref_cy = reference.GetK()[5];
    params_.ref_inv_fx = 1.0f / params_.ref_fx;
    params_.ref_neg_cx_inv_fx = -params_.ref_cx / params_.ref_fx;
    params_.ref_inv_fy = 1.0f / params_.ref_fy;
    params_.ref_neg_cy_inv_fy = -params_.ref_cy / params_.ref_fy;
    params_.filter_cos_min_triangulation_angle =
        std::cos(options_.filter_min_triangulation_angle * 3.14159265358979323846 / 180.0);
    params_.cos_min_triangulation_angle =
        std::cos(options_.min_triangulation_angle * 3.14159265358979323846 / 180.0);
    const float incident_angle_sigma = static_cast<float>(options_.incident_angle_sigma);
    params_.inv_incident_angle_sigma_square =
        -0.5f / std::max(1e-8f, incident_angle_sigma * incident_angle_sigma);
    const float ncc_sigma = static_cast<float>(options_.ncc_sigma);
    params_.ncc_norm_factor = 2.0f / (std::sqrt(2.0f * 3.14159265358979323846f) * ncc_sigma *
                                      std::erf(2.0f / (ncc_sigma * 1.414213562f)));
  }

  void InitBuffers() {
    reference_buffer_ = MakeBuffer(device_, reference_image_, "COLMAP reference image");
    source_depth_buffer_ = MakeBuffer(device_, source_depths_, "COLMAP source depth maps");
    pose_buffer_ = MakeBuffer(device_, poses_, "COLMAP camera transforms");

    const int radius = options_.window_radius;
    const int spatial_stride = 2 * radius + 1;
    const float sigma_spatial = static_cast<float>(options_.sigma_spatial);
    const float inv_spatial = 0.5f / std::max(1e-6f, sigma_spatial * sigma_spatial);
    std::vector<float> spatial_weights(spatial_stride * spatial_stride);
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const size_t index = static_cast<size_t>(dy + radius) * spatial_stride + dx + radius;
        spatial_weights[index] = std::exp(-(dx * dx + dy * dy) * inv_spatial);
      }
    }
    spatial_weight_buffer_ =
        MakeBuffer(device_, spatial_weights, "COLMAP bilateral spatial weights");

    const float sigma_color = static_cast<float>(options_.sigma_color);
    const float inv_color = 0.5f / std::max(1e-6f, sigma_color * sigma_color);
    std::vector<float> color_weights(256);
    for (size_t delta = 0; delta < color_weights.size(); ++delta) {
      const float normalized_delta = static_cast<float>(delta) / 255.0f;
      color_weights[delta] = std::exp(-normalized_delta * normalized_delta * inv_color);
    }
    color_weight_buffer_ = MakeBuffer(device_, color_weights, "COLMAP bilateral color weights");

    MTLTextureDescriptor* source_descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:source_width_
                                                          height:source_height_
                                                       mipmapped:NO];
    source_descriptor.textureType = MTLTextureType2DArray;
    source_descriptor.arrayLength = num_sources_;
    source_descriptor.storageMode = MTLStorageModeShared;
    source_descriptor.usage = MTLTextureUsageShaderRead;
    source_texture_ = [device_ newTextureWithDescriptor:source_descriptor];
    THROW_CHECK(source_texture_ != nil) << "Failed to allocate Metal source-image texture";
    source_texture_.label = @"COLMAP source images";
    const size_t source_layer_stride = source_width_ * source_height_;
    const MTLRegion source_region = MTLRegionMake2D(0, 0, source_width_, source_height_);
    for (size_t source = 0; source < num_sources_; ++source) {
      [source_texture_ replaceRegion:source_region
                         mipmapLevel:0
                               slice:source
                           withBytes:source_images_.data() + source * source_layer_stride
                         bytesPerRow:source_width_
                       bytesPerImage:source_layer_stride];
    }

    const size_t pixels = width_ * height_;
    std::vector<float> initial_depth(pixels, 0.0f);
    std::vector<float> initial_normal(pixels * 3, 0.0f);
    if (options_.geom_consistency) {
      const DepthMap& depth_map = problem_.depth_maps->at(problem_.ref_image_idx);
      std::memcpy(initial_depth.data(), depth_map.GetPtr(), pixels * sizeof(float));
      const NormalMap& normal_map = problem_.normal_maps->at(problem_.ref_image_idx);
      for (size_t row = 0; row < height_; ++row) {
        for (size_t col = 0; col < width_; ++col) {
          const size_t index = row * width_ + col;
          for (size_t channel = 0; channel < 3; ++channel) {
            initial_normal[index * 3 + channel] = normal_map.Get(row, col, channel);
          }
        }
      }
    }
    std::vector<float> initial_cost(pixels, std::numeric_limits<float>::infinity());
    for (int i = 0; i < 2; ++i) {
      depth_buffers_[i] = MakeBuffer(device_, initial_depth, "COLMAP Metal depth");
      normal_buffers_[i] = MakeBuffer(device_, initial_normal, "COLMAP Metal normal");
      cost_buffers_[i] = MakeBuffer(device_, initial_cost, "COLMAP Metal cost");
    }
    std::vector<float> probability(pixels * num_sources_, 0.5f);
    std::vector<uint8_t> mask(pixels * num_sources_, 0);
    probability_buffer_ = MakeBuffer(device_, probability, "COLMAP selection probability");
    source_cost_buffer_ = MakeBuffer(
        device_,
        std::vector<float>(pixels * num_sources_, std::numeric_limits<float>::infinity()),
        "COLMAP per-source matching costs");
    selection_probability_buffers_[0] =
        MakeBuffer(device_, probability, "COLMAP previous selection probabilities");
    selection_probability_buffers_[1] =
        MakeBuffer(device_, probability, "COLMAP current selection probabilities");
    std::vector<uint32_t> random_states(pixels);
    for (size_t index = 0; index < pixels; ++index) {
      random_states[index] = static_cast<uint32_t>(index) ^ params_.seed;
    }
    random_state_buffer_ = MakeBuffer(device_, random_states, "COLMAP persistent random states");
    sweep_workspace_buffer_ =
        MakeBuffer(device_,
                   std::vector<float>(2 * std::max(width_, height_) * num_sources_, 0.5f),
                   "COLMAP source-selection sweep workspace");
    mask_buffer_ = MakeBuffer(device_, mask, "COLMAP consistency mask");
  }

  id<MTLCommandBuffer> BeginCommand(const char* label,
                                    id<MTLComputeCommandEncoder>* encoder,
                                    id<MTLComputePipelineState> pipeline) {
    id<MTLCommandBuffer> command = [command_queue_ commandBuffer];
    command.label = [NSString stringWithUTF8String:label];
    *encoder = [command computeCommandEncoder];
    (*encoder).label = command.label;
    [*encoder setComputePipelineState:pipeline];
    return command;
  }

  CommandTiming EndCommand(id<MTLCommandBuffer> command,
                           id<MTLComputeCommandEncoder> encoder,
                           const MTLSize grid,
                           const MTLSize threads) {
    Timer timer;
    timer.Start();
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    THROW_CHECK(command.status == MTLCommandBufferStatusCompleted)
        << "Metal PatchMatch command failed: "
        << (command.error == nil ? "unknown error"
                                 : [[command.error localizedDescription] UTF8String]);
    return CommandTiming{timer.ElapsedSeconds(),
                         std::max(0.0, command.GPUEndTime - command.GPUStartTime)};
  }

  CommandTiming DispatchInitialize() {
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLCommandBuffer> command =
        BeginCommand("COLMAP PatchMatch initialize", &encoder, initialize_pipeline_);
    [encoder setBuffer:reference_buffer_ offset:0 atIndex:0];
    [encoder setTexture:source_texture_ atIndex:0];
    [encoder setBuffer:source_depth_buffer_ offset:0 atIndex:2];
    [encoder setBuffer:pose_buffer_ offset:0 atIndex:3];
    [encoder setBytes:&params_ length:sizeof(params_) atIndex:4];
    [encoder setBuffer:depth_buffers_[0] offset:0 atIndex:5];
    [encoder setBuffer:normal_buffers_[0] offset:0 atIndex:6];
    [encoder setBuffer:source_cost_buffer_ offset:0 atIndex:7];
    [encoder setBuffer:random_state_buffer_ offset:0 atIndex:10];
    [encoder setBuffer:spatial_weight_buffer_ offset:0 atIndex:11];
    [encoder setBuffer:color_weight_buffer_ offset:0 atIndex:12];
    return EndCommand(command, encoder, MTLSizeMake(width_, height_, 1), MTLSizeMake(16, 16, 1));
  }

  CommandTiming DispatchSweep() {
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLCommandBuffer> command =
        BeginCommand("COLMAP PatchMatch sweep", &encoder, sweep_pipeline_);
    [encoder setBuffer:reference_buffer_ offset:0 atIndex:0];
    [encoder setTexture:source_texture_ atIndex:0];
    [encoder setBuffer:source_depth_buffer_ offset:0 atIndex:2];
    [encoder setBuffer:pose_buffer_ offset:0 atIndex:3];
    [encoder setBytes:&params_ length:sizeof(params_) atIndex:4];
    [encoder setBuffer:depth_buffers_[0] offset:0 atIndex:5];
    [encoder setBuffer:normal_buffers_[0] offset:0 atIndex:6];
    [encoder setBuffer:source_cost_buffer_ offset:0 atIndex:7];
    [encoder setBuffer:selection_probability_buffers_[0] offset:0 atIndex:8];
    [encoder setBuffer:selection_probability_buffers_[1] offset:0 atIndex:9];
    [encoder setBuffer:random_state_buffer_ offset:0 atIndex:10];
    [encoder setBuffer:spatial_weight_buffer_ offset:0 atIndex:11];
    [encoder setBuffer:color_weight_buffer_ offset:0 atIndex:12];
    [encoder setBuffer:sweep_workspace_buffer_ offset:0 atIndex:13];
    const size_t logical_width = (params_.sweep_direction & 1u) == 0 ? width_ : height_;
    const NSUInteger simd_width = sweep_pipeline_.threadExecutionWidth;
    THROW_CHECK_EQ(simd_width, 32) << "CUDA-fidelity Metal sweep requires 32-lane SIMD groups";
    const NSUInteger max_threads = sweep_pipeline_.maxTotalThreadsPerThreadgroup;
    const NSUInteger simdgroups_per_threadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(4, max_threads / simd_width));
    const NSUInteger candidate_scratch_size =
        simdgroups_per_threadgroup * 5 * params_.num_samples * sizeof(float);
    THROW_CHECK_LE(candidate_scratch_size + sweep_pipeline_.staticThreadgroupMemoryLength,
                   device_.maxThreadgroupMemoryLength)
        << "CUDA-fidelity sweep candidate scratch exceeds Metal threadgroup memory";
    [encoder setThreadgroupMemoryLength:candidate_scratch_size atIndex:0];
    return EndCommand(command,
                      encoder,
                      MTLSizeMake(logical_width * simd_width, 1, 1),
                      MTLSizeMake(simdgroups_per_threadgroup * simd_width, 1, 1));
  }

  CommandTiming DispatchFinalize() {
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLCommandBuffer> command =
        BeginCommand("COLMAP PatchMatch finalize", &encoder, finalize_pipeline_);
    [encoder setBuffer:source_depth_buffer_ offset:0 atIndex:2];
    [encoder setBuffer:pose_buffer_ offset:0 atIndex:3];
    [encoder setBytes:&params_ length:sizeof(params_) atIndex:4];
    [encoder setBuffer:depth_buffers_[0] offset:0 atIndex:5];
    [encoder setBuffer:normal_buffers_[0] offset:0 atIndex:6];
    [encoder setBuffer:selection_probability_buffers_[0] offset:0 atIndex:7];
    [encoder setBuffer:mask_buffer_ offset:0 atIndex:8];
    return EndCommand(command, encoder, MTLSizeMake(width_, height_, 1), MTLSizeMake(16, 16, 1));
  }

  void DownloadOutputs() {
    const size_t pixels = width_ * height_;
    depth_.resize(pixels);
    normal_.resize(pixels * 3);
    selection_probability_.resize(pixels * num_sources_);
    consistency_mask_.resize(pixels * num_sources_);
    std::memcpy(depth_.data(), [depth_buffers_[0] contents], depth_.size() * sizeof(float));
    std::memcpy(normal_.data(), [normal_buffers_[0] contents], normal_.size() * sizeof(float));
    std::memcpy(selection_probability_.data(),
                [selection_probability_buffers_[0] contents],
                selection_probability_.size() * sizeof(float));
    std::memcpy(consistency_mask_.data(),
                [mask_buffer_ contents],
                consistency_mask_.size() * sizeof(uint8_t));
  }

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  size_t width_ = 0;
  size_t height_ = 0;
  size_t source_width_ = 0;
  size_t source_height_ = 0;
  size_t num_sources_ = 0;
  MetalParams params_{};

  std::vector<uint8_t> reference_image_;
  std::vector<uint8_t> source_images_;
  std::vector<float> source_depths_;
  std::vector<float> poses_;

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> initialize_pipeline_ = nil;
  id<MTLComputePipelineState> sweep_pipeline_ = nil;
  id<MTLComputePipelineState> finalize_pipeline_ = nil;

  id<MTLBuffer> reference_buffer_ = nil;
  id<MTLTexture> source_texture_ = nil;
  id<MTLBuffer> source_depth_buffer_ = nil;
  id<MTLBuffer> pose_buffer_ = nil;
  id<MTLBuffer> spatial_weight_buffer_ = nil;
  id<MTLBuffer> color_weight_buffer_ = nil;
  id<MTLBuffer> depth_buffers_[2] = {nil, nil};
  id<MTLBuffer> normal_buffers_[2] = {nil, nil};
  id<MTLBuffer> cost_buffers_[2] = {nil, nil};
  id<MTLBuffer> probability_buffer_ = nil;
  id<MTLBuffer> source_cost_buffer_ = nil;
  id<MTLBuffer> selection_probability_buffers_[2] = {nil, nil};
  id<MTLBuffer> random_state_buffer_ = nil;
  id<MTLBuffer> sweep_workspace_buffer_ = nil;
  id<MTLBuffer> mask_buffer_ = nil;

  std::vector<float> depth_;
  std::vector<float> normal_;
  std::vector<float> selection_probability_;
  std::vector<uint8_t> consistency_mask_;
};

PatchMatchMetal::PatchMatchMetal(const PatchMatchOptions& options,
                                 const PatchMatch::Problem& problem)
    : impl_(std::make_unique<Impl>(options, problem)) {}

PatchMatchMetal::~PatchMatchMetal() = default;

void PatchMatchMetal::Run() { impl_->Run(); }

DepthMap PatchMatchMetal::GetDepthMap() const { return impl_->GetDepthMap(); }

NormalMap PatchMatchMetal::GetNormalMap() const { return impl_->GetNormalMap(); }

Mat<float> PatchMatchMetal::GetSelProbMap() const { return impl_->GetSelProbMap(); }

std::vector<int> PatchMatchMetal::GetConsistentImageIdxs() const {
  return impl_->GetConsistentImageIdxs();
}

}  // namespace mvs
}  // namespace colmap
