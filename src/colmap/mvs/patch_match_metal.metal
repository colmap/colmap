// SPDX-License-Identifier: BSD-3-Clause

// Metal Shading Language port of the device code in patch_match_cuda.cu,
// gpu_mat_ref_image.cu and gpu_mat.h. The structure mirrors the CUDA code
// one to one (see the corresponding functions there for the documentation of
// the algorithm); only the differences are commented here:
//
//  - This source is embedded into the library as a string and compiled at
//    runtime (see patch_match_metal.mm), so no Metal toolchain is needed to
//    build COLMAP.
//  - The CUDA template parameters (window radius and step, geometric
//    consistency term, photometric / geometric filtering) are function
//    constants, specialized when the pipeline is created.
//  - The shared memory window of the reference image is a threadgroup memory
//    argument, because its size depends on the window radius.
//  - Metal has no cuRAND. Every column of a sweep draws from a PCG32 stream
//    seeded with a hash of the column and a per-sweep seed. Results therefore
//    differ from CUDA like two different random seeds would.
//  - MSL forbids recursion, so PerturbNormal retries in a loop.
//  - MSL has no erf, so the NCC likelihood normalization is computed on the
//    host and passed in `Params::ncc_norm`.
//  - The reference image and the maps are plain buffers (the rotation between
//    sweeps is done by dedicated kernels, ping-ponging between two buffers).

#include <metal_stdlib>
using namespace metal;

// Threads per threadgroup. Equal to the SIMD-group width of Apple GPUs and to
// THREADS_PER_BLOCK in patch_match_cuda.cu.
#define THREADS_PER_BLOCK 32

constant int kWindowRadius [[function_constant(0)]];
constant int kWindowStep [[function_constant(1)]];
constant bool kGeomConsistencyTerm [[function_constant(2)]];
constant bool kFilterPhotoConsistency [[function_constant(3)]];
constant bool kFilterGeomConsistency [[function_constant(4)]];

constant int kWindowSize = 2 * kWindowRadius + 1;
// The local reference image caches kWindowSize rows of three blocks of
// THREADS_PER_BLOCK columns each.
constant int kNumLocalColumns = 3 * THREADS_PER_BLOCK;

// Number of floats per source image in the poses buffer:
// K(4), R(9), T(3), C(3), P(12), P^-1(12).
constant int kNumPoseParams = 43;

// Keep in sync with PatchMatchMetal::Impl::Params in patch_match_metal.mm.
struct Params {
  int width;
  int height;
  int num_src_images;
  int num_samples;
  float ref_K[4];
  float ref_inv_K[4];
  float perturbation;
  float spatial_norm;  // 1 / (2 sigma_spatial^2)
  float color_norm;    // 1 / (2 sigma_color^2)
  float cos_min_triangulation_angle;
  float inv_incident_angle_sigma_sq;  // -1 / (2 sigma^2)
  float inv_ncc_sigma_sq;             // -1 / (2 sigma^2)
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
  uint seed;
  int src_width;  // Size of the layers of the source image array.
  int src_height;
};

////////////////////////////////////////////////////////////////////////////////
// Random numbers
////////////////////////////////////////////////////////////////////////////////

inline uint PcgHash(const uint v) {
  const uint state = v * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Uniform in (0, 1], like curand_uniform.
inline float RandUniform(thread uint& state) {
  state = state * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  word = (word >> 22u) ^ word;
  return float((word >> 8) + 1u) * (1.0f / 16777216.0f);
}

////////////////////////////////////////////////////////////////////////////////
// Geometry helpers
////////////////////////////////////////////////////////////////////////////////

inline float Dot3(thread const float* a, thread const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void Mat33DotVec3Homogeneous(thread const float* mat,
                                    thread const float* vec,
                                    thread float* result) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

inline void GenerateRandomNormal(const int row,
                                 const int col,
                                 thread uint& rand_state,
                                 constant Params& p,
                                 thread float* normal) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * RandUniform(rand_state) - 1.0f;
    v2 = 2.0f * RandUniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  // Make sure normal is looking away from camera.
  const float view_ray[3] = {p.ref_inv_K[0] * col + p.ref_inv_K[1],
                             p.ref_inv_K[2] * row + p.ref_inv_K[3],
                             1.0f};
  if (Dot3(normal, view_ray) > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

inline float PerturbDepth(const float perturbation,
                          const float depth,
                          thread uint& rand_state) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return RandUniform(rand_state) * (depth_max - depth_min) + depth_min;
}

// The CUDA version calls itself with half the perturbation (at most three
// times) if the perturbed normal faces away from the camera.
inline void PerturbNormal(const int row,
                          const int col,
                          float perturbation,
                          thread const float* normal,
                          thread uint& rand_state,
                          constant Params& p,
                          thread float* perturbed_normal) {
  const float view_ray[3] = {p.ref_inv_K[0] * col + p.ref_inv_K[1],
                             p.ref_inv_K[2] * row + p.ref_inv_K[3],
                             1.0f};

  for (int num_trials = 0;; ++num_trials) {
    // Perturbation rotation angles.
    const float a1 = (RandUniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (RandUniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (RandUniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    // R = Rx * Ry * Rz
    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = -cos_a2 * sin_a3;
    R[2] = sin_a2;
    R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
    R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
    R[5] = -cos_a2 * sin_a1;
    R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
    R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
    R[8] = cos_a1 * cos_a2;

    // Perturb the normal vector.
    perturbed_normal[0] =
        R[0] * normal[0] + R[1] * normal[1] + R[2] * normal[2];
    perturbed_normal[1] =
        R[3] * normal[0] + R[4] * normal[1] + R[5] * normal[2];
    perturbed_normal[2] =
        R[6] * normal[0] + R[7] * normal[1] + R[8] * normal[2];

    // Make sure the perturbed normal is still looking in the same direction as
    // the viewing direction, otherwise try again with a smaller perturbation.
    if (Dot3(perturbed_normal, view_ray) >= 0.0f) {
      if (num_trials < 3) {
        perturbation *= 0.5f;
        continue;
      }
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }

    // Normalize the perturbed normal.
    const float inv_norm = rsqrt(Dot3(perturbed_normal, perturbed_normal));
    perturbed_normal[0] *= inv_norm;
    perturbed_normal[1] *= inv_norm;
    perturbed_normal[2] *= inv_norm;
    return;
  }
}

inline void ComputePointAtDepth(const float row,
                                const float col,
                                const float depth,
                                constant Params& p,
                                thread float* point) {
  point[0] = depth * (p.ref_inv_K[0] * col + p.ref_inv_K[1]);
  point[1] = depth * (p.ref_inv_K[2] * row + p.ref_inv_K[3]);
  point[2] = depth;
}

// Transfer depth on plane from viewing ray at row1 to row2. The returned
// depth is the intersection of the viewing ray through row2 with the plane at
// row1 defined by the given depth and normal.
inline float PropagateDepth(const float depth1,
                            thread const float* normal1,
                            const float row1,
                            const float row2,
                            constant Params& p) {
  // Point along first viewing ray.
  const float x1 = depth1 * (p.ref_inv_K[2] * row1 + p.ref_inv_K[3]);
  const float y1 = depth1;
  // Point on plane defined by point along first viewing ray and plane normal1.
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];

  // Origin of second viewing ray.
  // const float x3 = 0.0f;
  // const float y3 = 0.0f;
  // Point on second viewing ray.
  const float x4 = p.ref_inv_K[2] * row2 + p.ref_inv_K[3];
  // const float y4 = 1.0f;

  // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
  const float denom = x2 - x1 + x4 * (y1 - y2);
  const float kEps = 1e-5f;
  if (fabs(denom) < kEps) {
    return depth1;
  }
  const float nom = y1 * x2 - x1 * y2;
  return nom / denom;
}

inline void ComputeViewingAngles(device const float* poses,
                                 thread const float* point,
                                 thread const float* normal,
                                 const int image_idx,
                                 thread float* cos_triangulation_angle,
                                 thread float* cos_incident_angle) {
  device const float* C = poses + image_idx * kNumPoseParams + 16;

  // Ray from point to camera.
  const float SX[3] = {C[0] - point[0], C[1] - point[1], C[2] - point[2]};

  // Length of ray from reference image to point.
  const float RX_inv_norm = rsqrt(Dot3(point, point));

  // Length of ray from source image to point.
  const float SX_inv_norm = rsqrt(Dot3(SX, SX));

  *cos_incident_angle = Dot3(SX, normal) * SX_inv_norm;
  *cos_triangulation_angle = -Dot3(SX, point) * RX_inv_norm * SX_inv_norm;
}

inline void ComposeHomography(device const float* poses,
                              const int image_idx,
                              const int row,
                              const int col,
                              const float depth,
                              thread const float* normal,
                              constant Params& p,
                              thread float* H) {
  device const float* K = poses + image_idx * kNumPoseParams;
  device const float* R = K + 4;
  device const float* T = K + 13;

  // Distance to the plane.
  const float dist =
      depth * (normal[0] * (p.ref_inv_K[0] * col + p.ref_inv_K[1]) +
               normal[1] * (p.ref_inv_K[2] * row + p.ref_inv_K[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;

  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  // Homography as H = K * (R - T * n' / d) * Kref^-1.
  H[0] = p.ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                           K[1] * (R[6] + inv_dist_N0 * T[2]));
  H[1] = p.ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                           K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
         K[1] * (R[8] + inv_dist_N2 * T[2]) +
         p.ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                           K[1] * (R[6] + inv_dist_N0 * T[2])) +
         p.ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                           K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[3] = p.ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                           K[3] * (R[6] + inv_dist_N0 * T[2]));
  H[4] = p.ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                           K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
         K[3] * (R[8] + inv_dist_N2 * T[2]) +
         p.ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                           K[3] * (R[6] + inv_dist_N0 * T[2])) +
         p.ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                           K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[6] = p.ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
  H[7] = p.ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
  H[8] = R[8] + p.ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
         p.ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
}

////////////////////////////////////////////////////////////////////////////////
// Local reference image window
////////////////////////////////////////////////////////////////////////////////

inline float ReadRefImage(device const uchar* ref_image,
                          const int width,
                          const int height,
                          const int col,
                          const int row) {
  // Equivalent to the border address mode of the CUDA texture.
  if (col < 0 || row < 0 || col >= width || row >= height) {
    return 0.0f;
  }
  return float(ref_image[row * width + col]) * (1.0f / 255.0f);
}

// Cache of kWindowSize rows of the reference image for a block of
// THREADS_PER_BLOCK columns, extended by one block to the left and to the
// right. Each thread holds the columns tid, tid + THREADS_PER_BLOCK, and
// tid + 2 * THREADS_PER_BLOCK of all rows. Moving to the next row shifts the
// cached rows up and reads one new row.
inline void ReadLocalRefImage(threadgroup float* data,
                              device const uchar* ref_image,
                              const int width,
                              const int height,
                              const int row,
                              const uint tid,
                              const uint block_first_col) {
  const int local_col_start = int(tid);
  const int global_col_start =
      int(block_first_col) - THREADS_PER_BLOCK + int(tid);

  // CUDA relies on the threads of a warp executing in lockstep when the
  // window is shifted. Make the dependency explicit.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (row == 0) {
    int global_row = row - kWindowRadius;
    for (int local_row = 0; local_row < kWindowSize;
         ++local_row, ++global_row) {
      for (int block = 0; block < 3; ++block) {
        data[local_row * kNumLocalColumns + local_col_start +
             block * THREADS_PER_BLOCK] =
            ReadRefImage(ref_image,
                         width,
                         height,
                         global_col_start + block * THREADS_PER_BLOCK,
                         global_row);
      }
    }
  } else {
    // Move rows up by one.
    for (int local_row = 1; local_row < kWindowSize; ++local_row) {
      for (int block = 0; block < 3; ++block) {
        data[(local_row - 1) * kNumLocalColumns + local_col_start +
             block * THREADS_PER_BLOCK] =
            data[local_row * kNumLocalColumns + local_col_start +
                 block * THREADS_PER_BLOCK];
      }
    }

    // Read the next row into the last row.
    const int local_row = kWindowSize - 1;
    const int global_row = row + kWindowRadius;
    for (int block = 0; block < 3; ++block) {
      data[local_row * kNumLocalColumns + local_col_start +
           block * THREADS_PER_BLOCK] =
          ReadRefImage(ref_image,
                       width,
                       height,
                       global_col_start + block * THREADS_PER_BLOCK,
                       global_row);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

////////////////////////////////////////////////////////////////////////////////
// Photometric and geometric consistency
////////////////////////////////////////////////////////////////////////////////

// Linear filtering and zero border, sampled at pixel centers like the CUDA
// texture (which uses unnormalized coordinates).
constexpr sampler kSourceImageSampler(coord::normalized,
                                      address::clamp_to_zero,
                                      filter::linear);

// Bilaterally weighted NCC between the reference patch and the patch in the
// source image: 1 - NCC, clamped to [0, 2].
inline float ComputeNCC(threadgroup const float* local_ref_image,
                        const uint tid,
                        const float local_ref_sum,
                        const float local_ref_squared_sum,
                        texture2d_array<float, access::sample> src_images,
                        const int src_image_idx,
                        device const float* poses,
                        const int row,
                        const int col,
                        const float depth,
                        thread const float* normal,
                        constant Params& p) {
  const float kMaxCost = 2.0f;

  float tform[9];
  ComposeHomography(poses, src_image_idx, row, col, depth, normal, p, tform);

  float tform_step[8];
  for (int i = 0; i < 8; ++i) {
    tform_step[i] = kWindowStep * tform[i];
  }

  const int row_start = row - kWindowRadius;
  const int col_start = col - kWindowRadius;
  float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
  float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
  float z = tform[6] * col_start + tform[7] * row_start + tform[8];
  float base_col_src = col_src;
  float base_row_src = row_src;
  float base_z = z;

  int ref_image_idx = THREADS_PER_BLOCK - kWindowRadius + int(tid);
  int ref_image_base_idx = ref_image_idx;

  const float ref_center_color =
      local_ref_image[ref_image_idx + kWindowRadius * kNumLocalColumns +
                      kWindowRadius];

  float src_sum = 0.0f;
  float src_squared_sum = 0.0f;
  float src_ref_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  const float inv_src_width = 1.0f / float(p.src_width);
  const float inv_src_height = 1.0f / float(p.src_height);

  for (int row_offset = -kWindowRadius; row_offset <= kWindowRadius;
       row_offset += kWindowStep) {
    for (int col_offset = -kWindowRadius; col_offset <= kWindowRadius;
         col_offset += kWindowStep) {
      const float inv_z = 1.0f / z;
      const float norm_col_src = (inv_z * col_src + 0.5f) * inv_src_width;
      const float norm_row_src = (inv_z * row_src + 0.5f) * inv_src_height;
      const float ref_color = local_ref_image[ref_image_idx];
      const float src_color = src_images
                                  .sample(kSourceImageSampler,
                                          float2(norm_col_src, norm_row_src),
                                          src_image_idx)
                                  .r;

      const float color_diff = ref_center_color - ref_color;
      const float bilateral_weight =
          exp(-float(row_offset * row_offset + col_offset * col_offset) *
                  p.spatial_norm -
              color_diff * color_diff * p.color_norm);

      const float bilateral_weight_src = bilateral_weight * src_color;

      src_sum += bilateral_weight_src;
      src_squared_sum += bilateral_weight_src * src_color;
      src_ref_sum += bilateral_weight_src * ref_color;
      bilateral_weight_sum += bilateral_weight;

      ref_image_idx += kWindowStep;

      // Accumulate warped source coordinates per row to reduce numerical
      // errors.
      col_src += tform_step[0];
      row_src += tform_step[3];
      z += tform_step[6];
    }

    ref_image_base_idx += kWindowStep * kNumLocalColumns;
    ref_image_idx = ref_image_base_idx;

    base_col_src += tform_step[1];
    base_row_src += tform_step[4];
    base_z += tform_step[7];

    col_src = base_col_src;
    row_src = base_row_src;
    z = base_z;
  }

  const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
  src_sum *= inv_bilateral_weight_sum;
  src_squared_sum *= inv_bilateral_weight_sum;
  src_ref_sum *= inv_bilateral_weight_sum;

  const float ref_color_var =
      local_ref_squared_sum - local_ref_sum * local_ref_sum;
  const float src_color_var = src_squared_sum - src_sum * src_sum;

  // Based on Jensen's Inequality for convex functions, the variance should
  // always be larger than 0. Hence, the check below is only needed to avoid
  // numerical issues.
  const float kMinVar = 1e-5f;
  if (ref_color_var < kMinVar || src_color_var < kMinVar) {
    return kMaxCost;
  }

  const float src_ref_color_covar = src_ref_sum - local_ref_sum * src_sum;
  const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
  return max(0.0f,
             min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
}

// Reprojection error of the point of the reference image after warping it to
// the source image, reading the source depth, and warping it back.
inline float ComputeGeomConsistencyCost(device const float* poses,
                                        device const float* src_depth_maps,
                                        constant Params& p,
                                        const float row,
                                        const float col,
                                        const float depth,
                                        const int image_idx,
                                        const float max_cost) {
  device const float* P = poses + image_idx * kNumPoseParams + 19;
  device const float* inv_P = poses + image_idx * kNumPoseParams + 31;

  float forward_point[3];
  ComputePointAtDepth(row, col, depth, p, forward_point);

  // Project point in reference image to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image (nearest neighbor, zero border).
  const int src_col_idx = int(floor(src_col + 0.5f));
  const int src_row_idx = int(floor(src_row + 0.5f));
  float src_depth = 0.0f;
  if (src_col_idx >= 0 && src_row_idx >= 0 && src_col_idx < p.src_width &&
      src_row_idx < p.src_height) {
    src_depth = src_depth_maps[(size_t(image_idx) * p.src_height +
                                size_t(src_row_idx)) *
                                   p.src_width +
                               size_t(src_col_idx)];
  }

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to reference image.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;
  const float backward_col =
      inv_backward_point_z *
      (p.ref_K[0] * backward_point_x + p.ref_K[1] * backward_point_z);
  const float backward_row =
      inv_backward_point_z *
      (p.ref_K[2] * backward_point_y + p.ref_K[3] * backward_point_z);

  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

////////////////////////////////////////////////////////////////////////////////
// Likelihoods and priors
////////////////////////////////////////////////////////////////////////////////

inline float ComputeNCCProb(const float cost, constant Params& p) {
  return exp(cost * cost * p.inv_ncc_sigma_sq) * p.ncc_norm;
}

inline float ComputeForwardMessage(const float cost,
                                   const float prev,
                                   constant Params& p) {
  const float kUniformProb = 0.5f;
  const float kNoChangeProb = 0.99999f;
  const float kChangeProb = 1.0f - kNoChangeProb;
  const float emission = ComputeNCCProb(cost, p);

  float zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb);
  float zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb);
  zn0 *= kUniformProb;
  zn1 *= emission;

  // Normalize.
  return zn1 / (zn0 + zn1);
}

inline float ComputeBackwardMessage(const float cost,
                                    const float prev,
                                    constant Params& p) {
  const float kUniformProb = 0.5f;
  const float kNoChangeProb = 0.99999f;
  const float kChangeProb = 1.0f - kNoChangeProb;
  const float emission = ComputeNCCProb(cost, p);

  float zn0 = prev * emission * kChangeProb +
              (1.0f - prev) * kUniformProb * kNoChangeProb;
  float zn1 = prev * emission * kNoChangeProb +
              (1.0f - prev) * kUniformProb * kChangeProb;

  // Normalize.
  return zn1 / (zn0 + zn1);
}

inline float ComputeSelProb(const float alpha,
                            const float beta,
                            const float prev,
                            const float prev_weight) {
  const float zn0 = (1.0f - alpha) * (1.0f - beta);
  const float zn1 = alpha * beta;
  const float curr = zn1 / (zn0 + zn1);
  return prev_weight * prev + (1.0f - prev_weight) * curr;
}

inline float ComputeTriProb(const float cos_triangulation_angle,
                            constant Params& p) {
  const float kMinTriProb = 0.0f;
  if (cos_triangulation_angle > p.cos_min_triangulation_angle) {
    const float scaled = 1.0f - (1.0f - cos_triangulation_angle) /
                                    (1.0f - p.cos_min_triangulation_angle);
    return min(1.0f, max(kMinTriProb, 1.0f - scaled * scaled));
  }
  return 1.0f;
}

inline float ComputeIncProb(const float cos_incident_angle,
                            constant Params& p) {
  const float x = 1.0f - max(0.0f, cos_incident_angle);
  return exp(x * x * p.inv_incident_angle_sigma_sq);
}

inline float ComputeResolutionProb(thread const float* H,
                                   const float row,
                                   const float col) {
  const int kWindowRadiusInt = kWindowRadius;
  float S1[2];
  float S2[2];
  float S3[2];
  float S4[2];

  // Project the four corners of the window into the source image.
  const float R1[2] = {col - kWindowRadiusInt, row - kWindowRadiusInt};
  const float R2[2] = {col - kWindowRadiusInt, row + kWindowRadiusInt};
  const float R3[2] = {col + kWindowRadiusInt, row + kWindowRadiusInt};
  const float R4[2] = {col + kWindowRadiusInt, row - kWindowRadiusInt};
  Mat33DotVec3Homogeneous(H, R1, S1);
  Mat33DotVec3Homogeneous(H, R2, S2);
  Mat33DotVec3Homogeneous(H, R3, S3);
  Mat33DotVec3Homogeneous(H, R4, S4);

  // Compute area of the projected window and the ratio to the reference
  // window area.
  const float ref_area = float(kWindowSize * kWindowSize);
  const float src_area = fabs(
      0.5f * (S1[0] * S2[1] - S2[0] * S1[1] - S1[0] * S4[1] + S2[0] * S3[1] -
              S3[0] * S2[1] + S4[0] * S1[1] + S3[0] * S4[1] - S4[0] * S3[1]));
  if (ref_area > src_area) {
    return src_area / ref_area;
  } else {
    return ref_area / src_area;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Element-wise kernels
////////////////////////////////////////////////////////////////////////////////

// Bilateral pre-filter of the reference image (FilterKernel in
// gpu_mat_ref_image.cu): weighted mean and mean of squares in the window.
kernel void FilterRefImage(device const uchar* image [[buffer(0)]],
                           device float* sum_image [[buffer(1)]],
                           device float* squared_sum_image [[buffer(2)]],
                           constant Params& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const int col = int(gid.x);
  const int row = int(gid.y);
  if (col >= p.width || row >= p.height) {
    return;
  }

  const float center_color = ReadRefImage(image, p.width, p.height, col, row);

  float color_sum = 0.0f;
  float color_squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int row_offset = -kWindowRadius; row_offset <= kWindowRadius;
       row_offset += kWindowStep) {
    for (int col_offset = -kWindowRadius; col_offset <= kWindowRadius;
         col_offset += kWindowStep) {
      const float color = ReadRefImage(
          image, p.width, p.height, col + col_offset, row + row_offset);
      const float color_diff = center_color - color;
      const float bilateral_weight =
          exp(-float(row_offset * row_offset + col_offset * col_offset) *
                  p.spatial_norm -
              color_diff * color_diff * p.color_norm);
      color_sum += bilateral_weight * color;
      color_squared_sum += bilateral_weight * color * color;
      bilateral_weight_sum += bilateral_weight;
    }
  }

  color_sum /= bilateral_weight_sum;
  color_squared_sum /= bilateral_weight_sum;

  sum_image[row * p.width + col] = color_sum;
  squared_sum_image[row * p.width + col] = color_squared_sum;
}

// Random initialization of the depth and normal maps (FillWithRandomNumbers
// and InitNormalMap in the CUDA code).
kernel void InitDepthAndNormalMap(device float* depth_map [[buffer(0)]],
                                  device float* normal_map [[buffer(1)]],
                                  constant Params& p [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const int col = int(gid.x);
  const int row = int(gid.y);
  if (col >= p.width || row >= p.height) {
    return;
  }

  uint rand_state = PcgHash(uint(row * p.width + col) ^ PcgHash(p.seed));
  depth_map[row * p.width + col] =
      RandUniform(rand_state) * (p.depth_max - p.depth_min) + p.depth_min;

  float normal[3];
  GenerateRandomNormal(row, col, rand_state, p, normal);
  const size_t plane = size_t(p.width) * p.height;
  for (int i = 0; i < 3; ++i) {
    normal_map[i * plane + row * p.width + col] = normal[i];
  }
}

// Rotate a width x height x depth map by 90 degrees in counter-clockwise
// direction, as GpuMat::Rotate does. The output has height x width pixels.
kernel void RotateFloat(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant int4& dims [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
  const int width = dims.x;
  const int height = dims.y;
  if (int(gid.x) >= width || int(gid.y) >= height || int(gid.z) >= dims.z) {
    return;
  }
  const size_t plane = size_t(width) * height;
  output[gid.z * plane + size_t(width - 1 - gid.x) * height + gid.y] =
      input[gid.z * plane + size_t(gid.y) * width + gid.x];
}

kernel void RotateUint8(device const uchar* input [[buffer(0)]],
                        device uchar* output [[buffer(1)]],
                        constant int4& dims [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
  const int width = dims.x;
  const int height = dims.y;
  if (int(gid.x) >= width || int(gid.y) >= height || int(gid.z) >= dims.z) {
    return;
  }
  const size_t plane = size_t(width) * height;
  output[gid.z * plane + size_t(width - 1 - gid.x) * height + gid.y] =
      input[gid.z * plane + size_t(gid.y) * width + gid.x];
}

// RotateNormalMap followed by the rotation of the map, fused: the normal
// vector is rotated by 90 degrees around the z-axis and written to the
// rotated pixel.
kernel void RotateNormalMap(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant int4& dims [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int width = dims.x;
  const int height = dims.y;
  if (int(gid.x) >= width || int(gid.y) >= height) {
    return;
  }
  const size_t plane = size_t(width) * height;
  const size_t in_idx = size_t(gid.y) * width + gid.x;
  const size_t out_idx = size_t(width - 1 - gid.x) * height + gid.y;
  const float n0 = input[in_idx];
  const float n1 = input[plane + in_idx];
  const float n2 = input[2 * plane + in_idx];
  output[out_idx] = n1;
  output[plane + out_idx] = -n0;
  output[2 * plane + out_idx] = n2;
}

////////////////////////////////////////////////////////////////////////////////
// Column sweep kernels (one thread per column, rows are processed in order)
////////////////////////////////////////////////////////////////////////////////

kernel void ComputeInitialCost(device float* cost_map [[buffer(0)]],
                               device const float* depth_map [[buffer(1)]],
                               device const float* normal_map [[buffer(2)]],
                               device const uchar* ref_image [[buffer(3)]],
                               device const float* ref_sum_image [[buffer(4)]],
                               device const float* ref_squared_sum_image
                               [[buffer(5)]],
                               texture2d_array<float, access::sample> src_images
                               [[texture(0)]],
                               device const float* poses [[buffer(6)]],
                               constant Params& p [[buffer(7)]],
                               threadgroup float* local_ref_image
                               [[threadgroup(0)]],
                               uint col_u [[thread_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint block [[threadgroup_position_in_grid]]) {
  const int col = int(col_u);
  const size_t plane = size_t(p.width) * p.height;

  for (int row = 0; row < p.height; ++row) {
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    ReadLocalRefImage(local_ref_image,
                      ref_image,
                      p.width,
                      p.height,
                      row,
                      tid,
                      block * THREADS_PER_BLOCK);

    if (col >= p.width) {
      continue;
    }

    const size_t pix = size_t(row) * p.width + col;
    const float depth = depth_map[pix];
    const float normal[3] = {
        normal_map[pix], normal_map[plane + pix], normal_map[2 * plane + pix]};
    for (int image_idx = 0; image_idx < p.num_src_images; ++image_idx) {
      cost_map[image_idx * plane + pix] = ComputeNCC(local_ref_image,
                                                     tid,
                                                     ref_sum_image[pix],
                                                     ref_squared_sum_image[pix],
                                                     src_images,
                                                     image_idx,
                                                     poses,
                                                     row,
                                                     col,
                                                     depth,
                                                     normal,
                                                     p);
    }
  }
}

kernel void SweepFromTopToBottom(
    device float* global_workspace [[buffer(0)]],
    device float* cost_map [[buffer(1)]],
    device float* depth_map [[buffer(2)]],
    device float* normal_map [[buffer(3)]],
    device uchar* consistency_mask [[buffer(4)]],
    device float* sel_prob_map [[buffer(5)]],
    device const float* prev_sel_prob_map [[buffer(6)]],
    device const uchar* ref_image [[buffer(7)]],
    device const float* ref_sum_image [[buffer(8)]],
    device const float* ref_squared_sum_image [[buffer(9)]],
    texture2d_array<float, access::sample> src_images [[texture(0)]],
    device const float* src_depth_maps [[buffer(10)]],
    device const float* poses [[buffer(11)]],
    constant Params& p [[buffer(12)]],
    threadgroup float* local_ref_image [[threadgroup(0)]],
    uint col_u [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const int col = int(col_u);
  const int width = p.width;
  const int height = p.height;
  const int num_src_images = p.num_src_images;
  const size_t plane = size_t(width) * height;

  // Probability for boundary pixels.
  const float kUniformProb = 0.5f;

  const int max_dim = max(width, height);
  device float* forward_message =
      global_workspace + size_t(col) * num_src_images;
  device float* sampling_probs = global_workspace +
                                 size_t(max_dim) * num_src_images +
                                 size_t(col) * num_src_images;

  //////////////////////////////////////////////////////////////////////////////
  // Compute backward message for all rows. Note that the backward messages are
  // temporarily stored in the sel_prob_map and replaced row by row as the
  // updated forward messages are computed further below.
  //////////////////////////////////////////////////////////////////////////////

  if (col < width) {
    for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
      // Compute backward message.
      float beta = kUniformProb;
      for (int row = height - 1; row >= 0; --row) {
        const size_t idx = image_idx * plane + size_t(row) * width + col;
        beta = ComputeBackwardMessage(cost_map[idx], beta, p);
        sel_prob_map[idx] = beta;
      }

      // Initialize forward message.
      forward_message[image_idx] = kUniformProb;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Estimate parameters for remaining rows and compute selection probabilities.
  //////////////////////////////////////////////////////////////////////////////

  // Parameters of previous pixel in column.
  float prev_depth = 0.0f;
  float prev_normal[3] = {0.0f, 0.0f, 0.0f};
  // Parameters of current pixel in column.
  float curr_depth = 0.0f;
  float curr_normal[3];
  // Randomly sampled parameters.
  float rand_depth = 0.0f;
  float rand_normal[3];

  uint rand_state = PcgHash(uint(col) * 9781u + PcgHash(p.seed));

  if (col < width) {
    // Parameters for first row in column.
    prev_depth = depth_map[col];
    prev_normal[0] = normal_map[col];
    prev_normal[1] = normal_map[plane + col];
    prev_normal[2] = normal_map[2 * plane + col];
  }

  for (int row = 0; row < height; ++row) {
    // Note that this must be executed even for pixels outside the borders,
    // since pixels are used in the local neighborhood of the current pixel.
    ReadLocalRefImage(local_ref_image,
                      ref_image,
                      width,
                      height,
                      row,
                      tid,
                      block * THREADS_PER_BLOCK);

    if (col >= width) {
      continue;
    }

    const size_t pix = size_t(row) * width + col;
    const float local_ref_sum = ref_sum_image[pix];
    const float local_ref_squared_sum = ref_squared_sum_image[pix];

    // Propagate the depth at which the current ray intersects with the plane
    // of the normal of the previous ray. This helps to better estimate
    // the depth of very oblique structures, i.e. pixels whose normal direction
    // is significantly different from their viewing direction.
    prev_depth = PropagateDepth(prev_depth, prev_normal, row - 1, row, p);

    // Read parameters for current pixel from previous sweep.
    curr_depth = depth_map[pix];
    curr_normal[0] = normal_map[pix];
    curr_normal[1] = normal_map[plane + pix];
    curr_normal[2] = normal_map[2 * plane + pix];

    // Generate random parameters.
    rand_depth = PerturbDepth(p.perturbation, curr_depth, rand_state);
    PerturbNormal(row,
                  col,
                  p.perturbation * M_PI_F,
                  curr_normal,
                  rand_state,
                  p,
                  rand_normal);

    // Read in the backward message, compute selection probabilities and
    // modulate selection probabilities with priors.

    float point[3];
    ComputePointAtDepth(row, col, curr_depth, p, point);

    for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
      const size_t idx = image_idx * plane + pix;
      const float cost = cost_map[idx];
      const float alpha =
          ComputeForwardMessage(cost, forward_message[image_idx], p);
      const float beta = sel_prob_map[idx];
      const float prev_prob = prev_sel_prob_map[idx];
      const float sel_prob =
          ComputeSelProb(alpha, beta, prev_prob, p.prev_sel_prob_weight);

      float cos_triangulation_angle;
      float cos_incident_angle;
      ComputeViewingAngles(poses,
                           point,
                           curr_normal,
                           image_idx,
                           &cos_triangulation_angle,
                           &cos_incident_angle);
      const float tri_prob = ComputeTriProb(cos_triangulation_angle, p);
      const float inc_prob = ComputeIncProb(cos_incident_angle, p);

      float H[9];
      ComposeHomography(
          poses, image_idx, row, col, curr_depth, curr_normal, p, H);
      const float res_prob = ComputeResolutionProb(H, row, col);

      sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
    }

    // Transform the PDF of the sampling probabilities to a CDF.
    {
      float sum = 0.0f;
      for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
        sum += sampling_probs[image_idx];
      }
      const float inv_sum = 1.0f / sum;
      float cum_value = 0.0f;
      for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
        cum_value += sampling_probs[image_idx] * inv_sum;
        sampling_probs[image_idx] = cum_value;
      }
    }

    // Compute matching cost using Monte Carlo sampling of source images. Images
    // with higher selection probability are more likely to be sampled.

    const int kNumCosts = 5;
    float costs[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float depths[5] = {
        curr_depth, prev_depth, rand_depth, curr_depth, rand_depth};
    // Hypothesis -> normal: current, previous, random, random, current.
    const float normals[3][3] = {
        {curr_normal[0], curr_normal[1], curr_normal[2]},
        {prev_normal[0], prev_normal[1], prev_normal[2]},
        {rand_normal[0], rand_normal[1], rand_normal[2]}};
    const int normal_idxs[5] = {0, 1, 2, 2, 0};

    for (int sample = 0; sample < p.num_samples; ++sample) {
      const float rand_prob = RandUniform(rand_state) - FLT_EPSILON;

      int src_image_idx = -1;
      for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
        if (sampling_probs[image_idx] > rand_prob) {
          src_image_idx = image_idx;
          break;
        }
      }

      if (src_image_idx == -1) {
        continue;
      }

      costs[0] += cost_map[src_image_idx * plane + pix];
      if (kGeomConsistencyTerm) {
        costs[0] += p.geom_consistency_regularizer *
                    ComputeGeomConsistencyCost(poses,
                                               src_depth_maps,
                                               p,
                                               row,
                                               col,
                                               depths[0],
                                               src_image_idx,
                                               p.geom_consistency_max_cost);
      }

      for (int i = 1; i < kNumCosts; ++i) {
        float normal[3] = {normals[normal_idxs[i]][0],
                           normals[normal_idxs[i]][1],
                           normals[normal_idxs[i]][2]};
        costs[i] += ComputeNCC(local_ref_image,
                               tid,
                               local_ref_sum,
                               local_ref_squared_sum,
                               src_images,
                               src_image_idx,
                               poses,
                               row,
                               col,
                               depths[i],
                               normal,
                               p);
        if (kGeomConsistencyTerm) {
          costs[i] += p.geom_consistency_regularizer *
                      ComputeGeomConsistencyCost(poses,
                                                 src_depth_maps,
                                                 p,
                                                 row,
                                                 col,
                                                 depths[i],
                                                 src_image_idx,
                                                 p.geom_consistency_max_cost);
        }
      }
    }

    // Find the parameters of the minimum cost.
    int min_cost_idx = 0;
    {
      float min_cost = costs[0];
      for (int i = 1; i < kNumCosts; ++i) {
        if (costs[i] <= min_cost) {
          min_cost = costs[i];
          min_cost_idx = i;
        }
      }
    }
    const float best_depth = depths[min_cost_idx];
    const float best_normal[3] = {normals[normal_idxs[min_cost_idx]][0],
                                  normals[normal_idxs[min_cost_idx]][1],
                                  normals[normal_idxs[min_cost_idx]][2]};

    // Save best new parameters.
    depth_map[pix] = best_depth;
    normal_map[pix] = best_normal[0];
    normal_map[plane + pix] = best_normal[1];
    normal_map[2 * plane + pix] = best_normal[2];

    // Use the new cost to recompute the updated forward message and
    // the selection probability.
    for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
      const size_t idx = image_idx * plane + pix;

      // Determine the cost for best depth.
      float cost;
      if (min_cost_idx == 0) {
        cost = cost_map[idx];
      } else {
        cost = ComputeNCC(local_ref_image,
                          tid,
                          local_ref_sum,
                          local_ref_squared_sum,
                          src_images,
                          image_idx,
                          poses,
                          row,
                          col,
                          best_depth,
                          best_normal,
                          p);
        cost_map[idx] = cost;
      }

      const float alpha =
          ComputeForwardMessage(cost, forward_message[image_idx], p);
      const float beta = sel_prob_map[idx];
      const float prev_prob = prev_sel_prob_map[idx];
      const float prob =
          ComputeSelProb(alpha, beta, prev_prob, p.prev_sel_prob_weight);
      forward_message[image_idx] = alpha;
      sel_prob_map[idx] = prob;
    }

    if (kFilterPhotoConsistency || kFilterGeomConsistency) {
      int num_consistent = 0;

      float best_point[3];
      ComputePointAtDepth(row, col, best_depth, p, best_point);

      const float min_ncc_prob = ComputeNCCProb(1.0f - p.filter_min_ncc, p);

      for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
        float cos_triangulation_angle;
        float cos_incident_angle;
        ComputeViewingAngles(poses,
                             best_point,
                             best_normal,
                             image_idx,
                             &cos_triangulation_angle,
                             &cos_incident_angle);
        if (cos_triangulation_angle > p.cos_filter_min_triangulation_angle ||
            cos_incident_angle <= 0.0f) {
          continue;
        }

        bool consistent = true;
        if (kFilterPhotoConsistency) {
          consistent = consistent &&
                       sel_prob_map[image_idx * plane + pix] >= min_ncc_prob;
        }
        if (kFilterGeomConsistency) {
          consistent = consistent && ComputeGeomConsistencyCost(
                                         poses,
                                         src_depth_maps,
                                         p,
                                         row,
                                         col,
                                         best_depth,
                                         image_idx,
                                         p.geom_consistency_max_cost) <=
                                         p.filter_geom_consistency_max_cost;
        }
        if (consistent) {
          consistency_mask[image_idx * plane + pix] = 1;
          num_consistent += 1;
        }
      }

      if (num_consistent < p.filter_min_num_consistent) {
        depth_map[pix] = 0.0f;
        normal_map[pix] = 0.0f;
        normal_map[plane + pix] = 0.0f;
        normal_map[2 * plane + pix] = 0.0f;
        for (int image_idx = 0; image_idx < num_src_images; ++image_idx) {
          consistency_mask[image_idx * plane + pix] = 0;
        }
      }
    }

    // Update previous depth for next row.
    prev_depth = best_depth;
    prev_normal[0] = best_normal[0];
    prev_normal[1] = best_normal[1];
    prev_normal[2] = best_normal[2];
  }
}
