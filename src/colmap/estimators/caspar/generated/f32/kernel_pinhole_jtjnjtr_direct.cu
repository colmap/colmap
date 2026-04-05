#include "kernel_pinhole_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_jtjnjtr_direct_kernel(float* pose_njtr,
                                  unsigned int pose_njtr_num_alloc,
                                  SharedIndex* pose_njtr_indices,
                                  float* pose_jac,
                                  unsigned int pose_jac_num_alloc,
                                  float* calib_njtr,
                                  unsigned int calib_njtr_num_alloc,
                                  SharedIndex* calib_njtr_indices,
                                  float* calib_jac,
                                  unsigned int calib_jac_num_alloc,
                                  float* point_njtr,
                                  unsigned int point_njtr_num_alloc,
                                  SharedIndex* point_njtr_indices,
                                  float* point_jac,
                                  unsigned int point_jac_num_alloc,
                                  float* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  float* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  float* const out_point_njtr,
                                  unsigned int out_point_njtr_num_alloc,
                                  size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5,
                         r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r7, r8);
    read_idx_4<1024, float, float, float4>(point_jac,
                                           0 * point_jac_num_alloc,
                                           global_thread_idx,
                                           r9,
                                           r10,
                                           r11,
                                           r12);
    r13 = fmaf(r5, r11, r6 * r7);
    r13 = fmaf(r4, r9, r13);
  };
  load_shared<4, float, float>(calib_njtr,
                               0 * calib_njtr_num_alloc,
                               calib_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         calib_njtr_indices_loc[threadIdx.x].target,
                         r14,
                         r15,
                         r16,
                         r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r18, r19);
    r14 = fmaf(r14, r18, r16);
    r16 = r13 + r14;
    r5 = fmaf(r5, r12, r6 * r8);
    r5 = fmaf(r4, r10, r5);
    r15 = fmaf(r15, r19, r17);
    r17 = r5 + r15;
    r4 = fmaf(r1, r17, r0 * r16);
    r6 = fmaf(r3, r17, r2 * r16);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r20,
                                           r21,
                                           r22,
                                           r23);
    r24 = fmaf(r21, r17, r20 * r16);
    r25 = r22 * r16;
    write_sum_4<float, float>((float*)inout_shared, r4, r6, r24, r25);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r23 * r17;
    read_idx_2<1024, float, float, float2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r24, r6);
    r16 = fmaf(r24, r16, r6 * r17);
    write_sum_2<float, float>((float*)inout_shared, r25, r16);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<2, float, float>(pose_njtr,
                               4 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r16,
                         r25);
  };
  __syncthreads();
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r17,
                         r4,
                         r26,
                         r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = fmaf(r26, r20, r25 * r24);
    r20 = fmaf(r27, r22, r20);
    r20 = fmaf(r17, r0, r20);
    r20 = fmaf(r4, r2, r20);
    r13 = r20 + r13;
    r18 = r18 * r13;
    r21 = fmaf(r26, r21, r25 * r6);
    r21 = fmaf(r17, r1, r21);
    r21 = fmaf(r4, r3, r21);
    r21 = fmaf(r16, r23, r21);
    r5 = r21 + r5;
    r19 = r19 * r5;
    write_sum_4<float, float>((float*)inout_shared, r18, r19, r13, r5);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r20 + r14;
    r15 = r21 + r15;
    r10 = fmaf(r10, r15, r9 * r14);
    r12 = fmaf(r12, r15, r11 * r14);
    r15 = fmaf(r8, r15, r7 * r14);
    write_sum_3<float, float>((float*)inout_shared, r10, r12, r15);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void pinhole_jtjnjtr_direct(float* pose_njtr,
                            unsigned int pose_njtr_num_alloc,
                            SharedIndex* pose_njtr_indices,
                            float* pose_jac,
                            unsigned int pose_jac_num_alloc,
                            float* calib_njtr,
                            unsigned int calib_njtr_num_alloc,
                            SharedIndex* calib_njtr_indices,
                            float* calib_jac,
                            unsigned int calib_jac_num_alloc,
                            float* point_njtr,
                            unsigned int point_njtr_num_alloc,
                            SharedIndex* point_njtr_indices,
                            float* point_jac,
                            unsigned int point_jac_num_alloc,
                            float* const out_pose_njtr,
                            unsigned int out_pose_njtr_num_alloc,
                            float* const out_calib_njtr,
                            unsigned int out_calib_njtr_num_alloc,
                            float* const out_point_njtr,
                            unsigned int out_point_njtr_num_alloc,
                            size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(pose_njtr,
                                                    pose_njtr_num_alloc,
                                                    pose_njtr_indices,
                                                    pose_jac,
                                                    pose_jac_num_alloc,
                                                    calib_njtr,
                                                    calib_njtr_num_alloc,
                                                    calib_njtr_indices,
                                                    calib_jac,
                                                    calib_jac_num_alloc,
                                                    point_njtr,
                                                    point_njtr_num_alloc,
                                                    point_njtr_indices,
                                                    point_jac,
                                                    point_jac_num_alloc,
                                                    out_pose_njtr,
                                                    out_pose_njtr_num_alloc,
                                                    out_calib_njtr,
                                                    out_calib_njtr_num_alloc,
                                                    out_point_njtr,
                                                    out_point_njtr_num_alloc,
                                                    problem_size);
}

}  // namespace caspar