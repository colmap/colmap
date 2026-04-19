#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_res_jac_kernel(
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35;
  load_shared<2, float, float>(principal_point,
                               0 * principal_point_num_alloc,
                               principal_point_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         principal_point_indices_loc[threadIdx.x].target,
                         r0,
                         r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r5, r6);
  };
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r7,
                         r8,
                         r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r11, r12, r13, r14);
    r15 = r13 * r14;
    r16 = 2.00000000000000000e+00;
    r17 = r11 * r16;
    r18 = r12 * r17;
    r19 = fmaf(r10, r15, r18);
    r0 = fmaf(r8, r19, r0);
    r20 = r12 * r14;
    r21 = r13 * r17;
    r20 = fmaf(r16, r20, r21);
    r22 = r13 * r13;
    r22 = r22 * r10;
    r23 = 1.00000000000000000e+00;
    r24 = r12 * r12;
    r24 = fmaf(r10, r24, r23);
    r25 = r22 + r24;
    r0 = fmaf(r9, r20, r0);
    r0 = fmaf(r7, r25, r0);
  };
  load_shared<2, float, float>(focal_and_extra,
                               0 * focal_and_extra_num_alloc,
                               focal_and_extra_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         focal_and_extra_indices_loc[threadIdx.x].target,
                         r26,
                         r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r16, r15, r18);
    r5 = fmaf(r7, r15, r5);
    r13 = r12 * r13;
    r13 = r13 * r16;
    r18 = r11 * r14;
    r18 = fmaf(r10, r18, r13);
    r22 = r23 + r22;
    r28 = r11 * r11;
    r28 = r28 * r10;
    r22 = r22 + r28;
    r5 = fmaf(r9, r18, r5);
    r5 = fmaf(r8, r22, r5);
    r29 = r5 * r5;
    r30 = 9.99999999999999955e-07;
    r17 = fmaf(r14, r17, r13);
    r8 = fmaf(r8, r17, r6);
    r6 = r12 * r14;
    r6 = fmaf(r10, r6, r21);
    r24 = r28 + r24;
    r8 = fmaf(r7, r6, r8);
    r8 = fmaf(r9, r24, r8);
    r9 = copysign(1.0, r8);
    r9 = fmaf(r30, r9, r8);
    r30 = r9 * r9;
    r8 = 1.0 / r30;
    r29 = r29 * r8;
    r7 = r0 * r0;
    r28 = r8 * r7;
    r21 = r29 + r28;
    r13 = fmaf(r27, r21, r23);
    r31 = r0 * r13;
    r32 = 1.0 / r9;
    r33 = r26 * r32;
    r2 = fmaf(r33, r31, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r5 * r13;
    r3 = fmaf(r33, r1, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r13;
    r1 = r1 * r32;
    r31 = r5 * r13;
    r31 = r31 * r32;
    r34 = r0 * r21;
    r34 = r34 * r33;
    r35 = r5 * r21;
    r35 = r35 * r33;
    write_idx_4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r1,
        r31,
        r34,
        r35);
    r35 = r5 * r3;
    r34 = r13 * r4;
    r35 = r35 * r32;
    r31 = r0 * r2;
    r31 = r31 * r32;
    r31 = fmaf(r34, r31, r34 * r35);
    r35 = r0 * r21;
    r35 = r35 * r4;
    r35 = r35 * r2;
    r32 = r5 * r21;
    r32 = r32 * r4;
    r32 = r32 * r3;
    r32 = fmaf(r33, r32, r33 * r35);
    write_sum_2<float, float>((float*)inout_shared, r31, r32);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_njtr,
                             0 * out_focal_and_extra_njtr_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r13 * r29;
    r31 = r13 * r13;
    r31 = fmaf(r28, r31, r13 * r32);
    r35 = r26 * r21;
    r1 = r26 * r21;
    r35 = r35 * r1;
    r35 = fmaf(r29, r35, r28 * r35);
    write_sum_2<float, float>((float*)inout_shared, r31, r35);
  };
  flush_sum_shared<2, float>(out_focal_and_extra_precond_diag,
                             0 * out_focal_and_extra_precond_diag_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r13 * r28;
    r35 = fmaf(r1, r35, r1 * r32);
    write_sum_1<float, float>((float*)inout_shared, r35);
  };
  flush_sum_shared<1, float>(out_focal_and_extra_precond_tril,
                             0 * out_focal_and_extra_precond_tril_num_alloc,
                             focal_and_extra_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r4 * r2;
    r32 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r35, r32);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r23, r23);
  };
  flush_sum_shared<2, float>(out_principal_point_precond_diag,
                             0 * out_principal_point_precond_diag_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r0 * r8;
    r34 = r26 * r34;
    r23 = r23 * r34;
    r26 = r6 * r7;
    r30 = r9 * r30;
    r30 = 1.0 / r30;
    r30 = r10 * r30;
    r10 = r16 * r25;
    r10 = r10 * r0;
    r10 = fmaf(r8, r10, r30 * r26);
    r26 = r5 * r5;
    r26 = r26 * r6;
    r10 = fmaf(r30, r26, r10);
    r9 = r16 * r15;
    r32 = r5 * r8;
    r10 = fmaf(r32, r9, r10);
    r9 = r0 * r10;
    r27 = r27 * r33;
    r9 = fmaf(r27, r9, r6 * r23);
    r26 = r25 * r13;
    r9 = fmaf(r33, r26, r9);
    r26 = r6 * r32;
    r35 = r5 * r10;
    r35 = fmaf(r27, r35, r34 * r26);
    r26 = r15 * r13;
    r35 = fmaf(r33, r26, r35);
    r26 = r19 * r13;
    r26 = fmaf(r33, r26, r17 * r23);
    r1 = r16 * r19;
    r1 = r1 * r0;
    r31 = r17 * r30;
    r1 = fmaf(r7, r31, r8 * r1);
    r29 = r5 * r5;
    r1 = fmaf(r31, r29, r1);
    r31 = r16 * r22;
    r1 = fmaf(r32, r31, r1);
    r1 = r1 * r27;
    r26 = fmaf(r0, r1, r26);
    r31 = r17 * r32;
    r31 = fmaf(r34, r31, r5 * r1);
    r1 = r22 * r13;
    r31 = fmaf(r33, r1, r31);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r35,
                                            r26,
                                            r31);
    r1 = r16 * r20;
    r1 = r1 * r0;
    r29 = r24 * r7;
    r29 = fmaf(r30, r29, r8 * r1);
    r1 = r16 * r18;
    r29 = fmaf(r32, r1, r29);
    r8 = r5 * r5;
    r8 = r8 * r24;
    r29 = fmaf(r30, r8, r29);
    r8 = r0 * r29;
    r8 = fmaf(r27, r8, r24 * r23);
    r23 = r20 * r13;
    r8 = fmaf(r33, r23, r8);
    r23 = r24 * r32;
    r1 = r5 * r29;
    r1 = fmaf(r27, r1, r34 * r23);
    r23 = r18 * r13;
    r1 = fmaf(r33, r23, r1);
    write_idx_2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r8, r1);
    r23 = r4 * r3;
    r33 = r4 * r2;
    r33 = fmaf(r9, r33, r35 * r23);
    r23 = r4 * r3;
    r27 = r4 * r2;
    r27 = fmaf(r26, r27, r31 * r23);
    r23 = r4 * r3;
    r34 = r4 * r2;
    r34 = fmaf(r8, r34, r1 * r23);
    write_sum_3<float, float>((float*)inout_shared, r33, r27, r34);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r9, r9, r35 * r35);
    r27 = fmaf(r31, r31, r26 * r26);
    r33 = fmaf(r8, r8, r1 * r1);
    write_sum_3<float, float>((float*)inout_shared, r34, r27, r33);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r35, r31, r9 * r26);
    r35 = fmaf(r35, r1, r9 * r8);
    r1 = fmaf(r31, r1, r26 * r8);
    write_sum_3<float, float>((float*)inout_shared, r33, r35, r1);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_res_jac(
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_res_jac_kernel<<<n_blocks, 1024>>>(
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar