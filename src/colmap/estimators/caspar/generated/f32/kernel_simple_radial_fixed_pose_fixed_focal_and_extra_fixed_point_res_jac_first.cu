#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first_kernel(
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
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
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r10, r11, r12, r13);
    r14 = -2.00000000000000000e+00;
    r15 = r13 * r14;
    r16 = 2.00000000000000000e+00;
    r17 = r10 * r16;
    r18 = r11 * r17;
    r19 = fmaf(r12, r15, r18);
    r19 = fmaf(r8, r19, r0);
    r0 = r11 * r13;
    r20 = r12 * r17;
    r0 = fmaf(r16, r0, r20);
    r21 = r12 * r12;
    r21 = r14 * r21;
    r22 = 1.00000000000000000e+00;
    r23 = r11 * r11;
    r23 = fmaf(r14, r23, r22);
    r24 = r21 + r23;
    r19 = fmaf(r9, r0, r19);
    r19 = fmaf(r7, r24, r19);
    read_idx_2<1024, float, float, float2>(focal_and_extra,
                                           0 * focal_and_extra_num_alloc,
                                           global_thread_idx,
                                           r24,
                                           r0);
    r25 = 9.99999999999999955e-07;
    r26 = r11 * r12;
    r26 = r26 * r16;
    r17 = fmaf(r13, r17, r26);
    r17 = fmaf(r8, r17, r6);
    r20 = fmaf(r11, r15, r20);
    r6 = r10 * r10;
    r6 = r6 * r14;
    r23 = r6 + r23;
    r17 = fmaf(r7, r20, r17);
    r17 = fmaf(r9, r23, r17);
    r23 = copysign(1.0, r17);
    r23 = fmaf(r25, r23, r17);
    r25 = r23 * r23;
    r25 = 1.0 / r25;
    r17 = r12 * r13;
    r17 = fmaf(r16, r17, r18);
    r17 = fmaf(r7, r17, r5);
    r15 = fmaf(r10, r15, r26);
    r21 = r22 + r21;
    r21 = r21 + r6;
    r17 = fmaf(r9, r15, r17);
    r17 = fmaf(r8, r21, r17);
    r21 = r17 * r17;
    r8 = r19 * r19;
    r8 = fmaf(r25, r8, r25 * r21);
    r8 = fmaf(r0, r8, r22);
    r8 = r24 * r8;
    r23 = 1.0 / r23;
    r8 = r8 * r23;
    r2 = fmaf(r19, r8, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r17, r8, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r8 = fmaf(r3, r3, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r8);
  if (global_thread_idx < problem_size) {
    r2 = r4 * r2;
    r3 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r2, r3);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r22, r22);
  };
  flush_sum_shared<2, float>(out_principal_point_precond_diag,
                             0 * out_principal_point_precond_diag_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first(
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(principal_point,
              principal_point_num_alloc,
              principal_point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_principal_point_njtr,
              out_principal_point_njtr_num_alloc,
              out_principal_point_precond_diag,
              out_principal_point_precond_diag_num_alloc,
              out_principal_point_precond_tril,
              out_principal_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar