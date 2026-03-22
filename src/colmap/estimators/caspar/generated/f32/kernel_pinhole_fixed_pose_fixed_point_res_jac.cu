#include "kernel_pinhole_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_point_res_jac_kernel(
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
  load_shared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2,
                         r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r2, r7, r8);
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13, r14, r15);
    r16 = -2.00000000000000000e+00;
    r17 = r15 * r16;
    r18 = 2.00000000000000000e+00;
    r19 = r12 * r18;
    r20 = r13 * r19;
    r21 = fmaf(r14, r17, r20);
    r21 = fmaf(r10, r21, r2);
    r2 = r13 * r15;
    r22 = r14 * r19;
    r2 = fmaf(r18, r2, r22);
    r23 = r14 * r14;
    r23 = r16 * r23;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fmaf(r16, r25, r24);
    r26 = r23 + r25;
    r21 = fmaf(r11, r2, r21);
    r21 = fmaf(r9, r26, r21);
    r26 = 9.99999999999999955e-07;
    r2 = r13 * r14;
    r2 = r2 * r18;
    r19 = fmaf(r15, r19, r2);
    r19 = fmaf(r10, r19, r8);
    r22 = fmaf(r13, r17, r22);
    r8 = r12 * r12;
    r8 = r8 * r16;
    r25 = r8 + r25;
    r19 = fmaf(r9, r22, r19);
    r19 = fmaf(r11, r25, r19);
    r25 = copysign(1.0, r19);
    r25 = fmaf(r26, r25, r19);
    r26 = 1.0 / r25;
    r19 = r21 * r26;
    r4 = fmaf(r0, r19, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r14 * r15;
    r3 = fmaf(r18, r3, r20);
    r3 = fmaf(r9, r3, r7);
    r17 = fmaf(r12, r17, r2);
    r23 = r24 + r23;
    r23 = r23 + r8;
    r3 = fmaf(r11, r17, r3);
    r3 = fmaf(r10, r23, r3);
    r23 = r1 * r3;
    r5 = fmaf(r26, r23, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r4 = r6 * r4;
    r23 = r6 * r5;
    r10 = r19 * r4;
    r6 = r6 * r3;
    r6 = r6 * r5;
    r6 = r6 * r26;
    write_sum_4<float, float>((float*)inout_shared, r10, r6, r4, r23);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r21 * r21;
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r21 = r21 * r25;
    r23 = r3 * r3;
    r23 = r25 * r23;
    write_sum_4<float, float>((float*)inout_shared, r21, r23, r24, r24);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = 0.00000000000000000e+00;
    write_sum_4<float, float>((float*)inout_shared, r24, r19, r24, r24);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r3 * r26;
    write_sum_2<float, float>((float*)inout_shared, r26, r24);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_pose_fixed_point_res_jac(
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar