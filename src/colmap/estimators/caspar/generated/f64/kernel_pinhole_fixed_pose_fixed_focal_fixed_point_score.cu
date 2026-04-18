#include "kernel_pinhole_fixed_pose_fixed_focal_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_focal_fixed_point_score_kernel(
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;
  load_shared<2, double, double>(extra_calib,
                                 0 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r0,
                          r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
    read_idx_2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r1, r5);
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r6, r7);
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r10, r11);
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r14, r15);
    r16 = r14 * r13;
    r17 = fma(r15, r16, r12);
    r17 = fma(r8, r17, r7);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r7);
    r18 = r11 * r16;
    r19 = -2.00000000000000000e+00;
    r20 = r15 * r19;
    r21 = fma(r10, r20, r18);
    r22 = r14 * r14;
    r22 = r22 * r19;
    r23 = 1.00000000000000000e+00;
    r24 = r10 * r10;
    r24 = fma(r19, r24, r23);
    r25 = r22 + r24;
    r17 = fma(r7, r21, r17);
    r17 = fma(r9, r25, r17);
    r25 = r5 * r17;
    r21 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r26);
    r27 = r10 * r15;
    r27 = fma(r13, r27, r18);
    r27 = fma(r9, r27, r26);
    r16 = r10 * r16;
    r26 = fma(r11, r20, r16);
    r18 = r11 * r11;
    r18 = r19 * r18;
    r24 = r18 + r24;
    r27 = fma(r8, r26, r27);
    r27 = fma(r7, r24, r27);
    r24 = copysign(1.0, r27);
    r24 = fma(r21, r24, r27);
    r24 = 1.0 / r24;
    r3 = fma(r24, r25, r3);
    r4 = fma(r2, r4, r0);
    r20 = fma(r14, r20, r12);
    r20 = fma(r9, r20, r6);
    r9 = r11 * r15;
    r9 = fma(r13, r9, r16);
    r22 = r23 + r22;
    r22 = r22 + r18;
    r20 = fma(r7, r9, r20);
    r20 = fma(r8, r22, r20);
    r22 = r1 * r20;
    r4 = fma(r24, r22, r4);
    r4 = fma(r4, r4, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r4);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_pose_fixed_focal_fixed_point_score(
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_focal_fixed_point_score_kernel<<<n_blocks, 1024>>>(
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar