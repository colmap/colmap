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
    PinholeFixedPoseFixedPointResJacKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r7);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r12, r13);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = fma(r14, r17, r11 * r12);
    r19 = r15 * r16;
    r18 = fma(r4, r19, r18);
    r18 = fma(r10, r13, r18);
    r19 = 2.00000000000000000e+00;
    r20 = fma(r14, r13, r11 * r16);
    r21 = r10 * r17;
    r20 = fma(r4, r21, r20);
    r20 = fma(r15, r12, r20);
    r21 = r19 * r20;
    r22 = r18 * r21;
    r23 = r14 * r12;
    r23 = fma(r4, r23, r11 * r17);
    r23 = fma(r15, r13, r23);
    r23 = fma(r10, r16, r23);
    r24 = -2.00000000000000000e+00;
    r25 = fma(r15, r17, r14 * r16);
    r25 = fma(r10, r12, r25);
    r25 = fma(r4, r25, r11 * r13);
    r13 = r24 * r25;
    r26 = fma(r23, r13, r22);
    r26 = fma(r8, r26, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r27);
    r28 = r14 * r10;
    r28 = r28 * r19;
    r29 = r15 * r11;
    r30 = fma(r24, r29, r28);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r31);
    r32 = 1.00000000000000000e+00;
    r33 = r15 * r15;
    r33 = r33 * r24;
    r34 = r32 + r33;
    r35 = r14 * r14;
    r35 = r24 * r35;
    r34 = r34 + r35;
    r36 = r15 * r10;
    r36 = r36 * r19;
    r37 = r14 * r11;
    r37 = fma(r19, r37, r36);
    r38 = r19 * r18;
    r38 = r38 * r23;
    r39 = fma(r25, r21, r38);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r40);
    r41 = r20 * r20;
    r41 = r41 * r24;
    r42 = r23 * r23;
    r42 = fma(r24, r42, r32);
    r43 = r41 + r42;
    r26 = fma(r7, r30, r26);
    r26 = fma(r31, r34, r26);
    r26 = fma(r27, r37, r26);
    r26 = fma(r9, r39, r26);
    r26 = fma(r40, r43, r26);
    r43 = copysign(1.0, r26);
    r43 = fma(r6, r43, r26);
    r6 = 1.0 / r43;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r26,
                                            r39);
    r37 = r18 * r18;
    r37 = r24 * r37;
    r42 = r37 + r42;
    r42 = fma(r8, r42, r26);
    r21 = r23 * r21;
    r26 = fma(r18, r13, r21);
    r34 = r19 * r23;
    r34 = fma(r25, r34, r22);
    r29 = fma(r19, r29, r28);
    r28 = r10 * r11;
    r22 = r14 * r15;
    r22 = r22 * r19;
    r28 = fma(r24, r28, r22);
    r33 = r32 + r33;
    r30 = r10 * r10;
    r30 = r24 * r30;
    r33 = r33 + r30;
    r42 = fma(r9, r26, r42);
    r42 = fma(r40, r34, r42);
    r42 = fma(r31, r29, r42);
    r42 = fma(r27, r28, r42);
    r42 = fma(r7, r33, r42);
    r33 = r6 * r42;
    r2 = fma(r0, r33, r2);
    r3 = fma(r3, r4, r1);
    r1 = r19 * r18;
    r1 = fma(r25, r1, r21);
    r1 = fma(r8, r1, r39);
    r8 = r10 * r11;
    r8 = fma(r19, r8, r22);
    r30 = r32 + r30;
    r30 = r30 + r35;
    r35 = r14 * r11;
    r35 = fma(r24, r35, r36);
    r13 = fma(r20, r13, r38);
    r37 = r32 + r37;
    r37 = r37 + r41;
    r1 = fma(r7, r8, r1);
    r1 = fma(r27, r30, r1);
    r1 = fma(r31, r35, r1);
    r1 = fma(r40, r13, r1);
    r1 = fma(r9, r37, r1);
    r37 = r5 * r1;
    r3 = fma(r6, r37, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r2 = r4 * r2;
    r37 = r33 * r2;
    r9 = r4 * r1;
    r9 = r9 * r3;
    r9 = r9 * r6;
    WriteSum2<double, double>((double*)inout_shared, r37, r9);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r42 * r42;
    r43 = r43 * r43;
    r43 = 1.0 / r43;
    r42 = r42 * r43;
    r3 = r1 * r1;
    r3 = r43 * r3;
    WriteSum2<double, double>((double*)inout_shared, r42, r3);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r32, r32);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r32, r33);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r1 * r6;
    WriteSum2<double, double>((double*)inout_shared, r6, r32);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedPoseFixedPointResJac(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
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