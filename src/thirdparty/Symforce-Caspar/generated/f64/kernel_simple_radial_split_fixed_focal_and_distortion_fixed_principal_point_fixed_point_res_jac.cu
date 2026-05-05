#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(focal_and_distortion,
                                            0 * focal_and_distortion_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r5);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = r12 + r18;
    r6 = fma(r9, r19, r6);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r20);
    r21 = r10 * r14;
    r21 = r21 * r13;
    r22 = r11 * r15;
    r22 = r22 * r13;
    r23 = r21 + r22;
    r24 = r14 * r17;
    r25 = 1.00000000000000000e+00;
    r26 = r11 * r11;
    r27 = fma(r16, r26, r25);
    r28 = r24 + r27;
    r6 = fma(r20, r23, r6);
    r6 = fma(r8, r28, r6);
    r28 = r0 * r6;
    r29 = r6 * r6;
    r30 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r11 * r14;
    r32 = r32 * r13;
    r33 = r10 * r15;
    r33 = r33 * r13;
    r34 = r32 + r33;
    r31 = fma(r9, r34, r31);
    r35 = r11 * r15;
    r35 = r35 * r16;
    r21 = r21 + r35;
    r36 = r10 * r10;
    r37 = r16 * r36;
    r27 = r37 + r27;
    r31 = fma(r8, r21, r31);
    r31 = fma(r20, r27, r31);
    r27 = copysign(1.0, r31);
    r27 = fma(r30, r27, r31);
    r30 = r27 * r27;
    r31 = 1.0 / r30;
    r38 = r14 * r15;
    r38 = r38 * r13;
    r12 = r12 + r38;
    r7 = fma(r8, r12, r7);
    r39 = r10 * r15;
    r39 = r39 * r16;
    r32 = r32 + r39;
    r24 = r25 + r24;
    r24 = r24 + r37;
    r7 = fma(r20, r32, r7);
    r7 = fma(r9, r24, r7);
    r24 = r7 * r7;
    r24 = fma(r31, r24, r31 * r29);
    r24 = fma(r5, r24, r25);
    r25 = 1.0 / r27;
    r29 = r24 * r25;
    r2 = fma(r28, r29, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r24;
    r1 = r1 * r25;
    r3 = fma(r7, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r29 = r4 * r3;
    r37 = r14 * r14;
    r15 = r15 * r15;
    r40 = r4 * r15;
    r41 = r37 + r40;
    r42 = r4 * r26;
    r43 = r36 + r42;
    r44 = r41 + r43;
    r44 = fma(r20, r44, r9 * r32);
    r32 = r11 * r17;
    r39 = r39 + r32;
    r42 = r37 + r42;
    r37 = r4 * r36;
    r45 = r15 + r37;
    r42 = r42 + r45;
    r42 = fma(r9, r42, r20 * r39);
    r39 = r0 * r31;
    r39 = r39 * r7;
    r39 = r39 * r24;
    r39 = r39 * r4;
    r46 = fma(r42, r39, r44 * r1);
    r47 = r16 * r6;
    r30 = r27 * r30;
    r27 = 1.0 / r30;
    r47 = r47 * r6;
    r47 = r47 * r42;
    r48 = r16 * r7;
    r48 = r48 * r7;
    r48 = r48 * r42;
    r48 = fma(r27, r48, r27 * r47);
    r47 = r13 * r7;
    r47 = r47 * r44;
    r48 = fma(r31, r47, r48);
    r44 = r13 * r6;
    r11 = r10 * r11;
    r11 = r11 * r16;
    r38 = r38 + r11;
    r23 = fma(r9, r23, r20 * r38);
    r44 = r44 * r23;
    r48 = fma(r31, r44, r48);
    r44 = r0 * r48;
    r47 = r5 * r7;
    r44 = r44 * r25;
    r46 = fma(r47, r44, r46);
    r44 = r4 * r2;
    r38 = r5 * r48;
    r38 = r38 * r25;
    r23 = fma(r23, r1, r28 * r38);
    r38 = r42 * r28;
    r24 = r24 * r4;
    r24 = r24 * r31;
    r23 = fma(r24, r38, r23);
    r44 = fma(r23, r44, r46 * r29);
    r29 = r4 * r3;
    r32 = r33 + r32;
    r32 = fma(r8, r32, r20 * r12);
    r12 = r16 * r7;
    r40 = r36 + r40;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r36 = r26 + r14;
    r40 = r40 + r36;
    r40 = fma(r8, r40, r20 * r21);
    r12 = r12 * r7;
    r12 = r12 * r40;
    r21 = r13 * r7;
    r21 = r21 * r32;
    r21 = fma(r31, r21, r27 * r12);
    r12 = r16 * r6;
    r12 = r12 * r6;
    r12 = r12 * r40;
    r21 = fma(r27, r12, r21);
    r33 = r13 * r6;
    r17 = r10 * r17;
    r35 = r35 + r17;
    r14 = r15 + r14;
    r14 = r14 + r43;
    r14 = fma(r20, r14, r8 * r35);
    r33 = r33 * r14;
    r21 = fma(r31, r33, r21);
    r33 = r0 * r21;
    r33 = r33 * r25;
    r33 = fma(r47, r33, r32 * r1);
    r33 = fma(r40, r39, r33);
    r32 = r4 * r2;
    r12 = r40 * r28;
    r12 = fma(r24, r12, r14 * r1);
    r14 = r5 * r21;
    r14 = r14 * r25;
    r12 = fma(r28, r14, r12);
    r32 = fma(r12, r32, r33 * r29);
    WriteSum2<double, double>((double*)inout_shared, r44, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r2;
    r37 = r26 + r37;
    r37 = r37 + r41;
    r37 = fma(r9, r37, r8 * r19);
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r34 = r17 * r28;
    r34 = fma(r24, r34, r37 * r1);
    r22 = r13 * r7;
    r18 = r11 + r18;
    r36 = r45 + r36;
    r36 = fma(r8, r36, r9 * r18);
    r22 = r22 * r36;
    r8 = r16 * r6;
    r8 = r8 * r6;
    r8 = r8 * r17;
    r8 = fma(r27, r8, r31 * r22);
    r22 = r16 * r7;
    r22 = r22 * r7;
    r22 = r22 * r17;
    r8 = fma(r27, r22, r8);
    r18 = r13 * r6;
    r18 = r18 * r37;
    r8 = fma(r31, r18, r8);
    r18 = r5 * r8;
    r18 = r18 * r25;
    r34 = fma(r28, r18, r34);
    r18 = r4 * r3;
    r22 = r0 * r8;
    r22 = r22 * r25;
    r22 = fma(r17, r39, r47 * r22);
    r22 = fma(r36, r1, r22);
    r18 = fma(r22, r18, r34 * r32);
    r32 = r4 * r2;
    r36 = r27 * r28;
    r31 = r13 * r36;
    r37 = r5 * r6;
    r31 = fma(r37, r31, r1);
    r9 = r16 * r3;
    r45 = r47 * r36;
    r9 = fma(r45, r9, r31 * r32);
    WriteSum2<double, double>((double*)inout_shared, r18, r9);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r4 * r3;
    r18 = r16 * r6;
    r18 = r18 * r6;
    r32 = r16 * r7;
    r32 = r32 * r7;
    r32 = fma(r27, r32, r27 * r18);
    r18 = r0 * r32;
    r18 = r18 * r25;
    r18 = fma(r47, r18, r39);
    r39 = r4 * r2;
    r11 = r5 * r32;
    r11 = r11 * r25;
    r11 = fma(r28, r11, r28 * r24);
    r39 = fma(r11, r39, r18 * r9);
    r9 = r4 * r3;
    r24 = r0 * r13;
    r24 = r24 * r7;
    r24 = r24 * r27;
    r24 = fma(r47, r24, r1);
    r1 = r16 * r2;
    r1 = fma(r45, r1, r24 * r9);
    WriteSum2<double, double>((double*)inout_shared, r1, r39);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r46, r46, r23 * r23);
    r1 = fma(r33, r33, r12 * r12);
    WriteSum2<double, double>((double*)inout_shared, r39, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r34, r34, r22 * r22);
    r39 = r0 * r7;
    r9 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r39 = r39 * r9;
    r39 = r39 * r30;
    r39 = r39 * r28;
    r39 = r39 * r47;
    r39 = r39 * r37;
    r37 = fma(r31, r31, r39);
    WriteSum2<double, double>((double*)inout_shared, r1, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r24, r24, r39);
    r37 = fma(r18, r18, r11 * r11);
    WriteSum2<double, double>((double*)inout_shared, r39, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r23, r12, r46 * r33);
    r39 = fma(r46, r22, r23 * r34);
    WriteSum2<double, double>((double*)inout_shared, r37, r39);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r13 * r45;
    r39 = fma(r46, r45, r23 * r31);
    r37 = fma(r23, r45, r46 * r24);
    WriteSum2<double, double>((double*)inout_shared, r39, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r23, r11, r46 * r18);
    r46 = fma(r33, r22, r12 * r34);
    WriteSum2<double, double>((double*)inout_shared, r23, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r33, r45, r12 * r31);
    r23 = fma(r12, r45, r33 * r24);
    WriteSum2<double, double>((double*)inout_shared, r46, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r12, r11, r33 * r18);
    r33 = fma(r22, r45, r34 * r31);
    WriteSum2<double, double>((double*)inout_shared, r12, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r22, r18, r34 * r11);
    r34 = fma(r34, r45, r22 * r24);
    WriteSum2<double, double>((double*)inout_shared, r34, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r24, r45, r31 * r45);
    r31 = fma(r18, r45, r31 * r11);
    WriteSum2<double, double>((double*)inout_shared, r33, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r11, r45, r24 * r18);
    WriteSum1<double, double>((double*)inout_shared, r45);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar