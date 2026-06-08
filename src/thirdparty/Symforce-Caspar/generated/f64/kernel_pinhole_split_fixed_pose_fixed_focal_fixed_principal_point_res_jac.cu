#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

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
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r5);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r8,
                                            r9);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r13);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r14, r15);
    r16 = fma(r12, r15, r9 * r10);
    r17 = r13 * r14;
    r16 = fma(r4, r17, r16);
    r16 = fma(r8, r11, r16);
    r17 = 2.00000000000000000e+00;
    r18 = fma(r12, r11, r9 * r14);
    r19 = r8 * r15;
    r18 = fma(r4, r19, r18);
    r18 = fma(r13, r10, r18);
    r19 = r17 * r18;
    r20 = r16 * r19;
    r21 = r12 * r10;
    r21 = fma(r4, r21, r9 * r15);
    r21 = fma(r13, r11, r21);
    r21 = fma(r8, r14, r21);
    r22 = -2.00000000000000000e+00;
    r23 = fma(r13, r15, r12 * r14);
    r23 = fma(r8, r10, r23);
    r23 = fma(r4, r23, r9 * r11);
    r11 = r22 * r23;
    r24 = fma(r21, r11, r20);
    r5 = fma(r6, r24, r5);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r25, r26);
    r27 = r12 * r8;
    r27 = r27 * r17;
    r28 = r13 * r9;
    r29 = fma(r22, r28, r27);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r30);
    r31 = r12 * r12;
    r31 = r22 * r31;
    r32 = 1.00000000000000000e+00;
    r33 = r13 * r13;
    r33 = fma(r22, r33, r32);
    r34 = r31 + r33;
    r35 = r13 * r8;
    r35 = r35 * r17;
    r36 = r12 * r9;
    r36 = fma(r17, r36, r35);
    r37 = r17 * r16;
    r37 = r37 * r21;
    r38 = fma(r23, r19, r37);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r21 * r21;
    r40 = r22 * r40;
    r41 = r32 + r40;
    r42 = r18 * r18;
    r42 = r42 * r22;
    r41 = r41 + r42;
    r5 = fma(r25, r29, r5);
    r5 = fma(r30, r34, r5);
    r5 = fma(r26, r36, r5);
    r5 = fma(r7, r38, r5);
    r5 = fma(r39, r41, r5);
    r36 = copysign(1.0, r5);
    r36 = fma(r0, r36, r5);
    r0 = 1.0 / r36;
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r5, r34);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r29,
                                            r43);
    r40 = r32 + r40;
    r44 = r16 * r16;
    r44 = r22 * r44;
    r40 = r40 + r44;
    r29 = fma(r6, r40, r29);
    r19 = r21 * r19;
    r45 = fma(r16, r11, r19);
    r46 = r17 * r21;
    r46 = fma(r23, r46, r20);
    r28 = fma(r17, r28, r27);
    r27 = r8 * r9;
    r20 = r12 * r13;
    r20 = r20 * r17;
    r27 = fma(r22, r27, r20);
    r47 = r8 * r8;
    r47 = r22 * r47;
    r33 = r47 + r33;
    r29 = fma(r7, r45, r29);
    r29 = fma(r39, r46, r29);
    r29 = fma(r30, r28, r29);
    r29 = fma(r26, r27, r29);
    r29 = fma(r25, r33, r29);
    r29 = r5 * r29;
    r2 = fma(r0, r29, r2);
    r3 = fma(r3, r4, r1);
    r1 = r17 * r16;
    r1 = fma(r23, r1, r19);
    r6 = fma(r6, r1, r43);
    r43 = r8 * r9;
    r43 = fma(r17, r43, r20);
    r47 = r32 + r47;
    r47 = r47 + r31;
    r31 = r12 * r9;
    r31 = fma(r22, r31, r35);
    r11 = fma(r18, r11, r37);
    r44 = r32 + r44;
    r44 = r44 + r42;
    r6 = fma(r25, r43, r6);
    r6 = fma(r26, r47, r6);
    r6 = fma(r30, r31, r6);
    r6 = fma(r39, r11, r6);
    r6 = fma(r7, r44, r6);
    r6 = r34 * r6;
    r3 = fma(r0, r6, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r7 = r4 * r3;
    r39 = r34 * r1;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r36 = r4 * r36;
    r31 = r24 * r36;
    r31 = fma(r6, r31, r0 * r39);
    r39 = r4 * r2;
    r29 = r36 * r29;
    r30 = r5 * r40;
    r30 = fma(r0, r30, r24 * r29);
    r39 = fma(r30, r39, r31 * r7);
    r7 = r4 * r3;
    r47 = r34 * r44;
    r26 = r38 * r36;
    r26 = fma(r6, r26, r0 * r47);
    r47 = r4 * r2;
    r43 = r5 * r45;
    r43 = fma(r0, r43, r38 * r29);
    r47 = fma(r43, r47, r26 * r7);
    WriteSum2<double, double>((double*)inout_shared, r39, r47);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r4 * r2;
    r39 = r5 * r46;
    r39 = fma(r0, r39, r41 * r29);
    r29 = r4 * r3;
    r7 = r41 * r36;
    r25 = r34 * r11;
    r25 = fma(r0, r25, r6 * r7);
    r29 = fma(r25, r29, r39 * r47);
    WriteSum1<double, double>((double*)inout_shared, r29);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r30, r30, r31 * r31);
    r47 = fma(r43, r43, r26 * r26);
    WriteSum2<double, double>((double*)inout_shared, r29, r47);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r39, r39, r25 * r25);
    WriteSum1<double, double>((double*)inout_shared, r47);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r31, r26, r30 * r43);
    r31 = fma(r31, r25, r30 * r39);
    WriteSum2<double, double>((double*)inout_shared, r47, r31);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r26, r25, r43 * r39);
    WriteSum1<double, double>((double*)inout_shared, r25);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJac(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                   1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar