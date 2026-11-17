#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
        double *pixel, unsigned int pixel_num_alloc, double *pose,
        unsigned int pose_num_alloc, double *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        double *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r6, r7);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(pose, 2 * pose_num_alloc,
                                            global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
    ReadIdx2<1024, double, double, double2>(pose, 0 * pose_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r19 = fma(r4, r20, r19);
    r19 = fma(r11, r14, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r15 * r13;
    r22 = fma(r4, r22, r12 * r18);
    r22 = fma(r16, r14, r22);
    r22 = fma(r11, r17, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r15, r14, r12 * r17);
    r27 = r11 * r18;
    r26 = fma(r4, r27, r26);
    r26 = fma(r16, r13, r26);
    r27 = r25 * r26;
    r28 = r22 * r27;
    r29 = fma(r16, r18, r15 * r17);
    r29 = fma(r11, r13, r29);
    r29 = fma(r4, r29, r12 * r14);
    r30 = r10 * r29;
    r31 = fma(r19, r30, r28);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r25 * r22;
    r34 = r19 * r27;
    r33 = fma(r29, r33, r34);
    ReadIdx1<1024, double, double, double>(pose, 6 * pose_num_alloc,
                                           global_thread_idx, r35);
    r36 = r15 * r11;
    r36 = r36 * r25;
    r37 = r16 * r12;
    r38 = fma(r25, r37, r36);
    ReadIdx2<1024, double, double, double2>(pose, 4 * pose_num_alloc,
                                            global_thread_idx, r39, r40);
    r41 = r11 * r12;
    r42 = r15 * r16;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r16 * r16;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r11 * r11;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx, r44, r41);
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r15 * r15;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r16 * r11;
    r31 = r31 * r25;
    r46 = r15 * r12;
    r46 = fma(r25, r46, r31);
    r47 = r25 * r19;
    r47 = r47 * r22;
    r27 = fma(r29, r27, r47);
    r48 = r26 * r26;
    r48 = r48 * r10;
    r23 = r48 + r23;
    r34 = fma(r39, r37, r34);
    r34 = fma(r35, r43, r34);
    r34 = fma(r40, r46, r34);
    r34 = fma(r9, r27, r34);
    r34 = fma(r32, r23, r34);
    r23 = copysign(1.0, r34);
    r23 = fma(r38, r23, r34);
    r38 = r23 * r23;
    r38 = 1.0 / r38;
    r34 = r24 * r24;
    r27 = r25 * r19;
    r27 = fma(r29, r27, r28);
    r27 = fma(r8, r27, r7);
    r28 = r11 * r12;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r15 * r12;
    r36 = fma(r10, r36, r31);
    r30 = fma(r26, r30, r47);
    r20 = r21 + r20;
    r20 = r20 + r48;
    r27 = fma(r39, r28, r27);
    r27 = fma(r40, r45, r27);
    r27 = fma(r35, r36, r27);
    r27 = fma(r32, r30, r27);
    r27 = fma(r9, r20, r27);
    r20 = r27 * r27;
    r20 = fma(r38, r20, r38 * r34);
    r20 = fma(r41, r20, r21);
    r20 = r44 * r20;
    r23 = 1.0 / r23;
    r20 = r20 * r23;
    r5 = fma(r24, r20, r5);
    r4 = fma(r3, r4, r1);
    r4 = fma(r27, r20, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = -2.00000000000000000e+00;
    r5 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r5 = fma(r4, r20, r5);
    r5 = fma(r11, r14, r5);
    r20 = r5 * r5;
    r20 = r0 * r20;
    r27 = 1.00000000000000000e+00;
    r24 = r15 * r13;
    r24 = fma(r4, r24, r12 * r18);
    r24 = fma(r16, r14, r24);
    r24 = fma(r11, r17, r24);
    r23 = r24 * r24;
    r23 = fma(r0, r23, r27);
    r21 = r20 + r23;
    r6 = fma(r8, r21, r6);
    r38 = r5 * r0;
    r34 = fma(r16, r18, r15 * r17);
    r34 = fma(r11, r13, r34);
    r34 = fma(r4, r34, r12 * r14);
    r30 = 2.00000000000000000e+00;
    r14 = fma(r15, r14, r12 * r17);
    r36 = r11 * r18;
    r14 = fma(r4, r36, r14);
    r14 = fma(r16, r13, r14);
    r36 = r30 * r14;
    r45 = r24 * r36;
    r38 = fma(r34, r38, r45);
    r28 = r30 * r24;
    r48 = r5 * r36;
    r28 = fma(r34, r28, r48);
    r26 = r15 * r11;
    r26 = r26 * r30;
    r47 = r16 * r12;
    r31 = fma(r30, r47, r26);
    r10 = r11 * r12;
    r42 = r15 * r16;
    r42 = r42 * r30;
    r10 = fma(r0, r10, r42);
    r29 = r16 * r16;
    r29 = r29 * r0;
    r46 = r27 + r29;
    r43 = r11 * r11;
    r43 = r0 * r43;
    r46 = r46 + r43;
    r6 = fma(r9, r38, r6);
    r6 = fma(r32, r28, r6);
    r6 = fma(r35, r31, r6);
    r6 = fma(r40, r10, r6);
    r6 = fma(r39, r46, r6);
    r46 = 1.00000000000000008e-15;
    r10 = r0 * r24;
    r10 = fma(r34, r10, r48);
    r33 = fma(r8, r10, r33);
    r47 = fma(r0, r47, r26);
    r29 = r27 + r29;
    r26 = r15 * r15;
    r26 = r0 * r26;
    r29 = r29 + r26;
    r48 = r16 * r11;
    r48 = r48 * r30;
    r31 = r15 * r12;
    r31 = fma(r30, r31, r48);
    r37 = r30 * r5;
    r37 = r37 * r24;
    r36 = fma(r34, r36, r37);
    r49 = r14 * r14;
    r49 = r49 * r0;
    r23 = r49 + r23;
    r33 = fma(r39, r47, r33);
    r33 = fma(r35, r29, r33);
    r33 = fma(r40, r31, r33);
    r33 = fma(r9, r36, r33);
    r33 = fma(r32, r23, r33);
    r31 = copysign(1.0, r33);
    r31 = fma(r46, r31, r33);
    r46 = 1.0 / r31;
    r33 = r31 * r31;
    r29 = 1.0 / r33;
    r47 = r6 * r29;
    r50 = r30 * r5;
    r50 = fma(r34, r50, r45);
    r8 = fma(r8, r50, r7);
    r7 = r11 * r12;
    r7 = fma(r30, r7, r42);
    r43 = r27 + r43;
    r43 = r43 + r26;
    r26 = r15 * r12;
    r26 = fma(r0, r26, r48);
    r48 = r14 * r0;
    r48 = fma(r34, r48, r37);
    r20 = r27 + r20;
    r20 = r20 + r49;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r43, r8);
    r8 = fma(r35, r26, r8);
    r8 = fma(r32, r48, r8);
    r8 = fma(r9, r20, r8);
    r9 = r8 * r8;
    r32 = fma(r29, r9, r6 * r47);
    r32 = fma(r41, r32, r27);
    r32 = r44 * r32;
    r27 = r46 * r32;
    r2 = fma(r6, r27, r2);
    r26 = r4 * r2;
    r35 = r4 * r32;
    r35 = r35 * r47;
    r43 = fma(r21, r27, r10 * r35);
    r40 = r44 * r41;
    r7 = r30 * r21;
    r33 = r31 * r33;
    r33 = 1.0 / r33;
    r33 = r0 * r33;
    r31 = r10 * r33;
    r31 = fma(r9, r31, r47 * r7);
    r7 = r6 * r6;
    r7 = r7 * r33;
    r39 = r30 * r50;
    r39 = r39 * r8;
    r31 = fma(r29, r39, r31);
    r31 = fma(r10, r7, r31);
    r40 = r40 * r31;
    r40 = r40 * r46;
    r43 = fma(r6, r40, r43);
    r31 = r4 * r10;
    r31 = r31 * r8;
    r31 = r31 * r29;
    r31 = fma(r50, r27, r32 * r31);
    r31 = fma(r8, r40, r31);
    r40 = r4 * r31;
    r3 = fma(r3, r4, r1);
    r3 = fma(r8, r27, r3);
    r40 = fma(r3, r40, r43 * r26);
    r26 = r4 * r3;
    r1 = r4 * r36;
    r1 = r1 * r8;
    r1 = r1 * r29;
    r1 = fma(r32, r1, r20 * r27);
    r39 = r44 * r41;
    r49 = r30 * r38;
    r49 = fma(r47, r49, r36 * r7);
    r37 = r30 * r20;
    r37 = r37 * r8;
    r49 = fma(r29, r37, r49);
    r34 = r36 * r33;
    r49 = fma(r9, r34, r49);
    r39 = r39 * r8;
    r39 = r39 * r49;
    r1 = fma(r46, r39, r1);
    r39 = r4 * r2;
    r34 = fma(r38, r27, r36 * r35);
    r37 = r44 * r41;
    r37 = r37 * r6;
    r37 = r37 * r49;
    r34 = fma(r46, r37, r34);
    r39 = fma(r34, r39, r1 * r26);
    WriteSum2<double, double>((double *)inout_shared, r40, r39);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r4 * r3;
    r40 = r44 * r41;
    r26 = r30 * r28;
    r37 = r23 * r33;
    r37 = fma(r9, r37, r47 * r26);
    r26 = r30 * r48;
    r26 = r26 * r8;
    r37 = fma(r29, r26, r37);
    r37 = fma(r23, r7, r37);
    r40 = r40 * r8;
    r40 = r40 * r37;
    r26 = r4 * r23;
    r26 = r26 * r8;
    r26 = r26 * r29;
    r26 = fma(r32, r26, r46 * r40);
    r26 = fma(r48, r27, r26);
    r40 = r4 * r2;
    r32 = r44 * r41;
    r32 = r32 * r6;
    r32 = r32 * r37;
    r32 = fma(r46, r32, r23 * r35);
    r32 = fma(r28, r27, r32);
    r40 = fma(r32, r40, r26 * r39);
    WriteSum1<double, double>((double *)inout_shared, r40);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r31, r31, r43 * r43);
    r39 = fma(r1, r1, r34 * r34);
    WriteSum2<double, double>((double *)inout_shared, r40, r39);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r26, r26, r32 * r32);
    WriteSum1<double, double>((double *)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r31, r1, r43 * r34);
    r43 = fma(r43, r32, r31 * r26);
    WriteSum2<double, double>((double *)inout_shared, r39, r43);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r34, r32, r1 * r26);
    WriteSum1<double, double>((double *)inout_shared, r32);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *pose,
    unsigned int pose_num_alloc, double *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel<<<
      n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, point, point_num_alloc,
      point_indices, pixel, pixel_num_alloc, pose, pose_num_alloc,
      focal_and_extra, focal_and_extra_num_alloc, principal_point,
      principal_point_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar