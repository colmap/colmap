#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirstKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
        double *pixel, unsigned int pixel_num_alloc, double *pose,
        unsigned int pose_num_alloc, double *focal,
        unsigned int focal_num_alloc, double *principal_point,
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
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(focal, 0 * focal_num_alloc,
                                            global_thread_idx, r6, r7);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(pose, 2 * pose_num_alloc,
                                            global_thread_idx, r15, r16);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r17, r18);
    ReadIdx2<1024, double, double, double2>(pose, 0 * pose_num_alloc,
                                            global_thread_idx, r19, r20);
    r21 = fma(r17, r20, r14 * r15);
    r22 = r18 * r19;
    r21 = fma(r4, r22, r21);
    r21 = fma(r13, r16, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r17 * r15;
    r24 = fma(r4, r24, r14 * r20);
    r24 = fma(r18, r16, r24);
    r24 = fma(r13, r19, r24);
    r25 = r24 * r24;
    r25 = fma(r12, r25, r23);
    r26 = r22 + r25;
    r26 = fma(r10, r26, r8);
    r27 = 2.00000000000000000e+00;
    r28 = fma(r17, r16, r14 * r19);
    r29 = r13 * r20;
    r28 = fma(r4, r29, r28);
    r28 = fma(r18, r15, r28);
    r29 = r27 * r28;
    r30 = r24 * r29;
    r31 = fma(r18, r20, r17 * r19);
    r31 = fma(r13, r15, r31);
    r31 = fma(r4, r31, r14 * r16);
    r32 = r12 * r31;
    r33 = fma(r21, r32, r30);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r27 * r24;
    r36 = r21 * r29;
    r35 = fma(r31, r35, r36);
    ReadIdx1<1024, double, double, double>(pose, 6 * pose_num_alloc,
                                           global_thread_idx, r37);
    r38 = r17 * r13;
    r38 = r38 * r27;
    r39 = r18 * r14;
    r40 = fma(r27, r39, r38);
    ReadIdx2<1024, double, double, double2>(pose, 4 * pose_num_alloc,
                                            global_thread_idx, r41, r42);
    r43 = r13 * r14;
    r44 = r17 * r18;
    r44 = r44 * r27;
    r43 = fma(r12, r43, r44);
    r45 = r18 * r18;
    r45 = r45 * r12;
    r46 = r23 + r45;
    r47 = r13 * r13;
    r47 = r12 * r47;
    r46 = r46 + r47;
    r26 = fma(r11, r33, r26);
    r26 = fma(r34, r35, r26);
    r26 = fma(r37, r40, r26);
    r26 = fma(r42, r43, r26);
    r26 = fma(r41, r46, r26);
    r46 = r6 * r26;
    r43 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r40);
    r36 = fma(r24, r32, r36);
    r36 = fma(r10, r36, r40);
    r39 = fma(r12, r39, r38);
    r45 = r23 + r45;
    r38 = r17 * r17;
    r38 = r12 * r38;
    r45 = r45 + r38;
    r35 = r18 * r13;
    r35 = r35 * r27;
    r33 = r17 * r14;
    r33 = fma(r27, r33, r35);
    r48 = r27 * r21;
    r48 = r48 * r24;
    r29 = fma(r31, r29, r48);
    r49 = r28 * r28;
    r49 = r49 * r12;
    r25 = r49 + r25;
    r36 = fma(r41, r39, r36);
    r36 = fma(r37, r45, r36);
    r36 = fma(r42, r33, r36);
    r36 = fma(r11, r29, r36);
    r36 = fma(r34, r25, r36);
    r25 = copysign(1.0, r36);
    r25 = fma(r43, r25, r36);
    r25 = 1.0 / r25;
    r5 = fma(r25, r46, r5);
    r4 = fma(r3, r4, r1);
    r46 = r27 * r21;
    r46 = fma(r31, r46, r30);
    r46 = fma(r10, r46, r9);
    r30 = r13 * r14;
    r30 = fma(r27, r30, r44);
    r47 = r23 + r47;
    r47 = r47 + r38;
    r38 = r17 * r14;
    r38 = fma(r12, r38, r35);
    r32 = fma(r28, r32, r48);
    r22 = r23 + r22;
    r22 = r22 + r49;
    r46 = fma(r41, r30, r46);
    r46 = fma(r42, r47, r46);
    r46 = fma(r37, r38, r46);
    r46 = fma(r34, r32, r46);
    r46 = fma(r11, r22, r46);
    r22 = r7 * r46;
    r4 = fma(r25, r22, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
    r1 = 1.00000000000000008e-15;
    r5 = -2.00000000000000000e+00;
    r22 = r17 * r15;
    r22 = fma(r4, r22, r14 * r20);
    r22 = fma(r18, r16, r22);
    r22 = fma(r13, r19, r22);
    r25 = r5 * r22;
    r32 = fma(r18, r20, r17 * r19);
    r32 = fma(r13, r15, r32);
    r32 = fma(r4, r32, r14 * r16);
    r38 = fma(r17, r20, r14 * r15);
    r47 = r18 * r19;
    r38 = fma(r4, r47, r38);
    r38 = fma(r13, r16, r38);
    r47 = 2.00000000000000000e+00;
    r16 = fma(r17, r16, r14 * r19);
    r30 = r13 * r20;
    r16 = fma(r4, r30, r16);
    r16 = fma(r18, r15, r16);
    r30 = r47 * r16;
    r49 = r38 * r30;
    r25 = fma(r32, r25, r49);
    r40 = fma(r10, r25, r40);
    r23 = r17 * r13;
    r23 = r23 * r47;
    r28 = r14 * r5;
    r48 = fma(r18, r28, r23);
    r35 = r18 * r18;
    r35 = r5 * r35;
    r12 = 1.00000000000000000e+00;
    r44 = r17 * r17;
    r44 = fma(r5, r44, r12);
    r31 = r35 + r44;
    r43 = r18 * r13;
    r43 = r43 * r47;
    r36 = r17 * r14;
    r36 = fma(r47, r36, r43);
    r29 = r47 * r38;
    r29 = r29 * r22;
    r33 = fma(r32, r30, r29);
    r45 = r16 * r16;
    r45 = r45 * r5;
    r39 = r12 + r45;
    r50 = r22 * r22;
    r50 = r5 * r50;
    r39 = r39 + r50;
    r40 = fma(r41, r48, r40);
    r40 = fma(r37, r31, r40);
    r40 = fma(r42, r36, r40);
    r40 = fma(r11, r33, r40);
    r40 = fma(r34, r39, r40);
    r36 = copysign(1.0, r40);
    r36 = fma(r1, r36, r40);
    r1 = 1.0 / r36;
    r30 = r22 * r30;
    r40 = r38 * r32;
    r31 = fma(r47, r40, r30);
    r9 = fma(r10, r31, r9);
    r48 = r13 * r14;
    r51 = r17 * r18;
    r51 = r51 * r47;
    r48 = fma(r47, r48, r51);
    r52 = r13 * r13;
    r52 = r5 * r52;
    r44 = r52 + r44;
    r43 = fma(r17, r28, r43);
    r53 = r16 * r5;
    r53 = fma(r32, r53, r29);
    r45 = r12 + r45;
    r38 = r38 * r38;
    r38 = r38 * r5;
    r45 = r45 + r38;
    r9 = fma(r41, r48, r9);
    r9 = fma(r42, r44, r9);
    r9 = fma(r37, r43, r9);
    r9 = fma(r34, r53, r9);
    r9 = fma(r11, r45, r9);
    r9 = r7 * r9;
    r3 = fma(r1, r9, r3);
    r43 = r4 * r3;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r36 = r4 * r36;
    r9 = r36 * r9;
    r44 = r7 * r31;
    r44 = fma(r1, r44, r25 * r9);
    r2 = fma(r2, r4, r0);
    r38 = r12 + r38;
    r38 = r38 + r50;
    r10 = fma(r10, r38, r8);
    r40 = fma(r5, r40, r30);
    r30 = r47 * r22;
    r30 = fma(r32, r30, r49);
    r49 = r18 * r14;
    r49 = fma(r47, r49, r23);
    r28 = fma(r13, r28, r51);
    r52 = r12 + r52;
    r52 = r52 + r35;
    r10 = fma(r11, r40, r10);
    r10 = fma(r34, r30, r10);
    r10 = fma(r37, r49, r10);
    r10 = fma(r42, r28, r10);
    r10 = fma(r41, r52, r10);
    r10 = r6 * r10;
    r2 = fma(r1, r10, r2);
    r52 = r4 * r2;
    r41 = r25 * r36;
    r28 = r6 * r38;
    r28 = fma(r1, r28, r10 * r41);
    r52 = fma(r28, r52, r44 * r43);
    r43 = r4 * r3;
    r41 = r7 * r45;
    r41 = fma(r33, r9, r1 * r41);
    r42 = r4 * r2;
    r49 = r6 * r40;
    r37 = r33 * r36;
    r37 = fma(r10, r37, r1 * r49);
    r42 = fma(r37, r42, r41 * r43);
    WriteSum2<double, double>((double *)inout_shared, r52, r42);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r4 * r2;
    r52 = r39 * r36;
    r43 = r6 * r30;
    r43 = fma(r1, r43, r10 * r52);
    r52 = r4 * r3;
    r10 = r7 * r53;
    r10 = fma(r1, r10, r39 * r9);
    r52 = fma(r10, r52, r43 * r42);
    WriteSum1<double, double>((double *)inout_shared, r52);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r28, r28, r44 * r44);
    r42 = fma(r41, r41, r37 * r37);
    WriteSum2<double, double>((double *)inout_shared, r52, r42);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r43, r43, r10 * r10);
    WriteSum1<double, double>((double *)inout_shared, r42);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r28, r37, r44 * r41);
    r28 = fma(r28, r43, r44 * r10);
    WriteSum2<double, double>((double *)inout_shared, r42, r28);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r37, r43, r41 * r10);
    WriteSum1<double, double>((double *)inout_shared, r43);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirst(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *pose,
    unsigned int pose_num_alloc, double *focal, unsigned int focal_num_alloc,
    double *principal_point, unsigned int principal_point_num_alloc,
    double *out_res, unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacFirstKernel<<<
      n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, point, point_num_alloc,
      point_indices, pixel, pixel_num_alloc, pose, pose_num_alloc, focal,
      focal_num_alloc, principal_point, principal_point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar