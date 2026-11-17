#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPointResJacFirstKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, double *pixel,
        unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
        double *focal, unsigned int focal_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0;
  LoadShared<2, double, double>(principal_point, 0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target, r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(focal, 0 * focal_num_alloc,
                                            global_thread_idx, r6, r7);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r10, r11);
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
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r34);
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
    r2 = fma(r2, r4, r0);
    r0 = -2.00000000000000000e+00;
    r5 = fma(r17, r20, r14 * r15);
    r22 = r18 * r19;
    r5 = fma(r4, r22, r5);
    r5 = fma(r13, r16, r5);
    r22 = r5 * r5;
    r22 = r0 * r22;
    r25 = 1.00000000000000000e+00;
    r32 = r17 * r15;
    r32 = fma(r4, r32, r14 * r20);
    r32 = fma(r18, r16, r32);
    r32 = fma(r13, r19, r32);
    r38 = r32 * r32;
    r38 = fma(r0, r38, r25);
    r47 = r22 + r38;
    r47 = fma(r10, r47, r8);
    r8 = 2.00000000000000000e+00;
    r30 = fma(r17, r16, r14 * r19);
    r49 = r13 * r20;
    r30 = fma(r4, r49, r30);
    r30 = fma(r18, r15, r30);
    r49 = r8 * r30;
    r23 = r32 * r49;
    r28 = fma(r18, r20, r17 * r19);
    r28 = fma(r13, r15, r28);
    r28 = fma(r4, r28, r14 * r16);
    r16 = r0 * r28;
    r48 = fma(r5, r16, r23);
    r35 = r8 * r32;
    r12 = r5 * r49;
    r35 = fma(r28, r35, r12);
    r44 = r17 * r13;
    r44 = r44 * r8;
    r31 = r18 * r14;
    r43 = fma(r8, r31, r44);
    r36 = r13 * r14;
    r29 = r17 * r18;
    r29 = r29 * r8;
    r36 = fma(r0, r36, r29);
    r33 = r18 * r18;
    r33 = r33 * r0;
    r45 = r25 + r33;
    r39 = r13 * r13;
    r39 = r0 * r39;
    r45 = r45 + r39;
    r47 = fma(r11, r48, r47);
    r47 = fma(r34, r35, r47);
    r47 = fma(r37, r43, r47);
    r47 = fma(r42, r36, r47);
    r47 = fma(r41, r45, r47);
    r45 = r6 * r47;
    r36 = 1.00000000000000008e-15;
    r12 = fma(r32, r16, r12);
    r12 = fma(r10, r12, r40);
    r31 = fma(r0, r31, r44);
    r33 = r25 + r33;
    r44 = r17 * r17;
    r44 = r0 * r44;
    r33 = r33 + r44;
    r40 = r18 * r13;
    r40 = r40 * r8;
    r43 = r17 * r14;
    r43 = fma(r8, r43, r40);
    r35 = r8 * r5;
    r35 = r35 * r32;
    r49 = fma(r28, r49, r35);
    r48 = r30 * r30;
    r48 = r48 * r0;
    r38 = r48 + r38;
    r12 = fma(r41, r31, r12);
    r12 = fma(r37, r33, r12);
    r12 = fma(r42, r43, r12);
    r12 = fma(r11, r49, r12);
    r12 = fma(r34, r38, r12);
    r38 = copysign(1.0, r12);
    r38 = fma(r36, r38, r12);
    r38 = 1.0 / r38;
    r2 = fma(r38, r45, r2);
    r2 = r4 * r2;
    r3 = fma(r3, r4, r1);
    r1 = r8 * r5;
    r1 = fma(r28, r1, r23);
    r1 = fma(r10, r1, r9);
    r10 = r13 * r14;
    r10 = fma(r8, r10, r29);
    r39 = r25 + r39;
    r39 = r39 + r44;
    r44 = r17 * r14;
    r44 = fma(r0, r44, r40);
    r16 = fma(r30, r16, r35);
    r22 = r25 + r22;
    r22 = r22 + r48;
    r1 = fma(r41, r10, r1);
    r1 = fma(r42, r39, r1);
    r1 = fma(r37, r44, r1);
    r1 = fma(r34, r16, r1);
    r1 = fma(r11, r22, r1);
    r22 = r7 * r1;
    r3 = fma(r38, r22, r3);
    r3 = r4 * r3;
    WriteSum2<double, double>((double *)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(
      out_principal_point_njtr, 0 * out_principal_point_njtr_num_alloc,
      principal_point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r25, r25);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalFixedPointResJacFirst(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
    double *focal, unsigned int focal_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPointResJacFirstKernel<<<n_blocks,
                                                               1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, principal_point,
      principal_point_num_alloc, principal_point_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, focal, focal_num_alloc, point,
      point_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_principal_point_njtr, out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar