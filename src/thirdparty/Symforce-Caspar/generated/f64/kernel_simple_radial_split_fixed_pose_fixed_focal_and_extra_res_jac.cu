#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedFocalAndExtraResJacKernel(
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, double *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
        unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
        double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        double *out_res, unsigned int out_res_num_alloc,
        double *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        double *out_point_jac, unsigned int out_point_jac_num_alloc,
        double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        double *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0;
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
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    r20 = -2.00000000000000000e+00;
    r27 = fma(r15, r18, r12 * r13);
    r24 = r16 * r17;
    r27 = fma(r4, r24, r27);
    r27 = fma(r11, r14, r27);
    r24 = r27 * r27;
    r24 = r20 * r24;
    r23 = 1.00000000000000000e+00;
    r21 = r15 * r13;
    r21 = fma(r4, r21, r12 * r18);
    r21 = fma(r16, r14, r21);
    r21 = fma(r11, r17, r21);
    r38 = r21 * r21;
    r38 = fma(r20, r38, r23);
    r34 = r24 + r38;
    r34 = fma(r8, r34, r6);
    r30 = 2.00000000000000000e+00;
    r36 = fma(r15, r14, r12 * r17);
    r45 = r11 * r18;
    r36 = fma(r4, r45, r36);
    r36 = fma(r16, r13, r36);
    r45 = r30 * r36;
    r28 = r21 * r45;
    r48 = fma(r16, r18, r15 * r17);
    r48 = fma(r11, r13, r48);
    r48 = fma(r4, r48, r12 * r14);
    r26 = r20 * r48;
    r47 = fma(r27, r26, r28);
    r31 = r30 * r21;
    r10 = r27 * r45;
    r31 = fma(r48, r31, r10);
    r42 = r15 * r11;
    r42 = r42 * r30;
    r29 = r16 * r12;
    r46 = fma(r30, r29, r42);
    r43 = r11 * r12;
    r37 = r15 * r16;
    r37 = r37 * r30;
    r43 = fma(r20, r43, r37);
    r49 = r16 * r16;
    r49 = r49 * r20;
    r50 = r23 + r49;
    r51 = r11 * r11;
    r51 = r20 * r51;
    r50 = r50 + r51;
    r34 = fma(r9, r47, r34);
    r34 = fma(r32, r31, r34);
    r34 = fma(r35, r46, r34);
    r34 = fma(r40, r43, r34);
    r34 = fma(r39, r50, r34);
    r50 = 1.00000000000000008e-15;
    r10 = fma(r21, r26, r10);
    r10 = fma(r8, r10, r33);
    r29 = fma(r20, r29, r42);
    r49 = r23 + r49;
    r42 = r15 * r15;
    r42 = r20 * r42;
    r49 = r49 + r42;
    r43 = r16 * r11;
    r43 = r43 * r30;
    r46 = r15 * r12;
    r46 = fma(r30, r46, r43);
    r31 = r30 * r27;
    r31 = r31 * r21;
    r45 = fma(r48, r45, r31);
    r47 = r36 * r36;
    r47 = r47 * r20;
    r38 = r47 + r38;
    r10 = fma(r39, r29, r10);
    r10 = fma(r35, r49, r10);
    r10 = fma(r40, r46, r10);
    r10 = fma(r9, r45, r10);
    r10 = fma(r32, r38, r10);
    r38 = copysign(1.0, r10);
    r38 = fma(r50, r38, r10);
    r50 = r38 * r38;
    r50 = 1.0 / r50;
    r10 = r34 * r34;
    r45 = r30 * r27;
    r45 = fma(r48, r45, r28);
    r45 = fma(r8, r45, r7);
    r28 = r11 * r12;
    r28 = fma(r30, r28, r37);
    r51 = r23 + r51;
    r51 = r51 + r42;
    r42 = r15 * r12;
    r42 = fma(r20, r42, r43);
    r26 = fma(r36, r26, r31);
    r24 = r23 + r24;
    r24 = r24 + r47;
    r45 = fma(r39, r28, r45);
    r45 = fma(r40, r51, r45);
    r45 = fma(r35, r42, r45);
    r45 = fma(r32, r26, r45);
    r45 = fma(r9, r24, r45);
    r24 = r45 * r45;
    r24 = fma(r50, r24, r50 * r10);
    r24 = fma(r41, r24, r23);
    r24 = r44 * r24;
    r38 = 1.0 / r38;
    r24 = r24 * r38;
    r5 = fma(r34, r24, r5);
    r5 = r4 * r5;
    r34 = fma(r3, r4, r1);
    r34 = fma(r45, r24, r34);
    r34 = r4 * r34;
    WriteSum2<double, double>((double *)inout_shared, r5, r34);
  };
  FlushSumShared<2, double>(
      out_principal_point_njtr, 0 * out_principal_point_njtr_num_alloc,
      principal_point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r23, r23);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r15, r14, r12 * r17);
    r34 = r11 * r18;
    r5 = -1.00000000000000000e+00;
    r23 = fma(r5, r34, r23);
    r23 = fma(r16, r13, r23);
    r34 = 2.00000000000000000e+00;
    r4 = fma(r15, r18, r12 * r13);
    r24 = r16 * r17;
    r4 = fma(r5, r24, r4);
    r4 = fma(r11, r14, r4);
    r24 = r34 * r4;
    r45 = r23 * r24;
    r38 = -2.00000000000000000e+00;
    r50 = r15 * r13;
    r50 = fma(r5, r50, r12 * r18);
    r50 = fma(r16, r14, r50);
    r50 = fma(r11, r17, r50);
    r10 = fma(r16, r18, r15 * r17);
    r10 = fma(r11, r13, r10);
    r10 = fma(r5, r10, r12 * r14);
    r14 = r50 * r10;
    r26 = fma(r38, r14, r45);
    r42 = 1.00000000000000000e+00;
    r51 = r4 * r4;
    r51 = r51 * r38;
    r28 = r38 * r50;
    r28 = fma(r50, r28, r42);
    r47 = r51 + r28;
    r6 = fma(r8, r47, r6);
    r36 = r34 * r23;
    r36 = r36 * r50;
    r31 = r4 * r38;
    r31 = fma(r10, r31, r36);
    r14 = fma(r34, r14, r45);
    r45 = r15 * r11;
    r45 = r45 * r34;
    r43 = r16 * r12;
    r20 = fma(r34, r43, r45);
    r37 = r11 * r12;
    r48 = r15 * r16;
    r48 = r48 * r34;
    r37 = fma(r38, r37, r48);
    r46 = r16 * r16;
    r46 = r46 * r38;
    r49 = r42 + r46;
    r29 = r11 * r11;
    r29 = r38 * r29;
    r49 = r49 + r29;
    r6 = fma(r9, r31, r6);
    r6 = fma(r32, r14, r6);
    r6 = fma(r35, r20, r6);
    r6 = fma(r40, r37, r6);
    r6 = fma(r39, r49, r6);
    r49 = 1.00000000000000008e-15;
    r33 = fma(r8, r26, r33);
    r43 = fma(r38, r43, r45);
    r46 = r42 + r46;
    r45 = r15 * r15;
    r45 = r38 * r45;
    r46 = r46 + r45;
    r37 = r16 * r11;
    r37 = r37 * r34;
    r20 = r15 * r12;
    r20 = fma(r34, r20, r37);
    r52 = r34 * r23;
    r53 = r50 * r24;
    r52 = fma(r10, r52, r53);
    r54 = r23 * r23;
    r54 = r38 * r54;
    r28 = r54 + r28;
    r33 = fma(r39, r43, r33);
    r33 = fma(r35, r46, r33);
    r33 = fma(r40, r20, r33);
    r33 = fma(r9, r52, r33);
    r33 = fma(r32, r28, r33);
    r20 = copysign(1.0, r33);
    r20 = fma(r49, r20, r33);
    r49 = r20 * r20;
    r33 = 1.0 / r49;
    r46 = r6 * r33;
    r24 = fma(r10, r24, r36);
    r8 = fma(r8, r24, r7);
    r7 = r11 * r12;
    r7 = fma(r34, r7, r48);
    r29 = r42 + r29;
    r29 = r29 + r45;
    r45 = r15 * r12;
    r45 = fma(r38, r45, r37);
    r37 = r23 * r38;
    r37 = fma(r10, r37, r53);
    r51 = r42 + r51;
    r51 = r51 + r54;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r29, r8);
    r8 = fma(r35, r45, r8);
    r8 = fma(r32, r37, r8);
    r8 = fma(r9, r51, r8);
    r9 = r8 * r8;
    r32 = fma(r33, r9, r6 * r46);
    r32 = fma(r41, r32, r42);
    r32 = r44 * r32;
    r42 = r5 * r32;
    r42 = r42 * r46;
    r45 = 1.0 / r20;
    r35 = r45 * r32;
    r29 = fma(r47, r35, r26 * r42);
    r40 = r44 * r41;
    r7 = r34 * r47;
    r49 = r20 * r49;
    r49 = 1.0 / r49;
    r49 = r38 * r49;
    r20 = r26 * r49;
    r20 = fma(r9, r20, r46 * r7);
    r7 = r6 * r6;
    r7 = r7 * r49;
    r39 = r34 * r24;
    r39 = r39 * r8;
    r20 = fma(r33, r39, r20);
    r20 = fma(r26, r7, r20);
    r40 = r40 * r20;
    r40 = r40 * r45;
    r29 = fma(r6, r40, r29);
    r20 = r5 * r26;
    r20 = r20 * r8;
    r20 = r20 * r33;
    r20 = fma(r24, r35, r32 * r20);
    r20 = fma(r8, r40, r20);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r29, r20);
    r40 = fma(r31, r35, r52 * r42);
    r39 = r44 * r41;
    r54 = r34 * r31;
    r54 = fma(r46, r54, r52 * r7);
    r53 = r34 * r51;
    r53 = r53 * r8;
    r54 = fma(r33, r53, r54);
    r10 = r52 * r49;
    r54 = fma(r9, r10, r54);
    r39 = r39 * r6;
    r39 = r39 * r54;
    r40 = fma(r45, r39, r40);
    r39 = r5 * r52;
    r39 = r39 * r8;
    r39 = r39 * r33;
    r39 = fma(r32, r39, r51 * r35);
    r10 = r44 * r41;
    r10 = r10 * r8;
    r10 = r10 * r54;
    r39 = fma(r45, r10, r39);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r40, r39);
    r10 = r44 * r41;
    r54 = r34 * r14;
    r53 = r28 * r49;
    r53 = fma(r9, r53, r46 * r54);
    r54 = r34 * r37;
    r54 = r54 * r8;
    r53 = fma(r33, r54, r53);
    r53 = fma(r28, r7, r53);
    r10 = r10 * r6;
    r10 = r10 * r53;
    r10 = fma(r45, r10, r28 * r42);
    r10 = fma(r14, r35, r10);
    r42 = r44 * r41;
    r42 = r42 * r8;
    r42 = r42 * r53;
    r53 = r5 * r28;
    r53 = r53 * r8;
    r53 = r53 * r33;
    r53 = fma(r32, r53, r45 * r42);
    r53 = fma(r37, r35, r53);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r10, r53);
    r42 = r5 * r29;
    r2 = fma(r2, r5, r0);
    r2 = fma(r6, r35, r2);
    r6 = r5 * r20;
    r3 = fma(r3, r5, r1);
    r3 = fma(r8, r35, r3);
    r6 = fma(r3, r6, r2 * r42);
    r42 = r5 * r39;
    r35 = r5 * r40;
    r35 = fma(r2, r35, r3 * r42);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r5 * r53;
    r6 = r5 * r10;
    r6 = fma(r2, r6, r3 * r35);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r20, r20, r29 * r29);
    r35 = fma(r39, r39, r40 * r40);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r53, r53, r10 * r10);
    WriteSum1<double, double>((double *)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r20, r39, r29 * r40);
    r6 = fma(r29, r10, r20 * r53);
    WriteSum2<double, double>((double *)inout_shared, r35, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r40, r10, r39 * r53);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
}

void SimpleRadialSplitFixedPoseFixedFocalAndExtraResJac(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
    double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    double *out_res, unsigned int out_res_num_alloc,
    double *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    double *out_point_jac, unsigned int out_point_jac_num_alloc,
    double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPoseFixedFocalAndExtraResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, principal_point,
      principal_point_num_alloc, principal_point_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, focal_and_extra, focal_and_extra_num_alloc, out_res,
      out_res_num_alloc, out_principal_point_jac,
      out_principal_point_jac_num_alloc, out_principal_point_njtr,
      out_principal_point_njtr_num_alloc, out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar