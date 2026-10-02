#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, double *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
        unsigned int pixel_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double *out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double *const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double *const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        double *out_point_jac, unsigned int out_point_jac_num_alloc,
        double *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        double *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
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
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0, r63 = 0, r64 = 0,
         r65 = 0, r66 = 0, r67 = 0, r68 = 0, r69 = 0, r70 = 0, r71 = 0, r72 = 0,
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0,
         r81 = 0, r82 = 0, r83 = 0, r84 = 0, r85 = 0, r86 = 0, r87 = 0, r88 = 0,
         r89 = 0, r90 = 0, r91 = 0, r92 = 0, r93 = 0, r94 = 0, r95 = 0, r96 = 0,
         r97 = 0, r98 = 0, r99 = 0, r100 = 0;

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
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r13, r14);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r15, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r16, r17, r11 * r14);
    r20 = r15 * r18;
    r19 = fma(r4, r20, r19);
    r19 = fma(r12, r13, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r11 * r17;
    r22 = fma(r4, r22, r16 * r14);
    r22 = fma(r12, r18, r22);
    r22 = fma(r15, r13, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r12, r17, r15 * r14);
    r27 = r16 * r13;
    r26 = fma(r4, r27, r26);
    r26 = fma(r11, r18, r26);
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
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r17 * r13;
    r36 = r36 * r25;
    r37 = r18 * r14;
    r38 = fma(r25, r37, r36);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r39, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r13 * r14;
    r42 = r17 * r18;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r18 * r18;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r13 * r13;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
  };
  LoadShared<2, double, double>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target, r44,
                        r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r17 * r17;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r18 * r13;
    r31 = r31 * r25;
    r46 = r17 * r14;
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
    r28 = r13 * r14;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r17 * r14;
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
    r4 = fma(r16, r17, r11 * r14);
    r5 = r15 * r18;
    r20 = -1.00000000000000000e+00;
    r4 = fma(r20, r5, r4);
    r4 = fma(r12, r13, r4);
    r5 = r4 * r4;
    r27 = -2.00000000000000000e+00;
    r5 = r5 * r27;
    r24 = 1.00000000000000000e+00;
    r23 = fma(r12, r18, r16 * r14);
    r21 = r15 * r13;
    r38 = r11 * r17;
    r23 = r23 + r21;
    r23 = fma(r20, r38, r23);
    r34 = r27 * r23;
    r34 = fma(r23, r34, r24);
    r30 = r5 + r34;
    r30 = fma(r8, r30, r6);
    r36 = 2.00000000000000000e+00;
    r45 = fma(r12, r17, r15 * r14);
    r28 = r16 * r13;
    r45 = fma(r20, r28, r45);
    r45 = fma(r11, r18, r45);
    r28 = r36 * r45;
    r28 = r28 * r23;
    r48 = r4 * r27;
    r26 = fma(r16, r18, r15 * r17);
    r26 = fma(r11, r13, r26);
    r26 = fma(r20, r26, r12 * r14);
    r48 = fma(r26, r48, r28);
    r47 = r36 * r4;
    r47 = r47 * r45;
    r31 = r36 * r26;
    r10 = fma(r23, r31, r47);
    r42 = r17 * r13;
    r42 = r42 * r36;
    r29 = r18 * r14;
    r46 = fma(r36, r29, r42);
    r43 = r13 * r14;
    r37 = r17 * r18;
    r37 = r37 * r36;
    r43 = fma(r27, r43, r37);
    r49 = r18 * r18;
    r49 = r49 * r27;
    r50 = r24 + r49;
    r51 = r13 * r13;
    r51 = r51 * r27;
    r50 = r50 + r51;
    r30 = fma(r9, r48, r30);
    r30 = fma(r32, r10, r30);
    r30 = fma(r35, r46, r30);
    r30 = fma(r40, r43, r30);
    r30 = fma(r39, r50, r30);
    r10 = r15 * r14;
    r48 = -5.00000000000000000e-01;
    r52 = r12 * r17;
    r52 = fma(r48, r52, r48 * r10);
    r10 = r11 * r18;
    r52 = fma(r48, r10, r52);
    r53 = r16 * r13;
    r54 = 5.00000000000000000e-01;
    r52 = fma(r54, r53, r52);
    r53 = r23 * r52;
    r10 = r11 * r14;
    r55 = r16 * r17;
    r55 = fma(r54, r55, r54 * r10);
    r10 = r15 * r18;
    r55 = fma(r48, r10, r55);
    r56 = r12 * r54;
    r55 = fma(r13, r56, r55);
    r10 = fma(r55, r31, r36 * r53);
    r57 = r36 * r45;
    r58 = r12 * r18;
    r59 = r16 * r48;
    r58 = fma(r14, r59, r48 * r58);
    r58 = fma(r54, r38, r58);
    r58 = fma(r48, r21, r58);
    r60 = r36 * r4;
    r61 = r15 * r17;
    r62 = r11 * r13;
    r62 = fma(r48, r62, r48 * r61);
    r62 = fma(r14, r56, r62);
    r62 = fma(r18, r59, r62);
    r60 = r60 * r62;
    r57 = fma(r58, r57, r60);
    r10 = r10 + r57;
    r61 = r36 * r23;
    r61 = r61 * r62;
    r63 = r36 * r45;
    r63 = r63 * r55;
    r64 = r61 + r63;
    r65 = r4 * r27;
    r64 = fma(r52, r65, r64);
    r66 = r27 * r26;
    r64 = fma(r58, r66, r64);
    r64 = fma(r9, r64, r32 * r10);
    r10 = r23 * r55;
    r66 = -4.00000000000000000e+00;
    r10 = r10 * r66;
    r65 = r4 * r58;
    r67 = r66 * r65;
    r68 = r10 + r67;
    r64 = fma(r8, r68, r64);
    r68 = r36 * r64;
    r69 = 1.00000000000000008e-15;
    r70 = r27 * r23;
    r70 = fma(r26, r70, r47);
    r70 = fma(r8, r70, r33);
    r29 = fma(r27, r29, r42);
    r49 = r24 + r49;
    r42 = r17 * r17;
    r42 = r42 * r27;
    r49 = r49 + r42;
    r47 = r18 * r13;
    r47 = r47 * r36;
    r71 = r17 * r14;
    r71 = fma(r36, r71, r47);
    r72 = r36 * r4;
    r72 = r72 * r23;
    r73 = fma(r45, r31, r72);
    r74 = r45 * r45;
    r74 = r74 * r27;
    r34 = r74 + r34;
    r70 = fma(r39, r29, r70);
    r70 = fma(r35, r49, r70);
    r70 = fma(r40, r71, r70);
    r70 = fma(r9, r73, r70);
    r70 = fma(r32, r34, r70);
    r34 = copysign(1.0, r70);
    r34 = fma(r69, r34, r70);
    r69 = r34 * r34;
    r70 = 1.0 / r69;
    r73 = r30 * r70;
    r75 = r36 * r23;
    r75 = r75 * r58;
    r76 = r36 * r4;
    r76 = fma(r55, r76, r75);
    r77 = r36 * r45;
    r77 = r77 * r52;
    r78 = r62 * r31;
    r79 = r77 + r78;
    r80 = r76 + r79;
    r81 = r27 * r26;
    r81 = fma(r27, r53, r55 * r81);
    r81 = r81 + r57;
    r81 = fma(r8, r81, r9 * r80);
    r80 = r45 * r66;
    r55 = r62 * r80;
    r10 = r10 + r55;
    r81 = fma(r32, r10, r81);
    r10 = r30 * r30;
    r69 = r34 * r69;
    r69 = 1.0 / r69;
    r69 = r27 * r69;
    r10 = r10 * r69;
    r68 = fma(r81, r10, r73 * r68);
    r82 = r81 * r69;
    r28 = fma(r4, r31, r28);
    r28 = fma(r8, r28, r7);
    r83 = r13 * r14;
    r83 = fma(r36, r83, r37);
    r51 = r24 + r51;
    r51 = r51 + r42;
    r42 = r17 * r14;
    r42 = fma(r27, r42, r47);
    r47 = r45 * r27;
    r47 = fma(r26, r47, r72);
    r5 = r24 + r5;
    r5 = r5 + r74;
    r28 = fma(r39, r83, r28);
    r28 = fma(r40, r51, r28);
    r28 = fma(r35, r42, r28);
    r28 = fma(r32, r47, r28);
    r28 = fma(r9, r5, r28);
    r5 = r28 * r28;
    r68 = fma(r5, r82, r68);
    r47 = r36 * r28;
    r74 = r45 * r27;
    r72 = r27 * r26;
    r72 = r72 * r62;
    r74 = fma(r52, r74, r72);
    r74 = r74 + r76;
    r55 = r67 + r55;
    r55 = fma(r9, r55, r32 * r74);
    r63 = fma(r58, r31, r63);
    r74 = r36 * r4;
    r74 = fma(r52, r74, r61);
    r63 = r63 + r74;
    r55 = fma(r8, r63, r55);
    r47 = r47 * r55;
    r68 = fma(r70, r47, r68);
    r68 = r41 * r68;
    r34 = 1.0 / r34;
    r34 = r44 * r34;
    r68 = r68 * r34;
    r47 = fma(r70, r5, r30 * r73);
    r47 = fma(r41, r47, r24);
    r24 = r47 * r34;
    r82 = fma(r64, r24, r30 * r68);
    r63 = r44 * r20;
    r63 = r63 * r47;
    r63 = r63 * r73;
    r82 = fma(r81, r63, r82);
    r55 = fma(r55, r24, r28 * r68);
    r68 = r44 * r20;
    r68 = r68 * r81;
    r68 = r68 * r28;
    r68 = r68 * r47;
    r55 = fma(r70, r68, r55);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r82, r55);
    r68 = r41 * r30;
    r78 = r75 + r78;
    r75 = r36 * r4;
    r61 = r11 * r14;
    r67 = r15 * r18;
    r67 = fma(r54, r67, r48 * r61);
    r61 = r12 * r13;
    r67 = fma(r48, r61, r67);
    r67 = fma(r17, r59, r67);
    r75 = r75 * r67;
    r61 = r36 * r45;
    r76 = r15 * r14;
    r37 = r11 * r18;
    r37 = fma(r54, r37, r54 * r76);
    r37 = fma(r17, r56, r37);
    r37 = fma(r13, r59, r37);
    r61 = fma(r37, r61, r75);
    r78 = r78 + r61;
    r59 = r23 * r62;
    r59 = r59 * r66;
    r76 = r4 * r66;
    r76 = r76 * r37;
    r84 = r59 + r76;
    r84 = fma(r8, r84, r32 * r78);
    r78 = r27 * r26;
    r78 = fma(r27, r65, r37 * r78);
    r85 = r36 * r45;
    r85 = r85 * r62;
    r86 = r36 * r23;
    r86 = fma(r67, r86, r85);
    r78 = r78 + r86;
    r84 = fma(r9, r78, r84);
    r78 = r36 * r84;
    r87 = r36 * r28;
    r88 = r45 * r27;
    r88 = fma(r58, r88, r60);
    r60 = r36 * r23;
    r60 = r60 * r37;
    r89 = r27 * r26;
    r88 = fma(r67, r89, r88);
    r88 = r88 + r60;
    r37 = fma(r37, r31, r36 * r65);
    r37 = r37 + r86;
    r37 = fma(r8, r37, r32 * r88);
    r88 = r67 * r80;
    r76 = r76 + r88;
    r37 = fma(r9, r76, r37);
    r87 = r87 * r37;
    r87 = fma(r70, r87, r73 * r78);
    r78 = r27 * r23;
    r78 = fma(r58, r78, r72);
    r78 = r78 + r61;
    r60 = fma(r67, r31, r60);
    r60 = r60 + r57;
    r60 = fma(r9, r60, r8 * r78);
    r88 = r59 + r88;
    r60 = fma(r32, r88, r60);
    r88 = r60 * r69;
    r87 = fma(r5, r88, r87);
    r87 = fma(r60, r10, r87);
    r68 = r68 * r87;
    r68 = fma(r60, r63, r34 * r68);
    r68 = fma(r84, r24, r68);
    r88 = r41 * r28;
    r88 = r88 * r87;
    r88 = fma(r34, r88, r37 * r24);
    r37 = r44 * r20;
    r37 = r37 * r28;
    r37 = r37 * r47;
    r37 = r37 * r60;
    r88 = fma(r70, r37, r88);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r68, r88);
    r37 = r16 * r14;
    r38 = fma(r48, r38, r54 * r37);
    r38 = fma(r18, r56, r38);
    r38 = fma(r54, r21, r38);
    r80 = r38 * r80;
    r53 = r66 * r53;
    r21 = r80 + r53;
    r54 = r36 * r4;
    r54 = r54 * r38;
    r85 = r85 + r54;
    r56 = r27 * r23;
    r85 = fma(r67, r56, r85);
    r48 = r27 * r26;
    r85 = fma(r52, r48, r85);
    r85 = fma(r8, r85, r32 * r21);
    r21 = r36 * r45;
    r21 = fma(r38, r31, r67 * r21);
    r21 = r21 + r74;
    r85 = fma(r9, r21, r85);
    r72 = r77 + r72;
    r77 = r36 * r23;
    r77 = r77 * r38;
    r21 = r4 * r27;
    r72 = fma(r67, r21, r72);
    r72 = r72 + r77;
    r62 = r4 * r62;
    r62 = r62 * r66;
    r53 = r62 + r53;
    r53 = fma(r8, r53, r9 * r72);
    r31 = fma(r52, r31, r54);
    r31 = r31 + r86;
    r53 = fma(r32, r31, r53);
    r31 = fma(r53, r24, r85 * r63);
    r86 = r41 * r30;
    r52 = r36 * r28;
    r77 = r75 + r77;
    r77 = r77 + r79;
    r79 = r45 * r27;
    r75 = r27 * r26;
    r75 = fma(r38, r75, r67 * r79);
    r75 = r75 + r74;
    r75 = fma(r32, r75, r8 * r77);
    r80 = r62 + r80;
    r75 = fma(r9, r80, r75);
    r52 = r52 * r75;
    r52 = fma(r85, r10, r70 * r52);
    r80 = r36 * r53;
    r52 = fma(r73, r80, r52);
    r62 = r85 * r69;
    r52 = fma(r5, r62, r52);
    r86 = r86 * r52;
    r31 = fma(r34, r86, r31);
    r86 = r41 * r28;
    r86 = r86 * r52;
    r52 = r44 * r20;
    r52 = r52 * r28;
    r52 = r52 * r47;
    r52 = r52 * r85;
    r52 = fma(r70, r52, r34 * r86);
    r52 = fma(r75, r24, r52);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r52);
    r75 = r41 * r30;
    r86 = r36 * r83;
    r86 = r86 * r28;
    r62 = r29 * r69;
    r62 = fma(r5, r62, r70 * r86);
    r86 = r36 * r50;
    r62 = fma(r73, r86, r62);
    r62 = fma(r29, r10, r62);
    r75 = r75 * r62;
    r75 = fma(r34, r75, r29 * r63);
    r75 = fma(r50, r24, r75);
    r86 = r41 * r28;
    r86 = r86 * r62;
    r86 = fma(r34, r86, r83 * r24);
    r62 = r44 * r20;
    r62 = r62 * r29;
    r62 = r62 * r28;
    r62 = r62 * r47;
    r86 = fma(r70, r62, r86);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r75, r86);
    r62 = fma(r71, r63, r43 * r24);
    r80 = r41 * r30;
    r77 = r71 * r69;
    r74 = r36 * r51;
    r74 = r74 * r28;
    r74 = fma(r70, r74, r5 * r77);
    r77 = r36 * r43;
    r74 = fma(r73, r77, r74);
    r74 = fma(r71, r10, r74);
    r80 = r80 * r74;
    r62 = fma(r34, r80, r62);
    r80 = r41 * r28;
    r80 = r80 * r74;
    r80 = fma(r51, r24, r34 * r80);
    r74 = r44 * r20;
    r74 = r74 * r71;
    r74 = r74 * r28;
    r74 = r74 * r47;
    r80 = fma(r70, r74, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r62, r80);
    r74 = r41 * r30;
    r77 = r36 * r42;
    r77 = r77 * r28;
    r79 = r49 * r69;
    r79 = fma(r5, r79, r70 * r77);
    r77 = r36 * r46;
    r79 = fma(r73, r77, r79);
    r79 = fma(r49, r10, r79);
    r74 = r74 * r79;
    r74 = fma(r46, r24, r34 * r74);
    r74 = fma(r49, r63, r74);
    r63 = r44 * r20;
    r63 = r63 * r49;
    r63 = r63 * r28;
    r63 = r63 * r47;
    r47 = r41 * r28;
    r47 = r47 * r79;
    r47 = fma(r34, r47, r70 * r63);
    r47 = fma(r42, r24, r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r74, r47);
    r63 = r20 * r55;
    r34 = fma(r3, r20, r1);
    r34 = fma(r28, r24, r34);
    r70 = r20 * r82;
    r79 = fma(r2, r20, r0);
    r79 = fma(r30, r24, r79);
    r70 = fma(r79, r70, r34 * r63);
    r63 = r20 * r68;
    r24 = r20 * r88;
    r24 = fma(r34, r24, r79 * r63);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r20 * r52;
    r70 = r20 * r31;
    r70 = fma(r79, r70, r34 * r24);
    r24 = r20 * r75;
    r63 = r20 * r86;
    r63 = fma(r34, r63, r79 * r24);
    WriteSum2<double, double>((double *)inout_shared, r70, r63);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r20 * r80;
    r70 = r20 * r62;
    r70 = fma(r79, r70, r34 * r63);
    r63 = r20 * r74;
    r24 = r20 * r47;
    r24 = fma(r34, r24, r79 * r63);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r82, r82, r55 * r55);
    r70 = fma(r88, r88, r68 * r68);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r31, r31, r52 * r52);
    r24 = fma(r75, r75, r86 * r86);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r62, r62, r80 * r80);
    r70 = fma(r74, r74, r47 * r47);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r82, r68, r55 * r88);
    r24 = fma(r55, r52, r82 * r31);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r55, r86, r82 * r75);
    r70 = fma(r82, r62, r55 * r80);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r82, r74, r55 * r47);
    r24 = fma(r88, r52, r68 * r31);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r88, r86, r68 * r75);
    r70 = fma(r88, r80, r68 * r62);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r88, r47, r68 * r74);
    r24 = fma(r31, r75, r52 * r86);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r31, r62, r52 * r80);
    r70 = fma(r52, r47, r31 * r74);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r86, r80, r75 * r62);
    r24 = fma(r86, r47, r75 * r74);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r80, r47, r62 * r74);
    WriteSum1<double, double>((double *)inout_shared, r24);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = -2.00000000000000000e+00;
    r70 = fma(r16, r17, r11 * r14);
    r63 = r15 * r18;
    r34 = -1.00000000000000000e+00;
    r70 = fma(r34, r63, r70);
    r70 = fma(r12, r13, r70);
    r63 = r70 * r70;
    r63 = r24 * r63;
    r79 = 1.00000000000000000e+00;
    r10 = r11 * r17;
    r10 = fma(r34, r10, r16 * r14);
    r10 = fma(r12, r18, r10);
    r10 = fma(r15, r13, r10);
    r77 = r10 * r10;
    r77 = fma(r24, r77, r79);
    r73 = r63 + r77;
    r73 = fma(r8, r73, r6);
    r5 = 2.00000000000000000e+00;
    r38 = fma(r12, r17, r15 * r14);
    r67 = r16 * r13;
    r38 = fma(r34, r67, r38);
    r38 = fma(r11, r18, r38);
    r67 = r5 * r38;
    r54 = r10 * r67;
    r72 = fma(r16, r18, r15 * r17);
    r72 = fma(r11, r13, r72);
    r72 = fma(r34, r72, r12 * r14);
    r66 = r24 * r72;
    r21 = fma(r70, r66, r54);
    r48 = r5 * r10;
    r56 = r70 * r67;
    r48 = fma(r72, r48, r56);
    r37 = r17 * r13;
    r37 = r37 * r5;
    r87 = r18 * r14;
    r59 = fma(r5, r87, r37);
    r78 = r13 * r14;
    r57 = r17 * r18;
    r57 = r57 * r5;
    r78 = fma(r24, r78, r57);
    r61 = r18 * r18;
    r61 = r61 * r24;
    r58 = r79 + r61;
    r76 = r13 * r13;
    r76 = r24 * r76;
    r58 = r58 + r76;
    r73 = fma(r9, r21, r73);
    r73 = fma(r32, r48, r73);
    r73 = fma(r35, r59, r73);
    r73 = fma(r40, r78, r73);
    r73 = fma(r39, r58, r73);
    r58 = 1.00000000000000008e-15;
    r56 = fma(r10, r66, r56);
    r56 = fma(r8, r56, r33);
    r87 = fma(r24, r87, r37);
    r61 = r79 + r61;
    r37 = r17 * r17;
    r37 = r24 * r37;
    r61 = r61 + r37;
    r78 = r18 * r13;
    r78 = r78 * r5;
    r59 = r17 * r14;
    r59 = fma(r5, r59, r78);
    r48 = r5 * r70;
    r48 = r48 * r10;
    r67 = fma(r72, r67, r48);
    r21 = r38 * r38;
    r21 = r21 * r24;
    r77 = r21 + r77;
    r56 = fma(r39, r87, r56);
    r56 = fma(r35, r61, r56);
    r56 = fma(r40, r59, r56);
    r56 = fma(r9, r67, r56);
    r56 = fma(r32, r77, r56);
    r77 = copysign(1.0, r56);
    r77 = fma(r58, r77, r56);
    r58 = r77 * r77;
    r58 = 1.0 / r58;
    r56 = r73 * r73;
    r67 = r5 * r70;
    r67 = fma(r72, r67, r54);
    r67 = fma(r8, r67, r7);
    r54 = r13 * r14;
    r54 = fma(r5, r54, r57);
    r76 = r79 + r76;
    r76 = r76 + r37;
    r37 = r17 * r14;
    r37 = fma(r24, r37, r78);
    r66 = fma(r38, r66, r48);
    r63 = r79 + r63;
    r63 = r63 + r21;
    r67 = fma(r39, r54, r67);
    r67 = fma(r40, r76, r67);
    r67 = fma(r35, r37, r67);
    r67 = fma(r32, r66, r67);
    r67 = fma(r9, r63, r67);
    r63 = r67 * r67;
    r66 = fma(r58, r63, r58 * r56);
    r79 = fma(r41, r66, r79);
    r77 = 1.0 / r77;
    r37 = r79 * r77;
    r76 = r73 * r37;
    r54 = r67 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r76, r54);
    r54 = r73 * r77;
    r76 = r44 * r66;
    r54 = r54 * r76;
    r21 = r67 * r77;
    r21 = r21 * r76;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r54, r21);
    r21 = r34 * r73;
    r54 = fma(r2, r34, r0);
    r38 = r44 * r73;
    r54 = fma(r37, r38, r54);
    r21 = r21 * r54;
    r54 = r34 * r67;
    r38 = fma(r3, r34, r1);
    r48 = r44 * r67;
    r38 = fma(r37, r48, r38);
    r54 = r54 * r38;
    r54 = fma(r37, r54, r37 * r21);
    r37 = r77 * r76;
    r48 = r34 * r67;
    r48 = r48 * r38;
    r48 = r48 * r77;
    r48 = fma(r76, r48, r21 * r37);
    WriteSum2<double, double>((double *)inout_shared, r54, r48);
  };
  FlushSumShared<2, double>(
      out_focal_and_extra_njtr, 0 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r79 * r79;
    r48 = r48 * r58;
    r48 = fma(r63, r48, r56 * r48);
    r66 = r44 * r66;
    r58 = r58 * r76;
    r66 = r66 * r58;
    r66 = fma(r63, r66, r56 * r66);
    WriteSum2<double, double>((double *)inout_shared, r48, r66);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r79 * r56;
    r48 = r79 * r63;
    r48 = fma(r58, r48, r58 * r66);
    WriteSum1<double, double>((double *)inout_shared, r48);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r12, r17, r15 * r14);
    r66 = r16 * r13;
    r58 = -1.00000000000000000e+00;
    r48 = fma(r58, r66, r48);
    r48 = fma(r11, r18, r48);
    r66 = 2.00000000000000000e+00;
    r54 = fma(r16, r17, r11 * r14);
    r37 = r15 * r18;
    r54 = fma(r58, r37, r54);
    r54 = fma(r12, r13, r54);
    r37 = r66 * r54;
    r21 = r48 * r37;
    r38 = -2.00000000000000000e+00;
    r78 = r11 * r17;
    r78 = fma(r58, r78, r16 * r14);
    r78 = fma(r12, r18, r78);
    r78 = fma(r15, r13, r78);
    r24 = fma(r16, r18, r15 * r17);
    r24 = fma(r11, r13, r24);
    r24 = fma(r58, r24, r12 * r14);
    r57 = r78 * r24;
    r72 = fma(r38, r57, r21);
    r59 = 1.00000000000000000e+00;
    r61 = r54 * r54;
    r61 = r61 * r38;
    r87 = r38 * r78;
    r87 = fma(r78, r87, r59);
    r65 = r61 + r87;
    r6 = fma(r8, r65, r6);
    r89 = r66 * r48;
    r89 = r89 * r78;
    r90 = r54 * r38;
    r90 = fma(r24, r90, r89);
    r57 = fma(r66, r57, r21);
    r21 = r17 * r13;
    r21 = r21 * r66;
    r91 = r18 * r14;
    r92 = fma(r66, r91, r21);
    r93 = r13 * r14;
    r94 = r17 * r18;
    r94 = r94 * r66;
    r93 = fma(r38, r93, r94);
    r95 = r18 * r18;
    r95 = r95 * r38;
    r96 = r59 + r95;
    r97 = r13 * r13;
    r97 = r38 * r97;
    r96 = r96 + r97;
    r6 = fma(r9, r90, r6);
    r6 = fma(r32, r57, r6);
    r6 = fma(r35, r92, r6);
    r6 = fma(r40, r93, r6);
    r6 = fma(r39, r96, r6);
    r96 = 1.00000000000000008e-15;
    r33 = fma(r8, r72, r33);
    r91 = fma(r38, r91, r21);
    r95 = r59 + r95;
    r21 = r17 * r17;
    r21 = r38 * r21;
    r95 = r95 + r21;
    r93 = r18 * r13;
    r93 = r93 * r66;
    r92 = r17 * r14;
    r92 = fma(r66, r92, r93);
    r98 = r66 * r48;
    r99 = r78 * r37;
    r98 = fma(r24, r98, r99);
    r100 = r48 * r48;
    r100 = r38 * r100;
    r87 = r100 + r87;
    r33 = fma(r39, r91, r33);
    r33 = fma(r35, r95, r33);
    r33 = fma(r40, r92, r33);
    r33 = fma(r9, r98, r33);
    r33 = fma(r32, r87, r33);
    r92 = copysign(1.0, r33);
    r92 = fma(r96, r92, r33);
    r96 = r92 * r92;
    r33 = 1.0 / r96;
    r95 = r6 * r33;
    r37 = fma(r24, r37, r89);
    r8 = fma(r8, r37, r7);
    r7 = r13 * r14;
    r7 = fma(r66, r7, r94);
    r97 = r59 + r97;
    r97 = r97 + r21;
    r21 = r17 * r14;
    r21 = fma(r38, r21, r93);
    r93 = r48 * r38;
    r93 = fma(r24, r93, r99);
    r61 = r59 + r61;
    r61 = r61 + r100;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r97, r8);
    r8 = fma(r35, r21, r8);
    r8 = fma(r32, r93, r8);
    r8 = fma(r9, r61, r8);
    r9 = r8 * r8;
    r32 = fma(r33, r9, r6 * r95);
    r32 = fma(r41, r32, r59);
    r32 = r44 * r32;
    r59 = r58 * r32;
    r59 = r59 * r95;
    r21 = 1.0 / r92;
    r35 = r21 * r32;
    r97 = fma(r65, r35, r72 * r59);
    r40 = r44 * r41;
    r7 = r66 * r65;
    r96 = r92 * r96;
    r96 = 1.0 / r96;
    r96 = r38 * r96;
    r92 = r72 * r96;
    r92 = fma(r9, r92, r95 * r7);
    r7 = r6 * r6;
    r7 = r7 * r96;
    r39 = r66 * r37;
    r39 = r39 * r8;
    r92 = fma(r33, r39, r92);
    r92 = fma(r72, r7, r92);
    r40 = r40 * r92;
    r40 = r40 * r21;
    r97 = fma(r6, r40, r97);
    r92 = r58 * r72;
    r92 = r92 * r8;
    r92 = r92 * r33;
    r92 = fma(r37, r35, r32 * r92);
    r92 = fma(r8, r40, r92);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r97, r92);
    r40 = fma(r90, r35, r98 * r59);
    r39 = r44 * r41;
    r100 = r66 * r90;
    r100 = fma(r95, r100, r98 * r7);
    r99 = r66 * r61;
    r99 = r99 * r8;
    r100 = fma(r33, r99, r100);
    r24 = r98 * r96;
    r100 = fma(r9, r24, r100);
    r39 = r39 * r6;
    r39 = r39 * r100;
    r40 = fma(r21, r39, r40);
    r39 = r58 * r98;
    r39 = r39 * r8;
    r39 = r39 * r33;
    r39 = fma(r32, r39, r61 * r35);
    r24 = r44 * r41;
    r24 = r24 * r8;
    r24 = r24 * r100;
    r39 = fma(r21, r24, r39);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r40, r39);
    r24 = r44 * r41;
    r100 = r66 * r57;
    r99 = r87 * r96;
    r99 = fma(r9, r99, r95 * r100);
    r100 = r66 * r93;
    r100 = r100 * r8;
    r99 = fma(r33, r100, r99);
    r99 = fma(r87, r7, r99);
    r24 = r24 * r6;
    r24 = r24 * r99;
    r24 = fma(r21, r24, r87 * r59);
    r24 = fma(r57, r35, r24);
    r59 = r44 * r41;
    r59 = r59 * r8;
    r59 = r59 * r99;
    r99 = r58 * r87;
    r99 = r99 * r8;
    r99 = r99 * r33;
    r99 = fma(r32, r99, r21 * r59);
    r99 = fma(r93, r35, r99);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r24, r99);
    r59 = r58 * r97;
    r2 = fma(r2, r58, r0);
    r2 = fma(r6, r35, r2);
    r6 = r58 * r92;
    r3 = fma(r3, r58, r1);
    r3 = fma(r8, r35, r3);
    r6 = fma(r3, r6, r2 * r59);
    r59 = r58 * r39;
    r35 = r58 * r40;
    r35 = fma(r2, r35, r3 * r59);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r58 * r99;
    r6 = r58 * r24;
    r6 = fma(r2, r6, r3 * r35);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r92, r92, r97 * r97);
    r35 = fma(r39, r39, r40 * r40);
    WriteSum2<double, double>((double *)inout_shared, r6, r35);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r99, r99, r24 * r24);
    WriteSum1<double, double>((double *)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r92, r39, r97 * r40);
    r6 = fma(r97, r24, r92 * r99);
    WriteSum2<double, double>((double *)inout_shared, r35, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r40, r24, r39 * r99);
    WriteSum1<double, double>((double *)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double *out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double *const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double *const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal_and_extra, focal_and_extra_num_alloc,
      focal_and_extra_indices, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, principal_point, principal_point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar