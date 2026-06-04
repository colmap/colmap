#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        size_t problem_size) {
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83;

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
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r5);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    r16 = fma(r13, r14, r8 * r11);
    r17 = r12 * r15;
    r16 = fma(r4, r17, r16);
    r16 = fma(r9, r10, r16);
    r17 = r16 * r16;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = 1.00000000000000000e+00;
    r20 = r9 * r15;
    r21 = fma(r13, r11, r20);
    r22 = r12 * r10;
    r23 = r8 * r14;
    r21 = r21 + r22;
    r21 = fma(r4, r23, r21);
    r24 = r18 * r21;
    r24 = fma(r21, r24, r19);
    r25 = r17 + r24;
    r25 = fma(r6, r25, r0);
    r0 = 2.00000000000000000e+00;
    r26 = fma(r9, r14, r12 * r11);
    r27 = r13 * r10;
    r26 = fma(r4, r27, r26);
    r26 = fma(r8, r15, r26);
    r27 = r0 * r26;
    r27 = r27 * r21;
    r28 = r16 * r18;
    r29 = fma(r13, r15, r12 * r14);
    r29 = fma(r8, r10, r29);
    r29 = fma(r4, r29, r9 * r11);
    r28 = fma(r29, r28, r27);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r30);
    r31 = r0 * r16;
    r31 = r31 * r26;
    r32 = r0 * r29;
    r33 = fma(r21, r32, r31);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r14 * r10;
    r35 = r35 * r0;
    r36 = r15 * r11;
    r36 = fma(r0, r36, r35);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r37, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r10 * r11;
    r40 = r14 * r15;
    r40 = r40 * r0;
    r39 = fma(r18, r39, r40);
    r41 = r15 * r15;
    r41 = r41 * r18;
    r42 = r19 + r41;
    r43 = r10 * r10;
    r43 = r43 * r18;
    r42 = r42 + r43;
    r25 = fma(r7, r28, r25);
    r25 = fma(r30, r33, r25);
    r25 = fma(r34, r36, r25);
    r25 = fma(r38, r39, r25);
    r25 = fma(r37, r42, r25);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r33,
                        r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r45);
    r46 = r18 * r21;
    r46 = fma(r29, r46, r31);
    r46 = fma(r6, r46, r45);
    r45 = r15 * r11;
    r45 = fma(r18, r45, r35);
    r41 = r19 + r41;
    r35 = r14 * r14;
    r35 = r35 * r18;
    r41 = r41 + r35;
    r31 = r15 * r10;
    r31 = r31 * r0;
    r47 = r14 * r11;
    r47 = fma(r0, r47, r31);
    r48 = r0 * r16;
    r48 = r48 * r21;
    r49 = fma(r26, r32, r48);
    r50 = r26 * r26;
    r50 = r50 * r18;
    r24 = r50 + r24;
    r46 = fma(r37, r45, r46);
    r46 = fma(r34, r41, r46);
    r46 = fma(r38, r47, r46);
    r46 = fma(r7, r49, r46);
    r46 = fma(r30, r24, r46);
    r24 = copysign(1.0, r46);
    r24 = fma(r44, r24, r46);
    r44 = r24 * r24;
    r46 = 1.0 / r44;
    r49 = r25 * r46;
    r27 = fma(r16, r32, r27);
    r27 = fma(r6, r27, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r35;
    r35 = r14 * r11;
    r35 = fma(r18, r35, r31);
    r31 = r26 * r18;
    r31 = fma(r29, r31, r48);
    r17 = r19 + r17;
    r17 = r17 + r50;
    r27 = fma(r37, r5, r27);
    r27 = fma(r38, r43, r27);
    r27 = fma(r34, r35, r27);
    r27 = fma(r30, r31, r27);
    r27 = fma(r7, r17, r27);
    r17 = r27 * r27;
    r31 = fma(r46, r17, r25 * r49);
    r19 = fma(r28, r31, r19);
    r34 = r25 * r19;
    r38 = 1.0 / r24;
    r37 = r33 * r38;
    r2 = fma(r37, r34, r2);
    r3 = fma(r3, r4, r1);
    r1 = r27 * r19;
    r3 = fma(r37, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r21;
    r34 = -5.00000000000000000e-01;
    r50 = r13 * r34;
    r48 = 5.00000000000000000e-01;
    r40 = fma(r48, r23, r11 * r50);
    r40 = fma(r34, r20, r40);
    r40 = fma(r34, r22, r40);
    r1 = r1 * r40;
    r51 = r0 * r16;
    r52 = r13 * r14;
    r53 = r12 * r15;
    r53 = fma(r34, r53, r48 * r52);
    r52 = r9 * r10;
    r53 = fma(r48, r52, r53);
    r54 = r11 * r48;
    r53 = fma(r8, r54, r53);
    r51 = fma(r53, r51, r1);
    r52 = r0 * r26;
    r55 = r12 * r11;
    r56 = r9 * r14;
    r56 = fma(r34, r56, r34 * r55);
    r55 = r8 * r15;
    r56 = fma(r34, r55, r56);
    r57 = r13 * r10;
    r56 = fma(r48, r57, r56);
    r52 = r52 * r56;
    r57 = r12 * r14;
    r55 = r8 * r10;
    r55 = fma(r34, r55, r34 * r57);
    r55 = fma(r9, r54, r55);
    r55 = fma(r15, r50, r55);
    r57 = r55 * r32;
    r58 = r52 + r57;
    r59 = r51 + r58;
    r60 = r18 * r29;
    r61 = r21 * r56;
    r60 = fma(r18, r61, r53 * r60);
    r62 = r0 * r26;
    r63 = r0 * r16;
    r63 = r63 * r55;
    r62 = fma(r40, r62, r63);
    r60 = r60 + r62;
    r60 = fma(r6, r60, r7 * r59);
    r59 = r21 * r53;
    r64 = -4.00000000000000000e+00;
    r59 = r59 * r64;
    r65 = r26 * r64;
    r66 = r55 * r65;
    r67 = r59 + r66;
    r60 = fma(r30, r67, r60);
    r67 = r60 * r49;
    r68 = r4 * r19;
    r69 = r33 * r68;
    r70 = fma(r53, r32, r0 * r61);
    r70 = r70 + r62;
    r71 = r0 * r21;
    r71 = r71 * r55;
    r72 = r0 * r26;
    r72 = r72 * r53;
    r53 = r71 + r72;
    r73 = r16 * r18;
    r53 = fma(r56, r73, r53);
    r74 = r18 * r29;
    r53 = fma(r40, r74, r53);
    r53 = fma(r7, r53, r30 * r70);
    r70 = r16 * r40;
    r74 = r64 * r70;
    r59 = r59 + r74;
    r53 = fma(r6, r59, r53);
    r59 = r19 * r53;
    r59 = fma(r37, r59, r69 * r67);
    r67 = r0 * r53;
    r73 = r25 * r25;
    r44 = r24 * r44;
    r44 = 1.0 / r44;
    r44 = r18 * r44;
    r73 = r73 * r44;
    r67 = fma(r60, r73, r49 * r67);
    r24 = r60 * r44;
    r67 = fma(r17, r24, r67);
    r75 = r0 * r27;
    r76 = r26 * r18;
    r77 = r18 * r29;
    r77 = r77 * r55;
    r76 = fma(r56, r76, r77);
    r76 = r76 + r51;
    r74 = r66 + r74;
    r74 = fma(r7, r74, r30 * r76);
    r72 = fma(r40, r32, r72);
    r76 = r0 * r16;
    r76 = fma(r56, r76, r71);
    r72 = r72 + r76;
    r74 = fma(r6, r72, r74);
    r75 = r75 * r74;
    r67 = fma(r46, r75, r67);
    r28 = r28 * r37;
    r67 = r67 * r28;
    r59 = fma(r25, r67, r59);
    r75 = r19 * r74;
    r75 = fma(r37, r75, r27 * r67);
    r67 = r27 * r46;
    r67 = r67 * r69;
    r75 = fma(r60, r67, r75);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r59, r75);
    r57 = r1 + r57;
    r1 = r0 * r16;
    r24 = r8 * r11;
    r72 = r12 * r15;
    r72 = fma(r48, r72, r34 * r24);
    r24 = r9 * r10;
    r72 = fma(r34, r24, r72);
    r72 = fma(r14, r50, r72);
    r1 = r1 * r72;
    r24 = r0 * r26;
    r71 = r9 * r14;
    r66 = r8 * r15;
    r66 = fma(r48, r66, r48 * r71);
    r66 = fma(r12, r54, r66);
    r66 = fma(r10, r50, r66);
    r24 = fma(r66, r24, r1);
    r57 = r57 + r24;
    r50 = r21 * r55;
    r50 = r50 * r64;
    r71 = r16 * r64;
    r71 = r71 * r66;
    r51 = r50 + r71;
    r51 = fma(r6, r51, r30 * r57);
    r57 = r18 * r29;
    r57 = fma(r18, r70, r66 * r57);
    r78 = r0 * r26;
    r78 = r78 * r55;
    r79 = r0 * r21;
    r79 = fma(r72, r79, r78);
    r57 = r57 + r79;
    r51 = fma(r7, r57, r51);
    r57 = r19 * r51;
    r80 = r0 * r51;
    r81 = r0 * r27;
    r82 = r26 * r18;
    r82 = fma(r40, r82, r63);
    r63 = r0 * r21;
    r63 = r63 * r66;
    r83 = r18 * r29;
    r82 = fma(r72, r83, r82);
    r82 = r82 + r63;
    r66 = fma(r66, r32, r0 * r70);
    r66 = r66 + r79;
    r66 = fma(r6, r66, r30 * r82);
    r82 = r72 * r65;
    r71 = r71 + r82;
    r66 = fma(r7, r71, r66);
    r81 = r81 * r66;
    r81 = fma(r46, r81, r49 * r80);
    r80 = r18 * r21;
    r80 = fma(r40, r80, r77);
    r80 = r80 + r24;
    r63 = fma(r72, r32, r63);
    r63 = r63 + r62;
    r63 = fma(r7, r63, r6 * r80);
    r82 = r50 + r82;
    r63 = fma(r30, r82, r63);
    r82 = r63 * r44;
    r81 = fma(r17, r82, r81);
    r81 = fma(r63, r73, r81);
    r82 = r25 * r81;
    r82 = fma(r28, r82, r37 * r57);
    r57 = r63 * r49;
    r82 = fma(r69, r57, r82);
    r57 = r19 * r66;
    r50 = r27 * r81;
    r50 = fma(r28, r50, r37 * r57);
    r50 = fma(r63, r67, r50);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r82, r50);
    r57 = r0 * r27;
    r80 = r0 * r21;
    r23 = fma(r34, r23, r13 * r54);
    r23 = fma(r48, r20, r23);
    r23 = fma(r48, r22, r23);
    r80 = r80 * r23;
    r1 = r1 + r80;
    r1 = r1 + r58;
    r58 = r26 * r18;
    r22 = r18 * r29;
    r22 = fma(r23, r22, r72 * r58);
    r22 = r22 + r76;
    r22 = fma(r30, r22, r6 * r1);
    r55 = r16 * r55;
    r55 = r55 * r64;
    r65 = r23 * r65;
    r1 = r55 + r65;
    r22 = fma(r7, r1, r22);
    r57 = r57 * r22;
    r61 = r64 * r61;
    r65 = r65 + r61;
    r64 = r0 * r16;
    r64 = r64 * r23;
    r78 = r78 + r64;
    r1 = r18 * r21;
    r78 = fma(r72, r1, r78);
    r58 = r18 * r29;
    r78 = fma(r56, r58, r78);
    r78 = fma(r6, r78, r30 * r65);
    r65 = r0 * r26;
    r23 = fma(r23, r32, r72 * r65);
    r23 = r23 + r76;
    r78 = fma(r7, r23, r78);
    r57 = fma(r78, r73, r46 * r57);
    r77 = r52 + r77;
    r52 = r16 * r18;
    r77 = fma(r72, r52, r77);
    r77 = r77 + r80;
    r61 = r55 + r61;
    r61 = fma(r6, r61, r7 * r77);
    r32 = fma(r56, r32, r64);
    r32 = r32 + r79;
    r61 = fma(r30, r32, r61);
    r32 = r0 * r61;
    r57 = fma(r49, r32, r57);
    r30 = r78 * r44;
    r57 = fma(r17, r30, r57);
    r30 = r25 * r57;
    r32 = r19 * r61;
    r32 = fma(r37, r32, r28 * r30);
    r30 = r78 * r49;
    r32 = fma(r69, r30, r32);
    r30 = r27 * r57;
    r30 = fma(r78, r67, r28 * r30);
    r79 = r19 * r22;
    r30 = fma(r37, r79, r30);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r30);
    r79 = r42 * r19;
    r56 = r0 * r5;
    r56 = r56 * r27;
    r64 = r45 * r44;
    r64 = fma(r17, r64, r46 * r56);
    r56 = r0 * r42;
    r64 = fma(r49, r56, r64);
    r64 = fma(r45, r73, r64);
    r56 = r25 * r64;
    r56 = fma(r28, r56, r37 * r79);
    r79 = r45 * r49;
    r56 = fma(r69, r79, r56);
    r79 = r5 * r19;
    r79 = fma(r37, r79, r45 * r67);
    r6 = r27 * r64;
    r79 = fma(r28, r6, r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r56, r79);
    r6 = r47 * r49;
    r77 = r39 * r19;
    r77 = fma(r37, r77, r69 * r6);
    r6 = r47 * r44;
    r7 = r0 * r43;
    r7 = r7 * r27;
    r7 = fma(r46, r7, r17 * r6);
    r6 = r0 * r39;
    r7 = fma(r49, r6, r7);
    r7 = fma(r47, r73, r7);
    r6 = r25 * r7;
    r77 = fma(r28, r6, r77);
    r6 = r43 * r19;
    r6 = fma(r37, r6, r47 * r67);
    r55 = r27 * r7;
    r6 = fma(r28, r55, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r77, r6);
    r55 = r0 * r35;
    r55 = r55 * r27;
    r52 = r41 * r44;
    r52 = fma(r17, r52, r46 * r55);
    r55 = r0 * r36;
    r52 = fma(r49, r55, r52);
    r52 = fma(r41, r73, r52);
    r73 = r25 * r52;
    r55 = r36 * r19;
    r55 = fma(r37, r55, r28 * r73);
    r73 = r41 * r49;
    r55 = fma(r69, r73, r55);
    r73 = r35 * r19;
    r69 = r27 * r52;
    r69 = fma(r28, r69, r37 * r73);
    r69 = fma(r41, r67, r69);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r55, r69);
    r67 = r4 * r2;
    r73 = r4 * r3;
    r73 = fma(r75, r73, r59 * r67);
    r67 = r4 * r2;
    r28 = r4 * r3;
    r28 = fma(r50, r28, r82 * r67);
    WriteSum2<double, double>((double*)inout_shared, r73, r28);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r4 * r3;
    r73 = r4 * r2;
    r73 = fma(r32, r73, r30 * r28);
    r28 = r4 * r3;
    r67 = r4 * r2;
    r67 = fma(r56, r67, r79 * r28);
    WriteSum2<double, double>((double*)inout_shared, r73, r67);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r4 * r2;
    r73 = r4 * r3;
    r73 = fma(r6, r73, r77 * r67);
    r67 = r4 * r2;
    r28 = r4 * r3;
    r28 = fma(r69, r28, r55 * r67);
    WriteSum2<double, double>((double*)inout_shared, r73, r28);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r59, r59, r75 * r75);
    r73 = fma(r82, r82, r50 * r50);
    WriteSum2<double, double>((double*)inout_shared, r28, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r32, r32, r30 * r30);
    r28 = fma(r56, r56, r79 * r79);
    WriteSum2<double, double>((double*)inout_shared, r73, r28);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r6, r6, r77 * r77);
    r73 = fma(r55, r55, r69 * r69);
    WriteSum2<double, double>((double*)inout_shared, r28, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r59, r82, r75 * r50);
    r28 = fma(r59, r32, r75 * r30);
    WriteSum2<double, double>((double*)inout_shared, r73, r28);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r59, r56, r75 * r79);
    r73 = fma(r75, r6, r59 * r77);
    WriteSum2<double, double>((double*)inout_shared, r28, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r59, r55, r75 * r69);
    r75 = fma(r50, r30, r82 * r32);
    WriteSum2<double, double>((double*)inout_shared, r59, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r50, r79, r82 * r56);
    r59 = fma(r50, r6, r82 * r77);
    WriteSum2<double, double>((double*)inout_shared, r75, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r50, r69, r82 * r55);
    r82 = fma(r32, r56, r30 * r79);
    WriteSum2<double, double>((double*)inout_shared, r50, r82);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fma(r32, r77, r30 * r6);
    r30 = fma(r30, r69, r32 * r55);
    WriteSum2<double, double>((double*)inout_shared, r82, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r56, r77, r79 * r6);
    r79 = fma(r79, r69, r56 * r55);
    WriteSum2<double, double>((double*)inout_shared, r30, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r6, r69, r77 * r55);
    WriteSum1<double, double>((double*)inout_shared, r69);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r25 * r19;
    r69 = r69 * r38;
    r6 = r27 * r19;
    r6 = r6 * r38;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r69,
        r6);
    r6 = r25 * r31;
    r6 = r6 * r37;
    r69 = r27 * r31;
    r69 = r69 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r6,
        r69);
    r69 = r25 * r2;
    r69 = r69 * r38;
    r6 = r27 * r3;
    r6 = r6 * r38;
    r6 = fma(r68, r6, r68 * r69);
    r69 = r4 * r27;
    r69 = r69 * r31;
    r69 = r69 * r3;
    r68 = r4 * r25;
    r68 = r68 * r31;
    r68 = r68 * r2;
    r68 = fma(r37, r68, r37 * r69);
    WriteSum2<double, double>((double*)inout_shared, r6, r68);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r25 * r19;
    r68 = r68 * r19;
    r6 = r19 * r19;
    r6 = r6 * r46;
    r6 = fma(r17, r6, r49 * r68);
    r68 = r31 * r31;
    r69 = r33 * r68;
    r46 = r33 * r46;
    r46 = r46 * r17;
    r17 = r33 * r33;
    r17 = r17 * r25;
    r17 = r17 * r49;
    r17 = fma(r68, r17, r46 * r69);
    WriteSum2<double, double>((double*)inout_shared, r6, r17);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r33 * r25;
    r17 = r17 * r31;
    r17 = r17 * r19;
    r6 = r31 * r19;
    r6 = fma(r46, r6, r49 * r17);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks,
                                                                    1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar