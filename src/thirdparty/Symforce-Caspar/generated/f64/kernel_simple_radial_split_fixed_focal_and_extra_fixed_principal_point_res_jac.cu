#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88;

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
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
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
    r0 = fma(r6, r25, r0);
    r26 = 2.00000000000000000e+00;
    r27 = fma(r9, r14, r12 * r11);
    r28 = r13 * r10;
    r27 = fma(r4, r28, r27);
    r27 = fma(r8, r15, r27);
    r28 = r26 * r27;
    r28 = r28 * r21;
    r29 = r16 * r18;
    r30 = fma(r13, r15, r12 * r14);
    r30 = fma(r8, r10, r30);
    r30 = fma(r4, r30, r9 * r11);
    r29 = fma(r30, r29, r28);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r26 * r16;
    r32 = r32 * r27;
    r33 = r26 * r30;
    r34 = fma(r21, r33, r32);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r14 * r10;
    r36 = r36 * r26;
    r37 = r15 * r11;
    r37 = fma(r26, r37, r36);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r38, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r10 * r11;
    r41 = r14 * r15;
    r41 = r41 * r26;
    r40 = fma(r18, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r18;
    r43 = r19 + r42;
    r44 = r10 * r10;
    r44 = r44 * r18;
    r43 = r43 + r44;
    r0 = fma(r7, r29, r0);
    r0 = fma(r31, r34, r0);
    r0 = fma(r35, r37, r0);
    r0 = fma(r39, r40, r0);
    r0 = fma(r38, r43, r0);
    r45 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r46);
    r47 = r18 * r21;
    r47 = fma(r30, r47, r32);
    r46 = fma(r6, r47, r46);
    r32 = r15 * r11;
    r32 = fma(r18, r32, r36);
    r42 = r19 + r42;
    r36 = r14 * r14;
    r36 = r36 * r18;
    r42 = r42 + r36;
    r48 = r15 * r10;
    r48 = r48 * r26;
    r49 = r14 * r11;
    r49 = fma(r26, r49, r48);
    r50 = r26 * r16;
    r50 = r50 * r21;
    r51 = fma(r27, r33, r50);
    r52 = r27 * r27;
    r52 = r52 * r18;
    r24 = r52 + r24;
    r46 = fma(r38, r32, r46);
    r46 = fma(r35, r42, r46);
    r46 = fma(r39, r49, r46);
    r46 = fma(r7, r51, r46);
    r46 = fma(r31, r24, r46);
    r53 = copysign(1.0, r46);
    r53 = fma(r45, r53, r46);
    r45 = 1.0 / r53;
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r46,
                                            r54);
    r55 = r53 * r53;
    r56 = 1.0 / r55;
    r57 = r0 * r56;
    r28 = fma(r16, r33, r28);
    r5 = fma(r6, r28, r5);
    r58 = r10 * r11;
    r58 = fma(r26, r58, r41);
    r44 = r19 + r44;
    r44 = r44 + r36;
    r36 = r14 * r11;
    r36 = fma(r18, r36, r48);
    r48 = r27 * r18;
    r48 = fma(r30, r48, r50);
    r17 = r19 + r17;
    r17 = r17 + r52;
    r5 = fma(r38, r58, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r35, r36, r5);
    r5 = fma(r31, r48, r5);
    r5 = fma(r7, r17, r5);
    r35 = r5 * r5;
    r39 = fma(r56, r35, r0 * r57);
    r39 = fma(r54, r39, r19);
    r39 = r46 * r39;
    r19 = r45 * r39;
    r2 = fma(r0, r19, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r5, r19, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r26 * r21;
    r38 = -5.00000000000000000e-01;
    r52 = r13 * r38;
    r50 = 5.00000000000000000e-01;
    r41 = fma(r50, r23, r11 * r52);
    r41 = fma(r38, r20, r41);
    r41 = fma(r38, r22, r41);
    r1 = r1 * r41;
    r59 = r26 * r16;
    r60 = r13 * r14;
    r61 = r12 * r15;
    r61 = fma(r38, r61, r50 * r60);
    r60 = r9 * r10;
    r61 = fma(r50, r60, r61);
    r62 = r11 * r50;
    r61 = fma(r8, r62, r61);
    r59 = fma(r61, r59, r1);
    r60 = r26 * r27;
    r63 = r12 * r11;
    r64 = r9 * r14;
    r64 = fma(r38, r64, r38 * r63);
    r63 = r8 * r15;
    r64 = fma(r38, r63, r64);
    r65 = r13 * r10;
    r64 = fma(r50, r65, r64);
    r60 = r60 * r64;
    r65 = r12 * r14;
    r63 = r8 * r10;
    r63 = fma(r38, r63, r38 * r65);
    r63 = fma(r9, r62, r63);
    r63 = fma(r15, r52, r63);
    r65 = r63 * r33;
    r66 = r60 + r65;
    r67 = r59 + r66;
    r68 = r18 * r30;
    r69 = r21 * r64;
    r68 = fma(r18, r69, r61 * r68);
    r70 = r26 * r27;
    r71 = r26 * r16;
    r71 = r71 * r63;
    r70 = fma(r41, r70, r71);
    r68 = r68 + r70;
    r68 = fma(r6, r68, r7 * r67);
    r67 = r21 * r61;
    r72 = -4.00000000000000000e+00;
    r67 = r67 * r72;
    r73 = r27 * r72;
    r74 = r63 * r73;
    r75 = r67 + r74;
    r68 = fma(r31, r75, r68);
    r75 = r4 * r39;
    r75 = r75 * r57;
    r76 = fma(r61, r33, r26 * r69);
    r76 = r76 + r70;
    r77 = r26 * r21;
    r77 = r77 * r63;
    r78 = r26 * r27;
    r78 = r78 * r61;
    r61 = r77 + r78;
    r79 = r16 * r18;
    r61 = fma(r64, r79, r61);
    r80 = r18 * r30;
    r61 = fma(r41, r80, r61);
    r61 = fma(r7, r61, r31 * r76);
    r76 = r16 * r41;
    r80 = r72 * r76;
    r67 = r67 + r80;
    r61 = fma(r6, r67, r61);
    r67 = fma(r61, r19, r68 * r75);
    r79 = r46 * r54;
    r81 = r26 * r61;
    r82 = r0 * r0;
    r55 = r53 * r55;
    r55 = 1.0 / r55;
    r55 = r18 * r55;
    r82 = r82 * r55;
    r81 = fma(r68, r82, r57 * r81);
    r53 = r68 * r55;
    r81 = fma(r35, r53, r81);
    r83 = r26 * r5;
    r84 = r27 * r18;
    r85 = r18 * r30;
    r85 = r85 * r63;
    r84 = fma(r64, r84, r85);
    r84 = r84 + r59;
    r80 = r74 + r80;
    r80 = fma(r7, r80, r31 * r84);
    r78 = fma(r41, r33, r78);
    r84 = r26 * r16;
    r84 = fma(r64, r84, r77);
    r78 = r78 + r84;
    r80 = fma(r6, r78, r80);
    r83 = r83 * r80;
    r81 = fma(r56, r83, r81);
    r79 = r79 * r81;
    r79 = r79 * r45;
    r67 = fma(r0, r79, r67);
    r80 = fma(r80, r19, r5 * r79);
    r79 = r4 * r5;
    r79 = r79 * r68;
    r79 = r79 * r56;
    r80 = fma(r39, r79, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r67, r80);
    r65 = r1 + r65;
    r1 = r26 * r16;
    r79 = r8 * r11;
    r81 = r12 * r15;
    r81 = fma(r50, r81, r38 * r79);
    r79 = r9 * r10;
    r81 = fma(r38, r79, r81);
    r81 = fma(r14, r52, r81);
    r1 = r1 * r81;
    r79 = r26 * r27;
    r83 = r9 * r14;
    r53 = r8 * r15;
    r53 = fma(r50, r53, r50 * r83);
    r53 = fma(r12, r62, r53);
    r53 = fma(r10, r52, r53);
    r79 = fma(r53, r79, r1);
    r65 = r65 + r79;
    r52 = r21 * r63;
    r52 = r52 * r72;
    r83 = r16 * r72;
    r83 = r83 * r53;
    r78 = r52 + r83;
    r78 = fma(r6, r78, r31 * r65);
    r65 = r18 * r30;
    r65 = fma(r18, r76, r53 * r65);
    r77 = r26 * r27;
    r77 = r77 * r63;
    r74 = r26 * r21;
    r74 = fma(r81, r74, r77);
    r65 = r65 + r74;
    r78 = fma(r7, r65, r78);
    r65 = r46 * r54;
    r59 = r26 * r78;
    r86 = r26 * r5;
    r87 = r27 * r18;
    r87 = fma(r41, r87, r71);
    r71 = r26 * r21;
    r71 = r71 * r53;
    r88 = r18 * r30;
    r87 = fma(r81, r88, r87);
    r87 = r87 + r71;
    r53 = fma(r53, r33, r26 * r76);
    r53 = r53 + r74;
    r53 = fma(r6, r53, r31 * r87);
    r87 = r81 * r73;
    r83 = r83 + r87;
    r53 = fma(r7, r83, r53);
    r86 = r86 * r53;
    r86 = fma(r56, r86, r57 * r59);
    r59 = r18 * r21;
    r59 = fma(r41, r59, r85);
    r59 = r59 + r79;
    r71 = fma(r81, r33, r71);
    r71 = r71 + r70;
    r71 = fma(r7, r71, r6 * r59);
    r87 = r52 + r87;
    r71 = fma(r31, r87, r71);
    r87 = r71 * r55;
    r86 = fma(r35, r87, r86);
    r86 = fma(r71, r82, r86);
    r65 = r65 * r0;
    r65 = r65 * r86;
    r65 = fma(r45, r65, r78 * r19);
    r65 = fma(r71, r75, r65);
    r87 = r46 * r54;
    r87 = r87 * r5;
    r87 = r87 * r86;
    r87 = fma(r45, r87, r53 * r19);
    r53 = r4 * r5;
    r53 = r53 * r71;
    r53 = r53 * r56;
    r87 = fma(r39, r53, r87);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r65, r87);
    r53 = r46 * r54;
    r86 = r26 * r5;
    r52 = r26 * r21;
    r23 = fma(r38, r23, r13 * r62);
    r23 = fma(r50, r20, r23);
    r23 = fma(r50, r22, r23);
    r52 = r52 * r23;
    r1 = r1 + r52;
    r1 = r1 + r66;
    r66 = r27 * r18;
    r22 = r18 * r30;
    r22 = fma(r23, r22, r81 * r66);
    r22 = r22 + r84;
    r22 = fma(r31, r22, r6 * r1);
    r63 = r16 * r63;
    r63 = r63 * r72;
    r73 = r23 * r73;
    r1 = r63 + r73;
    r22 = fma(r7, r1, r22);
    r86 = r86 * r22;
    r69 = r72 * r69;
    r73 = r73 + r69;
    r72 = r26 * r16;
    r72 = r72 * r23;
    r77 = r77 + r72;
    r1 = r18 * r21;
    r77 = fma(r81, r1, r77);
    r66 = r18 * r30;
    r77 = fma(r64, r66, r77);
    r77 = fma(r6, r77, r31 * r73);
    r73 = r26 * r27;
    r23 = fma(r23, r33, r81 * r73);
    r23 = r23 + r84;
    r77 = fma(r7, r23, r77);
    r86 = fma(r77, r82, r56 * r86);
    r85 = r60 + r85;
    r60 = r16 * r18;
    r85 = fma(r81, r60, r85);
    r85 = r85 + r52;
    r69 = r63 + r69;
    r69 = fma(r6, r69, r7 * r85);
    r33 = fma(r64, r33, r72);
    r33 = r33 + r74;
    r69 = fma(r31, r33, r69);
    r33 = r26 * r69;
    r86 = fma(r57, r33, r86);
    r31 = r77 * r55;
    r86 = fma(r35, r31, r86);
    r53 = r53 * r0;
    r53 = r53 * r86;
    r53 = fma(r69, r19, r45 * r53);
    r53 = fma(r77, r75, r53);
    r31 = r46 * r54;
    r31 = r31 * r5;
    r31 = r31 * r86;
    r86 = r4 * r5;
    r86 = r86 * r77;
    r86 = r86 * r56;
    r86 = fma(r39, r86, r45 * r31);
    r86 = fma(r22, r19, r86);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r53, r86);
    r22 = r46 * r54;
    r31 = r26 * r58;
    r31 = r31 * r5;
    r33 = r32 * r55;
    r33 = fma(r35, r33, r56 * r31);
    r31 = r26 * r43;
    r33 = fma(r57, r31, r33);
    r33 = fma(r32, r82, r33);
    r22 = r22 * r0;
    r22 = r22 * r33;
    r22 = fma(r45, r22, r43 * r19);
    r22 = fma(r32, r75, r22);
    r31 = r4 * r32;
    r31 = r31 * r5;
    r31 = r31 * r56;
    r31 = fma(r58, r19, r39 * r31);
    r74 = r46 * r54;
    r74 = r74 * r5;
    r74 = r74 * r33;
    r31 = fma(r45, r74, r31);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r31);
    r74 = fma(r40, r19, r49 * r75);
    r33 = r46 * r54;
    r64 = r49 * r55;
    r72 = r26 * r44;
    r72 = r72 * r5;
    r72 = fma(r56, r72, r35 * r64);
    r64 = r26 * r40;
    r72 = fma(r57, r64, r72);
    r72 = fma(r49, r82, r72);
    r33 = r33 * r0;
    r33 = r33 * r72;
    r74 = fma(r45, r33, r74);
    r33 = r4 * r49;
    r33 = r33 * r5;
    r33 = r33 * r56;
    r33 = fma(r44, r19, r39 * r33);
    r64 = r46 * r54;
    r64 = r64 * r5;
    r64 = r64 * r72;
    r33 = fma(r45, r64, r33);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r74, r33);
    r64 = r46 * r54;
    r72 = r26 * r36;
    r72 = r72 * r5;
    r6 = r42 * r55;
    r6 = fma(r35, r6, r56 * r72);
    r72 = r26 * r37;
    r6 = fma(r57, r72, r6);
    r6 = fma(r42, r82, r6);
    r64 = r64 * r0;
    r64 = r64 * r6;
    r64 = fma(r37, r19, r45 * r64);
    r64 = fma(r42, r75, r64);
    r72 = r46 * r54;
    r72 = r72 * r5;
    r72 = r72 * r6;
    r72 = fma(r45, r72, r36 * r19);
    r6 = r4 * r42;
    r6 = r6 * r5;
    r6 = r6 * r56;
    r72 = fma(r39, r6, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r64, r72);
    r6 = r4 * r2;
    r85 = r4 * r3;
    r85 = fma(r80, r85, r67 * r6);
    r6 = r4 * r2;
    r7 = r4 * r3;
    r7 = fma(r87, r7, r65 * r6);
    WriteSum2<double, double>((double*)inout_shared, r85, r7);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r4 * r3;
    r85 = r4 * r2;
    r85 = fma(r53, r85, r86 * r7);
    r7 = r4 * r3;
    r6 = r4 * r2;
    r6 = fma(r22, r6, r31 * r7);
    WriteSum2<double, double>((double*)inout_shared, r85, r6);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r2;
    r85 = r4 * r3;
    r85 = fma(r33, r85, r74 * r6);
    r6 = r4 * r2;
    r7 = r4 * r3;
    r7 = fma(r72, r7, r64 * r6);
    WriteSum2<double, double>((double*)inout_shared, r85, r7);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r67, r67, r80 * r80);
    r85 = fma(r65, r65, r87 * r87);
    WriteSum2<double, double>((double*)inout_shared, r7, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r53, r53, r86 * r86);
    r7 = fma(r22, r22, r31 * r31);
    WriteSum2<double, double>((double*)inout_shared, r85, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r33, r33, r74 * r74);
    r85 = fma(r64, r64, r72 * r72);
    WriteSum2<double, double>((double*)inout_shared, r7, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r67, r65, r80 * r87);
    r7 = fma(r67, r53, r80 * r86);
    WriteSum2<double, double>((double*)inout_shared, r85, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r67, r22, r80 * r31);
    r85 = fma(r80, r33, r67 * r74);
    WriteSum2<double, double>((double*)inout_shared, r7, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r67, r64, r80 * r72);
    r80 = fma(r87, r86, r65 * r53);
    WriteSum2<double, double>((double*)inout_shared, r67, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r87, r31, r65 * r22);
    r67 = fma(r87, r33, r65 * r74);
    WriteSum2<double, double>((double*)inout_shared, r80, r67);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = fma(r87, r72, r65 * r64);
    r65 = fma(r53, r22, r86 * r31);
    WriteSum2<double, double>((double*)inout_shared, r87, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r53, r74, r86 * r33);
    r86 = fma(r86, r72, r53 * r64);
    WriteSum2<double, double>((double*)inout_shared, r65, r86);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r22, r74, r31 * r33);
    r31 = fma(r31, r72, r22 * r64);
    WriteSum2<double, double>((double*)inout_shared, r86, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r33, r72, r74 * r64);
    WriteSum1<double, double>((double*)inout_shared, r72);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r46 * r54;
    r33 = r26 * r25;
    r64 = r47 * r55;
    r64 = fma(r35, r64, r57 * r33);
    r33 = r26 * r28;
    r33 = r33 * r5;
    r64 = fma(r56, r33, r64);
    r64 = fma(r47, r82, r64);
    r72 = r72 * r0;
    r72 = r72 * r64;
    r72 = fma(r47, r75, r45 * r72);
    r72 = fma(r25, r19, r72);
    r33 = r4 * r47;
    r33 = r33 * r5;
    r33 = r33 * r56;
    r74 = r46 * r54;
    r74 = r74 * r5;
    r74 = r74 * r64;
    r74 = fma(r45, r74, r39 * r33);
    r74 = fma(r28, r19, r74);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r72,
                                             r74);
    r33 = r46 * r54;
    r64 = r26 * r29;
    r64 = fma(r57, r64, r51 * r82);
    r31 = r26 * r17;
    r31 = r31 * r5;
    r64 = fma(r56, r31, r64);
    r86 = r51 * r55;
    r64 = fma(r35, r86, r64);
    r33 = r33 * r0;
    r33 = r33 * r64;
    r33 = fma(r29, r19, r45 * r33);
    r33 = fma(r51, r75, r33);
    r86 = r46 * r54;
    r86 = r86 * r5;
    r86 = r86 * r64;
    r64 = r4 * r51;
    r64 = r64 * r5;
    r64 = r64 * r56;
    r64 = fma(r39, r64, r45 * r86);
    r64 = fma(r17, r19, r64);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r33,
                                             r64);
    r86 = r46 * r54;
    r31 = r26 * r34;
    r22 = r24 * r55;
    r22 = fma(r35, r22, r57 * r31);
    r31 = r26 * r48;
    r31 = r31 * r5;
    r22 = fma(r56, r31, r22);
    r22 = fma(r24, r82, r22);
    r86 = r86 * r0;
    r86 = r86 * r22;
    r86 = fma(r34, r19, r45 * r86);
    r86 = fma(r24, r75, r86);
    r75 = r46 * r54;
    r75 = r75 * r5;
    r75 = r75 * r22;
    r22 = r4 * r24;
    r22 = r22 * r5;
    r22 = r22 * r56;
    r22 = fma(r39, r22, r45 * r75);
    r22 = fma(r48, r19, r22);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r86,
                                             r22);
    r19 = r4 * r2;
    r75 = r4 * r3;
    r75 = fma(r74, r75, r72 * r19);
    r19 = r4 * r3;
    r39 = r4 * r2;
    r39 = fma(r33, r39, r64 * r19);
    WriteSum2<double, double>((double*)inout_shared, r75, r39);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r4 * r2;
    r75 = r4 * r3;
    r75 = fma(r22, r75, r86 * r39);
    WriteSum1<double, double>((double*)inout_shared, r75);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r74, r74, r72 * r72);
    r39 = fma(r64, r64, r33 * r33);
    WriteSum2<double, double>((double*)inout_shared, r75, r39);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r86, r86, r22 * r22);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r74, r64, r72 * r33);
    r74 = fma(r74, r22, r72 * r86);
    WriteSum2<double, double>((double*)inout_shared, r39, r74);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r33, r86, r64 * r22);
    WriteSum1<double, double>((double*)inout_shared, r86);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                       1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar