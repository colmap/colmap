#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78;
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r20 = fma(r9, r15, r13 * r11);
    r21 = r12 * r10;
    r22 = r8 * r14;
    r20 = r20 + r21;
    r20 = fma(r4, r22, r20);
    r23 = r18 * r20;
    r23 = fma(r20, r23, r19);
    r24 = r17 + r23;
    r24 = fma(r6, r24, r0);
    r0 = 2.00000000000000000e+00;
    r25 = fma(r9, r14, r12 * r11);
    r26 = r13 * r10;
    r25 = fma(r4, r26, r25);
    r25 = fma(r8, r15, r25);
    r26 = r0 * r25;
    r26 = r26 * r20;
    r27 = r16 * r18;
    r28 = fma(r13, r15, r12 * r14);
    r28 = fma(r8, r10, r28);
    r28 = fma(r4, r28, r9 * r11);
    r27 = fma(r28, r27, r26);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r29);
    r30 = r0 * r16;
    r30 = r30 * r25;
    r31 = r0 * r28;
    r32 = fma(r20, r31, r30);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r14 * r10;
    r34 = r34 * r0;
    r35 = r15 * r11;
    r36 = fma(r0, r35, r34);
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
    r24 = fma(r7, r27, r24);
    r24 = fma(r29, r32, r24);
    r24 = fma(r33, r36, r24);
    r24 = fma(r38, r39, r24);
    r24 = fma(r37, r42, r24);
    r32 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r27);
    r44 = r18 * r20;
    r44 = fma(r28, r44, r30);
    r44 = fma(r6, r44, r27);
    r35 = fma(r18, r35, r34);
    r41 = r19 + r41;
    r34 = r14 * r14;
    r34 = r34 * r18;
    r41 = r41 + r34;
    r27 = r15 * r10;
    r27 = r27 * r0;
    r30 = r14 * r11;
    r30 = fma(r0, r30, r27);
    r45 = r0 * r16;
    r45 = r45 * r20;
    r46 = fma(r25, r31, r45);
    r47 = r25 * r25;
    r47 = r47 * r18;
    r23 = r47 + r23;
    r44 = fma(r37, r35, r44);
    r44 = fma(r33, r41, r44);
    r44 = fma(r38, r30, r44);
    r44 = fma(r7, r46, r44);
    r44 = fma(r29, r23, r44);
    r23 = copysign(1.0, r44);
    r23 = fma(r32, r23, r44);
    r32 = 1.0 / r23;
    ReadIdx2<1024, double, double, double2>(focal_and_distortion,
                                            0 * focal_and_distortion_num_alloc,
                                            global_thread_idx,
                                            r44,
                                            r46);
    r48 = r23 * r23;
    r49 = 1.0 / r48;
    r50 = r24 * r49;
    r26 = fma(r16, r31, r26);
    r26 = fma(r6, r26, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r34;
    r34 = r14 * r11;
    r34 = fma(r18, r34, r27);
    r27 = r25 * r18;
    r27 = fma(r28, r27, r45);
    r17 = r19 + r17;
    r17 = r17 + r47;
    r26 = fma(r37, r5, r26);
    r26 = fma(r38, r43, r26);
    r26 = fma(r33, r34, r26);
    r26 = fma(r29, r27, r26);
    r26 = fma(r7, r17, r26);
    r17 = r26 * r26;
    r27 = fma(r49, r17, r24 * r50);
    r27 = fma(r46, r27, r19);
    r27 = r44 * r27;
    r33 = r32 * r27;
    r2 = fma(r24, r33, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r26, r33, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fma(r2, r2, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r12 * r11;
    r38 = -5.00000000000000000e-01;
    r37 = r9 * r14;
    r37 = fma(r38, r37, r38 * r1);
    r1 = r8 * r15;
    r37 = fma(r38, r1, r37);
    r47 = r13 * r10;
    r45 = 5.00000000000000000e-01;
    r37 = fma(r45, r47, r37);
    r47 = r20 * r37;
    r1 = r8 * r11;
    r40 = r13 * r14;
    r40 = fma(r45, r40, r45 * r1);
    r1 = r12 * r15;
    r40 = fma(r38, r1, r40);
    r51 = r9 * r45;
    r40 = fma(r10, r51, r40);
    r1 = fma(r40, r31, r0 * r47);
    r52 = r0 * r25;
    r53 = r9 * r15;
    r54 = r13 * r38;
    r53 = fma(r11, r54, r38 * r53);
    r53 = fma(r45, r22, r53);
    r53 = fma(r38, r21, r53);
    r55 = r0 * r16;
    r56 = r12 * r14;
    r57 = r8 * r10;
    r57 = fma(r38, r57, r38 * r56);
    r57 = fma(r11, r51, r57);
    r57 = fma(r15, r54, r57);
    r55 = r55 * r57;
    r52 = fma(r53, r52, r55);
    r1 = r1 + r52;
    r56 = r0 * r20;
    r56 = r56 * r57;
    r58 = r0 * r25;
    r58 = r58 * r40;
    r59 = r56 + r58;
    r60 = r16 * r18;
    r59 = fma(r37, r60, r59);
    r61 = r18 * r28;
    r59 = fma(r53, r61, r59);
    r59 = fma(r7, r59, r29 * r1);
    r1 = r20 * r40;
    r61 = -4.00000000000000000e+00;
    r1 = r1 * r61;
    r60 = r16 * r53;
    r62 = r61 * r60;
    r63 = r1 + r62;
    r59 = fma(r6, r63, r59);
    r63 = r0 * r20;
    r63 = r63 * r53;
    r64 = r0 * r16;
    r64 = fma(r40, r64, r63);
    r65 = r0 * r25;
    r65 = r65 * r37;
    r66 = r57 * r31;
    r67 = r65 + r66;
    r68 = r64 + r67;
    r69 = r18 * r28;
    r69 = fma(r18, r47, r40 * r69);
    r69 = r69 + r52;
    r69 = fma(r6, r69, r7 * r68);
    r68 = r25 * r61;
    r40 = r57 * r68;
    r1 = r1 + r40;
    r69 = fma(r29, r1, r69);
    r1 = r4 * r27;
    r1 = r1 * r50;
    r70 = fma(r69, r1, r59 * r33);
    r71 = r44 * r46;
    r72 = r0 * r59;
    r73 = r24 * r24;
    r48 = r23 * r48;
    r48 = 1.0 / r48;
    r48 = r18 * r48;
    r73 = r73 * r48;
    r72 = fma(r69, r73, r50 * r72);
    r23 = r69 * r48;
    r72 = fma(r17, r23, r72);
    r74 = r0 * r26;
    r75 = r25 * r18;
    r76 = r18 * r28;
    r76 = r76 * r57;
    r75 = fma(r37, r75, r76);
    r75 = r75 + r64;
    r40 = r62 + r40;
    r40 = fma(r7, r40, r29 * r75);
    r58 = fma(r53, r31, r58);
    r75 = r0 * r16;
    r75 = fma(r37, r75, r56);
    r58 = r58 + r75;
    r40 = fma(r6, r58, r40);
    r74 = r74 * r40;
    r72 = fma(r49, r74, r72);
    r71 = r71 * r72;
    r71 = r71 * r32;
    r70 = fma(r24, r71, r70);
    r72 = r4 * r26;
    r72 = r72 * r69;
    r72 = r72 * r49;
    r71 = fma(r26, r71, r27 * r72);
    r71 = fma(r40, r33, r71);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r70, r71);
    r40 = r18 * r20;
    r40 = fma(r53, r40, r76);
    r72 = r0 * r16;
    r74 = r8 * r11;
    r23 = r12 * r15;
    r23 = fma(r45, r23, r38 * r74);
    r74 = r9 * r10;
    r23 = fma(r38, r74, r23);
    r23 = fma(r14, r54, r23);
    r72 = r72 * r23;
    r74 = r0 * r25;
    r58 = r12 * r11;
    r56 = r8 * r15;
    r56 = fma(r45, r56, r45 * r58);
    r56 = fma(r14, r51, r56);
    r56 = fma(r10, r54, r56);
    r74 = fma(r56, r74, r72);
    r40 = r40 + r74;
    r54 = r0 * r20;
    r54 = r54 * r56;
    r58 = fma(r23, r31, r54);
    r58 = r58 + r52;
    r58 = fma(r7, r58, r6 * r40);
    r40 = r20 * r57;
    r40 = r40 * r61;
    r52 = r23 * r68;
    r62 = r40 + r52;
    r58 = fma(r29, r62, r58);
    r66 = r63 + r66;
    r66 = r66 + r74;
    r74 = r16 * r61;
    r74 = r74 * r56;
    r40 = r40 + r74;
    r40 = fma(r6, r40, r29 * r66);
    r66 = r18 * r28;
    r66 = fma(r18, r60, r56 * r66);
    r63 = r0 * r25;
    r63 = r63 * r57;
    r62 = r0 * r20;
    r62 = fma(r23, r62, r63);
    r66 = r66 + r62;
    r40 = fma(r7, r66, r40);
    r66 = fma(r40, r33, r58 * r1);
    r64 = r44 * r46;
    r77 = r0 * r40;
    r78 = r0 * r26;
    r54 = r55 + r54;
    r55 = r25 * r18;
    r54 = fma(r53, r55, r54);
    r53 = r18 * r28;
    r54 = fma(r23, r53, r54);
    r56 = fma(r56, r31, r0 * r60);
    r56 = r56 + r62;
    r56 = fma(r6, r56, r29 * r54);
    r52 = r74 + r52;
    r56 = fma(r7, r52, r56);
    r78 = r78 * r56;
    r78 = fma(r49, r78, r50 * r77);
    r77 = r58 * r48;
    r78 = fma(r17, r77, r78);
    r78 = fma(r58, r73, r78);
    r64 = r64 * r24;
    r64 = r64 * r78;
    r66 = fma(r32, r64, r66);
    r64 = r44 * r46;
    r64 = r64 * r26;
    r64 = r64 * r78;
    r56 = fma(r56, r33, r32 * r64);
    r64 = r4 * r26;
    r64 = r64 * r58;
    r64 = r64 * r49;
    r56 = fma(r27, r64, r56);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r66, r56);
    r76 = r65 + r76;
    r65 = r0 * r20;
    r64 = r13 * r11;
    r22 = fma(r38, r22, r45 * r64);
    r22 = fma(r15, r51, r22);
    r22 = fma(r45, r21, r22);
    r65 = r65 * r22;
    r21 = r16 * r18;
    r76 = fma(r23, r21, r76);
    r76 = r76 + r65;
    r57 = r16 * r57;
    r57 = r57 * r61;
    r47 = r61 * r47;
    r61 = r57 + r47;
    r61 = fma(r6, r61, r7 * r76);
    r76 = r0 * r16;
    r76 = r76 * r22;
    r21 = fma(r37, r31, r76);
    r21 = r21 + r62;
    r61 = fma(r29, r21, r61);
    r68 = r22 * r68;
    r47 = r47 + r68;
    r76 = r63 + r76;
    r63 = r18 * r20;
    r76 = fma(r23, r63, r76);
    r21 = r18 * r28;
    r76 = fma(r37, r21, r76);
    r76 = fma(r6, r76, r29 * r47);
    r47 = r0 * r25;
    r31 = fma(r22, r31, r23 * r47);
    r31 = r31 + r75;
    r76 = fma(r7, r31, r76);
    r31 = fma(r76, r1, r61 * r33);
    r47 = r44 * r46;
    r21 = r0 * r26;
    r65 = r72 + r65;
    r65 = r65 + r67;
    r67 = r25 * r18;
    r72 = r18 * r28;
    r72 = fma(r22, r72, r23 * r67);
    r72 = r72 + r75;
    r72 = fma(r29, r72, r6 * r65);
    r68 = r57 + r68;
    r72 = fma(r7, r68, r72);
    r21 = r21 * r72;
    r21 = fma(r76, r73, r49 * r21);
    r68 = r0 * r61;
    r21 = fma(r50, r68, r21);
    r7 = r76 * r48;
    r21 = fma(r17, r7, r21);
    r47 = r47 * r24;
    r47 = r47 * r21;
    r31 = fma(r32, r47, r31);
    r47 = r4 * r26;
    r47 = r47 * r76;
    r47 = r47 * r49;
    r72 = fma(r72, r33, r27 * r47);
    r47 = r44 * r46;
    r47 = r47 * r26;
    r47 = r47 * r21;
    r72 = fma(r32, r47, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r72);
    r47 = fma(r35, r1, r42 * r33);
    r21 = r44 * r46;
    r7 = r0 * r5;
    r7 = r7 * r26;
    r68 = r35 * r48;
    r68 = fma(r17, r68, r49 * r7);
    r7 = r0 * r42;
    r68 = fma(r50, r7, r68);
    r68 = fma(r35, r73, r68);
    r21 = r21 * r24;
    r21 = r21 * r68;
    r47 = fma(r32, r21, r47);
    r21 = r4 * r35;
    r21 = r21 * r26;
    r21 = r21 * r49;
    r7 = r44 * r46;
    r7 = r7 * r26;
    r7 = r7 * r68;
    r7 = fma(r32, r7, r27 * r21);
    r7 = fma(r5, r33, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r47, r7);
    r21 = r44 * r46;
    r68 = r30 * r48;
    r57 = r0 * r43;
    r57 = r57 * r26;
    r57 = fma(r49, r57, r17 * r68);
    r68 = r0 * r39;
    r57 = fma(r50, r68, r57);
    r57 = fma(r30, r73, r57);
    r21 = r21 * r24;
    r21 = r21 * r57;
    r21 = fma(r30, r1, r32 * r21);
    r21 = fma(r39, r33, r21);
    r68 = r4 * r30;
    r68 = r68 * r26;
    r68 = r68 * r49;
    r68 = fma(r27, r68, r43 * r33);
    r29 = r44 * r46;
    r29 = r29 * r26;
    r29 = r29 * r57;
    r68 = fma(r32, r29, r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r21, r68);
    r29 = r44 * r46;
    r57 = r0 * r34;
    r57 = r57 * r26;
    r65 = r41 * r48;
    r65 = fma(r17, r65, r49 * r57);
    r57 = r0 * r36;
    r65 = fma(r50, r57, r65);
    r65 = fma(r41, r73, r65);
    r29 = r29 * r24;
    r29 = r29 * r65;
    r29 = fma(r36, r33, r32 * r29);
    r29 = fma(r41, r1, r29);
    r1 = r4 * r41;
    r1 = r1 * r26;
    r1 = r1 * r49;
    r49 = r44 * r46;
    r49 = r49 * r26;
    r49 = r49 * r65;
    r49 = fma(r32, r49, r27 * r1);
    r49 = fma(r34, r33, r49);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r29, r49);
    r33 = r4 * r3;
    r2 = r4 * r2;
    r33 = fma(r70, r2, r71 * r33);
    r1 = r4 * r3;
    r1 = fma(r66, r2, r56 * r1);
    WriteSum2<double, double>((double*)inout_shared, r33, r1);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r3;
    r1 = fma(r31, r2, r72 * r1);
    r33 = r4 * r3;
    r33 = fma(r47, r2, r7 * r33);
    WriteSum2<double, double>((double*)inout_shared, r1, r33);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r3;
    r33 = fma(r21, r2, r68 * r33);
    r1 = r4 * r3;
    r1 = fma(r29, r2, r49 * r1);
    WriteSum2<double, double>((double*)inout_shared, r33, r1);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r70, r70, r71 * r71);
    r33 = fma(r66, r66, r56 * r56);
    WriteSum2<double, double>((double*)inout_shared, r1, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r31, r31, r72 * r72);
    r1 = fma(r47, r47, r7 * r7);
    WriteSum2<double, double>((double*)inout_shared, r33, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r68, r68, r21 * r21);
    r33 = fma(r49, r49, r29 * r29);
    WriteSum2<double, double>((double*)inout_shared, r1, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r71, r56, r70 * r66);
    r1 = fma(r71, r72, r70 * r31);
    WriteSum2<double, double>((double*)inout_shared, r33, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r71, r7, r70 * r47);
    r33 = fma(r70, r21, r71 * r68);
    WriteSum2<double, double>((double*)inout_shared, r1, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r70, r29, r71 * r49);
    r71 = fma(r66, r31, r56 * r72);
    WriteSum2<double, double>((double*)inout_shared, r70, r71);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fma(r66, r47, r56 * r7);
    r70 = fma(r66, r21, r56 * r68);
    WriteSum2<double, double>((double*)inout_shared, r71, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r66, r29, r56 * r49);
    r56 = fma(r31, r47, r72 * r7);
    WriteSum2<double, double>((double*)inout_shared, r66, r56);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r72, r68, r31 * r21);
    r31 = fma(r31, r29, r72 * r49);
    WriteSum2<double, double>((double*)inout_shared, r56, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r47, r21, r7 * r68);
    r7 = fma(r7, r49, r47 * r29);
    WriteSum2<double, double>((double*)inout_shared, r31, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r68, r49, r21 * r29);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r49);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r19, r19);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              principal_point,
              principal_point_num_alloc,
              principal_point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
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
              out_principal_point_jac,
              out_principal_point_jac_num_alloc,
              out_principal_point_njtr,
              out_principal_point_njtr_num_alloc,
              out_principal_point_precond_diag,
              out_principal_point_precond_diag_num_alloc,
              out_principal_point_precond_tril,
              out_principal_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar