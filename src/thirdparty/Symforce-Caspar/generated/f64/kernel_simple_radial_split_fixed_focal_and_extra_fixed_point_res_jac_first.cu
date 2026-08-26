#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirstKernel(
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
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
      r76, r77, r78, r79;
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
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
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
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r44 * r46;
    r38 = r12 * r11;
    r37 = -5.00000000000000000e-01;
    r47 = r9 * r14;
    r47 = fma(r37, r47, r37 * r38);
    r38 = r8 * r15;
    r47 = fma(r37, r38, r47);
    r45 = r13 * r10;
    r40 = 5.00000000000000000e-01;
    r47 = fma(r40, r45, r47);
    r45 = r20 * r47;
    r38 = r8 * r11;
    r51 = r13 * r14;
    r51 = fma(r40, r51, r40 * r38);
    r38 = r12 * r15;
    r51 = fma(r37, r38, r51);
    r52 = r9 * r40;
    r51 = fma(r10, r52, r51);
    r38 = fma(r51, r31, r0 * r45);
    r53 = r0 * r25;
    r54 = r9 * r15;
    r55 = r13 * r37;
    r54 = fma(r11, r55, r37 * r54);
    r54 = fma(r40, r22, r54);
    r54 = fma(r37, r21, r54);
    r56 = r0 * r16;
    r57 = r12 * r14;
    r58 = r8 * r10;
    r58 = fma(r37, r58, r37 * r57);
    r58 = fma(r11, r52, r58);
    r58 = fma(r15, r55, r58);
    r56 = r56 * r58;
    r53 = fma(r54, r53, r56);
    r38 = r38 + r53;
    r57 = r0 * r20;
    r57 = r57 * r58;
    r59 = r0 * r25;
    r59 = r59 * r51;
    r60 = r57 + r59;
    r61 = r16 * r18;
    r60 = fma(r47, r61, r60);
    r62 = r18 * r28;
    r60 = fma(r54, r62, r60);
    r60 = fma(r7, r60, r29 * r38);
    r38 = r20 * r51;
    r62 = -4.00000000000000000e+00;
    r38 = r38 * r62;
    r61 = r16 * r54;
    r63 = r62 * r61;
    r64 = r38 + r63;
    r60 = fma(r6, r64, r60);
    r64 = r0 * r60;
    r65 = r0 * r20;
    r65 = r65 * r54;
    r66 = r0 * r16;
    r66 = fma(r51, r66, r65);
    r67 = r0 * r25;
    r67 = r67 * r47;
    r68 = r58 * r31;
    r69 = r67 + r68;
    r70 = r66 + r69;
    r71 = r18 * r28;
    r71 = fma(r18, r45, r51 * r71);
    r71 = r71 + r53;
    r71 = fma(r6, r71, r7 * r70);
    r70 = r25 * r62;
    r51 = r58 * r70;
    r38 = r38 + r51;
    r71 = fma(r29, r38, r71);
    r38 = r24 * r24;
    r48 = r23 * r48;
    r48 = 1.0 / r48;
    r48 = r18 * r48;
    r38 = r38 * r48;
    r64 = fma(r71, r38, r50 * r64);
    r23 = r71 * r48;
    r64 = fma(r17, r23, r64);
    r72 = r0 * r26;
    r73 = r25 * r18;
    r74 = r18 * r28;
    r74 = r74 * r58;
    r73 = fma(r47, r73, r74);
    r73 = r73 + r66;
    r51 = r63 + r51;
    r51 = fma(r7, r51, r29 * r73);
    r59 = fma(r54, r31, r59);
    r73 = r0 * r16;
    r73 = fma(r47, r73, r57);
    r59 = r59 + r73;
    r51 = fma(r6, r59, r51);
    r72 = r72 * r51;
    r64 = fma(r49, r72, r64);
    r1 = r1 * r64;
    r1 = r1 * r32;
    r64 = fma(r60, r33, r24 * r1);
    r72 = r4 * r27;
    r72 = r72 * r50;
    r64 = fma(r71, r72, r64);
    r51 = fma(r51, r33, r26 * r1);
    r1 = r4 * r26;
    r1 = r1 * r71;
    r1 = r1 * r49;
    r51 = fma(r27, r1, r51);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r64, r51);
    r1 = r44 * r46;
    r68 = r65 + r68;
    r65 = r0 * r16;
    r23 = r8 * r11;
    r59 = r12 * r15;
    r59 = fma(r40, r59, r37 * r23);
    r23 = r9 * r10;
    r59 = fma(r37, r23, r59);
    r59 = fma(r14, r55, r59);
    r65 = r65 * r59;
    r23 = r0 * r25;
    r57 = r12 * r11;
    r63 = r8 * r15;
    r63 = fma(r40, r63, r40 * r57);
    r63 = fma(r14, r52, r63);
    r63 = fma(r10, r55, r63);
    r23 = fma(r63, r23, r65);
    r68 = r68 + r23;
    r55 = r20 * r58;
    r55 = r55 * r62;
    r57 = r16 * r62;
    r57 = r57 * r63;
    r66 = r55 + r57;
    r66 = fma(r6, r66, r29 * r68);
    r68 = r18 * r28;
    r68 = fma(r18, r61, r63 * r68);
    r75 = r0 * r25;
    r75 = r75 * r58;
    r76 = r0 * r20;
    r76 = fma(r59, r76, r75);
    r68 = r68 + r76;
    r66 = fma(r7, r68, r66);
    r68 = r0 * r66;
    r77 = r0 * r26;
    r78 = r25 * r18;
    r78 = fma(r54, r78, r56);
    r56 = r0 * r20;
    r56 = r56 * r63;
    r79 = r18 * r28;
    r78 = fma(r59, r79, r78);
    r78 = r78 + r56;
    r63 = fma(r63, r31, r0 * r61);
    r63 = r63 + r76;
    r63 = fma(r6, r63, r29 * r78);
    r78 = r59 * r70;
    r57 = r57 + r78;
    r63 = fma(r7, r57, r63);
    r77 = r77 * r63;
    r77 = fma(r49, r77, r50 * r68);
    r68 = r18 * r20;
    r68 = fma(r54, r68, r74);
    r68 = r68 + r23;
    r56 = fma(r59, r31, r56);
    r56 = r56 + r53;
    r56 = fma(r7, r56, r6 * r68);
    r78 = r55 + r78;
    r56 = fma(r29, r78, r56);
    r78 = r56 * r48;
    r77 = fma(r17, r78, r77);
    r77 = fma(r56, r38, r77);
    r1 = r1 * r24;
    r1 = r1 * r77;
    r1 = fma(r56, r72, r32 * r1);
    r1 = fma(r66, r33, r1);
    r78 = r44 * r46;
    r78 = r78 * r26;
    r78 = r78 * r77;
    r78 = fma(r32, r78, r63 * r33);
    r63 = r4 * r26;
    r63 = r63 * r56;
    r63 = r63 * r49;
    r78 = fma(r27, r63, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r1, r78);
    r63 = r13 * r11;
    r22 = fma(r37, r22, r40 * r63);
    r22 = fma(r15, r52, r22);
    r22 = fma(r40, r21, r22);
    r70 = r22 * r70;
    r45 = r62 * r45;
    r21 = r70 + r45;
    r40 = r0 * r16;
    r40 = r40 * r22;
    r75 = r75 + r40;
    r52 = r18 * r20;
    r75 = fma(r59, r52, r75);
    r37 = r18 * r28;
    r75 = fma(r47, r37, r75);
    r75 = fma(r6, r75, r29 * r21);
    r21 = r0 * r25;
    r21 = fma(r22, r31, r59 * r21);
    r21 = r21 + r73;
    r75 = fma(r7, r21, r75);
    r74 = r67 + r74;
    r67 = r0 * r20;
    r67 = r67 * r22;
    r21 = r16 * r18;
    r74 = fma(r59, r21, r74);
    r74 = r74 + r67;
    r58 = r16 * r58;
    r58 = r58 * r62;
    r45 = r58 + r45;
    r45 = fma(r6, r45, r7 * r74);
    r31 = fma(r47, r31, r40);
    r31 = r31 + r76;
    r45 = fma(r29, r31, r45);
    r31 = fma(r45, r33, r75 * r72);
    r76 = r44 * r46;
    r47 = r0 * r26;
    r67 = r65 + r67;
    r67 = r67 + r69;
    r69 = r25 * r18;
    r65 = r18 * r28;
    r65 = fma(r22, r65, r59 * r69);
    r65 = r65 + r73;
    r65 = fma(r29, r65, r6 * r67);
    r70 = r58 + r70;
    r65 = fma(r7, r70, r65);
    r47 = r47 * r65;
    r47 = fma(r75, r38, r49 * r47);
    r70 = r0 * r45;
    r47 = fma(r50, r70, r47);
    r7 = r75 * r48;
    r47 = fma(r17, r7, r47);
    r76 = r76 * r24;
    r76 = r76 * r47;
    r31 = fma(r32, r76, r31);
    r76 = r44 * r46;
    r76 = r76 * r26;
    r76 = r76 * r47;
    r47 = r4 * r26;
    r47 = r47 * r75;
    r47 = r47 * r49;
    r47 = fma(r27, r47, r32 * r76);
    r47 = fma(r65, r33, r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r47);
    r65 = r44 * r46;
    r76 = r0 * r5;
    r76 = r76 * r26;
    r7 = r35 * r48;
    r7 = fma(r17, r7, r49 * r76);
    r76 = r0 * r42;
    r7 = fma(r50, r76, r7);
    r7 = fma(r35, r38, r7);
    r65 = r65 * r24;
    r65 = r65 * r7;
    r65 = fma(r32, r65, r35 * r72);
    r65 = fma(r42, r33, r65);
    r76 = r44 * r46;
    r76 = r76 * r26;
    r76 = r76 * r7;
    r76 = fma(r32, r76, r5 * r33);
    r7 = r4 * r35;
    r7 = r7 * r26;
    r7 = r7 * r49;
    r76 = fma(r27, r7, r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r65, r76);
    r7 = fma(r30, r72, r39 * r33);
    r70 = r44 * r46;
    r58 = r30 * r48;
    r29 = r0 * r43;
    r29 = r29 * r26;
    r29 = fma(r49, r29, r17 * r58);
    r58 = r0 * r39;
    r29 = fma(r50, r58, r29);
    r29 = fma(r30, r38, r29);
    r70 = r70 * r24;
    r70 = r70 * r29;
    r7 = fma(r32, r70, r7);
    r70 = r44 * r46;
    r70 = r70 * r26;
    r70 = r70 * r29;
    r70 = fma(r43, r33, r32 * r70);
    r29 = r4 * r30;
    r29 = r29 * r26;
    r29 = r29 * r49;
    r70 = fma(r27, r29, r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r7, r70);
    r29 = r44 * r46;
    r58 = r0 * r34;
    r58 = r58 * r26;
    r67 = r41 * r48;
    r67 = fma(r17, r67, r49 * r58);
    r58 = r0 * r36;
    r67 = fma(r50, r58, r67);
    r67 = fma(r41, r38, r67);
    r29 = r29 * r24;
    r29 = r29 * r67;
    r29 = fma(r36, r33, r32 * r29);
    r29 = fma(r41, r72, r29);
    r72 = r4 * r41;
    r72 = r72 * r26;
    r72 = r72 * r49;
    r49 = r44 * r46;
    r49 = r49 * r26;
    r49 = r49 * r67;
    r49 = fma(r32, r49, r27 * r72);
    r49 = fma(r34, r33, r49);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r29, r49);
    r33 = r4 * r2;
    r3 = r4 * r3;
    r33 = fma(r51, r3, r64 * r33);
    r72 = r4 * r2;
    r72 = fma(r78, r3, r1 * r72);
    WriteSum2<double, double>((double*)inout_shared, r33, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r4 * r2;
    r72 = fma(r47, r3, r31 * r72);
    r33 = r4 * r2;
    r33 = fma(r76, r3, r65 * r33);
    WriteSum2<double, double>((double*)inout_shared, r72, r33);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r2;
    r33 = fma(r70, r3, r7 * r33);
    r72 = r4 * r2;
    r72 = fma(r49, r3, r29 * r72);
    WriteSum2<double, double>((double*)inout_shared, r33, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r64, r64, r51 * r51);
    r33 = fma(r78, r78, r1 * r1);
    WriteSum2<double, double>((double*)inout_shared, r72, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r31, r31, r47 * r47);
    r72 = fma(r65, r65, r76 * r76);
    WriteSum2<double, double>((double*)inout_shared, r33, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r7, r7, r70 * r70);
    r33 = fma(r29, r29, r49 * r49);
    WriteSum2<double, double>((double*)inout_shared, r72, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r64, r1, r51 * r78);
    r72 = fma(r51, r47, r64 * r31);
    WriteSum2<double, double>((double*)inout_shared, r33, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r51, r76, r64 * r65);
    r33 = fma(r64, r7, r51 * r70);
    WriteSum2<double, double>((double*)inout_shared, r72, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r64, r29, r51 * r49);
    r51 = fma(r78, r47, r1 * r31);
    WriteSum2<double, double>((double*)inout_shared, r64, r51);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r78, r76, r1 * r65);
    r64 = fma(r78, r70, r1 * r7);
    WriteSum2<double, double>((double*)inout_shared, r51, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r78, r49, r1 * r29);
    r1 = fma(r31, r65, r47 * r76);
    WriteSum2<double, double>((double*)inout_shared, r78, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r31, r7, r47 * r70);
    r47 = fma(r47, r49, r31 * r29);
    WriteSum2<double, double>((double*)inout_shared, r1, r47);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r76, r70, r65 * r7);
    r76 = fma(r76, r49, r65 * r29);
    WriteSum2<double, double>((double*)inout_shared, r47, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r70, r49, r7 * r29);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r2;
    WriteSum2<double, double>((double*)inout_shared, r49, r3);
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

void SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirst(
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
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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