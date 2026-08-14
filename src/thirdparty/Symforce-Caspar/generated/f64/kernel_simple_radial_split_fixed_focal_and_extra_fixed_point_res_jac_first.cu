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
      r76, r77, r78, r79, r80;
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
    r33 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r28);
    r44 = r18 * r21;
    r44 = fma(r29, r44, r31);
    r44 = fma(r6, r44, r28);
    r28 = r15 * r11;
    r28 = fma(r18, r28, r35);
    r41 = r19 + r41;
    r35 = r14 * r14;
    r35 = r35 * r18;
    r41 = r41 + r35;
    r31 = r15 * r10;
    r31 = r31 * r0;
    r45 = r14 * r11;
    r45 = fma(r0, r45, r31);
    r46 = r0 * r16;
    r46 = r46 * r21;
    r47 = fma(r26, r32, r46);
    r48 = r26 * r26;
    r48 = r48 * r18;
    r24 = r48 + r24;
    r44 = fma(r37, r28, r44);
    r44 = fma(r34, r41, r44);
    r44 = fma(r38, r45, r44);
    r44 = fma(r7, r47, r44);
    r44 = fma(r30, r24, r44);
    r24 = copysign(1.0, r44);
    r24 = fma(r33, r24, r44);
    r33 = 1.0 / r24;
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r44,
                                            r47);
    r49 = r24 * r24;
    r50 = 1.0 / r49;
    r51 = r25 * r50;
    r27 = fma(r16, r32, r27);
    r27 = fma(r6, r27, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r35;
    r35 = r14 * r11;
    r35 = fma(r18, r35, r31);
    r31 = r26 * r18;
    r31 = fma(r29, r31, r46);
    r17 = r19 + r17;
    r17 = r17 + r48;
    r27 = fma(r37, r5, r27);
    r27 = fma(r38, r43, r27);
    r27 = fma(r34, r35, r27);
    r27 = fma(r30, r31, r27);
    r27 = fma(r7, r17, r27);
    r17 = r27 * r27;
    r31 = fma(r50, r17, r25 * r51);
    r31 = fma(r47, r31, r19);
    r31 = r44 * r31;
    r34 = r33 * r31;
    r2 = fma(r25, r34, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r27, r34, r3);
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
    r38 = -5.00000000000000000e-01;
    r37 = r13 * r38;
    r48 = 5.00000000000000000e-01;
    r46 = fma(r48, r23, r11 * r37);
    r46 = fma(r38, r20, r46);
    r46 = fma(r38, r22, r46);
    r1 = r1 * r46;
    r40 = r0 * r16;
    r52 = r13 * r14;
    r53 = r12 * r15;
    r53 = fma(r38, r53, r48 * r52);
    r52 = r9 * r10;
    r53 = fma(r48, r52, r53);
    r54 = r11 * r48;
    r53 = fma(r8, r54, r53);
    r40 = fma(r53, r40, r1);
    r52 = r0 * r26;
    r55 = r12 * r11;
    r56 = r9 * r14;
    r56 = fma(r38, r56, r38 * r55);
    r55 = r8 * r15;
    r56 = fma(r38, r55, r56);
    r57 = r13 * r10;
    r56 = fma(r48, r57, r56);
    r52 = r52 * r56;
    r57 = r12 * r14;
    r55 = r8 * r10;
    r55 = fma(r38, r55, r38 * r57);
    r55 = fma(r9, r54, r55);
    r55 = fma(r15, r37, r55);
    r57 = r55 * r32;
    r58 = r52 + r57;
    r59 = r40 + r58;
    r60 = r18 * r29;
    r61 = r21 * r56;
    r60 = fma(r18, r61, r53 * r60);
    r62 = r0 * r26;
    r63 = r0 * r16;
    r63 = r63 * r55;
    r62 = fma(r46, r62, r63);
    r60 = r60 + r62;
    r60 = fma(r6, r60, r7 * r59);
    r59 = r21 * r53;
    r64 = -4.00000000000000000e+00;
    r59 = r59 * r64;
    r65 = r26 * r64;
    r66 = r55 * r65;
    r67 = r59 + r66;
    r60 = fma(r30, r67, r60);
    r67 = r4 * r31;
    r67 = r67 * r51;
    r68 = fma(r53, r32, r0 * r61);
    r68 = r68 + r62;
    r69 = r0 * r21;
    r69 = r69 * r55;
    r70 = r0 * r26;
    r70 = r70 * r53;
    r53 = r69 + r70;
    r71 = r16 * r18;
    r53 = fma(r56, r71, r53);
    r72 = r18 * r29;
    r53 = fma(r46, r72, r53);
    r53 = fma(r7, r53, r30 * r68);
    r68 = r16 * r46;
    r72 = r64 * r68;
    r59 = r59 + r72;
    r53 = fma(r6, r59, r53);
    r59 = fma(r53, r34, r60 * r67);
    r71 = r44 * r47;
    r73 = r0 * r53;
    r74 = r25 * r25;
    r49 = r24 * r49;
    r49 = 1.0 / r49;
    r49 = r18 * r49;
    r74 = r74 * r49;
    r73 = fma(r60, r74, r51 * r73);
    r24 = r60 * r49;
    r73 = fma(r17, r24, r73);
    r75 = r0 * r27;
    r76 = r26 * r18;
    r77 = r18 * r29;
    r77 = r77 * r55;
    r76 = fma(r56, r76, r77);
    r76 = r76 + r40;
    r72 = r66 + r72;
    r72 = fma(r7, r72, r30 * r76);
    r70 = fma(r46, r32, r70);
    r76 = r0 * r16;
    r76 = fma(r56, r76, r69);
    r70 = r70 + r76;
    r72 = fma(r6, r70, r72);
    r75 = r75 * r72;
    r73 = fma(r50, r75, r73);
    r71 = r71 * r73;
    r71 = r71 * r33;
    r59 = fma(r25, r71, r59);
    r72 = fma(r72, r34, r27 * r71);
    r71 = r4 * r27;
    r71 = r71 * r60;
    r71 = r71 * r50;
    r72 = fma(r31, r71, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r59, r72);
    r57 = r1 + r57;
    r1 = r0 * r16;
    r71 = r8 * r11;
    r73 = r12 * r15;
    r73 = fma(r48, r73, r38 * r71);
    r71 = r9 * r10;
    r73 = fma(r38, r71, r73);
    r73 = fma(r14, r37, r73);
    r1 = r1 * r73;
    r71 = r0 * r26;
    r75 = r9 * r14;
    r24 = r8 * r15;
    r24 = fma(r48, r24, r48 * r75);
    r24 = fma(r12, r54, r24);
    r24 = fma(r10, r37, r24);
    r71 = fma(r24, r71, r1);
    r57 = r57 + r71;
    r37 = r21 * r55;
    r37 = r37 * r64;
    r75 = r16 * r64;
    r75 = r75 * r24;
    r70 = r37 + r75;
    r70 = fma(r6, r70, r30 * r57);
    r57 = r18 * r29;
    r57 = fma(r18, r68, r24 * r57);
    r69 = r0 * r26;
    r69 = r69 * r55;
    r66 = r0 * r21;
    r66 = fma(r73, r66, r69);
    r57 = r57 + r66;
    r70 = fma(r7, r57, r70);
    r57 = r44 * r47;
    r40 = r0 * r70;
    r78 = r0 * r27;
    r79 = r26 * r18;
    r79 = fma(r46, r79, r63);
    r63 = r0 * r21;
    r63 = r63 * r24;
    r80 = r18 * r29;
    r79 = fma(r73, r80, r79);
    r79 = r79 + r63;
    r24 = fma(r24, r32, r0 * r68);
    r24 = r24 + r66;
    r24 = fma(r6, r24, r30 * r79);
    r79 = r73 * r65;
    r75 = r75 + r79;
    r24 = fma(r7, r75, r24);
    r78 = r78 * r24;
    r78 = fma(r50, r78, r51 * r40);
    r40 = r18 * r21;
    r40 = fma(r46, r40, r77);
    r40 = r40 + r71;
    r63 = fma(r73, r32, r63);
    r63 = r63 + r62;
    r63 = fma(r7, r63, r6 * r40);
    r79 = r37 + r79;
    r63 = fma(r30, r79, r63);
    r79 = r63 * r49;
    r78 = fma(r17, r79, r78);
    r78 = fma(r63, r74, r78);
    r57 = r57 * r25;
    r57 = r57 * r78;
    r57 = fma(r33, r57, r70 * r34);
    r57 = fma(r63, r67, r57);
    r79 = r44 * r47;
    r79 = r79 * r27;
    r79 = r79 * r78;
    r79 = fma(r33, r79, r24 * r34);
    r24 = r4 * r27;
    r24 = r24 * r63;
    r24 = r24 * r50;
    r79 = fma(r31, r24, r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r57, r79);
    r24 = r44 * r47;
    r78 = r0 * r27;
    r37 = r0 * r21;
    r23 = fma(r38, r23, r13 * r54);
    r23 = fma(r48, r20, r23);
    r23 = fma(r48, r22, r23);
    r37 = r37 * r23;
    r1 = r1 + r37;
    r1 = r1 + r58;
    r58 = r26 * r18;
    r22 = r18 * r29;
    r22 = fma(r23, r22, r73 * r58);
    r22 = r22 + r76;
    r22 = fma(r30, r22, r6 * r1);
    r55 = r16 * r55;
    r55 = r55 * r64;
    r65 = r23 * r65;
    r1 = r55 + r65;
    r22 = fma(r7, r1, r22);
    r78 = r78 * r22;
    r61 = r64 * r61;
    r65 = r65 + r61;
    r64 = r0 * r16;
    r64 = r64 * r23;
    r69 = r69 + r64;
    r1 = r18 * r21;
    r69 = fma(r73, r1, r69);
    r58 = r18 * r29;
    r69 = fma(r56, r58, r69);
    r69 = fma(r6, r69, r30 * r65);
    r65 = r0 * r26;
    r23 = fma(r23, r32, r73 * r65);
    r23 = r23 + r76;
    r69 = fma(r7, r23, r69);
    r78 = fma(r69, r74, r50 * r78);
    r77 = r52 + r77;
    r52 = r16 * r18;
    r77 = fma(r73, r52, r77);
    r77 = r77 + r37;
    r61 = r55 + r61;
    r61 = fma(r6, r61, r7 * r77);
    r32 = fma(r56, r32, r64);
    r32 = r32 + r66;
    r61 = fma(r30, r32, r61);
    r32 = r0 * r61;
    r78 = fma(r51, r32, r78);
    r30 = r69 * r49;
    r78 = fma(r17, r30, r78);
    r24 = r24 * r25;
    r24 = r24 * r78;
    r24 = fma(r61, r34, r33 * r24);
    r24 = fma(r69, r67, r24);
    r30 = r44 * r47;
    r30 = r30 * r27;
    r30 = r30 * r78;
    r78 = r4 * r27;
    r78 = r78 * r69;
    r78 = r78 * r50;
    r78 = fma(r31, r78, r33 * r30);
    r78 = fma(r22, r34, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r24, r78);
    r22 = r44 * r47;
    r30 = r0 * r5;
    r30 = r30 * r27;
    r32 = r28 * r49;
    r32 = fma(r17, r32, r50 * r30);
    r30 = r0 * r42;
    r32 = fma(r51, r30, r32);
    r32 = fma(r28, r74, r32);
    r22 = r22 * r25;
    r22 = r22 * r32;
    r22 = fma(r33, r22, r42 * r34);
    r22 = fma(r28, r67, r22);
    r30 = r4 * r28;
    r30 = r30 * r27;
    r30 = r30 * r50;
    r30 = fma(r5, r34, r31 * r30);
    r66 = r44 * r47;
    r66 = r66 * r27;
    r66 = r66 * r32;
    r30 = fma(r33, r66, r30);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r30);
    r66 = fma(r39, r34, r45 * r67);
    r32 = r44 * r47;
    r56 = r45 * r49;
    r64 = r0 * r43;
    r64 = r64 * r27;
    r64 = fma(r50, r64, r17 * r56);
    r56 = r0 * r39;
    r64 = fma(r51, r56, r64);
    r64 = fma(r45, r74, r64);
    r32 = r32 * r25;
    r32 = r32 * r64;
    r66 = fma(r33, r32, r66);
    r32 = r4 * r45;
    r32 = r32 * r27;
    r32 = r32 * r50;
    r32 = fma(r43, r34, r31 * r32);
    r56 = r44 * r47;
    r56 = r56 * r27;
    r56 = r56 * r64;
    r32 = fma(r33, r56, r32);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r66, r32);
    r56 = r44 * r47;
    r64 = r0 * r35;
    r64 = r64 * r27;
    r6 = r41 * r49;
    r6 = fma(r17, r6, r50 * r64);
    r64 = r0 * r36;
    r6 = fma(r51, r64, r6);
    r6 = fma(r41, r74, r6);
    r56 = r56 * r25;
    r56 = r56 * r6;
    r56 = fma(r36, r34, r33 * r56);
    r56 = fma(r41, r67, r56);
    r67 = r44 * r47;
    r67 = r67 * r27;
    r67 = r67 * r6;
    r67 = fma(r33, r67, r35 * r34);
    r34 = r4 * r41;
    r34 = r34 * r27;
    r34 = r34 * r50;
    r67 = fma(r31, r34, r67);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r56, r67);
    r34 = r4 * r3;
    r2 = r4 * r2;
    r34 = fma(r59, r2, r72 * r34);
    r31 = r4 * r3;
    r31 = fma(r57, r2, r79 * r31);
    WriteSum2<double, double>((double*)inout_shared, r34, r31);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r4 * r3;
    r31 = fma(r24, r2, r78 * r31);
    r34 = r4 * r3;
    r34 = fma(r22, r2, r30 * r34);
    WriteSum2<double, double>((double*)inout_shared, r31, r34);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r4 * r3;
    r34 = fma(r66, r2, r32 * r34);
    r31 = r4 * r3;
    r31 = fma(r56, r2, r67 * r31);
    WriteSum2<double, double>((double*)inout_shared, r34, r31);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r59, r59, r72 * r72);
    r34 = fma(r57, r57, r79 * r79);
    WriteSum2<double, double>((double*)inout_shared, r31, r34);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r24, r24, r78 * r78);
    r31 = fma(r22, r22, r30 * r30);
    WriteSum2<double, double>((double*)inout_shared, r34, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r32, r32, r66 * r66);
    r34 = fma(r56, r56, r67 * r67);
    WriteSum2<double, double>((double*)inout_shared, r31, r34);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r59, r57, r72 * r79);
    r31 = fma(r59, r24, r72 * r78);
    WriteSum2<double, double>((double*)inout_shared, r34, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r59, r22, r72 * r30);
    r34 = fma(r72, r32, r59 * r66);
    WriteSum2<double, double>((double*)inout_shared, r31, r34);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r59, r56, r72 * r67);
    r72 = fma(r79, r78, r57 * r24);
    WriteSum2<double, double>((double*)inout_shared, r59, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r79, r30, r57 * r22);
    r59 = fma(r79, r32, r57 * r66);
    WriteSum2<double, double>((double*)inout_shared, r72, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r79, r67, r57 * r56);
    r57 = fma(r24, r22, r78 * r30);
    WriteSum2<double, double>((double*)inout_shared, r79, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r24, r66, r78 * r32);
    r78 = fma(r78, r67, r24 * r56);
    WriteSum2<double, double>((double*)inout_shared, r57, r78);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r22, r66, r30 * r32);
    r30 = fma(r30, r67, r22 * r56);
    WriteSum2<double, double>((double*)inout_shared, r78, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r32, r67, r66 * r56);
    WriteSum1<double, double>((double*)inout_shared, r67);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r67);
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