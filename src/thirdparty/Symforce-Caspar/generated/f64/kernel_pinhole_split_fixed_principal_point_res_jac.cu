#include "kernel_pinhole_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
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
        double* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
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
      r76, r77, r78, r79, r80, r81, r82;

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
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r5);
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
    r17 = 2.00000000000000000e+00;
    r18 = fma(r9, r14, r12 * r11);
    r19 = r13 * r10;
    r18 = fma(r4, r19, r18);
    r18 = fma(r8, r15, r18);
    r19 = r17 * r18;
    r20 = r16 * r19;
    r21 = -2.00000000000000000e+00;
    r22 = fma(r13, r15, r12 * r14);
    r22 = fma(r8, r10, r22);
    r22 = fma(r4, r22, r9 * r11);
    r23 = r21 * r22;
    r24 = r9 * r15;
    r25 = fma(r13, r11, r24);
    r26 = r12 * r10;
    r27 = r8 * r14;
    r25 = r25 + r26;
    r25 = fma(r4, r27, r25);
    r28 = fma(r25, r23, r20);
    r5 = fma(r6, r28, r5);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r14 * r10;
    r31 = r31 * r17;
    r32 = r15 * r11;
    r32 = fma(r21, r32, r31);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r14 * r14;
    r34 = r34 * r21;
    r35 = 1.00000000000000000e+00;
    r36 = r15 * r15;
    r36 = fma(r21, r36, r35);
    r37 = r34 + r36;
    r38 = r15 * r10;
    r38 = r38 * r17;
    r39 = r14 * r11;
    r39 = fma(r17, r39, r38);
    r40 = r17 * r16;
    r40 = r40 * r25;
    r41 = fma(r22, r19, r40);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r21 * r25;
    r43 = r43 * r25;
    r44 = r35 + r43;
    r45 = r18 * r18;
    r45 = r45 * r21;
    r44 = r44 + r45;
    r5 = fma(r29, r32, r5);
    r5 = fma(r33, r37, r5);
    r5 = fma(r30, r39, r5);
    r5 = fma(r7, r41, r5);
    r5 = fma(r42, r44, r5);
    r46 = copysign(1.0, r5);
    r46 = fma(r0, r46, r5);
    r0 = 1.0 / r46;
  };
  LoadShared<2, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r5, r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r49);
    r43 = r35 + r43;
    r50 = r16 * r16;
    r50 = r50 * r21;
    r43 = r43 + r50;
    r48 = fma(r6, r43, r48);
    r51 = r25 * r19;
    r52 = fma(r16, r23, r51);
    r53 = r17 * r25;
    r53 = fma(r22, r53, r20);
    r20 = r15 * r11;
    r20 = fma(r17, r20, r31);
    r31 = r10 * r11;
    r54 = r14 * r15;
    r54 = r54 * r17;
    r31 = fma(r21, r31, r54);
    r55 = r10 * r10;
    r55 = r55 * r21;
    r36 = r55 + r36;
    r48 = fma(r7, r52, r48);
    r48 = fma(r42, r53, r48);
    r48 = fma(r33, r20, r48);
    r48 = fma(r30, r31, r48);
    r48 = fma(r29, r36, r48);
    r56 = r5 * r48;
    r2 = fma(r0, r56, r2);
    r3 = fma(r3, r4, r1);
    r1 = r17 * r16;
    r1 = fma(r22, r1, r51);
    r49 = fma(r6, r1, r49);
    r51 = r10 * r11;
    r51 = fma(r17, r51, r54);
    r55 = r35 + r55;
    r55 = r55 + r34;
    r34 = r14 * r11;
    r34 = fma(r21, r34, r38);
    r40 = fma(r18, r23, r40);
    r50 = r35 + r50;
    r50 = r50 + r45;
    r49 = fma(r29, r51, r49);
    r49 = fma(r30, r55, r49);
    r49 = fma(r33, r34, r49);
    r49 = fma(r42, r40, r49);
    r49 = fma(r7, r50, r49);
    r33 = r47 * r49;
    r3 = fma(r0, r33, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r30 = r17 * r25;
    r29 = -5.00000000000000000e-01;
    r45 = r13 * r29;
    r35 = 5.00000000000000000e-01;
    r38 = fma(r35, r27, r11 * r45);
    r38 = fma(r29, r24, r38);
    r38 = fma(r29, r26, r38);
    r30 = r30 * r38;
    r54 = r17 * r16;
    r57 = r13 * r14;
    r58 = r12 * r15;
    r58 = fma(r29, r58, r35 * r57);
    r57 = r9 * r10;
    r58 = fma(r35, r57, r58);
    r59 = r11 * r35;
    r58 = fma(r8, r59, r58);
    r54 = fma(r58, r54, r30);
    r57 = r17 * r22;
    r60 = r12 * r14;
    r61 = r8 * r10;
    r61 = fma(r29, r61, r29 * r60);
    r61 = fma(r9, r59, r61);
    r61 = fma(r15, r45, r61);
    r57 = r57 * r61;
    r60 = r12 * r11;
    r62 = r9 * r14;
    r62 = fma(r29, r62, r29 * r60);
    r60 = r8 * r15;
    r62 = fma(r29, r60, r62);
    r63 = r13 * r10;
    r62 = fma(r35, r63, r62);
    r63 = r62 * r19;
    r60 = r57 + r63;
    r64 = r54 + r60;
    r65 = r25 * r62;
    r66 = fma(r58, r23, r21 * r65);
    r67 = r17 * r16;
    r67 = r67 * r61;
    r68 = fma(r38, r19, r67);
    r66 = r66 + r68;
    r66 = fma(r6, r66, r7 * r64);
    r64 = r25 * r58;
    r69 = -4.00000000000000000e+00;
    r64 = r64 * r69;
    r70 = r61 * r69;
    r71 = r18 * r70;
    r72 = r64 + r71;
    r66 = fma(r42, r72, r66);
    r46 = r46 * r46;
    r46 = 1.0 / r46;
    r72 = r4 * r46;
    r56 = r72 * r56;
    r73 = r17 * r22;
    r73 = fma(r17, r65, r58 * r73);
    r73 = r73 + r68;
    r74 = r17 * r25;
    r74 = r74 * r61;
    r75 = r16 * r21;
    r75 = fma(r62, r75, r74);
    r58 = r58 * r19;
    r75 = r75 + r58;
    r75 = fma(r38, r23, r75);
    r75 = fma(r7, r75, r42 * r73);
    r73 = r16 * r38;
    r76 = r69 * r73;
    r64 = r64 + r76;
    r75 = fma(r6, r64, r75);
    r64 = r5 * r75;
    r64 = fma(r0, r64, r66 * r56);
    r77 = r18 * r21;
    r78 = r61 * r23;
    r77 = fma(r62, r77, r78);
    r77 = r77 + r54;
    r76 = r71 + r76;
    r76 = fma(r7, r76, r42 * r77);
    r77 = r17 * r22;
    r77 = fma(r38, r77, r58);
    r58 = r17 * r16;
    r58 = fma(r62, r58, r74);
    r77 = r77 + r58;
    r76 = fma(r6, r77, r76);
    r77 = r47 * r76;
    r74 = r66 * r72;
    r74 = fma(r33, r74, r0 * r77);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r64, r74);
    r77 = r21 * r25;
    r77 = fma(r38, r77, r78);
    r71 = r17 * r16;
    r54 = r8 * r11;
    r79 = r12 * r15;
    r79 = fma(r35, r79, r29 * r54);
    r54 = r9 * r10;
    r79 = fma(r29, r54, r79);
    r79 = fma(r14, r45, r79);
    r71 = r71 * r79;
    r54 = r9 * r14;
    r80 = r8 * r15;
    r80 = fma(r35, r80, r35 * r54);
    r80 = fma(r12, r59, r80);
    r80 = fma(r10, r45, r80);
    r45 = fma(r80, r19, r71);
    r77 = r77 + r45;
    r54 = r17 * r25;
    r54 = r54 * r80;
    r81 = r17 * r22;
    r81 = fma(r79, r81, r54);
    r81 = r81 + r68;
    r81 = fma(r7, r81, r6 * r77);
    r77 = r18 * r69;
    r77 = r77 * r79;
    r68 = r25 * r70;
    r82 = r77 + r68;
    r81 = fma(r42, r82, r81);
    r57 = r30 + r57;
    r57 = r57 + r45;
    r45 = r16 * r69;
    r45 = r45 * r80;
    r68 = r45 + r68;
    r68 = fma(r6, r68, r42 * r57);
    r57 = fma(r80, r23, r21 * r73);
    r30 = r17 * r25;
    r61 = r61 * r19;
    r30 = fma(r79, r30, r61);
    r57 = r57 + r30;
    r68 = fma(r7, r57, r68);
    r57 = r5 * r68;
    r57 = fma(r0, r57, r81 * r56);
    r82 = r81 * r72;
    r54 = r67 + r54;
    r67 = r18 * r21;
    r54 = fma(r38, r67, r54);
    r54 = fma(r79, r23, r54);
    r67 = r17 * r22;
    r73 = fma(r17, r73, r80 * r67);
    r73 = r73 + r30;
    r73 = fma(r6, r73, r42 * r54);
    r45 = r77 + r45;
    r73 = fma(r7, r45, r73);
    r45 = r47 * r73;
    r45 = fma(r0, r45, r33 * r82);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r57, r45);
    r82 = r18 * r69;
    r27 = fma(r29, r27, r13 * r59);
    r27 = fma(r35, r24, r27);
    r27 = fma(r35, r26, r27);
    r82 = r82 * r27;
    r65 = r69 * r65;
    r69 = r82 + r65;
    r26 = r17 * r16;
    r26 = r26 * r27;
    r35 = r21 * r25;
    r35 = fma(r79, r35, r26);
    r35 = r35 + r61;
    r35 = fma(r62, r23, r35);
    r35 = fma(r6, r35, r42 * r69);
    r69 = r17 * r22;
    r19 = fma(r79, r19, r27 * r69);
    r19 = r19 + r58;
    r35 = fma(r7, r19, r35);
    r19 = r17 * r25;
    r19 = r19 * r27;
    r69 = r16 * r21;
    r69 = fma(r79, r69, r19);
    r69 = r69 + r63;
    r69 = r69 + r78;
    r70 = r16 * r70;
    r65 = r65 + r70;
    r65 = fma(r6, r65, r7 * r69);
    r69 = r17 * r22;
    r69 = fma(r62, r69, r26);
    r69 = r69 + r30;
    r65 = fma(r42, r69, r65);
    r69 = r5 * r65;
    r69 = fma(r0, r69, r35 * r56);
    r19 = r71 + r19;
    r19 = r19 + r60;
    r60 = r18 * r21;
    r23 = fma(r27, r23, r79 * r60);
    r23 = r23 + r58;
    r23 = fma(r42, r23, r6 * r19);
    r70 = r82 + r70;
    r23 = fma(r7, r70, r23);
    r70 = r47 * r23;
    r7 = r35 * r72;
    r7 = fma(r33, r7, r0 * r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r69, r7);
    r70 = r5 * r36;
    r70 = fma(r32, r56, r0 * r70);
    r82 = r32 * r72;
    r42 = r47 * r51;
    r42 = fma(r0, r42, r33 * r82);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r70, r42);
    r82 = r5 * r31;
    r82 = fma(r0, r82, r39 * r56);
    r19 = r39 * r72;
    r6 = r47 * r55;
    r6 = fma(r0, r6, r33 * r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r82, r6);
    r19 = r5 * r20;
    r19 = fma(r0, r19, r37 * r56);
    r58 = r47 * r34;
    r27 = r37 * r72;
    r27 = fma(r33, r27, r0 * r58);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r19, r27);
    r58 = r4 * r2;
    r60 = r4 * r3;
    r60 = fma(r74, r60, r64 * r58);
    r58 = r4 * r3;
    r79 = r4 * r2;
    r79 = fma(r57, r79, r45 * r58);
    WriteSum2<double, double>((double*)inout_shared, r60, r79);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r4 * r2;
    r60 = r4 * r3;
    r60 = fma(r7, r60, r69 * r79);
    r79 = r4 * r3;
    r58 = r4 * r2;
    r58 = fma(r70, r58, r42 * r79);
    WriteSum2<double, double>((double*)inout_shared, r60, r58);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r4 * r3;
    r60 = r4 * r2;
    r60 = fma(r82, r60, r6 * r58);
    r58 = r4 * r2;
    r79 = r4 * r3;
    r79 = fma(r27, r79, r19 * r58);
    WriteSum2<double, double>((double*)inout_shared, r60, r79);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r74, r74, r64 * r64);
    r60 = fma(r57, r57, r45 * r45);
    WriteSum2<double, double>((double*)inout_shared, r79, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r69, r69, r7 * r7);
    r79 = fma(r70, r70, r42 * r42);
    WriteSum2<double, double>((double*)inout_shared, r60, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r6, r6, r82 * r82);
    r60 = fma(r27, r27, r19 * r19);
    WriteSum2<double, double>((double*)inout_shared, r79, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r64, r57, r74 * r45);
    r79 = fma(r64, r69, r74 * r7);
    WriteSum2<double, double>((double*)inout_shared, r60, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r74, r42, r64 * r70);
    r60 = fma(r64, r82, r74 * r6);
    WriteSum2<double, double>((double*)inout_shared, r79, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r74, r27, r64 * r19);
    r64 = fma(r45, r7, r57 * r69);
    WriteSum2<double, double>((double*)inout_shared, r74, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r57, r70, r45 * r42);
    r74 = fma(r57, r82, r45 * r6);
    WriteSum2<double, double>((double*)inout_shared, r64, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r57, r19, r45 * r27);
    r45 = fma(r7, r42, r69 * r70);
    WriteSum2<double, double>((double*)inout_shared, r57, r45);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r69, r82, r7 * r6);
    r7 = fma(r7, r27, r69 * r19);
    WriteSum2<double, double>((double*)inout_shared, r45, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r70, r82, r42 * r6);
    r42 = fma(r42, r27, r70 * r19);
    WriteSum2<double, double>((double*)inout_shared, r7, r42);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r6, r27, r82 * r19);
    WriteSum1<double, double>((double*)inout_shared, r27);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r48 * r0;
    r6 = r49 * r0;
    WriteIdx2<1024, double, double, double2>(
        out_focal_jac, 0 * out_focal_jac_num_alloc, global_thread_idx, r27, r6);
    r6 = r4 * r48;
    r6 = r6 * r2;
    r6 = r6 * r0;
    r27 = r4 * r49;
    r27 = r27 * r3;
    r27 = r27 * r0;
    WriteSum2<double, double>((double*)inout_shared, r6, r27);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r48 * r48;
    r48 = r48 * r46;
    r49 = r49 * r49;
    r49 = r49 * r46;
    WriteSum2<double, double>((double*)inout_shared, r48, r49);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r5 * r43;
    r49 = fma(r0, r49, r28 * r56);
    r48 = r47 * r1;
    r46 = r28 * r72;
    r46 = fma(r33, r46, r0 * r48);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r46);
    r48 = r5 * r52;
    r48 = fma(r0, r48, r41 * r56);
    r27 = r47 * r50;
    r6 = r41 * r72;
    r6 = fma(r33, r6, r0 * r27);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r48, r6);
    r27 = r5 * r53;
    r27 = fma(r0, r27, r44 * r56);
    r56 = r44 * r72;
    r19 = r47 * r40;
    r19 = fma(r0, r19, r33 * r56);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r27,
                                             r19);
    r56 = r4 * r3;
    r0 = r4 * r2;
    r0 = fma(r49, r0, r46 * r56);
    r56 = r4 * r3;
    r33 = r4 * r2;
    r33 = fma(r48, r33, r6 * r56);
    WriteSum2<double, double>((double*)inout_shared, r0, r33);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r19, r0, r27 * r33);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r49, r49, r46 * r46);
    r33 = fma(r48, r48, r6 * r6);
    WriteSum2<double, double>((double*)inout_shared, r0, r33);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r27, r27, r19 * r19);
    WriteSum1<double, double>((double*)inout_shared, r33);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r46, r6, r49 * r48);
    r46 = fma(r46, r19, r49 * r27);
    WriteSum2<double, double>((double*)inout_shared, r33, r46);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r6, r19, r48 * r27);
    WriteSum1<double, double>((double*)inout_shared, r19);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPrincipalPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
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
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
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
  PinholeSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
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
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
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