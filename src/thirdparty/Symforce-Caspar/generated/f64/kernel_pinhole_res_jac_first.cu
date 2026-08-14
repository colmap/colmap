#include "kernel_pinhole_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeResJacFirstKernel(double* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             double* sensor_from_rig,
                             unsigned int sensor_from_rig_num_alloc,
                             double* calib,
                             unsigned int calib_num_alloc,
                             SharedIndex* calib_indices,
                             double* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             double* pixel,
                             unsigned int pixel_num_alloc,
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
                             double* out_calib_jac,
                             unsigned int out_calib_jac_num_alloc,
                             double* const out_calib_njtr,
                             unsigned int out_calib_njtr_num_alloc,
                             double* const out_calib_precond_diag,
                             unsigned int out_calib_precond_diag_num_alloc,
                             double* const out_calib_precond_tril,
                             unsigned int out_calib_precond_tril_num_alloc,
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

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r5, r47);
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
    r30 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r30);
  if (global_thread_idx < problem_size) {
    r30 = r17 * r25;
    r29 = -5.00000000000000000e-01;
    r45 = r13 * r29;
    r38 = 5.00000000000000000e-01;
    r54 = fma(r38, r27, r11 * r45);
    r54 = fma(r29, r24, r54);
    r54 = fma(r29, r26, r54);
    r30 = r30 * r54;
    r57 = r17 * r16;
    r58 = r13 * r14;
    r59 = r12 * r15;
    r59 = fma(r29, r59, r38 * r58);
    r58 = r9 * r10;
    r59 = fma(r38, r58, r59);
    r60 = r11 * r38;
    r59 = fma(r8, r60, r59);
    r57 = fma(r59, r57, r30);
    r58 = r17 * r22;
    r61 = r12 * r14;
    r62 = r8 * r10;
    r62 = fma(r29, r62, r29 * r61);
    r62 = fma(r9, r60, r62);
    r62 = fma(r15, r45, r62);
    r58 = r58 * r62;
    r61 = r12 * r11;
    r63 = r9 * r14;
    r63 = fma(r29, r63, r29 * r61);
    r61 = r8 * r15;
    r63 = fma(r29, r61, r63);
    r64 = r13 * r10;
    r63 = fma(r38, r64, r63);
    r64 = r63 * r19;
    r61 = r58 + r64;
    r65 = r57 + r61;
    r66 = r25 * r63;
    r67 = fma(r59, r23, r21 * r66);
    r68 = r17 * r16;
    r68 = r68 * r62;
    r69 = fma(r54, r19, r68);
    r67 = r67 + r69;
    r67 = fma(r6, r67, r7 * r65);
    r65 = r25 * r59;
    r70 = -4.00000000000000000e+00;
    r65 = r65 * r70;
    r71 = r62 * r70;
    r72 = r18 * r71;
    r73 = r65 + r72;
    r67 = fma(r42, r73, r67);
    r46 = r46 * r46;
    r46 = 1.0 / r46;
    r73 = r4 * r46;
    r56 = r73 * r56;
    r74 = r17 * r22;
    r74 = fma(r17, r66, r59 * r74);
    r74 = r74 + r69;
    r75 = r17 * r25;
    r75 = r75 * r62;
    r76 = r16 * r21;
    r76 = fma(r63, r76, r75);
    r59 = r59 * r19;
    r76 = r76 + r59;
    r76 = fma(r54, r23, r76);
    r76 = fma(r7, r76, r42 * r74);
    r74 = r16 * r54;
    r77 = r70 * r74;
    r65 = r65 + r77;
    r76 = fma(r6, r65, r76);
    r65 = r5 * r76;
    r65 = fma(r0, r65, r67 * r56);
    r78 = r18 * r21;
    r79 = r62 * r23;
    r78 = fma(r63, r78, r79);
    r78 = r78 + r57;
    r77 = r72 + r77;
    r77 = fma(r7, r77, r42 * r78);
    r78 = r17 * r22;
    r78 = fma(r54, r78, r59);
    r59 = r17 * r16;
    r59 = fma(r63, r59, r75);
    r78 = r78 + r59;
    r77 = fma(r6, r78, r77);
    r78 = r47 * r77;
    r75 = r67 * r73;
    r75 = fma(r33, r75, r0 * r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r65, r75);
    r78 = r21 * r25;
    r78 = fma(r54, r78, r79);
    r72 = r17 * r16;
    r57 = r8 * r11;
    r80 = r12 * r15;
    r80 = fma(r38, r80, r29 * r57);
    r57 = r9 * r10;
    r80 = fma(r29, r57, r80);
    r80 = fma(r14, r45, r80);
    r72 = r72 * r80;
    r57 = r9 * r14;
    r81 = r8 * r15;
    r81 = fma(r38, r81, r38 * r57);
    r81 = fma(r12, r60, r81);
    r81 = fma(r10, r45, r81);
    r45 = fma(r81, r19, r72);
    r78 = r78 + r45;
    r57 = r17 * r25;
    r57 = r57 * r81;
    r82 = r17 * r22;
    r82 = fma(r80, r82, r57);
    r82 = r82 + r69;
    r82 = fma(r7, r82, r6 * r78);
    r78 = r18 * r70;
    r78 = r78 * r80;
    r69 = r25 * r71;
    r83 = r78 + r69;
    r82 = fma(r42, r83, r82);
    r58 = r30 + r58;
    r58 = r58 + r45;
    r45 = r16 * r70;
    r45 = r45 * r81;
    r69 = r45 + r69;
    r69 = fma(r6, r69, r42 * r58);
    r58 = fma(r81, r23, r21 * r74);
    r30 = r17 * r25;
    r62 = r62 * r19;
    r30 = fma(r80, r30, r62);
    r58 = r58 + r30;
    r69 = fma(r7, r58, r69);
    r58 = r5 * r69;
    r58 = fma(r0, r58, r82 * r56);
    r57 = r68 + r57;
    r68 = r18 * r21;
    r57 = fma(r54, r68, r57);
    r57 = fma(r80, r23, r57);
    r68 = r17 * r22;
    r74 = fma(r17, r74, r81 * r68);
    r74 = r74 + r30;
    r74 = fma(r6, r74, r42 * r57);
    r45 = r78 + r45;
    r74 = fma(r7, r45, r74);
    r45 = r47 * r74;
    r78 = r82 * r73;
    r78 = fma(r33, r78, r0 * r45);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r58, r78);
    r45 = r17 * r25;
    r27 = fma(r29, r27, r13 * r60);
    r27 = fma(r38, r24, r27);
    r27 = fma(r38, r26, r27);
    r45 = r45 * r27;
    r26 = r16 * r21;
    r26 = fma(r80, r26, r45);
    r26 = r26 + r64;
    r26 = r26 + r79;
    r71 = r16 * r71;
    r66 = r70 * r66;
    r79 = r71 + r66;
    r79 = fma(r6, r79, r7 * r26);
    r26 = r17 * r16;
    r26 = r26 * r27;
    r64 = r17 * r22;
    r64 = fma(r63, r64, r26);
    r64 = r64 + r30;
    r79 = fma(r42, r64, r79);
    r64 = r5 * r79;
    r70 = r18 * r70;
    r70 = r70 * r27;
    r66 = r70 + r66;
    r30 = r21 * r25;
    r30 = fma(r80, r30, r26);
    r30 = r30 + r62;
    r30 = fma(r63, r23, r30);
    r30 = fma(r6, r30, r42 * r66);
    r66 = r17 * r22;
    r19 = fma(r80, r19, r27 * r66);
    r19 = r19 + r59;
    r30 = fma(r7, r19, r30);
    r64 = fma(r30, r56, r0 * r64);
    r45 = r72 + r45;
    r45 = r45 + r61;
    r61 = r18 * r21;
    r23 = fma(r27, r23, r80 * r61);
    r23 = r23 + r59;
    r23 = fma(r42, r23, r6 * r45);
    r71 = r70 + r71;
    r23 = fma(r7, r71, r23);
    r71 = r47 * r23;
    r7 = r30 * r73;
    r7 = fma(r33, r7, r0 * r71);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r64, r7);
    r71 = r5 * r36;
    r71 = fma(r32, r56, r0 * r71);
    r70 = r47 * r51;
    r42 = r32 * r73;
    r42 = fma(r33, r42, r0 * r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r71, r42);
    r70 = r5 * r31;
    r70 = fma(r0, r70, r39 * r56);
    r45 = r39 * r73;
    r6 = r47 * r55;
    r6 = fma(r0, r6, r33 * r45);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r70, r6);
    r45 = r5 * r20;
    r45 = fma(r0, r45, r37 * r56);
    r59 = r37 * r73;
    r27 = r47 * r34;
    r27 = fma(r0, r27, r33 * r59);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r45, r27);
    r59 = r4 * r3;
    r61 = r4 * r2;
    r61 = fma(r65, r61, r75 * r59);
    r59 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r58, r80, r78 * r59);
    WriteSum2<double, double>((double*)inout_shared, r61, r80);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r4 * r3;
    r61 = r4 * r2;
    r61 = fma(r64, r61, r7 * r80);
    r80 = r4 * r2;
    r59 = r4 * r3;
    r59 = fma(r42, r59, r71 * r80);
    WriteSum2<double, double>((double*)inout_shared, r61, r59);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r4 * r3;
    r61 = r4 * r2;
    r61 = fma(r70, r61, r6 * r59);
    r59 = r4 * r2;
    r80 = r4 * r3;
    r80 = fma(r27, r80, r45 * r59);
    WriteSum2<double, double>((double*)inout_shared, r61, r80);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r75, r75, r65 * r65);
    r61 = fma(r58, r58, r78 * r78);
    WriteSum2<double, double>((double*)inout_shared, r80, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r64, r64, r7 * r7);
    r80 = fma(r42, r42, r71 * r71);
    WriteSum2<double, double>((double*)inout_shared, r61, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r70, r70, r6 * r6);
    r61 = fma(r27, r27, r45 * r45);
    WriteSum2<double, double>((double*)inout_shared, r80, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r65, r58, r75 * r78);
    r80 = fma(r65, r64, r75 * r7);
    WriteSum2<double, double>((double*)inout_shared, r61, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r65, r71, r75 * r42);
    r61 = fma(r65, r70, r75 * r6);
    WriteSum2<double, double>((double*)inout_shared, r80, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r75, r27, r65 * r45);
    r65 = fma(r58, r64, r78 * r7);
    WriteSum2<double, double>((double*)inout_shared, r75, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r78, r42, r58 * r71);
    r75 = fma(r58, r70, r78 * r6);
    WriteSum2<double, double>((double*)inout_shared, r65, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r78, r27, r58 * r45);
    r58 = fma(r7, r42, r64 * r71);
    WriteSum2<double, double>((double*)inout_shared, r78, r58);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r7, r6, r64 * r70);
    r64 = fma(r64, r45, r7 * r27);
    WriteSum2<double, double>((double*)inout_shared, r58, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r42, r6, r71 * r70);
    r42 = fma(r42, r27, r71 * r45);
    WriteSum2<double, double>((double*)inout_shared, r64, r42);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r70, r45, r6 * r27);
    WriteSum1<double, double>((double*)inout_shared, r45);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r48 * r0;
    r70 = r49 * r0;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r45,
                                             r70);
    r27 = r4 * r48;
    r27 = r27 * r2;
    r27 = r27 * r0;
    r6 = r4 * r49;
    r6 = r6 * r3;
    r6 = r6 * r0;
    WriteSum2<double, double>((double*)inout_shared, r27, r6);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r2;
    r27 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r6, r27);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r48 * r48;
    r48 = r48 * r46;
    r49 = r49 * r49;
    r49 = r49 * r46;
    WriteSum2<double, double>((double*)inout_shared, r48, r49);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r35, r35);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r35, r45);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r70, r35);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r5 * r43;
    r35 = fma(r28, r56, r0 * r35);
    r70 = r28 * r73;
    r45 = r47 * r1;
    r45 = fma(r0, r45, r33 * r70);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r35,
                                             r45);
    r70 = r5 * r52;
    r70 = fma(r0, r70, r41 * r56);
    r49 = r47 * r50;
    r48 = r41 * r73;
    r48 = fma(r33, r48, r0 * r49);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r70,
                                             r48);
    r49 = r5 * r53;
    r49 = fma(r0, r49, r44 * r56);
    r56 = r47 * r40;
    r46 = r44 * r73;
    r46 = fma(r33, r46, r0 * r56);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r46);
    r56 = r4 * r2;
    r33 = r4 * r3;
    r33 = fma(r45, r33, r35 * r56);
    r56 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r48, r0, r70 * r56);
    WriteSum2<double, double>((double*)inout_shared, r33, r0);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r4 * r3;
    r33 = r4 * r2;
    r33 = fma(r49, r33, r46 * r0);
    WriteSum1<double, double>((double*)inout_shared, r33);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r45, r45, r35 * r35);
    r0 = fma(r48, r48, r70 * r70);
    WriteSum2<double, double>((double*)inout_shared, r33, r0);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r46, r46, r49 * r49);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r35, r70, r45 * r48);
    r45 = fma(r45, r46, r35 * r49);
    WriteSum2<double, double>((double*)inout_shared, r0, r45);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r48, r46, r70 * r49);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeResJacFirst(double* pose,
                        unsigned int pose_num_alloc,
                        SharedIndex* pose_indices,
                        double* sensor_from_rig,
                        unsigned int sensor_from_rig_num_alloc,
                        double* calib,
                        unsigned int calib_num_alloc,
                        SharedIndex* calib_indices,
                        double* point,
                        unsigned int point_num_alloc,
                        SharedIndex* point_indices,
                        double* pixel,
                        unsigned int pixel_num_alloc,
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
                        double* out_calib_jac,
                        unsigned int out_calib_jac_num_alloc,
                        double* const out_calib_njtr,
                        unsigned int out_calib_njtr_num_alloc,
                        double* const out_calib_precond_diag,
                        unsigned int out_calib_precond_diag_num_alloc,
                        double* const out_calib_precond_tril,
                        unsigned int out_calib_precond_tril_num_alloc,
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
  PinholeResJacFirstKernel<<<n_blocks, 1024>>>(pose,
                                               pose_num_alloc,
                                               pose_indices,
                                               sensor_from_rig,
                                               sensor_from_rig_num_alloc,
                                               calib,
                                               calib_num_alloc,
                                               calib_indices,
                                               point,
                                               point_num_alloc,
                                               point_indices,
                                               pixel,
                                               pixel_num_alloc,
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
                                               out_calib_jac,
                                               out_calib_jac_num_alloc,
                                               out_calib_njtr,
                                               out_calib_njtr_num_alloc,
                                               out_calib_precond_diag,
                                               out_calib_precond_diag_num_alloc,
                                               out_calib_precond_tril,
                                               out_calib_precond_tril_num_alloc,
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