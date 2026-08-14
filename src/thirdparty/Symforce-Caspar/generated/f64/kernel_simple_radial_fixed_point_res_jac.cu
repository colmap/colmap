#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFixedPointResJacKernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* point,
    unsigned int point_num_alloc,
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
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82;
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
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r32, r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r45);
    r46 = r18 * r20;
    r46 = fma(r28, r46, r30);
    r46 = fma(r6, r46, r45);
    r35 = fma(r18, r35, r34);
    r41 = r19 + r41;
    r34 = r14 * r14;
    r34 = r34 * r18;
    r41 = r41 + r34;
    r45 = r15 * r10;
    r45 = r45 * r0;
    r30 = r14 * r11;
    r30 = fma(r0, r30, r45);
    r47 = r0 * r16;
    r47 = r47 * r20;
    r48 = fma(r25, r31, r47);
    r49 = r25 * r25;
    r49 = r49 * r18;
    r23 = r49 + r23;
    r46 = fma(r37, r35, r46);
    r46 = fma(r33, r41, r46);
    r46 = fma(r38, r30, r46);
    r46 = fma(r7, r48, r46);
    r46 = fma(r29, r23, r46);
    r23 = copysign(1.0, r46);
    r23 = fma(r44, r23, r46);
    r44 = r23 * r23;
    r46 = 1.0 / r44;
    r48 = r24 * r46;
    r26 = fma(r16, r31, r26);
    r26 = fma(r6, r26, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r34;
    r34 = r14 * r11;
    r34 = fma(r18, r34, r45);
    r45 = r25 * r18;
    r45 = fma(r28, r45, r47);
    r17 = r19 + r17;
    r17 = r17 + r49;
    r26 = fma(r37, r5, r26);
    r26 = fma(r38, r43, r26);
    r26 = fma(r33, r34, r26);
    r26 = fma(r29, r45, r26);
    r26 = fma(r7, r17, r26);
    r17 = r26 * r26;
    r45 = fma(r46, r17, r24 * r48);
    r33 = fma(r27, r45, r19);
    r38 = r24 * r33;
    r37 = 1.0 / r23;
    r49 = r32 * r37;
    r2 = fma(r49, r38, r2);
    r3 = fma(r3, r4, r1);
    r1 = r26 * r33;
    r3 = fma(r49, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r12 * r11;
    r38 = -5.00000000000000000e-01;
    r47 = r9 * r14;
    r47 = fma(r38, r47, r38 * r1);
    r1 = r8 * r15;
    r47 = fma(r38, r1, r47);
    r40 = r13 * r10;
    r50 = 5.00000000000000000e-01;
    r47 = fma(r50, r40, r47);
    r40 = r20 * r47;
    r1 = r8 * r11;
    r51 = r13 * r14;
    r51 = fma(r50, r51, r50 * r1);
    r1 = r12 * r15;
    r51 = fma(r38, r1, r51);
    r52 = r9 * r50;
    r51 = fma(r10, r52, r51);
    r1 = fma(r51, r31, r0 * r40);
    r53 = r0 * r25;
    r54 = r9 * r15;
    r55 = r13 * r38;
    r54 = fma(r11, r55, r38 * r54);
    r54 = fma(r50, r22, r54);
    r54 = fma(r38, r21, r54);
    r56 = r0 * r16;
    r57 = r12 * r14;
    r58 = r8 * r10;
    r58 = fma(r38, r58, r38 * r57);
    r58 = fma(r11, r52, r58);
    r58 = fma(r15, r55, r58);
    r56 = r56 * r58;
    r53 = fma(r54, r53, r56);
    r1 = r1 + r53;
    r57 = r0 * r20;
    r57 = r57 * r58;
    r59 = r0 * r25;
    r59 = r59 * r51;
    r60 = r57 + r59;
    r61 = r16 * r18;
    r60 = fma(r47, r61, r60);
    r62 = r18 * r28;
    r60 = fma(r54, r62, r60);
    r60 = fma(r7, r60, r29 * r1);
    r1 = r20 * r51;
    r62 = -4.00000000000000000e+00;
    r1 = r1 * r62;
    r61 = r16 * r54;
    r63 = r62 * r61;
    r64 = r1 + r63;
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
    r71 = fma(r18, r40, r51 * r71);
    r71 = r71 + r53;
    r71 = fma(r6, r71, r7 * r70);
    r70 = r25 * r62;
    r51 = r58 * r70;
    r1 = r1 + r51;
    r71 = fma(r29, r1, r71);
    r1 = r24 * r24;
    r44 = r23 * r44;
    r44 = 1.0 / r44;
    r44 = r18 * r44;
    r1 = r1 * r44;
    r64 = fma(r71, r1, r48 * r64);
    r23 = r71 * r44;
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
    r64 = fma(r46, r72, r64);
    r72 = r24 * r64;
    r27 = r27 * r49;
    r23 = r71 * r48;
    r59 = r4 * r33;
    r57 = r32 * r59;
    r23 = fma(r57, r23, r27 * r72);
    r72 = r33 * r60;
    r23 = fma(r49, r72, r23);
    r72 = r33 * r51;
    r63 = r26 * r64;
    r63 = fma(r27, r63, r49 * r72);
    r72 = r26 * r46;
    r72 = r72 * r57;
    r63 = fma(r71, r72, r63);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r23, r63);
    r66 = r18 * r20;
    r66 = fma(r54, r66, r74);
    r75 = r0 * r16;
    r76 = r8 * r11;
    r77 = r12 * r15;
    r77 = fma(r50, r77, r38 * r76);
    r76 = r9 * r10;
    r77 = fma(r38, r76, r77);
    r77 = fma(r14, r55, r77);
    r75 = r75 * r77;
    r76 = r0 * r25;
    r78 = r12 * r11;
    r79 = r8 * r15;
    r79 = fma(r50, r79, r50 * r78);
    r79 = fma(r14, r52, r79);
    r79 = fma(r10, r55, r79);
    r76 = fma(r79, r76, r75);
    r66 = r66 + r76;
    r55 = r0 * r20;
    r55 = r55 * r79;
    r78 = fma(r77, r31, r55);
    r78 = r78 + r53;
    r78 = fma(r7, r78, r6 * r66);
    r66 = r20 * r58;
    r66 = r66 * r62;
    r53 = r77 * r70;
    r80 = r66 + r53;
    r78 = fma(r29, r80, r78);
    r80 = r78 * r48;
    r68 = r65 + r68;
    r68 = r68 + r76;
    r76 = r16 * r62;
    r76 = r76 * r79;
    r66 = r66 + r76;
    r66 = fma(r6, r66, r29 * r68);
    r68 = r18 * r28;
    r68 = fma(r18, r61, r79 * r68);
    r65 = r0 * r25;
    r65 = r65 * r58;
    r81 = r0 * r20;
    r81 = fma(r77, r81, r65);
    r68 = r68 + r81;
    r66 = fma(r7, r68, r66);
    r68 = r33 * r66;
    r68 = fma(r49, r68, r57 * r80);
    r80 = r0 * r66;
    r82 = r0 * r26;
    r55 = r56 + r55;
    r56 = r25 * r18;
    r55 = fma(r54, r56, r55);
    r54 = r18 * r28;
    r55 = fma(r77, r54, r55);
    r79 = fma(r79, r31, r0 * r61);
    r79 = r79 + r81;
    r79 = fma(r6, r79, r29 * r55);
    r53 = r76 + r53;
    r79 = fma(r7, r53, r79);
    r82 = r82 * r79;
    r82 = fma(r46, r82, r48 * r80);
    r80 = r78 * r44;
    r82 = fma(r17, r80, r82);
    r82 = fma(r78, r1, r82);
    r80 = r24 * r82;
    r68 = fma(r27, r80, r68);
    r80 = r26 * r82;
    r80 = fma(r78, r72, r27 * r80);
    r53 = r33 * r79;
    r80 = fma(r49, r53, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r68, r80);
    r74 = r67 + r74;
    r67 = r0 * r20;
    r53 = r13 * r11;
    r22 = fma(r38, r22, r50 * r53);
    r22 = fma(r15, r52, r22);
    r22 = fma(r50, r21, r22);
    r67 = r67 * r22;
    r21 = r16 * r18;
    r74 = fma(r77, r21, r74);
    r74 = r74 + r67;
    r58 = r16 * r58;
    r58 = r58 * r62;
    r40 = r62 * r40;
    r62 = r58 + r40;
    r62 = fma(r6, r62, r7 * r74);
    r74 = r0 * r16;
    r74 = r74 * r22;
    r21 = fma(r47, r31, r74);
    r21 = r21 + r81;
    r62 = fma(r29, r21, r62);
    r21 = r33 * r62;
    r70 = r22 * r70;
    r40 = r40 + r70;
    r74 = r65 + r74;
    r65 = r18 * r20;
    r74 = fma(r77, r65, r74);
    r81 = r18 * r28;
    r74 = fma(r47, r81, r74);
    r74 = fma(r6, r74, r29 * r40);
    r40 = r0 * r25;
    r31 = fma(r22, r31, r77 * r40);
    r31 = r31 + r73;
    r74 = fma(r7, r31, r74);
    r31 = r74 * r48;
    r31 = fma(r57, r31, r49 * r21);
    r21 = r0 * r26;
    r67 = r75 + r67;
    r67 = r67 + r69;
    r69 = r25 * r18;
    r75 = r18 * r28;
    r75 = fma(r22, r75, r77 * r69);
    r75 = r75 + r73;
    r75 = fma(r29, r75, r6 * r67);
    r70 = r58 + r70;
    r75 = fma(r7, r70, r75);
    r21 = r21 * r75;
    r21 = fma(r74, r1, r46 * r21);
    r70 = r0 * r62;
    r21 = fma(r48, r70, r21);
    r7 = r74 * r44;
    r21 = fma(r17, r7, r21);
    r7 = r24 * r21;
    r31 = fma(r27, r7, r31);
    r7 = r26 * r21;
    r7 = fma(r74, r72, r27 * r7);
    r70 = r33 * r75;
    r7 = fma(r49, r70, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r7);
    r70 = r42 * r33;
    r58 = r35 * r48;
    r58 = fma(r57, r58, r49 * r70);
    r70 = r0 * r5;
    r70 = r70 * r26;
    r29 = r35 * r44;
    r29 = fma(r17, r29, r46 * r70);
    r70 = r0 * r42;
    r29 = fma(r48, r70, r29);
    r29 = fma(r35, r1, r29);
    r70 = r24 * r29;
    r58 = fma(r27, r70, r58);
    r70 = r26 * r29;
    r70 = fma(r35, r72, r27 * r70);
    r67 = r5 * r33;
    r70 = fma(r49, r67, r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r58, r70);
    r67 = r30 * r44;
    r6 = r0 * r43;
    r6 = r6 * r26;
    r6 = fma(r46, r6, r17 * r67);
    r67 = r0 * r39;
    r6 = fma(r48, r67, r6);
    r6 = fma(r30, r1, r6);
    r6 = r6 * r27;
    r67 = r30 * r48;
    r67 = fma(r57, r67, r24 * r6);
    r73 = r39 * r33;
    r67 = fma(r49, r73, r67);
    r73 = r43 * r33;
    r6 = fma(r26, r6, r49 * r73);
    r6 = fma(r30, r72, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r67, r6);
    r73 = r0 * r34;
    r73 = r73 * r26;
    r69 = r41 * r44;
    r69 = fma(r17, r69, r46 * r73);
    r73 = r0 * r36;
    r69 = fma(r48, r73, r69);
    r69 = fma(r41, r1, r69);
    r1 = r24 * r69;
    r73 = r41 * r48;
    r73 = fma(r57, r73, r27 * r1);
    r1 = r36 * r33;
    r73 = fma(r49, r1, r73);
    r1 = r34 * r33;
    r72 = fma(r41, r72, r49 * r1);
    r1 = r26 * r69;
    r72 = fma(r27, r1, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r73, r72);
    r1 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r63, r27, r23 * r1);
    r1 = r4 * r2;
    r57 = r4 * r3;
    r57 = fma(r80, r57, r68 * r1);
    WriteSum2<double, double>((double*)inout_shared, r27, r57);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r31, r27, r7 * r57);
    r57 = r4 * r2;
    r1 = r4 * r3;
    r1 = fma(r70, r1, r58 * r57);
    WriteSum2<double, double>((double*)inout_shared, r27, r1);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r67, r27, r6 * r1);
    r1 = r4 * r2;
    r57 = r4 * r3;
    r57 = fma(r72, r57, r73 * r1);
    WriteSum2<double, double>((double*)inout_shared, r27, r57);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r63, r63, r23 * r23);
    r27 = fma(r68, r68, r80 * r80);
    WriteSum2<double, double>((double*)inout_shared, r57, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r7, r7, r31 * r31);
    r57 = fma(r70, r70, r58 * r58);
    WriteSum2<double, double>((double*)inout_shared, r27, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r67, r67, r6 * r6);
    r27 = fma(r72, r72, r73 * r73);
    WriteSum2<double, double>((double*)inout_shared, r57, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r63, r80, r23 * r68);
    r57 = fma(r63, r7, r23 * r31);
    WriteSum2<double, double>((double*)inout_shared, r27, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r23, r58, r63 * r70);
    r27 = fma(r23, r67, r63 * r6);
    WriteSum2<double, double>((double*)inout_shared, r57, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r63, r72, r23 * r73);
    r23 = fma(r80, r7, r68 * r31);
    WriteSum2<double, double>((double*)inout_shared, r63, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r68, r58, r80 * r70);
    r63 = fma(r80, r6, r68 * r67);
    WriteSum2<double, double>((double*)inout_shared, r23, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r80, r72, r68 * r73);
    r68 = fma(r7, r70, r31 * r58);
    WriteSum2<double, double>((double*)inout_shared, r80, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r31, r67, r7 * r6);
    r31 = fma(r31, r73, r7 * r72);
    WriteSum2<double, double>((double*)inout_shared, r68, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r58, r67, r70 * r6);
    r70 = fma(r70, r72, r58 * r73);
    WriteSum2<double, double>((double*)inout_shared, r31, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r67, r73, r6 * r72);
    WriteSum1<double, double>((double*)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r24 * r33;
    r73 = r73 * r37;
    r67 = r26 * r33;
    r67 = r67 * r37;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r73,
                                             r67);
    r72 = r24 * r45;
    r72 = r72 * r49;
    r6 = r26 * r45;
    r6 = r6 * r49;
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 2 * out_calib_jac_num_alloc, global_thread_idx, r72, r6);
    r70 = r26 * r3;
    r70 = r70 * r37;
    r31 = r24 * r2;
    r31 = r31 * r37;
    r31 = fma(r59, r31, r59 * r70);
    r70 = r4 * r26;
    r70 = r70 * r45;
    r70 = r70 * r3;
    r59 = r4 * r24;
    r59 = r59 * r45;
    r59 = r59 * r2;
    r59 = fma(r49, r59, r49 * r70);
    WriteSum2<double, double>((double*)inout_shared, r31, r59);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r4 * r2;
    r31 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r59, r31);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r24 * r33;
    r31 = r31 * r33;
    r59 = r33 * r46;
    r59 = r59 * r17;
    r31 = fma(r33, r59, r48 * r31);
    r70 = r46 * r17;
    r49 = r32 * r32;
    r37 = r45 * r45;
    r49 = r49 * r37;
    r37 = r24 * r48;
    r37 = fma(r49, r37, r49 * r70);
    WriteSum2<double, double>((double*)inout_shared, r31, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r19, r19);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r32 * r24;
    r19 = r19 * r45;
    r19 = r19 * r33;
    r37 = r32 * r45;
    r37 = fma(r59, r37, r48 * r19);
    WriteSum2<double, double>((double*)inout_shared, r37, r73);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r67, r72);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r6, r72);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialFixedPointResJac(double* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  double* sensor_from_rig,
                                  unsigned int sensor_from_rig_num_alloc,
                                  double* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  double* pixel,
                                  unsigned int pixel_num_alloc,
                                  double* point,
                                  unsigned int point_num_alloc,
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
                                  double* out_calib_jac,
                                  unsigned int out_calib_jac_num_alloc,
                                  double* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  double* const out_calib_precond_diag,
                                  unsigned int out_calib_precond_diag_num_alloc,
                                  double* const out_calib_precond_tril,
                                  unsigned int out_calib_precond_tril_num_alloc,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar