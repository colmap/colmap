#include "kernel_spherical_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalResJacFirstKernel(double* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               double* sensor_from_rig,
                               unsigned int sensor_from_rig_num_alloc,
                               double* wh,
                               unsigned int wh_num_alloc,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = r13 * r12;
    r22 = fma(r10, r15, r21);
    r23 = r14 * r11;
    r24 = r9 * r16;
    r22 = r22 + r23;
    r22 = fma(r18, r24, r22);
    r25 = r8 * r22;
    r25 = fma(r22, r25, r20);
    r26 = r19 + r25;
    r4 = fma(r6, r26, r4);
    r27 = fma(r14, r15, r9 * r12);
    r28 = r10 * r11;
    r27 = fma(r18, r28, r27);
    r27 = fma(r13, r16, r27);
    r28 = 2.00000000000000000e+00;
    r29 = r28 * r17;
    r30 = r27 * r29;
    r31 = fma(r10, r16, r9 * r15);
    r31 = fma(r13, r11, r31);
    r31 = fma(r18, r31, r14 * r12);
    r32 = r8 * r31;
    r33 = fma(r22, r32, r30);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r22 * r28;
    r35 = r35 * r27;
    r36 = fma(r31, r29, r35);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r15 * r11;
    r38 = r38 * r28;
    r39 = r16 * r12;
    r39 = fma(r28, r39, r38);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r40, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r11 * r12;
    r43 = r15 * r16;
    r43 = r43 * r28;
    r42 = fma(r8, r42, r43);
    r44 = r11 * r11;
    r44 = r44 * r8;
    r45 = r20 + r44;
    r46 = r16 * r16;
    r46 = r46 * r8;
    r45 = r45 + r46;
    r4 = fma(r7, r33, r4);
    r4 = fma(r34, r36, r4);
    r4 = fma(r37, r39, r4);
    r4 = fma(r41, r42, r4);
    r4 = fma(r40, r45, r4);
    r47 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r48);
    r35 = fma(r17, r32, r35);
    r48 = fma(r6, r35, r48);
    r49 = r16 * r12;
    r49 = fma(r8, r49, r38);
    r46 = r20 + r46;
    r38 = r15 * r15;
    r38 = r8 * r38;
    r46 = r46 + r38;
    r50 = r16 * r11;
    r50 = r50 * r28;
    r51 = r15 * r12;
    r51 = fma(r28, r51, r50);
    r52 = r28 * r27;
    r53 = r22 * r29;
    r52 = fma(r31, r52, r53);
    r19 = r20 + r19;
    r54 = r8 * r27;
    r54 = r54 * r27;
    r19 = r19 + r54;
    r48 = fma(r40, r49, r48);
    r48 = fma(r37, r46, r48);
    r48 = fma(r41, r51, r48);
    r48 = fma(r7, r52, r48);
    r48 = fma(r34, r19, r48);
    r55 = copysign(r47, r48);
    r55 = r55 + r48;
    r56 = atan2(r4, r55);
    r56 = fma(r3, r56, r2);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r57, r58);
    r57 = fma(r57, r18, r0 * r56);
    r56 = -3.18309886183790691e-01;
    r59 = r22 * r28;
    r59 = fma(r31, r59, r30);
    r5 = fma(r6, r59, r5);
    r30 = r11 * r12;
    r30 = fma(r28, r30, r43);
    r44 = r20 + r44;
    r44 = r44 + r38;
    r38 = r15 * r12;
    r38 = fma(r8, r38, r50);
    r53 = fma(r27, r32, r53);
    r25 = r54 + r25;
    r5 = fma(r40, r30, r5);
    r5 = fma(r41, r44, r5);
    r5 = fma(r37, r38, r5);
    r5 = fma(r34, r53, r5);
    r5 = fma(r7, r25, r5);
    r37 = r18 * r5;
    r41 = r4 * r4;
    r40 = r47 + r41;
    r40 = fma(r48, r48, r40);
    r54 = sqrt(r40);
    r47 = copysign(r47, r54);
    r54 = r47 + r54;
    r37 = atan2(r37, r54);
    r37 = fma(r56, r37, r2);
    r58 = fma(r58, r18, r1 * r37);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r57, r58);
    r37 = fma(r58, r58, r57 * r57);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r37);
  if (global_thread_idx < problem_size) {
    r37 = 1.0 / r55;
    r47 = r28 * r31;
    r50 = r10 * r15;
    r50 = fma(r2, r21, r2 * r50);
    r20 = -5.00000000000000000e-01;
    r50 = fma(r20, r24, r50);
    r50 = fma(r2, r23, r50);
    r43 = r9 * r12;
    r60 = r14 * r15;
    r60 = fma(r20, r60, r20 * r43);
    r43 = r13 * r16;
    r60 = fma(r20, r43, r60);
    r61 = r10 * r11;
    r60 = fma(r2, r61, r60);
    r47 = fma(r60, r29, r50 * r47);
    r61 = r28 * r27;
    r43 = r13 * r15;
    r62 = r14 * r16;
    r62 = fma(r20, r62, r2 * r43);
    r43 = r9 * r11;
    r62 = fma(r20, r43, r62);
    r63 = r10 * r20;
    r62 = fma(r12, r63, r62);
    r43 = r22 * r28;
    r64 = r14 * r12;
    r65 = r9 * r15;
    r65 = fma(r20, r65, r2 * r64);
    r64 = r13 * r11;
    r65 = fma(r20, r64, r65);
    r65 = fma(r16, r63, r65);
    r43 = r43 * r65;
    r61 = fma(r62, r61, r43);
    r47 = r47 + r61;
    r64 = r28 * r27;
    r64 = r64 * r50;
    r66 = r8 * r22;
    r66 = fma(r60, r66, r64);
    r67 = r65 * r29;
    r66 = r66 + r67;
    r66 = fma(r62, r32, r66);
    r66 = fma(r7, r66, r34 * r47);
    r47 = r17 * r50;
    r68 = -4.00000000000000000e+00;
    r47 = r47 * r68;
    r69 = r22 * r62;
    r70 = r68 * r69;
    r71 = r47 + r70;
    r66 = fma(r6, r71, r66);
    r71 = r22 * r28;
    r72 = r62 * r29;
    r71 = fma(r50, r71, r72);
    r73 = r28 * r27;
    r73 = r73 * r60;
    r74 = r28 * r31;
    r74 = r74 * r65;
    r75 = r73 + r74;
    r76 = r71 + r75;
    r77 = r8 * r17;
    r50 = fma(r50, r32, r60 * r77);
    r50 = r50 + r61;
    r50 = fma(r6, r50, r7 * r76);
    r76 = r27 * r68;
    r77 = r65 * r76;
    r47 = r47 + r77;
    r50 = fma(r34, r47, r50);
    r47 = r18 * r4;
    r55 = r55 * r55;
    r78 = 1.0 / r55;
    r47 = r47 * r78;
    r78 = fma(r50, r47, r66 * r37);
    r79 = r3 * r78;
    r41 = r41 + r55;
    r80 = 1.0 / r41;
    r81 = r0 * r55;
    r79 = r79 * r80;
    r79 = r79 * r81;
    r82 = r28 * r4;
    r83 = r28 * r48;
    r83 = fma(r50, r83, r66 * r82);
    r82 = r2 * r5;
    r50 = r54 * r54;
    r66 = 1.0 / r50;
    r40 = rsqrt(r40);
    r82 = r82 * r66;
    r82 = r82 * r40;
    r40 = r8 * r27;
    r66 = r65 * r32;
    r40 = fma(r60, r40, r66);
    r40 = r40 + r71;
    r77 = r70 + r77;
    r77 = fma(r7, r77, r34 * r40);
    r40 = r28 * r31;
    r40 = fma(r62, r40, r64);
    r64 = r22 * r28;
    r64 = fma(r60, r64, r67);
    r40 = r40 + r64;
    r77 = fma(r6, r40, r77);
    r40 = r18 * r77;
    r54 = 1.0 / r54;
    r40 = fma(r54, r40, r83 * r82);
    r83 = r56 * r40;
    r5 = fma(r5, r5, r50);
    r67 = 1.0 / r5;
    r70 = r1 * r50;
    r83 = r83 * r67;
    r83 = r83 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r79, r83);
    r83 = r8 * r17;
    r83 = fma(r62, r83, r66);
    r79 = r22 * r28;
    r21 = fma(r15, r63, r20 * r21);
    r21 = fma(r2, r24, r21);
    r21 = fma(r20, r23, r21);
    r79 = r79 * r21;
    r23 = r28 * r27;
    r24 = r9 * r12;
    r71 = r14 * r15;
    r71 = fma(r2, r71, r2 * r24);
    r24 = r13 * r16;
    r71 = fma(r2, r24, r71);
    r71 = fma(r11, r63, r71);
    r23 = fma(r71, r23, r79);
    r83 = r83 + r23;
    r63 = r28 * r31;
    r24 = r71 * r29;
    r63 = fma(r21, r63, r24);
    r63 = r63 + r61;
    r63 = fma(r7, r63, r6 * r83);
    r83 = r17 * r65;
    r83 = r83 * r68;
    r61 = r21 * r76;
    r84 = r83 + r61;
    r63 = fma(r34, r84, r63);
    r72 = r74 + r72;
    r72 = r72 + r23;
    r23 = r22 * r68;
    r23 = r23 * r71;
    r83 = r83 + r23;
    r83 = fma(r6, r83, r34 * r72);
    r72 = fma(r71, r32, r8 * r69);
    r74 = r28 * r27;
    r74 = r74 * r65;
    r84 = fma(r21, r29, r74);
    r72 = r72 + r84;
    r83 = fma(r7, r72, r83);
    r72 = fma(r83, r37, r63 * r47);
    r85 = r3 * r72;
    r85 = r85 * r80;
    r85 = r85 * r81;
    r86 = r8 * r27;
    r86 = fma(r62, r86, r43);
    r86 = r86 + r24;
    r86 = fma(r21, r32, r86);
    r24 = r28 * r31;
    r69 = fma(r28, r69, r71 * r24);
    r69 = r69 + r84;
    r69 = fma(r6, r69, r34 * r86);
    r61 = r23 + r61;
    r69 = fma(r7, r61, r69);
    r61 = r18 * r69;
    r23 = r28 * r4;
    r86 = r28 * r48;
    r86 = fma(r63, r86, r83 * r23);
    r86 = fma(r86, r82, r54 * r61);
    r61 = r56 * r86;
    r61 = r61 * r67;
    r61 = r61 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r85, r61);
    r61 = r17 * r60;
    r61 = r61 * r68;
    r85 = r10 * r12;
    r23 = r13 * r15;
    r23 = fma(r20, r23, r2 * r85);
    r85 = r14 * r16;
    r23 = fma(r2, r85, r23);
    r20 = r9 * r11;
    r23 = fma(r2, r20, r23);
    r76 = r23 * r76;
    r20 = r61 + r76;
    r85 = r22 * r28;
    r85 = r85 * r23;
    r74 = r74 + r85;
    r2 = r8 * r17;
    r74 = fma(r21, r2, r74);
    r74 = fma(r60, r32, r74);
    r74 = fma(r6, r74, r34 * r20);
    r20 = r28 * r27;
    r2 = r28 * r31;
    r2 = fma(r23, r2, r21 * r20);
    r2 = r2 + r64;
    r74 = fma(r7, r2, r74);
    r2 = r8 * r22;
    r2 = fma(r21, r2, r73);
    r29 = r23 * r29;
    r2 = r2 + r66;
    r2 = r2 + r29;
    r65 = r22 * r65;
    r65 = r65 * r68;
    r61 = r61 + r65;
    r61 = fma(r6, r61, r7 * r2);
    r2 = r28 * r31;
    r2 = fma(r60, r2, r85);
    r2 = r2 + r84;
    r61 = fma(r34, r2, r61);
    r2 = fma(r61, r37, r74 * r47);
    r84 = r3 * r2;
    r84 = r84 * r80;
    r84 = r84 * r81;
    r85 = r28 * r48;
    r60 = r28 * r4;
    r60 = fma(r61, r60, r74 * r85);
    r29 = r79 + r29;
    r29 = r29 + r75;
    r75 = r8 * r27;
    r32 = fma(r23, r32, r21 * r75);
    r32 = r32 + r64;
    r32 = fma(r34, r32, r6 * r29);
    r76 = r65 + r76;
    r32 = fma(r7, r76, r32);
    r76 = r18 * r32;
    r76 = fma(r54, r76, r60 * r82);
    r60 = r56 * r76;
    r60 = r60 * r67;
    r60 = r60 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r84, r60);
    r60 = fma(r45, r37, r49 * r47);
    r84 = r3 * r60;
    r84 = r84 * r80;
    r84 = r84 * r81;
    r7 = r28 * r45;
    r65 = r28 * r49;
    r65 = fma(r48, r65, r4 * r7);
    r7 = r18 * r30;
    r7 = fma(r54, r7, r65 * r82);
    r65 = r56 * r7;
    r65 = r65 * r67;
    r65 = r65 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r84, r65);
    r65 = fma(r42, r37, r51 * r47);
    r84 = r3 * r65;
    r84 = r84 * r80;
    r84 = r84 * r81;
    r34 = r28 * r42;
    r29 = r28 * r51;
    r29 = fma(r48, r29, r4 * r34);
    r34 = r18 * r44;
    r34 = fma(r54, r34, r29 * r82);
    r29 = r56 * r34;
    r29 = r29 * r67;
    r29 = r29 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r84, r29);
    r29 = fma(r39, r37, r46 * r47);
    r84 = r3 * r29;
    r84 = r84 * r80;
    r84 = r84 * r81;
    r6 = r28 * r39;
    r64 = r28 * r46;
    r64 = fma(r48, r64, r4 * r6);
    r6 = r18 * r38;
    r6 = fma(r54, r6, r64 * r82);
    r64 = r56 * r6;
    r64 = r64 * r67;
    r64 = r64 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r84, r64);
    r64 = 3.18309886183790691e-01;
    r64 = r58 * r64;
    r64 = r64 * r67;
    r64 = r64 * r70;
    r58 = -1.59154943091895346e-01;
    r58 = r57 * r58;
    r58 = r58 * r80;
    r58 = r58 * r81;
    r57 = fma(r78, r58, r40 * r64);
    r84 = fma(r86, r64, r72 * r58);
    WriteSum2<double, double>((double*)inout_shared, r57, r84);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = fma(r76, r64, r2 * r58);
    r57 = fma(r60, r58, r7 * r64);
    WriteSum2<double, double>((double*)inout_shared, r84, r57);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r65, r58, r34 * r64);
    r84 = fma(r29, r58, r6 * r64);
    WriteSum2<double, double>((double*)inout_shared, r57, r84);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r78 * r78;
    r57 = 2.53302959105844473e-02;
    r57 = r0 * r57;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r57 = r57 * r41;
    r57 = r57 * r81;
    r57 = r57 * r55;
    r55 = r40 * r40;
    r41 = 1.01321183642337789e-01;
    r41 = r1 * r41;
    r5 = r5 * r5;
    r5 = 1.0 / r5;
    r41 = r41 * r5;
    r41 = r41 * r70;
    r41 = r41 * r50;
    r55 = fma(r41, r55, r57 * r84);
    r84 = r86 * r86;
    r50 = r72 * r57;
    r72 = fma(r72, r50, r41 * r84);
    WriteSum2<double, double>((double*)inout_shared, r55, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r76 * r41;
    r55 = r2 * r2;
    r55 = fma(r57, r55, r76 * r72);
    r76 = r7 * r7;
    r84 = r60 * r60;
    r84 = fma(r57, r84, r41 * r76);
    WriteSum2<double, double>((double*)inout_shared, r55, r84);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r65 * r65;
    r55 = r34 * r34;
    r55 = fma(r41, r55, r57 * r84);
    r84 = r29 * r29;
    r76 = r6 * r6;
    r76 = fma(r41, r76, r57 * r84);
    WriteSum2<double, double>((double*)inout_shared, r55, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r40 * r86;
    r76 = fma(r41, r76, r78 * r50);
    r55 = r78 * r2;
    r55 = fma(r40, r72, r57 * r55);
    WriteSum2<double, double>((double*)inout_shared, r76, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r78 * r60;
    r76 = r40 * r7;
    r76 = fma(r41, r76, r57 * r55);
    r55 = r40 * r34;
    r84 = r78 * r65;
    r84 = fma(r57, r84, r41 * r55);
    WriteSum2<double, double>((double*)inout_shared, r76, r84);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r78 * r29;
    r76 = r40 * r6;
    r76 = fma(r41, r76, r57 * r84);
    r84 = fma(r86, r72, r2 * r50);
    WriteSum2<double, double>((double*)inout_shared, r76, r84);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r86 * r7;
    r84 = fma(r60, r50, r41 * r84);
    r76 = r86 * r34;
    r76 = fma(r65, r50, r41 * r76);
    WriteSum2<double, double>((double*)inout_shared, r84, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r86 * r6;
    r76 = fma(r41, r76, r29 * r50);
    r50 = r2 * r60;
    r50 = fma(r57, r50, r7 * r72);
    WriteSum2<double, double>((double*)inout_shared, r76, r50);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r2 * r65;
    r50 = fma(r34, r72, r57 * r50);
    r76 = r2 * r29;
    r72 = fma(r6, r72, r57 * r76);
    WriteSum2<double, double>((double*)inout_shared, r50, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r60 * r65;
    r50 = r7 * r34;
    r50 = fma(r41, r50, r57 * r72);
    r72 = r7 * r6;
    r76 = r60 * r29;
    r76 = fma(r57, r76, r41 * r72);
    WriteSum2<double, double>((double*)inout_shared, r50, r76);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r65 * r29;
    r50 = r34 * r6;
    r50 = fma(r41, r50, r57 * r76);
    WriteSum1<double, double>((double*)inout_shared, r50);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r26, r37, r35 * r47);
    r76 = r3 * r50;
    r76 = r76 * r80;
    r76 = r76 * r81;
    r72 = r28 * r26;
    r84 = r28 * r35;
    r84 = fma(r48, r84, r4 * r72);
    r72 = r18 * r59;
    r72 = fma(r54, r72, r84 * r82);
    r84 = r56 * r72;
    r84 = r84 * r67;
    r84 = r84 * r70;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r76,
                                             r84);
    r84 = fma(r33, r37, r52 * r47);
    r76 = r3 * r84;
    r76 = r76 * r80;
    r76 = r76 * r81;
    r55 = r28 * r52;
    r5 = r28 * r33;
    r5 = fma(r4, r5, r48 * r55);
    r55 = r18 * r25;
    r55 = fma(r54, r55, r5 * r82);
    r5 = r56 * r55;
    r5 = r5 * r67;
    r5 = r5 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r76, r5);
    r47 = fma(r19, r47, r36 * r37);
    r3 = r3 * r47;
    r3 = r3 * r80;
    r3 = r3 * r81;
    r81 = r18 * r53;
    r80 = r28 * r36;
    r37 = r28 * r19;
    r37 = fma(r48, r37, r4 * r80);
    r82 = fma(r37, r82, r54 * r81);
    r56 = r56 * r82;
    r56 = r56 * r67;
    r56 = r56 * r70;
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r3, r56);
    r56 = fma(r50, r58, r72 * r64);
    r3 = fma(r84, r58, r55 * r64);
    WriteSum2<double, double>((double*)inout_shared, r56, r3);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r82, r64, r47 * r58);
    WriteSum1<double, double>((double*)inout_shared, r64);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r50 * r50;
    r58 = r72 * r72;
    r58 = fma(r41, r58, r57 * r64);
    r64 = r84 * r84;
    r3 = r55 * r55;
    r3 = fma(r41, r3, r57 * r64);
    WriteSum2<double, double>((double*)inout_shared, r58, r3);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r82 * r82;
    r58 = r47 * r47;
    r58 = fma(r57, r58, r41 * r3);
    WriteSum1<double, double>((double*)inout_shared, r58);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r72 * r55;
    r3 = r50 * r84;
    r3 = fma(r57, r3, r41 * r58);
    r58 = r72 * r82;
    r64 = r50 * r47;
    r64 = fma(r57, r64, r41 * r58);
    WriteSum2<double, double>((double*)inout_shared, r3, r64);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r55 * r82;
    r3 = r84 * r47;
    r3 = fma(r57, r3, r41 * r64);
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SphericalResJacFirst(double* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          double* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          double* wh,
                          unsigned int wh_num_alloc,
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
  SphericalResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      wh,
      wh_num_alloc,
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