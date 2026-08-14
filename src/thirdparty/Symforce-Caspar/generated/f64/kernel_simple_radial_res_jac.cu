#include "kernel_simple_radial_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialResJacKernel(double* pose,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91;
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
    r20 = fma(r9, r15, r13 * r11);
    r21 = r12 * r10;
    r22 = r8 * r14;
    r20 = r20 + r21;
    r20 = fma(r4, r22, r20);
    r23 = r18 * r20;
    r23 = fma(r20, r23, r19);
    r24 = r17 + r23;
    r0 = fma(r6, r24, r0);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r9, r14, r12 * r11);
    r27 = r13 * r10;
    r26 = fma(r4, r27, r26);
    r26 = fma(r8, r15, r26);
    r27 = r25 * r26;
    r27 = r27 * r20;
    r28 = r16 * r18;
    r29 = fma(r13, r15, r12 * r14);
    r29 = fma(r8, r10, r29);
    r29 = fma(r4, r29, r9 * r11);
    r28 = fma(r29, r28, r27);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r25 * r16;
    r31 = r31 * r26;
    r32 = r25 * r29;
    r33 = fma(r20, r32, r31);
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
    r35 = r35 * r25;
    r36 = r15 * r11;
    r37 = fma(r25, r36, r35);
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
    r41 = r41 * r25;
    r40 = fma(r18, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r18;
    r43 = r19 + r42;
    r44 = r10 * r10;
    r44 = r44 * r18;
    r43 = r43 + r44;
    r0 = fma(r7, r28, r0);
    r0 = fma(r30, r33, r0);
    r0 = fma(r34, r37, r0);
    r0 = fma(r39, r40, r0);
    r0 = fma(r38, r43, r0);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r45, r46);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r47 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r48);
    r49 = r18 * r20;
    r49 = fma(r29, r49, r31);
    r48 = fma(r6, r49, r48);
    r36 = fma(r18, r36, r35);
    r42 = r19 + r42;
    r35 = r14 * r14;
    r35 = r35 * r18;
    r42 = r42 + r35;
    r31 = r15 * r10;
    r31 = r31 * r25;
    r50 = r14 * r11;
    r50 = fma(r25, r50, r31);
    r51 = r25 * r16;
    r51 = r51 * r20;
    r52 = fma(r26, r32, r51);
    r53 = r26 * r26;
    r53 = r53 * r18;
    r23 = r53 + r23;
    r48 = fma(r38, r36, r48);
    r48 = fma(r34, r42, r48);
    r48 = fma(r39, r50, r48);
    r48 = fma(r7, r52, r48);
    r48 = fma(r30, r23, r48);
    r54 = copysign(1.0, r48);
    r54 = fma(r47, r54, r48);
    r47 = r54 * r54;
    r48 = 1.0 / r47;
    r55 = r0 * r48;
    r27 = fma(r16, r32, r27);
    r5 = fma(r6, r27, r5);
    r56 = r10 * r11;
    r56 = fma(r25, r56, r41);
    r44 = r19 + r44;
    r44 = r44 + r35;
    r35 = r14 * r11;
    r35 = fma(r18, r35, r31);
    r31 = r26 * r18;
    r31 = fma(r29, r31, r51);
    r17 = r19 + r17;
    r17 = r17 + r53;
    r5 = fma(r38, r56, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r34, r35, r5);
    r5 = fma(r30, r31, r5);
    r5 = fma(r7, r17, r5);
    r34 = r5 * r5;
    r39 = fma(r48, r34, r0 * r55);
    r38 = fma(r46, r39, r19);
    r53 = r0 * r38;
    r51 = 1.0 / r54;
    r41 = r45 * r51;
    r2 = fma(r41, r53, r2);
    r3 = fma(r3, r4, r1);
    r1 = r5 * r38;
    r3 = fma(r41, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r12 * r11;
    r53 = -5.00000000000000000e-01;
    r57 = r9 * r14;
    r57 = fma(r53, r57, r53 * r1);
    r1 = r8 * r15;
    r57 = fma(r53, r1, r57);
    r58 = r13 * r10;
    r59 = 5.00000000000000000e-01;
    r57 = fma(r59, r58, r57);
    r58 = r20 * r57;
    r1 = r8 * r11;
    r60 = r13 * r14;
    r60 = fma(r59, r60, r59 * r1);
    r1 = r12 * r15;
    r60 = fma(r53, r1, r60);
    r61 = r9 * r59;
    r60 = fma(r10, r61, r60);
    r1 = fma(r60, r32, r25 * r58);
    r62 = r25 * r26;
    r63 = r9 * r15;
    r64 = r13 * r53;
    r63 = fma(r11, r64, r53 * r63);
    r63 = fma(r59, r22, r63);
    r63 = fma(r53, r21, r63);
    r65 = r25 * r16;
    r66 = r12 * r14;
    r67 = r8 * r10;
    r67 = fma(r53, r67, r53 * r66);
    r67 = fma(r11, r61, r67);
    r67 = fma(r15, r64, r67);
    r65 = r65 * r67;
    r62 = fma(r63, r62, r65);
    r1 = r1 + r62;
    r66 = r25 * r20;
    r66 = r66 * r67;
    r68 = r25 * r26;
    r68 = r68 * r60;
    r69 = r66 + r68;
    r70 = r16 * r18;
    r69 = fma(r57, r70, r69);
    r71 = r18 * r29;
    r69 = fma(r63, r71, r69);
    r69 = fma(r7, r69, r30 * r1);
    r1 = r20 * r60;
    r71 = -4.00000000000000000e+00;
    r1 = r1 * r71;
    r70 = r16 * r63;
    r72 = r71 * r70;
    r73 = r1 + r72;
    r69 = fma(r6, r73, r69);
    r73 = r25 * r69;
    r74 = r25 * r20;
    r74 = r74 * r63;
    r75 = r25 * r16;
    r75 = fma(r60, r75, r74);
    r76 = r25 * r26;
    r76 = r76 * r57;
    r77 = r67 * r32;
    r78 = r76 + r77;
    r79 = r75 + r78;
    r80 = r18 * r29;
    r80 = fma(r18, r58, r60 * r80);
    r80 = r80 + r62;
    r80 = fma(r6, r80, r7 * r79);
    r79 = r26 * r71;
    r60 = r67 * r79;
    r1 = r1 + r60;
    r80 = fma(r30, r1, r80);
    r1 = r0 * r0;
    r47 = r54 * r47;
    r47 = 1.0 / r47;
    r47 = r18 * r47;
    r1 = r1 * r47;
    r73 = fma(r80, r1, r55 * r73);
    r54 = r80 * r47;
    r73 = fma(r34, r54, r73);
    r81 = r25 * r5;
    r82 = r26 * r18;
    r83 = r18 * r29;
    r83 = r83 * r67;
    r82 = fma(r57, r82, r83);
    r82 = r82 + r75;
    r60 = r72 + r60;
    r60 = fma(r7, r60, r30 * r82);
    r68 = fma(r63, r32, r68);
    r82 = r25 * r16;
    r82 = fma(r57, r82, r66);
    r68 = r68 + r82;
    r60 = fma(r6, r68, r60);
    r81 = r81 * r60;
    r73 = fma(r48, r81, r73);
    r81 = r0 * r73;
    r46 = r46 * r41;
    r54 = r80 * r55;
    r68 = r4 * r38;
    r66 = r45 * r68;
    r54 = fma(r66, r54, r46 * r81);
    r81 = r38 * r69;
    r54 = fma(r41, r81, r54);
    r81 = r38 * r60;
    r72 = r5 * r73;
    r72 = fma(r46, r72, r41 * r81);
    r81 = r5 * r48;
    r81 = r81 * r66;
    r72 = fma(r80, r81, r72);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r54, r72);
    r75 = r18 * r20;
    r75 = fma(r63, r75, r83);
    r84 = r25 * r16;
    r85 = r8 * r11;
    r86 = r12 * r15;
    r86 = fma(r59, r86, r53 * r85);
    r85 = r9 * r10;
    r86 = fma(r53, r85, r86);
    r86 = fma(r14, r64, r86);
    r84 = r84 * r86;
    r85 = r25 * r26;
    r87 = r12 * r11;
    r88 = r8 * r15;
    r88 = fma(r59, r88, r59 * r87);
    r88 = fma(r14, r61, r88);
    r88 = fma(r10, r64, r88);
    r85 = fma(r88, r85, r84);
    r75 = r75 + r85;
    r64 = r25 * r20;
    r64 = r64 * r88;
    r87 = fma(r86, r32, r64);
    r87 = r87 + r62;
    r87 = fma(r7, r87, r6 * r75);
    r75 = r20 * r67;
    r75 = r75 * r71;
    r62 = r86 * r79;
    r89 = r75 + r62;
    r87 = fma(r30, r89, r87);
    r89 = r87 * r55;
    r77 = r74 + r77;
    r77 = r77 + r85;
    r85 = r16 * r71;
    r85 = r85 * r88;
    r75 = r75 + r85;
    r75 = fma(r6, r75, r30 * r77);
    r77 = r18 * r29;
    r77 = fma(r18, r70, r88 * r77);
    r74 = r25 * r26;
    r74 = r74 * r67;
    r90 = r25 * r20;
    r90 = fma(r86, r90, r74);
    r77 = r77 + r90;
    r75 = fma(r7, r77, r75);
    r77 = r38 * r75;
    r77 = fma(r41, r77, r66 * r89);
    r89 = r25 * r75;
    r91 = r25 * r5;
    r64 = r65 + r64;
    r65 = r26 * r18;
    r64 = fma(r63, r65, r64);
    r63 = r18 * r29;
    r64 = fma(r86, r63, r64);
    r88 = fma(r88, r32, r25 * r70);
    r88 = r88 + r90;
    r88 = fma(r6, r88, r30 * r64);
    r62 = r85 + r62;
    r88 = fma(r7, r62, r88);
    r91 = r91 * r88;
    r91 = fma(r48, r91, r55 * r89);
    r89 = r87 * r47;
    r91 = fma(r34, r89, r91);
    r91 = fma(r87, r1, r91);
    r89 = r0 * r91;
    r77 = fma(r46, r89, r77);
    r89 = r5 * r91;
    r89 = fma(r87, r81, r46 * r89);
    r62 = r38 * r88;
    r89 = fma(r41, r62, r89);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r77, r89);
    r83 = r76 + r83;
    r76 = r25 * r20;
    r62 = r13 * r11;
    r22 = fma(r53, r22, r59 * r62);
    r22 = fma(r15, r61, r22);
    r22 = fma(r59, r21, r22);
    r76 = r76 * r22;
    r21 = r16 * r18;
    r83 = fma(r86, r21, r83);
    r83 = r83 + r76;
    r67 = r16 * r67;
    r67 = r67 * r71;
    r58 = r71 * r58;
    r71 = r67 + r58;
    r71 = fma(r6, r71, r7 * r83);
    r83 = r25 * r16;
    r83 = r83 * r22;
    r21 = fma(r57, r32, r83);
    r21 = r21 + r90;
    r71 = fma(r30, r21, r71);
    r21 = r38 * r71;
    r79 = r22 * r79;
    r58 = r58 + r79;
    r83 = r74 + r83;
    r74 = r18 * r20;
    r83 = fma(r86, r74, r83);
    r90 = r18 * r29;
    r83 = fma(r57, r90, r83);
    r83 = fma(r6, r83, r30 * r58);
    r58 = r25 * r26;
    r32 = fma(r22, r32, r86 * r58);
    r32 = r32 + r82;
    r83 = fma(r7, r32, r83);
    r32 = r83 * r55;
    r32 = fma(r66, r32, r41 * r21);
    r21 = r25 * r5;
    r76 = r84 + r76;
    r76 = r76 + r78;
    r78 = r26 * r18;
    r84 = r18 * r29;
    r84 = fma(r22, r84, r86 * r78);
    r84 = r84 + r82;
    r84 = fma(r30, r84, r6 * r76);
    r79 = r67 + r79;
    r84 = fma(r7, r79, r84);
    r21 = r21 * r84;
    r21 = fma(r83, r1, r48 * r21);
    r79 = r25 * r71;
    r21 = fma(r55, r79, r21);
    r7 = r83 * r47;
    r21 = fma(r34, r7, r21);
    r7 = r0 * r21;
    r32 = fma(r46, r7, r32);
    r7 = r5 * r21;
    r7 = fma(r83, r81, r46 * r7);
    r79 = r38 * r84;
    r7 = fma(r41, r79, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r7);
    r79 = r43 * r38;
    r67 = r36 * r55;
    r67 = fma(r66, r67, r41 * r79);
    r79 = r25 * r56;
    r79 = r79 * r5;
    r30 = r36 * r47;
    r30 = fma(r34, r30, r48 * r79);
    r79 = r25 * r43;
    r30 = fma(r55, r79, r30);
    r30 = fma(r36, r1, r30);
    r79 = r0 * r30;
    r67 = fma(r46, r79, r67);
    r79 = r5 * r30;
    r79 = fma(r36, r81, r46 * r79);
    r76 = r56 * r38;
    r79 = fma(r41, r76, r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r67, r79);
    r76 = r50 * r47;
    r6 = r25 * r44;
    r6 = r6 * r5;
    r6 = fma(r48, r6, r34 * r76);
    r76 = r25 * r40;
    r6 = fma(r55, r76, r6);
    r6 = fma(r50, r1, r6);
    r76 = r0 * r6;
    r82 = r50 * r55;
    r82 = fma(r66, r82, r46 * r76);
    r76 = r40 * r38;
    r82 = fma(r41, r76, r82);
    r76 = r44 * r38;
    r78 = r5 * r6;
    r78 = fma(r46, r78, r41 * r76);
    r78 = fma(r50, r81, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r82, r78);
    r76 = r25 * r35;
    r76 = r76 * r5;
    r22 = r42 * r47;
    r22 = fma(r34, r22, r48 * r76);
    r76 = r25 * r37;
    r22 = fma(r55, r76, r22);
    r22 = fma(r42, r1, r22);
    r76 = r0 * r22;
    r86 = r42 * r55;
    r86 = fma(r66, r86, r46 * r76);
    r76 = r37 * r38;
    r86 = fma(r41, r76, r86);
    r76 = r35 * r38;
    r76 = fma(r42, r81, r41 * r76);
    r58 = r5 * r22;
    r76 = fma(r46, r58, r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r86, r76);
    r58 = r4 * r2;
    r90 = r4 * r3;
    r90 = fma(r72, r90, r54 * r58);
    r58 = r4 * r2;
    r74 = r4 * r3;
    r74 = fma(r89, r74, r77 * r58);
    WriteSum2<double, double>((double*)inout_shared, r90, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r4 * r3;
    r90 = r4 * r2;
    r90 = fma(r32, r90, r7 * r74);
    r74 = r4 * r2;
    r58 = r4 * r3;
    r58 = fma(r79, r58, r67 * r74);
    WriteSum2<double, double>((double*)inout_shared, r90, r58);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r4 * r3;
    r90 = r4 * r2;
    r90 = fma(r82, r90, r78 * r58);
    r58 = r4 * r2;
    r74 = r4 * r3;
    r74 = fma(r76, r74, r86 * r58);
    WriteSum2<double, double>((double*)inout_shared, r90, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r72, r72, r54 * r54);
    r90 = fma(r77, r77, r89 * r89);
    WriteSum2<double, double>((double*)inout_shared, r74, r90);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = fma(r7, r7, r32 * r32);
    r74 = fma(r79, r79, r67 * r67);
    WriteSum2<double, double>((double*)inout_shared, r90, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r82, r82, r78 * r78);
    r90 = fma(r76, r76, r86 * r86);
    WriteSum2<double, double>((double*)inout_shared, r74, r90);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = fma(r72, r89, r54 * r77);
    r74 = fma(r72, r7, r54 * r32);
    WriteSum2<double, double>((double*)inout_shared, r90, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r54, r67, r72 * r79);
    r90 = fma(r54, r82, r72 * r78);
    WriteSum2<double, double>((double*)inout_shared, r74, r90);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r72, r76, r54 * r86);
    r54 = fma(r89, r7, r77 * r32);
    WriteSum2<double, double>((double*)inout_shared, r72, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r77, r67, r89 * r79);
    r72 = fma(r89, r78, r77 * r82);
    WriteSum2<double, double>((double*)inout_shared, r54, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = fma(r89, r76, r77 * r86);
    r77 = fma(r7, r79, r32 * r67);
    WriteSum2<double, double>((double*)inout_shared, r89, r77);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r32, r82, r7 * r78);
    r32 = fma(r32, r86, r7 * r76);
    WriteSum2<double, double>((double*)inout_shared, r77, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r67, r82, r79 * r78);
    r79 = fma(r79, r76, r67 * r86);
    WriteSum2<double, double>((double*)inout_shared, r32, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r82, r86, r78 * r76);
    WriteSum1<double, double>((double*)inout_shared, r86);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r0 * r38;
    r86 = r86 * r51;
    r82 = r5 * r38;
    r82 = r82 * r51;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r86,
                                             r82);
    r76 = r0 * r39;
    r76 = r76 * r41;
    r78 = r5 * r39;
    r78 = r78 * r41;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r76,
                                             r78);
    r79 = r5 * r3;
    r79 = r79 * r51;
    r32 = r0 * r2;
    r32 = r32 * r51;
    r32 = fma(r68, r32, r68 * r79);
    r79 = r4 * r5;
    r79 = r79 * r39;
    r79 = r79 * r3;
    r68 = r4 * r0;
    r68 = r68 * r39;
    r68 = r68 * r2;
    r68 = fma(r41, r68, r41 * r79);
    WriteSum2<double, double>((double*)inout_shared, r32, r68);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r4 * r2;
    r32 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r68, r32);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r0 * r38;
    r32 = r32 * r38;
    r68 = r38 * r48;
    r68 = r68 * r34;
    r32 = fma(r38, r68, r55 * r32);
    r79 = r48 * r34;
    r51 = r45 * r45;
    r67 = r39 * r39;
    r51 = r51 * r67;
    r67 = r0 * r55;
    r67 = fma(r51, r67, r51 * r79);
    WriteSum2<double, double>((double*)inout_shared, r32, r67);
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
    r19 = r45 * r0;
    r19 = r19 * r39;
    r19 = r19 * r38;
    r67 = r45 * r39;
    r67 = fma(r68, r67, r55 * r19);
    WriteSum2<double, double>((double*)inout_shared, r67, r86);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r82, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r78, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r49 * r55;
    r78 = r24 * r38;
    r78 = fma(r41, r78, r66 * r76);
    r76 = r25 * r24;
    r82 = r49 * r47;
    r82 = fma(r34, r82, r55 * r76);
    r76 = r25 * r27;
    r76 = r76 * r5;
    r82 = fma(r48, r76, r82);
    r82 = fma(r49, r1, r82);
    r82 = r82 * r46;
    r78 = fma(r0, r82, r78);
    r76 = r27 * r38;
    r76 = fma(r41, r76, r5 * r82);
    r76 = fma(r49, r81, r76);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r78,
                                             r76);
    r82 = r28 * r38;
    r86 = r25 * r28;
    r86 = fma(r55, r86, r52 * r1);
    r67 = r25 * r17;
    r67 = r67 * r5;
    r86 = fma(r48, r67, r86);
    r19 = r52 * r47;
    r86 = fma(r34, r19, r86);
    r19 = r0 * r86;
    r19 = fma(r46, r19, r41 * r82);
    r82 = r52 * r55;
    r19 = fma(r66, r82, r19);
    r82 = r17 * r38;
    r82 = fma(r41, r82, r52 * r81);
    r67 = r5 * r86;
    r82 = fma(r46, r67, r82);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r19,
                                             r82);
    r67 = r25 * r33;
    r68 = r23 * r47;
    r68 = fma(r34, r68, r55 * r67);
    r67 = r25 * r31;
    r67 = r67 * r5;
    r68 = fma(r48, r67, r68);
    r68 = fma(r23, r1, r68);
    r67 = r0 * r68;
    r1 = r33 * r38;
    r1 = fma(r41, r1, r46 * r67);
    r67 = r23 * r55;
    r1 = fma(r66, r67, r1);
    r67 = r31 * r38;
    r66 = r5 * r68;
    r66 = fma(r46, r66, r41 * r67);
    r66 = fma(r23, r81, r66);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r66);
    r81 = r4 * r3;
    r67 = r4 * r2;
    r67 = fma(r78, r67, r76 * r81);
    r81 = r4 * r2;
    r46 = r4 * r3;
    r46 = fma(r82, r46, r19 * r81);
    WriteSum2<double, double>((double*)inout_shared, r67, r46);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r2;
    r67 = r4 * r3;
    r67 = fma(r66, r67, r1 * r46);
    WriteSum1<double, double>((double*)inout_shared, r67);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r78, r78, r76 * r76);
    r46 = fma(r19, r19, r82 * r82);
    WriteSum2<double, double>((double*)inout_shared, r67, r46);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r1, r1, r66 * r66);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r76, r82, r78 * r19);
    r76 = fma(r76, r66, r78 * r1);
    WriteSum2<double, double>((double*)inout_shared, r46, r76);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r19, r1, r82 * r66);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialResJac(double* pose,
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
  SimpleRadialResJacKernel<<<n_blocks, 1024>>>(pose,
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