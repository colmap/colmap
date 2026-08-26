#include "kernel_simple_radial_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialResJacFirstKernel(double* pose,
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
    r1 = fma(r2, r2, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
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
    r73 = r38 * r69;
    r74 = r25 * r69;
    r75 = r25 * r20;
    r75 = r75 * r63;
    r76 = r25 * r16;
    r76 = fma(r60, r76, r75);
    r77 = r25 * r26;
    r77 = r77 * r57;
    r78 = r67 * r32;
    r79 = r77 + r78;
    r80 = r76 + r79;
    r81 = r18 * r29;
    r81 = fma(r18, r58, r60 * r81);
    r81 = r81 + r62;
    r81 = fma(r6, r81, r7 * r80);
    r80 = r26 * r71;
    r60 = r67 * r80;
    r1 = r1 + r60;
    r81 = fma(r30, r1, r81);
    r1 = r0 * r0;
    r47 = r54 * r47;
    r47 = 1.0 / r47;
    r47 = r18 * r47;
    r1 = r1 * r47;
    r74 = fma(r81, r1, r55 * r74);
    r54 = r81 * r47;
    r74 = fma(r34, r54, r74);
    r82 = r25 * r5;
    r83 = r26 * r18;
    r84 = r18 * r29;
    r84 = r84 * r67;
    r83 = fma(r57, r83, r84);
    r83 = r83 + r76;
    r60 = r72 + r60;
    r60 = fma(r7, r60, r30 * r83);
    r68 = fma(r63, r32, r68);
    r83 = r25 * r16;
    r83 = fma(r57, r83, r66);
    r68 = r68 + r83;
    r60 = fma(r6, r68, r60);
    r82 = r82 * r60;
    r74 = fma(r48, r82, r74);
    r82 = r0 * r74;
    r46 = r46 * r41;
    r82 = fma(r46, r82, r41 * r73);
    r73 = r81 * r55;
    r54 = r4 * r38;
    r68 = r45 * r54;
    r82 = fma(r68, r73, r82);
    r73 = r5 * r74;
    r66 = r5 * r48;
    r66 = r66 * r68;
    r73 = fma(r81, r66, r46 * r73);
    r72 = r38 * r60;
    r73 = fma(r41, r72, r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r82, r73);
    r78 = r75 + r78;
    r75 = r25 * r16;
    r72 = r8 * r11;
    r76 = r12 * r15;
    r76 = fma(r59, r76, r53 * r72);
    r72 = r9 * r10;
    r76 = fma(r53, r72, r76);
    r76 = fma(r14, r64, r76);
    r75 = r75 * r76;
    r72 = r25 * r26;
    r85 = r12 * r11;
    r86 = r8 * r15;
    r86 = fma(r59, r86, r59 * r85);
    r86 = fma(r14, r61, r86);
    r86 = fma(r10, r64, r86);
    r72 = fma(r86, r72, r75);
    r78 = r78 + r72;
    r64 = r20 * r67;
    r64 = r64 * r71;
    r85 = r16 * r71;
    r85 = r85 * r86;
    r87 = r64 + r85;
    r87 = fma(r6, r87, r30 * r78);
    r78 = r18 * r29;
    r78 = fma(r18, r70, r86 * r78);
    r88 = r25 * r26;
    r88 = r88 * r67;
    r89 = r25 * r20;
    r89 = fma(r76, r89, r88);
    r78 = r78 + r89;
    r87 = fma(r7, r78, r87);
    r78 = r38 * r87;
    r90 = r18 * r20;
    r90 = fma(r63, r90, r84);
    r90 = r90 + r72;
    r72 = r25 * r20;
    r72 = r72 * r86;
    r91 = fma(r76, r32, r72);
    r91 = r91 + r62;
    r91 = fma(r7, r91, r6 * r90);
    r90 = r76 * r80;
    r64 = r64 + r90;
    r91 = fma(r30, r64, r91);
    r64 = r91 * r55;
    r64 = fma(r68, r64, r41 * r78);
    r78 = r25 * r87;
    r62 = r25 * r5;
    r72 = r65 + r72;
    r65 = r26 * r18;
    r72 = fma(r63, r65, r72);
    r63 = r18 * r29;
    r72 = fma(r76, r63, r72);
    r86 = fma(r86, r32, r25 * r70);
    r86 = r86 + r89;
    r86 = fma(r6, r86, r30 * r72);
    r90 = r85 + r90;
    r86 = fma(r7, r90, r86);
    r62 = r62 * r86;
    r62 = fma(r48, r62, r55 * r78);
    r78 = r91 * r47;
    r62 = fma(r34, r78, r62);
    r62 = fma(r91, r1, r62);
    r78 = r0 * r62;
    r64 = fma(r46, r78, r64);
    r78 = r5 * r62;
    r78 = fma(r91, r66, r46 * r78);
    r90 = r38 * r86;
    r78 = fma(r41, r90, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r64, r78);
    r90 = r13 * r11;
    r22 = fma(r53, r22, r59 * r90);
    r22 = fma(r15, r61, r22);
    r22 = fma(r59, r21, r22);
    r80 = r22 * r80;
    r58 = r71 * r58;
    r21 = r80 + r58;
    r59 = r25 * r16;
    r59 = r59 * r22;
    r88 = r88 + r59;
    r61 = r18 * r20;
    r88 = fma(r76, r61, r88);
    r53 = r18 * r29;
    r88 = fma(r57, r53, r88);
    r88 = fma(r6, r88, r30 * r21);
    r21 = r25 * r26;
    r21 = fma(r22, r32, r76 * r21);
    r21 = r21 + r83;
    r88 = fma(r7, r21, r88);
    r21 = r88 * r55;
    r84 = r77 + r84;
    r77 = r25 * r20;
    r77 = r77 * r22;
    r53 = r16 * r18;
    r84 = fma(r76, r53, r84);
    r84 = r84 + r77;
    r67 = r16 * r67;
    r67 = r67 * r71;
    r58 = r67 + r58;
    r58 = fma(r6, r58, r7 * r84);
    r32 = fma(r57, r32, r59);
    r32 = r32 + r89;
    r58 = fma(r30, r32, r58);
    r32 = r38 * r58;
    r32 = fma(r41, r32, r68 * r21);
    r21 = r25 * r5;
    r77 = r75 + r77;
    r77 = r77 + r79;
    r79 = r26 * r18;
    r75 = r18 * r29;
    r75 = fma(r22, r75, r76 * r79);
    r75 = r75 + r83;
    r75 = fma(r30, r75, r6 * r77);
    r80 = r67 + r80;
    r75 = fma(r7, r80, r75);
    r21 = r21 * r75;
    r21 = fma(r88, r1, r48 * r21);
    r80 = r25 * r58;
    r21 = fma(r55, r80, r21);
    r7 = r88 * r47;
    r21 = fma(r34, r7, r21);
    r7 = r0 * r21;
    r32 = fma(r46, r7, r32);
    r7 = r38 * r75;
    r7 = fma(r88, r66, r41 * r7);
    r80 = r5 * r21;
    r7 = fma(r46, r80, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r7);
    r80 = r25 * r56;
    r80 = r80 * r5;
    r67 = r36 * r47;
    r67 = fma(r34, r67, r48 * r80);
    r80 = r25 * r43;
    r67 = fma(r55, r80, r67);
    r67 = fma(r36, r1, r67);
    r80 = r0 * r67;
    r30 = r43 * r38;
    r30 = fma(r41, r30, r46 * r80);
    r80 = r36 * r55;
    r30 = fma(r68, r80, r30);
    r80 = r56 * r38;
    r77 = r5 * r67;
    r77 = fma(r46, r77, r41 * r80);
    r77 = fma(r36, r66, r77);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r30, r77);
    r80 = r50 * r55;
    r6 = r50 * r47;
    r83 = r25 * r44;
    r83 = r83 * r5;
    r83 = fma(r48, r83, r34 * r6);
    r6 = r25 * r40;
    r83 = fma(r55, r6, r83);
    r83 = fma(r50, r1, r83);
    r6 = r0 * r83;
    r6 = fma(r46, r6, r68 * r80);
    r80 = r40 * r38;
    r6 = fma(r41, r80, r6);
    r80 = r5 * r83;
    r80 = fma(r46, r80, r50 * r66);
    r79 = r44 * r38;
    r80 = fma(r41, r79, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r6, r80);
    r79 = r25 * r35;
    r79 = r79 * r5;
    r22 = r42 * r47;
    r22 = fma(r34, r22, r48 * r79);
    r79 = r25 * r37;
    r22 = fma(r55, r79, r22);
    r22 = fma(r42, r1, r22);
    r79 = r0 * r22;
    r76 = r42 * r55;
    r76 = fma(r68, r76, r46 * r79);
    r79 = r37 * r38;
    r76 = fma(r41, r79, r76);
    r79 = r5 * r22;
    r89 = r35 * r38;
    r89 = fma(r41, r89, r46 * r79);
    r89 = fma(r42, r66, r89);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r76, r89);
    r79 = r4 * r3;
    r57 = r4 * r2;
    r57 = fma(r82, r57, r73 * r79);
    r79 = r4 * r3;
    r59 = r4 * r2;
    r59 = fma(r64, r59, r78 * r79);
    WriteSum2<double, double>((double*)inout_shared, r57, r59);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r4 * r3;
    r57 = r4 * r2;
    r57 = fma(r32, r57, r7 * r59);
    r59 = r4 * r3;
    r79 = r4 * r2;
    r79 = fma(r30, r79, r77 * r59);
    WriteSum2<double, double>((double*)inout_shared, r57, r79);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r4 * r3;
    r57 = r4 * r2;
    r57 = fma(r6, r57, r80 * r79);
    r79 = r4 * r3;
    r59 = r4 * r2;
    r59 = fma(r76, r59, r89 * r79);
    WriteSum2<double, double>((double*)inout_shared, r57, r59);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r73, r73, r82 * r82);
    r57 = fma(r64, r64, r78 * r78);
    WriteSum2<double, double>((double*)inout_shared, r59, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r7, r7, r32 * r32);
    r59 = fma(r30, r30, r77 * r77);
    WriteSum2<double, double>((double*)inout_shared, r57, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r80, r80, r6 * r6);
    r57 = fma(r76, r76, r89 * r89);
    WriteSum2<double, double>((double*)inout_shared, r59, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r82, r64, r73 * r78);
    r59 = fma(r73, r7, r82 * r32);
    WriteSum2<double, double>((double*)inout_shared, r57, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r82, r30, r73 * r77);
    r57 = fma(r82, r6, r73 * r80);
    WriteSum2<double, double>((double*)inout_shared, r59, r57);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r73, r89, r82 * r76);
    r82 = fma(r78, r7, r64 * r32);
    WriteSum2<double, double>((double*)inout_shared, r73, r82);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fma(r78, r77, r64 * r30);
    r73 = fma(r78, r80, r64 * r6);
    WriteSum2<double, double>((double*)inout_shared, r82, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r78, r89, r64 * r76);
    r64 = fma(r7, r77, r32 * r30);
    WriteSum2<double, double>((double*)inout_shared, r78, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r7, r80, r32 * r6);
    r7 = fma(r7, r89, r32 * r76);
    WriteSum2<double, double>((double*)inout_shared, r64, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r77, r80, r30 * r6);
    r77 = fma(r77, r89, r30 * r76);
    WriteSum2<double, double>((double*)inout_shared, r7, r77);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r6, r76, r80 * r89);
    WriteSum1<double, double>((double*)inout_shared, r76);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r0 * r38;
    r76 = r76 * r51;
    r6 = r5 * r38;
    r6 = r6 * r51;
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r76, r6);
    r89 = r0 * r39;
    r89 = r89 * r41;
    r80 = r5 * r39;
    r80 = r80 * r41;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r89,
                                             r80);
    r77 = r5 * r3;
    r77 = r77 * r51;
    r7 = r0 * r2;
    r7 = r7 * r51;
    r7 = fma(r54, r7, r54 * r77);
    r77 = r4 * r0;
    r77 = r77 * r39;
    r77 = r77 * r2;
    r54 = r4 * r5;
    r54 = r54 * r39;
    r54 = r54 * r3;
    r54 = fma(r41, r54, r41 * r77);
    WriteSum2<double, double>((double*)inout_shared, r7, r54);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r4 * r2;
    r7 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r54, r7);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r0 * r38;
    r7 = r7 * r38;
    r54 = r38 * r48;
    r54 = r54 * r34;
    r7 = fma(r38, r54, r55 * r7);
    r77 = r48 * r34;
    r51 = r45 * r45;
    r30 = r39 * r39;
    r51 = r51 * r30;
    r30 = r0 * r55;
    r30 = fma(r51, r30, r51 * r77);
    WriteSum2<double, double>((double*)inout_shared, r7, r30);
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
    r30 = r45 * r39;
    r30 = fma(r54, r30, r55 * r19);
    WriteSum2<double, double>((double*)inout_shared, r30, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r6, r89);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r80, r89);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r25 * r24;
    r80 = r49 * r47;
    r80 = fma(r34, r80, r55 * r89);
    r89 = r25 * r27;
    r89 = r89 * r5;
    r80 = fma(r48, r89, r80);
    r80 = fma(r49, r1, r80);
    r80 = r80 * r46;
    r89 = r49 * r55;
    r89 = fma(r68, r89, r0 * r80);
    r6 = r24 * r38;
    r89 = fma(r41, r6, r89);
    r6 = r27 * r38;
    r6 = fma(r49, r66, r41 * r6);
    r6 = fma(r5, r80, r6);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r89, r6);
    r80 = r28 * r38;
    r76 = r52 * r55;
    r76 = fma(r68, r76, r41 * r80);
    r80 = r25 * r28;
    r80 = fma(r55, r80, r52 * r1);
    r30 = r25 * r17;
    r30 = r30 * r5;
    r80 = fma(r48, r30, r80);
    r19 = r52 * r47;
    r80 = fma(r34, r19, r80);
    r19 = r0 * r80;
    r76 = fma(r46, r19, r76);
    r19 = r5 * r80;
    r19 = fma(r52, r66, r46 * r19);
    r30 = r17 * r38;
    r19 = fma(r41, r30, r19);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r76,
                                             r19);
    r30 = r25 * r33;
    r54 = r23 * r47;
    r54 = fma(r34, r54, r55 * r30);
    r30 = r25 * r31;
    r30 = r30 * r5;
    r54 = fma(r48, r30, r54);
    r54 = fma(r23, r1, r54);
    r30 = r0 * r54;
    r1 = r23 * r55;
    r1 = fma(r68, r1, r46 * r30);
    r30 = r33 * r38;
    r1 = fma(r41, r30, r1);
    r30 = r5 * r54;
    r68 = r31 * r38;
    r68 = fma(r41, r68, r46 * r30);
    r68 = fma(r23, r66, r68);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r68);
    r66 = r4 * r2;
    r30 = r4 * r3;
    r30 = fma(r6, r30, r89 * r66);
    r66 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r76, r41, r19 * r66);
    WriteSum2<double, double>((double*)inout_shared, r30, r41);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r4 * r3;
    r30 = r4 * r2;
    r30 = fma(r1, r30, r68 * r41);
    WriteSum1<double, double>((double*)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r6, r6, r89 * r89);
    r41 = fma(r19, r19, r76 * r76);
    WriteSum2<double, double>((double*)inout_shared, r30, r41);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r1, r1, r68 * r68);
    WriteSum1<double, double>((double*)inout_shared, r41);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r6, r19, r89 * r76);
    r89 = fma(r89, r1, r6 * r68);
    WriteSum2<double, double>((double*)inout_shared, r41, r89);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r76, r1, r19 * r68);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialResJacFirst(double* pose,
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
  SimpleRadialResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
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