#include "cuda_to_hip.h"

#include "kernel_simple_radial_res_jac_first.h"
#include "memops.cuh"

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialResJacFirstKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91;
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r2, r7, r8);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r9, r10, r11);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r16, r17, r18, r19);
    r20 = fmaf(r13, r16, r14 * r19);
    r21 = r12 * r17;
    r20 = fmaf(r6, r21, r20);
    r20 = fmaf(r15, r18, r20);
    r21 = r20 * r20;
    r22 = -2.00000000000000000e+00;
    r21 = r21 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r15 * r17;
    r25 = fmaf(r13, r19, r24);
    r26 = r12 * r18;
    r27 = r14 * r16;
    r25 = r25 + r26;
    r25 = fmaf(r6, r27, r25);
    r28 = r22 * r25;
    r28 = fmaf(r25, r28, r23);
    r29 = r21 + r28;
    r2 = fmaf(r9, r29, r2);
    r30 = 2.00000000000000000e+00;
    r31 = fmaf(r15, r16, r12 * r19);
    r32 = r13 * r18;
    r31 = fmaf(r6, r32, r31);
    r31 = fmaf(r14, r17, r31);
    r32 = r30 * r31;
    r32 = r32 * r25;
    r33 = r20 * r22;
    r34 = fmaf(r13, r17, r12 * r16);
    r34 = fmaf(r14, r18, r34);
    r34 = fmaf(r6, r34, r15 * r19);
    r33 = fmaf(r34, r33, r32);
    r35 = r30 * r20;
    r35 = r35 * r31;
    r36 = r30 * r34;
    r37 = fmaf(r25, r36, r35);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r38, r39, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r16 * r18;
    r41 = r41 * r30;
    r42 = r17 * r19;
    r42 = fmaf(r30, r42, r41);
    r43 = r18 * r19;
    r44 = r16 * r17;
    r44 = r44 * r30;
    r43 = fmaf(r22, r43, r44);
    r45 = r17 * r17;
    r45 = r45 * r22;
    r46 = r23 + r45;
    r47 = r18 * r18;
    r47 = r47 * r22;
    r46 = r46 + r47;
    r2 = fmaf(r10, r33, r2);
    r2 = fmaf(r11, r37, r2);
    r2 = fmaf(r40, r42, r2);
    r2 = fmaf(r39, r43, r2);
    r2 = fmaf(r38, r46, r2);
    r48 = 9.99999999999999955e-07;
    r49 = r22 * r25;
    r49 = fmaf(r34, r49, r35);
    r8 = fmaf(r9, r49, r8);
    r35 = r17 * r19;
    r35 = fmaf(r22, r35, r41);
    r45 = r23 + r45;
    r41 = r16 * r16;
    r41 = r41 * r22;
    r45 = r45 + r41;
    r50 = r17 * r18;
    r50 = r50 * r30;
    r51 = r16 * r19;
    r51 = fmaf(r30, r51, r50);
    r52 = r30 * r20;
    r52 = r52 * r25;
    r53 = fmaf(r31, r36, r52);
    r54 = r31 * r31;
    r54 = r54 * r22;
    r28 = r54 + r28;
    r8 = fmaf(r38, r35, r8);
    r8 = fmaf(r40, r45, r8);
    r8 = fmaf(r39, r51, r8);
    r8 = fmaf(r10, r53, r8);
    r8 = fmaf(r11, r28, r8);
    r55 = copysign(1.0, r8);
    r55 = fmaf(r48, r55, r8);
    r48 = r55 * r55;
    r8 = 1.0 / r48;
    r56 = r2 * r8;
    r32 = fmaf(r20, r36, r32);
    r7 = fmaf(r9, r32, r7);
    r57 = r18 * r19;
    r57 = fmaf(r30, r57, r44);
    r47 = r23 + r47;
    r47 = r47 + r41;
    r41 = r16 * r19;
    r41 = fmaf(r22, r41, r50);
    r50 = r31 * r22;
    r50 = fmaf(r34, r50, r52);
    r21 = r23 + r21;
    r21 = r21 + r54;
    r7 = fmaf(r38, r57, r7);
    r7 = fmaf(r39, r47, r7);
    r7 = fmaf(r40, r41, r7);
    r7 = fmaf(r11, r50, r7);
    r7 = fmaf(r10, r21, r7);
    r40 = r7 * r7;
    r39 = fmaf(r8, r40, r2 * r56);
    r38 = fmaf(r1, r39, r23);
    r54 = r2 * r38;
    r52 = 1.0 / r55;
    r44 = r0 * r52;
    r4 = fmaf(r44, r54, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r7 * r38;
    r5 = fmaf(r44, r3, r5);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r4, r5);
    r3 = fmaf(r5, r5, r4 * r4);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r3);
  if (global_thread_idx < problem_size) {
    r3 = r30 * r7;
    r54 = r31 * r22;
    r58 = r12 * r19;
    r59 = -5.00000000000000000e-01;
    r60 = r15 * r16;
    r60 = fmaf(r59, r60, r59 * r58);
    r58 = r14 * r17;
    r60 = fmaf(r59, r58, r60);
    r61 = r13 * r18;
    r62 = 5.00000000000000000e-01;
    r60 = fmaf(r62, r61, r60);
    r61 = r22 * r34;
    r58 = r12 * r16;
    r63 = r14 * r18;
    r63 = fmaf(r59, r63, r59 * r58);
    r58 = r19 * r62;
    r64 = r13 * r59;
    r63 = fmaf(r15, r58, r63);
    r63 = fmaf(r17, r64, r63);
    r61 = r61 * r63;
    r54 = fmaf(r60, r54, r61);
    r65 = r30 * r25;
    r66 = fmaf(r62, r27, r19 * r64);
    r66 = fmaf(r59, r24, r66);
    r66 = fmaf(r59, r26, r66);
    r65 = r65 * r66;
    r67 = r30 * r20;
    r68 = r13 * r16;
    r69 = r12 * r17;
    r69 = fmaf(r59, r69, r62 * r68);
    r68 = r15 * r18;
    r69 = fmaf(r62, r68, r69);
    r69 = fmaf(r14, r58, r69);
    r67 = fmaf(r69, r67, r65);
    r54 = r54 + r67;
    r68 = -4.00000000000000000e+00;
    r70 = r31 * r68;
    r71 = r63 * r70;
    r72 = r20 * r66;
    r73 = r68 * r72;
    r74 = r71 + r73;
    r74 = fmaf(r10, r74, r11 * r54);
    r54 = r30 * r31;
    r54 = r54 * r69;
    r75 = fmaf(r66, r36, r54);
    r76 = r30 * r25;
    r76 = r76 * r63;
    r77 = r30 * r20;
    r77 = fmaf(r60, r77, r76);
    r75 = r75 + r77;
    r74 = fmaf(r9, r75, r74);
    r3 = r3 * r74;
    r75 = r25 * r60;
    r78 = fmaf(r69, r36, r30 * r75);
    r79 = r30 * r31;
    r80 = r30 * r20;
    r80 = r80 * r63;
    r79 = fmaf(r66, r79, r80);
    r78 = r78 + r79;
    r54 = r76 + r54;
    r76 = r20 * r22;
    r54 = fmaf(r60, r76, r54);
    r81 = r22 * r34;
    r54 = fmaf(r66, r81, r54);
    r54 = fmaf(r10, r54, r11 * r78);
    r78 = r25 * r69;
    r78 = r78 * r68;
    r73 = r78 + r73;
    r54 = fmaf(r9, r73, r54);
    r73 = r30 * r54;
    r73 = fmaf(r56, r73, r8 * r3);
    r3 = r30 * r31;
    r3 = r3 * r60;
    r81 = r63 * r36;
    r76 = r3 + r81;
    r67 = r67 + r76;
    r82 = r22 * r34;
    r82 = fmaf(r22, r75, r69 * r82);
    r82 = r82 + r79;
    r82 = fmaf(r9, r82, r10 * r67);
    r71 = r78 + r71;
    r82 = fmaf(r11, r71, r82);
    r48 = r55 * r48;
    r48 = 1.0 / r48;
    r48 = r22 * r48;
    r55 = r82 * r48;
    r73 = fmaf(r40, r55, r73);
    r71 = r2 * r2;
    r71 = r71 * r48;
    r73 = fmaf(r82, r71, r73);
    r1 = r1 * r44;
    r73 = r73 * r1;
    r55 = r82 * r56;
    r78 = r6 * r38;
    r67 = r0 * r78;
    r55 = fmaf(r67, r55, r2 * r73);
    r69 = r38 * r54;
    r55 = fmaf(r44, r69, r55);
    r69 = r38 * r74;
    r83 = r7 * r8;
    r83 = r83 * r67;
    r69 = fmaf(r82, r83, r44 * r69);
    r69 = fmaf(r7, r73, r69);
    r81 = r65 + r81;
    r65 = r30 * r20;
    r73 = r14 * r19;
    r84 = r12 * r17;
    r84 = fmaf(r62, r84, r59 * r73);
    r73 = r15 * r18;
    r84 = fmaf(r59, r73, r84);
    r84 = fmaf(r16, r64, r84);
    r65 = r65 * r84;
    r73 = r30 * r31;
    r85 = r15 * r16;
    r86 = r14 * r17;
    r86 = fmaf(r62, r86, r62 * r85);
    r86 = fmaf(r12, r58, r86);
    r86 = fmaf(r18, r64, r86);
    r73 = fmaf(r86, r73, r65);
    r81 = r81 + r73;
    r64 = r25 * r63;
    r64 = r64 * r68;
    r85 = r20 * r68;
    r85 = r85 * r86;
    r87 = r64 + r85;
    r87 = fmaf(r9, r87, r11 * r81);
    r81 = r22 * r34;
    r81 = fmaf(r22, r72, r86 * r81);
    r88 = r30 * r31;
    r88 = r88 * r63;
    r89 = r30 * r25;
    r89 = fmaf(r84, r89, r88);
    r81 = r81 + r89;
    r87 = fmaf(r10, r81, r87);
    r81 = r30 * r87;
    r90 = r22 * r25;
    r90 = fmaf(r66, r90, r61);
    r90 = r90 + r73;
    r73 = r30 * r25;
    r73 = r73 * r86;
    r91 = fmaf(r84, r36, r73);
    r91 = r91 + r79;
    r91 = fmaf(r10, r91, r9 * r90);
    r90 = r84 * r70;
    r64 = r64 + r90;
    r91 = fmaf(r11, r64, r91);
    r81 = fmaf(r91, r71, r56 * r81);
    r64 = r91 * r48;
    r81 = fmaf(r40, r64, r81);
    r79 = r30 * r7;
    r73 = r80 + r73;
    r80 = r31 * r22;
    r73 = fmaf(r66, r80, r73);
    r66 = r22 * r34;
    r73 = fmaf(r84, r66, r73);
    r86 = fmaf(r86, r36, r30 * r72);
    r86 = r86 + r89;
    r86 = fmaf(r9, r86, r11 * r73);
    r90 = r85 + r90;
    r86 = fmaf(r10, r90, r86);
    r79 = r79 * r86;
    r81 = fmaf(r8, r79, r81);
    r79 = r2 * r81;
    r64 = r91 * r56;
    r64 = fmaf(r67, r64, r1 * r79);
    r79 = r38 * r87;
    r64 = fmaf(r44, r79, r64);
    r79 = r38 * r86;
    r90 = r7 * r81;
    r90 = fmaf(r1, r90, r44 * r79);
    r90 = fmaf(r91, r83, r90);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r55, r69,
        r64, r90);
    r27 = fmaf(r59, r27, r13 * r58);
    r27 = fmaf(r62, r24, r27);
    r27 = fmaf(r62, r26, r27);
    r70 = r27 * r70;
    r75 = r68 * r75;
    r26 = r70 + r75;
    r62 = r30 * r20;
    r62 = r62 * r27;
    r88 = r88 + r62;
    r24 = r22 * r25;
    r88 = fmaf(r84, r24, r88);
    r59 = r22 * r34;
    r88 = fmaf(r60, r59, r88);
    r88 = fmaf(r9, r88, r11 * r26);
    r26 = r30 * r31;
    r26 = fmaf(r27, r36, r84 * r26);
    r26 = r26 + r77;
    r88 = fmaf(r10, r26, r88);
    r26 = r88 * r56;
    r3 = r61 + r3;
    r61 = r30 * r25;
    r61 = r61 * r27;
    r59 = r20 * r22;
    r3 = fmaf(r84, r59, r3);
    r3 = r3 + r61;
    r63 = r20 * r63;
    r63 = r63 * r68;
    r75 = r63 + r75;
    r75 = fmaf(r9, r75, r10 * r3);
    r36 = fmaf(r60, r36, r62);
    r36 = r36 + r89;
    r75 = fmaf(r11, r36, r75);
    r36 = r30 * r75;
    r36 = fmaf(r56, r36, r88 * r71);
    r89 = r88 * r48;
    r36 = fmaf(r40, r89, r36);
    r60 = r30 * r7;
    r61 = r65 + r61;
    r61 = r61 + r76;
    r76 = r31 * r22;
    r65 = r22 * r34;
    r65 = fmaf(r27, r65, r84 * r76);
    r65 = r65 + r77;
    r65 = fmaf(r11, r65, r9 * r61);
    r70 = r63 + r70;
    r65 = fmaf(r10, r70, r65);
    r60 = r60 * r65;
    r36 = fmaf(r8, r60, r36);
    r60 = r2 * r36;
    r60 = fmaf(r1, r60, r67 * r26);
    r26 = r38 * r75;
    r60 = fmaf(r44, r26, r60);
    r26 = r38 * r65;
    r26 = fmaf(r44, r26, r88 * r83);
    r89 = r7 * r36;
    r26 = fmaf(r1, r89, r26);
    r89 = r35 * r48;
    r70 = r30 * r57;
    r70 = r70 * r7;
    r70 = fmaf(r8, r70, r40 * r89);
    r89 = r30 * r46;
    r70 = fmaf(r56, r89, r70);
    r70 = fmaf(r35, r71, r70);
    r89 = r2 * r70;
    r10 = r35 * r56;
    r10 = fmaf(r67, r10, r1 * r89);
    r89 = r46 * r38;
    r10 = fmaf(r44, r89, r10);
    r89 = r7 * r70;
    r63 = r57 * r38;
    r63 = fmaf(r44, r63, r1 * r89);
    r63 = fmaf(r35, r83, r63);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r60, r26,
        r10, r63);
    r89 = r43 * r38;
    r11 = r51 * r48;
    r61 = r30 * r47;
    r61 = r61 * r7;
    r61 = fmaf(r8, r61, r40 * r11);
    r11 = r30 * r43;
    r61 = fmaf(r56, r11, r61);
    r61 = fmaf(r51, r71, r61);
    r11 = r2 * r61;
    r11 = fmaf(r1, r11, r44 * r89);
    r89 = r51 * r56;
    r11 = fmaf(r67, r89, r11);
    r89 = r47 * r38;
    r89 = fmaf(r44, r89, r51 * r83);
    r9 = r7 * r61;
    r89 = fmaf(r1, r9, r89);
    r9 = r42 * r38;
    r77 = r45 * r56;
    r77 = fmaf(r67, r77, r44 * r9);
    r9 = r30 * r41;
    r9 = r9 * r7;
    r76 = r45 * r48;
    r76 = fmaf(r40, r76, r8 * r9);
    r9 = r30 * r42;
    r76 = fmaf(r56, r9, r76);
    r76 = fmaf(r45, r71, r76);
    r9 = r2 * r76;
    r77 = fmaf(r1, r9, r77);
    r9 = r41 * r38;
    r9 = fmaf(r45, r83, r44 * r9);
    r27 = r7 * r76;
    r9 = fmaf(r1, r27, r9);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r11, r89, r77, r9);
    r27 = r6 * r4;
    r84 = r6 * r5;
    r84 = fmaf(r69, r84, r55 * r27);
    r27 = r6 * r5;
    r62 = r6 * r4;
    r62 = fmaf(r64, r62, r90 * r27);
    r27 = r6 * r5;
    r3 = r6 * r4;
    r3 = fmaf(r60, r3, r26 * r27);
    r27 = r6 * r4;
    r68 = r6 * r5;
    r68 = fmaf(r63, r68, r10 * r27);
    WriteSum4<float, float>((float *)inout_shared, r84, r62, r3, r68);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r6 * r4;
    r3 = r6 * r5;
    r3 = fmaf(r89, r3, r11 * r68);
    r68 = r6 * r5;
    r62 = r6 * r4;
    r62 = fmaf(r77, r62, r9 * r68);
    WriteSum2<float, float>((float *)inout_shared, r3, r62);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r55, r55, r69 * r69);
    r3 = fmaf(r64, r64, r90 * r90);
    r68 = fmaf(r26, r26, r60 * r60);
    r84 = fmaf(r63, r63, r10 * r10);
    WriteSum4<float, float>((float *)inout_shared, r62, r3, r68, r84);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = fmaf(r11, r11, r89 * r89);
    r68 = fmaf(r9, r9, r77 * r77);
    WriteSum2<float, float>((float *)inout_shared, r84, r68);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r55, r64, r69 * r90);
    r84 = fmaf(r55, r60, r69 * r26);
    r3 = fmaf(r55, r10, r69 * r63);
    r62 = fmaf(r55, r11, r69 * r89);
    WriteSum4<float, float>((float *)inout_shared, r68, r84, r3, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r55, r77, r69 * r9);
    r69 = fmaf(r64, r60, r90 * r26);
    r62 = fmaf(r64, r10, r90 * r63);
    r3 = fmaf(r64, r11, r90 * r89);
    WriteSum4<float, float>((float *)inout_shared, r55, r69, r62, r3);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r64, r77, r90 * r9);
    r90 = fmaf(r26, r63, r60 * r10);
    r3 = fmaf(r26, r89, r60 * r11);
    r60 = fmaf(r60, r77, r26 * r9);
    WriteSum4<float, float>((float *)inout_shared, r64, r90, r3, r60);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fmaf(r10, r11, r63 * r89);
    r63 = fmaf(r63, r9, r10 * r77);
    r77 = fmaf(r11, r77, r89 * r9);
    WriteSum3<float, float>((float *)inout_shared, r60, r63, r77);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r2 * r38;
    r77 = r77 * r52;
    r63 = r7 * r38;
    r63 = r63 * r52;
    r60 = r2 * r39;
    r60 = r60 * r44;
    r11 = r7 * r39;
    r11 = r11 * r44;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r77, r63,
        r60, r11);
    r9 = r6 * r4;
    r89 = r6 * r5;
    r10 = r2 * r4;
    r10 = r10 * r52;
    r3 = r7 * r5;
    r3 = r3 * r52;
    r3 = fmaf(r78, r3, r78 * r10);
    r10 = r6 * r2;
    r10 = r10 * r39;
    r10 = r10 * r4;
    r78 = r6 * r7;
    r78 = r78 * r39;
    r78 = r78 * r5;
    r78 = fmaf(r44, r78, r44 * r10);
    WriteSum4<float, float>((float *)inout_shared, r3, r78, r9, r89);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r2 * r38;
    r89 = r89 * r38;
    r9 = r38 * r8;
    r9 = r9 * r40;
    r89 = fmaf(r38, r9, r56 * r89);
    r78 = r8 * r40;
    r3 = r0 * r0;
    r10 = r39 * r39;
    r3 = r3 * r10;
    r10 = r2 * r56;
    r10 = fmaf(r3, r10, r3 * r78);
    WriteSum4<float, float>((float *)inout_shared, r89, r10, r23, r23);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r0 * r2;
    r23 = r23 * r39;
    r23 = r23 * r38;
    r10 = r0 * r39;
    r10 = fmaf(r9, r10, r56 * r23);
    WriteSum4<float, float>((float *)inout_shared, r10, r77, r63, r60);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = 0.00000000000000000e+00;
    WriteSum2<float, float>((float *)inout_shared, r11, r60);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r30 * r32;
    r60 = r60 * r7;
    r11 = r30 * r29;
    r11 = fmaf(r56, r11, r8 * r60);
    r60 = r49 * r48;
    r11 = fmaf(r40, r60, r11);
    r11 = fmaf(r49, r71, r11);
    r60 = r2 * r11;
    r63 = r49 * r56;
    r63 = fmaf(r67, r63, r1 * r60);
    r60 = r29 * r38;
    r63 = fmaf(r44, r60, r63);
    r60 = r32 * r38;
    r60 = fmaf(r44, r60, r49 * r83);
    r77 = r7 * r11;
    r60 = fmaf(r1, r77, r60);
    r77 = r33 * r38;
    r10 = r53 * r48;
    r23 = r30 * r33;
    r23 = fmaf(r56, r23, r40 * r10);
    r10 = r30 * r21;
    r10 = r10 * r7;
    r23 = fmaf(r8, r10, r23);
    r23 = fmaf(r53, r71, r23);
    r10 = r2 * r23;
    r10 = fmaf(r1, r10, r44 * r77);
    r77 = r53 * r56;
    r10 = fmaf(r67, r77, r10);
    r77 = r21 * r38;
    r77 = fmaf(r53, r83, r44 * r77);
    r9 = r7 * r23;
    r77 = fmaf(r1, r9, r77);
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r63, r60,
        r10, r77);
    r9 = r28 * r56;
    r89 = r37 * r38;
    r89 = fmaf(r44, r89, r67 * r9);
    r9 = r30 * r50;
    r9 = r9 * r7;
    r67 = r28 * r48;
    r67 = fmaf(r40, r67, r8 * r9);
    r9 = r30 * r37;
    r67 = fmaf(r56, r9, r67);
    r67 = fmaf(r28, r71, r67);
    r9 = r2 * r67;
    r89 = fmaf(r1, r9, r89);
    r9 = r50 * r38;
    r71 = r7 * r67;
    r71 = fmaf(r1, r71, r44 * r9);
    r71 = fmaf(r28, r83, r71);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r89, r71);
    r83 = r6 * r4;
    r9 = r6 * r5;
    r9 = fmaf(r60, r9, r63 * r83);
    r83 = r6 * r5;
    r1 = r6 * r4;
    r1 = fmaf(r10, r1, r77 * r83);
    r83 = r6 * r4;
    r44 = r6 * r5;
    r44 = fmaf(r71, r44, r89 * r83);
    WriteSum3<float, float>((float *)inout_shared, r9, r1, r44);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r60, r60, r63 * r63);
    r1 = fmaf(r77, r77, r10 * r10);
    r9 = fmaf(r89, r89, r71 * r71);
    WriteSum3<float, float>((float *)inout_shared, r44, r1, r9);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r60, r77, r63 * r10);
    r60 = fmaf(r60, r71, r63 * r89);
    r71 = fmaf(r77, r71, r10 * r89);
    WriteSum3<float, float>((float *)inout_shared, r9, r60, r71);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar