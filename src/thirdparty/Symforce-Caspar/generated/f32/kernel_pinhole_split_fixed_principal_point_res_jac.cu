#include "cuda_to_hip.h"

#include "kernel_pinhole_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"

namespace caspar {

__global__ void
__launch_bounds__(1024, 1) PinholeSplitFixedPrincipalPointResJacKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal, unsigned int focal_num_alloc, SharedIndex *focal_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_focal_jac,
    unsigned int out_focal_jac_num_alloc, float *const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc, float *const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float *const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc, float *out_point_jac,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r5, r6, r7);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r8, r9, r10);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r11, r12, r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    r19 = fmaf(r12, r15, r13 * r18);
    r20 = r11 * r16;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r14, r17, r19);
    r20 = 2.00000000000000000e+00;
    r21 = fmaf(r14, r15, r11 * r18);
    r22 = r12 * r17;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r13, r16, r21);
    r22 = r20 * r21;
    r23 = r19 * r22;
    r24 = -2.00000000000000000e+00;
    r25 = fmaf(r12, r16, r11 * r15);
    r25 = fmaf(r13, r17, r25);
    r25 = fmaf(r4, r25, r14 * r18);
    r26 = r24 * r25;
    r27 = fmaf(r14, r16, r12 * r18);
    r28 = r11 * r17;
    r29 = r13 * r15;
    r27 = r27 + r28;
    r27 = fmaf(r4, r29, r27);
    r30 = fmaf(r27, r26, r23);
    r7 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r31, r32, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r15 * r17;
    r34 = r34 * r20;
    r35 = r16 * r18;
    r36 = fmaf(r24, r35, r34);
    r37 = r15 * r15;
    r37 = r37 * r24;
    r38 = 1.00000000000000000e+00;
    r39 = r16 * r16;
    r39 = fmaf(r24, r39, r38);
    r40 = r37 + r39;
    r41 = r16 * r17;
    r41 = r41 * r20;
    r42 = r15 * r18;
    r42 = fmaf(r20, r42, r41);
    r43 = r20 * r19;
    r43 = r43 * r27;
    r44 = fmaf(r25, r22, r43);
    r45 = r24 * r27;
    r45 = r45 * r27;
    r46 = r38 + r45;
    r47 = r21 * r21;
    r47 = r47 * r24;
    r46 = r46 + r47;
    r7 = fmaf(r31, r36, r7);
    r7 = fmaf(r33, r40, r7);
    r7 = fmaf(r32, r42, r7);
    r7 = fmaf(r9, r44, r7);
    r7 = fmaf(r10, r46, r7);
    r48 = copysign(1.0, r7);
    r48 = fmaf(r0, r48, r7);
    r0 = 1.0 / r48;
  };
  LoadShared<2, float, float>(focal, 0 * focal_num_alloc, focal_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_indices_loc[threadIdx.x].target, r7, r49);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r45 = r38 + r45;
    r50 = r19 * r19;
    r50 = r50 * r24;
    r45 = r45 + r50;
    r5 = fmaf(r8, r45, r5);
    r51 = r27 * r22;
    r52 = fmaf(r19, r26, r51);
    r53 = r20 * r27;
    r53 = fmaf(r25, r53, r23);
    r35 = fmaf(r20, r35, r34);
    r34 = r17 * r18;
    r23 = r15 * r16;
    r23 = r23 * r20;
    r34 = fmaf(r24, r34, r23);
    r54 = r17 * r17;
    r54 = r54 * r24;
    r39 = r54 + r39;
    r5 = fmaf(r9, r52, r5);
    r5 = fmaf(r10, r53, r5);
    r5 = fmaf(r33, r35, r5);
    r5 = fmaf(r32, r34, r5);
    r5 = fmaf(r31, r39, r5);
    r55 = r7 * r5;
    r2 = fmaf(r0, r55, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r25, r1, r51);
    r6 = fmaf(r8, r1, r6);
    r51 = r17 * r18;
    r51 = fmaf(r20, r51, r23);
    r54 = r38 + r54;
    r54 = r54 + r37;
    r37 = r15 * r18;
    r37 = fmaf(r24, r37, r41);
    r43 = fmaf(r21, r26, r43);
    r50 = r38 + r50;
    r50 = r50 + r47;
    r6 = fmaf(r31, r51, r6);
    r6 = fmaf(r32, r54, r6);
    r6 = fmaf(r33, r37, r6);
    r6 = fmaf(r10, r43, r6);
    r6 = fmaf(r9, r50, r6);
    r33 = r49 * r6;
    r3 = fmaf(r0, r33, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r32 = r20 * r25;
    r31 = r13 * r18;
    r47 = 5.00000000000000000e-01;
    r38 = r12 * r15;
    r38 = fmaf(r47, r38, r47 * r31);
    r31 = r11 * r16;
    r41 = -5.00000000000000000e-01;
    r38 = fmaf(r41, r31, r38);
    r23 = r14 * r47;
    r38 = fmaf(r17, r23, r38);
    r31 = r11 * r18;
    r56 = r14 * r15;
    r56 = fmaf(r41, r56, r41 * r31);
    r31 = r13 * r16;
    r56 = fmaf(r41, r31, r56);
    r57 = r12 * r17;
    r56 = fmaf(r47, r57, r56);
    r57 = r27 * r56;
    r32 = fmaf(r20, r57, r38 * r32);
    r31 = r20 * r19;
    r58 = r11 * r15;
    r59 = r13 * r17;
    r59 = fmaf(r41, r59, r41 * r58);
    r58 = r12 * r41;
    r59 = fmaf(r18, r23, r59);
    r59 = fmaf(r16, r58, r59);
    r31 = r31 * r59;
    r60 = r14 * r16;
    r60 = fmaf(r18, r58, r41 * r60);
    r60 = fmaf(r47, r29, r60);
    r60 = fmaf(r41, r28, r60);
    r61 = fmaf(r60, r22, r31);
    r32 = r32 + r61;
    r62 = r20 * r27;
    r62 = r62 * r59;
    r63 = r19 * r24;
    r63 = fmaf(r56, r63, r62);
    r64 = r38 * r22;
    r63 = r63 + r64;
    r63 = fmaf(r60, r26, r63);
    r63 = fmaf(r9, r63, r10 * r32);
    r32 = r27 * r38;
    r65 = -4.00000000000000000e+00;
    r32 = r32 * r65;
    r66 = r19 * r60;
    r67 = r65 * r66;
    r68 = r32 + r67;
    r63 = fmaf(r8, r68, r63);
    r68 = r7 * r63;
    r69 = r20 * r27;
    r69 = r69 * r60;
    r70 = r20 * r19;
    r70 = fmaf(r38, r70, r69);
    r71 = r20 * r25;
    r71 = r71 * r59;
    r72 = r56 * r22;
    r73 = r71 + r72;
    r74 = r70 + r73;
    r38 = fmaf(r38, r26, r24 * r57);
    r38 = r38 + r61;
    r38 = fmaf(r8, r38, r9 * r74);
    r74 = r59 * r65;
    r75 = r21 * r74;
    r32 = r32 + r75;
    r38 = fmaf(r10, r32, r38);
    r48 = r48 * r48;
    r48 = 1.0 / r48;
    r32 = r4 * r48;
    r55 = r32 * r55;
    r68 = fmaf(r38, r55, r0 * r68);
    r76 = r38 * r32;
    r77 = r21 * r24;
    r78 = r59 * r26;
    r77 = fmaf(r56, r77, r78);
    r77 = r77 + r70;
    r75 = r67 + r75;
    r75 = fmaf(r9, r75, r10 * r77);
    r77 = r20 * r25;
    r77 = fmaf(r60, r77, r64);
    r64 = r20 * r19;
    r64 = fmaf(r56, r64, r62);
    r77 = r77 + r64;
    r75 = fmaf(r8, r77, r75);
    r77 = r49 * r75;
    r77 = fmaf(r0, r77, r33 * r76);
    r76 = r24 * r27;
    r76 = fmaf(r60, r76, r78);
    r62 = r20 * r19;
    r67 = r13 * r18;
    r70 = r11 * r16;
    r70 = fmaf(r47, r70, r41 * r67);
    r67 = r14 * r17;
    r70 = fmaf(r41, r67, r70);
    r70 = fmaf(r15, r58, r70);
    r62 = r62 * r70;
    r67 = r11 * r18;
    r79 = r13 * r16;
    r79 = fmaf(r47, r79, r47 * r67);
    r79 = fmaf(r15, r23, r79);
    r79 = fmaf(r17, r58, r79);
    r58 = fmaf(r79, r22, r62);
    r76 = r76 + r58;
    r67 = r20 * r27;
    r67 = r67 * r79;
    r80 = r20 * r25;
    r80 = fmaf(r70, r80, r67);
    r80 = r80 + r61;
    r80 = fmaf(r9, r80, r8 * r76);
    r76 = r21 * r65;
    r76 = r76 * r70;
    r61 = r27 * r74;
    r81 = r76 + r61;
    r80 = fmaf(r10, r81, r80);
    r71 = r69 + r71;
    r71 = r71 + r58;
    r58 = r19 * r65;
    r58 = r58 * r79;
    r61 = r58 + r61;
    r61 = fmaf(r8, r61, r10 * r71);
    r71 = fmaf(r79, r26, r24 * r66);
    r69 = r20 * r27;
    r59 = r59 * r22;
    r69 = fmaf(r70, r69, r59);
    r71 = r71 + r69;
    r61 = fmaf(r9, r71, r61);
    r71 = r7 * r61;
    r71 = fmaf(r0, r71, r80 * r55);
    r81 = r80 * r32;
    r67 = r31 + r67;
    r31 = r21 * r24;
    r67 = fmaf(r60, r31, r67);
    r67 = fmaf(r70, r26, r67);
    r31 = r20 * r25;
    r66 = fmaf(r20, r66, r79 * r31);
    r66 = r66 + r69;
    r66 = fmaf(r8, r66, r10 * r67);
    r58 = r76 + r58;
    r66 = fmaf(r9, r58, r66);
    r58 = r49 * r66;
    r58 = fmaf(r0, r58, r33 * r81);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r68, r77,
        r71, r58);
    r81 = r21 * r65;
    r76 = r12 * r18;
    r29 = fmaf(r41, r29, r47 * r76);
    r29 = fmaf(r16, r23, r29);
    r29 = fmaf(r47, r28, r29);
    r81 = r81 * r29;
    r57 = r65 * r57;
    r65 = r81 + r57;
    r28 = r20 * r19;
    r28 = r28 * r29;
    r47 = r24 * r27;
    r47 = fmaf(r70, r47, r28);
    r47 = r47 + r59;
    r47 = fmaf(r56, r26, r47);
    r47 = fmaf(r8, r47, r10 * r65);
    r65 = r20 * r25;
    r22 = fmaf(r70, r22, r29 * r65);
    r22 = r22 + r64;
    r47 = fmaf(r9, r22, r47);
    r22 = r20 * r27;
    r22 = r22 * r29;
    r65 = r19 * r24;
    r65 = fmaf(r70, r65, r22);
    r65 = r65 + r72;
    r65 = r65 + r78;
    r74 = r19 * r74;
    r57 = r57 + r74;
    r57 = fmaf(r8, r57, r9 * r65);
    r65 = r20 * r25;
    r65 = fmaf(r56, r65, r28);
    r65 = r65 + r69;
    r57 = fmaf(r10, r65, r57);
    r65 = r7 * r57;
    r65 = fmaf(r0, r65, r47 * r55);
    r22 = r62 + r22;
    r22 = r22 + r73;
    r73 = r21 * r24;
    r26 = fmaf(r29, r26, r70 * r73);
    r26 = r26 + r64;
    r26 = fmaf(r10, r26, r8 * r22);
    r74 = r81 + r74;
    r26 = fmaf(r9, r74, r26);
    r74 = r49 * r26;
    r9 = r47 * r32;
    r9 = fmaf(r33, r9, r0 * r74);
    r74 = r7 * r39;
    r74 = fmaf(r36, r55, r0 * r74);
    r81 = r49 * r51;
    r10 = r36 * r32;
    r10 = fmaf(r33, r10, r0 * r81);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx, r65, r9, r74, r10);
    r81 = r7 * r34;
    r81 = fmaf(r0, r81, r42 * r55);
    r22 = r42 * r32;
    r8 = r49 * r54;
    r8 = fmaf(r0, r8, r33 * r22);
    r22 = r7 * r35;
    r22 = fmaf(r40, r55, r0 * r22);
    r64 = r49 * r37;
    r29 = r40 * r32;
    r29 = fmaf(r33, r29, r0 * r64);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r81, r8, r22, r29);
    r64 = r4 * r3;
    r73 = r4 * r2;
    r73 = fmaf(r68, r73, r77 * r64);
    r64 = r4 * r2;
    r70 = r4 * r3;
    r70 = fmaf(r58, r70, r71 * r64);
    r64 = r4 * r2;
    r62 = r4 * r3;
    r62 = fmaf(r9, r62, r65 * r64);
    r64 = r4 * r2;
    r69 = r4 * r3;
    r69 = fmaf(r10, r69, r74 * r64);
    WriteSum4<float, float>((float *)inout_shared, r73, r70, r62, r69);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r4 * r2;
    r62 = r4 * r3;
    r62 = fmaf(r8, r62, r81 * r69);
    r69 = r4 * r3;
    r70 = r4 * r2;
    r70 = fmaf(r22, r70, r29 * r69);
    WriteSum2<float, float>((float *)inout_shared, r62, r70);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r68, r68, r77 * r77);
    r62 = fmaf(r58, r58, r71 * r71);
    r69 = fmaf(r9, r9, r65 * r65);
    r73 = fmaf(r10, r10, r74 * r74);
    WriteSum4<float, float>((float *)inout_shared, r70, r62, r69, r73);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r8, r8, r81 * r81);
    r69 = fmaf(r22, r22, r29 * r29);
    WriteSum2<float, float>((float *)inout_shared, r73, r69);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fmaf(r68, r71, r77 * r58);
    r73 = fmaf(r68, r65, r77 * r9);
    r62 = fmaf(r77, r10, r68 * r74);
    r70 = fmaf(r68, r81, r77 * r8);
    WriteSum4<float, float>((float *)inout_shared, r69, r73, r62, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r77, r29, r68 * r22);
    r68 = fmaf(r71, r65, r58 * r9);
    r70 = fmaf(r58, r10, r71 * r74);
    r62 = fmaf(r71, r81, r58 * r8);
    WriteSum4<float, float>((float *)inout_shared, r77, r68, r70, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fmaf(r58, r29, r71 * r22);
    r71 = fmaf(r65, r74, r9 * r10);
    r62 = fmaf(r9, r8, r65 * r81);
    r65 = fmaf(r65, r22, r9 * r29);
    WriteSum4<float, float>((float *)inout_shared, r58, r71, r62, r65);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r74, r81, r10 * r8);
    r74 = fmaf(r74, r22, r10 * r29);
    r29 = fmaf(r8, r29, r81 * r22);
    WriteSum3<float, float>((float *)inout_shared, r65, r74, r29);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r5 * r0;
    r74 = r6 * r0;
    WriteIdx2<1024, float, float, float2>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx, r29, r74);
    r74 = r4 * r5;
    r74 = r74 * r2;
    r74 = r74 * r0;
    r29 = r4 * r6;
    r29 = r29 * r3;
    r29 = r29 * r0;
    WriteSum2<float, float>((float *)inout_shared, r74, r29);
  };
  FlushSumShared<2, float>(out_focal_njtr, 0 * out_focal_njtr_num_alloc,
                           focal_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r5 * r5;
    r5 = r5 * r48;
    r6 = r6 * r6;
    r6 = r6 * r48;
    WriteSum2<float, float>((float *)inout_shared, r5, r6);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r7 * r45;
    r6 = fmaf(r0, r6, r30 * r55);
    r5 = r49 * r1;
    r48 = r30 * r32;
    r48 = fmaf(r33, r48, r0 * r5);
    r5 = r7 * r52;
    r5 = fmaf(r0, r5, r44 * r55);
    r29 = r49 * r50;
    r74 = r44 * r32;
    r74 = fmaf(r33, r74, r0 * r29);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx, r6, r48, r5, r74);
    r29 = r7 * r53;
    r55 = fmaf(r46, r55, r0 * r29);
    r29 = r49 * r43;
    r65 = r46 * r32;
    r65 = fmaf(r33, r65, r0 * r29);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r55, r65);
    r29 = r4 * r2;
    r33 = r4 * r3;
    r33 = fmaf(r48, r33, r6 * r29);
    r29 = r4 * r3;
    r0 = r4 * r2;
    r0 = fmaf(r5, r0, r74 * r29);
    r29 = r4 * r3;
    r8 = r4 * r2;
    r8 = fmaf(r55, r8, r65 * r29);
    WriteSum3<float, float>((float *)inout_shared, r33, r0, r8);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r48, r48, r6 * r6);
    r0 = fmaf(r74, r74, r5 * r5);
    r33 = fmaf(r65, r65, r55 * r55);
    WriteSum3<float, float>((float *)inout_shared, r8, r0, r33);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r6, r5, r48 * r74);
    r48 = fmaf(r48, r65, r6 * r55);
    r55 = fmaf(r5, r55, r74 * r65);
    WriteSum3<float, float>((float *)inout_shared, r33, r48, r55);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
}

void PinholeSplitFixedPrincipalPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal, unsigned int focal_num_alloc, SharedIndex *focal_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_focal_jac,
    unsigned int out_focal_jac_num_alloc, float *const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc, float *const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float *const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal, focal_num_alloc, focal_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, principal_point,
      principal_point_num_alloc, out_res, out_res_num_alloc, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc, out_focal_jac,
      out_focal_jac_num_alloc, out_focal_njtr, out_focal_njtr_num_alloc,
      out_focal_precond_diag, out_focal_precond_diag_num_alloc,
      out_focal_precond_tril, out_focal_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar