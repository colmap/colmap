#include "kernel_ThinPrismFisheyeCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibNormalizeKernel(float* precond_diag,
                                         unsigned int precond_diag_num_alloc,
                                         float* precond_tril,
                                         unsigned int precond_tril_num_alloc,
                                         float* njtr,
                                         unsigned int njtr_num_alloc,
                                         const float* const diag,
                                         float* out_normalized,
                                         unsigned int out_normalized_num_alloc,
                                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         28 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r1,
                                         r2,
                                         r3,
                                         r4);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         8 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         0 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r7,
                                         r6,
                                         r8,
                                         r9);
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         0 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r7,
                                         r10,
                                         r11);
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 1.00000000000000000e+00;
    r13 = r12 + r13;
    r14 = 9.99999999999999955e-07;
    r14 = r12 * r14;
    r8 = fmaf(r8, r13, r14);
    r8 = 1.0 / r8;
    r12 = r0 * r8;
    r15 = r6 * r12;
    r1 = fmaf(r5, r15, r1);
    r16 = r0 * r1;
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         8 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r19 = fmaf(r19, r13, r14);
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         4 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r21,
                                         r22,
                                         r23,
                                         r24);
    r21 = fmaf(r21, r13, r14);
    r10 = fmaf(r10, r13, r14);
    r10 = fmaf(r6, r15, r10);
    r10 = 1.0 / r10;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         20 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r6,
                                         r25,
                                         r26,
                                         r27);
    r26 = fmaf(r9, r15, r26);
    r25 = r26 * r26;
    r11 = fmaf(r11, r13, r14);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         12 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r28,
                                         r29,
                                         r30,
                                         r31);
    r32 = r28 * r0;
    r7 = fmaf(r7, r13, r14);
    r7 = 1.0 / r7;
    r32 = r32 * r7;
    r11 = fmaf(r28, r32, r11);
    r11 = 1.0 / r11;
    r3 = fmaf(r29, r32, r3);
    r28 = r3 * r3;
    r28 = fmaf(r11, r28, r10 * r25);
    r25 = r9 * r9;
    r33 = r29 * r29;
    r28 = fmaf(r8, r25, r28);
    r28 = fmaf(r7, r33, r28);
    r21 = fmaf(r0, r28, r21);
    r21 = 1.0 / r21;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         40 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r28,
                                         r33,
                                         r25,
                                         r34);
    r35 = r9 * r5;
    r36 = r26 * r1;
    r36 = fmaf(r10, r36, r8 * r35);
    r36 = fmaf(r0, r36, r34);
    r34 = r36 * r36;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         60 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r35,
                                         r37,
                                         r38,
                                         r39);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         24 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r40,
                                         r41,
                                         r42,
                                         r43);
    r43 = fmaf(r2, r15, r43);
    r44 = r43 * r1;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         56 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r45,
                                         r46,
                                         r47,
                                         r48);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         48 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r49,
                                         r50,
                                         r51,
                                         r52);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         32 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r53,
                                         r54,
                                         r55,
                                         r56);
    r53 = fmaf(r31, r32, r53);
    r57 = r53 * r11;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         16 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r58,
                                         r59,
                                         r60,
                                         r61);
    r54 = fmaf(r58, r32, r54);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         44 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r61,
                                         r62,
                                         r63,
                                         r64);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         36 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r65,
                                         r66,
                                         r67,
                                         r68);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         4 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r65,
                                         r69,
                                         r70,
                                         r71);
    r27 = fmaf(r65, r15, r27);
    r72 = r26 * r27;
    r4 = fmaf(r30, r32, r4);
    r73 = r3 * r4;
    r73 = fmaf(r11, r73, r10 * r72);
    r72 = r9 * r65;
    r73 = fmaf(r8, r72, r73);
    r74 = r29 * r30;
    r73 = fmaf(r7, r74, r73);
    r73 = fmaf(r0, r73, r67);
    r67 = r3 * r54;
    r41 = fmaf(r70, r15, r41);
    r74 = r26 * r41;
    r74 = fmaf(r10, r74, r11 * r67);
    r67 = r9 * r70;
    r74 = fmaf(r8, r67, r74);
    r72 = r29 * r58;
    r74 = fmaf(r7, r72, r74);
    r74 = fmaf(r0, r74, r28);
    r28 = r73 * r74;
    r72 = r4 * r54;
    r72 = fmaf(r11, r72, r21 * r28);
    r28 = r65 * r70;
    r72 = fmaf(r8, r28, r72);
    r67 = r27 * r41;
    r72 = fmaf(r10, r67, r72);
    r75 = r30 * r58;
    r72 = fmaf(r7, r75, r72);
    r72 = fmaf(r0, r72, r63);
    r40 = fmaf(r69, r15, r40);
    r63 = r40 * r10;
    r75 = fmaf(r26, r63, r3 * r57);
    r67 = r9 * r69;
    r75 = fmaf(r8, r67, r75);
    r28 = r29 * r31;
    r75 = fmaf(r7, r28, r75);
    r75 = fmaf(r0, r75, r68);
    r68 = r75 * r21;
    r28 = fmaf(r4, r57, r73 * r68);
    r67 = r65 * r69;
    r28 = fmaf(r8, r67, r28);
    r76 = r30 * r31;
    r28 = fmaf(r7, r76, r28);
    r28 = fmaf(r27, r63, r28);
    r28 = fmaf(r0, r28, r62);
    r22 = fmaf(r22, r13, r14);
    r62 = r27 * r27;
    r76 = r73 * r73;
    r76 = fmaf(r21, r76, r10 * r62);
    r62 = r4 * r4;
    r67 = r65 * r65;
    r77 = r30 * r30;
    r76 = fmaf(r11, r62, r76);
    r76 = fmaf(r8, r67, r76);
    r76 = fmaf(r7, r77, r76);
    r22 = fmaf(r0, r76, r22);
    r22 = 1.0 / r22;
    r76 = r28 * r22;
    r77 = fmaf(r72, r76, r54 * r57);
    r67 = r69 * r70;
    r77 = fmaf(r8, r67, r77);
    r62 = r31 * r58;
    r77 = fmaf(r7, r62, r77);
    r77 = fmaf(r74, r68, r77);
    r77 = fmaf(r41, r63, r77);
    r77 = fmaf(r0, r77, r52);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         52 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r52,
                                         r62,
                                         r67,
                                         r78);
    r56 = fmaf(r60, r32, r56);
    r79 = r4 * r56;
    r80 = r3 * r56;
    r81 = r26 * r43;
    r81 = fmaf(r10, r81, r11 * r80);
    r80 = r9 * r2;
    r81 = fmaf(r8, r80, r81);
    r82 = r29 * r60;
    r81 = fmaf(r7, r82, r81);
    r81 = fmaf(r0, r81, r25);
    r25 = r73 * r81;
    r25 = fmaf(r21, r25, r11 * r79);
    r79 = r27 * r43;
    r25 = fmaf(r10, r79, r25);
    r82 = r65 * r2;
    r25 = fmaf(r8, r82, r25);
    r80 = r30 * r60;
    r25 = fmaf(r7, r80, r25);
    r25 = fmaf(r0, r25, r49);
    r49 = fmaf(r56, r57, r25 * r76);
    r80 = r69 * r2;
    r49 = fmaf(r8, r80, r49);
    r82 = r31 * r60;
    r49 = fmaf(r7, r82, r49);
    r49 = fmaf(r81, r68, r49);
    r49 = fmaf(r43, r63, r49);
    r49 = fmaf(r0, r49, r62);
    r62 = r77 * r49;
    r23 = fmaf(r23, r13, r14);
    r28 = fmaf(r28, r76, r53 * r57);
    r53 = r69 * r69;
    r82 = r31 * r31;
    r28 = fmaf(r40, r63, r28);
    r28 = fmaf(r75, r68, r28);
    r28 = fmaf(r8, r53, r28);
    r28 = fmaf(r7, r82, r28);
    r23 = fmaf(r0, r28, r23);
    r23 = 1.0 / r23;
    r28 = r54 * r56;
    r28 = fmaf(r11, r28, r23 * r62);
    r62 = r74 * r81;
    r28 = fmaf(r21, r62, r28);
    r82 = r70 * r2;
    r28 = fmaf(r8, r82, r28);
    r53 = r58 * r60;
    r28 = fmaf(r7, r53, r28);
    r75 = r41 * r43;
    r28 = fmaf(r10, r75, r28);
    r40 = r72 * r25;
    r28 = fmaf(r22, r40, r28);
    r28 = fmaf(r0, r28, r46);
    r46 = r41 * r1;
    r40 = r74 * r36;
    r40 = fmaf(r21, r40, r10 * r46);
    r46 = r65 * r5;
    r75 = r73 * r36;
    r75 = fmaf(r21, r75, r8 * r46);
    r46 = r27 * r1;
    r75 = fmaf(r10, r46, r75);
    r75 = fmaf(r0, r75, r50);
    r50 = fmaf(r75, r76, r1 * r63);
    r46 = r69 * r5;
    r50 = fmaf(r8, r46, r50);
    r50 = fmaf(r36, r68, r50);
    r50 = fmaf(r0, r50, r67);
    r67 = r50 * r23;
    r46 = r70 * r5;
    r40 = fmaf(r8, r46, r40);
    r53 = r72 * r75;
    r40 = fmaf(r22, r53, r40);
    r40 = fmaf(r77, r67, r40);
    r40 = fmaf(r0, r40, r47);
    r47 = r28 * r40;
    r24 = fmaf(r24, r13, r14);
    r53 = r54 * r54;
    r46 = r72 * r72;
    r46 = fmaf(r22, r46, r11 * r53);
    r53 = r41 * r41;
    r82 = r74 * r74;
    r62 = r70 * r70;
    r80 = r58 * r58;
    r79 = r77 * r77;
    r46 = fmaf(r10, r53, r46);
    r46 = fmaf(r21, r82, r46);
    r46 = fmaf(r8, r62, r46);
    r46 = fmaf(r7, r80, r46);
    r46 = fmaf(r23, r79, r46);
    r24 = fmaf(r0, r46, r24);
    r24 = 1.0 / r24;
    r47 = fmaf(r24, r47, r10 * r44);
    r42 = fmaf(r71, r15, r42);
    r44 = r42 * r1;
    r55 = fmaf(r59, r32, r55);
    r46 = r3 * r55;
    r79 = r26 * r42;
    r79 = fmaf(r10, r79, r11 * r46);
    r46 = r9 * r71;
    r79 = fmaf(r8, r46, r79);
    r80 = r29 * r59;
    r79 = fmaf(r7, r80, r79);
    r79 = fmaf(r0, r79, r33);
    r33 = fmaf(r55, r57, r79 * r68);
    r80 = r69 * r71;
    r33 = fmaf(r8, r80, r33);
    r46 = r79 * r73;
    r62 = r55 * r4;
    r62 = fmaf(r11, r62, r21 * r46);
    r46 = r42 * r27;
    r62 = fmaf(r10, r46, r62);
    r82 = r65 * r71;
    r62 = fmaf(r8, r82, r62);
    r53 = r30 * r59;
    r62 = fmaf(r7, r53, r62);
    r62 = fmaf(r0, r62, r64);
    r64 = r31 * r59;
    r33 = fmaf(r7, r64, r33);
    r33 = fmaf(r62, r76, r33);
    r33 = fmaf(r42, r63, r33);
    r33 = fmaf(r0, r33, r52);
    r44 = fmaf(r33, r67, r10 * r44);
    r52 = r62 * r75;
    r44 = fmaf(r22, r52, r44);
    r64 = r71 * r5;
    r44 = fmaf(r8, r64, r44);
    r80 = r79 * r36;
    r44 = fmaf(r21, r80, r44);
    r53 = r55 * r54;
    r82 = r70 * r71;
    r82 = fmaf(r8, r82, r11 * r53);
    r53 = r79 * r74;
    r82 = fmaf(r21, r53, r82);
    r46 = r33 * r77;
    r82 = fmaf(r23, r46, r82);
    r83 = r42 * r41;
    r82 = fmaf(r10, r83, r82);
    r84 = r58 * r59;
    r82 = fmaf(r7, r84, r82);
    r85 = r62 * r72;
    r82 = fmaf(r22, r85, r82);
    r82 = fmaf(r0, r82, r45);
    r45 = r82 * r24;
    r44 = fmaf(r40, r45, r44);
    r44 = fmaf(r0, r44, r37);
    r37 = r62 * r25;
    r80 = r55 * r56;
    r80 = fmaf(r11, r80, r22 * r37);
    r37 = r33 * r49;
    r80 = fmaf(r23, r37, r80);
    r64 = r71 * r2;
    r80 = fmaf(r8, r64, r80);
    r52 = r79 * r81;
    r80 = fmaf(r21, r52, r80);
    r85 = r42 * r43;
    r80 = fmaf(r10, r85, r80);
    r84 = r59 * r60;
    r80 = fmaf(r7, r84, r80);
    r80 = fmaf(r28, r45, r80);
    r80 = fmaf(r0, r80, r35);
    r17 = fmaf(r17, r13, r14);
    r35 = r79 * r79;
    r84 = r33 * r33;
    r84 = fmaf(r23, r84, r21 * r35);
    r35 = r55 * r55;
    r85 = r62 * r62;
    r52 = r42 * r42;
    r64 = r71 * r71;
    r37 = r59 * r59;
    r84 = fmaf(r11, r35, r84);
    r84 = fmaf(r22, r85, r84);
    r84 = fmaf(r10, r52, r84);
    r84 = fmaf(r82, r45, r84);
    r84 = fmaf(r8, r64, r84);
    r84 = fmaf(r7, r37, r84);
    r17 = fmaf(r0, r84, r17);
    r17 = 1.0 / r17;
    r84 = r80 * r17;
    r37 = r2 * r5;
    r47 = fmaf(r8, r37, r47);
    r64 = r25 * r75;
    r47 = fmaf(r22, r64, r47);
    r82 = r81 * r36;
    r47 = fmaf(r21, r82, r47);
    r47 = fmaf(r49, r67, r47);
    r47 = fmaf(r44, r84, r47);
    r47 = fmaf(r0, r47, r39);
    r18 = fmaf(r18, r13, r14);
    r39 = r56 * r56;
    r82 = r25 * r25;
    r82 = fmaf(r22, r82, r11 * r39);
    r39 = r49 * r49;
    r64 = r43 * r43;
    r37 = r81 * r81;
    r52 = r28 * r28;
    r85 = r2 * r2;
    r35 = r60 * r60;
    r82 = fmaf(r23, r39, r82);
    r82 = fmaf(r10, r64, r82);
    r82 = fmaf(r21, r37, r82);
    r82 = fmaf(r80, r84, r82);
    r82 = fmaf(r24, r52, r82);
    r82 = fmaf(r8, r85, r82);
    r82 = fmaf(r7, r35, r82);
    r18 = fmaf(r0, r82, r18);
    r18 = 1.0 / r18;
    r82 = r47 * r18;
    r47 = fmaf(r47, r82, r21 * r34);
    r34 = r40 * r40;
    r35 = r1 * r1;
    r85 = r75 * r75;
    r52 = r5 * r5;
    r80 = r44 * r44;
    r47 = fmaf(r24, r34, r47);
    r47 = fmaf(r10, r35, r47);
    r47 = fmaf(r50, r67, r47);
    r47 = fmaf(r22, r85, r47);
    r47 = fmaf(r8, r52, r47);
    r47 = fmaf(r17, r80, r47);
    r19 = fmaf(r0, r47, r19);
    r19 = 1.0 / r19;
    ReadIdx4<1024, float, float, float4>(
        njtr, 8 * njtr_num_alloc, global_thread_idx, r47, r80, r52, r85);
    r50 = r0 * r67;
    ReadIdx4<1024, float, float, float4>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r35, r34, r37, r64);
    ReadIdx4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r39, r83, r46, r53);
    r86 = r31 * r83;
    r86 = r86 * r0;
    r86 = fmaf(r7, r86, r37);
    r46 = fmaf(r39, r15, r46);
    r37 = r0 * r46;
    r86 = fmaf(r63, r37, r86);
    r87 = r69 * r39;
    r86 = fmaf(r12, r87, r86);
    r53 = fmaf(r83, r32, r53);
    r88 = r0 * r53;
    r86 = fmaf(r57, r88, r86);
    r89 = r30 * r83;
    r89 = r89 * r0;
    r89 = fmaf(r7, r89, r34);
    r34 = r0 * r27;
    r34 = r34 * r46;
    r89 = fmaf(r10, r34, r89);
    r90 = r65 * r39;
    r89 = fmaf(r12, r90, r89);
    r91 = r0 * r4;
    r91 = r91 * r53;
    r89 = fmaf(r11, r91, r89);
    r92 = r0 * r73;
    r93 = r29 * r83;
    r93 = r93 * r0;
    r93 = fmaf(r7, r93, r35);
    r35 = r9 * r39;
    r93 = fmaf(r12, r35, r93);
    r94 = r0 * r3;
    r94 = r94 * r53;
    r93 = fmaf(r11, r94, r93);
    r95 = r0 * r26;
    r95 = r95 * r46;
    r93 = fmaf(r10, r95, r93);
    r92 = r92 * r93;
    r89 = fmaf(r21, r92, r89);
    r92 = r0 * r89;
    r86 = fmaf(r76, r92, r86);
    r91 = r0 * r93;
    r86 = fmaf(r68, r91, r86);
    r52 = fmaf(r86, r50, r52);
    r91 = r5 * r39;
    r52 = fmaf(r12, r91, r52);
    r92 = r0 * r93;
    r92 = r92 * r36;
    r52 = fmaf(r21, r92, r52);
    r88 = r0 * r89;
    r88 = r88 * r75;
    r52 = fmaf(r22, r88, r52);
    r87 = r0 * r79;
    r87 = r87 * r93;
    r87 = fmaf(r21, r87, r47);
    r47 = r0 * r33;
    r47 = r47 * r86;
    r87 = fmaf(r23, r47, r87);
    r37 = r0 * r77;
    r37 = r37 * r86;
    r37 = fmaf(r23, r37, r64);
    r64 = r58 * r83;
    r64 = r64 * r0;
    r37 = fmaf(r7, r64, r37);
    r90 = r0 * r54;
    r90 = r90 * r53;
    r37 = fmaf(r11, r90, r37);
    r34 = r0 * r41;
    r34 = r34 * r46;
    r37 = fmaf(r10, r34, r37);
    r95 = r70 * r39;
    r37 = fmaf(r12, r95, r37);
    r94 = r0 * r74;
    r94 = r94 * r93;
    r37 = fmaf(r21, r94, r37);
    r35 = r0 * r72;
    r35 = r35 * r89;
    r37 = fmaf(r22, r35, r37);
    r35 = r0 * r37;
    r87 = fmaf(r45, r35, r87);
    r94 = r59 * r83;
    r94 = r94 * r0;
    r87 = fmaf(r7, r94, r87);
    r95 = r0 * r62;
    r95 = r95 * r89;
    r87 = fmaf(r22, r95, r87);
    r34 = r0 * r42;
    r34 = r34 * r46;
    r87 = fmaf(r10, r34, r87);
    r90 = r71 * r39;
    r87 = fmaf(r12, r90, r87);
    r64 = r0 * r55;
    r64 = r64 * r53;
    r87 = fmaf(r11, r64, r87);
    r64 = r0 * r87;
    r64 = fmaf(r84, r64, r80);
    r80 = r0 * r43;
    r80 = r80 * r46;
    r64 = fmaf(r10, r80, r64);
    r90 = r60 * r83;
    r90 = r90 * r0;
    r64 = fmaf(r7, r90, r64);
    r34 = r0 * r25;
    r34 = r34 * r89;
    r64 = fmaf(r22, r34, r64);
    r95 = r0 * r49;
    r95 = r95 * r86;
    r64 = fmaf(r23, r95, r64);
    r94 = r0 * r28;
    r94 = r94 * r37;
    r64 = fmaf(r24, r94, r64);
    r35 = r0 * r81;
    r35 = r35 * r93;
    r64 = fmaf(r21, r35, r64);
    r47 = r2 * r39;
    r64 = fmaf(r12, r47, r64);
    r96 = r0 * r56;
    r96 = r96 * r53;
    r64 = fmaf(r11, r96, r64);
    r96 = r0 * r64;
    r52 = fmaf(r82, r96, r52);
    r47 = r0 * r87;
    r47 = r47 * r44;
    r52 = fmaf(r17, r47, r52);
    r35 = r0 * r37;
    r35 = r35 * r40;
    r52 = fmaf(r24, r35, r52);
    r94 = r0 * r46;
    r94 = r94 * r1;
    r52 = fmaf(r10, r94, r52);
    r66 = fmaf(r6, r32, r66);
    r94 = r3 * r66;
    r35 = r29 * r6;
    r35 = fmaf(r7, r35, r11 * r94);
    r35 = fmaf(r0, r35, r61);
    r61 = r35 * r36;
    r94 = r79 * r35;
    r47 = r73 * r35;
    r96 = r30 * r6;
    r96 = fmaf(r7, r96, r21 * r47);
    r47 = r4 * r66;
    r96 = fmaf(r11, r47, r96);
    r96 = fmaf(r0, r96, r51);
    r51 = r31 * r6;
    r51 = fmaf(r7, r51, r96 * r76);
    r51 = fmaf(r35, r68, r51);
    r51 = fmaf(r66, r57, r51);
    r51 = fmaf(r0, r51, r78);
    r78 = r33 * r51;
    r78 = fmaf(r23, r78, r21 * r94);
    r94 = r77 * r51;
    r47 = r74 * r35;
    r47 = fmaf(r21, r47, r23 * r94);
    r94 = r58 * r6;
    r47 = fmaf(r7, r94, r47);
    r88 = r72 * r96;
    r47 = fmaf(r22, r88, r47);
    r92 = r54 * r66;
    r47 = fmaf(r11, r92, r47);
    r47 = fmaf(r0, r47, r48);
    r48 = r62 * r96;
    r78 = fmaf(r22, r48, r78);
    r92 = r59 * r6;
    r78 = fmaf(r7, r92, r78);
    r88 = r55 * r66;
    r78 = fmaf(r11, r88, r78);
    r78 = fmaf(r47, r45, r78);
    r78 = fmaf(r0, r78, r38);
    r38 = r78 * r44;
    r38 = fmaf(r17, r38, r21 * r61);
    r61 = r96 * r75;
    r38 = fmaf(r22, r61, r38);
    r88 = r47 * r40;
    r38 = fmaf(r24, r88, r38);
    ReadIdx2<1024, float, float, float2>(
        precond_tril, 64 * precond_tril_num_alloc, global_thread_idx, r92, r48);
    r48 = r81 * r35;
    r48 = fmaf(r21, r48, r78 * r84);
    r94 = r28 * r47;
    r48 = fmaf(r24, r94, r48);
    r91 = r25 * r96;
    r48 = fmaf(r22, r91, r48);
    r95 = r49 * r51;
    r48 = fmaf(r23, r95, r48);
    r34 = r60 * r6;
    r48 = fmaf(r7, r34, r48);
    r90 = r56 * r66;
    r48 = fmaf(r11, r90, r48);
    r48 = fmaf(r0, r48, r92);
    r38 = fmaf(r51, r67, r38);
    r38 = fmaf(r48, r82, r38);
    r88 = r38 * r19;
    r13 = fmaf(r20, r13, r14);
    r20 = r66 * r66;
    r14 = r48 * r48;
    r14 = fmaf(r18, r14, r11 * r20);
    r20 = r51 * r51;
    r67 = r96 * r96;
    r61 = r78 * r78;
    r92 = r35 * r35;
    r90 = r6 * r6;
    r34 = r47 * r47;
    r14 = fmaf(r23, r20, r14);
    r14 = fmaf(r22, r67, r14);
    r14 = fmaf(r17, r61, r14);
    r14 = fmaf(r21, r92, r14);
    r14 = fmaf(r38, r88, r14);
    r14 = fmaf(r7, r90, r14);
    r14 = fmaf(r24, r34, r14);
    r13 = fmaf(r0, r14, r13);
    r13 = 1.0 / r13;
    r85 = fmaf(r52, r88, r85);
    r14 = r6 * r83;
    r14 = r14 * r0;
    r85 = fmaf(r7, r14, r85);
    r34 = r0 * r93;
    r34 = r34 * r35;
    r85 = fmaf(r21, r34, r85);
    r90 = r0 * r53;
    r90 = r90 * r66;
    r85 = fmaf(r11, r90, r85);
    r38 = r0 * r64;
    r38 = r38 * r48;
    r85 = fmaf(r18, r38, r85);
    r92 = r0 * r87;
    r92 = r92 * r78;
    r85 = fmaf(r17, r92, r85);
    r61 = r0 * r86;
    r61 = r61 * r51;
    r85 = fmaf(r23, r61, r85);
    r67 = r0 * r89;
    r67 = r67 * r96;
    r85 = fmaf(r22, r67, r85);
    r20 = r0 * r37;
    r20 = r20 * r47;
    r85 = fmaf(r24, r20, r85);
    r85 = r13 * r85;
    r88 = fmaf(r85, r88, r52 * r19);
    r16 = r16 * r88;
    r19 = r0 * r41;
    r52 = r0 * r44;
    r52 = r52 * r88;
    r13 = r48 * r18;
    r20 = r0 * r85;
    r13 = fmaf(r20, r13, r64 * r18);
    r67 = r0 * r88;
    r13 = fmaf(r82, r67, r13);
    r67 = r0 * r13;
    r67 = fmaf(r84, r67, r17 * r52);
    r52 = r78 * r17;
    r67 = fmaf(r20, r52, r67);
    r67 = fmaf(r87, r17, r67);
    r52 = r0 * r67;
    r52 = fmaf(r45, r52, r37 * r24);
    r45 = r0 * r40;
    r45 = r45 * r88;
    r52 = fmaf(r24, r45, r52);
    r84 = r47 * r24;
    r52 = fmaf(r20, r84, r52);
    r82 = r0 * r28;
    r82 = r82 * r13;
    r52 = fmaf(r24, r82, r52);
    r19 = r19 * r52;
    r19 = fmaf(r10, r19, r10 * r16);
    r16 = r0 * r77;
    r16 = r16 * r52;
    r16 = fmaf(r23, r16, r88 * r50);
    r50 = r0 * r33;
    r50 = r50 * r67;
    r16 = fmaf(r23, r50, r16);
    r82 = r0 * r49;
    r82 = r82 * r13;
    r16 = fmaf(r23, r82, r16);
    r84 = r51 * r23;
    r16 = fmaf(r20, r84, r16);
    r16 = fmaf(r86, r23, r16);
    r84 = r0 * r16;
    r19 = fmaf(r63, r84, r19);
    r63 = r0 * r42;
    r63 = r63 * r67;
    r19 = fmaf(r10, r63, r19);
    r82 = r0 * r27;
    r50 = r0 * r16;
    r45 = r0 * r62;
    r45 = r45 * r67;
    r45 = fmaf(r22, r45, r76 * r50);
    r50 = r0 * r25;
    r50 = r50 * r13;
    r45 = fmaf(r22, r50, r45);
    r76 = r0 * r72;
    r76 = r76 * r52;
    r45 = fmaf(r22, r76, r45);
    r61 = r0 * r75;
    r61 = r61 * r88;
    r45 = fmaf(r22, r61, r45);
    r92 = r96 * r22;
    r45 = fmaf(r20, r92, r45);
    r45 = fmaf(r89, r22, r45);
    r82 = r82 * r45;
    r19 = fmaf(r10, r82, r19);
    r92 = r0 * r43;
    r92 = r92 * r13;
    r19 = fmaf(r10, r92, r19);
    r61 = r0 * r26;
    r76 = r0 * r74;
    r76 = r76 * r52;
    r50 = r0 * r36;
    r50 = r50 * r88;
    r50 = fmaf(r21, r50, r21 * r76);
    r76 = r35 * r21;
    r50 = fmaf(r20, r76, r50);
    r38 = r0 * r16;
    r50 = fmaf(r68, r38, r50);
    r68 = r0 * r81;
    r68 = r68 * r13;
    r50 = fmaf(r21, r68, r50);
    r90 = r0 * r79;
    r90 = r90 * r67;
    r50 = fmaf(r21, r90, r50);
    r34 = r0 * r73;
    r34 = r34 * r45;
    r50 = fmaf(r21, r34, r50);
    r50 = fmaf(r93, r21, r50);
    r61 = r61 * r50;
    r19 = fmaf(r10, r61, r19);
    r19 = fmaf(r46, r10, r19);
    r61 = r2 * r13;
    r8 = fmaf(r39, r8, r12 * r61);
    r61 = r65 * r45;
    r8 = fmaf(r12, r61, r8);
    r92 = r9 * r50;
    r8 = fmaf(r12, r92, r8);
    r82 = r71 * r67;
    r8 = fmaf(r12, r82, r8);
    r63 = r5 * r88;
    r8 = fmaf(r12, r63, r8);
    r10 = r70 * r52;
    r8 = fmaf(r12, r10, r8);
    r84 = r69 * r16;
    r8 = fmaf(r12, r84, r8);
    r8 = fmaf(r19, r15, r8);
    r84 = r0 * r4;
    r84 = r84 * r45;
    r84 = fmaf(r11, r84, r53 * r11);
    r10 = r66 * r11;
    r84 = fmaf(r20, r10, r84);
    r63 = r0 * r3;
    r63 = r63 * r50;
    r84 = fmaf(r11, r63, r84);
    r82 = r0 * r54;
    r82 = r82 * r52;
    r84 = fmaf(r11, r82, r84);
    r92 = r0 * r16;
    r84 = fmaf(r57, r92, r84);
    r57 = r0 * r56;
    r57 = r57 * r13;
    r84 = fmaf(r11, r57, r84);
    r15 = r0 * r55;
    r15 = r15 * r67;
    r84 = fmaf(r11, r15, r84);
    r15 = r6 * r7;
    r15 = fmaf(r20, r15, r83 * r7);
    r20 = r59 * r0;
    r20 = r20 * r67;
    r15 = fmaf(r7, r20, r15);
    r57 = r31 * r0;
    r57 = r57 * r16;
    r15 = fmaf(r7, r57, r15);
    r92 = r29 * r0;
    r92 = r92 * r50;
    r15 = fmaf(r7, r92, r15);
    r82 = r60 * r0;
    r82 = r82 * r13;
    r15 = fmaf(r7, r82, r15);
    r63 = r58 * r0;
    r63 = r63 * r52;
    r15 = fmaf(r7, r63, r15);
    r10 = r30 * r0;
    r10 = r10 * r45;
    r15 = fmaf(r7, r10, r15);
    r15 = fmaf(r84, r32, r15);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          0 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r8,
                                          r15,
                                          r19,
                                          r84);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          4 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r50,
                                          r45,
                                          r16,
                                          r52);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          8 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r13,
                                          r88,
                                          r85);
  };
}

void ThinPrismFisheyeCalibNormalize(float* precond_diag,
                                    unsigned int precond_diag_num_alloc,
                                    float* precond_tril,
                                    unsigned int precond_tril_num_alloc,
                                    float* njtr,
                                    unsigned int njtr_num_alloc,
                                    const float* const diag,
                                    float* out_normalized,
                                    unsigned int out_normalized_num_alloc,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibNormalizeKernel<<<n_blocks, 1024>>>(
      precond_diag,
      precond_diag_num_alloc,
      precond_tril,
      precond_tril_num_alloc,
      njtr,
      njtr_num_alloc,
      diag,
      out_normalized,
      out_normalized_num_alloc,
      problem_size);
}

}  // namespace caspar