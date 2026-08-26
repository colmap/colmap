#include "kernel_ThinPrismFisheyeFocalAndExtra_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraNormalizeKernel(
        float* precond_diag,
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
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         32 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r1,
                                         r2,
                                         r3,
                                         r4);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         24 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7,
                                         r8);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         0 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r10,
                                         r11,
                                         r12);
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         0 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r13,
                                         r14,
                                         r15);
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = 1.00000000000000000e+00;
    r17 = r16 + r17;
    r18 = 9.99999999999999955e-07;
    r18 = r16 * r18;
    r9 = fmaf(r9, r17, r18);
    r9 = 1.0 / r9;
    r16 = r11 * r9;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         8 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r19,
                                         r20,
                                         r21,
                                         r22);
    r19 = r21 * r22;
    r13 = fmaf(r13, r17, r18);
    r13 = 1.0 / r13;
    r19 = fmaf(r13, r19, r12 * r16);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         16 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r23,
                                         r24,
                                         r25,
                                         r26);
    r27 = r20 * r22;
    r28 = r10 * r12;
    r28 = fmaf(r9, r28, r13 * r27);
    r28 = fmaf(r0, r28, r25);
    r25 = r20 * r21;
    r25 = fmaf(r10, r16, r13 * r25);
    r25 = fmaf(r0, r25, r24);
    r24 = r20 * r20;
    r27 = r10 * r10;
    r27 = fmaf(r9, r27, r13 * r24);
    r27 = fmaf(r0, r27, r18);
    r27 = fmaf(r14, r17, r27);
    r27 = 1.0 / r27;
    r14 = r25 * r27;
    r19 = fmaf(r28, r14, r19);
    r19 = fmaf(r0, r19, r5);
    r15 = fmaf(r15, r17, r18);
    r11 = fmaf(r11, r16, r25 * r14);
    r25 = r21 * r21;
    r11 = fmaf(r13, r25, r11);
    r15 = fmaf(r0, r11, r15);
    r15 = 1.0 / r15;
    r11 = r19 * r15;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         28 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r25,
                                         r5,
                                         r24,
                                         r29);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         20 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r30,
                                         r31,
                                         r32,
                                         r33);
    r34 = r20 * r23;
    r35 = r0 * r13;
    r34 = fmaf(r35, r34, r33);
    r33 = r21 * r23;
    r33 = fmaf(r13, r33, r34 * r14);
    r33 = fmaf(r0, r33, r5);
    r5 = r28 * r34;
    r5 = fmaf(r27, r5, r33 * r11);
    r36 = r22 * r23;
    r5 = fmaf(r13, r36, r5);
    r5 = fmaf(r0, r5, r3);
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         4 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r36,
                                         r37,
                                         r38);
    r3 = fmaf(r3, r17, r18);
    r39 = r28 * r28;
    r39 = fmaf(r27, r39, r19 * r11);
    r19 = r12 * r12;
    r40 = r22 * r22;
    r39 = fmaf(r9, r19, r39);
    r39 = fmaf(r13, r40, r39);
    r3 = fmaf(r0, r39, r3);
    r3 = 1.0 / r3;
    r39 = r5 * r3;
    ReadIdx2<1024, float, float, float2>(
        precond_diag, 8 * precond_diag_num_alloc, global_thread_idx, r40, r19);
    r19 = fmaf(r19, r17, r18);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         4 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r41,
                                         r42,
                                         r43,
                                         r44);
    r45 = r10 * r44;
    r45 = r45 * r0;
    r45 = fmaf(r9, r45, r32);
    r32 = r45 * r34;
    r46 = fmaf(r45, r14, r44 * r16);
    r46 = fmaf(r0, r46, r25);
    r25 = r46 * r33;
    r25 = fmaf(r15, r25, r27 * r32);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         40 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r32,
                                         r47,
                                         r48,
                                         r49);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         36 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r50,
                                         r51,
                                         r52,
                                         r53);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         12 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r54,
                                         r55,
                                         r56,
                                         r57);
    r57 = r20 * r54;
    r58 = r10 * r41;
    r58 = fmaf(r9, r58, r13 * r57);
    r58 = fmaf(r0, r58, r26);
    r26 = r58 * r28;
    r57 = fmaf(r58, r14, r41 * r16);
    r59 = r21 * r54;
    r57 = fmaf(r13, r59, r57);
    r57 = fmaf(r0, r57, r6);
    r26 = fmaf(r57, r11, r27 * r26);
    r6 = r12 * r41;
    r26 = fmaf(r9, r6, r26);
    r59 = r22 * r54;
    r26 = fmaf(r13, r59, r26);
    r26 = fmaf(r0, r26, r24);
    r24 = r45 * r28;
    r24 = fmaf(r46, r11, r27 * r24);
    r59 = r12 * r44;
    r24 = fmaf(r9, r59, r24);
    r24 = fmaf(r0, r24, r2);
    r2 = r26 * r24;
    r59 = r45 * r58;
    r59 = fmaf(r27, r59, r3 * r2);
    r2 = r46 * r57;
    r59 = fmaf(r15, r2, r59);
    r6 = r41 * r44;
    r59 = fmaf(r9, r6, r59);
    r59 = fmaf(r0, r59, r51);
    r51 = r20 * r55;
    r6 = r10 * r42;
    r6 = fmaf(r9, r6, r13 * r51);
    r6 = fmaf(r0, r6, r30);
    r30 = fmaf(r42, r16, r6 * r14);
    r51 = r21 * r55;
    r30 = fmaf(r13, r51, r30);
    r30 = fmaf(r0, r30, r7);
    r7 = r12 * r42;
    r7 = fmaf(r9, r7, r30 * r11);
    r51 = r22 * r55;
    r7 = fmaf(r13, r51, r7);
    r2 = r28 * r6;
    r7 = fmaf(r27, r2, r7);
    r7 = fmaf(r0, r7, r29);
    r29 = r26 * r7;
    r2 = r41 * r42;
    r2 = fmaf(r9, r2, r3 * r29);
    r29 = r57 * r30;
    r2 = fmaf(r15, r29, r2);
    r51 = r54 * r55;
    r2 = fmaf(r13, r51, r2);
    r60 = r58 * r6;
    r2 = fmaf(r27, r60, r2);
    r2 = fmaf(r0, r2, r4);
    r4 = r59 * r2;
    r36 = fmaf(r36, r17, r18);
    r60 = r57 * r57;
    r51 = r26 * r26;
    r51 = fmaf(r3, r51, r15 * r60);
    r60 = r58 * r58;
    r29 = r41 * r41;
    r61 = r54 * r54;
    r51 = fmaf(r27, r60, r51);
    r51 = fmaf(r9, r29, r51);
    r51 = fmaf(r13, r61, r51);
    r36 = fmaf(r0, r51, r36);
    r36 = 1.0 / r36;
    r51 = r45 * r6;
    r51 = fmaf(r27, r51, r36 * r4);
    r4 = r42 * r44;
    r51 = fmaf(r9, r4, r51);
    r61 = r24 * r7;
    r51 = fmaf(r3, r61, r51);
    r29 = r46 * r30;
    r51 = fmaf(r15, r29, r51);
    r51 = fmaf(r0, r51, r32);
    r32 = r7 * r5;
    r29 = r30 * r33;
    r29 = fmaf(r15, r29, r3 * r32);
    r32 = r6 * r34;
    r29 = fmaf(r27, r32, r29);
    r61 = r26 * r5;
    r4 = r57 * r33;
    r4 = fmaf(r15, r4, r3 * r61);
    r61 = r54 * r23;
    r4 = fmaf(r13, r61, r4);
    r60 = r58 * r34;
    r4 = fmaf(r27, r60, r4);
    r4 = fmaf(r0, r4, r52);
    r52 = r2 * r4;
    r29 = fmaf(r36, r52, r29);
    r60 = r55 * r23;
    r29 = fmaf(r13, r60, r29);
    r29 = fmaf(r0, r29, r47);
    r47 = r51 * r29;
    r37 = fmaf(r37, r17, r18);
    r60 = r30 * r30;
    r52 = r2 * r2;
    r52 = fmaf(r36, r52, r15 * r60);
    r60 = r42 * r42;
    r32 = r6 * r6;
    r61 = r55 * r55;
    r62 = r7 * r7;
    r52 = fmaf(r9, r60, r52);
    r52 = fmaf(r27, r32, r52);
    r52 = fmaf(r13, r61, r52);
    r52 = fmaf(r3, r62, r52);
    r37 = fmaf(r0, r52, r37);
    r37 = 1.0 / r37;
    r25 = fmaf(r37, r47, r25);
    r52 = r20 * r56;
    r62 = r10 * r43;
    r62 = fmaf(r9, r62, r13 * r52);
    r62 = fmaf(r0, r62, r31);
    r31 = r45 * r62;
    r52 = fmaf(r43, r16, r62 * r14);
    r61 = r21 * r56;
    r52 = fmaf(r13, r61, r52);
    r52 = fmaf(r0, r52, r8);
    r8 = r12 * r43;
    r8 = fmaf(r9, r8, r52 * r11);
    r61 = r22 * r56;
    r8 = fmaf(r13, r61, r8);
    r32 = r28 * r62;
    r8 = fmaf(r27, r32, r8);
    r8 = fmaf(r0, r8, r1);
    r1 = r8 * r3;
    r31 = fmaf(r24, r1, r27 * r31);
    r32 = r30 * r52;
    r61 = r41 * r43;
    r61 = fmaf(r9, r61, r26 * r1);
    r60 = r54 * r56;
    r61 = fmaf(r13, r60, r61);
    r63 = r57 * r52;
    r61 = fmaf(r15, r63, r61);
    r64 = r58 * r62;
    r61 = fmaf(r27, r64, r61);
    r61 = fmaf(r0, r61, r50);
    r50 = r61 * r36;
    r32 = fmaf(r2, r50, r15 * r32);
    r64 = r6 * r62;
    r32 = fmaf(r27, r64, r32);
    r63 = r42 * r43;
    r32 = fmaf(r9, r63, r32);
    r60 = r55 * r56;
    r32 = fmaf(r13, r60, r32);
    r32 = fmaf(r7, r1, r32);
    r32 = fmaf(r0, r32, r53);
    r53 = r32 * r37;
    r60 = r43 * r44;
    r31 = fmaf(r9, r60, r31);
    r63 = r46 * r52;
    r31 = fmaf(r15, r63, r31);
    r31 = fmaf(r51, r53, r31);
    r31 = fmaf(r59, r50, r31);
    r31 = fmaf(r0, r31, r48);
    r48 = r52 * r33;
    r48 = fmaf(r5, r1, r15 * r48);
    r63 = r62 * r34;
    r48 = fmaf(r27, r63, r48);
    r60 = r56 * r23;
    r48 = fmaf(r13, r60, r48);
    r48 = fmaf(r4, r50, r48);
    r48 = fmaf(r29, r53, r48);
    r48 = fmaf(r0, r48, r49);
    r38 = fmaf(r38, r17, r18);
    r8 = fmaf(r8, r1, r32 * r53);
    r32 = r43 * r43;
    r49 = r52 * r52;
    r60 = r62 * r62;
    r63 = r56 * r56;
    r8 = fmaf(r61, r50, r8);
    r8 = fmaf(r9, r32, r8);
    r8 = fmaf(r15, r49, r8);
    r8 = fmaf(r27, r60, r8);
    r8 = fmaf(r13, r63, r8);
    r38 = fmaf(r0, r8, r38);
    r38 = 1.0 / r38;
    r8 = r48 * r38;
    r63 = r59 * r4;
    r25 = fmaf(r36, r63, r25);
    r60 = r24 * r5;
    r25 = fmaf(r3, r60, r25);
    r25 = fmaf(r31, r8, r25);
    r17 = fmaf(r40, r17, r18);
    r40 = r46 * r46;
    r18 = r59 * r59;
    r18 = fmaf(r36, r18, r15 * r40);
    r40 = r24 * r24;
    r60 = r45 * r45;
    r63 = r51 * r51;
    r47 = r31 * r31;
    r49 = r44 * r44;
    r18 = fmaf(r3, r40, r18);
    r18 = fmaf(r27, r60, r18);
    r18 = fmaf(r37, r63, r18);
    r18 = fmaf(r38, r47, r18);
    r18 = fmaf(r9, r49, r18);
    r17 = fmaf(r0, r18, r17);
    r17 = 1.0 / r17;
    r18 = r25 * r17;
    r48 = fmaf(r48, r8, r25 * r18);
    r25 = r4 * r4;
    r49 = r29 * r29;
    r47 = r34 * r34;
    r63 = r33 * r33;
    r60 = r23 * r23;
    r40 = r5 * r5;
    r48 = fmaf(r36, r25, r48);
    r48 = fmaf(r37, r49, r48);
    r48 = fmaf(r27, r47, r48);
    r48 = fmaf(r15, r63, r48);
    r48 = fmaf(r13, r60, r48);
    r48 = fmaf(r3, r40, r48);
    r19 = fmaf(r0, r48, r19);
    r19 = 1.0 / r19;
    ReadIdx2<1024, float, float, float2>(
        njtr, 8 * njtr_num_alloc, global_thread_idx, r48, r40);
    ReadIdx4<1024, float, float, float4>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r60, r63, r47, r49);
    ReadIdx4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r25, r32, r61, r64);
    r65 = r32 * r35;
    r49 = fmaf(r56, r65, r49);
    r66 = r43 * r25;
    r66 = r66 * r0;
    r49 = fmaf(r9, r66, r49);
    r67 = r0 * r52;
    r64 = fmaf(r21, r65, r64);
    r68 = r25 * r0;
    r64 = fmaf(r16, r68, r64);
    r69 = r10 * r25;
    r69 = r69 * r0;
    r69 = fmaf(r9, r69, r61);
    r69 = fmaf(r20, r65, r69);
    r61 = r0 * r69;
    r64 = fmaf(r14, r61, r64);
    r67 = r67 * r64;
    r49 = fmaf(r15, r67, r49);
    r53 = r0 * r53;
    r47 = fmaf(r55, r65, r47);
    r61 = r42 * r25;
    r61 = r61 * r0;
    r47 = fmaf(r9, r61, r47);
    r68 = r0 * r2;
    r70 = r0 * r58;
    r70 = r70 * r69;
    r70 = fmaf(r27, r70, r63);
    r63 = r0 * r57;
    r63 = r63 * r64;
    r70 = fmaf(r15, r63, r70);
    r71 = r41 * r25;
    r71 = r71 * r0;
    r70 = fmaf(r9, r71, r70);
    r72 = r0 * r26;
    r60 = fmaf(r22, r65, r60);
    r73 = r0 * r28;
    r73 = r73 * r69;
    r60 = fmaf(r27, r73, r60);
    r74 = r12 * r25;
    r74 = r74 * r0;
    r60 = fmaf(r9, r74, r60);
    r75 = r0 * r64;
    r60 = fmaf(r11, r75, r60);
    r72 = r72 * r60;
    r70 = fmaf(r3, r72, r70);
    r70 = fmaf(r54, r65, r70);
    r68 = r68 * r70;
    r47 = fmaf(r36, r68, r47);
    r72 = r0 * r6;
    r72 = r72 * r69;
    r47 = fmaf(r27, r72, r47);
    r71 = r0 * r7;
    r71 = r71 * r60;
    r47 = fmaf(r3, r71, r47);
    r63 = r0 * r30;
    r63 = r63 * r64;
    r47 = fmaf(r15, r63, r47);
    r63 = r0 * r62;
    r63 = r63 * r69;
    r49 = fmaf(r27, r63, r49);
    r71 = r0 * r60;
    r49 = fmaf(r1, r71, r49);
    r72 = r0 * r70;
    r49 = fmaf(r50, r72, r49);
    r49 = fmaf(r47, r53, r49);
    r72 = r0 * r49;
    r72 = fmaf(r8, r72, r40);
    r40 = r0 * r33;
    r40 = r40 * r64;
    r72 = fmaf(r15, r40, r72);
    r71 = r0 * r46;
    r71 = r71 * r64;
    r71 = fmaf(r15, r71, r48);
    r48 = r0 * r45;
    r48 = r48 * r69;
    r71 = fmaf(r27, r48, r71);
    r63 = r44 * r25;
    r63 = r63 * r0;
    r71 = fmaf(r9, r63, r71);
    r67 = r0 * r24;
    r67 = r67 * r60;
    r71 = fmaf(r3, r67, r71);
    r66 = r0 * r31;
    r66 = r66 * r49;
    r71 = fmaf(r38, r66, r71);
    r68 = r0 * r59;
    r68 = r68 * r70;
    r71 = fmaf(r36, r68, r71);
    r61 = r0 * r51;
    r61 = r61 * r47;
    r71 = fmaf(r37, r61, r71);
    r61 = r0 * r5;
    r61 = r61 * r60;
    r72 = fmaf(r3, r61, r72);
    r68 = r0 * r4;
    r68 = r68 * r70;
    r72 = fmaf(r36, r68, r72);
    r66 = r0 * r29;
    r66 = r66 * r47;
    r72 = fmaf(r37, r66, r72);
    r67 = r0 * r34;
    r67 = r67 * r69;
    r72 = fmaf(r27, r67, r72);
    r72 = fmaf(r23, r65, r72);
    r72 = fmaf(r71, r18, r72);
    r72 = r19 * r72;
    r19 = r0 * r72;
    r67 = r0 * r24;
    r17 = fmaf(r71, r17, r72 * r18);
    r67 = r67 * r17;
    r67 = fmaf(r3, r67, r19 * r39);
    r39 = r0 * r7;
    r8 = fmaf(r49, r38, r19 * r8);
    r71 = r0 * r31;
    r71 = r71 * r17;
    r8 = fmaf(r38, r71, r8);
    r47 = fmaf(r47, r37, r8 * r53);
    r53 = r0 * r51;
    r53 = r53 * r17;
    r47 = fmaf(r37, r53, r47);
    r71 = r29 * r37;
    r47 = fmaf(r19, r71, r47);
    r39 = r39 * r47;
    r67 = fmaf(r3, r39, r67);
    r71 = r0 * r26;
    r53 = r0 * r2;
    r53 = r53 * r47;
    r38 = r0 * r59;
    r38 = r38 * r17;
    r38 = fmaf(r36, r38, r36 * r53);
    r53 = r0 * r8;
    r38 = fmaf(r50, r53, r38);
    r50 = r4 * r36;
    r38 = fmaf(r19, r50, r38);
    r38 = fmaf(r70, r36, r38);
    r71 = r71 * r38;
    r67 = fmaf(r3, r71, r67);
    r50 = r0 * r8;
    r67 = fmaf(r1, r50, r67);
    r67 = fmaf(r60, r3, r67);
    r50 = r0 * r67;
    r71 = r0 * r46;
    r71 = r71 * r17;
    r71 = fmaf(r15, r71, r11 * r50);
    r50 = r33 * r15;
    r71 = fmaf(r19, r50, r71);
    r11 = r0 * r52;
    r11 = r11 * r8;
    r71 = fmaf(r15, r11, r71);
    r39 = r0 * r57;
    r39 = r39 * r38;
    r71 = fmaf(r15, r39, r71);
    r1 = r0 * r30;
    r1 = r1 * r47;
    r71 = fmaf(r15, r1, r71);
    r71 = fmaf(r64, r15, r71);
    r1 = r0 * r62;
    r1 = r1 * r8;
    r39 = r0 * r58;
    r39 = r39 * r38;
    r39 = fmaf(r27, r39, r27 * r1);
    r1 = r34 * r27;
    r39 = fmaf(r19, r1, r39);
    r19 = r0 * r45;
    r19 = r19 * r17;
    r39 = fmaf(r27, r19, r39);
    r11 = r0 * r71;
    r39 = fmaf(r14, r11, r39);
    r14 = r0 * r6;
    r14 = r14 * r47;
    r39 = fmaf(r27, r14, r39);
    r50 = r0 * r28;
    r50 = r50 * r67;
    r39 = fmaf(r27, r50, r39);
    r39 = fmaf(r69, r27, r39);
    r50 = r44 * r0;
    r50 = r50 * r17;
    r50 = fmaf(r9, r50, r25 * r9);
    r14 = r43 * r0;
    r14 = r14 * r8;
    r50 = fmaf(r9, r14, r50);
    r11 = r0 * r71;
    r50 = fmaf(r16, r11, r50);
    r16 = r41 * r0;
    r16 = r16 * r38;
    r50 = fmaf(r9, r16, r50);
    r19 = r42 * r0;
    r19 = r19 * r47;
    r50 = fmaf(r9, r19, r50);
    r1 = r10 * r0;
    r1 = r1 * r39;
    r50 = fmaf(r9, r1, r50);
    r53 = r12 * r0;
    r53 = r53 * r67;
    r50 = fmaf(r9, r53, r50);
    r53 = r56 * r8;
    r53 = fmaf(r35, r53, r32 * r13);
    r13 = r55 * r47;
    r53 = fmaf(r35, r13, r53);
    r32 = r21 * r71;
    r53 = fmaf(r35, r32, r53);
    r1 = r54 * r38;
    r53 = fmaf(r35, r1, r53);
    r19 = r20 * r39;
    r53 = fmaf(r35, r19, r53);
    r16 = r22 * r67;
    r53 = fmaf(r35, r16, r53);
    r11 = r23 * r35;
    r53 = fmaf(r72, r11, r53);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          0 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r50,
                                          r53,
                                          r39,
                                          r71);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          4 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r38,
                                          r47,
                                          r8);
    WriteIdx2<1024, float, float, float2>(out_normalized,
                                          8 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r17,
                                          r72);
  };
}

void ThinPrismFisheyeFocalAndExtraNormalize(
    float* precond_diag,
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
  ThinPrismFisheyeFocalAndExtraNormalizeKernel<<<n_blocks, 1024>>>(
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