////////////////////////////////////////////////////////////////////////////
//  File:   SiftMatchCU.cpp
//  Author:   Changchang Wu
//  Description : implementation of the SiftMatchCU class.
//        CUDA-based implementation of SiftMatch
//
//  Copyright (c) 2007 University of North Carolina at Chapel Hill
//  All Rights Reserved
//
//  Permission to use, copy, modify and distribute this software and its
//  documentation for educational, research and non-profit purposes, without
//  fee, and without a written agreement is hereby granted, provided that
// the   above copyright notice and the following paragraph appear in all
// copies.
//
//  The University of North Carolina at Chapel Hill make no representations
//  about the suitability of this software for any purpose. It is provided
//  'as is' without express or implied warranty.
//
//  Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#if defined(SIFTGPU_CUDA_ENABLED)

#include "GL/glew.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
using namespace std;

#include <cuda_runtime.h>

#include "CuTexImage.h"
#include "GlobalUtil.h"
#include "ProgramCU.h"
#include "SiftGPU.h"
#include "SiftMatchCU.h"

#define MULT_TBLOCK_DIMX 128
#define MULT_TBLOCK_DIMY 1
#define MULT_BLOCK_DIMX (MULT_TBLOCK_DIMX)
#define MULT_BLOCK_DIMY (8 * MULT_TBLOCK_DIMY)

SiftMatchCU::SiftMatchCU(int max_sift) : SiftMatchGPU() {
  _num_sift[0] = _num_sift[1] = 0;
  _id_sift[0] = _id_sift[1] = 0;
  _have_loc[0] = _have_loc[1] = 0;
  __max_sift = max_sift <= 0 ? 4096 : ((max_sift + 31) / 32 * 32);
  _initialized = 0;
}

bool SiftMatchCU::Allocate(int max_sift, int mbm) {
  SetMaxSift(max_sift);

  for (int index = 0; index < 2; ++index) {
    if (!_texDes[index].InitTexture(8 * __max_sift, 1, 4) ||
        !_texLoc[index].InitTexture(__max_sift, 1, 2)) {
      return false;
    }
  }

  if (!_texDot.InitTexture(__max_sift, __max_sift) ||
      !_texMatch[0].InitTexture(__max_sift, 1)) {
    return false;
  }

  if (mbm) {
    const int cols = (__max_sift + MULT_BLOCK_DIMY - 1) / MULT_BLOCK_DIMY;
    if (!_texCRT.InitTexture(__max_sift, cols, 32) ||
        !_texMatch[1].InitTexture(__max_sift, 1)) {
      return false;
    }
  }

  _num_sift[0] = __max_sift;
  _num_sift[1] = __max_sift;

  return true;
}

void SiftMatchCU::SetMaxSift(int max_sift) {
  max_sift = ((max_sift + 31) / 32) * 32;
  __max_sift = max_sift;
}

int SiftMatchCU::CheckCudaDevice(int device) {
  return ProgramCU::CheckCudaDevice(device);
}

void SiftMatchCU::InitSiftMatch() {
  if (_initialized) return;
  GlobalUtil::_GoodOpenGL = max(GlobalUtil::_GoodOpenGL, 1);
  _initialized = 1;
}

void SiftMatchCU::SetDescriptors(int index, int num,
                                 const unsigned char* descriptors, int id) {
  if (_initialized == 0) return;
  if (index > 1) index = 1;
  if (index < 0) index = 0;
  _have_loc[index] = 0;
  // the same feature is already set
  if (id != -1 && id == _id_sift[index]) return;
  _id_sift[index] = id;
  if (num > __max_sift) num = __max_sift;
  _num_sift[index] = num;
  _texDes[index].InitTexture(8 * num, 1, 4);
  _texDes[index].CopyFromHost((void*)descriptors);
}

void SiftMatchCU::SetDescriptors(int index, int num, const float* descriptors,
                                 int id) {
  if (_initialized == 0) return;
  if (index > 1) index = 1;
  if (index < 0) index = 0;
  if (num > __max_sift) num = __max_sift;

  sift_buffer.resize(num * 128 / 4);
  unsigned char* pub = (unsigned char*)&sift_buffer[0];
  for (int i = 0; i < 128 * num; ++i) {
    pub[i] = int(512 * descriptors[i] + 0.5);
  }
  SetDescriptors(index, num, pub, id);
}

void SiftMatchCU::SetFeautreLocation(int index, const float* locations,
                                     int gap) {
  if (_num_sift[index] <= 0) return;
  _texLoc[index].InitTexture(_num_sift[index], 1, 2);
  if (gap == 0) {
    _texLoc[index].CopyFromHost(locations);
  } else {
    sift_buffer.resize(_num_sift[index] * 2);
    float* pbuf = (float*)(&sift_buffer[0]);
    for (int i = 0; i < _num_sift[index]; ++i) {
      pbuf[i * 2] = *locations++;
      pbuf[i * 2 + 1] = *locations++;
      locations += gap;
    }
    _texLoc[index].CopyFromHost(pbuf);
  }
  _have_loc[index] = 1;
}

int SiftMatchCU::GetGuidedSiftMatch(int max_match, uint32_t match_buffer[][2],
                                    float* H, float* F, float distmax,
                                    float ratiomax, float hdistmax,
                                    float fdistmax, int mbm) {
  if (_initialized == 0) return 0;
  if (_num_sift[0] <= 0 || _num_sift[1] <= 0) return 0;
  if (_have_loc[0] == 0 || _have_loc[1] == 0) return 0;
  ProgramCU::MultiplyDescriptorG(_texDes, _texDes + 1, _texLoc, _texLoc + 1,
                                 &_texDot, (mbm ? &_texCRT : NULL), H, hdistmax,
                                 F, fdistmax);
  return GetBestMatch(max_match, match_buffer, distmax, ratiomax, mbm);
}

int SiftMatchCU::GetSiftMatch(int max_match, uint32_t match_buffer[][2],
                              float distmax, float ratiomax, int mbm) {
  if (_initialized == 0) return 0;
  if (_num_sift[0] <= 0 || _num_sift[1] <= 0) return 0;
  ProgramCU::MultiplyDescriptor(_texDes, _texDes + 1, &_texDot,
                                (mbm ? &_texCRT : NULL));
  return GetBestMatch(max_match, match_buffer, distmax, ratiomax, mbm);
}

int SiftMatchCU::GetBestMatch(int max_match, uint32_t match_buffer[][2],
                              float distmax, float ratiomax, int mbm) {
  sift_buffer.resize(_num_sift[0] + _num_sift[1]);
  int *buffer1 = (int*)&sift_buffer[0],
      *buffer2 = (int*)&sift_buffer[_num_sift[0]];
  _texMatch[0].InitTexture(_num_sift[0], 1);
  ProgramCU::GetRowMatch(&_texDot, _texMatch, distmax, ratiomax);
  _texMatch[0].CopyToHost(buffer1);
  if (mbm) {
    _texMatch[1].InitTexture(_num_sift[1], 1);
    ProgramCU::GetColMatch(&_texCRT, _texMatch + 1, distmax, ratiomax);
    _texMatch[1].CopyToHost(buffer2);
  }
  int nmatch = 0, j;
  for (int i = 0; i < _num_sift[0] && nmatch < max_match; ++i) {
    j = int(buffer1[i]);
    if (j >= 0 && (!mbm || int(buffer2[j]) == i)) {
      match_buffer[nmatch][0] = i;
      match_buffer[nmatch][1] = j;
      nmatch++;
    }
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return -1;
  }

  return nmatch;
}

#endif
