////////////////////////////////////////////////////////////////////////////
//  File:           CuTexImage.cpp
//  Author:         Changchang Wu
//  Description :   implementation of the CuTexImage class.
//
//  Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)
//    and the University of Washington at Seattle
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation; either
//  Version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <fstream>
using namespace std;

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "CuTexImage.h"

#if CUDA_VERSION <= 2010
#error "Require CUDA 2.2 or higher"
#endif

namespace pba {

CuTexImage::CuTexImage() {
  _owner = true;
  _cuData = NULL;
  _numBytes = _numChannel = 0;
  _imgWidth = _imgHeight = 0;
}

CuTexImage::~CuTexImage() {
  if (_cuData && _owner) cudaFree(_cuData);
}

void CuTexImage::ReleaseData() {
  if (_cuData && _owner) cudaFree(_cuData);
  _cuData = NULL;
  _numBytes = 0;
}

void CuTexImage::SwapData(CuTexImage& src) {
  if (_cuData == src._cuData) return;

  void* cuData = _cuData;
  unsigned int numChannel = _numChannel;
  unsigned int imgWidth = _imgWidth;
  unsigned int imgHeight = _imgHeight;
  bool owner = _owner;
  size_t numBytes = _numBytes;

  _cuData = src._cuData;
  _numChannel = src._numChannel;
  _numBytes = src._numBytes;
  _imgWidth = src._imgWidth;
  _imgHeight = src._imgHeight;
  _owner = src._owner;

  src._cuData = cuData;
  src._numChannel = numChannel;
  src._numBytes = numBytes;
  src._imgWidth = imgWidth;
  src._imgHeight = imgHeight;
  src._owner = owner;
}

bool CuTexImage::InitTexture(unsigned int width, unsigned int height,
                             unsigned int nchannel) {
  size_t size = sizeof(float) * width * height * nchannel;
  _imgWidth = width;
  _imgHeight = height;
  _numChannel = nchannel;

  if (size <= _numBytes) return true;

  if (_cuData && _owner) cudaFree(_cuData);

  // allocate the array data
  cudaError_t e = cudaMalloc(&_cuData, size);
  _numBytes = e == cudaSuccess ? size : 0;
  _owner = true;
  return e == cudaSuccess;
}

void CuTexImage::SetTexture(void* data, unsigned int width,
                            unsigned int nchannel) {
  if (_cuData && _owner) cudaFree(_cuData);
  _imgWidth = width;
  _imgHeight = 1;
  _numChannel = nchannel;
  _numBytes = sizeof(float) * width * _imgHeight * _numChannel;
  _cuData = data;
  _owner = false;
}

void CuTexImage::CopyFromHost(const void* buf) {
  if (_cuData == NULL || buf == NULL || GetDataSize() == 0) return;
  cudaMemcpy(_cuData, buf, _imgWidth * _imgHeight * _numChannel * sizeof(float),
             cudaMemcpyHostToDevice);
}

void CuTexImage::CopyFromDevice(const void* buf) {
  if (_cuData == NULL) return;
  cudaMemcpy((char*)_cuData, buf,
             _imgWidth * _imgHeight * _numChannel * sizeof(float),
             cudaMemcpyDeviceToDevice);
}

void CuTexImage::CopyToHost(void* buf) {
  if (_cuData == NULL) return;
  size_t sz = _imgWidth * _imgHeight * _numChannel * sizeof(float);
  cudaMemcpy(buf, _cuData, sz, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

void CuTexImage::SaveToFile(const char* name) {
  ofstream out(name);
  vector<float> value(GetLength());
  CopyToHost(&value[0]);
  for (size_t i = 0; i < value.size(); ++i) out << value[i] << '\n';
}

}  // namespace pba
