////////////////////////////////////////////////////////////////////////////
//  File:           pba.cpp
//  Author:         Changchang Wu
//  Description :   implementation of ParallelBA, which is a wrapper around
//                  the GPU-based and CPU-based implementations
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
#include <stdlib.h>
#include <new>
#include "pba.h"
#include "SparseBundleCU.h"
#include "SparseBundleCPU.h"

namespace pba {

ParallelBA::ParallelBA(DeviceT device, const int num_threads) {
  // The wrapper intends to provide different implementations.

  if (device >= PBA_CUDA_DEVICE_DEFAULT)
#ifndef PBA_NO_GPU
  {
    SparseBundleCU* cuba = new SparseBundleCU(device - PBA_CUDA_DEVICE0);
    if (cuba->GetMemCapacity() > 0) {
      _optimizer = cuba;
    } else {
      device = PBA_CPU_FLOAT;
      _optimizer = NewSparseBundleCPU(false, num_threads);
      delete cuba;
    }
  } else
#else
    device = PBA_CPU_FLOAT;
#endif
      if (device == PBA_CPU_FLOAT)
    _optimizer = NewSparseBundleCPU(false, num_threads);
  else if (device == PBA_CPU_DOUBLE)
    _optimizer = NewSparseBundleCPU(true, num_threads);
  else
    _optimizer = NULL;
}

ParallelBA::~ParallelBA() {
  if (_optimizer) delete _optimizer;
}

void ParallelBA::ParseParam(int narg, char** argv) {
  _optimizer->ParseParam(narg, argv);
}

ConfigBA* ParallelBA::GetInternalConfig() {
  if (_optimizer)
    return _optimizer->GetInternalConfig();
  else
    return NULL;
}

void ParallelBA::SetFixedIntrinsics(bool fixed) {
  _optimizer->SetFixedIntrinsics(fixed);
}
void ParallelBA::EnableRadialDistortion(DistortionT enabled) {
  _optimizer->EnableRadialDistortion(enabled);
}
void ParallelBA::SetNextTimeBudget(int seconds) {
  _optimizer->SetNextTimeBudget(seconds);
}

void ParallelBA::SetNextBundleMode(BundleModeT mode) {
  _optimizer->SetNextBundleMode(mode);
}

void ParallelBA::SetCameraData(size_t ncam, CameraT* cams) {
  _optimizer->SetCameraData(ncam, cams);
}

void ParallelBA::SetPointData(size_t npoint, Point3D* pts) {
  _optimizer->SetPointData(npoint, pts);
}

void ParallelBA::SetProjection(size_t nproj, const Point2D* imgpts,
                               const int* point_idx, const int* cam_idx) {
  _optimizer->SetProjection(nproj, imgpts, point_idx, cam_idx);
}
int ParallelBA::RunBundleAdjustment() {
  return _optimizer->RunBundleAdjustment();
}

float ParallelBA::GetMeanSquaredError() {
  return _optimizer->GetMeanSquaredError();
}

int ParallelBA::GetCurrentIteration() {
  return _optimizer->GetCurrentIteration();
}
void ParallelBA::AbortBundleAdjustment() {
  return _optimizer->AbortBundleAdjustment();
}

void ParallelBA::ReserveStorage(size_t ncam, size_t npt, size_t nproj) {
  if (_optimizer) _optimizer->ReserveStorage(ncam, npt, nproj);
}

void ParallelBA::SetFocalMask(const int* fmask, float weight) {
  if (_optimizer && weight > 0) _optimizer->SetFocalMask(fmask, weight);
}

void* ParallelBA::operator new(size_t size) {
  void* p = malloc(size);
  if (p == 0) {
    const std::bad_alloc ba;
    throw ba;
  }
  return p;
}

ParallelBA* NewParallelBA(ParallelBA::DeviceT device) {
  return new ParallelBA(device);
}

int ParallelBA_GetVersion() { return 105; }

}  // namespace pba
