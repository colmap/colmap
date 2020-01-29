////////////////////////////////////////////////////////////////////////////
//  File:       pba.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   interface of class ParallelBA, which has two
//implementations
//                  SparseBundleCU for CUDA-based version, and
//                  SparseBundleCPU<Float> for CPU multi-threading version
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

#ifndef PARALLEL_BA_H
#define PARALLEL_BA_H

#if defined(_WIN32)
#ifdef PBA_DLL
#ifdef DLL_EXPORT
#define PBA_EXPORT __declspec(dllexport)
#else
#define PBA_EXPORT __declspec(dllimport)
#endif
#else
#define PBA_EXPORT
#endif

#define PBA_EXPORT_EXTERN PBA_EXPORT

#if _MSC_VER > 1000
#pragma once
#endif
#else
#define PBA_EXPORT
#define PBA_EXPORT_EXTERN extern "C"
#endif

// filetype definitions for points and camera
#include "DataInterface.h"
#include "ConfigBA.h"

namespace pba {

class ParallelBA {
 public:
  enum StatusT {
    STATUS_SUCCESS = 0,
    STATUS_CAMERA_MISSING = 1,
    STATUS_POINT_MISSING,
    STATUS_PROJECTION_MISSING,
    STATUS_MEASURMENT_MISSING,
    STATUS_ALLOCATION_FAIL
  };
  enum DeviceT {
    PBA_INVALID_DEVICE = -4,
    PBA_CPU_DOUBLE = -3,
    PBA_CPU_FLOAT = -2,
    PBA_CUDA_DEVICE_DEFAULT = -1,
    PBA_CUDA_DEVICE0 = 0
  };
  enum DistortionT {
    PBA_MEASUREMENT_DISTORTION = -1,  // single parameter, apply to measurements
    PBA_NO_DISTORTION = 0,  // no radial distortion
    PBA_PROJECTION_DISTORTION = 1  // single parameter, apply to projectino
  };
  enum BundleModeT {
    BUNDLE_FULL = 0,
    BUNDLE_ONLY_MOTION = 1,
    BUNDLE_ONLY_STRUCTURE = 2,
  };

 private:
  ParallelBA* _optimizer;

 public:
  ////////////////////////////////////////////////////
  // methods for changing bundle adjustment settings
  PBA_EXPORT virtual void ParseParam(int narg, char** argv);  // indirect method
  PBA_EXPORT virtual ConfigBA* GetInternalConfig();  // direct method
  PBA_EXPORT virtual void SetFixedIntrinsics(
      bool fixed);  // call this for calibrated system
  PBA_EXPORT virtual void EnableRadialDistortion(
      DistortionT type);  // call this to enable radial distortion
  PBA_EXPORT virtual void SetNextTimeBudget(
      int seconds);  //# of seconds for next run (0 = no limit)
  PBA_EXPORT virtual void ReserveStorage(size_t ncam, size_t npt, size_t nproj);

 public:
  // function name change; the old one is mapped as inline function
  inline void SetFocalLengthFixed(bool fixed) { SetFixedIntrinsics(fixed); }
  inline void ResetBundleStorage() {
    ReserveStorage(0, 0, 0); /*Reset devide for CUDA*/
  }

 public:
  /////////////////////////////////////////////////////
  // optimizer interface, input and run
  PBA_EXPORT virtual void SetCameraData(size_t ncam,
                                        CameraT* cams);  // set camera data
  PBA_EXPORT virtual void SetPointData(size_t npoint,
                                       Point3D* pts);  // set 3D point data
  PBA_EXPORT virtual void SetProjection(size_t nproj, const Point2D* imgpts,
                                        const int* point_idx,
                                        const int* cam_idx);  // set projections
  PBA_EXPORT virtual void SetNextBundleMode(
      BundleModeT
          mode = BUNDLE_FULL);  // mode of the next bundle adjustment call
  PBA_EXPORT virtual int RunBundleAdjustment();  // start bundle adjustment,
                                                 // return number of successful
                                                 // LM iterations
 public:
  //////////////////////////////////////////////////
  // Query optimzer runing status for Multi-threading
  // Three functions below can be called from a differnt thread while bundle is
  // running
  PBA_EXPORT virtual float
  GetMeanSquaredError();  // read back results during/after BA
  PBA_EXPORT virtual void
  AbortBundleAdjustment();  // tell bundle adjustment to abort ASAP
  PBA_EXPORT virtual int
  GetCurrentIteration();  // which iteration is it working on?
 public:
  PBA_EXPORT ParallelBA(DeviceT device = PBA_CUDA_DEVICE_DEFAULT,
                        const int num_threads = -1);
  PBA_EXPORT void* operator new(size_t size);
  PBA_EXPORT virtual ~ParallelBA();

 public:
  //////////////////////////////////////////////
  // Future functions will be added to the end for compatiability with old
  // version.
  PBA_EXPORT virtual void SetFocalMask(const int* fmask, float weight = 1.0f);
};

// function for dynamic loading of library
PBA_EXPORT_EXTERN ParallelBA* NewParallelBA(
    ParallelBA::DeviceT device = ParallelBA::PBA_CUDA_DEVICE_DEFAULT);
typedef ParallelBA* (*NEWPARALLELBAPROC)(ParallelBA::DeviceT);

///////////////////////////////////////////////
// older versions do not have this function.
PBA_EXPORT_EXTERN int ParallelBA_GetVersion();

}  // namespace pba

#endif
