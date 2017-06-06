////////////////////////////////////////////////////////////////////////////
//  File:           SparseBundleCU.cpp
//  Author:         Changchang Wu
//  Description :   implementation of the CUDA-based multicore bundle adjustment
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

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>
#include <iomanip>
using std::vector;
using std::cout;
using std::pair;
using std::ofstream;

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "pba.h"
#include "SparseBundleCU.h"

#include "ProgramCU.h"

using namespace pba::ProgramCU;

#ifdef _WIN32
#define finite _finite
#endif

namespace pba {

typedef float float_t;  // data type for host computation; double doesn't make
                        // much difference

#define CHECK_VEC(v1, v2)                                                 \
  for (size_t j = 0; j < v1.size(); ++j) {                                \
    if (v1[j] != v2[j]) {                                                 \
      different++;                                                        \
      std::cout << i << ' ' << j << ' ' << v1[j] << ' ' << v2[j] << '\n'; \
    }                                                                     \
  }
#define DEBUG_FUNCN(v, func, input, N)                                  \
  if (__debug_pba && v.IsValid()) {                                     \
    vector<float> buf(v.GetLength()), buf_(v.GetLength());              \
    for (int i = 0; i < N; ++i) {                                       \
      int different = 0;                                                \
      func input;                                                       \
      ProgramCU::FinishWorkCUDA();                                      \
      if (i > 0) {                                                      \
        v.CopyToHost(&buf_[0]);                                         \
        CHECK_VEC(buf, buf_);                                           \
      } else {                                                          \
        v.CopyToHost(&buf[0]);                                          \
      }                                                                 \
      if (different != 0)                                               \
        std::cout << #func << " : " << i << " : " << different << '\n'; \
    }                                                                   \
  }
#define DEBUG_FUNC(v, func, input) DEBUG_FUNCN(v, func, input, 2)

SparseBundleCU::SparseBundleCU(int device)
    : ParallelBA(PBA_INVALID_DEVICE),
      _num_camera(0),
      _num_point(0),
      _num_imgpt(0),
      _num_imgpt_q(0),
      _camera_data(NULL),
      _point_data(NULL),
      _imgpt_data(NULL),
      _camera_idx(NULL),
      _point_idx(NULL),
      _projection_sse(0) {
  __selected_device = device;
}

size_t SparseBundleCU::GetMemCapacity() {
  if (__selected_device != __current_device) SetCudaDevice(__selected_device);
  size_t sz = ProgramCU::GetCudaMemoryCap();
  if (sz < 1024) std::cout << "ERROR: CUDA is unlikely to be supported!\n";
  return sz < 1024 ? 0 : sz;
}

void SparseBundleCU::SetCameraData(size_t ncam, CameraT* cams) {
  if (sizeof(CameraT) != 16 * sizeof(float)) exit(0);  // never gonna happen...?
  _num_camera = (int)ncam;
  _camera_data = cams;
  _focal_mask = NULL;
}

void SparseBundleCU::SetFocalMask(const int* fmask, float weight) {
  _focal_mask = fmask;
  _weight_q = weight;
}

void SparseBundleCU::SetPointData(size_t npoint, Point3D* pts) {
  _num_point = (int)npoint;
  _point_data = (float*)pts;
}

void SparseBundleCU::SetProjection(size_t nproj, const Point2D* imgpts,
                                   const int* point_idx, const int* cam_idx) {
  _num_imgpt = (int)nproj;
  _imgpt_data = (float*)imgpts;
  _camera_idx = cam_idx;
  _point_idx = point_idx;
  _imgpt_datax.resize(0);
}

float SparseBundleCU::GetMeanSquaredError() {
  return float(_projection_sse /
               (_num_imgpt * __focal_scaling * __focal_scaling));
}

void SparseBundleCU::BundleAdjustment() {
  if (ValidateInputData() != STATUS_SUCCESS) return;

  //

  ////////////////////////
  TimerBA timer(this, TIMER_OVERALL);

  NormalizeData();
  if (InitializeBundle() != STATUS_SUCCESS) {
    // failed to allocate gpu storage
  } else if (__profile_pba) {
    // profiling some stuff
    RunProfileSteps();
  } else {
    // real optimization
    AdjustBundleAdjsutmentMode();
    NonlinearOptimizeLM();
    TransferDataToHost();
  }
  DenormalizeData();
}

int SparseBundleCU::RunBundleAdjustment() {
  if (__warmup_device) WarmupDevice();
  ResetBundleStatistics();
  BundleAdjustment();
  if (__num_lm_success > 0)
    SaveBundleStatistics(_num_camera, _num_point, _num_imgpt);
  if (__num_lm_success > 0) PrintBundleStatistics();
  ResetTemporarySetting();
  return __num_lm_success;
}

bool SparseBundleCU::InitializeBundleGPU() {
  bool previous_allocated = __memory_usage > 0;

  bool success = TransferDataToGPU() && InitializeStorageForCG();
  if (!success && previous_allocated) {
    if (__verbose_level) std::cout << "WARNING: try clean allocation\n";
    ClearPreviousError();
    ReleaseAllocatedData();
    success = TransferDataToGPU() && InitializeStorageForCG();
  }

  if (!success && __jc_store_original) {
    if (__verbose_level) std::cout << "WARNING: try not storing original JC\n";
    __jc_store_original = false;
    ClearPreviousError();
    ReleaseAllocatedData();
    success = TransferDataToGPU() && InitializeStorageForCG();
  }
  if (!success && __jc_store_transpose) {
    if (__verbose_level) std::cout << "WARNING: try not storing transpose JC\n";
    __jc_store_transpose = false;
    ClearPreviousError();
    ReleaseAllocatedData();
    success = TransferDataToGPU() && InitializeStorageForCG();
  }
  if (!success && !__no_jacobian_store) {
    if (__verbose_level) std::cout << "WARNING: switch to memory saving mode\n";
    __no_jacobian_store = true;
    ClearPreviousError();
    ReleaseAllocatedData();
    success = TransferDataToGPU() && InitializeStorageForCG();
  }
  return success;
}

int SparseBundleCU::ValidateInputData() {
  if (_camera_data == NULL) return STATUS_CAMERA_MISSING;
  if (_point_data == NULL) return STATUS_POINT_MISSING;
  if (_imgpt_data == NULL) return STATUS_MEASURMENT_MISSING;
  if (_camera_idx == NULL || _point_idx == NULL)
    return STATUS_PROJECTION_MISSING;
  return STATUS_SUCCESS;
}

void SparseBundleCU::WarmupDevice() {
  std::cout << "Warm up device with storage allocation...\n";
  if (__selected_device != __current_device) SetCudaDevice(__selected_device);
  CheckRequiredMemX();
  InitializeBundleGPU();
}

int SparseBundleCU::InitializeBundle() {
  /////////////////////////////////////////////////////
  TimerBA timer(this, TIMER_GPU_ALLOCATION);
  if (__selected_device != __current_device) SetCudaDevice(__selected_device);
  CheckRequiredMemX();
  ReserveStorageAuto();
  if (!InitializeBundleGPU()) return STATUS_ALLOCATION_FAIL;
  return STATUS_SUCCESS;
}

int SparseBundleCU::GetParameterLength() {
  return _num_camera * 8 + 4 * _num_point;
}

bool SparseBundleCU::CheckRequiredMemX() {
  if (CheckRequiredMem(0)) return true;
  if (__jc_store_original) {
    if (__verbose_level) std::cout << "NOTE: not storing original JC\n";
    __jc_store_original = false;
    if (CheckRequiredMem(1)) return true;
  }
  if (__jc_store_transpose) {
    if (__verbose_level) std::cout << "NOTE:  not storing camera Jacobian\n";
    __jc_store_transpose = false;
    if (CheckRequiredMem(1)) return true;
  }
  if (!__no_jacobian_store) {
    if (__verbose_level) std::cout << "NOTE: not storing any Jacobian\n";
    __no_jacobian_store = true;
    if (CheckRequiredMem(1)) return true;
  }
  return false;
}

bool SparseBundleCU::CheckRequiredMem(int fresh) {
  int m = _num_camera, n = _num_point, k = _num_imgpt;
#ifdef PBA_CUDA_ALLOCATE_MORE
  if (!fresh) {
    int m0 = _cuCameraData.GetReservedWidth();
    m = std::max(m, m0);
    int n0 = _cuPointData.GetReservedWidth();
    n = std::max(n, n0);
    int k0 = _cuMeasurements.GetReservedWidth();
    k = std::max(k, k0);
  }
#endif

  int p = 8 * m + 4 * n, q = _num_imgpt_q;
  size_t szn, total = GetCudaMemoryCap();
  size_t sz0 = 800 * 600 * 2 * 4 * sizeof(float);  //
  size_t szq = q > 0 ? (sizeof(float) * (q + m) * 4) : 0;
  size_t sz = sizeof(float) * (258 + 9 * n + 33 * m + 7 * k) + sz0;

  /////////////////////////////////// CG
  sz += p * 6 * sizeof(float);
  sz += ((__use_radial_distortion ? 64 : 56) * m + 12 * n) * sizeof(float);
  sz += (2 * (k + q) * sizeof(float));
  if (sz > total) return false;

  /////////////////////////////////////
  szn = (__no_jacobian_store ? 0 : (sizeof(float) * 8 * k));
  if (sz + szn > total)
    __no_jacobian_store = false;
  else
    sz += szn;
  /////////////////////////////
  szn = ((!__no_jacobian_store && __jc_store_transpose) ? 16 * k * sizeof(float)
                                                        : 0);
  if (sz + szn > total)
    __jc_store_transpose = false;
  else
    sz += szn;
  ///////////////////////////
  szn = ((!__no_jacobian_store && __jc_store_original) ? 16 * k * sizeof(float)
                                                       : 0);
  if (sz + szn > total)
    __jc_store_original = false;
  else
    sz += szn;
  ///////////////////////////////
  szn = ((!__no_jacobian_store && __jc_store_transpose && !__jc_store_original)
             ? k * sizeof(int)
             : 0);
  if (sz + szn > total) {
    __jc_store_transpose = false;
    sz -= (16 * k * sizeof(float));
  } else
    sz += szn;

  return sz <= total;
}

void SparseBundleCU::ReserveStorage(size_t ncam, size_t npt, size_t nproj) {
  if (ncam <= 1 || npt <= 1 || nproj <= 1) {
    ReleaseAllocatedData();
    // Reset the memory strategy to the default.
    __jc_store_transpose = true;
    __jc_store_original = true;
    __no_jacobian_store = false;
  } else {
    const int* camidx = _camera_idx;
    const int* ptidx = _point_idx;
    int ncam_ = _num_camera;
    int npt_ = _num_point;
    int nproj_ = _num_imgpt;

#ifdef PBA_CUDA_ALLOCATE_MORE
    size_t ncam_reserved = _cuCameraData.GetReservedWidth();
    size_t npt_reserved = _cuPointData.GetReservedWidth();
    size_t nproj_reserved = _cuMeasurements.GetReservedWidth();
    ncam = std::max(ncam, ncam_reserved);
    npt = std::max(npt, npt_reserved);
    nproj = std::max(nproj, nproj_reserved);
#endif

    _camera_idx = NULL;
    _point_idx = NULL;
    _num_camera = (int)ncam;
    _num_point = (int)npt;
    _num_imgpt = (int)nproj;

    if (__verbose_level)
      std::cout << "Reserving storage for ncam = " << ncam << "; npt = " << npt
                << "; nproj = " << nproj << '\n';
    InitializeBundleGPU();

    _num_camera = ncam_;
    _num_point = npt_;
    _num_imgpt = nproj_;
    _camera_idx = camidx;
    _point_idx = ptidx;
  }
}

static size_t upgrade_dimension(size_t sz) {
  size_t x = 1;
  while (x < sz) x <<= 1;
  return x;
}

void SparseBundleCU::ReserveStorageAuto() {
  if (_cuCameraData.data() == NULL || _cuPointData.data() == NULL ||
      _cuMeasurements.data() == NULL)
    return;
  ReserveStorage(upgrade_dimension(_num_camera), upgrade_dimension(_num_point),
                 upgrade_dimension(_num_imgpt));
}

#define REPORT_ALLOCATION(NAME)                                   \
  if (__verbose_allocation && NAME.GetDataSize() > 1024)          \
    std::cout << (NAME.GetDataSize() > 1024 * 1024                \
                      ? NAME.GetDataSize() / 1024 / 1024          \
                      : NAME.GetDataSize() / 1024)                \
              << (NAME.GetDataSize() > 1024 * 1024 ? "MB" : "KB") \
              << "\t allocated for " #NAME "\n";

#define ASSERT_ALLOCATION(NAME)                                    \
  if (!success) {                                                  \
    std::cerr << "WARNING: failed to allocate "                    \
              << (__verbose_allocation ? #NAME "; size = " : "")   \
              << (total_sz / 1024 / 1024) << "MB + "               \
              << (NAME.GetRequiredSize() / 1024 / 1024) << "MB\n"; \
    return false;                                                  \
  } else {                                                         \
    total_sz += NAME.GetDataSize();                                \
    REPORT_ALLOCATION(NAME);                                       \
  }

#define CHECK_ALLOCATION(NAME)                                     \
  if (NAME.GetDataSize() == 0 && NAME.GetRequiredSize() > 0) {     \
    ClearPreviousError();                                          \
    std::cerr << "WARNING: unable to allocate " #NAME ": "         \
              << (NAME.GetRequiredSize() / 1024 / 1024) << "MB\n"; \
  } else {                                                         \
    total_sz += NAME.GetDataSize();                                \
    REPORT_ALLOCATION(NAME);                                       \
  }

#define ALLOCATE_REQUIRED_DATA(NAME, num, channels) \
  {                                                 \
    success &= NAME.InitTexture(num, 1, channels);  \
    ASSERT_ALLOCATION(NAME);                        \
  }

#define ALLOCATE_OPTIONAL_DATA(NAME, num, channels, option) \
  if (option) {                                             \
    option = NAME.InitTexture(num, 1, channels);            \
    CHECK_ALLOCATION(NAME);                                 \
  } else {                                                  \
    NAME.InitTexture(0, 0, 0);                              \
  }

bool SparseBundleCU::TransferDataToGPU() {
  // given m camera, npoint, k measurements.. the number of float is
  bool success = true;
  size_t total_sz = 0;

  /////////////////////////////////////////////////////////////////////////////
  vector<int> qmap, qlist;
  vector<float> qmapw, qlistw;
  ProcessIndexCameraQ(qmap, qlist);

  //////////////////////////////////////////////////////////////////////////////
  ALLOCATE_REQUIRED_DATA(_cuBufferData, 256, 1);  // 256
  ALLOCATE_REQUIRED_DATA(_cuPointData, _num_point, 4);  // 4n
  ALLOCATE_REQUIRED_DATA(_cuCameraData, _num_camera, 16);  // 16m
  ALLOCATE_REQUIRED_DATA(_cuCameraDataEX, _num_camera, 16);  // 16m

  ////////////////////////////////////////////////////////////////
  ALLOCATE_REQUIRED_DATA(_cuCameraMeasurementMap, _num_camera + 1, 1);  // m
  ALLOCATE_REQUIRED_DATA(_cuCameraMeasurementList, _num_imgpt, 1);  // k
  ALLOCATE_REQUIRED_DATA(_cuPointMeasurementMap, _num_point + 1, 1);  // n
  ALLOCATE_REQUIRED_DATA(_cuProjectionMap, _num_imgpt, 2);  // 2k
  ALLOCATE_REQUIRED_DATA(_cuImageProj, _num_imgpt + _num_imgpt_q, 2);  // 2k
  ALLOCATE_REQUIRED_DATA(_cuPointDataEX, _num_point, 4);  // 4n
  ALLOCATE_REQUIRED_DATA(_cuMeasurements, _num_imgpt, 2);  // 2k

  //
  ALLOCATE_REQUIRED_DATA(_cuCameraQMap, _num_imgpt_q, 2);
  ALLOCATE_REQUIRED_DATA(_cuCameraQMapW, _num_imgpt_q, 2);
  ALLOCATE_REQUIRED_DATA(_cuCameraQList, (_num_imgpt_q > 0 ? _num_camera : 0),
                         2);
  ALLOCATE_REQUIRED_DATA(_cuCameraQListW, (_num_imgpt_q > 0 ? _num_camera : 0),
                         2);

  if (__no_jacobian_store) {
    _cuJacobianCamera.ReleaseData();
    _cuJacobianCameraT.ReleaseData();
    _cuJacobianPoint.ReleaseData();
    _cuCameraMeasurementListT.ReleaseData();
  } else {
    ALLOCATE_REQUIRED_DATA(_cuJacobianPoint, _num_imgpt * 2, 4);  // 8k
    ALLOCATE_OPTIONAL_DATA(_cuJacobianCameraT, _num_imgpt * 2, 8,
                           __jc_store_transpose);  // 16k
    ALLOCATE_OPTIONAL_DATA(_cuJacobianCamera, _num_imgpt * 2, 8,
                           __jc_store_original);  // 16k

    if ((!__jc_store_original || __profile_pba) && __jc_store_transpose) {
      ALLOCATE_OPTIONAL_DATA(_cuCameraMeasurementListT, _num_imgpt, 1,
                             __jc_store_transpose);  // k
      if (!__jc_store_transpose) _cuJacobianCameraT.ReleaseData();
    } else {
      _cuCameraMeasurementListT.ReleaseData();
    }
  }

  /////////////////////////////////////////////////
  if (_camera_idx && _point_idx) {
    //////////////////////////////////////////
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);
    ////mapping from camera to measuremnts
    vector<int> cpi(_num_camera + 1), cpidx(_num_imgpt);
    vector<int> cpnum(_num_camera, 0);
    cpi[0] = 0;
    for (int i = 0; i < _num_imgpt; ++i) cpnum[_camera_idx[i]]++;
    for (int i = 1; i <= _num_camera; ++i) cpi[i] = cpi[i - 1] + cpnum[i - 1];
    vector<int> cptidx = cpi;
    for (int i = 0; i < _num_imgpt; ++i) cpidx[cptidx[_camera_idx[i]]++] = i;
    if (_num_imgpt_q > 0) ProcessWeightCameraQ(cpnum, qmap, qmapw, qlistw);
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);

    ///////////////////////////////////////////////////////////////////////////////
    BundleTimerSwap(TIMER_GPU_UPLOAD, TIMER_GPU_ALLOCATION);
    _cuMeasurements.CopyFromHost(_imgpt_datax.size() > 0 ? &_imgpt_datax[0]
                                                         : _imgpt_data);
    _cuCameraData.CopyFromHost(_camera_data);
    _cuPointData.CopyFromHost(_point_data);
    _cuCameraMeasurementMap.CopyFromHost(&cpi[0]);
    _cuCameraMeasurementList.CopyFromHost(&cpidx[0]);
    if (_cuCameraMeasurementListT.IsValid()) {
      vector<int> ridx(_num_imgpt);
      for (int i = 0; i < _num_imgpt; ++i) ridx[cpidx[i]] = i;
      _cuCameraMeasurementListT.CopyFromHost(&ridx[0]);
    }
    if (_num_imgpt_q > 0) {
      _cuCameraQMap.CopyFromHost(&qmap[0]);
      _cuCameraQMapW.CopyFromHost(&qmapw[0]);
      _cuCameraQList.CopyFromHost(&qlist[0]);
      _cuCameraQListW.CopyFromHost(&qlistw[0]);
    }
    BundleTimerSwap(TIMER_GPU_UPLOAD, TIMER_GPU_ALLOCATION);

    ////////////////////////////////////////////
    ///////mapping from point to measurment
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);
    vector<int> ppi(_num_point + 1);
    for (int i = 0, last_point = -1; i < _num_imgpt; ++i) {
      int pt = _point_idx[i];
      while (last_point < pt) ppi[++last_point] = i;
    }
    ppi[_num_point] = _num_imgpt;

    //////////projection map
    vector<int> projection_map(_num_imgpt * 2);
    for (int i = 0; i < _num_imgpt; ++i) {
      int* imp = &projection_map[i * 2];
      imp[0] = _camera_idx[i] * 2;
      imp[1] = _point_idx[i];
    }
    BundleTimerSwap(TIMER_PREPROCESSING, TIMER_GPU_ALLOCATION);

    //////////////////////////////////////////////////////////////
    BundleTimerSwap(TIMER_GPU_UPLOAD, TIMER_GPU_ALLOCATION);
    _cuPointMeasurementMap.CopyFromHost(&ppi[0]);
    _cuProjectionMap.CopyFromHost(&projection_map[0]);
    BundleTimerSwap(TIMER_GPU_UPLOAD, TIMER_GPU_ALLOCATION);
  }

  __memory_usage = total_sz;
  if (__verbose_level > 1)
    std::cout << "Memory for Motion/Structure/Jacobian:\t"
              << (total_sz / 1024 / 1024) << "MB\n";
  return success;
}

bool SparseBundleCU::ProcessIndexCameraQ(vector<int>& qmap,
                                         vector<int>& qlist) {
  // reset q-data
  qmap.resize(0);
  qlist.resize(0);
  _num_imgpt_q = 0;

  // verify input
  if (_camera_idx == NULL) return true;
  if (_point_idx == NULL) return true;
  if (_focal_mask == NULL) return true;
  if (_num_camera == 0) return true;
  if (_weight_q <= 0) return true;

  ///////////////////////////////////////

  int error = 0;
  vector<int> temp(_num_camera * 2, -1);

  for (int i = 0; i < _num_camera; ++i) {
    int iq = _focal_mask[i];
    if (iq > i) {
      error = 1;
      break;
    }
    if (iq < 0) continue;
    if (iq == i) continue;
    int ip = temp[2 * iq];
    // float ratio = _camera_data[i].f / _camera_data[iq].f;
    // if(ratio < 0.01 || ratio > 100)
    //{
    //  std::cout << "Warning: constaraints on largely different camreas\n";
    //  continue;
    //}else
    if (_focal_mask[iq] != iq) {
      error = 1;
      break;
    } else if (ip == -1) {
      temp[2 * iq] = i;
      temp[2 * iq + 1] = i;
      temp[2 * i] = iq;
      temp[2 * i + 1] = iq;
    } else {
      // maintain double-linked list
      temp[2 * i] = ip;
      temp[2 * i + 1] = iq;
      temp[2 * ip + 1] = i;
      temp[2 * iq] = i;
    }
  }

  if (error) {
    std::cout << "Error: incorrect constraints\n";
    _focal_mask = NULL;
    return false;
  }

  qlist.resize(_num_camera * 2, -1);
  for (int i = 0; i < _num_camera; ++i) {
    int inext = temp[2 * i + 1];
    if (inext == -1) continue;
    qlist[2 * i] = _num_imgpt + _num_imgpt_q;
    qlist[2 * inext + 1] = _num_imgpt + _num_imgpt_q;
    qmap.push_back(i);
    qmap.push_back(inext);
    _num_imgpt_q++;
  }
  return true;
}

void SparseBundleCU::ProcessWeightCameraQ(vector<int>& cpnum, vector<int>& qmap,
                                          vector<float>& qmapw,
                                          vector<float>& qlistw) {
  // set average focal length and average radial distortion
  vector<float> qpnum(_num_camera, 0), qcnum(_num_camera, 0);
  vector<float> fs(_num_camera, 0), rs(_num_camera, 0);

  for (int i = 0; i < _num_camera; ++i) {
    int qi = _focal_mask[i];
    if (qi == -1) continue;
    // float ratio = _camera_data[i].f / _camera_data[qi].f;
    // if(ratio < 0.01 || ratio > 100)      continue;
    fs[qi] += _camera_data[i].f;
    rs[qi] += _camera_data[i].radial;
    qpnum[qi] += cpnum[i];
    qcnum[qi] += 1.0f;
  }

  // this seems not really matter..they will converge anyway
  for (int i = 0; i < _num_camera; ++i) {
    int qi = _focal_mask[i];
    if (qi == -1) continue;
    // float ratio = _camera_data[i].f / _camera_data[qi].f;
    // if(ratio < 0.01 || ratio > 100)      continue;
    _camera_data[i].f = fs[qi] / qcnum[qi];
    _camera_data[i].radial = rs[qi] / qcnum[qi];
  }

  qmapw.resize(_num_imgpt_q * 2, 0);
  qlistw.resize(_num_camera * 2, 0);
  for (int i = 0; i < _num_imgpt_q; ++i) {
    int cidx = qmap[i * 2], qi = _focal_mask[cidx];
    float wi = sqrt(qpnum[qi] / qcnum[qi]) * _weight_q;
    float wr = (__use_radial_distortion ? wi * _camera_data[qi].f : 0.0);
    qmapw[i * 2] = wi;
    qmapw[i * 2 + 1] = wr;
    qlistw[cidx * 2] = wi;
    qlistw[cidx * 2 + 1] = wr;
  }
}

void SparseBundleCU::ReleaseAllocatedData() {
  _cuCameraData.ReleaseData();
  _cuCameraDataEX.ReleaseData();
  _cuPointData.ReleaseData();
  _cuPointDataEX.ReleaseData();
  _cuMeasurements.ReleaseData();
  _cuImageProj.ReleaseData();
  _cuJacobianCamera.ReleaseData();
  _cuJacobianPoint.ReleaseData();
  _cuJacobianCameraT.ReleaseData();
  _cuProjectionMap.ReleaseData();
  _cuPointMeasurementMap.ReleaseData();
  _cuCameraMeasurementMap.ReleaseData();
  _cuCameraMeasurementList.ReleaseData();
  _cuCameraMeasurementListT.ReleaseData();
  _cuBufferData.ReleaseData();
  _cuBlockPC.ReleaseData();
  _cuVectorJtE.ReleaseData();
  _cuVectorJJ.ReleaseData();
  _cuVectorJX.ReleaseData();
  _cuVectorXK.ReleaseData();
  _cuVectorPK.ReleaseData();
  _cuVectorZK.ReleaseData();
  _cuVectorRK.ReleaseData();
  _cuVectorSJ.ReleaseData();
  _cuCameraQList.ReleaseData();
  _cuCameraQMap.ReleaseData();
  _cuCameraQMapW.ReleaseData();
  _cuCameraQListW.ReleaseData();
  ProgramCU::ResetCurrentDevice();
}

void SparseBundleCU::NormalizeDataF() {
  int incompatible_radial_distortion = 0;
  if (__focal_normalize) {
    if (__focal_scaling == 1.0f) {
      //------------------------------------------------------------------
      //////////////////////////////////////////////////////////////
      vector<float> focals(_num_camera);
      for (int i = 0; i < _num_camera; ++i) focals[i] = _camera_data[i].f;
      std::nth_element(focals.begin(), focals.begin() + _num_camera / 2,
                       focals.end());
      float median_focal_length = focals[_num_camera / 2];
      __focal_scaling = __data_normalize_median / median_focal_length;
      float radial_factor = median_focal_length * median_focal_length * 4.0f;

      ///////////////////////////////
      _imgpt_datax.resize(_num_imgpt * 2);
      for (int i = 0; i < _num_imgpt * 2; ++i)
        _imgpt_datax[i] = _imgpt_data[i] * __focal_scaling;
      for (int i = 0; i < _num_camera; ++i) {
        _camera_data[i].f *= __focal_scaling;
        if (!__use_radial_distortion) {
        } else if (__reset_initial_distortion) {
          _camera_data[i].radial = 0;
        } else if (_camera_data[i].distortion_type != __use_radial_distortion) {
          incompatible_radial_distortion++;
          _camera_data[i].radial = 0;
        } else if (__use_radial_distortion == -1) {
          _camera_data[i].radial *= radial_factor;
        }
      }
      if (__verbose_level > 2)
        std::cout << "Focal length normalized by " << __focal_scaling << '\n';
      __reset_initial_distortion = false;
    }
  } else {
    if (__use_radial_distortion) {
      for (int i = 0; i < _num_camera; ++i) {
        if (__reset_initial_distortion) {
          _camera_data[i].radial = 0;
        } else if (_camera_data[i].distortion_type != __use_radial_distortion) {
          _camera_data[i].radial = 0;
          incompatible_radial_distortion++;
        }
      }
      __reset_initial_distortion = false;
    }
    _imgpt_datax.resize(0);
  }

  if (incompatible_radial_distortion) {
    std::cout << "ERROR: incompatible radial distortion input; reset to 0;\n";
  }
}

void SparseBundleCU::NormalizeDataD() {
  if (__depth_scaling == 1.0f) {
    const float dist_bound = 1.0f;
    vector<float> oz(_num_imgpt);
    vector<float> cpdist1(_num_camera, dist_bound);
    vector<float> cpdist2(_num_camera, -dist_bound);
    vector<int> camnpj(_num_camera, 0), cambpj(_num_camera, 0);
    int bad_point_count = 0;
    for (int i = 0; i < _num_imgpt; ++i) {
      int cmidx = _camera_idx[i];
      CameraT* cam = _camera_data + cmidx;
      float* rz = cam->m[2];
      float* x = _point_data + 4 * _point_idx[i];
      oz[i] = (rz[0] * x[0] + rz[1] * x[1] + rz[2] * x[2] + cam->t[2]);

      /////////////////////////////////////////////////
      // points behind camera may causes big problem
      float ozr = oz[i] / cam->t[2];
      if (fabs(ozr) < __depth_check_epsilon) {
        bad_point_count++;
        float px = cam->f * (cam->m[0][0] * x[0] + cam->m[0][1] * x[1] +
                             cam->m[0][2] * x[2] + cam->t[0]);
        float py = cam->f * (cam->m[1][0] * x[0] + cam->m[1][1] * x[1] +
                             cam->m[1][2] * x[2] + cam->t[1]);
        float mx = _imgpt_data[i * 2], my = _imgpt_data[2 * i + 1];
        bool checkx = fabs(mx) > fabs(my);
        if ((checkx && px * oz[i] * mx < 0 && fabs(mx) > 64) ||
            (!checkx && py * oz[i] * my < 0 && fabs(my) > 64)) {
          if (__verbose_level > 3)
            std::cout << "Warning: proj of #" << cmidx
                      << " on the wrong side, oz = " << oz[i] << " ("
                      << (px / oz[i]) << ',' << (py / oz[i]) << ") (" << mx
                      << ',' << my << ")\n";
          /////////////////////////////////////////////////////////////////////////
          if (oz[i] > 0)
            cpdist2[cmidx] = 0;
          else
            cpdist1[cmidx] = 0;
        }
        if (oz[i] >= 0)
          cpdist1[cmidx] = std::min(cpdist1[cmidx], oz[i]);
        else
          cpdist2[cmidx] = std::max(cpdist2[cmidx], oz[i]);
      }
      if (oz[i] < 0) {
        __num_point_behind++;
        cambpj[cmidx]++;
      }
      camnpj[cmidx]++;
    }
    if (bad_point_count > 0 && __depth_degeneracy_fix) {
      if (!__focal_normalize || !__depth_normalize)
        std::cout << "Enable data normalization on degeneracy\n";
      __focal_normalize = true;
      __depth_normalize = true;
    }
    if (__depth_normalize) {
      std::nth_element(oz.begin(), oz.begin() + _num_imgpt / 2, oz.end());
      float oz_median = oz[_num_imgpt / 2];
      float shift_min = std::min(oz_median * 0.001f, 1.0f);
      float dist_threshold = shift_min * 0.1f;
      __depth_scaling = (1.0f / oz_median) / __data_normalize_median;
      if (__verbose_level > 2)
        std::cout << "Depth normalized by " << __depth_scaling << " ("
                  << oz_median << ")\n";

      for (int i = 0; i < _num_camera; ++i) {
        // move the camera a little bit?
        if (!__depth_degeneracy_fix) {
        } else if ((cpdist1[i] < dist_threshold ||
                    cpdist2[i] > -dist_threshold)) {
          float shift = shift_min;  //(cpdist1[i] <= -cpdist2[i] ? shift_min :
                                    //-shift_min);
          // if(cpdist1[i] < dist_bound && cpdist2[i] > - dist_bound) shift = -
          // 0.5f * (cpdist1[i] + cpdist2[i]);
          bool boths =
              cpdist1[i] < dist_threshold && cpdist2[i] > -dist_threshold;
          _camera_data[i].t[2] += shift;
          if (__verbose_level > 3)
            std::cout << "Adjust C" << std::setw(5) << i << " by "
                      << std::setw(12) << shift << " [B" << std::setw(2)
                      << cambpj[i] << "/" << std::setw(5) << camnpj[i] << "] ["
                      << (boths ? 'X' : ' ') << "][" << cpdist1[i] << ", "
                      << cpdist2[i] << "]\n";
          __num_camera_modified++;
        }
        _camera_data[i].t[0] *= __depth_scaling;
        _camera_data[i].t[1] *= __depth_scaling;
        _camera_data[i].t[2] *= __depth_scaling;
      }
      for (int i = 0; i < _num_point; ++i) {
        /////////////////////////////////
        _point_data[4 * i + 0] *= __depth_scaling;
        _point_data[4 * i + 1] *= __depth_scaling;
        _point_data[4 * i + 2] *= __depth_scaling;
      }
    }
    if (__num_point_behind > 0)
      std::cout << "WARNING: " << __num_point_behind
                << " points are behind cameras.\n";
    if (__num_camera_modified > 0)
      std::cout << "WARNING: " << __num_camera_modified
                << " camera moved to avoid degeneracy.\n";
  }
}

void SparseBundleCU::NormalizeData() {
  TimerBA timer(this, TIMER_PREPROCESSING);
  NormalizeDataD();
  NormalizeDataF();
}

void SparseBundleCU::DenormalizeData() {
  if (__focal_normalize && __focal_scaling != 1.0f) {
    float squared_focal_factor = (__focal_scaling * __focal_scaling);
    for (int i = 0; i < _num_camera; ++i) {
      _camera_data[i].f /= __focal_scaling;
      if (__use_radial_distortion == -1)
        _camera_data[i].radial *= squared_focal_factor;
      _camera_data[i].distortion_type = __use_radial_distortion;
    }
    _projection_sse /= squared_focal_factor;
    __focal_scaling = 1.0f;
    _imgpt_datax.resize(0);
  } else if (__use_radial_distortion) {
    for (int i = 0; i < _num_camera; ++i)
      _camera_data[i].distortion_type = __use_radial_distortion;
  }

  if (__depth_normalize && __depth_scaling != 1.0f) {
    for (int i = 0; i < _num_camera; ++i) {
      _camera_data[i].t[0] /= __depth_scaling;
      _camera_data[i].t[1] /= __depth_scaling;
      _camera_data[i].t[2] /= __depth_scaling;
    }
    for (int i = 0; i < _num_point; ++i) {
      _point_data[4 * i + 0] /= __depth_scaling;
      _point_data[4 * i + 1] /= __depth_scaling;
      _point_data[4 * i + 2] /= __depth_scaling;
    }
    __depth_scaling = 1.0f;
  }
}

void SparseBundleCU::TransferDataToHost() {
  TimerBA timer(this, TIMER_GPU_DOWNLOAD);
  _cuCameraData.CopyToHost(_camera_data);
  _cuPointData.CopyToHost(_point_data);
}

float SparseBundleCU::EvaluateProjection(CuTexImage& cam, CuTexImage& point,
                                         CuTexImage& proj) {
  ++__num_projection_eval;
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_PJ, true);
  ComputeProjection(cam, point, _cuMeasurements, _cuProjectionMap, proj,
                    __use_radial_distortion);
  if (_num_imgpt_q > 0)
    ComputeProjectionQ(cam, _cuCameraQMap, _cuCameraQMapW, proj, _num_imgpt);
  return (float)ComputeVectorNorm(proj, _cuBufferData);
}

float SparseBundleCU::EvaluateProjectionX(CuTexImage& cam, CuTexImage& point,
                                          CuTexImage& proj) {
  ++__num_projection_eval;
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_PJ, true);
  ComputeProjectionX(cam, point, _cuMeasurements, _cuProjectionMap, proj,
                     __use_radial_distortion);
  if (_num_imgpt_q > 0)
    ComputeProjectionQ(cam, _cuCameraQMap, _cuCameraQMapW, proj, _num_imgpt);
  return (float)ComputeVectorNorm(proj, _cuBufferData);
}

void SparseBundleCU::DebugProjections() {
  double e1 = 0, e2 = 0;
  for (int i = 0; i < _num_imgpt; ++i) {
    float* c = (float*)(_camera_data + _camera_idx[i]);
    float* p = _point_data + 4 * _point_idx[i];
    const float* m = _imgpt_datax.size() > 0 ? (&_imgpt_datax[i * 2])
                                             : (_imgpt_data + 2 * i);
    float* r = c + 4;
    float* t = c + 1;
    float dx1, dy1;
    ////////////////////////////////////////////////////////////////////////////////
    float z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
    float xx = (r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0]);
    float yy = (r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1]);
    float x = xx / z;
    float y = yy / z;
    if (__use_radial_distortion == -1) {
      float rn = (m[0] * m[0] + m[1] * m[1]) * c[13] + 1.0f;
      dx1 = c[0] * x - m[0] * rn;
      dy1 = c[0] * y - m[1] * rn;
      e1 += (dx1 * dx1 + dy1 * dy1);
      e2 += (dx1 * dx1 + dy1 * dy1) / (rn * rn);
    } else if (__use_radial_distortion) {
      float rn = (x * x + y * y) * c[13] + 1.0f;
      dx1 = c[0] * x * rn - m[0];
      dy1 = c[0] * y * rn - m[1];
      e1 += (dx1 * dx1 + dy1 * dy1) / (rn * rn);
      e2 += (dx1 * dx1 + dy1 * dy1);
    } else {
      dx1 = c[0] * x - m[0];
      dy1 = c[0] * y - m[1];
      e1 += (dx1 * dx1 + dy1 * dy1);
      e2 += (dx1 * dx1 + dy1 * dy1);
    }
    if (!isfinite(dx1) || !isfinite(dy1))
      std::cout << "x = " << xx << " y = " << yy << " z = " << z << '\n'
                << "t0 = " << t[0] << " t1 = " << t[1] << " t2 = " << t[2]
                << '\n' << "p0 = " << p[0] << " p1 = " << p[1]
                << " p2 = " << p[2] << '\n';
  }
  e1 = e1 / (__focal_scaling * __focal_scaling) / _num_imgpt;
  e2 = e2 / (__focal_scaling * __focal_scaling) / _num_imgpt;
  std::cout << "DEBUG: mean squared error = " << e1
            << " in undistorted domain;\n";
  std::cout << "DEBUG: mean squared error = " << e2
            << " in distorted domain.\n";
}

bool SparseBundleCU::InitializeStorageForCG() {
  bool success = true;
  size_t total_sz = 0;
  int plen = GetParameterLength();  // q = 8m + 4n

  //////////////////////////////////////////// 6q
  ALLOCATE_REQUIRED_DATA(_cuVectorJtE, plen, 1);
  ALLOCATE_REQUIRED_DATA(_cuVectorXK, plen, 1);
  ALLOCATE_REQUIRED_DATA(_cuVectorPK, plen, 1);
  ALLOCATE_REQUIRED_DATA(_cuVectorRK, plen, 1);
  ALLOCATE_REQUIRED_DATA(_cuVectorJJ, plen, 1);
  ALLOCATE_REQUIRED_DATA(_cuVectorZK, plen, 1);

  /////////////////////////////////
  unsigned int cblock_len = (__use_radial_distortion ? 64 : 56);
  ALLOCATE_REQUIRED_DATA(_cuBlockPC, _num_camera * cblock_len + 12 * _num_point,
                         1);  // 64m + 12n
  if (__accurate_gain_ratio) {
    ALLOCATE_REQUIRED_DATA(_cuVectorJX, _num_imgpt + _num_imgpt_q, 2);  // 2k
  } else {
    _cuVectorJX.SetTexture(_cuImageProj.data(), _num_imgpt + _num_imgpt_q, 2);
  }
  ALLOCATE_OPTIONAL_DATA(_cuVectorSJ, plen, 1, __jacobian_normalize);

  /////////////////////////////////////////
  __memory_usage += total_sz;
  if (__verbose_level > 1)
    std::cout << "Memory for Conjugate Gradient Solver:\t"
              << (total_sz / 1024 / 1024) << "MB\n";
  return success;
}

void SparseBundleCU::PrepareJacobianNormalization() {
  if (!_cuVectorSJ.IsValid()) return;

  if ((__jc_store_transpose || __jc_store_original) &&
      _cuJacobianPoint.IsValid() && !__bundle_current_mode) {
    CuTexImage null;
    null.SwapData(_cuVectorSJ);
    EvaluateJacobians();
    null.SwapData(_cuVectorSJ);
    ComputeDiagonal(_cuVectorJJ, _cuVectorSJ);
    ComputeSQRT(_cuVectorSJ);
  } else {
    CuTexImage null;
    null.SwapData(_cuVectorSJ);
    EvaluateJacobians();
    ComputeBlockPC(0, true);
    null.SwapData(_cuVectorSJ);
    _cuVectorJJ.SwapData(_cuVectorSJ);
    ProgramCU::ComputeRSQRT(_cuVectorSJ);
  }
}

void SparseBundleCU::EvaluateJacobians(bool shuffle) {
  if (__no_jacobian_store) return;
  if (__bundle_current_mode == BUNDLE_ONLY_MOTION && !__jc_store_original &&
      !__jc_store_transpose)
    return;
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_JJ, true);

  if (__jc_store_original || !__jc_store_transpose) {
    ComputeJacobian(_cuCameraData, _cuPointData, _cuJacobianCamera,
                    _cuJacobianPoint, _cuProjectionMap, _cuVectorSJ,
                    _cuMeasurements, _cuCameraMeasurementList,
                    __fixed_intrinsics, __use_radial_distortion, false);
    if (shuffle && __jc_store_transpose && _cuJacobianCameraT.IsValid())
      ShuffleCameraJacobian(_cuJacobianCamera, _cuCameraMeasurementList,
                            _cuJacobianCameraT);
  } else {
    ComputeJacobian(_cuCameraData, _cuPointData, _cuJacobianCameraT,
                    _cuJacobianPoint, _cuProjectionMap, _cuVectorSJ,
                    _cuMeasurements, _cuCameraMeasurementListT,
                    __fixed_intrinsics, __use_radial_distortion, true);
  }
  ++__num_jacobian_eval;
}

void SparseBundleCU::ComputeJtE(CuTexImage& E, CuTexImage& JtE, int mode) {
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_JTE, true);
  if (mode == 0) mode = __bundle_current_mode;
  if (__no_jacobian_store || (!__jc_store_original && !__jc_store_transpose)) {
    ProgramCU::ComputeJtE_(E, JtE, _cuCameraData, _cuPointData, _cuMeasurements,
                           _cuCameraMeasurementMap, _cuCameraMeasurementList,
                           _cuPointMeasurementMap, _cuProjectionMap,
                           _cuJacobianPoint, __fixed_intrinsics,
                           __use_radial_distortion, mode);

    ////////////////////////////////////////////////////////////////////////////////////
    if (!_cuVectorSJ.IsValid()) {
    } else if (mode == 2) {
      if (!_cuJacobianPoint.IsValid())
        ComputeVXY(JtE, _cuVectorSJ, JtE, _num_point * 4, _num_camera * 8);
    } else if (mode == 1)
      ComputeVXY(JtE, _cuVectorSJ, JtE, _num_camera * 8);
    else
      ComputeVXY(JtE, _cuVectorSJ, JtE,
                 _cuJacobianPoint.IsValid() ? _num_camera * 8 : 0);

  } else if (__jc_store_transpose) {
    ProgramCU::ComputeJtE(E, _cuJacobianCameraT, _cuCameraMeasurementMap,
                          _cuCameraMeasurementList, _cuJacobianPoint,
                          _cuPointMeasurementMap, JtE, true, mode);
  } else {
    ProgramCU::ComputeJtE(E, _cuJacobianCamera, _cuCameraMeasurementMap,
                          _cuCameraMeasurementList, _cuJacobianPoint,
                          _cuPointMeasurementMap, JtE, false, mode);
  }

  if (mode != 2 && _num_imgpt_q > 0)
    ProgramCU::ComputeJQtEC(E, _cuCameraQList, _cuCameraQListW, _cuVectorSJ,
                            JtE);
}

void SparseBundleCU::SaveBundleRecord(int iter, float res, float damping,
                                      float& g_norm, float& g_inf) {
  // do not really compute if parameter not specified...
  // for large dataset, it never converges..
  g_inf =
      __lm_check_gradient ? ComputeVectorMax(_cuVectorJtE, _cuBufferData) : 0;
  g_norm = __save_gradient_norm
               ? float(ComputeVectorNorm(_cuVectorJtE, _cuBufferData))
               : g_inf;
  ConfigBA::SaveBundleRecord(iter, res, damping, g_norm, g_inf);
}

void SparseBundleCU::ComputeJX(CuTexImage& X, CuTexImage& JX, int mode) {
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_JX, true);
  if (__no_jacobian_store || (__multiply_jx_usenoj && mode != 2) ||
      !__jc_store_original) {
    if (_cuVectorSJ.IsValid()) {
      if (mode == 0)
        ProgramCU::ComputeVXY(X, _cuVectorSJ, _cuVectorZK);
      else if (mode == 1)
        ProgramCU::ComputeVXY(X, _cuVectorSJ, _cuVectorZK, _num_camera * 8);
      else if (mode == 2)
        ProgramCU::ComputeVXY(X, _cuVectorSJ, _cuVectorZK, _num_point * 4,
                              _num_camera * 8);
      ProgramCU::ComputeJX_(_cuVectorZK, JX, _cuCameraData, _cuPointData,
                            _cuMeasurements, _cuProjectionMap,
                            __fixed_intrinsics, __use_radial_distortion, mode);
    } else {
      ProgramCU::ComputeJX_(X, JX, _cuCameraData, _cuPointData, _cuMeasurements,
                            _cuProjectionMap, __fixed_intrinsics,
                            __use_radial_distortion, mode);
    }
  } else {
    ProgramCU::ComputeJX(_num_camera * 2, X, _cuJacobianCamera,
                         _cuJacobianPoint, _cuProjectionMap, JX, mode);
  }

  if (_num_imgpt_q > 0 && mode != 2) {
    ProgramCU::ComputeJQX(X, _cuCameraQMap, _cuCameraQMapW, _cuVectorSJ, JX,
                          _num_imgpt);
  }
}

void SparseBundleCU::ComputeBlockPC(float lambda, bool dampd) {
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_BC, true);

  bool use_diagonal_q = _cuCameraQListW.IsValid() && __bundle_current_mode != 2;
  if (use_diagonal_q)
    ComputeDiagonalQ(_cuCameraQListW, _cuVectorSJ, _cuVectorJJ);

  if (__no_jacobian_store || (!__jc_store_original && !__jc_store_transpose)) {
    ComputeDiagonalBlock_(
        lambda, dampd, _cuCameraData, _cuPointData, _cuMeasurements,
        _cuCameraMeasurementMap, _cuCameraMeasurementList,
        _cuPointMeasurementMap, _cuProjectionMap, _cuJacobianPoint, _cuVectorSJ,
        _cuVectorJJ, _cuBlockPC, __fixed_intrinsics, __use_radial_distortion,
        use_diagonal_q, __bundle_current_mode);
  } else if (__jc_store_transpose) {
    ComputeDiagonalBlock(lambda, dampd, _cuJacobianCameraT,
                         _cuCameraMeasurementMap, _cuJacobianPoint,
                         _cuPointMeasurementMap, _cuCameraMeasurementList,
                         _cuVectorJJ, _cuBlockPC, __use_radial_distortion, true,
                         use_diagonal_q, __bundle_current_mode);
  } else {
    ComputeDiagonalBlock(lambda, dampd, _cuJacobianCamera,
                         _cuCameraMeasurementMap, _cuJacobianPoint,
                         _cuPointMeasurementMap, _cuCameraMeasurementList,
                         _cuVectorJJ, _cuBlockPC, __use_radial_distortion,
                         false, use_diagonal_q, __bundle_current_mode);
  }
}

void SparseBundleCU::ApplyBlockPC(CuTexImage& v, CuTexImage& pv, int mode) {
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_MP, true);
  MultiplyBlockConditioner(_num_camera, _num_point, _cuBlockPC, v, pv,
                           __use_radial_distortion, mode);
}

void SparseBundleCU::ComputeDiagonal(CuTexImage& JJ, CuTexImage& JJI) {
  ////////////////////checking the impossible.
  if (__no_jacobian_store) return;
  if (!__jc_store_transpose && !__jc_store_original) return;

  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_DD, true);
  bool use_diagonal_q = _cuCameraQListW.IsValid();
  if (use_diagonal_q) {
    CuTexImage null;
    ComputeDiagonalQ(_cuCameraQListW, null, JJ);
  }
  if (__jc_store_transpose) {
    ProgramCU::ComputeDiagonal(_cuJacobianCameraT, _cuCameraMeasurementMap,
                               _cuJacobianPoint, _cuPointMeasurementMap,
                               _cuCameraMeasurementList, JJ, JJI, true,
                               __use_radial_distortion, use_diagonal_q);
  } else {
    ProgramCU::ComputeDiagonal(_cuJacobianCamera, _cuCameraMeasurementMap,
                               _cuJacobianPoint, _cuPointMeasurementMap,
                               _cuCameraMeasurementList, JJ, JJI, false,
                               __use_radial_distortion, use_diagonal_q);
  }
}

int SparseBundleCU::SolveNormalEquationPCGX(float lambda) {
  //----------------------------------------------------------
  //(Jt * J + lambda * diag(Jt * J)) X = Jt * e
  //-------------------------------------------------------------
  TimerBA timer(this, TIMER_CG_ITERATION);
  __recent_cg_status = ' ';

  // diagonal for jacobian preconditioning...
  int plen = GetParameterLength();
  CuTexImage null;
  CuTexImage& VectorDP =
      __lm_use_diagonal_damp ? _cuVectorJJ : null;  // diagonal
  ComputeBlockPC(lambda, __lm_use_diagonal_damp);

  ///////////////////////////////////////////////////////
  // B = [BC 0 ; 0 BP]
  // m = [mc 0; 0 mp];
  // A x= BC * x - JcT * Jp * mp * JpT * Jc * x
  //   = JcT * Jc x + lambda * D * x + ........
  ////////////////////////////////////////////////////////////

  CuTexImage r;
  r.SetTexture(_cuVectorRK.data(), 8 * _num_camera);
  CuTexImage p;
  p.SetTexture(_cuVectorPK.data(), 8 * _num_camera);
  CuTexImage z;
  z.SetTexture(_cuVectorZK.data(), 8 * _num_camera);
  CuTexImage x;
  x.SetTexture(_cuVectorXK.data(), 8 * _num_camera);
  CuTexImage d;
  d.SetTexture(VectorDP.data(), 8 * _num_camera);

  CuTexImage& u = _cuVectorRK;
  CuTexImage& v = _cuVectorPK;
  CuTexImage up;
  up.SetTexture(u.data() + 8 * _num_camera, 4 * _num_point);
  CuTexImage vp;
  vp.SetTexture(v.data() + 8 * _num_camera, 4 * _num_point);
  CuTexImage uc;
  uc.SetTexture(z.data(), 8 * _num_camera);

  CuTexImage& e = _cuVectorJX;
  CuTexImage& e2 = _cuImageProj;

  ApplyBlockPC(_cuVectorJtE, u, 2);
  ComputeJX(u, e, 2);
  ComputeJtE(e, uc, 1);
  ComputeSAXPY(-1.0f, uc, _cuVectorJtE, r);  // r
  ApplyBlockPC(r, p, 1);  // z = p = M r

  float_t rtz0 = (float_t)ComputeVectorDot(r, p, _cuBufferData);  // r(0)' *
                                                                  // z(0)
  ComputeJX(p, e, 1);  // Jc * x
  ComputeJtE(e, u, 2);  // JpT * jc * x
  ApplyBlockPC(u, v, 2);
  float_t qtq0 = (float_t)ComputeVectorNorm(e, _cuBufferData);  // q(0)' * q(0)
  float_t pdp0 =
      (float_t)ComputeVectorNormW(p, d, _cuBufferData);  // p(0)' * DDD * p(0)
  float_t uv0 = (float_t)ComputeVectorDot(up, vp, _cuBufferData);
  float_t alpha0 = rtz0 / (qtq0 + lambda * pdp0 - uv0);

  if (__verbose_cg_iteration)
    std::cout << " --0,\t alpha = " << alpha0
              << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";
  if (!isfinite(alpha0)) {
    return 0;
  }
  if (alpha0 == 0) {
    __recent_cg_status = 'I';
    return 1;
  }

  ////////////////////////////////////////////////////////////
  ComputeSAX((float)alpha0, p, x);  // x(k+1) = x(k) + a(k) * p(k)
  ComputeJX(v, e2, 2);  //                          //Jp * mp * JpT * JcT * p
  ComputeSAXPY(-1.0f, e2, e, e);
  ComputeJtE(e, uc, 1);  // JcT * ....
  ComputeSXYPZ(lambda, d, p, uc, uc);
  ComputeSAXPY((float)-alpha0, uc, r, r);  // r(k + 1) = r(k) - a(k) * A * pk

  //////////////////////////////////////////////////////////////////////////
  float_t rtzk = rtz0, rtz_min = rtz0, betak;
  int iteration = 1;
  ++__num_cg_iteration;

  while (true) {
    ApplyBlockPC(r, z, 1);

    ///////////////////////////////////////////////////////////////////////////
    float_t rtzp = rtzk;
    rtzk = (float_t)ComputeVectorDot(
        r, z, _cuBufferData);  //[r(k + 1) = M^(-1) * z(k + 1)] * z(k+1)
    float_t rtz_ratio = sqrt(fabs(rtzk / rtz0));

    if (rtz_ratio < __cg_norm_threshold) {
      if (__recent_cg_status == ' ')
        __recent_cg_status = iteration < std::min(10, __cg_min_iteration)
                                 ? '0' + iteration
                                 : 'N';
      if (iteration >= __cg_min_iteration) break;
    }
    ////////////////////////////////////////////////////////////////////////////
    betak = rtzk / rtzp;  // beta
    rtz_min = std::min(rtz_min, rtzk);

    ComputeSAXPY((float)betak, p, z, p);  // p(k) = z(k) + b(k) * p(k - 1)
    ComputeJX(p, e, 1);  // Jc * p
    ComputeJtE(e, u, 2);  // JpT * jc * p
    ApplyBlockPC(u, v, 2);
    //////////////////////////////////////////////////////////////////////

    float_t qtqk = (float_t)ComputeVectorNorm(e, _cuBufferData);  // q(k)' q(k)
    float_t pdpk =
        (float_t)ComputeVectorNormW(p, d, _cuBufferData);  // p(k)' * DDD * p(k)
    float_t uvk = (float_t)ComputeVectorDot(up, vp, _cuBufferData);
    float_t alphak = rtzk / (qtqk + lambda * pdpk - uvk);

    /////////////////////////////////////////////////////
    if (__verbose_cg_iteration)
      std::cout << " --" << iteration << ",\t alpha= " << alphak
                << ", rtzk/rtz0 = " << rtz_ratio
                << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";

    ///////////////////////////////////////////////////
    if (!isfinite(alphak) || rtz_ratio > __cg_norm_guard) {
      __recent_cg_status = 'X';
      break;
    }  // something doesn't converge..

    ////////////////////////////////////////////////
    ComputeSAXPY((float)alphak, p, x, x);  // x(k+1) = x(k) + a(k) * p(k)

    /////////////////////////////////////////////////
    ++iteration;
    ++__num_cg_iteration;
    if (iteration >= std::min(__cg_max_iteration, plen)) break;

    ComputeJX(v, e2, 2);  //                          //Jp * mp * JpT * JcT * p
    ComputeSAXPY(-1.0f, e2, e, e);
    ComputeJtE(e, uc, 1);  // JcT * ....
    ComputeSXYPZ(lambda, d, p, uc, uc);
    ComputeSAXPY((float)-alphak, uc, r, r);  // r(k + 1) = r(k) - a(k) * A * pk
  }

  // if(__recent_cg_status == 'X')     return iteration;

  ComputeJX(x, e, 1);
  ComputeJtE(e, u, 2);
  CuTexImage jte_p;
  jte_p.SetTexture(_cuVectorJtE.data() + 8 * _num_camera, _num_point * 4);
  ComputeSAXPY(-1.0f, up, jte_p, vp);
  ApplyBlockPC(v, _cuVectorXK, 2);
  return iteration;
}
int SparseBundleCU::SolveNormalEquationPCGB(float lambda) {
  //----------------------------------------------------------
  //(Jt * J + lambda * diag(Jt * J)) X = Jt * e
  //-------------------------------------------------------------
  TimerBA timer(this, TIMER_CG_ITERATION);
  __recent_cg_status = ' ';

  // diagonal for jacobian preconditioning...
  int plen = GetParameterLength();
  CuTexImage null;
  CuTexImage& VectorDP =
      __lm_use_diagonal_damp ? _cuVectorJJ : null;  // diagonal
  CuTexImage& VectorQK = _cuVectorZK;  // temporary
  ComputeBlockPC(lambda, __lm_use_diagonal_damp);

  ////////////////////////////////////////////////////////
  ApplyBlockPC(_cuVectorJtE,
               _cuVectorPK);  // z(0) = p(0) = M * r(0)//r(0) = Jt * e
  ComputeJX(_cuVectorPK, _cuVectorJX);  // q(0) = J * p(0)

  //////////////////////////////////////////////////
  float_t rtz0 = (float_t)ComputeVectorDot(_cuVectorJtE, _cuVectorPK,
                                           _cuBufferData);  // r(0)' * z(0)
  float_t qtq0 =
      (float_t)ComputeVectorNorm(_cuVectorJX, _cuBufferData);  // q(0)' * q(0)
  float_t ptdp0 = (float_t)ComputeVectorNormW(
      _cuVectorPK, VectorDP, _cuBufferData);  // p(0)' * DDD * p(0)
  float_t alpha0 = rtz0 / (qtq0 + lambda * ptdp0);

  if (__verbose_cg_iteration)
    std::cout << " --0,\t alpha = " << alpha0
              << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";
  if (!isfinite(alpha0)) {
    return 0;
  }
  if (alpha0 == 0) {
    __recent_cg_status = 'I';
    return 1;
  }

  ////////////////////////////////////////////////////////////
  ComputeSAX((float)alpha0, _cuVectorPK,
             _cuVectorXK);  // x(k+1) = x(k) + a(k) * p(k)
  ComputeJtE(_cuVectorJX, VectorQK);  // Jt * (J * p0)

  ComputeSXYPZ(lambda, VectorDP, _cuVectorPK, VectorQK,
               VectorQK);  // Jt * J * p0 + lambda * DDD * p0
  ComputeSAXPY(
      (float)-alpha0, VectorQK, _cuVectorJtE,
      _cuVectorRK);  // r(k+1) = r(k) - a(k) * (Jt * q(k)  + DDD * p(k)) ;

  float_t rtzk = rtz0, rtz_min = rtz0, betak;
  int iteration = 1;
  ++__num_cg_iteration;

  while (true) {
    ApplyBlockPC(_cuVectorRK, _cuVectorZK);

    ///////////////////////////////////////////////////////////////////////////
    float_t rtzp = rtzk;
    rtzk = (float_t)ComputeVectorDot(
        _cuVectorRK, _cuVectorZK,
        _cuBufferData);  //[r(k + 1) = M^(-1) * z(k + 1)] * z(k+1)
    float_t rtz_ratio = sqrt(fabs(rtzk / rtz0));
    if (rtz_ratio < __cg_norm_threshold) {
      if (__recent_cg_status == ' ')
        __recent_cg_status = iteration < std::min(10, __cg_min_iteration)
                                 ? '0' + iteration
                                 : 'N';
      if (iteration >= __cg_min_iteration) break;
    }

    ////////////////////////////////////////////////////////////////////////////
    betak = rtzk / rtzp;  // beta
    rtz_min = std::min(rtz_min, rtzk);

    ComputeSAXPY((float)betak, _cuVectorPK, _cuVectorZK,
                 _cuVectorPK);  // p(k) = z(k) + b(k) * p(k - 1)
    ComputeJX(_cuVectorPK, _cuVectorJX);  // q(k) = J * p(k)
    //////////////////////////////////////////////////////////////////////

    float_t qtqk =
        (float_t)ComputeVectorNorm(_cuVectorJX, _cuBufferData);  // q(k)' q(k)
    float_t ptdpk = (float_t)ComputeVectorNormW(
        _cuVectorPK, VectorDP, _cuBufferData);  // p(k)' * DDD * p(k)
    float_t alphak = rtzk / (qtqk + lambda * ptdpk);

    /////////////////////////////////////////////////////
    if (__verbose_cg_iteration)
      std::cout << " --" << iteration << ",\t alpha= " << alphak
                << ", rtzk/rtz0 = " << rtz_ratio
                << ", t = " << BundleTimerGetNow(TIMER_CG_ITERATION) << "\n";

    ///////////////////////////////////////////////////
    if (!isfinite(alphak) || rtz_ratio > __cg_norm_guard) {
      __recent_cg_status = 'X';
      break;
    }  // something doesn't converge..

    ////////////////////////////////////////////////
    ComputeSAXPY((float)alphak, _cuVectorPK, _cuVectorXK,
                 _cuVectorXK);  // x(k+1) = x(k) + a(k) * p(k)

    /////////////////////////////////////////////////
    ++iteration;
    ++__num_cg_iteration;
    if (iteration >= std::min(__cg_max_iteration, plen)) break;

    // if(iteration == 2 && rtz_ratio < __cg_norm_threshold)
    if (__cg_recalculate_freq > 0 && iteration % __cg_recalculate_freq == 0) {
      ////r = JtE - (Jt J + lambda * D) x
      ComputeJX(_cuVectorXK, _cuVectorJX);
      ComputeJtE(_cuVectorJX, VectorQK);
      ComputeSXYPZ(lambda, VectorDP, _cuVectorXK, VectorQK, VectorQK);
      ComputeSAXPY(-1.0f, VectorQK, _cuVectorJtE, _cuVectorRK);
    } else {
      ComputeJtE(_cuVectorJX, VectorQK);
      ComputeSXYPZ(lambda, VectorDP, _cuVectorPK, VectorQK, VectorQK);  //
      ComputeSAXPY(
          (float)-alphak, VectorQK, _cuVectorRK,
          _cuVectorRK);  // r(k+1) = r(k) - a(k) * (Jt * q(k)  + DDD * p(k)) ;
    }
  }
  return iteration;
}

int SparseBundleCU::SolveNormalEquation(float lambda) {
  if (__bundle_current_mode == BUNDLE_ONLY_MOTION) {
    ComputeBlockPC(lambda, __lm_use_diagonal_damp);
    ApplyBlockPC(_cuVectorJtE, _cuVectorXK, 1);
    return 1;
  } else if (__bundle_current_mode == BUNDLE_ONLY_STRUCTURE) {
    ComputeBlockPC(lambda, __lm_use_diagonal_damp);
    ApplyBlockPC(_cuVectorJtE, _cuVectorXK, 2);
    return 1;
  } else {
    ////solve linear system using Conjugate Gradients
    return __cg_schur_complement ? SolveNormalEquationPCGX(lambda)
                                 : SolveNormalEquationPCGB(lambda);
  }
}

void SparseBundleCU::RunTestIterationLM(bool reduced) {
  EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
  EvaluateJacobians();
  ComputeJtE(_cuImageProj, _cuVectorJtE);
  if (reduced)
    SolveNormalEquationPCGX(__lm_initial_damp);
  else
    SolveNormalEquationPCGB(__lm_initial_damp);
  UpdateCameraPoint(_cuVectorZK, _cuImageProj);
  ComputeVectorDot(_cuVectorXK, _cuVectorJtE, _cuBufferData);
  ComputeJX(_cuVectorXK, _cuVectorJX);
  ComputeVectorNorm(_cuVectorJX, _cuBufferData);
}

float SparseBundleCU::UpdateCameraPoint(CuTexImage& dx,
                                        CuTexImage& cuImageTempProj) {
  ConfigBA::TimerBA timer(this, TIMER_FUNCTION_UP, true);
  if (__bundle_current_mode == BUNDLE_ONLY_MOTION) {
    if (__jacobian_normalize)
      ComputeVXY(_cuVectorXK, _cuVectorSJ, dx, 8 * _num_camera);
    ProgramCU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx,
                                 _cuCameraDataEX, _cuPointDataEX,
                                 __bundle_current_mode);
    return EvaluateProjection(_cuCameraDataEX, _cuPointData, cuImageTempProj);
  } else if (__bundle_current_mode == BUNDLE_ONLY_STRUCTURE) {
    if (__jacobian_normalize)
      ComputeVXY(_cuVectorXK, _cuVectorSJ, dx, 4 * _num_point, 8 * _num_camera);
    ProgramCU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx,
                                 _cuCameraDataEX, _cuPointDataEX,
                                 __bundle_current_mode);
    return EvaluateProjection(_cuCameraData, _cuPointDataEX, cuImageTempProj);
  } else {
    if (__jacobian_normalize) ComputeVXY(_cuVectorXK, _cuVectorSJ, dx);
    ProgramCU::UpdateCameraPoint(_num_camera, _cuCameraData, _cuPointData, dx,
                                 _cuCameraDataEX, _cuPointDataEX,
                                 __bundle_current_mode);
    return EvaluateProjection(_cuCameraDataEX, _cuPointDataEX, cuImageTempProj);
  }
}

float SparseBundleCU::SaveUpdatedSystem(float residual_reduction,
                                        float dx_sqnorm, float damping) {
  float expected_reduction;
  if (__bundle_current_mode == BUNDLE_ONLY_MOTION) {
    CuTexImage xk;
    xk.SetTexture(_cuVectorXK.data(), 8 * _num_camera);
    CuTexImage jte;
    jte.SetTexture(_cuVectorJtE.data(), 8 * _num_camera);
    float dxtg = (float)ComputeVectorDot(xk, jte, _cuBufferData);
    if (__lm_use_diagonal_damp) {
      CuTexImage jj;
      jj.SetTexture(_cuVectorJJ.data(), 8 * _num_camera);
      float dq = (float)ComputeVectorNormW(xk, jj, _cuBufferData);
      expected_reduction = damping * dq + dxtg;
    } else {
      expected_reduction = damping * dx_sqnorm + dxtg;
    }
    _cuCameraData.SwapData(_cuCameraDataEX);
  } else if (__bundle_current_mode == BUNDLE_ONLY_STRUCTURE) {
    CuTexImage xk;
    xk.SetTexture(_cuVectorXK.data() + 8 * _num_camera, 4 * _num_point);
    CuTexImage jte;
    jte.SetTexture(_cuVectorJtE.data() + 8 * _num_camera, 4 * _num_point);
    float dxtg = (float)ComputeVectorDot(xk, jte, _cuBufferData);
    if (__lm_use_diagonal_damp) {
      CuTexImage jj;
      jj.SetTexture(_cuVectorJJ.data() + 8 * _num_camera, 4 * _num_point);
      float dq = (float)ComputeVectorNormW(xk, jj, _cuBufferData);
      expected_reduction = damping * dq + dxtg;
    } else {
      expected_reduction = damping * dx_sqnorm + dxtg;
    }
    _cuPointData.SwapData(_cuPointDataEX);
  } else {
    float dxtg =
        (float)ComputeVectorDot(_cuVectorXK, _cuVectorJtE, _cuBufferData);

    if (__accurate_gain_ratio) {
      ComputeJX(_cuVectorXK, _cuVectorJX);
      float njx = (float)ComputeVectorNorm(_cuVectorJX, _cuBufferData);
      expected_reduction = 2.0f * dxtg - njx;
      // could the expected reduction be negative??? not sure
      if (expected_reduction <= 0)
        expected_reduction = 0.001f * residual_reduction;
    } else if (__lm_use_diagonal_damp) {
      float dq =
          (float)ComputeVectorNormW(_cuVectorXK, _cuVectorJJ, _cuBufferData);
      expected_reduction = damping * dq + dxtg;
    } else {
      expected_reduction = damping * dx_sqnorm + dxtg;
    }

    /// save the new motion/struture
    _cuCameraData.SwapData(_cuCameraDataEX);
    _cuPointData.SwapData(_cuPointDataEX);

    //_cuCameraData.CopyToHost(_camera_data);
    //_cuPointData.CopyToHost(_point_data);
    // DebugProjections();
  }
  ////////////////////////////////////////////
  return float(residual_reduction / expected_reduction);
}

void SparseBundleCU::AdjustBundleAdjsutmentMode() {
  if (__bundle_current_mode == BUNDLE_ONLY_STRUCTURE) {
    _cuJacobianCamera.InitTexture(0, 0);
    _cuJacobianCameraT.InitTexture(0, 0);
  }
}

float SparseBundleCU::EvaluateDeltaNorm() {
  if (__bundle_current_mode == BUNDLE_ONLY_MOTION) {
    CuTexImage temp;
    temp.SetTexture(_cuVectorXK.data(), 8 * _num_camera);
    return ComputeVectorNorm(temp, _cuBufferData);

  } else if (__bundle_current_mode == BUNDLE_ONLY_STRUCTURE) {
    CuTexImage temp;
    temp.SetTexture(_cuVectorXK.data() + 8 * _num_camera, 4 * _num_point);
    return ComputeVectorNorm(temp, _cuBufferData);
  } else {
    return (float)ComputeVectorNorm(_cuVectorXK, _cuBufferData);
  }
}

void SparseBundleCU::NonlinearOptimizeLM() {
  ////////////////////////////////////////
  TimerBA timer(this, TIMER_OPTIMIZATION);

  ////////////////////////////////////////////////
  float mse_convert_ratio =
      1.0f / (_num_imgpt * __focal_scaling * __focal_scaling);
  float error_display_ratio = __verbose_sse ? _num_imgpt : 1.0f;
  const int edwidth = __verbose_sse ? 12 : 8;
  _projection_sse =
      EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
  __initial_mse = __final_mse = _projection_sse * mse_convert_ratio;

  // compute jacobian diagonals for normalization
  if (__jacobian_normalize) PrepareJacobianNormalization();

  // evalaute jacobian
  EvaluateJacobians();
  ComputeJtE(_cuImageProj, _cuVectorJtE);
  ///////////////////////////////////////////////////////////////
  if (__verbose_level)
    std::cout << "Initial " << (__verbose_sse ? "sumed" : "mean")
              << " squared error = " << __initial_mse * error_display_ratio
              << "\n----------------------------------------------\n";

  //////////////////////////////////////////////////
  CuTexImage& cuImageTempProj = _cuVectorJX;
  // CuTexImage& cuVectorTempJX  =   _cuVectorJX;
  CuTexImage& cuVectorDX = _cuVectorSJ.IsValid() ? _cuVectorZK : _cuVectorXK;

  //////////////////////////////////////////////////
  float damping_adjust = 2.0f, damping = __lm_initial_damp, g_norm, g_inf;
  SaveBundleRecord(0, _projection_sse * mse_convert_ratio, damping, g_norm,
                   g_inf);

  ////////////////////////////////////
  std::cout << std::left;
  for (int i = 0; i < __lm_max_iteration && !__abort_flag;
       __current_iteration = (++i)) {
    ////solve linear system
    int num_cg_iteration = SolveNormalEquation(damping);

    // there must be NaN somewhere
    if (num_cg_iteration == 0) {
      if (__verbose_level)
        std::cout << "#" << std::setw(3) << i << " quit on numeric errors\n";
      __pba_return_code = 'E';
      break;
    }

    // there must be infinity somewhere
    if (__recent_cg_status == 'I') {
      std::cout << "#" << std::setw(3) << i << " 0  I e=" << std::setw(edwidth)
                << "------- "
                << " u=" << std::setprecision(3) << std::setw(9) << damping
                << '\n' << std::setprecision(6);
      /////////////increase damping factor
      damping = damping * damping_adjust;
      damping_adjust = 2.0f * damping_adjust;
      --i;
      continue;
    }

    /////////////////////
    ++__num_lm_iteration;

    ////////////////////////////////////
    float dx_sqnorm = EvaluateDeltaNorm(), dx_norm = sqrt(dx_sqnorm);

    // In this library, we check absolute difference instead of realtive
    // difference
    if (dx_norm <= __lm_delta_threshold) {
      // damping factor must be way too big...or it converges
      if (__verbose_level > 1)
        std::cout << "#" << std::setw(3) << i << " " << std::setw(3)
                  << num_cg_iteration << char(__recent_cg_status)
                  << " quit on too small change (" << dx_norm << "  < "
                  << __lm_delta_threshold << ")\n";
      __pba_return_code = 'S';
      break;
    }
    ///////////////////////////////////////////////////////////////////////
    // update structure and motion, check reprojection error
    float new_residual = UpdateCameraPoint(cuVectorDX, cuImageTempProj);
    float average_residual = new_residual * mse_convert_ratio;
    float residual_reduction = _projection_sse - new_residual;

    // do we find a better solution?
    if (isfinite(new_residual) && residual_reduction > 0) {
      ////compute relative norm change
      float relative_reduction = 1.0f - (new_residual / _projection_sse);

      ////////////////////////////////////
      __num_lm_success++;  // increase counter
      _projection_sse = new_residual;  // save the new residual
      _cuImageProj.SwapData(cuImageTempProj);  // save the new projection

      ///////////////gain ratio////////////////////
      float gain_ratio =
          SaveUpdatedSystem(residual_reduction, dx_sqnorm, damping);

      /////////////////////////////////////
      SaveBundleRecord(i + 1, _projection_sse * mse_convert_ratio, damping,
                       g_norm, g_inf);

      /////////////////////////////////////////////
      if (__verbose_level > 1)
        std::cout << "#" << std::setw(3) << i << " " << std::setw(3)
                  << num_cg_iteration << char(__recent_cg_status)
                  << " e=" << std::setw(edwidth)
                  << average_residual * error_display_ratio
                  << " u=" << std::setprecision(3) << std::setw(9) << damping
                  << " r=" << std::setw(6)
                  << floor(gain_ratio * 1000.f) * 0.001f
                  << " g=" << std::setw(g_norm > 0 ? 9 : 1) << g_norm << " "
                  << std::setw(9) << relative_reduction << ' ' << std::setw(9)
                  << dx_norm << " t=" << int(BundleTimerGetNow()) << "\n"
                  << std::setprecision(6);

      /////////////////////////////
      if (!IsTimeBudgetAvailable()) {
        if (__verbose_level > 1)
          std::cout << "#" << std::setw(3) << i << " used up time budget.\n";
        __pba_return_code = 'T';
        break;
      } else if (__lm_check_gradient && g_inf < __lm_gradient_threshold) {
        if (__verbose_level > 1)
          std::cout << "#" << std::setw(3) << i
                    << " converged with small gradient\n";
        __pba_return_code = 'G';
        break;
      } else if (average_residual * error_display_ratio <= __lm_mse_threshold) {
        if (__verbose_level > 1)
          std::cout << "#" << std::setw(3) << i << " satisfies MSE threshold\n";
        __pba_return_code = 'M';
        break;
      } else {
        /////////////////////////////adjust damping factor
        float temp = gain_ratio * 2.0f - 1.0f;
        float adaptive_adjust = 1.0f - temp * temp * temp;  // powf(, 3.0f); //
        float auto_adjust = std::max(1.0f / 3.0f, adaptive_adjust);

        //////////////////////////////////////////////////
        damping = damping * auto_adjust;
        damping_adjust = 2.0f;
        if (damping < __lm_minimum_damp)
          damping = __lm_minimum_damp;
        else if (__lm_damping_auto_switch == 0 && damping > __lm_maximum_damp &&
                 __lm_use_diagonal_damp)
          damping = __lm_maximum_damp;

        EvaluateJacobians();
        ComputeJtE(_cuImageProj, _cuVectorJtE);
      }
    } else {
      if (__verbose_level > 1)
        std::cout << "#" << std::setw(3) << i << " " << std::setw(3)
                  << num_cg_iteration << char(__recent_cg_status)
                  << " e=" << std::setw(edwidth) << std::left
                  << average_residual * error_display_ratio
                  << " u=" << std::setprecision(3) << std::setw(9) << damping
                  << " r=----- " << (__lm_check_gradient || __save_gradient_norm
                                         ? " g=---------"
                                         : " g=0")
                  << " --------- " << std::setw(9) << dx_norm
                  << " t=" << int(BundleTimerGetNow()) << "\n"
                  << std::setprecision(6);

      if (__lm_damping_auto_switch > 0 && __lm_use_diagonal_damp &&
          damping > __lm_damping_auto_switch) {
        __lm_use_diagonal_damp = false;
        damping = __lm_damping_auto_switch;
        damping_adjust = 2.0f;
        if (__verbose_level > 1)
          std::cout << "NOTE: switch to damping with an identity matix\n";
      } else {
        /////////////increase damping factor
        damping = damping * damping_adjust;
        damping_adjust = 2.0f * damping_adjust;
      }
    }

    if (__verbose_level == 1) std::cout << '.';
  }

  __final_mse = float(_projection_sse * mse_convert_ratio);
  __final_mse_x =
      __use_radial_distortion
          ? EvaluateProjectionX(_cuCameraData, _cuPointData, _cuImageProj) *
                mse_convert_ratio
          : __final_mse;
}

#define PROFILE_(A, B)                    \
  BundleTimerStart(TIMER_PROFILE_STEP);   \
  for (int i = 0; i < repeat; ++i) {      \
    B;                                    \
    FinishWorkCUDA();                     \
  }                                       \
  BundleTimerSwitch(TIMER_PROFILE_STEP);  \
  std::cout << std::setw(24) << A << ": " \
            << (BundleTimerGet(TIMER_PROFILE_STEP) / repeat) << "\n";

#define PROFILE(A, B) PROFILE_(#A, A B)
#define PROXILE(A, B) PROFILE_(A, B)

void SparseBundleCU::RunProfileSteps() {
  const int repeat = __profile_pba;
  std::cout << "---------------------------------\n"
               "|    Run profiling steps ("
            << repeat << ")  |\n"
                         "---------------------------------\n"
            << std::left;
  ;

  ///////////////////////////////////////////////
  PROXILE("Upload Measurements",
          _cuMeasurements.CopyFromHost(
              _imgpt_datax.size() > 0 ? &_imgpt_datax[0] : _imgpt_data));
  PROXILE("Upload Point Data", _cuPointData.CopyToHost(_point_data));
  std::cout << "---------------------------------\n";

  /////////////////////////////////////////////
  EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
  PrepareJacobianNormalization();
  EvaluateJacobians();
  ComputeJtE(_cuImageProj, _cuVectorJtE);
  ComputeBlockPC(__lm_initial_damp, true);
  FinishWorkCUDA();

  do {
    if (SolveNormalEquationPCGX(__lm_initial_damp) == 10 &&
        SolveNormalEquationPCGB(__lm_initial_damp) == 10)
      break;
    __lm_initial_damp *= 2.0f;
  } while (__lm_initial_damp < 1024.0f);
  std::cout << "damping set to " << __lm_initial_damp << " for profiling\n"
            << "---------------------------------\n";

  {
    int repeat = 10, cgmin = __cg_min_iteration, cgmax = __cg_max_iteration;
    __cg_max_iteration = __cg_min_iteration = 10;
    __num_cg_iteration = 0;
    PROFILE(SolveNormalEquationPCGX, (__lm_initial_damp));
    if (__num_cg_iteration != 100)
      std::cout << __num_cg_iteration << " cg iterations in all\n";

    /////////////////////////////////////////////////////////////////////
    __num_cg_iteration = 0;
    PROFILE(SolveNormalEquationPCGB, (__lm_initial_damp));
    if (__num_cg_iteration != 100)
      std::cout << __num_cg_iteration << " cg iterations in all\n";
    std::cout << "---------------------------------\n";
    //////////////////////////////////////////////////////
    __num_cg_iteration = 0;
    PROXILE("Single iteration LMX", RunTestIterationLM(true));
    if (__num_cg_iteration != 100)
      std::cout << __num_cg_iteration << " cg iterations in all\n";
    ////////////////////////////////////////////////////////
    __num_cg_iteration = 0;
    PROXILE("Single iteration LMB", RunTestIterationLM(false));
    if (__num_cg_iteration != 100)
      std::cout << __num_cg_iteration << " cg iterations in all\n";
    std::cout << "---------------------------------\n";
    __cg_max_iteration = cgmax;
    __cg_min_iteration = cgmin;
  }
  /////////////////////////////////////////////////////
  PROFILE(UpdateCameraPoint, (_cuVectorZK, _cuImageProj));
  PROFILE(ComputeVectorNorm, (_cuVectorXK, _cuBufferData));
  PROFILE(ComputeVectorDot, (_cuVectorXK, _cuVectorRK, _cuBufferData));
  PROFILE(ComputeVectorNormW, (_cuVectorXK, _cuVectorRK, _cuBufferData));
  PROFILE(ComputeSAXPY, (0.01f, _cuVectorXK, _cuVectorRK, _cuVectorZK));
  PROFILE(ComputeSXYPZ,
          (0.01f, _cuVectorXK, _cuVectorPK, _cuVectorRK, _cuVectorZK));
  std::cout << "---------------------------------\n";
  PROFILE(ComputeVectorNorm, (_cuImageProj, _cuBufferData));
  PROFILE(ComputeSAXPY, (0.000f, _cuImageProj, _cuVectorJX, _cuVectorJX));
  std::cout << "---------------------------------\n";

  __multiply_jx_usenoj = false;
  ///////////////////////////////////////////////////////
  PROFILE(EvaluateProjection, (_cuCameraData, _cuPointData, _cuImageProj));
  PROFILE(ApplyBlockPC, (_cuVectorJtE, _cuVectorPK));
  /////////////////////////////////////////////////
  if (!__no_jacobian_store) {
    if (__jc_store_original) {
      PROFILE(ComputeJX, (_cuVectorJtE, _cuVectorJX));
      PROFILE(EvaluateJacobians, (false));

      if (__jc_store_transpose) {
        PROFILE(
            ShuffleCameraJacobian,
            (_cuJacobianCamera, _cuCameraMeasurementList, _cuJacobianCameraT));
        PROFILE(ComputeDiagonal, (_cuVectorJJ, _cuVectorPK));
        PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
        PROFILE(ComputeBlockPC, (0.001f, true));

        std::cout << "---------------------------------\n"
                     "|   Not storing original  JC    | \n"
                     "---------------------------------\n";
        __jc_store_original = false;
        PROFILE(EvaluateJacobians, ());
        __jc_store_original = true;
      }
      //////////////////////////////////////////////////

      std::cout << "---------------------------------\n"
                   "|   Not storing transpose JC    | \n"
                   "---------------------------------\n";
      __jc_store_transpose = false;
      PROFILE(ComputeDiagonal, (_cuVectorJJ, _cuVectorPK));
      PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
      PROFILE(ComputeBlockPC, (0.001f, true));

      //////////////////////////////////////

    } else if (__jc_store_transpose) {
      PROFILE(ComputeDiagonal, (_cuVectorJJ, _cuVectorPK));
      PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
      PROFILE(ComputeBlockPC, (0.001f, true));
      std::cout << "---------------------------------\n"
                   "|   Not storing original  JC    | \n"
                   "---------------------------------\n";
      PROFILE(EvaluateJacobians, ());
    }
  }

  if (!__no_jacobian_store) {
    std::cout << "---------------------------------\n"
                 "| Not storing Camera Jacobians  | \n"
                 "---------------------------------\n";
    __jc_store_transpose = false;
    __jc_store_original = false;
    _cuJacobianCamera.ReleaseData();
    _cuJacobianCameraT.ReleaseData();
    PROFILE(EvaluateJacobians, ());
    PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
    PROFILE(ComputeBlockPC, (0.001f, true));
  }

  ///////////////////////////////////////////////

  std::cout << "---------------------------------\n"
               "|   Not storing any jacobians   |\n"
               "---------------------------------\n";
  __no_jacobian_store = true;
  _cuJacobianPoint.ReleaseData();
  PROFILE(ComputeJX, (_cuVectorJtE, _cuVectorJX));
  PROFILE(ComputeJtE, (_cuImageProj, _cuVectorJtE));
  PROFILE(ComputeBlockPC, (0.001f, true));

  std::cout << "---------------------------------\n";
}

void SparseBundleCU::RunDebugSteps() {
  EvaluateProjection(_cuCameraData, _cuPointData, _cuImageProj);
  EvaluateJacobians();
  ComputeJtE(_cuImageProj, _cuVectorJtE);
  // DEBUG_FUNCN(_cuVectorXK, SolveNormalEquationPCGB, (0.001f), 100);
  DEBUG_FUNCN(_cuVectorJtE, ComputeJtE, (_cuImageProj, _cuVectorJtE), 100);
  DEBUG_FUNCN(_cuVectorJX, ComputeJX, (_cuVectorJtE, _cuVectorJX), 100);
}

void SparseBundleCU::SaveNormalEquation(float lambda) {
  ofstream out1("../../matlab/cg_j.txt");
  ofstream out2("../../matlab/cg_b.txt");
  ofstream out3("../../matlab/cg_x.txt");

  out1 << std::setprecision(20);
  out2 << std::setprecision(20);
  out3 << std::setprecision(20);

  int plen = GetParameterLength();
  vector<float> jc(16 * _num_imgpt);
  vector<float> jp(8 * _num_imgpt);
  vector<float> ee(2 * _num_imgpt);
  vector<float> dx(plen);

  _cuJacobianCamera.CopyToHost(&jc[0]);
  _cuJacobianPoint.CopyToHost(&jp[0]);
  _cuImageProj.CopyToHost(&ee[0]);
  _cuVectorXK.CopyToHost(&dx[0]);

  for (int i = 0; i < _num_imgpt; ++i) {
    out2 << ee[i * 2] << ' ' << ee[i * 2 + 1] << ' ';
    int cidx = _camera_idx[i], pidx = _point_idx[i];
    float *cp = &jc[i * 16], *pp = &jp[i * 8];
    int cmin = cidx * 8, pmin = 8 * _num_camera + pidx * 4;
    for (int j = 0; j < 8; ++j)
      out1 << (i * 2 + 1) << ' ' << (cmin + j + 1) << ' ' << cp[j] << '\n';
    for (int j = 0; j < 8; ++j)
      out1 << (i * 2 + 2) << ' ' << (cmin + j + 1) << ' ' << cp[j + 8] << '\n';
    for (int j = 0; j < 4; ++j)
      out1 << (i * 2 + 1) << ' ' << (pmin + j + 1) << ' ' << pp[j] << '\n';
    for (int j = 0; j < 4; ++j)
      out1 << (i * 2 + 2) << ' ' << (pmin + j + 1) << ' ' << pp[j + 4] << '\n';
  }

  for (size_t i = 0; i < dx.size(); ++i) out3 << dx[i] << ' ';

  std::cout << "lambda = " << std::setprecision(20) << lambda << '\n';
}

}  // namespace pba
