////////////////////////////////////////////////////////////////////////////
//  File:       SparseBundleCU.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   interface of the CUDA-version of multicore bundle
// adjustment
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

#if !defined(SPARSE_BUNDLE_CU_H)
#define SPARSE_BUNDLE_CU_H

#include "ConfigBA.h"
#include "CuTexImage.h"
#include "DataInterface.h"

namespace pba {

class SparseBundleCU : public ParallelBA, public ConfigBA {
 protected:  // cpu data
  int _num_camera;
  int _num_point;
  int _num_imgpt;
  CameraT* _camera_data;
  float* _point_data;
  ////////////////////////////////
  const float* _imgpt_data;
  const int* _camera_idx;
  const int* _point_idx;
  const int* _focal_mask;
  std::vector<float> _imgpt_datax;
  ////////////////////////
  float _projection_sse;  // sumed square error
 protected:               // cuda data
  CuTexImage _cuCameraData;
  CuTexImage _cuCameraDataEX;
  CuTexImage _cuPointData;
  CuTexImage _cuPointDataEX;
  CuTexImage _cuMeasurements;
  CuTexImage _cuImageProj;
  CuTexImage _cuJacobianCamera;
  CuTexImage _cuJacobianPoint;
  CuTexImage _cuJacobianCameraT;
  CuTexImage _cuProjectionMap;
  CuTexImage _cuPointMeasurementMap;
  CuTexImage _cuCameraMeasurementMap;
  CuTexImage _cuCameraMeasurementList;
  CuTexImage _cuCameraMeasurementListT;

  ///////////////////////////////
  CuTexImage _cuBufferData;
  ////////////////////////////
  CuTexImage _cuBlockPC;
  CuTexImage _cuVectorSJ;

  /// LM normal    equation
  CuTexImage _cuVectorJtE;
  CuTexImage _cuVectorJJ;
  CuTexImage _cuVectorJX;
  CuTexImage _cuVectorXK;
  CuTexImage _cuVectorPK;
  CuTexImage _cuVectorZK;
  CuTexImage _cuVectorRK;

  ///////////////////////
 protected:
  int _num_imgpt_q;
  float _weight_q;
  CuTexImage _cuCameraQList;
  CuTexImage _cuCameraQMap;
  CuTexImage _cuCameraQMapW;
  CuTexImage _cuCameraQListW;

 protected:
  bool ProcessIndexCameraQ(std::vector<int>& qmap, std::vector<int>& qlist);
  void ProcessWeightCameraQ(std::vector<int>& cpnum, std::vector<int>& qmap,
                            std::vector<float>& qmapw,
                            std::vector<float>& qlistw);

 protected:  // internal functions
  int GetParameterLength();
  int InitializeBundle();
  int ValidateInputData();
  void ReleaseAllocatedData();
  bool InitializeStorageForCG();
  bool InitializeBundleGPU();
  bool TransferDataToGPU();
  void TransferDataToHost();
  void DenormalizeData();
  void NormalizeData();
  void NormalizeDataF();
  void NormalizeDataD();
  void DebugProjections();
  void RunDebugSteps();
  bool CheckRequiredMem(int fresh = 1);
  bool CheckRequiredMemX();
  void ReserveStorage(size_t ncam, size_t npt, size_t nproj);
  void ReserveStorageAuto();

 protected:
  float EvaluateProjection(CuTexImage& cam, CuTexImage& point,
                           CuTexImage& proj);
  float EvaluateProjectionX(CuTexImage& cam, CuTexImage& point,
                            CuTexImage& proj);
  float UpdateCameraPoint(CuTexImage& dx, CuTexImage& cuImageTempProj);
  float SaveUpdatedSystem(float residual_reduction, float dx_sqnorm,
                          float damping);
  float EvaluateDeltaNorm();
  void EvaluateJacobians(bool shuffle = true);
  void PrepareJacobianNormalization();
  void ComputeJtE(CuTexImage& E, CuTexImage& JtE, int mode = 0);
  void ComputeJX(CuTexImage& X, CuTexImage& JX, int mode = 0);
  void ComputeDiagonal(CuTexImage& JJ, CuTexImage& JJI);
  void ComputeBlockPC(float lambda, bool dampd = true);
  void ApplyBlockPC(CuTexImage& v, CuTexImage& pv, int mode = 0);
  int SolveNormalEquationPCGB(float lambda);
  int SolveNormalEquationPCGX(float lambda);
  int SolveNormalEquation(float lambda);
  void AdjustBundleAdjsutmentMode();
  void NonlinearOptimizeLM();
  void BundleAdjustment();
  void RunTestIterationLM(bool reduced);
  void SaveBundleRecord(int iter, float res, float damping, float& g_norm,
                        float& g_inf);
  /////////////////////////////////
  void SaveNormalEquation(float lambda);
  void RunProfileSteps();
  void WarmupDevice();

 public:
  virtual float GetMeanSquaredError();
  virtual void SetCameraData(size_t ncam, CameraT* cams);
  virtual void SetPointData(size_t npoint, Point3D* pts);
  virtual void SetProjection(size_t nproj, const Point2D* imgpts,
                             const int* point_idx, const int* cam_idx);
  virtual void SetFocalMask(const int* fmask, float weight);
  virtual int RunBundleAdjustment();

  ///
  virtual void AbortBundleAdjustment() { __abort_flag = true; }
  virtual int GetCurrentIteration() { return __current_iteration; }
  virtual void SetNextTimeBudget(int seconds) {
    __bundle_time_budget = seconds;
  }
  virtual void SetNextBundleMode(BundleModeT mode) {
    __bundle_mode_next = mode;
  }
  virtual void SetFixedIntrinsics(bool fixed) { __fixed_intrinsics = fixed; }
  virtual void EnableRadialDistortion(DistortionT type) {
    __use_radial_distortion = type;
  }
  virtual void ParseParam(int narg, char** argv) {
    ConfigBA::ParseParam(narg, argv);
  }
  virtual ConfigBA* GetInternalConfig() { return this; }

 public:
  SparseBundleCU(int device);
  size_t GetMemCapacity();
};

}  // namespace pba

#endif
