////////////////////////////////////////////////////////////////////////////
//  File:           ProgramCU.h
//  Author:         Changchang Wu
//  Description :   interface for the ProgramCU classes.
//                  It is basically a wrapper around all the CUDA kernels
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

#ifndef _PROGRAM_CU_H
#define _PROGRAM_CU_H

class CuTexImage;

namespace pba {
namespace ProgramCU {

int SetCudaDevice(int device);
size_t GetCudaMemoryCap();
int CheckErrorCUDA(const char* location);
void FinishWorkCUDA();
void ClearPreviousError();
void ResetCurrentDevice();
void GetBlockConfiguration(unsigned int nblock, unsigned int& bw,
                           unsigned int& bh);

//////////////////////////////////////////////////////////
void ComputeSQRT(CuTexImage& tex);
void ComputeRSQRT(CuTexImage& tex);
void ComputeVXY(CuTexImage& texX, CuTexImage& texY, CuTexImage& result,
                unsigned int part = 0, unsigned int skip = 0);
void ComputeSAXPY(float a, CuTexImage& texX, CuTexImage& texY,
                  CuTexImage& result);
void ComputeSAX(float a, CuTexImage& texX, CuTexImage& result);
void ComputeSXYPZ(float a, CuTexImage& texX, CuTexImage& texY, CuTexImage& texZ,
                  CuTexImage& result);
float ComputeVectorMax(CuTexImage& vector, CuTexImage& buf);
float ComputeVectorSum(CuTexImage& vector, CuTexImage& buf, int skip);
double ComputeVectorNorm(CuTexImage& vector, CuTexImage& buf);
double ComputeVectorNormW(CuTexImage& vector, CuTexImage& weight,
                          CuTexImage& buf);
double ComputeVectorDot(CuTexImage& vector1, CuTexImage& vector2,
                        CuTexImage& buf);

//////////////////////////////////////////////////////////////////////////
void UncompressCamera(int ncam, CuTexImage& camera0, CuTexImage& result);
void CompressCamera(int ncam, CuTexImage& camera0, CuTexImage& result);
void UpdateCameraPoint(int ncam, CuTexImage& camera, CuTexImage& point,
                       CuTexImage& delta, CuTexImage& new_camera,
                       CuTexImage& new_point, int mode = 0);

/////////////////////////////////////////////////////////////////////////
void ComputeJacobian(CuTexImage& camera, CuTexImage& point, CuTexImage& jc,
                     CuTexImage& jp, CuTexImage& proj_map, CuTexImage& sj,
                     CuTexImage& meas, CuTexImage& cmlist, bool intrinsic_fixed,
                     int radial_distortion, bool shuffle);
void ComputeProjection(CuTexImage& camera, CuTexImage& point, CuTexImage& meas,
                       CuTexImage& proj_map, CuTexImage& proj, int radial);
void ComputeProjectionX(CuTexImage& camera, CuTexImage& point, CuTexImage& meas,
                        CuTexImage& proj_map, CuTexImage& proj, int radial);

bool ShuffleCameraJacobian(CuTexImage& jc, CuTexImage& map, CuTexImage& result);

/////////////////////////////////////////////////////////////
void ComputeDiagonal(CuTexImage& jc, CuTexImage& cmap, CuTexImage& jp,
                     CuTexImage& pmap, CuTexImage& cmlist, CuTexImage& jtjd,
                     CuTexImage& jtjdi, bool jc_transpose, int radial,
                     bool add_existing_diagc);
void MultiplyBlockConditioner(int ncam, int npoint, CuTexImage& blocks,
                              CuTexImage& vector, CuTexImage& result,
                              int radial, int mode = 0);

////////////////////////////////////////////////////////////////////////////////
void ComputeProjectionQ(CuTexImage& camera, CuTexImage& qmap, CuTexImage& qw,
                        CuTexImage& proj, int offset);
void ComputeJQX(CuTexImage& x, CuTexImage& qmap, CuTexImage& wq, CuTexImage& sj,
                CuTexImage& jx, int offset);
void ComputeJQtEC(CuTexImage& pe, CuTexImage& qlist, CuTexImage& wq,
                  CuTexImage& sj, CuTexImage& result);
void ComputeDiagonalQ(CuTexImage& qlistw, CuTexImage& sj, CuTexImage& diag);

//////////////////////////////////////////////////////////////////////////
void ComputeJX(int point_offset, CuTexImage& x, CuTexImage& jc, CuTexImage& jp,
               CuTexImage& jmap, CuTexImage& result, int mode = 0);
void ComputeJtE(CuTexImage& pe, CuTexImage& jc, CuTexImage& cmap,
                CuTexImage& cmlist, CuTexImage& jp, CuTexImage& pmap,
                CuTexImage& jte, bool jc_transpose, int mode = 0);
void ComputeDiagonalBlock(float lambda, bool dampd, CuTexImage& jc,
                          CuTexImage& cmap, CuTexImage& jp, CuTexImage& pmap,
                          CuTexImage& cmlist, CuTexImage& diag,
                          CuTexImage& blocks, int radial_distortion,
                          bool jc_transpose, bool add_existing_diagc,
                          int mode = 0);

/////////////////////////////////////////////////////////////////////
void ComputeJX_(CuTexImage& x, CuTexImage& jx, CuTexImage& camera,
                CuTexImage& point, CuTexImage& meas, CuTexImage& pjmap,
                bool intrinsic_fixed, int radial_distortion, int mode = 0);
void ComputeJtE_(CuTexImage& e, CuTexImage& jte, CuTexImage& camera,
                 CuTexImage& point, CuTexImage& meas, CuTexImage& cmap,
                 CuTexImage& cmlist, CuTexImage& pmap, CuTexImage& jmap,
                 CuTexImage& jp, bool intrinsic_fixed, int radial_distortion,
                 int mode = 0);
void ComputeDiagonalBlock_(float lambda, bool dampd, CuTexImage& camera,
                           CuTexImage& point, CuTexImage& meas,
                           CuTexImage& cmap, CuTexImage& cmlist,
                           CuTexImage& pmap, CuTexImage& jmap, CuTexImage& jp,
                           CuTexImage& sj, CuTexImage& diag, CuTexImage& blocks,
                           bool intrinsic_fixed, int radial_distortion,
                           bool add_existing_diagc, int mode = 0);

}  // namespace ProgramCU
}  // namespace pba

#endif
