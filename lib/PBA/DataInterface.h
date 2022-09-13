////////////////////////////////////////////////////////////////////////////
//  File:       DataInterface.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   data interface, the data format been uploaded to GPU
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

#ifndef DATA_INTERFACE_GPU_H
#define DATA_INTERFACE_GPU_H

#include <math.h>

// ----------------------------WARNING------------------------------
// -----------------------------------------------------------------
// ROTATION CONVERSION:
// The internal rotation representation is 3x3 float matrix. Reading
// back the rotations as quaternion or Rodrigues's representation will
// cause inaccuracy, IF you have wrongly reconstructed cameras with
// a very very large focal length (typically also very far away).
// In this case, any small change in the rotation matrix, will cause
// a large reprojection error.
//
// ---------------------------------------------------------------------
// RADIAL distortion is NOT enabled by default, use parameter "-md", -pd"
// or set ConfigBA::__use_radial_distortion to 1 or -1 to enable it.
// ---------------------------------------------------------------------------

namespace pba {

// transfer data type with 4-float alignment
#define CameraT CameraT_
#define Point3D Point3D_
template <class FT>

struct CameraT_ {
  typedef FT float_t;
  //////////////////////////////////////////////////////
  float_t f;        // single focal length, K = [f, 0, 0; 0 f 0; 0 0 1]
  float_t t[3];     // T in  P = K[R T], T = - RC
  float_t m[3][3];  // R in  P = K[R T].
  float_t radial;   // WARNING: BE careful with the radial distortion model.
  int distortion_type;
  float_t constant_camera;

  //////////////////////////////////////////////////////////
  CameraT_() {
    radial = 0;
    distortion_type = 0;
    constant_camera = 0;
  }

  //////////////////////////////////////////////
  template <class CameraX>
  void SetCameraT(const CameraX& cam) {
    f = (float_t)cam.f;
    t[0] = (float_t)cam.t[0];
    t[1] = (float_t)cam.t[1];
    t[2] = (float_t)cam.t[2];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) m[i][j] = (float_t)cam.m[i][j];
    radial = (float_t)cam.radial;
    distortion_type = cam.distortion_type;
    constant_camera = cam.constant_camera;
  }

  //////////////////////////////////////////
  void SetConstantCamera() { constant_camera = 1.0f; }
  void SetVariableCamera() { constant_camera = 0.0f; }
  void SetFixedIntrinsic() { constant_camera = 2.0f; }
  // void SetFixedExtrinsic() {constant_camera = 3.0f;}

  //////////////////////////////////////
  template <class Float>
  void SetFocalLength(Float F) {
    f = (float_t)F;
  }
  float_t GetFocalLength() const { return f; }

  template <class Float>
  void SetMeasurementDistortion(Float r) {
    radial = (float_t)r;
    distortion_type = -1;
  }
  float_t GetMeasurementDistortion() const {
    return distortion_type == -1 ? radial : 0;
  }

  // normalize radial distortion that applies to angle will be (radial * f * f);
  template <class Float>
  void SetNormalizedMeasurementDistortion(Float r) {
    SetMeasurementDistortion(r / (f * f));
  }
  float_t GetNormalizedMeasurementDistortion() const {
    return GetMeasurementDistortion() * (f * f);
  }

  // use projection distortion
  template <class Float>
  void SetProjectionDistortion(Float r) {
    radial = float_t(r);
    distortion_type = 1;
  }
  template <class Float>
  void SetProjectionDistortion(const Float* r) {
    SetProjectionDistortion(r[0]);
  }
  float_t GetProjectionDistortion() const {
    return distortion_type == 1 ? radial : 0;
  }

  template <class Float>
  void SetRodriguesRotation(const Float r[3]) {
    double a = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    double ct = a == 0.0 ? 0.5 : (1.0 - cos(a)) / a / a;
    double st = a == 0.0 ? 1 : sin(a) / a;
    m[0][0] = float_t(1.0 - (r[1] * r[1] + r[2] * r[2]) * ct);
    m[0][1] = float_t(r[0] * r[1] * ct - r[2] * st);
    m[0][2] = float_t(r[2] * r[0] * ct + r[1] * st);
    m[1][0] = float_t(r[0] * r[1] * ct + r[2] * st);
    m[1][1] = float_t(1.0 - (r[2] * r[2] + r[0] * r[0]) * ct);
    m[1][2] = float_t(r[1] * r[2] * ct - r[0] * st);
    m[2][0] = float_t(r[2] * r[0] * ct - r[1] * st);
    m[2][1] = float_t(r[1] * r[2] * ct + r[0] * st);
    m[2][2] = float_t(1.0 - (r[0] * r[0] + r[1] * r[1]) * ct);
  }
  template <class Float>
  void GetRodriguesRotation(Float r[3]) const {
    double a = (m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0;
    const double epsilon = 0.01;
    if (fabs(m[0][1] - m[1][0]) < epsilon &&
        fabs(m[1][2] - m[2][1]) < epsilon &&
        fabs(m[0][2] - m[2][0]) < epsilon) {
      if (fabs(m[0][1] + m[1][0]) < 0.1 && fabs(m[1][2] + m[2][1]) < 0.1 &&
          fabs(m[0][2] + m[2][0]) < 0.1 && a > 0.9) {
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
      } else {
        const Float ha = Float(sqrt(0.5) * 3.14159265358979323846);
        double xx = (m[0][0] + 1.0) / 2.0;
        double yy = (m[1][1] + 1.0) / 2.0;
        double zz = (m[2][2] + 1.0) / 2.0;
        double xy = (m[0][1] + m[1][0]) / 4.0;
        double xz = (m[0][2] + m[2][0]) / 4.0;
        double yz = (m[1][2] + m[2][1]) / 4.0;

        if ((xx > yy) && (xx > zz)) {
          if (xx < epsilon) {
            r[0] = 0;
            r[1] = r[2] = ha;
          } else {
            double t = sqrt(xx);
            r[0] = Float(t * 3.14159265358979323846);
            r[1] = Float(xy / t * 3.14159265358979323846);
            r[2] = Float(xz / t * 3.14159265358979323846);
          }
        } else if (yy > zz) {
          if (yy < epsilon) {
            r[0] = r[2] = ha;
            r[1] = 0;
          } else {
            double t = sqrt(yy);
            r[0] = Float(xy / t * 3.14159265358979323846);
            r[1] = Float(t * 3.14159265358979323846);
            r[2] = Float(yz / t * 3.14159265358979323846);
          }
        } else {
          if (zz < epsilon) {
            r[0] = r[1] = ha;
            r[2] = 0;
          } else {
            double t = sqrt(zz);
            r[0] = Float(xz / t * 3.14159265358979323846);
            r[1] = Float(yz / t * 3.14159265358979323846);
            r[2] = Float(t * 3.14159265358979323846);
          }
        }
      }
    } else {
      a = acos(a);
      double b = 0.5 * a / sin(a);
      r[0] = Float(b * (m[2][1] - m[1][2]));
      r[1] = Float(b * (m[0][2] - m[2][0]));
      r[2] = Float(b * (m[1][0] - m[0][1]));
    }
  }
  ////////////////////////
  template <class Float>
  void SetQuaternionRotation(const Float q[4]) {
    double qq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double qw, qx, qy, qz;
    if (qq > 0) {
      qw = q[0] / qq;
      qx = q[1] / qq;
      qy = q[2] / qq;
      qz = q[3] / qq;
    } else {
      qw = 1;
      qx = qy = qz = 0;
    }
    m[0][0] = float_t(qw * qw + qx * qx - qz * qz - qy * qy);
    m[0][1] = float_t(2 * qx * qy - 2 * qz * qw);
    m[0][2] = float_t(2 * qy * qw + 2 * qz * qx);
    m[1][0] = float_t(2 * qx * qy + 2 * qw * qz);
    m[1][1] = float_t(qy * qy + qw * qw - qz * qz - qx * qx);
    m[1][2] = float_t(2 * qz * qy - 2 * qx * qw);
    m[2][0] = float_t(2 * qx * qz - 2 * qy * qw);
    m[2][1] = float_t(2 * qy * qz + 2 * qw * qx);
    m[2][2] = float_t(qz * qz + qw * qw - qy * qy - qx * qx);
  }
  template <class Float>
  void GetQuaternionRotation(Float q[4]) const {
    q[0] = 1 + m[0][0] + m[1][1] + m[2][2];
    if (q[0] > 0.000000001) {
      q[0] = sqrt(q[0]) / 2.0;
      q[1] = (m[2][1] - m[1][2]) / (4.0 * q[0]);
      q[2] = (m[0][2] - m[2][0]) / (4.0 * q[0]);
      q[3] = (m[1][0] - m[0][1]) / (4.0 * q[0]);
    } else {
      double s;
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        q[1] = 0.25 * s;
        q[2] = (m[0][1] + m[1][0]) / s;
        q[3] = (m[0][2] + m[2][0]) / s;
        q[0] = (m[1][2] - m[2][1]) / s;
      } else if (m[1][1] > m[2][2]) {
        s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        q[1] = (m[0][1] + m[1][0]) / s;
        q[2] = 0.25 * s;
        q[3] = (m[1][2] + m[2][1]) / s;
        q[0] = (m[0][2] - m[2][0]) / s;
      } else {
        s = 2.0 * sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        q[1] = (m[0][2] + m[2][0]) / s;
        q[2] = (m[1][2] + m[2][1]) / s;
        q[3] = 0.25f * s;
        q[0] = (m[0][1] - m[1][0]) / s;
      }
    }
  }
  ////////////////////////////////////////////////
  template <class Float>
  void SetMatrixRotation(const Float* r) {
    int k = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m[i][j] = float_t(r[k++]);
      }
    }
  }
  template <class Float>
  void GetMatrixRotation(Float* r) const {
    int k = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r[k++] = Float(m[i][j]);
      }
    }
  }
  float GetRotationMatrixDeterminant() const {
    return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
           m[0][2] * m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0] -
           m[0][1] * m[1][0] * m[2][2] - m[0][0] * m[1][2] * m[2][1];
  }
  ///////////////////////////////////////
  template <class Float>
  void SetTranslation(const Float T[3]) {
    t[0] = (float_t)T[0];
    t[1] = (float_t)T[1];
    t[2] = (float_t)T[2];
  }
  template <class Float>
  void GetTranslation(Float T[3]) const {
    T[0] = (Float)t[0];
    T[1] = (Float)t[1];
    T[2] = (Float)t[2];
  }
  /////////////////////////////////////////////
  template <class Float>
  void SetCameraCenterAfterRotation(const Float c[3]) {
    // t = - R * C
    for (int j = 0; j < 3; ++j)
      t[j] = -float_t(m[j][0] * c[0] + m[j][1] * c[1] + m[j][2] * c[2]);
  }
  template <class Float>
  void GetCameraCenter(Float c[3]) {
    // C = - R' * t
    for (int j = 0; j < 3; ++j)
      c[j] = -float_t(m[0][j] * t[0] + m[1][j] * t[1] + m[2][j] * t[2]);
  }
  ////////////////////////////////////////////
  template <class Float>
  void SetInvertedRT(const Float e[3], const Float T[3]) {
    SetRodriguesRotation(e);
    for (int i = 3; i < 9; ++i) m[0][i] = -m[0][i];
    SetTranslation(T);
    t[1] = -t[1];
    t[2] = -t[2];
  }

  template <class Float>
  void GetInvertedRT(Float e[3], Float T[3]) const {
    CameraT ci;
    ci.SetMatrixRotation(m[0]);
    for (int i = 3; i < 9; ++i) ci.m[0][i] = -ci.m[0][i];
    // for(int i = 1; i < 3; ++i) for(int j = 0; j < 3; ++j) ci.m[i][j] = -
    // ci.m[i][j];
    ci.GetRodriguesRotation(e);
    GetTranslation(T);
    T[1] = -T[1];
    T[2] = -T[2];
  }
  template <class Float>
  void SetInvertedR9T(const Float e[9], const Float T[3]) {
    // for(int i = 0; i < 9; ++i) m[0][i] = (i < 3 ? e[i] : - e[i]);
    // SetTranslation(T); t[1] = - t[1]; t[2] = -t[2];
    m[0][0] = e[0];
    m[0][1] = e[1];
    m[0][2] = e[2];
    m[1][0] = -e[3];
    m[1][1] = -e[4];
    m[1][2] = -e[5];
    m[2][0] = -e[6];
    m[2][1] = -e[7];
    m[2][2] = -e[8];
    t[0] = T[0];
    t[1] = -T[1];
    t[2] = -T[2];
  }
  template <class Float>
  void GetInvertedR9T(Float e[9], Float T[3]) const {
    e[0] = m[0][0];
    e[1] = m[0][1];
    e[2] = m[0][2];
    e[3] = -m[1][0];
    e[4] = -m[1][1];
    e[5] = -m[1][2];
    e[6] = -m[2][0];
    e[7] = -m[2][1];
    e[8] = -m[2][2];
    T[0] = t[0];
    T[1] = -t[1];
    T[2] = -t[2];
  }
};

template <class FT>
struct Point3D {
  typedef FT float_t;
  float_t xyz[3];  // 3D point location
  float_t reserved;  // alignment
  ////////////////////////////////
  template <class Float>
  void SetPoint(Float x, Float y, Float z) {
    xyz[0] = (float_t)x;
    xyz[1] = (float_t)y;
    xyz[2] = (float_t)z;
    reserved = 0;
  }
  template <class Float>
  void SetPoint(const Float* p) {
    xyz[0] = (float_t)p[0];
    xyz[1] = (float_t)p[1];
    xyz[2] = (float_t)p[2];
    reserved = 0;
  }
  template <class Float>
  void GetPoint(Float* p) const {
    p[0] = (Float)xyz[0];
    p[1] = (Float)xyz[1];
    p[2] = (Float)xyz[2];
  }
  template <class Float>
  void GetPoint(Float& x, Float& y, Float& z) const {
    x = (Float)xyz[0];
    y = (Float)xyz[1];
    z = (Float)xyz[2];
  }
};

#undef CameraT
#undef Point3D

typedef CameraT_<float> CameraT;
typedef Point3D_<float> Point3D;

struct Point2D {
  float x, y;
  ////////////////////////////////////////////////////////
  Point2D() {}
  template <class Float>
  Point2D(Float X, Float Y) {
    SetPoint2D(X, Y);
  }
  template <class Float>
  void SetPoint2D(Float X, Float Y) {
    x = (float)X;
    y = (float)Y;
  }
  template <class Float>
  void GetPoint2D(Float& X, Float& Y) const {
    X = (Float)x;
    Y = (Float)y;
  }
};

}  // namespace pba

#endif
