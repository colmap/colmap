////////////////////////////////////////////////////////////////////////////
//  File:       util.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   some utility functions for reading/writing SfM data
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
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <algorithm>
using namespace std;
#include "DataInterface.h"

namespace pba {

// File loader supports .nvm format and bundler format
bool LoadModelFile(const char* name, vector<CameraT>& camera_data,
                   vector<Point3D>& point_data, vector<Point2D>& measurements,
                   vector<int>& ptidx, vector<int>& camidx,
                   vector<string>& names, vector<int>& ptc);
void SaveNVM(const char* filename, vector<CameraT>& camera_data,
             vector<Point3D>& point_data, vector<Point2D>& measurements,
             vector<int>& ptidx, vector<int>& camidx, vector<string>& names,
             vector<int>& ptc);
void SaveBundlerModel(const char* filename, vector<CameraT>& camera_data,
                      vector<Point3D>& point_data,
                      vector<Point2D>& measurements, vector<int>& ptidx,
                      vector<int>& camidx);

//////////////////////////////////////////////////////////////////
void AddNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data,
              float percent);
void AddStableNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data,
                    const vector<int>& ptidx, const vector<int>& camidx,
                    float percent);
bool RemoveInvisiblePoints(vector<CameraT>& camera_data,
                           vector<Point3D>& point_data, vector<int>& ptidx,
                           vector<int>& camidx, vector<Point2D>& measurements,
                           vector<string>& names, vector<int>& ptc);

/////////////////////////////////////////////////////////////////////////////
bool LoadNVM(ifstream& in, vector<CameraT>& camera_data,
             vector<Point3D>& point_data, vector<Point2D>& measurements,
             vector<int>& ptidx, vector<int>& camidx, vector<string>& names,
             vector<int>& ptc) {
  int rotation_parameter_num = 4;
  bool format_r9t = false;
  string token;
  if (in.peek() == 'N') {
    in >> token;  // file header
    if (strstr(token.c_str(), "R9T")) {
      rotation_parameter_num = 9;  // rotation as 3x3 matrix
      format_r9t = true;
    }
  }

  int ncam = 0, npoint = 0, nproj = 0;
  // read # of cameras
  in >> ncam;
  if (ncam <= 1) return false;

  // read the camera parameters
  camera_data.resize(ncam);  // allocate the camera data
  names.resize(ncam);
  for (int i = 0; i < ncam; ++i) {
    double f, q[9], c[3], d[2];
    in >> token >> f;
    for (int j = 0; j < rotation_parameter_num; ++j) in >> q[j];
    in >> c[0] >> c[1] >> c[2] >> d[0] >> d[1];

    camera_data[i].SetFocalLength(f);
    if (format_r9t) {
      camera_data[i].SetMatrixRotation(q);
      camera_data[i].SetTranslation(c);
    } else {
      // older format for compability
      camera_data[i].SetQuaternionRotation(q);  // quaternion from the file
      camera_data[i].SetCameraCenterAfterRotation(
          c);  // camera center from the file
    }
    camera_data[i].SetNormalizedMeasurementDistortion(d[0]);
    names[i] = token;
  }

  //////////////////////////////////////
  in >> npoint;
  if (npoint <= 0) return false;

  // read image projections and 3D points.
  point_data.resize(npoint);
  for (int i = 0; i < npoint; ++i) {
    float pt[3];
    int cc[3], npj;
    in >> pt[0] >> pt[1] >> pt[2] >> cc[0] >> cc[1] >> cc[2] >> npj;
    for (int j = 0; j < npj; ++j) {
      int cidx, fidx;
      float imx, imy;
      in >> cidx >> fidx >> imx >> imy;

      camidx.push_back(cidx);  // camera index
      ptidx.push_back(i);  // point index

      // add a measurment to the vector
      measurements.push_back(Point2D(imx, imy));
      nproj++;
    }
    point_data[i].SetPoint(pt);
    ptc.insert(ptc.end(), cc, cc + 3);
  }
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << ncam << " cameras; " << npoint << " 3D points; " << nproj
            << " projections\n";

  return true;
}

void SaveNVM(const char* filename, vector<CameraT>& camera_data,
             vector<Point3D>& point_data, vector<Point2D>& measurements,
             vector<int>& ptidx, vector<int>& camidx, vector<string>& names,
             vector<int>& ptc) {
  std::cout << "Saving model to " << filename << "...\n";
  ofstream out(filename);

  out << "NVM_V3_R9T\n" << camera_data.size() << '\n' << std::setprecision(12);
  if (names.size() < camera_data.size())
    names.resize(camera_data.size(), string("unknown"));
  if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

  ////////////////////////////////////
  for (size_t i = 0; i < camera_data.size(); ++i) {
    CameraT& cam = camera_data[i];
    out << names[i] << ' ' << cam.GetFocalLength() << ' ';
    for (int j = 0; j < 9; ++j) out << cam.m[0][j] << ' ';
    out << cam.t[0] << ' ' << cam.t[1] << ' ' << cam.t[2] << ' '
        << cam.GetNormalizedMeasurementDistortion() << " 0\n";
  }

  out << point_data.size() << '\n';

  for (size_t i = 0, j = 0; i < point_data.size(); ++i) {
    Point3D& pt = point_data[i];
    int* pc = &ptc[i * 3];
    out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << ' ' << pc[0]
        << ' ' << pc[1] << ' ' << pc[2] << ' ';

    size_t je = j;
    while (je < ptidx.size() && ptidx[je] == (int)i) je++;

    out << (je - j) << ' ';

    for (; j < je; ++j)
      out << camidx[j] << ' ' << " 0 " << measurements[j].x << ' '
          << measurements[j].y << ' ';

    out << '\n';
  }
}

bool LoadBundlerOut(const char* name, ifstream& in,
                    vector<CameraT>& camera_data, vector<Point3D>& point_data,
                    vector<Point2D>& measurements, vector<int>& ptidx,
                    vector<int>& camidx, vector<string>& names,
                    vector<int>& ptc) {
  int rotation_parameter_num = 9;
  string token;
  while (in.peek() == '#') std::getline(in, token);

  char listpath[1024], filepath[1024];
  strcpy(listpath, name);
  char* ext = strstr(listpath, ".out");
  strcpy(ext, "-list.txt\0");

  ///////////////////////////////////
  ifstream listin(listpath);
  if (!listin.is_open()) {
    listin.close();
    listin.clear();
    char* slash = strrchr(listpath, '/');
    if (slash == NULL) slash = strrchr(listpath, '\\');
    slash = slash ? slash + 1 : listpath;
    strcpy(slash, "image_list.txt");
    listin.open(listpath);
  }
  if (listin) std::cout << "Using image list: " << listpath << '\n';

  // read # of cameras
  int ncam = 0, npoint = 0, nproj = 0;
  in >> ncam >> npoint;
  if (ncam <= 1 || npoint <= 1) return false;
  std::cout << ncam << " cameras; " << npoint << " 3D points;\n";

  // read the camera parameters
  camera_data.resize(ncam);  // allocate the camera data
  names.resize(ncam);

  bool det_checked = false;
  for (int i = 0; i < ncam; ++i) {
    float f, q[9], c[3], d[2];
    in >> f >> d[0] >> d[1];
    for (int j = 0; j < rotation_parameter_num; ++j) in >> q[j];
    in >> c[0] >> c[1] >> c[2];

    camera_data[i].SetFocalLength(f);
    camera_data[i].SetInvertedR9T(q, c);
    camera_data[i].SetProjectionDistortion(d[0]);

    if (listin >> filepath && f != 0) {
      char* slash = strrchr(filepath, '/');
      if (slash == NULL) slash = strchr(filepath, '\\');
      names[i] = (slash ? (slash + 1) : filepath);
      std::getline(listin, token);

      if (!det_checked) {
        float det = camera_data[i].GetRotationMatrixDeterminant();
        std::cout << "Check rotation matrix: " << det << '\n';
        det_checked = true;
      }
    } else {
      names[i] = "unknown";
    }
  }

  // read image projections and 3D points.
  point_data.resize(npoint);
  for (int i = 0; i < npoint; ++i) {
    float pt[3];
    int cc[3], npj;
    in >> pt[0] >> pt[1] >> pt[2] >> cc[0] >> cc[1] >> cc[2] >> npj;
    for (int j = 0; j < npj; ++j) {
      int cidx, fidx;
      float imx, imy;
      in >> cidx >> fidx >> imx >> imy;

      camidx.push_back(cidx);  // camera index
      ptidx.push_back(i);  // point index

      // add a measurment to the vector
      measurements.push_back(Point2D(imx, -imy));
      nproj++;
    }
    point_data[i].SetPoint(pt[0], pt[1], pt[2]);
    ptc.insert(ptc.end(), cc, cc + 3);
  }
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << ncam << " cameras; " << npoint << " 3D points; " << nproj
            << " projections\n";
  return true;
}

void SaveBundlerOut(const char* filename, vector<CameraT>& camera_data,
                    vector<Point3D>& point_data, vector<Point2D>& measurements,
                    vector<int>& ptidx, vector<int>& camidx,
                    vector<string>& names, vector<int>& ptc) {
  char listpath[1024];
  strcpy(listpath, filename);
  char* ext = strstr(listpath, ".out");
  if (ext == NULL) return;
  strcpy(ext, "-list.txt\0");

  ofstream out(filename);
  out << "# Bundle file v0.3\n";
  out << std::setprecision(12);  // need enough precision
  out << camera_data.size() << " " << point_data.size() << '\n';

  // save camera data
  for (size_t i = 0; i < camera_data.size(); ++i) {
    float q[9], c[3];
    CameraT& ci = camera_data[i];
    out << ci.GetFocalLength() << ' ' << ci.GetProjectionDistortion() << " 0\n";
    ci.GetInvertedR9T(q, c);
    for (int j = 0; j < 9; ++j) out << q[j] << (((j % 3) == 2) ? '\n' : ' ');
    out << c[0] << ' ' << c[1] << ' ' << c[2] << '\n';
  }
  ///
  for (size_t i = 0, j = 0; i < point_data.size(); ++i) {
    int npj = 0, *ci = &ptc[i * 3];
    Point3D& pt = point_data[i];
    while (j + npj < point_data.size() && ptidx[j + npj] == ptidx[j]) npj++;
    ///////////////////////////
    out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << '\n';
    out << ci[0] << ' ' << ci[1] << ' ' << ci[2] << '\n';
    out << npj << ' ';
    for (int k = 0; k < npj; ++k)
      out << camidx[j + k] << " 0 " << measurements[j + k].x << ' '
          << -measurements[j + k].y << '\n';
    out << '\n';
    j += npj;
  }

  ofstream listout(listpath);
  for (size_t i = 0; i < names.size(); ++i) listout << names[i] << '\n';
}

template <class CameraT, class Point3D>
bool LoadBundlerModel(ifstream& in, vector<CameraT>& camera_data,
                      vector<Point3D>& point_data,
                      vector<Point2D>& measurements, vector<int>& ptidx,
                      vector<int>& camidx) {
  // read bundle data from a file
  size_t ncam = 0, npt = 0, nproj = 0;
  if (!(in >> ncam >> npt >> nproj)) return false;
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << ncam << " cameras; " << npt << " 3D points; " << nproj
            << " projections\n";

  camera_data.resize(ncam);
  point_data.resize(npt);
  measurements.resize(nproj);
  camidx.resize(nproj);
  ptidx.resize(nproj);

  for (size_t i = 0; i < nproj; ++i) {
    double x, y;
    int cidx, pidx;
    in >> cidx >> pidx >> x >> y;
    if (((size_t)pidx) == npt && camidx.size() > i) {
      camidx.resize(i);
      ptidx.resize(i);
      measurements.resize(i);
      std::cout << "Truncate measurements to " << i << '\n';
    } else if (((size_t)pidx) >= npt) {
      continue;
    } else {
      camidx[i] = cidx;
      ptidx[i] = pidx;
      measurements[i].SetPoint2D(x, -y);
    }
  }

  for (size_t i = 0; i < ncam; ++i) {
    double p[9];
    for (int j = 0; j < 9; ++j) in >> p[j];
    CameraT& cam = camera_data[i];
    cam.SetFocalLength(p[6]);
    cam.SetInvertedRT(p, p + 3);
    cam.SetProjectionDistortion(p[7]);
  }

  for (size_t i = 0; i < npt; ++i) {
    double pt[3];
    in >> pt[0] >> pt[1] >> pt[2];
    point_data[i].SetPoint(pt);
  }
  return true;
}

void SaveBundlerModel(const char* filename, vector<CameraT>& camera_data,
                      vector<Point3D>& point_data,
                      vector<Point2D>& measurements, vector<int>& ptidx,
                      vector<int>& camidx) {
  std::cout << "Saving model to " << filename << "...\n";
  ofstream out(filename);
  out << std::setprecision(12);  // need enough precision
  out << camera_data.size() << ' ' << point_data.size() << ' '
      << measurements.size() << '\n';
  for (size_t i = 0; i < measurements.size(); ++i) {
    out << camidx[i] << ' ' << ptidx[i] << ' ' << measurements[i].x << ' '
        << -measurements[i].y << '\n';
  }

  for (size_t i = 0; i < camera_data.size(); ++i) {
    CameraT& cam = camera_data[i];
    double r[3], t[3];
    cam.GetInvertedRT(r, t);
    out << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << t[0] << ' ' << t[1]
        << ' ' << t[2] << ' ' << cam.f << ' ' << cam.GetProjectionDistortion()
        << " 0\n";
  }

  for (size_t i = 0; i < point_data.size(); ++i) {
    Point3D& pt = point_data[i];
    out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << '\n';
  }
}

bool LoadModelFile(const char* name, vector<CameraT>& camera_data,
                   vector<Point3D>& point_data, vector<Point2D>& measurements,
                   vector<int>& ptidx, vector<int>& camidx,
                   vector<string>& names, vector<int>& ptc) {
  if (name == NULL) return false;
  ifstream in(name);

  std::cout << "Loading cameras/points: " << name << "\n";
  if (!in.is_open()) return false;

  if (strstr(name, ".nvm"))
    return LoadNVM(in, camera_data, point_data, measurements, ptidx, camidx,
                   names, ptc);
  else if (strstr(name, ".out"))
    return LoadBundlerOut(name, in, camera_data, point_data, measurements,
                          ptidx, camidx, names, ptc);
  else
    return LoadBundlerModel(in, camera_data, point_data, measurements, ptidx,
                            camidx);
}

float random_ratio(float percent) {
  return (rand() % 101 - 50) * 0.02f * percent + 1.0f;
}

void AddNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data,
              float percent) {
  std::srand((unsigned int)time(NULL));
  for (size_t i = 0; i < camera_data.size(); ++i) {
    camera_data[i].f *= random_ratio(percent);
    camera_data[i].t[0] *= random_ratio(percent);
    camera_data[i].t[1] *= random_ratio(percent);
    camera_data[i].t[2] *= random_ratio(percent);
    double e[3];
    camera_data[i].GetRodriguesRotation(e);
    e[0] *= random_ratio(percent);
    e[1] *= random_ratio(percent);
    e[2] *= random_ratio(percent);
    camera_data[i].SetRodriguesRotation(e);
  }

  for (size_t i = 0; i < point_data.size(); ++i) {
    point_data[i].xyz[0] *= random_ratio(percent);
    point_data[i].xyz[1] *= random_ratio(percent);
    point_data[i].xyz[2] *= random_ratio(percent);
  }
}

void AddStableNoise(vector<CameraT>& camera_data, vector<Point3D>& point_data,
                    const vector<int>& ptidx, const vector<int>& camidx,
                    float percent) {
  ///
  std::srand((unsigned int)time(NULL));
  // do not modify the visibility status..
  vector<float> zz0(ptidx.size());
  vector<CameraT> backup = camera_data;
  vector<float> vx(point_data.size()), vy(point_data.size()),
      vz(point_data.size());
  for (size_t i = 0; i < point_data.size(); ++i) {
    Point3D& pt = point_data[i];
    vx[i] = pt.xyz[0];
    vy[i] = pt.xyz[1];
    vz[i] = pt.xyz[2];
  }

  // find out the median location of all the 3D points.
  size_t median_idx = point_data.size() / 2;

  std::nth_element(vx.begin(), vx.begin() + median_idx, vx.end());
  std::nth_element(vy.begin(), vy.begin() + median_idx, vy.end());
  std::nth_element(vz.begin(), vz.begin() + median_idx, vz.end());
  float cx = vx[median_idx], cy = vy[median_idx], cz = vz[median_idx];

  for (size_t i = 0; i < ptidx.size(); ++i) {
    CameraT& cam = camera_data[camidx[i]];
    Point3D& pt = point_data[ptidx[i]];
    zz0[i] = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] +
             cam.m[2][2] * pt.xyz[2] + cam.t[2];
  }

  vector<float> z2 = zz0;
  median_idx = ptidx.size() / 2;
  std::nth_element(z2.begin(), z2.begin() + median_idx, z2.end());
  float mz = z2[median_idx];  // median depth
  float dist_noise_base = mz * 0.2f;

  /////////////////////////////////////////////////
  // modify points first..
  for (size_t i = 0; i < point_data.size(); ++i) {
    Point3D& pt = point_data[i];
    pt.xyz[0] = pt.xyz[0] - cx + dist_noise_base * random_ratio(percent);
    pt.xyz[1] = pt.xyz[1] - cy + dist_noise_base * random_ratio(percent);
    pt.xyz[2] = pt.xyz[2] - cz + dist_noise_base * random_ratio(percent);
  }

  vector<bool> need_modification(camera_data.size(), true);
  int invalid_count = 0, modify_iteration = 1;

  do {
    if (invalid_count)
      std::cout << "NOTE" << std::setw(2) << modify_iteration << ": modify "
                << invalid_count << " camera to fix visibility\n";

    //////////////////////////////////////////////////////
    for (size_t i = 0; i < camera_data.size(); ++i) {
      if (!need_modification[i]) continue;
      CameraT& cam = camera_data[i];
      double e[3], c[3];
      cam = backup[i];
      cam.f *= random_ratio(percent);

      ///////////////////////////////////////////////////////////
      cam.GetCameraCenter(c);
      c[0] = c[0] - cx + dist_noise_base * random_ratio(percent);
      c[1] = c[1] - cy + dist_noise_base * random_ratio(percent);
      c[2] = c[2] - cz + dist_noise_base * random_ratio(percent);

      ///////////////////////////////////////////////////////////
      cam.GetRodriguesRotation(e);
      e[0] *= random_ratio(percent);
      e[1] *= random_ratio(percent);
      e[2] *= random_ratio(percent);

      ///////////////////////////////////////////////////////////
      cam.SetRodriguesRotation(e);
      cam.SetCameraCenterAfterRotation(c);
    }
    vector<bool> invalidc(camera_data.size(), false);

    invalid_count = 0;
    for (size_t i = 0; i < ptidx.size(); ++i) {
      int cid = camidx[i];
      if (need_modification[cid] == false) continue;
      if (invalidc[cid]) continue;
      CameraT& cam = camera_data[cid];
      Point3D& pt = point_data[ptidx[i]];
      float z = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] +
                cam.m[2][2] * pt.xyz[2] + cam.t[2];
      if (z * zz0[i] > 0) continue;
      if (zz0[i] == 0 && z > 0) continue;
      invalid_count++;
      invalidc[cid] = true;
    }

    need_modification = invalidc;
    modify_iteration++;

  } while (invalid_count && modify_iteration < 20);
}

void ExamineVisiblity(const char* input_filename) {
  //////////////
  vector<CameraD> camera_data;
  vector<Point3B> point_data;
  vector<int> ptidx, camidx;
  vector<Point2D> measurements;
  ifstream in(input_filename);
  LoadBundlerModel(in, camera_data, point_data, measurements, ptidx, camidx);

  ////////////////
  int count = 0;
  double d1 = 100, d2 = 100;
  std::cout << "checking visibility...\n";
  vector<double> zz(ptidx.size());
  for (size_t i = 0; i < ptidx.size(); ++i) {
    CameraD& cam = camera_data[camidx[i]];
    Point3B& pt = point_data[ptidx[i]];
    double dz = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] +
                cam.m[2][2] * pt.xyz[2] + cam.t[2];
    // double dx = cam.m[0][0] * pt.xyz[0] + cam.m[0][1] * pt.xyz[1] +
    // cam.m[0][2] * pt.xyz[2] + cam.t[0];
    // double dy = cam.m[1][0] * pt.xyz[0] + cam.m[1][1] * pt.xyz[1] +
    // cam.m[1][2] * pt.xyz[2] + cam.t[1];

    ////////////////////////////////////////
    float c[3];
    cam.GetCameraCenter(c);

    CameraT camt;
    camt.SetCameraT(cam);
    Point3D ptt;
    ptt.SetPoint(pt.xyz);
    double fz = camt.m[2][0] * ptt.xyz[0] + camt.m[2][1] * ptt.xyz[1] +
                camt.m[2][2] * ptt.xyz[2] + camt.t[2];
    double fz2 = camt.m[2][0] * (ptt.xyz[0] - c[0]) +
                 camt.m[2][1] * (ptt.xyz[1] - c[1]) +
                 camt.m[2][2] * (ptt.xyz[2] - c[2]);

    // if(dz == 0 && fz == 0) continue;

    if (dz * fz <= 0 || fz == 0) {
      std::cout << "cam "
                << camidx[i]  //<<// "; dx = " << dx << "; dy = " << dy
                << "; double: " << dz << "; float " << fz << "; float2 " << fz2
                << "\n";
      // std::cout << cam.m[2][0] << " "<<cam.m[2][1]<< " " <<  cam.m[2][2] << "
      // "<<cam.t[2] << "\n";
      // std::cout << camt.m[2][0] << " "<<camt.m[2][1]<< " " <<  camt.m[2][2]
      // << " "<<camt.t[2] << "\n";
      // std::cout << cam.m[2][0] - camt.m[2][0] << " " <<cam.m[2][1] -
      // camt.m[2][1]<< " "
      //          << cam.m[2][2] - camt.m[2][2] << " " <<cam.t[2] - camt.t[2]<<
      //          "\n";
    }

    zz[i] = dz;
    d1 = std::min(fabs(dz), d1);
    d2 = std::min(fabs(fz), d2);
  }

  std::cout << count << " points moved to wrong side " << d1 << ", " << d2
            << "\n";
}

bool RemoveInvisiblePoints(vector<CameraT>& camera_data,
                           vector<Point3D>& point_data, vector<int>& ptidx,
                           vector<int>& camidx, vector<Point2D>& measurements,
                           vector<string>& names, vector<int>& ptc) {
  vector<float> zz(ptidx.size());
  for (size_t i = 0; i < ptidx.size(); ++i) {
    CameraT& cam = camera_data[camidx[i]];
    Point3D& pt = point_data[ptidx[i]];
    zz[i] = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] +
            cam.m[2][2] * pt.xyz[2] + cam.t[2];
  }
  size_t median_idx = ptidx.size() / 2;
  std::nth_element(zz.begin(), zz.begin() + median_idx, zz.end());
  float dist_threshold = zz[median_idx] * 0.001f;

  // keep removing 3D points. until all of them are infront of the cameras..
  vector<bool> pmask(point_data.size(), true);
  int points_removed = 0;
  for (size_t i = 0; i < ptidx.size(); ++i) {
    int cid = camidx[i], pid = ptidx[i];
    if (!pmask[pid]) continue;
    CameraT& cam = camera_data[cid];
    Point3D& pt = point_data[pid];
    bool visible = (cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] +
                        cam.m[2][2] * pt.xyz[2] + cam.t[2] >
                    dist_threshold);
    pmask[pid] = visible;  // this point should be removed
    if (!visible) points_removed++;
  }
  if (points_removed == 0) return false;
  vector<int> cv(camera_data.size(), 0);
  // should any cameras be removed ?
  int min_observation = 20;  // cameras should see at leat 20 points

  do {
    // count visible points for each camera
    std::fill(cv.begin(), cv.end(), 0);
    for (size_t i = 0; i < ptidx.size(); ++i) {
      int cid = camidx[i], pid = ptidx[i];
      if (pmask[pid]) cv[cid]++;
    }

    // check if any more points should be removed
    vector<int> pv(point_data.size(), 0);
    for (size_t i = 0; i < ptidx.size(); ++i) {
      int cid = camidx[i], pid = ptidx[i];
      if (!pmask[pid]) continue;  // point already removed
      if (cv[cid] < min_observation)  // this camera shall be removed.
      {
        ///
      } else {
        pv[pid]++;
      }
    }

    points_removed = 0;
    for (size_t i = 0; i < point_data.size(); ++i) {
      if (pmask[i] == false) continue;
      if (pv[i] >= 2) continue;
      pmask[i] = false;
      points_removed++;
    }
  } while (points_removed > 0);

  ////////////////////////////////////
  vector<bool> cmask(camera_data.size(), true);
  for (size_t i = 0; i < camera_data.size(); ++i)
    cmask[i] = cv[i] >= min_observation;
  ////////////////////////////////////////////////////////

  vector<int> cidx(camera_data.size());
  vector<int> pidx(point_data.size());

  /// modified model.
  vector<CameraT> camera_data2;
  vector<Point3D> point_data2;
  vector<int> ptidx2;
  vector<int> camidx2;
  vector<Point2D> measurements2;
  vector<string> names2;
  vector<int> ptc2;

  //
  if (names.size() < camera_data.size())
    names.resize(camera_data.size(), string("unknown"));
  if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

  //////////////////////////////
  int new_camera_count = 0, new_point_count = 0;
  for (size_t i = 0; i < camera_data.size(); ++i) {
    if (!cmask[i]) continue;
    camera_data2.push_back(camera_data[i]);
    names2.push_back(names[i]);
    cidx[i] = new_camera_count++;
  }

  for (size_t i = 0; i < point_data.size(); ++i) {
    if (!pmask[i]) continue;
    point_data2.push_back(point_data[i]);
    ptc.push_back(ptc[i]);
    pidx[i] = new_point_count++;
  }

  int new_observation_count = 0;
  for (size_t i = 0; i < ptidx.size(); ++i) {
    int pid = ptidx[i], cid = camidx[i];
    if (!pmask[pid] || !cmask[cid]) continue;
    ptidx2.push_back(pidx[pid]);
    camidx2.push_back(cidx[cid]);
    measurements2.push_back(measurements[i]);
    new_observation_count++;
  }

  std::cout << "NOTE: removing " << (camera_data.size() - new_camera_count)
            << " cameras; " << (point_data.size() - new_point_count)
            << " 3D Points; " << (measurements.size() - new_observation_count)
            << " Observations;\n";

  camera_data2.swap(camera_data);
  names2.swap(names);
  point_data2.swap(point_data);
  ptc2.swap(ptc);
  ptidx2.swap(ptidx);
  camidx2.swap(camidx);
  measurements2.swap(measurements);

  return true;
}

void SaveModelFile(const char* outpath, vector<CameraT>& camera_data,
                   vector<Point3D>& point_data, vector<Point2D>& measurements,
                   vector<int>& ptidx, vector<int>& camidx,
                   vector<string>& names, vector<int>& ptc) {
  if (outpath == NULL) return;
  if (strstr(outpath, ".nvm"))
    SaveNVM(outpath, camera_data, point_data, measurements, ptidx, camidx,
            names, ptc);
  else if (strstr(outpath, ".out"))
    SaveBundlerOut(outpath, camera_data, point_data, measurements, ptidx,
                   camidx, names, ptc);
  else
    SaveBundlerModel(outpath, camera_data, point_data, measurements, ptidx,
                     camidx);
}

}  // namespace pba
