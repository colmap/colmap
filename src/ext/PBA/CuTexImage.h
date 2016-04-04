////////////////////////////////////////////////////////////////////////////
//  File:           CuTexImage.h
//  Author:         Changchang Wu
//  Description :   interface for the CuTexImage class.
//                  class for storing data in CUDA.
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

#ifndef CU_TEX_IMAGE_H
#define CU_TEX_IMAGE_H

struct textureReference;

namespace pba {

class CuTexImage {
 protected:
  bool _owner;
  void* _cuData;
  unsigned int _numChannel;
  unsigned int _imgWidth;
  unsigned int _imgHeight;
  size_t _numBytes;

 public:
  bool InitTexture(unsigned int width, unsigned int height,
                   unsigned int nchannel = 1);
  void SetTexture(void* data, unsigned int width, unsigned int nchannel = 1);
  void BindTexture(textureReference& texRef);
  void BindTexture(textureReference& texRef, int offset, size_t size);
  void BindTexture2(textureReference& texRef1, textureReference& texRef2);
  void BindTexture4(textureReference& texRef1, textureReference& texRef2,
                    textureReference& texRef3, textureReference& texRef4);
  int BindTextureX(textureReference& texRef1, textureReference& texRef2,
                   textureReference& texRef3, textureReference& texRef4,
                   bool force4);
  void SwapData(CuTexImage& src);
  void CopyToHost(void* buf);
  void CopyFromDevice(const void* buf);
  void CopyFromHost(const void* buf);
  void SaveToFile(const char* name);
  void ReleaseData();

 public:
  inline float* data() { return GetRequiredSize() ? ((float*)_cuData) : NULL; }
  inline bool IsValid() { return _cuData != NULL && GetDataSize() > 0; }
  inline unsigned int GetLength() {
    return _imgWidth * _imgHeight * _numChannel;
  }
  inline unsigned int GetImgWidth() { return _imgWidth; }
  inline unsigned int GetImgHeight() { return _imgHeight; }
  inline size_t GetReservedWidth() {
    return _numBytes == 0
               ? 0
               : (_numBytes / (_imgHeight * _numChannel * sizeof(float)));
  }
  inline size_t GetDataSize() { return _numBytes == 0 ? 0 : GetRequiredSize(); }
  inline size_t GetRequiredSize() {
    return sizeof(float) * _imgWidth * _imgHeight * _numChannel;
  }
  inline unsigned int IsHugeData() { return (GetLength() - 1) / (1 << 27); }

 public:
  CuTexImage();
  virtual ~CuTexImage();
};

}  // namespace pba

#endif  // !defined(CU_TEX_IMAGE_H)
