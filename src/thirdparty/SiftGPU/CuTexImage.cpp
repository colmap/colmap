////////////////////////////////////////////////////////////////////////////
//	File:		CuTexImage.cpp
//	Author:		Changchang Wu
//	Description : implementation of the CuTexImage class.
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty.
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#if defined(SIFTGPU_CUDA_ENABLED)

#include "GL/glew.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cstring>
using namespace std;


#if defined(COLMAP_HIP_ENABLED)
// On ROCm builds, route cuda* symbols through the compat header. HIP does
// not ship the legacy cudaGL* OpenGL interop API used below; those call
// sites are guarded with !COLMAP_HIP_ENABLED.
#include "colmap/util/cuda_to_hip.h"
#else
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#endif

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h"
#include "ProgramCU.h"

CuTexImage::CuTexObj::~CuTexObj()
{
	// handle == 0 means the object was default-constructed and never bound;
	// guard the call so the destructor is safe on HIP (where destroying a
	// bogus handle segfaults rather than silently no-op'ing as on CUDA).
	if (handle != 0) {
		cudaDestroyTextureObject(handle);
	}
}

CuTexImage::CuTexObj CuTexImage::BindTexture(const cudaTextureDesc& textureDesc,
											   										 const cudaChannelFormatDesc& channelFmtDesc)
{
	CuTexObj texObj;

	cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(resourceDesc));
  resourceDesc.resType = cudaResourceTypeLinear;
  resourceDesc.res.linear.devPtr = _cuData;
	resourceDesc.res.linear.desc = channelFmtDesc;
	resourceDesc.res.linear.sizeInBytes = _numBytes;

	cudaCreateTextureObject(&texObj.handle, &resourceDesc, &textureDesc, nullptr);
	ProgramCU::CheckErrorCUDA("CuTexImage::BindTexture");

	return texObj;
}

CuTexImage::CuTexObj CuTexImage::BindTexture2D(const cudaTextureDesc& textureDesc,
											   											 const cudaChannelFormatDesc& channelFmtDesc)
{
	CuTexObj texObj;

	cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(resourceDesc));
	resourceDesc.resType = cudaResourceTypePitch2D;
  resourceDesc.res.pitch2D.devPtr = _cuData;
	resourceDesc.res.pitch2D.width = _imgWidth;
	resourceDesc.res.pitch2D.height = _imgHeight;
	resourceDesc.res.pitch2D.pitchInBytes = _imgWidth * _numChannel * sizeof(float);
	resourceDesc.res.pitch2D.desc = channelFmtDesc;

	cudaCreateTextureObject(&texObj.handle, &resourceDesc, &textureDesc, nullptr);
	ProgramCU::CheckErrorCUDA("CuTexImage::BindTexture2D");

	return texObj;
}

CuTexImage::CuTexImage()
{
	_cuData = NULL;
	_cuData2D = NULL;
	_fromPBO = 0;
	_numChannel = _numBytes = 0;
	_imgWidth = _imgHeight = _texWidth = _texHeight = 0;
}

CuTexImage::CuTexImage(int width, int height, int nchannel, GLuint pbo)
{
	_cuData = NULL;

#if defined(COLMAP_HIP_ENABLED)
	// HIP does not expose the legacy cudaGL* PBO interop API. SiftGPU does
	// not need it for the headless feature-extraction path used by colmap;
	// the regular cudaMalloc path is used when the image is sized later.
	(void)width; (void)height; (void)nchannel; (void)pbo;
	_fromPBO = 0;
	_numBytes = 0;
	_imgWidth = 0;
	_imgHeight = 0;
	_numChannel = 0;
#else
	//check size of pbo
	GLint bsize, esize = width * height * nchannel * sizeof(float);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
	{
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize,	NULL, GL_STATIC_DRAW_ARB);
		glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	if(bsize >=esize)
	{

		cudaGLRegisterBufferObject(pbo);
		cudaGLMapBufferObject(&_cuData, pbo);
		ProgramCU::CheckErrorCUDA("cudaGLMapBufferObject");
		_fromPBO = pbo;
	}else
	{
		_cuData = NULL;
		_fromPBO = 0;
	}
	if(_cuData)
	{
		_numBytes = bsize;
		_imgWidth = width;
		_imgHeight = height;
		_numChannel = nchannel;
	}else
	{
		_numBytes = 0;
		_imgWidth = 0;
		_imgHeight = 0;
		_numChannel = 0;
	}
#endif

	_texWidth = _texHeight =0;

	_cuData2D = NULL;
}

CuTexImage::~CuTexImage()
{
#if !defined(COLMAP_HIP_ENABLED)
	if(_fromPBO)
	{
		cudaGLUnmapBufferObject(_fromPBO);
		cudaGLUnregisterBufferObject(_fromPBO);
	}else
#endif
	if(_cuData)
	{
		cudaFree(_cuData);
	}
	if(_cuData2D)  cudaFreeArray(_cuData2D);
}

void CuTexImage::SetImageSize(int width, int height)
{
	_imgWidth = width;
	_imgHeight = height;
}

bool CuTexImage::InitTexture(int width, int height, int nchannel)
{
	_imgWidth = width;
	_imgHeight = height;
	_numChannel = min(max(nchannel, 1), 4);

	const size_t size = width * height * _numChannel * sizeof(float);

  if (size < 0) {
    return false;
  }

  // SiftGPU uses int for all indexes and
  // this ensures that all elements can be accessed.
  if (size >= INT_MAX * sizeof(float)) {
    return false;
  }

	if(size <= _numBytes) return true;

	if(_cuData) cudaFree(_cuData);

	//allocate the array data
	const cudaError_t status = cudaMalloc(&_cuData, _numBytes = size);

  if (status != cudaSuccess) {
    _cuData = NULL;
    _numBytes = 0;
    return false;
  }

  return true;
}

void CuTexImage::CopyFromHost(const void * buf)
{
	if(_cuData == NULL) return;
	cudaMemcpy( _cuData, buf, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyHostToDevice);
}

void CuTexImage::CopyToHost(void * buf)
{
	if(_cuData == NULL) return;
	cudaMemcpy(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyDeviceToHost);
}

void CuTexImage::CopyToHost(void * buf, int stream)
{
	if(_cuData == NULL) return;
	cudaMemcpyAsync(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

void CuTexImage::InitTexture2D()
{
#if !defined(SIFTGPU_ENABLE_LINEAR_TEX2D)
	if(_cuData2D && (_texWidth < _imgWidth || _texHeight < _imgHeight))
	{
		cudaFreeArray(_cuData2D);
		_cuData2D = NULL;
	}
	if(_cuData2D == NULL)
	{
		_texWidth = max(_texWidth, _imgWidth);
		_texHeight = max(_texHeight, _imgHeight);
		cudaChannelFormatDesc desc;
		desc.f = cudaChannelFormatKindFloat;
		desc.x = sizeof(float) * 8;
		desc.y = _numChannel >=2 ? sizeof(float) * 8 : 0;
		desc.z = _numChannel >=3 ? sizeof(float) * 8 : 0;
		desc.w = _numChannel >=4 ? sizeof(float) * 8 : 0;
		const cudaError_t status = cudaMallocArray(&_cuData2D, &desc, _texWidth, _texHeight);
    if (status != cudaSuccess) {
      _cuData = NULL;
      _numBytes = 0;
    }
		ProgramCU::CheckErrorCUDA("CuTexImage::InitTexture2D");
	}
#endif
}

void CuTexImage::CopyToTexture2D()
{
#if !defined(SIFTGPU_ENABLE_LINEAR_TEX2D)
	InitTexture2D();
	if(_cuData2D)
	{
		cudaMemcpy2DToArray(_cuData2D, 0, 0, _cuData, _imgWidth* _numChannel* sizeof(float) ,
		_imgWidth * _numChannel*sizeof(float), _imgHeight,	cudaMemcpyDeviceToDevice);
		ProgramCU::CheckErrorCUDA("cudaMemcpy2DToArray");
	}
#endif
}

void CuTexImage::CopyFromPBO(int width, int height, GLuint pbo)
{
#if defined(COLMAP_HIP_ENABLED)
	// Legacy cudaGL* PBO interop is not available on HIP. Headless feature
	// extraction does not exercise this path.
	(void)width; (void)height; (void)pbo;
#else
	void* pbuf =NULL;
	GLint esize = width * height * sizeof(float);
	cudaGLRegisterBufferObject(pbo);
	cudaGLMapBufferObject(&pbuf, pbo);

	cudaMemcpy(_cuData, pbuf, esize, cudaMemcpyDeviceToDevice);

	cudaGLUnmapBufferObject(pbo);
	cudaGLUnregisterBufferObject(pbo);
#endif
}

int CuTexImage::CopyToPBO(GLuint pbo)
{
#if defined(COLMAP_HIP_ENABLED)
	(void)pbo;
	return 0;
#else
	void* pbuf =NULL;
	GLint bsize, esize = _imgWidth * _imgHeight * sizeof(float) * _numChannel;
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
	{
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize*3/2,	NULL, GL_STATIC_DRAW_ARB);
		glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

	if(bsize >= esize)
	{
		cudaGLRegisterBufferObject(pbo);
		cudaGLMapBufferObject(&pbuf, pbo);
		cudaMemcpy(pbuf, _cuData, esize, cudaMemcpyDeviceToDevice);
		cudaGLUnmapBufferObject(pbo);
		cudaGLUnregisterBufferObject(pbo);
		return 1;
	}else
	{
		return 0;
	}
#endif
}

#endif
