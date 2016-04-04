////////////////////////////////////////////////////////////////////////////
//	File:		CLTexImage.cpp
//	Author:		Changchang Wu
//	Description : implementation of the CLTexImage class.
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

#if defined(CL_SIFTGPU_ENABLED)

#include "GL/glew.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
using namespace std;


#include <CL/OpenCL.h>
#include "CLTexImage.h" 
#include "ProgramCL.h"
#include "GlobalUtil.h"


CLTexImage::CLTexImage()
{
    _context = NULL;
    _queue = NULL;
	_clData = NULL;
	_numChannel = _bufferLen = _fromGL = 0;
    _imgWidth = _imgHeight = _texWidth = _texHeight = 0;
}

CLTexImage::CLTexImage(cl_context context, cl_command_queue queue)
{
	_context = context;
	_queue  = queue;
	_clData = NULL;
	_numChannel = _bufferLen = _fromGL = 0;
    _imgWidth = _imgHeight = _texWidth = _texHeight = 0;
}

void CLTexImage::SetContext(cl_context context, cl_command_queue queue)
{
    _context = context;
    _queue   = queue;
}


CLTexImage::~CLTexImage()
{
    ReleaseTexture();
}

void CLTexImage::ReleaseTexture()
{
    if(_fromGL)     clEnqueueReleaseGLObjects(_queue, 1, &_clData, 0, NULL, NULL); 
	if(_clData) 	clReleaseMemObject(_clData);
}

void CLTexImage::SetImageSize(int width, int height)
{
	_imgWidth = width;
	_imgHeight = height;
}

void CLTexImage::InitBufferTex(int width, int height, int nchannel)
{
    if(width == 0 || height == 0 || nchannel <= 0 || _fromGL) return; 

	_imgWidth = width;	_imgHeight = height;
    _texWidth = _texHeight = _fromGL = 0; 
	_numChannel = min(nchannel, 4);

	int size = width * height * _numChannel * sizeof(float);
	if (size <= _bufferLen) return;
	
	//allocate the buffer data
    cl_int status; 
	if(_clData) status = clReleaseMemObject(_clData);

	_clData = clCreateBuffer(_context, CL_MEM_READ_WRITE, 
                            _bufferLen = size, NULL, &status);

    ProgramBagCL::CheckErrorCL(status, "CLTexImage::InitBufferTex");

}

void CLTexImage::InitTexture(int width, int height, int nchannel)
{
    if(width == 0 || height == 0 || nchannel <= 0 || _fromGL) return; 
	if(_clData && width == _texWidth && height == _texHeight && _numChannel == nchannel) return;
	if(_clData) clReleaseMemObject(_clData);

	_texWidth = _imgWidth =  width;
	_texHeight = _imgHeight =  height;
	_numChannel = nchannel; 
    _bufferLen = _fromGL = 0; 

    cl_int status;    cl_image_format format;

    if(nchannel == 1) format.image_channel_order = CL_R;
    else if(nchannel == 2) format.image_channel_order = CL_RG;
    else if(nchannel == 3) format.image_channel_order = CL_RGB;
    else format.image_channel_order = CL_RGBA;

    format.image_channel_data_type = CL_FLOAT;
    _clData = clCreateImage2D(_context, CL_MEM_READ_WRITE, & format, 
                    _texWidth, _texHeight, 0, 0, &status);
    ProgramBagCL::CheckErrorCL(status, "CLTexImage::InitTexture");
}

void CLTexImage::InitPackedTex(int width, int height, int packed)
{
    if(packed) InitTexture((width + 1) >> 1, (height + 1) >> 1, 4);
    else InitTexture(width, height, 1);
}

void CLTexImage::SetPackedSize(int width, int height, int packed)
{
    if(packed)  SetImageSize((width + 1) >> 1, (height + 1) >> 1);
    else SetImageSize(width, height);
}

void CLTexImage::InitTextureGL(GLuint tex, int width, int height, int nchannel)
{
    if(tex == 0) return;
    if(_clData) clReleaseMemObject(_clData); 

    ////create the memory object
    cl_int status;
    _clData = clCreateFromGLTexture2D(_context, CL_MEM_WRITE_ONLY, 
        GlobalUtil::_texTarget, 0 , tex, &status);
    ProgramBagCL::CheckErrorCL(status, "CLTexImage::InitTextureGL->clCreateFromGLTexture2D");
    if(status != CL_SUCCESS) return; 

    _texWidth = _imgWidth = width;
    _texHeight = _imgHeight = height;
    _numChannel = nchannel;
    _bufferLen = 0;    _fromGL = 1; 

    ////acquire object
    status = clEnqueueAcquireGLObjects(_queue, 1, &_clData, 0, NULL, NULL); 
    ProgramBagCL::CheckErrorCL(status, "CLTexImage::InitTextureGL->clEnqueueAcquireGLObjects");

}

void CLTexImage::CopyFromHost(const void * buf)
{
	if(_clData == NULL) return;
    cl_int status; 
    if(_bufferLen)
    {
	    status = clEnqueueWriteBuffer(_queue, _clData, false,  0, 
            _imgWidth * _imgHeight * _numChannel * sizeof(float),  buf,  0, NULL, NULL);
    }else
    {
        size_t origin[3] = {0, 0, 0}, region[3] = {_imgWidth, _imgHeight, 1};
        size_t row_pitch = _imgWidth * _numChannel * sizeof(float);
        status = clEnqueueWriteImage(_queue, _clData, false, origin,
            region, row_pitch, 0, buf, 0, 0, 0);  
    }
    ProgramBagCL::CheckErrorCL(status, "CLTexImage::CopyFromHost");
}

int CLTexImage::GetImageDataSize()
{
    return _imgWidth * _imgHeight * _numChannel * sizeof(float);
}

int CLTexImage::CopyToPBO(GLuint pbo)
{
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);  

    int esize = GetImageDataSize(), bsize;
	glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
    {
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, esize,	NULL, GL_STATIC_DRAW_ARB);
        glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
    }
    if(bsize >= esize)
    {
        // map the buffer object into client's memory
        void* ptr = glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
        CopyToHost(ptr); 
        clFinish(_queue);
        glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
    }
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);  
    GlobalUtil::CheckErrorsGL("CLTexImage::CopyToPBO");
    return esize >= bsize;
}

void CLTexImage::CopyToHost(void * buf)
{
	if(_clData == NULL) return;
    cl_int status;
    if(_bufferLen)
    { 
	    status = clEnqueueReadBuffer(_queue, _clData, true,  0, 
            _imgWidth * _imgHeight * _numChannel * sizeof(float), buf,  0, NULL, NULL);
    }else
    {
        size_t origin[3] = {0, 0, 0}, region[3] = {_imgWidth, _imgHeight, 1};
        size_t row_pitch = _imgWidth * _numChannel * sizeof(float);
        status = clEnqueueReadImage(_queue, _clData, true, origin,
            region, row_pitch, 0, buf, 0, 0, 0);
    }

    ProgramBagCL::CheckErrorCL(status, "CLTexImage::CopyToHost");
}

#endif

