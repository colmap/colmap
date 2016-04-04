////////////////////////////////////////////////////////////////////////////
//	File:		CLTexImage.h
//	Author:		Changchang Wu
//	Description :	interface for the CLTexImage class.
//					class for storing data in CUDA.
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

#ifndef CL_TEX_IMAGE_H
#define CL_TEX_IMAGE_H

class GLTexImage;

class CLTexImage
{	
protected:
	cl_context  		_context;
	cl_command_queue	_queue;
	cl_mem		        _clData;
	int			        _numChannel;
	int			        _imgWidth;
	int			        _imgHeight;
	int			        _texWidth;
	int			        _texHeight;
	int			        _bufferLen;
    int                 _fromGL;
private:
    void ReleaseTexture(); 
public:
	void SetImageSize(int width, int height);
    void SetPackedSize(int width, int height, int packed);
	void InitBufferTex(int width, int height, int nchannel);
	void InitTexture(int width, int height, int nchannel);
    void InitPackedTex(int width, int height, int packed);
    void InitTextureGL(GLuint tex, int width, int height, int nchannel);
	void CopyToHost(void* buf);
	void CopyFromHost(const void* buf);
public:
    int CopyToPBO(GLuint pbo);
    int GetImageDataSize();
public:
    inline operator cl_mem(){return _clData; }
	inline int GetImgWidth(){return _imgWidth;}
	inline int GetImgHeight(){return _imgHeight;}
	inline int GetTexWidth(){return _texWidth;}
    inline int GetTexHeight(){return _texHeight;}
	inline int GetDataSize(){return _bufferLen;}
    inline bool IsImage2D() {return _bufferLen == 0;}
	inline int GetImgPixelCount(){return _imgWidth*_imgHeight;}
	inline int GetTexPixelCount(){return _texWidth*_texHeight;}
public:
    CLTexImage();
	CLTexImage(cl_context context, cl_command_queue queue);
    void SetContext(cl_context context, cl_command_queue queue);
	virtual ~CLTexImage();
	friend class ProgramCL;
	friend class PyramidCL;
    friend class ProgramBagCL;
    friend class ProgramBagCLN;
};

//////////////////////////////////////////////////
//transfer OpenGL Texture to PBO, then to CUDA vector
//#endif 
#endif // !defined(CU_TEX_IMAGE_H)
#endif


