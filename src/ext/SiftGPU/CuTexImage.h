////////////////////////////////////////////////////////////////////////////
//	File:		CuTexImage.h
//	Author:		Changchang Wu
//	Description :	interface for the CuTexImage class.
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


#ifndef CU_TEX_IMAGE_H
#define CU_TEX_IMAGE_H

class GLTexImage;
struct cudaArray;
struct textureReference;

//using texture2D from linear memory

#define SIFTGPU_ENABLE_LINEAR_TEX2D

class CuTexImage
{	
protected:
	void*		_cuData;
	cudaArray*	_cuData2D;
	int			_numChannel;
	int			_numBytes;
	int			_imgWidth;
	int			_imgHeight;
	int			_texWidth;
	int			_texHeight;
	GLuint		_fromPBO;
public:
	virtual void SetImageSize(int width, int height);
	virtual void InitTexture(int width, int height, int nchannel = 1);
	void InitTexture2D();
	inline void BindTexture(textureReference& texRef);
	inline void BindTexture2D(textureReference& texRef);
	void CopyToTexture2D();
	void CopyToHost(void* buf);
	void CopyToHost(void* buf, int stream);
	void CopyFromHost(const void* buf);
	int  CopyToPBO(GLuint pbo);
	void CopyFromPBO(int width, int height, GLuint pbo);
	static int DebugCopyToTexture2D();
public:
	inline int GetImgWidth(){return _imgWidth;}
	inline int GetImgHeight(){return _imgHeight;}
	inline int GetDataSize(){return _numBytes;}
public:
	CuTexImage();
	CuTexImage(int width, int height, int nchannel, GLuint pbo);
	virtual ~CuTexImage();
	friend class ProgramCU;
	friend class PyramidCU;
};

//////////////////////////////////////////////////
//transfer OpenGL Texture to PBO, then to CUDA vector
//#endif 
#endif // !defined(CU_TEX_IMAGE_H)

