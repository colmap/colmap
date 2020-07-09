////////////////////////////////////////////////////////////////////////////
//	File:		GLTexImage.h
//	Author:		Changchang Wu
//	Description : interface for the GLTexImage class.
//		GLTexImage:		naive texture class. 
//						sevral different quad drawing functions are provied
//		GLTexPacked:	packed version (four value packed as four channels of a pixel)
//		GLTexInput:		GLTexImage + some input information
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


#ifndef GL_TEX_IMAGE_H
#define GL_TEX_IMAGE_H

class GlobalUtil;
class GLTexImage :public GlobalUtil 
{	
protected:
	GLuint	_texID;
	int		_imgWidth;
	int		_imgHeight;
	int		_texWidth;
	int		_texHeight;
	int		_drawWidth;
	int		_drawHeight;
public:
	static void DetachFBO(int i);
	static void UnbindTex();
	static void UnbindMultiTex(int n);
	static void DrawQuad(float x1, float x2, float y1, float y2);

public:
	virtual void DrawQuadUS(int scale);
	virtual void DrawQuadDS(int scale);
	virtual void DrawImage();
	virtual void TexConvertRGB();
	virtual void ZeroHistoMargin();
	virtual void SetImageSize(int width, int height);
	virtual void InitTexture(int width, int height, int clamp_to_edge =1 );
	void InitTexture(int width, int height, int clamp_to_edge, GLuint format);
	virtual void FillMargin(int marginx, int marginy);
public:
	void DrawScaledQuad(float scale);
	int  CopyToPBO(GLuint pbo, int width, int height, GLenum format = GL_RGBA);
	void CopyFromPBO(GLuint pbo, int width, int height, GLenum format = GL_RGBA);
	void FitRealTexViewPort();
	void DrawQuadMT8();
	void DrawQuadMT4();
	void DrawQuadReduction();
	void DrawQuadReduction(int w, int h);
	void DrawMargin(int right, int bottom);
	void DrawQuad();
	void FitTexViewPort();
	void ZeroHistoMargin(int hw, int hh);
	int  CheckTexture();
	void SaveToASCII(const char* path);
public:
	void AttachToFBO(int i );
	void BindTex();
	operator GLuint (){return _texID;}	
	GLuint GetTexID(){return _texID;}
	int	GetImgPixelCount(){return _imgWidth*_imgHeight;}
	int GetTexPixelCount(){return _texWidth*_texHeight;}
	int	GetImgWidth(){return _imgWidth;}
	int GetImgHeight(){return _imgHeight;}
	int	GetTexWidth(){return _texWidth;}
	int GetTexHeight(){return _texHeight;}
	int	GetDrawWidth(){return _drawWidth;}
	int GetDrawHeight(){return _drawHeight;}
	//int	IsTexTight(){return _texWidth == _drawWidth && _texHeight == _drawHeight;}
	int	IsTexPacked(){return _drawWidth != _imgWidth;}
	GLTexImage();
	virtual ~GLTexImage();
	friend class SiftGPU;
};

//class for handle data input, to support all openGL-supported data format, 
//data are first uploaded to an openGL texture then converted, and optionally
//when the datatype is simple, we downsample/convert on cpu
class GLTexInput:public GLTexImage
{
public:
	int      _down_sampled;
	int      _rgb_converted;
    int      _data_modified;

    //////////////////////////
	float *        _converted_data;
    const void*    _pixel_data;
public:
	static int  IsSimpleGlFormat(unsigned int gl_format, unsigned int gl_type)
	{
		//the formats there is a cpu code to conver rgb and downsample
		 return (gl_format ==GL_LUMINANCE ||gl_format == GL_LUMINANCE_ALPHA||
				gl_format == GL_RGB||	gl_format == GL_RGBA||
				gl_format == GL_BGR || gl_format == GL_BGRA) && 
				(gl_type == GL_UNSIGNED_BYTE || gl_type == GL_FLOAT || gl_type == GL_UNSIGNED_SHORT); 
	}
//in vc6, template member function doesn't work
#if !defined(_MSC_VER) || _MSC_VER > 1200
	template <class Uint> 
	static int DownSamplePixelDataI(unsigned int gl_format, int width, int height, 
		int ds, const Uint * pin, Uint * pout);
	template <class Uint> 
	static int DownSamplePixelDataI2F(unsigned int gl_format, int width, int height, 
		int ds, const Uint * pin, float * pout, int skip  = 0);
#endif
	static int DownSamplePixelDataF(unsigned int gl_format, int width, int height, 
		int ds, const float * pin, float * pout, int skip = 0);
    static int TruncateWidthCU(int w) {return  w & 0xfffffffc; }
public:
	GLTexInput() : _down_sampled(0), _rgb_converted(0), _data_modified(0), 
                    _converted_data(0), _pixel_data(0){}
	int SetImageData(int width, int height, const void * data, 
					unsigned int gl_format, unsigned int gl_type);
	int LoadImageFile(char * imagepath, int & w, int &h);
    void VerifyTexture();
    virtual ~GLTexInput();
};

//GLTexPacked doesn't have any data
//so that we can use the GLTexImage* pointer to index a GLTexPacked Vector

class GLTexPacked:public GLTexImage
{
public:
	virtual void	DrawImage();
	virtual void	DrawQuadUS(int scale);
	virtual void	DrawQuadDS(int scale);
	virtual void	FillMargin(int marginx, int marginy);
	virtual void	InitTexture(int width, int height, int clamp_to_edge =1);
	virtual void	TexConvertRGB();
	virtual void	SetImageSize(int width, int height);
	virtual void	ZeroHistoMargin();
	//virtual void	GetHistWH(int& w, int& h){return w = (3 + sz)>>1;}
public:
	void	DrawMargin(int right, int bottom, int mx, int my);
	GLTexPacked():GLTexImage(){}
};


#endif // !defined(GL_TEX_IMAGE_H)

