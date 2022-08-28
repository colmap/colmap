////////////////////////////////////////////////////////////////////////////
//	File:		GLTexImage.cpp
//	Author:		Changchang Wu
//	Description : implementation of the GLTexImage class.
//
//
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


#include "GL/glew.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
using namespace std;



#include "GlobalUtil.h"

#include "GLTexImage.h"
#include "FrameBufferObject.h"
#include "ShaderMan.h"


//#define SIFTGPU_NO_DEVIL

#ifndef SIFTGPU_NO_DEVIL
    #include "IL/il.h"
#else
	#include <string.h>
#endif
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


GLTexImage::GLTexImage()
{
	_imgWidth = _imgHeight = 0;
	_texWidth = _texHeight = 0;
	_drawWidth = _drawHeight = 0;
	_texID = 0;

}

GLTexImage::~GLTexImage()
{
	if(_texID) glDeleteTextures(1, &_texID);
}

int GLTexImage::CheckTexture()
{
	if(_texID)
	{
		GLint tw, th;
		BindTex();
		glGetTexLevelParameteriv(_texTarget, 0, GL_TEXTURE_WIDTH , &tw);
		glGetTexLevelParameteriv(_texTarget, 0, GL_TEXTURE_HEIGHT , &th);
		UnbindTex();
		return tw == _texWidth && th == _texHeight;
	}else
	{
		return _texWidth == 0 && _texHeight ==0;

	}
}
//set a dimension that is smaller than the actuall size
//for drawQuad
void GLTexImage::SetImageSize( int width,  int height)
{
	_drawWidth  = _imgWidth =  width;
	_drawHeight = _imgHeight =  height;
}

void GLTexImage::InitTexture( int width,  int height, int clamp_to_edge)
{

	if(_texID && width == _texWidth && height == _texHeight ) return;
	if(_texID==0)	glGenTextures(1, &_texID);

	_texWidth = _imgWidth = _drawWidth = width;
	_texHeight = _imgHeight = _drawHeight = height;

	BindTex();

	if(clamp_to_edge)
	{
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}else
	{
		//out of bound tex read returns 0??
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	}
	glTexParameteri(_texTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(_texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glTexImage2D(_texTarget, 0, _iTexFormat,
		_texWidth, _texHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	CheckErrorsGL("glTexImage2D");


	UnbindTex();

}


void GLTexImage::InitTexture( int width,  int height, int clamp_to_edge, GLuint format)
{

	if(_texID && width == _texWidth && height == _texHeight ) return;
	if(_texID==0)	glGenTextures(1, &_texID);

	_texWidth = _imgWidth = _drawWidth = width;
	_texHeight = _imgHeight = _drawHeight = height;

	BindTex();

	if(clamp_to_edge)
	{
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}else
	{
		//out of bound tex read returns 0??
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	}
	glTexParameteri(_texTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(_texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glTexImage2D(_texTarget, 0, format, _texWidth, _texHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	UnbindTex();

}
void  GLTexImage::BindTex()
{
	glBindTexture(_texTarget, _texID);
}

void  GLTexImage::UnbindTex()
{
	glBindTexture(_texTarget, 0);
}


void  GLTexImage::DrawQuad()
{
	glBegin (GL_QUADS);
		glTexCoord2i ( 0			,   0   ); 				glVertex2i   ( 0			,		0   );
		glTexCoord2i ( 0			,   _drawHeight  );		glVertex2i   ( 0			,		_drawHeight   );
 		glTexCoord2i ( _drawWidth   ,   _drawHeight  ); 	glVertex2i   ( _drawWidth	,		_drawHeight   );
		glTexCoord2i ( _drawWidth	,   0   ); 				glVertex2i   ( _drawWidth	,		0   );
	glEnd ();
	glFlush();
}

void GLTexImage::FillMargin(int marginx, int marginy)
{
	//
	marginx = min(marginx, _texWidth - _imgWidth);
	marginy = min(marginy, _texHeight - _imgHeight);
	if(marginx >0 || marginy > 0)
	{
		GlobalUtil::FitViewPort(_imgWidth + marginx, _imgHeight + marginy);
		AttachToFBO(0);
		BindTex();
		ShaderMan::UseShaderMarginCopy(_imgWidth, _imgHeight);
		DrawMargin(_imgWidth + marginx, _imgHeight + marginy);
	}
}

void GLTexImage::ZeroHistoMargin()
{
	ZeroHistoMargin(_imgWidth, _imgHeight);
}

void GLTexImage::ZeroHistoMargin(int width, int height)
{
	int marginx = width & 0x01;
	int marginy = height & 0x01;
	if(marginx >0 || marginy > 0)
	{
		int right = width + marginx;
		int bottom = height + marginy;
		GlobalUtil::FitViewPort(right, bottom);
		AttachToFBO(0);
		ShaderMan::UseShaderZeroPass();
		glBegin(GL_QUADS);
		if(right > width && _texWidth > width)
		{
			glTexCoord2i ( width	,   0   ); 				glVertex2i   ( width	,		0   );
			glTexCoord2i ( width	,   bottom  );			glVertex2i   ( width	,		bottom   );
			glTexCoord2i ( right	,   bottom  ); 			glVertex2i   ( right	,		bottom   );
			glTexCoord2i ( right	,   0   ); 				glVertex2i   ( right	,		0   );
		}
		if(bottom>height && _texHeight > height)
		{
			glTexCoord2i ( 0		,   height ); 		glVertex2i   ( 0		,		height   );
			glTexCoord2i ( 0		,   bottom	);		glVertex2i   ( 0		,		bottom		 );
			glTexCoord2i ( width	,   bottom	); 		glVertex2i   ( width	,		bottom		 );
			glTexCoord2i ( width	,   height	); 		glVertex2i   ( width	,		height	 );
		}
		glEnd();
		glFlush();
	}

}

void GLTexImage::DrawMargin(int right, int bottom)
{
	glBegin(GL_QUADS);
	if(right > _drawWidth)
	{
		glTexCoord2i ( _drawWidth	,   0   ); 				glVertex2i   ( _drawWidth	,		0   );
		glTexCoord2i ( _drawWidth	,   bottom  );			glVertex2i   ( _drawWidth	,		bottom   );
		glTexCoord2i ( right		,   bottom  ); 			glVertex2i   ( right		,		bottom   );
		glTexCoord2i ( right		,   0   ); 				glVertex2i   ( right		,		0   );
	}
	if(bottom>_drawHeight)
	{
		glTexCoord2i ( 0			,   _drawHeight ); 		glVertex2i   ( 0			,		_drawHeight   );
		glTexCoord2i ( 0			,   bottom		);		glVertex2i   ( 0			,		bottom		 );
		glTexCoord2i ( _drawWidth	,   bottom		); 		glVertex2i   ( _drawWidth	,		bottom		 );
		glTexCoord2i ( _drawWidth	,   _drawHeight	); 		glVertex2i   ( _drawWidth	,		_drawHeight	 );
	}
	glEnd();
	glFlush();


}


void GLTexImage::DrawQuadMT4()
{
	int w = _drawWidth, h = _drawHeight;
	glBegin (GL_QUADS);
		glMultiTexCoord2i( GL_TEXTURE0, 0		,   0  );
		glMultiTexCoord2i( GL_TEXTURE1, -1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE2, 1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE3, 0		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE4, 0		,   1  );
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2i( GL_TEXTURE0, 0		,   h  );
		glMultiTexCoord2i( GL_TEXTURE1, -1		,   h  );
		glMultiTexCoord2i( GL_TEXTURE2, 1		,   h );
		glMultiTexCoord2i( GL_TEXTURE3, 0		,   h -1 );
		glMultiTexCoord2i( GL_TEXTURE4, 0		,   h +1 );
		glVertex2i   ( 0			,		h   );


		glMultiTexCoord2i( GL_TEXTURE0, w		,   h  );
		glMultiTexCoord2i( GL_TEXTURE1, w-1		,   h  );
		glMultiTexCoord2i( GL_TEXTURE2, w+1		,   h  );
		glMultiTexCoord2i( GL_TEXTURE3, w		,   h-1  );
		glMultiTexCoord2i( GL_TEXTURE4, w		,   h+1  );
		glVertex2i   ( w	,		h   );

		glMultiTexCoord2i( GL_TEXTURE0, w		,   0  );
		glMultiTexCoord2i( GL_TEXTURE1, w-1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE2, w+1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE3, w		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE4, w		,   1  );
		glVertex2i   ( w	,		0   );
	glEnd ();
	glFlush();
}


void GLTexImage::DrawQuadMT8()
{
	int w = _drawWidth;
	int h = _drawHeight;
	glBegin (GL_QUADS);
		glMultiTexCoord2i( GL_TEXTURE0, 0		,   0  );
		glMultiTexCoord2i( GL_TEXTURE1, -1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE2, 1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE3, 0		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE4, 0		,   1  );
		glMultiTexCoord2i( GL_TEXTURE5, -1		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE6, -1		,   1  );
		glMultiTexCoord2i( GL_TEXTURE7, 1		,   -1  );
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2i( GL_TEXTURE0, 0		,   h    );
		glMultiTexCoord2i( GL_TEXTURE1, -1		,   h    );
		glMultiTexCoord2i( GL_TEXTURE2, 1		,   h    );
		glMultiTexCoord2i( GL_TEXTURE3, 0		,   h  -1  );
		glMultiTexCoord2i( GL_TEXTURE4, 0		,   h  +1  );
		glMultiTexCoord2i( GL_TEXTURE5, -1		,   h  -1  );
		glMultiTexCoord2i( GL_TEXTURE6, -1		,   h  +1  );
		glMultiTexCoord2i( GL_TEXTURE7, 1		,   h  -1  );
		glVertex2i   ( 0			,		h   );


		glMultiTexCoord2i( GL_TEXTURE0, w		,   h    );
		glMultiTexCoord2i( GL_TEXTURE1, w-1		,   h    );
		glMultiTexCoord2i( GL_TEXTURE2, w+1		,   h    );
		glMultiTexCoord2i( GL_TEXTURE3, w		,   h  -1  );
		glMultiTexCoord2i( GL_TEXTURE4, w		,   h  +1  );
		glMultiTexCoord2i( GL_TEXTURE5, w-1		,   h  -1  );
		glMultiTexCoord2i( GL_TEXTURE6, w-1		,   h  +1  );
		glMultiTexCoord2i( GL_TEXTURE7, w+1		,   h  -1  );
		glVertex2i   ( w	,		h   );

		glMultiTexCoord2i( GL_TEXTURE0, w		,   0  );
		glMultiTexCoord2i( GL_TEXTURE1, w-1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE2, w+1		,   0  );
		glMultiTexCoord2i( GL_TEXTURE3, w		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE4, w		,   1  );
		glMultiTexCoord2i( GL_TEXTURE5, w-1		,   -1  );
		glMultiTexCoord2i( GL_TEXTURE6, w-1		,   1  );
		glMultiTexCoord2i( GL_TEXTURE7, w+1		,   -1  );
		glVertex2i   ( w	,		0   );
	glEnd ();
	glFlush();
}




void GLTexImage::DrawImage()
{
	DrawQuad();
}



void GLTexImage::FitTexViewPort()
{
	GlobalUtil::FitViewPort(_drawWidth, _drawHeight);
}

void GLTexImage::FitRealTexViewPort()
{
	GlobalUtil::FitViewPort(_texWidth, _texHeight);
}

void  GLTexImage::AttachToFBO(int i)
{
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, i+GL_COLOR_ATTACHMENT0_EXT, _texTarget, _texID, 0 );
}

void  GLTexImage::DetachFBO(int i)
{
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, i+GL_COLOR_ATTACHMENT0_EXT, _texTarget, 0, 0 );
}


void GLTexImage::DrawQuad(float x1, float x2, float y1, float y2)
{

	glBegin (GL_QUADS);
		glTexCoord2f ( x1	,   y1   ); 	glVertex2f   ( x1	,		y1   );
		glTexCoord2f ( x1	,   y2  );		glVertex2f   ( x1	,		y2   );
 		glTexCoord2f ( x2   ,   y2  ); 		glVertex2f   ( x2	,		y2   );
		glTexCoord2f ( x2	,   y1   ); 	glVertex2f   ( x2	,		y1   );
	glEnd ();
	glFlush();
}

void GLTexImage::TexConvertRGB()
{
	//change 3/22/09
	FrameBufferObject fbo;
	//GlobalUtil::FitViewPort(1, 1);
	FitTexViewPort();

	AttachToFBO(0);
	ShaderMan::UseShaderRGB2Gray();
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	DrawQuad();
	ShaderMan::UnloadProgram();
	DetachFBO(0);
}

void GLTexImage::DrawQuadDS(int scale)
{
	DrawScaledQuad(float(scale));
}

void GLTexImage::DrawQuadUS(int scale)
{
	DrawScaledQuad(1.0f/scale);
}

void GLTexImage::DrawScaledQuad(float texscale)
{

	////the texture coordinate for 0.5 is to + 0.5*texscale
	float to = 0.5f -0.5f * texscale;
	float tx =  _imgWidth*texscale +to;
	float ty = _imgHeight*texscale +to;
	glBegin (GL_QUADS);
		glTexCoord2f ( to	,   to   ); 	glVertex2i   ( 0			,		0   );
		glTexCoord2f ( to	,   ty  );		glVertex2i   ( 0			,		_imgHeight   );
 		glTexCoord2f ( tx	,	ty ); 		glVertex2i   ( _imgWidth	,		_imgHeight   );
		glTexCoord2f ( tx	,   to   ); 	glVertex2i   ( _imgWidth	,		0   );
	glEnd ();
	glFlush();
}


void GLTexImage::DrawQuadReduction(int w , int h)
{
	float to = -0.5f;
	float tx = w*2 +to;
	float ty = h*2 +to;
	glBegin (GL_QUADS);
		glMultiTexCoord2f ( GL_TEXTURE0, to	,	to   );
		glMultiTexCoord2f ( GL_TEXTURE1, to	+1,	to   );
		glMultiTexCoord2f ( GL_TEXTURE2, to	,	to+1  );
		glMultiTexCoord2f ( GL_TEXTURE3, to	+1,	to+1  );
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2f ( GL_TEXTURE0, to	,   ty  );
		glMultiTexCoord2f ( GL_TEXTURE1, to	+1, ty  );
		glMultiTexCoord2f ( GL_TEXTURE2, to	,   ty +1 );
		glMultiTexCoord2f ( GL_TEXTURE3, to	+1, ty +1 );
		glVertex2i   ( 0			,		h   );

 		glMultiTexCoord2f ( GL_TEXTURE0, tx	,	ty );
 		glMultiTexCoord2f ( GL_TEXTURE1, tx	+1,	ty );
 		glMultiTexCoord2f ( GL_TEXTURE2, tx	,	ty +1);
 		glMultiTexCoord2f ( GL_TEXTURE3, tx	+1,	ty +1);

		glVertex2i   ( w	,		h   );

		glMultiTexCoord2f ( GL_TEXTURE0, tx	,   to   );
		glMultiTexCoord2f ( GL_TEXTURE1, tx	+1, to   );
		glMultiTexCoord2f ( GL_TEXTURE2, tx	,   to +1  );
		glMultiTexCoord2f ( GL_TEXTURE3, tx	+1, to +1  );
		glVertex2i   ( w	,		0   );
	glEnd ();

	glFlush();
}


void GLTexImage::DrawQuadReduction()
{
	float to = -0.5f;
	float tx = _drawWidth*2 +to;
	float ty = _drawHeight*2 +to;
	glBegin (GL_QUADS);
		glMultiTexCoord2f ( GL_TEXTURE0, to	,	to   );
		glMultiTexCoord2f ( GL_TEXTURE1, to	+1,	to   );
		glMultiTexCoord2f ( GL_TEXTURE2, to	,	to+1  );
		glMultiTexCoord2f ( GL_TEXTURE3, to	+1,	to+1  );
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2f ( GL_TEXTURE0, to	,   ty  );
		glMultiTexCoord2f ( GL_TEXTURE1, to	+1, ty  );
		glMultiTexCoord2f ( GL_TEXTURE2, to	,   ty +1 );
		glMultiTexCoord2f ( GL_TEXTURE3, to	+1, ty +1 );
		glVertex2i   ( 0			,		_drawHeight   );

 		glMultiTexCoord2f ( GL_TEXTURE0, tx	,	ty );
 		glMultiTexCoord2f ( GL_TEXTURE1, tx	+1,	ty );
 		glMultiTexCoord2f ( GL_TEXTURE2, tx	,	ty +1);
 		glMultiTexCoord2f ( GL_TEXTURE3, tx	+1,	ty +1);

		glVertex2i   ( _drawWidth	,		_drawHeight   );

		glMultiTexCoord2f ( GL_TEXTURE0, tx	,   to   );
		glMultiTexCoord2f ( GL_TEXTURE1, tx	+1, to   );
		glMultiTexCoord2f ( GL_TEXTURE2, tx	,   to +1  );
		glMultiTexCoord2f ( GL_TEXTURE3, tx	+1, to +1  );
		glVertex2i   ( _drawWidth	,		0   );
	glEnd ();

	glFlush();
}

void GLTexPacked::TexConvertRGB()
{
	//update the actual size of daw area
	_drawWidth  = (1 + _imgWidth) >> 1;
	_drawHeight = (1 + _imgHeight) >> 1;
	///
	FrameBufferObject fbo;
	GLuint oldTexID = _texID;
	glGenTextures(1, &_texID);
	glBindTexture(_texTarget, _texID);
	glTexImage2D(_texTarget, 0, _iTexFormat, _texWidth,	_texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE,	NULL);

	//input
	glBindTexture(_texTarget, oldTexID);
	//output
	AttachToFBO(0);
	//program
	ShaderMan::UseShaderRGB2Gray();
	//draw buffer
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	//run
	DrawQuadDS(2);
	ShaderMan::UnloadProgram();

	glDeleteTextures(1, &oldTexID);
	DetachFBO(0);
}


void GLTexPacked::SetImageSize( int width,  int height)
{
	_imgWidth =  width;		_drawWidth = (width + 1) >> 1;
	_imgHeight =  height;	_drawHeight = (height + 1) >> 1;
}

void GLTexPacked::InitTexture( int width,  int height, int clamp_to_edge)
{

	if(_texID && width == _imgWidth && height == _imgHeight ) return;
	if(_texID==0)	glGenTextures(1, &_texID);

	_imgWidth = width;
	_imgHeight = height;
	if(GlobalUtil::_PreciseBorder)
	{
		_texWidth = (width + 2) >> 1;
		_texHeight = (height + 2) >> 1;
	}else
	{
		_texWidth = (width + 1) >> 1;
		_texHeight = (height + 1) >> 1;
	}
	_drawWidth = (width + 1) >> 1;
	_drawHeight = (height + 1) >> 1;

	BindTex();

	if(clamp_to_edge)
	{
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}else
	{
		//out of bound tex read returns 0??
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	}
	glTexParameteri(_texTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(_texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glTexImage2D(_texTarget, 0, _iTexFormat,
		_texWidth, _texHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	UnbindTex();

}


void  GLTexPacked::DrawImage()
{
	float x1 =0,  y1 = 0; //border..
	float x2 = _imgWidth*0.5f +x1;
	float y2 = _imgHeight*0.5f + y1;
	glBegin (GL_QUADS);
		glTexCoord2f ( x1	,   y1  ); 			glVertex2i   ( 0			,		0   );
		glTexCoord2f ( x1	,   y2 );			glVertex2i   ( 0			,		_imgHeight   );
 		glTexCoord2f ( x2   ,   y2  ); 			glVertex2i   ( _imgWidth	,		_imgHeight   );
		glTexCoord2f ( x2	,   y1   ); 		glVertex2i   ( _imgWidth	,		0   );
	glEnd ();
	glFlush();
}

void GLTexPacked::DrawQuadUS(int scale)
{
	int tw =_drawWidth, th = _drawHeight;
	float texscale = 1.0f / scale;
	float x1 = 0.5f - 0.5f*scale, y1 = x1;
	float x2 = tw * texscale + x1;
	float y2 = th * texscale + y1;
	float step = texscale *0.5f;
	glBegin (GL_QUADS);
		glMultiTexCoord2f( GL_TEXTURE0, x1		,   y1      );
		glMultiTexCoord2f( GL_TEXTURE1, x1+step	,   y1      );
		glMultiTexCoord2f( GL_TEXTURE2, x1   	,   y1 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x1+step	,   y1 +step);
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2f( GL_TEXTURE0, x1		,	y2      );
		glMultiTexCoord2f( GL_TEXTURE1, x1+step	,   y2      );
		glMultiTexCoord2f( GL_TEXTURE2, x1   	,   y2 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x1+step	,   y2 +step);
		glVertex2i   ( 0			,	th   );

		glMultiTexCoord2f( GL_TEXTURE0, x2		,   y2      );
		glMultiTexCoord2f( GL_TEXTURE1, x2+step	,   y2      );
		glMultiTexCoord2f( GL_TEXTURE2, x2   	,   y2 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x2+step	,   y2 +step);
		glVertex2i   ( tw	,	th   );

		glMultiTexCoord2f( GL_TEXTURE0, x2		,   y1      );
		glMultiTexCoord2f( GL_TEXTURE1, x2+step	,   y1      );
		glMultiTexCoord2f( GL_TEXTURE2, x2   	,   y1 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x2+step	,   y1 +step);
		glVertex2i   ( tw	,	0   );
	glEnd ();
}

void GLTexPacked::DrawQuadDS(int scale)
{
	int tw = _drawWidth;
	int th = _drawHeight;
	float x1 = 0.5f - 0.5f*scale;
	float x2 = tw * scale + x1;
	float y1 = 0.5f - 0.5f * scale;
	float y2 = th * scale + y1;
	int step = scale / 2;

	glBegin (GL_QUADS);
		glMultiTexCoord2f( GL_TEXTURE0, x1		,   y1      );
		glMultiTexCoord2f( GL_TEXTURE1, x1+step	,   y1      );
		glMultiTexCoord2f( GL_TEXTURE2, x1   	,   y1 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x1+step	,   y1 +step);
		glVertex2i   ( 0			,		0   );

		glMultiTexCoord2f( GL_TEXTURE0, x1		,	y2      );
		glMultiTexCoord2f( GL_TEXTURE1, x1+step	,   y2      );
		glMultiTexCoord2f( GL_TEXTURE2, x1   	,   y2 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x1+step	,   y2 +step);
		glVertex2i   ( 0			,	th   );

		glMultiTexCoord2f( GL_TEXTURE0, x2		,   y2      );
		glMultiTexCoord2f( GL_TEXTURE1, x2+step	,   y2      );
		glMultiTexCoord2f( GL_TEXTURE2, x2   	,   y2 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x2+step	,   y2 +step);
		glVertex2i   ( tw	,	th   );

		glMultiTexCoord2f( GL_TEXTURE0, x2		,   y1      );
		glMultiTexCoord2f( GL_TEXTURE1, x2+step	,   y1      );
		glMultiTexCoord2f( GL_TEXTURE2, x2   	,   y1 +step);
		glMultiTexCoord2f( GL_TEXTURE3, x2+step	,   y1 +step);
		glVertex2i   ( tw	,	0   );
	glEnd ();
}

void GLTexPacked::ZeroHistoMargin()
{
	int marginx = (((_imgWidth  + 3) /4)*4) - _imgWidth;
	int marginy = (((-_imgHeight + 3)/4)*4) - _imgHeight;
	if(marginx >0 || marginy > 0)
	{
		int tw = (_imgWidth + marginx ) >> 1;
		int th = (_imgHeight + marginy ) >> 1;
		tw = min(_texWidth, tw );
		th = min(_texHeight, th);
		GlobalUtil::FitViewPort(tw, th);
		AttachToFBO(0);
		BindTex();
		ShaderMan::UseShaderZeroPass();
		DrawMargin(tw, th, 1, 1);
	}
}


void GLTexPacked::FillMargin(int marginx, int marginy)
{
	//
	marginx = min(marginx, _texWidth * 2 - _imgWidth);
	marginy = min(marginy, _texHeight * 2 - _imgHeight);
	if(marginx >0 || marginy > 0)
	{
		int tw = (_imgWidth + marginx + 1) >> 1;
		int th = (_imgHeight + marginy + 1) >> 1;
		GlobalUtil::FitViewPort(tw, th);
		BindTex();
		AttachToFBO(0);
		ShaderMan::UseShaderMarginCopy(_imgWidth , _imgHeight);
		DrawMargin(tw, th, marginx, marginy);
	}
}
void GLTexPacked::DrawMargin(int right, int bottom, int mx, int my)
{
	int tw = (_imgWidth >>1);
	int th = (_imgHeight >>1);
	glBegin(GL_QUADS);
	if(right>tw && mx)
	{
		glTexCoord2i ( tw	,   0   ); 				glVertex2i   ( tw	,		0   );
		glTexCoord2i ( tw	,   bottom  );			glVertex2i   ( tw	,		bottom   );
		glTexCoord2i ( right,   bottom  ); 			glVertex2i   ( right,		bottom   );
		glTexCoord2i ( right,   0   ); 				glVertex2i   ( right,		0   );
	}
	if(bottom>th && my)
	{
		glTexCoord2i ( 0	,   th  ); 		glVertex2i   ( 0	,		th   );
		glTexCoord2i ( 0	,   bottom	);	glVertex2i   ( 0	,		bottom	 );
		glTexCoord2i ( tw	,   bottom	); 	glVertex2i   ( tw	,		bottom	 );
		glTexCoord2i ( tw	,   th	); 		glVertex2i   ( tw	,		th	 );
	}
	glEnd();
	glFlush();

}


void GLTexImage::UnbindMultiTex(int n)
{
	for(int i = n-1; i>=0; i--)
	{
		glActiveTexture(GL_TEXTURE0+i);
		glBindTexture(_texTarget, 0);
	}
}

template <class Uint> int

#if !defined(_MSC_VER) || _MSC_VER > 1200
GLTexInput::
#endif

DownSamplePixelDataI(unsigned int gl_format, int width, int height, int ds,
									const Uint * pin, Uint * pout)
{
	int step, linestep;
	int i, j;
	int ws = width/ds;
	int hs = height/ds;
	const Uint * line = pin, * p;
	Uint *po = pout;
	switch(gl_format)
	{
	case GL_LUMINANCE:
	case GL_LUMINANCE_ALPHA:
		step = ds * (gl_format == GL_LUMINANCE? 1: 2);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = *p;
			}
		}
		break;
	case GL_RGB:
	case GL_RGBA:
		step = ds * (gl_format == GL_RGB? 3: 4);
		linestep = width * step;

		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				//*po++ = int(p[0]*0.299 + p[1] * 0.587 + p[2]* 0.114 + 0.5);
				*po++ = ((19595*p[0] + 38470*p[1] + 7471*p[2]+ 32768)>>16);
			}
		}
		break;
	case GL_BGR:
	case GL_BGRA:
		step = ds * (gl_format == GL_BGR? 3: 4);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = ((7471*p[0] + 38470*p[1] + 19595*p[2]+ 32768)>>16);
			}
		}
		break;
	default:
		return 0;
	}

	return 1;

}


template <class Uint> int

#if !defined(_MSC_VER) || _MSC_VER > 1200
GLTexInput::
#endif

DownSamplePixelDataI2F(unsigned int gl_format, int width, int height, int ds,
									const Uint * pin, float * pout, int skip)
{
	int step, linestep;
	int i, j;
	int ws = width/ds - skip;
	int hs = height/ds;
	const Uint * line = pin, * p;
	float *po = pout;
    const float factor = (sizeof(Uint) == 1? 255.0f : 65535.0f);
	switch(gl_format)
	{
	case GL_LUMINANCE:
	case GL_LUMINANCE_ALPHA:
		step = ds * (gl_format == GL_LUMINANCE? 1: 2);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = (*p) / factor;
			}
		}
		break;
	case GL_RGB:
	case GL_RGBA:
		step = ds * (gl_format == GL_RGB? 3: 4);
		linestep = width * step;

		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				//*po++ = int(p[0]*0.299 + p[1] * 0.587 + p[2]* 0.114 + 0.5);
				*po++ = ((19595*p[0] + 38470*p[1] + 7471*p[2]) / (65535.0f * factor));
			}
		}
		break;
	case GL_BGR:
	case GL_BGRA:
		step = ds * (gl_format == GL_BGR? 3: 4);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = ((7471*p[0] + 38470*p[1] + 19595*p[2]) / (65535.0f * factor));
			}
		}
		break;
	default:
		return 0;
	}
	return 1;
}

int GLTexInput::DownSamplePixelDataF(unsigned int gl_format, int width, int height, int ds, const float * pin, float * pout, int skip)
{
	int step, linestep;
	int i, j;
	int ws = width/ds - skip;
	int hs = height/ds;
	const float * line = pin, * p;
	float *po = pout;
	switch(gl_format)
	{
	case GL_LUMINANCE:
	case GL_LUMINANCE_ALPHA:
		step = ds * (gl_format == GL_LUMINANCE? 1: 2);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = *p;
			}
		}
		break;
	case GL_RGB:
	case GL_RGBA:
		step = ds * (gl_format == GL_RGB? 3: 4);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = (0.299f*p[0] + 0.587f*p[1] + 0.114f*p[2]);
			}
		}
		break;
	case GL_BGR:
	case GL_BGRA:
		step = ds * (gl_format == GL_BGR? 3: 4);
		linestep = width * step;
		for(i = 0 ; i < hs; i++, line+=linestep)
		{
			for(j = 0, p = line; j < ws; j++, p+=step)
			{
				*po++ = (0.114f*p[0] + 0.587f*p[1] + 0.299f * p[2]);
			}
		}
		break;
	default:
		return 0;
	}

	return 1;

}

int GLTexInput::SetImageData( int width,  int height, const void * data,
							 unsigned int gl_format, unsigned int gl_type )
{
	int simple_format = IsSimpleGlFormat(gl_format, gl_type);//no cpu code to handle other formats
	int ws, hs, done = 1;

	if(_converted_data) {delete [] _converted_data; _converted_data  = NULL; }

	_rgb_converted = 1;
    _data_modified = 0;

	if( simple_format
		&& ( width > _texMaxDim || height > _texMaxDim || GlobalUtil::_PreProcessOnCPU)
		&& GlobalUtil::_octave_min_default >0   )
	{
		_down_sampled = GlobalUtil::_octave_min_default;
		ws = width >> GlobalUtil::_octave_min_default;
		hs = height >> GlobalUtil::_octave_min_default;
	}else
	{
		_down_sampled = 0;
		ws = width;
		hs = height;
	}

	if ( ws > _texMaxDim || hs > _texMaxDim)
	{
		if(simple_format)
		{
			if(GlobalUtil::_verbose) std::cout<<"Automatic down-sampling is used\n";
			do
			{
				_down_sampled ++;
				ws >>= 1;
				hs >>= 1;
			}while(ws > _texMaxDim || hs > _texMaxDim);
		}else
		{
			std::cerr<<"Input images is too big to fit into a texture\n";
			return 0;
		}
	}

	_texWidth = _imgWidth = _drawWidth = ws;
	_texHeight = _imgHeight = _drawHeight = hs;

	if(GlobalUtil::_verbose)
	{
		std::cout<<"Image size :\t"<<width<<"x"<<height<<"\n";
		if(_down_sampled >0) 	std::cout<<"Down sample to \t"<<ws<<"x"<<hs<<"\n";
	}


    if(GlobalUtil::_UseCUDA || GlobalUtil::_UseOpenCL)
    {
        //////////////////////////////////////
        int tWidth = TruncateWidthCU(_imgWidth);
        int skip = _imgWidth - tWidth;
        //skip = 0;
        if(!simple_format)
        {
            std::cerr << "Input format not supported under current settings.\n";
            return 0;
        }else if(_down_sampled > 0 || gl_format != GL_LUMINANCE || gl_type != GL_FLOAT)
        {
		    _converted_data = new float [_imgWidth * _imgHeight];
            if(gl_type == GL_UNSIGNED_BYTE)
		        DownSamplePixelDataI2F(gl_format, width, height, 1<<_down_sampled,
                                        ((const unsigned char*) data), _converted_data, skip);
	        else if(gl_type == GL_UNSIGNED_SHORT)
		        DownSamplePixelDataI2F(gl_format, width, height, 1<<_down_sampled,
                                        ((const unsigned short*) data), _converted_data, skip);
	        else
		        DownSamplePixelDataF(gl_format, width, height, 1<<_down_sampled, (float*)data, _converted_data, skip);
            _rgb_converted = 2;  //indidates a new data copy
            _pixel_data = _converted_data;
        }else
        {
            //Luminance data that doesn't need to down sample
            _rgb_converted = 1;
            _pixel_data = data;
            if(skip > 0)
            {
                for(int i = 1; i < _imgHeight; ++i)
                {
                    float * dst = ((float*)data) + i * tWidth, * src = ((float*)data) + i * _imgWidth;
                    for(int j = 0; j < tWidth; ++j) *dst++ = * src++;
                }
            }
        }
        _texWidth = _imgWidth = _drawWidth = tWidth;
        _data_modified = 1;
    }else
    {
	    if(_texID ==0)		glGenTextures(1, &_texID);
	    glBindTexture(_texTarget, _texID);
	    CheckErrorsGL("glBindTexture");
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glPixelStorei(GL_UNPACK_ALIGNMENT , 1);

	    if(simple_format && ( _down_sampled> 0 || (gl_format != GL_LUMINANCE && GlobalUtil::_PreProcessOnCPU) ))
	    {

		    if(gl_type == GL_UNSIGNED_BYTE)
		    {
			    unsigned char * newdata = new unsigned char [_imgWidth * _imgHeight];
			    DownSamplePixelDataI(gl_format, width, height, 1<<_down_sampled, ((const unsigned char*) data), newdata);
			    glTexImage2D(_texTarget, 0, GL_LUMINANCE32F_ARB, //internal format changed
                    _imgWidth, _imgHeight, 0,
					GL_LUMINANCE, GL_UNSIGNED_BYTE, newdata);
			    delete[] newdata;
		    }else if(gl_type == GL_UNSIGNED_SHORT)
		    {
			    unsigned short * newdata = new unsigned short [_imgWidth * _imgHeight];
			    DownSamplePixelDataI(gl_format, width, height, 1<<_down_sampled, ((const unsigned short*) data), newdata);

			    glTexImage2D(_texTarget, 0, GL_LUMINANCE32F_ARB,   //internal format changed
                    _imgWidth, _imgHeight, 0,
					GL_LUMINANCE, GL_UNSIGNED_SHORT, newdata);
			    delete[] newdata;
		    }else if(gl_type == GL_FLOAT)
		    {
			    float * newdata = new float [_imgWidth * _imgHeight];
			    DownSamplePixelDataF(gl_format, width, height, 1<<_down_sampled, (float*)data, newdata);
			    glTexImage2D(_texTarget, 0, GL_LUMINANCE32F_ARB, //internal format changed
                    _imgWidth, _imgHeight, 0,
					GL_LUMINANCE, GL_FLOAT, newdata);
			    delete[] newdata;
		    }else
		    {
			    //impossible
			    done = 0;
			    _rgb_converted = 0;
		    }
		    GlobalUtil::FitViewPort(1, 1);  //this used to be necessary
	    }else
	    {
		    //ds must be 0 here if not simpleformat
		    if(gl_format == GL_LUMINANCE || gl_format == GL_LUMINANCE_ALPHA)
            {
                //use one channel internal format if data is intensity image
                glTexImage2D(_texTarget, 0, GL_LUMINANCE32F_ARB,
                _imgWidth, _imgHeight, 0, gl_format,	gl_type, data);
			    GlobalUtil::FitViewPort(1, 1); //this used to be necessary
            }
		    else
            {
	    	    //convert RGB 2 GRAY if needed
                glTexImage2D(_texTarget, 0,  _iTexFormat, _imgWidth, _imgHeight, 0, gl_format, gl_type, data);
                if(ShaderMan::HaveShaderMan())
			        TexConvertRGB();
		        else
			        _rgb_converted = 0;  //In CUDA mode, the conversion will be done by CUDA kernel
            }
	    }
	    UnbindTex();
    }
	return done;
}


GLTexInput::~GLTexInput()
{
    if(_converted_data) delete [] _converted_data;
}


int GLTexInput::LoadImageFile(char *imagepath, int &w, int &h )
{
#ifndef SIFTGPU_NO_DEVIL
    static int devil_loaded = 0;
	unsigned int imID;
	int done = 1;

    if(devil_loaded == 0)
    {
	    ilInit();
	    ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
	    ilEnable(IL_ORIGIN_SET);
        devil_loaded = 1;
    }

	///
	ilGenImages(1, &imID);
	ilBindImage(imID);

	if(ilLoadImage(imagepath))
	{
		w = ilGetInteger(IL_IMAGE_WIDTH);
		h = ilGetInteger(IL_IMAGE_HEIGHT);
		int ilformat = ilGetInteger(IL_IMAGE_FORMAT);

		if(SetImageData(w, h, ilGetData(), ilformat, GL_UNSIGNED_BYTE)==0)
		{
			done =0;
		}else 	if(GlobalUtil::_verbose)
		{
			std::cout<<"Image loaded :\t"<<imagepath<<"\n";
		}

	}else
	{
		std::cerr<<"Unable to open image [code = "<<ilGetError()<<"]\n";
		done = 0;
	}

	ilDeleteImages(1, &imID);

	return done;
#else
	FILE * file = fopen(imagepath, "rb"); if (file ==NULL) return 0;

	char buf[8];	int  width, height, cn, g, done = 1;

	if(fscanf(file, "%s %d %d %d", buf, &width, &height, &cn )<4 ||  cn > 255 || width < 0 || height < 0)
	{
		fclose(file);
        std::cerr << "ERROR: fileformat not supported\n";
		return 0;
	}else
    {
        w = width;
        h = height;
    }
    unsigned char * data = new unsigned char[width * height];
	unsigned char * pixels = data;
	if (strcmp(buf, "P5")==0 )
	{
		fscanf(file, "%c",buf);//skip one byte
		fread(pixels, 1, width*height, file);
	}else if (strcmp(buf, "P2")==0 )
	{
		for (int i = 0 ; i< height; i++)
		{
			for ( int j = 0; j < width; j++)
			{
				fscanf(file, "%d", &g);
				*pixels++ = (unsigned char) g;
			}
		}
	}else if (strcmp(buf, "P6")==0 )
	{
		fscanf(file, "%c", buf);//skip one byte
		int j, num = height*width;
        unsigned char buf[3];
		for ( j =0 ; j< num; j++)
		{
			fread(buf,1,3, file);
			*pixels++=int(0.10454f* buf[2]+0.60581f* buf[1]+0.28965f* buf[0]);
		}
	}else if (strcmp(buf, "P3")==0 )
	{
		int r, g, b;
		int i , num =height*width;
		for ( i = 0 ; i< num; i++)
		{
			fscanf(file, "%d %d %d", &r, &g, &b);
			*pixels++ = int(0.10454f* b+0.60581f* g+0.28965f* r);
		}

	}else
	{
        std::cerr << "ERROR: fileformat not supported\n";
		done = 0;
	}
    if(done)    SetImageData(width, height, data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
	fclose(file);
    delete data;
    if(GlobalUtil::_verbose && done) std::cout<< "Image loaded :\t" << imagepath << "\n";
	return 1;
#endif
}

int GLTexImage::CopyToPBO(GLuint pbo, int width, int height, GLenum format)
{
    /////////
    if(format != GL_RGBA && format != GL_LUMINANCE) return 0;

	FrameBufferObject fbo;
    GLint bsize, esize = width * height * sizeof(float) * (format == GL_RGBA ? 4 : 1);
	AttachToFBO(0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	if(bsize < esize)
	{
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize,	NULL, GL_STATIC_DRAW_ARB);
		glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
	}
	if(bsize >= esize)
	{
		glReadPixels(0, 0, width, height, format, GL_FLOAT, 0);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	DetachFBO(0);

	return bsize >= esize;
}

void GLTexImage::SaveToASCII(const char* path)
{
	vector<float> buf(GetImgWidth() * GetImgHeight() * 4);
	FrameBufferObject fbo;
	AttachToFBO(0);
	glReadPixels(0, 0, GetImgWidth(), GetImgHeight(), GL_RGBA, GL_FLOAT, &buf[0]);
	ofstream out(path);

	for(int i = 0, idx = 0; i < GetImgHeight(); ++i)
	{
		for(int j = 0; j < GetImgWidth(); ++j, idx += 4)
		{
			out << i << " " << j << " " << buf[idx] << " " << buf[idx + 1] << " "
				<< buf[idx + 2] <<  " " << buf[idx + 3] << "\n";
		}
	}
}


void GLTexInput::VerifyTexture()
{
    //for CUDA or OpenCL the texture is not generated by default
    if(!_data_modified) return;
    if(_pixel_data== NULL) return;
    InitTexture(_imgWidth, _imgHeight);
    BindTex();
	glTexImage2D(   _texTarget, 0, GL_LUMINANCE32F_ARB, //internal format changed
                    _imgWidth, _imgHeight, 0,
					GL_LUMINANCE, GL_FLOAT, _pixel_data);
    UnbindTex();
    _data_modified = 0;
}

void GLTexImage::CopyFromPBO(GLuint pbo, int width, int height, GLenum format)
{
	InitTexture(max(width, _texWidth), max(height, _texHeight));
	SetImageSize(width, height);
	if(width > 0 && height > 0)
	{
		BindTex();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, width, height, format, GL_FLOAT, 0);
        GlobalUtil::CheckErrorsGL("GLTexImage::CopyFromPBO->glTexSubImage2D");
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		UnbindTex();
	}
}

