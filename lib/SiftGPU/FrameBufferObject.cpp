////////////////////////////////////////////////////////////////////////////
//	File:		FrameBufferObject.cpp
//	Author:		Changchang Wu
//	Description : Implementation of FrameBufferObject Class
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
#include <stdlib.h>
#include "GlobalUtil.h"
#include "FrameBufferObject.h"

//whether use only one FBO globally
int		FrameBufferObject::UseSingleFBO=1;
GLuint	FrameBufferObject::GlobalFBO=0;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

FrameBufferObject::FrameBufferObject(int autobind)
{
	if(UseSingleFBO && GlobalFBO)
	{
		_fboID = GlobalFBO;
	}else
	{
		glGenFramebuffersEXT(1, &_fboID);
		if(UseSingleFBO )GlobalFBO = _fboID;
	}
	if(autobind )   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fboID);
}

FrameBufferObject::~FrameBufferObject()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	if(!UseSingleFBO )
	{
		glDeleteFramebuffersEXT (1,&_fboID);
	}
}

void FrameBufferObject::DeleteGlobalFBO()
{
	if(UseSingleFBO)
	{
		glDeleteFramebuffersEXT (1,&GlobalFBO);
		GlobalFBO = 0;
	}
}

void FrameBufferObject::AttachDepthTexture(GLenum textureTarget, GLuint texID)
{

  glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, textureTarget, texID, 0);
}

void FrameBufferObject::AttachTexture(GLenum textureTarget, GLenum attachment, GLuint texId)
{
  glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, attachment, textureTarget, texId, 0);
}

void FrameBufferObject::BindFBO()
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fboID);
}

void FrameBufferObject::UnbindFBO()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void FrameBufferObject::UnattachTex(GLenum attachment)
{
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_2D, 0, 0 );
}

void FrameBufferObject::AttachRenderBuffer(GLenum attachment, GLuint buffID)
{
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, buffID);

}

void FrameBufferObject:: UnattachRenderBuffer(GLenum attachment)
{
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, 0);
}

