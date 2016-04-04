////////////////////////////////////////////////////////////////////////////
//	File:		FrameBufferObject.h
//	Author:		Changchang Wu
//	Description : interface for the FrameBufferObject class.
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


#if !defined(_FRAME_BUFFER_OBJECT_H)
#define _FRAME_BUFFER_OBJECT_H

class FrameBufferObject  
{
	static GLuint	GlobalFBO;   //not thread-safe
	GLuint _fboID;
public:
	static int		UseSingleFBO;
public:
	static void DeleteGlobalFBO();
	static void UnattachTex(GLenum attachment);
	static void UnbindFBO();
	static void AttachDepthTexture(GLenum textureTarget, GLuint texID);
	static void AttachTexture( GLenum textureTarget, GLenum attachment, GLuint texID);
	static void AttachRenderBuffer(GLenum attachment,  GLuint buffID  );
	static void UnattachRenderBuffer(GLenum attachment);
public:
	void BindFBO();
	FrameBufferObject(int autobind = 1);
	~FrameBufferObject();

};

#endif 
