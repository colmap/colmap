////////////////////////////////////////////////////////////////////////////
//	File:		ProgramGPU.h
//	Author:		Changchang Wu
//	Description : Based class for GPU programs
//		ProgramGPU:	base class of ProgramGLSL
//		FilterProgram:	base class of FilterGLSL, FilterPKSL
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


#ifndef _PROGRAM_GPU_H
#define _PROGRAM_GPU_H

////////////////////////////////////////////////////////////////////////////
//class		ProgramGPU
//description:	pure virtual class
//				provides a common interface for shader programs
///////////////////////////////////////////////////////////////////////////
class ProgramGPU
{
public:
	//use a gpu program
	virtual int     UseProgram() = 0;
    virtual void*   GetProgramID() = 0;
	//not used
	virtual ~ProgramGPU(){};
};

///////////////////////////////////////////////////////////////////////////
//class			FilterProgram
///////////////////////////////////////////////////////////////////////////
class  FilterProgram
{
public:
	ProgramGPU*  s_shader_h;
	ProgramGPU*  s_shader_v;
	int			 _size;
	int			 _id; 
public:
    FilterProgram()          {  s_shader_h = s_shader_v = NULL; _size = _id = 0; }
    virtual ~FilterProgram() {	if(s_shader_h) delete s_shader_h;	if(s_shader_v) delete s_shader_v;}
};

#endif

