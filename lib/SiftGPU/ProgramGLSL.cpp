////////////////////////////////////////////////////////////////////////////
//	File:		ProgramGLSL.cpp
//	Author:		Changchang Wu
//	Description : GLSL related classes
//		class ProgramGLSL		A simple wrapper of GLSL programs
//		class ShaderBagGLSL		GLSL shaders for SIFT
//		class FilterGLSL		GLSL gaussian filters for SIFT
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
#include <string.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

#include "GlobalUtil.h"
#include "ProgramGLSL.h"
#include "GLTexImage.h"
#include "ShaderMan.h"
#include "SiftGPU.h"

ProgramGLSL::ShaderObject::ShaderObject(int shadertype, const char * source, int filesource)
{


	_type = shadertype;
	_compiled = 0;


	_shaderID = glCreateShader(shadertype);
	if(_shaderID == 0) return;

	if(source)
	{

		GLint				code_length;
		if(filesource ==0)
		{
			const char* code  = source;
			code_length = (GLint) strlen(code);
			glShaderSource(_shaderID, 1, (const char **) &code, &code_length);
		}else
		{
			char * code;
			if((code_length= ReadShaderFile(source, code)) ==0) return;
			glShaderSource(_shaderID, 1, (const char **) &code, &code_length);
			delete code;
		}

		glCompileShader(_shaderID);

		CheckCompileLog();

		if(!_compiled) 		std::cout << source;
	}




}

int ProgramGLSL::ShaderObject::ReadShaderFile(const char *sourcefile,  char*& code )
{
	code = NULL;
	FILE * file;
	int    len=0;

	if(sourcefile == NULL) return 0;

	file = fopen(sourcefile,"rt");
	if(file == NULL) return 0;


	fseek(file, 0, SEEK_END);
	len = ftell(file);
	rewind(file);
	if(len >1)
	{
		code = new  char[len+1];
		fread(code, sizeof( char), len, file);
		code[len] = 0;
	}else
	{
		len = 0;
	}

	fclose(file);

	return len;

}

void ProgramGLSL::ShaderObject::CheckCompileLog()
{

	GLint status;
	glGetShaderiv(_shaderID, GL_COMPILE_STATUS, &status);
	_compiled = (status ==GL_TRUE);

	if(_compiled == 0)	PrintCompileLog(std::cout);


}

ProgramGLSL::ShaderObject::~ShaderObject()
{
	if(_shaderID)	glDeleteShader(_shaderID);

}

int ProgramGLSL::ShaderObject::IsValidFragmentShader()
{
	return _type == GL_FRAGMENT_SHADER && _shaderID && _compiled;
}

int  ProgramGLSL::ShaderObject::IsValidVertexShader()
{
	return _type == GL_VERTEX_SHADER && _shaderID && _compiled;
}


void ProgramGLSL::ShaderObject::PrintCompileLog(ostream&os)
{
	GLint len = 0;

	glGetShaderiv(_shaderID, GL_INFO_LOG_LENGTH , &len);
	if(len <=1) return;

	char * compileLog = new char[len+1];
	if(compileLog == NULL) return;

	glGetShaderInfoLog(_shaderID, len, &len, compileLog);


	os<<"Compile Log\n"<<compileLog<<"\n";

	delete[] compileLog;
}


ProgramGLSL::ProgramGLSL()
{
	_linked = 0;
	_TextureParam0 = -1;
	_programID = glCreateProgram();
}
ProgramGLSL::~ProgramGLSL()
{
	if(_programID)glDeleteProgram(_programID);
}
void ProgramGLSL::AttachShaderObject(ShaderObject &shader)
{
	if(_programID  && shader.IsValidShaderObject())
		glAttachShader(_programID, shader.GetShaderID());
}
void ProgramGLSL::DetachShaderObject(ShaderObject &shader)
{
	if(_programID  && shader.IsValidShaderObject())
		glDetachShader(_programID, shader.GetShaderID());
}
int ProgramGLSL::LinkProgram()
{
	_linked = 0;

	if(_programID==0) return 0;

	glLinkProgram(_programID);

	CheckLinkLog();

//	GlobalUtil::StartTimer("100 link test");
//	for(int i = 0; i<100; i++) glLinkProgram(_programID);
//	GlobalUtil::StopTimer();

	return _linked;
}

void ProgramGLSL::CheckLinkLog()
{
	GLint status;
	glGetProgramiv(_programID, GL_LINK_STATUS, &status);

	_linked = (status == GL_TRUE);

}


int ProgramGLSL::ValidateProgram()
{
	if(_programID && _linked)
	{
///		GLint status;
//		glValidateProgram(_programID);
//		glGetProgramiv(_programID, GL_VALIDATE_STATUS, &status);
//		return status == GL_TRUE;
		return 1;
	}
	else
		return 0;
}

void ProgramGLSL::PrintLinkLog(std::ostream &os)
{
	GLint len = 0;

	glGetProgramiv(_programID, GL_INFO_LOG_LENGTH , &len);
	if(len <=1) return;

	char* linkLog = new char[len+1];
	if(linkLog == NULL) return;

	glGetProgramInfoLog(_programID, len, &len, linkLog);

	linkLog[len] = 0;

	if(strstr(linkLog, "failed"))
	{
		os<<linkLog + (linkLog[0] == ' '? 1:0)<<"\n";
		_linked = 0;
	}

	delete[] linkLog;
}

int ProgramGLSL::UseProgram()
{
	if(ValidateProgram())
	{
		glUseProgram(_programID);
		if (_TextureParam0 >= 0) glUniform1i(_TextureParam0, 0);
		return true;
	}
	else
	{
		return false;
	}
}


ProgramGLSL::ProgramGLSL(const char *frag_source)
{
	_linked = 0;
	_programID = glCreateProgram();
	_TextureParam0 = -1;
	ShaderObject shader(GL_FRAGMENT_SHADER, frag_source);

	if(shader.IsValidFragmentShader())
	{
		AttachShaderObject(shader);
		LinkProgram();

		if(!_linked)
		{
			//shader.PrintCompileLog(std::cout);
			PrintLinkLog(std::cout);
		} else
		{
			_TextureParam0 = glGetUniformLocation(_programID, "tex");
		}
	}else
	{
		_linked = 0;
	}

}

/*
ProgramGLSL::ProgramGLSL(char*frag_source, char * vert_source)
{
	_used = 0;
	_linked = 0;
	_programID = glCreateProgram();
	ShaderObject shader(GL_FRAGMENT_SHADER, frag_source);
	ShaderObject vertex_shader(GL_VERTEX_SHADER, vert_source);
	AttachShaderObject(shader);
	AttachShaderObject(vertex_shader);
	LinkProgram();
	if(!_linked)
	{
		shader.PrintCompileLog(std::cout);
		vertex_shader.PrintCompileLog(std::cout);
		PrintLinkLog(std::cout);
		std::cout<<vert_source;
		std::cout<<frag_source;
	}

}
*/



void ProgramGLSL::ReLink()
{
	glLinkProgram(_programID);
}

int ProgramGLSL::IsNative()
{
	return _linked;
}

FilterGLSL::FilterGLSL(float sigma)
{
	//pixel inside 3*sigma box
	int sz = int( ceil( GlobalUtil::_FilterWidthFactor * sigma -0.5) ) ;//
	int width = 2*sz + 1;

	//filter size truncation
	if(GlobalUtil::_MaxFilterWidth >0 && width > GlobalUtil::_MaxFilterWidth)
	{
		std::cout<<"Filter size truncated from "<<width<<" to "<<GlobalUtil::_MaxFilterWidth<<endl;
		sz = GlobalUtil::_MaxFilterWidth>>1;
		width = 2 * sz + 1;
	}

	int i;
	float * kernel = new float[width];
	float   rv = 1.0f/(sigma*sigma);
	float   v, ksum =0;

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i)
	{
		kernel[i+sz] =  v = exp(-0.5f * i * i *rv) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f / ksum;
	for(i = 0; i< width ;i++) kernel[i]*=rv;
	//

    MakeFilterProgram(kernel, width);

	_size = sz;

	delete[] kernel;
    if(GlobalUtil::_verbose && GlobalUtil::_timingL) std::cout<<"Filter: sigma = "<<sigma<<", size = "<<width<<"x"<<width<<endl;
}


void FilterGLSL::MakeFilterProgram(float kernel[], int width)
{
	if(GlobalUtil::_usePackedTex)
	{
		s_shader_h = CreateFilterHPK(kernel, width);
		s_shader_v = CreateFilterVPK(kernel, width);
	}else
	{
		s_shader_h = CreateFilterH(kernel, width);
		s_shader_v = CreateFilterV(kernel, width);
	}
}

ProgramGPU* FilterGLSL::CreateFilterH(float kernel[], int width)
{
	ostringstream out;
	out<<setprecision(8);

	out<<  "uniform sampler2DRect tex;";
	out<< "\nvoid main(void){ float intensity = 0.0 ;  vec2 pos;\n";

    int half_width = width / 2;
	for(int i = 0; i< width; i++)
	{
		if(i == half_width)
		{

			out<<"float or = texture2DRect(tex, gl_TexCoord[0].st).r;\n";
			out<<"intensity+= or * "<<kernel[i]<<";\n";
		}else
		{
			out<<"pos = gl_TexCoord[0].st + vec2(float("<< (i - half_width) <<") , 0);\n";
			out<<"intensity+= "<<kernel[i]<<"*texture2DRect(tex, pos).r;\n";
		}
	}

	//copy original data to red channel
	out<<"gl_FragColor.r = or;\n";
	out<<"gl_FragColor.b  = intensity;}\n"<<'\0';

	return new ProgramGLSL(out.str().c_str());
}


ProgramGPU* FilterGLSL::CreateFilterV(float kernel[], int height)
{
	ostringstream out;
	out<<setprecision(8);

	out<<  "uniform sampler2DRect tex;";
	out<< "\nvoid main(void){ float intensity = 0.0;vec2 pos; \n";
    int half_height = height / 2;
	for(int i = 0; i< height; i++)
	{

		if(i == half_height)
		{
			out<<"vec2 orb = texture2DRect(tex, gl_TexCoord[0].st).rb;\n";
			out<<"intensity+= orb.y * "<<kernel[i]<<";\n";

		}else
		{
			out<<"pos = gl_TexCoord[0].st + vec2(0, float("<<(i - half_height) <<") );\n";
			out<<"intensity+= texture2DRect(tex, pos).b * "<<kernel[i]<<";\n";
		}

	}

	out<<"gl_FragColor.b = orb.y;\n";
	out<<"gl_FragColor.g = intensity - orb.x;\n"; // difference of gaussian..
	out<<"gl_FragColor.r = intensity;}\n"<<'\0';

//	std::cout<<buffer<<endl;
	return new ProgramGLSL(out.str().c_str());
}



ProgramGPU* FilterGLSL::CreateFilterHPK(float kernel[], int width)
{
	//both h and v are packed...
	int i, j , xw, xwn;

	int halfwidth  = width >>1;
	float * pf = kernel + halfwidth;
	int nhpixel = (halfwidth+1)>>1;	//how many neighbour pixels need to be looked up
	int npixel  = (nhpixel<<1)+1;//
	float weight[3];
	ostringstream out;;
	out<<setprecision(8);

	out<<  "uniform sampler2DRect tex;";
	out<< "\nvoid main(void){ vec4 result = vec4(0, 0, 0, 0);\n";
	///use multi texture coordinate because nhpixels can be at most 3
	out<<"vec4 pc; vec2 coord; \n";
	for( i = 0 ; i < npixel ; i++)
	{
		out<<"coord = gl_TexCoord[0].xy + vec2(float("<<i-nhpixel<<"),0);\n";
		out<<"pc=texture2DRect(tex, coord);\n";
		if(GlobalUtil::_PreciseBorder)		out<<"if(coord.x < 0.0) pc = pc.rrbb;\n";
		//for each sub-pixel j  in center, the weight of sub-pixel k
		xw = (i - nhpixel)*2;
		for( j = 0; j < 3; j++)
		{
			xwn = xw  + j  -1;
			weight[j] = xwn < -halfwidth || xwn > halfwidth? 0 : pf[xwn];
		}
		if(weight[1] == 0.0)
		{
			out<<"result += vec4("<<weight[2]<<","<<weight[0]<<","<<weight[2]<<","<<weight[0]<<")*pc.grab;\n";
		}
		else
		{
			out<<"result += vec4("<<weight[1]<<", "<<weight[0]<<", "<<weight[1]<<", "<<weight[0]<<")*pc.rrbb;\n";
			out<<"result += vec4("<<weight[2]<<", "<<weight[1]<<", "<<weight[2]<<", "<<weight[1]<<")*pc.ggaa;\n";
		}

	}
	out<<"gl_FragColor = result;}\n"<<'\0';

	return new ProgramGLSL(out.str().c_str());


}


ProgramGPU* FilterGLSL::CreateFilterVPK(float kernel[], int height)
{

	//both h and v are packed...
	int i, j, yw, ywn;

	int halfh  = height >>1;
	float * pf = kernel + halfh;
	int nhpixel = (halfh+1)>>1;	//how many neighbour pixels need to be looked up
	int npixel  = (nhpixel<<1)+1;//
	float weight[3];
	ostringstream out;;
	out<<setprecision(8);

	out<<  "uniform sampler2DRect tex;";
	out<< "\nvoid main(void){ vec4 result = vec4(0, 0, 0, 0);\n";
	///use multi texture coordinate because nhpixels can be at most 3
	out<<"vec4 pc; vec2 coord;\n";
	for( i = 0 ; i < npixel ; i++)
	{
		out<<"coord = gl_TexCoord[0].xy + vec2(0, float("<<i-nhpixel<<"));\n";
		out<<"pc=texture2DRect(tex, coord);\n";
		if(GlobalUtil::_PreciseBorder)	out<<"if(coord.y < 0.0) pc = pc.rgrg;\n";

		//for each sub-pixel j  in center, the weight of sub-pixel k
		yw = (i - nhpixel)*2;
		for( j = 0; j < 3; j++)
		{
			ywn = yw + j  -1;
			weight[j] = ywn < -halfh || ywn > halfh? 0 : pf[ywn];
		}
		if(weight[1] == 0.0)
		{
			out<<"result += vec4("<<weight[2]<<","<<weight[2]<<","<<weight[0]<<","<<weight[0]<<")*pc.barg;\n";
		}else
		{
			out<<"result += vec4("<<weight[1]<<","<<weight[1]<<","<<weight[0]<<","<<weight[0]<<")*pc.rgrg;\n";
			out<<"result += vec4("<<weight[2]<<","<<weight[2]<<","<<weight[1]<<","<<weight[1]<<")*pc.baba;\n";
		}
	}
	out<<"gl_FragColor = result;}\n"<<'\0';

	return new ProgramGLSL(out.str().c_str());
}



ShaderBag::ShaderBag()
{
	s_debug = 0;
	s_orientation = 0;
	s_display_gaussian = 0;
	s_display_dog = 0;
	s_display_grad = 0;
	s_display_keys = 0;
	s_sampling = 0;
	s_grad_pass = 0;
	s_dog_pass = 0;
	s_keypoint = 0;
	s_genlist_init_tight = 0;
	s_genlist_init_ex = 0;
	s_genlist_histo = 0;
	s_genlist_start = 0;
	s_genlist_step = 0;
	s_genlist_end = 0;
	s_vertex_list = 0;
	s_descriptor_fp = 0;
	s_margin_copy = 0;
    ////////////
    f_gaussian_skip0 = NULL;
    f_gaussian_skip1 = NULL;
    f_gaussian_step = NULL;
    _gaussian_step_num = 0;

}

ShaderBag::~ShaderBag()
{
	if(s_debug)delete s_debug;
	if(s_orientation)delete s_orientation;
	if(s_display_gaussian)delete s_display_gaussian;
	if(s_display_dog)delete s_display_dog;
	if(s_display_grad)delete s_display_grad;
	if(s_display_keys)delete s_display_keys;
	if(s_sampling)delete s_sampling;
	if(s_grad_pass)delete s_grad_pass;
	if(s_dog_pass) delete s_dog_pass;
	if(s_keypoint)delete s_keypoint;
	if(s_genlist_init_tight)delete s_genlist_init_tight;
	if(s_genlist_init_ex)delete s_genlist_init_ex;
	if(s_genlist_histo)delete s_genlist_histo;
	if(s_genlist_start)delete s_genlist_start;
	if(s_genlist_step)delete s_genlist_step;
	if(s_genlist_end)delete s_genlist_end;
	if(s_vertex_list)delete s_vertex_list;
	if(s_descriptor_fp)delete s_descriptor_fp;
	if(s_margin_copy) delete s_margin_copy;

    //////////////////////////////////////////////
    if(f_gaussian_skip1) delete f_gaussian_skip1;

    for(unsigned int i = 0; i < f_gaussian_skip0_v.size(); i++)
    {
	    if(f_gaussian_skip0_v[i]) delete f_gaussian_skip0_v[i];
    }
    if(f_gaussian_step && _gaussian_step_num > 0)
    {
	    for(int i = 0; i< _gaussian_step_num; i++)
	    {
		    delete f_gaussian_step[i];
	    }
	    delete[] f_gaussian_step;
    }
}


void ShaderBag::SelectInitialSmoothingFilter(int octave_min, SiftParam&param)
{
    float sigma = param.GetInitialSmoothSigma(octave_min);
    if(sigma == 0)
    {
       f_gaussian_skip0 = NULL;
    }else
    {
	    for(unsigned int i = 0; i < f_gaussian_skip0_v.size(); i++)
	    {
		    if(f_gaussian_skip0_v[i]->_id == octave_min)
		    {
			    f_gaussian_skip0 = f_gaussian_skip0_v[i];
			    return ;
		    }
	    }
	    FilterGLSL * filter = new FilterGLSL(sigma);
	    filter->_id = octave_min;
	    f_gaussian_skip0_v.push_back(filter);
	    f_gaussian_skip0 = filter;
    }
}

void ShaderBag::CreateGaussianFilters(SiftParam&param)
{
	if(param._sigma_skip0>0.0f)
	{
        FilterGLSL * filter;
		f_gaussian_skip0 = filter = new FilterGLSL(param._sigma_skip0);
		filter->_id = GlobalUtil::_octave_min_default;
		f_gaussian_skip0_v.push_back(filter);
	}
	if(param._sigma_skip1>0.0f)
	{
		f_gaussian_skip1 = new FilterGLSL(param._sigma_skip1);
	}

	f_gaussian_step = new FilterProgram*[param._sigma_num];
	for(int i = 0; i< param._sigma_num; i++)
	{
		f_gaussian_step[i] =  new FilterGLSL(param._sigma[i]);
	}
    _gaussian_step_num = param._sigma_num;
}


void ShaderBag::LoadDynamicShaders(SiftParam& param)
{
    LoadKeypointShader(param._dog_threshold, param._edge_threshold);
    LoadGenListShader(param._dog_level_num, 0);
    CreateGaussianFilters(param);
}


void ShaderBagGLSL::LoadFixedShaders()
{


	s_gray = new ProgramGLSL(
		"uniform sampler2DRect tex; void main(void){\n"
		"float intensity = dot(vec3(0.299, 0.587, 0.114), texture2DRect(tex, gl_TexCoord[0].st ).rgb);\n"
		"gl_FragColor = vec4(intensity, intensity, intensity, 1.0);}");


	s_debug = new ProgramGLSL( "void main(void){gl_FragColor.rg =  gl_TexCoord[0].st;}");


	s_sampling = new ProgramGLSL(
		"uniform sampler2DRect tex; void main(void){gl_FragColor.rg= texture2DRect(tex, gl_TexCoord[0].st).rg;}");

	//
	s_grad_pass = new ProgramGLSL(
	"uniform sampler2DRect tex; void main ()\n"
	"{\n"
	"	vec4 v1, v2, gg;\n"
	"	vec4 cc  = texture2DRect(tex, gl_TexCoord[0].xy);\n"
	"	gg.x = texture2DRect(tex, gl_TexCoord[1].xy).r;\n"
	"	gg.y = texture2DRect(tex, gl_TexCoord[2].xy).r;\n"
	"	gg.z = texture2DRect(tex, gl_TexCoord[3].xy).r;\n"
	"	gg.w = texture2DRect(tex, gl_TexCoord[4].xy).r;\n"
	"	vec2 dxdy = (gg.yw - gg.xz); \n"
	"	float grad = 0.5*length(dxdy);\n"
	"	float theta = grad==0.0? 0.0: atan(dxdy.y, dxdy.x);\n"
	"	gl_FragData[0] = vec4(cc.rg, grad, theta);\n"
	"}\n\0");

	ProgramGLSL * program;
	s_margin_copy = program = new ProgramGLSL(
	"uniform sampler2DRect tex; uniform vec2 truncate;\n"
	"void main(){ gl_FragColor = texture2DRect(tex, min(gl_TexCoord[0].xy, truncate)); }");

	_param_margin_copy_truncate = glGetUniformLocation(*program, "truncate");


	GlobalUtil::_OrientationPack2 = 0;
	LoadOrientationShader();

	if(s_orientation == NULL)
	{
		//Load a simplified version if the right version is not supported
		s_orientation = program =  new ProgramGLSL(
		"uniform sampler2DRect tex; uniform sampler2DRect oTex;\n"
	"	uniform float size; void main(){\n"
	"	vec4 cc = texture2DRect(tex, gl_TexCoord[0].st);\n"
	"	vec4 oo = texture2DRect(oTex, cc.rg);\n"
	"	gl_FragColor.rg = cc.rg;\n"
	"	gl_FragColor.b = oo.a;\n"
	"	gl_FragColor.a = size;}");

		_param_orientation_gtex = glGetUniformLocation(*program, "oTex");
		_param_orientation_size = glGetUniformLocation(*program, "size");
		GlobalUtil::_MaxOrientation = 0;
		GlobalUtil::_FullSupported = 0;
		std::cerr<<"Orientation simplified on this hardware"<<endl;
	}

	if(GlobalUtil::_DescriptorPPT) LoadDescriptorShader();
	if(s_descriptor_fp == NULL)
	{
		GlobalUtil::_DescriptorPPT = GlobalUtil::_FullSupported = 0;
		std::cerr<<"Descriptor ignored on this hardware"<<endl;
	}

	s_zero_pass = new ProgramGLSL("void main(){gl_FragColor = vec4(0.0);}");
}


void ShaderBagGLSL::LoadDisplayShaders()
{
	s_copy_key = new ProgramGLSL(
		"uniform sampler2DRect tex; void main(){\n"
	"gl_FragColor.rg= texture2DRect(tex, gl_TexCoord[0].st).rg; gl_FragColor.ba = vec2(0.0,1.0);	}");


	ProgramGLSL * program;
	s_vertex_list = program = new ProgramGLSL(
	"uniform vec4 sizes; uniform sampler2DRect tex;\n"
	"void main(void){\n"
	"float fwidth = sizes.y; float twidth = sizes.z; float rwidth = sizes.w; \n"
	"float index = 0.1*(fwidth*floor(gl_TexCoord[0].y) + gl_TexCoord[0].x);\n"
	"float px = mod(index, twidth);\n"
	"vec2 tpos= floor(vec2(px, index*rwidth))+0.5;\n"
	"vec4 cc = texture2DRect(tex, tpos );\n"
	"float size = 3.0 * cc.a; //sizes.x;// \n"
	"gl_FragColor.zw = vec2(0.0, 1.0);\n"
	"if(any(lessThan(cc.xy,vec2(0.0))))  {gl_FragColor.xy = cc.xy; }\n"
	"else {float type = fract(px);\n"
	"vec2 dxy = vec2(0); \n"
	"dxy.x = type < 0.1 ? 0.0 : (((type <0.5) || (type > 0.9))? size : -size);\n"
	"dxy.y = type < 0.2 ? 0.0 : (((type < 0.3) || (type > 0.7) )? -size :size); \n"
	"float s = sin(cc.b); float c = cos(cc.b); \n"
	"gl_FragColor.x = cc.x + c*dxy.x-s*dxy.y;\n"
	"gl_FragColor.y = cc.y + c*dxy.y+s*dxy.x;}\n}\n");

	_param_genvbo_size = glGetUniformLocation(*program, "sizes");

	s_display_gaussian =  new ProgramGLSL(
	"uniform sampler2DRect tex; void main(void){float r = texture2DRect(tex, gl_TexCoord[0].st).r;\n"
	"gl_FragColor = vec4(r, r, r, 1);}" );

	s_display_dog =  new ProgramGLSL(
	"uniform sampler2DRect tex; void main(void){float g = 0.5+(20.0*texture2DRect(tex, gl_TexCoord[0].st).g);\n"
	"gl_FragColor = vec4(g, g, g, 0.0);}" );

	s_display_grad = new ProgramGLSL(
		"uniform sampler2DRect tex; void main(void){\n"
    "	vec4 cc = texture2DRect(tex, gl_TexCoord[0].st);gl_FragColor = vec4(5.0* cc.bbb, 1.0);}");

	s_display_keys= new ProgramGLSL(
		"uniform sampler2DRect tex; void main(void){\n"
	"	vec4 cc = texture2DRect(tex, gl_TexCoord[0].st);\n"
	"	if(cc.r ==0.0) discard; gl_FragColor =  (cc.r==1.0? vec4(1.0, 0.0, 0,1.0):vec4(0.0,1.0,0.0,1.0));}");
}

void ShaderBagGLSL::LoadKeypointShader(float threshold, float edge_threshold)
{
	float threshold0 = threshold* (GlobalUtil::_SubpixelLocalization?0.8f:1.0f);
	float threshold1 = threshold;
	float threshold2 = (edge_threshold+1)*(edge_threshold+1)/edge_threshold;
	ostringstream out;;
	streampos pos;

	//tex(X)(Y)
	//X: (CLR) (CENTER 0, LEFT -1, RIGHT +1)
	//Y: (CDU) (CENTER 0, DOWN -1, UP    +1)
	if(GlobalUtil::_DarknessAdaption)
	{
		out <<	"#define THRESHOLD0 (" << threshold0 << " * min(2.0 * cc.r + 0.1, 1.0))\n"
				"#define THRESHOLD1 (" << threshold1 << " * min(2.0 * cc.r + 0.1, 1.0))\n"
				"#define THRESHOLD2 " << threshold2 << "\n";
	}else
	{
		out <<	"#define THRESHOLD0 " << threshold0 << "\n"
				"#define THRESHOLD1 " << threshold1 << "\n"
				"#define THRESHOLD2 " << threshold2 << "\n";
	}

	out<<
	"uniform sampler2DRect tex, texU, texD; void main ()\n"
	"{\n"
	"	vec4 v1, v2, gg, temp;\n"
	"	vec2 TexRU = vec2(gl_TexCoord[2].x, gl_TexCoord[4].y); \n"
	"	vec4 cc  = texture2DRect(tex, gl_TexCoord[0].xy);\n"
	"	temp =  texture2DRect(tex, gl_TexCoord[1].xy);\n"
	"	v1.x =  temp.g;			gg.x = temp.r;\n"
	"	temp = texture2DRect(tex, gl_TexCoord[2].xy) ;\n"
	"	v1.y = temp.g;			gg.y = temp.r;\n"
	"	temp = texture2DRect(tex, gl_TexCoord[3].xy) ;\n"
	"	v1.z = temp.g;			gg.z = temp.r;\n"
	"	temp = texture2DRect(tex, gl_TexCoord[4].xy) ;\n"
	"	v1.w = temp.g;			gg.w = temp.r;\n"
	"	v2.x = texture2DRect(tex, gl_TexCoord[5].xy).g;\n"
	"	v2.y = texture2DRect(tex, gl_TexCoord[6].xy).g;\n"
	"	v2.z = texture2DRect(tex, gl_TexCoord[7].xy).g;\n"
	"	v2.w = texture2DRect(tex, TexRU.xy).g;\n"
	"	vec2 dxdy = (gg.yw - gg.xz); \n"
	"	float grad = 0.5*length(dxdy);\n"
	"	float theta = grad==0.0? 0.0: atan(dxdy.y, dxdy.x);\n"
	"	gl_FragData[0] = vec4(cc.rg, grad, theta);\n"

	//test against 8 neighbours
	//use variable to identify type of extremum
	//1.0 for local maximum and 0.5 for minimum
	<<
	"	float dog = 0.0; \n"
	"	gl_FragData[1] = vec4(0, 0, 0, 0); \n"
	"	dog = cc.g > float(THRESHOLD0) && all(greaterThan(cc.gggg, max(v1, v2)))?1.0: 0.0;\n"
	"	dog = cc.g < float(-THRESHOLD0) && all(lessThan(cc.gggg, min(v1, v2)))?0.5: dog;\n"
	"	if(dog == 0.0) return;\n";

	pos = out.tellp();
	//do edge supression first..
	//vector v1 is < (-1, 0), (1, 0), (0,-1), (0, 1)>
	//vector v2 is < (-1,-1), (-1,1), (1,-1), (1, 1)>

	out<<
	"	float fxx, fyy, fxy; \n"
	"	vec4 D2 = v1.xyzw - cc.gggg;\n"
	"	vec2 D4 = v2.xw - v2.yz;\n"
	"	fxx = D2.x + D2.y;\n"
	"	fyy = D2.z + D2.w;\n"
	"	fxy = 0.25*(D4.x + D4.y);\n"
	"	float fxx_plus_fyy = fxx + fyy;\n"
	"	float score_up = fxx_plus_fyy*fxx_plus_fyy; \n"
	"	float score_down = (fxx*fyy - fxy*fxy);\n"
	"	if( score_down <= 0.0 || score_up > THRESHOLD2 * score_down)return;\n";

	//...
	out<<" \n"
	"	vec2 D5 = 0.5*(v1.yw-v1.xz); \n"
	"	float fx = D5.x, fy = D5.y ; \n"
	"	float fs, fss , fxs, fys ; \n"
	"	vec2 v3; vec4 v4, v5, v6;\n"
	//read 9 pixels of upper level
	<<
	"	v3.x = texture2DRect(texU, gl_TexCoord[0].xy).g;\n"
	"	v4.x = texture2DRect(texU, gl_TexCoord[1].xy).g;\n"
	"	v4.y = texture2DRect(texU, gl_TexCoord[2].xy).g;\n"
	"	v4.z = texture2DRect(texU, gl_TexCoord[3].xy).g;\n"
	"	v4.w = texture2DRect(texU, gl_TexCoord[4].xy).g;\n"
	"	v6.x = texture2DRect(texU, gl_TexCoord[5].xy).g;\n"
	"	v6.y = texture2DRect(texU, gl_TexCoord[6].xy).g;\n"
	"	v6.z = texture2DRect(texU, gl_TexCoord[7].xy).g;\n"
	"	v6.w = texture2DRect(texU, TexRU.xy).g;\n"
	//compare with 9 pixels of upper level
	//read and compare with 9 pixels of lower level
	//the maximum case
	<<
	"	if(dog == 1.0)\n"
	"	{\n"
	"		if(cc.g < v3.x || any(lessThan(cc.gggg, v4)) ||any(lessThan(cc.gggg, v6)))return; \n"
	"		v3.y = texture2DRect(texD, gl_TexCoord[0].xy).g;\n"
	"		v5.x = texture2DRect(texD, gl_TexCoord[1].xy).g;\n"
	"		v5.y = texture2DRect(texD, gl_TexCoord[2].xy).g;\n"
	"		v5.z = texture2DRect(texD, gl_TexCoord[3].xy).g;\n"
	"		v5.w = texture2DRect(texD, gl_TexCoord[4].xy).g;\n"
	"		v6.x = texture2DRect(texD, gl_TexCoord[5].xy).g;\n"
	"		v6.y = texture2DRect(texD, gl_TexCoord[6].xy).g;\n"
	"		v6.z = texture2DRect(texD, gl_TexCoord[7].xy).g;\n"
	"		v6.w = texture2DRect(texD, TexRU.xy).g;\n"
	"		if(cc.g < v3.y || any(lessThan(cc.gggg, v5)) ||any(lessThan(cc.gggg, v6)))return; \n"
	"	}\n"
	//the minimum case
	<<
	"	else{\n"
	"	if(cc.g > v3.x || any(greaterThan(cc.gggg, v4)) ||any(greaterThan(cc.gggg, v6)))return; \n"
	"		v3.y = texture2DRect(texD, gl_TexCoord[0].xy).g;\n"
	"		v5.x = texture2DRect(texD, gl_TexCoord[1].xy).g;\n"
	"		v5.y = texture2DRect(texD, gl_TexCoord[2].xy).g;\n"
	"		v5.z = texture2DRect(texD, gl_TexCoord[3].xy).g;\n"
	"		v5.w = texture2DRect(texD, gl_TexCoord[4].xy).g;\n"
	"		v6.x = texture2DRect(texD, gl_TexCoord[5].xy).g;\n"
	"		v6.y = texture2DRect(texD, gl_TexCoord[6].xy).g;\n"
	"		v6.z = texture2DRect(texD, gl_TexCoord[7].xy).g;\n"
	"		v6.w = texture2DRect(texD, TexRU.xy).g;\n"
	"		if(cc.g > v3.y || any(greaterThan(cc.gggg, v5)) ||any(greaterThan(cc.gggg, v6)))return; \n"
	"	}\n";

	if(GlobalUtil::_SubpixelLocalization)

	// sub-pixel localization FragData1 = vec4(dog, 0, 0, 0); return;
	out <<
	"	fs = 0.5*( v3.x - v3.y );  \n"
	"	fss = v3.x + v3.y - cc.g - cc.g;\n"
	"	fxs = 0.25 * ( v4.y + v5.x - v4.x - v5.y);\n"
	"	fys = 0.25 * ( v4.w + v5.z - v4.z - v5.w);\n"

	//
	// let dog difference be quatratic function  of dx, dy, ds;
	// df(dx, dy, ds) = fx * dx + fy*dy + fs * ds +
	//				  + 0.5 * ( fxx * dx * dx + fyy * dy * dy + fss * ds * ds)
	//				  + (fxy * dx * dy + fxs * dx * ds + fys * dy * ds)
	// (fx, fy, fs, fxx, fyy, fss, fxy, fxs, fys are the derivatives)

	//the local extremum satisfies
	// df/dx = 0, df/dy = 0, df/dz = 0

	//that is
	// |-fx|     | fxx fxy fxs |   |dx|
	// |-fy|  =  | fxy fyy fys | * |dy|
	// |-fs|     | fxs fys fss |   |ds|
	// need to solve dx, dy, ds

	// Use Gauss elimination to solve the linear system
    <<
	"	vec3 dxys = vec3(0.0);		\n"
	"	vec4 A0, A1, A2 ;			\n"
	"	A0 = vec4(fxx, fxy, fxs, -fx);	\n"
	"	A1 = vec4(fxy, fyy, fys, -fy);	\n"
	"	A2 = vec4(fxs, fys, fss, -fs);	\n"
	"	vec3 x3 = abs(vec3(fxx, fxy, fxs));		\n"
	"	float maxa = max(max(x3.x, x3.y), x3.z);	\n"
	"	if(maxa >= 1e-10 ) {						\n"
	"		if(x3.y ==maxa )							\n"
	"		{											\n"
	"			vec4 TEMP = A1; A1 = A0; A0 = TEMP;	\n"
	"		}else if( x3.z == maxa )					\n"
	"		{											\n"
	"			vec4 TEMP = A2; A2 = A0; A0 = TEMP;	\n"
	"		}											\n"
	"		A0 /= A0.x;									\n"
	"		A1 -= A1.x * A0;							\n"
	"		A2 -= A2.x * A0;							\n"
	"		vec2 x2 = abs(vec2(A1.y, A2.y));		\n"
	"		if( x2.y > x2.x )							\n"
	"		{											\n"
	"			vec3 TEMP = A2.yzw;					\n"
	"			A2.yzw = A1.yzw;						\n"
	"			A1.yzw = TEMP;							\n"
	"			x2.x = x2.y;							\n"
	"		}											\n"
	"		if(x2.x >= 1e-10) {						\n"
	"			A1.yzw /= A1.y;								\n"
	"			A2.yzw -= A2.y * A1.yzw;					\n"
	"			if(abs(A2.z) >= 1e-10) {		\n"
	// compute dx, dy, ds:
	<<
	"				\n"
	"				dxys.z = A2.w /A2.z;				    \n"
	"				dxys.y = A1.w - dxys.z*A1.z;			    \n"
	"				dxys.x = A0.w - dxys.z*A0.z - dxys.y*A0.y;	\n"

	//one more threshold which I forgot in versions prior to 286
	<<
	"				bool dog_test = (abs(cc.g + 0.5*dot(vec3(fx, fy, fs), dxys ))<= float(THRESHOLD1)) ;\n"
	"				if(dog_test || any(greaterThan(abs(dxys), vec3(1.0)))) dog = 0.0;\n"
	"			}\n"
	"		}\n"
	"	}\n"
    //keep the point when the offset is less than 1
	<<
	"	gl_FragData[1] = vec4( dog, dxys); \n";
	else

	out<<
	"	gl_FragData[1] =  vec4( dog, 0.0, 0.0, 0.0) ;	\n";

	out<<
	"}\n" <<'\0';



	ProgramGLSL * program = new ProgramGLSL(out.str().c_str());
	if(program->IsNative())
	{
		s_keypoint = program ;
		//parameter
	}else
	{
		delete program;
		out.seekp(pos);
		out <<
	"	gl_FragData[1] =  vec4(dog, 0.0, 0.0, 0.0) ;	\n"
	"}\n" <<'\0';
		s_keypoint = program = new ProgramGLSL(out.str().c_str());
		GlobalUtil::_SubpixelLocalization = 0;
		std::cerr<<"Detection simplified on this hardware"<<endl;
	}

	_param_dog_texu = glGetUniformLocation(*program, "texU");
	_param_dog_texd = glGetUniformLocation(*program, "texD");
}


void ShaderBagGLSL::SetDogTexParam(int texU, int texD)
{
	glUniform1i(_param_dog_texu, 1);
	glUniform1i(_param_dog_texd, 2);
}

void ShaderBagGLSL::SetGenListStepParam(int tex, int tex0)
{
	glUniform1i(_param_genlist_step_tex0, 1);
}
void ShaderBagGLSL::SetGenVBOParam( float width, float fwidth,  float size)
{
	float sizes[4] = {size*3.0f, fwidth, width, 1.0f/width};
	glUniform4fv(_param_genvbo_size, 1, sizes);

}



void ShaderBagGLSL::UnloadProgram()
{
	glUseProgram(0);
}



void ShaderBagGLSL::LoadGenListShader(int ndoglev, int nlev)
{
	ProgramGLSL * program;

	s_genlist_init_tight = new ProgramGLSL(
	"uniform sampler2DRect tex; void main (void){\n"
	"vec4 helper = vec4( texture2DRect(tex, gl_TexCoord[0].xy).r,  texture2DRect(tex, gl_TexCoord[1].xy).r,\n"
	"texture2DRect(tex, gl_TexCoord[2].xy).r, texture2DRect(tex, gl_TexCoord[3].xy).r);\n"
	"gl_FragColor = vec4(greaterThan(helper, vec4(0.0,0.0,0.0,0.0)));\n"
	"}");


	s_genlist_init_ex = program = new ProgramGLSL(
	"uniform sampler2DRect tex;uniform vec2 bbox;\n"
	"void main (void ){\n"
	"vec4 helper = vec4( texture2DRect(tex, gl_TexCoord[0].xy).r,  texture2DRect(tex, gl_TexCoord[1].xy).r,\n"
	"texture2DRect(tex, gl_TexCoord[2].xy).r, texture2DRect(tex, gl_TexCoord[3].xy).r);\n"
	"bvec4 helper2 = bvec4( \n"
	"all(lessThan(gl_TexCoord[0].xy , bbox)) && helper.x >0.0,\n"
	"all(lessThan(gl_TexCoord[1].xy , bbox)) && helper.y >0.0,\n"
	"all(lessThan(gl_TexCoord[2].xy , bbox)) && helper.z >0.0,\n"
	"all(lessThan(gl_TexCoord[3].xy , bbox)) && helper.w >0.0);\n"
	"gl_FragColor = vec4(helper2);\n"
	"}");
	_param_genlist_init_bbox = glGetUniformLocation( *program, "bbox");


	//reduction ...
	s_genlist_histo = new ProgramGLSL(
	"uniform sampler2DRect tex; void main (void){\n"
	"vec4 helper; vec4 helper2; \n"
	"helper = texture2DRect(tex, gl_TexCoord[0].xy); helper2.xy = helper.xy + helper.zw; \n"
	"helper = texture2DRect(tex, gl_TexCoord[1].xy); helper2.zw = helper.xy + helper.zw; \n"
	"gl_FragColor.rg = helper2.xz + helper2.yw;\n"
	"helper = texture2DRect(tex, gl_TexCoord[2].xy); helper2.xy = helper.xy + helper.zw; \n"
	"helper = texture2DRect(tex, gl_TexCoord[3].xy); helper2.zw = helper.xy + helper.zw; \n"
	"gl_FragColor.ba= helper2.xz+helper2.yw;\n"
	"}");


	//read of the first part, which generates tex coordinates
	s_genlist_start= program =  LoadGenListStepShader(1, 1);
	_param_ftex_width= glGetUniformLocation(*program, "width");
	_param_genlist_start_tex0 = glGetUniformLocation(*program, "tex0");
	//stepping
	s_genlist_step = program = LoadGenListStepShader(0, 1);
	_param_genlist_step_tex0= glGetUniformLocation(*program, "tex0");

}

void ShaderBagGLSL::SetMarginCopyParam(int xmax, int ymax)
{
	float truncate[2] = {xmax - 0.5f , ymax - 0.5f};
	glUniform2fv(_param_margin_copy_truncate, 1, truncate);
}

void ShaderBagGLSL::SetGenListInitParam(int w, int h)
{
	float bbox[2] = {w - 1.0f, h - 1.0f};
	glUniform2fv(_param_genlist_init_bbox, 1, bbox);
}
void ShaderBagGLSL::SetGenListStartParam(float width, int tex0)
{
	glUniform1f(_param_ftex_width, width);
	glUniform1i(_param_genlist_start_tex0, 0);
}


ProgramGLSL* ShaderBagGLSL::LoadGenListStepShader(int start, int step)
{
	int i;
	// char chanels[5] = "rgba";
	ostringstream out;

	for(i = 0; i < step; i++) out<<"uniform sampler2DRect tex"<<i<<";\n";
	if(start)
	{
		out<<"uniform float width;\n";
		out<<"void main(void){\n";
		out<<"float  index = floor(gl_TexCoord[0].y) * width + floor(gl_TexCoord[0].x);\n";
		out<<"vec2 pos = vec2(0.5, 0.5);\n";
	}else
	{
		out<<"uniform sampler2DRect tex;\n";
		out<<"void main(void){\n";
		out<<"vec4 tc = texture2DRect( tex, gl_TexCoord[0].xy);\n";
		out<<"vec2 pos = tc.rg; float index = tc.b;\n";
	}
	out<<"vec2 sum; 	vec4 cc;\n";


	if(step>0)
	{
		out<<"vec2 cpos = vec2(-0.5, 0.5);\t vec2 opos;\n";
		for(i = 0; i < step; i++)
		{

			out<<"cc = texture2DRect(tex"<<i<<", pos);\n";
			out<<"sum.x = cc.r + cc.g; sum.y = sum.x + cc.b;  \n";
			out<<"if (index <cc.r){ opos = cpos.xx;}\n";
			out<<"else if(index < sum.x ) {opos = cpos.yx; index -= cc.r;}\n";
			out<<"else if(index < sum.y ) {opos = cpos.xy; index -= sum.x;}\n";
			out<<"else {opos = cpos.yy; index -= sum.y;}\n";
			out<<"pos = (pos + pos + opos);\n";
		}
	}
	out<<"gl_FragColor = vec4(pos, index, 1.0);\n";
	out<<"}\n"<<'\0';
	return new ProgramGLSL(out.str().c_str());
}


void ShaderBagGLSL::LoadOrientationShader()
{
	ostringstream out;

	if(GlobalUtil::_IsNvidia)
	{
	out <<	"#pragma optionNV(ifcvt none)\n"
			"#pragma optionNV(unroll all)\n";
	}

	out<<"\n"
	"#define GAUSSIAN_WF float("<<GlobalUtil::_OrientationGaussianFactor<<") \n"
	"#define SAMPLE_WF float("<<GlobalUtil::_OrientationWindowFactor<< " )\n"
	"#define ORIENTATION_THRESHOLD "<< GlobalUtil::_MulitiOrientationThreshold << "\n"
	"uniform sampler2DRect tex;					\n"
	"uniform sampler2DRect gradTex;				\n"
	"uniform vec4 size;						\n"
	<< ((GlobalUtil::_SubpixelLocalization || GlobalUtil::_KeepExtremumSign)? "	uniform sampler2DRect texS;	\n" : " ")	<<
	"void main()		\n"
	"{													\n"
	"	vec4 bins[10];								\n"
	"	bins[0] = vec4(0.0);bins[1] = vec4(0.0);bins[2] = vec4(0.0);	\n"
	"	bins[3] = vec4(0.0);bins[4] = vec4(0.0);bins[5] = vec4(0.0);	\n"
	"	bins[6] = vec4(0.0);bins[7] = vec4(0.0);bins[8] = vec4(0.0);	\n"
	"	vec4 loc = texture2DRect(tex, gl_TexCoord[0].xy);	\n"
	"	vec2 pos = loc.xy;		\n"
	"	bool orientation_mode = (size.z != 0.0);			\n"
	"	float sigma = orientation_mode? abs(size.z) : loc.w; \n";
	if(GlobalUtil::_SubpixelLocalization || GlobalUtil::_KeepExtremumSign)
	{
		out<<
	"	if(orientation_mode){\n"
	"		vec4 offset = texture2DRect(texS, pos);\n"
	"		pos.xy = pos.xy + offset.yz; \n"
	"		sigma = sigma * pow(size.w, offset.w);\n"
	"		#if "<< GlobalUtil::_KeepExtremumSign << "\n"
	"			if(offset.x < 0.6) sigma = -sigma; \n"
	"		#endif\n"
	"	}\n";
	}
	out<<
	"	//bool fixed_orientation = (size.z < 0.0);		\n"
	"	if(size.z < 0.0) {gl_FragData[0] = vec4(pos, 0.0, sigma); return;}"
	"	float gsigma = sigma * GAUSSIAN_WF;				\n"
	"	vec2 win = abs(vec2(sigma * (SAMPLE_WF * GAUSSIAN_WF))) ;	\n"
	"	vec2 dim = size.xy;							\n"
	"	float dist_threshold = win.x*win.x+0.5;			\n"
	"	float factor = -0.5/(gsigma*gsigma);			\n"
	"	vec4 sz;	vec2 spos;						\n"
	"	//if(any(pos.xy <= 1)) discard;					\n"
	"	sz.xy = max( pos - win, vec2(1,1));			\n"
	"	sz.zw = min( pos + win, dim-vec2(2, 2));				\n"
	"	sz = floor(sz)+0.5;";
	//loop to get the histogram

	out<<"\n"
	"	for(spos.y = sz.y; spos.y <= sz.w;	spos.y+=1.0)				\n"
	"	{																\n"
	"		for(spos.x = sz.x; spos.x <= sz.z;	spos.x+=1.0)			\n"
	"		{															\n"
	"			vec2 offset = spos - pos;								\n"
	"			float sq_dist = dot(offset,offset);						\n"
	"			if( sq_dist < dist_threshold){							\n"
	"				vec4 cc = texture2DRect(gradTex, spos);				\n"
	"				float grad = cc.b;	float theta = cc.a;				\n"
	"				float idx = floor(degrees(theta)*0.1);				\n"
	"				if(idx < 0.0 ) idx += 36.0;									\n"
	"				float weight = grad*exp(sq_dist * factor);				\n"
	"				float vidx = fract(idx * 0.25) * 4.0;//mod(idx, 4.0) ;							\n"
	"				vec4 inc = weight*vec4(equal(vec4(vidx), vec4(0.0,1.0,2.0,3.0)));";

	if(GlobalUtil::_UseDynamicIndexing)
	{
		//dynamic indexing may not be faster
		out<<"\n"
	"				int iidx = int((idx*0.25));	\n"
	"				bins[iidx]+=inc;					\n"
	"			}										\n"
	"		}											\n"
	"	}";

	}else
	{
		//nvfp40 still does not support dynamic array indexing
		//unrolled binary search...
		out<<"\n"
	"				if(idx < 16.0)							\n"
	"				{										\n"
	"					if(idx < 8.0)							\n"
	"					{									\n"
	"						if(idx < 4.0)	{	bins[0]+=inc;}	\n"
	"						else		{	bins[1]+=inc;}	\n"
	"					}else								\n"
	"					{									\n"
	"						if(idx < 12.0){	bins[2]+=inc;}	\n"
	"						else		{	bins[3]+=inc;}	\n"
	"					}									\n"
	"				}else if(idx < 32.0)						\n"
	"				{										\n"
	"					if(idx < 24.0)						\n"
	"					{									\n"
	"						if(idx <20.0)	{	bins[4]+=inc;}	\n"
	"						else		{	bins[5]+=inc;}	\n"
	"					}else								\n"
	"					{									\n"
	"						if(idx < 28.0){	bins[6]+=inc;}	\n"
	"						else		{	bins[7]+=inc;}	\n"
	"					}									\n"
	"				}else 						\n"
	"				{										\n"
	"					bins[8]+=inc;						\n"
	"				}										\n"
	"			}										\n"
	"		}											\n"
	"	}";

	}

	WriteOrientationCodeToStream(out);

	ProgramGLSL * program = new ProgramGLSL(out.str().c_str());
	if(program->IsNative())
	{
		s_orientation = program ;
		_param_orientation_gtex = glGetUniformLocation(*program, "gradTex");
		_param_orientation_size = glGetUniformLocation(*program, "size");
		_param_orientation_stex = glGetUniformLocation(*program, "texS");
	}else
	{
		delete program;
	}
}


void ShaderBagGLSL::WriteOrientationCodeToStream(std::ostream& out)
{
	//smooth histogram and find the largest
/*
	smoothing kernel:	 (1 3 6 7 6 3 1 )/27
	the same as 3 pass of (1 1 1)/3 averaging
	maybe better to use 4 pass on the vectors...
*/


	//the inner loop on different array numbers is always unrolled in fp40

	//bug fixed here:)
	out<<"\n"
	"	//mat3 m1 = mat3(1, 0, 0, 3, 1, 0, 6, 3, 1)/27.0;  \n"
	"	mat3 m1 = mat3(1, 3, 6, 0, 1, 3,0, 0, 1)/27.0;  \n"
	"	mat4 m2 = mat4(7, 6, 3, 1, 6, 7, 6, 3, 3, 6, 7, 6, 1, 3, 6, 7)/27.0;\n"
	"	#define FILTER_CODE(i) {						\\\n"
	"			vec4 newb	=	(bins[i]* m2);			\\\n"
	"			newb.xyz	+=	( prev.yzw * m1);		\\\n"
	"			prev = bins[i];							\\\n"
	"			newb.wzy	+=	( bins[i+1].zyx *m1);	\\\n"
	"			bins[i] = newb;}\n"
	"	for (int j=0; j<2; j++)								\n"
	"	{												\n"
	"		vec4 prev  = bins[8];						\n"
	"		bins[9]		 = bins[0];						\n";

	if(GlobalUtil::_KeepShaderLoop)
	{
		out<<
	"		for (int i=0; i<9; i++)							\n"
	"		{												\n"
	"			FILTER_CODE(i);								\n"
	"		}												\n"
	"	}";

	}else
	{
		//manually unroll the loop for ATI.
		out <<
	"	   FILTER_CODE(0);\n"
	"	   FILTER_CODE(1);\n"
	"	   FILTER_CODE(2);\n"
	"	   FILTER_CODE(3);\n"
	"	   FILTER_CODE(4);\n"
	"	   FILTER_CODE(5);\n"
	"	   FILTER_CODE(6);\n"
	"	   FILTER_CODE(7);\n"
	"	   FILTER_CODE(8);\n"
	"	}\n";
	}
	//find the maximum voting
	out<<"\n"
	"	vec4 maxh; vec2 maxh2; 	\n"
	"	vec4 maxh4 = max(max(max(max(max(max(max(max(bins[0], bins[1]), bins[2]), \n"
	"			bins[3]), bins[4]), bins[5]), bins[6]), bins[7]), bins[8]);\n"
	"	maxh2 = max(maxh4.xy, maxh4.zw); maxh = vec4(max(maxh2.x, maxh2.y));";

	std::string testpeak_code;
	std::string savepeak_code;

	//save two/three/four orientations with the largest votings?

	if(GlobalUtil::_MaxOrientation>1)
	{
		out<<"\n"
		"	vec4 Orientations = vec4(0.0, 0.0, 0.0, 0.0);				\n"
		"	vec4 weights = vec4(0.0,0.0,0.0,0.0);		";

		testpeak_code = "\\\n"
		"	{test = greaterThan(bins[i], hh);";

		//save the orientations in weight-decreasing order
		if(GlobalUtil::_MaxOrientation ==2)
		{
		savepeak_code = "\\\n"
		"			if(weight <=weights.g){}\\\n"
		"			else if(weight >weights.r)\\\n"
		"			{weights.rg = vec2(weight, weights.r); Orientations.rg = vec2(th, Orientations.r);}\\\n"
		"			else {weights.g = weight; Orientations.g = th;}";
		}else if(GlobalUtil::_MaxOrientation ==3)
		{
		savepeak_code = "\\\n"
		"			if(weight <=weights.b){}\\\n"
		"			else if(weight >weights.r)\\\n"
		"			{weights.rgb = vec3(weight, weights.rg); Orientations.rgb = vec3(th, Orientations.rg);}\\\n"
		"			else if(weight >weights.g)\\\n"
		"			{weights.gb = vec2(weight, weights.g); Orientations.gb = vec2(th, Orientations.g);}\\\n"
		"			else {weights.b = weight; Orientations.b = th;}";
		}else
		{
		savepeak_code = "\\\n"
		"			if(weight <=weights.a){}\\\n"
		"			else if(weight >weights.r)\\\n"
		"			{weights = vec4(weight, weights.rgb); Orientations = vec4(th, Orientations.rgb);}\\\n"
		"			else if(weight >weights.g)\\\n"
		"			{weights.gba = vec3(weight, weights.gb); Orientations.gba = vec3(th, Orientations.gb);}\\\n"
		"			else if(weight >weights.b)\\\n"
		"			{weights.ba = vec2(weight, weights.b); Orientations.ba = vec2(th, Orientations.b);}\\\n"
		"			else {weights.a = weight; Orientations.a = th;}";
		}

	}else
	{
		out<<"\n"
		"	float Orientation;				";
		testpeak_code ="\\\n"
		"	if(npeaks<=0.0){\\\n"
		"	test = equal(bins[i], maxh)	;";
		savepeak_code="\\\n"
		"			npeaks++;	\\\n"
		"			Orientation = th;";

	}
	//find the peaks
	out <<"\n"
	"	#define FINDPEAK(i, k)"	<<testpeak_code<<"\\\n"
	"	if( any ( test) )							\\\n"
	"	{											\\\n"
	"		if(test.r && bins[i].x > prevb && bins[i].x > bins[i].y )	\\\n"
	"		{											\\\n"
	"		    float	di = -0.5 * (bins[i].y-prevb) / (bins[i].y+prevb-bins[i].x - bins[i].x) ; \\\n"
	"		    float	th = (k+di+0.5);	float weight = bins[i].x;"
				<<savepeak_code<<"\\\n"
	"		}\\\n"
	"		else if(test.g && all( greaterThan(bins[i].yy , bins[i].xz)) )	\\\n"
	"		{											\\\n"
	"		    float	di = -0.5 * (bins[i].z-bins[i].x) / (bins[i].z+bins[i].x-bins[i].y- bins[i].y) ; \\\n"
	"		    float	th = (k+di+1.5);	float weight = bins[i].y;				"
				<<savepeak_code<<"	\\\n"
	"		}\\\n"
	"		if(test.b && all( greaterThan( bins[i].zz , bins[i].yw)) )	\\\n"
	"		{											\\\n"
	"		    float	di = -0.5 * (bins[i].w-bins[i].y) / (bins[i].w+bins[i].y-bins[i].z- bins[i].z) ; \\\n"
	"		    float	th = (k+di+2.5);	float weight = bins[i].z;				"
				<<savepeak_code<<"	\\\n"
	"		}\\\n"
	"		else if(test.a && bins[i].w > bins[i].z && bins[i].w > bins[i+1].x )	\\\n"
	"		{											\\\n"
	"		    float	di = -0.5 * (bins[i+1].x-bins[i].z) / (bins[i+1].x+bins[i].z-bins[i].w - bins[i].w) ; \\\n"
	"		    float	th = (k+di+3.5);	float weight = bins[i].w;				"
				<<savepeak_code<<"	\\\n"
	"		}\\\n"
	"	}}\\\n"
	"	prevb = bins[i].w;";
	//the following loop will be unrolled anyway in fp40,
	//taking more than 1000 instrucsions..
	//....
	if(GlobalUtil::_KeepShaderLoop)
	{
	out<<"\n"
	"	vec4 hh = maxh * ORIENTATION_THRESHOLD;	bvec4 test;	\n"
	"	bins[9] = bins[0];								\n"
	"	float npeaks = 0.0, k = 0.0;						\n"
	"	float prevb	= bins[8].w;						\n"
	"	for (int i = 0; i < 9; i++)						\n"
	"	{\n"
	"		FINDPEAK(i, k);\n"
	"		k = k + 4.0;	\n"
	"	}";
	}else
	{
		//loop unroll for ATI.
	out <<"\n"
	"	vec4 hh = maxh * ORIENTATION_THRESHOLD; bvec4 test;\n"
	"	bins[9] = bins[0];								\n"
	"	float npeaks = 0.0;								\n"
	"	float prevb	= bins[8].w;						\n"
	"	FINDPEAK(0, 0.0);\n"
	"	FINDPEAK(1, 4.0);\n"
	"	FINDPEAK(2, 8.0);\n"
	"	FINDPEAK(3, 12.0);\n"
	"	FINDPEAK(4, 16.0);\n"
	"	FINDPEAK(5, 20.0);\n"
	"	FINDPEAK(6, 24.0);\n"
	"	FINDPEAK(7, 28.0);\n"
	"	FINDPEAK(8, 32.0);\n";
	}
	//WRITE output
	if(GlobalUtil::_MaxOrientation>1)
	{
	out<<"\n"
	"	if(orientation_mode){\n"
	"		npeaks = dot(vec4(1,1,"
			<<(GlobalUtil::_MaxOrientation>2 ? 1 : 0)<<","
			<<(GlobalUtil::_MaxOrientation >3? 1 : 0)<<"), vec4(greaterThan(weights, hh)));\n"
	"		gl_FragData[0] = vec4(pos, npeaks, sigma);\n"
	"		gl_FragData[1] = radians((Orientations )*10.0);\n"
	"	}else{\n"
	"		gl_FragData[0] = vec4(pos, radians((Orientations.x)*10.0), sigma);\n"
	"	}\n";
	}else
	{
	out<<"\n"
	"	 gl_FragData[0] = vec4(pos, radians((Orientation)*10.0), sigma);\n";
	}
	//end
	out<<"\n"
	"}\n"<<'\0';


}

void ShaderBagGLSL::SetSimpleOrientationInput(int oTex, float sigma, float sigma_step)
{
	glUniform1i(_param_orientation_gtex, 1);
	glUniform1f(_param_orientation_size, sigma);
}




void ShaderBagGLSL::SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex, float step)
{
	///
	glUniform1i(_param_orientation_gtex, 1);

	if((GlobalUtil::_SubpixelLocalization || GlobalUtil::_KeepExtremumSign)&& stex)
	{
		//specify texutre for subpixel subscale localization
		glUniform1i(_param_orientation_stex, 2);
	}

	float size[4];
	size[0] = (float)width;
	size[1] = (float)height;
	size[2] = sigma;
	size[3] = step;
	glUniform4fv(_param_orientation_size, 1, size);
}


void ShaderBagGLSL::LoadDescriptorShaderF2()
{
	//one shader outpout 128/8 = 16 , each fragout encodes 4
	//const double twopi = 2.0*3.14159265358979323846;
	//const double rpi  = 8.0/twopi;
	ostringstream out;
	out<<setprecision(8);

	out<<"\n"
	"#define M_PI 3.14159265358979323846\n"
	"#define TWO_PI (2.0*M_PI)\n"
	"#define RPI 1.2732395447351626861510701069801\n"
	"#define WF  size.z\n"
	"uniform sampler2DRect tex;				\n"
	"uniform sampler2DRect gradTex;			\n"
	"uniform vec4 dsize;						\n"
	"uniform vec3 size;						\n"
	"void main()		\n"
	"{\n"
	"	vec2 dim	= size.xy;	//image size			\n"
	"	float index = dsize.x*floor(gl_TexCoord[0].y * 0.5) + gl_TexCoord[0].x;\n"
	"	float idx = 8.0 * fract(index * 0.125) + 8.0 * floor(2.0 * fract(gl_TexCoord[0].y * 0.5));		\n"
	"	index = floor(index*0.125) + 0.49;  \n"
	"	vec2 coord = floor( vec2( mod(index, dsize.z), index*dsize.w)) + 0.5 ;\n"
	"	vec2 pos = texture2DRect(tex, coord).xy;		\n"
	"	if(any(lessThanEqual(pos.xy,  vec2(1.0))) || any(greaterThanEqual(pos.xy, dim-1.0)))// discard;	\n"
	"	{ gl_FragData[0] = gl_FragData[1] = vec4(0.0); return; }\n"
	"	float  anglef = texture2DRect(tex, coord).z;\n"
	"	if(anglef > M_PI) anglef -= TWO_PI;\n"
	"	float sigma = texture2DRect(tex, coord).w; \n"
	"	float spt  = abs(sigma * WF);	//default to be 3*sigma	\n";

	//rotation
	out<<
	"	vec4 cscs, rots;								\n"
	"	cscs.y = sin(anglef);	cscs.x = cos(anglef);	\n"
	"	cscs.zw = - cscs.xy;							\n"
	"	rots = cscs /spt;								\n"
	"	cscs *= spt; \n";

	//here cscs is actually (cos, sin, -cos, -sin) * (factor: 3)*sigma
	//and rots is  (cos, sin, -cos, -sin ) /(factor*sigma)
	//devide the 4x4 sift grid into 16 1x1 block, and each corresponds to a shader thread
	//To use linear interoplation, 1x1 is increased to 2x2, by adding 0.5 to each side

	out<<
	"vec4 temp; vec2 pt, offsetpt;				\n"
	"	/*the fraction part of idx is .5*/			\n"
	"	offsetpt.x = 4.0* fract(idx*0.25) - 2.0;				\n"
	"	offsetpt.y = floor(idx*0.25) - 1.5;			\n"
	"	temp = cscs.xwyx*offsetpt.xyxy;				\n"
	"	pt = pos + temp.xz + temp.yw;				\n";

	//get a horizontal bounding box of the rotated rectangle
	out<<
	"	vec2 bwin = abs(cscs.xy);					\n"
	"	float bsz = bwin.x + bwin.y;					\n"
	"	vec4 sz;					\n"
	"	sz.xy = max(pt - vec2(bsz), vec2(1,1));\n"
	"	sz.zw = min(pt + vec2(bsz), dim - vec2(2, 2));		\n"
	"	sz = floor(sz)+0.5;"; //move sample point to pixel center
	//get voting for two box

	out<<"\n"
	"	vec4 DA, DB; vec2 spos;			\n"
	"	DA = DB  = vec4(0.0, 0.0, 0.0, 0.0);		\n"
	"	for(spos.y = sz.y; spos.y <= sz.w;	spos.y+=1.0)				\n"
	"	{																\n"
	"		for(spos.x = sz.x; spos.x <= sz.z;	spos.x+=1.0)			\n"
	"		{															\n"
	"			vec2 diff = spos - pt;								\n"
	"			temp = rots.xywx * diff.xyxy;\n"
	"			vec2 nxy = (temp.xz + temp.yw); \n"
	"			vec2 nxyn = abs(nxy);			\n"
	"			if(all( lessThan(nxyn, vec2(1.0)) ))\n"
	"			{\n"
	"				vec4 cc = texture2DRect(gradTex, spos);						\n"
	"				float mod = cc.b;	float angle = cc.a;					\n"
	"				float theta0 = RPI * (anglef - angle);				\n"
	"				float theta = theta0 < 0.0? theta0 + 8.0 : theta0;;\n"
	"				diff = nxy + offsetpt.xy;								\n"
	"				float ww = exp(-0.125*dot(diff, diff));\n"
	"				vec2 weights = vec2(1) - nxyn;\n"
	"				float weight = weights.x * weights.y *mod*ww; \n"
	"				float theta1 = floor(theta); \n"
	"				float weight2 = (theta - theta1) * weight;\n"
	"				float weight1 = weight - weight2;\n"
	"				DA += vec4(equal(vec4(theta1),  vec4(0, 1, 2, 3)))*weight1;\n"
	"				DA += vec4(equal(vec4(theta1),  vec4(7, 0, 1, 2)))*weight2; \n"
	"				DB += vec4(equal(vec4(theta1),  vec4(4, 5, 6, 7)))*weight1;\n"
	"				DB += vec4(equal(vec4(theta1),  vec4(3, 4, 5, 6)))*weight2; \n"
	"			}\n"
	"		}\n"
	"	}\n";

	out<<
	"	 gl_FragData[0] = DA; gl_FragData[1] = DB;\n"
	"}\n"<<'\0';

	ProgramGLSL * program =  new ProgramGLSL(out.str().c_str());

	if(program->IsNative())
	{
		s_descriptor_fp = program ;
		_param_descriptor_gtex = glGetUniformLocation(*program, "gradTex");
		_param_descriptor_size = glGetUniformLocation(*program, "size");
		_param_descriptor_dsize = glGetUniformLocation(*program, "dsize");
	}else
	{
		delete program;
	}


}

void ShaderBagGLSL::LoadDescriptorShader()
{
	GlobalUtil::_DescriptorPPT = 16;
	LoadDescriptorShaderF2();
}


void ShaderBagGLSL::SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth,  float width, float height, float sigma)
{
	///
	glUniform1i(_param_descriptor_gtex, 1);

	float dsize[4] ={dwidth, 1.0f/dwidth, fwidth, 1.0f/fwidth};
	glUniform4fv(_param_descriptor_dsize, 1, dsize);
	float size[3];
	size[0] = width;
	size[1] = height;
	size[2] = GlobalUtil::_DescriptorWindowFactor;
	glUniform3fv(_param_descriptor_size, 1, size);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ShaderBagPKSL::LoadFixedShaders()
{
	ProgramGLSL * program;


	s_gray = new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
	"float intensity = dot(vec3(0.299, 0.587, 0.114), texture2DRect(tex,gl_TexCoord[0].xy ).rgb);\n"
	"gl_FragColor= vec4(intensity, intensity, intensity, 1.0);}"	);


	s_sampling = new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
	"gl_FragColor= vec4(	texture2DRect(tex,gl_TexCoord[0].st ).r,texture2DRect(tex,gl_TexCoord[1].st ).r,\n"
	"						texture2DRect(tex,gl_TexCoord[2].st ).r,texture2DRect(tex,gl_TexCoord[3].st ).r);}"	);


	s_margin_copy = program = new ProgramGLSL(
	"uniform sampler2DRect tex;  uniform vec4 truncate; void main(){\n"
	"vec4 cc = texture2DRect(tex, min(gl_TexCoord[0].xy, truncate.xy)); \n"
	"bvec2 ob = lessThan(gl_TexCoord[0].xy, truncate.xy);\n"
	"if(ob.y) { gl_FragColor = (truncate.z ==0.0 ? cc.rrbb : cc.ggaa); } \n"
	"else if(ob.x) {gl_FragColor = (truncate.w <1.5 ? cc.rgrg : cc.baba);} \n"
	"else {	vec4 weights = vec4(vec4(0.0, 1.0, 2.0, 3.0) == truncate.wwww);\n"
	"float v = dot(weights, cc); gl_FragColor = vec4(v);}}");

	_param_margin_copy_truncate = glGetUniformLocation(*program, "truncate");



	s_zero_pass = new ProgramGLSL("void main(){gl_FragColor = vec4(0.0);}");



	s_grad_pass = program = new ProgramGLSL(
	"uniform sampler2DRect tex; uniform sampler2DRect texp; void main ()\n"
	"{\n"
	"	vec4 v1, v2, gg;\n"
	"	vec4 cc = texture2DRect(tex, gl_TexCoord[0].xy);\n"
	"	vec4 cp = texture2DRect(texp, gl_TexCoord[0].xy);\n"
	"	gl_FragData[0] = cc - cp; \n"
	"	vec4 cl = texture2DRect(tex, gl_TexCoord[1].xy); vec4 cr = texture2DRect(tex, gl_TexCoord[2].xy);\n"
	"	vec4 cd = texture2DRect(tex, gl_TexCoord[3].xy); vec4 cu = texture2DRect(tex, gl_TexCoord[4].xy);\n"
	"	vec4 dx = (vec4(cr.rb, cc.ga) - vec4(cc.rb, cl.ga)).zxwy;\n"
	"	vec4 dy = (vec4(cu.rg, cc.ba) - vec4(cc.rg, cd.ba)).zwxy;\n"
	"	vec4 grad = 0.5 * sqrt(dx*dx + dy * dy);\n"
	"	gl_FragData[1] = grad;\n"
	"	vec4 invalid = vec4(equal(grad, vec4(0.0)));	\n"
	"	vec4 ov = atan(dy, dx + invalid);		\n"
	"	gl_FragData[2] = ov; \n"
	"}\n\0"); //when

	_param_grad_pass_texp = glGetUniformLocation(*program, "texp");


	GlobalUtil::_OrientationPack2 = 0;
	LoadOrientationShader();

	if(s_orientation == NULL)
	{
		//Load a simplified version if the right version is not supported
		s_orientation = program =  new ProgramGLSL(
		"uniform sampler2DRect tex; uniform sampler2DRect oTex; uniform vec2 size; void main(){\n"
		"	vec4 cc = texture2DRect(tex, gl_TexCoord[0].xy);\n"
		"	vec2 co = cc.xy * 0.5; \n"
		"	vec4 oo = texture2DRect(oTex, co);\n"
		"	bvec2 bo = lessThan(fract(co), vec2(0.5)); \n"
		"	float o = bo.y? (bo.x? oo.r : oo.g) : (bo.x? oo.b : oo.a); \n"
		"	gl_FragColor = vec4(cc.rg, o, size.x * pow(size.y, cc.a));}");

		_param_orientation_gtex= glGetUniformLocation(*program, "oTex");
		_param_orientation_size= glGetUniformLocation(*program, "size");
		GlobalUtil::_MaxOrientation = 0;
		GlobalUtil::_FullSupported = 0;
		std::cerr<<"Orientation simplified on this hardware"<<endl;
	}

	if(GlobalUtil::_DescriptorPPT)
	{
		LoadDescriptorShader();
		if(s_descriptor_fp == NULL)
		{
			GlobalUtil::_DescriptorPPT = GlobalUtil::_FullSupported = 0;
			std::cerr<<"Descriptor ignored on this hardware"<<endl;
		}
	}
}


void ShaderBagPKSL::LoadDisplayShaders()
{
	ProgramGLSL * program;

	s_copy_key = new ProgramGLSL(
	"uniform sampler2DRect tex;void main(){\n"
	"gl_FragColor= vec4(texture2DRect(tex, gl_TexCoord[0].xy).rg, 0,1);}");

	//shader used to write a vertex buffer object
	//which is used to draw the quads of each feature
	s_vertex_list = program = new ProgramGLSL(
	"uniform sampler2DRect tex; uniform vec4 sizes; void main(){\n"
	"float fwidth = sizes.y; \n"
	"float twidth = sizes.z; \n"
	"float rwidth = sizes.w; \n"
	"float index = 0.1*(fwidth*floor(gl_TexCoord[0].y) + gl_TexCoord[0].x);\n"
	"float px = mod(index, twidth);\n"
	"vec2 tpos= floor(vec2(px, index*rwidth))+0.5;\n"
	"vec4 cc = texture2DRect(tex, tpos );\n"
	"float size = 3.0 * cc.a; \n"
	"gl_FragColor.zw = vec2(0.0, 1.0);\n"
	"if(any(lessThan(cc.xy,vec2(0.0)))) {gl_FragColor.xy = cc.xy;}else \n"
	"{\n"
	"	float type = fract(px);\n"
	"	vec2 dxy; float s, c;\n"
	"	dxy.x = type < 0.1 ? 0.0 : (((type <0.5) || (type > 0.9))? size : -size);\n"
	"	dxy.y = type < 0.2 ? 0.0 : (((type < 0.3) || (type > 0.7) )? -size :size); \n"
	"	s = sin(cc.b); c = cos(cc.b); \n"
	"	gl_FragColor.x = cc.x + c*dxy.x-s*dxy.y;\n"
	"	gl_FragColor.y = cc.y + c*dxy.y+s*dxy.x;}\n"
	"}\n\0");
	/*gl_FragColor = vec4(tpos, 0.0, 1.0);}\n\0");*/

	_param_genvbo_size = glGetUniformLocation(*program, "sizes");

	s_display_gaussian = new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
    "vec4 pc = texture2DRect(tex, gl_TexCoord[0].xy);	bvec2 ff = lessThan(fract(gl_TexCoord[0].xy), vec2(0.5));\n"
    "float v = ff.y?(ff.x? pc.r : pc.g):(ff.x?pc.b:pc.a); gl_FragColor = vec4(vec3(v), 1.0);}");

	s_display_dog =  new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
	"vec4 pc = texture2DRect(tex, gl_TexCoord[0].xy); bvec2 ff = lessThan(fract(gl_TexCoord[0].xy), vec2(0.5));\n"
	"float v = ff.y ?(ff.x ? pc.r : pc.g):(ff.x ? pc.b : pc.a);float g = (0.5+20.0*v);\n"
	"gl_FragColor = vec4(g, g, g, 1.0);}" );


	s_display_grad = new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
	"vec4 pc = texture2DRect(tex, gl_TexCoord[0].xy); bvec2 ff = lessThan(fract(gl_TexCoord[0].xy), vec2(0.5));\n"
	"float v = ff.y ?(ff.x ? pc.r : pc.g):(ff.x ? pc.b : pc.a); gl_FragColor = vec4(5.0 *vec3(v), 1.0); }");

	s_display_keys= new ProgramGLSL(
	"uniform sampler2DRect tex; void main(){\n"
	"vec4 oc = texture2DRect(tex, gl_TexCoord[0].xy); \n"
	"vec4 cc = vec4(equal(abs(oc.rrrr), vec4(1.0, 2.0, 3.0, 4.0))); \n"
	"bvec2 ff = lessThan(fract(gl_TexCoord[0].xy) , vec2(0.5));\n"
	"float v = ff.y ?(ff.x ? cc.r : cc.g):(ff.x ? cc.b : cc.a);\n"
	"if(v == 0.0) discard;	\n"
	"else if(oc.r > 0.0) gl_FragColor = vec4(1.0, 0.0, 0,1.0); \n"
	"else gl_FragColor = vec4(0.0,1.0,0.0,1.0);	}" );
}

void ShaderBagPKSL::LoadOrientationShader(void)
{
	ostringstream out;
	if(GlobalUtil::_IsNvidia)
	{
		out <<	"#pragma optionNV(ifcvt none)\n"
				"#pragma optionNV(unroll all)\n";
	}
	out<<"\n"
	"#define GAUSSIAN_WF float("<<GlobalUtil::_OrientationGaussianFactor<<") \n"
	"#define SAMPLE_WF float("<<GlobalUtil::_OrientationWindowFactor<< " )\n"
	"#define ORIENTATION_THRESHOLD "<< GlobalUtil::_MulitiOrientationThreshold << "\n"
	"uniform sampler2DRect tex;	uniform sampler2DRect gtex;\n"
	"uniform sampler2DRect otex; uniform vec4 size;\n"
	"void main()		\n"
	"{													\n"
	"	vec4 bins[10];								\n"
	"	bins[0] = vec4(0.0);bins[1] = vec4(0.0);bins[2] = vec4(0.0);	\n"
	"	bins[3] = vec4(0.0);bins[4] = vec4(0.0);bins[5] = vec4(0.0);	\n"
	"	bins[6] = vec4(0.0);bins[7] = vec4(0.0);bins[8] = vec4(0.0);	\n"
	"	vec4 sift = texture2DRect(tex, gl_TexCoord[0].xy);	\n"
	"	vec2 pos = sift.xy; \n"
	"	bool orientation_mode = (size.z != 0.0);		\n"
	"	float sigma = orientation_mode? (abs(size.z) * pow(size.w, sift.w) * sift.z) : (sift.w); \n"
	"	//bool fixed_orientation = (size.z < 0.0);		\n"
	"	if(size.z < 0.0) {gl_FragData[0] = vec4(pos, 0.0, sigma); return;}"
	"	float gsigma = sigma * GAUSSIAN_WF;				\n"
	"	vec2 win = abs(vec2(sigma * (SAMPLE_WF * GAUSSIAN_WF)));	\n"
	"	vec2 dim = size.xy;							\n"
	"	vec4 dist_threshold = vec4(win.x*win.x+0.5);			\n"
	"	float factor = -0.5/(gsigma*gsigma);			\n"
	"	vec4 sz;	vec2 spos;						\n"
	"	//if(any(pos.xy <= float(1))) discard;					\n"
	"	sz.xy = max( pos - win, vec2(2.0,2.0));			\n"
	"	sz.zw = min( pos + win, dim-vec2(3.0));				\n"
	"	sz = floor(sz*0.5) + 0.5; ";
		//loop to get the histogram

	out<<"\n"
	"	for(spos.y = sz.y; spos.y <= sz.w;	spos.y+=1.0)				\n"
	"	{																\n"
	"		for(spos.x = sz.x; spos.x <= sz.z;	spos.x+=1.0)			\n"
	"		{															\n"
	"			vec2 offset = 2.0 * spos - pos - vec2(0.5);					\n"
	"			vec4 off = vec4(offset, offset + vec2(1));				\n"
	"			vec4 distsq = off.xzxz * off.xzxz + off.yyww * off.yyww;	\n"
	"			bvec4 inside = lessThan(distsq, dist_threshold);			\n"
	"			if(any(inside))										\n"
	"			{														\n"
	"				vec4 gg = texture2DRect(gtex, spos);				\n"
	"				vec4 oo = texture2DRect(otex, spos);				\n"
	"				vec4 weight = gg * exp(distsq * factor);			\n"
	"				vec4 idxv  = floor(degrees(oo)*0.1); 				\n"
	"				idxv+= (vec4(lessThan(idxv, vec4(0.0)))*36.0); 			\n"
	"				vec4 vidx = fract(idxv * 0.25) * 4.0;//mod(idxv, 4.0);	\n";
	//
	if(GlobalUtil::_UseDynamicIndexing)
	{
		// it might be slow on some GPUs
		out<<"\n"
	"				for(int i = 0 ; i < 4; i++)\n"
	"				{\n"
	"					if(inside[i])\n"
	"					{\n"
	"						float idx = idxv[i];								\n"
	"						vec4 inc = weight[i] * vec4(equal(vec4(vidx[i]), vec4(0.0,1.0,2.0,3.0)));	\n"
	"						int iidx = int(floor(idx*0.25));	\n"
	"						bins[iidx]+=inc;					\n"
	"					}										\n"
	"				}											\n"
	"			}												\n"
	"		}													\n"
	"	}";

	}else
	{
		//nvfp40 still does not support dynamic array indexing
		//unrolled binary search
		//it seems to be faster than the dyanmic indexing version on some GPUs
		out<<"\n"
	"				for(int i = 0 ; i < 4; i++)\n"
	"				{\n"
	"					if(inside[i])\n"
	"					{\n"
	"						float idx = idxv[i]; 										\n"
	"						vec4 inc = weight[i] * vec4(equal(vec4(vidx[i]), vec4(0,1,2,3)));	\n"
	"						if(idx < 16.0)							\n"
	"						{										\n"
	"							if(idx < 8.0)							\n"
	"							{									\n"
	"								if(idx < 4.0)	{	bins[0]+=inc;}	\n"
	"								else		{	bins[1]+=inc;}	\n"
	"							}else								\n"
	"							{									\n"
	"								if(idx < 12.0){	bins[2]+=inc;}	\n"
	"								else		{	bins[3]+=inc;}	\n"
	"							}									\n"
	"						}else if(idx < 32.0)						\n"
	"						{										\n"
	"							if(idx < 24.0)						\n"
	"							{									\n"
	"								if(idx <20.0)	{	bins[4]+=inc;}	\n"
	"								else		{	bins[5]+=inc;}	\n"
	"							}else								\n"
	"							{									\n"
	"								if(idx < 28.0){	bins[6]+=inc;}	\n"
	"								else		{	bins[7]+=inc;}	\n"
	"							}									\n"
	"						}else 						\n"
	"						{										\n"
	"							bins[8]+=inc;						\n"
	"						}										\n"
	"					}											\n"
	"				}												\n"
	"			}										\n"
	"		}											\n"
	"	}";

	}

	//reuse the code from the unpacked version..
	ShaderBagGLSL::WriteOrientationCodeToStream(out);



	ProgramGLSL * program = new ProgramGLSL(out.str().c_str());
	if(program->IsNative())
	{
		s_orientation = program ;
		_param_orientation_gtex = glGetUniformLocation(*program, "gtex");
		_param_orientation_otex = glGetUniformLocation(*program, "otex");
		_param_orientation_size = glGetUniformLocation(*program, "size");
	}else
	{
		delete program;
	}
}

void ShaderBagPKSL::SetGenListStartParam(float width, int tex0)
{
	glUniform1f(_param_ftex_width, width);
	glUniform1i(_param_genlist_start_tex0, 0);
}

void ShaderBagPKSL::LoadGenListShader(int ndoglev,int nlev)
{
	ProgramGLSL * program;

	s_genlist_init_tight = new ProgramGLSL(
	"uniform sampler2DRect tex; void main ()\n"
	"{\n"
	"	vec4 key = vec4(texture2DRect(tex, gl_TexCoord[0].xy).r, \n"
	"					texture2DRect(tex, gl_TexCoord[1].xy).r, \n"
	"					texture2DRect(tex, gl_TexCoord[2].xy).r, \n"
	"					texture2DRect(tex, gl_TexCoord[3].xy).r); \n"
	"					gl_FragColor = vec4(notEqual(key, vec4(0.0))); \n"
	"}");

	s_genlist_init_ex = program = new ProgramGLSL(
	"uniform sampler2DRect tex; uniform vec4 bbox; void main ()\n"
	"{\n"
	"	vec4 helper1 = vec4(equal(vec4(abs(texture2DRect(tex, gl_TexCoord[0].xy).r)), vec4(1.0, 2.0, 3.0, 4.0)));\n"
	"	vec4 helper2 = vec4(equal(vec4(abs(texture2DRect(tex, gl_TexCoord[1].xy).r)), vec4(1.0, 2.0, 3.0, 4.0)));\n"
	"	vec4 helper3 = vec4(equal(vec4(abs(texture2DRect(tex, gl_TexCoord[2].xy).r)), vec4(1.0, 2.0, 3.0, 4.0)));\n"
	"	vec4 helper4 = vec4(equal(vec4(abs(texture2DRect(tex, gl_TexCoord[3].xy).r)), vec4(1.0, 2.0, 3.0, 4.0)));\n"
	"	vec4 bx1 = vec4(lessThan(gl_TexCoord[0].xxyy, bbox)); \n"
	"	vec4 bx4 = vec4(lessThan(gl_TexCoord[3].xxyy, bbox)); \n"
	"	vec4 bx2 = vec4(bx4.xy, bx1.zw); \n"
	"	vec4 bx3 = vec4(bx1.xy, bx4.zw);\n"
	"	helper1 = min(min(bx1.xyxy, bx1.zzww), helper1);\n"
	"	helper2 = min(min(bx2.xyxy, bx2.zzww), helper2);\n"
	"	helper3 = min(min(bx3.xyxy, bx3.zzww), helper3);\n"
	"	helper4 = min(min(bx4.xyxy, bx4.zzww), helper4);\n"
	"	gl_FragColor.r = float(any(greaterThan(max(helper1.xy, helper1.zw), vec2(0.0))));	\n"
	"	gl_FragColor.g = float(any(greaterThan(max(helper2.xy, helper2.zw), vec2(0.0))));	\n"
	"	gl_FragColor.b = float(any(greaterThan(max(helper3.xy, helper3.zw), vec2(0.0))));	\n"
	"	gl_FragColor.a = float(any(greaterThan(max(helper4.xy, helper4.zw), vec2(0.0))));	\n"
	"}");
	_param_genlist_init_bbox = glGetUniformLocation( *program, "bbox");

	s_genlist_end = program = new ProgramGLSL(
		GlobalUtil::_KeepExtremumSign == 0 ?

	"uniform sampler2DRect tex; uniform sampler2DRect ktex; void main()\n"
	"{\n"
	"	vec4 tc = texture2DRect( tex, gl_TexCoord[0].xy);\n"
	"	vec2 pos = tc.rg; float index = tc.b;\n"
	"	vec4 tk = texture2DRect( ktex, pos); \n"
	"	vec4 keys = vec4(equal(abs(tk.rrrr), vec4(1.0, 2.0, 3.0, 4.0))); \n"
	"	vec2 opos; \n"
	"	opos.x = dot(keys, vec4(-0.5, 0.5, -0.5, 0.5));\n"
	"	opos.y = dot(keys, vec4(-0.5, -0.5, 0.5, 0.5));\n"
	"	gl_FragColor = vec4(opos + pos * 2.0 + tk.yz, 1.0, tk.w);\n"
	"}" :

	"uniform sampler2DRect tex; uniform sampler2DRect ktex; void main()\n"
	"{\n"
	"	vec4 tc = texture2DRect( tex, gl_TexCoord[0].xy);\n"
	"	vec2 pos = tc.rg; float index = tc.b;\n"
	"	vec4 tk = texture2DRect( ktex, pos); \n"
	"	vec4 keys = vec4(equal(abs(tk.rrrr), vec4(1.0, 2.0, 3.0, 4.0))) \n"
	"	vec2 opos; \n"
	"	opos.x = dot(keys, vec4(-0.5, 0.5, -0.5, 0.5));\n"
	"	opos.y = dot(keys, vec4(-0.5, -0.5, 0.5, 0.5));\n"
	"	gl_FragColor = vec4(opos + pos * 2.0 + tk.yz, sign(tk.r), tk.w);\n"
	"}"
	);

	_param_genlist_end_ktex = glGetUniformLocation(*program, "ktex");

	//reduction ...
	s_genlist_histo = new ProgramGLSL(
	"uniform sampler2DRect tex; void main ()\n"
	"{\n"
	"	vec4 helper; vec4 helper2; \n"
	"	helper = texture2DRect(tex, gl_TexCoord[0].xy); helper2.xy = helper.xy + helper.zw; \n"
	"	helper = texture2DRect(tex, gl_TexCoord[1].xy); helper2.zw = helper.xy + helper.zw; \n"
	"	gl_FragColor.rg = helper2.xz + helper2.yw;\n"
	"	helper = texture2DRect(tex, gl_TexCoord[2].xy); helper2.xy = helper.xy + helper.zw; \n"
	"	helper = texture2DRect(tex, gl_TexCoord[3].xy); helper2.zw = helper.xy + helper.zw; \n"
	"	gl_FragColor.ba= helper2.xz+helper2.yw;\n"
	"}");


	//read of the first part, which generates tex coordinates

	s_genlist_start= program =  ShaderBagGLSL::LoadGenListStepShader(1, 1);
	_param_ftex_width= glGetUniformLocation(*program, "width");
	_param_genlist_start_tex0 = glGetUniformLocation(*program, "tex0");
	//stepping
	s_genlist_step = program = ShaderBagGLSL::LoadGenListStepShader(0, 1);
	_param_genlist_step_tex0= glGetUniformLocation(*program, "tex0");

}
void ShaderBagPKSL::UnloadProgram(void)
{
	glUseProgram(0);
}
void ShaderBagPKSL::LoadKeypointShader(float dog_threshold, float edge_threshold)
{
	float threshold0 = dog_threshold* (GlobalUtil::_SubpixelLocalization?0.8f:1.0f);
	float threshold1 = dog_threshold;
	float threshold2 = (edge_threshold+1)*(edge_threshold+1)/edge_threshold;
	ostringstream out;;
	out<<setprecision(8);

	if(GlobalUtil::_IsNvidia)
	{
		out << "#pragma optionNV(ifcvt none)\n"
				"#pragma optionNV(unroll all)\n";

	}
	if(GlobalUtil::_KeepShaderLoop)
	{
		out <<  "#define REPEAT4(FUNCTION)\\\n"
				"for(int i = 0; i < 4; ++i)\\\n"
				"{\\\n"
				"	FUNCTION(i);\\\n"
				"}\n";
	}else
	{
		//loop unroll
		out <<  "#define REPEAT4(FUNCTION)\\\n"
				"FUNCTION(0);\\\n"
				"FUNCTION(1);\\\n"
				"FUNCTION(2);\\\n"
				"FUNCTION(3);\n";
	}
	//tex(X)(Y)
	//X: (CLR) (CENTER 0, LEFT -1, RIGHT +1)
	//Y: (CDU) (CENTER 0, DOWN -1, UP    +1)

	if(GlobalUtil::_DarknessAdaption)
	{
		out <<	"#define THRESHOLD0(i) (" << threshold0 << "* ii[i])\n"
				"#define THRESHOLD1 (" << threshold1 << "* ii[0])\n"
				"#define THRESHOLD2 " << threshold2 << "\n"
				"#define DEFINE_EXTRA() vec4 ii = texture2DRect(texI, gl_TexCoord[0].xy); "
				"ii = min(2.0 * ii + 0.1, 1.0) \n"
				"#define MOVE_EXTRA(idx)	ii[0] = ii[idx]\n";
		out << "uniform sampler2DRect texI;\n";
	}else
	{
		out <<	"#define THRESHOLD0(i) " << threshold0 << "\n"
				"#define THRESHOLD1 " << threshold1 << "\n"
				"#define THRESHOLD2 " << threshold2 << "\n"
				"#define DEFINE_EXTRA()\n"
				"#define MOVE_EXTRA(idx) \n"	;
	}

	out<<
	"uniform sampler2DRect tex; uniform sampler2DRect texU;\n"
	"uniform sampler2DRect texD; void main ()\n"
	"{\n"
	"	vec2 TexRU = vec2(gl_TexCoord[2].x, gl_TexCoord[4].y); \n"
	"	vec4 ccc = texture2DRect(tex, gl_TexCoord[0].xy);\n"
	"	vec4 clc = texture2DRect(tex, gl_TexCoord[1].xy);\n"
	"	vec4 crc = texture2DRect(tex, gl_TexCoord[2].xy);\n"
	"	vec4 ccd = texture2DRect(tex, gl_TexCoord[3].xy);\n"
	"	vec4 ccu = texture2DRect(tex, gl_TexCoord[4].xy);\n"
	"	vec4 cld = texture2DRect(tex, gl_TexCoord[5].xy);\n"
	"	vec4 clu = texture2DRect(tex, gl_TexCoord[6].xy);\n"
	"	vec4 crd = texture2DRect(tex, gl_TexCoord[7].xy);\n"
	"	vec4 cru = texture2DRect(tex, TexRU.xy);\n"
	"	vec4  cc = ccc;\n"
	"	vec4  v1[4], v2[4];\n"
	"	v1[0] = vec4(clc.g, ccc.g, ccd.b, ccc.b);\n"
	"	v1[1] = vec4(ccc.r, crc.r, ccd.a, ccc.a);\n"
	"	v1[2] = vec4(clc.a, ccc.a, ccc.r, ccu.r);\n"
	"	v1[3] = vec4(ccc.b, crc.b, ccc.g, ccu.g);\n"
	"	v2[0] = vec4(cld.a, clc.a, ccd.a, ccc.a);\n"
	"	v2[1] = vec4(ccd.b, ccc.b, crd.b, crc.b);\n"
	"	v2[2] = vec4(clc.g, clu.g, ccc.g, ccu.g);\n"
	"	v2[3] = vec4(ccc.r, ccu.r, crc.r, cru.r);\n"
	"	DEFINE_EXTRA();\n";

	//test against 8 neighbours
	//use variable to identify type of extremum
	//1.0 for local maximum and -1.0 for minimum
	out <<
	"	vec4 key = vec4(0.0); \n"
	"	#define KEYTEST_STEP0(i) \\\n"
	"	{\\\n"
	"		bvec4 test1 = greaterThan(vec4(cc[i]), max(v1[i], v2[i])), test2 = lessThan(vec4(cc[i]), min(v1[i], v2[i]));\\\n"
	"		key[i] = cc[i] > float(THRESHOLD0(i)) && all(test1)?1.0: 0.0;\\\n"
	"		key[i] = cc[i] < float(-THRESHOLD0(i)) && all(test2)? -1.0: key[i];\\\n"
	"	}\n"
	"	REPEAT4(KEYTEST_STEP0);\n"
	"	if(gl_TexCoord[0].x < 1.0) {key.rb = vec2(0.0);}\n"
	"	if(gl_TexCoord[0].y < 1.0) {key.rg = vec2(0.0);}\n"
	"	gl_FragColor = vec4(0.0);\n"
	"	if(any(notEqual(key, vec4(0.0)))) {\n";

	//do edge supression first..
	//vector v1 is < (-1, 0), (1, 0), (0,-1), (0, 1)>
	//vector v2 is < (-1,-1), (-1,1), (1,-1), (1, 1)>

	out<<
	"	float fxx[4], fyy[4], fxy[4], fx[4], fy[4];\n"
	"	#define EDGE_SUPPRESION(i) \\\n"
	"	if(key[i] != 0.0)\\\n"
	"	{\\\n"
	"		vec4 D2 = v1[i].xyzw - cc[i];\\\n"
	"		vec2 D4 = v2[i].xw - v2[i].yz;\\\n"
	"		vec2 D5 = 0.5*(v1[i].yw-v1[i].xz); \\\n"
	"		fx[i] = D5.x;	fy[i] = D5.y ;\\\n"
	"		fxx[i] = D2.x + D2.y;\\\n"
	"		fyy[i] = D2.z + D2.w;\\\n"
	"		fxy[i] = 0.25*(D4.x + D4.y);\\\n"
	"		float fxx_plus_fyy = fxx[i] + fyy[i];\\\n"
	"		float score_up = fxx_plus_fyy*fxx_plus_fyy; \\\n"
	"		float score_down = (fxx[i]*fyy[i] - fxy[i]*fxy[i]);\\\n"
	"		if( score_down <= 0.0 || score_up > THRESHOLD2 * score_down)key[i] = 0.0;\\\n"
	"	}\n"
	"	REPEAT4(EDGE_SUPPRESION);\n"
	"	if(any(notEqual(key, vec4(0.0)))) {\n";

	////////////////////////////////////////////////
	//read 9 pixels of upper/lower level
	out<<
	"	vec4  v4[4], v5[4], v6[4];\n"
	"	ccc = texture2DRect(texU, gl_TexCoord[0].xy);\n"
	"	clc = texture2DRect(texU, gl_TexCoord[1].xy);\n"
	"	crc = texture2DRect(texU, gl_TexCoord[2].xy);\n"
	"	ccd = texture2DRect(texU, gl_TexCoord[3].xy);\n"
	"	ccu = texture2DRect(texU, gl_TexCoord[4].xy);\n"
	"	cld = texture2DRect(texU, gl_TexCoord[5].xy);\n"
	"	clu = texture2DRect(texU, gl_TexCoord[6].xy);\n"
	"	crd = texture2DRect(texU, gl_TexCoord[7].xy);\n"
	"	cru = texture2DRect(texU, TexRU.xy);\n"
	"	vec4 cu = ccc;\n"
	"	v4[0] = vec4(clc.g, ccc.g, ccd.b, ccc.b);\n"
	"	v4[1] = vec4(ccc.r, crc.r, ccd.a, ccc.a);\n"
	"	v4[2] = vec4(clc.a, ccc.a, ccc.r, ccu.r);\n"
	"	v4[3] = vec4(ccc.b, crc.b, ccc.g, ccu.g);\n"
	"	v6[0] = vec4(cld.a, clc.a, ccd.a, ccc.a);\n"
	"	v6[1] = vec4(ccd.b, ccc.b, crd.b, crc.b);\n"
	"	v6[2] = vec4(clc.g, clu.g, ccc.g, ccu.g);\n"
	"	v6[3] = vec4(ccc.r, ccu.r, crc.r, cru.r);\n"
	<<
	"	#define KEYTEST_STEP1(i)\\\n"
	"	if(key[i] == 1.0)\\\n"
	"	{\\\n"
	"		bvec4 test = lessThan(vec4(cc[i]), max(v4[i], v6[i])); \\\n"
	"		if(cc[i] < cu[i] || any(test))key[i] = 0.0; \\\n"
	"	}else if(key[i] == -1.0)\\\n"
	"	{\\\n"
	"		bvec4 test = greaterThan(vec4(cc[i]), min(v4[i], v6[i])); \\\n"
	"		if(cc[i] > cu[i] || any(test) )key[i] = 0.0; \\\n"
	"	}\n"
	"	REPEAT4(KEYTEST_STEP1);\n"
	"	if(any(notEqual(key, vec4(0.0)))) { \n"
	<<
	"	ccc = texture2DRect(texD, gl_TexCoord[0].xy);\n"
	"	clc = texture2DRect(texD, gl_TexCoord[1].xy);\n"
	"	crc = texture2DRect(texD, gl_TexCoord[2].xy);\n"
	"	ccd = texture2DRect(texD, gl_TexCoord[3].xy);\n"
	"	ccu = texture2DRect(texD, gl_TexCoord[4].xy);\n"
	"	cld = texture2DRect(texD, gl_TexCoord[5].xy);\n"
	"	clu = texture2DRect(texD, gl_TexCoord[6].xy);\n"
	"	crd = texture2DRect(texD, gl_TexCoord[7].xy);\n"
	"	cru = texture2DRect(texD, TexRU.xy);\n"
	"	vec4 cd = ccc;\n"
	"	v5[0] = vec4(clc.g, ccc.g, ccd.b, ccc.b);\n"
	"	v5[1] = vec4(ccc.r, crc.r, ccd.a, ccc.a);\n"
	"	v5[2] = vec4(clc.a, ccc.a, ccc.r, ccu.r);\n"
	"	v5[3] = vec4(ccc.b, crc.b, ccc.g, ccu.g);\n"
	"	v6[0] = vec4(cld.a, clc.a, ccd.a, ccc.a);\n"
	"	v6[1] = vec4(ccd.b, ccc.b, crd.b, crc.b);\n"
	"	v6[2] = vec4(clc.g, clu.g, ccc.g, ccu.g);\n"
	"	v6[3] = vec4(ccc.r, ccu.r, crc.r, cru.r);\n"
	<<
	"	#define KEYTEST_STEP2(i)\\\n"
	"	if(key[i] == 1.0)\\\n"
	"	{\\\n"
	"		bvec4 test = lessThan(vec4(cc[i]), max(v5[i], v6[i]));\\\n"
	"		if(cc[i] < cd[i] || any(test))key[i] = 0.0; \\\n"
	"	}else if(key[i] == -1.0)\\\n"
	"	{\\\n"
	"		bvec4 test = greaterThan(vec4(cc[i]), min(v5[i], v6[i]));\\\n"
	"		if(cc[i] > cd[i] || any(test))key[i] = 0.0; \\\n"
	"	}\n"
	"	REPEAT4(KEYTEST_STEP2);\n"
	"	float keysum = dot(abs(key), vec4(1, 1, 1, 1)) ;\n"
	"	//assume there is only one keypoint in the four. \n"
	"	if(keysum==1.0) {\n";

	//////////////////////////////////////////////////////////////////////
	if(GlobalUtil::_SubpixelLocalization)

	out <<
	"	vec3 offset = vec3(0.0, 0.0, 0.0); \n"
	"	#define TESTMOVE_KEYPOINT(idx) \\\n"
	"	if(key[idx] != 0.0) \\\n"
	"	{\\\n"
	"		cu[0] = cu[idx];	cd[0] = cd[idx];	cc[0] = cc[idx];	\\\n"
	"		v4[0] = v4[idx];	v5[0] = v5[idx];						\\\n"
	"		fxy[0] = fxy[idx];	fxx[0] = fxx[idx];	fyy[0] = fyy[idx];	\\\n"
	"		fx[0] = fx[idx];	fy[0] = fy[idx];	MOVE_EXTRA(idx);  \\\n"
	"	}\n"
	"	TESTMOVE_KEYPOINT(1);\n"
	"	TESTMOVE_KEYPOINT(2);\n"
	"	TESTMOVE_KEYPOINT(3);\n"
	<<

	"	float fs = 0.5*( cu[0] - cd[0] );				\n"
	"	float fss = cu[0] + cd[0] - cc[0] - cc[0];\n"
	"	float fxs = 0.25 * (v4[0].y + v5[0].x - v4[0].x - v5[0].y);\n"
	"	float fys = 0.25 * (v4[0].w + v5[0].z - v4[0].z - v5[0].w);\n"
	"	vec4 A0, A1, A2 ;			\n"
	"	A0 = vec4(fxx[0], fxy[0], fxs, -fx[0]);	\n"
	"	A1 = vec4(fxy[0], fyy[0], fys, -fy[0]);	\n"
	"	A2 = vec4(fxs, fys, fss, -fs);	\n"
	"	vec3 x3 = abs(vec3(fxx[0], fxy[0], fxs));		\n"
	"	float maxa = max(max(x3.x, x3.y), x3.z);	\n"
	"	if(maxa >= 1e-10 ) \n"
	"	{												\n"
	"		if(x3.y ==maxa )							\n"
	"		{											\n"
	"			vec4 TEMP = A1; A1 = A0; A0 = TEMP;	\n"
	"		}else if( x3.z == maxa )					\n"
	"		{											\n"
	"			vec4 TEMP = A2; A2 = A0; A0 = TEMP;	\n"
	"		}											\n"
	"		A0 /= A0.x;									\n"
	"		A1 -= A1.x * A0;							\n"
	"		A2 -= A2.x * A0;							\n"
	"		vec2 x2 = abs(vec2(A1.y, A2.y));		\n"
	"		if( x2.y > x2.x )							\n"
	"		{											\n"
	"			vec3 TEMP = A2.yzw;					\n"
	"			A2.yzw = A1.yzw;						\n"
	"			A1.yzw = TEMP;							\n"
	"			x2.x = x2.y;							\n"
	"		}											\n"
	"		if(x2.x >= 1e-10) {								\n"
	"			A1.yzw /= A1.y;								\n"
	"			A2.yzw -= A2.y * A1.yzw;					\n"
	"			if(abs(A2.z) >= 1e-10) {\n"
	"				offset.z = A2.w /A2.z;				    \n"
	"				offset.y = A1.w - offset.z*A1.z;			    \n"
	"				offset.x = A0.w - offset.z*A0.z - offset.y*A0.y;	\n"
	"				bool test = (abs(cc[0] + 0.5*dot(vec3(fx[0], fy[0], fs), offset ))>float(THRESHOLD1)) ;\n"
	"				if(!test || any( greaterThan(abs(offset), vec3(1.0)))) key = vec4(0.0);\n"
	"			}\n"
	"		}\n"
	"	}\n"
	<<"\n"
	"	float keyv = dot(key, vec4(1.0, 2.0, 3.0, 4.0));\n"
	"	gl_FragColor = vec4(keyv,  offset);\n"
	"	}}}}\n"
	"}\n"	<<'\0';

	else out << "\n"
	"	float keyv = dot(key, vec4(1.0, 2.0, 3.0, 4.0));\n"
	"	gl_FragColor =  vec4(keyv, 0.0, 0.0, 0.0);\n"
	"	}}}}\n"
	"}\n"	<<'\0';

	ProgramGLSL * program = new ProgramGLSL(out.str().c_str());
	s_keypoint = program ;

	//parameter
	_param_dog_texu = glGetUniformLocation(*program, "texU");
	_param_dog_texd = glGetUniformLocation(*program, "texD");
	if(GlobalUtil::_DarknessAdaption) 	_param_dog_texi = glGetUniformLocation(*program, "texI");
}
void ShaderBagPKSL::SetDogTexParam(int texU, int texD)
{
	glUniform1i(_param_dog_texu, 1);
	glUniform1i(_param_dog_texd, 2);
	if(GlobalUtil::_DarknessAdaption)glUniform1i(_param_dog_texi, 3);
}
void ShaderBagPKSL::SetGenListStepParam(int tex, int tex0)
{
	glUniform1i(_param_genlist_step_tex0, 1);
}

void ShaderBagPKSL::SetGenVBOParam(float width, float fwidth,float size)
{
	float sizes[4] = {size*3.0f, fwidth, width, 1.0f/width};
	glUniform4fv(_param_genvbo_size, 1, sizes);
}
void ShaderBagPKSL::SetGradPassParam(int texP)
{
	glUniform1i(_param_grad_pass_texp, 1);
}

void ShaderBagPKSL::LoadDescriptorShader()
{
	GlobalUtil::_DescriptorPPT = 16;
	LoadDescriptorShaderF2();
    s_rect_description = LoadDescriptorProgramRECT();
}

ProgramGLSL* ShaderBagPKSL::LoadDescriptorProgramRECT()
{
	//one shader outpout 128/8 = 16 , each fragout encodes 4
	//const double twopi = 2.0*3.14159265358979323846;
	//const double rpi  = 8.0/twopi;
	ostringstream out;
	out<<setprecision(8);
	if(GlobalUtil::_KeepShaderLoop)
	{
		out << 	"#define REPEAT4(FUNCTION)\\\n"
				"for(int i = 0; i < 4; ++i)\\\n"
				"{\\\n"
				"	FUNCTION(i);\\\n"
				"}\n";
	}else
	{
		//loop unroll for ATI
		out <<  "#define REPEAT4(FUNCTION)\\\n"
				"FUNCTION(0);\\\n"
				"FUNCTION(1);\\\n"
				"FUNCTION(2);\\\n"
				"FUNCTION(3);\n";
	}

	out<<"\n"
	"#define M_PI 3.14159265358979323846\n"
	"#define TWO_PI (2.0*M_PI)\n"
	"#define RPI 1.2732395447351626861510701069801\n"
	"#define WF size.z\n"
	"uniform sampler2DRect tex;			\n"
	"uniform sampler2DRect gtex;			\n"
	"uniform sampler2DRect otex;			\n"
	"uniform vec4		dsize;				\n"
	"uniform vec3		size;				\n"
	"void main()			\n"
	"{\n"
	"	vec2 dim	= size.xy;	//image size			\n"
	"	float index = dsize.x*floor(gl_TexCoord[0].y * 0.5) + gl_TexCoord[0].x;\n"
	"	float idx = 8.0* fract(index * 0.125) + 8.0 * floor(2.0* fract(gl_TexCoord[0].y * 0.5));		\n"
	"	index = floor(index*0.125)+ 0.49;  \n"
	"	vec2 coord = floor( vec2( mod(index, dsize.z), index*dsize.w)) + 0.5 ;\n"
	"	vec2 pos = texture2DRect(tex, coord).xy;		\n"
	"	vec2 wsz = texture2DRect(tex, coord).zw;\n"
    "   float aspect_ratio = wsz.y / wsz.x;\n"
    "   float aspect_sq = aspect_ratio * aspect_ratio; \n"
	"	vec2 spt  = wsz * 0.25; vec2 ispt = 1.0 / spt; \n";

	//here cscs is actually (cos, sin, -cos, -sin) * (factor: 3)*sigma
	//and rots is  (cos, sin, -cos, -sin ) /(factor*sigma)
	//devide the 4x4 sift grid into 16 1x1 block, and each corresponds to a shader thread
	//To use linear interoplation, 1x1 is increased to 2x2, by adding 0.5 to each side
	out<<
	"	vec4 temp; vec2 pt;				\n"
    "	pt.x = pos.x + fract(idx*0.25) * wsz.x;				\n"
	"	pt.y = pos.y + (floor(idx*0.25) + 0.5) * spt.y;			\n";

	//get a horizontal bounding box of the rotated rectangle
	out<<
    "	vec4 sz;					\n"
	"	sz.xy = max(pt - spt, vec2(2,2));\n"
	"	sz.zw = min(pt + spt, dim - vec2(3));		\n"
	"	sz = floor(sz * 0.5)+0.5;"; //move sample point to pixel center
	//get voting for two box

	out<<"\n"
	"	vec4 DA, DB;   vec2 spos;			\n"
	"	DA = DB  = vec4(0.0, 0.0, 0.0, 0.0);		\n"
	"	vec4 nox = vec4(0.0, 1.0, 0.0, 1.0);					\n"
	"	vec4 noy = vec4(0.0, 0.0, 1.0, 1.0);					\n"
	"	for(spos.y = sz.y; spos.y <= sz.w;	spos.y+=1.0)				\n"
	"	{																\n"
	"		for(spos.x = sz.x; spos.x <= sz.z;	spos.x+=1.0)			\n"
	"		{															\n"
	"			vec2 tpt = spos * 2.0 - pt - 0.5;					\n"
    "			vec4 nx = (tpt.x + nox) * ispt.x;								\n"
    "			vec4 ny = (tpt.y + noy) * ispt.y;			\n"
	"			vec4 nxn = abs(nx), nyn = abs(ny);						\n"
    "			bvec4 inside = lessThan(max(nxn, nyn) , vec4(1.0));	\n"
	"			if(any(inside))\n"
	"			{\n"
	"				vec4 gg = texture2DRect(gtex, spos);\n"
	"				vec4 oo = texture2DRect(otex, spos);\n"
    //"               vec4 cc = cos(oo), ss = sin(oo); \n"
    //"               oo = atan(ss* aspect_ratio, cc); \n"
    //"               gg = gg * sqrt(ss * ss * aspect_sq + cc * cc); \n "
	"				vec4 theta0 = (- oo)*RPI;\n"
	"				vec4 theta = 8.0 * fract(1.0 + 0.125 * theta0);			\n"
	"				vec4 theta1 = floor(theta);								\n"
	"				vec4 weight = (vec4(1) - nxn) * (vec4(1) - nyn) * gg; \n"
	"				vec4 weight2 = (theta - theta1) * weight;				\n"
	"				vec4 weight1 = weight - weight2;						\n"
	"				#define ADD_DESCRIPTOR(i) \\\n"
	"				if(inside[i])\\\n"
	"				{\\\n"
	"					DA += vec4(equal(vec4(theta1[i]), vec4(0, 1, 2, 3)))*weight1[i]; \\\n"
	"					DA += vec4(equal(vec4(theta1[i]), vec4(7, 0, 1, 2)))*weight2[i]; \\\n"
	"					DB += vec4(equal(vec4(theta1[i]), vec4(4, 5, 6, 7)))*weight1[i]; \\\n"
	"					DB += vec4(equal(vec4(theta1[i]), vec4(3, 4, 5, 6)))*weight2[i]; \\\n"
	"				}\n"
	"				REPEAT4(ADD_DESCRIPTOR);\n"
	"			}\n"
	"		}\n"
	"	}\n";
	out<<
	"	 gl_FragData[0] = DA; gl_FragData[1] = DB;\n"
	"}\n"<<'\0';

	ProgramGLSL * program =  new ProgramGLSL(out.str().c_str());
	if(program->IsNative())
	{
		return program;
	}
	else
	{
		delete program;
		return NULL;
	}
}

ProgramGLSL* ShaderBagPKSL::LoadDescriptorProgramPKSL()
{
	//one shader outpout 128/8 = 16 , each fragout encodes 4
	//const double twopi = 2.0*3.14159265358979323846;
	//const double rpi  = 8.0/twopi;
	ostringstream out;
	out<<setprecision(8);

	if(GlobalUtil::_KeepShaderLoop)
	{
		out << 	"#define REPEAT4(FUNCTION)\\\n"
				"for(int i = 0; i < 4; ++i)\\\n"
				"{\\\n"
				"	FUNCTION(i);\\\n"
				"}\n";
	}else
	{
		//loop unroll for ATI
		out <<  "#define REPEAT4(FUNCTION)\\\n"
				"FUNCTION(0);\\\n"
				"FUNCTION(1);\\\n"
				"FUNCTION(2);\\\n"
				"FUNCTION(3);\n";
	}

	out<<"\n"
	"#define M_PI 3.14159265358979323846\n"
	"#define TWO_PI (2.0*M_PI)\n"
	"#define RPI 1.2732395447351626861510701069801\n"
	"#define WF size.z\n"
	"uniform sampler2DRect tex;			\n"
	"uniform sampler2DRect gtex;			\n"
	"uniform sampler2DRect otex;			\n"
	"uniform vec4		dsize;				\n"
	"uniform vec3		size;				\n"
	"void main()			\n"
	"{\n"
	"	vec2 dim	= size.xy;	//image size			\n"
	"	float index = dsize.x*floor(gl_TexCoord[0].y * 0.5) + gl_TexCoord[0].x;\n"
	"	float idx = 8.0* fract(index * 0.125) + 8.0 * floor(2.0* fract(gl_TexCoord[0].y * 0.5));		\n"
	"	index = floor(index*0.125)+ 0.49;  \n"
	"	vec2 coord = floor( vec2( mod(index, dsize.z), index*dsize.w)) + 0.5 ;\n"
	"	vec2 pos = texture2DRect(tex, coord).xy;		\n"
	"	if(any(lessThan(pos.xy, vec2(1.0))) || any(greaterThan(pos.xy, dim-1.0))) "
	"	//discard;	\n"
	"	{ gl_FragData[0] = gl_FragData[1] = vec4(0.0); return; }\n"
	"	float anglef = texture2DRect(tex, coord).z;\n"
	"	if(anglef > M_PI) anglef -= TWO_PI;\n"
	"	float sigma = texture2DRect(tex, coord).w; \n"
	"	float spt  = abs(sigma * WF);	//default to be 3*sigma	\n";
	//rotation
	out<<
	"	vec4 cscs, rots;						\n"
	"	cscs.x = cos(anglef); cscs.y = sin(anglef);	\n"
	"	cscs.zw = - cscs.xy;							\n"
	"	rots = cscs /spt;								\n"
	"	cscs *= spt; \n";

	//here cscs is actually (cos, sin, -cos, -sin) * (factor: 3)*sigma
	//and rots is  (cos, sin, -cos, -sin ) /(factor*sigma)
	//devide the 4x4 sift grid into 16 1x1 block, and each corresponds to a shader thread
	//To use linear interoplation, 1x1 is increased to 2x2, by adding 0.5 to each side
	out<<
	"	vec4 temp; vec2 pt, offsetpt;				\n"
	"	/*the fraction part of idx is .5*/			\n"
	"	offsetpt.x = 4.0* fract(idx*0.25) - 2.0;				\n"
	"	offsetpt.y = floor(idx*0.25) - 1.5;			\n"
	"	temp = cscs.xwyx*offsetpt.xyxy;				\n"
	"	pt = pos + temp.xz + temp.yw;				\n";

	//get a horizontal bounding box of the rotated rectangle
	out<<
	"	vec2 bwin = abs(cscs.xy);					\n"
	"	float bsz = bwin.x + bwin.y;					\n"
	"	vec4 sz;					\n"
	"	sz.xy = max(pt - vec2(bsz), vec2(2,2));\n"
	"	sz.zw = min(pt + vec2(bsz), dim - vec2(3));		\n"
	"	sz = floor(sz * 0.5)+0.5;"; //move sample point to pixel center
	//get voting for two box

	out<<"\n"
	"	vec4 DA, DB;   vec2 spos;			\n"
	"	DA = DB  = vec4(0.0, 0.0, 0.0, 0.0);		\n"
	"	vec4 nox = vec4(0.0, rots.xy, rots.x + rots.y);					\n"
	"	vec4 noy = vec4(0.0, rots.wx, rots.w + rots.x);					\n"
	"	for(spos.y = sz.y; spos.y <= sz.w;	spos.y+=1.0)				\n"
	"	{																\n"
	"		for(spos.x = sz.x; spos.x <= sz.z;	spos.x+=1.0)			\n"
	"		{															\n"
	"			vec2 tpt = spos * 2.0 - pt - 0.5;					\n"
	"			vec4 temp = rots.xywx * tpt.xyxy;						\n"
	"			vec2 temp2 = temp.xz + temp.yw;						\n"
	"			vec4 nx = temp2.x + nox;								\n"
	"			vec4 ny = temp2.y + noy;			\n"
	"			vec4 nxn = abs(nx), nyn = abs(ny);						\n"
	"			bvec4 inside = lessThan(max(nxn, nyn) , vec4(1.0));	\n"
	"			if(any(inside))\n"
	"			{\n"
	"				vec4 gg = texture2DRect(gtex, spos);\n"
	"				vec4 oo = texture2DRect(otex, spos);\n"
	"				vec4 theta0 = (anglef - oo)*RPI;\n"
	"				vec4 theta = 8.0 * fract(1.0 + 0.125 * theta0);			\n"
	"				vec4 theta1 = floor(theta);								\n"
	"				vec4 diffx = nx + offsetpt.x, diffy = ny + offsetpt.y;	\n"
	"				vec4 ww = exp(-0.125 * (diffx * diffx + diffy * diffy ));	\n"
	"				vec4 weight = (vec4(1) - nxn) * (vec4(1) - nyn) * gg * ww; \n"
	"				vec4 weight2 = (theta - theta1) * weight;				\n"
	"				vec4 weight1 = weight - weight2;						\n"
	"	#define ADD_DESCRIPTOR(i) \\\n"
	"				if(inside[i])\\\n"
	"				{\\\n"
	"					DA += vec4(equal(vec4(theta1[i]), vec4(0, 1, 2, 3)))*weight1[i]; \\\n"
	"					DA += vec4(equal(vec4(theta1[i]), vec4(7, 0, 1, 2)))*weight2[i]; \\\n"
	"					DB += vec4(equal(vec4(theta1[i]), vec4(4, 5, 6, 7)))*weight1[i]; \\\n"
	"					DB += vec4(equal(vec4(theta1[i]), vec4(3, 4, 5, 6)))*weight2[i]; \\\n"
	"				}\n"
	"				REPEAT4(ADD_DESCRIPTOR);\n"
	"			}\n"
	"		}\n"
	"	}\n";
	out<<
	"	 gl_FragData[0] = DA; gl_FragData[1] = DB;\n"
	"}\n"<<'\0';

	ProgramGLSL * program =  new ProgramGLSL(out.str().c_str());
	if(program->IsNative())
	{
		return program;
	}
	else
	{
		delete program;
		return NULL;
	}
}

void ShaderBagPKSL::LoadDescriptorShaderF2()
{

	ProgramGLSL * program = LoadDescriptorProgramPKSL();
	if( program )
	{
		s_descriptor_fp = program;
		_param_descriptor_gtex = glGetUniformLocation(*program, "gtex");
		_param_descriptor_otex = glGetUniformLocation(*program, "otex");
		_param_descriptor_size = glGetUniformLocation(*program, "size");
		_param_descriptor_dsize = glGetUniformLocation(*program, "dsize");
	}
}



void ShaderBagPKSL::SetSimpleOrientationInput(int oTex, float sigma, float sigma_step)
{
	glUniform1i(_param_orientation_gtex, 1);
	glUniform2f(_param_orientation_size, sigma, sigma_step);
}


void ShaderBagPKSL::SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int otex, float step)
{
	///
	glUniform1i(_param_orientation_gtex, 1);
	glUniform1i(_param_orientation_otex, 2);

	float size[4];
	size[0] = (float)width;
	size[1] = (float)height;
	size[2] = sigma;
	size[3] = step;
	glUniform4fv(_param_orientation_size, 1, size);
}

void ShaderBagPKSL::SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth,  float width, float height, float sigma)
{
    if(sigma == 0 && s_rect_description)
    {
        //rectangle description mode
        s_rect_description->UseProgram();
        GLint param_descriptor_gtex = glGetUniformLocation(*s_rect_description, "gtex");
		GLint param_descriptor_otex = glGetUniformLocation(*s_rect_description, "otex");
		GLint param_descriptor_size = glGetUniformLocation(*s_rect_description, "size");
		GLint param_descriptor_dsize = glGetUniformLocation(*s_rect_description, "dsize");
	    ///
	    glUniform1i(param_descriptor_gtex, 1);
	    glUniform1i(param_descriptor_otex, 2);

	    float dsize[4] ={dwidth, 1.0f/dwidth, fwidth, 1.0f/fwidth};
	    glUniform4fv(param_descriptor_dsize, 1, dsize);
	    float size[3];
	    size[0] = width;
	    size[1] = height;
	    size[2] = GlobalUtil::_DescriptorWindowFactor;
	    glUniform3fv(param_descriptor_size, 1, size);
    }else
    {
	    ///
	    glUniform1i(_param_descriptor_gtex, 1);
	    glUniform1i(_param_descriptor_otex, 2);


	    float dsize[4] ={dwidth, 1.0f/dwidth, fwidth, 1.0f/fwidth};
	    glUniform4fv(_param_descriptor_dsize, 1, dsize);
	    float size[3];
	    size[0] = width;
	    size[1] = height;
	    size[2] = GlobalUtil::_DescriptorWindowFactor;
	    glUniform3fv(_param_descriptor_size, 1, size);
    }

}


void ShaderBagPKSL::SetGenListEndParam(int ktex)
{
	glUniform1i(_param_genlist_end_ktex, 1);
}
void ShaderBagPKSL::SetGenListInitParam(int w, int h)
{
	float bbox[4] = {(w -1.0f) * 0.5f +0.25f, (w-1.0f) * 0.5f - 0.25f,  (h - 1.0f) * 0.5f + 0.25f, (h-1.0f) * 0.5f - 0.25f};
	glUniform4fv(_param_genlist_init_bbox, 1, bbox);
}

void ShaderBagPKSL::SetMarginCopyParam(int xmax, int ymax)
{
	float truncate[4];
	truncate[0] = (xmax - 0.5f) * 0.5f; //((xmax + 1)  >> 1) - 0.5f;
	truncate[1] = (ymax - 0.5f) * 0.5f; //((ymax + 1)  >> 1) - 0.5f;
	truncate[2] = (xmax %2 == 1)? 0.0f: 1.0f;
	truncate[3] = truncate[2] +  (((ymax % 2) == 1)? 0.0f : 2.0f);
	glUniform4fv(_param_margin_copy_truncate, 1,  truncate);
}
