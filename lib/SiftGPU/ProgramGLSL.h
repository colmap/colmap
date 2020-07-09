////////////////////////////////////////////////////////////////////////////
//	File:		ProgramGLSL.h
//	Author:		Changchang Wu
//	Description : Interface for ProgramGLSL classes
//		ProgramGLSL:	Glsl Program
//		FilterGLSL:		Glsl Gaussian Filters
//		ShaderBag:	    base class of ShaderBagPKSL and ShaderBagGLSL
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


#ifndef _PROGRAM_GLSL_H
#define _PROGRAM_GLSL_H


#include "ProgramGPU.h"

class ProgramGLSL:public ProgramGPU
{
	class ShaderObject
	{
		GLuint		_shaderID;
		int			_type;
		int			_compiled;
		static int ReadShaderFile(const char * source,  char *& code);
		void CheckCompileLog();
	public:
		void PrintCompileLog(ostream & os  );
		int inline IsValidShaderObject(){	return _shaderID && _compiled;}
		int IsValidVertexShader();
		int IsValidFragmentShader();
		GLuint GetShaderID(){return _shaderID;}
		~ShaderObject();
		ShaderObject(int shadertype,  const char * source, int filesource =0);
	};

protected:
	int			_linked;
	GLint		_TextureParam0;
	GLuint		_programID;
private:
	void AttachShaderObject(ShaderObject& shader);
	void DetachShaderObject(ShaderObject& shader);

public:
	void ReLink();
	int IsNative();
	int	UseProgram();
	void PrintLinkLog(std::ostream&os);
	int ValidateProgram();
	void CheckLinkLog();
	int LinkProgram();
	operator GLuint (){return _programID;}
    virtual void * GetProgramID() { return (void*) _programID; }
public:
	ProgramGLSL();
	~ProgramGLSL();
	ProgramGLSL(const char* frag_source);
};


class GLTexImage;
class FilterGLSL : public FilterProgram
{
private:
	ProgramGPU* CreateFilterH(float kernel[], int width);
	ProgramGPU* CreateFilterV(float kernel[], int height);
	ProgramGPU* CreateFilterHPK(float kernel[], int width);
	ProgramGPU* CreateFilterVPK(float kernel[], int height);
public:
    void MakeFilterProgram(float kernel[],  int width);
public:
    FilterGLSL(float sigma) ;
};

class SiftParam;

/////////////////////////////////////////////////////////////////////////////////
//class ShaderBag
//desciption:	pure virtual class
//				provides storage and usage interface of all the shaders for SIFT
//				two implementations are  ShaderBagPKSL and ShaderBagGLSL
/////////////////////////////////////////////////////////////////////////////////
class ShaderBag
{
public:
	//shader:	rgb to gray
	ProgramGPU  * s_gray;
	//shader:	copy keypoint to PBO
	ProgramGPU  * s_copy_key;
	//shader:	debug view
	ProgramGPU  * s_debug;
	//shader:	orientation
	//shader:	assign simple orientation to keypoints if hardware is low
	ProgramGPU  * s_orientation;
	//shader:	display gaussian levels
	ProgramGPU  * s_display_gaussian;
	//shader:	display difference of gassian
	ProgramGPU  * s_display_dog;
	//shader:	display  gradient
	ProgramGPU  * s_display_grad;
	//shader:	display keypoints as red(maximum) and blue (minimum)
	ProgramGPU  * s_display_keys;
	//shader:	up/down-sample
	ProgramGPU  * s_sampling;
	//shader:	compute gradient/dog
	ProgramGPU  * s_grad_pass;
	ProgramGPU  * s_dog_pass;
	//shader:   keypoint detection in one pass
	ProgramGPU  * s_keypoint;
	ProgramGPU  * s_seperate_sp;
	//shader:   feature list generations..
	ProgramGPU	* s_genlist_init_tight;
	ProgramGPU	* s_genlist_init_ex;
	ProgramGPU	* s_genlist_histo;
	ProgramGPU	* s_genlist_start;
	ProgramGPU	* s_genlist_step;
	ProgramGPU	* s_genlist_end;
	ProgramGPU	* s_zero_pass;
	//shader:	generate vertex to display SIFT as a square
	ProgramGPU  * s_vertex_list;
	//shader:	descriptor
	ProgramGPU  * s_descriptor_fp;
	//shader:	copy pixels to margin
	ProgramGPU	* s_margin_copy;
public:
	FilterProgram  *         f_gaussian_skip0;
	vector<FilterProgram*>   f_gaussian_skip0_v;
	FilterProgram  *         f_gaussian_skip1;
	FilterProgram  **        f_gaussian_step;
    int                     _gaussian_step_num;
public:
	virtual void SetGenListInitParam(int w, int h){};
	virtual void SetGenListEndParam(int ktex){};
	virtual void SetMarginCopyParam(int xmax, int ymax){};
	virtual void LoadDescriptorShader(){};
	virtual void SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth, float width, float height, float sigma){};
	virtual void SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex, float step){};
	virtual void SetSimpleOrientationInput(int oTex, float sigma, float sigma_step){};
	virtual void LoadOrientationShader() =0;
	virtual void SetGenListStartParam(float width, int tex0) =0;
	virtual void LoadGenListShader(int ndoglev, int nlev)=0;
	virtual void UnloadProgram()=0;
	virtual void LoadKeypointShader(float threshold, float edgeTrheshold) = 0;
	virtual void LoadFixedShaders()=0;
	virtual void LoadDisplayShaders() = 0;
	virtual void SetDogTexParam(int texU, int texD)=0;
	virtual void SetGradPassParam(int texP=0){}
	virtual void SetGenListStepParam(int tex, int tex0) = 0;
	virtual void SetGenVBOParam( float width, float fwidth, float size)=0;
public:
    void CreateGaussianFilters(SiftParam&param);
    void SelectInitialSmoothingFilter(int octave_min, SiftParam&param);
    void LoadDynamicShaders(SiftParam& param);
	ShaderBag();
	virtual ~ShaderBag();
};


class ShaderBagGLSL:public ShaderBag
{
	GLint _param_dog_texu;
	GLint _param_dog_texd;
	GLint _param_ftex_width;
	GLint _param_genlist_start_tex0;
	GLint _param_genlist_step_tex0;
	GLint _param_genvbo_size;
	GLint _param_orientation_gtex;
	GLint _param_orientation_size;
	GLint _param_orientation_stex;
	GLint _param_margin_copy_truncate;
	GLint _param_genlist_init_bbox;
	GLint _param_descriptor_gtex;
	GLint _param_descriptor_size;
	GLint _param_descriptor_dsize;
public:
	virtual void SetMarginCopyParam(int xmax, int ymax);
	void SetSimpleOrientationInput(int oTex, float sigma, float sigma_step);
	void LoadOrientationShader();
	void LoadDescriptorShaderF2();
	virtual void LoadDescriptorShader();
	virtual void SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex = 0, float step = 1.0f);
	virtual void SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth, float width, float height, float sigma);
	static void  WriteOrientationCodeToStream(ostream& out);
	static ProgramGLSL* LoadGenListStepShader(int start, int step);
	virtual void SetGenListInitParam(int w, int h);
	virtual void SetGenListStartParam(float width, int tex0);
	virtual void LoadGenListShader(int ndoglev, int nlev);
	virtual void UnloadProgram();
	virtual void LoadKeypointShader(float threshold, float edgeTrheshold);
	virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
	virtual void SetDogTexParam(int texU, int texD);
	virtual void SetGenListStepParam(int tex, int tex0);
	virtual void SetGenVBOParam( float width, float fwidth, float size);
	virtual ~ShaderBagGLSL(){}
};


class ShaderBagPKSL:public ShaderBag
{
private:
	GLint	_param_dog_texu;
	GLint	_param_dog_texd;
	GLint	_param_dog_texi;
	GLint	_param_margin_copy_truncate;
	GLint	_param_grad_pass_texp;
	GLint	_param_genlist_init_bbox;
	GLint	_param_genlist_start_tex0;
	GLint	_param_ftex_width;
	GLint	_param_genlist_step_tex0;
	GLint	_param_genlist_end_ktex;
	GLint	_param_genvbo_size;
	GLint	_param_orientation_gtex;
	GLint	_param_orientation_otex;
	GLint	_param_orientation_size;
	GLint	_param_descriptor_gtex;
	GLint	_param_descriptor_otex;
	GLint	_param_descriptor_size;
	GLint	_param_descriptor_dsize;

    //
    ProgramGLSL* s_rect_description;
public:
    ShaderBagPKSL () {s_rect_description = NULL; }
	virtual ~ShaderBagPKSL() {if(s_rect_description) delete s_rect_description; }
	virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
	virtual void LoadOrientationShader() ;
	virtual void SetGenListStartParam(float width, int tex0) ;
	virtual void LoadGenListShader(int ndoglev, int nlev);
	virtual void UnloadProgram();
	virtual void LoadKeypointShader(float threshold, float edgeTrheshold) ;
	virtual void LoadDescriptorShader();
	virtual void LoadDescriptorShaderF2();
    static ProgramGLSL* LoadDescriptorProgramRECT();
	static ProgramGLSL* LoadDescriptorProgramPKSL();
/////////////////
	virtual void SetDogTexParam(int texU, int texD);
	virtual void SetGradPassParam(int texP);
	virtual void SetGenListStepParam(int tex, int tex0);
	virtual void SetGenVBOParam( float width, float fwidth, float size);
	virtual void SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth, float width, float height, float sigma);
	virtual void SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex, float step);
	virtual void SetSimpleOrientationInput(int oTex, float sigma, float sigma_step);
	virtual void SetGenListEndParam(int ktex);
	virtual void SetGenListInitParam(int w, int h);
	virtual void SetMarginCopyParam(int xmax, int ymax);
};


#endif

