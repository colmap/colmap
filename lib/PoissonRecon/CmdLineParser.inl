/* -*- C++ -*-
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

/////////////////////
// cmdLineIntArray //
/////////////////////
template<int Dim>
cmdLineIntArray<Dim>::cmdLineIntArray(const char* name) : cmdLineReadable(name)
{
	for(int i=0;i<Dim;i++)	values[i]=0;
}
template<int Dim>
cmdLineIntArray<Dim>::cmdLineIntArray(const char* name,const int v[Dim]) : cmdLineReadable(name)
{
	for(int i=0;i<Dim;i++)	values[i]=v[i];
}
template<int Dim>
int cmdLineIntArray<Dim>::read(char** argv,int argc)
{
	if(argc>=Dim)
	{
		for(int i=0;i<Dim;i++)	values[i]=atoi(argv[i]);
		set=true;
		return Dim;
	}
	else{return 0;}
}
template<int Dim>
void cmdLineIntArray<Dim>::writeValue(char* str)
{
	char* temp=str;
	for(int i=0;i<Dim;i++)
	{
		sprintf(temp,"%d ",values[i]);
		temp=str+strlen(str);
	}
}

///////////////////////
// cmdLineFloatArray //
///////////////////////
template<int Dim>
cmdLineFloatArray<Dim>::cmdLineFloatArray(const char* name) : cmdLineReadable(name)
{
	for(int i=0;i<Dim;i++)	values[i]=0;
}
template<int Dim>
cmdLineFloatArray<Dim>::cmdLineFloatArray(const char* name,const float f[Dim]) : cmdLineReadable(name)
{
	for(int i=0;i<Dim;i++)	values[i]=f[i];
}
template<int Dim>
int cmdLineFloatArray<Dim>::read(char** argv,int argc)
{
	if(argc>=Dim)
	{
		for(int i=0;i<Dim;i++)	values[i]=(float)atof(argv[i]);
		set=true;
		return Dim;
	}
	else{return 0;}
}
template<int Dim>
void cmdLineFloatArray<Dim>::writeValue(char* str)
{
	char* temp=str;
	for(int i=0;i<Dim;i++)
	{
		sprintf(temp,"%f ",values[i]);
		temp=str+strlen(str);
	}
}


////////////////////////
// cmdLineStringArray //
////////////////////////
template<int Dim>
cmdLineStringArray<Dim>::cmdLineStringArray(const char* name) : cmdLineReadable(name)
{
	for(int i=0;i<Dim;i++)	values[i]=NULL;
}
template<int Dim>
cmdLineStringArray<Dim>::~cmdLineStringArray(void)
{
	for(int i=0;i<Dim;i++)
	{
		if(values[i])	delete[] values[i];
		values[i]=NULL;
	}
}
template<int Dim>
int cmdLineStringArray<Dim>::read(char** argv,int argc)
{
	if(argc>=Dim)
	{
		for(int i=0;i<Dim;i++)
		{
			values[i]=new char[strlen(argv[i])+1];
			strcpy(values[i],argv[i]);
		}
		set=true;
		return Dim;
	}
	else{return 0;}
}
template<int Dim>
void cmdLineStringArray<Dim>::writeValue(char* str)
{
	char* temp=str;
	for(int i=0;i<Dim;i++)
	{
		sprintf(temp,"%s ",values[i]);
		temp=str+strlen(str);
	}
}
