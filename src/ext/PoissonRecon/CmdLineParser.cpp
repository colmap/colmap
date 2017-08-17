/*
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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CmdLineParser.h"


#ifdef _WIN32
int strcasecmp(char* c1,char* c2){return _stricmp(c1,c2);}
#endif

cmdLineReadable::cmdLineReadable(const char* name)
{
	set=false;
	this->name=new char[strlen(name)+1];
	strcpy(this->name,name);
}
cmdLineReadable::~cmdLineReadable(void)
{
	if(name) delete[] name;
	name=NULL;
}
int cmdLineReadable::read(char**,int){
	set=true;
	return 0;
}
void cmdLineReadable::writeValue(char* str)
{
	str[0] = 0;
}

////////////////
// cmdLineInt //
////////////////
cmdLineInt::cmdLineInt(const char* name) : cmdLineReadable(name) {value=0;}
cmdLineInt::cmdLineInt(const char* name,const int& v) : cmdLineReadable(name) {value=v;}
int cmdLineInt::read(char** argv,int argc){
	if(argc>0){
		value=atoi(argv[0]);
		set=true;
		return 1;
	}
	else{return 0;}
}
void cmdLineInt::writeValue(char* str)
{
	sprintf(str,"%d",value);
}

//////////////////
// cmdLineFloat //
//////////////////
cmdLineFloat::cmdLineFloat(const char* name) : cmdLineReadable(name) {value=0;}
cmdLineFloat::cmdLineFloat(const char* name, const float& v) : cmdLineReadable(name) {value=v;}
int cmdLineFloat::read(char** argv,int argc){
	if(argc>0){
		value=(float)atof(argv[0]);
		set=true;
		return 1;
	}
	else{return 0;}
}
void cmdLineFloat::writeValue(char* str)
{
	sprintf(str,"%f",value);
}

///////////////////
// cmdLineString //
///////////////////
cmdLineString::cmdLineString(const char* name) : cmdLineReadable(name) {value=NULL;}
cmdLineString::~cmdLineString(void)
{
	if(value)	delete[] value;
	value=NULL;
}
int cmdLineString::read(char** argv,int argc){
	if(argc>0)
	{
		value=new char[strlen(argv[0])+1];
		strcpy(value,argv[0]);
		set=true;
		return 1;
	}
	else{return 0;}
}
void cmdLineString::writeValue(char* str)
{
	sprintf(str,"%s",value);
}

////////////////////
// cmdLineStrings //
////////////////////
cmdLineStrings::cmdLineStrings(const char* name,int Dim) : cmdLineReadable(name)
{
	this->Dim=Dim;
	values=new char*[Dim];
	for(int i=0;i<Dim;i++)	values[i]=NULL;
}
cmdLineStrings::~cmdLineStrings(void)
{
	for(int i=0;i<Dim;i++)
	{
		if(values[i])	delete[] values[i];
		values[i]=NULL;
	}
	delete[] values;
	values=NULL;
}
int cmdLineStrings::read(char** argv,int argc)
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
	else	return 0;
}
void cmdLineStrings::writeValue(char* str)
{
	char* temp=str;
	for(int i=0;i<Dim;i++)
	{
		sprintf(temp,"%s ",values[i]);
		temp=str+strlen(str);
	}
}


char* GetFileExtension(char* fileName){
	char* fileNameCopy;
	char* ext=NULL;
	char* temp;

	fileNameCopy=new char[strlen(fileName)+1];
	assert(fileNameCopy);
	strcpy(fileNameCopy,fileName);
	temp=strtok(fileNameCopy,".");
	while(temp!=NULL)
	{
		if(ext!=NULL){delete[] ext;}
		ext=new char[strlen(temp)+1];
		assert(ext);
		strcpy(ext,temp);
		temp=strtok(NULL,".");
	}
	delete[] fileNameCopy;
	return ext;
}
char* GetLocalFileName(char* fileName){
	char* fileNameCopy;
	char* name=NULL;
	char* temp;

	fileNameCopy=new char[strlen(fileName)+1];
	assert(fileNameCopy);
	strcpy(fileNameCopy,fileName);
	temp=strtok(fileNameCopy,"\\");
	while(temp!=NULL){
		if(name!=NULL){delete[] name;}
		name=new char[strlen(temp)+1];
		assert(name);
		strcpy(name,temp);
		temp=strtok(NULL,"\\");
	}
	delete[] fileNameCopy;
	return name;
}

void cmdLineParse(int argc, char **argv,int num,cmdLineReadable** readable,int dumpError)
{
	int i,j;
	while (argc > 0)
	{
		if (argv[0][0] == '-' && argv[0][1]=='-')
		{
			for(i=0;i<num;i++)
			{
				if (!strcmp(&argv[0][2],readable[i]->name))
				{
					argv++, argc--;
					j=readable[i]->read(argv,argc);
					argv+=j,argc-=j;
					break;
				}
			}
			if(i==num){
				if(dumpError)
				{
					fprintf(stderr, "invalid option: %s\n",*argv);
					fprintf(stderr, "possible options are:\n");
					for(i=0;i<num;i++)	fprintf(stderr, "  %s\n",readable[i]->name);
				}
				argv++, argc--;
			}
		}
		else
		{
			if(dumpError)
			{
				fprintf(stderr, "invalid option: %s\n", *argv);
				fprintf(stderr, "  options must start with a \'--\'\n");
			}
			argv++, argc--;
		}
	}
}
char** ReadWords(const char* fileName,int& cnt)
{
	char** names;
	char temp[500];
	FILE* fp;

	fp=fopen(fileName,"r");
	if(!fp){return NULL;}
	cnt=0;
	while(fscanf(fp," %s ",temp)==1){cnt++;}
	fclose(fp);

	names=new char*[cnt];
	if(!names){return NULL;}

	fp=fopen(fileName,"r");
	if(!fp){
		delete[] names;
		cnt=0;
		return NULL;
	}
	cnt=0;
	while(fscanf(fp," %s ",temp)==1){
		names[cnt]=new char[strlen(temp)+1];
		if(!names){
			for(int j=0;j<cnt;j++){delete[] names[j];}
			delete[] names;
			cnt=0;
			fclose(fp);
			return NULL;
		}
		strcpy(names[cnt],temp);
		cnt++;
	}
	fclose(fp);
	return names;
}
