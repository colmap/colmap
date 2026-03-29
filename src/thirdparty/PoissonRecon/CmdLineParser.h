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

#ifndef CMD_LINE_PARSER_INCLUDED
#define CMD_LINE_PARSER_INCLUDED

#include <stdarg.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cassert>
#include <string.h>
#include <vector>
#include "Geometry.h"

namespace PoissonRecon
{
#ifdef WIN32
	int strcasecmp( const char* c1 , const char* c2 );
#endif // WIN32

	class CmdLineReadable
	{
	public:
		bool set;
		char *name;
		CmdLineReadable( const char *name );
		virtual ~CmdLineReadable( void );
		virtual int read( char** argv , int argc );
		virtual void writeValue( char* str ) const;
	};

	template< typename Type > struct CmdLineType;

	template< typename Type >
	struct CmdLineType
	{
		static void WriteValue( Type t , char* str );
		static void CleanUp( Type* t ){}
		static Type Initialize( void ){ return Type(); }
		static Type Copy( Type t ){ return t; }
		static Type StringToType( const char *str );
	};

	template< typename Real , unsigned int Dim >
	struct CmdLineType< Point< Real , Dim > >
	{
		using Type = Point< Real , Dim >;
		static void WriteValue( Type t , char* str );
		static void CleanUp( Type* t ){}
		static Type Initialize( void ){ return Type(); }
		static Type Copy( Type t ){ return t; }
		static Type StringToType( const char *str );
	};
	template< class Type >
	class CmdLineParameter : public CmdLineReadable
	{
	public:
		Type value;
		CmdLineParameter( const char *name );
		CmdLineParameter( const char *name , Type v );
		~CmdLineParameter( void );
		int read( char** argv , int argc );
		void writeValue( char* str ) const;
		bool expectsArg( void ) const { return true; }
	};

	template< class Type , int Dim >
	class CmdLineParameterArray : public CmdLineReadable
	{
	public:
		Type values[Dim];
		CmdLineParameterArray( const char *name, const Type* v=NULL );
		~CmdLineParameterArray( void );
		int read( char** argv , int argc );
		void writeValue( char* str ) const;
		bool expectsArg( void ) const { return true; }
	};

	template< class Type >
	class CmdLineParameters : public CmdLineReadable
	{
	public:
		int count;
		Type *values;
		CmdLineParameters( const char* name );
		~CmdLineParameters( void );
		int read( char** argv , int argc );
		void writeValue( char* str ) const;
		bool expectsArg( void ) const { return true; }
	};

	void CmdLineParse( int argc , char **argv, CmdLineReadable** params );
	char* FileExtension( char* fileName );
	char* LocalFileName( char* fileName );
	char* DirectoryName( char* fileName );
	char* GetFileExtension( const char* fileName );
	char* GetLocalFileName( const char* fileName );
	char** ReadWords( const char* fileName , int& cnt );

#include "CmdLineParser.inl"
}
#endif // CMD_LINE_PARSER_INCLUDED
