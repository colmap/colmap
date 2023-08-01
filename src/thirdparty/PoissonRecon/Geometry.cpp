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
#include "Geometry.h"
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#endif // _WIN32

///////////////////
// CoredMeshData //
///////////////////

TriangulationEdge::TriangulationEdge(void){pIndex[0]=pIndex[1]=tIndex[0]=tIndex[1]=-1;}
TriangulationTriangle::TriangulationTriangle(void){eIndex[0]=eIndex[1]=eIndex[2]=-1;}

///////////////////////////
// BufferedReadWriteFile //
///////////////////////////
BufferedReadWriteFile::BufferedReadWriteFile( char* fileName , int bufferSize )
{
	_bufferIndex = 0;
	_bufferSize = bufferSize;
	if( fileName ) strcpy( _fileName , fileName ) , tempFile = false , _fp = fopen( _fileName , "w+b" );
	else
	{
		strcpy( _fileName , "PR_XXXXXX" );
#ifdef _WIN32
		_mktemp( _fileName );
		_fp = fopen( _fileName , "w+b" );
#else // !_WIN32
		_fp = fdopen( mkstemp( _fileName ) , "w+b" );
#endif // _WIN32
		tempFile = true;
	}
	if( !_fp ) fprintf( stderr , "[ERROR] Failed to open file: %s\n" , _fileName ) , exit( 0 );
	_buffer = (char*) malloc( _bufferSize );
}
BufferedReadWriteFile::~BufferedReadWriteFile( void )
{
	free( _buffer );
	fclose( _fp );
	if( tempFile ) remove( _fileName );
}
void BufferedReadWriteFile::reset( void )
{
	if( _bufferIndex ) fwrite( _buffer , 1 , _bufferIndex , _fp );
	_bufferIndex = 0;
	fseek( _fp , 0 , SEEK_SET );
	_bufferIndex = 0;
	_bufferSize = fread( _buffer , 1 , _bufferSize , _fp );
}
bool BufferedReadWriteFile::write( const void* data , size_t size )
{
	if( !size ) return true;
	char* _data = (char*) data;
	size_t sz = _bufferSize - _bufferIndex;
	while( sz<=size )
	{
		memcpy( _buffer+_bufferIndex , _data , sz );
		fwrite( _buffer , 1 , _bufferSize , _fp );
		_data += sz;
		size -= sz;
		_bufferIndex = 0;
		sz = _bufferSize;
	}
	if( size )
	{
		memcpy( _buffer+_bufferIndex , _data , size );
		_bufferIndex += size;
	}
	return true;
}
bool BufferedReadWriteFile::read( void* data , size_t size )
{
	if( !size ) return true;
	char *_data = (char*) data;
	size_t sz = _bufferSize - _bufferIndex;
	while( sz<=size )
	{
		if( size && !_bufferSize ) return false;
		memcpy( _data , _buffer+_bufferIndex , sz );
		_bufferSize = fread( _buffer , 1 , _bufferSize , _fp );
		_data += sz;
		size -= sz;
		_bufferIndex = 0;
		if( !size ) return true;
		sz = _bufferSize;
	}
	if( size )
	{
		if( !_bufferSize ) return false;
		memcpy( _data , _buffer+_bufferIndex , size );
		_bufferIndex += size;
	}
	return true;
}