/*
Copyright (c) 2010, Michael Kazhdan
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

#ifndef PNG_INCLUDED
#define PNG_INCLUDED

#include <stdio.h>
#include <vector>
#include "Image.h"
#include "PNG/png.h"

namespace PoissonRecon
{
	struct PNGReader : public ImageReader
	{
		PNGReader( const char* fileName , unsigned int& width , unsigned int& height , unsigned int& channels );
		~PNGReader( void );
		unsigned int nextRow( unsigned char* row );
		static bool GetInfo( const char* fileName , unsigned int& width , unsigned int& height , unsigned int& channels );
	protected:
		png_structp _png_ptr;
		png_infop _info_ptr;
		png_infop _end_info ;
		FILE* _fp;
		unsigned char* _scratchRow;
		unsigned int _currentRow;
	};

	struct PNGWriter : public ImageWriter
	{
		PNGWriter( const char* fileName , unsigned int width , unsigned int height , unsigned int channels , unsigned int quality=100 );
		~PNGWriter( void );
		unsigned int nextRow( const unsigned char* row );
		unsigned int nextRows( const unsigned char* rows , unsigned int rowNum );
	protected:
		FILE* _fp;
		png_structp _png_ptr;
		png_infop _info_ptr;
		unsigned int _currentRow;
	};

#include "PNG.inl"
}

#endif //PNG_INCLUDED
