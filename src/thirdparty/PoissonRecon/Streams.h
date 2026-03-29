/*
Copyright (c) 2022, Michael Kazhdan
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
#ifndef STREAMS_INCLUDED
#define STREAMS_INCLUDED

#include "Array.h"

namespace PoissonRecon
{
	//////////////////
	// BinaryStream //
	//////////////////
	struct BinaryStream
	{
		size_t ioBytes;
		BinaryStream( void ) : ioBytes(0) {}

		template< typename C > bool  read(       C &c ){ return  __read( (      Pointer( unsigned char ) )GetPointer( c ) , sizeof(C) ); }
		template< typename C > bool write( const C &c ){ return __write( ( ConstPointer( unsigned char ) )GetPointer( c ) , sizeof(C) ); }
		template< typename C > bool write(       C &c ){ return __write( ( ConstPointer( unsigned char ) )GetPointer( c ) , sizeof(C) ); }
		template< typename C > bool  read(      Pointer( C ) c , size_t sz ){ return  __read( (      Pointer( unsigned char ) )c , sizeof(C)*sz ); }
		template< typename C > bool write( ConstPointer( C ) c , size_t sz ){ return __write( ( ConstPointer( unsigned char ) )c , sizeof(C)*sz ); }
		template< typename C > bool write(      Pointer( C ) c , size_t sz ){ return __write( ( ConstPointer( unsigned char ) )c , sizeof(C)*sz ); }

		template< typename C >
		bool read( std::vector< std::vector< C > > &c )
		{
			size_t sz;
			if( !read( sz ) ) return false;
			c.resize( sz );
			bool ret = true;
			for( size_t i=0 ; i<sz && ret ; i++ ) ret &= read( c[i] );
			return ret;
		}

		template< typename C >
		bool read( std::vector< C > &c )
		{
			size_t sz;
			if( !read( sz ) ) return false;
			c.resize( sz );
			if( sz ) return read( GetPointer( c ) , sz );
			else return true;
		}

		template< typename C >
		bool write( const std::vector< std::vector< C > > &c )
		{
			size_t sz = c.size();
			if( !write( sz ) ) return false;
			bool ret = true;
			for( size_t i=0 ; i<sz && ret ; i++ ) ret &= write( c[i] );
			return ret;
		}

		template< typename C >
		bool write( const std::vector< C > &c )
		{
			size_t sz = c.size();
			if( !write( sz ) ) return false;
			if( sz ) return write( GetPointer( c ) , sz );
			else return true;
		}

		template< typename C >
		bool write( std::vector< std::vector< C > > &c )
		{
			size_t sz = c.size();
			if( !write( sz ) ) return false;
			bool ret = true;
			for( size_t i=0 ; i<sz && ret ; i++ ) ret &= write( c[i] );
			return ret;
		}

		template< typename C >
		bool write( std::vector< C > &c )
		{
			size_t sz = c.size();
			if( !write( sz ) ) return false;
			if( sz ) return write( GetPointer( c )  , sz );
			else return true;
		}

		bool read( std::string &str )
		{
			size_t sz;
			if( !read( sz ) ) return false;
			char *_str = new char[sz+1];
			bool ret = read( GetPointer( _str , sz+1 ) , sz+1 );
			if( ret ) str = std::string( _str );
			delete[] _str;
			return ret;
		}

		bool write( const std::string &str )
		{
			size_t sz = strlen( str.c_str() );
			if( !write( sz ) ) return false;
			return write( GetPointer( str.c_str() , sz+1 ) , sz+1 );
		}

		bool write( std::string &str )
		{
			size_t sz = strlen( str.c_str() );
			if( !write( sz ) ) return false;
			return write( GetPointer( str.c_str() , sz+1 ) , sz+1 );
		}
	protected:
		virtual bool  _read(      Pointer( unsigned char ) ptr , size_t sz ) = 0;
		virtual bool _write( ConstPointer( unsigned char ) ptr , size_t sz ) = 0;
		bool _write( Pointer( unsigned char ) ptr , size_t sz ){ return _write( ( ConstPointer( unsigned char ) )ptr , sz ); }
		bool  __read(      Pointer( unsigned char ) ptr , size_t sz ){ ioBytes += sz ; return  _read( ptr , sz ); }
		bool __write( ConstPointer( unsigned char ) ptr , size_t sz ){ ioBytes += sz ; return _write( ptr , sz ); }
		bool __write(      Pointer( unsigned char ) ptr , size_t sz ){ ioBytes += sz ; return _write( ( ConstPointer( unsigned char ) )ptr , sz ); }
	};

	struct FileStream : public BinaryStream
	{
		FileStream( FILE *fp ) : _fp(fp){}
		void reset( void ){ std::rewind( _fp ); }
	protected:
		FILE *_fp;
		bool  _read(      Pointer( unsigned char ) ptr , size_t sz ){ return  fread( ptr , sizeof(unsigned char) , sz , _fp )==sz; }
		bool _write( ConstPointer( unsigned char ) ptr , size_t sz ){ return fwrite( ptr , sizeof(unsigned char) , sz , _fp )==sz; }
	};

	template< typename C >
	struct Serializer
	{
		virtual size_t size( void ) const = 0;
		virtual void   serialize( const C & , Pointer( char ) buffer ) const = 0;
		virtual void deserialize( ConstPointer( char ) buffer , C & ) const = 0;
	};
}

#endif // STREAMS_INCLUDED
