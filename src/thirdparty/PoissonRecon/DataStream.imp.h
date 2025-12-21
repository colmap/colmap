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

#ifndef DATA_STREAM_IMPLEMENTATION_INCLUDED
#define DATA_STREAM_IMPLEMENTATION_INCLUDED

#include <vector>
#include "DataStream.h"
#include "VertexFactory.h"
#include "Ply.h"

namespace PoissonRecon
{
	////////////////////////////////
	// Vector-backed data streams //
	////////////////////////////////
	template< typename Data >
	struct VectorBackedInputDataStream : public InputDataStream< Data >
	{
		VectorBackedInputDataStream( const std::vector< Data > &data ) : _data(data) , _current(0) {}
		void reset( void ) { _current = 0; }

		bool read( Data &d ){ if( _current<_data.size() ){ d = _data[_current++] ; return true; } else return false; }

	protected:
		const std::vector< Data > &_data;
		size_t _current;
	};

	template< typename Data >
	struct VectorBackedInputIndexedDataStream : public InputIndexedDataStream< Data >
	{
		VectorBackedInputIndexedDataStream( const std::vector< std::pair< size_t , Data > > &data ) : _data(data) , _current(0) {}
		void reset( void ) { _current = 0; }

		bool read( size_t &idx , Data &d )
		{
			if( _current<_data.size() )
			{
				idx = _data[_current].first;
				d = _data[_current].second;
				_current++;
				return true;
			}
			else return false;
		}

	protected:
		const std::vector< std::pair< size_t , Data > > &_data;
		size_t _current;
	};

	template< typename Data >
	struct VectorBackedOutputDataStream : public OutputDataStream< Data >
	{
		VectorBackedOutputDataStream( std::vector< Data > &data ) : _data(data) {}
		size_t write( const Data &d ){ size_t idx = _data.size() ; _data.push_back(d) ; return idx; }
		size_t size( void ) const { return _data.size(); }
	protected:
		std::vector< Data > &_data;
	};

	template< typename Data >
	struct VectorBackedOutputIndexedDataStream : public OutputIndexedDataStream< Data >
	{
		VectorBackedOutputIndexedDataStream( std::vector< std::pair< size_t , Data > > &data ) : _data(data) {}
		size_t write( const size_t &idx , const Data &d )
		{
			size_t sz = _data.size();
			_data.push_back( std::make_pair( idx , d ) );
			return idx;
		}
		size_t size( void ) const { return _data.size(); }
	protected:
		std::vector< std::pair< size_t , Data > > &_data;
	};


	////////////////////////////////////////////////////////////////////
	// File-backed data stream (assumes Data is statically allocated) //
	////////////////////////////////////////////////////////////////////
	template< typename Data >
	struct FileBackedInputDataStream : public InputDataStream< Data >
	{
		// It is assumed that the file pointer was open for binary reading
		FileBackedInputDataStream( FILE *fp ) : _fp(fp) {}
		void reset( void ){ fseek( _fp , 0 , SEEK_SET ); }
		bool read( Data &d ){ return fread( &d , sizeof(Data) , 1 , _fp )==1; }

	protected:
		FILE *_fp;
	};

	template< typename Data >
	struct FileBackedOutputDataStream : public OutputDataStream< Data >
	{
		// It is assumed that the file pointer was open for binary writing
		FileBackedOutputDataStream( FILE *fp ) : _fp(fp) {}
		size_t write( const Data &d ){ fwrite( &d , sizeof(Data) , 1 , _fp ) ; return _sz++; }
		size_t size( void ) const { return _sz; }

	protected:
		size_t _sz = 0;
		FILE *_fp;
	};

	/////////////////////////////////////////////////////////////////////////////////////////////
	// File-backed data stream  specialized for vectors (assumes Data is statically allocated) //
	/////////////////////////////////////////////////////////////////////////////////////////////
	template< typename Data >
	struct FileBackedInputDataStream< std::vector< Data > > : public InputDataStream< std::vector< Data > >
	{
		// It is assumed that the file pointer was open for binary reading
		FileBackedInputDataStream( FILE *fp ) : _fp(fp) {}
		void reset( void ){ fseek( _fp , 0 , SEEK_SET ); }
		bool read( std::vector< Data > &d )
		{
			unsigned int pSize;
			if( fread( &pSize , sizeof(unsigned int) , 1 , _fp )==1 )
			{
				d.resize( pSize );
				if( fread( &d[0] , sizeof(Data) , pSize , _fp )==pSize ) return true;
				MK_THROW( "Failed to read polygon from file" );
				return true;
			}
			else return false;
		}

	protected:
		FILE *_fp;
	};

	template< typename Data >
	struct FileBackedOutputDataStream< std::vector< Data > > : public OutputDataStream< std::vector< Data > >
	{
		// It is assumed that the file pointer was open for binary writing
		FileBackedOutputDataStream( FILE *fp ) : _fp(fp) {}
		size_t write( const std::vector< Data > &d )
		{
			unsigned int pSize = (unsigned int)d.size();
			fwrite( &pSize , sizeof(unsigned int) , 1 , _fp );
			fwrite( &d[0] , sizeof(Data) , pSize , _fp );
			return _sz++;
		}
		size_t size( void ) const { return _sz; }

	protected:
		size_t _sz = 0;
		FILE *_fp;
	};

	////////////////////////////////////////////////////////
	// File-backed stream with data desribed by a factory //
	////////////////////////////////////////////////////////
	template< typename Factory >
	struct FileBackedInputFactoryTypeStream : public InputDataStream< typename Factory::VertexType >
	{
		using Data = typename Factory::VertexType;
		// It is assumed that the file pointer was open for binary reading
		FileBackedInputFactoryTypeStream( FILE *fp , const Factory &factory ) : _fp(fp) , _factory(factory) , _buffer( NewPointer< char >( _factory.bufferSize() ) ) , _bufferSize( _factory.bufferSize() ) {}
		~FileBackedInputFactoryTypeStream( void ){ DeletePointer( _buffer ); }
		void reset( void ){ fseek( _fp , 0 , SEEK_SET ); }
		bool read( Data &d )
		{
			if( fread( _buffer , sizeof(unsigned char) , _bufferSize , _fp )==_bufferSize )
			{
				_factory.fromBuffer( _buffer , d );
				return true;
			}
			else return false;
			return _sz++;
		}

	protected:
		size_t _sz = 0;
		FILE *_fp;
		const Factory _factory;
		Pointer( char ) _buffer;
		const size_t _bufferSize;
	};

	template< typename Factory >
	struct FileBackedInputIndexedFactoryTypeStream : public InputIndexedDataStream< typename Factory::VertexType >
	{
		using Data = typename Factory::VertexType;
		// It is assumed that the file pointer was open for binary reading
		FileBackedInputIndexedFactoryTypeStream( FILE *fp , const Factory &factory ) : _fp(fp) , _factory(factory) , _buffer( NewPointer< char >( _factory.bufferSize() ) ) , _bufferSize( _factory.bufferSize() ) {}
		~FileBackedInputIndexedFactoryTypeStream( void ){ DeletePointer( _buffer ); }
		void reset( void ){ fseek( _fp , 0 , SEEK_SET ); }
		bool read( size_t &idx , Data &d )
		{
			if( fread( &idx , sizeof(size_t) , 1 , _fp )!=1 ) return false;
			if( fread( _buffer , sizeof(unsigned char) , _bufferSize , _fp )==_bufferSize )
			{
				_factory.fromBuffer( _buffer , d );
				return true;
			}
			else return false;
			return _sz++;
		}

	protected:
		size_t _sz = 0;
		FILE *_fp;
		const Factory _factory;
		Pointer( char ) _buffer;
		const size_t _bufferSize;
	};

	template< typename Factory >
	struct FileBackedOutputFactoryTypeStream : public OutputDataStream< typename Factory::VertexType >
	{
		using Data = typename Factory::VertexType;
		// It is assumed that the file pointer was open for binary reading
		FileBackedOutputFactoryTypeStream( FILE *fp , const Factory &factory ) : _fp(fp) , _factory(factory) , _buffer( NewPointer< char >( _factory.bufferSize() ) ) , _bufferSize( _factory.bufferSize() ) {}
		~FileBackedOutputFactoryTypeStream( void ){ DeletePointer( _buffer ); }
		size_t write( const Data &d )
		{
			_factory.toBuffer( d , _buffer );
			fwrite( _buffer , sizeof(unsigned char) , _bufferSize , _fp );
			return _sz++;
		}
		size_t size( void ) const { return _sz; }
	protected:
		size_t _sz = 0;
		FILE *_fp;
		const Factory _factory;
		Pointer( char ) _buffer;
		const size_t _bufferSize;
	};

	template< typename Factory >
	struct FileBackedOutputIndexedFactoryTypeStream : public OutputIndexedDataStream< typename Factory::VertexType >
	{
		using Data = typename Factory::VertexType;
		// It is assumed that the file pointer was open for binary reading
		FileBackedOutputIndexedFactoryTypeStream( FILE *fp , const Factory &factory ) : _fp(fp) , _factory(factory) , _buffer( NewPointer< char >( _factory.bufferSize() ) ) , _bufferSize( _factory.bufferSize() ) {}
		~FileBackedOutputIndexedFactoryTypeStream( void ){ DeletePointer( _buffer ); }
		size_t write( const size_t &idx , const Data &d )
		{
			fwrite( &idx , sizeof(size_t) , 1 , _fp );
			_factory.toBuffer( d , _buffer );
			fwrite( _buffer , sizeof(unsigned char) , _bufferSize , _fp );
			return _sz++;
		}
		size_t size( void ) const { return _sz; }
	protected:
		size_t _sz = 0;
		FILE *_fp;
		const Factory _factory;
		Pointer( char ) _buffer;
		const size_t _bufferSize;
	};

	///////////////////////////////////////////////////////////////////////////////////
	// File-backed data streams, with functionality for reading/writing from/to disk //
	///////////////////////////////////////////////////////////////////////////////////

	template< typename Factory >
	struct ASCIIInputDataStream : public InputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		ASCIIInputDataStream( const char* fileName , const Factory &factory );
		~ASCIIInputDataStream( void );
		void reset( void );
		bool read( Data &d );

	protected:
		const Factory _factory;
		FILE *_fp;
	};

	template< typename Factory >
	struct ASCIIOutputDataStream : public OutputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		ASCIIOutputDataStream( const char* fileName , const Factory &factory );
		~ASCIIOutputDataStream( void );
		size_t write( const Data &d );
		size_t size( void ) const { return _sz; }

	protected:
		size_t _sz = 0;
		const Factory _factory;
		FILE *_fp;
	};

	template< typename Factory >
	struct BinaryInputDataStream : public InputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		BinaryInputDataStream( const char* filename , const Factory &factory );
		~BinaryInputDataStream( void ){ fclose( _fp ) , _fp=NULL; }
		void reset( void );
		bool read( Data &d );

	protected:
		const Factory _factory;
		FILE* _fp;
	};

	template< typename Factory >
	struct BinaryOutputDataStream : public OutputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		BinaryOutputDataStream( const char* filename , const Factory &factory );
		~BinaryOutputDataStream( void ){ fclose( _fp ) , _fp=NULL; }
		void reset( void ){ fseek( _fp , 0 , SEEK_SET ); }
		size_t write( const Data &d );
		size_t size( void ) const { return _sz; }

	protected:
		size_t _sz = 0;
		const Factory _factory;
		FILE* _fp;
	};

	////////////////////////////////////////////
	// File-backed PLY-described data streams //
	////////////////////////////////////////////

	template< typename Factory >
	struct PLYInputDataStream : public InputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		PLYInputDataStream( const char* fileName , const Factory &factory );
		PLYInputDataStream( const char* fileName , const Factory &factory , size_t &count );
		~PLYInputDataStream( void );
		void reset( void );
		bool read( Data &d );

	protected:
		const Factory _factory;
		char* _fileName;
		PlyFile *_ply;
		std::vector< std::string > _elist;
		Pointer( char ) _buffer;

		size_t _pCount , _pIdx;
		void _free( void );
	};

	template< typename Factory >
	struct PLYOutputDataStream : public OutputDataStream< typename Factory::VertexType >
	{
		typedef typename Factory::VertexType Data;

		PLYOutputDataStream( const char* fileName , const Factory &factory , size_t count , int fileType=PLY_BINARY_NATIVE );
		~PLYOutputDataStream( void );
		size_t write( const Data &d );
		size_t size( void ) const { return _pIdx; }

	protected:
		const Factory _factory;
		PlyFile *_ply;
		size_t _pCount , _pIdx;
		Pointer( char ) _buffer;
	};

#include "DataStream.imp.inl"
}

#endif // DATA_STREAM_IMPLEMENTATION_INCLUDED
