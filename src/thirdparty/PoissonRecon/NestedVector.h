/*
Copyright (c) 2024, Michael Kazhdan
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


#ifndef NESTED_VECTOR_INCLUDED
#define NESTED_VECTOR_INCLUDED

#include <mutex>
#include "MyAtomic.h"
#include "MyMiscellany.h"

namespace PoissonRecon
{
	// This represents a vector that can only grow in size.
	// It has the property that once a reference to an element is returned, that reference remains valid until the vector is destroyed.
	template< typename T , unsigned int Depth , unsigned int LogSize=20 >
	struct NestedVector;

	// The base case, an array with a maximum of 1<<LogSize
	template< typename T , unsigned int LogSize >
	struct NestedVector< T , 0 , LogSize >
	{
		static const unsigned int Depth = 0;
		NestedVector( void ) : _size(0) { _data = new _DataType[ _Size ]; }

		~NestedVector( void ){ delete[] _data; }

		NestedVector( const NestedVector &nv ) : NestedVector()
		{
			_size = nv._size.load();
			for( size_t i=0 ; i<_size ; i++ ) operator[](i) = nv[i];
		}

		NestedVector &operator = ( const NestedVector & nv )
		{
			_size = nv._size.load();
			for( size_t i=0 ; i<_size ; i++ ) operator[](i) = nv[i];
			return *this;
		}

		NestedVector( NestedVector &&nv )
		{
			_data = nv._data;
			_size = nv._size.load();
			nv._size = 0;
			nv._data = nullptr;
		}

		NestedVector& operator = ( NestedVector &&nv )
		{
			size_t foo = _size;
			_size = nv._size.load();
			nv._size = foo;
			std::swap( _data , nv._data );
			return *this;
		}

		// This function is guaranteed to return a lower-bound on the actual size
		size_t size( void ) const { return _size; }

		const T& operator[]( size_t idx ) const { return _data[idx]; }
		T& operator[]( size_t idx ){ return _data[idx]; }

		size_t resize( size_t sz ){ return resize( sz , T{} ); }

		size_t resize( size_t sz , const T &defaultValue )
		{
			if( sz>_MaxSize ) MK_THROW( "Resize size exceeds max size, considering increasing nesting: " , sz , " > " , _MaxSize );

			// Quick check to see if anything needs doing
			if( sz<_size ) return size();

			// Otherwise lock it down and get to work
			std::lock_guard lock( _mutex );

			// Check if the size got changed
			if( sz>_size )
			{
				for( size_t i=_size ; i<sz ; i++ ) _data[i] = defaultValue;
				_size = sz;
			}
			return size();
		}

		void write( BinaryStream &stream ) const
		{
			size_t sz = _size;
			stream.write( sz );
			stream.write( GetPointer( _data , _Size ) , sz );
		}

		void read( BinaryStream &stream )
		{
			size_t sz;
			if( !stream.read( sz ) ) MK_THROW( "Failed to read _size" );
			resize( sz );
			if( !stream.read( GetPointer( _data , _Size ) , _size ) ) MK_THROW( "Failed to read _data" );
		}

		void write( BinaryStream &stream , const Serializer< T > &serializer ) const
		{
			const size_t serializedSize = serializer.size();

			size_t sz = _size;
			stream.write( sz );
			if( _size )
			{
				char *buffer = new char[ _size * serializedSize ];
				for( size_t i=0 ; i<_size ; i++ ) serializer.serialize( operator[]( i ) , buffer+i*serializedSize );
				stream.write( buffer , serializedSize*_size );
				delete[] buffer ;
			}
		}
		void read( BinaryStream &stream , const Serializer< T > &serializer )
		{
			const size_t serializedSize = serializer.size();

			size_t sz;
			if( !stream.read( sz ) ) MK_THROW( "Failed to read _size" );
			if( _size )
			{
				resize( sz );
				char *buffer = new char[ _size * serializedSize ];
				if( !stream.read( buffer , serializedSize*_size ) ) MK_THROW( "Failed tor read in data" );
				for( size_t i=0 ; i<_size ; i++ ) serializer.deserialize( buffer+i*serializedSize , operator[]( i ) );
				delete[] buffer;
			}
		}

	protected:
		template< typename _T , unsigned int _Depth , unsigned int _LogSize > friend struct NestedVector;

		using _DataType = T;
		static const size_t _MaxSize = ((size_t)1)<<LogSize;
		static const size_t _Size = ((size_t)1)<<LogSize;
		static const size_t _Mask = _MaxSize-1;

		std::mutex _mutex;
		std::atomic< size_t > _size;
		_DataType *_data;
	};

	// The derived case, an array a maximum of 1<<LogSize elements
	template< typename T , unsigned int Depth , unsigned int LogSize >
	struct NestedVector
	{
		NestedVector( void ) : _size(0)
		{
			_data = new _DataType*[ _Size ];
			for( size_t i=0 ; i<_Size ; i++ ) _data[i] = nullptr;
		}

		~NestedVector( void )
		{
			for( size_t i=0 ; i<_size ; i++ ) delete _data[i];
			delete[] _data;
		}

		NestedVector( const NestedVector &nv ) : NestedVector()
		{
			_size = nv._size.load();
			for( size_t i=0 ; i<_size ; i++ ) _data[i] = new _DataType( *nv._data[i] );
		}

		NestedVector &operator = ( const NestedVector &nv )
		{
			for( size_t i=0 ; i<_size ; i++ ){ delete _data[i] ; _data[i] = nullptr; }
			_size = nv._size.load();
			for( size_t i=0 ; i<_size ; i++ ) _data[i] = new _DataType( *nv._data[i] );
			return *this;
		}

		NestedVector( NestedVector &&nv )
		{
			_size = nv._size.load();
			_data = nv._data;
			nv._size = 0;
			nv._data = nullptr;
		}

		NestedVector& operator = ( NestedVector &&nv )
		{
			size_t foo = _size;
			_size = nv._size.load();
			nv._size = foo;
			std::swap( _data , nv._data );
			return *this;
		}

		// This function is guaranteed to return a lower-bound on the actual size
		size_t size( void ) const
		{
			if( !_size ) return 0;
			else return ( (_size-1)<<(Depth*LogSize) ) + _data[_size-1]->size();
		}

		const T& operator[]( size_t idx ) const { return ( *_data[ idx>>(LogSize*Depth) ] )[ idx & NestedVector< T , Depth-1 , LogSize >::_Mask ] ; }
		T& operator[]( size_t idx ){ return ( *( _data[ idx>>(LogSize*Depth) ] ) )[ idx & NestedVector< T , Depth-1 , LogSize >::_Mask ] ; }

		size_t resize( size_t sz ){ return resize( sz , T{} ); }

		size_t resize( size_t sz , const T &defaultValue )
		{
			if( sz>_MaxSize ) MK_THROW( "Resize size exceeds max size, considering increasing nesting: " , sz , " > " , _MaxSize );

			size_t _sz = (sz+NestedVector< T , Depth-1 , LogSize >::_Mask)>>(LogSize*Depth);

			// Quick check to see if anything needs doing
			if( !_sz || _sz<_size ) return size();

			// Otherwise lock it down and get to work
			std::lock_guard lock( _mutex );

			// Check if the size got changed
			if( _sz>_size )
			{
				// Complete the initialization for the last existing
				if( _size )
				{
					size_t i = _size-1;
					size_t __sz = _sz==(i+1) ? ( sz - NestedVector< T , Depth-1 , LogSize >::_MaxSize * i ) : ( NestedVector< T , Depth-1 , LogSize >::_MaxSize );
					_data[i]->resize( __sz , defaultValue );
				}
				for( size_t i=_size ; i<_sz ; i++ )
				{
					_data[i] = new _DataType();
					size_t __sz = _sz==(i+1) ? ( sz - NestedVector< T , Depth-1 , LogSize >::_MaxSize * i ) : ( NestedVector< T , Depth-1 , LogSize >::_MaxSize );
					_data[i]->resize( __sz , defaultValue );
				}
				_size = _sz;
			}
			else if( _sz==_size )
			{
				size_t i = _sz-1;
				size_t __sz = _sz==(i+1) ? ( sz - NestedVector< T , Depth-1 , LogSize >::_MaxSize * i ) : ( NestedVector< T , Depth-1 , LogSize >::_MaxSize );
				_data[i]->resize( __sz , defaultValue );
			}
			return size();
		}

		void write( BinaryStream &stream ) const
		{
			stream.write( size() );
			for( size_t i=0 ; i<_size ; i++ ) _data[i]->write( stream );
		}

		void read( BinaryStream &stream )
		{
			size_t sz;
			if( !stream.read( sz ) ) MK_THROW( "Failed to read _size" );
			resize( sz );
			for( size_t i=0 ; i<_size ; i++ ) _data[i]->read(stream);
		}

		void write( BinaryStream &stream , const Serializer< T > &serializer ) const
		{
			const size_t serializedSize = serializer.size();

			stream.write( size() );
			for( size_t i=0 ; i<_size ; i++ ) _data[i]->write( stream , serializer );
		}
		void read( BinaryStream &stream , const Serializer< T > &serializer )
		{
			const size_t serializedSize = serializer.size();

			size_t sz;
			if( !stream.read( sz ) ) MK_THROW( "Failed to read _size" );
			resize( sz );
			for( size_t i=0 ; i<_size ; i++ ) _data[i]->read( stream , serializer );
		}

	protected:
		template< typename _T , unsigned int _Depth , unsigned int _LogSize > friend struct NestedVector;

		using _DataType = NestedVector< T , Depth-1 , LogSize >;
		static const size_t _MaxSize = ((size_t)1)<<(LogSize*(Depth+1));
		static const size_t _Size = ((size_t)1)<<LogSize;
		static const size_t _Mask = _MaxSize-1;

		std::mutex _mutex;
		std::atomic< size_t > _size;
		_DataType **_data;
	};
}
#endif // NESTED_VECTOR_INCLUDED
