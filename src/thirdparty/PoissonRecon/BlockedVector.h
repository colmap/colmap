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


#ifndef BLOCKED_VECTOR_INCLUDED
#define BLOCKED_VECTOR_INCLUDED

#include "MyMiscellany.h"

namespace PoissonRecon
{
	// This represents a vector that can only grow in size.
	// It has the property that once a reference to an element is returned, that reference remains valid until the vector is destroyed.
	template< typename T , unsigned int LogBlockSize=10 , unsigned int InitialBlocks=10 , unsigned int AllocationMultiplier=2 >
	struct BlockedVector
	{
		BlockedVector( size_t sz=0 , T defaultValue=T() ) : _defaultValue( defaultValue )
		{
			_reservedBlocks = InitialBlocks;
			_blocks = NewPointer< Pointer( T ) >( _reservedBlocks );
			for( size_t i=0 ; i<_reservedBlocks ; i++ ) _blocks[i] = NullPointer( T );
			_allocatedBlocks = _size = 0;
			if( sz ) resize( sz );
		}

		~BlockedVector( void )
		{
			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) DeletePointer( _blocks[i] );
			DeletePointer( _blocks );
		}

		BlockedVector( const BlockedVector& v )
		{
#ifdef SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size.load() , _defaultValue = v._defaultValue;
#else // !SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size , _defaultValue = v._defaultValue;
#endif // SANITIZED_PR
			_blocks = NewPointer< Pointer( T ) >( _reservedBlocks );
			for( size_t i=0 ; i<_allocatedBlocks ; i++ )
			{
				_blocks[i] = NewPointer< T >( _BlockSize );
				memcpy( _blocks[i] , v._blocks[i] , sizeof(T)*_BlockSize );
			}
			for( size_t i=_allocatedBlocks ; i<_reservedBlocks ; i++ ) _blocks[i] = NullPointer( Pointer ( T ) );
		}

		BlockedVector& operator = ( const BlockedVector&  v )
		{
			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) DeletePointer( _blocks[i] );
			DeletePointer( _blocks );
#ifdef SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _blocks = v._blocks.load() , _allocatedBlocks = v._allocatedBlocks , _size = v._size.load() , _defaultValue = v._defaultValue;
#else // !SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _blocks = v._blocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size , _defaultValue = v._defaultValue;
#endif // SANITIZED_PR
			_blocks = NewPointer< Pointer( T ) >( _reservedBlocks );
			for( size_t i=0 ; i<_allocatedBlocks ; i++ )
			{
				_blocks[i] = NewPointer< T >( _BlockSize );
				memcpy( _blocks[i] , v._blocks[i] , sizeof(T)*_BlockSize );
			}
			for( size_t i=_allocatedBlocks ; i<_reservedBlocks ; i++ ) _blocks[i] = NullPointer( T );
			return *this;
		}

		BlockedVector( BlockedVector&& v )
		{
#ifdef SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size.load() , _defaultValue = v._defaultValue , _blocks = v._blocks.load();
#else // !SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size , _defaultValue = v._defaultValue , _blocks = v._blocks;
#endif // SANITIZED_PR
			v._reservedBlocks = v._allocatedBlocks = v._size = 0 , v._blocks = NullPointer( Pointer( T ) );
		}

		BlockedVector& operator = ( BlockedVector&& v )
		{
			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) DeletePointer( _blocks[i] );
			DeletePointer( _blocks );
#ifdef SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size.load() , _defaultValue = v._defaultValue , _blocks = v._blocks.load();
#else // !SANITIZED_PR
			_reservedBlocks = v._reservedBlocks , _allocatedBlocks = v._allocatedBlocks , _size = v._size , _defaultValue = v._defaultValue , _blocks = v._blocks;
#endif // SANITIZED_PR
			v._reservedBlocks = v._allocatedBlocks = v._size = 0 , v._blocks = NullPointer( Pointer( T ) );
			return *this;
		}

		size_t size( void ) const { return _size; }

		const T& operator[]( size_t idx ) const { return _blocks[idx>>LogBlockSize][idx&_Mask]; }
		T& operator[]( size_t idx ){ return _blocks[idx>>LogBlockSize][idx&_Mask]; }

		void resize( size_t size ){ resize( size , _defaultValue ); }
		void resize( size_t size , const T& defaultValue )
		{
			reserve( size , defaultValue );
			_size = size;
		}

		void clear( void ){ _size = 0; }

		size_t reserved( void ) const { return _allocatedBlocks * _BlockSize; }

		void reserve( size_t size ){ reserve( size , _defaultValue ); }
		void reserve( size_t size , const T& defaultValue )
		{
			if( size<=_allocatedBlocks * _BlockSize ) return;
			size_t index = size-1;
			size_t block = index >> LogBlockSize;
			size_t blockIndex = index & _Mask;

			// If there are insufficiently many blocks
			if( block>=_reservedBlocks )
			{
				size_t newReservedSize = std::max< size_t >( _reservedBlocks * AllocationMultiplier , block+1 );
				Pointer( Pointer( T ) ) __blocks = NewPointer< Pointer( T ) >( newReservedSize );
				memcpy( __blocks , _blocks , sizeof( Pointer( T ) ) * _reservedBlocks );
				for( size_t i=_reservedBlocks ; i<newReservedSize ; i++ ) __blocks[i] = NullPointer( T );
				Pointer( Pointer( T ) ) _oldBlocks = _blocks;
				_blocks = __blocks;
				_reservedBlocks = newReservedSize;
				DeletePointer( _oldBlocks );
			}

			// If the block hasn't been allocated
			if( block>=_allocatedBlocks )
			{
				for( size_t b=_allocatedBlocks ; b<=block ; b++ )
				{
					_blocks[b] = NewPointer< T >( _BlockSize );
					for( size_t i=0 ; i<_BlockSize ; i++ ) _blocks[b][i] = defaultValue;
				}
				_allocatedBlocks = block+1;
			}
		}
		void push_back( const T &value )
		{
			resize( _size+1 );
			operator[]( _size-1 ) = value;
		}
		T &back( void ){ return operator[]( _size-1 ); }
		const T &back( void ) const { return operator[]( _size-1 ); }
		void pop_back( void ){ _size--; }

		void write( BinaryStream &stream ) const
		{
			stream.write( _size );
			stream.write( _defaultValue );
			stream.write( _reservedBlocks );
			stream.write( _allocatedBlocks );
			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) stream.write( _blocks[i] , _BlockSize );
		}

		void read( BinaryStream &stream )
		{
			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) DeletePointer( _blocks[i] );
			DeletePointer( _blocks );
			if( !stream.read( _size ) ) MK_THROW( "Failed to read _size" );
			if( !stream.read( _defaultValue ) ) MK_THROW( "Failed to read _defaultValue" );
			if( !stream.read( _reservedBlocks ) ) MK_THROW( "Failed to read _reservedBlocks" );
			if( !stream.read( _allocatedBlocks ) ) MK_THROW( "Failed to read _allocatedBlocks" );

			_blocks = NewPointer< Pointer( T ) >( _reservedBlocks );
			if( !_blocks ) MK_THROW( "Failed to allocate _blocks: " , _reservedBlocks );

			for( size_t i=0 ; i<_allocatedBlocks ; i++ )
			{
				_blocks[i] = NewPointer< T >( _BlockSize );
				if( !_blocks[i] ) MK_THROW( "Failed to allocate _blocks[" , i , "]" );
				if( !stream.read( _blocks[i] , _BlockSize ) ) MK_THROW( "Failed to read _blocks[" , i , "]" );
			}
			for( size_t i=_allocatedBlocks ; i<_reservedBlocks ; i++ ) _blocks[i] = NullPointer( T );
		}

		void read( BinaryStream &stream , const Serializer< T > &serializer )
		{
			const size_t serializedSize = serializer.size();

			for( size_t i=0 ; i<_allocatedBlocks ; i++ ) DeletePointer( _blocks[i] );
			DeletePointer( _blocks );
			_size = _allocatedBlocks = _reservedBlocks = 0;

			if( !stream.read( _size ) ) MK_THROW( "Failed to read _size" );
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Should deserialize default value" )
#endif // SHOW_WARNINGS
			if( !stream.read( _defaultValue ) ) MK_THROW( "Failed to read _defaultValue" );
			if( !stream.read( _reservedBlocks ) ) MK_THROW( "Failed to read _reservedBlocks" );
			if( !stream.read( _allocatedBlocks ) ) MK_THROW( "Failed to read _allocatedBlocks" );
			_blocks = NewPointer< Pointer( T ) >( _reservedBlocks );
			if( !_blocks ) MK_THROW( "Failed to allocate _blocks: " , _reservedBlocks );
			for( size_t i=0 ; i<_allocatedBlocks ; i++ )
			{
				_blocks[i] = NewPointer< T >( _BlockSize );
				if( !_blocks[i] ) MK_THROW( "Failed to allocate _blocks[" , i , "]" );
			}
			if( _size )
			{
				Pointer( char ) buffer = NewPointer< char >( _size * serializedSize );
				if( !stream.read( buffer , serializedSize*_size ) ) MK_THROW( "Failed tor read in data" );
				for( unsigned int i=0 ; i<_size ; i++ ) serializer.deserialize( buffer+i*serializedSize , operator[]( i ) );
				DeletePointer( buffer );
			}
		}

		void write( BinaryStream &stream , const Serializer< T > &serializer ) const
		{
			const size_t serializedSize = serializer.size();

			stream.write( _size );
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Should serialize default value" )
#endif // SHOW_WARNINGS
			stream.write( _defaultValue );
			stream.write( _reservedBlocks );
			stream.write( _allocatedBlocks );
			if( _size )
			{
				Pointer( char ) buffer = NewPointer< char >( _size * serializedSize );
				for( unsigned int i=0 ; i<_size ; i++ ) serializer.serialize( operator[]( i ) , buffer+i*serializedSize );
				stream.write( buffer , serializedSize*_size );
				DeletePointer( buffer );
			}
		}


	protected:
		static const size_t _BlockSize = 1<<LogBlockSize;
		static const size_t _Mask = (1<<LogBlockSize)-1;

		T _defaultValue;
		size_t _allocatedBlocks , _reservedBlocks;
#ifdef SANITIZED_PR
		std::atomic< size_t > _size;
		std::atomic< Pointer( Pointer( T ) ) > _blocks;
#else // !SANITIZED_PR
		size_t _size;
		Pointer( Pointer( T ) ) _blocks;
#endif // SANITIZED_PR
	};
}
#endif // BLOCKED_VECTOR_INCLUDED
