/*
Copyright (c) 2011, Michael Kazhdan and Ming Chuang
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
#define FULL_ARRAY_DEBUG    0	// Note that this is not thread-safe

#include <stdio.h>
#include <emmintrin.h>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif // _WIN32
#include <stddef.h>

inline bool isfinitef( float fp ){ float f=fp; return ((*(unsigned *)&f)&0x7f800000)!=0x7f800000; }


template< class C >        bool IsValid( const C& c );
#if _DEBUG
template< >         inline bool IsValid< float >( const float& f ) { return isfinitef( f ) &&  ( f==0.f || abs(f)>1e-31f ); }
#else // !_DEBUG
template< >         inline bool IsValid< float >( const float& f ) { return isfinitef( f ); }
#endif // _DEBUG
template< >         inline bool IsValid< __m128 >( const __m128& m )
{
	const __m128* addr = &m;
	if( size_t(addr) & 15 ) return false;
	else                    return true;
}
template< class C > inline bool IsValid( const C& c ){ return true; }


#if FULL_ARRAY_DEBUG
class DebugMemoryInfo
{
public:
	const void* address;
	char name[512];
};
static std::vector< DebugMemoryInfo > memoryInfo;
#endif // FULL_ARRAY_DEBUG

template< class C >
class Array
{
	void _assertBounds( long long idx ) const
	{
		if( idx<min || idx>=max )
		{
			fprintf( stderr , "Array index out-of-bounds: %lld <= %lld < %lld\n" , min , idx , max );
			ASSERT( 0 );
			exit( 0 );
		}
	}
protected:
	C *data , *_data;
	long long min , max;
#if FULL_ARRAY_DEBUG
	static void _AddMemoryInfo( const void* ptr , const char* name )
	{
		size_t sz = memoryInfo.size();
		memoryInfo.resize( sz + 1 );
		memoryInfo[sz].address = ptr;
		if( name ) strcpy( memoryInfo[sz].name , name );
		else memoryInfo[sz].name[0] = 0;
	}
	static void _RemoveMemoryInfo( const void* ptr )
	{
		{
			size_t idx;
			for( idx=0 ; idx<memoryInfo.size( ) ; idx++ ) if( memoryInfo[idx].address==ptr ) break;
			if( idx==memoryInfo.size() )
			{
				fprintf( stderr , "Could not find memory in address table\n" );
				ASSERT( 0 );
			}
			else
			{
				memoryInfo[idx] = memoryInfo[memoryInfo.size()-1];
				memoryInfo.pop_back( );
			}
		}
	}
#endif // FULL_ARRAY_DEBUG

public:
	long long minimum( void ) const { return min; }
	long long maximum( void ) const { return max; }

	static inline Array New( size_t size , const char* name=NULL )
	{
		Array a;
		a._data = a.data = new C[size];
		a.min = 0;
#pragma message( "[WARNING] Casting unsigned to signed" )
		a.max = ( long long ) size;
#if FULL_ARRAY_DEBUG
		_AddMemoryInfo( a._data , name );
#endif // FULL_ARRAY_DEBUG
		return a;
	}
	static inline Array Alloc( size_t size , bool clear , const char* name=NULL )
	{
		Array a;
		a._data = a.data = ( C* ) malloc( size * sizeof( C ) );
		if( clear ) memset( a.data ,  0 , size * sizeof( C ) );
//		else        memset( a.data , -1 , size * sizeof( C ) );
		a.min = 0;
#pragma message( "[WARNING] Casting unsigned to signed" )
		a.max = ( long long ) size;
#if FULL_ARRAY_DEBUG
		_AddMemoryInfo( a._data , name );
#endif // FULL_ARRAY_DEBUG
		return a;
	}
	static inline Array AlignedAlloc( size_t size , size_t alignment , bool clear , const char* name=NULL )
	{
		Array a;
		a.data = ( C* ) aligned_malloc( sizeof(C) * size , alignment );
		a._data = ( C* )( ( ( void** )a.data )[-1] );
		if( clear ) memset( a.data ,  0 , size * sizeof( C ) );
//		else        memset( a.data , -1 , size * sizeof( C ) );
		a.min = 0;
#pragma message( "[WARNING] Casting unsigned to signed" )
		a.max = ( long long ) size;
#if FULL_ARRAY_DEBUG
		_AddMemoryInfo( a._data , name );
#endif // FULL_ARRAY_DEBUG
		return a;
	}
	static inline Array ReAlloc( Array& a , size_t size , bool clear , const char* name=NULL )
	{
		Array _a;
		_a._data = _a.data = ( C* ) realloc( a.data , size * sizeof( C ) );
		if( clear ) memset( _a.data ,  0 , size * sizeof( C ) );
#if FULL_ARRAY_DEBUG
		_RemoveMemoryInfo( a._data );
#endif // FULL_ARRAY_DEBUG
		a._data = NULL;
		_a.min = 0;
#pragma message( "[WARNING] Casting unsigned to signed" )
		_a.max = ( long long ) size;
#if FULL_ARRAY_DEBUG
		_AddMemoryInfo( _a._data , name );
#endif // FULL_ARRAY_DEBUG
		return _a;
	}

	Array( void )
	{
		data = _data = NULL;
		min = max = 0;
	}
	template< class D >
	Array( Array< D >& a )
	{
		_data = NULL;
		if( !a )
		{
			data =  NULL;
			min = max = 0;
		}
		else
		{
			// [WARNING] Chaning szC and szD to size_t causes some really strange behavior.
			long long szC = sizeof( C );
			long long szD = sizeof( D );
			data = (C*)a.data;
			min = ( a.minimum() * szD ) / szC;
			max = ( a.maximum() * szD ) / szC;
			if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD )
			{
				fprintf( stderr , "Could not convert array [ %lld , %lld ] * %lld => [ %lld , %lld ] * %lld\n" , a.minimum() , a.maximum() , szD , min , max , szC );
				ASSERT( 0 );
				exit( 0 );
			}
		}
	}
	static Array FromPointer( C* data , long long max )
	{
		Array a;
		a._data = NULL;
		a.data = data;
		a.min = 0;
		a.max = max;
		return a;
	}
	static Array FromPointer( C* data , long long min , long long max )
	{
		Array a;
		a._data = NULL;
		a.data = data;
		a.min = min;
		a.max = max;
		return a;
	}
	inline bool operator == ( const Array< C >& a ) const { return data==a.data; }
	inline bool operator != ( const Array< C >& a ) const { return data!=a.data; }
	inline bool operator == ( const C* c ) const { return data==c; }
	inline bool operator != ( const C* c ) const { return data!=c; }
	inline C* operator -> ( void )
	{
		_assertBounds( 0 );
		return data;
	}
	inline const C* operator -> ( ) const
	{
		_assertBounds( 0 );
		return data;
	}
	inline C& operator[]( long long idx )
	{
		_assertBounds( idx );
		return data[idx];
	}
	inline const C& operator[]( long long idx ) const
	{
		_assertBounds( idx );
		return data[idx];
	}
	inline Array operator + ( int idx ) const
	{
		Array a;
		a._data = _data;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline Array operator + ( long long idx ) const
	{
		Array a;
		a._data = _data;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline Array operator + ( unsigned int idx ) const
	{
		Array a;
		a._data = _data;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline Array operator + ( unsigned long long idx ) const
	{
		Array a;
		a._data = _data;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline Array& operator += ( int idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline Array& operator += ( long long idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline Array& operator += ( unsigned int idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline Array& operator += ( unsigned long long idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline Array& operator ++ ( void  ) { return (*this) += 1; }
	inline Array operator++( int ){ Array< C > temp = (*this) ; (*this) +=1 ; return temp; }
	Array  operator -  ( int idx ) const { return (*this) +  (-idx); }
	Array  operator -  ( long long idx ) const { return (*this) +  (-idx); }
	Array  operator -  ( unsigned int idx ) const { return (*this) +  (-idx); }
	Array  operator -  ( unsigned long long idx ) const { return (*this) +  (-idx); }
	Array& operator -= ( int idx )    { return (*this) += (-idx); }
	Array& operator -= ( long long idx )    { return (*this) += (-idx); }
	Array& operator -= ( unsigned int idx )    { return (*this) += (-idx); }
	Array& operator -= ( unsigned long long idx )    { return (*this) += (-idx); }
	Array& operator -- ( void ) { return (*this) -= 1; }
	inline Array operator--( int ){ Array< C > temp = (*this) ; (*this) -=1 ; return temp; }
	long long operator - ( const Array& a ) const { return ( long long )( data - a.data ); }

	void Free( void )
	{
		if( _data )
		{
			free( _data );
#if FULL_ARRAY_DEBUG
			_RemoveMemoryInfo( _data );
#endif // FULL_ARRAY_DEBUG
		}
		(*this) = Array( );
	}
	void Delete( void )
	{
		if( _data )
		{
			delete[] _data;
#if FULL_ARRAY_DEBUG
			_RemoveMemoryInfo( _data );
#endif // FULL_ARRAY_DEBUG
		}
		(*this) = Array( );
	}
	C* pointer( void ){ return data; }
	const C* pointer( void ) const { return data; }
	bool operator !( void ) const { return data==NULL; }
	operator bool( ) const { return data!=NULL; }
};

template< class C >
class ConstArray
{
	void _assertBounds( long long idx ) const
	{
		if( idx<min || idx>=max )
		{
			fprintf( stderr , "ConstArray index out-of-bounds: %lld <= %lld < %lld\n" , min , idx , max );
			ASSERT( 0 );
			exit( 0 );
		}
	}
protected:
	const C *data;
	long long min , max;
public:
	long long minimum( void ) const { return min; }
	long long maximum( void ) const { return max; }

	inline ConstArray( void )
	{
		data = NULL;
		min = max = 0;
	}
	inline ConstArray( const Array< C >& a )
	{
		// [WARNING] Changing szC and szD to size_t causes some really strange behavior.
		data = ( const C* )a.pointer( );
		min = a.minimum();
		max = a.maximum();
	}
	template< class D >
	inline ConstArray( const Array< D >& a )
	{
		// [WARNING] Changing szC and szD to size_t causes some really strange behavior.
		long long szC = ( long long ) sizeof( C );
		long long szD = ( long long ) sizeof( D );
		data = ( const C* )a.pointer( );
		min = ( a.minimum() * szD ) / szC;
		max = ( a.maximum() * szD ) / szC;
		if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD )
		{
//			fprintf( stderr , "Could not convert const array [ %lld , %lld ] * %lld => [ %lld , %lld ] * %lld\n" , a.minimum() , a.maximum() , szD , min , max , szC );
			fprintf( stderr , "Could not convert const array [ %lld , %lld ] * %lld => [ %lld , %lld ] * %lld\n %lld %lld %lld\n" , a.minimum() , a.maximum() , szD , min , max , szC , a.minimum() , a.minimum()*szD , (a.minimum()*szD)/szC );
			ASSERT( 0 );
			exit( 0 );
		}
	}
	template< class D >
	inline ConstArray( const ConstArray< D >& a )
	{
		// [WARNING] Chaning szC and szD to size_t causes some really strange behavior.
		long long szC = sizeof( C );
		long long szD = sizeof( D );
		data = ( const C*)a.pointer( );
		min = ( a.minimum() * szD ) / szC;
		max = ( a.maximum() * szD ) / szC;
		if( min*szC!=a.minimum()*szD || max*szC!=a.maximum()*szD )
		{
			fprintf( stderr , "Could not convert array [ %lld , %lld ] * %lld => [ %lld , %lld ] * %lld\n" , a.minimum() , a.maximum() , szD , min , max , szC );
			ASSERT( 0 );
			exit( 0 );
		}
	}
	static ConstArray FromPointer( const C* data , long long max )
	{
		ConstArray a;
		a.data = data;
		a.min = 0;
		a.max = max;
		return a;
	}
	static ConstArray FromPointer( const C* data , long long min , long long max )
	{
		ConstArray a;
		a.data = data;
		a.min = min;
		a.max = max;
		return a;
	}

	inline bool operator == ( const ConstArray< C >& a ) const { return data==a.data; }
	inline bool operator != ( const ConstArray< C >& a ) const { return data!=a.data; }
	inline bool operator == ( const C* c ) const { return data==c; }
	inline bool operator != ( const C* c ) const { return data!=c; }
	inline const C* operator -> ( void )
	{
		_assertBounds( 0 );
		return data;
	}
	inline const C& operator[]( long long idx ) const
	{
		_assertBounds( idx );
		return data[idx];
	}
	inline ConstArray operator + ( int idx ) const
	{
		ConstArray a;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline ConstArray operator + ( long long idx ) const
	{
		ConstArray a;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline ConstArray operator + ( unsigned int idx ) const
	{
		ConstArray a;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline ConstArray operator + ( unsigned long long idx ) const
	{
		ConstArray a;
		a.data = data+idx;
		a.min = min-idx;
		a.max = max-idx;
		return a;
	}
	inline ConstArray& operator += ( int idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline ConstArray& operator += ( long long idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline ConstArray& operator += ( unsigned int idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline ConstArray& operator += ( unsigned long long idx  )
	{
		min -= idx;
		max -= idx;
		data += idx;
		return (*this);
	}
	inline ConstArray& operator ++ ( void ) { return (*this) += 1; }
	inline ConstArray operator++( int ){ ConstArray< C > temp = (*this) ; (*this) +=1 ; return temp; }
	ConstArray  operator -  ( int idx ) const { return (*this) +  (-idx); }
	ConstArray  operator -  ( long long idx ) const { return (*this) +  (-idx); }
	ConstArray  operator -  ( unsigned int idx ) const { return (*this) +  (-idx); }
	ConstArray  operator -  ( unsigned long long idx ) const { return (*this) +  (-idx); }
	ConstArray& operator -= ( int idx )    { return (*this) += (-idx); }
	ConstArray& operator -= ( long long idx )    { return (*this) += (-idx); }
	ConstArray& operator -= ( unsigned int idx )    { return (*this) += (-idx); }
	ConstArray& operator -= ( unsigned long long idx )    { return (*this) += (-idx); }
	ConstArray& operator -- ( void ) { return (*this) -= 1; }
	inline ConstArray operator--( int ){ ConstArray< C > temp = (*this) ; (*this) -=1 ; return temp; }
	long long operator - ( const ConstArray& a ) const { return ( long long )( data - a.data ); }
	long long operator - ( const Array< C >& a ) const { return ( long long )( data - a.pointer() ); }

	const C* pointer( void ) const { return data; }
	bool operator !( void ) { return data==NULL; }
	operator bool( ) { return data!=NULL; }
};

#if FULL_ARRAY_DEBUG
inline void PrintMemoryInfo( void ){ for( size_t i=0 ; i<memoryInfo.size() ; i++ ) printf( "%d] %s\n" , i , memoryInfo[i].name ); }
#endif // FULL_ARRAY_DEBUG
template< class C >
Array< C > memcpy( Array< C > destination , const void* source , size_t size )
{
	if( size>destination.maximum()*sizeof(C) )
	{
		fprintf( stderr , "Size of copy exceeds destination maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( destination.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memcpy( &destination[0] , source , size );
	return destination;
}
template< class C , class D >
Array< C > memcpy( Array< C > destination , Array< D > source , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of copy exceeds destination maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( destination.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size>source.maximum()*sizeof( D ) )
	{
		fprintf( stderr , "Size of copy exceeds source maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( source.maximum()*sizeof( D ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memcpy( &destination[0] , &source[0] , size );
	return destination;
}
template< class C , class D >
Array< C > memcpy( Array< C > destination , ConstArray< D > source , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of copy exceeds destination maximum: %lld > %lld\n" , ( long long )( size ) , ( long  long )( destination.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size>source.maximum()*sizeof( D ) )
	{
		fprintf( stderr , "Size of copy exceeds source maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( source.maximum()*sizeof( D ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memcpy( &destination[0] , &source[0] , size );
	return destination;
}
template< class D >
void* memcpy( void* destination , Array< D > source , size_t size )
{
	if( size>source.maximum()*sizeof( D ) )
	{
		fprintf( stderr , "Size of copy exceeds source maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( source.maximum()*sizeof( D ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memcpy( destination , &source[0] , size );
	return destination;
}
template< class D >
void* memcpy( void* destination , ConstArray< D > source , size_t size )
{
	if( size>source.maximum()*sizeof( D ) )
	{
		fprintf( stderr , "Size of copy exceeds source maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( source.maximum()*sizeof( D ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memcpy( destination , &source[0] , size );
	return destination;
}
template< class C >
Array< C > memset( Array< C > destination , int value , size_t size )
{
	if( size>destination.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of set exceeds destination maximum: %lld > %lld\n" , ( long long )( size ) , ( long long )( destination.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( size ) memset( &destination[0] , value , size );
	return destination;
}

template< class C >
size_t fread( Array< C > destination , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>destination.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of read exceeds source maximum: %lld > %lld\n" , ( long long )( count*eSize ) , ( long long )( destination.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	return fread( &destination[0] , eSize , count , fp );
}
template< class C >
size_t fwrite( Array< C > source , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>source.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of write exceeds source maximum: %lld > %lld\n" , ( long long )( count*eSize ) , ( long long )( source.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	return fwrite( &source[0] , eSize , count , fp );
}
template< class C >
size_t fwrite( ConstArray< C > source , size_t eSize , size_t count , FILE* fp )
{
	if( count*eSize>source.maximum()*sizeof( C ) )
	{
		fprintf( stderr , "Size of write exceeds source maximum: %lld > %lld\n" , ( long long )( count*eSize ) , ( long long )( source.maximum()*sizeof( C ) ) );
		ASSERT( 0 );
		exit( 0 );
	}
	return fwrite( &source[0] , eSize , count , fp );
}
template< class C >
void qsort( Array< C > base , size_t numElements , size_t elementSize , int (*compareFunction)( const void* , const void* ) )
{
	if( sizeof(C)!=elementSize )
	{
		fprintf( stderr , "Element sizes differ: %lld != %lld\n" , ( long long )( sizeof(C) ) , ( long long )( elementSize ) );
		ASSERT( 0 );
		exit( 0 );
	}
	if( base.minimum()>0 || base.maximum()<numElements )
	{
		fprintf( stderr , "Array access out of bounds: %lld <= 0 <= %lld <= %lld\n" , base.minimum() , base.maximum() , ( long long )( numElements ) );
		ASSERT( 0 );
		exit( 0 );
	}
	qsort( base.pointer() , numElements , elementSize , compareFunction );
}
