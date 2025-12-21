/*
Copyright (c) 2019, Michael Kazhdan
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

#ifndef REGULAR_GRID_INCLUDED
#define REGULAR_GRID_INCLUDED

#include <type_traits>
#include "Array.h"
#include "Geometry.h"
#include "MyMiscellany.h"

namespace PoissonRecon
{

	template< typename ... Type > struct RegularGridDataType{};

	template<>
	struct RegularGridDataType<>
	{
		static inline void Write( FILE *fp , unsigned int dim , std::string name );
		static inline bool Read( FILE *fp , unsigned int dim , std::string name );
	};

	template<> struct RegularGridDataType< char >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } ; static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< char >::Name = "CHAR";
	template<> struct RegularGridDataType< unsigned char >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< unsigned char >::Name = "UNSIGNED_CHAR";
	template<> struct RegularGridDataType< int >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< int >::Name = "INT";
	template<> struct RegularGridDataType< unsigned int >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< unsigned int >::Name = "UNSIGNED_INT";
	template<> struct RegularGridDataType< float >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< float >::Name = "FLOAT";
	template<> struct RegularGridDataType< double >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , 1 , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , 1 , Name ); } };
	inline const std::string RegularGridDataType< double >::Name = "DOUBLE";

	template< typename Real , unsigned int Dim > struct RegularGridDataType< Point< Real , Dim > >{ static const std::string Name ; static void Write( FILE *fp ){ RegularGridDataType<>::Write( fp , Dim , Name ); } static bool Read( FILE *fp ){ return RegularGridDataType<>::Read( fp , Dim , Name ); } };
	template< typename Real , unsigned int Dim > const std::string RegularGridDataType< Point< Real , Dim > >::Name = RegularGridDataType< Real >::Name;


	template< typename DataType , unsigned int Dim >
	struct RegularGrid
	{
		static bool ReadHeader( std::string fileName , unsigned int &dim , std::string &type );

		RegularGrid( void ) : _values( NullPointer( DataType ) ){ for( unsigned int d=0 ; d<Dim ; d++ ) _res[d] = 0; }
		~RegularGrid( void ){ DeletePointer( _values ); }

		template< typename Real > static void Read( std::string fileName , unsigned int res[Dim] , Pointer( DataType ) &values , XForm< Real , Dim+1 > &gridToModel );
		template< typename Real > static void Write( std::string fileName , const unsigned int res[Dim] , ConstPointer( DataType ) values , XForm< Real , Dim+1 > gridToModel );

		template< typename Real > void read( std::string fileName , XForm< Real , Dim+1 > &gridToModel );
		template< typename Real > void write( std::string fileName , XForm< Real , Dim+1 > gridToModel ) const;

		Pointer( DataType ) operator()( void ){ return _values; }
		ConstPointer( DataType ) operator()( void ) const { return _values; }

		DataType &operator[]( size_t idx ){ return _values[idx]; }
		const DataType &operator[]( size_t idx ) const { return _values[idx]; }

		// Take the Dim coordinates and transform them into an index
		template< typename Int  >                    typename std::enable_if< std::is_integral< Int >::value , size_t >::type index( const Int coords[] ) const { return _index( coords , 0 ); }
		template< typename Int  >                    typename std::enable_if< std::is_integral< Int >::value , size_t >::type index(       Int coords[] ) const { return _index( coords , 0 ); }
		template< typename Int , typename ... Ints > typename std::enable_if< std::is_integral< Int >::value , size_t >::type index( Int coord , Ints ... coords ) const { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Int c[] = { coord , coords ... } ; return index( c ); }

		// Take the index and transform it into the Dim coordinate
		template< typename Int  >                    typename std::enable_if< std::is_integral< Int >::value >::type setIndex( size_t idx , Int coords[] ) const { _setIndex( idx , coords ); }

		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value >::type resize( const Int res[] );
		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value >::type resize(       Int res[] );
		template< typename Int , typename ... Ints > typename std::enable_if< std::is_integral< Int >::value >::type resize( Int res , Ints ...  ress );

		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value ,       DataType & >::type operator()( const Int coords[] )       { return operator[]( index( coords ) ); }
		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value ,       DataType & >::type operator()(       Int coords[] )       { return operator[]( index( coords ) ); }
		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value , const DataType & >::type operator()( const Int coords[] ) const { return operator[]( index( coords ) ); }
		template< typename Int >                     typename std::enable_if< std::is_integral< Int >::value , const DataType & >::type operator()(       Int coords[] ) const { return operator[]( index( coords ) ); }
		template< typename Int , typename ... Ints > typename std::enable_if< std::is_integral< Int >::value ,       DataType & >::type operator()( Int coord , Ints ... coords )       { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Int c[] = { coord , coords ... } ; return operator()( c ); }
		template< typename Int , typename ... Ints > typename std::enable_if< std::is_integral< Int >::value , const DataType & >::type operator()( Int coord , Ints ... coords ) const { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Int c[] = { coord , coords ... } ; return operator()( c ); }

		template< typename Real > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()(       Real coords[] )       { return _Sample( _res , coords , _values ); }
		template< typename Real > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()( const Real coords[] )       { return _Sample( _res , coords , _values ); }
		template< typename Real > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()(       Real coords[] ) const { return _Sample( _res , coords , _values ); }
		template< typename Real > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()( const Real coords[] ) const { return _Sample( _res , coords , _values ); }
		template< typename Real , typename ... Reals > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()( Real coord , Reals ...  coords  )       { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Real c[] = { coord , coords ... } ; return operator()( c ); }
		template< typename Real , typename ... Reals > typename std::enable_if< !std::is_integral< Real >::value , ProjectiveData< Real , DataType > >::type operator()( Real coord , Reals ...  coords  ) const { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Real c[] = { coord , coords ... } ; return operator()( c ); }
		template< typename Real > ProjectiveData< Real , DataType > operator()( Point< Real , Dim > coords )       { return operator()( &coords[0] ); }
		template< typename Real > ProjectiveData< Real , DataType > operator()( Point< Real , Dim > coords ) const { return operator()( &coords[0] ); }

		template< typename Int >                     typename std::enable_if< std::is_integral< int >::value , bool >::type inBounds( const Int coords[] ) const { return _inBounds< Dim-1 >( coords ); }
		template< typename Int >                     typename std::enable_if< std::is_integral< int >::value , bool >::type inBounds(       Int coords[] ) const { return _inBounds< Dim-1 >( coords ); }
		template< typename Int , typename ... Ints > typename std::enable_if< std::is_integral< int >::value , bool >::type inBounds( Ints ...  coords   ) const { static_assert( sizeof...(coords)+1==Dim , "[ERROR] number of coordinates does not match the number of dimensions" ) ; const Int c[] = { coords ... } ; return inBounds( c ); }

		const unsigned int *res( void ) const { return _res; }
		unsigned int res( unsigned int d ) const { return _res[d]; }
		size_t resolution( void ) const { return _Resolution< Dim >( _res ); }

	protected:
		template< unsigned int D=Dim > static typename std::enable_if< D==1 , size_t >::type _Resolution( const unsigned int res[] ) { return res[0]; }
		template< unsigned int D=Dim > static typename std::enable_if< D!=1 , size_t >::type _Resolution( const unsigned int res[] ) { return res[D-1] * _Resolution<D-1>(res); }

		template< typename Real , unsigned int D=Dim > static typename std::enable_if< D==1 , ProjectiveData< Real , DataType > >::type _Sample( const unsigned int res[] , const Real coords[] , ConstPointer( DataType ) values );
		template< typename Real , unsigned int D=Dim > static typename std::enable_if< D!=1 , ProjectiveData< Real , DataType > >::type _Sample( const unsigned int res[] , const Real coords[] , ConstPointer( DataType ) values );

		// _index( x[0] , ... , x[D-1] ) = x[0] + ... + x[D-1] * ( res[0] * ... * res[D-1] )
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==Dim-1 && D!=0 , size_t >::type _index(       Int coords[] , size_t idx ) const { return _index< Int , D-1 >( coords , coords[D]               ); }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=Dim-1 && D!=0 , size_t >::type _index(       Int coords[] , size_t idx ) const { return _index< Int , D-1 >( coords , coords[D] + idx*_res[D] ); }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=Dim-1 && D==0 , size_t >::type _index(       Int coords[] , size_t idx ) const { return                               coords[D] + idx*_res[D]  ; }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==Dim-1 && D==0 , size_t >::type _index(       Int coords[] , size_t idx ) const { return                               coords[D]                ; }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==Dim-1 && D!=0 , size_t >::type _index( const Int coords[] , size_t idx ) const { return _index< Int , D-1 >( coords , coords[D]               ); }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=Dim-1 && D!=0 , size_t >::type _index( const Int coords[] , size_t idx ) const { return _index< Int , D-1 >( coords , coords[D] + idx*_res[D] ); }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=Dim-1 && D==0 , size_t >::type _index( const Int coords[] , size_t idx ) const { return                               coords[D] + idx*_res[D]  ; }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==Dim-1 && D==0 , size_t >::type _index( const Int coords[] , size_t idx ) const { return                               coords[D]                ; }

		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=0 , size_t >::type _setIndex( size_t idx , Int coords[] ) const { idx = _setIndex< Int , D-1 >( idx , coords ) ; coords[D] = (Int)( idx % _res[D] ) ; return idx / _res[D]; }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==0 , size_t >::type _setIndex( size_t idx , Int coords[] ) const {                                                coords[D] = (Int)( idx % _res[D] ) ; return idx / _res[D]; }

		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==0 , bool >::type _inBounds(       Int coords[] ) const { return coords[0]>=0 && coords[0]<_res[0];                               }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=0 , bool >::type _inBounds(       Int coords[] ) const { return coords[D]>=0 && coords[D]<_res[D] && _inBounds< D-1 >( coords ); }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D==0 , bool >::type _inBounds( const Int coords[] ) const { return coords[0]>=0 && coords[0]<_res[0];                               }
		template< typename Int , unsigned int D=Dim-1 > typename std::enable_if< D!=0 , bool >::type _inBounds( const Int coords[] ) const { return coords[D]>=0 && coords[D]<_res[D] && _inBounds< D-1 >( coords ); }


		unsigned int _res[Dim];
		Pointer( DataType ) _values;
	};
#include "RegularGrid.inl"
}

#endif // REGULAR_GRID_INCLUDED
