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

#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

#include <stdio.h>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <unordered_map>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#endif // _WIN32
#include "MyMiscellany.h"
#include "Array.h"

namespace PoissonRecon
{

	// An empty type
	template< typename Real >
	struct EmptyVectorType
	{
		EmptyVectorType& operator += ( const EmptyVectorType& p ){ return *this; }
		EmptyVectorType& operator -= ( const EmptyVectorType& p ){ return *this; }
		EmptyVectorType& operator *= ( Real s )                  { return *this; }
		EmptyVectorType& operator /= ( Real s )                  { return *this; }
		EmptyVectorType  operator +  ( const EmptyVectorType& p ) const { EmptyVectorType _p = *this ; _p += p ; return _p; }
		EmptyVectorType  operator -  ( const EmptyVectorType& p ) const { EmptyVectorType _p = *this ; _p -= p ; return _p; }
		EmptyVectorType  operator *  ( Real s )                   const { EmptyVectorType _p = *this ; _p *= s ; return _p; }
		EmptyVectorType  operator /  ( Real s )                   const { EmptyVectorType _p = *this ; _p /= s ; return _p; }

		friend std::ostream &operator << ( std::ostream &os , const EmptyVectorType &v ){ return os; }
	};
	template< typename Real > EmptyVectorType< Real > operator * ( Real s , EmptyVectorType< Real > v ){ return v*s; }

	template< typename _Real , typename ... VectorTypes > struct DirectSum;

	template< typename _Real , typename FirstType , typename ... RestTypes >
	struct DirectSum< _Real , FirstType , RestTypes... >
	{
		friend Atomic< DirectSum< _Real , FirstType , RestTypes... > >;

		using Real = _Real;

		template< unsigned int I > using VectorType = std::tuple_element_t< I , std::tuple< FirstType , RestTypes ... > >;

		template< unsigned int I > VectorType< I >& get( void )
		{
			if constexpr( I==0 ) return _first;
			else return _rest.template get< I-1 >();
		}
		template< unsigned int I > const VectorType< I >& get( void ) const
		{
			if constexpr( I==0 ) return _first;
			else return _rest.template get< I-1 >();
		}

		DirectSum& operator += ( const DirectSum& p ){ _first += p._first ; _rest += p._rest ; return *this; }
		DirectSum& operator -= ( const DirectSum& p ){ _first -= p._first ; _rest -= p._rest ; return *this; }
		DirectSum& operator *= ( Real s )            { _first *= s ; _rest *= s ; return *this; }
		DirectSum& operator /= ( Real s )            { _first /= s ; _rest /= s ; return *this; }
		DirectSum  operator +  ( const DirectSum& p ) const { DirectSum _p = *this ; _p += p ; return _p; }
		DirectSum  operator -  ( const DirectSum& p ) const { DirectSum _p = *this ; _p -= p ; return _p; }
		DirectSum  operator *  ( Real s )             const { DirectSum _p = *this ; _p *= s ; return _p; }
		DirectSum  operator /  ( Real s )             const { DirectSum _p = *this ; _p /= s ; return _p; }

		DirectSum( void ){}

		template< typename _FirstType , typename ... _RestTypes >
		DirectSum( const _FirstType &first , const _RestTypes & ... rest ) : _first(first) , _rest(rest...){}

		template< typename ComponentFunctor /*=std::function< void (VectorTypes&...)>*/ >
		void process( ComponentFunctor f ){ _process( f ); }

		template< typename ComponentFunctor /*=std::function< void (const VectorTypes&...)>*/ >
		void process( ComponentFunctor f ) const { _process( f ); }

		friend std::ostream &operator << ( std::ostream &os , const DirectSum &v )
		{
			os << "{ ";
			v._write( os );
			os << " }";
			return os;
		}

	protected:
		FirstType _first;
		DirectSum< Real , RestTypes... > _rest;

		void _write( std::ostream &os ) const
		{
			os << _first;
			if constexpr( sizeof...(RestTypes) )
			{
				os << " , ";
				_rest._write( os );
			}
		}

		template< typename ComponentFunctor /*=std::function< void ( FirstType& , RestTypes&... )>*/ , typename ... Components >
		void _process( ComponentFunctor &f , Components ... c )
		{
			if constexpr( sizeof...(Components)==sizeof...(RestTypes)+1 ) f( c... );
			else _process( f , c... , this->template get< sizeof...(Components) >() );
		}
		template< typename ComponentFunctor /*=std::function< void ( const FirstType& , const RestTypes&... )>*/ , typename ... Components >
		void _process( ComponentFunctor &f , Components ... c ) const
		{
			if constexpr( sizeof...(Components)==sizeof...(RestTypes)+1 ) f( c... );
			else _process( f , c... , this->template get< sizeof...(Components) >() );
		}
	};

	template< typename _Real >
	struct DirectSum< _Real >
	{
		using Real = _Real;

		DirectSum& operator += ( const DirectSum& p ){ return *this; }
		DirectSum& operator -= ( const DirectSum& p ){ return *this; }
		DirectSum& operator *= ( Real s )            { return *this; }
		DirectSum& operator /= ( Real s )            { return *this; }
		DirectSum  operator +  ( const DirectSum& p ) const { return DirectSum(); }
		DirectSum  operator -  ( const DirectSum& p ) const { return DirectSum(); }
		DirectSum  operator *  ( Real s )             const { return DirectSum(); }
		DirectSum  operator /  ( Real s )             const { return DirectSum(); }

		DirectSum( void ){}

		template< typename ComponentFunctor /*=std::function< void (VectorTypes&...)>*/ >
		void process( ComponentFunctor f ){ f(); }

		template< typename ComponentFunctor /*=std::function< void (const VectorTypes&...)>*/ >
		void process( ComponentFunctor f ) const { f(); }

		friend std::ostream &operator << ( std::ostream &os , const DirectSum &v ){ return os << "{ }"; }
	};

	template< typename Real , typename ... Vectors >
	DirectSum< Real , Vectors ... > operator * ( Real s , DirectSum< Real , Vectors ... > vu ){ return vu * s; }

	template< class Real > Real Random( void );

	template< class Real , unsigned int Dim > struct XForm;


	template< typename Real , unsigned int ... Dims > struct Point;

	template< class Real , unsigned int Dim >
	struct Point< Real , Dim >
	{
		void _init( unsigned int d )
		{
			if( !d ) memset( coords , 0 , sizeof(Real)*Dim );
			else MK_THROW( "Should never be called" );
		}
		template< class _Real , class ... _Reals > void _init( unsigned int d , _Real v , _Reals ... values )
		{
			coords[d] = (Real)v;
			if( d+1<Dim ) _init( d+1 , values... );
		}
		template< class ... Points >
		static void _AddColumnVector( XForm< Real , Dim >& x , unsigned int c , Point point , Points ... points )
		{
			for( unsigned int r=0 ; r<Dim ; r++ ) x( c , r ) = point[r];
			_AddColumnVector( x , c+1 , points ... );
		}
		static void _AddColumnVector( XForm< Real , Dim >& x , unsigned int c ){ ; }
	public:
		Real coords[Dim];
		Point( void ) { memset( coords , 0 , sizeof(Real)*Dim ); }
		Point( const Point& p ){ memcpy( coords , p.coords , sizeof(Real)*Dim ); }
		template< class ... _Reals > Point( _Reals ... values ){ static_assert( sizeof...(values)==Dim || sizeof...(values)==0 , "[ERROR] Point::Point: Invalid number of coefficients" ) ; _init( 0 , values... ); }
		template< class _Real > Point( const Point< _Real , Dim >& p ){ for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] = (Real) p.coords[d]; }
		Point( Real *values ){ for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] = values[d]; }
		Point( const Real *values ){ for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] = values[d]; }
		inline       Real& operator[] ( unsigned int i )       { return coords[i]; }
		inline const Real& operator[] ( unsigned int i ) const { return coords[i]; }
		inline Point  operator - ( void ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = - coords[d] ; return q; }

		template< class _Real > inline Point& operator += ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] += (Real)p.coords[d] ; return *this; }
		template< class _Real > inline Point  operator +  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] + (Real)p.coords[d] ; return q; }
		template< class _Real > inline Point& operator -= ( Point< _Real , Dim > p )       { return (*this)+=(-p); }
		template< class _Real > inline Point  operator -  ( Point< _Real , Dim > p ) const { return (*this)+ (-p); }
		template< class Scalar > inline Point& operator *= ( Scalar r )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] *= r ; return *this; }
		template< class Scalar > inline Point  operator *  ( Scalar r ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] * r ; return q; }
		template< class Scalar > inline Point& operator /= ( Scalar r )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] /= r ; return *this; }
		template< class Scalar > inline Point  operator /  ( Scalar r ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] / r ; return q; }
		template< class _Real > inline Point& operator *= ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] *= p.coords[d] ; return *this; }
		template< class _Real > inline Point  operator *  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] * p.coords[d] ; return q; }
		template< class _Real > inline Point& operator /= ( Point< _Real , Dim > p )       { for( unsigned int d=0 ; d<Dim ; d++ ) coords[d] /= p.coords[d] ; return *this; }
		template< class _Real > inline Point  operator /  ( Point< _Real , Dim > p ) const { Point q ; for( unsigned int d=0 ; d<Dim ; d++ ) q.coords[d] = coords[d] / p.coords[d] ; return q; }

		static Real Dot( const Point& p1 , const Point& p2 ){ Real dot = {} ; for( unsigned int d=0 ; d<Dim ; d++ ) dot += p1.coords[d] * p2.coords[d] ; return dot; }
		static Real SquareNorm( const Point& p ){ return Dot( p , p ); }
		template< class ... Points > static Point CrossProduct( Points ... points )
		{
			static_assert( sizeof ... ( points )==Dim-1 , "Number of points in cross-product must be one less than the dimension" );
			XForm< Real , Dim > x;
			_AddColumnVector( x , 0 , points ... );
			Point p;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( d&1 ) ? -x.subDeterminant( Dim-1 , d ) : x.subDeterminant( Dim-1 , d );
			return p;
		}
		static Point CrossProduct( const Point* points )
		{
			XForm< Real , Dim > x;
			for( unsigned int d=0 ; d<Dim-1 ; d++ ) for( unsigned int c=0 ; c<Dim ; c++ ) x(d,c) = points[d][c];
			Point p;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( d&1 ) ? -x.subDeterminant( Dim-1 , d ) : x.subDeterminant( Dim-1 , d );
			return p;
		}
		static Point CrossProduct( Point* points ){ return CrossProduct( ( const Point* )points ); }

		template< class _Real , unsigned int _Dim >
		friend std::ostream &operator << ( std::ostream &os , const Point< _Real , _Dim > &p );
	};
	template< class Real , unsigned int Dim > Point< Real , Dim > operator * ( Real r , Point< Real , Dim > p ){ return p*r; }
	template< class Real , unsigned int Dim > Point< Real , Dim > operator / ( Real r , Point< Real , Dim > p ){ return p/r; }
	template< class Real , unsigned int Dim >
	std::ostream &operator << ( std::ostream &os , const Point< Real , Dim > &p )
	{
		os << "( ";
		for( int d=0 ; d<Dim ; d++ )
		{
			if( d ) os << " , ";
			os << p[d];
		}
		return os << " )";
	}

	// This class represents a point whose is size is allocated dynamically.
	// The size can be set by:
	// 1. Construction
	// 2. Copying
	// 3. Adding / subtracting (if the size has not been set yet)
	template< class Real >
	struct Point< Real >
	{
		friend struct Atomic< Point< Real > >;

		Point( void ) : _coords( NullPointer(Real) ) , _dim(0){}
		Point( size_t dim ) : _coords( NullPointer(Real) ) , _dim(0) { if( dim ){ _resize( (unsigned int)dim ) ; memset( _coords , 0 , sizeof(Real)*_dim ); } }
		Point( const Point &p ) : _coords( NullPointer(Real) ) , _dim(0) { if( p._dim ){ _resize( p._dim ) ; memcpy( _coords , p._coords , sizeof(Real)*_dim ); } }
		~Point( void ){ DeletePointer( _coords ); }

		Point &operator = ( const Point &p )
		{
			if( !_dim || !p._dim ){ _resize( p._dim ) ; memcpy( _coords , p._coords , sizeof(Real)*_dim ); }
			else if( _dim==p._dim ) memcpy( _coords , p._coords , sizeof(Real)*_dim );
			else MK_THROW( "Dimensions don't match: " , _dim , " != " , p._dim );
			return *this;
		}

		unsigned int dim( void ) const { return _dim; }
		Real &operator[]( size_t idx ){ return _coords[idx]; }
		const Real &operator[]( size_t idx ) const { return _coords[idx]; }


		Point& operator += ( const Point& p )
		{
			if( !_dim ){ _resize( p._dim ) ; for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] = p._coords[i]; }
			else if( _dim==p._dim ) for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] += p._coords[i];
			else MK_THROW( "Dimensions don't match: " , _dim , " != " , p._dim );
			return *this;
		}
		Point& operator -= ( const Point& p )
		{
			if( !_dim ){ _resize( p._dim ) ; for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] = -p._coords[i]; }
			else if( _dim==p._dim ) for( unsigned int i=0 ; i<_dim ; i++ ) _coords[i] -= p._coords[i];
			else MK_THROW( "Dimensions don't match: " , _dim , " != " , p._dim );
			return *this;
		}
		Point& operator *= ( Real s )
		{
			for( unsigned int i=0 ; i<_dim ; i++ ) (*this)[i] *=  s;
			return *this;
		}
		Point& operator /= ( Real s )
		{
			for( unsigned int i=0 ; i<_dim ; i++ ) (*this)[i] /=  s;
			return *this;
		}
		Point operator +  ( const Point& p ) const { Point _p = *this ; _p += p ; return _p; }
		Point operator -  ( const Point& p ) const { Point _p = *this ; _p -= p ; return _p; }
		Point operator *  ( Real s )         const { Point _p = *this ; _p *= s ; return _p; }
		Point operator /  ( Real s )         const { Point _p = *this ; _p /= s ; return _p; }

		static Real Dot( const Point &p1 , const Point &p2 )
		{
			Real dot;
			if( p1._dim!=p2._dim ) MK_THROW( "Dimensions differ: " , p1._dim , " != " , p2._dim );
			for( size_t d=0 ; d<p1._dim ; d++ ) dot += p1[d] * p2[d];
			return dot;
		}
		static Real SquareNorm( const Point& p ){ return Dot( p , p ); }

		friend std::ostream &operator << ( std::ostream &os , const Point &p )
		{
			os << "( ";
			for( size_t i=0 ; i<p._dim ; i++ )
			{
				if( i ) os << " , ";
				os << p[i];
			}
			return os << " )";
		}
	protected:
		Pointer( Real ) _coords;
		unsigned int _dim;
		void _resize( unsigned int dim )
		{
			DeletePointer( _coords );
			if( dim ) _coords = NewPointer< Real >( dim );
			_dim = dim;
		}
	};
	template< class Real > Point< Real > operator * ( Real s , Point< Real > p ){ return p*s; }

	/** This templated class represents a Ray.*/
	template< class Real , unsigned int Dim >
	class Ray
	{
	public:
		/** The starting point of the ray */
		Point< Real , Dim > position;

		/** The direction of the ray */
		Point< Real , Dim > direction;

		/** The default constructor */
		Ray( void ){}

		/** The constructor settign the the position and direction of the ray */
		Ray( const Point< Real , Dim > &p , const Point< Real , Dim > &d ) : position(p) , direction(d){}

		/** This method computes the translation of the ray by p and returns the translated ray.*/
		Ray  operator +  ( const Point< Real , Dim > &p ) const { return Ray( position+p , direction ); }

		/** This method translates the current ray by p.*/
		Ray &operator += ( const Point< Real , Dim > &p ){ position +=p ; return *this; }

		/** This method computes the translation of the ray by -p and returns the translated ray.*/
		Ray  operator -  ( const Point< Real , Dim > &p ) const { return Ray( position-p , direction ); }

		/** This method translates the current ray by -p.*/
		Ray &operator -= ( const Point< Real , Dim > &p ){ position -= p ; return *this; }

		/** This method returns the point at a distance of t along the ray. */
		Point< Real , Dim > operator() ( double t ) const { return position + direction * (Real)t; }
	};

	/** This function prints out the ray.*/
	template< class Real , unsigned int Dim >
	std::ostream &operator << ( std::ostream &stream , const Ray< Real , Dim > &ray )
	{
		stream << "[ " << ray.position << " ] [ " << ray.direction << " ]";
		return stream;
	}

	template< class Real , unsigned int _Columns , unsigned int _Rows >
	struct Matrix
	{
		static const unsigned int Columns = _Columns;
		static const unsigned int Rows = _Rows;
		Real coords[Columns][Rows];
		Matrix( void ) { memset( coords , 0 , sizeof(coords) ); }
		inline       Real& operator() ( unsigned int c , unsigned int r )       { return coords[c][r]; }
		inline const Real& operator() ( unsigned int c , unsigned int r ) const { return coords[c][r]; }
		inline       Real* operator[] ( unsigned int c                  )       { return coords[c]   ; }
		inline const Real* operator[] ( unsigned int c                  ) const { return coords[c]   ; }

		inline Matrix  operator - ( void ) const { Matrix m ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) m.coords[c][r] = - coords[c][r] ; return m; }

		inline Matrix& operator += ( const Matrix& m ){ for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) coords[c][r] += m.coords[c][r] ; return *this; }
		inline Matrix  operator +  ( const Matrix& m ) const { Matrix n ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) n.coords[c][r] = coords[c][r] + m.coords[c][r] ; return n; }
		inline Matrix& operator *= ( Real s ) { for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) coords[c][r] *= s ; return *this; }
		inline Matrix  operator *  ( Real s ) const { Matrix n ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) n.coords[c][r] = coords[c][r] * s ; return n; }

		inline Matrix& operator -= ( const Matrix& m ){ return ( (*this)+=(-m) ); }
		inline Matrix  operator -  ( const Matrix& m ) const { return (*this)+(-m); }
		inline Matrix& operator /= ( Real s ){ return ( (*this)*=(Real)(1./s) ); }
		inline Matrix  operator /  ( Real s ) const { return (*this) * ( (Real)(1./s) ); }

		static Real Dot( const Matrix& m1 , const Matrix& m2 ){ Real dot = (Real)0 ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) dot += m1.coords[c][r] * m2.coords[c][r] ; return dot; }

		template< typename T >
		inline Point< T , Rows > operator* ( const Point< T , Columns >& p ) const { Point< T , Rows > q ; for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) q[r] += (T)( p[c] * coords[c][r] ) ; return q; }

		template< unsigned int Cols >
		inline Matrix< Real , Cols , Rows > operator * ( const Matrix< Real , Cols , Columns > &M ) const
		{
			Matrix< Real , Cols , Rows > prod;
			for( unsigned int c=0 ; c<Cols ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) for( unsigned int i=0 ; i<Columns ; i++ ) prod(c,r) += coords[i][r] * M(c,i);
			return prod;
		}

		inline Matrix< Real , Rows , Columns > transpose( void ) const
		{
			Matrix< Real , Rows , Columns > t;
			for( unsigned int c=0 ; c<Columns ; c++ ) for( unsigned int r=0 ; r<Rows ; r++ ) t(r,c) = coords[c][r];
			return t;
		}
		friend std::ostream &operator << ( std::ostream &os , const Matrix &m )
		{
			os << "{ ";
			for( int r=0 ; r<Rows ; r++ )
			{
				if( r ) os << " , ";
				os << "{ ";
				for( int c=0 ; c<Columns ; c++ )
				{
					if( c ) os << " , ";
					os << m.coords[c][r];
				}
				os << " }";
			}
			return os << " }";
		}
	};

	template< class Real , unsigned int Dim >
	struct XForm
	{
		Real coords[Dim][Dim];
		XForm( void ) { memset( coords , 0 , sizeof(Real) * Dim * Dim ); }
		XForm( const Matrix< Real , Dim , Dim > &M ){ memcpy( coords , M.coords , sizeof(Real)*Dim*Dim ); }
		XForm( const XForm &M ){ memcpy( coords , M.coords , sizeof(Real)*Dim*Dim ); }
		XForm( const Matrix< Real , Dim+1 , Dim+1 > &M ){ for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) this->operator()(i,j) = M(i,j); }
		XForm( const XForm< Real , Dim+1 > &M ){ for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) this->operator()(i,j) = M(i,j); }
		XForm &operator = ( const Matrix< Real , Dim , Dim > &M ){ memcpy( coords , M.coords , sizeof(Real)*Dim*Dim ) ; return *this; }
		XForm &operator = ( const  XForm< Real , Dim >       &M ){ memcpy( coords , M.coords , sizeof(Real)*Dim*Dim ) ; return *this; }
		XForm &operator = ( const Matrix< Real , Dim+1 , Dim+1> &M ){ for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) this->operator()(i,j) = M(i,j) ; return *this; }
		XForm &operator = ( const  XForm< Real , Dim+1 >        &M ){ for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) this->operator()(i,j) = M(i,j) ; return *this; }

		static XForm Identity( void )
		{
			XForm xForm;
			for( unsigned int d=0 ; d<Dim ; d++ ) xForm(d,d) = (Real)1.;
			return xForm;
		}
		Real& operator() ( unsigned int i , unsigned int j ){ return coords[i][j]; }
		const Real& operator() ( unsigned int i , unsigned int j ) const { return coords[i][j]; }
		template< class _Real > Point< _Real , Dim-1 > operator * ( const Point< _Real , Dim-1 >& p ) const
		{
			Point< _Real , Dim-1 > q;
			for( unsigned int i=0 ; i<Dim-1 ; i++ )
			{
				for( unsigned int j=0 ; j<Dim-1 ; j++ ) q[i] += (_Real)( coords[j][i] * p[j] );
				q[i] += (_Real)coords[Dim-1][i];
			}
			return q;
		}
		template< class _Real > Point< _Real , Dim > operator * ( const Point< _Real , Dim >& p ) const
		{
			Point< _Real , Dim > q;
			for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) q[i] += (_Real)( coords[j][i] * p[j] );
			return q;
		}
		XForm operator * ( const XForm& m ) const
		{
			XForm n;
			for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) for( unsigned int k=0 ; k<Dim ; k++ ) n.coords[i][j] += m.coords[i][k]*coords[k][j];
			return n;
		}
		XForm transpose( void ) const
		{
			XForm xForm;
			for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) xForm( i , j ) = coords[j][i];
			return xForm;
		}
		Real determinant( void ) const
		{
			Real det = (Real)0.;
			for( unsigned int d=0 ; d<Dim ; d++ ) 
				if( d&1 ) det -= coords[d][0] * subDeterminant( d , 0 );
				else      det += coords[d][0] * subDeterminant( d , 0 );
			return det;
		}
		XForm inverse( void ) const
		{
			XForm xForm;
			Real d = determinant();
			for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ )
				if( (i+j)%2==0 ) xForm.coords[j][i] =  subDeterminant( i , j ) / d;
				else             xForm.coords[j][i] = -subDeterminant( i , j ) / d;
			return xForm;
		}
		Real subDeterminant( unsigned int i , unsigned int j ) const
		{
			XForm< Real , Dim-1 > xForm;
			unsigned int ii[Dim-1] , jj[Dim-1];
			for( unsigned int a=0 , _i=0 , _j=0 ; a<Dim ; a++ )
			{
				if( a!=i ) ii[_i++] = a;
				if( a!=j ) jj[_j++] = a;
			}
			for( unsigned int _i=0 ; _i<Dim-1 ; _i++ ) for( unsigned int _j=0 ; _j<Dim-1 ; _j++ ) xForm( _i , _j ) = coords[ ii[_i] ][ jj[_j] ];
			return xForm.determinant();
		}

		inline XForm  operator - ( void ) const { XForm m ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) m.coords[c][r] = - coords[c][r] ; return m; }

		inline XForm& operator += ( const XForm& m ){ for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) coords[c][r] += m.coords[c][r] ; return *this; }
		inline XForm  operator +  ( const XForm& m ) const { XForm n ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) n.coords[c][r] = coords[c][r] + m.coords[c][r] ; return n; }
		inline XForm& operator *= ( Real s ) { for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) coords[c][r] *= s ; return *this; }
		inline XForm  operator *  ( Real s ) const { XForm n ; for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) n.coords[c][r] = coords[c][r] * s ; return n; }

		inline XForm& operator -= ( const XForm& m ){ return ( (*this)+=(-m) ); }
		inline XForm  operator -  ( const XForm& m ) const { return (*this)+(-m); }
		inline XForm& operator /= ( Real s ){ return ( (*this)*=(Real)(1./s) ); }
		inline XForm  operator /  ( Real s ) const { return (*this) * ( (Real)(1./s) ); }

		friend std::ostream &operator << ( std::ostream &os , const XForm &x )
		{
			os << "{ ";
			for( int r=0 ; r<Dim ; r++ )
			{
				if( r ) os << " , ";
				os << "{ ";
				for( int c=0 ; c<Dim ; c++ )
				{
					if( c ) os << " , ";
					os << x.coords[c][r];
				}
				os << " }";
			}
			return os << " }";
		}
	};
	template<>
	inline XForm< float , 1 > XForm< float , 1 >::inverse( void ) const
	{
		XForm< float , 1 > x;
		x.coords[0][0] = (float)(1./coords[0][0] );
		return x;
	}
	template<>
	inline XForm< double , 1 > XForm< double , 1 >::inverse( void ) const
	{
		XForm< double , 1 > x;
		x.coords[0][0] = (double)(1./coords[0][0] );
		return x;
	}
	template<> inline float  XForm< float  , 1 >::determinant( void ) const { return coords[0][0]; }
	template<> inline double XForm< double , 1 >::determinant( void ) const { return coords[0][0]; }

	template< class Real , unsigned int Dim >
	struct OrientedPoint
	{
		Point< Real , Dim > p , n;
		OrientedPoint( Point< Real , Dim > pp = Point< Real , Dim >() , Point< Real , Dim > nn=Point< Real , Dim >() ) : p(pp) , n(nn) { ; }
		template< class _Real > OrientedPoint( const OrientedPoint< _Real , Dim>& p ) : OrientedPoint( Point< Real , Dim >( p.p ) , Point< Real , Dim >( p.n ) ){ ; }

		template< class _Real > inline OrientedPoint& operator += ( OrientedPoint< _Real , Dim > _p ){ p += _p.p , n += _p.n ; return *this; }
		template< class _Real > inline OrientedPoint  operator +  ( OrientedPoint< _Real , Dim > _p ) const { return OrientedPoint< Real , Dim >( p+_p.p , n+_p.n ); }
		template< class _Real > inline OrientedPoint& operator *= ( _Real r ) { p *= r , n *= r ; return *this; }
		template< class _Real > inline OrientedPoint  operator *  ( _Real r ) const { return OrientedPoint< Real , Dim >( p*r , n*r ); }

		template< class _Real > inline OrientedPoint& operator -= ( OrientedPoint< _Real , Dim > p ){ return ( (*this)+=(-p) ); }
		template< class _Real > inline OrientedPoint  operator -  ( OrientedPoint< _Real , Dim > p ) const { return (*this)+(-p); }
		template< class _Real > inline OrientedPoint& operator /= ( _Real r ){ return ( (*this)*=Real(1./r) ); }
		template< class _Real > inline OrientedPoint  operator /  ( _Real r ) const { return (*this) * ( Real(1.)/r ); }
	};


	template< class Data , class Real >
	struct ProjectiveData
	{
		Data data;
		Real weight;
		ProjectiveData( Data d=Data() , Real w=(Real)0 ) : data(d) , weight(w) { ; }
		operator Data (){ return weight!=0 ? data/weight : data*weight; }
		Data value( void ) const { return weight!=0 ? data/weight : data*weight; }
		ProjectiveData& operator += ( const ProjectiveData& p ){ data += p.data , weight += p.weight ; return *this; }
		ProjectiveData& operator -= ( const ProjectiveData& p ){ data -= p.data , weight -= p.weight ; return *this; }
		ProjectiveData& operator *= ( Real s ){ data *= s , weight *= s ; return *this; }
		ProjectiveData& operator /= ( Real s ){ data /= s , weight /= s ; return *this; }
		ProjectiveData  operator +  ( const ProjectiveData& p ) const { return ProjectiveData( data+p.data , weight+p.weight ); }
		ProjectiveData  operator -  ( const ProjectiveData& p ) const { return ProjectiveData( data-p.data , weight-p.weight ); }
		ProjectiveData  operator *  ( Real s ) const { return ProjectiveData( data*s , weight*s ); }
		ProjectiveData  operator /  ( Real s ) const { return ProjectiveData( data/s , weight/s ); }
	};

	template< class Real , unsigned int Dim > Point< Real , Dim > RandomBallPoint( void );
	template< class Real , unsigned int Dim > Point< Real , Dim > RandomSpherePoint( void );
	template< class Real , unsigned int Dim > Real Length( Point< Real , Dim > p ){ return (Real)sqrt( Point< Real , Dim >::SquareNorm( p ) ); }
	template< class Real , unsigned int Dim > Real SquareLength( Point< Real , Dim > p ){ return Point< Real , Dim >::SquareNorm( p ); }
	template< class Real , unsigned int Dim > Real Distance( Point< Real , Dim > p1 , Point< Real , Dim > p2 ){ return Length(p1-p2); }
	template< class Real , unsigned int Dim > Real SquareDistance( Point< Real , Dim > p1 , Point< Real , Dim > p2 ){ return SquareLength( p1-p2 ); }
	template< class Real > Point< Real , 3 > CrossProduct( Point< Real , 3 > p1 , Point< Real , 3 > p2 ){ return Point< Real , 3 >::CrossProduct( p1 , p2 ); }

	template< class Real , unsigned int Dim > Real SquareArea( Point< Real , Dim > p1 , Point< Real , Dim > p2 , Point< Real , Dim > p3 )
	{
		Point< Real , Dim > v1 = p2-p1 , v2 = p3-p1;
		// Area^2 = ( |v1|^2 * |v2|^2 * sin^2( < v1 ,v2 ) ) / 4
		//        = ( |v1|^2 * |v2|^2 * ( 1 - cos^2( < v1 ,v2 ) ) ) / 4
		//        = ( |v1|^2 * |v2|^2 * ( 1 - < v1 , v2 >^2 / ( |v1|^2 * |v2|^2 ) ) ) / 4
		//        = ( |v1|^2 * |v2|^2 - < v1 , v2 >^2 ) / 4
		Real dot = Point< Real , Dim >::Dot( v1 , v2 );
		Real l1 = Point< Real , Dim >::SquareNorm( v1 ) , l2 = Point< Real , Dim >::SquareNorm( v2 );
		return ( l1 * l2 - dot * dot ) / 4;
	}
	template< class Real , unsigned int Dim > Real Area( Point< Real , Dim > p1 , Point< Real , Dim > p2 , Point< Real , Dim > p3 ){ return (Real)sqrt( SquareArea( p1 , p2 , p3 ) ); }

	template< unsigned int K > struct Factorial{ static const unsigned long long Value = Factorial< K-1 >::Value * K; };
	template<> struct Factorial< 0 >{ static const unsigned long long Value = 1; };

	template< class Real , unsigned int Dim , unsigned int K >
	struct Simplex
	{
		Point< Real , Dim > p[K+1];
		Simplex( void ){ static_assert( K<=Dim , "[ERROR] Bad simplex dimension" ); }
		Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
		const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
		Real measure( void ) const { return (Real)sqrt( squareMeasure() ); }
		Real squareMeasure( void ) const
		{
			XForm< Real , K > mass;
			for( unsigned int i=1 ; i<=K ; i++ ) for( unsigned int j=1 ; j<=K ; j++ ) mass(i-1,j-1) = Point< Real , Dim >::Dot( p[i]-p[0] , p[j]-p[0] );
			return mass.determinant() / ( Factorial< K >::Value * Factorial< K >::Value );
		}
		Point< Real , Dim > center( void ) const
		{
			Point< Real , Dim > c;
			for( unsigned int k=0 ; k<=K ; k++ ) c += p[k];
			return c / (K+1);
		}
		void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const;

		template< unsigned int _K=K >
		typename std::enable_if< _K==Dim-1 , Point< Real , Dim > >::type normal( void ) const
		{
			static_assert( K==Dim-1 , "[ERROR] Co-dimension is not one" );
			Point< Real , Dim > d[Dim-1];
			for( int k=1 ; k<Dim ; k++ ) d[k-1] = p[k] - p[0];
			return Point< Real , Dim >::CrossProduct( d );
		}
		template< unsigned int _K=K >
		typename std::enable_if< _K==Dim-1 , bool >::type intersect( Ray< Real , Dim > ray , Real &t , Real barycentricCoordinates[Dim] ) const
		{
			static_assert( K==Dim-1 , "[ERROR] Co-dimension is not one" );
			Point< Real , Dim > n = normal();
			Real denominator = Point< Real , Dim >::Dot( n , ray.direction );
			if( denominator==0 ) return false;
			// Solve for t s.t. < ray(t) , n > = < p[0] , n >
			t = Point< Real , Dim >::Dot( n , p[0] - ray.position ) / denominator;
			Point< Real,  Dim > q = ray(t);

			// Let C be the matrix whose columns are the simplex vertices.
			// Solve for the barycentric coordinates, b, minimizing:
			//		E(b) = || C * b - q ||^2
			//		     = b^t * C^t * C * b - 2 * b^t * C^t * q + || q ||^2
			// Taking the gradient with respect to b gives:
			//		   b = ( C^t * C )^{-1} * C^t * q
			Matrix< Real , Dim-1 , Dim > C;
			for( int c=0 ; c<Dim-1 ; c++ ) for( int r=0 ; r<Dim ; r++ ) C(c,r) = p[c+1][r] - p[0][r];
			Matrix< Real , Dim , Dim-1 > C_t = C.transpose();
			XForm< Real , Dim-1 > M = C_t * C;
			Point< Real , Dim-1 > bc = M.inverse() * ( C_t * ( q - p[0] ) );
			barycentricCoordinates[0] = (Real)1.;
			for( unsigned int d=0 ; d<Dim-1 ; d++ ) barycentricCoordinates[d+1] = bc[d] , barycentricCoordinates[0] -= bc[d];
			for( unsigned int d=0 ; d<Dim ; d++ ) if( barycentricCoordinates[d]<0 ) return false;
			return true;
		}

		template< unsigned int _K=K >
		static typename std::enable_if< _K==Dim-1 , bool >::type IsInterior( Point< Real , Dim > p , const std::vector< Simplex< Real , Dim , Dim-1 > > &simplices )
		{
			if( !simplices.size() ) MK_THROW( "No simplices provided" );

			// Create a ray that intersecting the largest simplex
			int idx;
			{
				Real maxMeasure = 0;
				for( int i=0 ; i<simplices.size() ; i++ )
				{
					Real measure = simplices[i].squareMeasure();
					if( measure>maxMeasure ) maxMeasure = measure , idx = i;
				}
			}
			Ray< Real , Dim > ray( p , simplices[idx].center() - p );
			Real l = (Real)Point< Real , Dim >::SquareNorm( ray.direction );
			if( !l ) MK_THROW( "point is on simplex" );
			l = (Real)sqrt(l);
			ray.direction /= l;

			// Make the assessment based on which side of the simplex the point p is
			Real min_t = l;
			bool isInside = Point< Real , Dim >::Dot( simplices[idx].normal() , ray.direction )*min_t>0;

			// Look for intersections with closer simplices
			for( size_t i=0 ; i<simplices.size() ; i++ )
			{
				Real t , barycentricCoordinates[ Dim ];
				if( simplices[i].intersect( ray , t , barycentricCoordinates ) && t<min_t )	min_t = t , isInside = Point< Real , Dim >::Dot( simplices[i].normal() , ray.direction )*t>0;
			}
			return isInside;
		}

		template< unsigned int _K=K >
		static typename std::enable_if< _K==Dim-1 , bool >::type IsInterior( Point< Real , Dim > p , const std::vector< Simplex< Real , Dim , Dim-1 > > &simplices , const std::vector< Point< Real , Dim > > &normals )
		{
			if( !simplices.size() ) MK_THROW( "No simplices provided" );

#if 0
			// A more conservative approach for ray-tracing, sending a ray for each simplex and using the consensus solution
			Real insideMeasure = 0;
			for( int idx=0 ; idx<simplices.size() ; idx++ )
			{
				Ray< Real , Dim > ray( p , simplices[idx].center() - p );
				Real l = (Real)Point< Real , Dim >::SquareNorm( ray.direction );
				if( !l ){ MK_WARN( "point is on simplex" ) ; continue; }
				l = (Real)sqrt(l);
				ray.direction /= l;

				// Make the assessment based on which side of the simplex the point p is
				Real min_t = l;
				bool isInside = Point< Real , Dim >::Dot( normals[idx] , ray.direction )*min_t>0;

				// Look for intersections with closer simplices
				for( size_t i=0 ; i<simplices.size() ; i++ )
				{
					Real t , barycentricCoordinates[ Dim ];
					if( simplices[i].intersect( ray , t , barycentricCoordinates ) && t<min_t )	min_t = t , isInside = Point< Real , Dim >::Dot( normals[i] , ray.direction )*t>0;
				}
				Real measure = simplices[idx].measure();
				if( isInside ) insideMeasure += measure;
				else           insideMeasure -= measure;
			}
			return insideMeasure>0;
#else
			// Create a ray that intersecting the largest simplex
			int idx;
			{
				Real maxMeasure = 0;
				for( int i=0 ; i<simplices.size() ; i++ )
				{
					Real measure = simplices[i].squareMeasure();
					if( measure>maxMeasure ) maxMeasure = measure , idx = i;
				}
			}
			Ray< Real , Dim > ray( p , simplices[idx].center() - p );
			Real l = (Real)Point< Real , Dim >::SquareNorm( ray.direction );
			if( !l ) MK_THROW( "point is on simplex" );
			l = (Real)sqrt(l);
			ray.direction /= l;

			// Make the assessment based on which side of the simplex the point p is
			Real min_t = l;
			bool isInside = Point< Real , Dim >::Dot( normals[idx] , ray.direction )*min_t>0;

			// Look for intersections with closer simplices
			for( size_t i=0 ; i<simplices.size() ; i++ )
			{
				Real t , barycentricCoordinates[ Dim ];
				if( simplices[i].intersect( ray , t , barycentricCoordinates ) && t<min_t ) min_t = t , isInside = Point< Real , Dim >::Dot( normals[i] , ray.direction )*t>0;
			}
			return isInside;
#endif
		}
	};


	template< class Real , unsigned int Dim >	
	struct Simplex< Real , Dim , 0 >
	{
		static const unsigned int K=0;
		Point< Real , Dim > p[1];
		Point< Real , Dim >& operator[]( unsigned int k ){ return p[k]; }
		const Point< Real , Dim >& operator[]( unsigned int k ) const { return p[k]; }
		Real squareMeasure( void ) const { return (Real)1.; }
		Real measure( void ) const { return (Real)1.; }
		Point< Real , Dim > center( void ) const { return p[0]; }
		void split( Point< Real , Dim > pNormal , Real pOffset , std::vector< Simplex >& back , std::vector< Simplex >& front ) const
		{
			if( Point< Real , Dim >::Dot( p[0] , pNormal ) < pOffset ) back.push_back( *this );
			else                                                       front.push_back( *this );
		}
		template< unsigned int _K=K >
		typename std::enable_if< _K==Dim-1 , bool >::type intersect( Ray< Real , Dim > ray , Real &t , Real barycentricCoordinates[Dim] ) const
		{
			static_assert( K==Dim-1 , "[ERROR] Co-dimension is not one" );
			if( !ray.direction[0] ) return false;
			// Solve for t s.t. ray(t) = p[0]
			t = ( p[0][0] - ray.position[0] ) / ray.direction[0];
			barycentricCoordinates[0] = (Real)1.;
			return true;
		}

		template< unsigned int _K=0 >
		static typename std::enable_if< _K==Dim-1 , bool >::type IsInterior( Point< Real , Dim > p , const std::vector< Simplex< Real , Dim , Dim-1 > > &simplices , const std::vector< Point< Real , Dim > > &normals )
		{
			if( !simplices.size() ) MK_THROW( "No simplices provided" );

			Ray< Real , Dim > ray( p , simplices[0].center() - p );
			Real l = (Real)Point< Real , Dim >::SquareNorm( ray.direction );
			if( !l ) MK_THROW( "point is on simplex" );
			l = (Real)sqrt(l);
			ray.direction /= l;

			// Make the assessment based on which side of the simplex the point p is
			Real min_t = l;
			bool isInside = Point< Real , Dim >::Dot( normals[0] , ray.direction )*min_t>0;

			// Look for intersections with closer simplices
			for( size_t i=0 ; i<simplices.size() ; i++ )
			{
				Real t , barycentricCoordinates[ Dim ];
				if( simplices[i].intersect( ray , t , barycentricCoordinates ) && t<min_t ) min_t = t , isInside = Point< Real , Dim >::Dot( normals[i] , ray.direction )*t>0;
			}
			return isInside;
		}
	};
	template< typename Real , unsigned int Dim , unsigned int K >
	std::ostream &operator << ( std::ostream &os , const Simplex< Real , Dim , K > &s )
	{
		for( unsigned int k=0 ; k<K ; k++ ) os << s.p[k] << " , ";
		return os << s.p[K];
	}


	template< class Real , unsigned int Dim > using Edge = Simplex< Real , Dim , 1 >;
	template< class Real , unsigned int Dim > using Triangle = Simplex< Real , Dim , 2 >;

	template< unsigned int K , typename Index >
	struct SimplexIndex
	{
		Index idx[K+1];
		template< class ... Ints >
		SimplexIndex( Ints ... values ){ static_assert( sizeof...(values)==K+1 || sizeof...(values)==0 , "[ERROR] Invalid number of coefficients" ) ; _init( 0 , values ... ); }
		Index &operator[] ( unsigned int i ) { return idx[i] ;}
		const Index &operator[] ( unsigned int i ) const { return idx[i]; }
	protected:
		void _init( unsigned int k )
		{
			if( !k ) memset( idx , 0 , sizeof(idx) );
			else MK_THROW( "Should never be called" );
		}
		template< class ... Ints > void _init( unsigned int k , Index v , Ints ... values )
		{
			idx[k] = v;
			if( k<K ) _init( k+1 , values ... );
		}
	};
	template< typename Index > using EdgeIndex = SimplexIndex< 1 , Index >;
	template< typename Index > using TriangleIndex = SimplexIndex< 2 , Index >;

	template< typename Real , unsigned int Dim , unsigned int K >
	struct SimplicialComplex
	{
		SimplicialComplex( const std::vector< Simplex< Real , Dim , K > > &simplices ) : _simplices( simplices ){}
		virtual size_t size( void ) const { return _simplices.size(); }
		virtual Simplex< Real , Dim , K > operator[]( size_t idx ) const { return _simplices[idx]; }
	protected:
		SimplicialComplex( void ) :_simplices(__simplices) {}
		const std::vector< Simplex< Real , Dim , K > > &_simplices;
		const std::vector< Simplex< Real , Dim , K > > __simplices;
	};

	template< typename Real , unsigned int Dim , unsigned int K , typename IndexType >
	struct IndexedSimplicialComplex : public SimplicialComplex< Real , Dim , K >
	{
		IndexedSimplicialComplex( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , IndexType > > &simplices ) : _vertices(vertices) , _simplices(simplices){}
		IndexedSimplicialComplex( IndexedSimplicialComplex && isc )
		{
			std::swap( _vertices , isc._vertices );
			std::swap( _simplices , isc._simplices );
		}

		size_t size( void ) const { return _simplices.size(); }
		Simplex< Real , Dim , K > operator[]( size_t idx ) const
		{
			Simplex< Real , Dim , K > s;
			for( unsigned int k=0 ; k<=K ; k++ ) s[k] = _vertices[ _simplices[idx][k] ];
			return s;
		}
	protected:
		const std::vector< Point< Real , Dim > > &_vertices;
		const std::vector< SimplexIndex< K , IndexType > > &_simplices;
	};


	template< typename Index >
	class TriangulationEdge
	{
	public:
		TriangulationEdge( void ){ pIndex[0] = pIndex[1] = tIndex[0] = tIndex[1] = -1; }
		Index pIndex[2] , tIndex[2];
	};

	template< typename Index >
	class TriangulationTriangle
	{
	public:
		TriangulationTriangle( void ){ eIndex[0] = eIndex[1] = eIndex[2] = -1; }
		Index eIndex[3];
	};

#include "Geometry.inl"
}

#endif // GEOMETRY_INCLUDED
