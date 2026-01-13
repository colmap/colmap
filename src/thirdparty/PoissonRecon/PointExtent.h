/*
Copyright (c) 2023, Michael Kazhdan
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

#ifndef POINT_EXTENT_INCLUDED
#define POINT_EXTENT_INCLUDED

#include <ostream>
#include "Geometry.h"
#include "DataStream.h"

namespace PoissonRecon
{
	namespace PointExtent
	{
		template< typename Real , unsigned int Dim , bool ExtendedAxes >
		struct Frame
		{
			static const unsigned int DirectionN = ExtendedAxes ? ( Dim==2 ? 4 : 9 ) : Dim;
			Point< Real , Dim > directions[ DirectionN ];
			unsigned int frames[DirectionN][Dim];
			Frame( void );
		};

		template< typename Real , unsigned int Dim , bool ExtendedAxes=true >
		struct Extent
		{
			static const unsigned int DirectionN = Frame< Real , Dim , ExtendedAxes >::DirectionN;
			static Point< Real , Dim > Direction( unsigned int d ){ return _Frame.directions[d]; }
			static const unsigned int *Frame( unsigned int d ){ return _Frame.frames[d]; }

			std::pair< Real , Real > extents[ DirectionN ];
			std::pair< Real , Real > &operator[]( unsigned int d ){ return extents[d]; }
			const std::pair< Real , Real > &operator[]( unsigned int d ) const { return extents[d]; }

			Extent( void );
			void add( Point< Real , Dim > p );
			Extent operator + ( const Extent &e ) const;
		protected:
			static const PointExtent::Frame< Real , Dim , ExtendedAxes > _Frame;

			template< typename _Real , unsigned int _Dim , bool _ExtendedAxes >
			friend std::ostream &operator << ( std::ostream & , const Extent< _Real , _Dim , _ExtendedAxes > & );
		};

		template< typename Real , unsigned int Dim , bool ExtendedAxes >
		const Frame< Real , Dim , ExtendedAxes > Extent< Real , Dim , ExtendedAxes >::_Frame;

		template< class Real , unsigned int Dim , bool ExtendedAxes , typename ... Data >
		Extent< Real , Dim , ExtendedAxes > GetExtent( InputDataStream< Point< Real , Dim > , Data ... > &stream , Data ... data );

		template< class Real , unsigned int Dim >
		XForm< Real , Dim+1 > GetXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor , bool rotate=true );

		template< class Real , unsigned int Dim , bool ExtendedAxes >
		XForm< Real , Dim+1 > GetXForm( const Extent< Real , Dim , ExtendedAxes > &extent , Real scaleFactor , unsigned int dir );

		template< class Real , unsigned int Dim , bool ExtendedAxes , typename ... Data >
		XForm< Real , Dim+1 > GetXForm( InputDataStream< Point< Real , Dim > , Data ... > &stream , Data ... data , Real scaleFactor , unsigned int dir );

#include "PointExtent.inl"
	}
}

#endif // POINT_EXTENT_INCLUDED