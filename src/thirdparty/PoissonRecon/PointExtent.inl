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


////////////////////
// GetBoundingBox //
////////////////////
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor , bool rotate )
{
	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = max[0] - min[0];
	for( int d=1 ; d<Dim ; d++ ) scale = std::max< Real >( scale , max[d]-min[d] );
	scale *= scaleFactor;
	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity() , rXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	unsigned int maxDim = 0;
	for( int i=1 ; i<Dim ; i++ ) if( (max[i]-min[i])>(max[maxDim]-min[maxDim]) ) maxDim = i;
	if( rotate )
	{
		for( int i=0 ; i<Dim ; i++ ) rXForm(i,i) = 0;
		for( int i=0 ; i<Dim ; i++ ) rXForm((maxDim+i)%Dim,(Dim-1+i)%Dim) = 1;
	}
	return rXForm * sXForm * tXForm;
}

template< class Real , unsigned int Dim , bool ExtendedAxes >
XForm< Real , Dim+1 > GetXForm( const Extent< Real , Dim , ExtendedAxes > &extent , Real scaleFactor , unsigned int dir )
{
	using _Extent = Extent< Real , Dim , ExtendedAxes >;
	bool rotate = ( dir >= _Extent::DirectionN );

	if( rotate ) // Find the direction of maximal extent
	{
		dir = 0;
		for( unsigned int d=1 ; d<_Extent::DirectionN ; d++ ) if( extent[d].second - extent[d].first > extent[dir].second - extent[dir].first ) dir = d;
	}

	const unsigned int *frame = _Extent::Frame(dir);

	// Compute the rotation taking the direction of maximal extent to the last axis
	XForm< Real , Dim+1 > R = XForm< Real , Dim+1 >::Identity();
	for( unsigned int c=0 ; c<Dim ; c++ ) for( unsigned int r=0 ; r<Dim ; r++ ) R(r,c) = _Extent::Direction( frame[c] )[r];

	// Get the bounding box with respect to the maximal extent direction's frame
	Point< Real , Dim >  bBox[2];
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		bBox[0][d] = extent[ frame[d] ].first;
		bBox[1][d] = extent[ frame[d] ].second;
	}
	return GetXForm( bBox[0] , bBox[1] , scaleFactor , rotate ) * R;
}

///////////
// Frame //
///////////
template< typename Real , unsigned int Dim , bool ExtendedAxes >
Frame< Real , Dim , ExtendedAxes >::Frame( void )
{
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		directions[d][d] = 1;
		for( unsigned int dd=0 ; dd<Dim ; dd++ ) frames[d][dd] = (d+1+dd) % Dim;
	}

	if constexpr( ExtendedAxes )
	{
		static_assert( Dim==2 || Dim==3 , "[ERROR] Non-axis extents only supported for dimensions 2 and 3" );
		if constexpr( Dim==2 )
		{
			directions[2] = Point< Real , 2 >( 1 , 1 )/(Real)sqrt(2.);
			directions[3] = Point< Real , 2 >( 1 ,-1 )/(Real)sqrt(2.);
			frames[2][0] = 2 , frames[2][1] = 3;
			frames[3][0] = 3 , frames[3][1] = 2;
		}
		else
		{
			directions[3] = Point< Real , 3 >( 1 , 1 , 0 )/(Real)sqrt(2.);
			directions[4] = Point< Real , 3 >( 1 , 0 , 1 )/(Real)sqrt(2.);
			directions[5] = Point< Real , 3 >( 0 , 1 , 1 )/(Real)sqrt(2.);
			directions[6] = Point< Real , 3 >( 1 ,-1 , 0 )/(Real)sqrt(2.);
			directions[7] = Point< Real , 3 >( 1 , 0 ,-1 )/(Real)sqrt(2.);
			directions[8] = Point< Real , 3 >( 0 , 1 ,-1 )/(Real)sqrt(2.);
			frames[3][0] = 2 , frames[3][1] = 6 , frames[3][2] = 3;
			frames[4][0] = 7 , frames[4][1] = 1 , frames[4][2] = 4;
			frames[5][0] = 0 , frames[5][1] = 8 , frames[5][2] = 5;
			frames[6][0] = 3 , frames[6][1] = 2 , frames[6][2] = 6;
			frames[7][0] = 1 , frames[7][1] = 4 , frames[7][2] = 7;
			frames[8][0] = 5 , frames[8][1] = 0 , frames[8][2] = 8;
		}
	}
}

////////////
// Extent //
////////////
template< typename Real , unsigned int Dim , bool ExtendedAxes >
Extent< Real , Dim , ExtendedAxes >::Extent( void )
{
	Real inf = std::numeric_limits< Real >::infinity();
	for( unsigned int d=0 ; d<DirectionN ; d++ ) extents[d].first = inf , extents[d].second = -inf;
}

template< typename Real , unsigned int Dim , bool ExtendedAxes >
void Extent< Real , Dim , ExtendedAxes >::add( Point< Real , Dim > p )
{
	for( unsigned int d=0 ; d<DirectionN ; d++ )
	{
		extents[d].first  = std::min< Real >( extents[d].first  , Point< Real , Dim >::Dot( p , _Frame.directions[d] ) );
		extents[d].second = std::max< Real >( extents[d].second , Point< Real , Dim >::Dot( p , _Frame.directions[d] ) );
	}
}

template< typename Real , unsigned int Dim , bool ExtendedAxes >
Extent< Real , Dim , ExtendedAxes > Extent< Real , Dim , ExtendedAxes >::operator + ( const Extent &e ) const
{
	Extent _e;
	for( unsigned int d=0 ; d<DirectionN ; d++ )
	{
		_e.extents[d].first  = std::min< Real >( extents[d].first  , e.extents[d].first  );
		_e.extents[d].second = std::max< Real >( extents[d].second , e.extents[d].second );
	}
	return _e;
}

template< typename Real , unsigned int Dim , bool ExtendedAxes >
std::ostream &operator << ( std::ostream &os , const Extent< Real , Dim , ExtendedAxes > &e )
{
	for( unsigned int d=0 ; d<Extent< Real , Dim , ExtendedAxes >::DirectionN ; d++ )
	{
		os << Extent< Real , Dim , ExtendedAxes >::_Frame.directions[d] << " : [ " << e.extents[d].first << " , " << e.extents[d].second << " ]";
		os << "\t(" << e.extents[d].second - e.extents[d].first << " )" << std::endl;
	}
	return os;
}

template< class Real , unsigned int Dim , bool ExtendedAxes , typename ... Data >
Extent< Real , Dim , ExtendedAxes > GetExtent( InputDataStream< Point< Real , Dim > , Data ... > &stream , Data ... data )
{
	Point< Real,  Dim > p;
	Extent< Real , Dim , ExtendedAxes > e;
	while( stream.read( p , data... ) ) e.add(p);
	return e;
}

template< class Real , unsigned int Dim , bool ExtendedAxes , typename ... Data >
XForm< Real , Dim+1 > GetXForm( InputDataStream< Point< Real , Dim > , Data ... > &stream , Data ... data , Real scaleFactor , unsigned int dir )
{
	return GetXForm( GetExtent< Real , Dim , ExtendedAxes >( stream  , data... ) , scaleFactor , dir );
}
