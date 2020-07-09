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

///////////////////////////
// BSplineEvaluationData //
///////////////////////////
template< int Degree >
double BSplineEvaluationData< Degree >::Value( int depth , int off , double s , bool dirichlet , bool derivative )
{
	if( s<0 || s>1 ) return 0.;

	int dim = Dimension(depth) , res = 1<<depth;
	if( off<0 || off>=dim ) return 0;

	BSplineComponents components = BSplineComponents( depth , off , dirichlet );

	// [NOTE] This is an ugly way to ensure that when s=1 we evaluate using a B-Spline component within the valid range.
	int ii = std::max< int >( 0 , std::min< int >( res-1 , (int)floor( s * res ) ) ) - off;

	if( ii<SupportStart || ii>SupportEnd ) return 0;
	if( derivative ) return components[ii-SupportStart].derivative()(s);
	else             return components[ii-SupportStart](s);
}
template< int Degree >
void BSplineEvaluationData< Degree >::SetCenterEvaluator( typename CenterEvaluator::Evaluator& evaluator , int depth , bool dirichlet )
{
	evaluator._depth = depth;
	int dim = BSplineEvaluationData< Degree >::Dimension( depth ) , res = 1<<depth;
	for( int i=0 ; i<CenterEvaluator::Size ; i++ ) for( int j=SupportStart ; j<=SupportEnd ; j++ )
	{
		int ii = ( i<=CenterEvaluator::Start ? i : ( dim - CenterEvaluator::Size + i ) );
		double s = 0.5 + ii + j;
		for( int d1=0 ; d1<2 ; d1++ ) evaluator._ccValues[d1][i][j-SupportStart] = Value( depth , ii , s/res , dirichlet , d1!=0 );
	}
}
template< int Degree >
void BSplineEvaluationData< Degree >::SetChildCenterEvaluator( typename CenterEvaluator::ChildEvaluator& evaluator , int parentDepth , bool dirichlet )
{
	evaluator._parentDepth = parentDepth;
	int dim = BSplineEvaluationData< Degree >::Dimension( parentDepth ) , res = 1<<(parentDepth+1);
	for( int i=0 ; i<CenterEvaluator::Size ; i++ ) for( int j=ChildSupportStart ; j<=ChildSupportEnd ; j++ )
	{
		int ii = ( i<=CenterEvaluator::Start ? i : ( dim - CenterEvaluator::Size + i ) );
		double s = 0.5 + 2*ii + j;
		for( int d1=0 ; d1<2 ; d1++ ) evaluator._pcValues[d1][i][j-ChildSupportStart] = Value( parentDepth , ii , s/res , dirichlet , d1!=0 );
	}
}
template< int Degree >
double BSplineEvaluationData< Degree >::CenterEvaluator::Evaluator::value( int fIdx , int cIdx , bool d ) const
{
	int dd = cIdx-fIdx , res = 1<<(_depth) , dim = Dimension(_depth);
	if( cIdx<0 || fIdx<0 || cIdx>=res || fIdx>=dim || dd<SupportStart || dd>SupportEnd ) return 0;
	return _ccValues[d?1:0][ CenterEvaluator::Index( _depth , fIdx ) ][dd-SupportStart];
}
template< int Degree >
double BSplineEvaluationData< Degree >::CenterEvaluator::ChildEvaluator::value( int fIdx , int cIdx , bool d ) const
{
	int dd = cIdx-2*fIdx , res = 1<<(_parentDepth+1) , dim = Dimension(_parentDepth);
	if( cIdx<0 || fIdx<0 || cIdx>=res || fIdx>=dim || dd<ChildSupportStart || dd>ChildSupportEnd ) return 0;
	return _pcValues[d?1:0][ CenterEvaluator::Index( _parentDepth , fIdx ) ][dd-ChildSupportStart];
}
template< int Degree >
void BSplineEvaluationData< Degree >::SetCornerEvaluator( typename CornerEvaluator::Evaluator& evaluator , int depth , bool dirichlet )
{
	evaluator._depth = depth;
	int dim = BSplineEvaluationData< Degree >::Dimension( depth ) , res = 1<<depth;
	for( int i=0 ; i<CornerEvaluator::Size ; i++ ) for( int j=CornerStart ; j<=CornerEnd ; j++ )
	{
		int ii = ( i<=CornerEvaluator::Start ? i : ( dim - CornerEvaluator::Size + i ) );
		double s = ii + j;
		for( int d1=0 ; d1<2 ; d1++ ) evaluator._ccValues[d1][i][j-CornerStart] = Value( depth , ii , s/res , dirichlet , d1!=0 );
	}
}
template< int Degree >
void BSplineEvaluationData< Degree >::SetChildCornerEvaluator( typename CornerEvaluator::ChildEvaluator& evaluator , int parentDepth , bool dirichlet )
{
	evaluator._parentDepth = parentDepth;
	int dim = BSplineEvaluationData< Degree >::Dimension( parentDepth ) ,  res = 1<<(parentDepth+1);
	for( int i=0 ; i<CornerEvaluator::Size ; i++ ) for( int j=ChildCornerStart ; j<=ChildCornerEnd ; j++ )
	{
		int ii = ( i<=CornerEvaluator::Start ? i : ( dim - CornerEvaluator::Size + i ) );
		double s = 2*ii + j;
		for( int d1=0 ; d1<2 ; d1++ ) evaluator._pcValues[d1][i][j-ChildCornerStart] = Value( parentDepth , ii , s/res , dirichlet , d1!=0 );
	}
}
template< int Degree >
void BSplineEvaluationData< Degree >::SetUpSampleEvaluator( UpSampleEvaluator& evaluator , int lowDepth , bool dirichlet )
{
	evaluator._lowDepth = lowDepth;
	int lowDim = Dimension(lowDepth);
	for( int i=0 ; i<UpSampleEvaluator::Size ; i++ )
	{
		int ii = ( i<=UpSampleEvaluator::Start ? i : ( lowDim - UpSampleEvaluator::Size + i ) );
		BSplineUpSamplingCoefficients b( lowDepth , ii , dirichlet );
		for( int j=0 ; j<UpSampleSize ; j++ ) evaluator._pcValues[i][j] = b[j];
	}
}
template< int Degree >
double BSplineEvaluationData< Degree >::CornerEvaluator::Evaluator::value( int fIdx , int cIdx , bool d ) const
{
	int dd = cIdx-fIdx , res = ( 1<<_depth ) + 1 , dim = Dimension(_depth);
	if( cIdx<0 || fIdx<0 || cIdx>=res || fIdx>=dim || dd<CornerStart || dd>CornerEnd ) return 0;
	return _ccValues[d?1:0][ CornerEvaluator::Index( _depth , fIdx ) ][dd-CornerStart];
}
template< int Degree >
double BSplineEvaluationData< Degree >::CornerEvaluator::ChildEvaluator::value( int fIdx , int cIdx , bool d ) const
{
	int dd = cIdx-2*fIdx , res = ( 1<<(_parentDepth+1) ) + 1 , dim = Dimension(_parentDepth);
	if( cIdx<0 || fIdx<0 || cIdx>=res || fIdx>=dim || dd<ChildCornerStart || dd>ChildCornerEnd ) return 0;
	return _pcValues[d?1:0][ CornerEvaluator::Index( _parentDepth , fIdx ) ][dd-ChildCornerStart];
}
template< int Degree >
double BSplineEvaluationData< Degree >::UpSampleEvaluator::value( int pIdx , int cIdx ) const
{
	int dd = cIdx-2*pIdx , pDim = Dimension( _lowDepth ) , cDim = Dimension( _lowDepth+1 );
	if( cIdx<0 || pIdx<0 || cIdx>=cDim || pIdx>=pDim || dd<UpSampleStart || dd>UpSampleEnd ) return 0;
	return _pcValues[ UpSampleEvaluator::Index( _lowDepth , pIdx ) ][dd-UpSampleStart];
}

//////////////////////////////////////////////
// BSplineEvaluationData::BSplineComponents //
//////////////////////////////////////////////
template< int Degree >
BSplineEvaluationData< Degree >::BSplineComponents::BSplineComponents( int depth , int offset , bool dirichlet )
{
	int res = 1<<depth;
	BSplineElements< Degree > elements( res , offset , dirichlet );

	// The first index is the position, the second is the element type
	Polynomial< Degree > components[Degree+1][Degree+1];
	// Generate the elements that can appear in the base function corresponding to the base function at (depth,offset) = (0,0)
	for( int d=0 ; d<=Degree ; d++ ) for( int dd=0 ; dd<=Degree ; dd++ ) components[d][dd] = Polynomial< Degree >::BSplineComponent( Degree-dd ).shift( -( (Degree+1)/2 ) + d );

	// Now adjust to the desired depth and offset
	double width = 1. / res;
	for( int d=0 ; d<=Degree ; d++ ) for( int dd=0 ; dd<=Degree ; dd++ ) components[d][dd] = components[d][dd].scale( width ).shift( width*offset );

	// Now write in the polynomials
	for( int d=0 ; d<=Degree ; d++ )
	{
		int idx = offset + SupportStart + d;
		_polys[d] = Polynomial< Degree >();

		if( idx>=0 && idx<res ) for( int dd=0 ; dd<=Degree ; dd++ ) _polys[d] += components[d][dd] * ( ( double )( elements[idx][dd] ) ) / elements.denominator;
	}
}

//////////////////////////////////////////////////////////
// BSplineEvaluationData::BSplineUpSamplingCoefficients //
//////////////////////////////////////////////////////////
template< int Degree >
BSplineEvaluationData< Degree >::BSplineUpSamplingCoefficients::BSplineUpSamplingCoefficients( int depth , int offset , bool dirichlet )
{
	// [ 1/8 1/2 3/4 1/2 1/8]
	// [ 1 , 1 ] ->  [ 3/4 , 1/2 , 1/8 ] + [ 1/8 , 1/2 , 3/4 ] = [ 7/8 , 1 , 7/8 ]
	int dim = Dimension(depth) , _dim = Dimension(depth+1);
	bool reflect;
	offset = BSplineData< Degree >::RemapOffset( depth , offset , reflect );
	int multiplier = ( dirichlet && reflect ) ? -1 : 1;
	bool useReflected = Inset || ( offset % ( dim-1 ) );
	int b[ UpSampleSize ];
	Polynomial< Degree+1 >::BinomialCoefficients( b );

	// Clear the values
	memset( _coefficients , 0 , sizeof(int) * UpSampleSize );

	// Get the array of coefficients, relative to the origin
	int* coefficients = _coefficients - ( 2*offset + UpSampleStart );
	for( int i=UpSampleStart ; i<=UpSampleEnd ; i++ )
	{
		int _offset = 2*offset+i;
		_offset = BSplineData< Degree >::RemapOffset( depth+1 , _offset , reflect );
		if( useReflected || !reflect )
		{
			int _multiplier = multiplier * ( ( dirichlet && reflect ) ? -1 : 1 );
			coefficients[ _offset ] += b[ i-UpSampleStart ] * _multiplier;
		}
		// If we are not inset and we are at the boundary, use the reflection as well
		if( !Inset && ( offset % (dim-1) ) && !( _offset % (_dim-1) ) )
		{
			_offset = BSplineData< Degree >::RemapOffset( depth+1 , _offset , reflect );
			int _multiplier = multiplier * ( ( dirichlet && reflect ) ? -1 : 1 );
			if( dirichlet ) _multiplier *= -1;
			coefficients[ _offset ] += b[ i-UpSampleStart ] * _multiplier;
		}
	}
}

////////////////////////////
// BSplineIntegrationData //
////////////////////////////
template< int Degree1 , int Degree2 >
double BSplineIntegrationData< Degree1 , Degree2 >::Dot( int depth1 ,  int off1 , bool dirichlet1 , bool d1 , int depth2 , int off2 , bool dirichlet2 , bool d2 )
{
	const int _Degree1 = (d1 ? (Degree1-1) : Degree1) , _Degree2 = (d2 ? (Degree2-1) : Degree2);
	int sums[ Degree1+1 ][ Degree2+1 ];

	int depth = std::max< int >( depth1 , depth2 );

	BSplineElements< Degree1 > b1( 1<<depth1 , off1 , dirichlet1 );
	BSplineElements< Degree2 > b2( 1<<depth2 , off2 , dirichlet2 );

	{
		BSplineElements< Degree1 > b;
		while( depth1<depth ) b=b1 , b.upSample( b1 ) , depth1++;
	}
	{
		BSplineElements< Degree2 > b;
		while( depth2<depth ) b=b2 , b.upSample( b2 ) , depth2++;
	}

	BSplineElements< Degree1-1 > db1;
	BSplineElements< Degree2-1 > db2;
	b1.differentiate( db1 ) , b2.differentiate( db2 );

	int start1=-1 , end1=-1 , start2=-1 , end2=-1;
	for( int i=0 ; i<int( b1.size() ) ; i++ )
	{
		for( int j=0 ; j<=Degree1 ; j++ )
		{
			if( b1[i][j] && start1==-1 ) start1 = i;
			if( b1[i][j] ) end1 = i+1;
		}
		for( int j=0 ; j<=Degree2 ; j++ )
		{
			if( b2[i][j] && start2==-1 ) start2 = i;
			if( b2[i][j] ) end2 = i+1;
		}
	}
	if( start1==end1 || start2==end2 || start1>=end2 || start2>=end1 ) return 0.;
	int start = std::max< int >( start1 , start2 ) , end = std::min< int >( end1 , end2 );
	memset( sums , 0 , sizeof( sums ) );

	// Iterate over the support
	for( int i=start ; i<end ; i++ )
		// Iterate over all pairs of elements within a node
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ )
			// Accumulate the product of the coefficients
			sums[j][k] += ( d1 ?  db1[i][j] : b1[i][j] ) * ( d2 ? db2[i][k] : b2[i][k] );

	double _dot = 0;
	if( d1 && d2 )
	{
		double integrals[ Degree1 ][ Degree2 ];
		SetBSplineElementIntegrals< Degree1-1 , Degree2-1 >( integrals );
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ ) _dot += integrals[j][k] * sums[j][k];
	}
	else if( d1 )
	{
		double integrals[ Degree1 ][ Degree2+1 ];
		SetBSplineElementIntegrals< Degree1-1 , Degree2 >( integrals );
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ ) _dot += integrals[j][k] * sums[j][k];
	}
	else if( d2 )
	{
		double integrals[ Degree1+1 ][ Degree2 ];
		SetBSplineElementIntegrals< Degree1 , Degree2-1 >( integrals );
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ ) _dot += integrals[j][k] * sums[j][k];
	}
	else
	{
		double integrals[ Degree1+1 ][ Degree2+1 ];
		SetBSplineElementIntegrals< Degree1 , Degree2 >( integrals );
		for( int j=0 ; j<=_Degree1 ; j++ ) for( int k=0 ; k<=_Degree2 ; k++ ) _dot += integrals[j][k] * sums[j][k];
	}

	_dot /= b1.denominator;
	_dot /= b2.denominator;
	if     ( d1 && d2 ) return _dot * (1<<depth);
	else if( d1 || d2 ) return _dot;
	else                return _dot / (1<<depth);
}
template< int Degree1 , int Degree2 >
void BSplineIntegrationData< Degree1, Degree2 >::SetIntegrator( typename FunctionIntegrator::Integrator& integrator , int depth , bool dirichlet1 , bool dirichlet2 )
{
	integrator._depth = depth;
	int dim = BSplineEvaluationData< Degree2 >::Dimension( depth );
	for( int i=0 ; i<FunctionIntegrator::Size ; i++ ) for( int j=OverlapStart ; j<=OverlapEnd ; j++ )
	{
		int ii = ( i<=FunctionIntegrator::Start ? i : ( dim - FunctionIntegrator::Size + i ) );
		for( int d1=0 ; d1<2 ; d1++ ) for( int d2=0 ; d2<2 ; d2++ ) integrator._ccIntegrals[d1][d2][i][j-OverlapStart] = Dot( depth , ii , dirichlet1 , d1!=0 , depth , ii+j , dirichlet2 , d2!=0 );
	}
}
template< int Degree1 , int Degree2 >
void BSplineIntegrationData< Degree1, Degree2 >::SetChildIntegrator( typename FunctionIntegrator::ChildIntegrator& integrator , int parentDepth , bool dirichlet1 , bool dirichlet2 )
{
	integrator._parentDepth = parentDepth;
	int dim = BSplineEvaluationData< Degree2 >::Dimension( parentDepth );
	for( int i=0 ; i<FunctionIntegrator::Size ; i++ ) for( int j=ChildOverlapStart ; j<=ChildOverlapEnd ; j++ )
	{
		int ii = ( i<=FunctionIntegrator::Start ? i : ( dim - FunctionIntegrator::Size + i ) );
		for( int d1=0 ; d1<2 ; d1++ ) for( int d2=0 ; d2<2 ; d2++ ) integrator._pcIntegrals[d1][d2][i][j-ChildOverlapStart] = Dot( parentDepth , ii , dirichlet1 , d1!=0 , parentDepth+1 , 2*ii+j , dirichlet2 , d2!=0 );
	}
}
template< int Degree1 , int Degree2 >
double BSplineIntegrationData< Degree1 , Degree2 >::FunctionIntegrator::Integrator::dot( int off1 , int off2 , bool d1 , bool d2 ) const
{
	int d = off2-off1 , dim1 = BSplineEvaluationData< Degree1 >::Dimension( _depth ) , dim2 = BSplineEvaluationData< Degree2 >::Dimension( _depth );
	if( off1<0 || off2<0 || off1>=dim1 || off2>=dim2 || d<OverlapStart || d>OverlapEnd ) return 0;
	return _ccIntegrals[d1?1:0][d2?1:0][ FunctionIntegrator::Index( _depth , off1 ) ][d-OverlapStart];
}
template< int Degree1 , int Degree2 >
double BSplineIntegrationData< Degree1 , Degree2 >::FunctionIntegrator::ChildIntegrator::dot( int off1 , int off2 , bool d1 , bool d2 ) const
{
	int d = off2-2*off1 , dim1 = BSplineEvaluationData< Degree1 >::Dimension( _parentDepth ) , dim2 = BSplineEvaluationData< Degree2 >::Dimension( _parentDepth+1 );
	if( off1<0 || off2<0 || off1>=dim1 || off2>=dim2 || d<ChildOverlapStart || d>ChildOverlapEnd ) return 0;
	return _pcIntegrals[d1?1:0][d2?1:0][ FunctionIntegrator::Index( _parentDepth , off1 ) ][d-ChildOverlapStart];
}
/////////////////
// BSplineData //
/////////////////
#define MODULO( A , B ) ( (A)<0 ? ( (B)-((-(A))%(B)) ) % (B) : (A) % (B) )
template< int Degree >
int BSplineData< Degree >::RemapOffset( int depth , int offset , bool& reflect )
{
	const int I = ( Degree&1 ) ? 0 : 1;
	int dim = Dimension( depth );
	offset = MODULO( offset , 2*(dim-1+I) );
	reflect = offset>=dim;
	if( reflect ) return 2*(dim-1+I) - (offset+I);
	else          return offset;
}
#undef MODULO

template< int Degree > BSplineData< Degree >::BSplineData( void ){ functionCount = sampleCount = 0; }


template< int Degree >
void BSplineData< Degree >::set( int maxDepth , bool dirichlet )
{
	_dirichlet = dirichlet;

	depth = maxDepth;
	functionCount = TotalFunctionCount( depth );
	sampleCount = TotalSampleCount( depth );
	baseBSplines = NewPointer< typename BSplineEvaluationData< Degree >::BSplineComponents >( functionCount );

	for( size_t i=0 ; i<functionCount ; i++ )
	{
		int d , off;
		FactorFunctionIndex( (int)i , d , off );
		baseBSplines[i] = typename BSplineEvaluationData< Degree >::BSplineComponents( d , off , _dirichlet );
	}
}

/////////////////////
// BSplineElements //
/////////////////////
template< int Degree >
BSplineElements< Degree >::BSplineElements( int res , int offset , bool dirichlet )
{
	denominator = 1;
	std::vector< BSplineElementCoefficients< Degree > >::resize( res , BSplineElementCoefficients< Degree >() );

	// If we have primal dirichlet constraints, the boundary functions are necessarily zero
	if( _Primal && dirichlet && !(offset%res) ) return;

	// Construct the B-Spline
	for( int i=0 ; i<=Degree ; i++ )
	{
		int idx = -_Off + offset + i;
		if( idx>=0 && idx<res ) (*this)[idx][i] = 1;
	}
	// Fold in the periodic instances (which cancels the negation)
	_addPeriodic< true >( _RotateLeft ( offset , res ) , false ) , _addPeriodic< false >( _RotateRight( offset , res ) , false );

	// Recursively fold in the boundaries
	if( _Primal && !(offset%res) ) return;

	// Fold in the reflected instance (which may require negation)
	_addPeriodic< true >( _ReflectLeft( offset , res ) , dirichlet ) , _addPeriodic< false >( _ReflectRight( offset , res ) , dirichlet );
}
template< int Degree > int BSplineElements< Degree >::_ReflectLeft ( int offset , int res ){ return (Degree&1) ?      -offset :      -1-offset; }
template< int Degree > int BSplineElements< Degree >::_ReflectRight( int offset , int res ){ return (Degree&1) ? 2*res-offset : 2*res-1-offset; }
template< int Degree > int BSplineElements< Degree >::_RotateLeft  ( int offset , int res ){ return offset-2*res; }
template< int Degree > int BSplineElements< Degree >::_RotateRight ( int offset , int res ){ return offset+2*res; }

template< int Degree >
template< bool Left >
void BSplineElements< Degree >::_addPeriodic( int offset , bool negate )
{
	int res = int( std::vector< BSplineElementCoefficients< Degree > >::size() );
	bool set = false;
	// Add in the corresponding B-spline elements (possibly negated)
	for( int i=0 ; i<=Degree ; i++ )
	{
		int idx = -_Off + offset + i;
		if( idx>=0 && idx<res ) (*this)[idx][i] += negate ? -1 : 1 , set = true;
	}
	// If there is a change for additional overlap, give it a go
	if( set ) _addPeriodic< Left >( Left ? _RotateLeft( offset , res ) : _RotateRight( offset , res ) , negate );
}
template< int Degree >
void BSplineElements< Degree >::upSample( BSplineElements< Degree >& high ) const
{
	int bCoefficients[ BSplineEvaluationData< Degree >::UpSampleSize ];
	Polynomial< Degree+1 >::BinomialCoefficients( bCoefficients );

	high.resize( std::vector< BSplineElementCoefficients< Degree > >::size()*2 );
	high.assign( high.size() , BSplineElementCoefficients< Degree >() );
	// [NOTE] We have flipped the order of the B-spline elements
	for( int i=0 ; i<int(std::vector< BSplineElementCoefficients< Degree > >::size()) ; i++ ) for( int j=0 ; j<=Degree ; j++ )
	{
		// At index I , B-spline element J corresponds to a B-spline centered at:
		//		I - SupportStart - J
		int idx = i - BSplineEvaluationData< Degree >::SupportStart - j;
		for( int k=BSplineEvaluationData< Degree >::UpSampleStart ; k<=BSplineEvaluationData< Degree >::UpSampleEnd ; k++ )
		{
			// Index idx at the coarser resolution gets up-sampled into indices:
			//		2*idx + [UpSampleStart,UpSampleEnd]
			// at the finer resolution
			int _idx = 2*idx + k;
			// Compute the index of the B-spline element relative to 2*i and 2*i+1
			int _j1 = -_idx + 2*i - BSplineEvaluationData< Degree >::SupportStart , _j2 = -_idx + 2*i + 1 - BSplineEvaluationData< Degree >::SupportStart;
			if( _j1>=0 && _j1<=Degree ) high[2*i+0][_j1] += (*this)[i][j] * bCoefficients[k-BSplineEvaluationData< Degree >::UpSampleStart];
			if( _j2>=0 && _j2<=Degree ) high[2*i+1][_j2] += (*this)[i][j] * bCoefficients[k-BSplineEvaluationData< Degree >::UpSampleStart];
		}
	}
	high.denominator = denominator<<Degree;
}

template< int Degree >
void BSplineElements< Degree >::differentiate( BSplineElements< Degree-1 >& d ) const
{
	d.resize( std::vector< BSplineElementCoefficients< Degree > >::size() );
	d.assign( d.size()  , BSplineElementCoefficients< Degree-1 >() );
	for( int i=0 ; i<int(std::vector< BSplineElementCoefficients< Degree > >::size()) ; i++ ) for( int j=0 ; j<=Degree ; j++ )
	{
		if( j-1>=0 )   d[i][j-1] -= (*this)[i][j];
		if( j<Degree ) d[i][j  ] += (*this)[i][j];
	}
	d.denominator = denominator;
}

// If we were really good, we would implement this integral table to store
// rational values to improve precision...
template< int Degree1 , int Degree2 >
void SetBSplineElementIntegrals( double integrals[Degree1+1][Degree2+1] )
{
	for( int i=0 ; i<=Degree1 ; i++ )
	{
		Polynomial< Degree1 > p1 = Polynomial< Degree1 >::BSplineComponent( Degree1-i );
		for( int j=0 ; j<=Degree2 ; j++ )
		{
			Polynomial< Degree2 > p2 = Polynomial< Degree2 >::BSplineComponent( Degree2-j );
			integrals[i][j] = ( p1 * p2 ).integral( 0 , 1 );
		}
	}
}
