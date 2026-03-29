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

template< class Real , unsigned int DataDegree , unsigned int ... DataDegrees > typename std::enable_if< sizeof ... ( DataDegrees )==0 >::type __SetBSplineComponentValues( const Real* position , const Real* start , Real width , double* values , unsigned int stride )
{
	Polynomial< DataDegree >::BSplineComponentValues( ( position[0] - start[0] ) / width , values );
}
template< class Real , unsigned int DataDegree , unsigned int ... DataDegrees > typename std::enable_if< sizeof ... ( DataDegrees )!=0 >::type __SetBSplineComponentValues( const Real* position , const Real* start , Real width , double* values , unsigned int stride )
{
	Polynomial< DataDegree >::BSplineComponentValues( ( position[0] - start[0] ) / width , values );
	__SetBSplineComponentValues< Real , DataDegrees ... >( position+1 , start+1 , width , values + stride , stride );
}


// evaluate the result of splatting along a plane and then evaluating at a point on the plane.
template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int Degree >
Real FEMTree< Dim , Real >::_GetScaleValue( Point< Real , Dim > p ) const
{
	static_assert( ( Dim>=CoDim) , "[ERROR] Co-dimension exceeds dimension" );
	static const int PointSupportStart = -BSplineSupportSizes< Degree >::SupportEnd , PointSupportEnd = -BSplineSupportSizes< Degree >::SupportStart;
	static const int PointSupportSize = PointSupportEnd - PointSupportStart + 1;
	static const int BSplineSupportStart = BSplineSupportSizes< Degree >::SupportStart , BSplineSupportEnd = BSplineSupportSizes< Degree >::SupportEnd;
	static const int BSplineSupportSize = BSplineSupportEnd - BSplineSupportStart + 1;
	double splineValues[Dim][Degree+1];

	// Evaluate the B-spline component functions at the position
	for( int d=0 ; d<Dim ; d++ ) Polynomial< Degree >::BSplineComponentValues( p[d] , splineValues[d] );

	StaticWindow< double , IsotropicUIntPack< Dim , PointSupportSize > > splatValues , densityValues;

	// Get the values with which the center point splats into its neighbors
	{
		double scratch[Dim+1];
		scratch[0] = 1.;
		int idx[Dim];
		WindowLoop< Dim >::Run
		(
			PointSupportStart , PointSupportEnd+1 ,
			[&]( int d , int i ){ scratch[d+1] = scratch[d] * splineValues[d][i-PointSupportStart] ,  idx[d] = i; } ,
			[&]( void )
		{
			int _idx[Dim];
			for( int d=0 ; d<Dim ; d++ ) _idx[d] = idx[d] - PointSupportStart;
			splatValues( _idx ) = scratch[ Dim ] , densityValues( _idx ) = 0;
		}
		);
	}

	// Splat from points along the hyperplane
	// A point at node i will contribute to the evaluation of a point at node 0 if:
	//		0 <= i + PointSupportEnd + BSplineSupportEnd
	// and
	//		0 >= i + PointSupportStart + BSplineSupportStart
	// Or, equivalently:
	//		- PointSupportEnd - BSplineSupportEnd <= i <= - PointSupportStart - BSplineSupportStart
	{
		int neighborPointIndex[Dim];
		// Iterate over all points that can contribute
		WindowLoop< Dim >::Run
		(
			- PointSupportEnd - BSplineSupportEnd , - PointSupportStart - BSplineSupportStart + 1 ,
			[&]( int d , int i ){ neighborPointIndex[d] = i; } ,
			[&]( void )
		{
			// Check that the neighboring point's node lies on the hyperplane
			bool validNeighbor = true;
			for( int d=0 ; d<CoDim ; d++ ) if( neighborPointIndex[d]!=0 ) validNeighbor = false;

			if( validNeighbor )
			{
				int splineIndex[Dim] , _splineIndex[Dim];
				// Iterate over all B-Splines supported on the neighboring point
				WindowLoop< Dim >::Run
				(
					PointSupportStart , PointSupportEnd+1 ,
					[&]( int d , int i ){ splineIndex[d] = neighborPointIndex[d] + i , _splineIndex[d] = i; } ,
					[&]( void )
				{
					int idx[Dim] , _idx[Dim];
					bool inRange = true;
					for( int d=0 ; d<Dim ; d++ )
					{
						idx[d] = splineIndex[d] - PointSupportStart;
						_idx[d] = _splineIndex[d] - PointSupportStart;
						if( idx[d]<0 || idx[d]>=PointSupportSize ) inRange = false;
					}
					if( inRange ) densityValues( idx ) += splatValues( _idx );
				}
				);
			}
		}
		);
	}

	double scaleValue = 0;
	{
		int idx[Dim];
		WindowLoop< Dim >::Run
		(
			PointSupportStart , PointSupportEnd+1 ,
			[&]( int d , int i ){ idx[d] = i - PointSupportStart; } ,
			[&]( void ){ scaleValue += splatValues(idx) * densityValues(idx); }
		);
	}

	return (Real)( 1./scaleValue );
}

// Evaluate the result of splatting along a hyper-plane of co-dimension CoDim through points in the interior of the node and then evaluating at those points.
template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int Degree > Real FEMTree< Dim , Real >::_GetScaleValue( unsigned int res ) const
{
	Point< Real , Dim > p;
	Real dx = (Real)(1./res);
	unsigned int count = 0;
	Real scaleValueSum = 0;

	WindowLoop< Dim >::Run
	(
		0 , res ,
		[&]( int d , int i ){ p[d] = dx/2 + dx*i; } ,
		[&]( void ){ count++ ; scaleValueSum += _GetScaleValue< CoDim , Degree >(p); }
	);
	return scaleValueSum / count;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe , unsigned int CoDim , unsigned int WeightDegree >
void FEMTree< Dim , Real >::_addWeightContribution( Allocator< FEMTreeNode > *nodeAllocator , DensityEstimator< WeightDegree >& densityWeights , FEMTreeNode* node , Point< Real , Dim > position , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , Real weight )
{
	static const Real ScaleValue = _GetScaleValue< CoDim , WeightDegree >( 10 );
	double values[ Dim ][ BSplineSupportSizes< WeightDegree >::SupportSize ];
	typename FEMTreeNode::template Neighbors< IsotropicUIntPack< Dim , BSplineSupportSizes< WeightDegree >::SupportSize > >& neighbors = weightKey.template getNeighbors< true , ThreadSafe >( node , nodeAllocator , _nodeInitializer );

	densityWeights.reserve( nodeCount() );

	// Evaluate the B-spline components at the position
	{
		Point< Real , Dim > start;
		Real w;
		node->startAndWidth( start , w );
		for( int dim=0 ; dim<Dim ; dim++ ) Polynomial< WeightDegree >::BSplineComponentValues( ( position[dim]-start[dim] ) / w , values[dim] );
	}

	weight *= (Real)ScaleValue;
	double scratch[Dim+1];
	scratch[0] = weight;
	WindowLoop< Dim >::Run
	(
		IsotropicUIntPack< Dim , 0 >() , IsotropicUIntPack< Dim , BSplineSupportSizes< WeightDegree >::SupportSize >() ,
		[&]( int d , int i ){ scratch[d+1] = scratch[d] * values[d][i]; } ,
		[&]( FEMTreeNode* node )
	{
		if( node )
		{
			AddAtomic( densityWeights[ node ] , (Real)scratch[Dim] );
		}
	} ,
		neighbors.neighbors()
	);
}

template< unsigned int Dim , class Real >
template< unsigned int WeightDegree , class PointSupportKey >
Real FEMTree< Dim , Real >::_getSamplesPerNode( const DensityEstimator< WeightDegree >& densityWeights , const FEMTreeNode* node , Point< Real , Dim > position , PointSupportKey& weightKey ) const
{
	Real weight = 0;
	typedef typename PointSupportKey::NeighborType Neighbors;
	double values[ Dim ][ BSplineSupportSizes< WeightDegree >::SupportSize ];
	Neighbors neighbors = weightKey.getNeighbors( node );
	Point< Real , Dim > start;
	Real w;
	_startAndWidth( node , start , w );

	for( int dim=0 ; dim<Dim ; dim++ ) Polynomial< WeightDegree >::BSplineComponentValues( ( position[dim]-start[dim] ) / w , values[dim] );
	double scratch[Dim+1];
	scratch[0] = 1;
	WindowLoop< Dim >::Run
	(
		IsotropicUIntPack< Dim , 0 >() , IsotropicUIntPack< Dim , BSplineSupportSizes< WeightDegree >::SupportSize >() ,
		[&]( int d , int i ){ scratch[d+1] = scratch[d] * values[d][i]; } ,
		[&]( typename Neighbors::StaticWindow::data_type node ){ if( node ){ const Real *w = densityWeights( node ) ; if( w ) weight += (Real)( scratch[Dim] * (*w) ); } } ,
		neighbors.neighbors()
	);
	return weight;
}
template< unsigned int Dim , class Real >
template< unsigned int WeightDegree , class PointSupportKey >
void FEMTree< Dim , Real >::_getSampleDepthAndWeight( const DensityEstimator< WeightDegree >& densityWeights , const FEMTreeNode* node , Point< Real , Dim > position , PointSupportKey& weightKey , Real& depth , Real& weight ) const
{
	const FEMTreeNode* temp = node;
	while( _localDepth( temp )>densityWeights.kernelDepth() ) temp = temp->parent;
	// Goal:
	// Find the depth d at which the number of samples per node is equal to densityWeights.samplesPerNode.
	// Assume that the number of samples per node grows by a factor of 2^( Dim-CoDim ) as the depth is decreased by 1.
	// That is:
	//		SamplesPerNode( d ) = C / 2^( d * ( Dim - CoDim ) )
	// So, given a target spd, we have:
	//		spd = C / 2^( d * ( Dim - CoDim ) )
	//		log( spd ) = log( C ) / log( 2^( d * ( Dim - CoDim ) ) )
	//		log( spd ) = log( C ) - log( 2 ) * ( d * ( Dim - CoDim ) ) )
	//		d = [ log( C ) - log( spd ) ] / [ log(2) * ( Dim-CoDim ) ]
	// To get C, we note that if we know that we have spd_0 at depth d_0, this gives:
	//		spd_0 = C / 2^( d_0 * ( Dim - CoDim ) )
	//		C = spd_0 * 2^( d_0 * ( Dim - CoDim ) )
	// Putting these together, we get:
	//		d = [ log( spd_0 * 2^( d_0 * ( Dim - CoDim ) ) ) - log( spd ) ] / [ log(2) * ( Dim-CoDim ) ]
	//		d = [ log( spd_0 ) - log( spd ) + log(2) * ( d_0 * ( Dim - CoDim ) ) ) ] / [ log(2) * ( Dim-CoDim ) ]
	//		d = [ log( spd_0 / spd ) ] / [ log(2) * ( Dim-CoDim ) ]  + d_0

	Real samplesPerNode = _getSamplesPerNode( densityWeights , temp , position , weightKey );
	if( samplesPerNode>=densityWeights.samplesPerNode() ) depth = Real( _localDepth( temp ) + log( samplesPerNode / densityWeights.samplesPerNode() ) / ( log(2.) * ( Dim-densityWeights.coDimension() ) ) );
	else
	{
		Real fineSamplesPerNode , coarseSamplesPerNode;
		fineSamplesPerNode = coarseSamplesPerNode = samplesPerNode;
		while( coarseSamplesPerNode<densityWeights.samplesPerNode() && _localDepth(temp) )
		{
			temp = temp->parent;
			fineSamplesPerNode = coarseSamplesPerNode;
			coarseSamplesPerNode = _getSamplesPerNode( densityWeights , temp , position , weightKey );
		}
		// Rather than assuming that the number of samples per node scales by a factor of 2^(Dim-CoDim),
		// use the fact that between the coarse and fine levels the samples per node scaled by coarseSamplesPerNode / fineSamplesPerNode
		depth = Real( _localDepth( temp ) + log( coarseSamplesPerNode / densityWeights.samplesPerNode() ) / log( coarseSamplesPerNode / fineSamplesPerNode ) );
		samplesPerNode = coarseSamplesPerNode;
	}
	Real nodeWidth = (Real)( 1. / (1<<_localDepth(temp) ) );
	weight = (Real)pow( nodeWidth , Dim-densityWeights.coDimension() ) / samplesPerNode;
}
template< unsigned int Dim , class Real >
template< unsigned int WeightDegree , class PointSupportKey >
void FEMTree< Dim , Real >::_getSampleDepthAndWeight( const DensityEstimator< WeightDegree >& densityWeights , Point< Real , Dim > position , PointSupportKey& weightKey , Real& depth , Real& weight ) const
{
	FEMTreeNode* temp;
	Point< Real,  Dim > myCenter;
	for( int d=0 ; d<Dim ; d++ ) myCenter[d] = (Real)0.5;
	Real myWidth = Real( 1. );

	// Get the finest node with depth less than or equal to the splat depth that contains the point
	temp = _spaceRoot;
	while( _localDepth( temp )<densityWeights.kernelDepth() )
	{
		if( !IsActiveNode< Dim >( temp->children ) ) break; // MK_THROW( "" );
		int cIndex = FEMTreeNode::ChildIndex( myCenter , position );
		temp = temp->children + cIndex;
		myWidth /= 2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) myCenter[d] += myWidth/2;
			else                  myCenter[d] -= myWidth/2;
	}
	return _getSampleDepthAndWeight( densityWeights , temp , position , weightKey , depth , weight );
}

template< unsigned int Dim , class Real >
template< bool CreateNodes , bool ThreadSafe , class V , unsigned int ... DataSigs >
void FEMTree< Dim , Real >::_splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , FEMTreeNode* node , Point< Real , Dim > position , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& dataInfo , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey )
{
	typedef UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... > SupportSizes;
	double values[ Dim ][ SupportSizes::Max() ];
	typename FEMTreeNode::template Neighbors< UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... > >& neighbors = dataKey.template getNeighbors< CreateNodes , ThreadSafe >( node , nodeAllocator , _nodeInitializer );
	Point< Real , Dim > start;
	Real w;
	_startAndWidth( node , start , w );
	__SetBSplineComponentValues< Real , FEMSignature< DataSigs >::Degree ... >( &position[0] , &start[0] , w , &values[0][0] , SupportSizes::Max() );
	double scratch[Dim+1];
	scratch[0] = 1;
	WindowLoop< Dim >::Run
	(
		ZeroUIntPack< Dim >() , UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... >() ,
		[&]( int d , int i ){ scratch[d+1] = scratch[d] * values[d][i]; } ,
		[&]( FEMTreeNode* node ){ if( IsActiveNode< Dim >( node ) ) Atomic< V >::Add( dataInfo.at( node , zero ) , v * (Real)scratch[Dim] ); } ,
		neighbors.neighbors()
	);
}
template< unsigned int Dim , class Real >
template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs >
Point< Real , 2 > FEMTree< Dim , Real >::_splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >& densityWeights , Real minDepthCutoff , Point< Real , Dim > position , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& dataInfo , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , LocalDepth minDepth , LocalDepth maxDepth , int dim , Real depthBias )
{
	// Get the depth and weight at position
	Real weight , depth;
	FEMTreeNode *temp = _spaceRoot;
	Point< Real , Dim > myCenter;
	Real myWidth;
	{
		for( int d=0 ; d<Dim ; d++ ) myCenter[d] = (Real)0.5;
		myWidth = (Real)1.;
		while( _localDepth( temp )<densityWeights.kernelDepth() )
		{
			if( !IsActiveNode< Dim >( temp->children ) ) break;
			int cIndex = FEMTreeNode::ChildIndex( myCenter , position );
			temp = temp->children + cIndex;
			myWidth /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) myCenter[d] += myWidth/2;
				else                  myCenter[d] -= myWidth/2;
		}
		_getSampleDepthAndWeight( densityWeights , temp , position , weightKey , depth , weight );
		depth += depthBias;
	}

	if( depth<minDepthCutoff ) return Point< Real , 2 >( (Real)-1. , (Real)0. );
	Real rawDepth = depth;

	if( depth<minDepth ) depth = Real(minDepth);
	if( depth>maxDepth ) depth = Real(maxDepth);
	int topDepth = (int)ceil(depth);

	double dx = 1.0-(topDepth-depth);
	if     ( topDepth<=minDepth ) topDepth = minDepth , dx = 1;
	else if( topDepth> maxDepth ) topDepth = maxDepth , dx = 1;

	while( _localDepth( temp )>topDepth ) temp=temp->parent;
	while( _localDepth( temp )<topDepth )
	{
		if( !temp->children ) temp->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		int cIndex = FEMTreeNode::ChildIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth/=2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) myCenter[d] += myWidth/2;
			else                  myCenter[d] -= myWidth/2;
	}

	auto Splat = [&]( FEMTreeNode *node , Real dx )
	{
		double width = 1.0 / ( 1<<_localDepth( temp ) );
		// Scale by:
		//		weight: the area/volume associated with the sample
		//		dx: the fraction of the sample splatted into the current depth
		//		pow( width , -dim ): So that each sample is splatted with a unit volume
		V _v = v * weight / Real( pow( width , dim ) ) * dx;
//		V _v = v / Length(v) * dx;
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
		_splatPointData< CreateNodes , ThreadSafe , V >( zero , nodeAllocator , temp , position , _v , dataInfo , dataKey );
#else // !__GNUC__ || __GNUC__ >=5
		_splatPointData< CreateNodes , ThreadSafe , V ,  DataSigs ... >( zero , nodeAllocator , temp , position , _v , dataInfo , dataKey );
#endif // __GNUC__ || __GNUC__ < 4
	};
	Splat( temp , (Real)dx );
	if( fabs(1.-dx)>1e-6 ) Splat( temp->parent , (Real)(1.-dx) );
	return Point< Real , 2 >( rawDepth , weight );
}

template< unsigned int Dim , class Real >
template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs >
Point< Real , 2 > FEMTree< Dim , Real >::_splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >& densityWeights , Real minDepthCutoff , Point< Real , Dim > position , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& dataInfo , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , LocalDepth minDepth , std::function< int ( Point< Real , Dim > ) > &pointDepthFunctor , int dim , Real depthBias )
{
	// Get the depth and weight at position
	Real sampleWeight , sampleDepth;
	FEMTreeNode *temp = _spaceRoot;
	Point< Real , Dim > myCenter;
	Real myWidth;

	int maxDepth = pointDepthFunctor( position );
	{
		int depth = 0;
		for( int d=0 ; d<Dim ; d++ ) myCenter[d] = (Real)0.5;
		myWidth = (Real)1.;
		while( depth<maxDepth && depth<densityWeights.kernelDepth() )
		{
			if( !IsActiveNode< Dim >( temp->children ) ) break;
			int cIndex = FEMTreeNode::ChildIndex( myCenter , position );
			temp = temp->children + cIndex;
			myWidth /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) myCenter[d] += myWidth/2;
				else                  myCenter[d] -= myWidth/2;
			depth++;
		}
		_getSampleDepthAndWeight( densityWeights , temp , position , weightKey , sampleDepth , sampleWeight );
		sampleDepth += depthBias;
	}

	if( sampleDepth<minDepthCutoff ) return Point< Real , 2 >( (Real)-1. , (Real)0. );
	Real rawSampleDepth = sampleDepth;

	if( sampleDepth<minDepth ) sampleDepth = (Real)minDepth;
	if( sampleDepth>maxDepth ) sampleDepth = (Real)maxDepth;
	int topDepth = (int)ceil(sampleDepth);

	double dx = 1.0-(topDepth-sampleDepth);
	if     ( topDepth<=minDepth ) topDepth = minDepth , dx = 1;
	else if( topDepth> maxDepth ) topDepth = maxDepth , dx = 1;

	while( _localDepth( temp )>topDepth ) temp=temp->parent;
	while( _localDepth( temp )<topDepth )
	{
		if( !temp->children ) temp->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		int cIndex = FEMTreeNode::ChildIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth/=2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) myCenter[d] += myWidth/2;
			else                  myCenter[d] -= myWidth/2;
	}

	auto Splat = [&]( FEMTreeNode *node , Real dx )
	{
		double width = 1.0 / ( 1<<_localDepth( temp ) );
		// Scale by:
		//		weight: the area/volume associated with the sample
		//		dx: the fraction of the sample splatted into the current depth
		//		pow( width , -dim ): So that each sample is splatted with a unit volume
		V _v = v * sampleWeight / Real( pow( width , dim ) ) * dx;
		//		V _v = v / Length(v) * dx;
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
		#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
		_splatPointData< CreateNodes , ThreadSafe , V >( zero , nodeAllocator , temp , position , _v , dataInfo , dataKey );
#else // !__GNUC__ || __GNUC__ >=5
		_splatPointData< CreateNodes , ThreadSafe , V ,  DataSigs ... >( zero , nodeAllocator , temp , position , _v , dataInfo , dataKey );
#endif // __GNUC__ || __GNUC__ < 4
	};
	Splat( temp , (Real)dx );
	if( fabs(1.-dx)>1e-6 ) Splat( temp->parent , (Real)(1.-dx) );
	return Point< Real , 2 >( rawSampleDepth , sampleWeight );
}

template< unsigned int Dim , class Real >
template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs >
Point< Real , 2 > FEMTree< Dim , Real >::_multiSplatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >* densityWeights , FEMTreeNode* node , Point< Real , Dim > position , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& dataInfo , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , int dim )
{
	typedef UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... > SupportSizes;
	Real _depth , weight;
	if( densityWeights ) _getSampleDepthAndWeight( *densityWeights , position , weightKey , _depth , weight );
	else _depth=(Real)-1. , weight = (Real)1.;
	V _v = v * weight;

	double values[ Dim ][ SupportSizes::Max() ];
	dataKey.template getNeighbors< CreateNodes , ThreadSafe >( node , nodeAllocator , _nodeInitializer );

	for( FEMTreeNode* _node=node ; _localDepth( _node )>=0 ; _node=_node->parent )
	{
		V __v = _v * (Real)pow( 1<<_localDepth( _node ) , dim );
		Point< Real , Dim > start;
		Real w;
		_startAndWidth( _node , start , w );
		__SetBSplineComponentValues< Real , FEMSignature< DataSigs >::Degree ... >( &position[0] , &start[0] , w , &values[0][0] , SupportSizes::Max() );
		typename FEMTreeNode::template Neighbors< UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... > >& neighbors = dataKey.neighbors[ _localToGlobal( _localDepth( _node ) ) ];
		double scratch[Dim+1];
		scratch[0] = 1.;
		WindowLoop< Dim >::Run
		(
			ZeroUIntPack< Dim >() , UIntPack< BSplineSupportSizes< FEMSignature< DataSigs >::Degree >::SupportSize ... >() ,
			[&]( int d , int i ){ scratch[d+1] = scratch[d] * values[d][i]; } ,
			[&]( FEMTreeNode* node ){ if( IsActiveNode< Dim >( node ) ) Atomic< V >::Add( dataInfo.at( node , zero ) , __v * (Real)scratch[Dim] ) ; } ,
			neighbors.neighbors()
		);
	}
	return Point< Real , 2 >( _depth , weight );
}

template< unsigned int Dim , class Real >
template< unsigned int WeightDegree , class V , unsigned int ... DataSigs >
Real FEMTree< Dim , Real >::_nearestMultiSplatPointData( V zero , const DensityEstimator< WeightDegree >* densityWeights , FEMTreeNode* node , Point< Real , Dim > position , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& dataInfo , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , int dim )
{
	Real _depth , weight;
	if( densityWeights ) _getSampleDepthAndWeight( *densityWeights , position , weightKey , _depth , weight );
	else weight = (Real)1.;
	V _v = v * weight;

	for( FEMTreeNode* _node=node ; _localDepth( _node )>=0 ; _node=_node->parent ) if( IsActiveNode< Dim >( _node ) ) Atomic< V >::Add( dataInfo.at( _node , zero ) , _v * (Real)pow( 1<<_localDepth( _node ) , dim ) );
	return weight;
}
//////////////////////////////////
// MultiThreadedWeightEvaluator //
//////////////////////////////////
template< unsigned int Dim , class Real >
template< unsigned int DensityDegree >
FEMTree< Dim , Real >::MultiThreadedWeightEvaluator< DensityDegree >::MultiThreadedWeightEvaluator( const FEMTree< Dim , Real >* tree , const DensityEstimator< DensityDegree >& density , int threads ) : _density( density ) , _tree( tree )
{
	_threads = std::max< int >( 1 , threads );
	_neighborKeys.resize( _threads );
	for( int t=0 ; t<_neighborKeys.size() ; t++ ) _neighborKeys[t].set( tree->_localToGlobal( density.kernelDepth() ) );
}
template< unsigned int Dim , class Real >
template< unsigned int DensityDegree >
Real FEMTree< Dim , Real >::MultiThreadedWeightEvaluator< DensityDegree >::weight( Point< Real , Dim > p , int thread )
{
	ConstPointSupportKey< IsotropicUIntPack< Dim , DensityDegree > >& nKey = _neighborKeys[thread];
	Real depth , weight;
	_tree->_getSampleDepthAndWeight( _density , p , nKey , depth , weight );
	return weight;
}
