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

// Evaluate the result of splatting along a plane and then evaluating at a point on the plane.
template< int Degree > double GetScaleValue( void )
{
	double centerValues[Degree+1];
	Polynomial< Degree >::BSplineComponentValues( 0.5 , centerValues );
	double scaleValue = 0;
	for( int i=0 ; i<=Degree ; i++ ) scaleValue += centerValues[i] * centerValues[i];
	return 1./ scaleValue;
}
template< class Real >
template< int WeightDegree >
void Octree< Real >::_AddWeightContribution( SparseNodeData< Real , WeightDegree >& densityWeights , TreeOctNode* node , Point3D< Real > position , PointSupportKey< WeightDegree >& weightKey , Real weight )
{
	static const double ScaleValue = GetScaleValue< WeightDegree >();
	double dx[ DIMENSION ][ PointSupportKey< WeightDegree >::Size ];
	typename TreeOctNode::Neighbors< PointSupportKey< WeightDegree >::Size >& neighbors = weightKey.template getNeighbors< true >( node );

	if( densityWeights.indices.size()<TreeNodeData::NodeCount ) densityWeights.resize( TreeNodeData::NodeCount );

	Point3D< Real > start;
	Real w;
	_StartAndWidth( node , start , w );

	for( int dim=0 ; dim<DIMENSION ; dim++ ) Polynomial< WeightDegree >::BSplineComponentValues( ( position[dim]-start[dim] ) / w , dx[dim] );

	weight *= (Real)ScaleValue;

	for( int i=0 ; i<PointSupportKey< WeightDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< WeightDegree >::Size ; j++ )
	{
		double dxdy = dx[0][i] * dx[1][j] * weight;
		TreeOctNode** _neighbors = neighbors.neighbors[i][j];
		for( int k=0 ; k<PointSupportKey< WeightDegree >::Size ; k++ ) if( _neighbors[k] )
		{
			int idx = densityWeights.index( _neighbors[k] );
			if( idx<0 )
			{
				densityWeights.indices[ _neighbors[k]->nodeData.nodeIndex ] = (int)densityWeights.data.size();
				densityWeights.data.push_back( (Real)( dxdy * dx[2][k] ) );
			}
			else densityWeights.data[idx] += Real( dxdy * dx[2][k] );
		}
	}
}

template< class Real >
template< int WeightDegree >
Real Octree< Real >::_GetSamplesPerNode( const SparseNodeData< Real , WeightDegree >& densityWeights , TreeOctNode* node , Point3D< Real > position , PointSupportKey< WeightDegree >& weightKey )
{
	Real weight = 0;
	double dx[ DIMENSION ][ PointSupportKey< WeightDegree >::Size ];
	typename TreeOctNode::Neighbors< PointSupportKey< WeightDegree >::Size >& neighbors = weightKey.template getNeighbors< true >( node );

	Point3D< Real > start;
	Real w;
	_StartAndWidth( node , start , w );

	for( int dim=0 ; dim<DIMENSION ; dim++ ) Polynomial< WeightDegree >::BSplineComponentValues( ( position[dim]-start[dim] ) / w , dx[dim] );

	for( int i=0 ; i<PointSupportKey< WeightDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< WeightDegree >::Size ; j++ )
	{
		double dxdy = dx[0][i] * dx[1][j];
		for( int k=0 ; k<PointSupportKey< WeightDegree >::Size ; k++ ) if( neighbors.neighbors[i][j][k] )
		{
			int idx = densityWeights.index( neighbors.neighbors[i][j][k] );
			if( idx>=0 ) weight += Real( dxdy * dx[2][k] * densityWeights.data[idx] );
		}
	}
	return weight;
}
template< class Real >
template< int WeightDegree >
Real Octree< Real >::_GetSamplesPerNode( const SparseNodeData< Real , WeightDegree >& densityWeights , const TreeOctNode* node , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey ) const
{
	Real weight = 0;
	double dx[ DIMENSION ][ PointSupportKey< WeightDegree >::Size ];
	typename TreeOctNode::ConstNeighbors< PointSupportKey< WeightDegree >::Size >& neighbors = weightKey.getNeighbors( node );

	Point3D< Real > start;
	Real w;
	_StartAndWidth( node , start , w );

	for( int dim=0 ; dim<DIMENSION ; dim++ ) Polynomial< WeightDegree >::BSplineComponentValues( ( position[dim]-start[dim] ) / w , dx[dim] );

	for( int i=0 ; i<PointSupportKey< WeightDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< WeightDegree >::Size ; j++ )
	{
		double dxdy = dx[0][i] * dx[1][j];
		for( int k=0 ; k<PointSupportKey< WeightDegree >::Size ; k++ ) if( neighbors.neighbors[i][j][k] )
		{
			int idx = densityWeights.index( neighbors.neighbors[i][j][k] );
			if( idx>=0 ) weight += Real( dxdy * dx[2][k] * densityWeights.data[idx] );
		}
	}
	return weight;
}
template< class Real >
template< int WeightDegree >
void Octree< Real >::_GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , const TreeOctNode* node , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight ) const
{
	const TreeOctNode* temp = node;
	weight = _GetSamplesPerNode( densityWeights , temp , position , weightKey );
	if( weight>=(Real)1. ) depth = Real( _Depth( temp ) + log( weight ) / log(double(1<<(DIMENSION-1))) );
	else
	{
		Real oldWeight , newWeight;
		oldWeight = newWeight = weight;
		while( newWeight<(Real)1. && temp->parent )
		{
			temp=temp->parent;
			oldWeight = newWeight;
			newWeight = _GetSamplesPerNode( densityWeights , temp , position , weightKey );
		}
		depth = Real( _Depth( temp ) + log( newWeight ) / log( newWeight / oldWeight ) );
	}
	weight = Real( pow( double(1<<(DIMENSION-1)) , -double(depth) ) );
}
template< class Real >
template< int WeightDegree >
void Octree< Real >::_GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , TreeOctNode* node , Point3D< Real > position , PointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight )
{
	TreeOctNode* temp = node;
	weight = _GetSamplesPerNode( densityWeights , temp , position , weightKey );
	if( weight>=(Real)1. ) depth = Real( _Depth( temp ) + log( weight ) / log(double(1<<(DIMENSION-1))) );
	else
	{
		Real oldWeight , newWeight;
		oldWeight = newWeight = weight;
		while( newWeight<(Real)1. && temp->parent )
		{
			temp=temp->parent;
			oldWeight = newWeight;
			newWeight = _GetSamplesPerNode( densityWeights , temp , position, weightKey );
		}
		depth = Real( _Depth( temp ) + log( newWeight ) / log( newWeight / oldWeight ) );
	}
	weight = Real( pow( double(1<<(DIMENSION-1)) , -double(depth) ) );
}
template< class Real >
template< int WeightDegree >
void Octree< Real >::_GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight )
{
	TreeOctNode* temp;
	Point3D< Real > myCenter( (Real)0.5 , (Real)0.5 , (Real)0.5 );
	Real myWidth = Real( 1. );

	// Get the finest node with depth less than or equal to the splat depth that contains the point
	temp = _spaceRoot;
	while( _Depth( temp )<_splatDepth )
	{
		if( !temp->children ) break;// fprintf( stderr , "[ERROR] Octree::GetSampleDepthAndWeight\n" ) , exit( 0 );
		int cIndex = TreeOctNode::CornerIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth /= 2;
		if( cIndex&1 ) myCenter[0] += myWidth/2;
		else		   myCenter[0] -= myWidth/2;
		if( cIndex&2 ) myCenter[1] += myWidth/2;
		else		   myCenter[1] -= myWidth/2;
		if( cIndex&4 ) myCenter[2] += myWidth/2;
		else		   myCenter[2] -= myWidth/2;
	}
	return _GetSampleDepthAndWeight( densityWeights , temp , position , weightKey , depth , weight );
}
template< class Real >
template< int WeightDegree >
void Octree< Real >::_GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > position , PointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight )
{
	TreeOctNode* temp;
	Point3D< Real > myCenter( (Real)0.5 , (Real)0.5 , (Real)0.5 );
	Real myWidth = Real( 1. );

	// Get the finest node with depth less than or equal to the splat depth that contains the point
	temp = _spaceRoot;
	while( _Depth( temp )<_splatDepth )
	{
		if( !temp->children ) break;// fprintf( stderr , "[ERROR] Octree::GetSampleDepthAndWeight\n" ) , exit( 0 );
		int cIndex = TreeOctNode::CornerIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth /= 2;
		if( cIndex&1 ) myCenter[0] += myWidth/2;
		else		   myCenter[0] -= myWidth/2;
		if( cIndex&2 ) myCenter[1] += myWidth/2;
		else		   myCenter[1] -= myWidth/2;
		if( cIndex&4 ) myCenter[2] += myWidth/2;
		else		   myCenter[2] -= myWidth/2;
	}
	return _GetSampleDepthAndWeight( densityWeights , temp , position , weightKey , depth , weight );
}


template< class Real >
template< int DataDegree , class V >
void Octree< Real >::_SplatPointData( TreeOctNode* node , Point3D< Real > position , V v , SparseNodeData< V , DataDegree >& dataInfo , PointSupportKey< DataDegree >& dataKey )
{
	double dx[ DIMENSION ][ PointSupportKey< DataDegree >::Size ];
	typename TreeOctNode::Neighbors< PointSupportKey< DataDegree >::Size >& neighbors = dataKey.template getNeighbors< true >( node );

	Point3D< Real > start;
	Real w;
	_StartAndWidth( node , start , w );

	for( int dd=0 ; dd<DIMENSION ; dd++ ) Polynomial< DataDegree >::BSplineComponentValues( ( position[dd]-start[dd] ) / w , dx[dd] );

	for( int i=0 ; i<PointSupportKey< DataDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< DataDegree >::Size ; j++ )
	{
		double dxdy = dx[0][i] * dx[1][j];
		for( int k=0 ; k<PointSupportKey< DataDegree >::Size ; k++ )
			if( neighbors.neighbors[i][j][k] )
			{
				TreeOctNode* _node = neighbors.neighbors[i][j][k];

				double dxdydz = dxdy * dx[2][k];
				if( (int)dataInfo.indices.size()<TreeNodeData::NodeCount ) dataInfo.indices.resize( TreeNodeData::NodeCount , -1 );
				int idx = dataInfo.index( _node );
				if( idx<0 )
				{
					dataInfo.indices[ _node->nodeData.nodeIndex ] = (int)dataInfo.data.size();
					dataInfo.data.push_back( v * Real(dxdydz) );
				}
				else dataInfo.data[idx] += v * Real( dxdydz );
			}
	}
}
template< class Real >
template< int WeightDegree , int DataDegree , class V >
Real Octree< Real >::_SplatPointData( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > position , V v , SparseNodeData< V , DataDegree >& dataInfo , PointSupportKey< WeightDegree >& weightKey , PointSupportKey< DataDegree >& dataKey , int minDepth , int maxDepth , int dim )
{
	double dx;
	V _v;
	TreeOctNode* temp;
	int cnt=0;
	double width;
	Point3D< Real > myCenter( (Real)0.5 , (Real)0.5 , (Real)0.5 );
	Real myWidth = (Real)1.;

	temp = _spaceRoot;
	while( _Depth( temp )<_splatDepth )
	{
		if( !temp->children ) fprintf( stderr , "[ERROR] Octree::SplatPointData\n" ) , exit( 0 );
		int cIndex = TreeOctNode::CornerIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth /= 2;
		if( cIndex&1 ) myCenter[0] += myWidth/2;
		else		   myCenter[0] -= myWidth/2;
		if( cIndex&2 ) myCenter[1] += myWidth/2;
		else 	  	   myCenter[1] -= myWidth/2;
		if( cIndex&4 ) myCenter[2] += myWidth/2;
		else 		   myCenter[2] -= myWidth/2;
	}
	Real weight , depth;
	_GetSampleDepthAndWeight( densityWeights , temp , position , weightKey , depth , weight );

	if( depth<minDepth ) depth = Real(minDepth);
	if( depth>maxDepth ) depth = Real(maxDepth);
	int topDepth = int(ceil(depth));

	dx = 1.0-(topDepth-depth);
	if     ( topDepth<=minDepth ) topDepth = minDepth , dx = 1;
	else if( topDepth> maxDepth ) topDepth = maxDepth , dx = 1;

	while( _Depth( temp )>topDepth ) temp=temp->parent;
	while( _Depth( temp )<topDepth )
	{
		if( !temp->children ) temp->initChildren();
		int cIndex = TreeOctNode::CornerIndex( myCenter , position );
		temp = &temp->children[cIndex];
		myWidth/=2;
		if( cIndex&1 ) myCenter[0] += myWidth/2;
		else		   myCenter[0] -= myWidth/2;
		if( cIndex&2 ) myCenter[1] += myWidth/2;
		else		   myCenter[1] -= myWidth/2;
		if( cIndex&4 ) myCenter[2] += myWidth/2;
		else		   myCenter[2] -= myWidth/2;
	}
	width = 1.0 / ( ( 1<<( _Depth( temp ) ) ) );
	_v = v * weight / Real( pow( width , dim ) ) * Real( dx );
	_SplatPointData( temp , position , _v , dataInfo , dataKey );
	if( fabs(1.0-dx) > EPSILON )
	{
		dx = Real(1.0-dx);
		temp = temp->parent;
		width = 1.0 / ( ( 1<<( _Depth( temp ) ) ) );

		_v = v * weight / Real( pow( width , dim ) ) * Real( dx );
		_SplatPointData( temp , position , _v , dataInfo , dataKey );
	}
	return weight;
}
template< class Real >
template< int WeightDegree , int DataDegree , class V >
void Octree< Real >::_MultiSplatPointData( const SparseNodeData< Real , WeightDegree >* densityWeights , Point3D< Real > position , V v , SparseNodeData< V , DataDegree >& dataInfo , PointSupportKey< WeightDegree >& weightKey , PointSupportKey< DataDegree >& dataKey , int maxDepth , int dim )
{
	Real _depth , weight;
	if( densityWeights ) _GetSampleDepthAndWeight( *densityWeights , position , weightKey , _depth , weight );
	else weight = (Real)1. , _depth = (Real)maxDepth;
	int depth = std::min< int >( maxDepth , (int)ceil( _depth ) );
	V _v = v * weight;

	Point3D< Real > myCenter( (Real)0.5 , (Real)0.5 , (Real)0.5 );
	Real myWidth = (Real)1.;

	TreeOctNode* temp = _spaceRoot;
	while( _Depth( temp )<=depth )
	{
		_SplatPointData( temp , position , _v * Real( pow( 1<<_Depth( temp ) , dim ) ) , dataInfo , dataKey );
		if( _Depth( temp )<depth )
		{
			if( !temp->children ) temp->initChildren();
			int cIndex = TreeOctNode::CornerIndex( myCenter , position );
			temp = &temp->children[cIndex];
			myWidth /= 2;
			if( cIndex&1 ) myCenter[0] += myWidth/2;
			else		   myCenter[0] -= myWidth/2;
			if( cIndex&2 ) myCenter[1] += myWidth/2;
			else 	  	   myCenter[1] -= myWidth/2;
			if( cIndex&4 ) myCenter[2] += myWidth/2;
			else 		   myCenter[2] -= myWidth/2;
		}
		else break;
	}
}
template< class Real >
template< class V , int DataDegree >
V Octree< Real >::_Evaluate( const DenseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData , const ConstPointSupportKey< DataDegree >& neighborKey ) const
{
	V value = V(0);

	for( int d=0 ; d<=neighborKey.depth() ; d++ ) for( int i=0 ; i<PointSupportKey< DataDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< DataDegree >::Size ; j++ ) for( int k=0 ; k<PointSupportKey< DataDegree >::Size ; k++ )
	{
		const TreeOctNode* n = neighborKey.neighbors[d].neighbors[i][j][k];
		if( _IsValidNode< DataDegree >( n ) )
		{
			int fIdx[3];
			FunctionIndex< DataDegree >( n , fIdx );
			value +=
				(
				coefficients[ n->nodeData.nodeIndex ] *
				(Real)
				(
					bsData.baseBSplines[ fIdx[0] ][PointSupportKey< DataDegree >::Size-1-i]( p[0] ) *
					bsData.baseBSplines[ fIdx[1] ][PointSupportKey< DataDegree >::Size-1-j]( p[1] ) *
					bsData.baseBSplines[ fIdx[2] ][PointSupportKey< DataDegree >::Size-1-k]( p[2] )
				)
			);
		}
	}

	return value;
}
template< class Real >
template< class V , int DataDegree >
V Octree< Real >::_Evaluate( const SparseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData , const ConstPointSupportKey< DataDegree >& dataKey ) const
{
	V value = V(0);

	for( int d=0 ; d<=dataKey.depth() ; d++ )
	{
		double dx[ DIMENSION ][ PointSupportKey< DataDegree >::Size ];
		memset( dx , 0 , sizeof( double ) * DIMENSION * PointSupportKey< DataDegree >::Size );
		{
			const TreeOctNode* n = dataKey.neighbors[d].neighbors[ PointSupportKey< DataDegree >::LeftRadius ][ PointSupportKey< DataDegree >::LeftRadius ][ PointSupportKey< DataDegree >::LeftRadius ];
			if( !n ) fprintf( stderr , "[ERROR] Point is not centered on a node\n" ) , exit( 0 );
			int fIdx[3];
			FunctionIndex< DataDegree >( n , fIdx );
			int fStart , fEnd;
			BSplineData< DataDegree >::FunctionSpan( d-1 , fStart , fEnd );
			for( int dd=0 ; dd<DIMENSION ; dd++ ) for( int i=-PointSupportKey< DataDegree >::LeftRadius ; i<=PointSupportKey< DataDegree >::RightRadius ; i++ )
				if( fIdx[dd]+i>=fStart && fIdx[dd]+i<fEnd ) dx[dd][i] = bsData.baseBSplines[ fIdx[dd]+i ][ -i+PointSupportKey< DataDegree >::RightRadius ]( p[dd] );
		}
		for( int i=0 ; i<PointSupportKey< DataDegree >::Size ; i++ ) for( int j=0 ; j<PointSupportKey< DataDegree >::Size ; j++ ) for( int k=0 ; k<PointSupportKey< DataDegree >::Size ; k++ )
		{
			const TreeOctNode* n = dataKey.neighbors[d].neighbors[i][j][k];
			if( _IsValidNode< DataDegree >( n ) )
			{
				int idx = coefficients.index( n );
				if( idx>=0 ) value +=  coefficients.data[ idx ] * (Real) ( dx[0][i] * dx[1][j] * dx[2][k] );
			}
		}
	}

	return value;
}

template< class Real >
template< class V , int DataDegree >
V Octree< Real >::Evaluate( const DenseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData ) const
{
	static const int SupportSize = BSplineEvaluationData< DataDegree >::SupportSize;
	static const int  LeftSupportRadius = -BSplineEvaluationData< DataDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< DataDegree >::SupportEnd;
	V value = V(0);

	// [WARNING] This is required because the B-Spline components are not continuous at the domain boundaries
	// so we need to nudge things inward a tiny bit.
	for( int dd=0 ; dd<3 ; dd++ )
		if     ( p[dd]==0 ) p[dd] = 0.+1e-6;
		else if( p[dd]==1 ) p[dd] = 1.-1e-6;

	const TreeOctNode* n = _tree.nextNode();
	while( n )
	{
		Point3D< Real > s;
		Real w;
		_StartAndWidth( n , s , w );
		double left = (LeftSupportRadius+0.)*w , right = (RightSupportRadius+1.)*w;
		if(	p[0]<=s[0]-left || p[0]>=s[0]+right || p[1]<=s[1]-left || p[1]>=s[1]+right || p[2]<=s[2]-left || p[2]>=s[2]+right )
		{
			n = _tree.nextBranch( n );
			continue;
		}
		if( _IsValidNode< DataDegree >( n ) )
		{
			int d , fIdx[3] , pIdx[3];
			_DepthAndOffset( n , d , fIdx );
			for( int dd=0 ; dd<3 ; dd++ ) pIdx[dd] = std::max< int >( 0 , std::min< int >( SupportSize-1 , LeftSupportRadius + (int)floor( ( p[dd]-s[dd] ) / w ) ) );
			value +=
				coefficients[ n->nodeData.nodeIndex ] *
				(Real)
				(
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , fIdx[0] ) ][ pIdx[0] ]( p[0] ) *
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , fIdx[1] ) ][ pIdx[1] ]( p[1] ) *
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , fIdx[2] ) ][ pIdx[2] ]( p[2] )
				);
		}
		n = _tree.nextNode( n );
	}
	return value;
}
template< class Real >
template< class V , int DataDegree >
V Octree< Real >::Evaluate( const SparseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData ) const
{
	V value = V(0);

	const TreeOctNode* n = _tree.nextNode();
	while( n )
	{
		Point3D< Real > s;
		Real w;
		_StartAndWidth( n , s , w );
		if( !_IsValidNode< DataDegree >( n ) ||
			p[0]<s[0]+BSplineData< DataDegree >::SupportStart*w || p[0]>s[0]+(BSplineData< DataDegree >::SupportEnd+1.0)*w ||
			p[1]<s[1]+BSplineData< DataDegree >::SupportStart*w || p[1]>s[1]+(BSplineData< DataDegree >::SupportEnd+1.0)*w ||
			p[2]<s[2]+BSplineData< DataDegree >::SupportStart*w || p[2]>s[2]+(BSplineData< DataDegree >::SupportEnd+1.0)*w )
		{
			n = _tree.nextBranch( n );
			continue;
		}

		int idx = coefficients.index( n );
		if( idx>=0 )
		{
			int d , off[3] , pIdx[3];
			_DepthAndOffset( n , d , off );
			for( int dd=0 ; dd<3 ; dd++ ) pIdx[dd] =  std::max< int >( 0 , std::min< int >( BSplineData< DataDegree >::SupportSize-1 , -BSplineData< DataDegree >::SupportStart + (int)floor( ( p[dd]-s[dd] ) / w ) ) );
			value +=
				coefficients.data[idx] *
				(Real)
				(
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , off[0] ) ][ pIdx[0] ]( p[0] ) *
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , off[1] ) ][ pIdx[1] ]( p[1] ) *
					bsData.baseBSplines[ BSplineData< DataDegree >::FunctionIndex( d , off[2] ) ][ pIdx[2] ]( p[2] )
				);
		}
		n = _tree.nextNode( n );
	}
	return value;
}
template< class Real >
template< class V , int DataDegree >
Pointer( V ) Octree< Real >::Evaluate( const DenseNodeData< V , DataDegree >& coefficients , int& res , Real isoValue , int depth , bool primal )
{
	int dim;
	if( depth>=0 ) depth++;
	int maxDepth = _tree.maxDepth();
	if( depth<=0 || depth>maxDepth ) depth = maxDepth;

	BSplineData< DataDegree > fData;
	fData.set( depth , _dirichlet );

	// Initialize the coefficients at the coarsest level
	Pointer( V ) _coefficients = NullPointer( V );
	{
		int d = _minDepth;
		dim = _Dimension< DataDegree >( d );
		_coefficients = NewPointer< V >( dim * dim * dim );
		memset( _coefficients , 0 , sizeof( V ) * dim  * dim * dim );
#pragma omp parallel for num_threads( threads )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ ) if( _IsValidNode< DataDegree >( _sNodes.treeNodes[i] ) )
		{
			int _d , _off[3];
			_sNodes.treeNodes[i]->depthAndOffset( _d , _off );
			_coefficients[ _off[0] + _off[1]*dim + _off[2]*dim*dim ] = coefficients[i];
		}
	}

	// Up-sample and add in the existing coefficients
	for( int d=_minDepth+1 ; d<=depth ; d++ )
	{
		dim = _Dimension< DataDegree >( d );
		Pointer( V ) __coefficients = NewPointer< V >( dim * dim *dim );
		memset( __coefficients , 0 , sizeof( V ) * dim  * dim * dim );
#pragma omp parallel for num_threads( threads )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ ) if( _IsValidNode< DataDegree >( _sNodes.treeNodes[i] ) )
		{
			int _d , _off[3];
			_sNodes.treeNodes[i]->depthAndOffset( _d , _off );
			__coefficients[ _off[0] + _off[1]*dim + _off[2]*dim*dim ] = coefficients[i];
		}
		_UpSample< V , DataDegree >( d , ( ConstPointer(V) )_coefficients , __coefficients , _dirichlet , threads );
		DeletePointer( _coefficients );
		_coefficients = __coefficients;
	}

	res = 1<<(depth-1);
	if( primal ) res++;
	Pointer( V ) values = NewPointer< V >( res*res*res );
	memset( values , 0 , sizeof(V)*res*res*res );

	if( primal )
	{
		// Evaluate at the cell corners
		typename BSplineEvaluationData< DataDegree >::CornerEvaluator::Evaluator evaluator;
		BSplineEvaluationData< DataDegree >::SetCornerEvaluator( evaluator , depth-1 , _dirichlet );
#pragma omp parallel for num_threads( threads )
		for( int k=0 ; k<res ; k++ ) for( int j=0 ; j<res ; j++ ) for( int i=0 ; i<res ; i++ )
		{
			V value = values[ i + j*res + k*res*res ];
			for( int kk=-BSplineEvaluationData< DataDegree >::CornerEnd ; kk<=-BSplineEvaluationData< DataDegree >::CornerStart ; kk++ ) if( k+kk>=0 && k+kk<dim )
				for( int jj=-BSplineEvaluationData< DataDegree >::CornerEnd ; jj<=-BSplineEvaluationData< DataDegree >::CornerStart ; jj++ ) if( j+jj>=0 && j+jj<dim )
				{
					double weight = evaluator.value( k+kk , k , false ) * evaluator.value( j+jj , j , false );
					int idx = (j+jj)*dim + (k+kk)*dim*dim;
					for( int ii=-BSplineEvaluationData< DataDegree >::CornerEnd ; ii<=-BSplineEvaluationData< DataDegree >::CornerStart ; ii++ ) if( i+ii>=0 && i+ii<dim )
						value += _coefficients[ i+ii + idx ] * Real( weight * evaluator.value( i+ii , i , false ) );
				}
			values[ i + j*res + k*res*res ] = value;
		}
	}
	else
	{
		// Evaluate at the cell centers
		typename BSplineEvaluationData< DataDegree >::CenterEvaluator::Evaluator evaluator;
		BSplineEvaluationData< DataDegree >::SetCenterEvaluator( evaluator , depth-1 , _dirichlet );
#pragma omp parallel for num_threads( threads )
		for( int k=0 ; k<res ; k++ ) for( int j=0 ; j<res ; j++ ) for( int i=0 ; i<res ; i++ )
		{
			V& value = values[ i + j*res + k*res*res ];
			for( int kk=-BSplineEvaluationData< DataDegree >::SupportEnd ; kk<=-BSplineEvaluationData< DataDegree >::SupportStart ; kk++ ) if( k+kk>=0 && k+kk<dim )
				for( int jj=-BSplineEvaluationData< DataDegree >::SupportEnd ; jj<=-BSplineEvaluationData< DataDegree >::SupportStart ; jj++ ) if( j+jj>=0 && j+jj<dim )
				{
					double weight = evaluator.value( k+kk , k , false ) * evaluator.value( j+jj , j , false );
					int idx = (j+jj)*dim + (k+kk)*dim*dim;
					for( int ii=-BSplineEvaluationData< DataDegree >::SupportEnd ; ii<=-BSplineEvaluationData< DataDegree >::SupportStart ; ii++ ) if( i+ii>=0 && i+ii<dim )
						value += _coefficients[ i+ii + idx ] * Real( weight * evaluator.value( i+ii , i , false ) );
				}
		}
	}
	MemoryUsage();
	DeletePointer( _coefficients );
	for( int i=0 ; i<res*res*res ; i++ ) values[i] -= isoValue;

	return values;
}
