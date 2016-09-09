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

#include "PointStream.h"

#define ITERATION_POWER 1.0/3
#define MEMORY_ALLOCATOR_BLOCK_SIZE 1<<12


const double MATRIX_ENTRY_EPSILON = 0;
const double EPSILON              = 1e-6;
const double ROUND_EPS            = 1e-5;



//////////////////
// TreeNodeData //
//////////////////
size_t TreeNodeData::NodeCount = 0;
TreeNodeData::TreeNodeData( void ){ nodeIndex = (int)NodeCount ; NodeCount++; }
TreeNodeData::~TreeNodeData( void ) { }


////////////
// Octree //
////////////
template< class Real > double Octree< Real >::maxMemoryUsage=0;

template< class Real >
double Octree< Real >::MemoryUsage( void )
{
	double mem = double( MemoryInfo::Usage() ) / (1<<20);
	if( mem>maxMemoryUsage ) maxMemoryUsage=mem;
	return mem;
}

template< class Real >
Octree< Real >::Octree( void )
{
	threads = 1;
	_constrainValues = false;
	_multigridDegree = 0;
}
template< class Real >
template< int FEMDegree >
void Octree< Real >::FunctionIndex( const TreeOctNode* node , int idx[3] )
{
	int d;
	node->depthAndOffset( d , idx );
	for( int dd=0 ; dd<DIMENSION ; dd++ ) idx[dd] = BSplineData< FEMDegree >::FunctionIndex( d-1 , idx[dd] );
}

template< class Real > bool Octree< Real >::_InBounds( Point3D< Real > p ) const { return p[0]>=Real(0.) && p[0]<=Real(1.0) && p[1]>=Real(0.) && p[1]<=Real(1.0) && p[2]>=Real(0.) && p[2]<=Real(1.0); }
template< class Real >
template< int FEMDegree >
bool Octree< Real >::IsValidNode( const TreeOctNode* node , bool dirichlet  )
{
	if( !node || node->depth()<1 ) return false;
	int d , off[3];
	node->depthAndOffset( d , off );
	int dim = BSplineData< FEMDegree >::Dimension( d-1 );
	if( FEMDegree&1 && dirichlet ) return !( off[0]<=0 || off[1]<=0 || off[2]<=0 || off[0]>=dim-1 || off[1]>=dim-1 || off[2]>=dim-1 );
	else                           return !( off[0]< 0 || off[1]< 0 || off[2]< 0 || off[0]> dim-1 || off[1]> dim-1 || off[2]> dim-1 );
}
template< class Real >
void Octree< Real >::_SetFullDepth( TreeOctNode* node , int depth )
{
	if( node->depth()==0 || _Depth( node )<depth )
	{
		if( !node->children ) node->initChildren();
		for( int c=0 ; c<Cube::CORNERS ; c++ ) _SetFullDepth( node->children+c , depth );
	}
}
template< class Real >
void Octree< Real >::_setFullDepth( int depth )
{
	if( !_tree.children ) _tree.initChildren();
	for( int c=0 ; c<Cube::CORNERS ; c++ ) _SetFullDepth( _tree.children+c , depth );
}
template< class Real >
template< class PointReal , int NormalDegree , int WeightDegree , int DataDegree , class Data , class _Data >
int Octree< Real >::SetTree( OrientedPointStream< PointReal >* pointStream , int minDepth , int maxDepth , int fullDepth ,
							int splatDepth , Real samplesPerNode , Real scaleFactor ,
							bool useConfidence , bool useNormalWeights , Real constraintWeight , int adaptiveExponent ,
							SparseNodeData< Real , WeightDegree >& densityWeights , SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo , SparseNodeData< Real , NormalDegree >& nodeWeights ,
							SparseNodeData< ProjectiveData< _Data > , DataDegree >* dataValues ,
							XForm4x4< Real >& xForm , bool dirichlet , bool makeComplete )
{
	OrientedPointStreamWithData< PointReal , Data >* pointStreamWithData = ( OrientedPointStreamWithData< PointReal , Data >* )pointStream;
	_tree.initChildren() , _spaceRoot = _tree.children;
	splatDepth = std::max< int >( 0 , std::min< int >( splatDepth , maxDepth ) );

	_dirichlet = dirichlet;
	_constrainValues = (constraintWeight>0);

	XForm3x3< Real > xFormN;
	for( int i=0 ; i<3 ; i++ ) for( int j=0 ; j<3 ; j++ ) xFormN(i,j) = xForm(i,j);
	xFormN = xFormN.transpose().inverse();
	minDepth = std::max< int >( 0 , std::min< int >( minDepth , maxDepth ) ); // 0<=minDepth <= maxDepth
	fullDepth = std::max< int >( minDepth , std::min< int >( fullDepth , maxDepth ) );	// minDepth <= fullDepth <= maxDepth

#if 0
	// For Neumann constraints, the function at depth 0 is constant so the system matrix is zero if there is no screening.
	if( !_dirichlet && !_constrainValues ) minDepth = std::max< int >( minDepth , 1 );
#endif
	minDepth++ , maxDepth++;

	_minDepth = minDepth;
	_fullDepth = fullDepth;
	_splatDepth = splatDepth;
	double pointWeightSum = 0;
	int i , cnt=0;

	PointSupportKey< WeightDegree > weightKey;
	PointSupportKey< DataDegree > dataKey;
	PointSupportKey< NormalDegree > normalKey;
	weightKey.set( maxDepth ) , dataKey.set( maxDepth ) , normalKey.set( maxDepth );

	_setFullDepth( _fullDepth );

	// Read through once to get the center and scale
	{
		Point3D< Real > min , max;
		double t = Time();
		Point3D< Real > p;
		OrientedPoint3D< PointReal > _p;
		while( pointStream->nextPoint( _p ) )
		{
			p = xForm * Point3D< Real >(_p.p);
			for( i=0 ; i<DIMENSION ; i++ )
			{
				if( !cnt || p[i]<min[i] ) min[i] = p[i];
				if( !cnt || p[i]>max[i] ) max[i] = p[i];
			}
			cnt++;
		}

		_scale = std::max< Real >( max[0]-min[0] , std::max< Real >( max[1]-min[1] , max[2]-min[2] ) );
		_center = ( max+min ) /2;
	}

	_scale *= scaleFactor;
	for( i=0 ; i<DIMENSION ; i++ ) _center[i] -= _scale / 2;

	// Update the transformation
	{
		XForm4x4< Real > trans = XForm4x4< Real >::Identity() , scale = XForm4x4< Real >::Identity();
		for( int i=0 ; i<3 ; i++ ) scale(i,i) = (Real)(1./_scale ) , trans(3,i) = -_center[i];
		xForm = scale * trans * xForm;
	}

	{
		double t = Time();
		cnt = 0;
		pointStream->reset();
		Point3D< Real > p , n;
		OrientedPoint3D< PointReal > _p;
		while( pointStream->nextPoint( _p ) )
		{
			p = xForm * Point3D< Real >(_p.p) , n = xFormN * Point3D< Real >(_p.n);
			if( !_InBounds(p) ) continue;
			Point3D< Real > myCenter = Point3D< Real >( Real(0.5) , Real(0.5) , Real(0.5) );
			Real myWidth = Real(1.0);
			Real weight=Real( 1. );
			if( useConfidence ) weight = Real( Length(n) );
			if( samplesPerNode>0 ) weight /= (Real)samplesPerNode;
			TreeOctNode* temp = _spaceRoot;
			int d=0;
			while( d<splatDepth )
			{
				_AddWeightContribution( densityWeights , temp , p , weightKey , weight );
				if( !temp->children ) temp->initChildren();
				int cIndex=TreeOctNode::CornerIndex( myCenter , p );
				temp = temp->children + cIndex;
				myWidth/=2;
				if( cIndex&1 ) myCenter[0] += myWidth/2;
				else           myCenter[0] -= myWidth/2;
				if( cIndex&2 ) myCenter[1] += myWidth/2;
				else           myCenter[1] -= myWidth/2;
				if( cIndex&4 ) myCenter[2] += myWidth/2;
				else           myCenter[2] -= myWidth/2;
				d++;
			}
			_AddWeightContribution( densityWeights , temp , p , weightKey , weight );
			cnt++;
		}
	}

	std::vector< PointData< Real > >& points = pointInfo.data;

	cnt = 0;
	pointStream->reset();
	Point3D< Real > p , n;
	OrientedPoint3D< PointReal > _p;
	Data _d;
	while( ( dataValues ? pointStreamWithData->nextPoint( _p , _d ) : pointStream->nextPoint( _p ) ) )
	{
		p = xForm * Point3D< Real >(_p.p) , n = xFormN * Point3D< Real >(_p.n);
		n *= Real(-1.);
		if( !_InBounds(p) ) continue;
		Real normalLength = Real( Length( n ) );
		if(std::isnan( normalLength ) || !std::isfinite( normalLength ) || normalLength<=EPSILON ) continue;
		if( !useConfidence ) n /= normalLength;

		Real pointWeight = Real(1.f);
		if( samplesPerNode>0 )
		{
			if( dataValues ) _MultiSplatPointData( &densityWeights , p , ProjectiveData< _Data >( _Data( _d ) , (Real)1. ) , *dataValues , weightKey , dataKey , maxDepth-1 , 2 );
			pointWeight = _SplatPointData( densityWeights , p , n , normalInfo , weightKey , normalKey , _minDepth-1 , maxDepth-1 , 3 );
		}
		else
		{
			if( dataValues ) _MultiSplatPointData( ( SparseNodeData< Real , WeightDegree >* )NULL , p , ProjectiveData< _Data >( _Data( _d ) , (Real)1. ) , *dataValues , weightKey , dataKey , maxDepth-1 , 2 );

			Point3D< Real > myCenter = Point3D< Real >( Real(0.5) , Real(0.5) , Real(0.5) );
			Real myWidth = Real(1.0);
			TreeOctNode* temp = _spaceRoot;
			int d=0;
			if( splatDepth )
			{
				while( d<splatDepth )
				{
					int cIndex=TreeOctNode::CornerIndex(myCenter,p);
					temp = &temp->children[cIndex];
					myWidth /= 2;
					if(cIndex&1) myCenter[0] += myWidth/2;
					else		 myCenter[0] -= myWidth/2;
					if(cIndex&2) myCenter[1] += myWidth/2;
					else		 myCenter[1] -= myWidth/2;
					if(cIndex&4) myCenter[2] += myWidth/2;
					else		 myCenter[2] -= myWidth/2;
					d++;
				}
				pointWeight = (Real)1.0/_GetSamplesPerNode( densityWeights , temp , p , weightKey );
			}
			for( i=0 ; i<DIMENSION ; i++ ) n[i] *= pointWeight;
			// [WARNING] mixing depth definitions
			while( d<maxDepth-1 )
			{
				if( !temp->children ) temp->initChildren();
				int cIndex=TreeOctNode::CornerIndex( myCenter , p );
				temp=&temp->children[cIndex];
				myWidth/=2;
				if(cIndex&1) myCenter[0] += myWidth/2;
				else		 myCenter[0] -= myWidth/2;
				if(cIndex&2) myCenter[1] += myWidth/2;
				else		 myCenter[1] -= myWidth/2;
				if(cIndex&4) myCenter[2] += myWidth/2;
				else		 myCenter[2] -= myWidth/2;
				d++;
			}
			_SplatPointData( temp , p , n , normalInfo , normalKey );
		}
		pointWeightSum += pointWeight;
		if( _constrainValues )
		{
			Real pointScreeningWeight = useNormalWeights ? Real( normalLength ) : Real(1.f);
			TreeOctNode* temp = _spaceRoot;
			Point3D< Real > myCenter = Point3D< Real >( Real(0.5) , Real(0.5) , Real(0.5) );
			Real myWidth = Real(1.0);
			while( 1 )
			{
				if( (int)pointInfo.indices.size()<TreeNodeData::NodeCount ) pointInfo.indices.resize( TreeNodeData::NodeCount , -1 );
				int idx = pointInfo.index( temp );

				if( idx==-1 )
				{
					idx = (int)points.size();
					points.push_back( PointData< Real >( p*pointScreeningWeight , pointScreeningWeight ) );
					pointInfo.indices[ temp->nodeData.nodeIndex ] = idx;
				}
				else
				{
					points[idx].position += p*pointScreeningWeight;
					points[idx].weight += pointScreeningWeight;
				}

				int cIndex = TreeOctNode::CornerIndex( myCenter , p );
				if( !temp->children ) break;
				temp = &temp->children[cIndex];
				myWidth /= 2;
				if( cIndex&1 ) myCenter[0] += myWidth/2;
				else		   myCenter[0] -= myWidth/2;
				if( cIndex&2 ) myCenter[1] += myWidth/2;
				else		   myCenter[1] -= myWidth/2;
				if( cIndex&4 ) myCenter[2] += myWidth/2;
				else		   myCenter[2] -= myWidth/2;
			}
		}
		cnt++;
	}

	constraintWeight *= Real( pointWeightSum );
	constraintWeight /= cnt;

	MemoryUsage( );
	if( _constrainValues )
		// Set the average position and scale the weights
		for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode(node) )
			if( pointInfo.index( node )!=-1 )
			{
				int idx = pointInfo.index( node );
				points[idx].position /= points[idx].weight;
				int e = _Depth( node ) * adaptiveExponent - ( maxDepth - 1 ) * (adaptiveExponent-1);
				if( e<0 ) points[idx].weight /= Real( 1<<(-e) );
				else      points[idx].weight *= Real( 1<<  e  );
				points[idx].weight *= Real( constraintWeight );
			}
#if FORCE_NEUMANN_FIELD
// #pragma message( "[WARNING] This is likely wrong as it only forces the normal component of the coefficient to be zero, not the actual vector-field" )
	if( !_dirichlet )
		for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode( node ) )
		{
			int d , off[3] , res;
			node->depthAndOffset( d , off );
			res = 1<<d;
			int idx = normalInfo.index( node );
			if( idx<0 ) continue;
			Point3D< Real >& normal = normalInfo.data[ idx ];
			for( int d=0 ; d<3 ; d++ ) if( off[d]==0 || off[d]==res-1 ) normal[d] = 0;
		}
#endif // FORCE_NEUMANN_FIELD
	nodeWeights.resize( TreeNodeData::NodeCount );
	// Set the point weights for evaluating the iso-value
	for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode(node) )
	{
		int nIdx = normalInfo.index( node );
		if( nIdx>=0 )
		{
			Real l = Real( Length( normalInfo.data[ nIdx ] ) );
			if( l )
			{
				int nIdx = nodeWeights.index( node );
				if( nIdx<0 )
				{
					nodeWeights.indices[ node->nodeData.nodeIndex ] = (int)nodeWeights.data.size();
					nodeWeights.data.push_back( l );
				}
				else nodeWeights.data[ nIdx ] = l;
			}
		}
	}
	MemoryUsage();
	if( makeComplete ) _MakeComplete( );
	else _ClipTree< NormalDegree >( normalInfo );
	_maxDepth = _tree.maxDepth();
	return cnt;
}

template< class Real >
void Octree< Real >::_SetValidityFlags( void )
{
	for( int i=0 ; i<_sNodes.end( _sNodes.levels()-1 ) ; i++ )
	{
		_sNodes.treeNodes[i]->nodeData.flags = 0;
		if( IsValidNode< 0 >( _sNodes.treeNodes[i] , _dirichlet ) ) _sNodes.treeNodes[i]->nodeData.flags |= (1<<0);
		if( IsValidNode< 1 >( _sNodes.treeNodes[i] , _dirichlet ) ) _sNodes.treeNodes[i]->nodeData.flags |= (1<<1);
	}
}
template< class Real > void Octree< Real >::_MakeComplete( void ){ _tree.setFullDepth( _spaceRoot->maxDepth() ) ; MemoryUsage(); }
// Trim off the branches of the tree (finer than _fullDepth) that don't contain normal points
template< class Real >
template< int NormalDegree >
void Octree< Real >::_ClipTree( const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo )
{
#if NEW_NEW_CODE
#define ABS_INDEX( idx ) ( (idx<0) ? -(idx) : (idx) )
	static const int      SupportSize   =  BSplineEvaluationData< NormalDegree >::SupportSize;
	static const int  LeftSupportRadius = -BSplineEvaluationData< NormalDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< NormalDegree >::SupportEnd;
	int maxDepth = _tree.maxDepth();
	typename TreeOctNode::NeighborKey< LeftSupportRadius , RightSupportRadius > neighborKey;
	neighborKey.set( maxDepth );

	// Set all nodes to invalid (negative indices)
	for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode(node) ) node->nodeData.nodeIndex = -node->nodeData.nodeIndex;

	// Iterate over the nodes and, if they contain a normal, make sure that the supported nodes exist and are set to valid
	for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode(node) ) if( normalInfo.index( ABS_INDEX( node->nodeData.nodeIndex ) )>=0 )
	{
		int depth = node->depth();
		neighborKey.getNeighbors< true >( node );
		for( int d=0 ; d<=depth ; d++ )
		{
			TreeOctNode::template Neighbors< SupportSize >& neighbors = neighborKey.neighbors[d];
			for( int i=0 ; i<SupportSize ; i++ ) for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
				if( neighbors.neighbors[i][j][k] ) neighbors.neighbors[i][j][k]->nodeData.nodeIndex = ABS_INDEX( neighbors.neighbors[i][j][k]->nodeData.nodeIndex );
		}
	}

	// Remove the invalid nodes
	for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode( node ) )
	{
		if( node->children && _Depth(node)>=_fullDepth )
		{
			bool hasValidChildren = false;
			for( int c=0 ; c<Cube::CORNERS ; c++ ) hasValidChildren |= ( node->children[c].nodeData.nodeIndex>0 );
			if( !hasValidChildren ) node->children = NULL;
		}
		node->nodeData.nodeIndex = ABS_INDEX( node->nodeData.nodeIndex );
	}

	MemoryUsage();
#undef ABS_INDEX
#else // !NEW_NEW_CODE
	int maxDepth = _tree.maxDepth();
	for( TreeOctNode* temp=_tree.nextNode() ; temp ; temp=_tree.nextNode(temp) )
		if( temp->children && _Depth( temp )>=_fullDepth )
		{
			int hasNormals=0;
			for( int i=0 ; i<Cube::CORNERS && !hasNormals ; i++ ) hasNormals = _HasNormals( &temp->children[i] , normalInfo );
			if( !hasNormals ) temp->children=NULL;
		}
	MemoryUsage();
#endif // NEW_NEW_CODE
}

template< class Real >
template< int FEMDegree >
void Octree< Real >::EnableMultigrid( std::vector< int >* map )
{
	if( FEMDegree<=_multigridDegree ) return;
	_multigridDegree = FEMDegree;
	const int OverlapRadius = -BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	int maxDepth = _tree.maxDepth( );
	typename TreeOctNode::NeighborKey< OverlapRadius , OverlapRadius > neighborKey;
	neighborKey.set( maxDepth-1 );
	for( int d=maxDepth-1 ; d>=0 ; d-- )
		for( TreeOctNode* node=_tree.nextNode() ; node ; node=_tree.nextNode( node ) ) if( node->depth()==d && node->children )
			neighborKey.template getNeighbors< true >( node );
	_sNodes.set( _tree , map );
	_SetValidityFlags();
}

template< class Real >
template< int NormalDegree >
int Octree< Real >::_HasNormals( TreeOctNode* node , const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo )
{
	int idx = normalInfo.index( node );
	if( idx>=0 )
	{
		const Point3D< Real >& normal = normalInfo.data[ idx ];
		if( normal[0]!=0 || normal[1]!=0 || normal[2]!=0 ) return 1;
	}
	if( node->children ) for( int i=0 ; i<Cube::CORNERS ; i++ ) if( _HasNormals( &node->children[i] , normalInfo ) ) return 1;
	return 0;
}

////////////////
// VertexData //
////////////////
long long VertexData::CenterIndex(const TreeOctNode* node,int maxDepth)
{
	int idx[DIMENSION];
	return CenterIndex(node,maxDepth,idx);
}
long long VertexData::CenterIndex(const TreeOctNode* node,int maxDepth,int idx[DIMENSION])
{
	int d,o[3];
	node->depthAndOffset(d,o);
	for(int i=0;i<DIMENSION;i++) idx[i]=BinaryNode::CornerIndex( maxDepth+1 , d+1 , o[i]<<1 , 1 );
	return (long long)(idx[0]) | (long long)(idx[1])<<VERTEX_COORDINATE_SHIFT | (long long)(idx[2])<<(2*VERTEX_COORDINATE_SHIFT);
}
long long VertexData::CenterIndex( int depth , const int offSet[DIMENSION] , int maxDepth , int idx[DIMENSION] )
{
	for(int i=0;i<DIMENSION;i++) idx[i]=BinaryNode::CornerIndex( maxDepth+1 , depth+1 , offSet[i]<<1 , 1 );
	return (long long)(idx[0]) | (long long)(idx[1])<<VERTEX_COORDINATE_SHIFT | (long long)(idx[2])<<(2*VERTEX_COORDINATE_SHIFT);
}
long long VertexData::CornerIndex(const TreeOctNode* node,int cIndex,int maxDepth)
{
	int idx[DIMENSION];
	return CornerIndex(node,cIndex,maxDepth,idx);
}
long long VertexData::CornerIndex( const TreeOctNode* node , int cIndex , int maxDepth , int idx[DIMENSION] )
{
	int x[DIMENSION];
	Cube::FactorCornerIndex( cIndex , x[0] , x[1] , x[2] );
	int d , o[3];
	node->depthAndOffset( d , o );
	for( int i=0 ; i<DIMENSION ; i++ ) idx[i] = BinaryNode::CornerIndex( maxDepth+1 , d , o[i] , x[i] );
	return CornerIndexKey( idx );
}
long long VertexData::CornerIndex( int depth , const int offSet[DIMENSION] , int cIndex , int maxDepth , int idx[DIMENSION] )
{
	int x[DIMENSION];
	Cube::FactorCornerIndex( cIndex , x[0] , x[1] , x[2] );
	for( int i=0 ; i<DIMENSION ; i++ ) idx[i] = BinaryNode::CornerIndex( maxDepth+1 , depth , offSet[i] , x[i] );
	return CornerIndexKey( idx );
}
long long VertexData::CornerIndexKey( const int idx[DIMENSION] )
{
	return (long long)(idx[0]) | (long long)(idx[1])<<VERTEX_COORDINATE_SHIFT | (long long)(idx[2])<<(2*VERTEX_COORDINATE_SHIFT);
}
long long VertexData::FaceIndex(const TreeOctNode* node,int fIndex,int maxDepth){
	int idx[DIMENSION];
	return FaceIndex(node,fIndex,maxDepth,idx);
}
long long VertexData::FaceIndex(const TreeOctNode* node,int fIndex,int maxDepth,int idx[DIMENSION])
{
	int dir,offset;
	Cube::FactorFaceIndex(fIndex,dir,offset);
	int d,o[3];
	node->depthAndOffset(d,o);
	for(int i=0;i<DIMENSION;i++){idx[i]=BinaryNode::CornerIndex(maxDepth+1,d+1,o[i]<<1,1);}
	idx[dir]=BinaryNode::CornerIndex(maxDepth+1,d,o[dir],offset);
	return (long long)(idx[0]) | (long long)(idx[1])<<VERTEX_COORDINATE_SHIFT | (long long)(idx[2])<<(2*VERTEX_COORDINATE_SHIFT);
}
long long VertexData::EdgeIndex( const TreeOctNode* node , int eIndex , int maxDepth ){ int idx[DIMENSION] ; return EdgeIndex( node , eIndex , maxDepth , idx ); }
long long VertexData::EdgeIndex( const TreeOctNode* node , int eIndex , int maxDepth , int idx[DIMENSION] )
{
	int o , i1 , i2;
	int d , off[3];
	node->depthAndOffset( d ,off );
	Cube::FactorEdgeIndex( eIndex , o , i1 , i2 );
	for( int i=0 ; i<DIMENSION ; i++ ) idx[i] = BinaryNode::CornerIndex( maxDepth+1 , d+1 , off[i]<<1 , 1 );
	switch(o)
	{
		case 0:
			idx[1] = BinaryNode::CornerIndex( maxDepth+1 , d , off[1] , i1 );
			idx[2] = BinaryNode::CornerIndex( maxDepth+1 , d , off[2] , i2 );
			break;
		case 1:
			idx[0] = BinaryNode::CornerIndex( maxDepth+1 , d , off[0] , i1 );
			idx[2] = BinaryNode::CornerIndex( maxDepth+1 , d , off[2] , i2 );
			break;
		case 2:
			idx[0] = BinaryNode::CornerIndex( maxDepth+1 , d , off[0] , i1 );
			idx[1] = BinaryNode::CornerIndex( maxDepth+1 , d , off[1] , i2 );
			break;
	};
	return (long long)(idx[0]) | (long long)(idx[1])<<VERTEX_COORDINATE_SHIFT | (long long)(idx[2])<<(2*VERTEX_COORDINATE_SHIFT);
}
