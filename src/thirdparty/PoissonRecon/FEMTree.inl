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

/////////////////////
// FEMTreeNodeData //
/////////////////////
#ifdef SANITIZED_PR
FEMTreeNodeData::FEMTreeNodeData( void ) : flags(0){}
FEMTreeNodeData &FEMTreeNodeData::operator = ( const FEMTreeNodeData &data )
{
	nodeIndex = data.nodeIndex;
	flags = data.flags.load();
	return *this;
}
#else // !SANITIZED_PR
FEMTreeNodeData::FEMTreeNodeData( void ){ flags = 0; }
#endif // SANITIZED_PR
FEMTreeNodeData::~FEMTreeNodeData( void ) { }


/////////////
// FEMTree //
/////////////

template< unsigned int Dim , class Real >
void FEMTree< Dim , Real >::_init( void )
{
	// Reset the depths and offsets
	int offset[Dim];
	for( int d=0 ; d<Dim ; d++ ) offset[d] = 0;
	RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >::ResetDepthAndOffset( &_tree , 0 , offset );

	// Set the _spaceRoot
	_spaceRoot = &_tree;
	for( int d=0 ; d<_depthOffset ; d++ )
	{
		if( !_spaceRoot->children ) MK_THROW( "Expected child node: " , d , " / " , _depthOffset );
		else if( d==0 ) _spaceRoot = _spaceRoot->children + (1<<Dim)-1;
		else            _spaceRoot = _spaceRoot->children;
	}
}

template< unsigned int Dim , class Real >
FEMTree< Dim , Real > *FEMTree< Dim , Real >::Merge( const FEMTree< Dim , Real > &tree1 , const FEMTree< Dim , Real > &tree2 , size_t blockSize )
{
	if( tree1._baseDepth != tree2._baseDepth ) MK_THROW( "Base depths differ: " , tree1._baseDepth , " != " , tree2._baseDepth );
	if( tree1._depthOffset != tree2._depthOffset ) MK_THROW( "Depth offsets differ: " , tree1._depthOffset , " != " , tree2._depthOffset );
	FEMTree< Dim , Real > *mergeTree = new FEMTree( blockSize );

	// have support overlapping the slice.
	std::function< void ( const FEMTreeNode *node1 , const FEMTreeNode *node2 , FEMTreeNode *mergeNode ) > mergeTrees =
		[&]( const FEMTreeNode *node1 , const FEMTreeNode *node2 , FEMTreeNode *mergeNode )
	{
		if( ( node1 && node1->children ) || ( node2 && node2->children ) )
		{
			if( !mergeNode->children ) mergeNode->template initChildren< false >( mergeTree->nodeAllocators.size() ? mergeTree->nodeAllocators[0] : NULL , mergeTree->_nodeInitializer );

			for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
				if     ( node1 && node1->children && node2 && node2->children ) mergeTrees( node1->children+c , node2->children+c , mergeNode->children+c );
				else if( node1 && node1->children                             ) mergeTrees( node1->children+c , NULL              , mergeNode->children+c );
				else if(                             node2 && node2->children ) mergeTrees( NULL              , node2->children+c , mergeNode->children+c );
		}
	};
	mergeTrees( &tree1._tree , &tree2._tree , &mergeTree->_tree );

	int d=0 , off[Dim];
	for( int d=0 ; d<Dim ; d++ ) off[d] = 0;
	FEMTreeNode::ResetDepthAndOffset( &mergeTree->_tree , d , off );
	mergeTree->_depthOffset = tree1._depthOffset;
	mergeTree->_baseDepth = tree1._baseDepth;

	mergeTree->_init();
	mergeTree->_maxDepth = mergeTree->_spaceRoot->maxDepth();

	std::vector< node_index_type > map;
	mergeTree->_sNodes.reset( mergeTree->_tree , map );
	mergeTree->_setSpaceValidityFlags();

	return mergeTree;
}

template< unsigned int Dim , class Real >
template< unsigned int CrossDegree , unsigned int Pad >
FEMTree< Dim , Real > *FEMTree< Dim , Real >::Slice( const FEMTree< Dim+1 , Real > &tree , unsigned int sliceDepth , unsigned int sliceIndex , bool includeBounds , size_t blockSize )
{
	if( sliceIndex>(unsigned int)(1<<sliceDepth) ) MK_THROW( "Slice index out of bounds: 0 <= " , sliceIndex , " <= " , (1<<sliceDepth) );
	FEMTree< Dim , Real > *sliceTree = new FEMTree( blockSize );

	unsigned int maxDepth = tree.maxDepth();
	if( sliceDepth<maxDepth+1 )
	{
		sliceIndex <<= ( maxDepth+1-sliceDepth );
		sliceDepth = maxDepth+1;
	}
	const int StartOffset = BSplineSupportSizes< CrossDegree >::SupportStart-(int)Pad;
	const int   EndOffset = BSplineSupportSizes< CrossDegree >::SupportEnd+1+(int)Pad;

	// A function returning true if the function indexed by the node has support overlapping the slice
	auto OverlapsSlice = [&]( const typename FEMTree< Dim+1 , Real >::FEMTreeNode *node )
	{
		typename FEMTree< Dim+1 , Real >::LocalDepth d ; typename FEMTree< Dim+1 , Real >::LocalOffset off;
		tree.depthAndOffset( node , d , off );
		if( d<0 ) return true;
		else
		{
			int start = ( off[Dim] + StartOffset )<<(sliceDepth-d);
			int end   = ( off[Dim] +   EndOffset )<<(sliceDepth-d);
			if( includeBounds ) return (int)sliceIndex>=start && (int)sliceIndex<=end;
			else                return (int)sliceIndex> start && (int)sliceIndex< end;
		}
	};

	// Walk through the two trees in tandem, adding children to the slice-tree if the associated children in the full tree
	// have support overlapping the slice.
	std::function< void ( const typename FEMTree< Dim+1 , Real >::FEMTreeNode *node , typename FEMTree< Dim , Real >::FEMTreeNode *sliceNode ) > refineSliceTree =
		[&]( const typename FEMTree< Dim+1 , Real >::FEMTreeNode *node , typename FEMTree< Dim , Real >::FEMTreeNode *sliceNode )
	{
		if( !GetGhostFlag( node->children ) )
		{
			bool overlaps = false;
			for( unsigned int c=0 ; c<(1<<(Dim+1)) ; c++ ) overlaps |= OverlapsSlice( node->children+c );
			if( overlaps )
			{
				if( !sliceNode->children ) sliceNode->template initChildren< false >( sliceTree->nodeAllocators.size() ? sliceTree->nodeAllocators[0] : NULL , sliceTree->_nodeInitializer );
				for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
				{
					refineSliceTree( node->children+( c|(1<<Dim) ) , sliceNode->children+c );
					refineSliceTree( node->children+( c          ) , sliceNode->children+c );
				}
			}
		}
	};

	refineSliceTree( &tree._tree , &sliceTree->_tree );

	int d=0 , off[Dim];
	for( int d=0 ; d<Dim ; d++ ) off[d] = 0;
	FEMTreeNode::ResetDepthAndOffset( &sliceTree->_tree , d , off );
	sliceTree->_depthOffset = tree._depthOffset;
	sliceTree->_baseDepth = tree._baseDepth;

	sliceTree->_init();
	sliceTree->_maxDepth = sliceTree->_spaceRoot->maxDepth();

	std::vector< node_index_type > map;
	sliceTree->_sNodes.reset( sliceTree->_tree , map );
	sliceTree->_setSpaceValidityFlags();
	return sliceTree;
}

template< unsigned int Dim , class Real >
template< unsigned int ... FEMSigs  , typename Data >
void FEMTree< Dim , Real >::merge( const FEMTree< Dim , Real > &tree , const DenseNodeData< Data , UIntPack< FEMSigs ... > > &coefficients , DenseNodeData< Data , UIntPack< FEMSigs ... > > &mergeCoefficients ) const
{
	static_assert( sizeof ... ( FEMSigs )==Dim , "[ERROR] Signature count and dimension don't match" );

	std::function< void ( const FEMTreeNode * , const FEMTreeNode * ) > accumulateCoefficients =
		[&]( const FEMTreeNode *node , const FEMTreeNode *mergeNode )
	{
		if( node && node->nodeData.nodeIndex!=-1 )
		{
			if( !mergeNode || mergeNode->nodeData.nodeIndex==-1 ) MK_THROW( "Merge node not set" );
			LocalDepth d ; LocalOffset off;
			tree.depthAndOffset( node , d , off );
			mergeCoefficients[ mergeNode->nodeData.nodeIndex ] += coefficients[ node->nodeData.nodeIndex ];
		}
		if( node && node->children ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) accumulateCoefficients( node->children+c , mergeNode->children+c );
	};
	accumulateCoefficients( &tree._tree , &_tree );
}

template< unsigned int Dim , class Real , unsigned int Pad , unsigned int FEMSig >
struct SliceEvaluator
{
	struct _SliceEvaluator
	{
		const int StartOffset = BSplineSupportSizes< FEMSignature< FEMSig >::Degree >::SupportStart-(int)Pad;
		const int   EndOffset = BSplineSupportSizes< FEMSignature< FEMSig >::Degree >::SupportEnd+1+(int)Pad;
		int start;
		Real values[ BSplineSupportSizes< FEMSignature< FEMSig >::Degree >::SupportSize+2*Pad ];
		void init( unsigned int depth , double x , unsigned int d )
		{
			// off @ depthsupports the slice if 
			//		x>(off+StartOffset)/(1<<depth) && x<(off+EndOffset)/(1<<depth)
			// <=>	x*(1<<depth)-StartOffset>off && x*(1<<depth)-EndOfset<off
			// <=>	off \in ( x*(1<<depth)-EndOffset , x*(1<<depth)-StartOffset )
			// <=	off \in [ ceil( x*(1<<depth) )-EndOffset , ceil( x*(1<<depth) ) - StartOffset )
			start = (int)ceil( x*(1<<depth)-EndOffset );

			// The derivative is lower than the degree, the derivative will be continuous so we can evaluate it directly
			if( d<FEMSignature< FEMSig >::Degree ) for( int i=0 ; i<EndOffset-StartOffset ; i++ ) values[i] = (Real)BSplineEvaluationData< FEMSig >::Value( depth , start+i , x , d );
			else if( d==FEMSignature< FEMSig >::Degree )
			{
				double eps = 1e-4/(1<<depth);
#ifdef SHOW_WARNINGS
				MK_WARN_ONCE( "Using discrete derivative: " , eps );
#endif // SHOW_WARNINGS
				// If we are dealing with a degree-zero polynomial, we offset
				if( d==0 )
				{
					for( int i=0 ; i<EndOffset-StartOffset ; i++ )
					{
						double value = 0;
						value += BSplineEvaluationData< FEMSig >::Value( depth , start+i , x-eps , d );
						value += BSplineEvaluationData< FEMSig >::Value( depth , start+i , x+eps , d );
						values[i] = (Real)(value/2);
					}
				}
				else // Otherwise we compute the discrete derivative
				{
					for( int i=0 ; i<EndOffset-StartOffset ; i++ )
					{
						double value = 0;
						value += BSplineEvaluationData< FEMSig >::Value( depth , start+i , x , d-1 ) - BSplineEvaluationData< FEMSig >::Value( depth , start+i , x-eps , d-1 );
						value += BSplineEvaluationData< FEMSig >::Value( depth , start+i , x+eps , d-1 ) - BSplineEvaluationData< FEMSig >::Value( depth , start+i , x , d-1 );
						values[i] = (Real)(value/(2*eps) );
					}
				}

			}
			else MK_THROW( "Derivative exceeds degree: " , d , " > " , FEMSignature< FEMSig >::Degree );
		}
		Real operator()( int off ) const
		{
			if( off<start || off>=start+(EndOffset-StartOffset ) ) return 0;
			else return values[off-start];
		}
	};

	std::vector< _SliceEvaluator > evaluators;
	void init( unsigned int maxDepth , double x , unsigned int d )
	{
		evaluators.resize( maxDepth+1 );
		for( unsigned int depth=0 ; depth<=maxDepth ; depth++ ) evaluators[depth].init( depth , x , d );
	}
	Real operator()( int d , int off ) const
	{
		if( d<0 || d>=(int)evaluators.size() ) return 0;
		else return evaluators[d]( off );
	}
};

template< unsigned int Dim , class Real >
template< unsigned int Pad , unsigned int FEMSig , unsigned int ... FEMSigs , typename Data >
void FEMTree< Dim , Real >::slice( const FEMTree< Dim+1 , Real > &tree , unsigned int d , const DenseNodeData< Data , UIntPack< FEMSigs ... , FEMSig > > &coefficients , DenseNodeData< Data , UIntPack< FEMSigs ... > > &sliceCoefficients , unsigned int sliceDepth , unsigned int sliceIndex ) const
{
	static_assert( sizeof ... ( FEMSigs )==Dim , "[ERROR] Signature count and dimension don't match" );
#ifdef __GNUC__
#ifdef SHOW_WARNINGS
	#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
#else // !__GNUC__
	static const unsigned int CrossDegree = FEMSignature< FEMSig >::Degree;
	static const unsigned int _Pad = Pad;
#endif // __GNUC__

	unsigned int maxDepth = tree.maxDepth();
	if( sliceDepth<maxDepth+1 )
	{
		sliceIndex <<= ( maxDepth+1-sliceDepth );
		sliceDepth = maxDepth+1;
	}

	double s = (double)sliceIndex/(1<<sliceDepth);

#ifdef __GNUC__
	const int StartOffset = BSplineSupportSizes< FEMSignature< FEMSig >::Degree >::SupportStart-(int)Pad;
	const int   EndOffset = BSplineSupportSizes< FEMSignature< FEMSig >::Degree >::SupportEnd+1+(int)Pad;
#else // !__GNUC__
	const int StartOffset = BSplineSupportSizes< CrossDegree >::SupportStart-(int)Pad;
	const int   EndOffset = BSplineSupportSizes< CrossDegree >::SupportEnd+1+(int)Pad;
#endif // __GNUC__

	SliceEvaluator< Dim , Real , Pad , FEMSig > sliceEvaluator;
	sliceEvaluator.init( maxDepth , s , d );

	// A function return true if the function indexed by the node has support overlapping the slice
	auto OverlapsSlice = [&]( const typename FEMTree< Dim+1 , Real >::FEMTreeNode *node )
	{
		typename FEMTree< Dim+1 , Real >::LocalDepth d ; typename FEMTree< Dim+1 , Real >::LocalOffset off;
		tree.depthAndOffset( node , d , off );
		if( d<0 ) return true;
		else
		{
			int start = ( off[Dim] + StartOffset )<<(sliceDepth-d);
			int end   = ( off[Dim] +   EndOffset )<<(sliceDepth-d);
			return (int)sliceIndex>start && (int)sliceIndex<end;
		}
	};

	std::function< void ( const typename FEMTree< Dim+1 , Real >::FEMTreeNode * , const typename FEMTree< Dim , Real >::FEMTreeNode * ) > accumulateSliceCoefficients =
		[&]( const typename FEMTree< Dim+1 , Real >::FEMTreeNode *node , const typename FEMTree< Dim , Real >::FEMTreeNode *sliceNode )
	{
		if( node->nodeData.nodeIndex!=-1 )
		{
			if( sliceNode->nodeData.nodeIndex==-1 ) MK_THROW( "Slice node not set" );
			typename FEMTree< Dim+1 , Real >::LocalDepth d ; typename FEMTree< Dim+1 , Real >::LocalOffset off;
			tree.depthAndOffset( node , d , off );
			sliceCoefficients[ sliceNode->nodeData.nodeIndex ] += coefficients[ node->nodeData.nodeIndex ] * sliceEvaluator( d , off[Dim] );
		}
		if( !GetGhostFlag( node->children ) )
		{
			if( OverlapsSlice( node->children          ) ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) accumulateSliceCoefficients( node->children+( c          ) , sliceNode->children+c );
			if( OverlapsSlice( node->children+(1<<Dim) ) ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) accumulateSliceCoefficients( node->children+( c|(1<<Dim) ) , sliceNode->children+c );
		}
	};
	accumulateSliceCoefficients( &tree._tree , &_tree );
}

template< unsigned int Dim , class Real >
FEMTree< Dim , Real >::FEMTree( size_t blockSize ) : _nodeInitializer( *this ) , _depthOffset(1)
{
	if( blockSize )
	{
		nodeAllocators.resize( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<nodeAllocators.size() ; i++ )
		{
			nodeAllocators[i] = new Allocator< FEMTreeNode >();
			nodeAllocators[i]->set( blockSize );
		}
	}
	_nodeCount = 0;
	// Initialize the root
	_nodeInitializer( _tree );
	_tree.template initChildren< false >( nodeAllocators.size() ? nodeAllocators[0] : NULL , _nodeInitializer );
	_init();
	memset( _femSigs1 , -1 , sizeof( _femSigs1 ) );
	memset( _femSigs2 , -1 , sizeof( _femSigs2 ) );
}

template< unsigned int Dim , class Real >
FEMTree< Dim , Real >::FEMTree( BinaryStream &stream , size_t blockSize ) : FEMTree( blockSize )
{
	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;
	node_index_type nodeCount;
	if( !stream.read( nodeCount ) ) MK_THROW( "Failed to read nodeCount" );
	_nodeCount = nodeCount;
	if( !stream.read( _maxDepth ) ) MK_THROW( "Failed to read _maxDepth" );
	if( !stream.read( _depthOffset ) ) MK_THROW( "Failed to read _depthOffset" );
	if( !stream.read( _baseDepth ) ) MK_THROW( "Failed to read _baseDepth" );
	_tree.read( stream , nodeAllocator );
	_init();
	_sNodes.read( stream , _tree );
}

template< unsigned int Dim , class Real > void FEMTree< Dim , Real >::write( BinaryStream &stream , bool serialize ) const
{
	node_index_type nodeCount = _nodeCount;
	stream.write( nodeCount );
	stream.write( _maxDepth );
	stream.write( _depthOffset );
	stream.write( _baseDepth );
	_tree.write( stream , serialize );
	_sNodes.write( stream );
}

template< unsigned int Dim , class Real >
const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* FEMTree< Dim , Real >::leaf( Point< Real , Dim > p ) const
{
	if( !_InBounds( p ) ) return NULL;
	Point< Real , Dim > center;
	for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
	Real width = Real(1.0);
	FEMTreeNode* node = _spaceRoot;
	while( node->children )
	{
		int cIndex = FEMTreeNode::ChildIndex( center , p );
		node = node->children + cIndex;
		width /= 2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) center[d] += width/2;
			else                  center[d] -= width/2;
	}
	return node;
}
template< unsigned int Dim , class Real >
template< bool ThreadSafe >
RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* FEMTree< Dim , Real >::_leaf( Allocator< FEMTreeNode > *nodeAllocator , Point< Real , Dim > p , LocalDepth maxDepth )
{
	if( !_InBounds( p ) ) return NULL;
	Point< Real , Dim > center;
	for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
	Real width = Real(1.0);
	FEMTreeNode* node = _spaceRoot;
	LocalDepth d = _localDepth( node );
	while( ( d<0 && node->children ) || ( d>=0 && d<maxDepth ) )
	{
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		int cIndex = FEMTreeNode::ChildIndex( center , p );
		node = node->children + cIndex;
		d++;
		width /= 2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) center[d] += width/2;
			else                  center[d] -= width/2;
	}
	return node;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* FEMTree< Dim , Real >::_leaf( Allocator< FEMTreeNode > *nodeAllocator , Point< Real , Dim > p , std::function< int ( Point< Real , Dim > ) > &pointDepthFunctor )
{
	if( !_InBounds( p ) ) return NULL;
	int maxDepth = pointDepthFunctor( p );
	Point< Real , Dim > center;
	for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
	Real width = Real(1.0);
	FEMTreeNode* node = _spaceRoot;
	LocalDepth depth = 0;
	int cIndex = FEMTreeNode::ChildIndex( center , p );

	while( depth<maxDepth )
	{
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		int cIndex = FEMTreeNode::ChildIndex( center , p );
		node = node->children + cIndex;

		depth++;
		width /= 2;
		for( int d=0 ; d<Dim ; d++ )
			if( (cIndex>>d) & 1 ) center[d] += width/2;
			else                  center[d] -= width/2;
	}
	return node;
}

template< unsigned int Dim , class Real > bool FEMTree< Dim , Real >::_InBounds( Point< Real , Dim > p ){ for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return false ; return true; }
template< unsigned int Dim , class Real >
template< unsigned int ... FEMSignatures >
bool FEMTree< Dim , Real >::isValidFEMNode( UIntPack< FEMSignatures ... > , const FEMTreeNode* node ) const
{
	if( GetGhostFlag< Dim >( node ) ) return false;
	LocalDepth d ; LocalOffset off ; _localDepthAndOffset( node , d , off );
	if( d<0 ) return false;
	return FEMIntegrator::IsValidFEMNode( UIntPack< FEMSignatures ... >() , d , off );
}
template< unsigned int Dim , class Real >
bool FEMTree< Dim , Real >::isValidSpaceNode( const FEMTreeNode* node ) const
{
	if( !node ) return false;
	LocalDepth d ; LocalOffset off ; _localDepthAndOffset( node , d , off );
	if( d<0 ) return false;
	int res = 1<<d;
	for( int dd=0 ; dd<Dim ; dd++ ) if( off[dd]<0 || off[dd]>=res ) return false;
	return true;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe , typename AddNodeFunctor , unsigned int ... Degrees >
void FEMTree< Dim , Real >::_refine( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , const AddNodeFunctor &addNodeFunctor , FEMTreeNode *node )
{
	LocalDepth d , _d ; LocalOffset off , _off;
	_localDepthAndOffset( node , d , off );
	_d = d+1;

	bool refine = d<0;
	for( int c=0 ; c<(1<<Dim) ; c++ )
	{
		for( int d=0 ; d<Dim ; d++ )
			if( c&(1<<d) ) _off[d] = off[d]*2+1;
			else           _off[d] = off[d]*2+0;
		refine |= !FEMIntegrator::IsOutOfBounds( UIntPack< FEMDegreeAndBType< Degrees , BOUNDARY_FREE >::Signature ... >() , _d , _off ) && addNodeFunctor( _d , _off );
	}
	if( refine )
	{
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		for( int c=0 ; c<(1<<Dim) ; c++ ) _refine< ThreadSafe >( UIntPack< Degrees ... >() , nodeAllocator , addNodeFunctor , node->children+c );
	}
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe , typename AddNodeFunctor , unsigned int ... Degrees >
void FEMTree< Dim , Real >::_refine( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , const AddNodeFunctor &addNodeFunctor )
{
	if( !_tree.children ) _tree.template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
	for( int c=0 ; c<(1<<Dim) ; c++ ) _refine< ThreadSafe >( UIntPack< Degrees ... >() , nodeAllocator , addNodeFunctor , _tree.children+c );
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe , unsigned int ... Degrees >
void FEMTree< Dim , Real >::_setFullDepth( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , FEMTreeNode* node , LocalDepth depth )
{
	LocalDepth d ; LocalOffset off;
	_localDepthAndOffset( node , d , off );
	bool refine = d<depth && ( d<0 || !FEMIntegrator::IsOutOfBounds( UIntPack< FEMDegreeAndBType< Degrees , BOUNDARY_FREE >::Signature ... >() , d , off ) );
	if( refine )
	{
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
		for( int c=0 ; c<(1<<Dim) ; c++ ) _setFullDepth< ThreadSafe >( UIntPack< Degrees ... >() , nodeAllocator , node->children+c , depth );
	}
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe , unsigned int ... Degrees >
void FEMTree< Dim , Real >::_setFullDepth( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , LocalDepth depth )
{
	if( !_tree.children ) _tree.template initChildren< ThreadSafe >( nodeAllocator , _nodeInitializer );
	for( int c=0 ; c<(1<<Dim) ; c++ ) _setFullDepth< ThreadSafe >( UIntPack< Degrees ... >() , nodeAllocator , _tree.children+c , depth );
}

template< unsigned int Dim , class Real >
template< unsigned int ... Degrees >
typename FEMTree< Dim , Real >::LocalDepth FEMTree< Dim , Real >::_getFullDepth( UIntPack< Degrees ... > , const FEMTreeNode* node ) const
{
	LocalDepth d ; LocalOffset off;
	_localDepthAndOffset( node , d , off );
	bool refine = d<0 || !FEMIntegrator::IsOutOfBounds( UIntPack< FEMDegreeAndBType< Degrees , BOUNDARY_FREE >::Signature ... >() , d , off );

	if( refine )
	{
		if( !node->children ) return d;
		else
		{
			LocalDepth depth = INT_MAX;
			for( int c=0 ; c<(1<<Dim) ; c++ )
			{
				LocalDepth d = _getFullDepth( UIntPack< Degrees ... >() , node->children+c );
				if( d<depth ) depth = d;
			}
			return depth;
		}
	}
	else return INT_MAX;
}

template< unsigned int Dim , class Real >
template< unsigned int ... Degrees >
typename FEMTree< Dim , Real >::LocalDepth FEMTree< Dim , Real >::_getFullDepth( UIntPack< Degrees ... > , const LocalDepth depth , const LocalOffset begin , const LocalOffset end , const FEMTreeNode* node ) const
{
	LocalDepth d ; LocalOffset off;
	_localDepthAndOffset( node , d , off );

	// [NOTE]: Changing closed interval to half open interval
	const int StartOffsets[] = { BSplineSupportSizes< Degrees >::SupportStart ... };
	const int   EndOffsets[] = { BSplineSupportSizes< Degrees >::SupportEnd + 1 ... };

	auto IsSupported = [&]( LocalDepth d , LocalOffset off )
	{
		if( d>depth ) return false;
		LocalOffset supportStart , supportEnd;
		for( unsigned int dim=0 ; dim<Dim ; dim++ )
		{
			supportStart[dim] = ( off[dim] + StartOffsets[dim] )*(1<<(depth-d));
			supportEnd  [dim] = ( off[dim] +   EndOffsets[dim] )*(1<<(depth-d));
		}
		if( d>=0 ) for( unsigned int dim=0 ; dim<Dim ; dim++ ) if( supportStart[dim]>=end[dim] || supportEnd[dim]<=begin[dim] ) return false;
		return true;
	};

	{
		if( !node->children )
		{
			LocalDepth _d=d+1;
			LocalOffset _off;
			bool childrenSupported = false;
			for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
			{
				for( int dim=0 ; dim<Dim ; dim++ )
#ifdef SANITIZED_PR
					if( c&(1<<dim) ) _off[dim] = off[dim]*2 + 1;
					else             _off[dim] = off[dim]*2 + 0;
#else // !SANITIZED_PR
					if( c&(1<<dim) ) _off[dim] = (off[dim]<<1) | 1;
					else             _off[dim] = (off[dim]<<1);
#endif // SANITIZED_PR
				childrenSupported |= IsSupported( _d , _off );
			}
			if( childrenSupported ) return d;
			else return INT_MAX;
		}
		else
		{
			LocalDepth minDepth = INT_MAX;
			for( int c=0 ; c<(1<<Dim) ; c++ )
			{
				LocalDepth d = _getFullDepth( UIntPack< Degrees ... >() , depth , begin , end , node->children+c );
				if( d<minDepth ) minDepth = d;
			}
			return minDepth;
		}
	}
}

template< unsigned int Dim , class Real >
template< unsigned int ... Degrees >
typename FEMTree< Dim , Real >::LocalDepth FEMTree< Dim , Real >::getFullDepth( UIntPack< Degrees ... > ) const
{
	if( !_tree.children ) return -1;
	LocalDepth depth = INT_MAX;
	for( int c=0 ; c<(1<<Dim) ; c++ )
	{
		LocalDepth d = _getFullDepth( UIntPack< Degrees ... >() , _tree.children+c );
		if( d<depth ) depth = d;
	}
	return depth;
}

template< unsigned int Dim , class Real >
template< unsigned int ... Degrees >
typename FEMTree< Dim , Real >::LocalDepth FEMTree< Dim , Real >::getFullDepth( UIntPack< Degrees ... > , const LocalDepth depth , const LocalOffset begin , const LocalOffset end ) const
{
	// [NOTE] Need "+1" because _getFullDepth will test children of leaves
	LocalDepth maxDepth = this->maxDepth() + 1;
	LocalDepth _depth ; LocalOffset _begin , _end;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		if( begin[d]>end[d] ) MK_THROW( "Bad bounds [" , d , "]: " , begin[d] , " <= " , end[d] );
		if( begin[d]<0 ) MK_THROW( "Start bound cannot be negative [" , d , "]: 0 <= " , begin[d]  );
		if( end[d]>(1<<depth) ) MK_THROW( "End bound cannot exceed resolution [" , d , "]: " , end[d] , " <=" , (1<<depth) );

		// Push to max depth
		if( depth<maxDepth )
		{
			_begin[d] = begin[d]<<(maxDepth-depth);
			_end  [d] = end  [d]<<(maxDepth-depth);
		}
		else
		{
			_begin[d] = begin[d];
			_end  [d] = end  [d];
		}
	}
	if( depth<maxDepth ) _depth = maxDepth;
	else                 _depth = depth;

	if( !_tree.children ) return -1;

	LocalDepth minDepth = INT_MAX;
	for( int c=0 ; c<(1<<Dim) ; c++ )
	{
		LocalDepth d = _getFullDepth( UIntPack< Degrees ... >() , _depth , _begin , _end , _tree.children+c );
		if( d<minDepth ) minDepth = d;
	}

	return minDepth;
}

template< unsigned int Dim , class Real >
template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes , typename ProcessingNodeFunctor , typename ... DenseOrSparseNodeData , typename InitializeFunctor >
void FEMTree< Dim , Real >::processNeighbors( ProcessingNodeFunctor processingNode , std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize )
{
	std::vector< FEMTreeNode * > nodes;
	nodes.reserve( _spaceRoot->nodes() );
	_spaceRoot->processNodes( [&]( FEMTreeNode *node ){ if( processingNode(node) ) nodes.push_back( node ); } );
	processNeighbors< LeftRadius , RightRadius , CreateNodes >( &nodes[0] , nodes.size() , data , initialize );
}

template< unsigned int Dim , class Real >
template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes , typename ... DenseOrSparseNodeData , typename InitializeFunctor >
void FEMTree< Dim , Real >::processNeighbors( FEMTreeNode **nodes , size_t nodeCount, std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize )
{
	int maxDepth = 0;
	for( size_t i=0 ; i<nodeCount ; i++ ) maxDepth = std::max( maxDepth , nodes[i]->depth() );
	std::vector< node_index_type > map( _nodeCount );
	for( node_index_type i=0 ; i<_nodeCount ; i++ ) map[i] = i;
	typedef typename RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< IsotropicUIntPack< Dim , LeftRadius > , IsotropicUIntPack< Dim , RightRadius > > NeighborKey;

	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;
	//	typename RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< IsotropicUIntPack< Dim , LeftRadius > , IsotropicUIntPack< Dim , RightRadius > > neighborKey;
	NeighborKey neighborKey;
	neighborKey.set( maxDepth );
	for( size_t i=0 ; i<nodeCount ; i++ )
	{
		auto neighbors = neighborKey.template getNeighbors< CreateNodes , false >( nodes[i] , nodeAllocator , _nodeInitializer );
		for( unsigned int j=0 ; j<neighbors.neighbors.Size() ; j++ ) if( neighbors.neighbors.data[j] ) initialize( neighbors.neighbors.data[j] );
	}

	_reorderDenseOrSparseNodeData< 0 >( GetPointer( map ) , _nodeCount , data );
}

template< unsigned int Dim , class Real >
template< unsigned int LeftRadius , unsigned int RightRadius , typename IsProcessingNodeFunctor , typename ProcessingKernel >
void FEMTree< Dim , Real >::processNeighboringLeaves( IsProcessingNodeFunctor isProcessingNode , ProcessingKernel kernel , bool processSubTree )
{
	std::vector< FEMTreeNode * > nodes;
	nodes.reserve( _spaceRoot->nodes() );
	_spaceRoot->processNodes( [&]( FEMTreeNode *node ){ if( isProcessingNode(node) ) nodes.push_back( node ); } );
	processNeighboringLeaves< LeftRadius , RightRadius >( &nodes[0] , nodes.size() , kernel , processSubTree );
}

template< unsigned int Dim , class Real >
template< unsigned int LeftRadius , unsigned int RightRadius , typename ProcessingKernel >
void FEMTree< Dim , Real >::processNeighboringLeaves( FEMTreeNode **nodes , size_t nodeCount  , ProcessingKernel kernel , bool processSubTree )
{
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] may process the same leaf multiple times, if it is coarser" )
#endif // SHOW_WARNINGS
	typedef typename RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< IsotropicUIntPack< Dim , LeftRadius > , IsotropicUIntPack< Dim , RightRadius > > NeighborKey;
	// Suppose that we have a node at index I and we want the leaf nodes supported on the (possibly virtual) node K away
	// Case 1: The K-th neighbor exists
	// ---> Iterate over the leaf nodes of the sub-tree rooted at the K-th neighbor
	// Case 2: The K-th neighbor does not exist
	// ---> The index of the K-th neighbor is I+K
	// ---> The index of the parent is floor( I/2 )
	// ---> The index of the K-th neighbors parent is floor( (I+K)/2 )
	// ---> The parent of the K-th neighbor is the [ floor( (I+k)/2 ) - floor( I/2 ) ]-th neighbor of the parent

	std::function< void ( FEMTreeNode * ) > ProcessSubTree = [&]( FEMTreeNode *node )
	{
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) ProcessSubTree( node->children+c );
		kernel( node );
	};

	unsigned int maxDepth=0;
	for( size_t i=0 ; i<nodeCount ; i++ ) maxDepth = std::max( maxDepth , (unsigned int)nodes[i]->depth() );

	std::vector< NeighborKey > neighborKeys( ThreadPool::NumThreads() );
	for( int i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( maxDepth );

	ThreadPool::ParallelFor( 0 , nodeCount , [&]( unsigned int t , size_t  i )
		{
			typedef StaticWindow< FEMTreeNode * , IsotropicUIntPack< Dim , LeftRadius+RightRadius+1 > > NeighborLeafNodes;
			NeighborLeafNodes neighborLeafNodes;
			neighborKeys[t].setLeafNeighbors( nodes[i] , neighborLeafNodes );
			for( unsigned int i=0 ; i<NeighborLeafNodes::Size() ; i++ ) if( neighborLeafNodes.data[i] )
				if( processSubTree ) ProcessSubTree( neighborLeafNodes.data[i] );
				else kernel( neighborLeafNodes.data[i] );
		} );
}

template< unsigned int Dim , class Real >
template< class C , unsigned int ... FEMSigs >
DenseNodeData< C , UIntPack< FEMSigs ... > > FEMTree< Dim , Real >::trimToDepth( const DenseNodeData< C , UIntPack< FEMSigs ... > >& data , LocalDepth coarseDepth ) const
{
	if( coarseDepth>_maxDepth ) return data;
	return data._trim( _sNodesEnd( coarseDepth ) );
}

template< unsigned int Dim , class Real >
template< class C , unsigned int ... FEMSigs >
SparseNodeData< C , UIntPack< FEMSigs ... > > FEMTree< Dim , Real >::trimToDepth( const SparseNodeData< C , UIntPack< FEMSigs ... > >& data , LocalDepth coarseDepth ) const
{
	if( coarseDepth>_maxDepth ) return data;
	return data._trim( _sNodesEnd( coarseDepth ) );
}

template< unsigned int Dim , class Real >
template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
typename FEMTree< Dim , Real >::template ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual > FEMTree< Dim , Real >::trimToDepth( const ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual > &iInfo , LocalDepth coarseDepth ) const
{
	ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual > _iInfo;
	if( coarseDepth<_maxDepth ) _iInfo.iData = iInfo.iData._trim( _sNodesEnd( coarseDepth ) );
	else                        _iInfo.iData = iInfo.iData;
	_iInfo._constrainsDCTerm = iInfo._constrainsDCTerm;
	_iInfo._constraintDual = iInfo._constraintDual;
	_iInfo._systemDual = iInfo._systemDual;
	return _iInfo;
}

template< unsigned int Dim , class Real >
template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
typename FEMTree< Dim , Real >::template ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual > FEMTree< Dim , Real >::trimToDepth( const ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual > &iInfo , LocalDepth coarseDepth ) const
{
	ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual > _iInfo;
	if( coarseDepth<_maxDepth ) _iInfo.iData = iInfo.iData._trim( _sNodesEnd( coarseDepth ) );
	else                        _iInfo.iData = iInfo.iData;
	_iInfo._constrainsDCTerm = iInfo._constrainsDCTerm;
	_iInfo._constraintDual = iInfo._constraintDual;
	_iInfo._systemDual = iInfo._systemDual;
	return _iInfo;
}

template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int DensityDegree >
void FEMTree< Dim , Real >::updateDensityEstimator( typename FEMTree< Dim , Real >::template DensityEstimator< DensityDegree > &density , const std::vector< PointSample >& samples , LocalDepth minSplatDepth , LocalDepth maxSplatDepth )
{
	//	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( _spaceRoot );
	SubTreeExtractor subtreeExtractor( _spaceRoot , _depthOffset );
	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;
	LocalDepth maxDepth = _spaceRoot->maxDepth();
	maxSplatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( maxSplatDepth , maxDepth ) );
	minSplatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( minSplatDepth , maxDepth ) );
	if( minSplatDepth>maxSplatDepth ) MK_THROW( "Minimum splat depth exceeds maximum splat depth" );
	PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > densityKey;
	densityKey.set( maxSplatDepth );

	std::vector< node_index_type > sampleMap( nodeCount() , -1 );

	// Initialize the map from node indices to samples
	ThreadPool::ParallelFor( 0 , samples.size() , [&]( unsigned int , size_t i ){ if( samples[i].sample.weight>0 ) sampleMap[ samples[i].node->nodeData.nodeIndex ] = (node_index_type)i; } );

	std::function< ProjectiveData< Point< Real , Dim > , Real > ( FEMTreeNode* ) > SetDensity = [&] ( FEMTreeNode* node )
	{
		ProjectiveData< Point< Real , Dim > , Real > sample;
		LocalDepth d = node->depth();
		node_index_type idx = node->nodeData.nodeIndex;
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) sample += SetDensity( node->children + c );
		if( idx<(node_index_type)sampleMap.size() && sampleMap[idx]!=-1 ) sample += samples[ sampleMap[ idx ] ].sample;
		if( d>=minSplatDepth && d<=maxSplatDepth && sample.weight>0 ) _addWeightContribution< true , CoDim >( nodeAllocator , density , node , sample.data / sample.weight , densityKey , sample.weight );
		return sample;
	};
	SetDensity( _spaceRoot );
}

template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int DensityDegree >
void FEMTree< Dim , Real >::updateDensityEstimator( typename FEMTree< Dim , Real >::template DensityEstimator< DensityDegree > &density , const std::vector< PointSample >& samples , LocalDepth minSplatDepth , LocalDepth maxSplatDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor )
{
	//	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( _spaceRoot );
	SubTreeExtractor subtreeExtractor( _spaceRoot , _depthOffset );
	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;
	LocalDepth maxDepth = _spaceRoot->maxDepth();
	maxSplatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( maxSplatDepth , maxDepth ) );
	minSplatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( minSplatDepth , maxDepth ) );
	if( minSplatDepth>maxSplatDepth ) MK_THROW( "Minimum splat depth exceeds maximum splat depth" );
	PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > densityKey;
	densityKey.set( maxSplatDepth );

	std::vector< node_index_type > sampleMap( nodeCount() , -1 );

	// Initialize the map from node indices to samples
	ThreadPool::ParallelFor( 0 , samples.size() , [&]( unsigned int , size_t i ){ if( samples[i].sample.weight>0 ) sampleMap[ samples[i].node->nodeData.nodeIndex ] = (node_index_type)i; } );

	std::function< ProjectiveData< Point< Real , Dim > , Real > ( FEMTreeNode* ) > SetDensity = [&] ( FEMTreeNode* node )
	{
		ProjectiveData< Point< Real , Dim > , Real > sample;
		LocalDepth d = node->depth();
		node_index_type idx = node->nodeData.nodeIndex;
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) sample += SetDensity( node->children + c );
		if( idx<(node_index_type)sampleMap.size() && sampleMap[idx]!=-1 ) sample += samples[ sampleMap[ idx ] ].sample;

		if( d>=minSplatDepth && d<=maxSplatDepth && sample.weight>0 )
		{
			// The average position of samples accumulated in the node
			Point< Real , Dim > p = sample.data / sample.weight;
			if( d<=pointDepthFunctor( p ) ) _addWeightContribution< true , CoDim >( nodeAllocator , density , node , p , densityKey , sample.weight );
		}
		return sample;
	};
	SetDensity( _spaceRoot );
}

template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int DensityDegree >
typename FEMTree< Dim , Real >::template DensityEstimator< DensityDegree >* FEMTree< Dim , Real >::setDensityEstimator( const std::vector< PointSample >& samples , LocalDepth splatDepth , Real samplesPerNode )
{
	LocalDepth maxDepth = _spaceRoot->maxDepth();
	splatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( splatDepth , maxDepth ) );
	DensityEstimator< DensityDegree > *density = new DensityEstimator< DensityDegree >( splatDepth , CoDim , samplesPerNode );
	this->template updateDensityEstimator< CoDim , DensityDegree >( *density , samples , 0 , splatDepth );

	return density;
}

template< unsigned int Dim , class Real >
template< unsigned int CoDim , unsigned int DensityDegree >
typename FEMTree< Dim , Real >::template DensityEstimator< DensityDegree >* FEMTree< Dim , Real >::setDensityEstimator( const std::vector< PointSample >& samples , LocalDepth splatDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real samplesPerNode )
{
	LocalDepth maxDepth = _spaceRoot->maxDepth();
	splatDepth = std::max< LocalDepth >( 0 , std::min< LocalDepth >( splatDepth , maxDepth ) );
	DensityEstimator< DensityDegree > *density = new DensityEstimator< DensityDegree >( splatDepth , CoDim , samplesPerNode );
	this->template updateDensityEstimator< CoDim , DensityDegree >( *density , samples , 0 , splatDepth , pointDepthFunctor );

	return density;
}

template< unsigned int Dim , class Real >
template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
SparseNodeData< OutData , UIntPack< DataSigs ... > > FEMTree< Dim , Real >::setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData& ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction )
{
	std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction = [&]( InData in , OutData &out , Real &bias )
	{
		if( ConversionFunction( in , out ) )
		{
			bias = BiasFunction( in );
			return true;
		}
		else return false;
	};
	return setInterpolatedDataField( zero , UIntPack< DataSigs ... >() , samples , data , density , minDepth , maxDepth , minDepthCutoff , pointDepthAndWeight , ConversionAndBiasFunction );
}

template< unsigned int Dim , class Real >
template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
SparseNodeData< OutData , UIntPack< DataSigs ... > > FEMTree< Dim , Real >::setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction )
{
	//	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( _spaceRoot );
	SubTreeExtractor subtreeExtractor( _spaceRoot , _depthOffset );

	typedef PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > DensityKey;
	typedef UIntPack< FEMSignature< DataSigs >::Degree ... > DataDegrees;
	typedef PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > > DataKey;
	std::vector< DensityKey > densityKeys( ThreadPool::NumThreads() );
	std::vector<    DataKey >    dataKeys( ThreadPool::NumThreads() );
	bool oneKey = DensityDegree==DataDegrees::Min() && DensityDegree==DataDegrees::Max();
	for( size_t i=0 ; i<densityKeys.size() ; i++ ) densityKeys[i].set( _localToGlobal( maxDepth ) );
	if( !oneKey ) for( size_t i=0 ; i<dataKeys.size() ; i++ ) dataKeys[i].set( _localToGlobal( maxDepth ) );
	std::vector< Real > depthAndWeightSums( ThreadPool::NumThreads() , 0 );
	pointDepthAndWeight.data = Point< Real , 2 >();
	pointDepthAndWeight.weight = 0;
	SparseNodeData< OutData , UIntPack< DataSigs ... > > dataField;
	std::vector< Point< Real , 2 > > pointDepthAndWeightSums( ThreadPool::NumThreads() , Point< Real , 2 >() );
	ThreadPool::ParallelFor( 0 , samples.size() , [&]( unsigned int thread , size_t i )
		{
			DensityKey& densityKey = densityKeys[ thread ];
			DataKey& dataKey = dataKeys[ thread ];
			const ProjectiveData< Point< Real , Dim > , Real >& sample = samples[i].sample;
			if( sample.weight>0 )
			{
				Point< Real , Dim > p = sample.data / sample.weight;
				InData in = data[i] / sample.weight;
				OutData out;

				Real depthBias;
				if( !_InBounds(p) ) MK_WARN( "Point sample is out of bounds" );
				else if( ConversionAndBiasFunction( in , out , depthBias ) )
				{
					depthAndWeightSums[thread] += sample.weight;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Should the next line be commented out?" )
#endif // SHOW_WARNINGS
					out *= sample.weight;
					Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[ thread ] : NULL;
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
					if( density ) pointDepthAndWeightSums[thread] += _splatPointData< true , true , DensityDegree , OutData >( zero , nodeAllocator , *density , minDepthCutoff , p , out , dataField , densityKey , oneKey ? *( (DataKey*)&densityKey ) : dataKey , minDepth , maxDepth , Dim , depthBias ) * sample.weight;
#else // !__GNUC__ || __GNUC__ >=5
					if( density ) pointDepthAndWeightSums[thread] += _splatPointData< true , true , DensityDegree , OutData , DataSigs ... >( zero , nodeAllocator , *density , minDepthCutoff , p , out , dataField , densityKey , oneKey ? *( (DataKey*)&densityKey ) : dataKey , minDepth , maxDepth , Dim , depthBias ) * sample.weight;
#endif // __GNUC__ && __GNUC__ < 5
					else
					{
						Real width = (Real)( 1.0 / ( 1<<maxDepth ) );
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
						#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
						_splatPointData< true , true , OutData >( zero , nodeAllocator , _leaf< true >( nodeAllocator , p , maxDepth ) , p , out / (Real)pow( width , Dim ) , dataField , oneKey ? *( (DataKey*)&densityKey ) : dataKey );
#else // !__GNUC__ || __GNUC__ >=5
						_splatPointData< true , true , OutData , DataSigs ... >( zero , nodeAllocator , _leaf< true >( nodeAllocator , p , maxDepth ) , p , out / (Real)pow( width , Dim ) , dataField , oneKey ? *( (DataKey*)&densityKey ) : dataKey );
#endif // __GNUC__ && __GNUC__ < 5
						pointDepthAndWeightSums[thread] += Point< Real , 2 >( (Real)1. , (Real)maxDepth ) * sample.weight;
					}
				}
			}
		}
	);
	pointDepthAndWeight.data = Point< Real , 2 >();
	for( unsigned int i=0 ; i<depthAndWeightSums.size() ; i++ ) pointDepthAndWeight.data += pointDepthAndWeightSums[i];
	pointDepthAndWeight.weight = 0;
	for( unsigned int i=0 ; i<depthAndWeightSums.size() ; i++ ) pointDepthAndWeight.weight += depthAndWeightSums[i];
	return dataField;
}

template< unsigned int Dim , class Real >
template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
SparseNodeData< OutData , UIntPack< DataSigs ... > > FEMTree< Dim , Real >::setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData& ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction )
{
	std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction = [&]( InData in , OutData &out , Real &bias )
	{
		if( ConversionFunction( in , out ) )
		{
			bias = BiasFunction( in );
			return true;
		}
		else return false;
	};
	return setInterpolatedDataField( zero , UIntPack< DataSigs ... >() , samples , data , density , minDepth , maxDepth , pointDepthFunctor , minDepthCutoff , pointDepthAndWeight , ConversionAndBiasFunction );
}

template< unsigned int Dim , class Real >
template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
SparseNodeData< OutData , UIntPack< DataSigs ... > > FEMTree< Dim , Real >::setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction )
{
	//	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( _spaceRoot );
	SubTreeExtractor subtreeExtractor( _spaceRoot , _depthOffset );

	typedef PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > DensityKey;
	typedef UIntPack< FEMSignature< DataSigs >::Degree ... > DataDegrees;
	typedef PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > > DataKey;
	std::vector< DensityKey > densityKeys( ThreadPool::NumThreads() );
	std::vector<    DataKey >    dataKeys( ThreadPool::NumThreads() );
	bool oneKey = DensityDegree==DataDegrees::Min() && DensityDegree==DataDegrees::Max();
	for( size_t i=0 ; i<densityKeys.size() ; i++ ) densityKeys[i].set( _localToGlobal( maxDepth ) );
	if( !oneKey ) for( size_t i=0 ; i<dataKeys.size() ; i++ ) dataKeys[i].set( _localToGlobal( maxDepth ) );
	std::vector< Real > depthAndWeightSums( ThreadPool::NumThreads() , 0 );
	pointDepthAndWeight.data = Point< Real , 2 >();
	pointDepthAndWeight.weight = 0;
	SparseNodeData< OutData , UIntPack< DataSigs ... > > dataField;
	std::vector< Point< Real , 2 > > pointDepthAndWeightSums( ThreadPool::NumThreads() , Point< Real , 2 >() );
	ThreadPool::ParallelFor( 0 , samples.size() , [&]( unsigned int thread , size_t i )
		{
			DensityKey& densityKey = densityKeys[ thread ];
			DataKey& dataKey = dataKeys[ thread ];
			const ProjectiveData< Point< Real , Dim > , Real >& sample = samples[i].sample;
			if( sample.weight>0 )
			{
				Point< Real , Dim > p = sample.data / sample.weight;
				InData in = data[i] / sample.weight;
				OutData out;

				Real depthBias;
				if( !_InBounds(p) ) MK_WARN( "Point sample is out of bounds" );
				else if( ConversionAndBiasFunction( in , out , depthBias ) )
				{
					depthAndWeightSums[thread] += sample.weight;
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Should the next line be commented out?" )
#endif // SHOW_WARNINGS
					out *= sample.weight;
					Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[ thread ] : NULL;
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
					if( density ) pointDepthAndWeightSums[thread] += _splatPointData< true , true , DensityDegree , OutData >( zero , nodeAllocator , *density , minDepthCutoff , p , out , dataField , densityKey , oneKey ? *( (DataKey*)&densityKey ) : dataKey , minDepth , pointDepthFunctor , Dim , depthBias ) * sample.weight;
#else // !__GNUC__ || __GNUC__ >=5
					if( density ) pointDepthAndWeightSums[thread] += _splatPointData< true , true , DensityDegree , OutData , DataSigs ... >( zero , nodeAllocator , *density , minDepthCutoff , p , out , dataField , densityKey , oneKey ? *( (DataKey*)&densityKey ) : dataKey , minDepth , pointDepthFunctor , Dim , depthBias ) * sample.weight;
#endif // __GNUC__ && __GNUC__ < 5
					else
					{
						Real width = (Real)( 1.0 / ( 1<<maxDepth ) );
#if defined( __GNUC__ ) && __GNUC__ < 5
#ifdef SHOW_WARNINGS
						#warning "you've got me gcc version<5"
#endif // SHOW_WARNINGS
						_splatPointData< true , true , OutData >( zero , nodeAllocator , _leaf< true >( nodeAllocator , p , pointDepthFunctor ) , p , out / (Real)pow( width , Dim ) , dataField , oneKey ? *( (DataKey*)&densityKey ) : dataKey );
#else // !__GNUC__ || __GNUC__ >=5
						_splatPointData< true , true , OutData , DataSigs ... >( zero , nodeAllocator , _leaf< true >( nodeAllocator , p , pointDepthFunctor ) , p , out / (Real)pow( width , Dim ) , dataField , oneKey ? *( (DataKey*)&densityKey ) : dataKey );
#endif // __GNUC__ && __GNUC__ < 5
						pointDepthAndWeightSums[thread] += Point< Real , 2 >( (Real)1 , (Real)maxDepth ) * sample.weight;
					}
				}
			}
		}
	);
	pointDepthAndWeight.data = Point< Real , 2 >();
	for( unsigned int i=0 ; i<pointDepthAndWeightSums.size() ; i++ ) pointDepthAndWeight.data += pointDepthAndWeightSums[i];
	pointDepthAndWeight.weight = 0;
	for( unsigned int i=0 ; i<depthAndWeightSums.size() ; i++ ) pointDepthAndWeight.weight += depthAndWeightSums[i];
	return dataField;
}

template< unsigned int Dim , class Real >
template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data >
SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > FEMTree< Dim , Real >::setExtrapolatedDataField( Data zero , const std::vector< PointSample >& samples , const std::vector< Data >& sampleData , const DensityEstimator< DensityDegree >* density , bool nearest )
{
	return this->template setExtrapolatedDataField< DataSig , CreateNodes , DensityDegree , Data >( zero , samples.size() , [&]( size_t i ) -> const PointSample & { return samples[i]; } , [&]( size_t i ) -> const Data & { return sampleData[i]; } , density , nearest );
}

template< unsigned int Dim , class Real >
template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data >
void FEMTree< Dim , Real >::updateExtrapolatedDataField( Data zero , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > &dataField , const std::vector< PointSample >& samples , const std::vector< Data >& sampleData , const DensityEstimator< DensityDegree >* density , bool nearest )
{
	return this->template updateExtrapolatedDataField< DataSig , CreateNodes , DensityDegree , Data >( zero , dataField , samples.size() , [&]( size_t i ) -> const PointSample & { return samples[i]; } , [&]( size_t i ) -> const Data & { return sampleData[i]; } , density , nearest );
}

template< unsigned int Dim , class Real >
template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data , class SampleFunctor /* = std::function< const PointSample & (size_t) >*/ , class SampleDataFunctor /* = std::function< const Data & (size_t) > */ >
SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > FEMTree< Dim , Real >::setExtrapolatedDataField( Data zero , size_t sampleNum , SampleFunctor sampleFunctor , SampleDataFunctor sampleDataFunctor , const DensityEstimator< DensityDegree >* density , bool nearest )
{
	SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > dataField;
	this->template updateExtrapolatedDataField< DataSig , CreateNodes , DensityDegree , Data >( zero , dataField , sampleNum , sampleFunctor , sampleDataFunctor , density , nearest );
	return dataField;
}

template< unsigned int Dim , class Real >
template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data , class SampleFunctor /* = std::function< const PointSample & (size_t) >*/ , class SampleDataFunctor /* = std::function< const Data & (size_t) > */ >
void FEMTree< Dim , Real >::updateExtrapolatedDataField( Data zero , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > &dataField , size_t sampleNum , SampleFunctor sampleFunctor , SampleDataFunctor sampleDataFunctor , const DensityEstimator< DensityDegree >* density , bool nearest )
{
	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;
	LocalDepth maxDepth = _spaceRoot->maxDepth();

	if constexpr( CreateNodes )
	{
		PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > densityKey;
		PointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > > dataKey;
		densityKey.set( _localToGlobal( maxDepth ) ) , dataKey.set( _localToGlobal( maxDepth ) );
		for( node_index_type i=0 ; i<(node_index_type)sampleNum ; i++ )
		{
			const PointSample &sampleAndNode = sampleFunctor(i);
			const ProjectiveData< Point< Real , Dim > , Real >& sample = sampleAndNode.sample;
			const Data& data = sampleDataFunctor(i);
			Point< Real , Dim > p = sample.weight==0 ? sample.data : sample.data / sample.weight;
			if( !_InBounds(p) )
			{
				MK_WARN( "Point is out of bounds" );
				continue;
			}
			if( nearest ) _nearestMultiSplatPointData< DensityDegree >( ProjectiveData< Data , Real >( zero ) , density , (FEMTreeNode*)sampleAndNode.node , p , ProjectiveData< Data , Real >( data , sample.weight ) , dataField , densityKey , 2 );
			else          _multiSplatPointData< CreateNodes , false , DensityDegree >( ProjectiveData< Data , Real >( zero ) , nodeAllocator , density , (FEMTreeNode*)sampleAndNode.node , p , ProjectiveData< Data , Real >( data , sample.weight ) , dataField , densityKey , dataKey , 2 );
		}
	}
	else
	{
		std::vector< PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > > densityKeys( ThreadPool::NumThreads() );
		std::vector< PointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > > > dataKeys( ThreadPool::NumThreads() );
		for( unsigned int i=0 ; i<ThreadPool::NumThreads() ; i++ ) densityKeys[i].set( _localToGlobal( maxDepth ) ) , dataKeys[i].set( _localToGlobal( maxDepth ) );

		ThreadPool::ParallelFor( 0 , sampleNum , [&]( unsigned int thread , size_t i ) {
			PointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > &densityKey = densityKeys[thread];
			PointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > > &dataKey = dataKeys[thread];

			const PointSample &sampleAndNode = sampleFunctor(i);
			const ProjectiveData< Point< Real , Dim > , Real >& sample = sampleAndNode.sample;
			const Data& data = sampleDataFunctor(i);
			Point< Real , Dim > p = sample.weight==0 ? sample.data : sample.data / sample.weight;
			if( !_InBounds(p) ) MK_WARN( "Point is out of bounds" );
			else
			{
				if( nearest ) _nearestMultiSplatPointData< DensityDegree >( ProjectiveData< Data , Real >( zero ) , density , (FEMTreeNode*)sampleAndNode.node , p , ProjectiveData< Data , Real >( data , sample.weight ) , dataField , densityKey , 2 );
				else          _multiSplatPointData< CreateNodes , false , DensityDegree >( ProjectiveData< Data , Real >( zero ) , nodeAllocator , density , (FEMTreeNode*)sampleAndNode.node , p , ProjectiveData< Data , Real >( data , sample.weight ) , dataField , densityKey , dataKey , 2 );
			}
		} );
	}
}

template< unsigned int Dim , class Real >
template< unsigned int MaxDegree >
void FEMTree< Dim , Real >::_supportApproximateProlongation( void )
{
	// Refine the tree so that if an active element exists @{d}, all supporting elements exist @{d-1}
	const int OverlapRadius = -BSplineOverlapSizes< MaxDegree , MaxDegree >::OverlapStart;
	typedef typename FEMTreeNode::template NeighborKey< IsotropicUIntPack< Dim , OverlapRadius > , IsotropicUIntPack< Dim , OverlapRadius > > NeighborKey;

	std::vector< NeighborKey > neighborKeys( ThreadPool::NumThreads() );
	for( int i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( _localToGlobal( _maxDepth-1 ) );

	for( LocalDepth d=_maxDepth-1 ; d>_baseDepth ; d-- )
	{
		// Compute the set of nodes at depth d that have (non-ghost) children at depth d+1.
		std::vector< FEMTreeNode* > nodes;
		_tree.processNodes( [&]( FEMTreeNode *node ){ if( _localDepth( node )==d && IsActiveNode( node->children ) ) nodes.push_back( node ) ; return _localDepth(node)<d; } );

		// Make sure that all finite elements whose support overlaps the support of the finite elements indexed by those nodes are in the tree.
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] This may be overkill as we only need to check if the support overlaps the support of the children" )
#endif // SHOW_WARNINGS
		ThreadPool::ParallelFor( 0 , nodes.size() , [&]( unsigned int thread , size_t i )
			{
				NeighborKey& neighborKey = neighborKeys[ thread ];
				FEMTreeNode *node = nodes[i];

				// Create the neighbors if they are not already in the tree
				neighborKey.template getNeighbors< true , true >( node , nodeAllocators.size() ? nodeAllocators[ thread ] : NULL , _nodeInitializer );

				// Mark the neighbors as active
				Pointer( FEMTreeNode* ) nodes = neighborKey.neighbors[ _localToGlobal(d) ].neighbors().data;
				unsigned int size = neighborKey.neighbors[ _localToGlobal(d) ].neighbors.Size();
				for( unsigned int i=0 ; i<size ; i++ ) SetGhostFlag( nodes[i] , false );
			}
		);
	}
}

template< unsigned int Dim , typename Real >
template< unsigned int SystemDegree >
void FEMTree< Dim , Real >::_markNonBaseDirichletElements( void )
{
	const int LeftSupportRadius = -BSplineSupportSizes< SystemDegree >::SupportStart;
	const int RightSupportRadius = BSplineSupportSizes< SystemDegree >::SupportEnd;
	typedef typename FEMTreeNode::template NeighborKey< IsotropicUIntPack< Dim , LeftSupportRadius > , IsotropicUIntPack< Dim , RightSupportRadius > > SupportKey;
	typedef StaticWindow< FEMTreeNode * , IsotropicUIntPack< Dim , LeftSupportRadius + RightSupportRadius + 1 > > NeighborLeaves;

	std::vector< NeighborLeaves > neighborLeaves( ThreadPool::NumThreads() );
	std::vector< SupportKey > supportKeys( ThreadPool::NumThreads() );
	for( int i=0 ; i<supportKeys.size() ; i++ ) supportKeys[i].set( _localToGlobal( _maxDepth ) );

	// Get the list of nodes @{_baseDepth)
	std::vector< FEMTreeNode * > baseNodes;
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( _localDepth( node )==_baseDepth ) baseNodes.push_back( node ); } );

	// Process the sub-tree rooted at the node:
	// -- For each non-ghost node in the sub-tree check if the finite element associated to the node is supported on a Dirichlet node.
	//    If it is, mark the node as a Dirichlet element node.
	std::function< void ( FEMTreeNode * , SupportKey & , NeighborLeaves & ) > ProcessSubTree = [&]( FEMTreeNode *node , SupportKey &supportKey , NeighborLeaves &neighborLeaves )
	{
		if( !node->nodeData.getGhostFlag() )
		{
			supportKey.setLeafNeighbors( node , neighborLeaves );
			bool hasDirichletNeighbor = false;
			for( unsigned int i=0 ; i<NeighborLeaves::Size() ; i++ ) if( neighborLeaves.data[i] && neighborLeaves.data[i]->nodeData.getDirichletNodeFlag() ) hasDirichletNeighbor = true;
			node->nodeData.setDirichletElementFlag( hasDirichletNeighbor );

			if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) ProcessSubTree( node->children+c , supportKey , neighborLeaves );
		}
	};

	ThreadPool::ParallelFor( 0 , baseNodes.size() , [&]( unsigned int t , size_t i ){ ProcessSubTree( baseNodes[i] , supportKeys[t] , neighborLeaves[t] ); } );
}

template< unsigned int Dim , typename Real >
template< unsigned int SystemDegree >
void FEMTree< Dim , Real >::_markBaseDirichletElements( void )
{
	const int LeftSupportRadius = BSplineSupportSizes< SystemDegree >::SupportEnd;
	const int RightSupportRadius = -BSplineSupportSizes< SystemDegree >::SupportStart;

	typedef typename FEMTreeNode::template NeighborKey< IsotropicUIntPack< Dim , LeftSupportRadius > , IsotropicUIntPack< Dim , RightSupportRadius > > SupportKey;
	std::vector< SupportKey > supportKeys( ThreadPool::NumThreads() );
	for( int i=0 ; i<supportKeys.size() ; i++ ) supportKeys[i].set( _localToGlobal( _baseDepth ) );

	std::vector< FEMTreeNode* > nodes;
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( _localDepth( node )==_baseDepth && node->nodeData.getDirichletNodeFlag() ) nodes.push_back( node ) ; return _localDepth(node)<_baseDepth; } );

	ThreadPool::ParallelFor( 0 , nodes.size() , [&]( unsigned int thread , size_t i )
		{
			SupportKey &supportKey = supportKeys[ thread ];
			FEMTreeNode *node = nodes[i];
			supportKey.getNeighbors( node );
			for( LocalDepth d=0 ; d<=_baseDepth ; d++ )
			{
				Pointer( FEMTreeNode* ) _nodes = supportKey.neighbors[ _localToGlobal(d) ].neighbors().data;
				unsigned int size = supportKey.neighbors[ _localToGlobal(d) ].neighbors.Size();
				for( unsigned int i=0 ; i<size ; i++ ) if( _nodes[i] ) SetGhostFlag( _nodes[i] , false ) , _nodes[i]->nodeData.setDirichletElementFlag( true );
			}
		} );
}

template< unsigned int Dim , class Real >
template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
std::vector< node_index_type > FEMTree< Dim , Real >::finalizeForMultigrid( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data )
{
	auto isDirichletLeafFunctor = []( const FEMTreeNode * ){ return false; };
	return _finalizeForMultigrid< false , MaxDegree , SystemDegree >( baseDepth , addNodeFunctor , hasDataFunctor , isDirichletLeafFunctor , interpolationInfos , data );
}

template< unsigned int Dim , class Real >
template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename IsDirichletLeafFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
std::vector< node_index_type > FEMTree< Dim , Real >::finalizeForMultigridWithDirichlet( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , const IsDirichletLeafFunctor isDirichletLeafFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data )
{
	return _finalizeForMultigrid< true , MaxDegree , SystemDegree >( baseDepth , addNodeFunctor , hasDataFunctor , isDirichletLeafFunctor , interpolationInfos , data );
}


template< unsigned int Dim , class Real >
template< bool HasDirichlet , unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename IsDirichletLeafFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
std::vector< node_index_type > FEMTree< Dim , Real >::_finalizeForMultigrid( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , const IsDirichletLeafFunctor isDirichletLeafFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data )
{
	std::function< void ( FEMTreeNode * ) > pushFullDirichletFlag = [&]( FEMTreeNode *node )
	{
		if( node->children )
		{
			if( node->nodeData.getDirichletNodeFlag() )
			{
				for( int c=0 ; c<(1<<Dim) ; c++ ) node->children[c].nodeData.setDirichletNodeFlag( true );
				node->nodeData.setDirichletNodeFlag( false );
			}
			for( int c=0 ; c<(1<<Dim) ; c++ ) pushFullDirichletFlag( node->children + c );
		}
	};

	std::function< bool ( FEMTreeNode * ) > pullPartialDirichletFlag = [&]( FEMTreeNode *node )
	{
		if( node->children )
		{
			bool childDirichlet = false;
			for( int c=0 ; c<(1<<Dim) ; c++ ) childDirichlet |= pullPartialDirichletFlag( node->children + c );
			node->nodeData.setDirichletNodeFlag( childDirichlet );
		}
		return node->nodeData.getDirichletNodeFlag();
	};

	std::function< void ( FEMTreeNode * , node_index_type , bool ) > pushDirichletFlag = [&]( FEMTreeNode *node , node_index_type newNodeIndex , bool isDirichletNode )
	{
		if( node->nodeData.nodeIndex>=newNodeIndex ) node->nodeData.setDirichletNodeFlag( isDirichletNode );
		isDirichletNode = node->nodeData.getDirichletNodeFlag();
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) pushDirichletFlag( node->children+c , newNodeIndex , isDirichletNode );
	};

	_baseDepth = baseDepth;
	// Expand the tree to ensure supporting finite elements can be indexed by octree nodes

	Allocator< FEMTreeNode > *nodeAllocator = nodeAllocators.size() ? nodeAllocators[0] : NULL;

	while( _localInset( 0 ) + BSplineEvaluationData< FEMDegreeAndBType< MaxDegree >::Signature >::Begin( 0 )<0 || _localInset( 0 ) + BSplineEvaluationData< FEMDegreeAndBType< MaxDegree >::Signature >::End( 0 )>(1<<_depthOffset) )
	{
		//                       +-+-+-+-+-+-+-+-+
		//                       | | | | | | | | |
		//                       +-+-+-+-+-+-+-+-+
		//                       | | | | | | | | |
		//          +-+-+-+-+    +-+-+-+-+-+-+-+-+
		//          | | | | |    | | | | | | | | |
		// +-+-+    +-+-+-+-+    +-+-+-+-+-+-+-+-+
		// |*| |    | | | | |    | | | | | | | | |
		// +-o-+ -> +-+-o-+-+ -> +-+-+-+-o-+-+-+-+
		// | | |    | | |*| |    | | | | |*| | | |
		// +-+-+    +-+-+-+-+    +-+-+-+-+-+-+-+-+
		//          | | | | |    | | | | | | | | |
		//          +-+-+-+-+    +-+-+-+-+-+-+-+-+
		//                       | | | | | | | | |
		//                       +-+-+-+-+-+-+-+-+
		//                       | | | | | | | | |
		//                       +-+-+-+-+-+-+-+-+

		// Insert a new brood between the root of the tree and its (old) children:
		// -- Make the old children sit off the last node of the brood
		// -- Swap the children sitting off the last node of the old children to the first node of the old children
		FEMTreeNode *oldChildren = _tree.children;
		FEMTreeNode *newChildren = FEMTreeNode::NewBrood( nodeAllocator , _nodeInitializer );
		if( !oldChildren ) MK_THROW( "Expected children" );
		{
			if( oldChildren[(1<<Dim)-1].children )
			{
				for( int c=0 ; c<(1<<Dim) ; c++ ) oldChildren[(1<<Dim)-1].children[c].parent = oldChildren;
				oldChildren[0].children = oldChildren[(1<<Dim)-1].children;
				oldChildren[(1<<Dim)-1].children = NULL;
			}
			for( int c=0 ; c<(1<<Dim) ; c++ ) oldChildren[c].parent = newChildren+(1<<Dim)-1;
			newChildren[(1<<Dim)-1].children = oldChildren;
		}

		for( int c=0 ; c<(1<<Dim) ; c++ ) newChildren[c].parent = &_tree;
		_tree.children = newChildren;

		_depthOffset++;
	}
	_init();

	_maxDepth = _spaceRoot->maxDepth();

	// Mark leaf nodes that are Dirichlet constraints so they do not get clipped out.
	// Need to do this before introducing new nodes into the tree  (since isDirichletLeaf depends on the structure at input).
	if constexpr( HasDirichlet ) _spaceRoot->processLeaves( [&]( FEMTreeNode *leaf ){ leaf->nodeData.setDirichletNodeFlag( isDirichletLeafFunctor( leaf ) ); } );

	// Make the low-resolution part of the tree be complete
	_setFullDepth< false >( IsotropicUIntPack< Dim , MaxDegree >() , nodeAllocator , _baseDepth );
	_refine< false >( IsotropicUIntPack< Dim , MaxDegree >() , nodeAllocator , addNodeFunctor );

	if constexpr( HasDirichlet )
	{
		// Mark new leaf nodes
		pushFullDirichletFlag( _spaceRoot );

		// Pull the Dirichlet designator from the leaves so that nodes are now marked if they contain (possibly partial Dirichlet constraints
		pullPartialDirichletFlag( _spaceRoot );

		// Use the node Dirichlet designators to set the coarser finite element Dirichlet designators
		_markBaseDirichletElements< SystemDegree >();
	}

	// Clear all the flags and make everything that is not low-res a ghost node, and clip the tree
	auto _addNodeFunctor = [&]( const FEMTreeNode *node )
	{
		LocalDepth d ; LocalOffset off;
		_localDepthAndOffset( node , d , off );
		return addNodeFunctor( d , off );
	};

	auto nodeFunctor = [&]( FEMTreeNode *node )
	{
		if constexpr( HasDirichlet ) node->nodeData.flags &= ( FEMTreeNodeData::SCRATCH_FLAG | FEMTreeNodeData::DIRICHLET_NODE_FLAG | FEMTreeNodeData::DIRICHLET_ELEMENT_FLAG );
		else                         node->nodeData.flags &= ( FEMTreeNodeData::SCRATCH_FLAG );
		SetGhostFlag( node , !_addNodeFunctor( node ) && _localDepth( node )>_baseDepth );
	};
	_tree.processNodes( nodeFunctor );

	// Clip off nodes that not have data and do not contain geometry or Dirichlet constraints below the exactDepth
	if constexpr( HasDirichlet ) _clipTree( [&]( const FEMTreeNode *node ){ return _addNodeFunctor(node) || hasDataFunctor(node) || ( ( node->nodeData.getDirichletNodeFlag() || node->nodeData.getDirichletElementFlag() ) && _localDepth(node)<=_baseDepth ); } , _baseDepth );
	else                         _clipTree( [&]( const FEMTreeNode *node ){ return _addNodeFunctor(node) || hasDataFunctor(node); } , _baseDepth );

	// It is possible for the tree to have become shallower after clipping
	_maxDepth = _tree.maxDepth() - _depthOffset;

	node_index_type oldNodeCount = _nodeCount;

	// Refine the node so that finite elements @{depth-1} whose support overlaps finite elements @{depth} are in the tree
	_supportApproximateProlongation< MaxDegree >();

	// Mark new leaf nodes
	if constexpr( HasDirichlet )
	{
		pushDirichletFlag( _spaceRoot , oldNodeCount , _spaceRoot->nodeData.getDirichletNodeFlag() );
		_markNonBaseDirichletElements< SystemDegree >();
	}
	_markNonBaseDirichletElements< SystemDegree >();

	return setSortedTreeNodes( interpolationInfos , data );
}

template< unsigned int Dim , class Real >
template< typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
std::vector< node_index_type > FEMTree< Dim , Real >::setSortedTreeNodes( std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data )
{
	_maxDepth = _tree.maxDepth() - _depthOffset;
	std::vector< node_index_type > map;
	_sNodes.reset( _tree , map );
	_setSpaceValidityFlags();
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( !IsActiveNode( node ) ) node->nodeData.nodeIndex = -1; } );
	_reorderDenseOrSparseNodeData< 0 >( GetPointer( map ) , _sNodes.size() , data );
	_reorderInterpolationInfo< 0 >( GetPointer( map ) , _sNodes.size() , interpolationInfos );
	memset( _femSigs1 , -1 , sizeof( _femSigs1 ) );
	memset( _femSigs2 , -1 , sizeof( _femSigs2 ) );
	return map;
}

template< unsigned int Dim , class Real >
template< class ... DenseOrSparseNodeData >
void FEMTree< Dim , Real >::resetIndices( std::tuple< DenseOrSparseNodeData *... > data )
{
	std::vector< node_index_type > map;
	_sNodes.reset( _tree , map );
	_setSpaceValidityFlags();
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( !IsActiveNode< Dim >( node ) ) node->nodeData.nodeIndex = -1; } );
	_reorderDenseOrSparseNodeData< 0 >( GetPointer( map ) , _sNodes.size() , data );
}

template< unsigned int Dim , class Real >
template< typename PruneChildrenFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
void FEMTree< Dim , Real >::pruneChildren( const PruneChildrenFunctor pruneChildren , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data )
{
	_tree.pruneChildren( pruneChildren , false );
	std::vector< node_index_type > map;
	_sNodes.reset( _tree , map );
	_setSpaceValidityFlags();
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( !IsActiveNode< Dim >( node ) ) node->nodeData.nodeIndex = -1; } );
	_reorderDenseOrSparseNodeData< 0 >( GetPointer( map ) , _sNodes.size() , data );
	_reorderInterpolationInfo< 0 >( GetPointer( map ) , _sNodes.size() , interpolationInfos );
}

template< unsigned int Dim , class Real >
void FEMTree< Dim , Real >::_setSpaceValidityFlags( void ) const
{
	const unsigned char MASK = ~( FEMTreeNodeData::SPACE_FLAG );
	ThreadPool::ParallelFor( 0 , _sNodes.size() , [&]( unsigned int , size_t i )
		{
			_sNodes.treeNodes[i]->nodeData.flags &= MASK;
			if( isValidSpaceNode( _sNodes.treeNodes[i] ) ) _sNodes.treeNodes[i]->nodeData.flags |= FEMTreeNodeData::SPACE_FLAG;
		}
	);
}
template< unsigned int Dim , class Real >
template< unsigned int ... FEMSigs1 >
void FEMTree< Dim , Real >::_setFEM1ValidityFlags( UIntPack< FEMSigs1 ... > ) const
{
	bool needToReset;
	unsigned int femSigs1[] = { FEMSigs1 ... };
	{
		static std::mutex m;
		std::lock_guard< std::mutex > lock( m );
		needToReset = memcmp( femSigs1 , _femSigs1 , sizeof( _femSigs1 ) )!=0;
		if( needToReset ) memcpy( _femSigs1 , femSigs1 , sizeof( _femSigs1 ) );
	}
	if( needToReset )
		for( node_index_type i=0 ; i<(node_index_type)_sNodes.size() ; i++ )
		{
			const unsigned char MASK = ~( FEMTreeNodeData::FEM_FLAG_1 );
			_sNodes.treeNodes[i]->nodeData.flags &= MASK;
			if( isValidFEMNode( UIntPack< FEMSigs1 ... >() , _sNodes.treeNodes[i] ) ) _sNodes.treeNodes[i]->nodeData.flags |= FEMTreeNodeData::FEM_FLAG_1;
		}

}
template< unsigned int Dim , class Real >
template< unsigned int ... FEMSigs2 >
void FEMTree< Dim , Real >::_setFEM2ValidityFlags( UIntPack< FEMSigs2 ... > ) const
{
	bool needToReset;
	unsigned int femSigs2[] = { FEMSigs2 ... };
	{
		static std::mutex m;
		std::lock_guard< std::mutex > lock(m);
		needToReset = memcmp( femSigs2 , _femSigs2 , sizeof( _femSigs2 ) )!=0;
		if( needToReset ) memcpy( _femSigs2 , femSigs2 , sizeof( _femSigs2 ) );
	}
	if( needToReset )
		for( node_index_type i=0 ; i<(node_index_type)_sNodes.size() ; i++ )
		{
			const unsigned char MASK = ~( FEMTreeNodeData::FEM_FLAG_2 );
			_sNodes.treeNodes[i]->nodeData.flags &= MASK;
			if( isValidFEMNode( UIntPack< FEMSigs2 ... >() , _sNodes.treeNodes[i] ) ) _sNodes.treeNodes[i]->nodeData.flags |= FEMTreeNodeData::FEM_FLAG_2;
		}
}

template< unsigned int Dim , class Real >
template< class HasDataFunctor >
void FEMTree< Dim , Real >::_clipTree( const HasDataFunctor& f , LocalDepth fullDepth )
{
	std::vector< FEMTreeNode * > regularNodes;
	_tree.processNodes( [&]( FEMTreeNode *node ){ if( _localDepth( node )==fullDepth ) regularNodes.push_back( node ) ; return _localDepth( node )<fullDepth; } );

	// Get the data status of each node
	// [NOTE] Have to use an array of chars instead of bools because the latter is not thread safe
	node_index_type sz = nodeCount();
	Pointer( char ) nodeHasData = NewPointer< char >( sz );
	for( node_index_type i=0 ; i<sz ; i++ ) nodeHasData[i] = 0;
	ThreadPool::ParallelFor( 0 , regularNodes.size() , [&]( unsigned int , size_t i )
		{
			regularNodes[i]->processNodes( [&]( FEMTreeNode *node ){ if( node->nodeData.nodeIndex!=-1 ) nodeHasData[node->nodeData.nodeIndex] = f( node ) ? 1 : 0; } );
		} );

	// Pull the data status from the leaves
	std::function< char ( const FEMTreeNode * ) > PullHasDataFromChildren = [&]( const FEMTreeNode *node )
	{
		if( node->nodeData.nodeIndex==-1 ) return (char)0;
		char hasData = nodeHasData[node->nodeData.nodeIndex];
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) hasData |= PullHasDataFromChildren( node->children+c );
		nodeHasData[node->nodeData.nodeIndex] = hasData;
		return hasData;
	};

	ThreadPool::ParallelFor( 0 , regularNodes.size() , [&]( unsigned int , size_t i ){ PullHasDataFromChildren( regularNodes[i] ); } );

	// Mark all children of a node as ghost if none of them have data
	ThreadPool::ParallelFor( 0 , regularNodes.size() , [&]( unsigned int , size_t i )
		{
			auto nodeFunctor = [&]( FEMTreeNode *node )
			{
				if( node->children )
				{
					char childHasData = 0;
					for( int c=0 ; c<(1<<Dim) ; c++ ) if( node->children[c].nodeData.nodeIndex!=-1 ) childHasData |= nodeHasData[node->children[c].nodeData.nodeIndex];
					for( int c=0 ; c<(1<<Dim) ; c++ ) SetGhostFlag< Dim >( node->children+c , !childHasData );
				}
			};
			regularNodes[i]->processNodes( nodeFunctor );
		} );
	DeletePointer( nodeHasData );
}

template< unsigned int Dim , class Real >
template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
void FEMTree< Dim , Real >::_ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , bool noRescale )
{
	_sampleSpan.resize( tree.nodesSize() );
	ThreadPool::ParallelFor( 0 , tree.nodesSize() , [&]( unsigned int , size_t i ){ _sampleSpan[i] = std::pair< node_index_type , node_index_type >( 0 , 0 ); } );
	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) ) _sampleSpan[ leaf->nodeData.nodeIndex ].second++;
	}
	_iData.resize( samples.size() );

	std::function< void ( FEMTreeNode* , node_index_type & ) > SetRange = [&] ( FEMTreeNode* node , node_index_type &start )
	{
		std::pair< node_index_type , node_index_type >& span = _sampleSpan[ node->nodeData.nodeIndex ];
		if( tree._isValidSpaceNode( node->children ) )
		{
			for( int c=0 ; c<(1<<Dim) ; c++ ) SetRange( node->children + c , start );
			span.first  = _sampleSpan[ node->children[0           ].nodeData.nodeIndex ].first;
			span.second = _sampleSpan[ node->children[ (1<<Dim)-1 ].nodeData.nodeIndex ].second;
		}
		else
		{
			span.second = start + span.second - span.first;
			span.first = start;
			start += span.second - span.first;
		}
	};

	node_index_type start = 0;
	SetRange( tree._spaceRoot , start );
	auto nodeFunctor = [&]( FEMTreeNode *node )
	{
		if( tree._isValidSpaceNode( node ) && !tree._isValidSpaceNode( node->children ) ) _sampleSpan[ node->nodeData.nodeIndex ].second = _sampleSpan[ node->nodeData.nodeIndex ].first;
	};
	tree._spaceRoot->processNodes( nodeFunctor );

	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) )
		{
			const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
			DualPointAndDataInfo< Dim , Real , Data , T , PointD >& _pData = _iData[ _sampleSpan[ leaf->nodeData.nodeIndex ].second++ ];
			_pData.pointInfo.position = pData.data;
			_pData.pointInfo.weight = pData.weight;
			_pData.pointInfo.dualValues = _constraintDual( pData.data/pData.weight , sampleData[i]/pData.weight ) * pData.weight;
			_pData.data = sampleData[i];
		}
	}

	ThreadPool::ParallelFor( 0 , _iData.size() , [&]( unsigned int , size_t i  )
		{
			Real w = _iData[i].pointInfo.weight;
			_iData[i] /= w;
			if( noRescale ) _iData[i].pointInfo.weight = w;
			else            _iData[i].pointInfo.weight = w * ( 1<<tree._maxDepth );
			_iData[i].pointInfo.dualValues *= _iData[i].pointInfo.weight;
		} );
}

template< unsigned int Dim , class Real >
template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
void FEMTree< Dim , Real >::ExactPointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >::_init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , bool noRescale )
{
	_sampleSpan.resize( tree._nodeCount );
	ThreadPool::ParallelFor( 0 , tree.nodesSize() , [&]( unsigned int , size_t i ){ _sampleSpan[i] = std::pair< node_index_type , node_index_type >( 0 , 0 ); } );
	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) ) _sampleSpan[ leaf->nodeData.nodeIndex ].second++;
	}
	_iData.resize( samples.size() );

	std::function< void ( FEMTreeNode* , node_index_type & ) > SetRange = [&] ( FEMTreeNode* node , node_index_type &start )
	{
		std::pair< node_index_type , node_index_type >& span = _sampleSpan[ node->nodeData.nodeIndex ];
		if( tree._isValidSpaceNode( node->children ) )
		{
			for( int c=0 ; c<(1<<Dim) ; c++ ) SetRange( node->children + c , start );
			span.first  = _sampleSpan[ node->children[0           ].nodeData.nodeIndex ].first;
			span.second = _sampleSpan[ node->children[ (1<<Dim)-1 ].nodeData.nodeIndex ].second;
		}
		else
		{
			span.second = start + span.second - span.first;
			span.first = start;
			start += span.second - span.first;
		}
	};

	node_index_type start=0;
	SetRange( tree._spaceRoot , start );

	auto nodeFunctor = [&]( FEMTreeNode *node )
	{
		if( tree._isValidSpaceNode( node ) && !tree._isValidSpaceNode( node->children ) ) _sampleSpan[ node->nodeData.nodeIndex ].second = _sampleSpan[ node->nodeData.nodeIndex ].first;
	};
	tree._spaceRoot->processNodes( nodeFunctor );

	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) )
		{
			const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
			DualPointInfo< Dim , Real , T , PointD >& _pData = _iData[ _sampleSpan[ leaf->nodeData.nodeIndex ].second++ ];
			_pData.position = pData.data;
			_pData.dualValues = _constraintDual( pData.data/pData.weight ) * pData.weight;
			_pData.weight = pData.weight;
		}
	}

	ThreadPool::ParallelFor( 0 , _iData.size() , [&]( unsigned int , size_t i )
		{
			Real w = _iData[i].weight;
			_iData[i] /= w;
			if( noRescale ) _iData[i].weight = w;
			else            _iData[i].weight = w * ( 1<<tree._maxDepth );
			_iData[i].dualValues *= _iData[i].weight;
		} );
}

template< unsigned int Dim , class Real >
template< unsigned int PointD , typename ConstraintDual , typename SystemDual >
void FEMTree< Dim , Real >::ExactPointInterpolationInfo< double , PointD , ConstraintDual , SystemDual >::_init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , bool noRescale )
{
	_sampleSpan.resize( tree._nodeCount );
	ThreadPool::ParallelFor( 0 , tree.nodesSize() , [&]( unsigned int , size_t i ){ _sampleSpan[i] = std::pair< node_index_type , node_index_type >( 0 , 0 ); } );
	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) ) _sampleSpan[ leaf->nodeData.nodeIndex ].second++;
	}
	_iData.resize( samples.size() );

	std::function< void ( FEMTreeNode* , node_index_type & ) > SetRange = [&] ( FEMTreeNode *node , node_index_type &start )
	{
		std::pair< node_index_type , node_index_type >& span = _sampleSpan[ node->nodeData.nodeIndex ];
		if( tree._isValidSpaceNode( node->children ) )
		{
			for( int c=0 ; c<(1<<Dim) ; c++ ) SetRange( node->children + c , start );
			span.first  = _sampleSpan[ node->children[0           ].nodeData.nodeIndex ].first;
			span.second = _sampleSpan[ node->children[ (1<<Dim)-1 ].nodeData.nodeIndex ].second;
		}
		else
		{
			span.second = start + span.second - span.first;
			span.first = start;
			start += span.second - span.first;
		}
	};

	node_index_type start = 0;
	SetRange( tree._spaceRoot , start );
	auto nodeFunctor = [&]( FEMTreeNode *node )
	{
		if( tree._isValidSpaceNode( node ) && !tree._isValidSpaceNode( node->children ) ) _sampleSpan[ node->nodeData.nodeIndex ].second = _sampleSpan[ node->nodeData.nodeIndex ].first;
	};
	tree._spaceRoot->processNodes( nodeFunctor );

	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* leaf = samples[i].node;
		while( leaf && !tree._isValidSpaceNode( leaf ) ) leaf = leaf->parent;
		if( leaf && tree._isValidSpaceNode( leaf ) )
		{
			const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
			DualPointInfo< Dim , Real , T , PointD >& _pData = _iData[ _sampleSpan[ leaf->nodeData.nodeIndex ].second++ ];
			_pData.position = pData.data;
			_pData.dualValues = _constraintDual( pData.data/pData.weight ) * pData.weight;
			_pData.weight = pData.weight;
		}
	}

	ThreadPool::ParallelFor( 0 , _iData.size() , [&]( unsigned int , size_t i )
		{
			Real w = _iData[i].weight;
			_iData[i] /= w;
			if( noRescale ) _iData[i].weight = w;
			else            _iData[i].weight = w * ( 1<<tree._maxDepth );
			_iData[i].dualValues *= _iData[i].weight;
		}
	);
}

template< unsigned int Dim , class Real >
template< typename T >
bool FEMTree< Dim , Real >::_setInterpolationInfoFromChildren( FEMTreeNode* node , SparseNodeData< T , IsotropicUIntPack< Dim , FEMTrivialSignature > >& interpolationInfo ) const
{
	if( IsActiveNode< Dim >( node->children ) )
	{
		bool hasChildData = false;
		T t = {};
		for( int c=0 ; c<(1<<Dim) ; c++ )
			if( _setInterpolationInfoFromChildren( node->children + c , interpolationInfo ) )
			{
				t += interpolationInfo[ node->children + c ];
				hasChildData = true;
			}
		if( hasChildData && IsActiveNode< Dim >( node ) ) interpolationInfo[ node ] += t;
		return hasChildData;
	}
	else return interpolationInfo( node )!=NULL;
}

template< unsigned int Dim , class Real >
template< typename T , unsigned int PointD , typename ConstraintDual >
void FEMTree< Dim , Real >::_densifyInterpolationInfoAndSetDualConstraints( SparseNodeData< DualPointInfo< Dim , Real , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > & iInfo , const std::vector< PointSample >& samples , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const
{
	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* node = samples[i].node;
		const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
		while( !IsActiveNode< Dim >( node ) ) node = node->parent;
		if( pData.weight )
		{
			DualPointInfo< Dim , Real , T , PointD >& _pData = iInfo[node];
			_pData.position += pData.data;
			_pData.weight += pData.weight;
			_pData.dualValues += constraintDual( pData.data/pData.weight ) * pData.weight;
		}
	}

	// Set the interior values
	_setInterpolationInfoFromChildren( _spaceRoot , iInfo );

	ThreadPool::ParallelFor( 0 , iInfo.size() , [&]( unsigned int , size_t i )
		{
			Real w = iInfo[i].weight;
			iInfo[i] /= w ; iInfo[i].weight = w;
		} );

	// Set the average position and scale the weights
	auto nodeFunctor = [&]( const FEMTreeNode *node )
	{
		if( IsActiveNode< Dim >( node ) )
		{
			DualPointInfo< Dim , Real , T , PointD >* pData = iInfo( node );
			if( pData )
			{
				int e = _localDepth( node ) * adaptiveExponent - ( maxDepth ) * (adaptiveExponent-1);
				if( e<0 ) pData->weight /= Real( 1<<(-e) );
				else      pData->weight *= Real( 1<<  e  );
				pData->dualValues *= pData->weight;
			}
		}
	};
	_tree.processNodes( nodeFunctor );
}
template< unsigned int Dim , class Real >
template< typename T , typename Data , unsigned int PointD , typename ConstraintDual >
void FEMTree< Dim , Real >::_densifyInterpolationInfoAndSetDualConstraints( SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > &iInfo , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent  ) const
{
	for( node_index_type i=0 ; i<(node_index_type)samples.size() ; i++ )
	{
		const FEMTreeNode* node = samples[i].node;
		const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
		while( !IsActiveNode< Dim >( node ) ) node = node->parent;
		if( pData.weight )
		{
			DualPointAndDataInfo< Dim , Real , Data , T , PointD >& _pData = iInfo[node];
			_pData.pointInfo.position += pData.data;
			_pData.pointInfo.dualValues += constraintDual( pData.data/pData.weight , sampleData[i]/pData.weight ) * pData.weight;
			_pData.pointInfo.weight += pData.weight;
			_pData.data += sampleData[i];
		}
	}

	// Set the interior values
	_setInterpolationInfoFromChildren( _spaceRoot , iInfo );

	ThreadPool::ParallelFor( 0 , iInfo.size() , [&]( unsigned int , size_t i )
		{
			Real w = iInfo[i].pointInfo.weight;
			iInfo[i] /= w ; iInfo[i].pointInfo.weight = w;
		}
	);

	// Set the average position and scale the weights
	auto nodeFunctor = [&]( const FEMTreeNode *node )
	{
		if( IsActiveNode< Dim >( node ) )
		{
			DualPointAndDataInfo< Dim , Real , Data , T , PointD >* pData = iInfo( node );
			if( pData )
			{
				int e = _localDepth( node ) * adaptiveExponent - ( maxDepth ) * (adaptiveExponent-1);
				if( e<0 ) pData->pointInfo.weight /= Real( 1<<(-e) );
				else      pData->pointInfo.weight *= Real( 1<<  e  );
				pData->pointInfo.dualValues *= pData->pointInfo.weight;
			}
		}
	};
	_tree.processNodes( nodeFunctor );
}
template< unsigned int Dim , class Real >
template< typename T , unsigned int PointD , typename ConstraintDual >
SparseNodeData< DualPointInfo< Dim , Real , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > FEMTree< Dim , Real >::_densifyInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const
{
	SparseNodeData< DualPointInfo< Dim , Real , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > iInfo;
	_densifyInterpolationInfoAndSetDualConstraints( iInfo , samples , constraintDual , maxDepth , adaptiveExponent );
	return iInfo;
}
template< unsigned int Dim , class Real >
template< typename T , typename Data , unsigned int PointD , typename ConstraintDual >
SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > FEMTree< Dim , Real >::_densifyInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const
{
	SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > iInfo;
	_densifyInterpolationInfoAndSetDualConstraints( iInfo , samples , sampleData , constraintDual , maxDepth , adaptiveExponent );
	return iInfo;
}


template< unsigned int Dim , class Real >
template< typename T , unsigned int PointD , typename ConstraintDual >
SparseNodeData< DualPointInfoBrood< Dim , Real , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > FEMTree< Dim , Real >::_densifyChildInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstraintDual constraintDual , bool noRescale ) const
{
	SparseNodeData< DualPointInfoBrood< Dim , Real , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > iInfo;
	for( node_index_type i=0 ; i<samples.size() ; i++ )
	{
		const FEMTreeNode* node = samples[i].node;
		const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
		while( !IsActiveNode< Dim >( node ) ) node = node->parent;
		if( pData.weight )
		{
			DualPointInfoBrood< Dim , Real , T , PointD >& _pData = iInfo[node];
			Point< Real , Dim > p = pData.data/pData.weight;
			int cIdx = _childIndex( node , p );
			_pData[cIdx].position += pData.data;
			_pData[cIdx].weight += pData.weight;
			_pData[cIdx].dualValues += constraintDual( p ) * pData.weight;
		}
	}

	// Set the interior values
	_setInterpolationInfoFromChildren( _spaceRoot , iInfo );

	ThreadPool::ParallelFor( 0 , iInfo.size() , [&]( unsigned int , size_t i )
		{
			iInfo[i].finalize();
			for( size_t c=0 ; c<iInfo[i].size() ; c++ )
			{
				iInfo[i][c].position /= iInfo[i][c].weight;
				if( !noRescale )
				{
					iInfo[i][c].weight     *= ( 1<<_maxDepth );
					iInfo[i][c].dualValues *= ( 1<<_maxDepth );
				}
			}
		}
	);
	return iInfo;
}
template< unsigned int Dim , class Real >
template< typename T , typename Data , unsigned int PointD , typename ConstraintDual >
SparseNodeData< DualPointAndDataInfoBrood< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > FEMTree< Dim , Real >::_densifyChildInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , bool noRescale ) const
{
	SparseNodeData< DualPointAndDataInfoBrood< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > iInfo;
	for( node_index_type i=0 ; i<samples.size() ; i++ )
	{
		const FEMTreeNode* node = samples[i].node;
		const ProjectiveData< Point< Real , Dim > , Real >& pData = samples[i].sample;
		while( !IsActiveNode< Dim >( node ) ) node = node->parent;
		if( pData.weight )
		{
			DualPointAndDataInfoBrood< Dim , Real , Data , T , PointD >& _pData = iInfo[node];
			Point< Real , Dim > p = pData.data/pData.weight;
			int cIdx = _childIndex( node , p );
			_pData[cIdx].pointInfo.position += pData.data;
			_pData[cIdx].pointInfo.dualValues += constraintDual( p , sampleData[i]/pData.weight ) * pData.weight;
			_pData[cIdx].pointInfo.weight += pData.weight;
			_pData[cIdx].data += sampleData[i];
		}
	}

	// Set the interior values
	_setInterpolationInfoFromChildren( _spaceRoot , iInfo );

	ThreadPool::ParallelFor( 0 , iInfo.size() , [&]( unsigned int , size_t i )
		{
			iInfo[i].finalize();
			for( size_t c=0 ; c<iInfo[i].size() ; c++ )
			{
				iInfo[i][c].pointInfo.position /= iInfo[i][c].pointInfo.weight;
				iInfo[i][c].data /= iInfo[i][c].pointInfo.weight;
				if( !noRescale )
				{
					iInfo[i][c].pointInfo.weight     *= ( 1<<_maxDepth );
					iInfo[i][c].pointInfo.dualValues *= ( 1<<_maxDepth );
					iInfo[i][c].data                 *= ( 1<<_maxDepth );
				}
			}
		}
	);
	return iInfo;
}



template< unsigned int Dim , class Real >
std::vector< node_index_type > FEMTree< Dim , Real >::merge( FEMTree* tree )
{
	std::vector< node_index_type > map;
	if( _depthOffset!=tree->_depthOffset ) MK_THROW( "depthOffsets don't match: %d != %d" , _depthOffset , tree->_depthOffset );

	// Compute the next available index
	node_index_type nextIndex = 0;
	_tree.processNodes( [&]( const FEMTreeNode *node ){ nextIndex = std::max< node_index_type >( nextIndex , node->nodeData.nodeIndex+1 ); } );

	// Set the size of the map
	{
		node_index_type mapSize = 0;
		tree->_tree.processNodes( [&]( const FEMTreeNode *node ){ mapSize = std::max< node_index_type >( mapSize , node->nodeData.nodeIndex+1 ); } );
		map.resize( mapSize );
	}

	std::function< void ( FEMTreeNode* , FEMTreeNode* , std::vector< node_index_type > & , node_index_type & ) > MergeNodes = [&]( FEMTreeNode* node1 , FEMTreeNode* node2 , std::vector< node_index_type > &map , node_index_type &nextIndex )
	{
		if( node1 && node2 )
		{
			if( node2->nodeData.nodeIndex>=0 )
			{
				if( node1->nodeData.nodeIndex<0 ) node1->nodeData.nodeIndex = nextIndex++;
				map[ node2->nodeData.nodeIndex ] = node1->nodeData.nodeIndex;
			}
			if( node1->children && node2->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) MergeNodes( node1->children+c , node2->children+c , map , nextIndex );
			else if( node2->children )
			{
				for( int c=0 ; c<(1<<Dim) ; c++ ) MergeNodes( NULL , node2->children+c , map , nextIndex );
				node1->children = node2->children;
				node2->children = NULL;
				for( int c=0 ; c<(1<<Dim) ; c++ ) node1->children[c].parent = node1;
			}
		}
		else if( node2 )
		{
			if( node2->nodeData.nodeIndex>=0 ){ map[ node2->nodeData.nodeIndex ] = nextIndex ; node2->nodeData.nodeIndex = nextIndex++; }
			if( node2->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) MergeNodes( NULL , node2->children+c , map , nextIndex );
		}
	};

	MergeNodes( _tree , tree->_tree , map , nextIndex );
	return map;
}