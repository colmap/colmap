/*
Copyright (c) 2016, Michael Kazhdan
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

////////////////////////
// FEMTreeInitializer //
////////////////////////
template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , int maxDepth , std::function< bool ( int , int[] ) > Refine , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );
	return _Initialize( root , maxDepth , Refine , nodeAllocator , NodeInitializer );
}

template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::_Initialize( FEMTreeNode &node , int maxDepth , std::function< bool ( int , int[] ) > Refine , Allocator< FEMTreeNode > *nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	size_t count = 0;
	int d , off[3];
	node.depthAndOffset( d , off );
	if( d<maxDepth && Refine( d , off ) )
	{
		if( !node.children ) node.template initChildren< false >( nodeAllocator , NodeInitializer ) , count += 1<<Dim;
		for( int c=0 ; c<(1<<Dim) ; c++ ) count += _Initialize( node.children[c] , maxDepth , Refine , nodeAllocator , NodeInitializer );
	}
	return count;
}

template< unsigned int Dim , class Real >
template< typename IsValidFunctor /*=std::function< bool ( const Point< Real , Dim > & , const AuxData &... ) >*/ , typename ProcessFunctor/*=std::function< bool ( FEMTreeNode & , const Point< Real , Dim > & , const AuxData &... ) >*/ , typename ... AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData ... > &pointStream , AuxData ... d , int maxDepth ,                                                                  Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , IsValidFunctor IsValid , ProcessFunctor Process )
{
	return Initialize< IsValidFunctor , ProcessFunctor , AuxData ... >( root , pointStream , d... , maxDepth , [&]( Point< Real , Dim > ){ return maxDepth; } , nodeAllocator , NodeInitializer , IsValid , Process );
}

template< unsigned int Dim , class Real >
template< typename IsValidFunctor/*=std::function< bool ( const Point< Real , Dim > & , const AuxData &... ) >*/ , typename ProcessFunctor/*=std::function< bool ( FEMTreeNode & , const Point< Real , Dim > & , const AuxData &... ) >*/ , typename ... AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData ... > &pointStream , AuxData ... d , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , IsValidFunctor IsValid , ProcessFunctor Process )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );
	auto Leaf = [&]( FEMTreeNode& root , Point< Real , Dim > p , unsigned int maxDepth )
		{
			for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
			Point< Real , Dim > center;
			Real width;
			typename FEMTree< Dim , Real >::LocalDepth depth;
			typename FEMTree< Dim , Real >::LocalOffset offset;
			root.centerAndWidth( center , width );
			root.depthAndOffset( depth , offset );

			FEMTreeNode* node = &root;

			while( depth<(int)maxDepth )
			{
				if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
				int cIndex = FEMTreeNode::ChildIndex( center , p );
				node = node->children + cIndex;
				width /= 2;

				depth++;
				for( int dd=0 ; dd<Dim ; dd++ )
					if( (cIndex>>dd) & 1 ) center[dd] += width/2 , offset[dd] = (offset[dd]<<1) | 1;
					else                   center[dd] -= width/2 , offset[dd] = (offset[dd]<<1) | 0;
			}
			return node;
		};

	// Add the point data
	size_t outOfBoundPoints = 0 , badDataCount = 0 , pointCount = 0;
	Point< Real , Dim > p;
	while( pointStream.read( p , d... ) )
	{
		// Check if the data is good
		if( !IsValid( p , d... ) ){ badDataCount++ ; continue; }

		// Check that the position is in-range
		FEMTreeNode *leaf = Leaf( root , p , pointDepthFunctor(p) );
		if( !leaf ){ outOfBoundPoints++ ; continue; }

		// Process the data
		if( Process( *leaf , p , d ... ) ) pointCount++;
	}
	pointStream.reset();
	return pointCount;
}

template< unsigned int Dim , class Real >
template< typename AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData )
{
	struct StreamInitializationData sid;
	return Initialize< AuxData >( sid , root , pointStream , zeroData , maxDepth , samplePoints , sampleData , nodeAllocator , NodeInitializer , ProcessData );
}

template< unsigned int Dim , class Real >
template< typename AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData )
{
	struct StreamInitializationData sid;
	return Initialize< AuxData >( sid , root , pointStream , zeroData , maxDepth , pointDepthFunctor , samplePoints , sampleData , nodeAllocator , NodeInitializer , ProcessData );
}

template< unsigned int Dim , class Real >
template< typename AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( struct StreamInitializationData &sid , FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData )
{
	return Initialize< AuxData >( sid , root , pointStream , zeroData , maxDepth , [&]( Point< Real , Dim > ){ return maxDepth; } , samplePoints , sampleData , nodeAllocator , NodeInitializer , ProcessData );
}

template< unsigned int Dim , class Real >
template< typename AuxData >
size_t FEMTreeInitializer< Dim , Real >::Initialize( struct StreamInitializationData &sid , FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData )
{
	Real weight;
	std::vector< node_index_type > &nodeToIndexMap = sid._nodeToIndexMap;

	auto IsValid = [&]( const Point< Real , Dim > &p , AuxData & d )
		{
			weight = ProcessData( p , d );
			return weight>0;
		};
	auto Process = [&]( FEMTreeNode &node , const Point< Real , Dim > &p , AuxData &d )
		{
			node_index_type nodeIndex = node.nodeData.nodeIndex;
			// If the node's index exceeds what's stored in the node-to-index map, grow the node-to-index map
			if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );

			node_index_type idx = nodeToIndexMap[ nodeIndex ];
			if( idx==-1 )
			{
				idx = (node_index_type)samplePoints.size();
				nodeToIndexMap[ nodeIndex ] = idx;
				samplePoints.resize( idx+1 ) , samplePoints[idx].node = &node;
				sampleData.resize( idx+1 );
				samplePoints[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
				sampleData[idx] = d*weight;
			}
			else
			{
				samplePoints[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
				sampleData[ idx ] += d*weight;
			}
			return true;
		};
	return Initialize< decltype(IsValid) , decltype(Process) , AuxData >( root , pointStream , zeroData , maxDepth , pointDepthFunctor , nodeAllocator , NodeInitializer , IsValid , Process );
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , int maxDepth , std::vector< PointSample >& samples , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );

	std::vector< node_index_type > nodeToIndexMap;
	ThreadPool::ParallelFor( 0 , simplices.size() , [&]( unsigned int t , size_t  i )
	{
		Simplex< Real , Dim , Dim-1 > s;
		for( int k=0 ; k<Dim ; k++ ) s[k] = vertices[ simplices[i][k] ];
		_AddSimplex< true >( root , s , maxDepth , samples , &nodeToIndexMap , nodeAllocators.size() ? nodeAllocators[t] : NULL , NodeInitializer );
	} );
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , const std::vector< ProjectiveData< Point< Real , Dim > , Real > > &points , int maxDepth , std::vector< PointSample >& samples , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );

	std::vector< node_index_type > nodeToIndexMap;
	ThreadPool::ParallelFor( 0 , points.size() , [&]( unsigned int t , size_t  i )
	{
		_AddSample< true >( root , points[i] , maxDepth , samples , &nodeToIndexMap , nodeAllocators.size() ? nodeAllocators[t] : NULL , NodeInitializer );
	} );
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode &root , const std::vector< ProjectiveData< Point< Real , Dim > , Real > > &points , int maxDepth , std::vector< PointSample >& samples )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );

	std::vector< node_index_type > nodeToIndexMap;
	ThreadPool::ParallelFor( 0 , points.size() , [&]( unsigned int t , size_t  i )
	{
		_AddSample( root , points[i] , maxDepth , samples , &nodeToIndexMap );
	} );
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSample( FEMTreeNode& root , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d=0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};
	FEMTreeNode *node = Leaf( s.value() , maxDepth );
	if( !node ) return 0;
	else        return _AddSample( node , s , maxDepth , samples , nodeToIndexMap );
}

template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::_AddSample( FEMTreeNode& root , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap )
{
	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d=0;
		while( d<maxDepth && node->children )
		{
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};
	FEMTreeNode *node = Leaf( s.value() , maxDepth );
	if( !node ) return 0;
	else        return _AddSample( node , s , maxDepth , samples , nodeToIndexMap );
}

template< unsigned int Dim , class Real >
size_t FEMTreeInitializer< Dim , Real >::_AddSample( FEMTreeNode* node , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap )
{
	if( nodeToIndexMap )
	{
		node_index_type nodeIndex = node->nodeData.nodeIndex;
		{
			static std::mutex m;
			std::lock_guard< std::mutex > lock(m);
			if( nodeIndex>=(node_index_type)nodeToIndexMap->size() ) nodeToIndexMap->resize( nodeIndex+1 , -1 );
			node_index_type idx = (*nodeToIndexMap)[ nodeIndex ];
			if( idx==-1 )
			{
				idx = (node_index_type)samples.size();
				(*nodeToIndexMap)[ nodeIndex ] = idx;
				samples.resize( idx+1 );
				samples[idx].node = node;
			}
			samples[idx].sample += s;
		}
	}
	else
	{
		{
			static std::mutex m;
			std::lock_guard< std::mutex > lock(m);
			node_index_type idx = (node_index_type)samples.size();
			samples.resize( idx+1 );
			samples[idx].node = node;
			samples[idx].sample = s;
		}
	}
	return 1;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode& root , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	std::vector< Simplex< Real , Dim , Dim-1 > > subSimplices;
	subSimplices.push_back( s );

	// Clip the simplex to the unit cube
	{
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n;
			n[d] = 1;
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
				subSimplices = front;
			}
			{
				std::vector< Simplex< Real , Dim , Dim-1 > > back , front;
				for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
				subSimplices = back;
			}
		}
	}

	struct RegularGridIndex
	{
		int idx[Dim];
		bool operator != ( const RegularGridIndex& i ) const
		{
			for( int d=0 ; d<Dim ; d++ ) if( idx[d]!=i.idx[d] ) return true;
			return false;
		}
	};

	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		for( int d=0 ; d<Dim ; d++ ) center[d] = (Real)0.5;
		Real width = Real(1.0);
		FEMTreeNode* node = &root;
		int d=0;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};

	size_t sCount = 0;
	for( int i=0 ; i<subSimplices.size() ; i++ )
	{
		// Find the finest depth at which the simplex is entirely within a node
		int tDepth;
		RegularGridIndex idx0 , idx;
		for( tDepth=0 ; tDepth<maxDepth ; tDepth++ )
		{
			// Get the grid index of the first vertex of the simplex
			for( int d=0 ; d<Dim ; d++ ) idx0.idx[d] = idx.idx[d] = (int)( subSimplices[i][0][d] * (1<<(tDepth+1)) );
			bool done = false;
			for( int k=0 ; k<Dim && !done ; k++ )
			{
				for( int d=0 ; d<Dim ; d++ ) idx.idx[d] = (int)( subSimplices[i][k][d] * (1<<(tDepth+1)) );
				if( idx!=idx0 ) done = true;
			}
			if( done ) break;
		}

		// Generate a point in the middle of the simplex
		sCount += _AddSimplex< ThreadSafe >( Leaf( subSimplices[i].center() , tDepth ) , subSimplices[i] , maxDepth , samples , nodeToIndexMap , nodeAllocator , NodeInitializer );
	}
	return sCount;
}

template< unsigned int Dim , class Real >
template< bool ThreadSafe >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode* node , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	int d = node->depth();
	if( d==maxDepth )
	{
		Real weight = s.measure();
		Point< Real , Dim > position = s.center();
		if( weight && weight==weight )
		{
			if( nodeToIndexMap )
			{
				node_index_type nodeIndex = node->nodeData.nodeIndex;
				{
					static std::mutex m;
					std::lock_guard< std::mutex > lock(m);
					if( nodeIndex>=(node_index_type)nodeToIndexMap->size() ) nodeToIndexMap->resize( nodeIndex+1 , -1 );
					node_index_type idx = (*nodeToIndexMap)[ nodeIndex ];
					if( idx==-1 )
					{
						idx = (node_index_type)samples.size();
						(*nodeToIndexMap)[ nodeIndex ] = idx;
						samples.resize( idx+1 );
						samples[idx].node = node;
					}
					samples[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( position*weight , weight );
				}
			}
			else
			{
				{
					static std::mutex m;
					std::lock_guard< std::mutex > lock(m);
					node_index_type idx = (node_index_type)samples.size();
					samples.resize( idx+1 );
					samples[idx].node = node;
					samples[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( position*weight , weight );
				}
			}
		}
		return 1;
	}
	else
	{
		size_t sCount = 0;
		if( !node->children ) node->template initChildren< ThreadSafe >( nodeAllocator , NodeInitializer );

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		Real width;
		node->centerAndWidth( center , width );

		std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > childSimplices( 1 );
		childSimplices[0].push_back( s );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( size_t i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( size_t i=0 ; i<childSimplices[c].size() ; i++ ) if( childSimplices[c][i].measure() ) sCount += _AddSimplex< ThreadSafe >( node->children+c , childSimplices[c][i] , maxDepth , samples , nodeToIndexMap , nodeAllocator , NodeInitializer );
		return sCount;
	}
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< NodeSimplices< Dim , Real > >& nodeSimplices , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	if( regularGridDepth>maxDepth ) MK_THROW( "Regular grid depth cannot excceed maximum depth: " , regularGridDepth , " <= " , maxDepth );

	// Allocate the tree up to the prescribed depth
	const Real RegularGridWidth = (Real)( 1./(1<<regularGridDepth) );

	auto Leaf = [&]( FEMTreeNode *root , Point< Real , Dim > p , int depth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		Real width;
		root->centerAndWidth( center , width );
		FEMTreeNode *node = root;
		int d=node->depth();
		while( d<depth )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocators.size() ? nodeAllocators[0] : NULL , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};

	IndexedSimplicialComplex< Real , Dim , Dim-1 , node_index_type > sComplex( vertices , simplices );
	typename Rasterizer< Real , Dim >::template SimplexRasterizationGrid< node_index_type , Dim-1 > raster = Rasterizer< Real , Dim >::template Rasterize< node_index_type >( sComplex , regularGridDepth , typename Rasterizer< Real , Dim >::ThreadSafety( Rasterizer< Real , Dim >::ThreadSafety::MUTEXES , regularGridDepth ) );
//	typename Rasterizer< Real , Dim >::template SimplexRasterizationGrid< node_index_type , Dim-1 > raster = Rasterizer< Real , Dim >::template Rasterize< node_index_type >( sComplex , regularGridDepth , typename Rasterizer< Real , Dim >::ThreadSafety( Rasterizer< Real , Dim >::ThreadSafety::SINGLE_THREADED ) );
//	typename Rasterizer< Real , Dim >::template SimplexRasterizationGrid< node_index_type , Dim-1 > raster = Rasterizer< Real , Dim >::template Rasterize< node_index_type >( sComplex , regularGridDepth , typename Rasterizer< Real , Dim >::ThreadSafety( Rasterizer< Real , Dim >::ThreadSafety::MAP_REDUCE ) );

	size_t geometricCellCount = 0;
	for( size_t i=0 ; i<raster.resolution() ; i++ ) if( raster[i].size() ) geometricCellCount++;

	if( maxDepth==regularGridDepth )
	{
		nodeSimplices.resize( geometricCellCount );

		geometricCellCount = 0;
		for( size_t i=0 ; i<raster.resolution() ; i++ ) if( raster[i].size() )
		{
			int idx[Dim];
			raster.setIndex( i , idx );
			Point< Real , Dim > p;
			for( int d=0 ; d<Dim ; d++ ) p[d] = (Real)( idx[d] + 0.5 ) * RegularGridWidth;
			NodeSimplices< Dim , Real > &nSimplices = nodeSimplices[ geometricCellCount++ ];
			nSimplices.node = Leaf( &root , p , maxDepth );
			nSimplices.data = raster[i];
		}
	}
	else
	{
		// The indices of the grid cells containing geometry
		std::vector< size_t > cellIndices( geometricCellCount );
		// The list of nodes @{regularGridDepth} containing geometry
		std::vector< FEMTreeNode * > roots( geometricCellCount );
		std::vector< node_index_type > nodeToIndexMap;

		// Get the list of the indices of the regular grid containing geometry
		geometricCellCount = 0;
		// [WARNING] In principal, this could be done in parallel but then Leaf would need to be thread-safe.
		for( size_t i=0 ; i<raster.resolution() ; i++ ) if( raster[i].size() )
		{
			cellIndices[ geometricCellCount ] = i;
			int idx[Dim];
			raster.setIndex( i , idx );
			Point< Real , Dim > p;
			for( int d=0 ; d<Dim ; d++ ) p[d] = (Real)( idx[d] + 0.5 ) * RegularGridWidth;
			roots[ geometricCellCount ] = Leaf( &root , p , regularGridDepth );
			geometricCellCount++;
		}

		std::vector< Allocator< FEMTreeNode > * > _nodeAllocators( ThreadPool::NumThreads() );
		for( int i=0 ; i<_nodeAllocators.size() ; i++ ) _nodeAllocators[i] = nodeAllocators.size() ? nodeAllocators[i] : NULL;
		ThreadPool::ParallelFor( 0 , geometricCellCount , [&]( unsigned int t , size_t i )
		{
			auto &cellSimplices = raster[ cellIndices[i] ];
			for( int j=0 ; j<cellSimplices.size() ; j++ ) _AddSimplex< false , true >( *roots[i] , cellSimplices[j].first , cellSimplices[j].second , maxDepth , nodeSimplices , nodeToIndexMap , _nodeAllocators[t] , NodeInitializer );
		} );
	}
}

template< unsigned int Dim , class Real >
template< bool ThreadSafeAllocation , bool ThreadSafeSimplices >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode& root , node_index_type id , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	struct RegularGridIndex
	{
		int idx[Dim];
		bool operator != ( const RegularGridIndex& i ) const
		{
			for( int d=0 ; d<Dim ; d++ ) if( idx[d]!=i.idx[d] ) return true;
			return false;
		}
	};

	auto Leaf = [&]( Point< Real , Dim > p , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) return (FEMTreeNode*)NULL;
		Point< Real , Dim > center;
		Real width;
		root.centerAndWidth( center , width );
		int d = root.depth();
		FEMTreeNode* node = &root;
		while( d<maxDepth )
		{
			if( !node->children ) node->template initChildren< ThreadSafeAllocation >( nodeAllocator , NodeInitializer );
			int cIndex = FEMTreeNode::ChildIndex( center , p );
			node = node->children + cIndex;
			d++;
			width /= 2;
			for( int d=0 ; d<Dim ; d++ )
				if( (cIndex>>d) & 1 ) center[d] += width/2;
				else                  center[d] -= width/2;
		}
		return node;
	};

	// Find the finest depth at which the simplex is entirely within a node
	int tDepth;
	RegularGridIndex idx0 , idx;
	for( tDepth=0 ; tDepth<maxDepth ; tDepth++ )
	{
		// Get the grid index of the first vertex of the simplex
		for( int d=0 ; d<Dim ; d++ ) idx0.idx[d] = (int)( s[0][d] * (1<<(tDepth+1)) );
		bool done = false;
		for( int k=1 ; k<=Dim && !done ; k++ )
		{
			for( int d=0 ; d<Dim ; d++ ) idx.idx[d] = (int)( s[k][d] * (1<<(tDepth+1)) );
			if( idx!=idx0 ) done = true;
		}
		if( done ) break;
	}
	// Add the simplex to the node
	return _AddSimplex< ThreadSafeAllocation , ThreadSafeSimplices >( Leaf( s.center() , tDepth ) , id , s , maxDepth , simplices , nodeToIndexMap , nodeAllocator , NodeInitializer );
}

template< unsigned int Dim , class Real >
template< bool ThreadSafeAllocation , bool ThreadSafeSimplices >
size_t FEMTreeInitializer< Dim , Real >::_AddSimplex( FEMTreeNode* node , node_index_type id , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	int d = node->depth();
	if( d==maxDepth )
	{
		// If the simplex has non-zero size, add it to the list
		Real weight = s.measure();
		if( weight && weight==weight )
		{
			node_index_type nodeIndex = node->nodeData.nodeIndex;
			if( ThreadSafeSimplices )
			{
				static std::mutex m;
				std::lock_guard< std::mutex > lock(m);
				if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );
				node_index_type idx = nodeToIndexMap[ nodeIndex ];
				if( idx==-1 )
				{
					idx = (node_index_type)simplices.size();
					nodeToIndexMap[ nodeIndex ] = idx;
					simplices.resize( idx+1 );
					simplices[idx].node = node;
				}
				simplices[idx].data.push_back( std::pair< node_index_type , Simplex< Real , Dim , Dim-1 > >( id , s ) );
			}
			else
			{
				if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );
				node_index_type idx = nodeToIndexMap[ nodeIndex ];
				if( idx==-1 )
				{
					idx = (node_index_type)simplices.size();
					nodeToIndexMap[ nodeIndex ] = idx;
					simplices.resize( idx+1 );
					simplices[idx].node = node;
				}
				simplices[idx].data.push_back( std::pair< node_index_type , Simplex< Real , Dim , Dim-1 > >( id , s ) );
			}
		}
		return 1;
	}
	else
	{
		size_t sCount = 0;
		if( !node->children ) node->template initChildren< ThreadSafeAllocation >( nodeAllocator , NodeInitializer );

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		Real width;
		node->centerAndWidth( center , width );

		std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > childSimplices( 1 );
		childSimplices[0].push_back( s );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , Dim-1 > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) sCount += _AddSimplex< ThreadSafeAllocation , ThreadSafeSimplices >( node->children+c , id , childSimplices[c][i] , maxDepth , simplices , nodeToIndexMap , nodeAllocator , NodeInitializer );
		return sCount;
	}
}

template< unsigned int Dim , class Real >
template< class Data , class _Data , bool Dual >
size_t FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , ConstPointer( Data ) values , ConstPointer( int ) labels , int resolution[Dim] , std::vector< NodeSample< Dim , _Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< _Data ( const Data& ) > DataConverter )
{
	auto Leaf = [&]( FEMTreeNode& root , const int idx[Dim] , int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( idx[d]<0 || idx[d]>=(1<<maxDepth) ) return (FEMTreeNode*)NULL;
		FEMTreeNode* node = &root;
		for( int d=0 ; d<maxDepth ; d++ )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = 0;
			for( int dd=0 ; dd<Dim ; dd++ ) if( idx[dd]&(1<<(maxDepth-d-1)) ) cIndex |= 1<<dd;
			node = node->children + cIndex;
		}
		return node;
	};
	auto FactorIndex = []( size_t i , const int resolution[Dim] , int idx[Dim] )
	{
		size_t ii = i;
		for( int d=0 ; d<Dim ; d++ ) idx[d] = ii % resolution[d] , ii /= resolution[d];
	};
	auto MakeIndex = [] ( const int idx[Dim] , const int resolution[Dim] )
	{
		size_t i = 0;
		for( int d=0 ; d<Dim ; d++ ) i = i * resolution[Dim-1-d] + idx[Dim-1-d];
		return i;
	};


	int maxResolution = resolution[0];
	for( int d=1 ; d<Dim ; d++ ) maxResolution = std::max< int >( maxResolution , resolution[d] );
	int maxDepth = 0;
	while( ( (1<<maxDepth) + ( Dual ? 0 : 1 ) )<maxResolution ) maxDepth++;

	size_t totalRes = 1;
	for( int d=0 ; d<Dim ; d++ ) totalRes *= resolution[d];

	// Iterate over each direction
	for( int d=0 ; d<Dim ; d++ ) for( size_t i=0 ; i<totalRes ; i++ )
	{
		// Factor the index into directional components and get the index of the next cell
		int idx[Dim] ; FactorIndex( i , resolution , idx ) ; idx[d]++;

		if( idx[d]<resolution[d] )
		{
			// Get the index of the next cell
			size_t ii = MakeIndex( idx , resolution );

			// [NOTE] There are no derivatives across negative labels
			if( labels[i]!=labels[ii] && labels[i]>=0 && labels[ii]>=0 )
			{
				if( !Dual ) idx[d]--;
				NodeSample< Dim , _Data > nodeSample;
				nodeSample.node = Leaf( root , idx , maxDepth );
				nodeSample.data = DataConverter( values[ii] ) - DataConverter( values[i] );
				if( nodeSample.node ) derivatives[d].push_back( nodeSample );
			}
		}
	}
	return maxDepth;
}

template< unsigned int Dim , class Real >
template< bool Dual , class Data >
unsigned int FEMTreeInitializer< Dim , Real >::Initialize( FEMTreeNode& root , DerivativeStream< Data >& dStream , Data zeroData , std::vector< NodeSample< Dim , Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	// Note:
	// --   Dual: The difference between [i] and [i+1] is stored at cell [i+1]
	// -- Primal: The difference between [i] and [i+1] is stored at cell [i]

	// Find the leaf containing the specified cell index
	auto Leaf = [&]( FEMTreeNode& root , const unsigned int idx[Dim] , unsigned int maxDepth )
	{
		for( int d=0 ; d<Dim ; d++ ) if( idx[d]<0 || idx[d]>=(unsigned int)(1<<maxDepth) ) return (FEMTreeNode*)NULL;
		FEMTreeNode* node = &root;
		for( unsigned int d=0 ; d<maxDepth ; d++ )
		{
			if( !node->children ) node->template initChildren< false >( nodeAllocator , NodeInitializer );
			int cIndex = 0;
			for( int dd=0 ; dd<Dim ; dd++ ) if( idx[dd]&(1<<(maxDepth-d-1)) ) cIndex |= 1<<dd;
			node = node->children + cIndex;
		}
		return node;
	};

	unsigned int resolution[Dim];
	dStream.resolution( resolution );
	unsigned int maxResolution = resolution[0];
	for( int d=1 ; d<Dim ; d++ ) maxResolution = std::max< unsigned int >( maxResolution , resolution[d] );
	unsigned int maxDepth = 0;

	// If we are using a dual formulation, we need at least maxResolution cells.
	// Otherwise, we need at least maxResolution-1 cells.
	while( (unsigned int)( (1<<maxDepth) + ( Dual ? 0 : 1 ) )<maxResolution ) maxDepth++;

	unsigned int idx[Dim] , dir;
	Data dValue = zeroData;
	while( dStream.nextDerivative( idx , dir , dValue ) )
	{
		if( Dual ) idx[dir]++;
		NodeSample< Dim , Data > nodeSample;
		nodeSample.node = Leaf( root , idx , maxDepth );
		nodeSample.data = dValue;
		if( nodeSample.node ) derivatives[dir].push_back( nodeSample );
	}
	return maxDepth;
}

template< unsigned int Dim , class Real >
template< unsigned int _Dim >
typename std::enable_if< _Dim!=1 , DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > >::type FEMTreeInitializer< Dim , Real >::GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	static_assert( Dim==_Dim , "[ERROR] Dimensions don't match" );
	std::vector< Point< Real , Dim > > normals( simplices.size() );
	ThreadPool::ParallelFor
	(
		0 , simplices.size() ,
		[&]( unsigned int , size_t i )
	{
		Simplex< Real , Dim , Dim-1 > s;
		for( int j=0 ; j<Dim ; j++ ) s[j] = vertices[ simplices[i][j] ];
		normals[i] = s.normal();
	}
	);
	return _GetGeometryNodeDesignators( root , vertices , simplices , normals , regularGridDepth , maxDepth , nodeAllocators , NodeInitializer );
}
template< unsigned int Dim , class Real >
template< unsigned int _Dim >
typename std::enable_if< _Dim==1 , DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > >::type FEMTreeInitializer< Dim , Real >::GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	static_assert( Dim==_Dim , "[ERROR] Dimensions don't match" );
	if( simplices.size()%2 ) MK_THROW( "Expected even number of hull points: " , simplices.size() );
	struct HullPoint
	{
		Real x;
		size_t idx;
	};
	std::vector< HullPoint > hullPoints( simplices.size() );
	for( size_t i=0 ; i<simplices.size() ; i++ ) hullPoints[i].x = vertices[ simplices[i][0] ][0] , hullPoints[i].idx = i;
	std::sort( hullPoints.begin() , hullPoints.end() , []( const HullPoint &hp1 , const HullPoint &hp2 ){ return hp1.x<hp2.x; } );
	std::vector< Point< Real , Dim > > normals( simplices.size() );
	for( int i=0 ; i<hullPoints.size() ; i++ ) normals[ hullPoints[i].idx ][0] = (i%2) ? (Real)1. : (Real)-1.;
	return _GetGeometryNodeDesignators( root , vertices , simplices , normals , regularGridDepth , maxDepth , nodeAllocators , NodeInitializer );
}

template< unsigned int Dim , class Real >
DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > FEMTreeInitializer< Dim , Real >::_GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , const std::vector< Point< Real , Dim > > &normals , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer )
{
	typename FEMTreeNode::SubTreeExtractor subtreeExtractor( root );

	typedef typename FEMTreeNode::template ConstNeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > > NeighborKey;
	typedef typename FEMTreeNode::template ConstNeighbors< IsotropicUIntPack< Dim , 3 > > Neighbors;
	DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > geometryNodeDesignators;

	// Rasterize the geometry into the tree
	std::vector< NodeSimplices< Dim , Real > > nodeSimplices;
	FEMTreeInitializer< Dim , Real >::Initialize( *root , vertices , simplices , regularGridDepth , maxDepth , nodeSimplices , nodeAllocators , NodeInitializer );

	// Mark all the nodes containing geometry
	node_index_type nodeCount = 0;
	root->processNodes( [&]( FEMTreeNode *node ){ nodeCount = std::max< node_index_type >( nodeCount , node->nodeData.nodeIndex ); } );
	nodeCount++;

	geometryNodeDesignators.resize( nodeCount );

	ThreadPool::ParallelFor( 0 , nodeSimplices.size() , [&]( unsigned int , size_t i ){ for( FEMTreeNode *node=nodeSimplices[i].node ; node ; node=node->parent ) geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY; } );

	// Propagate out from the boundary nodes
	std::vector< const FEMTreeNode * > interiorNodes , exteriorNodes;
	std::vector< std::vector< const FEMTreeNode * > > _interiorNodes( ThreadPool::NumThreads() ) ,  _exteriorNodes( ThreadPool::NumThreads() );

	std::vector< NeighborKey > neighborKeys( ThreadPool::NumThreads() );
	for( int i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( maxDepth );

	// In the first pass, flood-fill from the geometry-containing nodes
	ThreadPool::ParallelFor( 0 , nodeSimplices.size() , [&]( unsigned int thread , size_t i )
	{
		std::vector< const FEMTreeNode * > &interiorNodes = _interiorNodes[thread];
		std::vector< const FEMTreeNode * > &exteriorNodes = _exteriorNodes[thread];
		NeighborKey &neighborKey = neighborKeys[thread];
		Point< Real , Dim > center ; Real width;
		nodeSimplices[i].node->centerAndWidth( center , width );
		Neighbors &neighbors = neighborKey.getNeighbors( nodeSimplices[i].node );

		// Iterate over the faces
		for( unsigned int d=0 ; d<Dim ; d++ ) for( int dir=0 ; dir<=2 ; dir+=2 )
		{
			unsigned int idx[Dim];
			for( unsigned int d=0 ; d<Dim ; d++ ) idx[d] = 1;
			idx[d] = dir;

			// Terminate early if the neighbor node exists but contains geometry
			const FEMTreeNode *node = neighbors.neighbors( idx );
			if( node && geometryNodeDesignators[node]==GeometryNodeType::BOUNDARY ) continue;

			// Compute the center of the face
			Point< Real , Dim > p = center;
			p[d] += (Real)(dir-1) * width / 2;

			// If the center of the face is outside and the face-adjacent node does not contain geometry, add the face-adjacent neighbor to the list of exterior nodes
			std::vector< Simplex< Real , Dim , Dim-1 > > _simplices( nodeSimplices[i].data.size() );
			std::vector< Point< Real , Dim > > _normals( nodeSimplices[i].data.size() ); 
			for( int j=0 ; j<nodeSimplices[i].data.size() ; j++ )
			{
				_simplices[j] = nodeSimplices[i].data[j].second;
				_normals[j] = normals[ nodeSimplices[i].data[j].first ];
			}

			bool interior = Simplex< Real , Dim , Dim-1 >::IsInterior( p , _simplices , _normals );
			{
				const FEMTreeNode *node = neighbors.neighbors( idx );

				// If the face-adjacent node exists and does not contain geometry, add it to the list of exterior nodes
				if( node )
				{
					if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN )
						if( interior ) interiorNodes.push_back( node );
						else           exteriorNodes.push_back( node );
				}
				// Otherwise, try the parents' face-adjacent neighbors
				else
				{
					for( int depth=maxDepth-1 ; depth>=0 ; depth-- )
					{
						node = neighborKey.neighbors[depth].neighbors( idx );
						if( node )
						{
							if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN )
								if( interior ) interiorNodes.push_back( node );
								else           exteriorNodes.push_back( node );
							break;
						}
					}
				}
			}
		}
	} );

	// Merge the exterior nodes computed by the different threads and mark them exterior
	{
		size_t interiorCount = 0 , exteriorCount = 0;;
		for( int i=0 ; i<_interiorNodes.size() ; i++ ) interiorCount += _interiorNodes[i].size();
		for( int i=0 ; i<_exteriorNodes.size() ; i++ ) exteriorCount += _exteriorNodes[i].size();
		interiorNodes.reserve( interiorCount ) , exteriorNodes.reserve( exteriorCount );
		for( int i=0 ; i<_interiorNodes.size() ; i++ ) for( int j=0 ; j<_interiorNodes[i].size() ; j++ )
		{
			if( geometryNodeDesignators[ _interiorNodes[i][j] ]==GeometryNodeType::BOUNDARY ) MK_THROW( "Interior node has geometry" );
			else if( geometryNodeDesignators[ _interiorNodes[i][j] ]==GeometryNodeType::UNKNOWN )
			{
				geometryNodeDesignators[ _interiorNodes[i][j] ] = GeometryNodeType::INTERIOR;
				interiorNodes.push_back( _interiorNodes[i][j] );
			}
		}
		for( int i=0 ; i<_exteriorNodes.size() ; i++ ) for( int j=0 ; j<_exteriorNodes[i].size() ; j++ )
		{
			if( geometryNodeDesignators[ _exteriorNodes[i][j] ]==GeometryNodeType::BOUNDARY ) MK_THROW( "Exterior node has geometry" );
			else if( geometryNodeDesignators[ _exteriorNodes[i][j] ]==GeometryNodeType::UNKNOWN )
			{
				geometryNodeDesignators[ _exteriorNodes[i][j] ] = GeometryNodeType::EXTERIOR;
				exteriorNodes.push_back( _exteriorNodes[i][j] );
			}
		}
	}

	// In subsequent passes, propagate from nodes marked as interior/exterior
	while( interiorNodes.size() || exteriorNodes.size() )
	{
		for( int i=0 ; i<_interiorNodes.size() ; i++ ) _interiorNodes[i].resize( 0 );
		for( int i=0 ; i<_exteriorNodes.size() ; i++ ) _exteriorNodes[i].resize( 0 );

		ThreadPool::ParallelFor( 0 , interiorNodes.size() , [&]( unsigned int thread , size_t i )
		{
			std::vector< const FEMTreeNode * > &__interiorNodes = _interiorNodes[thread];
			NeighborKey &neighborKey = neighborKeys[thread];
			Neighbors &neighbors = neighborKey.getNeighbors( interiorNodes[i] );

			// Iterate over the faces
			for( unsigned int d=0 ; d<Dim ; d++ ) for( int dir=0 ; dir<=2 ; dir+=2 )
			{
				unsigned int idx[Dim];
				for( unsigned int _d=0 ; _d<Dim ; _d++ ) idx[_d] = 1;
				idx[d] = dir;

				for( int depth=interiorNodes[i]->depth() ; depth>=0 ; depth-- )
				{
					const FEMTreeNode *node = neighborKey.neighbors[depth].neighbors( idx );
					if( node )
					{
						if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN ) __interiorNodes.push_back( node );
						break;
					}
				}
			}
		} );

		ThreadPool::ParallelFor( 0 , exteriorNodes.size() , [&]( unsigned int thread , size_t i )
		{
			std::vector< const FEMTreeNode * > &__exteriorNodes = _exteriorNodes[thread];
			NeighborKey &neighborKey = neighborKeys[thread];
			Neighbors &neighbors = neighborKey.getNeighbors( exteriorNodes[i] );

			// Iterate over the faces
			for( unsigned int d=0 ; d<Dim ; d++ ) for( int dir=0 ; dir<=2 ; dir+=2 )
			{
				unsigned int idx[Dim];
				for( unsigned int _d=0 ; _d<Dim ; _d++ ) idx[_d] = 1;
				idx[d] = dir;

				for( int depth=exteriorNodes[i]->depth() ; depth>=0 ; depth-- )
				{
					const FEMTreeNode *node = neighborKey.neighbors[depth].neighbors( idx );
					if( node )
					{
						if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN ) __exteriorNodes.push_back( node );
						break;
					}
				}
			}
		} );

		// Merge the interior/exterior nodes computed by the different threads
		{
			size_t interiorCount = 0 , exteriorCount = 0;
			for( int i=0 ; i<_interiorNodes.size() ; i++ ) interiorCount += _interiorNodes[i].size();
			for( int i=0 ; i<_exteriorNodes.size() ; i++ ) exteriorCount += _exteriorNodes[i].size();
			if( !interiorCount && !exteriorCount ) break;
			interiorNodes.resize( 0 ) , exteriorNodes.resize( 0 );
			interiorNodes.reserve( interiorCount ) , exteriorNodes.reserve( exteriorCount );

			for( int i=0 ; i<_interiorNodes.size() ; i++ ) for( int j=0 ; j<_interiorNodes[i].size() ; j++ )
			{
				if( geometryNodeDesignators[ _interiorNodes[i][j] ]==GeometryNodeType::BOUNDARY ) MK_THROW( "Interior node has geometry" );
				else if( geometryNodeDesignators[ _interiorNodes[i][j] ]==GeometryNodeType::UNKNOWN )
				{
					geometryNodeDesignators[ _interiorNodes[i][j] ] = GeometryNodeType::INTERIOR;
					interiorNodes.push_back( _interiorNodes[i][j] );
				}
			}
			for( int i=0 ; i<_exteriorNodes.size() ; i++ ) for( int j=0 ; j<_exteriorNodes[i].size() ; j++ )
			{
				if( geometryNodeDesignators[ _exteriorNodes[i][j] ]==GeometryNodeType::BOUNDARY ) MK_THROW( "Exterior node has geometry" );
				else if( geometryNodeDesignators[ _exteriorNodes[i][j] ]==GeometryNodeType::UNKNOWN )
				{
					geometryNodeDesignators[ _exteriorNodes[i][j] ] = GeometryNodeType::EXTERIOR;
					exteriorNodes.push_back( _exteriorNodes[i][j] );
				}
			}
		}
	}

	size_t correctionCount=0;
	std::function< void ( FEMTreeNode * ) > CorrectDesignatorsFromChildren = [&]( const FEMTreeNode *node )
	{
		if( node->children )
		{
			int interiorCount=0 , exteriorCount=0 , boundaryCount=0;
			for( int c=0 ; c<(1<<Dim) ; c++ )
			{
				CorrectDesignatorsFromChildren( node->children+c );
				if     ( geometryNodeDesignators[node->children+c]==GeometryNodeType::INTERIOR ) interiorCount++;
				else if( geometryNodeDesignators[node->children+c]==GeometryNodeType::EXTERIOR ) exteriorCount++;
				else if( geometryNodeDesignators[node->children+c]==GeometryNodeType::BOUNDARY ) boundaryCount++;
			}
			if( boundaryCount || ( exteriorCount && interiorCount ) )
			{
				if( geometryNodeDesignators[node]!=GeometryNodeType::UNKNOWN && geometryNodeDesignators[node]!=GeometryNodeType::BOUNDARY ) correctionCount++;
				geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY;
			}
			else if( interiorCount==(1<<Dim) )
			{
				if( geometryNodeDesignators[node]!=GeometryNodeType::UNKNOWN && geometryNodeDesignators[node]!=GeometryNodeType::INTERIOR ) correctionCount++;
				geometryNodeDesignators[node] = GeometryNodeType::INTERIOR;
			}
			else if( exteriorCount==(1<<Dim) )
			{
				if( geometryNodeDesignators[node]!=GeometryNodeType::UNKNOWN && geometryNodeDesignators[node]!=GeometryNodeType::EXTERIOR ) correctionCount++;
				geometryNodeDesignators[node] = GeometryNodeType::EXTERIOR;
			}
			else if( interiorCount )
			{
				if( geometryNodeDesignators[node]!=GeometryNodeType::UNKNOWN && geometryNodeDesignators[node]!=GeometryNodeType::INTERIOR && geometryNodeDesignators[node]!=GeometryNodeType::BOUNDARY ) correctionCount++;
				geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY;
			}
			else if( exteriorCount )
			{
				if( geometryNodeDesignators[node]!=GeometryNodeType::UNKNOWN && geometryNodeDesignators[node]!=GeometryNodeType::EXTERIOR && geometryNodeDesignators[node]!=GeometryNodeType::BOUNDARY ) correctionCount++;
				geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY;
			}
		}
	};
	CorrectDesignatorsFromChildren( root );
	if( correctionCount ) MK_WARN( "Adjusted designator inconsistencies: " , correctionCount );

	std::function< void ( FEMTreeNode * ) > SetUnknownDesignatorsFromParents = [&]( FEMTreeNode *node )
	{
		if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN && node->parent ) geometryNodeDesignators[node] = geometryNodeDesignators[node->parent];
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) SetUnknownDesignatorsFromParents( node->children + c );
	};
	std::function< void ( FEMTreeNode * ) > SetUnknownDesignatorsFromChildren = [&]( FEMTreeNode *node )
	{
		if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) SetUnknownDesignatorsFromChildren( node->children + c );
		if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN )
			if( node->children )
			{
				int interiorCount = 0 , exteriorCount = 0 , boundaryCount = 0;
				for( int c=0 ; c<(1<<Dim) ; c++ )
				{
					if     ( geometryNodeDesignators[node->children+c]==GeometryNodeType::INTERIOR ) interiorCount++;
					else if( geometryNodeDesignators[node->children+c]==GeometryNodeType::EXTERIOR ) exteriorCount++;
					else if( geometryNodeDesignators[node->children+c]==GeometryNodeType::BOUNDARY ) boundaryCount++;
				}
				if( interiorCount+exteriorCount+boundaryCount!=(1<<Dim) ) MK_THROW( "Children are unknown" );
				else if( boundaryCount==0 && interiorCount!=0 && exteriorCount!=0 ) MK_THROW( "Expected boundary between interior/exterior" );
				else if( boundaryCount!=0 ) geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY;
				else if( interiorCount!=0 ) geometryNodeDesignators[node] = GeometryNodeType::INTERIOR;
				else if( exteriorCount!=0 ) geometryNodeDesignators[node] = GeometryNodeType::INTERIOR;
			}
			else if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN ) MK_THROW( "Leaf node is unknown" );
	};
	SetUnknownDesignatorsFromParents( root );
	SetUnknownDesignatorsFromChildren( root );

	return geometryNodeDesignators;
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::TestGeometryNodeDesignators( const FEMTreeNode *root , const DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators )
{
	std::function< void ( const FEMTreeNode * ) > Test = [&]( const FEMTreeNode *node )
	{
		if( node->children )
		{
			if( node->nodeData.nodeIndex>=0 && node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node->nodeData.nodeIndex]!=GeometryNodeType::UNKNOWN )
			{
				GeometryNodeType type = geometryNodeDesignators[node->nodeData.nodeIndex];
				int interiorCount=0 , exteriorCount=0 , boundaryCount=0 , unknownCount=0;
				for( int c=0 ; c<(1<<Dim) ; c++ )
				{
					if( node->children[c].nodeData.nodeIndex>=0 && node->children[c].nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() )
					{
						if( geometryNodeDesignators[ node->children[c].nodeData.nodeIndex ]==GeometryNodeType::UNKNOWN  )  unknownCount++;
						if( geometryNodeDesignators[ node->children[c].nodeData.nodeIndex ]==GeometryNodeType::INTERIOR ) interiorCount++;
						if( geometryNodeDesignators[ node->children[c].nodeData.nodeIndex ]==GeometryNodeType::EXTERIOR ) exteriorCount++;
						if( geometryNodeDesignators[ node->children[c].nodeData.nodeIndex ]==GeometryNodeType::BOUNDARY ) boundaryCount++;
					}
				}
				if( boundaryCount || ( interiorCount && exteriorCount ) )
				{
					if( type!=GeometryNodeType::UNKNOWN && type!=GeometryNodeType::BOUNDARY ) MK_THROW( "Expected unknown or boundary, got: " , type , " | " , node->depthAndOffset() );
				}
				else if( interiorCount==(1<<Dim) )
				{
					if( type!=GeometryNodeType::UNKNOWN && type!=GeometryNodeType::INTERIOR ) MK_THROW( "Expected unknown or interior, got: " , type , " | " , node->depthAndOffset() );
				}
				else if( exteriorCount==(1<<Dim) )
				{
					if( type!=GeometryNodeType::UNKNOWN && type!=GeometryNodeType::EXTERIOR ) MK_THROW( "Expected unknown or exterior, got: " , type , " | " , node->depthAndOffset() );
				}
				else if( interiorCount )
				{
					if( type!=GeometryNodeType::UNKNOWN && type!=GeometryNodeType::INTERIOR && type!=GeometryNodeType::BOUNDARY ) MK_THROW( "Expected unknown, interior , or boundary, got: " , type , " | " , node->depthAndOffset() );
				}
				else if( exteriorCount==(1<<Dim) )
				{
					if( type!=GeometryNodeType::UNKNOWN && type!=GeometryNodeType::EXTERIOR && type!=GeometryNodeType::BOUNDARY ) MK_THROW( "Expected unknown, exterior, or boundary, got: " , type , " | " , node->depthAndOffset() );
				}
			}

			for( int c=0 ; c<(1<<Dim) ; c++ ) Test( node->children+c );
		}
	};

	Test( root );
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::PushGeometryNodeDesignatorsToFiner( const FEMTreeNode *root , DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators , unsigned int maxDepth )
{
	std::function< void ( const FEMTreeNode * ) > Push = [&]( const FEMTreeNode *node )
	{
		if( node->nodeData.nodeIndex>=0 && node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() )
		{
			if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN )
				if( node!=root ) geometryNodeDesignators[node] = geometryNodeDesignators[node->parent];
				else MK_THROW( "Root node should not be unknown" );
			else if( node!=root && geometryNodeDesignators[node]!=geometryNodeDesignators[node->parent] && geometryNodeDesignators[node->parent]!=GeometryNodeType::BOUNDARY )
			{
				int d , off[Dim];
				node->depthAndOffset( d , off );
				MK_THROW( "Child designator does not match parent: " , geometryNodeDesignators[node] , " != " , geometryNodeDesignators[node->parent] , " | " , d , " @ ( " , off[0] , " , " , off[1] , " , " , off[2] , " ) " );
			}
			if( node->depth()<(long long)maxDepth && node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) Push( node->children+c );
		}
	};

	Push( root );
}

template< unsigned int Dim , class Real >
void FEMTreeInitializer< Dim , Real >::PullGeometryNodeDesignatorsFromFiner( const FEMTreeNode *root , DenseNodeData< typename FEMTreeInitializer< Dim , Real >::GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators , unsigned int maxDepth )
{
	std::function< void ( const FEMTreeNode * ) > Pull = [&]( const FEMTreeNode *node )
	{
		if( node->nodeData.nodeIndex>=0 && node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() )
		{
			if( node->depth()<(long long)maxDepth && node->children && node->children->nodeData.nodeIndex>=0 && node->children->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() )
			{
				size_t interiorCount = 0 , exteriorCount = 0;
				for( int c=0 ; c<(1<<Dim) ; c++ )
				{
					Pull( node->children+c );
					if     ( geometryNodeDesignators[ node->children+c ]==GeometryNodeType::EXTERIOR ) exteriorCount++;
					else if( geometryNodeDesignators[ node->children+c ]==GeometryNodeType::INTERIOR ) interiorCount++;
				}
				if     ( interiorCount==(1<<Dim) ) geometryNodeDesignators[node] = GeometryNodeType::INTERIOR;
				else if( exteriorCount==(1<<Dim) ) geometryNodeDesignators[node] = GeometryNodeType::EXTERIOR;
				else                               geometryNodeDesignators[node] = GeometryNodeType::BOUNDARY;
			}
			else if( geometryNodeDesignators[node]==GeometryNodeType::UNKNOWN ) MK_THROW( "Should not have unknown nodes" );
		}
	};
	Pull( root );
}
