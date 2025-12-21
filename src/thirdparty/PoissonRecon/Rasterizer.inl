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

///////////////////////////////////
// Rasterizer::_RegularGridIndex //
///////////////////////////////////
template< typename Real , unsigned int Dim >
Rasterizer< Real , Dim >::_RegularGridIndex::_RegularGridIndex( void )
{
	depth = 0;
	for( int d=0 ; d<Dim ; d++ ) index[d] = 0;
}

template< typename Real , unsigned int Dim >
Rasterizer< Real , Dim >::_RegularGridIndex::_RegularGridIndex( unsigned int depth , Point< Real , Dim > point )
{
	this->depth = depth;
	// After clipping the coordinate could be equal to exactly one, in which case we want to round down.
	for( unsigned int d=0 ; d<Dim ; d++ ) index[d] = std::min< unsigned int >( (unsigned int)( point[d] * (1<<depth) ) , (1<<depth)-1 );
}

template< typename Real , unsigned int Dim >
template< unsigned int K >
Rasterizer< Real , Dim >::_RegularGridIndex::_RegularGridIndex( unsigned int maxDepth , Simplex< Real , Dim , K > simplex )
{
	for( depth=0 ; depth<maxDepth ; depth++ )
	{
		_RegularGridIndex idx( depth , simplex[0] );

		bool done = false;
		for( int k=1 ; k<=K && !done ; k++ ) if( _RegularGridIndex( depth , simplex[k] )!=idx ) done = true;
		if( done ) break;
	}
	if( depth==0 ) MK_THROW( "Simplex is not in unit cube: " , simplex );
	else *this = _RegularGridIndex( depth-1 , simplex[0] );
}


template< typename Real , unsigned int Dim >
bool Rasterizer< Real , Dim >::_RegularGridIndex::operator != ( const _RegularGridIndex &idx ) const
{
	if( depth!=idx.depth ) return false;
	for( int d=0 ; d<Dim ; d++ ) if( index[d]!=idx.index[d] ) return true;
	return false;
}

template< typename Real , unsigned int Dim >
typename Rasterizer< Real , Dim >::_RegularGridIndex Rasterizer< Real , Dim >::_RegularGridIndex::child( unsigned int c ) const
{
	_RegularGridIndex idx;
	idx.depth = depth+1;
	for( int d=0 ; d<Dim ; d++ ) idx.index[d] = index[d]*2 + ( c&(1<<d) ? 1 : 0 );
	return idx;
}

////////////////
// Rasterizer //
////////////////

template< typename Real , unsigned int Dim >
template< typename IndexType , unsigned int K >
size_t Rasterizer< Real , Dim >::_Rasterize( _RegularGridMutexes &mutexes , SimplexRasterizationGrid< IndexType , K > &raster , IndexType simplexIndex , Simplex< Real , Dim , K > simplex , unsigned int maxDepth , _RegularGridIndex idx )
{
	if( idx.depth==maxDepth )
	{
		// If the simplex has non-zero size, add it to the list
		Real weight = simplex.measure();
		if( weight && weight==weight )
		{
			std::lock_guard< std::mutex > lock( mutexes( idx.index ) );
			raster( idx.index ).push_back( std::pair< IndexType , Simplex< Real , Dim , K > >( simplexIndex , simplex ) );
		}
		return 1;
	}
	else
	{
		size_t sCount = 0;

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		for( unsigned int d=0 ; d<Dim ; d++ ) center[d] = (Real)( idx.index[d] + 0.5 ) / (1<<idx.depth);

		std::vector< std::vector< Simplex< Real , Dim , K > > > childSimplices( 1 );
		childSimplices[0].push_back( simplex );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , K > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) sCount += _Rasterize( mutexes , raster , simplexIndex , childSimplices[c][i] , maxDepth , idx.child(c) );
		return sCount;
	}
}

template< typename Real , unsigned int Dim >
template< typename IndexType , unsigned int K >
size_t Rasterizer< Real , Dim >::_Rasterize( SimplexRasterizationGrid< IndexType , K > &raster , IndexType simplexIndex , Simplex< Real , Dim , K > simplex , unsigned int maxDepth , _RegularGridIndex idx )
{
	if( idx.depth==maxDepth )
	{
		// If the simplex has non-zero size, add it to the list
		Real weight = simplex.measure();
		if( weight && weight==weight ) raster( idx.index ).push_back( std::pair< IndexType , Simplex< Real , Dim , K > >( simplexIndex , simplex ) );
		return 1;
	}
	else
	{
		size_t sCount = 0;

		// Split up the simplex and pass the parts on to the children
		Point< Real , Dim > center;
		for( unsigned int d=0 ; d<Dim ; d++ ) center[d] = (Real)( idx.index[d] + 0.5 ) / (1<<idx.depth);

		std::vector< std::vector< Simplex< Real , Dim , K > > > childSimplices( 1 );
		childSimplices[0].push_back( simplex );
		for( int d=0 ; d<Dim ; d++ )
		{
			Point< Real , Dim > n ; n[Dim-d-1] = 1;
			std::vector< std::vector< Simplex< Real , Dim , K > > > temp( (int)( 1<<(d+1) ) );
			for( int c=0 ; c<(1<<d) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) childSimplices[c][i].split( n , center[Dim-d-1] , temp[2*c] , temp[2*c+1] );
			childSimplices = temp;
		}
		for( int c=0 ; c<(1<<Dim) ; c++ ) for( int i=0 ; i<childSimplices[c].size() ; i++ ) sCount += _Rasterize( raster , simplexIndex , childSimplices[c][i] , maxDepth , idx.child(c) );
		return sCount;
	}
}
template< typename Real , unsigned int Dim >
template< typename IndexType , unsigned int K >
typename Rasterizer< Real , Dim >::template SimplexRasterizationGrid< IndexType , K > Rasterizer< Real , Dim >::Rasterize( const SimplicialComplex< Real , Dim , K > &simplicialComplex , unsigned int depth , ThreadSafety threadSafety )
{
	unsigned int res = 1<<depth;

	SimplexRasterizationGrid< IndexType , K > raster;
	{
		unsigned int _res[Dim];
		for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
		raster.resize( _res );
	}

	if( threadSafety.type==ThreadSafety::MUTEXES )
	{
		if( threadSafety.lockDepth>depth ) MK_THROW( "Lock depth cannot excceed depth: " , threadSafety.lockDepth , " <= " , depth );
		_RegularGridMutexes mutexes( threadSafety.lockDepth , depth );

		ThreadPool::ParallelFor( 0 , simplicialComplex.size() , [&]( unsigned int t , size_t  i )
		{
			std::vector< Simplex< Real , Dim , K > > subSimplices;
			subSimplices.push_back( simplicialComplex[i] );

			// Clip the simplex to the unit cube
			{
				for( int d=0 ; d<Dim ; d++ )
				{
					Point< Real , Dim > n;
					n[d] = 1;
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
						subSimplices = front;
					}
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
						subSimplices = back;
					}
				}
			}
			for( int j=0 ; j<subSimplices.size() ; j++ ) _Rasterize< IndexType , K >( mutexes , raster , (IndexType)i , subSimplices[j] , depth , _RegularGridIndex( depth , subSimplices[j] ) );
		} );
	}
	else if( threadSafety.type==ThreadSafety::SINGLE_THREADED )
	{
		for( size_t i=0 ; i<simplicialComplex.size() ; i++ )
		{
			std::vector< Simplex< Real , Dim , K > > subSimplices;
			subSimplices.push_back( simplicialComplex[i] );

			// Clip the simplex to the unit cube
			{
				for( int d=0 ; d<Dim ; d++ )
				{
					Point< Real , Dim > n;
					n[d] = 1;
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
						subSimplices = front;
					}
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
						subSimplices = back;
					}
				}
			}
			for( int j=0 ; j<subSimplices.size() ; j++ ) _Rasterize< IndexType , K >( raster , (IndexType)i , subSimplices[j] , depth , _RegularGridIndex( depth , subSimplices[j] ) );
		}
	}
	else if( threadSafety.type==ThreadSafety::MAP_REDUCE )
	{
		std::vector< SimplexRasterizationGrid< IndexType , K > > rasters( ThreadPool::NumThreads() );
		for( int t=0 ; t<rasters.size() ; t++ )
		{
			unsigned int _res[Dim];
			for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
			rasters[t].resize( _res );
		}

		// Map
		ThreadPool::ParallelFor( 0 , simplicialComplex.size() , [&]( unsigned int t , size_t  i )
		{
			std::vector< Simplex< Real , Dim , K > > subSimplices;
			subSimplices.push_back( simplicialComplex[i] );

			// Clip the simplex to the unit cube
			{
				for( int d=0 ; d<Dim ; d++ )
				{
					Point< Real , Dim > n;
					n[d] = 1;
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 0 , back , front );
						subSimplices = front;
					}
					{
						std::vector< Simplex< Real , Dim , K > > back , front;
						for( int i=0 ; i<subSimplices.size() ; i++ ) subSimplices[i].split( n , 1 , back , front );
						subSimplices = back;
					}
				}
			}
			for( int j=0 ; j<subSimplices.size() ; j++ ) _Rasterize< IndexType , K >( rasters[t] , (IndexType)i , subSimplices[j] , depth , _RegularGridIndex( depth , subSimplices[j] ) );
		} );

		// Reduce
		ThreadPool::ParallelFor( 0 , raster.resolution() , [&]( unsigned int , size_t i )
		{
			size_t count = 0;
			for( int t=0 ; t<rasters.size() ; t++ ) count += rasters[t][i].size();
			raster[i].reserve( count );
			for( int t=0 ; t<rasters.size() ; t++ ) for( int j=0 ; j<rasters[t][i].size() ; j++ ) raster[i].push_back( rasters[t][i][j] );
		} );
	}
	else MK_THROW( "Unrecognized thread safety type: " , threadSafety.type );
	return raster;
}
