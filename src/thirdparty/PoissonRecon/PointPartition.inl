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

std::string FileDir( std::string dir , std::string header )
{
	if( dir.back()!=FileSeparator ) dir += std::string(1,FileSeparator);
	return dir + header;
}

std::string FileDir( std::string dir , std::string header , unsigned int clientIndex )
{
	std::stringstream sStream;
	sStream << header << "_" << clientIndex;
	return FileDir( dir , sStream.str() );
}

std::string FileName( std::string dir , unsigned int slab , unsigned int slabs , unsigned int filesPerDir )
{
	if( filesPerDir<=1 ) MK_THROW( "Need at least two files per directory" );

	if( !dir.length() ) dir = std::string( "." );
	if( dir.back()!=FileSeparator ) dir.push_back( FileSeparator );

	std::vector< unsigned int > factors;
	factors.push_back(slab);
	while( slabs>filesPerDir )
	{
		factors.push_back( (slab / filesPerDir) % filesPerDir );
		slab /= filesPerDir;
		slabs /= filesPerDir;
	}

	std::stringstream sStream;
	sStream << dir;

	for( unsigned int i=(unsigned int)factors.size()-1 ; i!=0 ; i-- ) sStream << factors[i] << std::string(1,FileSeparator);
	sStream << factors[0] << ".points";
	return sStream.str();
}

std::string FileName( std::string dir , std::string header , unsigned int slab , unsigned int slabs , unsigned int filesPerDir )
{
	return FileName( FileDir( dir , header ) , slab , slabs , filesPerDir );
}

std::string FileName( std::string dir , std::string header , unsigned int clientIndex , unsigned int slab , unsigned int slabs , unsigned int filesPerDir )
{
	return FileName( FileDir( dir , header , clientIndex ) , slab , slabs , filesPerDir );
}

std::string PointSetInfoName( std::string dir , std::string header )
{
	if( dir.back()==FileSeparator ) return dir + header + std::string( ".psi" );
	else return dir + std::string(1,FileSeparator) + header + std::string( ".psi" );
}


//////////////////
// PointSetInfo //
//////////////////
template< typename Real , unsigned int Dim >
PointSetInfo< Real , Dim >::PointSetInfo( void ) : modelToUnitCube( XForm< Real , Dim+1 >::Identity() ) , filesPerDir(-1) {}

template< typename Real , unsigned int Dim >
PointSetInfo< Real , Dim >::PointSetInfo( unsigned int slabs ) : modelToUnitCube( XForm< Real , Dim+1 >::Identity() ) , filesPerDir(-1) 
{
	pointsPerSlab.resize( slabs , 0 );
}

template< typename Real , unsigned int Dim >
PointSetInfo< Real , Dim >::PointSetInfo( BinaryStream &stream )
{
	if( !stream.read( header ) ) MK_THROW( "Failed to read header" );
	if( !stream.read( modelToUnitCube ) ) MK_THROW( "Failed to read model-to-unit-cube transform" );
	if( !stream.read( pointsPerSlab ) ) MK_THROW( "Failed to read points-per-slab" );
	{
		size_t sz;
		if( !stream.read( sz ) ) MK_THROW( "Failed to read number of auxiliary properties" );
		auxiliaryProperties.resize(sz);
		for( size_t i=0 ; i<sz ; i++ ) auxiliaryProperties[i].read( stream );
	}
	if( !stream.read( filesPerDir ) ) MK_THROW( "Failed to read files-per-directory" );
}

template< typename Real , unsigned int Dim >
void PointSetInfo< Real , Dim >::write( BinaryStream &stream ) const
{
	stream.write( header );
	stream.write( modelToUnitCube );
	stream.write( pointsPerSlab );
	{
		size_t sz = auxiliaryProperties.size();
		stream.write( sz );
		for( size_t i=0 ; i<sz ; i++ ) auxiliaryProperties[i].write( stream );
	}
	stream.write( filesPerDir );
}

void RemovePointSlabDirs( std::string dir ){ std::filesystem::remove_all( dir ); }
void CreatePointSlabDirs( std::string dir , unsigned int count , unsigned int filesPerDir )
{
	if( filesPerDir<=1 ) MK_THROW( "Need at least two files per directory" );
	if( !dir.length() ) dir = std::string( "." );
	if( dir.back()!=FileSeparator ) dir += std::string(1,FileSeparator);

	try{ std::filesystem::create_directories( dir ); }
	catch( ... ){ MK_THROW( "Failed to create directory: " , dir ); }

	unsigned int depth = 0;
	{
		size_t _filesPerDir = filesPerDir;
		while( count>_filesPerDir ) _filesPerDir *= filesPerDir , depth++;
	}

	auto _exp = []( unsigned int filesPerDir , unsigned int depth )
	{
		unsigned int e = 1;
		for( unsigned int d=0 ; d<depth ; d++ ) e *= filesPerDir;
		return e;
	};

	std::function< void ( std::string , unsigned int , unsigned int , unsigned int ) > MakeDirs = [&]( std::string dir , unsigned int count , unsigned int depth , unsigned int filesPerDir )
	{
		if( depth )
		{
			unsigned int _filesPerDir = _exp( filesPerDir , depth );
			for( unsigned int i=0 ; i*_filesPerDir<count ; i++ )
			{
				std::stringstream sStream;
				sStream << dir << i << FileSeparator;
				std::string _dir = sStream.str();
				try{ std::filesystem::create_directories( _dir ); }
				catch( ... ){ MK_THROW( "Failed to create directory: " , _dir ); }
				MakeDirs( _dir , std::min< unsigned int >( count-(i*_filesPerDir) , _filesPerDir ) , depth-1 , filesPerDir );
			}
		}
	};
	MakeDirs( dir , count , depth , filesPerDir );
}
///////////////
// Partition //
///////////////
// Energy:
//		An object supporting bool operator < ( const Energy & ) const;
template< typename Energy >
struct _DynamicProgrammingPartition
{
	// EnergyFunctor:
	//		A functor that that takes the start and end of the range and returns the associated energy
	// MergeFunctor:
	//		A functor that takes two energies and turns them into one
	template< typename EnergyFunctor , typename MergeFunctor >
	_DynamicProgrammingPartition( const EnergyFunctor &energy , const MergeFunctor &merge , unsigned int slabs , unsigned int interiorBoundaries )
	{
		_solutions.resize( slabs+1 );
		for( unsigned int start=0 ; start<_solutions.size() ; start++ )
		{
			_solutions[start].resize( slabs+1 );
			for( unsigned int end=start+1 ; end<_solutions[start].size() ; end++ ) _solutions[start][end].resize( interiorBoundaries+1 );
		}
		_energies.resize( slabs+1 );
		for( unsigned int i=0 ; i<=slabs ; i++ )
		{
			_energies[i].resize( slabs+1 );
			for( unsigned int j=i ; j<=slabs ; j++ ) _energies[i][j] = energy( i , j );
		}
		_getSolution( merge , 0 , slabs , interiorBoundaries , Energy() );
	}

	std::pair< Energy , std::vector< unsigned int > > solution( void ) const
	{
		std::pair< Energy , std::vector< unsigned int > > sol;
		unsigned int start = 0 , end = (unsigned int)_solutions[0].size()-1;
		unsigned int interiorBoundaries = (unsigned int)_solutions[start][end].size()-1;
		sol.first = _solutions[0].back().back().e;
		while( _solutions[start][end][interiorBoundaries].idx!=-1 )
		{
			sol.second.push_back( _solutions[start][end][interiorBoundaries].idx );
			start = _solutions[start][end][interiorBoundaries].idx;
			interiorBoundaries--;
		}
		return sol;
	}

	template< typename EnergyFunctor , typename MergeFunctor >
	static std::pair< Energy , std::vector< unsigned int > > Partition( const EnergyFunctor &energy , const MergeFunctor &merge , unsigned int slabs , unsigned int interiorBoundaries )
	{
		return _DynamicProgrammingPartition( energy , merge , slabs , interiorBoundaries ).solution();
	}

protected:
	struct _Solution
	{
		unsigned int idx;
		Energy e;
		_Solution( void ) : idx(-1){}
	};


	// A solution is indexed by the start index, end index, and the number of interior boundaries
	// The extent is defined by [start,end)
	std::vector< std::vector< std::vector< _Solution > > > _solutions;
	std::vector< std::vector< Energy > > _energies;
	template< typename MergeFunctor >
	Energy _getSolution( const MergeFunctor &merge , unsigned int start , unsigned int end , unsigned int interiorBoundaries , Energy minEnergy )
	{
		// Assuming that:
		//		start<end
		//		interiorBoundaries+1<(end-start)

		// If we have a solution for this range, return it
		if( _solutions[start][end][interiorBoundaries].idx!=-1 ) return _solutions[start][end][interiorBoundaries].e;

		// If the range does not need to be partitioned
		else if( interiorBoundaries==0 ) return ( _solutions[start][end][interiorBoundaries].e = _energies[start][end] );

		// Otherwise
		else
		{
			// 1. Split [start,end) into [start,_start) and [_start,end)
			// 2. Find interiorBoundaries-1 interior boundaries for [_start,end)
			// 3. Merge
			unsigned int minIndex = -1;
			unsigned int _interiorBoundaries = interiorBoundaries-1;
			for( unsigned int _start=start+1 ; _start<end ; _start++ )
			{
				if( _interiorBoundaries<(end-_start) )
				{
					if( !(minEnergy<_energies[start][_start]) )
					{
						Energy e = merge( _energies[start][_start] , _getSolution( merge , _start , end , _interiorBoundaries , minEnergy ) );
						if( minIndex==-1 || e<minEnergy ) minEnergy = e , minIndex = _start;
					}
				}
			}
			if( minIndex==-1 ) MK_THROW( "Could not find a solution: [ " , start , " , " , end , " ) " , interiorBoundaries );
			_solutions[start][end][interiorBoundaries].e = minEnergy;
			_solutions[start][end][interiorBoundaries].idx = minIndex;
			return minEnergy;
		}
	}
};
Partition::Partition( void ) : _slabSizes(1,0) , _starts(0) {}

Partition::Partition( unsigned int dCount , const std::vector< size_t > &slabSizes ) : _slabSizes( slabSizes )
{
	unsigned int sCount = (unsigned int)_slabSizes.size();
	_starts.resize( dCount-1 );
	for( unsigned int i=1 ; i<dCount ; i++ ) _starts[i-1] = ( sCount * i ) / dCount;
}

#ifdef ADAPTIVE_PADDING
void Partition::optimize( bool useMax )
#else // !ADAPTIVE_PADDING
void Partition::optimize( bool useMax , unsigned int padSize )
#endif // ADAPTIVE_PADDING
{
#ifdef ADAPTIVE_PADDING
#else // !ADAPTIVE_PADDING
	auto paddedStart = [&]( unsigned int start ){ return start>padSize ?  (start-padSize) : 0; };
	auto paddedEnd = [&]( unsigned int end ){ return ( end+padSize<=_slabSizes.size() ) ? (end+padSize) : (unsigned int)_slabSizes.size(); };
#endif // ADAPTIVE_PADDING

	struct L2Energy
	{
		double e;
		L2Energy( void ) : e( std::numeric_limits<double>::infinity() ) {}
		L2Energy( double e ) : e(e) {}
		bool operator < ( const L2Energy &energy ) const { return e<energy.e; }
	};
	struct MaxEnergy
	{
		std::vector< double > e;
		MaxEnergy( void ){}
		MaxEnergy( double e ) { this->e.resize(1,e); }
		bool operator < ( const MaxEnergy &energy ) const
		{
			for( unsigned int i=0 ; i<e.size() && i<energy.e.size() ; i++ )
			{
				if     ( e[i]<energy.e[i] ) return true;
				else if( e[i]>energy.e[i] ) return false;
			}
			return false;
		}
	};

	auto energy = [&]( unsigned int start , unsigned int end )
	{
		double e = 0;
#ifdef ADAPTIVE_PADDING
		for( unsigned int i=start ; i<end ; i++ ) e += _slabSizes[i];
#else // !ADAPTIVE_PADDING
		for( unsigned int i=paddedStart(start) ; i<paddedEnd(end) ; i++ ) e += _slabSizes[i];
#endif // ADAPTIVE_PADDING
		return e*e;
	};
	auto  l2Energy = [&]( unsigned int start , unsigned int end ){ return L2Energy( energy(start,end) ); };
	auto maxEnergy = [&]( unsigned int start , unsigned int end ){ return MaxEnergy( energy(start,end) ); };

	auto  l2Merge = [](  L2Energy e1 ,  L2Energy e2 ){ return  L2Energy( e1.e + e2.e ); };
	auto maxMerge = []( MaxEnergy e1 , MaxEnergy e2 )
	{
		MaxEnergy e;
		e.e.reserve( e1.e.size() + e2.e.size() );
		for( unsigned int i=0 ; i<e1.e.size() ; i++ ) e.e.push_back( e1.e[i] );
		for( unsigned int i=0 ; i<e2.e.size() ; i++ ) e.e.push_back( e2.e[i] );
		std::sort( e.e.begin() , e.e.end() , []( double e1 , double e2 ){ return e1>e2; } );
		return e;
	};

	if( useMax ) _starts = _DynamicProgrammingPartition< MaxEnergy >::Partition( maxEnergy , maxMerge , (unsigned)_slabSizes.size() , (unsigned int)_starts.size() ).second;
	else         _starts = _DynamicProgrammingPartition<  L2Energy >::Partition(  l2Energy ,  l2Merge , (unsigned)_slabSizes.size() , (unsigned int)_starts.size() ).second;
}

#ifdef ADAPTIVE_PADDING
std::pair< unsigned int , unsigned int > Partition::range( unsigned int i ) const
#else // !ADAPTIVE_PADDING
std::pair< unsigned int , unsigned int > Partition::range( unsigned int i , unsigned int padSize ) const
#endif // ADAPTIVE_PADDING
{
	unsigned int begin = i==0 ? 0 : _starts[i-1];
	unsigned int   end = i==_starts.size() ? (unsigned int)_slabSizes.size() : _starts[i];
#ifdef ADAPTIVE_PADDING
#else // !ADAPTIVE_PADDING
	if( begin<padSize ) begin = 0;
	else begin -= padSize;
	if( end+padSize>_slabSizes.size() ) end = (unsigned int)_slabSizes.size();
	else end += padSize;
#endif // ADAPTIVE_PADDING
	return std::pair< unsigned int , unsigned int >( begin , end );
}

#ifdef ADAPTIVE_PADDING
size_t Partition::size( unsigned int i ) const
#else // !ADAPTIVE_PADDING
size_t Partition::size( unsigned int i , unsigned int padSize ) const
#endif // ADAPTIVE_PADDING
{
	if( i>_starts.size() ) MK_THROW( "Index out of bounds: 0 <= " , i , " <= " , _starts.size() );
#ifdef ADAPTIVE_PADDING
	std::pair< unsigned int , unsigned int > r = range( i );
#else // !ADAPTIVE_PADDING
	std::pair< unsigned int , unsigned int > r = range( i , padSize );
#endif // ADAPTIVE_PADDING
	size_t count = 0;
	for( unsigned int j=r.first ; j<r.second ; j++ ) count += _slabSizes[j];
	return count;
}

size_t Partition::size( void ) const
{
	size_t count = 0;
#ifdef ADAPTIVE_PADDING
	for( unsigned int i=0 ; i<=_starts.size() ; i++ ) count += size(i);
#else // !ADAPTIVE_PADDING
	for( unsigned int i=0 ; i<=_starts.size() ; i++ ) count += size(i,0);
#endif // ADAPTIVE_PADDING
	return count;
}

#ifdef ADAPTIVE_PADDING
void Partition::printDistribution( void ) const
#else // !ADAPTIVE_PADDING
void Partition::printDistribution( unsigned int padSize ) const
#endif // ADAPTIVE_PADDING
{
	for( unsigned int i=0 ; i<=_starts.size() ; i++ )
	{
#ifdef ADAPTIVE_PADDING
		std::pair< unsigned int , unsigned int > r = range(i);
		std::cout << "Slab[ " << i << " ] [ " << r.first << " , " << r.second << " ) " << size(i) << std::endl;
#else // !ADAPTIVE_PADDING
		std::pair< unsigned int , unsigned int > r = range(i,0) , _r = range( i , padSize );
		if( padSize )
			std::cout << "Slab[ " << i << " ] [ " << r.first << " , " << r.second << " ) [ " << _r.first << " , " << _r.second << " ) " << size(i,0) << " " << size(i,padSize) << std::endl;
		else
			std::cout << "Slab[ " << i << " ] [ " << r.first << " , " << r.second << " ) " << size(i,0) << std::endl;
#endif // ADAPTIVE_PADDING
	}
}

#ifdef ADAPTIVE_PADDING
double Partition::l2Energy( void ) const
#else // !ADAPTIVE_PADDING
double Partition::l2Energy( unsigned int padSize ) const
#endif // ADAPTIVE_PADDING
{
	double e = 0;
	for( unsigned int i=0 ; i<=_starts.size() ; i++ )
	{
#ifdef ADAPTIVE_PADDING
		double d = (double)size(i);
#else // !ADAPTIVE_PADDING
		double d = (double)size(i,padSize);
#endif // ADAPTIVE_PADDING
		e += d*d;
	}
	return sqrt(e);
}

#ifdef ADAPTIVE_PADDING
double Partition::maxEnergy( void ) const
#else // !ADAPTIVE_PADDING
double Partition::maxEnergy( unsigned int padSize ) const
#endif // ADAPTIVE_PADDING
{
	double e = 0;
#ifdef ADAPTIVE_PADDING
	for( unsigned int i=0 ; i<=_starts.size() ; i++ ) e = std::max< double >( e , (double)size(i) );
#else // !ADAPTIVE_PADDING
	for( unsigned int i=0 ; i<=_starts.size() ; i++ ) e = std::max< double >( e , (double)size(i,padSize) );
#endif // ADAPTIVE_PADDING
	return e;
}

unsigned int Partition::slabs( void ) const { return (unsigned int)_slabSizes.size(); }

unsigned int Partition::partitions( void ) const{ return (unsigned int)_starts.size()+1; }

///////////////////////////////
// Read/Write Ply Properties //
///////////////////////////////
long ReadPLYProperties( FILE *fp , std::vector< PlyProperty > &properties )
{
	size_t sz;
	if( fread( &sz , sizeof( size_t ) , 1 , fp )!=1 ) MK_THROW( "Failed to read property size" );
	properties.resize( sz );
	FileStream fs(fp);
	for( size_t i=0 ; i<sz ; i++ ) properties[i].read( fs );
	return ftell( fp );
}

long ReadPLYProperties( const char *fileName , std::vector< PlyProperty > &properties )
{
	FILE *fp = fopen( fileName , "rb" );
	if( !fp ) MK_THROW( "Could not open file for reading: " , fileName );
	long pos = ReadPLYProperties( fp , properties );
	fclose( fp );
	return pos;
}

long WritePLYProperties( FILE *fp , const std::vector< PlyProperty > &properties )
{
	size_t sz = properties.size();
	fwrite( &sz , sizeof( size_t ) , 1 , fp );
	FileStream fs(fp);
	for( size_t i=0 ; i<sz ; i++ ) properties[i].write( fs );
	return ftell( fp );
}

long WritePLYProperties( const char *fileName , const std::vector< PlyProperty > &properties )
{
	FILE *fp = fopen( fileName , "wb" );
	if( !fp ) MK_THROW( "Could not open file for writing: " , fileName );
	long pos = WritePLYProperties( fp , properties );
	fclose( fp );
	return pos;
}

///////////////////////////////////
// BufferedBinaryInputDataStream //
///////////////////////////////////
template< typename InputFactory >
BufferedBinaryInputDataStream< InputFactory >::BufferedBinaryInputDataStream( const char *fileName , const InputFactory &factory , size_t bufferSize ) : _factory(factory) , _bufferSize(bufferSize) , _current(0) , _inBuffer(0)
{

	if( !_bufferSize )
	{
		MK_WARN_ONCE( "BufferSize cannot be zero , setting to one" );
		_bufferSize = 1;
	}
	_elementSize = _factory.bufferSize();
	_buffer = AllocPointer< char >( _elementSize*_bufferSize );
	_fp = fopen( fileName , "rb" );
	if( !_fp ) MK_THROW( "Could not open file for reading: " , fileName );
	std::vector< PlyProperty > properties;
	_inset = ReadPLYProperties( _fp , properties );
}

template< typename InputFactory >
BufferedBinaryInputDataStream< InputFactory >::~BufferedBinaryInputDataStream( void )
{
	FreePointer( _buffer );
	fclose( _fp );
}

template< typename InputFactory >
void BufferedBinaryInputDataStream< InputFactory >::reset( void )
{
	fseek( _fp , _inset , SEEK_SET );
	_current = 0;
	_inBuffer = 0;
}

template< typename InputFactory >
bool BufferedBinaryInputDataStream< InputFactory >::read( Data &d )
{
	if( _current==_inBuffer ) 
	{
		_inBuffer = fread( _buffer , _elementSize , _bufferSize , _fp );
		_current = 0;
	}
	if( !_inBuffer ) return false;

	_factory.fromBuffer( _buffer + _elementSize*_current , d );
	_current++;

	return true;
}

////////////////////////////////////
// BufferedBinaryOutputDataStream //
////////////////////////////////////

template< typename OutputFactory >
BufferedBinaryOutputDataStream< OutputFactory >::BufferedBinaryOutputDataStream( const char *fileName , const OutputFactory &factory , size_t bufferSize ) : _factory(factory) , _bufferSize(bufferSize) , _current(0)
{
	if( !_bufferSize )
	{
		MK_WARN_ONCE( "BufferSize cannot be zero , setting to one" );
		_bufferSize = 1;
	}
	_elementSize = _factory.bufferSize();
	_buffer = AllocPointer< char >( _elementSize*_bufferSize );
	_fp = fopen( fileName , "wb" );
	if( !_fp ) MK_THROW( "Could not open file for writing: " , fileName );
	std::vector< PlyProperty > properties( factory.plyWriteNum() );
	for( unsigned int i=0 ; i<factory.plyWriteNum() ; i++ ) properties[i] = factory.plyWriteProperty(i);
	_inset = WritePLYProperties( _fp , properties );
}

template< typename OutputFactory >
BufferedBinaryOutputDataStream< OutputFactory >::~BufferedBinaryOutputDataStream( void )
{
	if( _current ) fwrite( _buffer , _elementSize , _current , _fp );
	FreePointer( _buffer );
	fclose( _fp );
}

template< typename OutputFactory >
void BufferedBinaryOutputDataStream< OutputFactory >::reset( void )
{
	fseek( _fp , _inset , SEEK_SET );
	_current = 0;
}

template< typename OutputFactory >
size_t BufferedBinaryOutputDataStream< OutputFactory >::write( const Data &d )
{
	if( _current==_bufferSize ) 
	{
		fwrite( _buffer , _elementSize , _bufferSize , _fp );
		_current = 0;
	}
	_factory.toBuffer( d , _buffer + _elementSize*_current );
	return _current++;
}

template< typename OutputFactory >
size_t BufferedBinaryOutputDataStream< OutputFactory >::size( void ) const { return _current; }
