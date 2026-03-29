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

#include "PreProcessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Ply.h"
#include "VertexFactory.h"
#include "Image.h"
#include "RegularGrid.h"
#include "DataStream.imp.h"
#include "Reconstructors.h"

using namespace PoissonRecon;

CmdLineParameter< char* >
	In( "in" ) ,
	Samples( "samples" ) ,
	OutMesh( "mesh" ) ,
	OutTree( "tree" ) ,
	OutSlice( "slice" ) ,
	OutGrid( "grid" );

CmdLineReadable
	PolygonMesh( "polygonMesh" ) ,
	NonManifold( "nonManifold" ) ,
	FlipOrientation( "flip" ) ,
	ASCII( "ascii" ) ,
	NonLinearFit( "nonLinearFit" ) ,
	PrimalGrid( "primalGrid" ) ,
	Verbose( "verbose" );

CmdLineParameter< int >
	ParallelType( "parallel" , 0 ) ,
	ScheduleType( "schedule" , (int)ThreadPool::Schedule ) ,
	ThreadChunkSize( "chunkSize" , (int)ThreadPool::ChunkSize ) ,
	IsoSlabDepth( "sDepth" , 0 ) ,
	IsoSlabStart( "sStart" , 0 ) ,
	IsoSlabEnd  ( "sEnd"   , 1 ) ,
	SliceDepth( "sliceDepth" ) ,
	SliceIndex( "sliceIndex" ) ,
	TreeScale( "treeScale" , 1 ) ,
	TreeDepth( "treeDepth" , -1 );


CmdLineParameter< float >
	IsoValue( "iso" , 0.f );

CmdLineReadable* params[] =
{
	&In , 
	&Samples ,
	&OutMesh , &NonManifold , &PolygonMesh , &FlipOrientation , &ASCII , &NonLinearFit , &IsoValue ,
	&OutGrid , &PrimalGrid ,
	&OutTree , &TreeScale , &TreeDepth ,
	&Verbose , 
	&ParallelType ,
	&ScheduleType ,
	&ThreadChunkSize ,
	&IsoSlabDepth , &IsoSlabStart , &IsoSlabEnd ,
	&OutSlice , &SliceDepth , &SliceIndex,
	NULL
};

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input tree>\n" , In.name );
	printf( "\t[--%s sample positions>]\n" , Samples.name );
	printf( "\t[--%s <ouput triangle mesh>]\n" , OutMesh.name );
	printf( "\t[--%s <ouput grid>]\n" , OutGrid.name );
	printf( "\t[--%s <ouput tree grid>]\n" , OutTree.name );
	printf( "\t[--%s <tree scale factor>=%d]\n" , TreeScale.name , TreeScale.value );
	printf( "\t[--%s <tree depth>=%d]\n" , TreeDepth.name , TreeDepth.value );
	printf( "\t[--%s <parallel type>=%d]\n" , ParallelType.name , ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , ScheduleType.name , ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , ThreadChunkSize.name , ThreadChunkSize.value );
	printf( "\t[--%s <iso-value for extraction>=%f]\n" , IsoValue.name , IsoValue.value );
	printf( "\t[-%s <extraction slab depth>=%d]\n" , IsoSlabDepth.name , IsoSlabDepth.value );
	printf( "\t[-%s <extraction slab start>=%d]\n" , IsoSlabStart.name , IsoSlabStart.value );
	printf( "\t[-%s <extraction slab end>=%d]\n"   , IsoSlabEnd  .name , IsoSlabEnd  .value );
	printf( "\t[--%s <output slice name>]\n" , OutSlice.name );
	printf( "\t[--%s <output slice depth>]\n" , SliceDepth.name );
	printf( "\t[--%s <output slice index>]\n" , SliceIndex.name );
	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s]\n" , NonLinearFit.name );
	printf( "\t[--%s]\n" , FlipOrientation.name );
	printf( "\t[--%s]\n" , PrimalGrid.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

template< typename Real , unsigned int Dim >
void WriteGrid( const char *fileName , ConstPointer( Real ) values , unsigned int res , XForm< Real , Dim+1 > voxelToModel , bool verbose , bool normalize=true )
{
	char *ext = GetFileExtension( fileName );

	if( Dim==2 && ImageWriter::ValidExtension( ext ) )
	{
		unsigned int totalResolution = 1;
		for( int d=0 ; d<Dim ; d++ ) totalResolution *= res;

		// Compute average
		Real avg = 0;
		std::vector< Real > avgs( ThreadPool::NumThreads() , 0 );
		ThreadPool::ParallelFor( 0 , totalResolution , [&]( unsigned int thread , size_t i ){ avgs[thread] += values[i]; } );
		for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) avg += avgs[t];
		avg /= (Real)totalResolution;

		// Compute standard deviation
		Real std = 0;
		std::vector< Real > stds( ThreadPool::NumThreads() , 0 );
		ThreadPool::ParallelFor( 0 , totalResolution , [&]( unsigned int thread , size_t i ){ stds[thread] += ( values[i] - avg ) * ( values[i] - avg ); } );
		for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) std += stds[t];
		std = (Real)sqrt( std / totalResolution );

		if( verbose )
		{
			printf( "Grid to image: [%.2f,%.2f] -> [0,255]\n" , avg - 2*std , avg + 2*std );
			printf( "Transform:\n" );
			for( int i=0 ; i<Dim+1 ; i++ )
			{
				printf( "\t" );
				for( int j=0 ; j<Dim+1 ; j++ ) printf( " %f" , voxelToModel(j,i) );
				printf( "\n" );
			}
		}

		unsigned char *pixels = new unsigned char[ totalResolution*3 ];
		ThreadPool::ParallelFor( 0 , totalResolution , [&]( unsigned int , size_t i )
		{
			Real v = (Real)std::min< Real >( (Real)1. , std::max< Real >( (Real)-1. , ( values[i] - avg ) / (2*std ) ) );
			v = (Real)( ( v + 1. ) / 2. * 256. );
			unsigned char color = (unsigned char )std::min< Real >( (Real)255. , std::max< Real >( (Real)0. , v ) );
			for( int c=0 ; c<3 ; c++ ) pixels[i*3+c ] = color;
		}
		);
		ImageWriter::Write( fileName , pixels , res , res , 3 );
		delete[] pixels;
	}
	else if( !strcasecmp( ext , "iso" ) )
	{
		FILE *fp = fopen( fileName , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , fileName );
		int r = (int)res;
		fwrite( &r , sizeof(int) , 1 , fp );
		size_t count = 1;
		for( unsigned int d=0 ; d<Dim ; d++ ) count *= res;
		fwrite( values , sizeof(Real) , count , fp );
		fclose( fp );
	}
	else
	{
		unsigned int _res[Dim];
		for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
		RegularGrid< Real , Dim >::Write( fileName , _res , values , voxelToModel );
	}
	delete[] ext;
}
template< unsigned int Dim , class Real , unsigned int FEMSig >
void _Execute( const FEMTree< Dim , Real > *tree , XForm< Real , Dim+1 > modelToUnitCube , BinaryStream &stream )
{
	ThreadPool::ParallelizationType= (ThreadPool::ParallelType)ParallelType.value;
	static const unsigned int Degree = FEMSignature< FEMSig >::Degree;
	DenseNodeData< Real , IsotropicUIntPack< Dim , FEMSig > > coefficients;

	coefficients.read( stream );

	// Evaluate at the sample positions
	if( Samples.set )
	{
		InputDataStream< Point< Real , Dim > > *pointStream;
		typedef VertexFactory::PositionFactory< Real , Dim > InFactory;
		char* ext = GetFileExtension( Samples.value );
		if     ( !strcasecmp( ext , "bpts" ) ) pointStream = new BinaryInputDataStream< InFactory >( Samples.value , InFactory() );
		else if( !strcasecmp( ext , "ply"  ) ) pointStream = new    PLYInputDataStream< InFactory >( Samples.value , InFactory() );
		else                                   pointStream = new  ASCIIInputDataStream< InFactory >( Samples.value , InFactory() );
		delete[] ext;
		typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< IsotropicUIntPack< Dim , FEMSig > , 0 > evaluator( tree , coefficients );
		static const unsigned int CHUNK_SIZE = 1024;
		Point< Real , Dim > points[ CHUNK_SIZE ];
		Real values[ CHUNK_SIZE ];
		size_t pointsRead;
		auto ReadBatch = [&]( void )
		{
			size_t c=0;
			for( size_t i=0 ; i<CHUNK_SIZE ; i++ , c++ ) if( !pointStream->read( points[i] ) ) break;
			return c;
		};
		while( ( pointsRead=ReadBatch() ) )
		{
			ThreadPool::ParallelFor( 0 , pointsRead , [&]( unsigned int thread , size_t j )
			{
				Point< Real , Dim > p = modelToUnitCube * points[j];
				bool inBounds = true;
				for( int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) inBounds = false;
				if( inBounds ) values[j] = evaluator.values( modelToUnitCube * points[j] , thread )[0];
				else           values[j] = (Real)nan( "" );
			}
			);
			for( int j=0 ; j<pointsRead ; j++ ) printf( "%g %g %g\n" , points[j][0] , points[j][1] , values[j] );
		}

		delete pointStream;
	}

	// Output the grid
	if( OutGrid.set )
	{
		int res = 0;
		double t = Time();
		Pointer( Real ) values = tree->template regularGridEvaluate< true >( coefficients , res , -1 , PrimalGrid.set );
		if( Verbose.set ) printf( "Got grid: %.2f(s)\n" , Time()-t );
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );
		WriteGrid< Real , Dim >( OutGrid.value , values , res , modelToUnitCube.inverse() * voxelToUnitCube , Verbose.set );
		DeletePointer( values );
	}

	// Output the tree
	if( OutTree.set )
	{
		DenseNodeData< Real , IsotropicUIntPack< Dim , FEMTrivialSignature > > _coefficients = tree->initDenseNodeData( IsotropicUIntPack< Dim , FEMTrivialSignature >() );
		ThreadPool::ParallelFor( 0 , _coefficients.size() , [&]( unsigned int , size_t i ){ _coefficients[i] = 0; } );

		if( TreeDepth.value!=-1 )
		{
			ThreadPool::ParallelFor
			(
				tree->nodesBegin(TreeDepth.value) , tree->nodesEnd(TreeDepth.value) ,
				[&]( unsigned int , size_t i )
			{
				const typename FEMTree< Dim , Real >::FEMTreeNode *node = tree->node( (node_index_type)i );
				if( tree->isValidFEMNode( IsotropicUIntPack< Dim , FEMTrivialSignature >() , node ) ) _coefficients[ node ] = (Real)i;
			}
			);
		}
		else
		{
			ThreadPool::ParallelFor
			(
				tree->nodesBegin(0) , tree->nodesEnd( tree->depth() ) ,
				[&]( unsigned int , size_t i )
			{
				const typename FEMTree< Dim , Real >::FEMTreeNode *node = tree->node( (node_index_type)i );
				if( tree->isValidFEMNode( IsotropicUIntPack< Dim , FEMTrivialSignature >() , node ) && !tree->isValidFEMNode( IsotropicUIntPack< Dim , FEMTrivialSignature >() , node->children ) )
					_coefficients[ node ] = (Real)i;
			}
			);
		}
		int res = 0;
		double t = Time();
		Pointer( Real ) values = tree->template regularGridEvaluate< true >( _coefficients , res , -1 , false );
		if( Verbose.set ) printf( "Got grid: %.2f(s)\n" , Time()-t );

		int _res = res * TreeScale.value + 1;
		size_t count = 1;
		for( int d=0 ; d<Dim ; d++ ) count *= _res;

		Pointer( Real ) dValues = NewPointer< Real >( count );

		// In the original indexing, index i corresponds to the position
		//		i -> (i+0.5)/res
		// In the new indexing, index j corresponds to the position
		//		j -> j/(_res-1)
		// The two neighbors of index j are:
		//		j0 = j/(_res-1) - 1/(2*scale)
		//		j1 = j/(_res-1) + 1/(2*scale)

		{
			bool boundary[Dim+1];
			int idx[Dim] , idx0[Dim] , idx1[Dim];
			boundary[0] = false;

			auto Index = [&]( const int i[] , unsigned int r )
			{
				size_t index = 0;
				for( int d=Dim-1 ; d>=0 ; d-- ) index = index*r + (size_t)i[d];
				return index;
			};
			auto _Index = [&]( const int i[] , unsigned int r )
			{
				size_t index = 0;
				for( int d=Dim-1 ; d>=0 ; d-- )
				{
					std::cout << index << " * " << r << " + " << i[d] << " = ";
					index = index*r + (size_t)i[d];
					std::cout <<index << std::endl;
				}
				return index;
			};

			WindowLoop< Dim >::Run
			(
				0 , _res ,
				[&]( int d , int i )
			{
				idx [d] = i;
				idx0[d] = (int)floor( ( (double)i/(_res-1) - 1./(2.*TreeScale.value*res ) ) * res );
				idx1[d] = (int)floor( ( (double)i/(_res-1) + 1./(2.*TreeScale.value*res ) ) * res );
				boundary[d+1] = boundary[d] || idx0[d]==-1 || idx1[d]==res;
			} ,
				[&]( void )
			{
				if( !boundary[Dim] )
				{
					Real v1 = values[ Index( idx0 , res ) ];
					int _idx[Dim];
					for( int d=0 ; d<Dim ; d++ ) _idx[d] = idx0[d];
					for( int d=0 ; d<Dim ; d++ )
					{
						_idx[d] = idx1[d];
						Real v2 = values[ Index( _idx , res ) ];
						_idx[d] = idx0[d];
						if( v1!=v2 ) boundary[Dim] = true;
					}
				}
				dValues[ Index( idx , _res ) ] = (Real)( boundary[Dim] ? 0 : 1 );
			}
			);
		}

		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );
		WriteGrid< Real , Dim >( OutTree.value , dValues , _res , modelToUnitCube.inverse() * voxelToUnitCube , Verbose.set , false );
		DeletePointer( values );
		DeletePointer( dValues );
	}

	if( OutSlice.set && SliceDepth.set && SliceIndex.set )
	{
		using SliceFEMTreeNode = RegularTreeNode< Dim-1 , FEMTreeNodeData , depth_and_offset_type >;

		FEMTree< Dim-1 , Real > *sliceTree = FEMTree< Dim-1 , Real >::template Slice< Degree , 0 >( *tree , SliceDepth.value , SliceIndex.value , true , MEMORY_ALLOCATOR_BLOCK_SIZE );

		DenseNodeData< Real , IsotropicUIntPack< Dim-1 , FEMSig > > sliceCoefficients = sliceTree->initDenseNodeData( IsotropicUIntPack< Dim-1 , FEMSig >() );
		sliceTree->template slice< 0 , FEMSig >( *tree , 0 , coefficients , sliceCoefficients , SliceDepth.value , SliceIndex.value );

		XForm< Real , Dim > sliceModelToUnitCube;
		sliceModelToUnitCube( Dim-1 , Dim-1 ) = 1.;
		for( unsigned int i=0 ; i<(Dim-1) ; i++ )
		{
			for( unsigned int j=0 ; j<(Dim-1) ; j++ ) sliceModelToUnitCube(i,j) = modelToUnitCube(i,j);
			sliceModelToUnitCube(Dim-1,i) = modelToUnitCube(Dim,i);
		}
		FILE *fp = fopen( OutSlice.value , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , OutSlice.value );
		FileStream fs(fp);
		FEMTree< Dim-1 , Real >::WriteParameter( fs );
		DenseNodeData< Real , IsotropicUIntPack< Dim-1 , FEMSig > >::WriteSignatures( fs );
		sliceTree->write( fs , false );
		fs.write( sliceModelToUnitCube );
		sliceCoefficients.write( fs );
		fclose( fp );

		delete sliceTree;
	}

	// Output the mesh
	if constexpr( Dim==2 || Dim==3 ) if( OutMesh.set )
	{
		double t = Time();

		// A description of the output vertex information
		using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , false , false >;

		// A factory generating the output vertices
		using Factory = typename VInfo::Factory;
		Factory factory = VInfo::GetFactory();

		// A backing stream for the vertices
		Reconstructor::OutputInputFactoryTypeStream< Real , Dim , Factory , false , true > vertexStream( factory , VInfo::Convert );
		Reconstructor::OutputInputFaceStream< Dim-1 , false , true > faceStream;

		// Extract the mesh
		Reconstructor::TransformedOutputLevelSetVertexStream< Real , Dim > _vertexStream( modelToUnitCube.inverse() , vertexStream );
		if      constexpr( Dim==3 ) LevelSetExtractor< Real , Dim >::Extract( IsotropicUIntPack< Dim , FEMSig >() , UIntPack< 0 >() , *tree , ( typename FEMTree< Dim , Real >::template DensityEstimator< 0 >* )NULL , coefficients , IsoValue.value , IsoSlabDepth.value , IsoSlabStart.value , IsoSlabEnd.value , _vertexStream , faceStream , NonLinearFit.set , false , !NonManifold.set , PolygonMesh.set , FlipOrientation.set );
		else if constexpr( Dim==2 ) LevelSetExtractor< Real , Dim >::Extract( IsotropicUIntPack< Dim , FEMSig >() , UIntPack< 0 >() , *tree , ( typename FEMTree< Dim , Real >::template DensityEstimator< 0 >* )NULL , coefficients , IsoValue.value ,                                                              _vertexStream , faceStream , NonLinearFit.set , false , FlipOrientation.set );

		if( Verbose.set ) printf( "Got level-set: %.2f(s)\n" , Time()-t );
		if( Verbose.set ) printf( "Vertices / Faces: %llu / %llu\n" , (unsigned long long)vertexStream.size() , (unsigned long long)faceStream.size() );

		// Write the mesh to a .ply file
		std::vector< std::string > noComments;
		PLY::Write< Factory , node_index_type , Real , Dim >( OutMesh.value , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ASCII.set ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
	}
}

template< unsigned int Dim , class Real , BoundaryType BType >
void Execute( BinaryStream &stream , int degree , FEMTree< Dim , Real > *tree , XForm< Real , Dim+1 > modelToUnitCube )
{
	switch( degree )
	{
		case 1: _Execute< Dim , Real , FEMDegreeAndBType< 1 , BType >::Signature >( tree , modelToUnitCube , stream ) ; break;
		case 2: _Execute< Dim , Real , FEMDegreeAndBType< 2 , BType >::Signature >( tree , modelToUnitCube , stream ) ; break;
		case 3: _Execute< Dim , Real , FEMDegreeAndBType< 3 , BType >::Signature >( tree , modelToUnitCube , stream ) ; break;
		case 4: _Execute< Dim , Real , FEMDegreeAndBType< 4 , BType >::Signature >( tree , modelToUnitCube , stream ) ; break;
		default: MK_THROW( "Only B-Splines of degree 1 - 4 are supported" );
	}
}

template< unsigned int Dim , class Real >
void Execute( BinaryStream &stream , int degree , BoundaryType bType )
{
	XForm< Real , Dim+1 > modelToUnitCube;
	FEMTree< Dim , Real > tree( stream , MEMORY_ALLOCATOR_BLOCK_SIZE );
	stream.read( modelToUnitCube );

	if( Verbose.set ) printf( "All Nodes / Active Nodes / Ghost Nodes: %llu / %llu / %llu\n" , (unsigned long long)tree.allNodes() , (unsigned long long)tree.activeNodes() , (unsigned long long)tree.ghostNodes() );

	switch( bType )
	{
		case BOUNDARY_FREE:      return Execute< Dim , Real , BOUNDARY_FREE      >( stream , degree , &tree , modelToUnitCube );
		case BOUNDARY_NEUMANN:   return Execute< Dim , Real , BOUNDARY_NEUMANN   >( stream , degree , &tree , modelToUnitCube );
		case BOUNDARY_DIRICHLET: return Execute< Dim , Real , BOUNDARY_DIRICHLET >( stream , degree , &tree , modelToUnitCube );
		default: MK_THROW( "Not a valid boundary type: " , bType );
	}
}

int main( int argc , char* argv[] )
{
#ifdef ARRAY_DEBUG
	MK_WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG
	CmdLineParse( argc-1 , &argv[1] , params );
	ThreadPool::ChunkSize = ThreadChunkSize.value;
	ThreadPool::Schedule = (ThreadPool::ScheduleType)ScheduleType.value;
	if( Verbose.set )
	{
		printf( "**************************************************\n" );
		printf( "**************************************************\n" );
		printf( "** Running Octree Visualization (Version %s) **\n" , ADAPTIVE_SOLVERS_VERSION );
		printf( "**************************************************\n" );
		printf( "**************************************************\n" );
	}

	if( !In.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	FILE* fp = fopen( In.value , "rb" );
	if( !fp ) MK_THROW( "Failed to open file for reading: " , In.value );
	FEMTreeRealType realType ; int degree ; BoundaryType bType;
	unsigned int dimension;
	FileStream fs(fp);
	ReadFEMTreeParameter( fs , realType , dimension );
	{
		unsigned int dim = dimension;
		unsigned int* sigs = ReadDenseNodeDataSignatures( fs , dim );
		if( dimension!=dim ) MK_THROW( "Octree and node data dimensions don't math: " , dimension , " != " , dim );
		for( unsigned int d=1 ; d<dim ; d++ ) if( sigs[0]!=sigs[d] ) MK_THROW( "Anisotropic signatures" );
		degree = FEMSignatureDegree( sigs[0] );
		bType = FEMSignatureBType( sigs[0] );
		delete[] sigs;
	}
	if( Verbose.set ) printf( "%d-dimension , %s-precision , degree-%d , %s-boundary\n" , dimension , FEMTreeRealNames[ realType ] , degree , BoundaryNames[ bType ] );

	switch( dimension )
	{
	case 2:
		switch( realType )
		{
			case FEM_TREE_REAL_FLOAT:  Execute< 2 , float  >( fs , degree , bType ) ; break;
			case FEM_TREE_REAL_DOUBLE: Execute< 2 , double >( fs , degree , bType ) ; break;
			default: MK_THROW( "Unrecognized real type: " , realType );
		}
		break;
	case 3:
		switch( realType )
		{
			case FEM_TREE_REAL_FLOAT:  Execute< 3 , float  >( fs , degree , bType ) ; break;
			case FEM_TREE_REAL_DOUBLE: Execute< 3 , double >( fs , degree , bType ) ; break;
			default: MK_THROW( "Unrecognized real type: " , realType );
		}
		break;
	default: MK_THROW( "Only dimensions 1-4 supported" );
	}

	fclose( fp );
	return EXIT_SUCCESS;
}
