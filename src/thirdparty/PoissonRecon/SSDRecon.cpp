/*
Copyright (c) 2014, Michael Kazhdan
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
#include "Reconstructors.h"

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
#include "RegularGrid.h"
#include "DataStream.imp.h"

#define DEFAULT_DIMENSION 3

using namespace PoissonRecon;

CmdLineParameter< char* >
	In( "in" ) ,
	Out( "out" ) ,
	TempDir( "tempDir" ) ,
	Grid( "grid" ) ,
	Tree( "tree" ) ,
	Transform( "xForm" );

CmdLineReadable
	Performance( "performance" ) ,
	ShowResidual( "showResidual" ) ,
	PolygonMesh( "polygonMesh" ) ,
	NonManifold( "nonManifold" ) ,
	ASCII( "ascii" ) ,
	Density( "density" ) ,
	NonLinearFit( "nonLinearFit" ) ,
	PrimalGrid( "primalGrid" ) ,
	ExactInterpolation( "exact" ) ,
	Colors( "colors" ) ,
	InCore( "inCore" ) ,
	Gradients( "gradients" ) ,
	GridCoordinates( "gridCoordinates" ) ,
	Confidence( "confidence" ) ,
	Verbose( "verbose" );

CmdLineParameter< int >
#ifndef FAST_COMPILE
	Degree( "degree" , Reconstructor::SSD::DefaultFEMDegree ) ,
#endif // !FAST_COMPILE
	Depth( "depth" , 8 ) ,
	KernelDepth( "kernelDepth" , -1 ) ,
	SolveDepth( "solveDepth" , -1 ) ,
	FullDepth( "fullDepth" , 5 ) ,
	BaseDepth( "baseDepth" , -1 ) ,
	Iters( "iters" , 8 ) ,
	BaseVCycles( "baseVCycles" , 4 ) ,
#ifndef FAST_COMPILE
	BType( "bType" , Reconstructor::SSD::DefaultFEMBoundary+1 ) ,
#endif // !FAST_COMPILE
	MaxMemoryGB( "maxMemory" , 0 ) ,
	ParallelType( "parallel" , 0 ) ,
	AlignmentDir( "alignDir" , DEFAULT_DIMENSION-1 ) ,
	ScheduleType( "schedule" , (int)ThreadPool::Schedule ) ,
	ThreadChunkSize( "chunkSize" , (int)ThreadPool::ChunkSize );

CmdLineParameter< float >
	DataX( "data" , 32.f ) ,
	SamplesPerNode( "samplesPerNode" , 1.5f ) ,
	Scale( "scale" , 1.1f ) ,
	Width( "width" , 0.f ) ,
	LowDepthCutOff( "lowDepthCutOff" , 0.f ) ,
	CGSolverAccuracy( "cgAccuracy" , 1e-3f ) ,
	ValueWeight   (    "valueWeight" , 1.f ) ,
	GradientWeight( "gradientWeight" , 1.f ) ,
	BiLapWeight   (    "biLapWeight" , 1.f );


CmdLineReadable* params[] =
{
#ifndef FAST_COMPILE
	&Degree , &BType ,
#endif // !FAST_COMPILE
	&SolveDepth ,
	&In , &Depth , &Out , &Transform ,
	&Width ,
	&Scale , &Verbose , &CGSolverAccuracy ,
	&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
	&ValueWeight , &GradientWeight , &BiLapWeight ,
	&Grid ,
	&Tree ,
	&Density ,
	&FullDepth ,
	&BaseDepth , &BaseVCycles ,
	&Iters ,
	&DataX ,
	&Colors ,
	&Gradients ,
	&NonLinearFit ,
	&PrimalGrid ,
	&TempDir ,
	&ExactInterpolation ,
	&Performance ,
	&MaxMemoryGB ,
	&InCore ,
	&ParallelType ,
	&ScheduleType ,
	&ThreadChunkSize ,
	&LowDepthCutOff ,
	&AlignmentDir ,
	&GridCoordinates ,
	NULL
};

void ShowUsage(char* ex)
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input points>\n" , In.name );
	printf( "\t[--%s <ouput triangle mesh>]\n" , Out.name );
	printf( "\t[--%s <ouput grid>]\n" , Grid.name );
	printf( "\t[--%s <ouput fem tree>]\n" , Tree.name );
#ifndef FAST_COMPILE
	printf( "\t[--%s <b-spline degree>=%d]\n" , Degree.name , Degree.value );
	printf( "\t[--%s <boundary type>=%d]\n" , BType.name , BType.value );
	for( int i=0 ; i<BOUNDARY_COUNT ; i++ ) printf( "\t\t%d] %s\n" , i+1 , BoundaryNames[i] );
#endif // !FAST_COMPILE
	printf( "\t[--%s <maximum reconstruction depth>=%d]\n" , Depth.name , Depth.value );
	printf( "\t[--%s <maximum solution depth>=%d]\n" , SolveDepth.name , SolveDepth.value );
	printf( "\t[--%s <grid width>]\n" , Width.name );
	printf( "\t[--%s <full depth>=%d]\n" , FullDepth.name , FullDepth.value );
	printf( "\t[--%s <coarse MG solver depth>]\n" , BaseDepth.name );
	printf( "\t[--%s <coarse MG solver v-cycles>=%d]\n" , BaseVCycles.name , BaseVCycles.value );
	printf( "\t[--%s <scale factor>=%f]\n" , Scale.name , Scale.value );
	printf( "\t[--%s <minimum number of samples per node>=%f]\n" , SamplesPerNode.name, SamplesPerNode.value );
	printf( "\t[--%s <zero-crossing weight>=%.3e]\n" , ValueWeight.name , ValueWeight.value );
	printf( "\t[--%s <gradient weight>=%.3e]\n" , GradientWeight.name , GradientWeight.value );
	printf( "\t[--%s <bi-laplacian weight>=%.3e]\n" , BiLapWeight.name , BiLapWeight.value );
	printf( "\t[--%s <iterations>=%d]\n" , Iters.name , Iters.value );
	printf( "\t[--%s]\n" , ExactInterpolation.name );
	printf( "\t[--%s <pull factor>=%f]\n" , DataX.name , DataX.value );
	printf( "\t[--%s]\n" , Colors.name );
	printf( "\t[--%s]\n" , Gradients.name );
	printf( "\t[--%s <parallel type>=%d]\n" , ParallelType.name , ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , ScheduleType.name , ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , ThreadChunkSize.name , ThreadChunkSize.value );
	printf( "\t[--%s <low depth cut-off>=%f]\n" , LowDepthCutOff.name , LowDepthCutOff.value );
	printf( "\t[--%s <slice direction>=%d]\n" , AlignmentDir.name , AlignmentDir.value );
	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s <cg solver accuracy>=%g]\n" , CGSolverAccuracy.name , CGSolverAccuracy.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , MaxMemoryGB.name , MaxMemoryGB.value );
	printf( "\t[--%s]\n" , Confidence.name );
	printf( "\t[--%s]\n" , GridCoordinates.name );
	printf( "\t[--%s]\n" , Performance.name );
	printf( "\t[--%s]\n" , Density.name );
	printf( "\t[--%s]\n" , NonLinearFit.name );
	printf( "\t[--%s]\n" , PrimalGrid.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , TempDir.name );
	printf( "\t[--%s]\n" , InCore.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool HasGradients , bool HasDensity , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , HasGradients , HasDensity , AuxDataFactories ... >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory( factories... );

	Reconstructor::OutputInputFactoryTypeStream< Real , Dim , Factory , InCore , true , typename AuxDataFactories::VertexType... > vertexStream( factory , VInfo::Convert );
	Reconstructor::OutputInputFaceStream< Dim-1 , InCore , true > faceStream;

	implicit.extractLevelSet( vertexStream , faceStream , meParams );

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	vertexStream.reset();
	PLY::Write< Factory , node_index_type , Real , Dim >( fileName , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ascii ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool HasDensity , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( hasGradients ) return WriteMesh< Real , Dim, FEMSig , true  , HasDensity , InCore , AuxDataFactories ... >( implicit , meParams , fileName , ascii , factories... );
	else               return WriteMesh< Real , Dim, FEMSig , false , HasDensity , InCore , AuxDataFactories ... >( implicit , meParams , fileName , ascii , factories... );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients , bool hasDensity ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( hasDensity ) return WriteMesh< Real , Dim, FEMSig , true  , InCore , AuxDataFactories ... >( hasGradients , implicit , meParams , fileName , ascii , factories... );
	else             return WriteMesh< Real , Dim, FEMSig , false , InCore , AuxDataFactories ... >( hasGradients , implicit , meParams , fileName , ascii , factories... );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients , bool hasDensity , bool inCore ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( inCore ) return WriteMesh< Real , Dim, FEMSig , true  , AuxDataFactories ... >( hasGradients , hasDensity , implicit , meParams , fileName , ascii , factories... );
	else         return WriteMesh< Real , Dim, FEMSig , false , AuxDataFactories ... >( hasGradients , hasDensity , implicit , meParams , fileName , ascii , factories... );
}

template< class Real , unsigned int Dim , unsigned int FEMSig , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	static const bool HasAuxData = !std::is_same< AuxDataFactory , VertexFactory::EmptyFactory< Real > >::value;

	///////////////
	// Types --> //
	typedef IsotropicUIntPack< Dim , FEMSig > Sigs;
	using namespace VertexFactory;

	// The factory for constructing an input sample's data
	typedef std::conditional_t< HasAuxData , Factory< Real , NormalFactory< Real , Dim > , AuxDataFactory > , NormalFactory< Real , Dim > > InputSampleDataFactory;

	// The factory for constructing an input sample
	typedef Factory< Real , PositionFactory< Real , Dim > , InputSampleDataFactory >  InputSampleFactory;

	typedef InputDataStream< typename InputSampleFactory::VertexType > InputPointStream;

	// The type storing the reconstruction solution (depending on whether auxiliary data is provided or not)
	using Implicit = std::conditional_t< HasAuxData , Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactory::VertexType > , Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > > >;
	using Solver = std::conditional_t< HasAuxData , Reconstructor::SSD::Solver< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactory::VertexType > , Reconstructor::SSD::Solver< Real , Dim , IsotropicUIntPack< Dim , FEMSig > > >;
	// <-- Types //
	///////////////

	if( Verbose.set )
	{
		std::cout << "************************************************" << std::endl;
		std::cout << "************************************************" << std::endl;
		std::cout << "** Running SSD Reconstruction (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "************************************************" << std::endl;
		std::cout << "************************************************" << std::endl;

		char str[1024];
		for( int i=0 ; params[i] ; i++ ) if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( strlen( str ) ) std::cout << "\t--" << params[i]->name << " " << str << std::endl;
			else                std::cout << "\t--" << params[i]->name << std::endl;
		}
	}

	Profiler profiler(20);
	Implicit *implicit = NULL;
	typename Reconstructor::SSD::SolutionParameters< Real > sParams;
	Reconstructor::LevelSetExtractionParameters meParams;

	sParams.verbose = Verbose.set;
	sParams.exactInterpolation = ExactInterpolation.set;
	sParams.showResidual = ShowResidual.set;
	sParams.confidence = Confidence.set;
	sParams.scale = (Real)Scale.value;
	sParams.lowDepthCutOff = (Real)LowDepthCutOff.value;
	sParams.width = (Real)Width.value;
	sParams.pointWeight = (Real)ValueWeight.value;
	sParams.gradientWeight = (Real)GradientWeight.value;
	sParams.biLapWeight = (Real)BiLapWeight.value;
	sParams.samplesPerNode = (Real)SamplesPerNode.value;
	sParams.cgSolverAccuracy = (Real)CGSolverAccuracy.value;
	sParams.perLevelDataScaleFactor = (Real)DataX.value;
	sParams.depth = (unsigned int)Depth.value;
	sParams.baseDepth = (unsigned int)BaseDepth.value;
	sParams.solveDepth = (unsigned int)SolveDepth.value;
	sParams.fullDepth = (unsigned int)FullDepth.value;
	sParams.kernelDepth = (unsigned int)KernelDepth.value;
	sParams.baseVCycles = (unsigned int)BaseVCycles.value;
	sParams.iters = (unsigned int)Iters.value;
	sParams.alignDir = (unsigned int)AlignmentDir.value;

	meParams.linearFit = !NonLinearFit.set;
	meParams.outputGradients = Gradients.set;
	meParams.forceManifold = !NonManifold.set;
	meParams.polygonMesh = PolygonMesh.set;
	meParams.gridCoordinates = GridCoordinates.set;
	meParams.outputDensity = Density.set;
	meParams.verbose = Verbose.set;

	double startTime = Time();

	InputSampleFactory *_inputSampleFactory;
	if constexpr( HasAuxData ) _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , InputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory ) );
	else _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , VertexFactory::NormalFactory< Real , Dim >() );
	InputSampleFactory &inputSampleFactory = *_inputSampleFactory;
	XForm< Real , Dim+1 > toModel = XForm< Real , Dim+1 >::Identity();

	// Read in the transform, if we want to apply one to the points before processing
	if( Transform.set )
	{
		FILE* fp = fopen( Transform.value , "r" );
		if( !fp ) MK_WARN( "Could not read x-form from: " , Transform.value );
		else
		{
			for( int i=0 ; i<Dim+1 ; i++ ) for( int j=0 ; j<Dim+1 ; j++ )
			{
				float f;
				if( fscanf( fp , " %f " , &f )!=1 ) MK_THROW( "Failed to read xform" );
				toModel(i,j) = (Real)f;
			}
			fclose( fp );
		}
	}
	std::vector< typename InputSampleFactory::VertexType > inCorePoints;
	InputPointStream *pointStream;

	// Get the point stream
	{
		profiler.reset();
		char *ext = GetFileExtension( In.value );

		if( InCore.set )
		{
			InputPointStream *_pointStream;
			if     ( !strcasecmp( ext , "bnpts" ) ) _pointStream = new BinaryInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) _pointStream = new    PLYInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else                                    _pointStream = new  ASCIIInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			typename InputSampleFactory::VertexType p = inputSampleFactory();
			while( _pointStream->read( p ) ) inCorePoints.push_back( p );
			delete _pointStream;

			pointStream = new VectorBackedInputDataStream< typename InputSampleFactory::VertexType >( inCorePoints );
		}
		else
		{
			if     ( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) pointStream = new    PLYInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
			else                                    pointStream = new  ASCIIInputDataStream< InputSampleFactory >( In.value , inputSampleFactory );
		}
		delete[] ext;
	}

	// A wrapper class to realize InputPointStream as an InputSampleWithDataStream
	struct _InputOrientedSampleStream : public Reconstructor::InputOrientedSampleStream< Real , Dim >
	{
		typedef Reconstructor::Normal< Real , Dim > DataType;
		typedef DirectSum< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;
		_InputOrientedSampleStream( _InputPointStream &pointStream ) : pointStream( pointStream )
		{
			scratch = SampleType( Reconstructor::Position< Real , Dim >() , Reconstructor::Normal< Real , Dim >() );
		}
		void reset( void ){ pointStream.reset(); }
		bool read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n )
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>();
			return ret;
		}
		bool read( unsigned int thread , Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n )
		{
			bool ret = pointStream.read( thread , scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>();
			return ret;
		}
	};

	// A wrapper class to realize InputPointStream as an InputSampleWithDataStream
	struct _InputOrientedSampleWithDataStream : public Reconstructor::InputOrientedSampleStream< Real , Dim , typename AuxDataFactory::VertexType >
	{
		typedef DirectSum< Real , Reconstructor::Normal< Real , Dim > , typename AuxDataFactory::VertexType > DataType;
		typedef DirectSum< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;
		_InputOrientedSampleWithDataStream( _InputPointStream &pointStream , typename AuxDataFactory::VertexType zero ) : pointStream( pointStream )
		{
			scratch = SampleType( Reconstructor::Position< Real , Dim >() , DataType( Reconstructor::Normal< Real , Dim >() , zero ) );
		}
		void reset( void ){ pointStream.reset(); }
		bool read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n , typename AuxDataFactory::VertexType &d )
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>().template get<0>() , d = scratch.template get<1>().template get<1>();
			return ret;
		}
		bool read( unsigned int thread , Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n , typename AuxDataFactory::VertexType &d )
		{
			bool ret = pointStream.read( thread , scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>().template get<0>() , d = scratch.template get<1>().template get<1>();
			return ret;
		}
	};

	if constexpr( HasAuxData )
	{
		_InputOrientedSampleWithDataStream sampleStream( *pointStream , auxDataFactory() );

		if( Transform.set )
		{
			Reconstructor::TransformedInputOrientedSampleStream< Real , Dim , typename AuxDataFactory::VertexType > _sampleStream( toModel , sampleStream );
			implicit = Solver::Solve( _sampleStream , sParams , auxDataFactory() );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit = Solver::Solve( sampleStream , sParams , auxDataFactory() );
	}
	else
	{
		_InputOrientedSampleStream sampleStream( *pointStream );

		if( Transform.set )
		{
			Reconstructor::TransformedInputOrientedSampleStream< Real , Dim > _sampleStream( toModel , sampleStream );
			implicit = Solver::Solve( _sampleStream , sParams );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit = Solver::Solve( sampleStream , sParams );
	}

	delete pointStream;
	delete _inputSampleFactory;

	if( Tree.set )
	{
		FILE* fp = fopen( Tree.value , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , Tree.value );
		FileStream fs(fp);
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		implicit->tree.write( fs , false );
		fs.write( implicit->unitCubeToModel.inverse() );
		implicit->solution.write( fs );
		fclose( fp );
	}


	if( Grid.set )
	{
		int res = 0;
		profiler.reset();
		Pointer( Real ) values = implicit->tree.template regularGridEvaluate< true >( implicit->solution , res , -1 , PrimalGrid.set );
		if( Verbose.set ) std::cout << "Got grid: " << profiler << std::endl;
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );

		unsigned int _res[Dim];
		for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
		RegularGrid< Real , Dim >::Write( Grid.value , _res , values , implicit->unitCubeToModel * voxelToUnitCube );

		DeletePointer( values );
	}


	if( Out.set && ( Dim==2 || Dim==3 ) )
	{
		// Create the output mesh
		char tempHeader[2048];
		{
			char tempPath[1024];
			tempPath[0] = 0;
			if( TempDir.set ) strcpy( tempPath , TempDir.value );
			else SetTempDirectory( tempPath , sizeof(tempPath) );
			if( strlen(tempPath)==0 ) sprintf( tempPath , ".%c" , FileSeparator );
			if( tempPath[ strlen( tempPath )-1 ]==FileSeparator ) sprintf( tempHeader , "%sPR_" , tempPath );
			else                                                  sprintf( tempHeader , "%s%cPR_" , tempPath , FileSeparator );
		}

		XForm< Real , Dim+1 > pXForm = implicit->unitCubeToModel;
		XForm< Real , Dim > nXForm = XForm< Real , Dim >( pXForm ).inverse().transpose();

		if constexpr( HasAuxData ) WriteMesh< Real , Dim , FEMSig >( Gradients.set , Density.set , InCore.set , *implicit , meParams , Out.value , ASCII.set , auxDataFactory );
		else                       WriteMesh< Real , Dim , FEMSig >( Gradients.set , Density.set , InCore.set , *implicit , meParams , Out.value , ASCII.set );
	}
	else MK_WARN( "Mesh extraction is only supported in dimensions 2 and 3" );

	if( Verbose.set ) std::cout << "#          Total Solve: " << Time()-startTime << " (s), " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
	delete implicit;
}

#ifndef FAST_COMPILE
template< unsigned int Dim , class Real , BoundaryType BType , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	switch( Degree.value )
	{
		case 2: return Execute< Real , Dim , FEMDegreeAndBType< 2 , BType >::Signature >( auxDataFactory );
		case 3: return Execute< Real , Dim , FEMDegreeAndBType< 3 , BType >::Signature >( auxDataFactory );
//		case 4: return Execute< Real , Dim , FEMDegreeAndBType< 4 , BType >::Signature >( auxDataFactory );
		default: MK_THROW( "Only B-Splines of degree 1 - 2 are supported" );
	}
}

template< unsigned int Dim , class Real , typename AuxDataFactory >
void Execute( const AuxDataFactory &auxDataFactory )
{
	switch( BType.value )
	{
		case BOUNDARY_FREE+1:      return Execute< Dim , Real , BOUNDARY_FREE      >( auxDataFactory );
		case BOUNDARY_NEUMANN+1:   return Execute< Dim , Real , BOUNDARY_NEUMANN   >( auxDataFactory );
		case BOUNDARY_DIRICHLET+1: return Execute< Dim , Real , BOUNDARY_DIRICHLET >( auxDataFactory );
		default: MK_THROW( "Not a valid boundary type: " , BType.value );
	}
}
#endif // !FAST_COMPILE

int main( int argc , char* argv[] )
{
	Timer timer;
#ifdef ARRAY_DEBUG
	MK_WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG

	CmdLineParse( argc-1 , &argv[1] , params );
	if( MaxMemoryGB.value>0 ) SetPeakMemoryMB( MaxMemoryGB.value<<10 );
	ThreadPool::ChunkSize = ThreadChunkSize.value;
	ThreadPool::Schedule = (ThreadPool::ScheduleType)ScheduleType.value;
	ThreadPool::ParallelizationType= (ThreadPool::ParallelType)ParallelType.value;

	if( !In.set )
	{
		ShowUsage( argv[0] );
		return 0;
	}
	if( GradientWeight.value<=0 ) MK_THROW( "Gradient weight must be positive: " , GradientWeight.value , "> 0" );
	if( BiLapWeight.value<=0 ) MK_THROW( "Bi-Laplacian weight must be positive: " , BiLapWeight.value , " > 0" );

	ValueWeight.value    *= (float)Reconstructor::SSD::WeightMultipliers[0];
	GradientWeight.value *= (float)Reconstructor::SSD::WeightMultipliers[1];
	BiLapWeight.value    *= (float)Reconstructor::SSD::WeightMultipliers[2];

#ifdef USE_DOUBLE
	typedef double Real;
#else // !USE_DOUBLE
	typedef float  Real;
#endif // USE_DOUBLE

#ifdef FAST_COMPILE
	static const int Degree = Reconstructor::SSD::DefaultFEMDegree;
	static const BoundaryType BType = Reconstructor::SSD::DefaultFEMBoundary;
	static const unsigned int Dim = DEFAULT_DIMENSION;
	static const unsigned int FEMSig = FEMDegreeAndBType< Degree , BType >::Signature;
	MK_WARN( "Compiled for degree-" , Degree , ", boundary-" , BoundaryNames[ BType ] , ", " , sizeof(Real)==4 ? "single" : "double" , "-precision _only_" );

	char *ext = GetFileExtension( In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties );
		if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
		if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
		delete[] readFlags;

		if( unprocessedProperties.size() ) Execute< Real , Dim , FEMSig >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< Real , Dim , FEMSig >( VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( Colors.set ) Execute< Real , Dim , FEMSig >( VertexFactory::RGBColorFactory< Real >() );
		else             Execute< Real , Dim , FEMSig >( VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#else // !FAST_COMPILE
	char *ext = GetFileExtension( In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties );
		if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
		if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
		delete[] readFlags;

		if( unprocessedProperties.size() ) Execute< DEFAULT_DIMENSION , Real >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< DEFAULT_DIMENSION , Real >( VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( Colors.set ) Execute< DEFAULT_DIMENSION , Real >( VertexFactory::RGBColorFactory< Real >() );
		else             Execute< DEFAULT_DIMENSION , Real >( VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#endif // FAST_COMPILE

	if( Performance.set )
	{
		printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
		printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
	}
	return EXIT_SUCCESS;
}
