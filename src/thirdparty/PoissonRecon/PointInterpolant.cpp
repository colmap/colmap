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

//#undef USE_DOUBLE								// If enabled, double-precesion is used
#define USE_DOUBLE								// If enabled, double-precesion is used
#define WEIGHT_DEGREE 2							// The order of the B-Spline used to splat in the weights for density estimation
#define DEFAULT_FEM_DEGREE 2					// The default finite-element degree
#define DEFAULT_FEM_BOUNDARY BOUNDARY_FREE		// The default finite-element boundary type
#define DEFAULT_DIMENSION 2						// The dimension of the system

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
	InValues( "inValues" ) ,
	InGradients( "inGradients" ) ,
	Out( "out" ) ,
	TempDir( "tempDir" ) ,
	Grid( "grid" ) ,	
	Tree( "tree" ) ,
	Transform( "xForm" );

CmdLineReadable
	Performance( "performance" ) ,
	ShowResidual( "showResidual" ) ,
	PrimalGrid( "primalGrid" ) ,
	ExactInterpolation( "exact" ) ,
	InCore( "inCore" ) ,
	PolygonMesh( "polygonMesh" ) ,
	NonManifold( "nonManifold" ) ,
	NonLinearFit( "nonLinearFit" ) ,
	ASCII( "ascii" ) ,
	Verbose( "verbose" );

CmdLineParameter< int >
#ifndef FAST_COMPILE
	Degree( "degree" , DEFAULT_FEM_DEGREE ) ,
#endif // !FAST_COMPILE
	Depth( "depth" , 8 ) ,
	SolveDepth( "solveDepth" , -1 ) ,
	Iters( "iters" , 8 ) ,
	FullDepth( "fullDepth" , 5 ) ,
	BaseDepth( "baseDepth" ) ,
	BaseVCycles( "baseVCycles" , 4 ) ,
#ifndef FAST_COMPILE
	BType( "bType" , DEFAULT_FEM_BOUNDARY+1 ) ,
	Dimension( "dim" , DEFAULT_DIMENSION ) ,
#endif // !FAST_COMPILE
	MaxMemoryGB( "maxMemory" , 0 ) ,
	ParallelType( "parallel" , 0 ) ,
	ScheduleType( "schedule" , (int)ThreadPool::Schedule ) ,
	ThreadChunkSize( "chunkSize" , (int)ThreadPool::ChunkSize );

CmdLineParameter< float >
	Scale( "scale" , 1.1f ) ,
	Width( "width" , 0.f ) ,
	CGSolverAccuracy( "cgAccuracy" , 1e-3f ) ,
	IsoValue( "iso" , 0.f ) ,
	ValueWeight   (    "valueWeight" , 1000.f ) ,
	GradientWeight( "gradientWeight" , 1.f ) ,
	LapWeight     (      "lapWeight" , 0.f ) ,
	BiLapWeight   (    "biLapWeight" , 1.f );

CmdLineReadable* params[] =
{
#ifndef FAST_COMPILE
	&Degree , &BType , &Dimension ,
#endif // !FAST_COMPILE
	&SolveDepth ,
	&InValues , &InGradients ,
	&Out , &Depth , &Transform ,
	&Width ,
	&Scale , &Verbose , &CGSolverAccuracy ,
	&NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
	&ValueWeight , &GradientWeight ,
	&LapWeight , &BiLapWeight ,
	&Grid ,
	&Tree ,
	&FullDepth ,
	&BaseDepth , &BaseVCycles ,
	&Iters ,
	&IsoValue ,
	&PrimalGrid ,
	&ExactInterpolation ,
	&Performance ,
	&MaxMemoryGB ,
	&InCore ,
	&ParallelType ,
	&ScheduleType ,
	&ThreadChunkSize ,
	&NonLinearFit ,
	NULL
};

void ShowUsage(char* ex)
{
	printf( "Usage: %s\n" , ex );
	printf( "\t[--%s <input point values>]\n" , InValues.name );
	printf( "\t[--%s <input point gradients>]\n" , InGradients.name );
	printf( "\t[--%s <ouput mesh>]\n" , Out.name );
	printf( "\t[--%s <ouput grid>]\n" , Grid.name );
	printf( "\t[--%s <ouput fem tree>]\n" , Tree.name );
#ifndef FAST_COMPILE
	printf( "\t[--%s <dimension>=%d]\n" , Dimension.name , Dimension.value );
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
	printf( "\t[--%s <zero-crossing weight>=%.3e]\n" , ValueWeight.name , ValueWeight.value );
	printf( "\t[--%s <gradient weight>=%.3e]\n" , GradientWeight.name , GradientWeight.value );
	printf( "\t[--%s <laplacian weight>=%.3e]\n" , LapWeight.name , LapWeight.value );
	printf( "\t[--%s <bi-laplacian weight>=%.3e]\n" , BiLapWeight.name , BiLapWeight.value );
	printf( "\t[--%s <iterations>=%d]\n" , Iters.name , Iters.value );
	printf( "\t[--%s]\n" , ExactInterpolation.name );
	printf( "\t[--%s <parallel type>=%d]\n" , ParallelType.name , ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , ScheduleType.name , ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , ThreadChunkSize.name , ThreadChunkSize.value );
	printf( "\t[--%s <cg solver accuracy>=%g]\n" , CGSolverAccuracy.name , CGSolverAccuracy.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , MaxMemoryGB.name , MaxMemoryGB.value );
	printf( "\t[--%s <iso-value>=%f]\n" , IsoValue.name , IsoValue.value );
	printf( "\t[--%s]\n" , Performance.name );
	printf( "\t[--%s]\n" , PrimalGrid.name );
	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t[--%s]\n" , NonLinearFit.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , InCore.name );
	printf( "\t[--%s]\n" , Verbose.name );
}


template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor )
{
	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = max[0] - min[0];
	for( int d=1 ; d<Dim ; d++ ) scale = std::max< Real >( scale , max[d]-min[d] );
	scale *= scaleFactor;
	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real width , Real scaleFactor , int& depth )
{
	// Get the target resolution (along the largest dimension)
	Real resolution = ( max[0]-min[0] ) / width;
	for( int d=1 ; d<Dim ; d++ ) resolution = std::max< Real >( resolution , ( max[d]-min[d] ) / width );
	resolution *= scaleFactor;
	depth = 0;
	while( (1<<depth)<resolution ) depth++;

	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = (1<<depth) * width;

	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}

template< typename Real , unsigned int Dim , typename FunctionValueType >
using InputPointStreamInfo = typename FEMTreeInitializer< Dim , Real >::template InputPointStream< FunctionValueType >;

template< typename Real , unsigned int Dim , typename FunctionValueType >
using InputPointStream = typename InputPointStreamInfo< Real , Dim , FunctionValueType >::StreamType;

template< class Real , unsigned int Dim , typename FunctionValueType >
XForm< Real , Dim+1 > GetPointXForm( InputPointStream< Real , Dim , FunctionValueType > &stream , Real width , Real scaleFactor , int& depth )
{
	Point< Real , Dim > min , max;
	InputPointStreamInfo< Real , Dim , FunctionValueType >::BoundingBox( stream , min , max );
	return GetBoundingBoxXForm( min , max , width , scaleFactor , depth );
}
template< class Real , unsigned int Dim , typename FunctionValueType >
XForm< Real , Dim+1 > GetPointXForm( InputPointStream< Real , Dim , FunctionValueType > &stream , Real scaleFactor )
{
	Point< Real , Dim > min , max;
	InputPointStreamInfo< Real , Dim , FunctionValueType >::BoundingBox( stream , min , max );
	return GetBoundingBoxXForm( min , max , scaleFactor );
}

template< unsigned int Dim , typename Real , typename PointSampleData > struct ConstraintDual;

template< unsigned int Dim , typename Real >
struct ConstraintDual< Dim , Real , DirectSum< Real , Real > >
{
	typedef DirectSum< Real , Real > PointSampleData;
	Real vWeight;
	ConstraintDual( Real v) : vWeight(v){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim > &p , const PointSampleData& data ) const 
	{
		Real value = data.template get<0>();
		CumulativeDerivativeValues< Real , Dim , 0 > cdv;
		cdv[0] = value*vWeight;
		return cdv;
	}
};

template< unsigned int Dim , typename Real >
struct ConstraintDual< Dim , Real , DirectSum< Real , Point< Real , Dim > > >
{
	typedef DirectSum< Real , Point< Real , Dim > > PointSampleData;
	Real gWeight;
	ConstraintDual( Real g ) : gWeight(g) { }
	CumulativeDerivativeValues< Real , Dim , 1 > operator()( const Point< Real , Dim >& p , const PointSampleData& data ) const 
	{
		Point< Real , Dim > gradient = data.template get<0>();
		CumulativeDerivativeValues< Real , Dim , 1 > cdv;
		for( int d=0 ; d<Dim ; d++ ) cdv[1+d] = gradient[d]*gWeight;
		return cdv;
	}
};

template< unsigned int Dim , typename Real , typename TotalPointSampleData > struct SystemDual;

template< unsigned int Dim , typename Real >
struct SystemDual< Dim , Real , DirectSum< Real , Real > >
{
	CumulativeDerivativeValues< Real , Dim , 0 > weight;
	SystemDual( Real v ){ weight[0] = v; }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< Real , Real > &data , const CumulativeDerivativeValues< Real , Dim , 0 > &dValues ) const
	{
		return dValues * weight;
	}
	CumulativeDerivativeValues< double , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< Real , Real > &data , const CumulativeDerivativeValues< double , Dim , 0 > &dValues ) const
	{
		return dValues * weight;
	};
};
template< unsigned int Dim >
struct SystemDual< Dim , double , DirectSum< double , double > >
{
	typedef double Real;
	CumulativeDerivativeValues< Real , Dim , 0 > weight;
	SystemDual( Real v ){ weight[0] = v; }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< double , double > &data , const CumulativeDerivativeValues< Real , Dim , 0 > &dValues ) const
	{
		return dValues * weight;
	}
};

template< unsigned int Dim , typename Real >
struct SystemDual< Dim , Real , DirectSum< Real , Point< Real , Dim > > >
{
	CumulativeDerivativeValues< Real , Dim , 1 > weight;
	SystemDual( Real g )
	{
		weight[0] = 0;
		for( int d=0 ; d<Dim ; d++ ) weight[d+1] = g;
	}
	CumulativeDerivativeValues< Real , Dim , 1 > operator()( Point< Real , Dim > p , const DirectSum< Real , Point< Real , Dim > > &data , const CumulativeDerivativeValues< Real , Dim , 1 > &dValues ) const
	{
		return dValues * weight;
	}
	CumulativeDerivativeValues< double , Dim , 1 > operator()( Point< Real , Dim > p , const DirectSum< Real , Point< Real , Dim > > &data , const CumulativeDerivativeValues< double , Dim , 1 > &dValues ) const
	{
		return dValues * weight;
	};
};
template< unsigned int Dim >
struct SystemDual< Dim , double , DirectSum< double , Point< double , Dim > > >
{
	typedef double Real;
	CumulativeDerivativeValues< Real , Dim , 1 > weight;
	SystemDual( Real g )
	{
		weight[0] = 0;
		for( int d=0 ; d<Dim ; d++) weight[1+d] = g;
	}
	CumulativeDerivativeValues< Real , Dim , 1 > operator()( Point< Real , Dim > p , const DirectSum< Real , Point< Real , Dim > > &data , const CumulativeDerivativeValues< Real , Dim , 1 > &dValues ) const
	{
		return dValues * weight;
	}
};

template< typename Real , unsigned int ... FEMSigs >
void ExtractLevelSet
(
	UIntPack< FEMSigs ... > ,
	FEMTree< sizeof ... ( FEMSigs ) , Real >& tree ,
	const DenseNodeData< Real , UIntPack< FEMSigs ... > >& solution ,
	Real isoValue ,
	XForm< Real , sizeof...(FEMSigs)+1 > unitCubeToModel ,
	std::vector< std::string > &comments
)
{
	static const int Dim = sizeof ... ( FEMSigs );
	typedef UIntPack< FEMSigs ... > Sigs;
	static const unsigned int DataSig = FEMDegreeAndBType< WEIGHT_DEGREE , BOUNDARY_FREE >::Signature;

	Profiler profiler(20);

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

	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , false , false >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory();

	// A backing stream for the vertices
	Reconstructor::OutputInputFactoryTypeStream< Real , Dim , Factory , false , true > vertexStream( factory , VInfo::Convert );
	Reconstructor::OutputInputFaceStream< Dim-1 , false , true > faceStream;
	typename LevelSetExtractor< Real , Dim >::Stats stats;

	Reconstructor::TransformedOutputLevelSetVertexStream< Real , Dim > _vertexStream( unitCubeToModel , vertexStream );

	// Extract the mesh
	stats = LevelSetExtractor< Real , Dim >::Extract( Sigs() , UIntPack< 0 >() , tree , ( typename FEMTree< Dim , Real >::template DensityEstimator< 0 >* )NULL , solution , isoValue , _vertexStream , faceStream , NonLinearFit.set , false , !NonManifold.set , PolygonMesh.set , false );

	if( Verbose.set )
	{
		std::cout << "Vertices / Faces: " << vertexStream.size() << " / " << faceStream.size() << std::endl;
		std::cout << stats.toString() << std::endl;
		std::cout << "#            Got faces: " << profiler << std::endl;
	}

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	PLY::Write< Factory , node_index_type , Real , Dim >( Out.value , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ASCII.set ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
}

template< typename Real , unsigned int Dim >
void WriteGrid( const char *fileName , ConstPointer( Real ) values , unsigned int res , XForm< Real , Dim+1 > voxelToModel , bool verbose )
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

template< class Real , unsigned int ... FEMSigs >
void Execute( UIntPack< FEMSigs ... > )
{
	static const int Dim = sizeof ... ( FEMSigs );
	typedef UIntPack< FEMSigs ... > Sigs;
	typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;
	typedef UIntPack< FEMDegreeAndBType< WEIGHT_DEGREE , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > DataSigs;
	typedef typename FEMTree< Dim , Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;

	typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 0 > ValueInterpolationInfo;
	typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 1 > GradientInterpolationInfo;

	// The factory for constructing the function's values
	typedef VertexFactory::Factory< Real , VertexFactory::ValueFactory< Real > > FunctionValueFactory;
	// The factory for constructing the function's gradients
	typedef VertexFactory::Factory< Real , VertexFactory::NormalFactory< Real , Dim > > FunctionGradientFactory;

	// The factory for constructing the value data
	typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , FunctionValueFactory > InputSampleValueFactory;
	// The factory for constructing the gradient data
	typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , FunctionGradientFactory > InputSampleGradientFactory;

	// The type of the function values
	typedef typename FunctionValueFactory::VertexType FunctionValueType;
	// The type of the function gradients
	typedef typename FunctionGradientFactory::VertexType FunctionGradientType;

	// The type of the input value
	typedef typename InputSampleValueFactory::VertexType InputSampleValueType;
	// The type of the input gradient
	typedef typename InputSampleGradientFactory::VertexType InputSampleGradientType;

	typedef InputDataStream< InputSampleValueType >  InputPointValueStream;
	typedef InputDataStream< InputSampleGradientType >  InputPointGradientStream;

	struct XInputPointValueStream : public InputDataStream< InputSampleValueType >
	{
		InputDataStream< InputSampleValueType > &stream;
		XForm< Real , Dim+1 > pointTransform;
		XInputPointValueStream( InputDataStream< InputSampleValueType > &stream , XForm< Real , Dim+1 > modelToUnitCube ) : stream(stream) , pointTransform( modelToUnitCube ){}
		bool read( InputSampleValueType &s )
		{
			if( stream.read( s ) ){ s.template get<0>() = pointTransform * s.template get<0>() ; return true; }
			else return false;
		}
		void reset( void ){ return stream.reset(); }
	};
	struct XInputPointGradientStream : public InputDataStream< InputSampleGradientType >
	{
		//    G(p) = F( A*p )
		// F(p+d) = F(p) + < \nabla F(p) , d >
		// G(p+d) = F( A*p ) + < \nabla F(A*p) , A*d >
		//        = G(p) + < A^t * \nabla F(A*p) , d >
		// => \nabla G(p) = A^t * \nabla F(A*p)

		InputDataStream< InputSampleGradientType > &stream;
		XForm< Real , Dim+1 > pointTransform;
		XForm< Real , Dim > gradientTransform;
		XInputPointGradientStream( InputDataStream< InputSampleGradientType > &stream , XForm< Real , Dim+1 > modelToUnitCube ) : stream(stream)
		{
			pointTransform = modelToUnitCube;
			gradientTransform = XForm< Real , Dim >( pointTransform ).inverse().transpose();
		}
		bool read( InputSampleGradientType &s )
		{
			if( stream.read( s ) )
			{
				s.template get<0>() = pointTransform * s.template get<0>();
				s.template get<1>() = gradientTransform * s.template get<1>().template get<0>();
				return true;
			}
			else return false;
		}
		void reset( void ){ return stream.reset(); }
	};
	FunctionValueFactory functionValueFactory;
	FunctionGradientFactory functionGradientFactory;

	InputSampleValueFactory inputSampleValueFactory( VertexFactory::PositionFactory< Real , Dim >() , functionValueFactory );
	InputSampleGradientFactory inputSampleGradientFactory( VertexFactory::PositionFactory< Real , Dim >() , functionGradientFactory );

	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
	std::vector< std::string > comments;
	if( Verbose.set )
	{
		std::cout << "***********************************************" << std::endl;
		std::cout << "***********************************************" << std::endl;
		std::cout << "** Running Point Interpolant (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "***********************************************" << std::endl;
		std::cout << "***********************************************" << std::endl;
	}

	ThreadPool::ParallelizationType= (ThreadPool::ParallelType)ParallelType.value;

	XForm< Real , Dim+1 > modelToUnitCube , unitCubeToModel;
	if( Transform.set )
	{
		FILE* fp = fopen( Transform.value , "r" );
		if( !fp )
		{
			MK_WARN( "Could not read x-form from: " , Transform.value );
			modelToUnitCube = XForm< Real , Dim+1 >::Identity();
		}
		else
		{
			for( int i=0 ; i<Dim+1 ; i++ ) for( int j=0 ; j<Dim+1 ; j++ )
			{
				float f;
				if( fscanf( fp , " %f " , &f )!=1 ) MK_THROW( "Failed to read xform" );
				modelToUnitCube(i,j) = (Real)f;
			}
			fclose( fp );
		}
	}
	else modelToUnitCube = XForm< Real , Dim+1 >::Identity();

	char str[1024];
	for( int i=0 ; params[i] ; i++ )
		if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( Verbose.set )
			{
				if( strlen( str ) ) std::cout << "\t--" << params[i]->name << " " << str << std::endl;
				else                std::cout << "\t--" << params[i]->name << std::endl;
			}
		}

	double startTime = Time();

	FEMTree< Dim , Real > tree( MEMORY_ALLOCATOR_BLOCK_SIZE );
	Profiler profiler(20);

	if( Depth.set && Width.value>0 )
	{
		MK_WARN( "Both --" , Depth.name , " and --" , Width.name , " set, ignoring --" , Width.name );
		Width.value = 0;
	}

	size_t pointValueCount , pointGradientCount;

	std::vector< typename FEMTree< Dim , Real >::PointSample > *valueSamples = NULL;
	std::vector< FunctionValueType > *valueSampleData = NULL;
	std::vector< typename FEMTree< Dim , Real >::PointSample > *gradientSamples = NULL;
	std::vector< FunctionGradientType > *gradientSampleData = NULL;

	// Read in the samples
	{
		profiler.reset();
		InputPointValueStream *pointValueStream = NULL;
		InputPointGradientStream *pointGradientStream = NULL;
		Point< Real , Dim > valueMin , valueMax , gradientMin , gradientMax;
		std::vector< InputSampleValueType > inCorePointsAndValues;
		std::vector< InputSampleGradientType > inCorePointsAndGradients;

		if( ValueWeight.value>0 )
		{
			char* ext = GetFileExtension( InValues.value );
			valueSampleData = new std::vector< FunctionValueType >();
			if( InCore.set )
			{
				InputPointValueStream *_pointValueStream;
				if     ( !strcasecmp( ext , "bnpts" ) ) _pointValueStream = new BinaryInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
				else if( !strcasecmp( ext , "ply"   ) ) _pointValueStream = new    PLYInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
				else                                    _pointValueStream = new  ASCIIInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
				InputSampleValueType s = inputSampleValueFactory();
				while( _pointValueStream->read( s ) ) inCorePointsAndValues.push_back( s );
				delete _pointValueStream;

				pointValueStream = new VectorBackedInputDataStream< InputSampleValueType >( inCorePointsAndValues );
			}
			else
			{
				if     ( !strcasecmp( ext , "bnpts" ) ) pointValueStream = new BinaryInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
				else if( !strcasecmp( ext , "ply"   ) ) pointValueStream = new    PLYInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
				else                                    pointValueStream = new  ASCIIInputDataStream< InputSampleValueFactory>( InValues.value , inputSampleValueFactory );
			}
			delete[] ext;

			XInputPointValueStream _pointStream( *pointValueStream , modelToUnitCube );
			FunctionValueType s = functionValueFactory();
			InputPointStreamInfo< Real , Dim , FunctionValueType >::BoundingBox( _pointStream , s , valueMin , valueMax );
		}
		if( GradientWeight.value>0 )
		{
			char* ext = GetFileExtension( InGradients.value );
			gradientSampleData = new std::vector< FunctionGradientType >();
			if( InCore.set )
			{
				InputPointGradientStream *_pointGradientStream;
				if     ( !strcasecmp( ext , "bnpts" ) ) _pointGradientStream = new BinaryInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
				else if( !strcasecmp( ext , "ply"   ) ) _pointGradientStream = new    PLYInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
				else                                    _pointGradientStream = new  ASCIIInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
				InputSampleGradientType s = inputSampleGradientFactory();
				while( _pointGradientStream->read( s ) ) inCorePointsAndGradients.push_back( s );
				delete _pointGradientStream;

				pointGradientStream = new VectorBackedInputDataStream< InputSampleGradientType >( inCorePointsAndGradients );
			}
			else
			{
				if     ( !strcasecmp( ext , "bnpts" ) ) pointGradientStream = new BinaryInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
				else if( !strcasecmp( ext , "ply"   ) ) pointGradientStream = new    PLYInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
				else                                    pointGradientStream = new  ASCIIInputDataStream< InputSampleGradientFactory>( InGradients.value , inputSampleGradientFactory );
			}
			delete[] ext;

			XInputPointGradientStream _pointStream( *pointGradientStream , modelToUnitCube );
			FunctionGradientType s = functionGradientFactory();
			InputPointStreamInfo< Real , Dim , FunctionGradientType >::BoundingBox( _pointStream , s , gradientMin , gradientMax );
		}

		{
			Point< Real , Dim > min , max;
			if( ValueWeight.value>0 && GradientWeight.value>0 ) for( int d=0 ; d<Dim ; d++ ) min[d] = std::min< Real >( valueMin[d] , gradientMin[d] ) , max[d] = std::max< Real >( valueMax[d] , gradientMax[d] );
			else if( ValueWeight.value>0 ) min = valueMin , max = valueMax;
			else if( GradientWeight.value>0 ) min = gradientMin , max = gradientMax;

			if( Width.value>0 )
			{
				modelToUnitCube = GetBoundingBoxXForm( min , max , (Real)Width.value , (Real)( Scale.value>0 ? Scale.value : 1. ) , Depth.value ) * modelToUnitCube;
				if( !SolveDepth.set || SolveDepth.value==-1 ) SolveDepth.value = Depth.value;
				if( SolveDepth.value>Depth.value )
				{
					MK_WARN( "Solution depth cannot exceed system depth: " , SolveDepth.value , " <= " , Depth.value );
					SolveDepth.value = Depth.value;
				}
				if( FullDepth.value>Depth.value )
				{
					MK_WARN( "Full depth cannot exceed system depth: " , FullDepth.value , " <= " , Depth.value );
					FullDepth.value = Depth.value;
				}
				if( BaseDepth.value>FullDepth.value )
				{
					if( BaseDepth.set ) MK_WARN( "Base depth must be smaller than full depth: " , BaseDepth.value , " <= " , FullDepth.value );
					BaseDepth.value = FullDepth.value;
				}
			}
			else modelToUnitCube = Scale.value>0 ? GetBoundingBoxXForm( min , max , (Real)Scale.value ) * modelToUnitCube : modelToUnitCube;
		}

		if( ValueWeight.value>0 )
		{
			valueSamples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
			XInputPointValueStream _pointStream( *pointValueStream , modelToUnitCube );
			auto ProcessData = []( const Point< Real , Dim > &p , FunctionValueType &d ){ return (Real)1.; };
			FunctionValueType zeroValue = functionValueFactory();
			typename FEMTreeInitializer< Dim , Real >::StreamInitializationData sid;
			{
				using ExternalType = std::tuple< Point< Real , Dim > , FunctionValueType >;
				using InternalType = std::tuple< InputSampleValueType >;
				auto converter = []( const InternalType &iType )
					{
						ExternalType xType;
						std::get< 0 >( xType ) = std::get< 0 >( iType ).template get<0>();
						std::get< 1 >( xType ) = std::get< 0 >( iType ).template get<1>();
						return xType;
					};
				InputDataStreamConverter< InternalType , ExternalType > __pointStream( _pointStream , converter , InputSampleValueType( Point< Real , Dim >() , zeroValue ) );
				pointValueCount = FEMTreeInitializer< Dim , Real >::template Initialize< FunctionValueType >( sid , tree.spaceRoot() , __pointStream , zeroValue , Depth.value , *valueSamples , *valueSampleData , tree.nodeAllocators.size() ? tree.nodeAllocators[0] : NULL , tree.initializer() , ProcessData );
			}
			delete pointValueStream;
		}
		else pointValueCount = 0;

		if( GradientWeight.value>0 )
		{
			gradientSamples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
			XInputPointGradientStream _pointStream( *pointGradientStream , modelToUnitCube );
			auto ProcessData = []( const Point< Real , Dim > &p , FunctionGradientType &d ){ return (Real)1.; };
			FunctionGradientType zeroGradient = functionGradientFactory();
			typename FEMTreeInitializer< Dim , Real >::StreamInitializationData sid;
			{
				using ExternalType = std::tuple< Point< Real , Dim > , FunctionGradientType >;
				using InternalType = std::tuple< InputSampleGradientType >;
				auto converter = []( const InternalType &iType )
					{
						ExternalType xType;
						std::get< 0 >( xType ) = std::get< 0 >( iType ).template get<0>();
						std::get< 1 >( xType ) = std::get< 0 >( iType ).template get<1>();
						return xType;
					};
				InputDataStreamConverter< InternalType , ExternalType > __pointGradientStream( _pointStream , converter , InputSampleGradientType( Point< Real , Dim >() , zeroGradient ) );
				pointGradientCount = FEMTreeInitializer< Dim , Real >::template Initialize< FunctionGradientType >( sid , tree.spaceRoot() , __pointGradientStream , zeroGradient , Depth.value , *gradientSamples , *gradientSampleData , tree.nodeAllocators.size() ? tree.nodeAllocators[0] : NULL , tree.initializer() , ProcessData );
			}
			delete pointGradientStream;
		}
		else pointGradientCount = 0;

		unitCubeToModel = modelToUnitCube.inverse();

		if( Verbose.set )
		{
			if( valueSamples ) std::cout << "Input Value Points / Value Samples: " << pointValueCount << " / " << valueSamples->size() << std::endl;
			if( gradientSamples ) std::cout << "Input Gradient Points / Gradient Samples: " << pointGradientCount << " / " << gradientSamples->size() << std::endl;
			std::cout << "# Read input into tree: " << profiler << std::endl;
		}
	}

	DenseNodeData< Real , Sigs > solution;
	{
		DenseNodeData< Real , Sigs > constraints;
		ValueInterpolationInfo *valueInterpolationInfo = NULL;
		GradientInterpolationInfo *gradientInterpolationInfo = NULL;
		int solveDepth = Depth.value;

		tree.resetNodeIndices( 0 );
	
		if( ValueWeight.value>0 )
		{
			if( ExactInterpolation.set ) valueInterpolationInfo = FEMTree< Dim , Real >::template       InitializeExactPointAndDataInterpolationInfo< Real , FunctionValueType , 0 >( tree , *valueSamples , GetPointer( *valueSampleData ) , ConstraintDual< Dim , Real , FunctionValueType >( (Real)ValueWeight.value ) , SystemDual< Dim , Real , FunctionValueType >( (Real)ValueWeight.value ) , true , false );
			else                         valueInterpolationInfo = FEMTree< Dim , Real >::template InitializeApproximatePointAndDataInterpolationInfo< Real , FunctionValueType , 0 >( tree , *valueSamples , GetPointer( *valueSampleData ) , ConstraintDual< Dim , Real , FunctionValueType >( (Real)ValueWeight.value ) , SystemDual< Dim , Real , FunctionValueType >( (Real)ValueWeight.value ) , true , Depth.value , 1 );
		}
		if( GradientWeight.value>0 )
		{
			if( ExactInterpolation.set ) gradientInterpolationInfo = FEMTree< Dim , Real >::template       InitializeExactPointAndDataInterpolationInfo< Real , FunctionGradientType , 1 >( tree , *gradientSamples , GetPointer( *gradientSampleData ) , ConstraintDual< Dim , Real , FunctionGradientType >( (Real)GradientWeight.value ) , SystemDual< Dim , Real , FunctionGradientType >( (Real)GradientWeight.value ) , true , false );
			else                         gradientInterpolationInfo = FEMTree< Dim , Real >::template InitializeApproximatePointAndDataInterpolationInfo< Real , FunctionGradientType , 1 >( tree , *gradientSamples , GetPointer( *gradientSampleData ) , ConstraintDual< Dim , Real , FunctionGradientType >( (Real)GradientWeight.value ) , SystemDual< Dim , Real , FunctionGradientType >( (Real)GradientWeight.value ) , true , Depth.value , 1 );
		}

		// Prepare for multigrid
		{
			profiler.reset();

			auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=FullDepth.value; };
			if( ValueWeight.value>0 && GradientWeight.value>0 )
				tree.template finalizeForMultigrid< Degrees::Max() , Degrees::Max() >( BaseDepth.value , addNodeFunctor , []( const FEMTreeNode * ){ return true; } , std::make_tuple( valueInterpolationInfo , gradientInterpolationInfo ) );
			else if( ValueWeight.value>0 )
				tree.template finalizeForMultigrid< Degrees::Max() , Degrees::Max() >( BaseDepth.value , addNodeFunctor , []( const FEMTreeNode * ){ return true; } , std::make_tuple( valueInterpolationInfo                             ) );
			else if( GradientWeight.value>0 )
				tree.template finalizeForMultigrid< Degrees::Max() , Degrees::Max() >( BaseDepth.value , addNodeFunctor , []( const FEMTreeNode * ){ return true; } , std::make_tuple(                          gradientInterpolationInfo ) );

			if( Verbose.set ) std::cout << "#       Finalized tree: " << profiler << std::endl;
		}

		// Add the interpolation constraints
		{
			profiler.reset();
			constraints = tree.initDenseNodeData( Sigs() );
			if( ValueWeight.value>0 && GradientWeight.value>0 ) tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( valueInterpolationInfo , gradientInterpolationInfo ) );
			else if( ValueWeight.value>0 )                      tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( valueInterpolationInfo                             ) );
			else if( GradientWeight.value>0 )                   tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple(                          gradientInterpolationInfo ) );
			if( Verbose.set ) std::cout << "#Set point constraints: " << profiler << std::endl;
		}

		if( Verbose.set )
		{
			std::cout << "All Nodes / Active Nodes / Ghost Nodes: " << tree.allNodes() << " / " << tree.activeNodes() << " / " << tree.ghostNodes() << std::endl;
			std::cout << "Memory Usage: " << float( MemoryInfo::Usage() )/(1<<20) << " MB" << std::endl;
		}

		// Solve the linear system
		{
			profiler.reset();
			typename FEMTree< Dim , Real >::SolverInfo sInfo;
			sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = Iters.value , sInfo.cgAccuracy = CGSolverAccuracy.value , sInfo.verbose = Verbose.set , sInfo.showResidual = ShowResidual.set , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
			sInfo.baseVCycles = BaseVCycles.value;
			typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 2 > > F( { 0. , (double)LapWeight.value , (double)BiLapWeight.value } );
			if( ValueWeight.value>0 && GradientWeight.value>0 ) solution = tree.solveSystem( Sigs() , F , constraints , BaseDepth.value , SolveDepth.value , sInfo , std::make_tuple( valueInterpolationInfo , gradientInterpolationInfo ) );
			else if( ValueWeight.value>0 )                      solution = tree.solveSystem( Sigs() , F , constraints , BaseDepth.value , SolveDepth.value , sInfo , std::make_tuple( valueInterpolationInfo                             ) );
			else if( GradientWeight.value>0 )                   solution = tree.solveSystem( Sigs() , F , constraints , BaseDepth.value , SolveDepth.value , sInfo , std::make_tuple(                          gradientInterpolationInfo ) );
			if( Verbose.set ) std::cout << "# Linear system solved: " << profiler << std::endl;
			delete valueInterpolationInfo , valueInterpolationInfo = NULL;
			delete gradientInterpolationInfo , gradientInterpolationInfo = NULL;
		}
	}

	if( Verbose.set )
	{
		if( valueSamples )
		{
			typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &tree , solution );
			std::pair< double , double > valueStat(0,0);
			std::vector< std::pair< double , double > > valueStats( ThreadPool::NumThreads() , std::pair< double , double >(0,0) );
			ThreadPool::ParallelFor( 0 , valueSamples->size() , [&]( unsigned int thread , size_t j )
			{
				ProjectiveData< Point< Real , Dim > , Real >& sample = (*valueSamples)[j].sample;
				Real w = sample.weight;
				if( w>0 )
				{
					CumulativeDerivativeValues< Real , Dim , 0 > values = evaluator.values( sample.data / sample.weight , thread , (*valueSamples)[j].node );
					Real v1 = values[0];
					Real v2 = (*valueSampleData)[j].template get<0>() / w;
					valueStats[ thread ].first += ( v1 - v2 ) * ( v1 - v2 ) * w;
					valueStats[ thread ].second += ( v1 * v1 + v2 * v2 ) * w;
				}
			}
			);
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) valueStat.first += valueStats[t].first , valueStat.second += valueStats[t].second;
			if( Verbose.set ) std::cout << "Value Error: " << sqrt( valueStat.first / valueStat.second ) << std::endl;
		}
		if( gradientSamples )
		{
			typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 1 > evaluator( &tree , solution );
			std::pair< double , double > gradientStat(0,0);
			std::vector< std::pair< double , double > > gradientStats( ThreadPool::NumThreads() , std::pair< double , double >(0,0) );
			ThreadPool::ParallelFor( 0 , gradientSamples->size() , [&]( unsigned int thread , size_t j )
			{
				ProjectiveData< Point< Real , Dim > , Real >& sample = (*gradientSamples)[j].sample;
				Real w = sample.weight;
				if( w>0 )
				{
					CumulativeDerivativeValues< Real , Dim , 1 > values = evaluator.values( sample.data / sample.weight , thread , (*gradientSamples)[j].node );
					Point< Real , Dim > g1;
					for( int d=0 ; d<Dim ; d++ ) g1[d] = values[d+1];
					Point< Real , Dim > g2 = (*gradientSampleData)[j].template get<0>() / w;
					gradientStats[ thread ].first += Point< Real , Dim >::SquareNorm( g1 - g2 ) * w;
					gradientStats[ thread ].second += ( Point< Real , Dim >::SquareNorm( g1 ) + Point< Real , Dim >::SquareNorm( g2 ) ) * w;
				}
			}
			);
			for( unsigned int t=0 ; t<ThreadPool::NumThreads() ; t++ ) gradientStat.first += gradientStats[t].first , gradientStat.second += gradientStats[t].second;
			if( Verbose.set ) std::cout << "Gradient Error: " << sqrt( gradientStat.first / gradientStat.second ) << std::endl;
		}
	}

	delete valueSamples , valueSamples = NULL;
	delete gradientSamples , gradientSamples = NULL;
	delete valueSampleData , valueSampleData = NULL;
	delete gradientSampleData , gradientSampleData = NULL;

	if( Tree.set )
	{
		FILE* fp = fopen( Tree.value , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , Tree.value );
		FileStream fs( fp );
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		tree.write( fs , false );
		fs.write( modelToUnitCube );
		solution.write( fs );
		fclose( fp );
	}

	if( Grid.set )
	{
		int res = 0;
		profiler.reset();
		Pointer( Real ) values = tree.template regularGridEvaluate< true >( solution , res , -1 , PrimalGrid.set );
		size_t resolution = 1;
		for( int d=0 ; d<Dim ; d++ ) resolution *= res;
		if( Verbose.set ) std::cout << "Got grid: " << profiler << std::endl;
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );
		WriteGrid< Real , Dim >( Grid.value , values , res , unitCubeToModel * voxelToUnitCube , Verbose.set );
		DeletePointer( values );
	}

	if( Out.set )
	{
		if constexpr ( Dim==3 )
		{
			ExtractLevelSet( UIntPack< FEMSigs ... >() , tree , solution , (Real)IsoValue.value , unitCubeToModel , comments );
		}
		else if constexpr ( Dim==2 )
		{
			typedef VertexFactory::PositionFactory< Real , 3 > VertexFactory;
			int res = 0;
			Pointer( Real ) values = tree.template regularGridEvaluate< true >( solution , res , -1 , true );
			res--;
			std::vector< std::vector< int > > polygons( res * res );
			std::vector< typename VertexFactory::VertexType > vertices( (res+1) * (res+1) );

			for( int i=0 ; i<res ; i++ ) for( int j=0 ; j<res ; j++ )
			{
				std::vector< int > &poly = polygons[ j*res+i ];
				poly.resize( 4 );
				poly[0] = (j+0)*(res+1)+(i+0);
				poly[1] = (j+0)*(res+1)+(i+1);
				poly[2] = (j+1)*(res+1)+(i+1);
				poly[3] = (j+1)*(res+1)+(i+0);
			}
			for( int i=0 ; i<=res ; i++ ) for( int j=0 ; j<=res ; j++ )
			{
				Point< Real , Dim > p;
				p[0] = (Real)i/res;
				p[1] = (Real)j/res;
				p = unitCubeToModel * p;
				vertices[ j*(res+1)+i ] = Point< float , 3 >( (float)p[0] , (float)p[1] , values[ j*(res+1)+i ] );
			}
			DeletePointer( values );

			std::vector< std::string > noComments;
			PLY::WritePolygons( Out.value , VertexFactory() , vertices , polygons , ASCII.set ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
		}
	}

	if( Verbose.set ) std::cout <<"#          Total Solve: " << Time()-startTime << " (s), " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
}

#ifndef FAST_COMPILE
template< unsigned int Dim , class Real , BoundaryType BType >
void Execute( void )
{
	switch( Degree.value )
	{
		//		case 1: return Execute< Real >( IsotropicUIntPack< Dim , FEMDegreeAndBType< 1 , BType >::Signature >() );
		case 2: return Execute< Real >( IsotropicUIntPack< Dim , FEMDegreeAndBType< 2 , BType >::Signature >() );
		case 3: return Execute< Real >( IsotropicUIntPack< Dim , FEMDegreeAndBType< 3 , BType >::Signature >() );
			//		case 4: return Execute< Real >( IsotropicUIntPack< Dim , FEMDegreeAndBType< 4 , BType >::Signature >() );
		default: MK_THROW( "Only B-Splines of degree 1 - 3 are supported" );
	}
}

template< unsigned int Dim , class Real >
void Execute( void )
{
	switch( BType.value )
	{
		case BOUNDARY_FREE+1:      return Execute< Dim , Real , BOUNDARY_FREE      >();
		case BOUNDARY_NEUMANN+1:   return Execute< Dim , Real , BOUNDARY_NEUMANN   >();
		case BOUNDARY_DIRICHLET+1: return Execute< Dim , Real , BOUNDARY_DIRICHLET >();
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

	if( !InValues.set && !InGradients.set )
	{
		ShowUsage( argv[0] );
		MK_THROW( "Either values or gradients need to be specified" );
		return 0;
	}
	if( !InValues.set ) ValueWeight.value = 0;
	if( !InGradients.set ) GradientWeight.value = 0;

	if( ValueWeight.value<0 ) MK_THROW( "Value weight must be non-negative: " , ValueWeight.value , "> 0" );
	if( GradientWeight.value<0 ) MK_THROW( "Gradient weight must be non-negative: " , GradientWeight.value , "> 0" );
	if( !ValueWeight.value && !GradientWeight.value ) MK_THROW( "Either value or gradient weight must be positive" );

	if( LapWeight.value<0 ) MK_THROW( "Laplacian weight must be non-negative: " , LapWeight.value , " > 0" );
	if( BiLapWeight.value<0 ) MK_THROW( "Bi-Laplacian weight must be non-negative: " , BiLapWeight.value , " > 0" );
	if( !LapWeight.value && !BiLapWeight.value ) MK_THROW( "Eiter Laplacian or bi-Laplacian weight must be positive" );

	if( !BaseDepth.set ) BaseDepth.value = FullDepth.value;
	if( BaseDepth.value>FullDepth.value )
	{
		if( BaseDepth.set ) MK_WARN( "Base depth must be smaller than full depth: " , BaseDepth.value , " <= " , FullDepth.value );
		BaseDepth.value = FullDepth.value;
	}
	if( !SolveDepth.set || SolveDepth.value==-1 ) SolveDepth.value = Depth.value;
	if( SolveDepth.value>Depth.value )
	{
		MK_WARN( "Solution depth cannot exceed system depth: " , SolveDepth.value , " <= " , Depth.value );
		SolveDepth.value = Depth.value;
	}

#ifdef USE_DOUBLE
	typedef double Real;
#else // !USE_DOUBLE
	typedef float  Real;
#endif // USE_DOUBLE

#ifdef FAST_COMPILE
	static const int Dimension = DEFAULT_DIMENSION;
	static const int Degree = DEFAULT_FEM_DEGREE;
	static const BoundaryType BType = DEFAULT_FEM_BOUNDARY;
	typedef IsotropicUIntPack< Dimension , FEMDegreeAndBType< Degree , BType >::Signature > FEMSigs;
	MK_WARN( "Compiled for degree-" , Degree , ", boundary-" , BoundaryNames[ BType ] , ", " , sizeof(Real)==4 ? "single" : "double" , "-precision _only_" );
	Execute< Real >( FEMSigs() );
#else // !FAST_COMPILE
	if     ( Dimension.value==2 ) Execute< 2 , Real >();
	else if( Dimension.value==3 ) Execute< 3 , Real >();
	else MK_THROW( "Only Degrees 2 and 3 are supported" );
#endif // FAST_COMPILE
	if( Performance.set )
	{
		printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
		printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
	}
	return EXIT_SUCCESS;
}
