/*
Copyright (c) 2024, Michael Kazhdan
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

#ifndef EXTRAPOLATOR_INCLUDED
#define EXTRAPOLATOR_INCLUDED

#include "PreProcessor.h"
#include "MyMiscellany.h"
#include "FEMTree.h"
#include "PointExtent.h"
#include "Reconstructors.streams.h"

namespace PoissonRecon
{
	namespace Extrapolator
	{
		static unsigned int ProfilerMS = 20;		// The number of ms at which to poll the performance (set to zero for no polling)

		template< typename Real , unsigned int Dim , typename AuxData >
		using InputStream = Reconstructor::InputSampleStream< Real , Dim , AuxData >;

		// Specialized solution information without auxiliary data
		template< typename Real , unsigned int Dim , typename AuxData , unsigned int DataDegree=1 >
		struct Implicit
		{
			struct Parameters
			{
				bool verbose = false;
				Real scale = (Real)1.1;
				Real width = (Real)0.;
				Real samplesPerNode = (Real)1.5;
				Real perLevelScaleFactor = (Real)32;
				unsigned int baseDepth = -1;
				unsigned int depth = 8;
				unsigned int fullDepth = 5;
				unsigned int alignDir = 0;
			};

			// The constructor
			Implicit( InputStream< Real , Dim , AuxData > &pointStream , Parameters params , AuxData zero );

			// The desctructor
			~Implicit( void ){ delete auxData ; auxData = nullptr ; delete _auxEvaluator ; _auxEvaluator = nullptr; }

			// The transformation taking points in the unit cube back to world coordinates
			XForm< Real , Dim+1 > unitCubeToModel;

			// The octree adapted to the points
			FEMTree< Dim , Real > tree;

			// The signature of the finite-element used for data extrapolation
			static const unsigned int DataSig = FEMDegreeAndBType< DataDegree , BOUNDARY_FREE >::Signature;

			// The auxiliary information stored with the oriented vertices
			SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *auxData;

			// An instance of "zero" AuxData
			AuxData zeroAuxData;

			struct OutOfUnitCubeException : public std::exception
			{
				OutOfUnitCubeException( Point< Real , Dim > world , Point< Real , Dim > cube )
				{
					std::stringstream sStream;
					sStream << "Out-of-unit-cube input: " << world << " -> " << cube;
					_message = sStream.str();						
				}
				const char * what( void ) const noexcept { return _message.c_str(); }
			protected:
				std::string _message;
			};

			XForm< Real , Dim+1 > worldToUnitCubeTransform( void ) const { return _worldToUnitCube; }

			AuxData operator()( unsigned int t , Point< Real , Dim > p ){ return _value(t,p); }
			AuxData operator()(                  Point< Real , Dim > p ){ return _value(0,p); }

			void evaluate( unsigned int t , Point< Real , Dim > p , AuxData &data ){ return _evaluate( t , p , data ); }
			void evaluate(                  Point< Real , Dim > p , AuxData &data ){ return _evaluate( 0 , p , data ); }

		protected:
			typename FEMTree< Dim , Real >::template MultiThreadedSparseEvaluator< IsotropicUIntPack< Dim , DataSig > , ProjectiveData< AuxData , Real > > *_auxEvaluator = nullptr;
			XForm< Real , Dim+1 > _worldToUnitCube;

			AuxData _value( unsigned int t , Point< Real , Dim > p )
			{
				Point< Real , Dim > q = _worldToUnitCube * p;
				for( unsigned int d=0 ; d<Dim ; d++ ) if( q[d]<0 || q[d]>1 ) throw OutOfUnitCubeException(p,q);
				ProjectiveData< AuxData , Real > pData( zeroAuxData );
				_auxEvaluator->addValue( q , pData , t );
				return pData.value();
			}

			void _evaluate( unsigned int t , Point< Real , Dim > p , AuxData &data )
			{
				Point< Real , Dim > q = _worldToUnitCube * p;
				for( unsigned int d=0 ; d<Dim ; d++ ) if( q[d]<0 || q[d]>1 ) throw OutOfUnitCubeException(p,q);

				Real weight = (Real)0.;
				data *= 0;

				auto Accumulation = [&weight,&data]( const ProjectiveData< AuxData , Real > &pData , Real scale )
					{
						weight += pData.weight * scale;
						data += pData.data * scale;
					};
				_auxEvaluator->accumulate( q , Accumulation , t );
				if( weight ) data /= weight;
			}
		};

		template< typename Real , unsigned int Dim , typename AuxData , unsigned int DataDegree >
		Implicit< Real , Dim , AuxData , DataDegree >::Implicit( InputStream< Real , Dim , AuxData > &pointStream , Parameters params , AuxData zeroAuxData )
			: tree(MEMORY_ALLOCATOR_BLOCK_SIZE) , unitCubeToModel( XForm< Real , Dim+1 >::Identity() ) , zeroAuxData(zeroAuxData)
		{
			// The finite-element tracking tree node
			using FEMTreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;

			// The type describing the sampling density
			using DensityEstimator = typename FEMTree< Dim , Real >::template DensityEstimator< 0 >;

			// The signature for the finite-elements representing the auxiliary data (if it's there)
			static const unsigned int DataSig = FEMDegreeAndBType< DataDegree , BOUNDARY_FREE >::Signature;

			XForm< Real , Dim+1 > modelToUnitCube = XForm< Real , Dim+1 >::Identity();

			Profiler profiler( ProfilerMS );

			size_t pointCount;

			ProjectiveData< Point< Real , 2 > , Real > pointDepthAndWeight;
			std::vector< typename FEMTree< Dim , Real >::PointSample > *samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
			std::vector< AuxData > *sampleAuxData = nullptr;

			// Read in the samples (and auxiliary data)
			{
				profiler.reset();

				pointStream.reset();
				sampleAuxData = new std::vector< AuxData >();

				modelToUnitCube = params.scale>0 ? PointExtent::GetXForm< Real , Dim , true , AuxData >( pointStream , zeroAuxData , params.scale , params.alignDir ) * modelToUnitCube : modelToUnitCube;
				pointStream.reset();

				if( params.width>0 )
				{
					// Assuming the transformation is rigid so that the (max) scale can be pulled from the Frobenius norm
					Real maxScale = 0;
					for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) maxScale += modelToUnitCube(i,j) * modelToUnitCube(i,j);
					maxScale = (Real)( 1. / sqrt( maxScale / Dim ) );
					params.depth = (unsigned int)ceil( std::max< double >( 0. , log( maxScale/params.width )/log(2.) ) );
				}

				{
					Reconstructor::TransformedInputSampleStream< Real , Dim , AuxData > _pointStream( modelToUnitCube , pointStream );
					auto ProcessData = []( const Point< Real , Dim > &p , AuxData &d ){ return (Real)1.; };
					pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< AuxData >( tree.spaceRoot() , _pointStream , zeroAuxData , params.depth , *samples , *sampleAuxData , tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr , tree.initializer() , ProcessData );
				}

				if( params.fullDepth>params.depth )
				{
					if( params.fullDepth!=-1 ) MK_WARN( "Full depth cannot exceed depth: " , params.fullDepth , " <= " , params.depth );
					params.fullDepth = params.depth;
				}
				if( params.baseDepth>params.fullDepth )
				{
					if( params.baseDepth!=-1 ) MK_WARN( "Base depth must be smaller than full depth: " , params.baseDepth , " <= " , params.fullDepth );
					params.baseDepth = params.fullDepth;
				}


				unitCubeToModel = modelToUnitCube.inverse();

				if( params.verbose )
				{
					std::cout << "Input Points / Samples: " << pointCount << " / " << samples->size() << std::endl;
					std::cout << "# Read input into tree: " << profiler << std::endl;
				}
			}
			tree.resetNodeIndices( 0 , std::make_tuple() );
			{
				profiler.reset();
				auto SampleFunctor = [&]( size_t i ) -> const typename FEMTree< Dim , Real >::PointSample & { return (*samples)[i]; };
				auto SampleDataFunctor = [&]( size_t i ) -> const AuxData & { return (*sampleAuxData)[i]; };
				auxData = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >( tree.template setExtrapolatedDataField< DataSig , false , 0 , AuxData >( zeroAuxData , samples->size() , SampleFunctor , SampleDataFunctor , (DensityEstimator*)nullptr ) );
				delete sampleAuxData;
				if( params.verbose ) std::cout << "#         Got aux data: " << profiler << std::endl;
			}

			{
				profiler.reset();
				auto hasDataFunctor = [&]( const FEMTreeNode *node )
					{
						ProjectiveData< AuxData , Real > *data = auxData->operator()( node );
						return data && data->weight;
					};
				auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)params.fullDepth; };
				tree.template finalizeForMultigrid< 0 , 0 >( 0 , addNodeFunctor , hasDataFunctor , std::make_tuple() , std::make_tuple( auxData ) );
			}

			if( params.verbose ) std::cout << "#       Finalized tree: " << profiler << std::endl;
			if( params.verbose ) std::cout << "All Nodes / Active Nodes / Ghost Nodes: " << tree.allNodes() << " / " << tree.activeNodes() << " / " << tree.ghostNodes() << std::endl;
			if( params.verbose ) std::cout << "Memory Usage: " << float( MemoryInfo::Usage())/(1<<20) << " MB" << std::endl;
			delete samples;

			auto nodeFunctor = [&]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *n )
				{
					ProjectiveData< AuxData , Real >* clr = (*auxData)( n );
					if( clr ) (*clr) *= (Real)pow( params.perLevelScaleFactor , tree.depth( n ) );
				};
			tree.tree().processNodes( nodeFunctor );

			_worldToUnitCube = unitCubeToModel.inverse();
			_auxEvaluator = new typename FEMTree< Dim , Real >::template MultiThreadedSparseEvaluator< IsotropicUIntPack< Dim , DataSig > , ProjectiveData< AuxData , Real > >( &tree , *auxData , ThreadPool::NumThreads() );

		}
	}
}

#endif // EXTRAPOLATOR_INCLUDED