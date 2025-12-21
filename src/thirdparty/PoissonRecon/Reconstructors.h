/*
Copyright (c) 2022, Michael Kazhdan and Matthew Bolitho
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

#ifndef RECONSTRUCTORS_INCLUDED
#define RECONSTRUCTORS_INCLUDED

#include "PreProcessor.h"
#include "MyMiscellany.h"
#include "DataStream.imp.h"
#include "FEMTree.h"
#include "PointExtent.h"
#include "Reconstructors.streams.h"

namespace PoissonRecon
{

	namespace Reconstructor
	{
		static unsigned int ProfilerMS = 20;		// The number of ms at which to poll the performance (set to zero for no polling)
		static const unsigned int DataDegree = 0;	// The order of the B-Spline used to splat in data for auxiliary data interpolation
		static const unsigned int WeightDegree = 2;	// The order of the B-Spline used to splat in the weights for density estimation

		// Declare a type for storing the solution information
		template< typename Real , unsigned int Dim , typename FEMSigPack /* = UIntPack< FEMSigs... >*/ , typename ... AuxData > struct Implicit;

		// Parameters for mesh extraction
		struct LevelSetExtractionParameters
		{
			bool linearFit;
			bool outputGradients;
			bool forceManifold;
			bool polygonMesh;
			bool gridCoordinates;
			bool outputDensity;
			bool verbose;
			LevelSetExtractionParameters( void ) : linearFit(false) , outputGradients(false) , forceManifold(true) , polygonMesh(false) , gridCoordinates(false) , verbose(false) , outputDensity(false) {}
		};

		// General solver parameters
		template< typename Real >
		struct SolutionParameters
		{
			bool verbose;
			bool exactInterpolation;
			bool showResidual;
			bool confidence;
			Real scale;
			Real lowDepthCutOff;
			Real width;
			Real samplesPerNode;
			Real cgSolverAccuracy;
			Real perLevelDataScaleFactor;
			unsigned int depth;
			unsigned int solveDepth;
			unsigned int baseDepth;
			unsigned int fullDepth;
			unsigned int kernelDepth;
			unsigned int baseVCycles;
			unsigned int iters;
			unsigned int alignDir;

			SolutionParameters( void ) :
				verbose(false) , exactInterpolation(false) , showResidual(false) , confidence(false) ,
				scale((Real)1.1) ,
				lowDepthCutOff((Real)0.) , width((Real)0.) ,
				samplesPerNode((Real)1.5) , cgSolverAccuracy((Real)1e-3 ) , perLevelDataScaleFactor((Real)32.) ,
				depth((unsigned int)8) , solveDepth((unsigned int)-1) , baseDepth((unsigned int)-1) , fullDepth((unsigned int)5) , kernelDepth((unsigned int)-1) ,
				baseVCycles((unsigned int)1) , iters((unsigned int)8) , alignDir(0)
			{}

			template< unsigned int Dim >
			void testAndSet( XForm< Real , Dim+1 > unitCubeToModel )
			{
				if( width>0 )
				{
					Real maxScale = 0;
					for( unsigned int i=0 ; i<Dim ; i++ )
					{
						Real l2 = 0;
						for( unsigned int j=0 ; j<Dim ; j++ ) l2 += unitCubeToModel(i,j) * unitCubeToModel(i,j);
						if( l2>maxScale ) maxScale = l2;
					}
					maxScale = sqrt( maxScale );
					depth = (unsigned int)ceil( std::max< double >( 0. , log( maxScale/width )/log(2.) ) );
				}

				if( solveDepth>depth )
				{
					if( solveDepth!=-1 ) MK_WARN( "Solution depth cannot exceed system depth: " , solveDepth , " <= " , depth );
					solveDepth = depth;
				}
				if( fullDepth>solveDepth )
				{
					if( fullDepth!=-1 ) MK_WARN( "Full depth cannot exceed system depth: " , fullDepth , " <= " , solveDepth );
					fullDepth = solveDepth;
				}
				if( baseDepth>fullDepth )
				{
					if( baseDepth!=-1 ) MK_WARN( "Base depth must be smaller than full depth: " , baseDepth , " <= " , fullDepth );
					baseDepth = fullDepth;
				}
				if( kernelDepth==-1 ) kernelDepth = depth>2 ? depth-2 : 0;
				if( kernelDepth>depth )
				{
					if( kernelDepth!=-1 ) MK_WARN( "Kernel depth cannot exceed system depth: " , kernelDepth , " <= " , depth );
					kernelDepth = depth;
				}
			}
		};
		struct Poisson;
		struct SSD;

		// Specialized solution information without auxiliary data
		template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
		struct Implicit< Real , Dim , UIntPack< FEMSigs ... > , AuxData ... >
		{
			static_assert( sizeof...(FEMSigs)==Dim , "[ERROR] Number of FEM signatures doesn't match dimension" );

			// The internal type representing the auxiliary data
			using InternalAuxData = DirectSum< Real , AuxData... >;

			// The internal type representing the normal and auxiliary data
			using InternalNormalAndAuxData = DirectSum< Real , Normal< Real , Dim > , InternalAuxData >;

			// The signature pack
			using Sigs = UIntPack< FEMSigs ... >;

			// The type representing the point sampling density
			typedef typename FEMTree< Dim , Real >::template DensityEstimator< Reconstructor::WeightDegree > DensityEstimator;

			// The signature of the finite-element used for data extrapolation
			static const unsigned int DataSig = FEMDegreeAndBType< Reconstructor::DataDegree , BOUNDARY_FREE >::Signature;

			// The constructor
			Implicit( AuxData ... zeroAuxData ) : density(nullptr) , isoValue(0) , tree(MEMORY_ALLOCATOR_BLOCK_SIZE) , unitCubeToModel( XForm< Real , Dim+1 >::Identity() ) , _auxData(nullptr) , _zeroAuxData(zeroAuxData...) { _zeroNormalAndAuxData = InternalNormalAndAuxData( Normal< Real , Dim >() , _zeroAuxData ); }

			// The desctructor
			~Implicit( void ){ delete density ; density = nullptr ; delete _auxData ; _auxData = nullptr; }

			// Write out to file
			void write( BinaryStream &stream ) const
			{
				tree.write( stream , false );
				stream.write( isoValue );
				stream.write( unitCubeToModel );
				solution.write( stream );
				density->write( stream );
				if constexpr( sizeof...(AuxData) ) _auxData->write( stream );
			}

			// The transformation taking points in the unit cube back to world coordinates
			XForm< Real , Dim+1 > unitCubeToModel;

			// The octree adapted to the points
			FEMTree< Dim , Real > tree;

			// The solution coefficients
			DenseNodeData< Real , Sigs > solution;

			// The average value at the sample positions
			Real isoValue;

			// The density estimator computed from the samples
			DensityEstimator *density;

			// A method that writes the extracted mesh to the streams
			void extractLevelSet( OutputLevelSetVertexStream< Real , Dim , AuxData... > &vertexStream , OutputFaceStream< Dim-1 > &faceStream , LevelSetExtractionParameters params ) const;

			struct Evaluator
			{
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

				Evaluator( const FEMTree< Dim , Real > *tree , const DenseNodeData< Real , Sigs > &coefficients , XForm< Real , Dim+1 > worldToUnitCube )
					: _evaluator( tree , coefficients , ThreadPool::NumThreads() ) , _xForm(worldToUnitCube) { _gxForm = XForm< Real , Dim >(_xForm).transpose(); }

				Point< Real , Dim > grad( unsigned int t , Point< Real , Dim > p )
				{
					CumulativeDerivativeValues< Real , Dim , 1 > v = _values( t , p );
					Point< Real , Dim > g;
					for( unsigned int d=0 ; d<Dim ; d++ ) g[d] = v[d+1];
					return _gxForm * g;
				}
				Real operator()( unsigned int t , Point< Real , Dim > p ){ return _values(t,p)[0]; }

				Real operator()( Point< Real , Dim > p ){ return operator()( 0 , p ); }
				Point< Real , Dim > grad( Point< Real , Dim > p ){ return grad( 0 , p ); }
			protected:
				CumulativeDerivativeValues< Real , Dim , 1 > _values( unsigned int t , Point< Real , Dim > p )
				{
					Point< Real , Dim > q = _xForm * p;
					for( unsigned int d=0 ; d<Dim ; d++ ) if( q[d]<0 || q[d]>1 ) throw OutOfUnitCubeException(p,q);
					return _evaluator.values( q , t );
				}

				typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 1 > _evaluator;
				XForm< Real , Dim+1 > _xForm;
				XForm< Real , Dim > _gxForm;
			};
			Evaluator evaluator( void ) const { return Evaluator( &tree , solution , unitCubeToModel.inverse() ); }

			struct AuxEvaluator
			{
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

				AuxEvaluator( const FEMTree< Dim , Real > *tree , const SparseNodeData< ProjectiveData< InternalAuxData , Real > , IsotropicUIntPack< Dim , DataSig > > &coefficients , XForm< Real , Dim+1 > worldToUnitCube , InternalAuxData zero )
					: _auxEvaluator( tree , coefficients , ThreadPool::NumThreads() ) , _xForm(worldToUnitCube) , _zero(zero) {}

				void set( unsigned int t , Point< Real , Dim > p , AuxData&...d ){ _SetFromInternal( _value(t,p) , d... ); }

				void set( Point< Real , Dim > p , AuxData&...d ){ return operator()( 0 , p , d... ); }

				XForm< Real , Dim+1 > worldToUnitCubeTransform( void ) const { return worldToUnitCubeTransform(); }
			protected:
				InternalAuxData _value( unsigned int t , Point< Real , Dim > p )
				{
					Point< Real , Dim > q = _xForm * p;
					for( unsigned int d=0 ; d<Dim ; d++ ) if( q[d]<0 || q[d]>1 ) throw OutOfUnitCubeException(p,q);
					ProjectiveData< InternalAuxData , Real > pData( _zero );
					_auxEvaluator.addValue( q , pData );
					return pData.value();
				}

				template< typename D , typename ... Ds >
				static void _SetFromInternal( InternalAuxData iData , D &d , Ds&... ds )
				{
					d = iData.template get< sizeof...(AuxData) - sizeof...(Ds) - 1 >();
					if constexpr( sizeof...(Ds) ) _SetFromInternal( iData , ds... );
				}

				typename FEMTree< Dim , Real >::template MultiThreadedSparseEvaluator< IsotropicUIntPack< Dim , DataSig > , ProjectiveData< InternalAuxData , Real > > _auxEvaluator;
				XForm< Real , Dim+1 > _xForm;
				InternalAuxData _zero;
			};

			AuxEvaluator auxEvaluator( void ) const
			{
				static_assert( sizeof...(AuxData) , "No auxiliary data" );
				return AuxEvaluator( &tree , *_auxData , unitCubeToModel.inverse() , _zeroAuxData );
			}

		protected:
			// The auxiliary information stored with the oriented vertices
			SparseNodeData< ProjectiveData< InternalAuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *_auxData;

			// An instance of "zero" AuxData
			InternalAuxData _zeroAuxData;

			InternalNormalAndAuxData _zeroNormalAndAuxData;

			struct _VertexTypeConverter
			{
				using ExternalVertexType = std::tuple< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , InternalAuxData >;
				using InternalVertexType = std::tuple< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , AuxData...  >;

				static ExternalVertexType ConvertI2X( const InternalVertexType &iType )
				{
					ExternalVertexType xType;

					// Copy the position information
					std::get< 0 >( xType ) = std::get< 0 >( iType );
					std::get< 1 >( xType ) = std::get< 1 >( iType );
					std::get< 2 >( xType ) = std::get< 2 >( iType );
					// Copy the remainder of the information
					_SetAuxData< 0 >( iType , xType );
					return xType;
				}
				static InternalVertexType ConvertX2I( const ExternalVertexType &xType )
				{
					InternalVertexType iType;

					// Copy the position information
					std::get< 0 >( iType ) = std::get< 0 >( xType );
					std::get< 1 >( iType ) = std::get< 1 >( xType );
					std::get< 2 >( iType ) = std::get< 2 >( xType );
					// Copy the remainder of the information
					_SetAuxData< 0 >( xType , iType );
					return iType;
				}
			protected:
				template< unsigned int I >
				static void _SetAuxData( const InternalVertexType &iType , ExternalVertexType &xType )
				{
					std::get< 3 >( xType ).template get< I >() = std::get< I+3 >( iType );
					if constexpr( I+1<sizeof...(AuxData) ) _SetAuxData< I+1 >( iType , xType );
				}
				template< unsigned int I >
				static void _SetAuxData( const ExternalVertexType &xType , InternalVertexType &iType )
				{
					std::get< I+3 >( iType ) = std::get< 3 >( xType ).template get< I >();
					if constexpr( I+1<sizeof...(AuxData) ) _SetAuxData< I+1 >( xType , iType );
				}
			};

			struct _SampleTypeConverter
			{
				using ExternalSampleType = std::tuple< Position< Real , Dim > , InternalNormalAndAuxData >;
				using InternalSampleType = std::tuple< Position< Real , Dim > , Normal< Real , Dim > , AuxData... >;

				static ExternalSampleType ConvertI2X( const InternalSampleType &iType )
				{
					ExternalSampleType xType;

					// Copy the position information
					std::get< 0 >( xType ) = std::get< 0 >( iType );
					// Copy the remainder of the information
					_SetNormalAndAuxData< 0 >( iType , xType );
					return xType;
				}
				static InternalSampleType ConvertX2I( const ExternalSampleType &xType )
				{
					InternalSampleType iType;

					// Copy the position information
					std::get< 0 >( iType ) = std::get< 0 >( xType );
					// Copy the remainder of the information
					_SetNormalAndAuxData< 0 >( xType , iType );
					return iType;
				}
			protected:
				template< unsigned int I >
				static void _SetNormalAndAuxData( const InternalSampleType &iType , ExternalSampleType &xType )
				{
					std::get< 1 >( xType ).template get< I >() = std::get< I+1 >( iType );
					if constexpr( I<sizeof...(AuxData) ) _SetNormalAndAuxData< I+1 >( iType , xType );
				}
				template< unsigned int I >
				static void _SetNormalAndAuxData( const ExternalSampleType &xType , InternalSampleType &iType )
				{
					std::get< I+1 >( iType ) = std::get< 1 >( xType ).template get< I >();
					if constexpr( I<sizeof...(AuxData) ) _SetNormalAndAuxData< I+1 >( xType , iType );
				}
			};

			friend Poisson;
			friend SSD;
		};


		struct Poisson
		{
			static const unsigned int NormalDegree = 2;							// The order of the B-Spline used to splat in the normals for constructing the Laplacian constraints
			static const unsigned int DefaultFEMDegree = 1;						// The default finite-element degree (has to be at least 1)
			static const BoundaryType DefaultFEMBoundary = BOUNDARY_NEUMANN;	// The default finite-element boundary type {BOUNDARY_FREE, BOUNDARY_DIRICHLET, BOUNDARY_NEUMANN}
			inline static const float WeightMultiplier = 2.f;					// The default degree-to-point-weight scaling

			template< unsigned int Dim , typename Real >
			struct ConstraintDual
			{
				Real target , weight;
				ConstraintDual( void ) : target(0) , weight(0) {}
				ConstraintDual( Real t , Real w ) : target(t) , weight(w) {}
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p ) const { return CumulativeDerivativeValues< Real , Dim , 0 >( target*weight ); };
			};

			template< unsigned int Dim , typename Real >
			struct SystemDual
			{
				Real weight;
				SystemDual( void ) : weight(0) {}
				SystemDual( Real w ) : weight(w) {}
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
				CumulativeDerivativeValues< double , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< double , Dim , 0 >& dValues ) const { return dValues * weight; };
			};

			template< unsigned int Dim >
			struct SystemDual< Dim , double >
			{
				typedef double Real;
				Real weight;
				SystemDual( void ) : weight(0) {}
				SystemDual( Real w ) : weight(w) {}
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
			};

			template< unsigned int Dim , typename Real >
			struct ValueInterpolationConstraintDual
			{
				typedef DirectSum< Real , Real > PointSampleData;
				Real vWeight;
				ValueInterpolationConstraintDual( Real v ) : vWeight(v){ }
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim > &p , const DirectSum< Real , Real >& data ) const 
				{
					Real value = data.template get<0>();
					CumulativeDerivativeValues< Real , Dim , 0 > cdv;
					cdv[0] = value*vWeight;
					return cdv;
				}
			};

			template< unsigned int Dim , typename Real >
			struct ValueInterpolationSystemDual
			{
				CumulativeDerivativeValues< Real , Dim , 0 > weight;
				ValueInterpolationSystemDual( Real v ){ weight[0] = v; }
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< Real , Real > &data , const CumulativeDerivativeValues< Real , Dim , 0 > &dValues ) const
				{
					return dValues * weight;
				}
				CumulativeDerivativeValues< double , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< Real , Real > &data , const CumulativeDerivativeValues< double , Dim , 0 > &dValues ) const
				{
					return dValues * weight;
				}
			};

			template< unsigned int Dim >
			struct ValueInterpolationSystemDual< Dim , double >
			{
				typedef double Real;
				Real weight;
				ValueInterpolationSystemDual( void ) : weight(0) {}
				ValueInterpolationSystemDual( Real v ) : weight(v) {}
				CumulativeDerivativeValues< Real , Dim , 0 > operator()( Point< Real , Dim > p , const DirectSum< Real , Real > &data , const CumulativeDerivativeValues< Real , Dim , 0 > &dValues ) const
				{
					return dValues * weight;
				}
			};

			template< typename Real >
			struct SolutionParameters : public Reconstructor::SolutionParameters< Real >
			{
				bool dirichletErode;
				Real pointWeight;
				Real valueInterpolationWeight;
				unsigned int envelopeDepth;

				SolutionParameters( void ) :
					dirichletErode(false) ,
					pointWeight((Real)( WeightMultiplier * DefaultFEMDegree ) ) , valueInterpolationWeight((Real)0.) ,
					envelopeDepth((unsigned int)-1)
				{}

				template< unsigned int Dim >
				void testAndSet( XForm< Real , Dim+1 > unitCubeToModel )
				{
					Reconstructor::SolutionParameters< Real >::template testAndSet< Dim >( unitCubeToModel );
					if( envelopeDepth==-1 ) envelopeDepth = Reconstructor::SolutionParameters< Real >::baseDepth;
					if( envelopeDepth>Reconstructor::SolutionParameters< Real >::depth )
					{
						if( envelopeDepth!=-1 ) MK_WARN( "Envelope depth cannot exceed system depth:  " , envelopeDepth , " <= " , Reconstructor::SolutionParameters< Real >::depth );
						envelopeDepth = Reconstructor::SolutionParameters< Real >::depth;
					}
					if( envelopeDepth<Reconstructor::SolutionParameters< Real >::baseDepth )
					{
						MK_WARN( "Envelope depth cannot be less than base depth: " , envelopeDepth , " >= " , Reconstructor::SolutionParameters< Real >::baseDepth );
						envelopeDepth = Reconstructor::SolutionParameters< Real >::baseDepth;
					}
				}
			};

			template< typename Real , unsigned int Dim >
			struct EnvelopeMesh
			{
				std::vector< Point< Real , Dim > > vertices;
				std::vector< SimplexIndex< Dim-1 , node_index_type > > simplices;
			};

			template< typename Real , unsigned int Dim , typename FEMSigPack , typename ... AuxData > struct Solver;

			template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
			struct Solver< Real , Dim , UIntPack< FEMSigs... > , AuxData... >
			{
				static Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *Solve( InputOrientedSampleStream< Real , Dim , AuxData... > &pointStream , SolutionParameters< Real > params , AuxData ... zero , const EnvelopeMesh< Real , Dim > *envelopeMesh=nullptr , InputValuedSampleStream< Real , Dim > *valueInterpolationStream=nullptr );
			};
		};

		struct SSD
		{
			static const unsigned int NormalDegree = 2;									// The order of the B-Spline used to splat in the normals for constructing the Laplacian constraints
			static const unsigned int DefaultFEMDegree = 2;								// The default finite-element degree (has to be at least 2)
			static const BoundaryType DefaultFEMBoundary = BOUNDARY_NEUMANN;			// The default finite-element boundary type {BOUNDARY_FREE, BOUNDARY_DIRICHLET, BOUNDARY_NEUMANN}
			inline static const double WeightMultipliers[] = { 5e+1f , 5e-4f , 1e-5f };	// The default weights for balancing the value, gradient, and laplacian energy terms

			template< typename Real , unsigned int Dim , typename ... AuxData >
			using NormalAndAuxData = DirectSum< Real , Normal< Real , Dim > , DirectSum< Real , AuxData... > >;

			template< unsigned int Dim , typename Real , typename ... AuxData >
			struct ConstraintDual
			{
				Real target , vWeight , gWeight;
				ConstraintDual( Real t , Real v , Real g ) : target(t) , vWeight(v) , gWeight(g) { }
				CumulativeDerivativeValues< Real , Dim , 1 > operator()( const Point< Real , Dim >& p , const NormalAndAuxData< Real , Dim , AuxData... > &normalAndAuxData ) const 
				{
					Point< Real , Dim > n = normalAndAuxData.template get<0>();
					CumulativeDerivativeValues< Real , Dim , 1 > cdv;
					cdv[0] = target*vWeight;
					for( int d=0 ; d<Dim ; d++ ) cdv[1+d] = -n[d]*gWeight;
					return cdv;
				}
			};

			template< unsigned int Dim , typename Real , typename ... AuxData >
			struct SystemDual
			{
				CumulativeDerivativeValues< Real , Dim , 1 > weight;
				SystemDual( Real v , Real g )
				{
					weight[0] = v;
					for( int d=0 ; d<Dim ; d++ ) weight[d+1] = g;
				}
				CumulativeDerivativeValues< Real , Dim , 1 > operator()( Point< Real , Dim > p , const NormalAndAuxData< Real , Dim , AuxData... > & , const CumulativeDerivativeValues< Real , Dim , 1 >& dValues ) const
				{
					return dValues * weight;
				}
				CumulativeDerivativeValues< double , Dim , 1 > operator()( Point< Real , Dim > p , const NormalAndAuxData< Real , Dim , AuxData... > & , const CumulativeDerivativeValues< double , Dim , 1 >& dValues ) const
				{
					return dValues * weight;
				};
			};

			template< unsigned int Dim , typename ... AuxData >
			struct SystemDual< Dim , double , AuxData... >
			{
				typedef double Real;
				CumulativeDerivativeValues< Real , Dim , 1 > weight;
				SystemDual( Real v , Real g ) : weight( v , g , g , g ) { }
				CumulativeDerivativeValues< Real , Dim , 1 > operator()( Point< Real , Dim > p , const NormalAndAuxData< Real , Dim , AuxData... > & , const CumulativeDerivativeValues< Real , Dim , 1 >& dValues ) const
				{
					return dValues * weight;
				}
			};

			template< typename Real >
			struct SolutionParameters : public Reconstructor::SolutionParameters< Real >
			{
				Real pointWeight;
				Real gradientWeight;
				Real biLapWeight;

				SolutionParameters( void ) :
					pointWeight((Real)WeightMultipliers[0]) , gradientWeight((Real)WeightMultipliers[1]) , biLapWeight((Real)WeightMultipliers[2])
				{}
			};

			template< typename Real , unsigned int Dim , typename FEMSigPack , typename ... AuxData > struct Solver;

			template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
			struct Solver< Real , Dim , UIntPack< FEMSigs... > , AuxData... >
			{
				static Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *Solve( InputOrientedSampleStream< Real , Dim , AuxData... > &pointStream , SolutionParameters< Real > params , AuxData ... zero );
			};
		};

		// Implementation of the base Implicit class's level-set extraction
		template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
		void Implicit< Real , Dim , UIntPack< FEMSigs ... > , AuxData ... >::extractLevelSet( OutputLevelSetVertexStream< Real , Dim , AuxData... > &vertexStream , OutputFaceStream< Dim-1 > &faceStream , LevelSetExtractionParameters params ) const
		{
			XForm< Real , Dim+1 > unitCubeToModel;
			if( params.gridCoordinates )
			{
				unitCubeToModel = XForm< Real , Dim+1 >::Identity();
				unsigned int res = 1<<tree.depth();
				for( unsigned int d=0 ; d<Dim ; d++ ) unitCubeToModel(d,d) = (Real)res;
			}
			else unitCubeToModel = this->unitCubeToModel;

			DensityEstimator *density = params.outputDensity ? this->density : nullptr;

			if constexpr( Dim==2 || Dim==3 )
			{
				Profiler profiler( ProfilerMS );

				std::string statsString;

				TransformedOutputLevelSetVertexStream< Real , Dim , AuxData... > _vertexStream( unitCubeToModel , vertexStream );
				if constexpr( sizeof...(AuxData) )
				{
					// Convert stream:
					// OutputDataStream< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , AuxData... >
					// -> OutputDataStream< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , InternalAuxData >
					OutputDataStreamConverter< typename _VertexTypeConverter::InternalVertexType , typename _VertexTypeConverter::ExternalVertexType > __vertexStream( _vertexStream , _VertexTypeConverter::ConvertX2I );
					typename LevelSetExtractor< Real , Dim , InternalAuxData >::Stats stats;
					if constexpr( Dim==3 )
						stats = LevelSetExtractor< Real , Dim , InternalAuxData >::Extract( Sigs() , UIntPack< Reconstructor::WeightDegree >() , UIntPack< DataSig >() , tree , density , _auxData , solution , isoValue , __vertexStream , faceStream , _zeroAuxData , !params.linearFit , params.outputGradients , params.forceManifold , params.polygonMesh , false );
					else if constexpr( Dim==2 )
						stats = LevelSetExtractor< Real , Dim , InternalAuxData >::Extract( Sigs() , UIntPack< Reconstructor::WeightDegree >() , UIntPack< DataSig >() , tree , density , _auxData , solution , isoValue , __vertexStream , faceStream , _zeroAuxData , !params.linearFit , params.outputGradients , false );
					statsString = stats.toString();
				}
				else
				{
					typename LevelSetExtractor< Real , Dim >::Stats stats;
					if constexpr( Dim==3 )
						stats = LevelSetExtractor< Real , Dim >::Extract( Sigs() , UIntPack< Reconstructor::WeightDegree >() , tree , density , solution , isoValue , _vertexStream , faceStream , !params.linearFit , params.outputGradients , params.forceManifold , params.polygonMesh , false );
					else if constexpr( Dim==2 )
						stats = LevelSetExtractor< Real , Dim >::Extract( Sigs() , UIntPack< Reconstructor::WeightDegree >() , tree , density , solution , isoValue , _vertexStream , faceStream , !params.linearFit , params.outputGradients , false );
					statsString = stats.toString();
				}

				if( params.verbose )
				{
					std::cout << "Vertices / Faces: " << vertexStream.size() << " / " << faceStream.size() << std::endl;
					std::cout << statsString << std::endl;
					std::cout << "#            Got Faces: " << profiler << std::endl;
				}
			}
			else MK_WARN( "Extraction only supported for dimensions 2 and 3" );
		}

		// Implementation of the derived Poisson::Implicit's constructor
		template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
		Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *Poisson::Solver< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::Solve( InputOrientedSampleStream< Real , Dim , AuxData... > &pointStream , SolutionParameters< Real > params , AuxData ... zero , const EnvelopeMesh< Real , Dim > *envelopeMesh , InputValuedSampleStream< Real , Dim > *valueInterpolationStream )
		{
			Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *implicitPtr = new Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >( zero... );
			Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > &implicit = *implicitPtr;

			if( params.valueInterpolationWeight<0 )
			{
				MK_WARN( "Negative value interpolation weight clamped to zero" );
				params.valueInterpolationWeight = 0;
			}
			if( valueInterpolationStream && !params.valueInterpolationWeight ) MK_WARN( "Value interpolation stream provided but interpolation weight is zero" );

			// The signature for the finite-elements representing the auxiliary data (if it's there)
			static const unsigned int DataSig = FEMDegreeAndBType< Reconstructor::DataDegree , BOUNDARY_FREE >::Signature;

			///////////////
			// Types --> //
			using InternalAuxData          = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalAuxData;
			using InternalNormalAndAuxData = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalNormalAndAuxData;
			using _SampleTypeConverter     = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::_SampleTypeConverter;
			using Sigs                     = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::Sigs;

			// The degrees of the finite elements across the different axes
			typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;

			// The signature describing the normals elements
			typedef UIntPack< FEMDegreeAndBType< Poisson::NormalDegree , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > NormalSigs;

			// Type for tracking sample interpolation
			typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 0 > InterpolationInfo;

			// The finite-element tracking tree node
			typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;

			typedef typename FEMTreeInitializer< Dim , Real >::GeometryNodeType GeometryNodeType;
			// <-- Types //
			///////////////

			XForm< Real , Dim+1 > modelToUnitCube = XForm< Real , Dim+1 >::Identity();

			Profiler profiler( ProfilerMS );

			size_t pointCount;

			ProjectiveData< Point< Real , 2 > , Real > pointDepthAndWeight;
			std::vector< typename FEMTree< Dim , Real >::PointSample > *valueInterpolationSamples = nullptr;
			std::vector< Real > *valueInterpolationSampleData = nullptr;
			DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > geometryNodeDesignators;
			SparseNodeData< Point< Real , Dim > , NormalSigs > *normalInfo = nullptr;
			std::vector< typename FEMTree< Dim , Real >::PointSample > *samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
			std::vector< InternalNormalAndAuxData > *sampleNormalAndAuxData = new std::vector< InternalNormalAndAuxData >();

			Real targetValue = (Real)0.5;

			// Read in the samples (and auxiliary data)
			{
				profiler.reset();

				pointStream.reset();
				modelToUnitCube = params.scale>0 ? PointExtent::GetXForm< Real , Dim , true , Normal< Real , Dim > , AuxData... >( pointStream , Normal< Real , Dim >() , zero... , params.scale , params.alignDir ) * modelToUnitCube : modelToUnitCube;
				implicit.unitCubeToModel = modelToUnitCube.inverse();
				pointStream.reset();

				params.template testAndSet< Dim >( implicit.unitCubeToModel );

				{
					// Apply the transformation
					TransformedInputOrientedSampleStream< Real , Dim , AuxData... > _pointStream( modelToUnitCube , pointStream );

					std::vector< node_index_type > nodeToIndexMap;

					auto IsValid = [&]( const Point< Real , Dim > &p , const Normal< Real , Dim > &n , AuxData ... d )
						{
							Real l = Point< Real , Dim >::SquareNorm( n );
							return l>0 && std::isfinite(l);
						};

					auto Process = [&]( FEMTreeNode &node , const Point< Real , Dim > &p , Normal< Real , Dim > &n , AuxData ... d )
						{
							Real l = (Real)Length( n );
							Real weight = params.confidence ? l : (Real)1.;
							n /= l;

							node_index_type nodeIndex = node.nodeData.nodeIndex;
							// If the node's index exceeds what's stored in the node-to-index map, grow the node-to-index map
							if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );

							node_index_type idx = nodeToIndexMap[ nodeIndex ];
							if( idx==-1 )
							{
								idx = (node_index_type)samples->size();
								nodeToIndexMap[ nodeIndex ] = idx;
								samples->resize( idx+1 ) , (*samples)[idx].node = &node;
								sampleNormalAndAuxData->resize( idx+1 );
								(*samples)[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
								(*sampleNormalAndAuxData)[idx] = InternalNormalAndAuxData( n , DirectSum< Real , AuxData... >( d... ) ) * weight;
							}
							else
							{
								(*samples)[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
								(*sampleNormalAndAuxData)[idx] += InternalNormalAndAuxData( n , DirectSum< Real , AuxData... >( d... ) ) * weight;
							}
							return true;
						};

					auto F = [&]( AuxData ... zeroAuxData )
						{
							pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< decltype(IsValid) , decltype(Process) , Normal< Real , Dim > , AuxData... >( implicit.tree.spaceRoot() , _pointStream , Normal< Real , Dim >() , zeroAuxData... , params.depth , implicit.tree.nodeAllocators.size() ? implicit.tree.nodeAllocators[0] : nullptr , implicit.tree.initializer() , IsValid , Process );
						};
					implicit._zeroAuxData.process( F );
				}


				if( params.verbose )
				{
					std::cout << "Input Points / Samples: " << pointCount << " / " << samples->size() << std::endl;
					std::cout << "# Read input into tree: " << profiler << std::endl;
				}
			}

			if( valueInterpolationStream && params.valueInterpolationWeight )
			{
				valueInterpolationSamples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
				valueInterpolationSampleData = new std::vector< Real >();
				// Wrap the point stream in a transforming stream
				TransformedInputValuedSampleStream< Real , Dim > _valueInterpolationStream( modelToUnitCube , *valueInterpolationStream );

				// Assign each sample a weight of 1.
				auto ProcessData = []( const Point< Real , Dim > &p , Real &d ){ return (Real)1.; };
				Real zeroValue = (Real)0.;
				size_t count = FEMTreeInitializer< Dim , Real >::template Initialize< Real >( implicit.tree.spaceRoot() , _valueInterpolationStream , zeroValue , params.depth , *valueInterpolationSamples , *valueInterpolationSampleData , implicit.tree.nodeAllocators.size() ? implicit.tree.nodeAllocators[0] : NULL , implicit.tree.initializer() , ProcessData );
				if( params.verbose ) std::cout << "Input Interpolation Points / Samples: " << count << " / " << valueInterpolationSamples->size() << std::endl;
			}

			{
				InterpolationInfo *valueInterpolationInfo = NULL;
				DenseNodeData< Real , Sigs > constraints;
				InterpolationInfo *iInfo = NULL;
				int solveDepth = params.depth;

				implicit.tree.resetNodeIndices( 0 , std::make_tuple() );

				// Get the kernel density estimator
				{
					profiler.reset();
					implicit.density = implicit.tree.template setDensityEstimator< 1 , Reconstructor::WeightDegree >( *samples , params.kernelDepth , params.samplesPerNode );
					if( params.verbose ) std::cout << "#   Got kernel density: " << profiler << std::endl;
				}

				// Transform the Hermite samples into a vector field
				{
					profiler.reset();
					normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();

					std::function< bool ( InternalNormalAndAuxData , Point< Real , Dim > & ) > ConversionFunction;
					std::function< bool ( InternalNormalAndAuxData , Point< Real , Dim > & , Real & ) > ConversionAndBiasFunction;
					ConversionFunction = []( InternalNormalAndAuxData in , Normal< Real , Dim > &out )
						{
							Normal< Real , Dim > n = in.template get<0>();
							Real l = (Real)Length( n );
							// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
							if( !l ) return false;
							out = n / l;
							return true;
						};
					*normalInfo = implicit.tree.setInterpolatedDataField( Point< Real , Dim >() , NormalSigs() , *samples , *sampleNormalAndAuxData , implicit.density , params.baseDepth , params.depth , params.lowDepthCutOff , pointDepthAndWeight , ConversionFunction );

					ThreadPool::ParallelFor( 0 , normalInfo->size() , [&]( unsigned int , size_t i ){ (*normalInfo)[i] *= (Real)-1.; } );
					if( params.verbose )
					{
						std::cout << "#     Got normal field: " << profiler << std::endl;
						std::cout << "Point depth / Point weight / Estimated measure: " << pointDepthAndWeight.value()[0] << " / " << pointDepthAndWeight.value()[1] << " / " << pointCount*pointDepthAndWeight.value()[1] << std::endl;
					}
				}

				// Get the geometry designators indicating if the space node are interior to, exterior to, or contain the envelope boundary
				if( envelopeMesh )
				{
					profiler.reset();
					{
						// Make the octree complete up to the base depth
						FEMTreeInitializer< Dim , Real >::Initialize( implicit.tree.spaceRoot() , params.baseDepth , []( int , int[] ){ return true; } , implicit.tree.nodeAllocators.size() ?  implicit.tree.nodeAllocators[0] : NULL , implicit.tree.initializer() );

						std::vector< Point< Real , Dim > > vertices( envelopeMesh->vertices.size() );
						for( unsigned int i=0 ; i<vertices.size() ; i++ ) vertices[i] = modelToUnitCube * envelopeMesh->vertices[i];
						geometryNodeDesignators = FEMTreeInitializer< Dim , Real >::GetGeometryNodeDesignators( &implicit.tree.spaceRoot() , vertices , envelopeMesh->simplices , params.baseDepth , params.envelopeDepth , implicit.tree.nodeAllocators , implicit.tree.initializer() );

						// Make nodes in the support of the vector field @{ExactDepth} interior
						if( params.dirichletErode )
						{
							// What to do if we find a node in the support of the vector field
							auto SetScratchFlag = [&]( FEMTreeNode *node )
								{
									if( node )
									{
										while( node->depth()>(int)params.baseDepth ) node = node->parent;
										node->nodeData.setScratchFlag( true );
									}
								};

							std::function< void ( FEMTreeNode * ) > PropagateToLeaves = [&]( const FEMTreeNode *node )
								{
									geometryNodeDesignators[ node ] = GeometryNodeType::INTERIOR;
									if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) PropagateToLeaves( node->children+c );
								};

							// Flags indicating if a node contains a non-zero vector field coefficient
							std::vector< bool > isVectorFieldElement( implicit.tree.nodeCount() , false );

							// Get the set of base nodes
							std::vector< FEMTreeNode * > baseNodes;
							auto nodeFunctor = [&]( FEMTreeNode *node )
								{
									if( node->depth()==params.baseDepth ) baseNodes.push_back( node );
									return node->depth()<(int)params.baseDepth;
								};
							implicit.tree.spaceRoot().processNodes( nodeFunctor );

							std::vector< node_index_type > vectorFieldElementCounts( baseNodes.size() );
							for( int i=0 ; i<vectorFieldElementCounts.size() ; i++ ) vectorFieldElementCounts[i] = 0;

							// In parallel, iterate over the base nodes and mark the nodes containing non-zero vector field coefficients
							ThreadPool::ParallelFor( 0 , baseNodes.size() , [&]( unsigned int t , size_t  i )
								{
									auto nodeFunctor = [&]( FEMTreeNode *node )
										{
											Point< Real , Dim > *n = (*normalInfo)( node );
											if( n && Point< Real , Dim >::SquareNorm( *n ) ) isVectorFieldElement[ node->nodeData.nodeIndex ] = true , vectorFieldElementCounts[i]++;
										};
									baseNodes[i]->processNodes( nodeFunctor );
								} );
							size_t vectorFieldElementCount = 0;
							for( int i=0 ; i<vectorFieldElementCounts.size() ; i++ ) vectorFieldElementCount += vectorFieldElementCounts[i];

							// Get the subset of nodes containing non-zero vector field coefficients and disable the "scratch" flag
							std::vector< FEMTreeNode * > vectorFieldElements;
							vectorFieldElements.reserve( vectorFieldElementCount );
							{
								std::vector< std::vector< FEMTreeNode * > > _vectorFieldElements( baseNodes.size() );
								for( int i=0 ; i<_vectorFieldElements.size() ; i++ ) _vectorFieldElements[i].reserve( vectorFieldElementCounts[i] );
								ThreadPool::ParallelFor( 0 , baseNodes.size() , [&]( unsigned int t , size_t  i )
									{
										auto nodeFunctor = [&]( FEMTreeNode *node )
											{
												if( isVectorFieldElement[ node->nodeData.nodeIndex ] ) _vectorFieldElements[i].push_back( node );
												node->nodeData.setScratchFlag( false );
											};
										baseNodes[i]->processNodes( nodeFunctor );
									} );
								for( int i=0 ; i<_vectorFieldElements.size() ; i++ ) vectorFieldElements.insert( vectorFieldElements.end() , _vectorFieldElements[i].begin() , _vectorFieldElements[i].end() );
							}

							// Set the scratch flag for the base nodes on which the vector field is supported
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] In principal, we should unlock finite elements whose support overlaps the vector field" )
#endif // SHOW_WARNINGS
							implicit.tree.template processNeighboringLeaves< -BSplineSupportSizes< Poisson::NormalDegree >::SupportStart , BSplineSupportSizes< Poisson::NormalDegree >::SupportEnd >( &vectorFieldElements[0] , vectorFieldElements.size() , SetScratchFlag , false );

							// Set sub-trees rooted at interior nodes @ ExactDepth to interior
							ThreadPool::ParallelFor( 0 , baseNodes.size() , [&]( unsigned int , size_t  i ){ if( baseNodes[i]->nodeData.getScratchFlag() ) PropagateToLeaves( baseNodes[i] ); } );

							// Adjust the coarser node designators in case exterior nodes have become boundary.
							ThreadPool::ParallelFor( 0 , baseNodes.size() , [&]( unsigned int , size_t  i ){ FEMTreeInitializer< Dim , Real >::PullGeometryNodeDesignatorsFromFiner( baseNodes[i] , geometryNodeDesignators ); } );
							FEMTreeInitializer< Dim , Real >::PullGeometryNodeDesignatorsFromFiner( &implicit.tree.spaceRoot() , geometryNodeDesignators , params.baseDepth );
						}
					}
					if( params.verbose ) std::cout << "#               Initialized envelope constraints: " << profiler << std::endl;
				}

				if constexpr( sizeof...(AuxData) )
				{
					profiler.reset();
					auto PointSampleFunctor = [&]( size_t i ) -> const typename FEMTree< Dim , Real >::PointSample & { return (*samples)[i]; };
					auto AuxDataSampleFunctor = [&]( size_t i ) -> const typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalAuxData & { return (*sampleNormalAndAuxData)[i].template get<1>(); };
					implicit._auxData = new SparseNodeData< ProjectiveData< InternalAuxData , Real > , IsotropicUIntPack< Dim , DataSig > >
						(
							implicit.tree.template setExtrapolatedDataField< DataSig , false , Reconstructor::WeightDegree , InternalAuxData >
							(
								InternalAuxData( zero... ) ,
								samples->size() ,
								PointSampleFunctor ,
								AuxDataSampleFunctor ,
								(typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::DensityEstimator*)nullptr
							)
						);
					auto nodeFunctor = [&]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *n )
						{
							ProjectiveData< InternalAuxData , Real >* clr = (*implicit._auxData)( n );
							if( clr ) (*clr) *= (Real)pow( (Real)params.perLevelDataScaleFactor , implicit.tree.depth( n ) );
						};
					implicit.tree.tree().processNodes( nodeFunctor );
					if( params.verbose ) std::cout << "#         Got aux data: " << profiler << std::endl;
				}

				delete sampleNormalAndAuxData;

				// Add the interpolation constraints
				if( params.pointWeight>0 )
				{
					profiler.reset();
					if( params.exactInterpolation ) iInfo = FEMTree< Dim , Real >::template       InitializeExactPointInterpolationInfo< Real , 0 > ( implicit.tree , *samples , Poisson::ConstraintDual< Dim , Real >( targetValue , params.pointWeight * pointDepthAndWeight.value()[1] ) , Poisson::SystemDual< Dim , Real >( params.pointWeight * pointDepthAndWeight.value()[1] ) , true , false );
					else                            iInfo = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( implicit.tree , *samples , Poisson::ConstraintDual< Dim , Real >( targetValue , params.pointWeight * pointDepthAndWeight.value()[1] ) , Poisson::SystemDual< Dim , Real >( params.pointWeight * pointDepthAndWeight.value()[1] ) , true , params.depth , 1 );
					if( params.verbose ) std::cout <<  "#Initialized point interpolation constraints: " << profiler << std::endl;
				}

				// Trim the tree and prepare for multigrid
				{
					profiler.reset();
					constexpr int MaxDegree = Poisson::NormalDegree > Degrees::Max() ? Poisson::NormalDegree : Degrees::Max();
					typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor( *normalInfo );
					auto hasDataFunctor = [&]( const FEMTreeNode *node ){ return hasNormalDataFunctor( node ); };
					auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)params.fullDepth; };
					if constexpr( sizeof...(AuxData) )
					{
						if( geometryNodeDesignators.size() ) implicit.tree.template finalizeForMultigridWithDirichlet< MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor , [&]( const FEMTreeNode *node ){ return node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node]==GeometryNodeType::EXTERIOR; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density , implicit._auxData , &geometryNodeDesignators ) );
						else                                 implicit.tree.template finalizeForMultigrid             < MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor ,                                                                                                                                                                                   std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density , implicit._auxData ) );
					}
					else
					{
						if( geometryNodeDesignators.size() ) implicit.tree.template finalizeForMultigridWithDirichlet< MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor , [&]( const FEMTreeNode *node ){ return node->nodeData.nodeIndex<(node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node]==GeometryNodeType::EXTERIOR; } , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density , &geometryNodeDesignators ) );
						else                                 implicit.tree.template finalizeForMultigrid             < MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor ,                                                                                                                                                                                   std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density ) );
					}

					if( params.verbose ) std::cout << "#       Finalized tree: " << profiler << std::endl;
				}

				// Add the FEM constraints
				{
					profiler.reset();
					constraints = implicit.tree.initDenseNodeData( Sigs() );

					// Add Poisson constraints
					{
						typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
						unsigned int derivatives2[Dim];
						for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
						typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
						typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
						for( int d=0 ; d<Dim ; d++ )
						{
							unsigned int derivatives1[Dim];
							for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ? 1 : 0;
							F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
						}
						implicit.tree.addFEMConstraints( F , *normalInfo , constraints , solveDepth );
					}
					if( params.verbose ) std::cout << "#  Set FEM constraints: " << profiler << std::endl;
				}

				// Free up the normal info
				delete normalInfo , normalInfo = NULL;

				if( params.pointWeight>0 )
				{
					profiler.reset();
					implicit.tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( iInfo ) );
					if( params.verbose ) std::cout << "#Set point constraints: " << profiler << std::endl;
				}

				if( valueInterpolationSamples && params.valueInterpolationWeight )
				{
					profiler.reset();
					if( params.exactInterpolation ) valueInterpolationInfo = FEMTree< Dim , Real >::template       InitializeExactPointAndDataInterpolationInfo< Real , Real , 0 > ( implicit.tree , *valueInterpolationSamples , GetPointer( *valueInterpolationSampleData ) , Poisson::ValueInterpolationConstraintDual< Dim , Real >( params.valueInterpolationWeight ) , Poisson::ValueInterpolationSystemDual< Dim , Real >( params.valueInterpolationWeight ) , true , false );
					else                            valueInterpolationInfo = FEMTree< Dim , Real >::template InitializeApproximatePointAndDataInterpolationInfo< Real , Real , 0 > ( implicit.tree , *valueInterpolationSamples , GetPointer( *valueInterpolationSampleData ) , Poisson::ValueInterpolationConstraintDual< Dim , Real >( params.valueInterpolationWeight ) , Poisson::ValueInterpolationSystemDual< Dim , Real >( params.valueInterpolationWeight ) , true , params.depth , 1 );
					delete valueInterpolationSamples , valueInterpolationSamples = NULL;
					delete valueInterpolationSampleData , valueInterpolationSampleData = NULL;

					implicit.tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( valueInterpolationInfo ) );
					if( params.verbose ) std::cout << "#Set value interpolation constraints: " << profiler << std::endl;
				}

				if( params.verbose ) std::cout << "All Nodes / Active Nodes / Ghost Nodes / Dirichlet Supported Nodes: " << implicit.tree.allNodes() << " / " << implicit.tree.activeNodes() << " / " << implicit.tree.ghostNodes() << " / " << implicit.tree.dirichletElements() << std::endl;
				if( params.verbose ) std::cout << "Memory Usage: " << float( MemoryInfo::Usage())/(1<<20) << " MB" << std::endl;

				// Solve the linear system
				{
					profiler.reset();
					typename FEMTree< Dim , Real >::SolverInfo _sInfo;
					_sInfo.cgDepth = 0 , _sInfo.cascadic = true , _sInfo.vCycles = 1 , _sInfo.iters = params.iters , _sInfo.cgAccuracy = params.cgSolverAccuracy , _sInfo.verbose = params.verbose , _sInfo.showResidual = params.showResidual , _sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , _sInfo.sliceBlockSize = 1;
					_sInfo.baseVCycles = params.baseVCycles;
					typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
					if( valueInterpolationInfo ) implicit.solution = implicit.tree.solveSystem( Sigs() , F , constraints , params.baseDepth , params.solveDepth , _sInfo , std::make_tuple( iInfo , valueInterpolationInfo ) );
					else                         implicit.solution = implicit.tree.solveSystem( Sigs() , F , constraints , params.baseDepth , params.solveDepth , _sInfo , std::make_tuple( iInfo ) );
					if( params.verbose ) std::cout << "# Linear system solved: " << profiler << std::endl;
					if( iInfo ) delete iInfo , iInfo = NULL;
					if( valueInterpolationInfo ) delete valueInterpolationInfo , valueInterpolationInfo = NULL;
				}
			}

			// Get the iso-value
			{
				profiler.reset();
				double valueSum = 0 , weightSum = 0;
				typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &implicit.tree , implicit.solution );
				std::vector< double > valueSums( ThreadPool::NumThreads() , 0 ) , weightSums( ThreadPool::NumThreads() , 0 );
				ThreadPool::ParallelFor( 0 , samples->size() , [&]( unsigned int thread , size_t j )
					{
						ProjectiveData< Point< Real , Dim > , Real >& sample = (*samples)[j].sample;
						Real w = sample.weight;
						if( w>0 ) weightSums[thread] += w , valueSums[thread] += evaluator.values( sample.data / sample.weight , thread , (*samples)[j].node )[0] * w;
					} );
				for( size_t t=0 ; t<valueSums.size() ; t++ ) valueSum += valueSums[t] , weightSum += weightSums[t];
				implicit.isoValue = (Real)( valueSum / weightSum );
				if( params.verbose )
				{
					std::cout << "Got average: " << profiler << std::endl;
					std::cout << "Iso-Value: " << implicit.isoValue << " = " << valueSum << " / " << weightSum << std::endl;
				}
			}
			delete samples;
			return implicitPtr;
		}

		// Implementation of the derived Poisson::Implicit's constructor
		template< typename Real , unsigned int Dim , unsigned int ... FEMSigs , typename ... AuxData >
		Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *SSD::Solver< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::Solve( InputOrientedSampleStream< Real , Dim , AuxData... > &pointStream , SolutionParameters< Real > params , AuxData ... zero )
		{
			Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > *implicitPtr = new Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >( zero... );
			Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... > &implicit = *implicitPtr;

			// The signature for the finite-elements representing the auxiliary data (if it's there)
			static const unsigned int DataSig = FEMDegreeAndBType< Reconstructor::DataDegree , BOUNDARY_FREE >::Signature;

			///////////////
			// Types --> //
			using InternalAuxData          = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalAuxData;
			using InternalNormalAndAuxData = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalNormalAndAuxData;
			using _SampleTypeConverter     = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::_SampleTypeConverter;
			using Sigs                     = typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::Sigs;

			// The degrees of the finite elements across the different axes
			typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;

			// The signature describing the normals elements
			typedef UIntPack< FEMDegreeAndBType< SSD::NormalDegree , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > NormalSigs;

			// Type for tracking sample interpolation
			typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 1 > InterpolationInfo;

			// The finite-element tracking tree node
			typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
			// <-- Types //
			///////////////

			XForm< Real , Dim+1 > modelToUnitCube = XForm< Real , Dim+1 >::Identity();

			Profiler profiler( ProfilerMS );

			size_t pointCount;

			ProjectiveData< Point< Real , 2 > , Real > pointDepthAndWeight;
			SparseNodeData< Point< Real , Dim > , NormalSigs > *normalInfo = nullptr;
			std::vector< typename FEMTree< Dim , Real >::PointSample > *samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
			std::vector< InternalNormalAndAuxData > *sampleNormalAndAuxData = new std::vector< InternalNormalAndAuxData >();;

			Real targetValue = (Real)0.0;

			// Read in the samples (and auxiliary data)
			{
				profiler.reset();

				pointStream.reset();
				modelToUnitCube = params.scale>0 ? PointExtent::GetXForm< Real , Dim , true , Normal< Real , Dim > , AuxData... >( pointStream , Normal< Real , Dim >() , zero... , params.scale , params.alignDir ) * modelToUnitCube : modelToUnitCube;
				implicit.unitCubeToModel = modelToUnitCube.inverse();
				pointStream.reset();

				params.template testAndSet< Dim >( implicit.unitCubeToModel );

				{
					// Apply the transformation
					TransformedInputOrientedSampleStream< Real , Dim , AuxData... > _pointStream( modelToUnitCube , pointStream );

					// Convert:
					// InputDataStream< Position< Real , Dim > , Normal< Real , Dim > , AuxData ... >
					//	 -> InputDataStream< Position< Real , Dim > , InternalNormalAndAuxData >
					InputDataStreamConverter< typename _SampleTypeConverter::InternalSampleType , typename _SampleTypeConverter::ExternalSampleType > __pointStream( _pointStream , _SampleTypeConverter::ConvertI2X , Position< Real , Dim >() , Normal< Real , Dim >() , zero... );

					std::vector< node_index_type > nodeToIndexMap;

					auto IsValid = [&]( const Point< Real , Dim > &p , const Normal< Real , Dim > &n , AuxData ... d )
						{
							Real l = Point< Real , Dim >::SquareNorm( n );
							return l>0 && std::isfinite(l);
						};

					auto Process = [&]( FEMTreeNode &node , const Point< Real , Dim > &p , Normal< Real , Dim > &n , AuxData ... d )
						{
							Real l = (Real)Length( n );
							Real weight = params.confidence ? l : (Real)1.;
							n /= l;

							node_index_type nodeIndex = node.nodeData.nodeIndex;
							// If the node's index exceeds what's stored in the node-to-index map, grow the node-to-index map
							if( nodeIndex>=(node_index_type)nodeToIndexMap.size() ) nodeToIndexMap.resize( nodeIndex+1 , -1 );

							node_index_type idx = nodeToIndexMap[ nodeIndex ];
							if( idx==-1 )
							{
								idx = (node_index_type)samples->size();
								nodeToIndexMap[ nodeIndex ] = idx;
								samples->resize( idx+1 ) , (*samples)[idx].node = &node;
								sampleNormalAndAuxData->resize( idx+1 );
								(*samples)[idx].sample = ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
								(*sampleNormalAndAuxData)[idx] = InternalNormalAndAuxData( n , DirectSum< Real , AuxData... >( d... ) ) * weight;
							}
							else
							{
								(*samples)[idx].sample += ProjectiveData< Point< Real , Dim > , Real >( p*weight , weight );
								(*sampleNormalAndAuxData)[idx] += InternalNormalAndAuxData( n , DirectSum< Real , AuxData... >( d... ) ) * weight;
							}
							return true;
						};
					auto F = [&]( AuxData ... zeroAuxData )
						{
							pointCount = FEMTreeInitializer< Dim , Real >::template Initialize< decltype(IsValid) , decltype(Process) , Normal< Real , Dim > , AuxData... >( implicit.tree.spaceRoot() , _pointStream , Normal< Real , Dim >() , zeroAuxData... , params.depth , implicit.tree.nodeAllocators.size() ? implicit.tree.nodeAllocators[0] : nullptr , implicit.tree.initializer() , IsValid , Process );
						};
					implicit._zeroAuxData.process( F );
				}

				if( params.verbose )
				{
					std::cout << "Input Points / Samples: " << pointCount << " / " << samples->size() << std::endl;
					std::cout << "# Read input into tree: " << profiler << std::endl;
				}
			}
			{
				DenseNodeData< Real , Sigs > constraints;
				InterpolationInfo *iInfo = NULL;
				int solveDepth = params.depth;

				implicit.tree.resetNodeIndices( 0 , std::make_tuple() );

				// Get the kernel density estimator
				{
					profiler.reset();
					implicit.density = implicit.tree.template setDensityEstimator< 1 , Reconstructor::WeightDegree >( *samples , params.kernelDepth , params.samplesPerNode );
					if( params.verbose ) std::cout << "#   Got kernel density: " << profiler << std::endl;
				}

				// Transform the Hermite samples into a vector field
				{
					profiler.reset();
					normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();

					std::function< bool ( InternalNormalAndAuxData , Point< Real , Dim > & ) > ConversionFunction;
					std::function< bool ( InternalNormalAndAuxData , Point< Real , Dim > & , Real & ) > ConversionAndBiasFunction;
					ConversionFunction = []( InternalNormalAndAuxData in , Normal< Real , Dim > &out )
						{
							Normal< Real , Dim > n = in.template get<0>();
							Real l = (Real)Length( n );
							// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
							if( !l ) return false;
							out = n / l;
							return true;
						};
					*normalInfo = implicit.tree.setInterpolatedDataField( Point< Real , Dim >() , NormalSigs() , *samples , *sampleNormalAndAuxData , implicit.density , params.baseDepth , params.depth , params.lowDepthCutOff , pointDepthAndWeight , ConversionFunction );

					if( params.verbose )
					{
						std::cout << "#     Got normal field: " << profiler << std::endl;
						std::cout << "Point depth / Point weight / Estimated measure: " << pointDepthAndWeight.value()[0] << " / " << pointDepthAndWeight.value()[1] << " / " << pointCount*pointDepthAndWeight.value()[1] << std::endl;
					}
				}

				if constexpr( sizeof...(AuxData) )
				{
					profiler.reset();
					auto PointSampleFunctor = [&]( size_t i ) -> const typename FEMTree< Dim , Real >::PointSample & { return (*samples)[i]; };
					auto AuxDataSampleFunctor = [&]( size_t i ) -> const typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::InternalAuxData & { return (*sampleNormalAndAuxData)[i].template get<1>(); };
					implicit._auxData = new SparseNodeData< ProjectiveData< InternalAuxData , Real > , IsotropicUIntPack< Dim , DataSig > >
						(
							implicit.tree.template setExtrapolatedDataField< DataSig , false , Reconstructor::WeightDegree , InternalAuxData >
							(
								InternalAuxData( zero... ) ,
								samples->size() ,
								PointSampleFunctor ,
								AuxDataSampleFunctor ,
								(typename Reconstructor::Implicit< Real , Dim , UIntPack< FEMSigs... > , AuxData... >::DensityEstimator*)nullptr
							)
						);
					auto nodeFunctor = [&]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *n )
						{
							ProjectiveData< InternalAuxData , Real >* clr = (*implicit._auxData)( n );
							if( clr ) (*clr) *= (Real)pow( (Real)params.perLevelDataScaleFactor , implicit.tree.depth( n ) );
						};
					implicit.tree.tree().processNodes( nodeFunctor );
					if( params.verbose ) std::cout << "#         Got aux data: " << profiler << std::endl;
				}
				// Add the interpolation constraints
				if( params.pointWeight>0 || params.gradientWeight>0 )
				{
					profiler.reset();
					if( params.exactInterpolation ) iInfo = FEMTree< Dim , Real >::template       InitializeExactPointAndDataInterpolationInfo< Real , InternalNormalAndAuxData , 1 >( implicit.tree , *samples , GetPointer( *sampleNormalAndAuxData ) , SSD::ConstraintDual< Dim , Real , AuxData... >( targetValue , params.pointWeight * pointDepthAndWeight.value()[1] , params.gradientWeight * pointDepthAndWeight.value()[1]  ) , SSD::SystemDual< Dim , Real , AuxData... >( params.pointWeight * pointDepthAndWeight.value()[1] , params.gradientWeight * pointDepthAndWeight.value()[1] ) , true , false );
					else                            iInfo = FEMTree< Dim , Real >::template InitializeApproximatePointAndDataInterpolationInfo< Real , InternalNormalAndAuxData , 1 >( implicit.tree , *samples , GetPointer( *sampleNormalAndAuxData ) , SSD::ConstraintDual< Dim , Real , AuxData... >( targetValue , params.pointWeight * pointDepthAndWeight.value()[1] , params.gradientWeight * pointDepthAndWeight.value()[1]  ) , SSD::SystemDual< Dim , Real , AuxData... >( params.pointWeight * pointDepthAndWeight.value()[1] , params.gradientWeight * pointDepthAndWeight.value()[1] ) , true , params.depth , 1 );
					if( params.verbose ) std::cout <<  "#Initialized point interpolation constraints: " << profiler << std::endl;
				}

				delete sampleNormalAndAuxData;


				// Trim the tree and prepare for multigrid
				{
					profiler.reset();
					constexpr int MaxDegree = SSD::NormalDegree > Degrees::Max() ? SSD::NormalDegree : Degrees::Max();
					typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor( *normalInfo );
					auto hasDataFunctor = [&]( const FEMTreeNode *node ){ return hasNormalDataFunctor( node ); };
					auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)params.fullDepth; };
					if constexpr( sizeof...(AuxData) ) implicit.tree.template finalizeForMultigrid< MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density , implicit._auxData ) );
					else implicit.tree.template finalizeForMultigrid< MaxDegree , Degrees::Max() >( params.baseDepth , addNodeFunctor , hasDataFunctor , std::make_tuple( iInfo ) , std::make_tuple( normalInfo , implicit.density ) );

					if( params.verbose ) std::cout << "#       Finalized tree: " << profiler << std::endl;
				}

				// Free up the normal info
				delete normalInfo , normalInfo = NULL;

				if( params.pointWeight>0 || params.gradientWeight>0 )
				{
					profiler.reset();
					constraints = implicit.tree.initDenseNodeData( Sigs() );
					implicit.tree.addInterpolationConstraints( constraints , solveDepth , std::make_tuple( iInfo ) );
					if( params.verbose ) std::cout << "#Set point constraints: " << profiler << std::endl;
				}

				if( params.verbose ) std::cout << "All Nodes / Active Nodes / Ghost Nodes: " << implicit.tree.allNodes() << " / " << implicit.tree.activeNodes() << " / " << implicit.tree.ghostNodes() << std::endl;
				if( params.verbose ) std::cout << "Memory Usage: " << float( MemoryInfo::Usage())/(1<<20) << " MB" << std::endl;

				// Solve the linear system
				{
					profiler.reset();
					typename FEMTree< Dim , Real >::SolverInfo _sInfo;
					_sInfo.cgDepth = 0 , _sInfo.cascadic = true , _sInfo.vCycles = 1 , _sInfo.iters = params.iters , _sInfo.cgAccuracy = params.cgSolverAccuracy , _sInfo.verbose = params.verbose , _sInfo.showResidual = params.showResidual , _sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , _sInfo.sliceBlockSize = 1;
					_sInfo.baseVCycles = params.baseVCycles;
					typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 2 > > F( { 0. , 0. , (double)params.biLapWeight } );
					implicit.solution = implicit.tree.solveSystem( Sigs() , F , constraints , params.baseDepth , params.solveDepth , _sInfo , std::make_tuple( iInfo ) );
					if( params.verbose ) std::cout << "# Linear system solved: " << profiler << std::endl;
					if( iInfo ) delete iInfo , iInfo = NULL;
				}
			}

			// Get the iso-value
			{
				profiler.reset();
				double valueSum = 0 , weightSum = 0;
				typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &implicit.tree , implicit.solution );
				std::vector< double > valueSums( ThreadPool::NumThreads() , 0 ) , weightSums( ThreadPool::NumThreads() , 0 );
				ThreadPool::ParallelFor( 0 , samples->size() , [&]( unsigned int thread , size_t j )
					{
						ProjectiveData< Point< Real , Dim > , Real >& sample = (*samples)[j].sample;
						Real w = sample.weight;
						if( w>0 ) weightSums[thread] += w , valueSums[thread] += evaluator.values( sample.data / sample.weight , thread , (*samples)[j].node )[0] * w;
					} );
				for( size_t t=0 ; t<valueSums.size() ; t++ ) valueSum += valueSums[t] , weightSum += weightSums[t];
				implicit.isoValue = (Real)( valueSum / weightSum );
				if( params.verbose )
				{
					std::cout << "Got average: " << profiler << std::endl;
					std::cout << "Iso-Value: " << implicit.isoValue << " = " << valueSum << " / " << weightSum << std::endl;
				}
			}
			delete samples;
			return implicitPtr;
		}
	}
}


#endif // RECONSTRUCTORS_INCLUDED