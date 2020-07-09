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

#ifndef BSPLINE_DATA_INCLUDED
#define BSPLINE_DATA_INCLUDED

#define NEW_BSPLINE_CODE 1

#include "BinaryNode.h"
#include "PPolynomial.h"
#include "Array.h"

// This class represents a function that is a linear combination of B-spline elements,
// with the coeff member indicating how much of each element is present.
// [WARNING] The ordering of B-spline elements is in the opposite order from that returned by Polynomial::BSplineComponent
template< int Degree >
struct BSplineElementCoefficients
{
	int coeffs[Degree+1];
	BSplineElementCoefficients( void ){ memset( coeffs , 0 , sizeof( int ) * ( Degree+1 ) ); }
	int& operator[]( int idx ){ return coeffs[idx]; }
	const int& operator[]( int idx ) const { return coeffs[idx]; }
};

// This class represents a function on the the interval, partitioned into "res" blocks.
// On each block, the function is a degree-Degree polynomial, represented by the coefficients
// in the associated BSplineElementCoefficients.
// [NOTE] This representation of a function is agnostic to the type of boundary conditions (though the constructor is not).
template< int Degree >
struct BSplineElements : public std::vector< BSplineElementCoefficients< Degree > >
{
	static const bool _Primal = (Degree&1)==1;
	static const int _Off = (Degree+1)/2;
	static int _ReflectLeft ( int offset , int res );
	static int _ReflectRight( int offset , int res );
	static int _RotateLeft  ( int offset , int res );
	static int _RotateRight ( int offset , int res );
	template< bool Left > void _addPeriodic( int offset , bool negate );
public:
	// Coefficients are ordered as "/" "-" "\"
	// [WARNING] This is the opposite of the order in Polynomial::BSplineComponent
	int denominator;

	BSplineElements( void ) { denominator = 1; }
	BSplineElements( int res , int offset , bool dirichlet );

	void upSample( BSplineElements& high ) const;
	void differentiate( BSplineElements< Degree-1 >& d ) const;

	void print( FILE* fp=stdout ) const
	{
		for( int i=0 ; i<std::vector< BSplineElementCoefficients< Degree > >::size() ; i++ )
		{
			printf( "%d]" , i );
			for( int j=0 ; j<=Degree ; j++ ) printf( " %d" , (*this)[i][j] );
			printf( " (%d)\n" , denominator );
		}
	}
};
#define BSPLINE_SET_BOUNDS( name , s , e ) \
	static const int name ## Start = (s); \
	static const int name ## End   = (e); \
	static const int name ## Size  = (e)-(s)+1

// Assumes that x is non-negative
#define _FLOOR_OF_HALF( x ) (   (x)    >>1 )
#define  _CEIL_OF_HALF( x ) ( ( (x)+1 )>>1 )
// Done with the assumption
#define FLOOR_OF_HALF( x ) ( (x)<0 ? -  _CEIL_OF_HALF( -(x) ) : _FLOOR_OF_HALF( x ) )
#define  CEIL_OF_HALF( x ) ( (x)<0 ? - _FLOOR_OF_HALF( -(x) ) :  _CEIL_OF_HALF( x ) )
#define SMALLEST_INTEGER_LARGER_THAN_HALF( x ) (  CEIL_OF_HALF( (x)+1 ) )
#define LARGEST_INTEGER_SMALLER_THAN_HALF( x ) ( FLOOR_OF_HALF( (x)-1 ) )
#define SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( x ) (  CEIL_OF_HALF( x ) )
#define LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( x ) ( FLOOR_OF_HALF( x ) )

template< int Degree >
class BSplineEvaluationData
{
public:
	BSplineEvaluationData( void );
	static double Value( int depth , int off , double s , bool dirichlet , bool derivative );

	static int Dimension( int depth ){ return ( 1<<depth ) + ( Degree&1 ); }
	// An index is interiorly supported if its support is in the range [0,1<<depth)
	inline static void InteriorSupportedSpan( int depth , int& begin , int& end ){ begin = -SupportStart , end = (1<<depth)-SupportEnd; }

	// If the degree is even, we use a dual basis and functions are centered at the center of the interval
	// It the degree is odd, we use a primal basis and functions are centered at the left end of the interval
	// The function at index I is supported in:
	//	Support( I ) = [ I - (Degree+1-Inset)/2 , I + (Degree+1+Inset)/2 ]
	// [NOTE] The value of ( Degree + 1 +/- Inset ) is always even
	static const int Inset = (Degree&1) ? 0 : 1;
	BSPLINE_SET_BOUNDS(      Support , -( (Degree+1)/2 ) , Degree/2           );
	BSPLINE_SET_BOUNDS( ChildSupport ,    2*SupportStart , 2*(SupportEnd+1)-1 );
	BSPLINE_SET_BOUNDS(       Corner ,    SupportStart+1 , SupportEnd         );
	BSPLINE_SET_BOUNDS(  ChildCorner ,  2*SupportStart+1 , 2*SupportEnd + 1   );

	// Setting I=0, we are looking for the smallest/largest integers J such that:
	//		Support( 0 ) CONTAINS Support( J )
	// <=>	[-(Degree+1-Inset) , (Degree+1+Inset) ] CONTAINS [ J-(Degree+1-Inset)/2 , J+(Degree+1+Inset)/2 ]
	// Which is the same as the smallest/largest integers J such that:
	//		J - (Degree+1-Inset)/2 >= -(Degree+1-Inset)	| J + (Degree+1+Inset)/2 <= (Degree+1+Inset)
	// <=>	J >= -(Degree+1-Inset)/2					| J <= (Degree+1+Inset)/2
	BSPLINE_SET_BOUNDS( UpSample , - ( Degree + 1 - Inset ) / 2 , ( Degree + 1 + Inset ) /2 );

	// Setting I=0/1, we are looking for the smallest/largest integers J such that:
	//		Support( J ) CONTAINS Support( 0/1 )
	// <=>	[ 2*J - (Degree+1-Inset) , 2*J + (Degree+1+Inset) ] CONTAINS [ 0/1 - (Degree+1-Inset)/2 , 0/1 + (Degree+1+Inset)/2 ]
	// Which is the same as the smallest/largest integers J such that:
	//		2*J + (Degree+1+Inset) >= 0/1 + (Degree+1+Inset)/2	| 2*J - (Degree+1-Inset) <= 0/1 - (Degree+1-Inset)/2
	// <=>	2*J >= 0/1 - (Degree+1+Inset)/2						| 2*J <= 0/1 + (Degree+1-Inset)/2
	BSPLINE_SET_BOUNDS( DownSample0 , SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( 0 - ( Degree + 1 + Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( 0 + ( Degree + 1 - Inset ) / 2 ) );
	BSPLINE_SET_BOUNDS( DownSample1 , SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF( 1 - ( Degree + 1 + Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF( 1 + ( Degree + 1 - Inset ) / 2 ) );
	static const int DownSampleStart[] , DownSampleEnd[] , DownSampleSize[];

	// Note that this struct stores the components in left-to-right order
	struct BSplineComponents
	{
	protected:
		Polynomial< Degree > _polys[Degree+1];
	public:
		BSplineComponents( void ){ ; }
		BSplineComponents( int depth , int offset , bool dirichlet );
		const Polynomial< Degree >& operator[] ( int idx ) const { return _polys[idx]; }
		void printnl( void ) const { for( int d=0 ; d<=Degree ; d++ ) printf( "[%d] " , d ) , _polys[d].printnl(); }
	};
	struct BSplineUpSamplingCoefficients
	{
	protected:
		int _coefficients[ UpSampleSize ];
	public:
		BSplineUpSamplingCoefficients( void ){ ; }
		BSplineUpSamplingCoefficients( int depth , int offset , bool dirichlet );
		double operator[] ( int idx ){ return (double)_coefficients[idx] / (1<<Degree); }
	};

	struct CenterEvaluator
	{
		static const int Start = -SupportStart , Stop = SupportEnd , Size = Start + Stop + 1;

		static const int Index( int depth , int offset  )
		{
			int dim = BSplineEvaluationData< Degree >::Dimension( depth );
			if     ( offset<Start )     return offset;
			else if( offset>=dim-Stop ) return Start + 1 + offset - ( dim-Stop );
			else                        return Start;
		}
		struct Evaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _depth;
			double _ccValues[2][Size][SupportSize];
		public:
			double value( int fIdx , int cIdx , bool d ) const;
			int depth( void ) const { return _depth; }
		};
		struct ChildEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _parentDepth;
			double _pcValues[2][Size][ChildSupportSize];
		public:
			double value( int fIdx , int cIdx , bool d ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
		};
	};
	static void SetCenterEvaluator( typename CenterEvaluator::Evaluator& evaluator , int depth , bool dirichlet );
	static void SetChildCenterEvaluator( typename CenterEvaluator::ChildEvaluator& evaluator , int parentDepth , bool dirichlet );

	struct CornerEvaluator
	{
		static const int Start = -SupportStart , Stop = SupportEnd , Size = Start + Stop + 1;

		static const int Index( int depth , int offset  )
		{
			int dim = BSplineEvaluationData< Degree >::Dimension( depth );
			if     ( offset<Start )     return offset;
			else if( offset>=dim-Stop ) return Start + 1 + offset - ( dim-Stop );
			else                        return Start;
		}
		struct Evaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _depth;
			double _ccValues[2][Size][CornerSize];
		public:
			double value( int fIdx , int cIdx , bool d ) const;
			int depth( void ) const { return _depth; }
		};
		struct ChildEvaluator
		{
		protected:
			friend BSplineEvaluationData;
			int _parentDepth;
			double _pcValues[2][Size][ChildCornerSize];
		public:
			double value( int fIdx , int cIdx , bool d ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
		};
	};
	static void SetCornerEvaluator( typename CornerEvaluator::Evaluator& evaluator , int depth , bool dirichlet );
	static void SetChildCornerEvaluator( typename CornerEvaluator::ChildEvaluator& evaluator , int parentDepth , bool dirichlet );

	struct Evaluator
	{
		typename CenterEvaluator::Evaluator centerEvaluator;
		typename CornerEvaluator::Evaluator cornerEvaluator;
		double centerValue( int fIdx , int cIdx , bool d ) const { return centerEvaluator.value( fIdx , cIdx , d ); }
		double cornerValue( int fIdx , int cIdx , bool d ) const { return cornerEvaluator.value( fIdx , cIdx , d ); }
	};
	static void SetEvaluator( Evaluator& evaluator , int depth , bool dirichlet ){ SetCenterEvaluator( evaluator.centerEvaluator , depth , dirichlet ) , SetCornerEvaluator( evaluator.cornerEvaluator , depth , dirichlet ); }
	struct ChildEvaluator
	{
		typename CenterEvaluator::ChildEvaluator centerEvaluator;
		typename CornerEvaluator::ChildEvaluator cornerEvaluator;
		double centerValue( int fIdx , int cIdx , bool d ) const { return centerEvaluator.value( fIdx , cIdx , d ); }
		double cornerValue( int fIdx , int cIdx , bool d ) const { return cornerEvaluator.value( fIdx , cIdx , d ); }
	};
	static void SetChildEvaluator( ChildEvaluator& evaluator , int depth , bool dirichlet ){ SetChildCenterEvaluator( evaluator.centerEvaluator , depth , dirichlet ) , SetChildCornerEvaluator( evaluator.cornerEvaluator , depth , dirichlet ); }

	struct UpSampleEvaluator
	{
		static const int Start = - SupportStart , Stop = SupportEnd , Size = Start + Stop + 1;
		static const int Index( int depth , int offset  )
		{
			int dim = BSplineEvaluationData< Degree >::Dimension( depth );
			if     ( offset<Start )     return offset;
			else if( offset>=dim-Stop ) return Start + 1 + offset - ( dim-Stop );
			else                        return Start;
		}
	protected:
		friend BSplineEvaluationData;
		int _lowDepth;
		double _pcValues[Size][UpSampleSize];
	public:
		double value( int pIdx , int cIdx ) const;
		int lowDepth( void ) const { return _lowDepth; }
	};
	static void SetUpSampleEvaluator( UpSampleEvaluator& evaluator , int lowDepth , bool dirichlet );
};
template< int Degree > const int BSplineEvaluationData< Degree >::DownSampleStart[] = { DownSample0Start , DownSample1Start };
template< int Degree > const int BSplineEvaluationData< Degree >::DownSampleEnd  [] = { DownSample0End   , DownSample1End   };
template< int Degree > const int BSplineEvaluationData< Degree >::DownSampleSize [] = { DownSample0Size  , DownSample1Size  };

template< int Degree1 , int Degree2 >
class BSplineIntegrationData
{
public:
	static double Dot( int depth1 , int off1 , bool dirichlet1 , bool d1 , int depth2 , int off2 , bool dirichlet2 , bool d2 );
	// An index is interiorly overlapped if the support of its overlapping neighbors is in the range [0,1<<depth)
	inline static void InteriorOverlappedSpan( int depth , int& begin , int& end ){ begin = -OverlapStart-BSplineEvaluationData< Degree2 >::SupportStart , end = (1<<depth)-OverlapEnd-BSplineEvaluationData< Degree2 >::SupportEnd; }

	typedef BSplineEvaluationData< Degree1 > EData1;
	typedef BSplineEvaluationData< Degree2 > EData2;
	BSPLINE_SET_BOUNDS(             Overlap , EData1::     SupportStart - EData2::SupportEnd , EData1::     SupportEnd - EData2::SupportStart );
	BSPLINE_SET_BOUNDS(        ChildOverlap , EData1::ChildSupportStart - EData2::SupportEnd , EData1::ChildSupportEnd - EData2::SupportStart );
	BSPLINE_SET_BOUNDS(      OverlapSupport ,      OverlapStart + EData2::SupportStart ,      OverlapEnd + EData2::SupportEnd );
	BSPLINE_SET_BOUNDS( ChildOverlapSupport , ChildOverlapStart + EData2::SupportStart , ChildOverlapEnd + EData2::SupportEnd );

	// Setting I=0/1, we are looking for the smallest/largest integers J such that:
	//		Support( 2*J ) * 2 INTERSECTION Support( 0/1 ) NON-EMPTY
	// <=>	[ 2*J - (Degree2+1-Inset2) , 2*J + (Degree2+1+Inset2) ] INTERSECTION [ 0/1 - (Degree1+1-Inset1)/2 , 0/1 + (Degree1+1+Inset1)/2 ] NON-EMPTY
	// Which is the same as the smallest/largest integers J such that:
	//		0/1 - (Degree1+1-Inset1)/2 < 2*J + (Degree2+1+Inset2)			| 0/1 + (Degree1+1+Inset1)/2 > 2*J - (Degree2+1-Inset2)	
	// <=>	2*J > 0/1 - ( 2*Degree2 + Degree1 + 3 + 2*Inset2 - Inset1 ) / 2	| 2*J < 0/1 + ( 2*Degree2 + Degree1 + 3 - 2*Inset2 + Inset1 ) / 2
	BSPLINE_SET_BOUNDS( ParentOverlap0 , SMALLEST_INTEGER_LARGER_THAN_HALF( 0 - ( 2*Degree2 + Degree1 + 3 + 2*EData2::Inset - EData1::Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_HALF( 0 + ( 2*Degree2 + Degree1 + 3 - 2*EData2::Inset + EData1::Inset ) / 2 ) );
	BSPLINE_SET_BOUNDS( ParentOverlap1 , SMALLEST_INTEGER_LARGER_THAN_HALF( 1 - ( 2*Degree2 + Degree1 + 3 + 2*EData2::Inset - EData1::Inset ) / 2 ) , LARGEST_INTEGER_SMALLER_THAN_HALF( 1 + ( 2*Degree2 + Degree1 + 3 - 2*EData2::Inset + EData1::Inset ) / 2 ) );
	static const int ParentOverlapStart[] , ParentOverlapEnd[] , ParentOverlapSize[];

	struct FunctionIntegrator
	{
		static const int Start = - OverlapSupportStart , Stop = OverlapSupportEnd , Size = Start + Stop + 1;
		static const int Index( int depth , int offset  )
		{
			int dim = BSplineEvaluationData< Degree2 >::Dimension( depth );
			if     ( offset<Start )     return offset;
			else if( offset>=dim-Stop ) return Start + 1 + offset - ( dim-Stop );
			else                        return Start;
		}
		struct Integrator
		{
		protected:
			friend BSplineIntegrationData;
			int _depth;
			double _ccIntegrals[2][2][Size][OverlapSize];
		public:
			double dot( int fIdx1 , int fidx2 , bool d1 , bool d2 ) const;
			int depth( void ) const { return _depth; }
		};
		struct ChildIntegrator
		{
		protected:
			friend BSplineIntegrationData;
			int _parentDepth;
			double _pcIntegrals[2][2][Size][ChildOverlapSize];
		public:
			double dot( int fIdx1 , int fidx2 , bool d1 , bool d2 ) const;
			int parentDepth( void ) const { return _parentDepth; }
			int childDepth( void ) const { return _parentDepth+1; }
		};
	};
	static void SetIntegrator( typename FunctionIntegrator::Integrator& integrator , int depth , bool dirichlet1 , bool dirichlet2 );
	static void SetChildIntegrator( typename FunctionIntegrator::ChildIntegrator& integrator , int parentDepth , bool dirichlet1 , bool dirichlet2 );
};
template< int Degree1 , int Degree2 > const int BSplineIntegrationData< Degree1 , Degree2 >::ParentOverlapStart[] = { ParentOverlap0Start , ParentOverlap1Start };
template< int Degree1 , int Degree2 > const int BSplineIntegrationData< Degree1 , Degree2 >::ParentOverlapEnd  [] = { ParentOverlap0End   , ParentOverlap1End   };
template< int Degree1 , int Degree2 > const int BSplineIntegrationData< Degree1 , Degree2 >::ParentOverlapSize [] = { ParentOverlap0Size  , ParentOverlap1Size  };
#undef BSPLINE_SET_BOUNDS
#undef _FLOOR_OF_HALF
#undef  _CEIL_OF_HALF
#undef FLOOR_OF_HALF
#undef  CEIL_OF_HALF
#undef SMALLEST_INTEGER_LARGER_THAN_HALF
#undef LARGEST_INTEGER_SMALLER_THAN_HALF
#undef SMALLEST_INTEGER_LARGER_THAN_OR_EQUAL_TO_HALF
#undef LARGEST_INTEGER_SMALLER_THAN_OR_EQUAL_TO_HALF

template< int Degree >
class BSplineData
{
	bool _dirichlet;
public:

	inline static int Centers  ( int depth ){ return (1<<depth); }
	inline static int Corners  ( int depth ){ return (1<<depth) + 1; }
	inline static int Dimension( int depth ){ return (1<<depth) + (Degree&1); }
	inline static int FunctionIndex( int depth , int offset ){ return (Degree&1) ? BinaryNode::CornerIndex( depth , offset ) : BinaryNode::CenterIndex( depth , offset ); }
	inline static void FactorFunctionIndex( int idx , int& depth , int& offset ){ return (Degree&1) ? BinaryNode::CornerDepthAndOffset( idx , depth , offset ) : BinaryNode::CenterDepthAndOffset( idx , depth , offset ); }
	inline static int TotalFunctionCount( int depth ){ return (Degree&1) ? BinaryNode::CumulativeCornerCount( depth ) : BinaryNode::CumulativeCenterCount( depth ); }
	inline static int TotalSampleCount( int depth ){ return BinaryNode::CenterCount( depth ) + BinaryNode::CornerCount( depth ); }
	inline static void FunctionSpan( int depth , int& fStart , int& fEnd ){ fStart = (depth>0) ? TotalFunctionCount(depth-1) : 0 , fEnd = TotalFunctionCount(depth); }
	inline static void SampleSpan( int depth , int& sStart , int& sEnd ){ sStart = (depth>0) ? TotalSampleCount(depth-1) : 0 , sEnd = TotalSampleCount(depth); }

	inline static int RemapOffset( int depth , int idx , bool& reflect );

	int depth;
	size_t functionCount , sampleCount;
	Pointer( typename BSplineEvaluationData< Degree >::BSplineComponents ) baseBSplines;

	BSplineData( void );

	void set( int maxDepth , bool dirichlet=false );
};

template< int Degree1 , int Degree2 > void SetBSplineElementIntegrals( double integrals[Degree1+1][Degree2+1] );


#include "BSplineData.inl"
#endif // BSPLINE_DATA_INCLUDED