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

#ifndef PARAMETER_PACK_INCLUDED
#define PARAMETER_PACK_INCLUDED

/////////////////////
// parameter packs //
/////////////////////

namespace ParameterPack
{
	////////////////////////////
	////////////////////////////
	//// Short declarations ////
	////////////////////////////
	////////////////////////////

	// A wrapper class for passing unsigned integer parameter packs
	template< typename IntegralType , IntegralType ... Values > struct Pack;

	// A class that identifies a single entry and its complement
	template< unsigned int I , typename Pack > struct Selection;

	// A class that splits a Pack into two sub-packs
	template< unsigned int I , typename Pack > struct Partition;

	// A class for comparing two Packs
	template< typename Pack1 , typename Pack2 > struct Comparison;

	// A class for adding/subtracting two Packs
	template< typename Pack1 , typename Pack2 > struct Arithmetic;

	// A helper class for defining a concatenation of multiple Packs
	template< typename ... Packs > struct _Concatenation;

	// A helper class for defining the permutation of a Pack
	template< typename Pack , typename PermutationPack > struct _Permutation;

	// A helper class for defining a Pack with the same value repeated Dim times
	template< typename IntegralType , unsigned int Dim , IntegralType Value > struct _IsotropicPack;

	// A helper class for defining a Pack with sequentially increasing values
	template< typename IntegralType , unsigned int Dim , IntegralType StartValue > struct _SequentialPack;

	// A Pack that is the concatenation of multiple Packs
	template< typename ... Packs > using Concatenation = typename _Concatenation< Packs ... >::type;

	// A Pack that is the permtuation of a Pack
	// [NOTE] The entry in the i-th position  of PermutationPack indicates where the i-th position comes from (not goes to)
	template< typename Pack , typename PermutationPack > using Permutation = typename _Permutation< Pack , PermutationPack >::type;

	// A Pack that has the same value repeated Dim times
	template< typename IntegralType , unsigned int Dim , IntegralType Value=0 > using IsotropicPack = typename _IsotropicPack< IntegralType , Dim , Value >::type;

	// A Pack with sequentially increasing values
	template< typename IntegralType , unsigned int Dim , IntegralType StartValue=0 > using SequentialPack = typename _SequentialPack< IntegralType , Dim , StartValue >::type;

	/////////////////////////
	/////////////////////////
	//// Specializations ////
	/////////////////////////
	/////////////////////////

	// A pack with int values
	template< int ... Values > using IntPack = Pack< int , Values... >;

	// A pack with unsigned int values
	template< unsigned int ... Values > using UIntPack = Pack< unsigned int , Values... >;

	// An isotropic pack with int values
	template< unsigned int Dim , int Value=0 > using IsotropicIntPack = IsotropicPack< int , Dim , Value >;

	// An isotropic pack with unsigned int values
	template< unsigned int Dim , unsigned int Value=0 > using IsotropicUIntPack = IsotropicPack< unsigned int , Dim , Value >;


	/////////////////////
	/////////////////////
	//// Definitions ////
	/////////////////////
	/////////////////////


	//////////
	// Pack //
	//////////

	// The general case
	template< typename IntegralType , IntegralType _Value , IntegralType ... _Values > struct Pack< IntegralType , _Value , _Values ... >
	{
		static const IntegralType First = _Value;
		typedef Pack< IntegralType , _Values ... > Rest;
		typedef typename Rest::Transpose::template Append< First > Transpose;

		static const unsigned int Size = 1 + sizeof ... ( _Values );

		template< IntegralType ... __Values > using  Append = Pack< IntegralType , _Value , _Values ... , __Values ... >;
		template< IntegralType ... __Values > using Prepend = Pack< IntegralType , __Values ... , _Value , _Values ... >;

		static const IntegralType Values[];

		template< unsigned int I > constexpr static IntegralType Get( void )
		{
			if constexpr( I==0 ) return _Value;
			else return Rest::template Get< I-1 >();
		}

		static constexpr IntegralType Min( void ){ return _Value < Rest::Min() ? _Value : Rest::Min(); }
		static constexpr IntegralType Max( void ){ return _Value > Rest::Max() ? _Value : Rest::Max(); }

		friend std::ostream &operator << ( std::ostream &os , Pack )
		{
			os << "< ";
			for( unsigned int i=0 ; i<Size ; i++ )
			{
				if( i ) os << " , ";
				os << Values[i];
			}
			return os << " >";
		}
	};

	// The specialized case with one entry
	template< typename IntegralType , IntegralType _Value > struct Pack< IntegralType , _Value >
	{
		static const IntegralType First = _Value;
		typedef Pack< IntegralType > Rest;
		typedef Pack< IntegralType , _Value > Transpose;

		static const unsigned int Size = 1;

		template< IntegralType ... __Values > using  Append = Pack< IntegralType , _Value , __Values ... >;
		template< IntegralType ... __Values > using Prepend = Pack< IntegralType , __Values ... , _Value >;

		static const IntegralType Values[];

		template< unsigned int I > constexpr static IntegralType Get( void )
		{
			static_assert( I==0 , "[ERROR] Pack< IntegralType , Value >::Get called with non-zero index" );
			return _Value;
		}

		static constexpr IntegralType Min( void ){ return _Value; }
		static constexpr IntegralType Max( void ){ return _Value; }

		friend std::ostream &operator << ( std::ostream &os , Pack )
		{
			return os << "< " << First << " >";
		}
	};

	// The specialized case with no entries
	template< typename IntegralType > struct Pack< IntegralType >
	{
		typedef Pack< IntegralType > Rest;
		static const unsigned int Size = 0;
		static constexpr IntegralType Values[] = { 0 };
		typedef Pack< IntegralType > Transpose;
		template< IntegralType ... __Values > using  Append = Pack< IntegralType , __Values ... >;
		template< IntegralType ... __Values > using Prepend = Pack< IntegralType , __Values ... >;
		friend std::ostream &operator << ( std::ostream &os , Pack ){ return os << "< >"; }
	};

	template< typename IntegralType , IntegralType _Value , IntegralType ... _Values > const IntegralType Pack< IntegralType , _Value , _Values ... >::Values[] = { _Value , _Values ... };
	template< typename IntegralType , IntegralType _Value > const IntegralType Pack< IntegralType , _Value >::Values[] = { _Value };

	///////////////
	// Selection //
	///////////////
	template< unsigned int I , typename IntegralType , IntegralType _Value , IntegralType ... _Values >
	struct Selection< I , Pack< IntegralType , _Value , _Values ... > >
	{
		static const IntegralType Value = Selection< I-1 , Pack< IntegralType , _Values ... > >::Value;
		typedef typename Selection< I-1 , Pack< IntegralType , _Values ... > >::Complement::template Prepend< _Value > Complement;
	};

	template< typename IntegralType , IntegralType _Value , IntegralType ... _Values >
	struct Selection< 0 , Pack< IntegralType , _Value , _Values ... > >
	{
		static const IntegralType Value = _Value;
		typedef Pack< IntegralType , _Values ... > Complement;
	};

	///////////////
	// Partition //
	///////////////
	template< typename IntegralType , IntegralType ... Values >
	struct Partition< 0 , Pack< IntegralType , Values ... > >
	{
		typedef Pack< IntegralType > First;
		typedef Pack< IntegralType , Values ... > Second;
	};

	template< unsigned int I , typename IntegralType , IntegralType ... Values >
	struct Partition< I , Pack< IntegralType , Values ... > >
	{
		typedef Concatenation< Pack< IntegralType , Pack< IntegralType , Values ... >::First > , typename Partition< I-1 , typename Pack< IntegralType , Values ... >::Rest >::First > First;
		typedef typename Partition< I-1 , typename Pack< IntegralType , Values ... >::Rest >::Second Second;
	};

	////////////////
	// Arithmetic //
	////////////////
	template< typename IntegralType >
	struct Arithmetic< Pack< IntegralType > , Pack< IntegralType > >
	{
		using Sum        = Pack< IntegralType >;
		using Difference = Pack< IntegralType >;
	};

	template< typename IntegralType , IntegralType Value1 , IntegralType ... Values1 , IntegralType Value2 , IntegralType ... Values2 >
	struct Arithmetic< Pack< IntegralType , Value1 , Values1 ... > , Pack< IntegralType , Value2 , Values2 ... > >
	{
		using Sum        = Concatenation< Pack< IntegralType , Value1+Value2 > , typename Arithmetic< Pack< IntegralType , Values1 ... > , Pack< IntegralType , Values2... > >::Sum >;
		using Difference = Concatenation< Pack< IntegralType , Value1-Value2 > , typename Arithmetic< Pack< IntegralType , Values1 ... > , Pack< IntegralType , Values2... > >::Difference >;
	};

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	typename Arithmetic< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::Sum operator + ( Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > ){ return typename Arithmetic< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::Sum(); }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	typename Arithmetic< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::Difference operator - ( Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > ){ return typename Arithmetic< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::Difference(); }


	////////////////
	// Comparison //
	////////////////
	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	struct Comparison< Pack< IntegralType , Values1 ... > , Pack< IntegralType , Values2 ... > >
	{
		typedef Pack< IntegralType , Values1 ... > Pack1;
		typedef Pack< IntegralType , Values2 ... > Pack2;
		static const bool              Equal = Pack1::First==Pack2::First && Comparison< typename Pack1::Rest , typename Pack2::Rest >::Equal;
		static const bool           NotEqual = Pack1::First!=Pack2::First || Comparison< typename Pack1::Rest , typename Pack2::Rest >::NotEqual;
		static const bool    LessThan        = Pack1::First< Pack2::First && Comparison< typename Pack1::Rest , typename Pack2::Rest >::LessThan;
		static const bool    LessThanOrEqual = Pack1::First<=Pack2::First && Comparison< typename Pack1::Rest , typename Pack2::Rest >::LessThanOrEqual;
		static const bool GreaterThan        = Pack1::First> Pack2::First && Comparison< typename Pack1::Rest , typename Pack2::Rest >::GreaterThan;
		static const bool GreaterThanOrEqual = Pack1::First>=Pack2::First && Comparison< typename Pack1::Rest , typename Pack2::Rest >::GreaterThanOrEqual;
	};

	template< typename IntegralType , IntegralType Value1 , IntegralType Value2 >
	struct Comparison< Pack< IntegralType , Value1 > , Pack< IntegralType , Value2 > >
	{
		static const bool Equal = Value1==Value2;
		static const bool NotEqual = Value1!=Value2;
		static const bool LessThan = Value1<Value2;
		static const bool LessThanOrEqual = Value1<=Value2;
		static const bool GreaterThan = Value1>Value2;
		static const bool GreaterThanOrEqual = Value1>=Value2;
	};

	template< typename IntegralType >
	struct Comparison< Pack< IntegralType > , Pack< IntegralType > >
	{
		static const bool Equal = true;
		static const bool NotEqual = false;
		static const bool LessThan = false;
		static const bool LessThanOrEqual = true;
		static const bool GreaterThan = false;
		static const bool GreaterThanOrEqual = true;
	};

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator==( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::Equal; }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator!=( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::NotEqual; }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator<( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::LessThan; }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator<=( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::LessThanOrEqual; }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator>( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::GreaterThan; }

	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 >
	bool operator>=( const Pack< IntegralType , Values1... > , const Pack< IntegralType , Values2... > ){ return Comparison< Pack< IntegralType , Values1... > , Pack< IntegralType , Values2... > >::GreaterThanOrEqual; }

	////////////////////
	// _Concatenation //
	////////////////////
	template< typename IntegralType , IntegralType ... Values1 , IntegralType ... Values2 , typename ... Packs >
	struct _Concatenation< Pack< IntegralType , Values1 ... > , Pack< IntegralType , Values2 ... > , Packs ... >
	{
		typedef typename _Concatenation< typename Pack< IntegralType , Values1 ... >::template Append< Values2 ... > , Packs ... >::type type;
	};
	template< typename IntegralType , IntegralType ... Values >
	struct _Concatenation< Pack< IntegralType , Values ... > >
	{
		typedef Pack< IntegralType , Values ... > type;
	};

	//////////////////
	// _Permutation //
	//////////////////
	template< typename IntegralType , IntegralType ... Values , unsigned int ... PermutationValues >
	struct _Permutation< Pack< IntegralType , Values ... > , Pack< unsigned int , PermutationValues ... > >
	{
		typedef Pack< IntegralType , PermutationValues ... > PPack;
		typedef Concatenation< Pack< IntegralType , Selection< PPack::First , Pack< IntegralType , Values ... > >::Value > , typename _Permutation< Pack< IntegralType , Values ... > , typename PPack::Rest >::type > type;
	};
	template< typename IntegralType , IntegralType ... Values >
	struct _Permutation< Pack< IntegralType , Values ... > , Pack< unsigned int > >
	{
		typedef Pack< IntegralType > type;
	};

	////////////////////
	// _IsotropicPack //
	////////////////////
	template< typename IntegralType , unsigned int Dim , IntegralType Value > struct _IsotropicPack                            { typedef typename _IsotropicPack< IntegralType , Dim-1 , Value >::type::template Append< Value > type; };
	template< typename IntegralType ,                    IntegralType Value > struct _IsotropicPack< IntegralType , 1 , Value >{ typedef Pack< IntegralType , Value > type; };
	template< typename IntegralType ,                    IntegralType Value > struct _IsotropicPack< IntegralType , 0 , Value >{ typedef Pack< IntegralType > type; };

	/////////////////////
	// _SequentialPack //
	/////////////////////
	template< typename IntegralType , unsigned int Dim , IntegralType Value >   struct _SequentialPack                            { typedef Concatenation< Pack< IntegralType , Value > , typename _SequentialPack< IntegralType , Dim-1 , Value+1 >::type > type; };
	template< typename IntegralType ,                    IntegralType Value >   struct _SequentialPack< IntegralType , 0 , Value >{ typedef Pack< IntegralType > type; };
}
#endif // PARAMETER_PACK_INCLUDED
