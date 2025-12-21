/*
Copyright (c) 2020, Michael Kazhdan
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

namespace VertexFactory
{
	int ToPlyType( TypeOnDisk typeOnDisk )
	{
		switch( typeOnDisk )
		{
			case TypeOnDisk::CHAR:    return PLY::Type<          char >();
			case TypeOnDisk::UCHAR:   return PLY::Type< unsigned char >();
			case TypeOnDisk::INT:     return PLY::Type<           int >();
			case TypeOnDisk::UINT:    return PLY::Type< unsigned  int >();
			case TypeOnDisk::FLOAT:   return PLY::Type<         float >();
			case TypeOnDisk::DOUBLE:  return PLY::Type<        double >();
			case TypeOnDisk::INT_8:   return PLY::Type<        int8_t >();
			case TypeOnDisk::UINT_8:  return PLY::Type<       uint8_t >();
			case TypeOnDisk::INT_16:  return PLY::Type<       int16_t >();
			case TypeOnDisk::UINT_16: return PLY::Type<      uint16_t >();
			case TypeOnDisk::INT_32:  return PLY::Type<       int32_t >();
			case TypeOnDisk::UINT_32: return PLY::Type<      uint32_t >();
			case TypeOnDisk::INT_64:  return PLY::Type<       int64_t >();
			case TypeOnDisk::UINT_64: return PLY::Type<      uint64_t >();
			default: MK_THROW( "Unrecognized type: " , typeOnDisk );
		}
		return -1;
	}

	TypeOnDisk FromPlyType( int plyType )
	{
		switch( plyType )
		{
			case PLY_INT:       return TypeOnDisk::INT;
			case PLY_UINT:      return TypeOnDisk::UINT;
			case PLY_CHAR:      return TypeOnDisk::CHAR;
			case PLY_UCHAR:     return TypeOnDisk::UCHAR;
			case PLY_FLOAT:     return TypeOnDisk::FLOAT;
			case PLY_DOUBLE:    return TypeOnDisk::DOUBLE;
			case PLY_INT_8:     return TypeOnDisk::INT_8;
			case PLY_UINT_8:    return TypeOnDisk::UINT_8;
			case PLY_INT_16:    return TypeOnDisk::INT_16;
			case PLY_UINT_16:   return TypeOnDisk::UINT_16;
			case PLY_INT_32:    return TypeOnDisk::INT_32;
			case PLY_UINT_32:   return TypeOnDisk::UINT_32;
			case PLY_INT_64:    return TypeOnDisk::INT_64;
			case PLY_UINT_64:   return TypeOnDisk::UINT_64;
			case PLY_FLOAT_32:  return TypeOnDisk::FLOAT;
			case PLY_FLOAT_64:  return TypeOnDisk::DOUBLE;
			default: MK_THROW( "Unrecognized type: " , plyType );
		}
		return TypeOnDisk::UNKNOWN;
	}

	template< typename Type > TypeOnDisk GetTypeOnDisk( void )
	{
		if      constexpr( std::is_same< Type ,          char >::value ) return TypeOnDisk::CHAR;
		else if constexpr( std::is_same< Type , unsigned char >::value ) return TypeOnDisk::UCHAR;
		else if constexpr( std::is_same< Type ,           int >::value ) return TypeOnDisk::INT;
		else if constexpr( std::is_same< Type , unsigned  int >::value ) return TypeOnDisk::UINT;
		else if constexpr( std::is_same< Type ,        int8_t >::value ) return TypeOnDisk::INT_8;
		else if constexpr( std::is_same< Type ,       uint8_t >::value ) return TypeOnDisk::UINT_8;
		else if constexpr( std::is_same< Type ,       int16_t >::value ) return TypeOnDisk::INT_16;
		else if constexpr( std::is_same< Type ,      uint16_t >::value ) return TypeOnDisk::UINT_16;
		else if constexpr( std::is_same< Type ,       int32_t >::value ) return TypeOnDisk::INT_32;
		else if constexpr( std::is_same< Type ,      uint32_t >::value ) return TypeOnDisk::UINT_32;
		else if constexpr( std::is_same< Type ,       int64_t >::value ) return TypeOnDisk::INT_64;
		else if constexpr( std::is_same< Type ,      uint64_t >::value ) return TypeOnDisk::UINT_64;
		else if constexpr( std::is_same< Type ,         float >::value ) return TypeOnDisk::FLOAT;
		else if constexpr( std::is_same< Type ,        double >::value ) return TypeOnDisk::DOUBLE;
		else MK_THROW( "Unrecognized type" );
		return TypeOnDisk::UNKNOWN;
	}

	template< typename Real >
	template< typename Type >
	bool VertexIO< Real >::_ReadBinary( FILE *fp , Real &r )
	{
		Type t;
		if( fread( &t , sizeof(Type) , 1 , fp )!=1 ) return false;
		r = (Real)t;
		return true;
	}

	template< typename Real >
	template< typename Type >
	void VertexIO< Real >::_WriteBinary( FILE *fp , Real r ){ Type t = (Type)r ; fwrite(  &t , sizeof(Type) , 1 , fp ); }

	template< typename Real >
	bool VertexIO< Real >::ReadASCII( FILE *fp , TypeOnDisk , Real &s )
	{
		double d;
		if( fscanf( fp , " %lf"  , &d )!=1 ) return false;
		s = (Real)d;
		return true;
	}

	template< typename Real >
	bool VertexIO< Real >::ReadBinary( FILE *fp , TypeOnDisk typeOnDisk , Real &s )
	{
		if( TypeOnDisk()==typeOnDisk ) return fread( &s , sizeof(Real) , 1 , fp )==1;
		switch( typeOnDisk )
		{
			case TypeOnDisk::CHAR:    return _ReadBinary<          char >( fp , s );
			case TypeOnDisk::UCHAR:   return _ReadBinary< unsigned char >( fp , s );
			case TypeOnDisk::INT:     return _ReadBinary<           int >( fp , s );
			case TypeOnDisk::UINT:    return _ReadBinary< unsigned  int >( fp , s );
			case TypeOnDisk::FLOAT:   return _ReadBinary<         float >( fp , s );
			case TypeOnDisk::DOUBLE:  return _ReadBinary<        double >( fp , s );
			case TypeOnDisk::INT_8:   return _ReadBinary<        int8_t >( fp , s );
			case TypeOnDisk::UINT_8:  return _ReadBinary<       uint8_t >( fp , s );
			case TypeOnDisk::INT_16:  return _ReadBinary<       int16_t >( fp , s );
			case TypeOnDisk::UINT_16: return _ReadBinary<      uint16_t >( fp , s );
			case TypeOnDisk::INT_32:  return _ReadBinary<       int32_t >( fp , s );
			case TypeOnDisk::UINT_32: return _ReadBinary<      uint32_t >( fp , s );
			case TypeOnDisk::INT_64:  return _ReadBinary<       int64_t >( fp , s );
			case TypeOnDisk::UINT_64: return _ReadBinary<      uint64_t >( fp , s );
			default: MK_THROW( "Unrecognized type: " , typeOnDisk );
		}
		return true;
	}
	template< typename Real >
	void VertexIO< Real >::WriteASCII( FILE *fp , TypeOnDisk typeOnDisk , const Real &s )
	{
		switch( typeOnDisk )
		{
			case TypeOnDisk::CHAR:    fprintf( fp , " %d"       , (         char)s ) ; break;
			case TypeOnDisk::UCHAR:   fprintf( fp , " %u"       , (unsigned char)s ) ; break;
			case TypeOnDisk::INT:     fprintf( fp , " %d"       , (          int)s ) ; break;
			case TypeOnDisk::UINT:    fprintf( fp , " %u"       , (unsigned  int)s ) ; break;
			case TypeOnDisk::FLOAT:   fprintf( fp , " %f"       , (        float)s ) ; break;
			case TypeOnDisk::DOUBLE:  fprintf( fp , " %f"       , (       double)s ) ; break;
			case TypeOnDisk::INT_8:   fprintf( fp , " %" PRId8  , (       int8_t)s ) ; break;
			case TypeOnDisk::UINT_8:  fprintf( fp , " %" PRIu8  , (      uint8_t)s ) ; break;
			case TypeOnDisk::INT_16:  fprintf( fp , " %" PRId16 , (      int16_t)s ) ; break;
			case TypeOnDisk::UINT_16: fprintf( fp , " %" PRIu16 , (     uint16_t)s ) ; break;
			case TypeOnDisk::INT_32:  fprintf( fp , " %" PRId32 , (      int32_t)s ) ; break;
			case TypeOnDisk::UINT_32: fprintf( fp , " %" PRIu32 , (     uint32_t)s ) ; break;
			case TypeOnDisk::INT_64:  fprintf( fp , " %" PRId64 , (      int64_t)s ) ; break;
			case TypeOnDisk::UINT_64: fprintf( fp , " %" PRIu64 , (     uint64_t)s ) ; break;
			default: MK_THROW( "Unrecongized type: " , typeOnDisk );
		}
	}

	template< typename Real >
	void VertexIO< Real >::WriteBinary( FILE *fp , TypeOnDisk typeOnDisk , const Real &s )
	{
		if( TypeOnDisk()==typeOnDisk ) fwrite( &s , sizeof(Real) , 1 , fp );
		switch( typeOnDisk )
		{
			case TypeOnDisk::CHAR:    _WriteBinary<          char >( fp , s ) ; break;
			case TypeOnDisk::UCHAR:   _WriteBinary< unsigned char >( fp , s ) ; break;
			case TypeOnDisk::INT:     _WriteBinary<           int >( fp , s ) ; break;
			case TypeOnDisk::UINT:    _WriteBinary< unsigned  int >( fp , s ) ; break;
			case TypeOnDisk::FLOAT:   _WriteBinary<         float >( fp , s ) ; break;
			case TypeOnDisk::DOUBLE:  _WriteBinary<        double >( fp , s ) ; break;
			case TypeOnDisk::INT_8:   _WriteBinary<        int8_t >( fp , s ) ; break;
			case TypeOnDisk::UINT_8:  _WriteBinary<       uint8_t >( fp , s ) ; break;
			case TypeOnDisk::INT_16:  _WriteBinary<       int16_t >( fp , s ) ; break;
			case TypeOnDisk::UINT_16: _WriteBinary<      uint16_t >( fp , s ) ; break;
			case TypeOnDisk::INT_32:  _WriteBinary<       int32_t >( fp , s ) ; break;
			case TypeOnDisk::UINT_32: _WriteBinary<      uint32_t >( fp , s ) ; break;
			case TypeOnDisk::INT_64:  _WriteBinary<       int64_t >( fp , s ) ; break;
			case TypeOnDisk::UINT_64: _WriteBinary<      uint64_t >( fp , s ) ; break;
			default: MK_THROW( "Unrecongized type: " , typeOnDisk );
		}
	}

	template< typename Real >
	bool VertexIO< Real >::ReadASCII( FILE *fp , TypeOnDisk typeOnDisk , size_t sz , Real *s )
	{
		for( size_t i=0 ; i<sz ; i++ ) if( !ReadASCII( fp , typeOnDisk , s[i] ) ) return false;
		return true;
	}

	template< typename Real >
	bool VertexIO< Real >::ReadBinary( FILE *fp , TypeOnDisk typeOnDisk , size_t sz , Real *s )
	{
		if( TypeOnDisk()==typeOnDisk ) return fread( s , sizeof(Real) , sz , fp )==sz;
		else for( size_t i=0 ; i<sz ; i++ ) if( !ReadBinary( fp , typeOnDisk , s[i] ) ) return false;
		return true;
	}
	template< typename Real >
	void VertexIO< Real >::WriteASCII( FILE *fp , TypeOnDisk typeOnDisk , size_t sz , const Real *s )
	{
		for( size_t i=0 ; i<sz ; i++ ) WriteASCII( fp , typeOnDisk , s[i] );
	}

	template< typename Real >
	void VertexIO< Real >::WriteBinary( FILE *fp , TypeOnDisk typeOnDisk , size_t sz , const Real *s )
	{
		if( TypeOnDisk()==typeOnDisk ) fwrite( s , sizeof(Real) , sz , fp );
		else for( size_t i=0 ; i<sz ; i++ ) WriteBinary( fp , typeOnDisk , s[i] );
	}

	/////////////////////
	// PositionFactory //
	/////////////////////
	template< typename Real , unsigned int Dim >
	PlyProperty PositionFactory< Real , Dim >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty PositionFactory< Real , Dim >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real , unsigned int Dim >
	PlyProperty PositionFactory< Real , Dim >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty PositionFactory< Real , Dim >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}

	template< typename Real , unsigned int Dim >
	std::string PositionFactory< Real , Dim >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "x" , "y" , "z" };
		return names[idx];
	}

	///////////////////
	// NormalFactory //
	///////////////////
	template< typename Real , unsigned int Dim >
	PlyProperty NormalFactory< Real , Dim >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty NormalFactory< Real , Dim >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real , unsigned int Dim >
	PlyProperty NormalFactory< Real , Dim >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty NormalFactory< Real , Dim >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}

	template< typename Real , unsigned int Dim >
	std::string NormalFactory< Real , Dim >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "nx" , "ny" , "nz" };
		return names[idx];
	}

	////////////////////
	// TextureFactory //
	////////////////////
	template< typename Real , unsigned int Dim >
	PlyProperty TextureFactory< Real , Dim >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty TextureFactory< Real , Dim >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real , unsigned int Dim >
	PlyProperty TextureFactory< Real , Dim >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}
	template< typename Real , unsigned int Dim >
	PlyProperty TextureFactory< Real , Dim >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}

	template< typename Real , unsigned int Dim >
	std::string TextureFactory< Real , Dim >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "u" , "v" , "w" };
		return names[idx];
	}

	/////////////////////
	// RGBColorFactory //
	/////////////////////
	template< typename Real >
	PlyProperty RGBColorFactory< Real >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}
	template< typename Real >
	PlyProperty RGBColorFactory< Real >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	PlyProperty RGBColorFactory< Real >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + (idx%3)*sizeof(Real) ) );
	}
	template< typename Real >
	PlyProperty RGBColorFactory< Real >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}

	template< typename Real >
	std::string RGBColorFactory< Real >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "red" , "green" , "blue" , "r" , "g" , "b" };
		return names[idx];
	}

	//////////////////////
	// RGBAColorFactory //
	//////////////////////
	template< typename Real >
	PlyProperty RGBAColorFactory< Real >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	PlyProperty RGBAColorFactory< Real >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	PlyProperty RGBAColorFactory< Real >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + (idx%4)*sizeof(Real) ) );
	}

	template< typename Real >
	PlyProperty RGBAColorFactory< Real >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , (int)( offsetof( VertexType , coords ) + idx*sizeof(Real) ) );
	}

	template< typename Real >
	std::string RGBAColorFactory< Real >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "red" , "green" , "blue" , "alpha" , "r" , "g" , "b" , "a" };
		return names[idx];
	}

	//////////////////
	// ValueFactory //
	//////////////////
	template< typename Real >
	PlyProperty ValueFactory< Real >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>= plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	PlyProperty ValueFactory< Real >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	PlyProperty ValueFactory< Real >::plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , 0 );
	}

	template< typename Real >
	PlyProperty ValueFactory< Real >::plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _PlyName(idx) , ToPlyType( _typeOnDisk ) , PLY::Type< Real >() , 0 );
	}

	template< typename Real >
	std::string ValueFactory< Real >::_PlyName( unsigned int idx )
	{
		static const std::string names[] = { "value" };
		return names[idx];
	}

	////////////////////
	// DynamicFactory //
	////////////////////
	template< typename Real >
	DynamicFactory< Real >::DynamicFactory( const std::vector< std::pair< std::string , TypeOnDisk > > &namesAndTypesOnDisk ) : _namesAndTypesOnDisk(namesAndTypesOnDisk)
	{
		_realTypeOnDisk = true;
		for( unsigned int i=0 ; i<_namesAndTypesOnDisk.size() ; i++ ) _realTypeOnDisk &= GetTypeOnDisk< Real>()!=_namesAndTypesOnDisk[i].second;
	}
	template< typename Real >
	DynamicFactory< Real >::DynamicFactory( const std::vector< PlyProperty > &plyProperties )
	{
		for( int i=0 ; i<plyProperties.size() ; i++ )
			if( !plyProperties[i].is_list ) _namesAndTypesOnDisk.push_back( std::pair< std::string , TypeOnDisk >( plyProperties[i].name , FromPlyType( plyProperties[i].external_type ) ) );
			else MK_WARN( "List property not supported: " , plyProperties[i].name );
		_realTypeOnDisk = true;
		for( unsigned int i=0 ; i<_namesAndTypesOnDisk.size() ; i++ ) _realTypeOnDisk &= GetTypeOnDisk< Real>()!=_namesAndTypesOnDisk[i].second;
	}

	template< typename Real >
	bool DynamicFactory< Real >::readASCII( FILE *fp , VertexType &dt ) const
	{
		for( unsigned int i=0 ; i<dt.dim() ; i++ ) if( !VertexIO< Real >::ReadASCII( fp , _namesAndTypesOnDisk[i].second , dt[i] ) ) return false;
		return true;
	}
	template< typename Real >
	bool DynamicFactory< Real >::readBinary( FILE *fp , VertexType &dt ) const
	{
		if( _realTypeOnDisk )
		{
			if( fread( &dt[0] , sizeof(Real) , dt.dim() , fp )!=dt.dim() ) return false;
		}
		else
		{
			for( unsigned int i=0 ; i<dt.dim() ; i++ ) if( !VertexIO< Real >::ReadBinary( fp , _namesAndTypesOnDisk[i].second , dt[i] ) ) return false;
		}
		return true;
	}
	template< typename Real >
	void DynamicFactory< Real >::writeASCII( FILE *fp , const VertexType &dt ) const
	{
		for( unsigned int i=0 ; i<dt.dim() ; i++ ) VertexIO< Real >::WriteASCII( fp , _namesAndTypesOnDisk[i].second , dt[i] );
	}
	template< typename Real >
	void DynamicFactory< Real >::writeBinary( FILE *fp , const VertexType &dt ) const
	{
		if( _realTypeOnDisk ) fwrite( &dt[0] , sizeof(Real) , dt.dim() , fp );
		else  for( unsigned int i=0 ; i<dt.dim() ; i++ ) VertexIO< Real >::WriteBinary( fp , _namesAndTypesOnDisk[i].second , dt[i] );
	}

	template< typename Real >
	PlyProperty DynamicFactory< Real >::plyReadProperty( unsigned int idx ) const
	{
		if( idx>=plyReadNum() ) MK_THROW( "read property out of bounds" );
		return PlyProperty( _namesAndTypesOnDisk[idx].first , ToPlyType( _namesAndTypesOnDisk[idx].second ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}
	template< typename Real >
	PlyProperty DynamicFactory< Real >::plyWriteProperty( unsigned int idx ) const
	{
		if( idx>=plyWriteNum() ) MK_THROW( "write property out of bounds" );
		return PlyProperty( _namesAndTypesOnDisk[idx].first , ToPlyType( _namesAndTypesOnDisk[idx].second ) , PLY::Type< Real >() , sizeof(Real)*idx );
	}

	template< typename Real >
	bool DynamicFactory< Real >::operator == ( const DynamicFactory< Real > &factory ) const
	{
		if( size()!=factory.size() ) return false;
		for( int i=0 ; i<size() ; i++ ) if( _namesAndTypesOnDisk[i].first!=factory._namesAndTypesOnDisk[i].first || _namesAndTypesOnDisk[i].second!=factory._namesAndTypesOnDisk[i].second ) return false;
		return true;
	}

	/////////////
	// Factory //
	/////////////
	template< typename Real , typename ... Factories >
	template< unsigned int I >
	typename std::enable_if< I!=sizeof...(Factories) , PlyProperty >::type Factory< Real , Factories ... >::_plyReadProperty( unsigned int idx , size_t offset ) const
	{
		if( idx<this->template get<I>().plyReadNum() )
		{
			PlyProperty prop = this->template get<I>().plyReadProperty(idx);
			prop.offset += (int)offset;
			return prop;
		}
		else return _plyReadProperty<I+1>( idx - this->template get<I>().plyReadNum() , offset + this->template get<I>().bufferSize() );
	}

	template< typename Real , typename ... Factories >
	template< unsigned int I >
	typename std::enable_if< I!=sizeof...(Factories) , PlyProperty >::type Factory< Real , Factories ... >::_plyWriteProperty( unsigned int idx , size_t offset ) const
	{
		if( idx<this->template get<I>().plyWriteNum() )
		{
			PlyProperty prop = this->template get<I>().plyWriteProperty(idx);
			prop.offset += (int)offset;
			return prop;
		}
		else return _plyWriteProperty<I+1>( idx - this->template get<I>().plyWriteNum() , offset + this->template get<I>().bufferSize() );
	}

	template< typename Real , typename ... Factories >
	template< unsigned int I >
	typename std::enable_if< I!=sizeof...(Factories) , PlyProperty >::type Factory< Real , Factories ... >::_plyStaticReadProperty( unsigned int idx ) const
	{
		if( idx<this->template get<I>().plyReadNum() )
		{
			VertexType v;
			PlyProperty prop = this->template get<I>().plyStaticReadProperty( idx );
			prop.offset += (int)( (size_t)&v.template get<I>() - (size_t)&v );
			return prop;
		}
		else return _plyStaticReadProperty<I+1>( idx - this->template get<I>().plyReadNum() );
	}

	template< typename Real , typename ... Factories >
	template< unsigned int I >
	typename std::enable_if< I!=sizeof...(Factories) , PlyProperty >::type Factory< Real , Factories ... >::_plyStaticWriteProperty( unsigned int idx ) const
	{
		if( idx<this->template get<I>().plyWriteNum() )
		{
			VertexType v;
			PlyProperty prop = this->template get<I>().plyStaticWriteProperty( idx );
			prop.offset += (int)( (size_t)&v.template get<I>() - (size_t)&v );
			return prop;
		}
		else return _plyStaticWriteProperty<I+1>( idx - this->template get<I>().plyWriteNum() ) ;
	}
}
