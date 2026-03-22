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

//////////////////////////
// ASCIIInputDataStream //
//////////////////////////
template< typename Factory >
ASCIIInputDataStream< Factory >::ASCIIInputDataStream( const char* fileName , const Factory &factory ) : _factory( factory )
{
	_fp = fopen( fileName , "r" );
	if( !_fp ) MK_THROW( "Failed to open file for reading: %s" , fileName );
}

template< typename Factory >
ASCIIInputDataStream< Factory >::~ASCIIInputDataStream( void )
{
	fclose( _fp );
	_fp = NULL;
}

template< typename Factory >
void ASCIIInputDataStream< Factory >::reset( void ) { fseek( _fp , 0 , SEEK_SET ); }

template< typename Factory >
bool ASCIIInputDataStream< Factory >::read( Data &d ){ return _factory.readASCII( _fp , d ); }

///////////////////////////
// ASCIIOutputDataStream //
///////////////////////////
template< typename Factory >
ASCIIOutputDataStream< Factory >::ASCIIOutputDataStream( const char* fileName , const Factory &factory ) : _factory( factory )
{
	_fp = fopen( fileName , "w" );
	if( !_fp ) MK_THROW( "Failed to open file for writing: %s" , fileName );
}

template< typename Factory >
ASCIIOutputDataStream< Factory >::~ASCIIOutputDataStream( void )
{
	fclose( _fp );
	_fp = NULL;
}

template< typename Factory >
size_t ASCIIOutputDataStream< Factory >::write( const Data &d ){ _factory.writeASCII( _fp , d ) ; return _sz++; }

///////////////////////////
// BinaryInputDataStream //
///////////////////////////
template< typename Factory >
BinaryInputDataStream< Factory >::BinaryInputDataStream( const char* fileName , const Factory &factory ) : _factory(factory)
{
	_fp = fopen( fileName , "rb" );
	if( !_fp ) MK_THROW( "Failed to open file for reading: %s" , fileName );
}

template< typename Factory >
void BinaryInputDataStream< Factory >::reset( void ) { fseek( _fp , 0 , SEEK_SET ); }

template< typename Factory >
bool BinaryInputDataStream< Factory >::read( Data &d ){ return _factory.readBinary( _fp , d ); }

////////////////////////////
// BinaryOutputDataStream //
////////////////////////////
template< typename Factory >
BinaryOutputDataStream< Factory >::BinaryOutputDataStream( const char* fileName , const Factory &factory ) : _factory(factory)
{
	_fp = fopen( fileName , "wb" );
	if( !_fp ) MK_THROW( "Failed to open file for writing: %s" , fileName );
}

template< typename Factory >
size_t BinaryOutputDataStream< Factory >::write( const Data &d ){ _factory.writeBinary( _fp , d ); return _sz++; }

////////////////////////
// PLYInputDataStream //
////////////////////////
template< typename Factory >
PLYInputDataStream< Factory >::PLYInputDataStream( const char* fileName , const Factory &factory , size_t &count ) : _factory(factory)
{
	_fileName = new char[ strlen( fileName )+1 ];
	strcpy( _fileName , fileName );
	_ply = NULL;
	if( factory.bufferSize() ) _buffer = NewPointer< char >( factory.bufferSize() );
	else _buffer = NullPointer( char );
	reset();
	count = _pCount;
}

template< typename Factory >
PLYInputDataStream< Factory >::PLYInputDataStream( const char* fileName , const Factory &factory ) : _factory(factory)
{
	_fileName = new char[ strlen( fileName )+1 ];
	strcpy( _fileName , fileName );
	_ply = NULL;
	if( factory.bufferSize() ) _buffer = NewPointer< char >( factory.bufferSize() );
	else _buffer = NullPointer( char );
	reset();
}

template< typename Factory >
void PLYInputDataStream< Factory >::reset( void )
{
	int fileType;
	float version;
	if( _ply ) _free();
	_ply = PlyFile::Read( _fileName, _elist, fileType, version );
	if( !_ply ) MK_THROW( "Failed to open ply file for reading: " , _fileName );

	bool foundData = false;
	for( int i=0 ; i<_elist.size() ; i++ )
	{
		std::string &elem_name = _elist[i];

		if( elem_name=="vertex" )
		{
			size_t num_elems;
			std::vector< PlyProperty > plist = _ply->get_element_description( elem_name , num_elems );
			if( !plist.size() ) MK_THROW( "Failed to get description for \"" , elem_name , "\"" );

			foundData = true;
			_pCount = num_elems , _pIdx = 0;

			bool* properties = new bool[ _factory.plyReadNum() ];
			for( unsigned int i=0 ; i<_factory.plyReadNum() ; i++ )
			{
				PlyProperty prop;
				if constexpr( Factory::IsStaticallyAllocated() ) prop = _factory.plyStaticReadProperty(i);
				else                                             prop = _factory.plyReadProperty(i);
				if( !_ply->get_property( elem_name , &prop ) ) properties[i] = false;
				else                                           properties[i] = true;
			}
			bool valid = _factory.plyValidReadProperties( properties );
			delete[] properties;
			if( !valid ) MK_THROW( "Failed to validate properties in file" );
		}
	}
	if( !foundData ) MK_THROW( "Could not find data in ply file" );
}

template< typename Factory >
void PLYInputDataStream< Factory >::_free( void ){ delete _ply; }

template< typename Factory >
PLYInputDataStream< Factory >::~PLYInputDataStream( void )
{
	_free();
	if( _fileName ) delete[] _fileName , _fileName = NULL;
	DeletePointer( _buffer );
}

template< typename Factory >
bool PLYInputDataStream< Factory >::read( Data &d )
{
	if( _pIdx<_pCount )
	{
		if constexpr( Factory::IsStaticallyAllocated() ) _ply->get_element( (void *)&d );
		else
		{
			_ply->get_element( PointerAddress( _buffer ) );
			_factory.fromBuffer( _buffer , d );
		}
		_pIdx++;
		return true;
	}
	else return false;
}

/////////////////////////
// PLYOutputDataStream //
/////////////////////////
template< typename Factory >
PLYOutputDataStream< Factory >::PLYOutputDataStream( const char* fileName , const Factory &factory , size_t count , int fileType ) : _factory(factory)
{
	float version;
	std::vector< std::string > elem_names = { std::string( "vertex" ) };
	_ply = PlyFile::Write( fileName , elem_names , fileType , version );
	if( !_ply ) MK_THROW( "Failed to open ply file for writing: " , fileName );

	_pIdx = 0;
	_pCount = count;
	_ply->element_count( "vertex" , _pCount );
	for( unsigned int i=0 ; i<_factory.plyWriteNum() ; i++ )
	{
		PlyProperty prop;
		if constexpr( Factory::IsStaticallyAllocated() ) prop = _factory.plyStaticWriteProperty(i);
		else                                             prop = _factory.plyWriteProperty(i);
		_ply->describe_property( "vertex" , &prop );
	}
	_ply->header_complete();
	_ply->put_element_setup( "vertex" );
	if( _factory.bufferSize() ) _buffer = NewPointer< char >( _factory.bufferSize() );
	else                        _buffer = NullPointer( char );
}

template< typename Factory >
PLYOutputDataStream< Factory >::~PLYOutputDataStream( void )
{
	if( _pIdx!=_pCount ) MK_THROW( "Streamed points not equal to total count: " , _pIdx , " != " , _pCount );
	delete _ply;
	DeletePointer( _buffer );
}

template< typename Factory >
size_t PLYOutputDataStream< Factory >::write( const Data &d )
{
	if( _pIdx==_pCount ) MK_THROW( "Trying to add more points than total: " , _pIdx , " < " , _pCount );
	if constexpr( Factory::IsStaticallyAllocated() ) _ply->put_element( (void *)&d );
	else
	{
		_factory.toBuffer( d , _buffer );
		_ply->put_element( PointerAddress( _buffer ) );
	}
	return _pIdx++;
}

