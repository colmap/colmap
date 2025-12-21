/*
Copyright (c) 2019, Michael Kazhdan
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

/////////////////////////
// RegularGridTypeData //
/////////////////////////

void RegularGridDataType<>::Write( FILE *fp , unsigned int dim , std::string name ){ fprintf( fp , "%d %s\n" , (int)dim , name.c_str() ); }
bool RegularGridDataType<>::Read( FILE *fp , unsigned int dim , std::string name )
{
	char line[1024];
	int d;
	if( fscanf( fp , " %d %s " , &d , line )!=2 ) return false;
	return d==dim && name==std::string(line);
}

/////////////////
// RegularGrid //
/////////////////

template< typename DataType , unsigned int Dim >
template< typename Int >
typename std::enable_if< std::is_integral< Int >::value >::type RegularGrid< DataType , Dim >::resize( Int res[] )
{
	if( _values ) DeletePointer( _values );
	size_t resolution = 1;
	for( int d=0 ; d<Dim ; d++ ) _res[d] = (unsigned int)res[d] , resolution *= (size_t)res[d];
	if( resolution ) _values = NewPointer< DataType >( resolution );
}

template< typename DataType , unsigned int Dim >
template< typename Int >
typename std::enable_if< std::is_integral< Int >::value >::type RegularGrid< DataType , Dim >::resize( const Int res[] )
{
	if( _values ) DeletePointer( _values );
	size_t resolution = 1;
	for( int d=0 ; d<Dim ; d++ ) _res[d] = (unsigned int)res[d] , resolution *= (size_t)res[d];
	if( resolution ) _values = NewPointer< DataType >( resolution );
}

template< typename DataType , unsigned int Dim >
template< typename Int , typename ... Ints >
typename std::enable_if< std::is_integral< Int >::value >::type RegularGrid< DataType , Dim >::resize( Int res , Ints ... ress )
{
	static_assert( sizeof...(ress)+1==Dim , "[ERROR] number of resolutions does not match the number of dimensions" );
	const Int r[] = { res , ress ... };
	return resize( r );
}

template< typename DataType , unsigned int Dim >
template< typename Real , unsigned int D >
typename std::enable_if< D==1 , ProjectiveData< Real , DataType > >::type RegularGrid< DataType , Dim >::_Sample( const unsigned int res[] , const Real coords[] , ConstPointer( DataType ) values )
{
	int iCoord1 = (int)floor(coords[0]) , iCoord2 = (int)floor(coords[0])+1;
	Real dx1 = (Real)( iCoord2 - coords[0] ) , dx2 = (Real)( coords[0] - iCoord1 );
	ProjectiveData< Real , DataType > d;
	if( iCoord1>=0 && iCoord1<(int)res[0] ) d += ProjectiveData< Real , DataType >( values[ iCoord1 ] * dx1 , dx1 );
	if( iCoord2>=0 && iCoord2<(int)res[0] ) d += ProjectiveData< Real , DataType >( values[ iCoord2 ] * dx2 , dx2 );
	return d;
}

template< typename DataType , unsigned int Dim >
template< typename Real , unsigned int D >
typename std::enable_if< D!=1 , ProjectiveData< Real , DataType > >::type RegularGrid< DataType , Dim >::_Sample( const unsigned int res[] , const Real coords[] , ConstPointer( DataType ) values )
{
	int iCoord1 = (int)floor(coords[D-1]) , iCoord2 = (int)floor(coords[D-1])+1;
	Real dx1 = (Real)( iCoord2 - coords[D-1] ) , dx2 = (Real)( coords[D-1] - iCoord1 );
	ProjectiveData< Real , DataType > d;
	if( iCoord1>=0 && iCoord1<(int)res[D-1] ) d += _Sample< Real , D-1 >( res , coords , values + _Resolution< D-1 >(res) * iCoord1 ) * dx1;
	if( iCoord2>=0 && iCoord2<(int)res[D-1] ) d += _Sample< Real , D-1 >( res , coords , values + _Resolution< D-1 >(res) * iCoord2 ) * dx2;
	return d;
}

template< typename DataType , unsigned int Dim >
bool RegularGrid< DataType , Dim >::ReadHeader( std::string fileName , unsigned int &dim , std::string &name )
{
	FILE *fp = fopen( fileName.c_str() , "rb" );
	if( !fp ) return false;
	else
	{
		// Write the magic number
		int d;
		if( fscanf( fp , " G%d " , &d )!=1 || d!=Dim ){ fclose(fp) ; return false; }

		char line[1024];
		if( fscanf( fp , " %d %s " , &d , line )!=2 ){ fclose(fp) ; return false; }
		dim = d , name =std::string( line );
		fclose( fp );
	}
	return true;
}

template< typename DataType , unsigned int Dim >
template< typename Real >
void RegularGrid< DataType , Dim >::Write( std::string fileName , const unsigned int res[Dim] , ConstPointer( DataType ) values , XForm< Real , Dim+1 > gridToModel )
{
	FILE *fp = fopen( fileName.c_str() , "wb" );
	if( !fp ) MK_THROW( "Failed to open grid file for writing: " , fileName );
	else
	{
		// Write the magic number
		fprintf( fp , "G%d\n" , (int)Dim );

		RegularGridDataType< DataType >::Write( fp );

		// Write the dimensions
		for( int d=0 ; d<Dim ; d++ )
		{
			fprintf( fp , "%d" , (int)res[d] );
			if( d==Dim-1 ) fprintf( fp , "\n" );
			else           fprintf( fp , " " );
		}

		// Write the transformation
		for( int j=0 ; j<Dim+1 ; j++ ) for( int i=0 ; i<Dim+1 ; i++ )
		{
			fprintf( fp , "%f" , (float)gridToModel(i,j) );
			if( i==Dim ) fprintf( fp , "\n" );
			else         fprintf( fp , " " );
		}

		// Write the grid values
		fwrite( values , sizeof(DataType) , _Resolution(res) , fp );
		fclose( fp );
	}
}


template< typename DataType , unsigned int Dim >
template< typename Real >
void RegularGrid< DataType , Dim >::write( std::string fileName , XForm< Real , Dim+1 > gridToModel ) const
{
	Write( fileName , _res , _values , gridToModel );
}


template< typename DataType , unsigned int Dim >
template< typename Real >
void RegularGrid< DataType , Dim >::Read( std::string fileName , unsigned int res[Dim] , Pointer( DataType ) &values , XForm< Real , Dim+1 > &gridToModel )
{
	FILE *fp = fopen( fileName.c_str() , "rb" );
	if( !fp ) MK_THROW( "Failed to open grid file for reading: " , fileName );
	else
	{
		// Read the magic number
		{
			int dim;
			if( fscanf( fp , " G%d " , &dim )!=1 ) MK_THROW( "Failed to read magic number: " , fileName );
			if( dim!=Dim ) MK_THROW( "Dimensions don't match: " , Dim , " != " , dim );
		}

		// Read the data type
		if( !RegularGridDataType< DataType >::Read( fp ) ) MK_THROW( "Failed to read type" );

		// Read the dimensions
		{
			int r;
			for( int d=0 ; d<Dim ; d++ )
			{
				if( fscanf( fp , " %d " , &r )!=1 ) MK_THROW( "Failed to read dimension[ " , d , " ]" );
				res[d] = r;
			}
		}

		// Read the transformation
		{
			float x;
			for( int j=0 ; j<Dim+1 ; j++ ) for( int i=0 ; i<Dim+1 ; i++ )
			{
				if( fscanf( fp , " %f" , &x )!=1 ) MK_THROW( "Failed to read xForm( " , i , " , " , j , " )" );
				gridToModel(i,j) = x;
			}
		}

		// Read through the end of the line
		{
			char line[1024];
			if( !fgets( line , sizeof(line)/sizeof(char) , fp ) ) MK_THROW( "Could not read end of line" );
		}

		values = NewPointer< DataType >( _Resolution(res) );

		// Write the grid values
		fread( values , sizeof(DataType) , _Resolution(res) , fp );
		fclose( fp );
	}
}


template< typename DataType , unsigned int Dim >
template< typename Real >
void RegularGrid< DataType , Dim >::read( std::string fileName , XForm< Real , Dim+1 > &gridToModel )
{
	Read( fileName , _res , _values , gridToModel );
}
