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


///////////////////////////////
// MemoryOrientedPointStream //
///////////////////////////////
template< class Real >
MemoryOrientedPointStream< Real >::MemoryOrientedPointStream( size_t pointCount , const OrientedPoint3D< Real >* points ){ _points = points , _pointCount = pointCount , _current = 0; }
template< class Real >
MemoryOrientedPointStream< Real >::~MemoryOrientedPointStream( void ){ ; }
template< class Real >
void MemoryOrientedPointStream< Real >::reset( void ) { _current=0; }
template< class Real >
bool MemoryOrientedPointStream< Real >::nextPoint( OrientedPoint3D< Real >& p )
{
	if( _current>=_pointCount ) return false;
	p = _points[_current];
	_current++;
	return true;
}

//////////////////////////////
// ASCIIOrientedPointStream //
//////////////////////////////
template< class Real >
ASCIIOrientedPointStream< Real >::ASCIIOrientedPointStream( const char* fileName )
{
	_fp = fopen( fileName , "r" );
	if( !_fp ) fprintf( stderr , "Failed to open file for reading: %s\n" , fileName ) , exit( 0 );
}
template< class Real >
ASCIIOrientedPointStream< Real >::~ASCIIOrientedPointStream( void )
{
	fclose( _fp );
	_fp = NULL;
}
template< class Real >
void ASCIIOrientedPointStream< Real >::reset( void ) { fseek( _fp , SEEK_SET , 0 ); }
template< class Real >
bool ASCIIOrientedPointStream< Real >::nextPoint( OrientedPoint3D< Real >& p )
{
	float c[2*3];
	if( fscanf( _fp , " %f %f %f %f %f %f " , &c[0] , &c[1] , &c[2] , &c[3] , &c[4] , &c[5] )!=2*3 ) return false;
	p.p[0] = c[0] , p.p[1] = c[1] , p.p[2] = c[2];
	p.n[0] = c[3] , p.n[1] = c[4] , p.n[2] = c[5];
	return true;
}

///////////////////////////////
// BinaryOrientedPointStream //
///////////////////////////////
template< class Real >
BinaryOrientedPointStream< Real >::BinaryOrientedPointStream( const char* fileName )
{
	_pointsInBuffer = _currentPointIndex = 0;
	_fp = fopen( fileName , "rb" );
	if( !_fp ) fprintf( stderr , "Failed to open file for reading: %s\n" , fileName ) , exit( 0 );
}
template< class Real >
BinaryOrientedPointStream< Real >::~BinaryOrientedPointStream( void )
{
	fclose( _fp );
	_fp = NULL;
}
template< class Real >
void BinaryOrientedPointStream< Real >::reset( void )
{
	fseek( _fp , SEEK_SET , 0 );
	_pointsInBuffer = _currentPointIndex = 0;
}
template< class Real >
bool BinaryOrientedPointStream< Real >::nextPoint( OrientedPoint3D< Real >& p )
{
	if( _currentPointIndex<_pointsInBuffer )
	{
		p = _pointBuffer[ _currentPointIndex ];
		_currentPointIndex++;
		return true;
	}
	else
	{
		_currentPointIndex = 0;
		_pointsInBuffer = int( fread( _pointBuffer , sizeof( OrientedPoint3D< Real > ) , POINT_BUFFER_SIZE , _fp ) );
		if( !_pointsInBuffer ) return false;
		else return nextPoint( p );
	}
}

////////////////////////////
// PLYOrientedPointStream //
////////////////////////////
template< class Real >
PLYOrientedPointStream< Real >::PLYOrientedPointStream( const char* fileName )
{
	_fileName = new char[ strlen( fileName )+1 ];
	strcpy( _fileName , fileName );
	_ply = NULL;
	reset();
}
template< class Real >
void PLYOrientedPointStream< Real >::reset( void )
{
	int fileType;
	float version;
	PlyProperty** plist;
	if( _ply ) _free();
	_ply = ply_open_for_reading( _fileName, &_nr_elems, &_elist, &fileType, &version );
	if( !_ply )
	{
		fprintf( stderr, "[ERROR] Failed to open ply file for reading: %s\n" , _fileName );
		exit( 0 );
	}
	bool foundVertices = false;
	for( int i=0 ; i<_nr_elems ; i++ )
	{
		int num_elems;
		int nr_props;
		char* elem_name = _elist[i];
		plist = ply_get_element_description( _ply , elem_name , &num_elems , &nr_props );
		if( !plist )
		{
			fprintf( stderr , "[ERROR] Failed to get element description: %s\n" , elem_name );
			exit( 0 );
		}	

		if( equal_strings( "vertex" , elem_name ) )
		{
			foundVertices = true;
			_pCount = num_elems , _pIdx = 0;
			for( int i=0 ; i<PlyOrientedVertex< Real >::ReadComponents ; i++ ) 
				if( !ply_get_property( _ply , elem_name , &(PlyOrientedVertex< Real >::ReadProperties[i]) ) )
				{
					fprintf( stderr , "[ERROR] Failed to find property in ply file: %s\n" , PlyOrientedVertex< Real >::ReadProperties[i].name );
					exit( 0 );
				}
		}
		for( int j=0 ; j<nr_props ; j++ )
		{
			free( plist[j]->name );
			free( plist[j] );
		}
		free( plist );
		if( foundVertices ) break;
	}
	if( !foundVertices )
	{
		fprintf( stderr , "[ERROR] Could not find vertices in ply file\n" );
		exit( 0 );
	}
}
template< class Real >
void PLYOrientedPointStream< Real >::_free( void )
{
	if( _ply ) ply_close( _ply ) , _ply = NULL;
	if( _elist )
	{
		for( int i=0 ; i<_nr_elems ; i++ ) free( _elist[i] );
		free( _elist );
	}
}
template< class Real >
PLYOrientedPointStream< Real >::~PLYOrientedPointStream( void )
{
	_free();
	if( _fileName ) delete[] _fileName , _fileName = NULL;
}
template< class Real >
bool PLYOrientedPointStream< Real >::nextPoint( OrientedPoint3D< Real >& p )
{
	if( _pIdx<_pCount )
	{
		PlyOrientedVertex< Real > op;
		ply_get_element( _ply, (void *)&op );
		p.p = op.point;
		p.n = op.normal;
		_pIdx++;
		return true;
	}
	else return false;
}

///////////////////////////////////////
// MemoryOrientedPointStreamWithData //
///////////////////////////////////////
template< class Real , class Data >
MemoryOrientedPointStreamWithData< Real , Data >::MemoryOrientedPointStreamWithData( size_t pointCount , const std::pair< OrientedPoint3D< Real > , Data >* points ){ _points = points , _pointCount = pointCount , _current = 0; }
template< class Real , class Data >
MemoryOrientedPointStreamWithData< Real , Data >::~MemoryOrientedPointStreamWithData( void ){ ; }
template< class Real , class Data >
void MemoryOrientedPointStreamWithData< Real , Data >::reset( void ) { _current=0; }
template< class Real , class Data >
bool MemoryOrientedPointStreamWithData< Real , Data >::nextPoint( OrientedPoint3D< Real >& p , Data& d )
{
	if( _current>=_pointCount ) return false;
	p = _points[_current].first;
	d = _points[_current].second;
	_current++;
	return true;
}

//////////////////////////////////////
// ASCIIOrientedPointStreamWithData //
//////////////////////////////////////
template< class Real , class Data >
ASCIIOrientedPointStreamWithData< Real , Data >::ASCIIOrientedPointStreamWithData( const char* fileName , Data (*readData)( FILE* ) ) : _readData( readData )
{
	_fp = fopen( fileName , "r" );
	if( !_fp ) fprintf( stderr , "Failed to open file for reading: %s\n" , fileName ) , exit( 0 );
}
template< class Real , class Data >
ASCIIOrientedPointStreamWithData< Real , Data >::~ASCIIOrientedPointStreamWithData( void )
{
	fclose( _fp );
	_fp = NULL;
}
template< class Real , class Data >
void ASCIIOrientedPointStreamWithData< Real , Data >::reset( void ) { fseek( _fp , SEEK_SET , 0 ); }
template< class Real , class Data >
bool ASCIIOrientedPointStreamWithData< Real , Data >::nextPoint( OrientedPoint3D< Real >& p , Data& d )
{
	float c[2*3];
	if( fscanf( _fp , " %f %f %f %f %f %f " , &c[0] , &c[1] , &c[2] , &c[3] , &c[4] , &c[5] )!=2*3 ) return false;
	p.p[0] = c[0] , p.p[1] = c[1] , p.p[2] = c[2];
	p.n[0] = c[3] , p.n[1] = c[4] , p.n[2] = c[5];
	d = _readData( _fp );
	return true;
}

///////////////////////////////////////
// BinaryOrientedPointStreamWithData //
///////////////////////////////////////
template< class Real , class Data >
BinaryOrientedPointStreamWithData< Real , Data >::BinaryOrientedPointStreamWithData( const char* fileName )
{
	_pointsInBuffer = _currentPointIndex = 0;
	_fp = fopen( fileName , "rb" );
	if( !_fp ) fprintf( stderr , "Failed to open file for reading: %s\n" , fileName ) , exit( 0 );
}
template< class Real , class Data >
BinaryOrientedPointStreamWithData< Real , Data >::~BinaryOrientedPointStreamWithData( void )
{
	fclose( _fp );
	_fp = NULL;
}
template< class Real , class Data >
void BinaryOrientedPointStreamWithData< Real , Data >::reset( void )
{
	fseek( _fp , SEEK_SET , 0 );
	_pointsInBuffer = _currentPointIndex = 0;
}
template< class Real , class Data >
bool BinaryOrientedPointStreamWithData< Real , Data >::nextPoint( OrientedPoint3D< Real >& p , Data& d )
{
	if( _currentPointIndex<_pointsInBuffer )
	{
		p = _pointBuffer[ _currentPointIndex ].first;
		d = _pointBuffer[ _currentPointIndex ].second;
		_currentPointIndex++;
		return true;
	}
	else
	{
		_currentPointIndex = 0;
		_pointsInBuffer = int( fread( _pointBuffer , sizeof( std::pair< OrientedPoint3D< Real > , Data > ) , POINT_BUFFER_SIZE , _fp ) );
		if( !_pointsInBuffer ) return false;
		else return nextPoint( p , d );
	}
}

////////////////////////////////////
// PLYOrientedPointStreamWithData //
////////////////////////////////////
template< class Real , class Data >
PLYOrientedPointStreamWithData< Real , Data >::PLYOrientedPointStreamWithData( const char* fileName , const PlyProperty* dataProperties , int dataPropertiesCount , bool (*validationFunction)( const bool* ) ) : _dataPropertiesCount( dataPropertiesCount ) , _validationFunction( validationFunction )
{
	_dataProperties = new PlyProperty[ _dataPropertiesCount ];
	memcpy( _dataProperties , dataProperties , sizeof(PlyProperty) * _dataPropertiesCount );
	for( int i=0 ; i<_dataPropertiesCount ; i++ ) _dataProperties[i].offset += sizeof( PlyOrientedVertex< Real > );
	_fileName = new char[ strlen( fileName )+1 ];
	strcpy( _fileName , fileName );
	_ply = NULL;
	reset();
}
template< class Real , class Data >
void PLYOrientedPointStreamWithData< Real , Data >::reset( void )
{
	int fileType;
	float version;
	PlyProperty** plist;
	if( _ply ) _free();
	_ply = ply_open_for_reading( _fileName, &_nr_elems, &_elist, &fileType, &version );
	if( !_ply )
	{
		fprintf( stderr, "[ERROR] Failed to open ply file for reading: %s\n" , _fileName );
		exit( 0 );
	}
	bool foundVertices = false;
	for( int i=0 ; i<_nr_elems ; i++ )
	{
		int num_elems;
		int nr_props;
		char* elem_name = _elist[i];
		plist = ply_get_element_description( _ply , elem_name , &num_elems , &nr_props );
		if( !plist )
		{
			fprintf( stderr , "[ERROR] Failed to get element description: %s\n" , elem_name );
			exit( 0 );
		}	

		if( equal_strings( "vertex" , elem_name ) )
		{
			foundVertices = true;
			_pCount = num_elems , _pIdx = 0;
			for( int i=0 ; i<PlyOrientedVertex< Real >::ReadComponents ; i++ ) 
				if( !ply_get_property( _ply , elem_name , &(PlyOrientedVertex< Real >::ReadProperties[i]) ) )
				{
					fprintf( stderr , "[ERROR] Failed to find property in ply file: %s\n" , PlyOrientedVertex< Real >::ReadProperties[i].name );
					exit( 0 );
				}
			if( _validationFunction )
			{
				bool* properties = new bool[_dataPropertiesCount];
				for( int i=0 ; i<_dataPropertiesCount ; i++ )
					if( !ply_get_property( _ply , elem_name , &(_dataProperties[i]) ) ) properties[i] = false;
					else                                                                properties[i] = true;
				bool valid = _validationFunction( properties );
				delete[] properties;
				if( !valid ) fprintf( stderr , "[ERROR] Failed to validate properties in file\n" ) , exit( 0 );
			}
			else
			{
				for( int i=0 ; i<_dataPropertiesCount ; i++ )
					if( !ply_get_property( _ply , elem_name , &(_dataProperties[i]) ) )
						fprintf( stderr , "[WARNING] Failed to find property in ply file: %s\n" , _dataProperties[i].name );
			}
		}
		for( int j=0 ; j<nr_props ; j++ )
		{
			free( plist[j]->name );
			free( plist[j] );
		}
		free( plist );
		if( foundVertices ) break;
	}
	if( !foundVertices )
	{
		fprintf( stderr , "[ERROR] Could not find vertices in ply file\n" );
		exit( 0 );
	}
}
template< class Real , class Data >
void PLYOrientedPointStreamWithData< Real , Data >::_free( void )
{
	if( _ply ) ply_close( _ply ) , _ply = NULL;
	if( _elist )
	{
		for( int i=0 ; i<_nr_elems ; i++ ) free( _elist[i] );
		free( _elist );
	}
}
template< class Real , class Data >
PLYOrientedPointStreamWithData< Real , Data >::~PLYOrientedPointStreamWithData( void )
{
	_free();
	if( _fileName ) delete[] _fileName , _fileName = NULL;
	if( _dataProperties ) delete[] _dataProperties , _dataProperties = NULL;
}
template< class Real , class Data >
bool PLYOrientedPointStreamWithData< Real , Data >::nextPoint( OrientedPoint3D< Real >& p , Data& d )
{
	if( _pIdx<_pCount )
	{
		_PlyOrientedVertexWithData op;
		ply_get_element( _ply, (void *)&op );
		p.p = op.point;
		p.n = op.normal;
		d = op.data;
		_pIdx++;
		return true;
	}
	else return false;
}
