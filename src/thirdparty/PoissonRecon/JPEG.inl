/*
Copyright (c) 2010, Michael Kazhdan
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

inline METHODDEF( void )
my_error_exit (j_common_ptr cinfo)
{
	// cinfo->err really points to a my_error_mgr struct, so coerce pointer
	my_error_ptr myerr = (my_error_ptr) cinfo->err;

	// Always display the message.
	// We could postpone this until after returning, if we chose.
	(*cinfo->err->output_message) (cinfo);

	// Return control to the setjmp point
	longjmp(myerr->setjmp_buffer, 1);
}

inline bool JPEGReader::GetInfo( const char* fileName , unsigned int& width , unsigned int& height , unsigned int& channels )
{
	FILE* fp = fopen( fileName , "rb" );
	if( !fp ) MK_THROW( "Failed to open: " , fileName );

	struct jpeg_decompress_struct cInfo;
	struct my_error_mgr jErr;

	cInfo.err = jpeg_std_error( &jErr.pub );
	jErr.pub.error_exit = my_error_exit;
	if( setjmp( jErr.setjmp_buffer ) )
	{
		jpeg_destroy_decompress( &cInfo );
		MK_THROW( "JPEG error occured" );
	}

	jpeg_create_decompress( &cInfo );
	jpeg_stdio_src( &cInfo , fp );

	(void) jpeg_read_header( &cInfo , TRUE );

	channels = cInfo.num_components;
	width  = cInfo.image_width;
	height = cInfo.image_height;
	jpeg_destroy_decompress( &cInfo );

	fclose( fp );
	return true;
}

inline JPEGReader::JPEGReader( const char* fileName , unsigned int& width , unsigned int& height , unsigned int& channels )
{
	_currentRow = 0;
	_fp = fopen( fileName , "rb" );
	if( !_fp ) MK_THROW( "Failed to open: " , fileName );

	_cInfo.err = jpeg_std_error( &_jErr.pub );
	_jErr.pub.error_exit = my_error_exit;
	if( setjmp( _jErr.setjmp_buffer ) )
	{
		jpeg_destroy_decompress( &_cInfo );
		MK_THROW( "JPEG error occured" );
	}

	jpeg_create_decompress( &_cInfo );
	jpeg_stdio_src( &_cInfo , _fp );

	(void) jpeg_read_header( &_cInfo , TRUE );
	(void) jpeg_start_decompress( &_cInfo );

	channels = _cInfo.output_components;
	width  = _cInfo.output_width;
	height = _cInfo.output_height;
}
inline JPEGReader::~JPEGReader( void )
{
	(void) jpeg_finish_decompress( &_cInfo );
	jpeg_destroy_decompress( &_cInfo );
	fclose( _fp );
}
inline unsigned int JPEGReader::nextRow( unsigned char* row )
{
	JSAMPROW row_pointers[1];
	row_pointers[0] = row;
	jpeg_read_scanlines( &_cInfo , row_pointers, 1 );
	return _currentRow++;
}

inline JPEGWriter::JPEGWriter( const char* fileName , unsigned int width , unsigned int height , unsigned int channels , unsigned int quality )
{
	_currentRow = 0;
	_fp = fopen( fileName , "wb" );
	if( !_fp ) MK_THROW( "Failed to open: " , fileName );

	_cInfo.err = jpeg_std_error( &_jErr.pub );
	jpeg_create_compress( &_cInfo );

	jpeg_stdio_dest( &_cInfo , _fp );

	_cInfo.image_width = width;
	_cInfo.image_height = height;
	_cInfo.input_components = channels;
	_cInfo.in_color_space = JCS_RGB;		/* colorspace of input image */

	jpeg_set_defaults( &_cInfo );
	jpeg_set_quality( &_cInfo , quality , TRUE );

	jpeg_start_compress( &_cInfo , TRUE );
}
inline JPEGWriter::~JPEGWriter( void )
{
	jpeg_finish_compress( &_cInfo );
	jpeg_destroy_compress( &_cInfo );
	fclose( _fp );
}
inline unsigned int JPEGWriter::nextRow( const unsigned char* row )
{
	JSAMPROW row_pointer[1];
	row_pointer[0] = ( unsigned char* )row;
	(void) jpeg_write_scanlines( &_cInfo , row_pointer , 1 );
	return _currentRow++;
}

inline unsigned int JPEGWriter::nextRows( const unsigned char* rows , unsigned int rowNum )
{
	JSAMPROW* row_pointers = new JSAMPROW[ rowNum ];
	for( unsigned int r=0 ; r<rowNum ; r++ ) row_pointers[r] = (unsigned char*)( rows + r * 3 * sizeof( unsigned char ) * _cInfo.image_width );
	(void) jpeg_write_scanlines( &_cInfo , row_pointers , rowNum );
	delete[] row_pointers;
	return _currentRow += rowNum;
}

