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

/////////////////////////////
// Variadic <-> std::tuple //
/////////////////////////////
template< typename ... Data >
void TupleConverter< Data ... >::FromTuple( const std::tuple< Data ... > &dTuple , Data& ... d ){ return _FromTuple< 0 >( dTuple , d... ); }
template< typename ... Data >
void TupleConverter< Data ... >::ToTuple( std::tuple< Data ... > &dTuple , const Data& ... d ){ return _ToTuple< 0 >( dTuple , d... ); }

template< typename ... Data >
template< unsigned int I , typename _Data , typename ... _Datas >
void TupleConverter< Data ... >::_FromTuple( const std::tuple< Data ... > &dTuple , _Data &d , _Datas& ... ds )
{
	d = std::get< I >( dTuple );
	if constexpr( sizeof...(_Datas) ) _FromTuple< I+1 >( dTuple , ds... );
}

template< typename ... Data >
template< unsigned int I , typename _Data , typename ... _Datas >
void TupleConverter< Data ... >::_ToTuple( std::tuple< Data ... > &dTuple , const _Data &d , const _Datas& ... ds )
{
	std::get< I >( dTuple ) = d;
	if constexpr( sizeof...(_Datas) ) _ToTuple< I+1 >( dTuple , ds... );
}

////////////////////////////
// Variadic <-> DirectSum //
////////////////////////////
template< typename Real , typename ... Data >
void DirectSumConverter< Real , Data ... >::FromDirectSum( const DirectSum< Real , Data ... > &dSum , Data& ... d ){ return _FromDirectSum< 0 >( dSum , d... ); }

template< typename Real , typename ... Data >
void DirectSumConverter< Real , Data ... >::ToDirectSum( DirectSum< Real , Data ... > &dSum , const Data& ... d ){ return _ToDirectSum< 0 >( dSum , d... ); }

template< typename Real , typename ... Data >
template< unsigned int I , typename _Data , typename ... _Datas >
void DirectSumConverter< Real , Data ... >::_FromDirectSum( const DirectSum< Real , Data ... > &dSum , _Data &d , _Datas& ... ds )
{
	d = dSum.template get<I>();
	if constexpr( sizeof...(_Datas) ) _FromDirectSum< I+1 >( dSum , ds... );
}

template< typename Real , typename ... Data >
template< unsigned int I , typename _Data , typename ... _Datas >
void DirectSumConverter< Real , Data ... >::_ToDirectSum( DirectSum< Real , Data ... > &dSum , const _Data &d , const _Datas& ... ds )
{
	dSum.template get<I>() + d;
	if constexpr( sizeof...(_Datas) ) _ToDirectSum< I+1 >( dSum , ds... );
}

////////////////////////
// Input data streams //
////////////////////////
template< typename ... Data >
bool InputDataStream< Data ... >::read( unsigned int thread , Data& ... d )
{
#ifdef SHOW_WARNINGS
	MK_WARN_ONCE( "Serializing read: " , typeid(*this).name() );
#endif // SHOW_WARNINGS
	std::lock_guard< std::mutex > lock( _insertionMutex );
	return read(d...);
}

template< typename ... Data >
bool InputDataStream< Data ... >::read(                       std::tuple< Data ... > &data ){ return _read(          data ); }

template< typename ... Data >
bool InputDataStream< Data ... >::read( unsigned int thread , std::tuple< Data ... > &data ){ return _read( thread , data ); }

template< typename ... Data >
template< typename Real >
bool InputDataStream< Data ... >::read(                       DirectSum< Real , Data ... > &data ){ return _read(          data ); }

template< typename ... Data >
template< typename Real >
bool InputDataStream< Data ... >::read( unsigned int thread , DirectSum< Real , Data ... > &data ){ return _read( thread , data ); }

template< typename ... Data >
template< typename ... _Data >
bool InputDataStream< Data ... >::_read( std::tuple< Data ... > &data , _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return read( _data... );
	else return _read( data , _data... , std::get< sizeof...(_Data) >( data ) );
}

template< typename ... Data >
template< typename ... _Data >
bool InputDataStream< Data ... >::_read( unsigned int thread , std::tuple< Data ... > &data , _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return read( thread , _data... );
	else return _read( thread , data , _data... , std::get< sizeof...(_Data) >( data ) );
}

template< typename ... Data >
template< typename Real , typename ... _Data >
bool InputDataStream< Data ... >::_read( DirectSum< Real , Data ... > &data , _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return read( _data... );
	else return _read( data , _data... , data.template get< sizeof...(_Data) >() );
}

template< typename ... Data >
template< typename Real , typename ... _Data >
bool InputDataStream< Data ... >::_read( unsigned int thread , DirectSum< Real , Data ... > &data , _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return read( thread , _data... );
	else return _read( thread , data , _data... , data.template get< sizeof...(_Data) >() );
}


/////////////////////////
// Output data streams //
/////////////////////////
template< typename ... Data >
size_t OutputDataStream< Data ... >::write( unsigned int thread , const Data& ... d )
{
#ifdef SHOW_WARNINGS
	MK_WARN_ONCE( "Serializing write: " , typeid(*this).name() );
#endif // SHOW_WARNINGS
	std::lock_guard< std::mutex > lock( _insertionMutex );
	return write(d...);
}

template< typename ... Data >
size_t OutputDataStream< Data ... >::write(                       const std::tuple< Data ... > &data ){ return _write(          data ); }

template< typename ... Data >
size_t OutputDataStream< Data ... >::write( unsigned int thread , const std::tuple< Data ... > &data ){ return _write( thread , data ); }

template< typename ... Data >
template< typename Real >
size_t OutputDataStream< Data ... >::write(                       const DirectSum< Real , Data ... > &data ){ return _write(          data ); }

template< typename ... Data >
template< typename Real >
size_t OutputDataStream< Data ... >::write( unsigned int thread , const DirectSum< Real , Data ... > &data ){ return _write( thread , data ); }

template< typename ... Data >
template< typename ... _Data >
size_t OutputDataStream< Data ... >::_write( const std::tuple< Data ... > &data , const _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return write( _data... );
	else return _write( data , _data... , std::get< sizeof...(_Data) >( data ) );
}

template< typename ... Data >
template< typename ... _Data >
size_t OutputDataStream< Data ... >::_write( unsigned int thread , const std::tuple< Data ... > &data , const _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return write( thread , _data... );
	else return _write( thread , data , _data... , std::get< sizeof...(_Data) >( data ) );
}

template< typename ... Data >
template< typename Real , typename ... _Data >
size_t OutputDataStream< Data ... >::_write( const DirectSum< Real , Data ... > &data , const _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return write( _data... );
	else return _write( data , _data... , data.template get< sizeof...(_Data) >() );
}

template< typename ... Data >
template< typename Real , typename ... _Data >
size_t OutputDataStream< Data ... >::_write( unsigned int thread , const DirectSum< Real , Data ... > &data , const _Data& ... _data )
{
	if constexpr( sizeof...(_Data)==sizeof...(Data) ) return write( thread , _data... );
	else return _write( thread , data , _data... , data.template get< sizeof...(_Data) >() );
}

////////////////////////////////
// Multiple input data stream //
////////////////////////////////
template< typename ...Data >
MultiInputDataStream< Data ... >::MultiInputDataStream( InputDataStream< Data ... > **streams , size_t N ) : _current(0) , _streams( streams , streams+N ) {}

template< typename ...Data >
MultiInputDataStream< Data ... >::MultiInputDataStream( const std::vector< InputDataStream< Data ... > * > &streams ) : _current(0) , _streams( streams ) {}

template< typename ...Data >
void MultiInputDataStream< Data ... >::reset( void ){ for( unsigned int i=0 ; i<_streams.size() ; i++ ) _streams[i]->reset(); }

template< typename ...Data >
unsigned int MultiInputDataStream< Data ... >::numStreams( void ) const { return (unsigned int)_streams.size(); }

template< typename ...Data >
bool MultiInputDataStream< Data ... >::read( unsigned int t , Data& ... d ){ return _streams[t]->read(d...); }

template< typename ...Data >
bool MultiInputDataStream< Data ... >::read(                  Data& ... d )
{
	while( _current<_streams.size() )
	{
		if( _streams[_current]->read( d... ) ) return true;
		else _current++;
	}
	return false;
}

/////////////////////////////////
// Multiple output data stream //
/////////////////////////////////

template< typename ... Data >
MultiOutputDataStream< Data ... >::MultiOutputDataStream( OutputDataStream< Data ... > **streams , size_t N ) : _size(0) , _streams( streams , streams+N ) {}

template< typename ... Data >
MultiOutputDataStream< Data ... >::MultiOutputDataStream( const std::vector< OutputDataStream< Data ... > * > &streams ) : _size(0) , _streams( streams ) {}

template< typename ... Data >
unsigned int MultiOutputDataStream< Data ... >::numStreams( void ) const { return (unsigned int)_streams.size(); }

template< typename ... Data >
size_t MultiOutputDataStream< Data ... >::write(                  const Data& ... d ){ size_t idx = _size++ ; _streams[0]->write(d...) ; return idx; }

template< typename ... Data >
size_t MultiOutputDataStream< Data ... >::write( unsigned int t , const Data& ... d ){ size_t idx = _size++ ; _streams[t]->write(d...) ; return idx; }

////////////////////////////////////////////////
// De-interleaved multiple output data stream //
////////////////////////////////////////////////

template< typename ... Data >
DeInterleavedMultiOutputIndexedDataStream< Data ... >::DeInterleavedMultiOutputIndexedDataStream( OutputIndexedDataStream< Data ... > **streams , size_t N ) : _size(0) , _multiStream( streams , streams+N ) {}

template< typename ... Data >
DeInterleavedMultiOutputIndexedDataStream< Data ... >::DeInterleavedMultiOutputIndexedDataStream( const std::vector< OutputIndexedDataStream<  Data ... > * > &streams ) : _size(0) , _multiStream( streams ) {}

template< typename ... Data >
unsigned int DeInterleavedMultiOutputIndexedDataStream< Data ... >::numStreams( void ) const { return _multiStream.numStreams(); }

template< typename ... Data >
size_t DeInterleavedMultiOutputIndexedDataStream< Data ... >::write(                       const Data& ... d ){ size_t idx = _size++ ; _multiStream.write(          idx , d... ) ; return idx; }

template< typename ... Data >
size_t DeInterleavedMultiOutputIndexedDataStream< Data ... >::write( unsigned int thread , const Data& ... d ){ size_t idx = _size++ ; _multiStream.write( thread , idx , d... ) ; return idx; }

/////////////////////////////////
// Input data stream converter //
/////////////////////////////////
template< typename ... IData , typename ... XData >
InputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::InputDataStreamConverter( InputDataStream< IData ... > &stream , std::function< std::tuple< XData ... > ( const std::tuple< IData ... >& ) > converter , IData ... zero )
	: _stream(stream) , _converter(converter) , _iZero( std::make_tuple( zero... ) ) {}

template< typename ... IData , typename ... XData >
void InputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::reset( void ){ _stream.reset(); }

template< typename ... IData , typename ... XData >
bool InputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::read( XData& ... xData )
{
	std::tuple< IData ... > _iData = _iZero;
	if( !_stream.read( _iData ) ) return false;
	std::tuple< XData ... > _xData = _converter( _iData );
	TupleConverter< XData ... >::FromTuple( _xData , xData ... );
	return true;
}

template< typename ... IData , typename ... XData >
bool InputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::read( unsigned int thread , XData& ... xData )
{
	std::tuple< IData ... > _iData = _iZero;
	if( !_stream.read( thread , _iData ) ) return false;
	std::tuple< XData ... > _xData = _converter( _iData );
	TupleConverter< XData ... >::FromTuple( _xData , xData ... );
	return true;
}

//////////////////////////////////
// Output data stream converter //
//////////////////////////////////
template< typename ... IData , typename ... XData >
OutputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::OutputDataStreamConverter( OutputDataStream< IData ... > &stream , std::function< std::tuple< IData ... > ( const std::tuple< XData ... >& ) > converter )
	: _stream(stream) , _converter(converter) {}

template< typename ... IData , typename ... XData >
size_t OutputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::write( const XData& ... xData )
{
	std::tuple< XData ... > _xData;
	TupleConverter< XData ... >::ToTuple( _xData , xData ... );
	std::tuple< IData ... > _iData = _converter( _xData );
	return _stream.write( _iData );
}

template< typename ... IData , typename ... XData >
size_t OutputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > >::write( unsigned int thread , const XData& ... xData )
{
	std::tuple< XData ... > _xData;
	TupleConverter< XData ... >::ToTuple( _xData , xData ... );
	std::tuple< IData ... > _iData = _converter( _xData );
	return _stream.write( thread , _iData );
}


////////////////////////////////////////////
// Interleaved multiple input data stream //
////////////////////////////////////////////

template< typename ... Data >
InterleavedMultiInputIndexedDataStream< Data ... >::InterleavedMultiInputIndexedDataStream( InputIndexedDataStream< Data ... > **streams , size_t N )
	: _multiStream( streams , N )
	, _firstTime(true)
{
	_nextValues.resize( _multiStream.numStreams() );
}

template< typename ... Data >
InterleavedMultiInputIndexedDataStream< Data ... >::InterleavedMultiInputIndexedDataStream( const std::vector< InputIndexedDataStream< Data ... > * > &streams )
	: _multiStream( streams )
	, _firstTime(true)
{
	_nextValues.resize( _multiStream.numStreams() );
}

template< typename ... Data >
void InterleavedMultiInputIndexedDataStream< Data ... >::reset( void ){ _multiStream.reset() , _firstTime = true; }

template< typename ... Data >
bool InterleavedMultiInputIndexedDataStream< Data ... >::_NextValue::Compare( const _NextValue &v1 , const _NextValue &v2 )
{
	if( !v2.validData ) return false;
	else if( !v1.validData && v2.validData ) return true;
	else return std::get<0>( v1.data ) > std::get<0>( v2.data );
}

template< typename ... Data >
void InterleavedMultiInputIndexedDataStream< Data ... >::_init( const std::tuple< Data ... > &data )
{
	std::tuple< size_t , Data ... > _data = std::tuple_cat( std::make_tuple( (size_t)0 ) , data );
	for( unsigned int i=0 ; i<_nextValues.size() ; i++ ) _nextValues[i].data = _data;
	for( unsigned int i=0 ; i<_nextValues.size() ; i++ )
	{
		_nextValues[i].validData = _multiStream.read( i , _nextValues[i].data );
		_nextValues[i].streamIndex = i;
	}
	std::make_heap( _nextValues.begin() , _nextValues.end() , _NextValue::Compare );
}

template< typename ... Data >
bool InterleavedMultiInputIndexedDataStream< Data ... >::read( Data& ... d )
{
	if( _firstTime ) _init(d...) , _firstTime = false;
	std::pop_heap( _nextValues.begin() , _nextValues.end() , _NextValue::Compare );
	_NextValue &next = _nextValues.back();
	if( !next.validData ) return false;
	size_t sz;
	TupleConverter< size_t , Data ... >::FromTuple( next.data , sz , d... );

	next.validData = _multiStream.read( next.streamIndex , next.data );

	std::push_heap( _nextValues.begin() , _nextValues.end() , _NextValue::Compare );
	return true;
}