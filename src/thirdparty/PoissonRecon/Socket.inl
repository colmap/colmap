/*
Copyright (c) 2008, Michael Kazhdan
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

template<class C>
bool ReceiveOnSocket( Socket& s , Pointer( C ) data , size_t dataSize )
{
#ifdef ARRAY_DEBUG
	if( dataSize>data.maximum()*sizeof( C ) ) MK_THROW( "Size of socket read exceeds source maximum: " , dataSize , " > " , data.maximum()*sizeof( C ) );
#endif // ARRAY_DEBUG
	unsigned long long rec=0;
	while( rec!=dataSize )
	{
		int tmp = socket_receive( s , ( ( Pointer( char ) ) data) + rec , dataSize-rec );
		if( tmp<=0 )
		{
			if( !tmp ) MK_THROW( "Connection Closed" );
			else       MK_THROW( "socket_receive from client failed: " , LastSocketError() );
			return false;
		}
		rec+=tmp;
	}
	return true;
}

template<class C>
bool SendOnSocket( Socket& s , ConstPointer( C ) data , size_t dataSize )
{
#ifdef ARRAY_DEBUG
	if( dataSize>data.maximum()*sizeof( C ) ) MK_THROW( "Size of socket write exceeds source maximum: " , dataSize , " > " , data.maximum()*sizeof( C ) );
#endif // ARRAY_DEBUG
	if( socket_send( s , ( ConstPointer( char ) )data , dataSize )<0 )
	{
		MK_THROW( "socket_send to client failed (" , s , "): " , LastSocketError() );
		return false;
	}
	return true;
}

template<class C>
bool SendOnSocket( Socket& s , Pointer( C ) data , size_t dataSize ){ return SendOnSocket( ( ConstPointer( C ) )data , dataSize ); }

template<class C>
void ReceiveOnSocket( Socket& s , Pointer( C ) data , size_t dataSize , const char* errorMessage , ... )
{
#ifdef ARRAY_DEBUG
	if( dataSize>data.maximum()*sizeof( C ) ) MK_THROW( "Size of socket read exceeds source maximum: " , dataSize , " > " , data.maximum()*sizeof( C ) );
#endif // ARRAY_DEBUG
	unsigned long long rec=0;
	while( rec!=dataSize )
	{
		int tmp = socket_receive( s , ( ( Pointer( char ) ) data) + rec , dataSize-rec );
		if( tmp<=0 )
		{
			if( !tmp ) MK_THROW( "Connection Closed" );
			else       MK_THROW( "socket_receive from client failed: " , LastSocketError() );
			{
				fprintf( stderr , "\t" );
				va_list args;
				va_start( args , errorMessage );
				vfprintf( stderr , errorMessage , args );
				va_end( args );
				fprintf( stderr , "\n" );
			}
			exit( EXIT_FAILURE );
		}
		rec+=tmp;
	}
}

template<class C>
void SendOnSocket( Socket& s , ConstPointer( C ) data , size_t dataSize , const char* errorMessage , ... )
{
#ifdef ARRAY_DEBUG
	if( dataSize>data.maximum()*sizeof( C ) ) MK_THROW( "Size of socket write exceeds source maximum: " , dataSize , " > " , data.maximum()*sizeof( C ) );
#endif // ARRAY_DEBUG
	if( socket_send( s , ( ConstPointer( char ) )data , dataSize )<0 )
		MK_THROW( "socket_send to client failed: " , LastSocketError() );
}

template<class C>
void SendOnSocket( Socket& s , Pointer( C ) data , size_t dataSize , const char* errorMessage , ... )
{
#ifdef ARRAY_DEBUG
	if( dataSize>data.maximum()*sizeof( C ) ) MK_THROW( "Size of socket write exceeds source maximum: " , dataSize , " > " , data.maximum()*sizeof( C ) );
#endif // ARRAY_DEBUG
	if( socket_send( s , ( ConstPointer( char ) )data , dataSize )<0 )
		MK_THROW( "socket_send to client failed: " , LastSocketError() );
}

inline bool GetHostEndpointAddress( EndpointAddress* address , const char* prefix )
{
	boost::asio::ip::tcp::resolver resolver( io_service );
	boost::asio::ip::tcp::resolver::query query( boost::asio::ip::host_name() , std::string( "" ) , boost::asio::ip::resolver_query_base::numeric_service );
	boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve( query ) , end;
	for( int count=0 ; iterator!=end ; )
	{
		if( (*iterator).endpoint().address().is_v4() )
		{
			std::string addrss_string = (*iterator).endpoint().address().to_string();
			const char* _address = addrss_string.c_str();
			if( !prefix || strstr( _address , prefix ) )
			{
				*address = (*iterator).endpoint().address();
				return true;
			}
		}
		iterator++;
	}
	return false;
}

inline bool GetHostAddress( char* address , const char* prefix )
{
	EndpointAddress _address;
	if( !GetHostEndpointAddress( &_address , prefix ) ) return false;
	strcpy( address , _address.to_string().c_str() );
	return true;
}

inline int GetLocalSocketPort( Socket& s )
{
	return s->local_endpoint().port();
}

inline EndpointAddress GetLocalSocketEndpointAddress( Socket& s )
{
	return s->local_endpoint().address();
}

inline int GetPeerSocketPort( Socket& s )
{
	return s->remote_endpoint().port();
}

inline EndpointAddress GetPeerSocketEndpointAddress( Socket& s )
{
	return s->remote_endpoint().address();
}

inline Socket GetConnectSocket( const char* address , int port , int ms , bool progress )
{
	char _port[128];
	sprintf( _port , "%d" , port );
	boost::asio::ip::tcp::resolver resolver( io_service );
	boost::asio::ip::tcp::resolver::query query( address , _port );
	boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve( query );
	Socket s = new boost::asio::ip::tcp::socket( io_service );
	boost::system::error_code ec;
	long long sleepCount = 0;
	do
	{
		boost::asio::connect( *s , resolver.resolve(query) , ec );
		sleepCount++;
		std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
		if( progress && !(sleepCount%ms) ) printf( "." );
	}
	while( ec );
	if( progress ) printf( "\n" ) , fflush( stdout );
	return s;
}

inline Socket GetConnectSocket( EndpointAddress address , int port , int ms , bool progress )
{
	char _port[128];
	sprintf( _port , "%d" , port );
	boost::asio::ip::tcp::resolver resolver( io_service );
	boost::asio::ip::tcp::resolver::query query( address.to_string().c_str() , _port );
	boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve( query );
	Socket s = new boost::asio::ip::tcp::socket( io_service );
	boost::system::error_code ec;
	long long sleepCount = 0;
	do
	{
		boost::asio::connect( *s , resolver.resolve(query) , ec );
		sleepCount++;
		std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
		if( progress && !(sleepCount%ms) ) std::cout << ".";
	}
	while( ec );
	if( progress ) std::cout << std::endl;
	return s;
}

inline Socket AcceptSocket( AcceptorSocket listen )
{
	Socket s = new boost::asio::ip::tcp::socket( io_service );
	listen->accept( *s );
	return s;
}

inline AcceptorSocket GetListenSocket( int &port )
{
	AcceptorSocket s = new boost::asio::ip::tcp::acceptor( io_service , boost::asio::ip::tcp::endpoint( boost::asio::ip::tcp::v4() , port ) );
	port = s->local_endpoint().port();
	return s;
}

inline void CloseSocket( Socket& s )
{
	delete s;
	s = _INVALID_SOCKET_;
}

inline void CloseAcceptorSocket( AcceptorSocket& s )
{
	delete s;
	s = _INVALID_ACCEPTOR_SOCKET_;
}
