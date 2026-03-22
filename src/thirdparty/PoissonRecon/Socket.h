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

#ifndef SOCKET_INCLUDED
#define SOCKET_INCLUDED

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif // _WIN32_WINNT
#endif // _WIN32

#include <iostream>
#include <boost/asio.hpp>
#include <stdarg.h>
#include <thread>
#include "Array.h"
#include "MyMiscellany.h"
#include "Streams.h"

namespace PoissonRecon
{

	static const unsigned int SOCKET_CONNECT_WAIT = 500;		// Default time to wait on a socket (in ms)

	typedef boost::asio::ip::tcp::socket *Socket;
	typedef boost::asio::ip::tcp::acceptor *AcceptorSocket;
	typedef boost::asio::ip::address EndpointAddress;
	const Socket _INVALID_SOCKET_ = (Socket)NULL;
	const AcceptorSocket _INVALID_ACCEPTOR_SOCKET_ = (AcceptorSocket)NULL;
	static boost::asio::io_service io_service;

	template< class C > int socket_receive( Socket &s , C *destination , size_t len )
	{
		boost::system::error_code ec;
		int ret = (int)( boost::asio::read( *s , boost::asio::buffer( destination , len ) , ec ) );
		if( ec ) MK_THROW( "Failed to read from socket" );
		return ret;
	}

	template< class C > int socket_send( Socket& s , const C* source , size_t len )
	{
		boost::system::error_code ec;
		int ret = (int)( boost::asio::write( *s , boost::asio::buffer( source , len ) , ec ) );
		if( ec ) MK_THROW( "Failed to write to socket" );
		return ret;
	}

	inline bool AddressesEqual( const EndpointAddress& a1 ,  const EndpointAddress& a2 ){ return a1.to_string()==a2.to_string(); }
	inline const char *LastSocketError( void ){ return ""; }
	inline void PrintHostAddresses( FILE* fp )
	{
		boost::asio::ip::tcp::resolver resolver( io_service );
		boost::asio::ip::tcp::resolver::query query( boost::asio::ip::host_name() , std::string( "" ) , boost::asio::ip::resolver_query_base::numeric_service );
		boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve( query ) , end;
		for( int count=0 ; iterator!=end ; )
		{
			if( (*iterator).endpoint().address().is_v4() ) fprintf( fp , "%d]  %s\n" , count++ , (*iterator).endpoint().address().to_string().c_str() );
			//		else                                           fprintf( fp , "%d]* %s\n" , count++ , (*iterator).endpoint().address().to_string().c_str() );
			iterator++;
		}
	}

#ifdef ARRAY_DEBUG
	template< class C >
	int socket_receive( Socket& s , Array< C > destination , size_t len )
	{
		if( len>destination.maximum()*sizeof( C ) )
			MK_THROW( "Size of socket_receive exceeds destination maximum: " , len , " > " , destination.maximum()*sizeof( C ) );
		return socket_receive( s , (char*)&destination[0] , len );
	}
	template< class C >
	int socket_send( Socket s , ConstArray< C > source , size_t len )
	{
		if( len>source.maximum()*sizeof( C ) )
			MK_THROW( "Size of socket_send exceeds source maximum: " , len , " > " , source.maximum()*sizeof( C ) );
		return socket_send( s , (char*)&source[0] , len );
	}
#endif // ARRAY_DEBUG

	class ConnectionData
	{
	public:
		EndpointAddress localAddr , peerAddr;
		int localPort , peerPort;
	};


	template< class C > bool ReceiveOnSocket( Socket &s ,      Pointer( C ) data , size_t dataSize );
	template< class C > bool SendOnSocket   ( Socket &s , ConstPointer( C ) data , size_t dataSize );
	template< class C > bool SendOnSocket   ( Socket &s ,      Pointer( C ) data , size_t dataSize );
	template< class C > void ReceiveOnSocket( Socket &s ,      Pointer( C ) data , size_t dataSize , const char *errorMessage , ... );
	template< class C > void SendOnSocket   ( Socket &s , ConstPointer( C ) data , size_t dataSize , const char *errorMessage , ... );
	template< class C > void SendOnSocket   ( Socket &s ,      Pointer( C ) data , size_t dataSize , const char *errorMessage , ... );

	AcceptorSocket GetListenSocket( int& port );
	Socket AcceptSocket( AcceptorSocket listen );
	Socket GetConnectSocket( const char* address , int port , int ms=5 , bool progress=false );
	Socket GetConnectSocket( EndpointAddress , int port , int ms=5 , bool progress=false );
	void CloseSocket( Socket& s );
	void CloseAcceptorSocket( AcceptorSocket& s );
	EndpointAddress GetLocalSocketEndpointAddress( Socket& s );
	int             GetLocalSocketPort           ( Socket& s );
	EndpointAddress GetLocalSocketEndpointAddress( Socket& s );
	int             GetPeerSocketPort            ( Socket& s );
	bool GetHostAddress( char* address , const char* prefix = NULL );
	bool GetHostEndpointAddress( EndpointAddress* address , const char* prefix=NULL );
	void PrintHostAddress( void );

	struct SocketStream : public BinaryStream
	{
		SocketStream( Socket socket=_INVALID_SOCKET_ ) : _socket(socket){}
	protected:
		Socket _socket;
		bool  _read(      Pointer( unsigned char ) ptr , size_t sz ){ return socket_receive( _socket , ptr , sizeof(unsigned char)*sz )==sz; }
		bool _write( ConstPointer( unsigned char ) ptr , size_t sz ){ return socket_send   ( _socket , ptr , sizeof(unsigned char)*sz )==sz; }
	};


#include "Socket.inl"
}
#endif // SOCKET_INCLUDED
