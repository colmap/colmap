/*
Copyright (c) 2023, Michael Kazhdan
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

#ifndef MERGE_PLY_CLIENT_SERVER_INCLUDED
#define MERGE_PLY_CLIENT_SERVER_INCLUDED

#include <string>
#include <functional>
#include "Socket.h"
#include "MyMiscellany.h"
#include "CmdLineParser.h"

namespace PoissonRecon
{

	namespace MergePlyClientServer
	{
		struct ClientMergePlyInfo
		{
			std::vector< PlyProperty > auxProperties;
			size_t bufferSize;
			bool keepSeparate , verbose;

			ClientMergePlyInfo( void );
			ClientMergePlyInfo( BinaryStream &stream );

			void write( BinaryStream &stream ) const;
		};

		template< typename Real , unsigned int Dim >
		void RunServer
		(
			std::string inDir , 
			std::string tempDir ,
			std::string header ,
			std::string out ,
			std::vector< Socket > &clientSockets ,
			const std::vector< unsigned int > &sharedVertexCounts ,
			ClientMergePlyInfo clientMergePlyInfo ,
			unsigned int sampleMS ,
			std::function< std::vector< std::string > (unsigned int) > commentFunctor=[](unsigned int){ return std::vector< std::string >(); }
		);

		template< typename Real , unsigned int Dim >
		void RunClients( std::vector< Socket > &serverSockets , unsigned int sampleMS );

#include "MergePlyClientServer.inl"
	}
}

#endif // MERGE_PLY_CLIENT_SERVER_INCLUDED