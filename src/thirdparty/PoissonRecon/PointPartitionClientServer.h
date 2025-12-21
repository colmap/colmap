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

#ifndef POINT_PARTITION_CLIENT_SERVER_INCLUDED
#define POINT_PARTITION_CLIENT_SERVER_INCLUDED

#include <string>
#include "PointPartition.h"
#include "Socket.h"
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "VertexFactory.h"
#include "Reconstructors.h"
#include "PointExtent.h"

namespace PoissonRecon
{
	namespace PointPartitionClientServer
	{
		template< typename Real >
		struct ClientPartitionInfo
		{
			std::string in , tempDir , outDir , outHeader;
			unsigned int slabs , filesPerDir , bufferSize , clientCount , sliceDir;
			Real scale;
			bool verbose;

			ClientPartitionInfo( void );
			ClientPartitionInfo( BinaryStream &stream );

			void write( BinaryStream &stream ) const;
		};


		template< typename Real , unsigned int Dim >
		std::pair< PointPartition::PointSetInfo< Real , Dim > , PointPartition::Partition > RunServer
		(
			std::vector< Socket > &clientSockets ,
			ClientPartitionInfo< Real > clientPartitionInfo ,
			bool loadBalance
		);

		template< typename Real , unsigned int Dim >
		void RunClients( std::vector< Socket > &serverSockets );

#include "PointPartitionClientServer.inl"
	}
}


#endif // POINT_PARTITION_CLIENT_SERVER_INCLUDED