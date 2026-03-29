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

#ifndef POISSON_RECON_CLIENT_SERVER_INCLUDED
#define POISSON_RECON_CLIENT_SERVER_INCLUDED


#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include "PointPartition.h"
#include "FEMTree.h"
#include "VertexFactory.h"
#include "Socket.h"
#include "Reconstructors.h"
#include "DataStream.imp.h"

namespace PoissonRecon
{
	namespace PoissonReconClientServer
	{
		template< typename Real , unsigned int Dim >
		struct ClientReconstructionInfo
		{
			enum ShareType
			{
				BACK ,
				CENTER ,
				FRONT
			};
			enum MergeType
			{
				TOPOLOGY_AND_FUNCTION ,		// Identical topology across slice
				FUNCTION ,					// Identical function across slice
				NONE
			};

			std::string header , inDir , tempDir , outDir;
			unsigned int solveDepth , reconstructionDepth , sharedDepth , distributionDepth , baseDepth , kernelDepth , iters , bufferSize , filesPerDir , padSize , verbose;
			Real pointWeight , samplesPerNode , dataX , cgSolverAccuracy , targetValue;
			MergeType mergeType;
			bool density , linearFit , ouputVoxelGrid , outputSolution , gridCoordinates , confidence;
			std::vector< PlyProperty > auxProperties;

			ClientReconstructionInfo( void );
			ClientReconstructionInfo( BinaryStream &stream );

			void write( BinaryStream &stream ) const;

			std::string sharedFile( unsigned int idx , ShareType shareType=CENTER ) const;
		};

		template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
		std::vector< unsigned int > RunServer
		(
			PointPartition::PointSetInfo< Real , Dim > pointSetInfo ,
			PointPartition::Partition pointPartition ,
			std::vector< Socket > clientSockets ,
			ClientReconstructionInfo< Real , Dim > clientReconInfo , 
			unsigned int baseVCycles , 
			unsigned int sampleMS ,
			bool showDiscontinuity=false ,
			bool outputBoundarySlices=false
		);

		template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
		void RunClient( std::vector< Socket > &serverSockets , unsigned int sampleMS );

#include "PoissonReconClientServer.inl"
#include "PoissonRecon.server.inl"
#include "PoissonRecon.client.inl"
	}
}

#endif // POISSON_RECON_CLIENT_SERVER_INCLUDED