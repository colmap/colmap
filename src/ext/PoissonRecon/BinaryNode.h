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

#ifndef BINARY_NODE_INCLUDED
#define BINARY_NODE_INCLUDED

class BinaryNode
{
public:
	static inline int CenterCount( int depth ) { return  1<<depth; }
	static inline int CornerCount( int depth ) { return (1<<depth)+1; }
	static inline int CumulativeCenterCount( int maxDepth ) { return (1<<(maxDepth+1))-1; }
	static inline int CumulativeCornerCount( int maxDepth ) { return (1<<(maxDepth+1))+maxDepth; }
	static inline int CenterIndex( int depth , int offSet ) { return (1<<depth)+offSet-1; }
	static inline int CornerIndex( int depth , int offSet ) { return (1<<depth)+offSet-1+depth; }

	static inline int CornerIndex( int maxDepth , int depth , int offSet , int forwardCorner ){ return (offSet+forwardCorner)<<(maxDepth-depth); }
	template< class Real > static inline Real Width( int depth ){ return Real(1.0/(1<<depth)); }
	template< class Real > static inline void CenterAndWidth( int depth , int offset , Real& center , Real& width ){ width = Real (1.0/(1<<depth) ) , center = Real((0.5+offset)*width); }
	template< class Real > static inline void CornerAndWidth( int depth , int offset , Real& corner , Real& width ){ width = Real(1.0/(1<<depth) ) , corner = Real(offset*width); }
	template< class Real > static inline void CenterAndWidth( int idx , Real& center , Real& width )
	{
		int depth , offset;
		CenterDepthAndOffset( idx , depth , offset );
		CenterAndWidth( depth , offset , center , width );
	}
	template< class Real > static inline void CornerAndWidth( int idx , Real& corner , Real& width )
	{
		int depth , offset;
		CornerDepthAndOffset( idx , depth , offset );
		CornerAndWidth( depth , offset , corner , width );
	}
	static inline void CenterDepthAndOffset( int idx , int& depth , int& offset )
	{
		offset = idx , depth = 0;
		while( offset>=(1<<depth) ) offset -= (1<<depth) , depth++;
	}
	static inline void CornerDepthAndOffset( int idx , int& depth , int& offset )
	{
		offset = idx , depth = 0;
		while( offset>=( (1<<depth)+1 ) ) offset -= ( (1<<depth)+1 ) , depth++;
	}
};

#endif // BINARY_NODE_INCLUDED
