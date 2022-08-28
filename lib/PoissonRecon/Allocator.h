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

#ifndef ALLOCATOR_INCLUDED
#define ALLOCATOR_INCLUDED
#include <vector>

class AllocatorState{
public:
	int index,remains;
};
/** This templated class assists in memory allocation and is well suited for instances
  * when it is known that the sequence of memory allocations is performed in a stack-based
  * manner, so that memory allocated last is released first. It also preallocates memory
  * in chunks so that multiple requests for small chunks of memory do not require separate
  * system calls to the memory manager.
  * The allocator is templated off of the class of objects that we would like it to allocate,
  * ensuring that appropriate constructors and destructors are called as necessary.
  */
template<class T>
class Allocator
{
	int blockSize;
	int index , remains;
	std::vector< T* > memory;
public:
	Allocator( void ){ blockSize = index = remains = 0; }
	~Allocator( void ){ reset(); }

	/** This method is the allocators destructor. It frees up any of the memory that
	  * it has allocated. */
	void reset( void )
	{
		for(size_t i=0;i<memory.size();i++){delete[] memory[i];}
		memory.clear();
		blockSize=index=remains=0;
	}
	/** This method returns the memory state of the allocator. */
	AllocatorState getState( void ) const
	{
		AllocatorState s;
		s.index=index;
		s.remains=remains;
		return s;
	}


	/** This method rolls back the allocator so that it makes all of the memory previously
	  * allocated available for re-allocation. Note that it does it not call the constructor
	  * again, so after this method has been called, assumptions about the state of the values
	  * in memory are no longer valid. */
	void rollBack(void)
	{
		if( memory.size() )
		{
			for( size_t i=0 ; i<memory.size() ; i++ )
			{
				for( int j=0 ; j<blockSize ; j++ )
				{
					memory[i][j].~T();
					new(&memory[i][j]) T();
				}
			}
			index=0;
			remains=blockSize;
		}
	}
	/** This method rolls back the allocator to the previous memory state and makes all of the memory previously
	  * allocated available for re-allocation. Note that it does it not call the constructor
	  * again, so after this method has been called, assumptions about the state of the values
	  * in memory are no longer valid. */
	void rollBack(const AllocatorState& state){
		if(state.index<index || (state.index==index && state.remains<remains)){
			if(state.index<index){
				for(int j=state.remains;j<blockSize;j++){
					memory[state.index][j].~T();
					new(&memory[state.index][j]) T();
				}
				for(int i=state.index+1;i<index-1;i++){
					for(int j=0;j<blockSize;j++){
						memory[i][j].~T();
						new(&memory[i][j]) T();
					}
				}
				for(int j=0;j<remains;j++){
					memory[index][j].~T();
					new(&memory[index][j]) T();
				}
				index=state.index;
				remains=state.remains;
			}
			else{
				for(int j=0;j<state.remains;j<remains){
					memory[index][j].~T();
					new(&memory[index][j]) T();
				}
				remains=state.remains;
			}
		}
	}

	/** This method initiallizes the constructor and the blockSize variable specifies the
	  * the number of objects that should be pre-allocated at a time. */
	void set( int blockSize)
	{
		reset();
		this->blockSize = blockSize;
		index=-1;
		remains=0;
	}

	/** This method returns a pointer to an array of elements objects. If there is left over pre-allocated
	  * memory, this method simply returns a pointer to the next free piece of memory, otherwise it pre-allocates
	  * more memory. Note that if the number of objects requested is larger than the value blockSize with which
	  * the allocator was initialized, the request for memory will fail.
	  */
	T* newElements( int elements=1 )
	{
		T* mem;
		if( !elements ) return NULL;
		if( elements>blockSize ) fprintf( stderr , "[ERROR] Allocator: elements bigger than block-size: %d>%d\n" , elements , blockSize ) , exit( 0 );
		if( remains<elements )
		{
			if( index==memory.size()-1 )
			{
				mem = new T[blockSize];
				if( !mem ) fprintf( stderr , "[ERROR] Failed to allocate memory\n" ) , exit(0);
				memory.push_back( mem );
			}
			index++;
			remains=blockSize;
		}
		mem = &(memory[index][blockSize-remains]);
		remains -= elements;
		return mem;
	}
};
#endif // ALLOCATOR_INCLUDE
