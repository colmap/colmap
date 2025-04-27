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

#include <float.h>
#include <string.h>


///////////////////
//  SparseMatrix //
///////////////////
///////////////////////////////////////
// SparseMatrix Methods and Memebers //
///////////////////////////////////////

template< class T >
void SparseMatrix< T >::_init( void )
{
	_contiguous = false;
	_maxEntriesPerRow = 0;
	rows = 0;
	rowSizes = NullPointer( int );
	m_ppElements = NullPointer( Pointer( MatrixEntry< T > ) );
}

template< class T > SparseMatrix< T >::SparseMatrix( void ){  _init(); }

template< class T > SparseMatrix< T >::SparseMatrix( int rows                        ){ _init() , Resize( rows ); }
template< class T > SparseMatrix< T >::SparseMatrix( int rows , int maxEntriesPerRow ){ _init() , Resize( rows , maxEntriesPerRow ); }

template< class T >
SparseMatrix< T >::SparseMatrix( const SparseMatrix& M )
{
	_init();
	if( M._contiguous ) Resize( M.rows , M._maxEntriesPerRow );
	else                Resize( M.rows );
	for( int i=0 ; i<rows ; i++ )
	{
		SetRowSize( i , M.rowSizes[i] );
		memcpy( (*this)[i] , M[i] , sizeof( MatrixEntry< T > ) * rowSizes[i] );
	}
}
template<class T>
int SparseMatrix<T>::Entries( void ) const
{
	int e = 0;
	for( int i=0 ; i<rows ; i++ ) e += int( rowSizes[i] );
	return e;
}
template<class T>
SparseMatrix<T>& SparseMatrix<T>::operator = (const SparseMatrix<T>& M)
{
	if( M._contiguous ) Resize( M.rows , M._maxEntriesPerRow );
	else                Resize( M.rows );
	for( int i=0 ; i<rows ; i++ )
	{
		SetRowSize( i , M.rowSizes[i] );
		memcpy( (*this)[i] , M[i] , sizeof( MatrixEntry< T > ) * rowSizes[i] );
	}
	return *this;
}

template<class T>
SparseMatrix<T>::~SparseMatrix( void ){ Resize( 0 ); }

template< class T >
bool SparseMatrix< T >::write( const char* fileName ) const
{
	FILE* fp = fopen( fileName , "wb" );
	if( !fp ) return false;
	bool ret = write( fp );
	fclose( fp );
	return ret;
}
template< class T >
bool SparseMatrix< T >::read( const char* fileName )
{
	FILE* fp = fopen( fileName , "rb" );
	if( !fp ) return false;
	bool ret = read( fp );
	fclose( fp );
	return ret;
}
template< class T >
bool SparseMatrix< T >::write( FILE* fp ) const
{
	if( fwrite( &rows , sizeof( int ) , 1 , fp )!=1 ) return false;
	if( fwrite( rowSizes , sizeof( int ) , rows , fp )!=rows ) return false;
	for( int i=0 ; i<rows ; i++ ) if( fwrite( (*this)[i] , sizeof( MatrixEntry< T > ) , rowSizes[i] , fp )!=rowSizes[i] ) return false;
	return true;
}
template< class T >
bool SparseMatrix< T >::read( FILE* fp )
{
	int r;
	if( fread( &r , sizeof( int ) , 1 , fp )!=1 ) return false;
	Resize( r );
	if( fread( rowSizes , sizeof( int ) , rows , fp )!=rows ) return false;
	for( int i=0 ; i<rows ; i++ )
	{
		r = rowSizes[i];
		rowSizes[i] = 0;
		SetRowSize( i , r );
		if( fread( (*this)[i] , sizeof( MatrixEntry< T > ) , rowSizes[i] , fp )!=rowSizes[i] ) return false;
	}
	return true;
}


template< class T >
void SparseMatrix< T >::Resize( int r )
{
	if( rows>0 )
	{
		if( _contiguous ){ if( _maxEntriesPerRow ) FreePointer( m_ppElements[0] ); }
		else for( int i=0 ; i<rows ; i++ ){ if( rowSizes[i] ) FreePointer( m_ppElements[i] ); }
		FreePointer( m_ppElements );
		FreePointer( rowSizes );
	}
	rows = r;
	if( r )
	{
		rowSizes = AllocPointer< int >( r );
		m_ppElements = AllocPointer< Pointer( MatrixEntry< T > ) >( r );
		memset( rowSizes , 0 , sizeof( int ) * r );
	}
	_contiguous = false;
	_maxEntriesPerRow = 0;
}
template< class T >
void SparseMatrix< T >::Resize( int r , int e )
{
	if( rows>0 )
	{
		if( _contiguous ){ if( _maxEntriesPerRow ) FreePointer( m_ppElements[0] ); }
		else for( int i=0 ; i<rows ; i++ ){ if( rowSizes[i] ) FreePointer( m_ppElements[i] ); }
		FreePointer( m_ppElements );
		FreePointer( rowSizes );
	}
	rows = r;
	if( r )
	{
		rowSizes = AllocPointer< int >( r );
		m_ppElements = AllocPointer< Pointer( MatrixEntry< T > ) >( r );
		m_ppElements[0] = AllocPointer< MatrixEntry< T > >( r * e );
		memset( rowSizes , 0 , sizeof( int ) * r );
		for( int i=1 ; i<r ; i++ ) m_ppElements[i] = m_ppElements[i-1] + e;
	}
	_contiguous = true;
	_maxEntriesPerRow = e;
}

template<class T>
void SparseMatrix< T >::SetRowSize( int row , int count )
{
	if( _contiguous )
	{
		if( count>_maxEntriesPerRow ) fprintf( stderr , "[ERROR] Cannot set row size on contiguous matrix: %d<=%d\n" , count , _maxEntriesPerRow ) , exit( 0 );
		rowSizes[row] = count;
	}
	else if( row>=0 && row<rows )
	{
		if( rowSizes[row] ) FreePointer( m_ppElements[row] );
		if( count>0 ) m_ppElements[row] = AllocPointer< MatrixEntry< T > >( count );
		// [WARNING] Why wasn't this line here before???
		rowSizes[row] = count;
	}
}


template<class T>
void SparseMatrix<T>::SetZero()
{
	Resize(this->rows, this->_maxEntriesPerRow);
}

template<class T>
SparseMatrix<T> SparseMatrix<T>::operator * (const T& V) const
{
	SparseMatrix<T> M(*this);
	M *= V;
	return M;
}

template<class T>
SparseMatrix<T>& SparseMatrix<T>::operator *= (const T& V)
{
	for( int i=0 ; i<rows ; i++ ) for( int ii=0 ; ii<rowSizes[i] ; i++ ) m_ppElements[i][ii].Value *= V;
	return *this;
}

template< class T >
template< class T2 >
void SparseMatrix< T >::Multiply( ConstPointer( T2 ) in , Pointer( T2 ) out , int threads ) const
{
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<rows ; i++ )
	{
		T2 _out(0);
		ConstPointer( MatrixEntry< T > ) start = m_ppElements[i];
		ConstPointer( MatrixEntry< T > ) end = start + rowSizes[i];
		ConstPointer( MatrixEntry< T > ) e;
		for( e=start ; e!=end ; e++ ) _out += in[ e->N ] * e->Value;
		out[i] = _out;
	}
}
template< class T >
template< class T2 >
void SparseMatrix< T >::MultiplyAndAddAverage( ConstPointer( T2 ) in , Pointer( T2 ) out , int threads ) const
{
	T2 average = 0;
	for( int i=0 ; i<rows ; i++ ) average += in[i];
	average /= rows;
	Multiply( in , out , threads );
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<rows ; i++ ) out[i] += average;
}


template< class T >
template< class T2 >
int SparseMatrix<T>::SolveJacobi( const SparseMatrix<T>& M , ConstPointer( T2 ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , Pointer( T2 ) Mx , T2 sor , int threads )
{
	M.Multiply( x , Mx , threads );
#if ZERO_TESTING_JACOBI
	for( int j=0 ; j<int(M.rows) ; j++ ) if( diagonal[j] ) x[j] += ( b[j]-Mx[j] ) * sor / diagonal[j];
#else // !ZERO_TESTING_JACOBI
	for( int j=0 ; j<int(M.rows) ; j++ ) x[j] += ( b[j]-Mx[j] ) * sor / diagonal[j];
#endif // ZERO_TESTING_JACOBI
	return M.rows;
}
template< class T >
template< class T2 >
int SparseMatrix<T>::SolveJacobi( const SparseMatrix<T>& M , ConstPointer( T2 ) b , Pointer( T2 ) x , Pointer( T2 ) Mx , T2 sor , int threads )
{
	M.Multiply( x , Mx , threads );
#if ZERO_TESTING_JACOBI
	for( int j=0 ; j<int(M.rows) ; j++ )
	{
		T diagonal = M[j][0].Value;
		if( diagonal ) x[j] += ( b[j]-Mx[j] ) * sor / diagonal;
	}
#else // !ZERO_TESTING_JACOBI
	for( int j=0 ; j<int(M.rows) ; j++ ) x[j] += ( b[j]-Mx[j] ) * sor / M[j][0].Value;
#endif // ZERO_TESTING_JACOBI
	return M.rows;
}
template<class T>
template<class T2>
int SparseMatrix<T>::SolveGS( const SparseMatrix<T>& M , ConstPointer( T2 ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward )
{
#define ITERATE                                                         \
	{                                                                   \
		ConstPointer( MatrixEntry< T > ) start = M[j];                  \
		ConstPointer( MatrixEntry< T > ) end = start + M.rowSizes[j];   \
		ConstPointer( MatrixEntry< T > ) e;                             \
		T2 _b = b[j];                                                   \
		for( e=start ; e!=end ; e++ ) _b -= x[ e->N ] * e->Value;       \
		x[j] += _b / diagonal[j];                                       \
	}

#if ZERO_TESTING_JACOBI
	if( forward ) for( int j=0 ; j<int(M.rows)    ; j++ ){ if( diagonal[j] ){ ITERATE; } }
	else          for( int j=int(M.rows)-1 ; j>=0 ; j-- ){ if( diagonal[j] ){ ITERATE; } }
#else // !ZERO_TESTING_JACOBI
	if( forward ) for( int j=0 ; j<int(M.rows) ; j++ ){ ITERATE; }
	else          for( int j=int(M.rows)-1 ; j>=0 ; j-- ){ ITERATE; }
#endif // ZERO_TESTING_JACOBI
#undef ITERATE
	return M.rows;
}
template<class T>
template<class T2>
int SparseMatrix<T>::SolveGS( const std::vector< std::vector< int > >& mcIndices , const SparseMatrix<T>& M , ConstPointer( T2 ) diagonal , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward , int threads )
{
	int sum=0;
#ifdef _WIN32
#define SetOMPParallel __pragma( omp parallel for num_threads( threads ) )
#else // !_WIN32
#define SetOMPParallel _Pragma( "omp parallel for num_threads( threads )" )
#endif // _WIN32
#if ZERO_TESTING_JACOBI
#define ITERATE( indices )                                                        \
	{                                                                             \
SetOMPParallel                                                                    \
		for( int k=0 ; k<int( indices.size() ) ; k++ ) if( diagonal[indices[k]] ) \
		{                                                                         \
			int jj = indices[k];                                                  \
			ConstPointer( MatrixEntry< T > ) start = M[jj];                       \
			ConstPointer( MatrixEntry< T > ) end = start + M.rowSizes[jj];        \
			ConstPointer( MatrixEntry< T > ) e;                                   \
			T2 _b = b[jj];                                                        \
			for( e=start ; e!=end ; e++ ) _b -= x[ e->N ] * e->Value;             \
			x[jj] += _b / diagonal[jj];                                           \
		}                                                                         \
	}
#else // !ZERO_TESTING_JACOBI
#define ITERATE( indices )                                                  \
	{                                                                       \
SetOMPParallel                                                              \
		for( int k=0 ; k<int( indices.size() ) ; k++ )                      \
		{                                                                   \
			int jj = indices[k];                                            \
			ConstPointer( MatrixEntry< T > ) start = M[jj];                 \
			ConstPointer( MatrixEntry< T > ) end = start + M.rowSizes[jj];  \
			ConstPointer( MatrixEntry< T > ) e;                             \
			T2 _b = b[jj];                                                  \
			for( e=start ; e!=end ; e++ ) _b -= x[ e->N ] * e->Value;       \
			x[jj] += _b / diagonal[jj];                                     \
		}                                                                   \
	}
#endif // ZERO_TESTING_JACOBI
	if( forward ) for( int j=0 ; j<mcIndices.size()  ; j++ ){ sum += int( mcIndices[j].size() ) ; ITERATE( mcIndices[j] ); }
	else for( int j=int( mcIndices.size() )-1 ; j>=0 ; j-- ){ sum += int( mcIndices[j].size() ) ; ITERATE( mcIndices[j] ); }
#undef ITERATE
#undef SetOMPParallel
	return sum;
}
template<class T>
template<class T2>
int SparseMatrix<T>::SolveGS( const SparseMatrix<T>& M , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward )
{
	int start = forward ? 0 : M.rows-1 , end = forward ? M.rows : -1 , dir = forward ? 1 : -1;
	for( int j=start ; j!=end ; j+=dir )
	{
		T diagonal = M[j][0].Value;
#if ZERO_TESTING_JACOBI
		if( diagonal )
#endif // ZERO_TESTING_JACOBI
		{
			ConstPointer( MatrixEntry< T > ) start = M[j];
			ConstPointer( MatrixEntry< T > ) end = start + M.rowSizes[j];
			ConstPointer( MatrixEntry< T > ) e;
			start++;
			T2 _b = b[j];
			for( e=start ; e!=end ; e++ ) _b -= x[ e->N ] * e->Value;
			x[j] = _b / diagonal;
		}
	}
	return M.rows;
}
template<class T>
template<class T2>
int SparseMatrix<T>::SolveGS( const std::vector< std::vector< int > >& mcIndices , const SparseMatrix<T>& M , ConstPointer( T2 ) b , Pointer( T2 ) x , bool forward , int threads )
{
	int sum=0 , start = forward ? 0 : int( mcIndices.size() )-1 , end = forward ? int( mcIndices.size() ) : -1 , dir = forward ? 1 : -1;
	for( int j=start ; j!=end ; j+=dir )
	{
		const std::vector< int >& _mcIndices = mcIndices[j];
		sum += int( _mcIndices.size() );
		{
#pragma omp parallel for num_threads( threads )
			for( int k=0 ; k<int( _mcIndices.size() ) ; k++ )
			{
				int jj = _mcIndices[k];
				T diagonal = M[jj][0].Value;
#if ZERO_TESTING_JACOBI
				if( diagonal )
#endif // ZERO_TESTING_JACOBI
				{
					ConstPointer( MatrixEntry< T > ) start = M[jj];
					ConstPointer( MatrixEntry< T > ) end = start + M.rowSizes[jj];
					ConstPointer( MatrixEntry< T > ) e;
					start++;
					T2 _b = b[jj];
					for( e=start ; e!=end ; e++ ) _b -= x[ e->N ] * e->Value;
					x[jj] = _b / diagonal;
				}                                   
			}
		}
	}
	return sum;
}

template< class T >
template< class T2 >
void SparseMatrix< T >::getDiagonal( Pointer( T2 ) diagonal , int threads ) const
{
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<rows ; i++ )
	{
		T2 d = 0.;
		ConstPointer( MatrixEntry< T > ) start = m_ppElements[i];
		ConstPointer( MatrixEntry< T > ) end = start + rowSizes[i];
		ConstPointer( MatrixEntry< T > ) e;
		for( e=start ; e!=end ; e++ ) if( e->N==i ) d += e->Value;
		diagonal[i] = d;
	}
}
template< class T >
template< class T2 >
int SparseMatrix< T >::SolveCG( const SparseMatrix<T>& A , ConstPointer( T2 ) b , int iters , Pointer( T2 ) x , T2 eps , int reset , bool addDCTerm , bool solveNormal , int threads )
{
	eps *= eps;
	int dim = A.rows;
	Pointer( T2 ) r = AllocPointer< T2 >( dim );
	Pointer( T2 ) d = AllocPointer< T2 >( dim );
	Pointer( T2 ) q = AllocPointer< T2 >( dim );
	Pointer( T2 ) temp = NullPointer( T2 );
	if( reset ) memset( x , 0 , sizeof(T2)* dim );
	if( solveNormal ) temp = AllocPointer< T2 >( dim );

	double delta_new = 0 , delta_0;
	if( solveNormal )
	{
		if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )x , temp , threads ) , A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )temp , r , threads ) , A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )b , temp , threads );
		else            A.Multiply( ( ConstPointer( T2 ) )x , temp , threads ) , A.Multiply( ( ConstPointer( T2 ) )temp , r , threads ) , A.Multiply( ( ConstPointer( T2 ) )b , temp , threads );
#pragma omp parallel for num_threads( threads ) reduction( + : delta_new )
		for( int i=0 ; i<dim ; i++ ) d[i] = r[i] = temp[i] - r[i] , delta_new += r[i] * r[i];
	}
	else
	{
		if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )x , r , threads );
		else            A.Multiply( ( ConstPointer( T2 ) )x , r , threads );
#pragma omp parallel for num_threads( threads )  reduction ( + : delta_new )
		for( int i=0 ; i<dim ; i++ ) d[i] = r[i] = b[i] - r[i] , delta_new += r[i] * r[i];
	}
	delta_0 = delta_new;
	if( delta_new<eps )
	{
//		fprintf( stderr , "[WARNING] Initial residual too low: %g < %f\n" , delta_new , eps );
		FreePointer( r );
		FreePointer( d );
		FreePointer( q );
		FreePointer( temp );
		return 0;
	}
	int ii;
	for( ii=0 ; ii<iters && delta_new>eps*delta_0 ; ii++ )
	{
		if( solveNormal )
			if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )d , temp , threads ) , A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )temp , q , threads );
			else            A.Multiply( ( ConstPointer( T2 ) )d , temp , threads ) , A.Multiply( ( ConstPointer( T2 ) )temp , q , threads );
		else
			if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )d , q , threads );
			else            A.Multiply( ( ConstPointer( T2 ) )d , q , threads );
        double dDotQ = 0;
#pragma omp parallel for num_threads( threads ) reduction( + : dDotQ )
		for( int i=0 ; i<dim ; i++ ) dDotQ += d[i] * q[i];
		T2 alpha = T2( delta_new / dDotQ );
		double delta_old = delta_new;
		delta_new = 0;
		if( (ii%50)==(50-1) )
		{
#pragma omp parallel for num_threads( threads )
			for( int i=0 ; i<dim ; i++ ) x[i] += d[i] * alpha;
			if( solveNormal )
				if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )x , temp , threads ) , A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )temp , r , threads );
				else            A.Multiply( ( ConstPointer( T2 ) )x , temp , threads ) , A.Multiply( ( ConstPointer( T2 ) )temp , r , threads );
			else
				if( addDCTerm ) A.MultiplyAndAddAverage( ( ConstPointer( T2 ) )x , r , threads );
				else            A.Multiply( ( ConstPointer( T2 ) )x , r , threads );
#pragma omp parallel for num_threads( threads ) reduction( + : delta_new )
			for( int i=0 ; i<dim ; i++ ) r[i] = b[i] - r[i] , delta_new += r[i] * r[i] , x[i] += d[i] * alpha;
		}
		else
#pragma omp parallel for num_threads( threads ) reduction( + : delta_new )
			for( int i=0 ; i<dim ; i++ ) r[i] -= q[i] * alpha , delta_new += r[i] * r[i] ,  x[i] += d[i] * alpha;

		T2 beta = T2( delta_new / delta_old );
#pragma omp parallel for num_threads( threads )
		for( int i=0 ; i<dim ; i++ ) d[i] = r[i] + d[i] * beta;
	}
	FreePointer( r );
	FreePointer( d );
	FreePointer( q );
	FreePointer( temp );
	return ii;
}
