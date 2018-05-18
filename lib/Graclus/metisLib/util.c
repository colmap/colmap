/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id: util.c,v 1.1 1998/11/27 17:59:32 karypis Exp $
 */

#include "metis.h"

/************************************************************************
 when command line fails, print out help info
************************************************************************/
void print_help(char *program_name){
  printf("\nHelp\nTo cluster a graph into a given number of clusters:\n %s [options] graph_file number_of_clusters\n", program_name);
  printf(" options: -o ncut|rassoc\n");
  printf(" \t\t ncut --- normalized cut (default)\n\t\t rassoc --- ratio association\n");
  printf("          -l number_of_local_search_steps (default is 0)\n");
  printf("          -b use only boundary points (default is to use all points)\n");
  // printf("          -s use spectral at coarsest level (default is METIS init.)\n");
  printf("\nTo compute objective function value for a given clustering:\n %s [options] -e clustering_file graph_file\n\n", program_name);
}


/************************************************************************
 find out the cluster size
************************************************************************/
void clusterSize(GraphType * graph, int *clustersize){
  idxtype *where, i, nvtxs;
  where = graph->where;
  nvtxs = graph->nvtxs;

  for (i=0; i<nvtxs; i++)
	  clustersize[where[i]] ++;
}

/*************************************************************************
* This function transform Metis graph matrix into a dense matrix,
* which is represented as an array initialized to be 0
**************************************************************************/
void sparse2dense(GraphType * graph, double * dense, float *m_adjwgt)
{
  int nvtxs, i, j;
  idxtype *adjwgt, *adjncy, *xadj;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  for (i=0; i<nvtxs; i++)
    for (j=0; j<nvtxs; j++)
      dense[i*nvtxs+j] =0;

  if (adjwgt == NULL)
    for (i=0; i<nvtxs; i++)
      for (j=xadj[i]; j<xadj[i+1]; j++)
	dense[i* nvtxs+adjncy[j]] = 1;
  else
    for (i=0; i<nvtxs; i++)
      for (j=xadj[i]; j<xadj[i+1]; j++)
	dense[i* nvtxs+adjncy[j]] = m_adjwgt[j];
}

/*************************************************************************
* This function extracts file name from a path
**************************************************************************/
void extractfilename(char *path, char *name)
{
  int length, i, j;
  length = strlen(path);
  for(i= length-1; i>=0; i--)
    if ((path[i] == '/') || (path[i] == '\\'))
      {
	i++;
	for (j=i; j<length; j++)
	  name[j-i]=path[j];
	name[j-i] = '\0';
	break;
      }
    else if (i==0)
      {
	for (j=i; j<length; j++)
	  name[j-i]=path[j];
	name[j] = '\0';
	break;
      }
}

/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void errexit(char *f_str,...)
{
  va_list argp;
  char out1[256], out2[256];

  va_start(argp, f_str);
  vsprintf(out1, f_str, argp);
  va_end(argp);

  sprintf(out2, "Error! %s", out1);

  fprintf(stdout, out2);
  fflush(stdout);

  abort();
}



#ifndef DMALLOC
/*************************************************************************
* The following function allocates an array of Chains
**************************************************************************/
Chains *chainmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (Chains *)GKmalloc(sizeof(Chains)*n, msg);
}

/*************************************************************************
* The following function allocates an 2-D array of floats
**************************************************************************/
float **f2malloc(int n, int m, char *msg)
{
  float ** temp;
  int i;
  if ((n == 0) || (m==0))
    return NULL;

  temp = (float **)GKmalloc(sizeof(float *)*n, msg);
  for (i=0; i<n; i++)
    temp[i] = (float *)GKmalloc(sizeof(float)*m, msg);
  return temp;
}

/*************************************************************************
* The following function allocates an 2-D array of ints
**************************************************************************/
int **i2malloc(int n, int m, char *msg)
{
  int ** temp;
  int i;
  if ((n == 0) || (m==0))
    return NULL;

  temp = (int **)GKmalloc(sizeof(int *)*n, msg);
  for (i=0; i<n; i++)
    temp[i] = (int *)GKmalloc(sizeof(int)*m, msg);
  return temp;
}

/*************************************************************************
* The following function allocates an array of integers
**************************************************************************/
int *imalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (int *)GKmalloc(sizeof(int)*n, msg);
}


/*************************************************************************
* The following function allocates an array of integers
**************************************************************************/
idxtype *idxmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (idxtype *)GKmalloc(sizeof(idxtype)*n, msg);
}


/*************************************************************************
* The following function allocates an array of float
**************************************************************************/
float *fmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (float *)GKmalloc(sizeof(float)*n, msg);
}


/*************************************************************************
* The follwoing function allocates an array of integers
**************************************************************************/
int *ismalloc(int n, int ival, char *msg)
{
  if (n == 0)
    return NULL;

  return iset(n, ival, (int *)GKmalloc(sizeof(int)*n, msg));
}



/*************************************************************************
* The follwoing function allocates an array of integers
**************************************************************************/
idxtype *idxsmalloc(int n, idxtype ival, char *msg)
{
  if (n == 0)
    return NULL;

  return idxset(n, ival, (idxtype *)GKmalloc(sizeof(idxtype)*n, msg));
}


/*************************************************************************
* This function is my wrapper around malloc
**************************************************************************/
void *GKmalloc(int nbytes, char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL)
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);

  return ptr;
}
#endif

/*************************************************************************
* This function is my wrapper around free, allows multiple pointers
**************************************************************************/
void GKfree(void **ptr1,...)
{
  va_list plist;
  void **ptr;

  if (*ptr1 != NULL)
    free(*ptr1);
  *ptr1 = NULL;

  va_start(plist, ptr1);

  /* while ((int)(ptr = va_arg(plist, void **)) != -1) { */
  while ((ptr = va_arg(plist, void **)) != LTERM) {
    if (*ptr != NULL)
      free(*ptr);
    *ptr = NULL;
  }

  va_end(plist);
}


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
int *iset(int n, int val, int *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
idxtype *idxset(int n, idxtype val, idxtype *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
float *sset(int n, float val, float *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}



/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int iamax(int n, int *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int idxamax(int n, idxtype *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}

/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int idxamax_strd(int n, idxtype *x, int incx)
{
  int i, max=0;

  n *= incx;
  for (i=incx; i<n; i+=incx)
    max = (x[i] > x[max] ? i : max);

  return max/incx;
}



/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int samax(int n, float *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}

/*************************************************************************
* These functions return the index of the almost maximum element in a vector
**************************************************************************/
int samax2(int n, float *x)
{
  int i, max1, max2;

  if (x[0] > x[1]) {
    max1 = 0;
    max2 = 1;
  }
  else {
    max1 = 1;
    max2 = 0;
  }

  for (i=2; i<n; i++) {
    if (x[i] > x[max1]) {
      max2 = max1;
      max1 = i;
    }
    else if (x[i] > x[max2])
      max2 = i;
  }

  return max2;
}


/*************************************************************************
* These functions return the index of the minimum element in a vector
**************************************************************************/
int idxamin(int n, idxtype *x)
{
  int i, min=0;

  for (i=1; i<n; i++)
    min = (x[i] < x[min] ? i : min);

  return min;
}


/*************************************************************************
* These functions return the index of the minimum element in a vector
**************************************************************************/
int samin(int n, float *x)
{
  int i, min=0;

  for (i=1; i<n; i++)
    min = (x[i] < x[min] ? i : min);

  return min;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int idxsum(int n, idxtype *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int idxsum_strd(int n, idxtype *x, int incx)
{
  int i, sum = 0;

  for (i=0; i<n; i++, x+=incx) {
    sum += *x;
  }

  return sum;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
void idxadd(int n, idxtype *x, idxtype *y)
{
  for (n--; n>=0; n--)
    y[n] += x[n];
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int charsum(int n, char *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int isum(int n, int *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
float ssum(int n, float *x)
{
  int i;
  float sum = 0.0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
float ssum_strd(int n, float *x, int incx)
{
  int i;
  float sum = 0.0;

  for (i=0; i<n; i++, x+=incx)
    sum += *x;

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
void sscale(int n, float alpha, float *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] *= alpha;
}


/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
float snorm2(int n, float *v)
{
  int i;
  float partial = 0;

  for (i = 0; i<n; i++)
    partial += v[i] * v[i];

  return sqrt(partial);
}



/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
float sdot(int n, float *x, float *y)
{
  int i;
  float partial = 0;

  for (i = 0; i<n; i++)
    partial += x[i] * y[i];

  return partial;
}


/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
void saxpy(int n, float alpha, float *x, int incx, float *y, int incy)
{
  int i;

  for (i=0; i<n; i++, x+=incx, y+=incy)
    *y += alpha*(*x);
}




/*************************************************************************
* This file randomly permutes the contents of an array.
* flag == 0, don't initialize perm
* flag == 1, set p[i] = i
**************************************************************************/
void RandomPermute(int n, idxtype *p, int flag)
{
  int i, j, u, v;
  idxtype tmp;

  if (flag == 1) {
    for (i=0; i<n; i++)
      p[i] = i;
  }

  for(i = 1; i < n; i++)
  {
    j = rand() % (i+1);
    tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }

/*
  if (flag == 1) {
    for (i=0; i<n; i++)
      p[i] = i;
  }

  if (n <= 4)
    return;

  for (i=0; i<n; i+=16) {
    u = RandomInRangeFast(n-4);
    v = RandomInRangeFast(n-4);
    SWAP(p[v], p[u], tmp);
    SWAP(p[v+1], p[u+1], tmp);
    SWAP(p[v+2], p[u+2], tmp);
    SWAP(p[v+3], p[u+3], tmp);
    }
*/
}


/*************************************************************************
* This function generates random initialization
**************************************************************************/
void RandomInit(int n, int k, idxtype *label)
{
  int i, chunksize, j;
  idxtype tmp;
  idxtype *p= idxmalloc(n, "Util: RandomInit\n");

  RandomPermute(n, p, 1);
  chunksize = n / k +1;
  j=0;
  for (i=0; i<n; i++){
    label[p[i]] = j;
    if ((i+1)% chunksize ==0)
      j++;
  }
  free (p);
}



/*************************************************************************
* This function returns true if the a is a power of 2
**************************************************************************/
int ispow2(int a)
{
  for (; a%2 != 1; a = a>>1);
  return (a > 1 ? 0 : 1);
}


/*************************************************************************
* This function initializes the random number generator
**************************************************************************/
void InitRandom(int seed)
{
  if (seed == -1) {
    srand(4321);
  }
  else {
    srand(seed);
  }
}

/*************************************************************************
* This function returns the log2(x)
**************************************************************************/
int log2_metis(int a)
{
  int i;

  for (i=1; a > 1; i++, a = a>>1);
  return i-1;
}

