/* kmeans.h */

#ifndef __KMEANS_H__
#define __KMEANS_H__

/* Run kmeans on a set of input vectors
 *
 * Inputs: n        : number of input vectors
 *         dim      : dimension of each input vector
 *         k        : number of means to compute
 *         restarts : number of random restarts to perform
 *         v        : set of pointers to input vectors (stored as arrays)
 *
 * Outputs: means      : vector of means (stored as a flat array,
 *                       i.e., the means are concatenated together in
 *                       memory
 *          clustering : array containing assignment of input points
 *                       to clusters -- clustering[i] contains the
 *                       cluster ID for point i
 */
double kmeans(int n, int dim, int k, int restarts, unsigned char **v,
              double *means, unsigned int *clustering);

#endif /* __KMEANS_H__ */
