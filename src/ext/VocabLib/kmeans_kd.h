/* kmeans_kd.h */

#ifndef __KMEANS_KD_H__
#define __KMEANS_KD_H__

int compute_clustering_kd_tree(int n, int dim, int k, unsigned char **v,
                               double *means, unsigned int *clustering, 
                               double &error_out);

#endif /* __KMEANS_KD_H__ */
