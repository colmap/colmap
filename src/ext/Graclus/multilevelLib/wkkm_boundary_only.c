/*
 * Copyright 2005, Yuqiang Guan
 *
 * wkkm.c
 *
 * This file contains weighted kernel k-means and refinement.
 *
 * Started 12/04
 * Yuqiang Guan
 *
 *
 */

#include "metis.h"
#include <float.h>

extern int cutType, memory_saving, boundary_points;

void Compute_Weights(CtrlType *ctrl, GraphType *graph, idxtype *w)
     /* compute the weights for WKKM; for the time, only Ncut. w is zero-initialized */
{
  int nvtxs, i, j;
  idxtype *xadj, *adjwgt;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjwgt = graph->adjwgt;

  if ((cutType == RASSO) || (cutType == RCUT))
    for (i=0; i<nvtxs; i++)
      w[i] = 1;
  else
    if (adjwgt == NULL)
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  w[i] ++;
    else
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  w[i] += adjwgt[j];
}

void transform_matrix(CtrlType *ctrl, GraphType *graph, idxtype *w, float *m_adjwgt)
     /* normalized the adjacency matrix for Ncut only*/
{
  int nvtxs, i, j;
  idxtype *xadj, *adjncy, *adjwgt, *where;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  if (cutType == RASSO){ // ratio asso.
    if (adjwgt == NULL)
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  m_adjwgt[j] =1;
    else
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  m_adjwgt[j] = adjwgt[j];
  }
  else{ //normalize rows and columns
    if (adjwgt == NULL){
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] = 1.0/w[i];
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] /=  w[adjncy[j]];
    }
    else{
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] = adjwgt[j] *1.0/w[i];
	  else
	    m_adjwgt[j] = 0;
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] /= w[adjncy[j]];
    }
  }
}

void pingpong(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, float *tpwgts, float ubfactor, int toplevel)
     // do batch-local search; chain_length is the search length
{

  int nvtxs, nedges, moves, iter;
  idxtype *w;
  //float *m_adjwgt;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;

  w = idxsmalloc(nvtxs, 0, "pingpong: weight");
  Compute_Weights(ctrl, graph, w);
  //m_adjwgt = fmalloc(nedges, "pingpong: normalized matrix");
  //transform_matrix(ctrl, graph, w, m_adjwgt);

  printf("Chain length is %d.\n", chain_length);

  moves =0;
  iter =0;

  //printf("Number of boundary points is %d\n", graph->nbnd);
  do{
    //Weighted_kernel_k_means(ctrl, graph, nparts, w, m_adjwgt, tpwgts, ubfactor);
    Weighted_kernel_k_means(ctrl, graph, nparts, w, tpwgts, ubfactor);
    if (chain_length>0){

      //moves = local_search(ctrl, graph, nparts, chain_length, w, m_adjwgt, tpwgts, ubfactor);
      moves = local_search(ctrl, graph, nparts, chain_length, w, tpwgts, ubfactor);
      printf("Number of local search moves is %d\n", moves);
      printf("Number of boundary points is %d\n", graph->nbnd);
    }
    iter ++;
    if (iter > MAXITERATIONS)
      break;
  }while(moves >0) ;
  if(memory_saving ==0){
    remove_empty_clusters_l1(ctrl, graph, nparts, w, tpwgts, ubfactor);
    if(toplevel>0)
      remove_empty_clusters_l2(ctrl, graph, nparts, w, tpwgts, ubfactor);
  }
  free(w);
  //free(m_adjwgt);
}

void Weighted_kernel_k_means(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *tpwgts, float ubfactor){
  // w is the weights


  int nvtxs, nbnd, nedges, me, i, j;
  idxtype *squared_sum, *sum, *xadj, *adjncy, *adjwgt, *where, *new_where, *bndptr, *bndind;
  float obj, old_obj, epsilon, *inv_sum, *squared_inv_sum;
  int change;
  int *linearTerm, ii;
  int loopend;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  nbnd = graph-> nbnd;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  // we need new_where because in kernel k-means distance is based on cluster label
  // if we change a label, distance to that cluster will change
  new_where = imalloc(nvtxs, "Weighted_kernel_k_means: new_where");
  for (i=0; i<nvtxs; i++)
    new_where[i] = where[i];

  sum = idxsmalloc(nparts,0, "Weighted_kernel_k_means: weight sum");
  inv_sum = fmalloc(nparts, "Weighted_kernel_k_means: sum inverse");
  squared_inv_sum = fmalloc(nparts, "Weighted_kernel_k_means: squared sum inverse");
  squared_sum = idxsmalloc(nparts,0, "Weighted_kernel_k_means: weight squared sum");

  //initialization

  obj = old_obj = 0;
  for (i=0; i<nvtxs; i++)
    sum[where[i]] += w[i];
  for (i=0; i<nparts; i++)
    if(sum[i] >0){
      inv_sum[i] = 1.0/sum[i];
      squared_inv_sum[i] = inv_sum[i]*inv_sum[i];
    }
    else
      inv_sum[i] = squared_inv_sum[i] = 0;

  /*
    if (adjwgt == NULL) //if graph has uniform edge weights
      for (i=0; i<nvtxs; i++){
        me = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (where[adjncy[j]] == me)
	  squared_sum[me] ++;
    }
    else*/
  for (i=0; i<nvtxs; i++){
    me = where[i];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      if (where[adjncy[j]] == me)
	squared_sum[me] += adjwgt[j];
  }

  //note the obj here is not obj. fun. of wkkm, just the second trace value to be maximized
  for (i=0; i<nparts; i++)
    if (sum[i] >0)
      obj +=  squared_sum[i]*1.0/sum[i];

  epsilon =.001;
  linearTerm = imalloc(nparts, "Weighted_kernel_k_means: new_where");

  do{
    float min_dist, dist;
    int min_ind, k;
    change =0;
    old_obj = obj;
    /*
    if (adjwgt == NULL) // if graph has uniform edge weights
      for (i=0; i<nvtxs; i++){ // compute linear term in distance from point i to all centers
	min_ind = me = where[i];

	for (k=0; k<nparts; k++)
	  linearTerm[k] =0;
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  linearTerm[where[adjncy[j]]] ++; // this line is the only difference between if and else
	for (k=0; k<nparts; k++)
	  printf("U%f ", linearTerm[k]*1.0/w[i]);
	printf("\n");
	min_dist = squared_sum[me]*w[i]-2*linearTerm[me]*sum[me];

	for (k=0; k<me; k++){
	  dist = squared_sum[k]*w[i] - 2*linearTerm[k]*sum[k];
	  if(dist*sum[min_ind]*sum[min_ind] < min_dist*sum[k]*sum[k]){
	    min_dist = dist;
	    min_ind = k;
	  }
	}
	for (k=me+1; k<nparts; k++){
	  dist = squared_sum[k]*w[i] - 2*linearTerm[k]*sum[k];
	  if(dist*sum[min_ind]*sum[min_ind] < min_dist*sum[k]*sum[k]){
	    min_dist = dist;
	    min_ind = k;
	  }
	}

	if(me != min_ind){
	  new_where[i] = min_ind; // note here we can not change where; otherwise we change the center
	  change ++;
	}
      }
    else // if graph weight is various
    */
    //printf("iteration of weighted kernel k-means...\n");
    if(boundary_points == 1)
      loopend = nbnd;
    else
      loopend = nvtxs;
    //for (i=0; i<nvtxs; i++){ // compute linear term in distance from point i to all centers
    for (ii=0; ii<loopend; ii++){
      if(boundary_points == 1)
	i = bndind[ii];
      else
	i = ii;
      if(w[i] >0){
	float inv_wi=1.0/w[i];
        for (k=0; k<nparts; k++)
 	  linearTerm[k] =0;
        for (j=xadj[i]; j<xadj[i+1]; j++)
	  linearTerm[where[adjncy[j]]] += adjwgt[j]; //only difference between if and else

	min_ind = me = where[i];
	min_dist = squared_sum[me]*squared_inv_sum[me] - 2*inv_wi*linearTerm[me]*inv_sum[me];
	/*if (sum[me] >0)
	  min_dist = squared_sum[me]*1.0/(1.0*sum[me]*sum[me])-2.0*linearTerm[me]/(sum[me]*w[i]);
	*/
	for (k=0; k<me; k++){
	  dist = squared_sum[k]*squared_inv_sum[k] -2*inv_wi*linearTerm[k]*inv_sum[k];
	  /*
	    dist = 0;
	    if (sum[k] >0)
	    dist = squared_sum[k]*1.0/(1.0*sum[k]*sum[k])-2.0*linearTerm[k]/(sum[k]*w[i]);
	  */
	  if(dist < min_dist){
	    //if(dist < min_dist){
	    min_dist = dist;
	    min_ind = k;
	  }
	}
	for (k=me+1; k<nparts; k++){
	  dist = squared_sum[k]*squared_inv_sum[k] -2*inv_wi*linearTerm[k]*inv_sum[k];
	  /*
	  dist = 0;
	  if (sum[k] >0)
	    dist = squared_sum[k]*1.0/(1.0*sum[k]*sum[k])-2.0*linearTerm[k]/(sum[k]*w[i]);
	  */
	  if(dist < min_dist){
	    //if(dist < min_dist){
	    min_dist = dist;
	    min_ind = k;
	  }
	}

	if(me != min_ind){
	  new_where[i] = min_ind; // note here we can not change where; otherwise we change the center
	  change ++;
	}
	//if(i==0)
	//printf("%d(%d->%d), sum=(%d, %d), linear=(%f, %f), square=(%d, %d), dis=(%f-%f, %f-%f)\n",i, me, min_ind, sum[me], sum[min_ind], linearTerm[me]*1.0/w[i],linearTerm[min_ind]*1.0/w[i],squared_sum[me], squared_sum[min_ind], squared_sum[me]*1.0/(1.0*sum[me]*sum[me]), 2.0*linearTerm[me]/(sum[me]*w[i]), squared_sum[min_ind]*1.0/(1.0*sum[min_ind]*sum[min_ind]), 2.0*linearTerm[min_ind]/(sum[min_ind]*w[i]));
      }
    }

    // update sum and squared_sum
    for (i=0; i<nparts; i++)
      sum[i] = squared_sum[i] = 0;
    for (i=0; i<nvtxs; i++)
      sum[new_where[i]] += w[i];
    for (i=0; i<nparts; i++)
      if(sum[i] >0){
	inv_sum[i] = 1.0/sum[i];
	squared_inv_sum[i] = inv_sum[i]*inv_sum[i];
      }
      else
	inv_sum[i] = squared_inv_sum[i] = 0;
    /*
      if (adjwgt == NULL)
         for (i=0; i<nvtxs; i++){
            me = new_where[i];
         for (j=xadj[i]; j<xadj[i+1]; j++)
            if (new_where[adjncy[j]] == me)
	      squared_sum[me] ++;
	}
      else
      */
    for (i=0; i<nvtxs; i++){
      me = new_where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (new_where[adjncy[j]] == me)
	  squared_sum[me] += adjwgt[j];
    }

    //update objective function (trace maximizatin)
    obj=0;
    for (i=0; i<nparts; i++)
      if (sum[i] >0)
	obj +=  squared_sum[i]*1.0/sum[i];
    //if matrix is not positive definite
    if (obj > old_obj)
      {
	if(boundary_points == 1)
	  loopend = nbnd;
	else
	  loopend = nvtxs;
      //for (i=0; i<nvtxs; i++){
	for (ii=0; ii<loopend; ii++){
	  if(boundary_points == 1)
	    i = bndind[ii];
	  else
	    i = ii;
	  where[i] = new_where[i];
	}
      }

  }while((obj - old_obj)> epsilon*obj);
  free(sum); free(squared_sum); free(new_where); free(linearTerm); free(inv_sum); free(squared_inv_sum);
}

/*
int local_search(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, idxtype *w, float *tpwgts, float ubfactor)
     //return # of points moved
{
  int nvtxs, nedges, nbnd, me, i, j, k, s, ii;
  idxtype *sum, *squared_sum, *xadj, *adjncy, *adjwgt, *where, *bndptr, *bndind;
  float change, obj, epsilon, **kDist, *accum_change;
  int moves, actual_length, *mark, fidx;
  Chains *chain;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;

  chain = chainmalloc(chain_length, "Local_search: local search chain");
  mark = ismalloc(nbnd, 0 , "Local_search: mark");
  sum = idxsmalloc(nparts,0, "Local_search: weight sum");
  squared_sum = idxsmalloc(nparts,0,"Local_search: weight squared sum");
  kDist = f2malloc(nbnd, nparts, "Local_search: distance matrix");
  accum_change = fmalloc(chain_length+1,"Local_search: accumulated change");

  //initialization
  for (i = 0; i<nparts; i++)
    sum[i] = squared_sum[i] = 0;
  for (i = 0; i<nbnd; i++)
    for (j = 0; j<nparts; j++)
      kDist[i][j] = 0;
  for (i = 0; i<chain_length+1; i++)
    accum_change[i] = 0;
  obj =  0;
  moves = 0;
  epsilon =.0001;
  actual_length = chain_length;

  for (i=0; i<nvtxs; i++)
    sum[where[i]] += w[i];
  for (i=0; i<nvtxs; i++){
    me = where[i];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      if (where[adjncy[j]] == me)
	squared_sum[me] += adjwgt[j];
  }

  //the diagonal entries won't affect the result so diagonal's assumed zero
  //for (i=0; i<nvtxs; i++)
  for (ii=0; ii<nbnd; ii++){
    i = bndind[ii];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      //kDist[i][where[adjncy[j]]] += 1.0*adjwgt[j]/w[i];
      kDist[ii][where[adjncy[j]]] += 1.0*adjwgt[j]/w[i];
  }
  for (k=0; k<nparts; k++)
    if (sum[k] >0)
      //for (i=0; i<nvtxs; i++)
      for (ii=0; ii<nbnd; ii++)
    	//kDist[i][k] = squared_sum[k]/(1.0*sum[k]*sum[k]) - 2*kDist[i][k]/sum[k];
	kDist[ii][k] = squared_sum[k]/(1.0*sum[k]*sum[k]) - 2*kDist[ii][k]/sum[k];

  for (i=0; i<nparts; i++)
    if (sum[i] >0)
      obj +=  squared_sum[i]*1.0/sum[i];

  for (i=0; i<chain_length; i++)
    {
      float tempMinChange, tempchange, temp_q;
      int tempid, tempMoveTo, from, to, tempbndind;

      tempMinChange = obj;
      tempchange =0;
      tempid =0;
      tempMoveTo = where[tempid];
      tempbndind =0;

      //for (j=0; j<nvtxs; j++)
      for (ii=0; ii<nbnd; ii++){
	j = bndind[ii];
        if (mark[ii] ==0){
	  me = where[j];
	  if (sum[me] > w[j]) // if this cluster where j belongs is not a singleton
	    for (k=0; k<nparts; k++)
	      if (k != me){
		//tempchange = -kDist[j][me]*sum[me]*w[j]/(sum[me]-w[j]) + kDist[j][k]*sum[k]*w[j]/(sum[k]+w[j]);
		tempchange = -kDist[ii][me]*sum[me]*w[j]/(sum[me]-w[j]) + kDist[ii][k]*sum[k]*w[j]/(sum[k]+w[j]);
		if (tempchange < tempMinChange){
		  tempMinChange = tempchange;
		  tempid = j;
		  tempbndind = ii;
		  tempMoveTo = k;
		}
	      }
	}
      }
      if ((tempMoveTo == where[tempid]) || (mark[tempbndind] >0)){
        actual_length = i;
	break;
      }
      else{
        // record which point is moved from its original cluster to new cluster
        chain[i].point = tempid;
        chain[i].from = where[tempid];
        chain[i].to = tempMoveTo;
        chain[i].change = tempMinChange;
	//mark the point moved
	mark[tempbndind] = 1;
	// update info
        accum_change[i+1] = accum_change[i] + tempMinChange;
	from = chain[i].from;
	to = chain[i].to;
	where[tempid] = to;
        sum[from] -=  w[tempid];
        sum[to] +=  w[tempid];

        for (j=xadj[tempid]; j<xadj[tempid+1]; j++)
	  if (where[adjncy[j]] == from)
	    squared_sum[from] -= 2*adjwgt[j];
	//for (s=0; s<nvtxs; s++){
	for (ii=0; ii<nbnd; ii++){
	  //kDist[s][from] = 0;
	  kDist[ii][from] = 0;
	  s = bndind[ii];
	  for (j=xadj[s]; j<xadj[s+1]; j++)
	    if (where[adjncy[j]] == from)
	      //kDist[s][from] += adjwgt[j]*1.0/w[s];
	      kDist[ii][from] += adjwgt[j]*1.0/w[s];
	}
	temp_q = squared_sum[from]/(1.0*sum[from]*sum[from]);

	//for (s=0; s<nvtxs; s++)
	for (ii=0; ii<nbnd; ii++)
	  kDist[ii][from] = temp_q - 2*kDist[ii][from]/sum[from];

	for (j=xadj[tempid]; j<xadj[tempid+1]; j++)
	  if (where[adjncy[j]] == to)
	    squared_sum[to] += 2*adjwgt[j];

	//for (s=0; s<nvtxs; s++){
	for (ii=0; ii<nbnd; ii++){
	  //kDist[s][to] = 0;
	  kDist[ii][to] = 0;
	  s = bndind[ii];
	  for (j=xadj[s]; j<xadj[s+1]; j++)
	    if (where[adjncy[j]] == to)
	      //kDist[s][to] += adjwgt[j]*1.0/w[s];
	      kDist[ii][to] += adjwgt[j]*1.0/w[s];
	}
	temp_q = squared_sum[to]/(1.0*sum[to]*sum[to]);

	//for (s=0; s<nvtxs; s++)
	for (ii=0; ii<nbnd; ii++)
	  //kDist[s][to] = temp_q - 2*kDist[s][to]/sum[to];
	  kDist[ii][to] = temp_q - 2*kDist[ii][to]/sum[to];
      }
    }
  fidx = samin(actual_length, accum_change);
  if (accum_change[fidx] < -epsilon * obj){
    for (i= fidx+1; i<=actual_length; i++)
      where[chain[i-1].point] = chain[i-1].from;
    moves = fidx;
    change = accum_change[fidx];
  }
  else{
    for (i= 0; i<actual_length; i++)
      where[chain[i].point] = chain[i].from;
    moves = 0;
    change = 0;
  }

  free(sum); free(squared_sum);free(accum_change); free(chain); free(mark);

  //for (i= 0; i<nvtxs; i++)
  for (i= 0; i<nbnd; i++)
    free(kDist[i]);
  free(kDist);

  return moves;
}
*/

float onePoint_move(GraphType *graph, int nparts, idxtype *sum, idxtype *squared_sum, idxtype *w, idxtype *self_sim, int **linearTerm, int ii){

  int k, j, s, i, nbnd, temp, q1, q2, minchange_id, new_squared_sum1, new_squared_sum2, me, nedges, nvtxs;
  float tempchange, minchange, obj, cut;
  idxtype *xadj, *adjncy, *adjwgt, *where, *bndptr, *bndind;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;

  s = bndind[ii];
  me= where[s];
  minchange_id = me;
  minchange = 0;
  new_squared_sum1=  new_squared_sum2= squared_sum[me];

  if (sum[me] > w[s]){ // if this cluster where j belongs is not a singleton
    float inv_sum1 = 1.0/sum[me], inv_sum2 = 1.0/(sum[me]-w[s]);

    for(k= 0; k<me; k++){
      q1 = squared_sum[me] - linearTerm[ii][me]*2 + self_sim[ii];
      q2 = squared_sum[k] + linearTerm[ii][k] *2 + self_sim[ii];
      if(sum[k] >0)
	tempchange = squared_sum[me]*inv_sum1 + squared_sum[k]*1.0/sum[k] - q1*inv_sum2- q2*1.0/(sum[k]+w[s]);
      else if(w[s]>0) // if sum[k] ==0 but w[s] >0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2 - q2*1.0/(sum[k]+w[s]);
      else // if sum[k] == 0 and w[s] ==0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2;

      if(tempchange < minchange){
	minchange = tempchange;
	minchange_id = k;
	new_squared_sum1 = q1;
	new_squared_sum2 = q2;
      }
    }
    for(k= me+1; k<nparts; k++){
      q1 = squared_sum[me] - linearTerm[ii][me]*2 +self_sim[ii];
      q2 = squared_sum[k] + linearTerm[ii][k]*2 + self_sim[ii];
      if(sum[k] >0)
	tempchange = squared_sum[me]*inv_sum1 + squared_sum[k]*1.0/sum[k] - q1*inv_sum2- q2*1.0/(sum[k]+w[s]);
      else if(w[s]>0) // if sum[k] ==0 but w[s] >0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2 - q2*1.0/(sum[k]+w[s]);
      else // if sum[k] == 0 and w[s] ==0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2;

      if(tempchange < minchange){
	minchange = tempchange;
	minchange_id = k;
	new_squared_sum1 = q1;
	new_squared_sum2 = q2;
      }
    }
    if (minchange < 0){
      where[s] = minchange_id;
      sum[me] -=  w[s];
      sum[minchange_id] +=  w[s];
      /*
      for (ii = 0; ii<nbnd; ii++)
	for (j = 0; j<nparts; j++)
	  linearTerm[ii][j] = 0;
      */
      for (j=xadj[s]; j<xadj[s+1]; j++){
	int boundary, adj_temp;
	adj_temp = adjncy[j];
	boundary = bndptr[adj_temp];
	if(boundary >-1) {
	  //if (where[adj_temp] == me){
	  //linearTerm[ii][me] -= adjwgt[j];
	    linearTerm[boundary][me] -= adjwgt[j];
	    //}
	    //if (where[adj_temp] == minchange_id){
	    //linearTerm[ii][minchange_id] += adjwgt[j];
	    linearTerm[boundary][minchange_id] += adjwgt[j];
	    //}
	}
      }
      /*
      for (ii =0; ii<nbnd; ii++){
	i = bndind[ii];
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  linearTerm[ii][where[adjncy[j]]] += adjwgt[j];
      }
      */


      squared_sum[me] = new_squared_sum1;
      squared_sum[minchange_id] = new_squared_sum2;
    }
  }
  return minchange;
}

void move1Point2EmptyCluster(GraphType *graph, int nparts, idxtype *sum, idxtype *squared_sum, idxtype *w, idxtype *self_sim, int **linearTerm, int k){
  int j, s, ii, nbnd, q1, q2, minchange_id, new_squared_sum1, new_squared_sum2, me, nvtxs;
  float tempchange, minchange;
  idxtype *xadj, *adjncy, *adjwgt, *where, *bndptr, *bndind;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;
  minchange_id = bndind[0];
  minchange = FLT_MAX;

  for(ii=0; ii<nbnd; ii++){
    s = bndind[ii];
    me= where[s];
    new_squared_sum1=  new_squared_sum2= squared_sum[me];

    if (sum[me] > w[s]){ // if this cluster where j belongs is not a singleton
      float inv_sum1 = 1.0/sum[me], inv_sum2 = 1.0/(sum[me]-w[s]);

      q1 = squared_sum[me] - linearTerm[ii][me]*2 + self_sim[ii];
      q2 = squared_sum[k] + linearTerm[ii][k] *2 + self_sim[ii];
      if(w[s]>0) // note that sum[k] ==0; if w[s] >0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2 - q2*1.0/(sum[k]+w[s]);
      else // if sum[k] == 0 and w[s] ==0
	tempchange = squared_sum[me]*inv_sum1 - q1*inv_sum2;

      if(tempchange < minchange){
	minchange = tempchange;
	minchange_id = s;
	new_squared_sum1 = q1;
	new_squared_sum2 = q2;
      }
    }
  }

  where[minchange_id] = k;
  sum[me] -=  w[minchange_id];
  sum[k] +=  w[minchange_id];

  for (j=xadj[minchange_id]; j<xadj[minchange_id+1]; j++){
    int boundary, adj_temp;
    adj_temp = adjncy[j];
    boundary = bndptr[adj_temp];
    if(boundary >-1) {
      linearTerm[boundary][me] -= adjwgt[j];
      linearTerm[boundary][k] += adjwgt[j];
    }
    squared_sum[me] = new_squared_sum1;
    squared_sum[k] = new_squared_sum2;
  }
}

int local_search(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, idxtype *w, float *tpwgts, float ubfactor)
     //return # of points moved
{
  int nvtxs, nedges, nbnd, me, i, j, k, s, ii;
  idxtype *sum, *squared_sum, *xadj, *adjncy, *adjwgt, *where, *bndptr, *bndind, *self_sim;
  float change, obj, epsilon, accum_change, minchange;
  int moves, loopTimes, **linearTerm;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  nbnd = graph->nbnd;
  bndind = graph->bndind;
  bndptr = graph->bndptr;

  sum = idxsmalloc(nparts,0, "Local_search: weight sum");
  squared_sum = idxsmalloc(nparts,0,"Local_search: weight squared sum");
  self_sim = idxsmalloc(nbnd, 0, "Local_search: self similarity");
  linearTerm = i2malloc(nbnd, nparts, "Local_search: linear term");


  moves = 0;
  epsilon =.001;

  for (i=0; i<nvtxs; i++)
    sum[where[i]] += w[i];
  for (i=0; i<nvtxs; i++){
    me = where[i];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      if (where[adjncy[j]] == me)
	squared_sum[me] += adjwgt[j];
  }

  for (ii = 0; ii<nbnd; ii++)
    for (j = 0; j<nparts; j++)
      linearTerm[ii][j] = 0;
  for (ii =0; ii<nbnd; ii++){
    s = bndind[ii];
    for (j=xadj[s]; j<xadj[s+1]; j++){
      //kDist[i][where[adjncy[j]]] += 1.0*adjwgt[j]/w[i];
      linearTerm[ii][where[adjncy[j]]] += adjwgt[j];
      if (adjncy[j] == s)
	self_sim[ii] = adjwgt[j];
    }
  }

  //the diagonal entries won't affect the result so diagonal's assumed zero
  obj =  0;
  for (i=0; i<nparts; i++)
    if (sum[i] >0)
      obj +=  squared_sum[i]*1.0/sum[i];

  srand(time(NULL));
  //temperature = DEFAULT_TEMP;
  loopTimes = 0;

  while (loopTimes < chain_length){
    accum_change =0;
    //for (j=0; j<nvtxs; j++){
    for (ii=0; ii<nbnd; ii++){
      minchange = onePoint_move(graph, nparts, sum, squared_sum, w, self_sim, linearTerm, ii);
      accum_change += minchange;
    }

    if (accum_change > -epsilon * obj){
      break;
    }
    moves ++;
    loopTimes ++;
  }

  free(sum); free(squared_sum); free(self_sim);

  //for (i= 0; i<nvtxs; i++)
  for (i= 0; i<nbnd; i++)
    free(linearTerm[i]);
  free(linearTerm);

  //printf("moves = %d\n", moves);
  return moves;
}


void remove_empty_clusters_l1(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *tpwgts, float ubfactor){
  int *clustersize=imalloc(nparts, "remove_empty_clusters: clusterSize");
  int number_of_empty_cluster=0, i, s;

  for(i=0; i<nparts; i++)
    clustersize[i] =0;
  clusterSize(graph, clustersize);
  for(i=0; i<nparts; i++)
    if(clustersize[i] ==0)
      number_of_empty_cluster ++;

  if(number_of_empty_cluster>0)
    local_search(ctrl, graph, nparts, 1, w, tpwgts, ubfactor);

  free(clustersize);
}

void remove_empty_clusters_l2(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *tpwgts, float ubfactor){
  int *clustersize=imalloc(nparts, "remove_empty_clusters: clusterSize");
  int number_of_empty_cluster=0, i, s;

  for(i=0; i<nparts; i++)
    clustersize[i] =0;
  clusterSize(graph, clustersize);
  for(i=0; i<nparts; i++)
    if(clustersize[i] ==0)
      number_of_empty_cluster ++;
  //printf("%d empty clusters; ", number_of_empty_cluster);

  if(number_of_empty_cluster>0){
    int nvtxs, me, j, k, ii;
    idxtype *sum, *squared_sum, *xadj, *adjncy, *adjwgt, *where, *bndptr, *bndind, *self_sim, nbnd;
    int **linearTerm;

    nvtxs = graph->nvtxs;
    xadj = graph->xadj;
    adjncy = graph->adjncy;
    adjwgt = graph->adjwgt;
    where = graph->where;
    nbnd = graph->nbnd;
    bndind = graph->bndind;
    bndptr = graph->bndptr;

    sum = idxsmalloc(nparts,0, "Local_search: weight sum");
    squared_sum = idxsmalloc(nparts,0,"Local_search: weight squared sum");
    self_sim = idxsmalloc(nbnd, 0, "Local_search: self similarity");
    linearTerm = i2malloc(nbnd, nparts, "Local_search: linear term");

    for (i=0; i<nvtxs; i++)
      sum[where[i]] += w[i];
    for (i=0; i<nvtxs; i++){
      me = where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (where[adjncy[j]] == me)
	  squared_sum[me] += adjwgt[j];
    }

    for (ii = 0; ii<nbnd; ii++)
      for (j = 0; j<nparts; j++)
	linearTerm[ii][j] = 0;
    for (ii =0; ii<nbnd; ii++){
      s = bndind[ii];
      for (j=xadj[s]; j<xadj[s+1]; j++){
	linearTerm[ii][where[adjncy[j]]] += adjwgt[j];
	if (adjncy[j] == s)
	  self_sim[ii] = adjwgt[j];
      }
    }
    for(k=0; k<nparts; k++)
      if(clustersize[k] ==0){
	move1Point2EmptyCluster(graph, nparts, sum, squared_sum, w, self_sim, linearTerm, k);
      }
    free(sum); free(squared_sum); free(self_sim);

    //for (i= 0; i<nvtxs; i++)
    for (i= 0; i<nbnd; i++)
      free(linearTerm[i]);
    free(linearTerm);
  }
  /*
  for(i=0; i<nparts; i++)
    clustersize[i] =0;
  number_of_empty_cluster=0;
  clusterSize(graph, clustersize);
  for(i=0; i<nparts; i++)
    if(clustersize[i] ==0)
      number_of_empty_cluster ++;
  printf("%d empty clusters\n", number_of_empty_cluster);
  */
  free(clustersize);
}

void MLKKMRefine(CtrlType *ctrl, GraphType *orggraph, GraphType *graph, int nparts, int chain_length, float *tpwgts, float ubfactor)
{
  int i, nlevels, mustfree=0, temp_cl;
  GraphType *ptr;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->UncoarsenTmr));

  /* Compute the parameters of the coarsest graph */
  ComputeKWayPartitionParams(ctrl, graph, nparts);
  temp_cl = chain_length;

  /* Determine how many levels are there */
  for (ptr=graph, nlevels=0; ptr!=orggraph; ptr=ptr->finer, nlevels++);
  //printf("Number of levels is %d\n", nlevels);

  for (i=0; ;i++) {
    timer tmr;
    float result;

    cleartimer(tmr);
    starttimer(tmr);

    //pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor);
    //chain_length /= 1.5;
    //printf("Level: %d\n", i+1);

    if (graph == orggraph){
      //chain_length = chain_length>0 ? chain_length : 1;
      pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor, 1);
      break;
    }
    else{
      //pingpong(ctrl, graph, nparts, 0, tpwgts, ubfactor, 0);
      pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor, 0);
      //chain_length /= 2;
    }


    //pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor);

    //    /* for time and quality each level

    stoptimer(tmr);
    //printf("Level %d: %7.3f", i+1, tmr);
    if (cutType == NCUT){
      result = ComputeNCut(graph, graph->where, nparts);
      //printf("   %7f", result);
    }
    else{
      result = ComputeRAsso(graph, graph->where, nparts);
      //printf("   %7f", result);
    }
    //printf(" (%d)\n\n", graph->nvtxs);
    //ends here*/

    if (graph == orggraph)
      break;
    /*
    if(i>1)
      chain_length /= 10;
    */

    GKfree((void **) &graph->gdata, LTERM);  /* Deallocate the graph related arrays */
    graph = graph->finer;
    IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));
    if (graph->vwgt == NULL) {
      graph->vwgt = idxsmalloc(graph->nvtxs, 1, "RefineKWay: graph->vwgt");
      graph->adjwgt = idxsmalloc(graph->nedges, 1, "RefineKWay: graph->adjwgt");
      mustfree = 1;
    }
    ProjectKWayPartition(ctrl, graph, nparts);
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
  }

  if (mustfree)
    GKfree((void **) &graph->vwgt, (void **) &graph->adjwgt, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->UncoarsenTmr));
}
