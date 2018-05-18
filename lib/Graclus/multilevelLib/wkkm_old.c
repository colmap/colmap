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
 * $Id: weighted kernel k-means,v 1.0 2005/1/10 $
 *
 */

#include "metis.h"

extern int cutType;

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
     /* normalized the adjacency matrix for Ncut only, D^-1*A*D^-1*/
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

void transform_matrix_half(CtrlType *ctrl, GraphType *graph, idxtype *w, float *m_adjwgt)
     /* normalized the adjacency matrix for Ncut only, D^-.5*A*D^-.5*/
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
	    m_adjwgt[j] = 1.0/sqrt(w[i]);
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] /=  sqrt(w[adjncy[j]]);
    }
    else{
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] = adjwgt[j] *1.0/sqrt(w[i]);
	  else
	    m_adjwgt[j] = 0;
      for (i=0; i<nvtxs; i++)
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (w[i]>0)
	    m_adjwgt[j] /= sqrt(w[adjncy[j]]);
    }
  }
}


void pingpong(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, float *tpwgts, float ubfactor)
     // do batch-local search; chain_length is the search length
{

  int nvtxs, nedges, moves, iter;
  idxtype *w;
  float *m_adjwgt;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;

  w = idxsmalloc(nvtxs, 0, "pingpong: weight");
  Compute_Weights(ctrl, graph, w);
  m_adjwgt = fmalloc(nedges, "pingpong: normalized matrix");
  transform_matrix(ctrl, graph, w, m_adjwgt);

  moves =0;
  iter =0;

  do{
    Weighted_kernel_k_means(ctrl, graph, nparts, w, m_adjwgt, tpwgts, ubfactor);
    if (chain_length>0)
      moves = local_search(ctrl, graph, nparts, chain_length, w, m_adjwgt, tpwgts, ubfactor);
    iter ++;
    if (iter > MAXITERATIONS)
      break;
  }while(moves >0) ;

  free(w); free(m_adjwgt);
}


void Weighted_kernel_k_means(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *m_adjwgt, float *tpwgts, float ubfactor)
     // w is the weights and m_adjwgt is the kernel matrix
{

  int nvtxs, nedges, me, i, j;
  idxtype *xadj, *adjncy, *adjwgt, *where, *new_where;
  float *sum, *squared_sum, obj, old_obj, epsilon;
  int change;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;

  // we need new_where because in kernel k-means distance is based on cluster label
  // if we change a label, distance to that cluster will change
  new_where = imalloc(nvtxs, "Weighted_kernel_k_means: new_where");
  for (i=0; i<nvtxs; i++)
    new_where[i] = where[i];

  sum = fmalloc(nparts,"Weighted_kernel_k_means: weight sum");
  squared_sum = fmalloc(nparts,"Weighted_kernel_k_means: weight squared sum");

  //initialization
  for (i=0; i<nparts; i++)
    sum[i] = squared_sum[i] =0;

  obj = old_obj = 0;
  for (i=0; i<nvtxs; i++)
    sum[where[i]] += w[i];
  for (i=0; i<nvtxs; i++){
    me = where[i];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      if (where[adjncy[j]] == me)
	squared_sum[me] += w[i]*w[adjncy[j]]*m_adjwgt[j];
  }

  //note the obj here is not obj. fun. of wkkm, just the second trace value to be maximized
  for (i=0; i<nparts; i++)
    if (sum[i] >0)
      obj +=  squared_sum[i]/sum[i];

  epsilon =.001;
  /*
  for (int i=0; i<nvtxs; i++)
    printf("%d ", where[i]);
  printf("$ ");
  printf("start: %f\n", obj);
  */
  do{
    float dist, temp, min_dist;
    int min_ind, *temp_where, k;
    change =0;
    old_obj = obj;

    //printf("   Obj: %f\n", obj);

    // for each point, find its closest center
    for (i=0; i<nvtxs; i++){
      // compute distance from a point to its own center
      min_ind = me = where[i];
      min_dist = 0;
      if (sum[me] >0){
	temp =0;
	for (j=xadj[i]; j<xadj[i+1]; j++)
	  if (where[adjncy[j]] == me)
	    temp +=w[adjncy[j]]*m_adjwgt[j];
	min_dist = squared_sum[me]/(sum[me]*sum[me])-2*temp/sum[me];
      }

      // compute distance from the point to other centers
      for (k=0; k<me; k++){
	dist =0;
	if (sum[k] >0){
	  temp =0;
	  for (j=xadj[i]; j<xadj[i+1]; j++)
	    if (where[adjncy[j]] == k)
	      temp +=w[adjncy[j]]*m_adjwgt[j];
	  dist = squared_sum[k]/(sum[k]*sum[k])-2*temp/sum[k];
	}

	if (dist <min_dist){
	  min_dist = dist;
	  min_ind = k;
	}
      }
      for (k=me+1; k<nparts; k++){
	dist =0;
	if (sum[k] >0){
	  temp =0;
	  for (j=xadj[i]; j<xadj[i+1]; j++)
	    if (where[adjncy[j]] == k)
	      temp +=w[adjncy[j]]*m_adjwgt[j];
	  dist = squared_sum[k]/(sum[k]*sum[k])-2*temp/sum[k];
	}

	if (dist <min_dist){
	  min_dist = dist;
	  min_ind = k;
	}
      }
      if(me != min_ind){
	new_where[i] = min_ind; // note here we can not change where; otherwise we change the center
	change ++;
      }
    }

    // update sum and squared_sum
    for (i=0; i<nparts; i++)
      sum[i] = squared_sum[i] =0;
    for (i=0; i<nvtxs; i++)
      sum[new_where[i]] += w[i];
    for (i=0; i<nvtxs; i++){
      me = new_where[i];
      for (j=xadj[i]; j<xadj[i+1]; j++)
	if (new_where[adjncy[j]] == me)
	  squared_sum[me] += w[i]*w[adjncy[j]]*m_adjwgt[j];
    }

    //update objective function (trace maximizatin)
    obj=0;
    for (i=0; i<nparts; i++)
      if (sum[i] >0)
	obj +=  squared_sum[i]/sum[i];
    //if matrix is not positive definite
    if (obj > old_obj)
      for (i=0; i<nvtxs; i++)
	where[i] = new_where[i];

  }while((obj - old_obj)> epsilon*obj);
  /*
  for (int i=0; i<nvtxs; i++)
    printf("%d ", where[i]);
  printf("$ ");
  printf("stop: %f\n", obj);
  */
  free(sum); free(squared_sum); free(new_where);
}


int local_search(CtrlType *ctrl, GraphType *graph, int nparts, int chain_length, idxtype *w, float *m_adjwgt, float *tpwgts, float ubfactor)
     //return # of points moved
{
  int nvtxs, nedges, me, i, j, k, s;
  idxtype *xadj, *adjncy, *adjwgt, *where;
  float change, *sum, *squared_sum, obj, epsilon, **kDist, *accum_change;
  int moves, actual_length, *mark, fidx;
  Chains *chain;

  nedges = graph->nedges;
  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;

  chain = chainmalloc(chain_length, "Local_search: local search chain");
  mark = ismalloc(nvtxs, 0 , "Local_search: mark");
  sum = fmalloc(nparts,"Local_search: weight sum");
  squared_sum = fmalloc(nparts,"Local_search: weight squared sum");
  kDist = f2malloc(nvtxs, nparts, "Local_search: distance matrix");
  accum_change = fmalloc(chain_length+1,"Local_search: accumulated change");

  //initialization
  for (i = 0; i<nparts; i++)
    sum[i] = squared_sum[i] = 0;
  for (i = 0; i<nvtxs; i++)
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
	squared_sum[me] += w[i]*w[adjncy[j]]*m_adjwgt[j];
  }

  //note this distance is only for METIS matrix, i.e., zero diagonal matrix
  for (i=0; i<nvtxs; i++)
    for (j=xadj[i]; j<xadj[i+1]; j++)
      kDist[i][where[adjncy[j]]] +=w[adjncy[j]]*m_adjwgt[j];
  for (k=0; k<nparts; k++)
    if (sum[k] >0)
      for (i=0; i<nvtxs; i++)
    	kDist[i][k] = squared_sum[k]/(sum[k]*sum[k]) - 2*kDist[i][k]/sum[k];

  for (i=0; i<nparts; i++)
    if (sum[i] >0)
      obj +=  squared_sum[i]/sum[i];

  for (i=0; i<chain_length; i++)
    {
      float tempMinChange, tempchange, temp_q;
      int tempid, tempMoveTo, from, to;

      tempMinChange = obj;
      tempchange =0;
      tempid =0;
      tempMoveTo = where[tempid];

      for (j=0; j<nvtxs; j++)
        if (mark[j] ==0){
	  me = where[j];
	  if (sum[me] > w[j]) // if this cluster where j belongs is not a singleton
	    for (k=0; k<nparts; k++)
	      if (k != me){
		tempchange = -sum[me]*w[j]/(sum[me]-w[j])*kDist[j][me]+sum[k]*w[j]/(sum[k]+w[j])*kDist[j][k];
		if (tempchange < tempMinChange){
		  tempMinChange = tempchange;
		  tempid = j;
		  tempMoveTo = k;
		}
	      }
	}

      if ((tempMoveTo == where[tempid]) || (mark[tempid] >0)){
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
	mark[tempid] = 1;
	// update info
        accum_change[i+1] = accum_change[i] + tempMinChange;
	from = chain[i].from;
	to = chain[i].to;
	where[tempid] = to;
        sum[from] -=  w[tempid];
        sum[to] +=  w[tempid];

        for (j=xadj[tempid]; j<xadj[tempid+1]; j++)
	  if (where[adjncy[j]] == from)
	    squared_sum[from] -= 2*w[tempid]*w[adjncy[j]]*m_adjwgt[j];
	for (s=0; s<nvtxs; s++){
	  kDist[s][from] = 0;
	  for (j=xadj[s]; j<xadj[s+1]; j++)
	    if (where[adjncy[j]] == from)
	      kDist[s][from] += w[adjncy[j]]*m_adjwgt[j];
	}
	temp_q = squared_sum[from]/(sum[from]*sum[from]);
	for (s=0; s<nvtxs; s++)
	  kDist[s][from] = temp_q - 2*kDist[s][from]/sum[from];

	for (j=xadj[tempid]; j<xadj[tempid+1]; j++)
	  if (where[adjncy[j]] == to)
	    squared_sum[to] += 2*w[tempid]*w[adjncy[j]]*m_adjwgt[j];
	for (s=0; s<nvtxs; s++){
	  kDist[s][to] = 0;
	  for (j=xadj[s]; j<xadj[s+1]; j++)
	    if (where[adjncy[j]] == to)
	      kDist[s][to] += w[adjncy[j]]*m_adjwgt[j];
	}
	temp_q = squared_sum[to]/(sum[to]*sum[to]);
	for (s=0; s<nvtxs; s++)
	  kDist[s][to] = temp_q - 2*kDist[s][to]/sum[to];
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
  for (i= 0; i<nvtxs; i++)
    free(kDist[i]);
  free(kDist);

  return moves;
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

  for (i=0; ;i++) {
    timer tmr;
    float result;

    cleartimer(tmr);
    starttimer(tmr);

    //pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor);
    //chain_length /= 1.5;
    //printf("Level: %d\n", i+1);
    /*
    if (graph == orggraph){
      pingpong(ctrl, graph, nparts, temp_cl, tpwgts, ubfactor);
      break;
    }
    else{
      pingpong(ctrl, graph, nparts, 0, tpwgts, ubfactor);
      //chain_length /= 2;
    }
    */

    pingpong(ctrl, graph, nparts, chain_length, tpwgts, ubfactor);

    /* for time and quality each level

    stoptimer(tmr);
    printf("\nLevel %d: %7.3f", i+1, tmr);
    if (cutType == NCUT){
      result = ComputeNCut(graph, graph->where, nparts);
      printf("   %7f", result);
    }
    else{
      result = ComputeRAsso(graph, graph->where, nparts);
      printf("   %7f", result);
    }
    printf(" (%d)", graph->nvtxs);
    ends here*/

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
