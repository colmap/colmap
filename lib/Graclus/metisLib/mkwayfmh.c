/*
 * mkwayfmh.c
 *
 * This file contains code that implements the multilevel k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: mkwayfmh.c,v 1.1 1998/11/27 17:59:24 karypis Exp $
 *
 */

#include "metis.h"



/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void MCRandom_KWayEdgeRefineHorizontal(CtrlType *ctrl, GraphType *graph, int nparts,
       float *orgubvec, int npasses)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, ncon, nmoves, nbnd, myndegrees, same;
  int from, me, to, oldcut, gain;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *perm, *bndptr, *bndind;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;
  float *npwgts, *nvwgt, *minwgt, *maxwgt, maxlb, minlb, ubvec[MAXNCON], tvec[MAXNCON];

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  bndptr = graph->bndptr;
  bndind = graph->bndind;

  where = graph->where;
  npwgts = graph->npwgts;

  /* Setup the weight intervals of the various subdomains */
  minwgt =  fwspacemalloc(ctrl, nparts*ncon);
  maxwgt = fwspacemalloc(ctrl, nparts*ncon);

  /* See if the orgubvec consists of identical constraints */
  maxlb = minlb = orgubvec[0];
  for (i=1; i<ncon; i++) {
    minlb = (orgubvec[i] < minlb ? orgubvec[i] : minlb);
    maxlb = (orgubvec[i] > maxlb ? orgubvec[i] : maxlb);
  }
  same = (fabs(maxlb-minlb) < .01 ? 1 : 0);


  /* Let's not get very optimistic. Let Balancing do the work */
  ComputeHKWayLoadImbalance(ncon, nparts, npwgts, ubvec);
  for (i=0; i<ncon; i++)
    ubvec[i] = amax(ubvec[i], orgubvec[i]);

  if (!same) {
    for (i=0; i<nparts; i++) {
      for (j=0; j<ncon; j++) {
        maxwgt[i*ncon+j] = ubvec[j]/nparts;
        minwgt[i*ncon+j] = 1.0/(ubvec[j]*nparts);
      }
    }
  }
  else {
    maxlb = ubvec[0];
    for (i=1; i<ncon; i++)
      maxlb = (ubvec[i] > maxlb ? ubvec[i] : maxlb);

    for (i=0; i<nparts; i++) {
      for (j=0; j<ncon; j++) {
        maxwgt[i*ncon+j] = maxlb/nparts;
        minwgt[i*ncon+j] = 1.0/(maxlb*nparts);
      }
    }
  }


  perm = idxwspacemalloc(ctrl, nvtxs);

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Partitions: [%5.4f %5.4f], Nv-Nb[%6d %6d]. Cut: %6d, LB: ",
            npwgts[samin(ncon*nparts, npwgts)], npwgts[samax(ncon*nparts, npwgts)],
            graph->nvtxs, graph->nbnd, graph->mincut);
    ComputeHKWayLoadImbalance(ncon, nparts, npwgts, tvec);
    for (i=0; i<ncon; i++)
      printf("%.3f ", tvec[i]);
    printf("\n");
  }

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    oldcut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (nmoves=iii=0; iii<graph->nbnd; iii++) {
      ii = perm[iii];
      if (ii >= nbnd)
        continue;
      i = bndind[ii];

      myrinfo = graph->rinfo+i;

      if (myrinfo->ed >= myrinfo->id) { /* Total ED is too high */
        from = where[i];
        nvwgt = graph->nvwgt+i*ncon;

        if (myrinfo->id > 0 && AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, -1.0, nvwgt, minwgt+from*ncon))
          continue;   /* This cannot be moved! */

        myedegrees = myrinfo->edegrees;
        myndegrees = myrinfo->ndegrees;

        for (k=0; k<myndegrees; k++) {
          to = myedegrees[k].pid;
          gain = myedegrees[k].ed - myrinfo->id;
          if (gain >= 0 &&
              (AreAllHVwgtsBelow(ncon, 1.0, npwgts+to*ncon, 1.0, nvwgt, maxwgt+to*ncon) ||
               IsHBalanceBetterFT(ncon, nparts, npwgts+from*ncon, npwgts+to*ncon, nvwgt, ubvec)))
            break;
        }
        if (k == myndegrees)
          continue;  /* break out if you did not find a candidate */

        for (j=k+1; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          if ((myedegrees[j].ed > myedegrees[k].ed &&
               (AreAllHVwgtsBelow(ncon, 1.0, npwgts+to*ncon, 1.0, nvwgt, maxwgt+to*ncon) ||
               IsHBalanceBetterFT(ncon, nparts, npwgts+from*ncon, npwgts+to*ncon, nvwgt, ubvec))) ||
              (myedegrees[j].ed == myedegrees[k].ed &&
               IsHBalanceBetterTT(ncon, nparts, npwgts+myedegrees[k].pid*ncon, npwgts+to*ncon, nvwgt, ubvec)))
            k = j;
        }

        to = myedegrees[k].pid;

        if (myedegrees[k].ed-myrinfo->id == 0
            && !IsHBalanceBetterFT(ncon, nparts, npwgts+from*ncon, npwgts+to*ncon, nvwgt, ubvec)
            && AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, 0.0, npwgts+from*ncon, maxwgt+from*ncon))
          continue;

        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to'
        *======================================================================*/
        graph->mincut -= myedegrees[k].ed-myrinfo->id;

        IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

        /* Update where, weight, and ID/ED information of the vertex you moved */
        saxpy(ncon, 1.0, nvwgt, 1, npwgts+to*ncon, 1);
        saxpy(ncon, -1.0, nvwgt, 1, npwgts+from*ncon, 1);
        where[i] = to;
        myrinfo->ed += myrinfo->id-myedegrees[k].ed;
        SWAP(myrinfo->id, myedegrees[k].ed, j);
        if (myedegrees[k].ed == 0)
          myedegrees[k] = myedegrees[--myrinfo->ndegrees];
        else
          myedegrees[k].pid = from;

        if (myrinfo->ed-myrinfo->id < 0)
          BNDDelete(nbnd, bndind, bndptr, i);

        /* Update the degrees of adjacent vertices */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          ii = adjncy[j];
          me = where[ii];

          myrinfo = graph->rinfo+ii;
          if (myrinfo->edegrees == NULL) {
            myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
            ctrl->wspace.cdegree += xadj[ii+1]-xadj[ii];
          }
          myedegrees = myrinfo->edegrees;

          ASSERT(CheckRInfo(myrinfo));

          if (me == from) {
            INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);

            if (myrinfo->ed-myrinfo->id >= 0 && bndptr[ii] == -1)
              BNDInsert(nbnd, bndind, bndptr, ii);
          }
          else if (me == to) {
            INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);

            if (myrinfo->ed-myrinfo->id < 0 && bndptr[ii] != -1)
              BNDDelete(nbnd, bndind, bndptr, ii);
          }

          /* Remove contribution from the .ed of 'from' */
          if (me != from) {
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (myedegrees[k].pid == from) {
                if (myedegrees[k].ed == adjwgt[j])
                  myedegrees[k] = myedegrees[--myrinfo->ndegrees];
                else
                  myedegrees[k].ed -= adjwgt[j];
                break;
              }
            }
          }

          /* Add contribution to the .ed of 'to' */
          if (me != to) {
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (myedegrees[k].pid == to) {
                myedegrees[k].ed += adjwgt[j];
                break;
              }
            }
            if (k == myrinfo->ndegrees) {
              myedegrees[myrinfo->ndegrees].pid = to;
              myedegrees[myrinfo->ndegrees++].ed = adjwgt[j];
            }
          }

          ASSERT(myrinfo->ndegrees <= xadj[ii+1]-xadj[ii]);
          ASSERT(CheckRInfo(myrinfo));

        }
        nmoves++;
      }
    }

    graph->nbnd = nbnd;

    if (ctrl->dbglvl&DBG_REFINE) {
      printf("\t [%5.4f %5.4f], Nb: %6d, Nmoves: %5d, Cut: %6d, LB: ",
              npwgts[samin(ncon*nparts, npwgts)], npwgts[samax(ncon*nparts, npwgts)],
              nbnd, nmoves, graph->mincut);
      ComputeHKWayLoadImbalance(ncon, nparts, npwgts, tvec);
      for (i=0; i<ncon; i++)
        printf("%.3f ", tvec[i]);
      printf("\n");
    }

    if (graph->mincut == oldcut)
      break;
  }

  fwspacefree(ctrl, ncon*nparts);
  fwspacefree(ctrl, ncon*nparts);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void MCGreedy_KWayEdgeBalanceHorizontal(CtrlType *ctrl, GraphType *graph, int nparts,
       float *ubvec, int npasses)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, ncon, nbnd, myndegrees, oldgain, gain, nmoves;
  int from, me, to, oldcut;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *perm, *bndptr, *bndind, *moved;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;
  PQueueType queue;
  float *npwgts, *nvwgt, *minwgt, *maxwgt, tvec[MAXNCON];

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;

  where = graph->where;
  npwgts = graph->npwgts;

  /* Setup the weight intervals of the various subdomains */
  minwgt =  fwspacemalloc(ctrl, ncon*nparts);
  maxwgt = fwspacemalloc(ctrl, ncon*nparts);

  for (i=0; i<nparts; i++) {
    for (j=0; j<ncon; j++) {
      maxwgt[i*ncon+j] = ubvec[j]/nparts;
      minwgt[i*ncon+j] = 1.0/(ubvec[j]*nparts);
    }
  }

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxwspacemalloc(ctrl, nvtxs);

  PQueueInit(ctrl, &queue, nvtxs, graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)]);

  if (ctrl->dbglvl&DBG_REFINE) {
    printf("Partitions: [%5.4f %5.4f], Nv-Nb[%6d %6d]. Cut: %6d, LB: ",
            npwgts[samin(ncon*nparts, npwgts)], npwgts[samax(ncon*nparts, npwgts)],
            graph->nvtxs, graph->nbnd, graph->mincut);
    ComputeHKWayLoadImbalance(ncon, nparts, npwgts, tvec);
    for (i=0; i<ncon; i++)
      printf("%.3f ", tvec[i]);
    printf("[B]\n");
  }


  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    /* Check to see if things are out of balance, given the tolerance */
    if (MocIsHBalanced(ncon, nparts, npwgts, ubvec))
      break;

    PQueueReset(&queue);
    idxset(nvtxs, -1, moved);

    oldcut = graph->mincut;
    nbnd = graph->nbnd;

    RandomPermute(nbnd, perm, 1);
    for (ii=0; ii<nbnd; ii++) {
      i = bndind[perm[ii]];
      PQueueInsert(&queue, i, graph->rinfo[i].ed - graph->rinfo[i].id);
      moved[i] = 2;
    }

    nmoves = 0;
    for (;;) {
      if ((i = PQueueGetMax(&queue)) == -1)
        break;
      moved[i] = 1;

      myrinfo = graph->rinfo+i;
      from = where[i];
      nvwgt = graph->nvwgt+i*ncon;

      if (AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, -1.0, nvwgt, minwgt+from*ncon))
        continue;   /* This cannot be moved! */

      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      for (k=0; k<myndegrees; k++) {
        to = myedegrees[k].pid;
        if (IsHBalanceBetterFT(ncon, nparts, npwgts+from*ncon, npwgts+to*ncon, nvwgt, ubvec))
          break;
      }
      if (k == myndegrees)
        continue;  /* break out if you did not find a candidate */

      for (j=k+1; j<myndegrees; j++) {
        to = myedegrees[j].pid;
        if (IsHBalanceBetterTT(ncon, nparts, npwgts+myedegrees[k].pid*ncon, npwgts+to*ncon, nvwgt, ubvec))
          k = j;
      }

      to = myedegrees[k].pid;

      j = 0;
      if (!AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, 0.0, nvwgt, maxwgt+from*ncon))
        j++;
      if (myedegrees[k].ed-myrinfo->id >= 0)
        j++;
      if (!AreAllHVwgtsAbove(ncon, 1.0, npwgts+to*ncon, 0.0, nvwgt, minwgt+to*ncon) &&
          AreAllHVwgtsBelow(ncon, 1.0, npwgts+to*ncon, 1.0, nvwgt, maxwgt+to*ncon))
        j++;
      if (j == 0)
        continue;

/* DELETE
      if (myedegrees[k].ed-myrinfo->id < 0 &&
          AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, 0.0, nvwgt, maxwgt+from*ncon) &&
          AreAllHVwgtsAbove(ncon, 1.0, npwgts+to*ncon, 0.0, nvwgt, minwgt+to*ncon) &&
          AreAllHVwgtsBelow(ncon, 1.0, npwgts+to*ncon, 1.0, nvwgt, maxwgt+to*ncon))
        continue;
*/
      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      graph->mincut -= myedegrees[k].ed-myrinfo->id;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

      /* Update where, weight, and ID/ED information of the vertex you moved */
      saxpy(ncon, 1.0, nvwgt, 1, npwgts+to*ncon, 1);
      saxpy(ncon, -1.0, nvwgt, 1, npwgts+from*ncon, 1);
      where[i] = to;
      myrinfo->ed += myrinfo->id-myedegrees[k].ed;
      SWAP(myrinfo->id, myedegrees[k].ed, j);
      if (myedegrees[k].ed == 0)
        myedegrees[k] = myedegrees[--myrinfo->ndegrees];
      else
        myedegrees[k].pid = from;

      if (myrinfo->ed == 0)
        BNDDelete(nbnd, bndind, bndptr, i);

      /* Update the degrees of adjacent vertices */
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        ii = adjncy[j];
        me = where[ii];

        myrinfo = graph->rinfo+ii;
        if (myrinfo->edegrees == NULL) {
          myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
          ctrl->wspace.cdegree += xadj[ii+1]-xadj[ii];
        }
        myedegrees = myrinfo->edegrees;

        ASSERT(CheckRInfo(myrinfo));

        oldgain = (myrinfo->ed-myrinfo->id);

        if (me == from) {
          INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);

          if (myrinfo->ed > 0 && bndptr[ii] == -1)
            BNDInsert(nbnd, bndind, bndptr, ii);
        }
        else if (me == to) {
          INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);

          if (myrinfo->ed == 0 && bndptr[ii] != -1)
            BNDDelete(nbnd, bndind, bndptr, ii);
        }

        /* Remove contribution from the .ed of 'from' */
        if (me != from) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (myedegrees[k].pid == from) {
              if (myedegrees[k].ed == adjwgt[j])
                myedegrees[k] = myedegrees[--myrinfo->ndegrees];
              else
                myedegrees[k].ed -= adjwgt[j];
              break;
            }
          }
        }

        /* Add contribution to the .ed of 'to' */
        if (me != to) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (myedegrees[k].pid == to) {
              myedegrees[k].ed += adjwgt[j];
              break;
            }
          }
          if (k == myrinfo->ndegrees) {
            myedegrees[myrinfo->ndegrees].pid = to;
            myedegrees[myrinfo->ndegrees++].ed = adjwgt[j];
          }
        }


        /* Update the queue */
        if (me == to || me == from) {
          gain = myrinfo->ed-myrinfo->id;
          if (moved[ii] == 2) {
            if (myrinfo->ed > 0)
              PQueueUpdate(&queue, ii, oldgain, gain);
            else {
              PQueueDelete(&queue, ii, oldgain);
              moved[ii] = -1;
            }
          }
          else if (moved[ii] == -1 && myrinfo->ed > 0) {
            PQueueInsert(&queue, ii, gain);
            moved[ii] = 2;
          }
        }

        ASSERT(myrinfo->ndegrees <= xadj[ii+1]-xadj[ii]);
        ASSERT(CheckRInfo(myrinfo));
      }
      nmoves++;
    }

    graph->nbnd = nbnd;

    if (ctrl->dbglvl&DBG_REFINE) {
      printf("\t [%5.4f %5.4f], Nb: %6d, Nmoves: %5d, Cut: %6d, LB: ",
              npwgts[samin(ncon*nparts, npwgts)], npwgts[samax(ncon*nparts, npwgts)],
              nbnd, nmoves, graph->mincut);
      ComputeHKWayLoadImbalance(ncon, nparts, npwgts, tvec);
      for (i=0; i<ncon; i++)
        printf("%.3f ", tvec[i]);
      printf("\n");
    }

    if (nmoves == 0)
      break;
  }

  PQueueFree(ctrl, &queue);

  fwspacefree(ctrl, ncon*nparts);
  fwspacefree(ctrl, ncon*nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}





/*************************************************************************
* This function checks if the vertex weights of two vertices are below
* a given set of values
**************************************************************************/
int AreAllHVwgtsBelow(int ncon, float alpha, float *vwgt1, float beta, float *vwgt2, float *limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] > limit[i])
      return 0;

  return 1;
}



/*************************************************************************
* This function checks if the vertex weights of two vertices are above
* a given set of values
**************************************************************************/
int AreAllHVwgtsAbove(int ncon, float alpha, float *vwgt1, float beta, float *vwgt2, float *limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] < limit[i])
      return 0;

  return 1;
}


/*************************************************************************
* This function computes the load imbalance over all the constrains
* For now assume that we just want balanced partitionings
**************************************************************************/
void ComputeHKWayLoadImbalance(int ncon, int nparts, float *npwgts, float *lbvec)
{
  int i, j;
  float max;

  for (i=0; i<ncon; i++) {
    max = 0.0;
    for (j=0; j<nparts; j++) {
      if (npwgts[j*ncon+i] > max)
        max = npwgts[j*ncon+i];
    }

    lbvec[i] = max*nparts;
  }
}


/*************************************************************************
* This function determines if a partitioning is horizontally balanced
**************************************************************************/
int MocIsHBalanced(int ncon, int nparts, float *npwgts, float *ubvec)
{
  int i, j;
  float max;

  for (i=0; i<ncon; i++) {
    max = 0.0;
    for (j=0; j<nparts; j++) {
      if (npwgts[j*ncon+i] > max)
        max = npwgts[j*ncon+i];
    }

    if (ubvec[i] < max*nparts)
      return 0;
  }

  return 1;
}





/*************************************************************************
* This function checks if the pairwise balance of the between the two
* partitions will improve by moving the vertex v from pfrom to pto,
* subject to the target partition weights of tfrom, and tto respectively
**************************************************************************/
int IsHBalanceBetterFT(int ncon, int nparts, float *pfrom, float *pto, float *vwgt, float *ubvec)
{
  int i, j, k;
  float blb1=0.0, alb1=0.0, sblb=0.0, salb=0.0;
  float blb2=0.0, alb2=0.0;
  float temp;

  for (i=0; i<ncon; i++) {
    temp = amax(pfrom[i], pto[i])*nparts/ubvec[i];
    if (blb1 < temp) {
      blb2 = blb1;
      blb1 = temp;
    }
    else if (blb2 < temp)
      blb2 = temp;
    sblb += temp;

    temp = amax(pfrom[i]-vwgt[i], pto[i]+vwgt[i])*nparts/ubvec[i];
    if (alb1 < temp) {
      alb2 = alb1;
      alb1 = temp;
    }
    else if (alb2 < temp)
      alb2 = temp;
    salb += temp;
  }

  if (alb1 < blb1)
    return 1;
  if (blb1 < alb1)
    return 0;
  if (alb2 < blb2)
    return 1;
  if (blb2 < alb2)
    return 0;

  return salb < sblb;

}




/*************************************************************************
* This function checks if it will be better to move a vertex to pt2 than
* to pt1 subject to their target weights of tt1 and tt2, respectively
* This routine takes into account the weight of the vertex in question
**************************************************************************/
int IsHBalanceBetterTT(int ncon, int nparts, float *pt1, float *pt2, float *vwgt, float *ubvec)
{
  int i;
  float m11=0.0, m12=0.0, m21=0.0, m22=0.0, sm1=0.0, sm2=0.0, temp;

  for (i=0; i<ncon; i++) {
    temp = (pt1[i]+vwgt[i])*nparts/ubvec[i];
    if (m11 < temp) {
      m12 = m11;
      m11 = temp;
    }
    else if (m12 < temp)
      m12 = temp;
    sm1 += temp;

    temp = (pt2[i]+vwgt[i])*nparts/ubvec[i];
    if (m21 < temp) {
      m22 = m21;
      m21 = temp;
    }
    else if (m22 < temp)
      m22 = temp;
    sm2 += temp;
  }

  if (m21 < m11)
    return 1;
  if (m21 > m11)
    return 0;
  if (m22 < m12)
    return 1;
  if (m22 > m12)
    return 0;

  return sm2 < sm1;
}

