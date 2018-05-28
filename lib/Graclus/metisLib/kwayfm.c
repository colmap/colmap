/*
 * kwayfm.c
 *
 * This file contains code that implements the multilevel k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: kwayfm.c,v 1.1 1998/11/27 17:59:16 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Random_KWayEdgeRefine(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor, int npasses, int ffactor)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, nmoves, nbnd, tvwgt, myndegrees;
  int from, me, to, oldcut, vwgt, gain;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  bndptr = graph->bndptr;
  bndind = graph->bndind;

  where = graph->where;
  pwgts = graph->pwgts;

  /* Setup the weight intervals of the various subdomains */
  minwgt =  idxwspacemalloc(ctrl, nparts);
  maxwgt = idxwspacemalloc(ctrl, nparts);
  itpwgts = idxwspacemalloc(ctrl, nparts);
  tvwgt = idxsum(nparts, pwgts);
  ASSERT(tvwgt == idxsum(nvtxs, graph->vwgt));

  for (i=0; i<nparts; i++) {
    itpwgts[i] = floor(tpwgts[i]*tvwgt);
    maxwgt[i] = floor(tpwgts[i]*tvwgt*ubfactor);
    minwgt[i] = floor(tpwgts[i]*tvwgt*(1.0/ubfactor));
  }

  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d]-[%6d %6d], Balance: %5.3f, Nv-Nb[%6d %6d]. Cut: %6d\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut));

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
        vwgt = graph->vwgt[i];

        if (myrinfo->id > 0 && pwgts[from]-vwgt < minwgt[from])
          continue;   /* This cannot be moved! */

        myedegrees = myrinfo->edegrees;
        myndegrees = myrinfo->ndegrees;

        j = myrinfo->id;
        for (k=0; k<myndegrees; k++) {
          to = myedegrees[k].pid;
          gain = myedegrees[k].ed-j; /* j = myrinfo->id. Allow good nodes to move */
          if (pwgts[to]+vwgt <= maxwgt[to]+ffactor*gain && gain >= 0)
            break;
        }
        if (k == myndegrees)
          continue;  /* break out if you did not find a candidate */

        for (j=k+1; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          if ((myedegrees[j].ed > myedegrees[k].ed && pwgts[to]+vwgt <= maxwgt[to]) ||
              (myedegrees[j].ed == myedegrees[k].ed &&
               itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid]))
            k = j;
        }

        to = myedegrees[k].pid;

        j = 0;
        if (myedegrees[k].ed-myrinfo->id > 0)
          j = 1;
        else if (myedegrees[k].ed-myrinfo->id == 0) {
          if ((iii&7) == 0 || pwgts[from] >= maxwgt[from] || itpwgts[from]*(pwgts[to]+vwgt) < itpwgts[to]*pwgts[from])
            j = 1;
        }
        if (j == 0)
          continue;

        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to'
        *======================================================================*/
        graph->mincut -= myedegrees[k].ed-myrinfo->id;

        IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

        /* Update where, weight, and ID/ED information of the vertex you moved */
        where[i] = to;
        INC_DEC(pwgts[to], pwgts[from], vwgt);
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

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, Vol: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut, ComputeVolume(graph, where)));

    if (graph->mincut == oldcut)
      break;
  }

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
}






/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Greedy_KWayEdgeRefine(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor, int npasses)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, nbnd, tvwgt, myndegrees, oldgain, gain;
  int from, me, to, oldcut, vwgt;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *moved, *itpwgts;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;
  PQueueType queue;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;

  where = graph->where;
  pwgts = graph->pwgts;

  /* Setup the weight intervals of the various subdomains */
  minwgt =  idxwspacemalloc(ctrl, nparts);
  maxwgt = idxwspacemalloc(ctrl, nparts);
  itpwgts = idxwspacemalloc(ctrl, nparts);
  tvwgt = idxsum(nparts, pwgts);
  ASSERT(tvwgt == idxsum(nvtxs, graph->vwgt));

  for (i=0; i<nparts; i++) {
    itpwgts[i] = floor(tpwgts[i]*tvwgt);
    maxwgt[i] = floor(tpwgts[i]*tvwgt*ubfactor);
    minwgt[i] = floor(tpwgts[i]*tvwgt*(1.0/ubfactor));
  }

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxwspacemalloc(ctrl, nvtxs);

  PQueueInit(ctrl, &queue, nvtxs, graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)]);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d]-[%6d %6d], Balance: %5.3f, Nv-Nb[%6d %6d]. Cut: %6d\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut));

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

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

    for (iii=0;;iii++) {
      if ((i = PQueueGetMax(&queue)) == -1)
        break;
      moved[i] = 1;

      myrinfo = graph->rinfo+i;
      from = where[i];
      vwgt = graph->vwgt[i];

      if (pwgts[from]-vwgt < minwgt[from])
        continue;   /* This cannot be moved! */

      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      j = myrinfo->id;
      for (k=0; k<myndegrees; k++) {
        to = myedegrees[k].pid;
        gain = myedegrees[k].ed-j; /* j = myrinfo->id. Allow good nodes to move */
        if (pwgts[to]+vwgt <= maxwgt[to]+gain && gain >= 0)
          break;
      }
      if (k == myndegrees)
        continue;  /* break out if you did not find a candidate */

      for (j=k+1; j<myndegrees; j++) {
        to = myedegrees[j].pid;
        if ((myedegrees[j].ed > myedegrees[k].ed && pwgts[to]+vwgt <= maxwgt[to]) ||
            (myedegrees[j].ed == myedegrees[k].ed &&
             itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid]))
          k = j;
      }

      to = myedegrees[k].pid;

      j = 0;
      if (myedegrees[k].ed-myrinfo->id > 0)
        j = 1;
      else if (myedegrees[k].ed-myrinfo->id == 0) {
        if ((iii&7) == 0 || pwgts[from] >= maxwgt[from] || itpwgts[from]*(pwgts[to]+vwgt) < itpwgts[to]*pwgts[from])
          j = 1;
      }
      if (j == 0)
        continue;

      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      graph->mincut -= myedegrees[k].ed-myrinfo->id;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

      /* Update where, weight, and ID/ED information of the vertex you moved */
      where[i] = to;
      INC_DEC(pwgts[to], pwgts[from], vwgt);
      myrinfo->ed += myrinfo->id-myedegrees[k].ed;
      SWAP(myrinfo->id, myedegrees[k].ed, j);
      if (myedegrees[k].ed == 0)
        myedegrees[k] = myedegrees[--myrinfo->ndegrees];
      else
        myedegrees[k].pid = from;

      if (myrinfo->ed < myrinfo->id)
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

        /* Update the queue */
        if (me == to || me == from) {
          gain = myrinfo->ed-myrinfo->id;
          if (moved[ii] == 2) {
            if (gain >= 0)
              PQueueUpdate(&queue, ii, oldgain, gain);
            else {
              PQueueDelete(&queue, ii, oldgain);
              moved[ii] = -1;
            }
          }
          else if (moved[ii] == -1 && gain >= 0) {
            PQueueInsert(&queue, ii, gain);
            moved[ii] = 2;
          }
        }

        ASSERT(myrinfo->ndegrees <= xadj[ii+1]-xadj[ii]);
        ASSERT(CheckRInfo(myrinfo));

      }
    }

    graph->nbnd = nbnd;

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Cut: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, graph->mincut));

    if (graph->mincut == oldcut)
      break;
  }

  PQueueFree(ctrl, &queue);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Greedy_KWayEdgeBalance(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor, int npasses)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, nbnd, tvwgt, myndegrees, oldgain, gain, nmoves;
  int from, me, to, oldcut, vwgt;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *moved, *itpwgts;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;
  PQueueType queue;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  bndind = graph->bndind;
  bndptr = graph->bndptr;

  where = graph->where;
  pwgts = graph->pwgts;

  /* Setup the weight intervals of the various subdomains */
  minwgt =  idxwspacemalloc(ctrl, nparts);
  maxwgt = idxwspacemalloc(ctrl, nparts);
  itpwgts = idxwspacemalloc(ctrl, nparts);
  tvwgt = idxsum(nparts, pwgts);
  ASSERT(tvwgt == idxsum(nvtxs, graph->vwgt));

  for (i=0; i<nparts; i++) {
    itpwgts[i] = floor(tpwgts[i]*tvwgt);
    maxwgt[i] = floor(tpwgts[i]*tvwgt*ubfactor);
    minwgt[i] = floor(tpwgts[i]*tvwgt*(1.0/ubfactor));
  }

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxwspacemalloc(ctrl, nvtxs);

  PQueueInit(ctrl, &queue, nvtxs, graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)]);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d]-[%6d %6d], Balance: %5.3f, Nv-Nb[%6d %6d]. Cut: %6d [B]\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut));

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    /* Check to see if things are out of balance, given the tolerance */
    for (i=0; i<nparts; i++) {
      if (pwgts[i] > maxwgt[i])
        break;
    }
    if (i == nparts) /* Things are balanced. Return right away */
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
      vwgt = graph->vwgt[i];

      if (pwgts[from]-vwgt < minwgt[from])
        continue;   /* This cannot be moved! */

      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      for (k=0; k<myndegrees; k++) {
        to = myedegrees[k].pid;
        if (pwgts[to]+vwgt <= maxwgt[to] || itpwgts[from]*(pwgts[to]+vwgt) <= itpwgts[to]*pwgts[from])
          break;
      }
      if (k == myndegrees)
        continue;  /* break out if you did not find a candidate */

      for (j=k+1; j<myndegrees; j++) {
        to = myedegrees[j].pid;
        if (itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid])
          k = j;
      }

      to = myedegrees[k].pid;

      if (pwgts[from] < maxwgt[from] && pwgts[to] > minwgt[to] && myedegrees[k].ed-myrinfo->id < 0)
        continue;

      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      graph->mincut -= myedegrees[k].ed-myrinfo->id;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

      /* Update where, weight, and ID/ED information of the vertex you moved */
      where[i] = to;
      INC_DEC(pwgts[to], pwgts[from], vwgt);
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

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut));
  }

  PQueueFree(ctrl, &queue);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}

