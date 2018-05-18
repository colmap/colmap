/*
 * kwayvolfm.c
 *
 * This file contains code that implements the multilevel k-way refinement
 *
 * Started 7/8/98
 * George
 *
 * $Id: kwayvolfm.c,v 1.1 1998/11/27 17:59:17 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Random_KWayVolRefine(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts,
                          float ubfactor, int npasses, int ffactor)
{
  int i, ii, iii, j, jj, k, kk, l, u, pass, nvtxs, nmoves, tvwgt, myndegrees, xgain;
  int from, me, to, oldcut, oldvol, vwgt;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts, *updind, *marker, *phtable;
  VEDegreeType *myedegrees;
  VRInfoType *myrinfo;

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

  updind = idxmalloc(nvtxs, "Random_KWayVolRefine: updind");
  marker = idxsmalloc(nvtxs, 0, "Random_KWayVolRefine: marker");
  phtable = idxsmalloc(nparts, -1, "Random_KWayVolRefine: phtable");

  for (i=0; i<nparts; i++) {
    itpwgts[i] = (int) tpwgts[i]*tvwgt;
    maxwgt[i] = (int) tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = (int) tpwgts[i]*tvwgt*(1.0/ubfactor);
  }

  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("VolPart: [%5d %5d]-[%5d %5d], Balance: %3.2f, Nv-Nb[%5d %5d]. Cut: %5d, Vol: %5d\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut, graph->minvol));

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    oldcut = graph->mincut;
    oldvol = graph->minvol;

    RandomPermute(graph->nbnd, perm, 1);
    for (nmoves=iii=0; iii<graph->nbnd; iii++) {
      ii = perm[iii];
      if (ii >= graph->nbnd)
        continue;
      i = bndind[ii];
      myrinfo = graph->vrinfo+i;

      if (myrinfo->gv >= 0) { /* Total volume gain is too high */
        from = where[i];
        vwgt = graph->vwgt[i];

        if (myrinfo->id > 0 && pwgts[from]-vwgt < minwgt[from])
          continue;   /* This cannot be moved! */

        xgain = (myrinfo->id == 0 && myrinfo->ed > 0 ? graph->vsize[i] : 0);

        myedegrees = myrinfo->edegrees;
        myndegrees = myrinfo->ndegrees;

        for (k=0; k<myndegrees; k++) {
          to = myedegrees[k].pid;
          if (pwgts[to]+vwgt <= maxwgt[to]+ffactor*myedegrees[k].gv && xgain+myedegrees[k].gv >= 0)
            break;
        }
        if (k == myndegrees)
          continue;  /* break out if you did not find a candidate */

        for (j=k+1; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          if (pwgts[to]+vwgt > maxwgt[to])
            continue;
          if (myedegrees[j].gv > myedegrees[k].gv ||
              (myedegrees[j].gv == myedegrees[k].gv && myedegrees[j].ed > myedegrees[k].ed) ||
              (myedegrees[j].gv == myedegrees[k].gv && myedegrees[j].ed == myedegrees[k].ed &&
               itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid]))
            k = j;
        }

        to = myedegrees[k].pid;

        j = 0;
        if (xgain+myedegrees[k].gv > 0 || myedegrees[k].ed-myrinfo->id > 0)
          j = 1;
        else if (myedegrees[k].ed-myrinfo->id == 0) {
          if ((iii&5) == 0 || pwgts[from] >= maxwgt[from] || itpwgts[from]*(pwgts[to]+vwgt) < itpwgts[to]*pwgts[from])
            j = 1;
        }
        if (j == 0)
          continue;

        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to'
        *======================================================================*/
        INC_DEC(pwgts[to], pwgts[from], vwgt);
        graph->mincut -= myedegrees[k].ed-myrinfo->id;
        graph->minvol -= (xgain+myedegrees[k].gv);
        where[i] = to;

        IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d from %3d to %3d. Gain: [%4d %4d]. Cut: %6d, Vol: %6d\n",
              i, from, to, xgain+myedegrees[k].gv, myedegrees[k].ed-myrinfo->id, graph->mincut, graph->minvol));

        KWayVolUpdate(ctrl, graph, i, from, to, marker, phtable, updind);

        nmoves++;

        /* CheckVolKWayPartitionParams(ctrl, graph, nparts); */
      }
    }

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, Vol: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut,
               graph->minvol));

    if (graph->minvol == oldvol && graph->mincut == oldcut)
      break;
  }

  GKfree((void **) &marker, (void **) &updind, (void **) &phtable, LTERM);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
}


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Random_KWayVolRefineMConn(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts,
            float ubfactor, int npasses, int ffactor)
{
  int i, ii, iii, j, jj, k, kk, l, u, pass, nvtxs, nmoves, tvwgt, myndegrees, xgain;
  int from, me, to, oldcut, oldvol, vwgt, nadd, maxndoms;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts, *updind, *marker, *phtable;
  idxtype *pmat, *pmatptr, *ndoms;
  VEDegreeType *myedegrees;
  VRInfoType *myrinfo;

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

  updind = idxmalloc(nvtxs, "Random_KWayVolRefine: updind");
  marker = idxsmalloc(nvtxs, 0, "Random_KWayVolRefine: marker");
  phtable = idxsmalloc(nparts, -1, "Random_KWayVolRefine: phtable");

  pmat = ctrl->wspace.pmat;
  ndoms = idxwspacemalloc(ctrl, nparts);

  ComputeVolSubDomainGraph(graph, nparts, pmat, ndoms);

  for (i=0; i<nparts; i++) {
    itpwgts[i] = (int) tpwgts[i]*tvwgt;
    maxwgt[i] = (int) tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = (int) tpwgts[i]*tvwgt*(1.0/ubfactor);
  }

  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("VolPart: [%5d %5d]-[%5d %5d], Balance: %3.2f, Nv-Nb[%5d %5d]. Cut: %5d, Vol: %5d\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut, graph->minvol));

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    maxndoms = ndoms[idxamax(nparts, ndoms)];

    oldcut = graph->mincut;
    oldvol = graph->minvol;

    RandomPermute(graph->nbnd, perm, 1);
    for (nmoves=iii=0; iii<graph->nbnd; iii++) {
      ii = perm[iii];
      if (ii >= graph->nbnd)
        continue;
      i = bndind[ii];
      myrinfo = graph->vrinfo+i;

      if (myrinfo->gv >= 0) { /* Total volume gain is too high */
        from = where[i];
        vwgt = graph->vwgt[i];

        if (myrinfo->id > 0 && pwgts[from]-vwgt < minwgt[from])
          continue;   /* This cannot be moved! */

        xgain = (myrinfo->id == 0 && myrinfo->ed > 0 ? graph->vsize[i] : 0);

        myedegrees = myrinfo->edegrees;
        myndegrees = myrinfo->ndegrees;

        /* Determine the valid domains */
        for (j=0; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          phtable[to] = 1;
          pmatptr = pmat + to*nparts;
          for (nadd=0, k=0; k<myndegrees; k++) {
            if (k == j)
              continue;

            l = myedegrees[k].pid;
            if (pmatptr[l] == 0) {
              if (ndoms[l] > maxndoms-1) {
                phtable[to] = 0;
                nadd = maxndoms;
                break;
              }
              nadd++;
            }
          }
          if (ndoms[to]+nadd > maxndoms)
            phtable[to] = 0;
          if (nadd == 0)
            phtable[to] = 2;
        }

        for (k=0; k<myndegrees; k++) {
          to = myedegrees[k].pid;
          if (!phtable[to])
            continue;
          if (pwgts[to]+vwgt <= maxwgt[to]+ffactor*myedegrees[k].gv && xgain+myedegrees[k].gv >= 0)
            break;
        }
        if (k == myndegrees)
          continue;  /* break out if you did not find a candidate */

        for (j=k+1; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          if (!phtable[to] || pwgts[to]+vwgt > maxwgt[to])
            continue;
          if (myedegrees[j].gv > myedegrees[k].gv ||
              (myedegrees[j].gv == myedegrees[k].gv && myedegrees[j].ed > myedegrees[k].ed) ||
              (myedegrees[j].gv == myedegrees[k].gv && myedegrees[j].ed == myedegrees[k].ed &&
               itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid]))
            k = j;
        }

        to = myedegrees[k].pid;

        j = 0;
        if (xgain+myedegrees[k].gv > 0 || myedegrees[k].ed-myrinfo->id > 0)
          j = 1;
        else if (myedegrees[k].ed-myrinfo->id == 0) {
          if ((iii&5) == 0 || phtable[myedegrees[k].pid] == 2 || pwgts[from] >= maxwgt[from] || itpwgts[from]*(pwgts[to]+vwgt) < itpwgts[to]*pwgts[from])
            j = 1;
        }

        if (j == 0)
          continue;

        for (j=0; j<myndegrees; j++)
          phtable[myedegrees[j].pid] = -1;


        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to'
        *======================================================================*/
        INC_DEC(pwgts[to], pwgts[from], vwgt);
        graph->mincut -= myedegrees[k].ed-myrinfo->id;
        graph->minvol -= (xgain+myedegrees[k].gv);
        where[i] = to;

        IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d from %3d to %3d. Gain: [%4d %4d]. Cut: %6d, Vol: %6d\n",
              i, from, to, xgain+myedegrees[k].gv, myedegrees[k].ed-myrinfo->id, graph->mincut, graph->minvol));

        /* Update pmat to reflect the move of 'i' */
        pmat[from*nparts+to] += (myrinfo->id-myedegrees[k].ed);
        pmat[to*nparts+from] += (myrinfo->id-myedegrees[k].ed);
        if (pmat[from*nparts+to] == 0) {
          ndoms[from]--;
          if (ndoms[from]+1 == maxndoms)
            maxndoms = ndoms[idxamax(nparts, ndoms)];
        }
        if (pmat[to*nparts+from] == 0) {
          ndoms[to]--;
          if (ndoms[to]+1 == maxndoms)
            maxndoms = ndoms[idxamax(nparts, ndoms)];
        }

        for (j=xadj[i]; j<xadj[i+1]; j++) {
          ii = adjncy[j];
          me = where[ii];

          /* Update pmat to reflect the move of 'i' for domains other than 'from' and 'to' */
          if (me != from && me != to) {
            pmat[me*nparts+from] -= adjwgt[j];
            pmat[from*nparts+me] -= adjwgt[j];
            if (pmat[me*nparts+from] == 0) {
              ndoms[me]--;
              if (ndoms[me]+1 == maxndoms)
                maxndoms = ndoms[idxamax(nparts, ndoms)];
            }
            if (pmat[from*nparts+me] == 0) {
              ndoms[from]--;
              if (ndoms[from]+1 == maxndoms)
                maxndoms = ndoms[idxamax(nparts, ndoms)];
            }

            if (pmat[me*nparts+to] == 0) {
              ndoms[me]++;
              if (ndoms[me] > maxndoms) {
                printf("You just increased the maxndoms: %d %d\n", ndoms[me], maxndoms);
                maxndoms = ndoms[me];
              }
            }
            if (pmat[to*nparts+me] == 0) {
              ndoms[to]++;
              if (ndoms[to] > maxndoms) {
                printf("You just increased the maxndoms: %d %d\n", ndoms[to], maxndoms);
                maxndoms = ndoms[to];
              }
            }
            pmat[me*nparts+to] += adjwgt[j];
            pmat[to*nparts+me] += adjwgt[j];
          }
        }

        KWayVolUpdate(ctrl, graph, i, from, to, marker, phtable, updind);

        nmoves++;

        /* CheckVolKWayPartitionParams(ctrl, graph, nparts); */
      }
    }

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, Vol: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut,
               graph->minvol));

    if (graph->minvol == oldvol && graph->mincut == oldcut)
      break;
  }

  GKfree((void **) &marker, (void **) &updind, (void **) &phtable, LTERM);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
}




/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Greedy_KWayVolBalance(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts,
                           float ubfactor, int npasses)
{
  int i, ii, iii, j, jj, k, kk, l, u, pass, nvtxs, nmoves, tvwgt, myndegrees, xgain;
  int from, me, to, vwgt, gain;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *moved, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts, *updind, *marker, *phtable;
  VEDegreeType *myedegrees;
  VRInfoType *myrinfo;
  PQueueType queue;

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

  updind = idxmalloc(nvtxs, "Random_KWayVolRefine: updind");
  marker = idxsmalloc(nvtxs, 0, "Random_KWayVolRefine: marker");
  phtable = idxsmalloc(nparts, -1, "Random_KWayVolRefine: phtable");

  for (i=0; i<nparts; i++) {
    itpwgts[i] = (int) tpwgts[i]*tvwgt;
    maxwgt[i] = (int) tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = (int) tpwgts[i]*tvwgt*(1.0/ubfactor);
  }

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxwspacemalloc(ctrl, nvtxs);

  PQueueInit(ctrl, &queue, nvtxs, graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)]);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("VolPart: [%5d %5d]-[%5d %5d], Balance: %3.2f, Nv-Nb[%5d %5d]. Cut: %5d, Vol: %5d [B]\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut, graph->minvol));


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

    RandomPermute(graph->nbnd, perm, 1);
    for (ii=0; ii<graph->nbnd; ii++) {
      i = bndind[perm[ii]];
      PQueueInsert(&queue, i, graph->vrinfo[i].gv);
      moved[i] = 2;
    }

    for (nmoves=0;;) {
      if ((i = PQueueGetMax(&queue)) == -1)
        break;
      moved[i] = 1;

      myrinfo = graph->vrinfo+i;
      from = where[i];
      vwgt = graph->vwgt[i];

      if (pwgts[from]-vwgt < minwgt[from])
        continue;   /* This cannot be moved! */

      xgain = (myrinfo->id == 0 && myrinfo->ed > 0 ? graph->vsize[i] : 0);

      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      for (k=0; k<myndegrees; k++) {
        to = myedegrees[k].pid;
        if (pwgts[to]+vwgt <= maxwgt[to] ||
            itpwgts[from]*(pwgts[to]+vwgt) <= itpwgts[to]*pwgts[from])
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

      if (pwgts[from] < maxwgt[from] && pwgts[to] > minwgt[to] &&
          (xgain+myedegrees[k].gv < 0 ||
           (xgain+myedegrees[k].gv == 0 &&  myedegrees[k].ed-myrinfo->id < 0))
         )
        continue;


      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      INC_DEC(pwgts[to], pwgts[from], vwgt);
      graph->mincut -= myedegrees[k].ed-myrinfo->id;
      graph->minvol -= (xgain+myedegrees[k].gv);
      where[i] = to;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d from %3d to %3d. Gain: [%4d %4d]. Cut: %6d, Vol: %6d\n",
            i, from, to, xgain+myedegrees[k].gv, myedegrees[k].ed-myrinfo->id, graph->mincut, graph->minvol));

      KWayVolUpdate(ctrl, graph, i, from, to, marker, phtable, updind);

      nmoves++;

      /*CheckVolKWayPartitionParams(ctrl, graph, nparts); */
    }

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, Vol: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut,
               graph->minvol));

  }

  GKfree((void **) &marker, (void **) &updind, (void **) &phtable, LTERM);

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
void Greedy_KWayVolBalanceMConn(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts,
                                float ubfactor, int npasses)
{
  int i, ii, iii, j, jj, k, kk, l, u, pass, nvtxs, nmoves, tvwgt, myndegrees, xgain;
  int from, me, to, vwgt, gain, maxndoms, nadd;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *moved, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts, *updind, *marker, *phtable;
  idxtype *pmat, *pmatptr, *ndoms;
  VEDegreeType *myedegrees;
  VRInfoType *myrinfo;
  PQueueType queue;

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

  updind = idxmalloc(nvtxs, "Random_KWayVolRefine: updind");
  marker = idxsmalloc(nvtxs, 0, "Random_KWayVolRefine: marker");
  phtable = idxsmalloc(nparts, -1, "Random_KWayVolRefine: phtable");

  pmat = ctrl->wspace.pmat;
  ndoms = idxwspacemalloc(ctrl, nparts);

  ComputeVolSubDomainGraph(graph, nparts, pmat, ndoms);

  for (i=0; i<nparts; i++) {
    itpwgts[i] = (int) tpwgts[i]*tvwgt;
    maxwgt[i] = (int) tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = (int) tpwgts[i]*tvwgt*(1.0/ubfactor);
  }

  perm = idxwspacemalloc(ctrl, nvtxs);
  moved = idxwspacemalloc(ctrl, nvtxs);

  PQueueInit(ctrl, &queue, nvtxs, graph->adjwgtsum[idxamax(nvtxs, graph->adjwgtsum)]);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("VolPart: [%5d %5d]-[%5d %5d], Balance: %3.2f, Nv-Nb[%5d %5d]. Cut: %5d, Vol: %5d [B]\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut, graph->minvol));


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

    RandomPermute(graph->nbnd, perm, 1);
    for (ii=0; ii<graph->nbnd; ii++) {
      i = bndind[perm[ii]];
      PQueueInsert(&queue, i, graph->vrinfo[i].gv);
      moved[i] = 2;
    }

    maxndoms = ndoms[idxamax(nparts, ndoms)];

    for (nmoves=0;;) {
      if ((i = PQueueGetMax(&queue)) == -1)
        break;
      moved[i] = 1;

      myrinfo = graph->vrinfo+i;
      from = where[i];
      vwgt = graph->vwgt[i];

      if (pwgts[from]-vwgt < minwgt[from])
        continue;   /* This cannot be moved! */

      xgain = (myrinfo->id == 0 && myrinfo->ed > 0 ? graph->vsize[i] : 0);

      myedegrees = myrinfo->edegrees;
      myndegrees = myrinfo->ndegrees;

      /* Determine the valid domains */
      for (j=0; j<myndegrees; j++) {
        to = myedegrees[j].pid;
        phtable[to] = 1;
        pmatptr = pmat + to*nparts;
        for (nadd=0, k=0; k<myndegrees; k++) {
          if (k == j)
            continue;

          l = myedegrees[k].pid;
          if (pmatptr[l] == 0) {
            if (ndoms[l] > maxndoms-1) {
              phtable[to] = 0;
              nadd = maxndoms;
              break;
            }
            nadd++;
          }
        }
        if (ndoms[to]+nadd > maxndoms)
          phtable[to] = 0;
      }

      for (k=0; k<myndegrees; k++) {
        to = myedegrees[k].pid;
        if (!phtable[to])
          continue;
        if (pwgts[to]+vwgt <= maxwgt[to] ||
            itpwgts[from]*(pwgts[to]+vwgt) <= itpwgts[to]*pwgts[from])
          break;
      }
      if (k == myndegrees)
        continue;  /* break out if you did not find a candidate */

      for (j=k+1; j<myndegrees; j++) {
        to = myedegrees[j].pid;
        if (!phtable[to])
          continue;
        if (itpwgts[myedegrees[k].pid]*pwgts[to] < itpwgts[to]*pwgts[myedegrees[k].pid])
          k = j;
      }

      to = myedegrees[k].pid;

      for (j=0; j<myndegrees; j++)
        phtable[myedegrees[j].pid] = -1;

      if (pwgts[from] < maxwgt[from] && pwgts[to] > minwgt[to] &&
          (xgain+myedegrees[k].gv < 0 ||
           (xgain+myedegrees[k].gv == 0 &&  myedegrees[k].ed-myrinfo->id < 0))
         )
        continue;


      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      INC_DEC(pwgts[to], pwgts[from], vwgt);
      graph->mincut -= myedegrees[k].ed-myrinfo->id;
      graph->minvol -= (xgain+myedegrees[k].gv);
      where[i] = to;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d from %3d to %3d. Gain: [%4d %4d]. Cut: %6d, Vol: %6d\n",
            i, from, to, xgain+myedegrees[k].gv, myedegrees[k].ed-myrinfo->id, graph->mincut, graph->minvol));

      /* Update pmat to reflect the move of 'i' */
      pmat[from*nparts+to] += (myrinfo->id-myedegrees[k].ed);
      pmat[to*nparts+from] += (myrinfo->id-myedegrees[k].ed);
      if (pmat[from*nparts+to] == 0) {
        ndoms[from]--;
        if (ndoms[from]+1 == maxndoms)
          maxndoms = ndoms[idxamax(nparts, ndoms)];
      }
      if (pmat[to*nparts+from] == 0) {
        ndoms[to]--;
        if (ndoms[to]+1 == maxndoms)
          maxndoms = ndoms[idxamax(nparts, ndoms)];
      }

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        ii = adjncy[j];
        me = where[ii];

        /* Update pmat to reflect the move of 'i' for domains other than 'from' and 'to' */
        if (me != from && me != to) {
          pmat[me*nparts+from] -= adjwgt[j];
          pmat[from*nparts+me] -= adjwgt[j];
          if (pmat[me*nparts+from] == 0) {
            ndoms[me]--;
            if (ndoms[me]+1 == maxndoms)
              maxndoms = ndoms[idxamax(nparts, ndoms)];
          }
          if (pmat[from*nparts+me] == 0) {
            ndoms[from]--;
            if (ndoms[from]+1 == maxndoms)
              maxndoms = ndoms[idxamax(nparts, ndoms)];
          }

          if (pmat[me*nparts+to] == 0) {
            ndoms[me]++;
            if (ndoms[me] > maxndoms) {
              printf("You just increased the maxndoms: %d %d\n", ndoms[me], maxndoms);
              maxndoms = ndoms[me];
            }
          }
          if (pmat[to*nparts+me] == 0) {
            ndoms[to]++;
            if (ndoms[to] > maxndoms) {
              printf("You just increased the maxndoms: %d %d\n", ndoms[to], maxndoms);
              maxndoms = ndoms[to];
            }
          }
          pmat[me*nparts+to] += adjwgt[j];
          pmat[to*nparts+me] += adjwgt[j];
        }
      }

      KWayVolUpdate(ctrl, graph, i, from, to, marker, phtable, updind);

      nmoves++;

      /*CheckVolKWayPartitionParams(ctrl, graph, nparts); */
    }

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, Vol: %6d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut,
               graph->minvol));

  }

  GKfree((void **) &marker, (void **) &updind, (void **) &phtable, LTERM);

  PQueueFree(ctrl, &queue);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
}




/*************************************************************************
* This function updates the edge and volume gains as a result of moving
* v from 'from' to 'to'.
* The working arrays marker and phtable are assumed to be initialized to
* -1, and they left to -1 upon return
**************************************************************************/
void KWayVolUpdate(CtrlType *ctrl, GraphType *graph, int v, int from, int to,
                   idxtype *marker, idxtype *phtable, idxtype *updind)
{
  int ii, iii, j, jj, k, kk, l, u, nupd, other, me, myidx;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *where;
  VEDegreeType *myedegrees, *oedegrees;
  VRInfoType *myrinfo, *orinfo;

  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  vsize = graph->vsize;
  where = graph->where;

  myrinfo = graph->vrinfo+v;
  myedegrees = myrinfo->edegrees;


  /*======================================================================
   * Remove the contributions on the gain made by 'v'.
   *=====================================================================*/
  for (k=0; k<myrinfo->ndegrees; k++)
    phtable[myedegrees[k].pid] = k;
  phtable[from] = k;

  myidx = phtable[to];  /* Keep track of the index in myedegrees of the 'to' domain */

  for (j=xadj[v]; j<xadj[v+1]; j++) {
    ii = adjncy[j];
    other = where[ii];
    orinfo = graph->vrinfo+ii;
    oedegrees = orinfo->edegrees;

    if (other == from) {
      for (k=0; k<orinfo->ndegrees; k++) {
        if (phtable[oedegrees[k].pid] == -1)
          oedegrees[k].gv += vsize[v];
      }
    }
    else {
      ASSERT(phtable[other] != -1);

      if (myedegrees[phtable[other]].ned > 1) {
        for (k=0; k<orinfo->ndegrees; k++) {
          if (phtable[oedegrees[k].pid] == -1)
            oedegrees[k].gv += vsize[v];
        }
      }
      else { /* There is only one connection */
        for (k=0; k<orinfo->ndegrees; k++) {
          if (phtable[oedegrees[k].pid] != -1)
            oedegrees[k].gv -= vsize[v];
        }
      }
    }
  }

  for (k=0; k<myrinfo->ndegrees; k++)
    phtable[myedegrees[k].pid] = -1;
  phtable[from] = -1;


  /*======================================================================
   * Update the id/ed of vertex 'v'
   *=====================================================================*/
  myrinfo->ed += myrinfo->id-myedegrees[myidx].ed;
  SWAP(myrinfo->id, myedegrees[myidx].ed, j);
  SWAP(myrinfo->nid, myedegrees[myidx].ned, j);
  if (myedegrees[myidx].ed == 0)
    myedegrees[myidx] = myedegrees[--myrinfo->ndegrees];
  else
    myedegrees[myidx].pid = from;

  /*======================================================================
   * Update the degrees of adjacent vertices and their volume gains
   *=====================================================================*/
  marker[v] = 1;
  updind[0] = v;
  nupd = 1;
  for (j=xadj[v]; j<xadj[v+1]; j++) {
    ii = adjncy[j];
    me = where[ii];

    if (!marker[ii]) {  /* The marking is done for boundary and max gv calculations */
      marker[ii] = 2;
      updind[nupd++] = ii;
    }

    myrinfo = graph->vrinfo+ii;
    if (myrinfo->edegrees == NULL) {
      myrinfo->edegrees = ctrl->wspace.vedegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += xadj[ii+1]-xadj[ii];
    }
    myedegrees = myrinfo->edegrees;

    if (me == from) {
      INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);
      myrinfo->nid--;
    }
    else if (me == to) {
      INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);
      myrinfo->nid++;
    }

    /* Remove the edgeweight from the 'pid == from' entry of the vertex */
    if (me != from) {
      for (k=0; k<myrinfo->ndegrees; k++) {
        if (myedegrees[k].pid == from) {
          if (myedegrees[k].ned == 1) {
            myedegrees[k] = myedegrees[--myrinfo->ndegrees];
            marker[ii] = 1;  /* You do a complete .gv calculation */

            /* All vertices adjacent to 'ii' need to be updated */
            for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
              u = adjncy[jj];
              other = where[u];
              orinfo = graph->vrinfo+u;
              oedegrees = orinfo->edegrees;

              for (kk=0; kk<orinfo->ndegrees; kk++) {
                if (oedegrees[kk].pid == from) {
                  oedegrees[kk].gv -= vsize[ii];
                  break;
                }
              }
            }
          }
          else {
            myedegrees[k].ed -= adjwgt[j];
            myedegrees[k].ned--;

            /* Update the gv due to single 'ii' connection to 'from' */
            if (myedegrees[k].ned == 1) {
              /* find the vertex 'u' that 'ii' was connected into 'from' */
              for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
                u = adjncy[jj];
                other = where[u];
                orinfo = graph->vrinfo+u;
                oedegrees = orinfo->edegrees;

                if (other == from) {
                  for (kk=0; kk<orinfo->ndegrees; kk++)
                    oedegrees[kk].gv += vsize[ii];
                  break;
                }
              }
            }
          }

          break;
        }
      }
    }

    /* Add the edgeweight to the 'pid == to' entry of the vertex */
    if (me != to) {
      for (k=0; k<myrinfo->ndegrees; k++) {
        if (myedegrees[k].pid == to) {
          myedegrees[k].ed += adjwgt[j];
          myedegrees[k].ned++;

          /* Update the gv due to non-single 'ii' connection to 'to' */
          if (myedegrees[k].ned == 2) {
            /* find the vertex 'u' that 'ii' was connected into 'to' */
            for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
              u = adjncy[jj];
              other = where[u];
              orinfo = graph->vrinfo+u;
              oedegrees = orinfo->edegrees;

              if (u != v && other == to) {
                for (kk=0; kk<orinfo->ndegrees; kk++)
                  oedegrees[kk].gv -= vsize[ii];
                break;
              }
            }
          }
          break;
        }
      }

      if (k == myrinfo->ndegrees) {
        myedegrees[myrinfo->ndegrees].pid = to;
        myedegrees[myrinfo->ndegrees].ed = adjwgt[j];
        myedegrees[myrinfo->ndegrees++].ned = 1;
        marker[ii] = 1;  /* You do a complete .gv calculation */

        /* All vertices adjacent to 'ii' need to be updated */
        for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
          u = adjncy[jj];
          other = where[u];
          orinfo = graph->vrinfo+u;
          oedegrees = orinfo->edegrees;

          for (kk=0; kk<orinfo->ndegrees; kk++) {
            if (oedegrees[kk].pid == to) {
              oedegrees[kk].gv += vsize[ii];
              if (!marker[u]) { /* Need to update boundary etc */
                marker[u] = 2;
                updind[nupd++] = u;
              }
              break;
            }
          }
        }
      }
    }

    ASSERT(myrinfo->ndegrees <= xadj[ii+1]-xadj[ii]);
  }

  /*======================================================================
   * Add the contributions on the volume gain due to 'v'
   *=====================================================================*/
  myrinfo = graph->vrinfo+v;
  myedegrees = myrinfo->edegrees;
  for (k=0; k<myrinfo->ndegrees; k++)
    phtable[myedegrees[k].pid] = k;
  phtable[to] = k;

  for (j=xadj[v]; j<xadj[v+1]; j++) {
    ii = adjncy[j];
    other = where[ii];
    orinfo = graph->vrinfo+ii;
    oedegrees = orinfo->edegrees;

    if (other == to) {
      for (k=0; k<orinfo->ndegrees; k++) {
        if (phtable[oedegrees[k].pid] == -1)
          oedegrees[k].gv -= vsize[v];
      }
    }
    else {
      ASSERT(phtable[other] != -1);

      if (myedegrees[phtable[other]].ned > 1) {
        for (k=0; k<orinfo->ndegrees; k++) {
          if (phtable[oedegrees[k].pid] == -1)
            oedegrees[k].gv -= vsize[v];
        }
      }
      else { /* There is only one connection */
        for (k=0; k<orinfo->ndegrees; k++) {
          if (phtable[oedegrees[k].pid] != -1)
            oedegrees[k].gv += vsize[v];
        }
      }
    }
  }
  for (k=0; k<myrinfo->ndegrees; k++)
    phtable[myedegrees[k].pid] = -1;
  phtable[to] = -1;


  /*======================================================================
   * Recompute the volume information of the 'hard' nodes, and update the
   * max volume gain for all the update vertices
   *=====================================================================*/
  ComputeKWayVolume(graph, nupd, updind, marker, phtable);


  /*======================================================================
   * Maintain a consistent boundary
   *=====================================================================*/
  for (j=0; j<nupd; j++) {
    k = updind[j];
    marker[k] = 0;
    myrinfo = graph->vrinfo+k;

    if ((myrinfo->gv >= 0 || myrinfo->ed-myrinfo->id >= 0) && graph->bndptr[k] == -1)
      BNDInsert(graph->nbnd, graph->bndind, graph->bndptr, k);

    if (myrinfo->gv < 0 && myrinfo->ed-myrinfo->id < 0 && graph->bndptr[k] != -1)
      BNDDelete(graph->nbnd, graph->bndind, graph->bndptr, k);
  }

}




/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void ComputeKWayVolume(GraphType *graph, int nupd, idxtype *updind, idxtype *marker, idxtype *phtable)
{
  int ii, iii, i, j, k, kk, l, nvtxs, me, other, pid;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *where;
  VRInfoType *rinfo, *myrinfo, *orinfo;
  VEDegreeType *myedegrees, *oedegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->vrinfo;


  /*------------------------------------------------------------
  / Compute now the iv/ev degrees
  /------------------------------------------------------------*/
  for (iii=0; iii<nupd; iii++) {
    i = updind[iii];
    me = where[i];

    myrinfo = rinfo+i;
    myedegrees = myrinfo->edegrees;

    if (marker[i] == 1) {  /* Only complete gain updates go through */
      for (k=0; k<myrinfo->ndegrees; k++)
        myedegrees[k].gv = 0;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        ii = adjncy[j];
        other = where[ii];
        orinfo = rinfo+ii;
        oedegrees = orinfo->edegrees;

        for (kk=0; kk<orinfo->ndegrees; kk++)
          phtable[oedegrees[kk].pid] = kk;
        phtable[other] = 1;

        if (me == other) {
          /* Find which domains 'i' is connected and 'ii' is not and update their gain */
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (phtable[myedegrees[k].pid] == -1)
              myedegrees[k].gv -= vsize[ii];
          }
        }
        else {
          ASSERT(phtable[me] != -1);

          /* I'm the only connection of 'ii' in 'me' */
          if (oedegrees[phtable[me]].ned == 1) {
            /* Increase the gains for all the common domains between 'i' and 'ii' */
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (phtable[myedegrees[k].pid] != -1)
                myedegrees[k].gv += vsize[ii];
            }
          }
          else {
            /* Find which domains 'i' is connected and 'ii' is not and update their gain */
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (phtable[myedegrees[k].pid] == -1)
                myedegrees[k].gv -= vsize[ii];
            }
          }
        }

        for (kk=0; kk<orinfo->ndegrees; kk++)
          phtable[oedegrees[kk].pid] = -1;
        phtable[other] = -1;

      }
    }

    myrinfo->gv = -MAXIDX;
    for (k=0; k<myrinfo->ndegrees; k++) {
      if (myedegrees[k].gv > myrinfo->gv)
        myrinfo->gv = myedegrees[k].gv;
    }
    if (myrinfo->ed > 0 && myrinfo->id == 0)
      myrinfo->gv += vsize[i];

  }

}



/*************************************************************************
* This function computes the total volume
**************************************************************************/
int ComputeVolume(GraphType *graph, idxtype *where)
{
  int i, j, k, me, nvtxs, nparts, totalv;
  idxtype *xadj, *adjncy, *vsize, *marker;


  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vsize = (graph->vsize == NULL ? graph->vwgt : graph->vsize);

  nparts = where[idxamax(nvtxs, where)]+1;
  marker = idxsmalloc(nparts, -1, "ComputeVolume: marker");

  totalv = 0;

  for (i=0; i<nvtxs; i++) {
    marker[where[i]] = i;
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = where[adjncy[j]];
      if (marker[k] != i) {
        marker[k] = i;
        totalv += vsize[i];
      }
    }
  }

  free(marker);

  return totalv;
}





/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void CheckVolKWayPartitionParams(CtrlType *ctrl, GraphType *graph, int nparts)
{
  int i, ii, j, k, kk, l, nvtxs, nbnd, mincut, minvol, me, other, pid;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *pwgts, *where, *bndind, *bndptr;
  VRInfoType *rinfo, *myrinfo, *orinfo, tmprinfo;
  VEDegreeType *myedegrees, *oedegrees, *tmpdegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->vrinfo;

  tmpdegrees = (VEDegreeType *)GKmalloc(nparts*sizeof(VEDegreeType), "CheckVolKWayPartitionParams: tmpdegrees");

  /*------------------------------------------------------------
  / Compute now the iv/ev degrees
  /------------------------------------------------------------*/
  for (i=0; i<nvtxs; i++) {
    me = where[i];

    myrinfo = rinfo+i;
    myedegrees = myrinfo->edegrees;

    for (k=0; k<myrinfo->ndegrees; k++)
      tmpdegrees[k] = myedegrees[k];

    tmprinfo.ndegrees = myrinfo->ndegrees;
    tmprinfo.id = myrinfo->id;
    tmprinfo.ed = myrinfo->ed;

    myrinfo = &tmprinfo;
    myedegrees = tmpdegrees;


    for (k=0; k<myrinfo->ndegrees; k++)
      myedegrees[k].gv = 0;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      ii = adjncy[j];
      other = where[ii];
      orinfo = rinfo+ii;
      oedegrees = orinfo->edegrees;

      if (me == other) {
        /* Find which domains 'i' is connected and 'ii' is not and update their gain */
        for (k=0; k<myrinfo->ndegrees; k++) {
          pid = myedegrees[k].pid;
          for (kk=0; kk<orinfo->ndegrees; kk++) {
            if (oedegrees[kk].pid == pid)
              break;
          }
          if (kk == orinfo->ndegrees)
            myedegrees[k].gv -= vsize[ii];
        }
      }
      else {
        /* Find the orinfo[me].ed and see if I'm the only connection */
        for (k=0; k<orinfo->ndegrees; k++) {
          if (oedegrees[k].pid == me)
            break;
        }

        if (oedegrees[k].ned == 1) { /* I'm the only connection of 'ii' in 'me' */
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (myedegrees[k].pid == other) {
              myedegrees[k].gv += vsize[ii];
              break;
            }
          }

          /* Increase the gains for all the common domains between 'i' and 'ii' */
          for (k=0; k<myrinfo->ndegrees; k++) {
            if ((pid = myedegrees[k].pid) == other)
              continue;
            for (kk=0; kk<orinfo->ndegrees; kk++) {
              if (oedegrees[kk].pid == pid) {
                myedegrees[k].gv += vsize[ii];
                break;
              }
            }
          }

        }
        else {
          /* Find which domains 'i' is connected and 'ii' is not and update their gain */
          for (k=0; k<myrinfo->ndegrees; k++) {
            if ((pid = myedegrees[k].pid) == other)
              continue;
            for (kk=0; kk<orinfo->ndegrees; kk++) {
              if (oedegrees[kk].pid == pid)
                break;
            }
            if (kk == orinfo->ndegrees)
              myedegrees[k].gv -= vsize[ii];
          }
        }
      }
    }

    myrinfo = rinfo+i;
    myedegrees = myrinfo->edegrees;

    for (k=0; k<myrinfo->ndegrees; k++) {
      pid = myedegrees[k].pid;
      for (kk=0; kk<tmprinfo.ndegrees; kk++) {
        if (tmpdegrees[kk].pid == pid) {
          if (tmpdegrees[kk].gv != myedegrees[k].gv)
            printf("[%d %d %d %d]\n", i, pid, myedegrees[k].gv, tmpdegrees[kk].gv);
          break;
        }
      }
    }

  }

  free(tmpdegrees);

}


/*************************************************************************
* This function computes the subdomain graph
**************************************************************************/
void ComputeVolSubDomainGraph(GraphType *graph, int nparts, idxtype *pmat, idxtype *ndoms)
{
  int i, j, k, me, nvtxs, ndegrees;
  idxtype *xadj, *adjncy, *adjwgt, *where;
  VRInfoType *rinfo;
  VEDegreeType *edegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->vrinfo;

  idxset(nparts*nparts, 0, pmat);

  for (i=0; i<nvtxs; i++) {
    if (rinfo[i].ed > 0) {
      me = where[i];
      ndegrees = rinfo[i].ndegrees;
      edegrees = rinfo[i].edegrees;

      k = me*nparts;
      for (j=0; j<ndegrees; j++)
        pmat[k+edegrees[j].pid] += edegrees[j].ed;
    }
  }

  for (i=0; i<nparts; i++) {
    ndoms[i] = 0;
    for (j=0; j<nparts; j++) {
      if (pmat[i*nparts+j] > 0)
        ndoms[i]++;
    }
  }
}



/*************************************************************************
* This function computes the subdomain graph
**************************************************************************/
void EliminateVolSubDomainEdges(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts)
{
  int i, ii, j, k, me, other, nvtxs, total, max, avg, totalout, nind, ncand, ncand2, target, target2, nadd;
  int min, move, cpwgt, tvwgt;
  idxtype *xadj, *adjncy, *vwgt, *adjwgt, *pwgts, *where, *maxpwgt, *pmat, *ndoms, *mypmat, *otherpmat, *ind;
  KeyValueType *cand, *cand2;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = idxset(nparts, 0, graph->pwgts);

  maxpwgt = idxwspacemalloc(ctrl, nparts);
  ndoms = idxwspacemalloc(ctrl, nparts);
  otherpmat = idxwspacemalloc(ctrl, nparts);
  ind = idxwspacemalloc(ctrl, nvtxs);
  pmat = idxset(nparts*nparts, 0, ctrl->wspace.pmat);

  cand = (KeyValueType *)GKmalloc(nparts*sizeof(KeyValueType), "EliminateSubDomainEdges: cand");
  cand2 = (KeyValueType *)GKmalloc(nparts*sizeof(KeyValueType), "EliminateSubDomainEdges: cand");

  /* Compute the pmat matrix */
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    pwgts[me] += vwgt[i];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (where[k] != me)
        pmat[me*nparts+where[k]] += adjwgt[j];
    }
  }

  /* Compute the maximum allowed weight for each domain */
  tvwgt = idxsum(nparts, pwgts);
  for (i=0; i<nparts; i++)
    maxpwgt[i] = (int) 1.25*tpwgts[i]*tvwgt;

  /* Determine the domain connectivity */
  for (i=0; i<nparts; i++) {
    for (k=0, j=0; j<nparts; j++) {
      if (pmat[i*nparts+j] > 0)
        k++;
    }
    ndoms[i] = k;
  }

  /* Get into the loop eliminating subdomain connections */
  for (;;) {
    total = idxsum(nparts, ndoms);
    avg = total/nparts;
    max = ndoms[idxamax(nparts, ndoms)];

    /* printf("Adjacent Subdomain Stats: Total: %3d, Max: %3d, Avg: %3d\n", total, max, avg); */

    if (max < 1.5*avg)
      break;

    me = idxamax(nparts, ndoms);
    mypmat = pmat + me*nparts;
    totalout = idxsum(nparts, mypmat);

    /*printf("Me: %d, TotalOut: %d,\n", me, totalout);*/

    /* Sort the connections according to their cut */
    for (ncand2=0, i=0; i<nparts; i++) {
      if (mypmat[i] > 0) {
        cand2[ncand2].key = mypmat[i];
        cand2[ncand2++].val = i;
      }
    }
    ikeysort(ncand2, cand2);

    move = 0;
    for (min=0; min<ncand2; min++) {
      if (cand2[min].key > totalout/(2*ndoms[me]))
        break;

      other = cand2[min].val;

      /*printf("\tMinOut: %d to %d\n", mypmat[other], other);*/

      idxset(nparts, 0, otherpmat);

      /* Go and find the vertices in 'other' that are connected in 'me' */
      for (nind=0, i=0; i<nvtxs; i++) {
        if (where[i] == other) {
          for (j=xadj[i]; j<xadj[i+1]; j++) {
            if (where[adjncy[j]] == me) {
              ind[nind++] = i;
              break;
            }
          }
        }
      }

      /* Go and construct the otherpmat to see where these nind vertices are connected to */
      for (cpwgt=0, ii=0; ii<nind; ii++) {
        i = ind[ii];
        cpwgt += vwgt[i];

        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = adjncy[j];
          if (where[k] != other)
            otherpmat[where[k]] += adjwgt[j];
        }
      }

      for (ncand=0, i=0; i<nparts; i++) {
        if (otherpmat[i] > 0) {
          cand[ncand].key = -otherpmat[i];
          cand[ncand++].val = i;
        }
      }
      ikeysort(ncand, cand);

      /*
       * Go through and the select the first domain that is common with 'me', and
       * does not increase the ndoms[target] higher than my ndoms, subject to the
       * maxpwgt constraint. Traversal is done from the mostly connected to the least.
       */
      target = target2 = -1;
      for (i=0; i<ncand; i++) {
        k = cand[i].val;

        if (mypmat[k] > 0) {
          if (pwgts[k] + cpwgt > maxpwgt[k])  /* Check if balance will go off */
            continue;

          for (j=0; j<nparts; j++) {
            if (otherpmat[j] > 0 && ndoms[j] >= ndoms[me]-1 && pmat[nparts*j+k] == 0)
              break;
          }
          if (j == nparts) { /* No bad second level effects */
            for (nadd=0, j=0; j<nparts; j++) {
              if (otherpmat[j] > 0 && pmat[nparts*k+j] == 0)
                nadd++;
            }

            /*printf("\t\tto=%d, nadd=%d, %d\n", k, nadd, ndoms[k]);*/
            if (target2 == -1 && ndoms[k]+nadd < ndoms[me]) {
              target2 = k;
            }
            if (nadd == 0) {
              target = k;
              break;
            }
          }
        }
      }
      if (target == -1 && target2 != -1)
        target = target2;

      if (target == -1) {
        /* printf("\t\tCould not make the move\n");*/
        continue;
      }

      /*printf("\t\tMoving to %d\n", target);*/

      /* Update the partition weights */
      INC_DEC(pwgts[target], pwgts[other], cpwgt);

      /* Set all nind vertices to belong to 'target' */
      for (ii=0; ii<nind; ii++) {
        i = ind[ii];
        where[i] = target;

        /* First remove any contribution that this vertex may have made */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = adjncy[j];
          if (where[k] != other) {
            if (pmat[nparts*other + where[k]] == 0)
              printf("Something wrong\n");
            pmat[nparts*other + where[k]] -= adjwgt[j];
            if (pmat[nparts*other + where[k]] == 0)
              ndoms[other]--;

            if (pmat[nparts*where[k] + other] == 0)
              printf("Something wrong\n");
            pmat[nparts*where[k] + other] -= adjwgt[j];
            if (pmat[nparts*where[k] + other] == 0)
              ndoms[where[k]]--;
          }
        }

        /* Next add the new contributions as a result of the move */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = adjncy[j];
          if (where[k] != target) {
            if (pmat[nparts*target + where[k]] == 0)
              ndoms[target]++;
            pmat[nparts*target + where[k]] += adjwgt[j];

            if (pmat[nparts*where[k] + target] == 0)
              ndoms[where[k]]++;
            pmat[nparts*where[k] + target] += adjwgt[j];
          }
        }
      }

      move = 1;
      break;
    }

    if (move == 0)
      break;
  }

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);

  GKfree((void **) &cand, (void **) &cand2, LTERM);
}



/*************************************************************************
* This function finds all the connected components induced by the
* partitioning vector in wgraph->where and tries to push them around to
* remove some of them
**************************************************************************/
void EliminateVolComponents(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor)
{
  int i, ii, j, jj, k, me, nvtxs, tvwgt, first, last, nleft, ncmps, cwgt, ncand, other, target, deltawgt;
  idxtype *xadj, *adjncy, *vwgt, *adjwgt, *where, *pwgts, *maxpwgt;
  idxtype *cpvec, *touched, *perm, *todo, *cind, *cptr, *npcmps;
  KeyValueType *cand;
  int recompute=0;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = idxset(nparts, 0, graph->pwgts);

  touched = idxset(nvtxs, 0, idxwspacemalloc(ctrl, nvtxs));
  cptr = idxwspacemalloc(ctrl, nvtxs);
  cind = idxwspacemalloc(ctrl, nvtxs);
  perm = idxwspacemalloc(ctrl, nvtxs);
  todo = idxwspacemalloc(ctrl, nvtxs);
  maxpwgt = idxwspacemalloc(ctrl, nparts);
  cpvec = idxwspacemalloc(ctrl, nparts);
  npcmps = idxset(nparts, 0, idxwspacemalloc(ctrl, nparts));

  for (i=0; i<nvtxs; i++)
    perm[i] = todo[i] = i;

  /* Find the connected componends induced by the partition */
  ncmps = -1;
  first = last = 0;
  nleft = nvtxs;
  while (nleft > 0) {
    if (first == last) { /* Find another starting vertex */
      cptr[++ncmps] = first;
      ASSERT(touched[todo[0]] == 0);
      i = todo[0];
      cind[last++] = i;
      touched[i] = 1;
      me = where[i];
      npcmps[me]++;
    }

    i = cind[first++];
    k = perm[i];
    j = todo[k] = todo[--nleft];
    perm[j] = k;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (where[k] == me && !touched[k]) {
        cind[last++] = k;
        touched[k] = 1;
      }
    }
  }
  cptr[++ncmps] = first;

  /* printf("I found %d components, for this %d-way partition\n", ncmps, nparts); */

  if (ncmps > nparts) { /* There are more components than processors */
    cand = (KeyValueType *)GKmalloc(nparts*sizeof(KeyValueType), "EliminateSubDomainEdges: cand");

    /* First determine the partition sizes and max allowed load imbalance */
    for (i=0; i<nvtxs; i++)
      pwgts[where[i]] += vwgt[i];
    tvwgt = idxsum(nparts, pwgts);
    for (i=0; i<nparts; i++)
      maxpwgt[i] = (int) ubfactor*tpwgts[i]*tvwgt;

    deltawgt = tvwgt/(100*nparts);
    deltawgt = 5;

    for (i=0; i<ncmps; i++) {
      me = where[cind[cptr[i]]];  /* Get the domain of this component */
      if (npcmps[me] == 1)
        continue;  /* Skip it because it is contigous */

      /*printf("Trying to move %d from %d\n", i, me); */

      /* Determine the connectivity */
      idxset(nparts, 0, cpvec);
      for (cwgt=0, j=cptr[i]; j<cptr[i+1]; j++) {
        ii = cind[j];
        cwgt += vwgt[ii];
        for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
          other = where[adjncy[jj]];
          if (me != other)
            cpvec[other] += adjwgt[jj];
        }
      }

      /*printf("\tCmp weight: %d\n", cwgt);*/

      if (cwgt > .30*pwgts[me])
        continue;  /* Skip the component if it is over 30% of the weight */

      for (ncand=0, j=0; j<nparts; j++) {
        if (cpvec[j] > 0) {
          cand[ncand].key = -cpvec[j];
          cand[ncand++].val = j;
        }
      }
      if (ncand == 0)
        continue;

      ikeysort(ncand, cand);

      target = -1;
      for (j=0; j<ncand; j++) {
        k = cand[j].val;
        if (cwgt < deltawgt || pwgts[k] + cwgt < maxpwgt[k]) {
          target = k;
          break;
        }
      }

      /*printf("\tMoving it to %d [%d]\n", target, cpvec[target]);*/

      if (target != -1) {
        /* Assign all the vertices of 'me' to 'target' and update data structures */
        pwgts[me] -= cwgt;
        pwgts[target] += cwgt;
        npcmps[me]--;

        for (j=cptr[i]; j<cptr[i+1]; j++)
          where[cind[j]] = target;

        graph->mincut -= cpvec[target];
        recompute = 1;
      }
    }

    free(cand);
  }

  if (recompute) {
    int ttlv;
    idxtype *marker;

    marker = idxset(nparts, -1, cpvec);
    for (ttlv=0, i=0; i<nvtxs; i++) {
      marker[where[i]] = i;
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (marker[where[adjncy[j]]] != i) {
          ttlv += graph->vsize[i];
          marker[where[adjncy[j]]] = i;
        }
      }
    }
    graph->minvol = ttlv;
  }

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}

