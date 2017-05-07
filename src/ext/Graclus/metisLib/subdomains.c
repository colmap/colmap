/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * subdomains.c
 *
 * This file contains functions that deal with prunning the number of
 * adjacent subdomains in KMETIS
 *
 * Started 7/15/98
 * George
 *
 * $Id: subdomains.c,v 1.1 1998/11/27 17:59:32 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Random_KWayEdgeRefineMConn(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor, int npasses, int ffactor)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, nmoves, nbnd, tvwgt, myndegrees;
  int from, me, to, oldcut, vwgt, gain;
  int maxndoms, nadd;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *itpwgts;
  idxtype *phtable, *pmat, *pmatptr, *ndoms;
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

  pmat = ctrl->wspace.pmat;
  phtable = idxwspacemalloc(ctrl, nparts);
  ndoms = idxwspacemalloc(ctrl, nparts);

  ComputeSubDomainGraph(graph, nparts, pmat, ndoms);

  /* Setup the weight intervals of the various subdomains */
  minwgt =  idxwspacemalloc(ctrl, nparts);
  maxwgt = idxwspacemalloc(ctrl, nparts);
  itpwgts = idxwspacemalloc(ctrl, nparts);
  tvwgt = idxsum(nparts, pwgts);
  ASSERT(tvwgt == idxsum(nvtxs, graph->vwgt));

  for (i=0; i<nparts; i++) {
    itpwgts[i] = tpwgts[i]*tvwgt;
    maxwgt[i] = tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = tpwgts[i]*tvwgt*(1.0/ubfactor);
  }

  perm = idxwspacemalloc(ctrl, nvtxs);

  IFSET(ctrl->dbglvl, DBG_REFINE,
     printf("Partitions: [%6d %6d]-[%6d %6d], Balance: %5.3f, Nv-Nb[%6d %6d]. Cut: %6d\n",
             pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)], minwgt[0], maxwgt[0],
             1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nvtxs, graph->nbnd,
             graph->mincut));

  for (pass=0; pass<npasses; pass++) {
    ASSERT(ComputeCut(graph, where) == graph->mincut);

    maxndoms = ndoms[idxamax(nparts, ndoms)];

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

        /* Find the first valid move */
        j = myrinfo->id;
        for (k=0; k<myndegrees; k++) {
          to = myedegrees[k].pid;
          if (!phtable[to])
            continue;
          gain = myedegrees[k].ed-j; /* j = myrinfo->id. Allow good nodes to move */
          if (pwgts[to]+vwgt <= maxwgt[to]+ffactor*gain && gain >= 0)
            break;
        }
        if (k == myndegrees)
          continue;  /* break out if you did not find a candidate */

        for (j=k+1; j<myndegrees; j++) {
          to = myedegrees[j].pid;
          if (!phtable[to])
            continue;
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
          if (/*(iii&7) == 0  ||*/ phtable[myedegrees[k].pid] == 2 || pwgts[from] >= maxwgt[from] || itpwgts[from]*(pwgts[to]+vwgt) < itpwgts[to]*pwgts[from])
            j = 1;
        }
        if (j == 0)
          continue;

        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to'
        *======================================================================*/
        graph->mincut -= myedegrees[k].ed-myrinfo->id;

        IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

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

          ASSERT(myrinfo->ndegrees <= xadj[ii+1]-xadj[ii]);
          ASSERT(CheckRInfo(myrinfo));

        }
        nmoves++;
      }
    }

    graph->nbnd = nbnd;

    IFSET(ctrl->dbglvl, DBG_REFINE,
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %5d, Vol: %5d, %d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves,
               graph->mincut, ComputeVolume(graph, where), idxsum(nparts, ndoms)));

    if (graph->mincut == oldcut)
      break;
  }

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
}



/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Greedy_KWayEdgeBalanceMConn(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor, int npasses)
{
  int i, ii, iii, j, jj, k, l, pass, nvtxs, nbnd, tvwgt, myndegrees, oldgain, gain, nmoves;
  int from, me, to, oldcut, vwgt, maxndoms, nadd;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *pwgts, *perm, *bndptr, *bndind, *minwgt, *maxwgt, *moved, *itpwgts;
  idxtype *phtable, *pmat, *pmatptr, *ndoms;
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

  pmat = ctrl->wspace.pmat;
  phtable = idxwspacemalloc(ctrl, nparts);
  ndoms = idxwspacemalloc(ctrl, nparts);

  ComputeSubDomainGraph(graph, nparts, pmat, ndoms);


  /* Setup the weight intervals of the various subdomains */
  minwgt =  idxwspacemalloc(ctrl, nparts);
  maxwgt = idxwspacemalloc(ctrl, nparts);
  itpwgts = idxwspacemalloc(ctrl, nparts);
  tvwgt = idxsum(nparts, pwgts);
  ASSERT(tvwgt == idxsum(nvtxs, graph->vwgt));

  for (i=0; i<nparts; i++) {
    itpwgts[i] = tpwgts[i]*tvwgt;
    maxwgt[i] = tpwgts[i]*tvwgt*ubfactor;
    minwgt[i] = tpwgts[i]*tvwgt*(1.0/ubfactor);
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

    maxndoms = ndoms[idxamax(nparts, ndoms)];

    for (nmoves=0;;) {
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
        if (pwgts[to]+vwgt <= maxwgt[to] || itpwgts[from]*(pwgts[to]+vwgt) <= itpwgts[to]*pwgts[from])
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

      if (pwgts[from] < maxwgt[from] && pwgts[to] > minwgt[to] && myedegrees[k].ed-myrinfo->id < 0)
        continue;

      /*=====================================================================
      * If we got here, we can now move the vertex from 'from' to 'to'
      *======================================================================*/
      graph->mincut -= myedegrees[k].ed-myrinfo->id;

      IFSET(ctrl->dbglvl, DBG_MOVEINFO, printf("\t\tMoving %6d to %3d. Gain: %4d. Cut: %6d\n", i, to, myedegrees[k].ed-myrinfo->id, graph->mincut));

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
       printf("\t[%6d %6d], Balance: %5.3f, Nb: %6d. Nmoves: %5d, Cut: %6d, %d\n",
               pwgts[idxamin(nparts, pwgts)], pwgts[idxamax(nparts, pwgts)],
               1.0*nparts*pwgts[idxamax(nparts, pwgts)]/tvwgt, graph->nbnd, nmoves, graph->mincut,idxsum(nparts, ndoms)));
  }

  PQueueFree(ctrl, &queue);

  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nparts);
  idxwspacefree(ctrl, nvtxs);
  idxwspacefree(ctrl, nvtxs);

}




/*************************************************************************
* This function computes the subdomain graph
**************************************************************************/
void PrintSubDomainGraph(GraphType *graph, int nparts, idxtype *where)
{
  int i, j, k, me, nvtxs, total, max;
  idxtype *xadj, *adjncy, *adjwgt, *pmat;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  pmat = idxsmalloc(nparts*nparts, 0, "ComputeSubDomainGraph: pmat");

  for (i=0; i<nvtxs; i++) {
    me = where[i];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (where[k] != me)
        pmat[me*nparts+where[k]] += adjwgt[j];
    }
  }

  /* printf("Subdomain Info\n"); */
  total = max = 0;
  for (i=0; i<nparts; i++) {
    for (k=0, j=0; j<nparts; j++) {
      if (pmat[i*nparts+j] > 0)
        k++;
    }
    total += k;

    if (k > max)
      max = k;
/*
    printf("%2d -> %2d  ", i, k);
    for (j=0; j<nparts; j++) {
      if (pmat[i*nparts+j] > 0)
        printf("[%2d %4d] ", j, pmat[i*nparts+j]);
    }
    printf("\n");
*/
  }
  printf("Total adjacent subdomains: %d, Max: %d\n", total, max);

  free(pmat);
}



/*************************************************************************
* This function computes the subdomain graph
**************************************************************************/
void ComputeSubDomainGraph(GraphType *graph, int nparts, idxtype *pmat, idxtype *ndoms)
{
  int i, j, k, me, nvtxs, ndegrees;
  idxtype *xadj, *adjncy, *adjwgt, *where;
  RInfoType *rinfo;
  EDegreeType *edegrees;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->rinfo;

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
void EliminateSubDomainEdges(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts)
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
  pwgts = graph->pwgts;  /* We assume that this is properly initialized */

  maxpwgt = idxwspacemalloc(ctrl, nparts);
  ndoms = idxwspacemalloc(ctrl, nparts);
  otherpmat = idxwspacemalloc(ctrl, nparts);
  ind = idxwspacemalloc(ctrl, nvtxs);
  pmat = ctrl->wspace.pmat;

  cand = (KeyValueType *)GKmalloc(nparts*sizeof(KeyValueType), "EliminateSubDomainEdges: cand");
  cand2 = (KeyValueType *)GKmalloc(nparts*sizeof(KeyValueType), "EliminateSubDomainEdges: cand");

  /* Compute the pmat matrix and ndoms */
  ComputeSubDomainGraph(graph, nparts, pmat, ndoms);


  /* Compute the maximum allowed weight for each domain */
  tvwgt = idxsum(nparts, pwgts);
  for (i=0; i<nparts; i++)
    maxpwgt[i] = 1.25*tpwgts[i]*tvwgt;


  /* Get into the loop eliminating subdomain connections */
  for (;;) {
    total = idxsum(nparts, ndoms);
    avg = total/nparts;
    max = ndoms[idxamax(nparts, ndoms)];

    /* printf("Adjacent Subdomain Stats: Total: %3d, Max: %3d, Avg: %3d [%5d]\n", total, max, avg, idxsum(nparts*nparts, pmat)); */

    if (max < 1.4*avg)
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

        for (j=xadj[i]; j<xadj[i+1]; j++)
          otherpmat[where[adjncy[j]]] += adjwgt[j];
      }
      otherpmat[other] = 0;

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

      MoveGroupMConn(ctrl, graph, ndoms, pmat, nparts, target, nind, ind);

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
* This function moves a collection of vertices and updates their rinfo
**************************************************************************/
void MoveGroupMConn(CtrlType *ctrl, GraphType *graph, idxtype *ndoms, idxtype *pmat,
                    int nparts, int to, int nind, idxtype *ind)
{
  int i, ii, iii, j, jj, k, l, nvtxs, nbnd, myndegrees;
  int from, me;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *bndptr, *bndind;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  nbnd = graph->nbnd;

  for (iii=0; iii<nind; iii++) {
    i = ind[iii];
    from = where[i];

    myrinfo = graph->rinfo+i;
    if (myrinfo->edegrees == NULL) {
      myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += xadj[i+1]-xadj[i];
      myrinfo->ndegrees = 0;
    }
    myedegrees = myrinfo->edegrees;

    /* find the location of 'to' in myrinfo or create it if it is not there */
    for (k=0; k<myrinfo->ndegrees; k++) {
      if (myedegrees[k].pid == to)
        break;
    }
    if (k == myrinfo->ndegrees) {
      myedegrees[k].pid = to;
      myedegrees[k].ed = 0;
      myrinfo->ndegrees++;
    }

    graph->mincut -= myedegrees[k].ed-myrinfo->id;

    /* Update pmat to reflect the move of 'i' */
    pmat[from*nparts+to] += (myrinfo->id-myedegrees[k].ed);
    pmat[to*nparts+from] += (myrinfo->id-myedegrees[k].ed);
    if (pmat[from*nparts+to] == 0)
      ndoms[from]--;
    if (pmat[to*nparts+from] == 0)
      ndoms[to]--;

    /* Update where, weight, and ID/ED information of the vertex you moved */
    where[i] = to;
    myrinfo->ed += myrinfo->id-myedegrees[k].ed;
    SWAP(myrinfo->id, myedegrees[k].ed, j);
    if (myedegrees[k].ed == 0)
      myedegrees[k] = myedegrees[--myrinfo->ndegrees];
    else
      myedegrees[k].pid = from;

    if (myrinfo->ed-myrinfo->id < 0 && bndptr[i] != -1)
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

      /* Update pmat to reflect the move of 'i' for domains other than 'from' and 'to' */
      if (me != from && me != to) {
        pmat[me*nparts+from] -= adjwgt[j];
        pmat[from*nparts+me] -= adjwgt[j];
        if (pmat[me*nparts+from] == 0)
          ndoms[me]--;
        if (pmat[from*nparts+me] == 0)
          ndoms[from]--;

        if (pmat[me*nparts+to] == 0)
          ndoms[me]++;
        if (pmat[to*nparts+me] == 0)
          ndoms[to]++;

        pmat[me*nparts+to] += adjwgt[j];
        pmat[to*nparts+me] += adjwgt[j];
      }

      ASSERT(CheckRInfo(myrinfo));
    }

    ASSERT(CheckRInfo(graph->rinfo+i));
  }

  graph->nbnd = nbnd;

}




/*************************************************************************
* This function finds all the connected components induced by the
* partitioning vector in wgraph->where and tries to push them around to
* remove some of them
**************************************************************************/
void EliminateComponents(CtrlType *ctrl, GraphType *graph, int nparts, float *tpwgts, float ubfactor)
{
  int i, ii, j, jj, k, me, nvtxs, tvwgt, first, last, nleft, ncmps, cwgt, other, target, deltawgt;
  idxtype *xadj, *adjncy, *vwgt, *adjwgt, *where, *pwgts, *maxpwgt;
  idxtype *cpvec, *touched, *perm, *todo, *cind, *cptr, *npcmps;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  vwgt = graph->vwgt;
  adjwgt = graph->adjwgt;

  where = graph->where;
  pwgts = graph->pwgts;

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
    /* First determine the max allowed load imbalance */
    tvwgt = idxsum(nparts, pwgts);
    for (i=0; i<nparts; i++)
      maxpwgt[i] = ubfactor*tpwgts[i]*tvwgt;

    deltawgt = 5;

    for (i=0; i<ncmps; i++) {
      me = where[cind[cptr[i]]];  /* Get the domain of this component */
      if (npcmps[me] == 1)
        continue;  /* Skip it because it is contigous */

      /*printf("Trying to move %d from %d\n", i, me); */

      /* Determine the weight of the block to be moved and abort if too high */
      for (cwgt=0, j=cptr[i]; j<cptr[i+1]; j++)
        cwgt += vwgt[cind[j]];

      if (cwgt > .30*pwgts[me])
        continue;  /* Skip the component if it is over 30% of the weight */

      /* Determine the connectivity */
      idxset(nparts, 0, cpvec);
      for (j=cptr[i]; j<cptr[i+1]; j++) {
        ii = cind[j];
        for (jj=xadj[ii]; jj<xadj[ii+1]; jj++)
          cpvec[where[adjncy[jj]]] += adjwgt[jj];
      }
      cpvec[me] = 0;

      target = -1;
      for (j=0; j<nparts; j++) {
        if (cpvec[j] > 0 && (cwgt < deltawgt || pwgts[j] + cwgt < maxpwgt[j])) {
          if (target == -1 || cpvec[target] < cpvec[j])
            target = j;
        }
      }

      /* printf("\tMoving it to %d [%d]\n", target, cpvec[target]);*/

      if (target != -1) {
        /* Assign all the vertices of 'me' to 'target' and update data structures */
        INC_DEC(pwgts[target], pwgts[me], cwgt);
        npcmps[me]--;

        MoveGroup(ctrl, graph, nparts, target, i, cptr, cind);
      }
    }

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


/*************************************************************************
* This function moves a collection of vertices and updates their rinfo
**************************************************************************/
void MoveGroup(CtrlType *ctrl, GraphType *graph, int nparts, int to, int gid, idxtype *ptr, idxtype *ind)
{
  int i, ii, iii, j, jj, k, l, nvtxs, nbnd, myndegrees;
  int from, me;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where, *bndptr, *bndind;
  EDegreeType *myedegrees;
  RInfoType *myrinfo;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  bndptr = graph->bndptr;
  bndind = graph->bndind;

  nbnd = graph->nbnd;

  for (iii=ptr[gid]; iii<ptr[gid+1]; iii++) {
    i = ind[iii];
    from = where[i];

    myrinfo = graph->rinfo+i;
    if (myrinfo->edegrees == NULL) {
      myrinfo->edegrees = ctrl->wspace.edegrees+ctrl->wspace.cdegree;
      ctrl->wspace.cdegree += xadj[i+1]-xadj[i];
      myrinfo->ndegrees = 0;
    }
    myedegrees = myrinfo->edegrees;

    /* find the location of 'to' in myrinfo or create it if it is not there */
    for (k=0; k<myrinfo->ndegrees; k++) {
      if (myedegrees[k].pid == to)
        break;
    }
    if (k == myrinfo->ndegrees) {
      myedegrees[k].pid = to;
      myedegrees[k].ed = 0;
      myrinfo->ndegrees++;
    }

    graph->mincut -= myedegrees[k].ed-myrinfo->id;


    /* Update where, weight, and ID/ED information of the vertex you moved */
    where[i] = to;
    myrinfo->ed += myrinfo->id-myedegrees[k].ed;
    SWAP(myrinfo->id, myedegrees[k].ed, j);
    if (myedegrees[k].ed == 0)
      myedegrees[k] = myedegrees[--myrinfo->ndegrees];
    else
      myedegrees[k].pid = from;

    if (myrinfo->ed-myrinfo->id < 0 && bndptr[i] != -1)
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

      ASSERT(CheckRInfo(myrinfo));
    }

    ASSERT(CheckRInfo(graph->rinfo+i));
  }

  graph->nbnd = nbnd;

}

