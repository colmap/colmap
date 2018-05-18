/*
 * coarsen.c
 *
 * This file contains the driving routines for the coarsening process
 *
 * Started 7/23/97
 * George
 *
 * $Id: coarsen.c,v 1.1 1998/11/27 17:59:12 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function takes a graph and creates a sequence of coarser graphs
**************************************************************************/
GraphType *Coarsen2Way(CtrlType *ctrl, GraphType *graph)
{
  int clevel;
  GraphType *cgraph;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->CoarsenTmr));

  cgraph = graph;

  /* The following is ahack to allow the multiple bisections to go through with correct
     coarsening */
  if (ctrl->CType > 20) {
    clevel = 1;
    ctrl->CType -= 20;
  }
  else
    clevel = 0;

  do {
    IFSET(ctrl->dbglvl, DBG_COARSEN, printf("%6d %7d [%d] [%d %d]\n",
          cgraph->nvtxs, cgraph->nedges, ctrl->CoarsenTo, ctrl->maxvwgt,
          (cgraph->vwgt ? idxsum(cgraph->nvtxs, cgraph->vwgt) : cgraph->nvtxs)));

    if (cgraph->adjwgt) {
      switch (ctrl->CType) {
        case MATCH_RM:
          Match_RM(ctrl, cgraph);
          break;
        case MATCH_HEM:
          if (clevel < 1)
            Match_RM(ctrl, cgraph);
          else
            Match_HEM(ctrl, cgraph);
          break;
        case MATCH_HEMN:
          if (clevel < 1)
            Match_RM(ctrl, cgraph);
          else
            Match_HEMN(ctrl, cgraph);
          break;
        case MATCH_SHEMN:
          //if (clevel < 1)
          //  Match_RM(ctrl, cgraph);
          //else
            Match_SHEMN(ctrl, cgraph);
          break;
        case MATCH_SHEM:
          if (clevel < 1)
            Match_RM(ctrl, cgraph);
          else
            Match_SHEM(ctrl, cgraph);
          break;
        case MATCH_SHEMKWAY:
          Match_SHEM(ctrl, cgraph);
          break;
        default:
          errexit("Unknown CType: %d\n", ctrl->CType);
      }
    }
    else {
      Match_RM_NVW(ctrl, cgraph);
    }

    cgraph = cgraph->coarser;
    clevel++;
    //printf("Coarsening to level %d, number of vertices is %d...\n", clevel, cgraph->nvtxs);
  } while (cgraph->nvtxs > ctrl->CoarsenTo && cgraph->nvtxs < COARSEN_FRACTION2*cgraph->finer->nvtxs && cgraph->nedges > cgraph->nvtxs/2);


  IFSET(ctrl->dbglvl, DBG_COARSEN, printf("%6d %7d [%d] [%d %d]\n",
        cgraph->nvtxs, cgraph->nedges, ctrl->CoarsenTo, ctrl->maxvwgt,
        (cgraph->vwgt ? idxsum(cgraph->nvtxs, cgraph->vwgt) : cgraph->nvtxs)));

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->CoarsenTmr));

  return cgraph;
}

