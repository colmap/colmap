/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * timing.c
 *
 * This file contains routines that deal with timing Metis
 *
 * Started 7/24/97
 * George
 *
 * $Id: timing.c,v 1.1 1998/11/27 17:59:32 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function clears the timers
**************************************************************************/
void InitTimers(CtrlType *ctrl)
{
  cleartimer(ctrl->TotalTmr);
  cleartimer(ctrl->InitPartTmr);
  cleartimer(ctrl->MatchTmr);
  cleartimer(ctrl->ContractTmr);
  cleartimer(ctrl->CoarsenTmr);
  cleartimer(ctrl->UncoarsenTmr);
  cleartimer(ctrl->RefTmr);
  cleartimer(ctrl->ProjectTmr);
  cleartimer(ctrl->SplitTmr);
  cleartimer(ctrl->SepTmr);
  cleartimer(ctrl->AuxTmr1);
  cleartimer(ctrl->AuxTmr2);
  cleartimer(ctrl->AuxTmr3);
  cleartimer(ctrl->AuxTmr4);
  cleartimer(ctrl->AuxTmr5);
  cleartimer(ctrl->AuxTmr6);
}



/*************************************************************************
* This function prints the various timers
**************************************************************************/
void PrintTimers(CtrlType *ctrl)
{
  printf("\nTiming Information -------------------------------------------------");
  printf("\n Multilevel: \t\t %7.3f", gettimer(ctrl->TotalTmr));
  printf("\n     Coarsening: \t\t %7.3f", gettimer(ctrl->CoarsenTmr));
  printf("\n            Matching: \t\t\t %7.3f", gettimer(ctrl->MatchTmr));
  printf("\n            Contract: \t\t\t %7.3f", gettimer(ctrl->ContractTmr));
  printf("\n     Initial Partition: \t %7.3f", gettimer(ctrl->InitPartTmr));
  printf("\n   Construct Separator: \t %7.3f", gettimer(ctrl->SepTmr));
  printf("\n     Uncoarsening: \t\t %7.3f", gettimer(ctrl->UncoarsenTmr));
  printf("\n          Refinement: \t\t\t %7.3f", gettimer(ctrl->RefTmr));
  printf("\n          Projection: \t\t\t %7.3f", gettimer(ctrl->ProjectTmr));
  printf("\n     Splitting: \t\t %7.3f", gettimer(ctrl->SplitTmr));
  printf("\n          AUX1: \t\t %7.3f", gettimer(ctrl->AuxTmr1));
  printf("\n          AUX2: \t\t %7.3f", gettimer(ctrl->AuxTmr2));
  printf("\n          AUX3: \t\t %7.3f", gettimer(ctrl->AuxTmr3));
  printf("\n********************************************************************\n");
}


/*************************************************************************
* This function returns the seconds
**************************************************************************/
double seconds(void)
{
  return((double) clock()/CLOCKS_PER_SEC);
}


