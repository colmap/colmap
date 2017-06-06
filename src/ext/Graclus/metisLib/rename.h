/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * $Id: rename.h,v 1.1 1998/11/27 17:59:29 karypis Exp $
 *
 */

/* balance.c */
#define Balance2Way			__Balance2Way
#define Bnd2WayBalance			__Bnd2WayBalance
#define General2WayBalance		__General2WayBalance


/* bucketsort.c */
#define BucketSortKeysInc		__BucketSortKeysInc


/* ccgraph.c */
#define CreateCoarseGraph		__CreateCoarseGraph
#define CreateCoarseGraphNoMask		__CreateCoarseGraphNoMask
#define CreateCoarseGraph_NVW 		__CreateCoarseGraph_NVW
#define SetUpCoarseGraph		__SetUpCoarseGraph
#define ReAdjustMemory			__ReAdjustMemory


/* coarsen.c */
#define Coarsen2Way			__Coarsen2Way


/* compress.c */
#define CompressGraph			__CompressGraph
#define PruneGraph			__PruneGraph


/* debug.c */
#define ComputeCut			__ComputeCut
#define CheckBnd			__CheckBnd
#define CheckBnd2			__CheckBnd2
#define CheckNodeBnd			__CheckNodeBnd
#define CheckRInfo			__CheckRInfo
#define CheckNodePartitionParams	__CheckNodePartitionParams
#define IsSeparable			__IsSeparable


/* estmem.c */
#define EstimateCFraction		__EstimateCFraction
#define ComputeCoarseGraphSize		__ComputeCoarseGraphSize


/* fm.c */
#define FM_2WayEdgeRefine		__FM_2WayEdgeRefine


/* fortran.c */
#define Change2CNumbering		__Change2CNumbering
#define Change2FNumbering		__Change2FNumbering
#define Change2FNumbering2		__Change2FNumbering2
#define Change2FNumberingOrder		__Change2FNumberingOrder
#define ChangeMesh2CNumbering		__ChangeMesh2CNumbering
#define ChangeMesh2FNumbering		__ChangeMesh2FNumbering
#define ChangeMesh2FNumbering2		__ChangeMesh2FNumbering2


/* graph.c */
#define SetUpGraph			__SetUpGraph
#define SetUpGraphKway 			__SetUpGraphKway
#define SetUpGraph2			__SetUpGraph2
#define VolSetUpGraph			__VolSetUpGraph
#define RandomizeGraph			__RandomizeGraph
#define IsConnectedSubdomain		__IsConnectedSubdomain
#define IsConnected			__IsConnected
#define IsConnected2			__IsConnected2
#define FindComponents			__FindComponents


/* initpart.c */
#define Init2WayPartition		__Init2WayPartition
#define InitSeparator			__InitSeparator
#define GrowBisection			__GrowBisection
#define GrowBisectionNode		__GrowBisectionNode
#define RandomBisection			__RandomBisection


/* kmetis.c */
#define MlevelKWayPartitioning		__MlevelKWayPartitioning


/* kvmetis.c */
#define MlevelVolKWayPartitioning	__MlevelVolKWayPartitioning


/* kwayfm.c */
#define Random_KWayEdgeRefine		__Random_KWayEdgeRefine
#define Greedy_KWayEdgeRefine		__Greedy_KWayEdgeRefine
#define Greedy_KWayEdgeBalance		__Greedy_KWayEdgeBalance


/* kwayrefine.c */
#define RefineKWay			__RefineKWay
#define AllocateKWayPartitionMemory	__AllocateKWayPartitionMemory
#define ComputeKWayPartitionParams	__ComputeKWayPartitionParams
#define ProjectKWayPartition		__ProjectKWayPartition
#define IsBalanced			__IsBalanced
#define ComputeKWayBoundary		__ComputeKWayBoundary
#define ComputeKWayBalanceBoundary	__ComputeKWayBalanceBoundary


/* kwayvolfm.c */
#define Random_KWayVolRefine		__Random_KWayVolRefine
#define Random_KWayVolRefineMConn	__Random_KWayVolRefineMConn
#define Greedy_KWayVolBalance		__Greedy_KWayVolBalance
#define Greedy_KWayVolBalanceMConn	__Greedy_KWayVolBalanceMConn
#define KWayVolUpdate			__KWayVolUpdate
#define ComputeKWayVolume		__ComputeKWayVolume
#define ComputeVolume			__ComputeVolume
#define CheckVolKWayPartitionParams	__CheckVolKWayPartitionParams
#define ComputeVolSubDomainGraph	__ComputeVolSubDomainGraph
#define EliminateVolSubDomainEdges	__EliminateVolSubDomainEdges


/* kwayvolrefine.c */
#define RefineVolKWay			__RefineVolKWay
#define AllocateVolKWayPartitionMemory	__AllocateVolKWayPartitionMemory
#define ComputeVolKWayPartitionParams	__ComputeVolKWayPartitionParams
#define ComputeKWayVolGains		__ComputeKWayVolGains
#define ProjectVolKWayPartition		__ProjectVolKWayPartition
#define ComputeVolKWayBoundary		__ComputeVolKWayBoundary
#define ComputeVolKWayBalanceBoundary	__ComputeVolKWayBalanceBoundary


/* match.c */
#define Match_RM			__Match_RM
#define Match_RM_NVW			__Match_RM_NVW
#define Match_HEM			__Match_HEM
#define Match_SHEM			__Match_SHEM


/* mbalance.c */
#define MocBalance2Way			__MocBalance2Way
#define MocGeneral2WayBalance		__MocGeneral2WayBalance


/* mbalance2.c */
#define MocBalance2Way2			__MocBalance2Way2
#define MocGeneral2WayBalance2		__MocGeneral2WayBalance2
#define SelectQueue3			__SelectQueue3


/* mcoarsen.c */
#define MCCoarsen2Way			__MCCoarsen2Way


/* memory.c */
#define AllocateWorkSpace		__AllocateWorkSpace
#define FreeWorkSpace			__FreeWorkSpace
#define WspaceAvail			__WspaceAvail
#define idxwspacemalloc			__idxwspacemalloc
#define idxwspacefree			__idxwspacefree
#define fwspacemalloc			__fwspacemalloc
#define CreateGraph			__CreateGraph
#define InitGraph			__InitGraph
#define FreeGraph			__FreeGraph


/* mesh.c */
#define TRIDUALMETIS			__TRIDUALMETIS
#define TETDUALMETIS			__TETDUALMETIS
#define HEXDUALMETIS			__HEXDUALMETIS
#define TRINODALMETIS			__TRINODALMETIS
#define TETNODALMETIS			__TETNODALMETIS
#define HEXNODALMETIS			__HEXNODALMETIS


/* mfm.c */
#define MocFM_2WayEdgeRefine		__MocFM_2WayEdgeRefine
#define SelectQueue			__SelectQueue
#define BetterBalance			__BetterBalance
#define Compute2WayHLoadImbalance	__Compute2WayHLoadImbalance
#define Compute2WayHLoadImbalanceVec	__Compute2WayHLoadImbalanceVec


/* mfm2.c */
#define MocFM_2WayEdgeRefine2		__MocFM_2WayEdgeRefine2
#define SelectQueue2			__SelectQueue2
#define IsBetter2wayBalance		__IsBetter2wayBalance


/* mincover.c */
#define MinCover			__MinCover
#define MinCover_Augment		__MinCover_Augment
#define MinCover_Decompose		__MinCover_Decompose
#define MinCover_ColDFS			__MinCover_ColDFS
#define MinCover_RowDFS			__MinCover_RowDFS


/* minitpart.c */
#define MocInit2WayPartition		__MocInit2WayPartition
#define MocGrowBisection		__MocGrowBisection
#define MocRandomBisection		__MocRandomBisection
#define MocInit2WayBalance		__MocInit2WayBalance
#define SelectQueueoneWay		__SelectQueueoneWay


/* minitpart2.c */
#define MocInit2WayPartition2		__MocInit2WayPartition2
#define MocGrowBisection2		__MocGrowBisection2
#define MocGrowBisectionNew2		__MocGrowBisectionNew2
#define MocInit2WayBalance2		__MocInit2WayBalance2
#define SelectQueueOneWay2		__SelectQueueOneWay2


/* mkmetis.c */
#define MCMlevelKWayPartitioning	__MCMlevelKWayPartitioning


/* mkwayfmh.c */
#define MCRandom_KWayEdgeRefineHorizontal	__MCRandom_KWayEdgeRefineHorizontal
#define MCGreedy_KWayEdgeBalanceHorizontal	__MCGreedy_KWayEdgeBalanceHorizontal
#define AreAllHVwgtsBelow			__AreAllHVwgtsBelow
#define AreAllHVwgtsAbove			__AreAllHVwgtsAbove
#define ComputeHKWayLoadImbalance		__ComputeHKWayLoadImbalance
#define MocIsHBalanced				__MocIsHBalanced
#define IsHBalanceBetterFT			__IsHBalanceBetterFT
#define IsHBalanceBetterTT			__IsHBalanceBetterTT


/* mkwayrefine.c */
#define MocRefineKWayHorizontal		__MocRefineKWayHorizontal
#define MocAllocateKWayPartitionMemory	__MocAllocateKWayPartitionMemory
#define MocComputeKWayPartitionParams	__MocComputeKWayPartitionParams
#define MocProjectKWayPartition		__MocProjectKWayPartition
#define MocComputeKWayBalanceBoundary	__MocComputeKWayBalanceBoundary


/* mmatch.c */
#define MCMatch_RM			__MCMatch_RM
#define MCMatch_HEM			__MCMatch_HEM
#define MCMatch_SHEM			__MCMatch_SHEM
#define MCMatch_SHEBM			__MCMatch_SHEBM
#define MCMatch_SBHEM			__MCMatch_SBHEM
#define BetterVBalance			__BetterVBalance
#define AreAllVwgtsBelowFast		__AreAllVwgtsBelowFast


/* mmd.c */
#define genmmd				__genmmd
#define mmdelm				__mmdelm
#define mmdint				__mmdint
#define mmdnum				__mmdnum
#define mmdupd				__mmdupd


/* mpmetis.c */
#define MCMlevelRecursiveBisection	__MCMlevelRecursiveBisection
#define MCHMlevelRecursiveBisection	__MCHMlevelRecursiveBisection
#define MCMlevelEdgeBisection		__MCMlevelEdgeBisection
#define MCHMlevelEdgeBisection		__MCHMlevelEdgeBisection


/* mrefine.c */
#define MocRefine2Way			__MocRefine2Way
#define MocAllocate2WayPartitionMemory	__MocAllocate2WayPartitionMemory
#define MocCompute2WayPartitionParams	__MocCompute2WayPartitionParams
#define MocProject2WayPartition		__MocProject2WayPartition


/* mrefine2.c */
#define MocRefine2Way2			__MocRefine2Way2


/* mutil.c */
#define AreAllVwgtsBelow		__AreAllVwgtsBelow
#define AreAnyVwgtsBelow		__AreAnyVwgtsBelow
#define AreAllVwgtsAbove		__AreAllVwgtsAbove
#define ComputeLoadImbalance		__ComputeLoadImbalance
#define AreAllBelow			__AreAllBelow


/* myqsort.c */
#define iidxsort			__iidxsort
#define iintsort			__iintsort
#define ikeysort			__ikeysort
#define ikeyvalsort			__ikeyvalsort


/* ometis.c */
#define MlevelNestedDissection		__MlevelNestedDissection
#define MlevelNestedDissectionCC	__MlevelNestedDissectionCC
#define MlevelNodeBisectionMultiple	__MlevelNodeBisectionMultiple
#define MlevelNodeBisection		__MlevelNodeBisection
#define SplitGraphOrder			__SplitGraphOrder
#define MMDOrder			__MMDOrder
#define SplitGraphOrderCC		__SplitGraphOrderCC


/* parmetis.c */
#define MlevelNestedDissectionP		__MlevelNestedDissectionP


/* pmetis.c */
#define MlevelRecursiveBisection	__MlevelRecursiveBisection
#define MlevelEdgeBisection		__MlevelEdgeBisection
#define SplitGraphPart			__SplitGraphPart
#define SetUpSplitGraph			__SetUpSplitGraph


/* pqueue.c */
#define PQueueInit			__PQueueInit
#define PQueueReset			__PQueueReset
#define PQueueFree			__PQueueFree
#define PQueueInsert			__PQueueInsert
#define PQueueDelete			__PQueueDelete
#define PQueueUpdate			__PQueueUpdate
#define PQueueUpdateUp			__PQueueUpdateUp
#define PQueueGetMax			__PQueueGetMax
#define PQueueSeeMax			__PQueueSeeMax
#define CheckHeap			__CheckHeap


/* refine.c */
#define Refine2Way			__Refine2Way
#define Allocate2WayPartitionMemory	__Allocate2WayPartitionMemory
#define Compute2WayPartitionParams	__Compute2WayPartitionParams
#define Project2WayPartition		__Project2WayPartition


/* separator.c */
#define ConstructSeparator		__ConstructSeparator
#define ConstructMinCoverSeparator0	__ConstructMinCoverSeparator0
#define ConstructMinCoverSeparator	__ConstructMinCoverSeparator


/* sfm.c */
#define FM_2WayNodeRefine		__FM_2WayNodeRefine
#define FM_2WayNodeRefineEqWgt		__FM_2WayNodeRefineEqWgt
#define FM_2WayNodeRefine_OneSided	__FM_2WayNodeRefine_OneSided
#define FM_2WayNodeBalance		__FM_2WayNodeBalance
#define ComputeMaxNodeGain		__ComputeMaxNodeGain


/* srefine.c */
#define Refine2WayNode			__Refine2WayNode
#define Allocate2WayNodePartitionMemory	__Allocate2WayNodePartitionMemory
#define Compute2WayNodePartitionParams	__Compute2WayNodePartitionParams
#define Project2WayNodePartition	__Project2WayNodePartition


/* stat.c */
#define ComputePartitionInfo		__ComputePartitionInfo
#define ComputePartitionBalance		__ComputePartitionBalance
#define ComputeElementBalance		__ComputeElementBalance


/* subdomains.c */
#define Random_KWayEdgeRefineMConn	__Random_KWayEdgeRefineMConn
#define Greedy_KWayEdgeBalanceMConn	__Greedy_KWayEdgeBalanceMConn
#define PrintSubDomainGraph		__PrintSubDomainGraph
#define ComputeSubDomainGraph		__ComputeSubDomainGraph
#define EliminateSubDomainEdges		__EliminateSubDomainEdges
#define MoveGroupMConn			__MoveGroupMConn
#define EliminateComponents		__EliminateComponents
#define MoveGroup			__MoveGroup


/* timing.c */
#define InitTimers			__InitTimers
#define PrintTimers			__PrintTimers
#define seconds				__seconds


/* util.c */
#define errexit				__errexit
#define GKfree				__GKfree
#ifndef DMALLOC
#define imalloc				__imalloc
#define idxmalloc			__idxmalloc
#define fmalloc				__fmalloc
#define ismalloc			__ismalloc
#define idxsmalloc			__idxsmalloc
#define GKmalloc			__GKmalloc
#endif
#define iset				__iset
#define idxset				__idxset
#define sset				__sset
#define iamax				__iamax
#define idxamax				__idxamax
#define idxamax_strd			__idxamax_strd
#define samax				__samax
#define samax2				__samax2
#define idxamin				__idxamin
#define samin				__samin
#define idxsum				__idxsum
#define idxsum_strd			__idxsum_strd
#define idxadd				__idxadd
#define charsum				__charsum
#define isum				__isum
#define ssum				__ssum
#define ssum_strd			__ssum_strd
#define sscale				__sscale
#define snorm2				__snorm2
#define sdot				__sdot
#define saxpy				__saxpy
#define RandomPermute			__RandomPermute
#define ispow2				__ispow2
#define InitRandom			__InitRandom
#define log2_metis			__log2_metis





