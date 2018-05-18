/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id: proto.h,v 1.1 1998/11/27 17:59:28 karypis Exp $
 *
 */

/* balance.c */
void Balance2Way(CtrlType *, GraphType *, int *, float);
void Bnd2WayBalance(CtrlType *, GraphType *, int *);
void General2WayBalance(CtrlType *, GraphType *, int *);

/* bucketsort.c */
void BucketSortKeysInc(int, int, idxtype *, idxtype *, idxtype *);

/* ccgraph.c */
void CreateCoarseGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *);
void CreateCoarseGraphNoMask(CtrlType *, GraphType *, int, idxtype *, idxtype *);
void CreateCoarseGraph_NVW(CtrlType *, GraphType *, int, idxtype *, idxtype *);
GraphType *SetUpCoarseGraph(GraphType *, int, int);
void ReAdjustMemory(GraphType *, GraphType *, int);

/* coarsen.c */
GraphType *Coarsen2Way(CtrlType *, GraphType *);

/* compress.c */
void CompressGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *, idxtype *, idxtype *);
void PruneGraph(CtrlType *, GraphType *, int, idxtype *, idxtype *, idxtype *, float);

/* debug.c */
int ComputeCut(GraphType *, idxtype *);
float ComputeRAsso(GraphType *graph, idxtype *where, int npart);
float ComputeNCut(GraphType *, idxtype *, int);
int CheckBnd(GraphType *);
int CheckBnd2(GraphType *);
int CheckNodeBnd(GraphType *, int);
int CheckRInfo(RInfoType *);
int CheckNodePartitionParams(GraphType *);
int IsSeparable(GraphType *);

/* estmem.c */
void METIS_EstimateMemory(int *, idxtype *, idxtype *, int *, int *, int *);
void EstimateCFraction(int, idxtype *, idxtype *, float *, float *);
int ComputeCoarseGraphSize(int, idxtype *, idxtype *, int, idxtype *, idxtype *, idxtype *);

/* fm.c */
void FM_2WayEdgeRefine(CtrlType *, GraphType *, int *, int);

/* fortran.c */
void Change2CNumbering(int, idxtype *, idxtype *);
void Change2FNumbering(int, idxtype *, idxtype *, idxtype *);
void Change2FNumbering2(int, idxtype *, idxtype *);
void Change2FNumberingOrder(int, idxtype *, idxtype *, idxtype *, idxtype *);
void ChangeMesh2CNumbering(int, idxtype *);
void ChangeMesh2FNumbering(int, idxtype *, int, idxtype *, idxtype *);
void ChangeMesh2FNumbering2(int, idxtype *, int, int, idxtype *, idxtype *);

/* frename.c */
void METIS_PARTGRAPHRECURSIVE(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphrecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphrecursive_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphrecursive__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPARTGRAPHRECURSIVE(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphrecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphrecursive_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphrecursive__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_PARTGRAPHKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPARTGRAPHKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_EDGEND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_edgend(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_edgend_(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_edgend__(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_NODEND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodend(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodend_(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodend__(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_NODEWND(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodewnd(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodewnd_(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_nodewnd__(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_PARTMESHNODAL(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal_(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshnodal__(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_PARTMESHDUAL(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual_(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void metis_partmeshdual__(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_MESHTONODAL(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal_(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtonodal__(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_MESHTODUAL(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual_(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void metis_meshtodual__(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_ESTIMATEMEMORY(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory_(int *, idxtype *, idxtype *, int *, int *, int *);
void metis_estimatememory__(int *, idxtype *, idxtype *, int *, int *, int *);
void METIS_MCPARTGRAPHRECURSIVE(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive_(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_mcpartgraphrecursive__(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_MCPARTGRAPHKWAY(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_mcpartgraphkway(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_mcpartgraphkway_(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_mcpartgraphkway__(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_PARTGRAPHVKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void metis_partgraphvkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPARTGRAPHVKWAY(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphvkway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphvkway_(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void metis_wpartgraphvkway__(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);

/* graph.c */
void SetUpGraph(GraphType *, int, int, int, idxtype *, idxtype *, idxtype *, idxtype *, int);
void SetUpGraphKway(GraphType *, int, idxtype *, idxtype *);
void SetUpGraph2(GraphType *, int, int, idxtype *, idxtype *, float *, idxtype *);
void VolSetUpGraph(GraphType *, int, int, int, idxtype *, idxtype *, idxtype *, idxtype *, int);
void RandomizeGraph(GraphType *);
int IsConnectedSubdomain(CtrlType *, GraphType *, int, int);
int IsConnected(CtrlType *, GraphType *, int);
int IsConnected2(GraphType *, int);
int FindComponents(CtrlType *, GraphType *, idxtype *, idxtype *);

/* initpart.c */
void Init2WayPartition(CtrlType *, GraphType *, int *, float);
void InitSeparator(CtrlType *, GraphType *, float);
void GrowBisection(CtrlType *, GraphType *, int *, float);
void GrowBisectionNode(CtrlType *, GraphType *, float);
void RandomBisection(CtrlType *, GraphType *, int *, float);

/* mlkkm.c */
/*void spectralInit(CtrlType *, GraphType *, int *);*/
void spectralInit(GraphType *, int, int *, int *);
void MLKKM_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, int *, idxtype *, int);
void MLKKM_WPartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, float *, int *, int *, idxtype *, int);
int MLKKMPartitioning(CtrlType *, GraphType *, int, int, idxtype *, float *, float);

/* weighted kernel k-means */
void Compute_Weights(CtrlType *ctrl, GraphType *graph, idxtype *w);
void transform_matrix(CtrlType *ctrl, GraphType *graph, idxtype *w, float *adjwgt);
void transform_matrix_half(CtrlType *ctrl, GraphType *graph, idxtype *w, float *adjwgt);
void pingpong(CtrlType *, GraphType *, int , int , float *, float, int );
void Weighted_kernel_k_means(CtrlType *, GraphType *, int , idxtype *, float *, float );
void remove_empty_clusters_l1(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *tpwgts, float ubfactor);
void remove_empty_clusters_l2(CtrlType *ctrl, GraphType *graph, int nparts, idxtype *w, float *tpwgts, float ubfactor);
/*void Weighted_kernel_k_means(CtrlType *, GraphType *, int , idxtype *, float *, float *, float ); */
void MLKKMRefine(CtrlType *, GraphType *, GraphType *, int, int, float *, float);
float onePoint_move(GraphType *graph, int nparts, idxtype *sum, idxtype *squared_sum, idxtype *w, idxtype *self_sim, int **linearTerm, int ii);
void move1Point2EmptyCluster(GraphType *graph, int nparts, idxtype *sum, idxtype *squared_sum, idxtype *w, idxtype *self_sim, int **linearTerm, int k);
int local_search(CtrlType *, GraphType *, int, int, idxtype *, float *, float);
/*int local_search(CtrlType *, GraphType *, int, int, idxtype *, float *, float *, float); */
/* kmetis.c */
void METIS_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
int MlevelKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, float *, float);

/* kvmetis.c */
void METIS_PartGraphVKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphVKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
int MlevelVolKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, float *, float);

/* kwayfm.c */
void Random_KWayEdgeRefine(CtrlType *, GraphType *, int, float *, float, int, int);
void Greedy_KWayEdgeRefine(CtrlType *, GraphType *, int, float *, float, int);
void Greedy_KWayEdgeBalance(CtrlType *, GraphType *, int, float *, float, int);

/* kwayrefine.c */
void RefineKWay(CtrlType *, GraphType *, GraphType *, int, float *, float);
void AllocateKWayPartitionMemory(CtrlType *, GraphType *, int);
void ComputeKWayPartitionParams(CtrlType *, GraphType *, int);
void ProjectKWayPartition(CtrlType *, GraphType *, int);
int IsBalanced(idxtype *, int, float *, float);
void ComputeKWayBoundary(CtrlType *, GraphType *, int);
void ComputeKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* kwayvolfm.c */
void Random_KWayVolRefine(CtrlType *, GraphType *, int, float *, float, int, int);
void Random_KWayVolRefineMConn(CtrlType *, GraphType *, int, float *, float, int, int);
void Greedy_KWayVolBalance(CtrlType *, GraphType *, int, float *, float, int);
void Greedy_KWayVolBalanceMConn(CtrlType *, GraphType *, int, float *, float, int);
void KWayVolUpdate(CtrlType *, GraphType *, int, int, int, idxtype *, idxtype *, idxtype *);
void ComputeKWayVolume(GraphType *, int, idxtype *, idxtype *, idxtype *);
int ComputeVolume(GraphType *, idxtype *);
void CheckVolKWayPartitionParams(CtrlType *, GraphType *, int);
void ComputeVolSubDomainGraph(GraphType *, int, idxtype *, idxtype *);
void EliminateVolSubDomainEdges(CtrlType *, GraphType *, int, float *);
void EliminateVolComponents(CtrlType *, GraphType *, int, float *, float);

/* kwayvolrefine.c */
void RefineVolKWay(CtrlType *, GraphType *, GraphType *, int, float *, float);
void AllocateVolKWayPartitionMemory(CtrlType *, GraphType *, int);
void ComputeVolKWayPartitionParams(CtrlType *, GraphType *, int);
void ComputeKWayVolGains(CtrlType *, GraphType *, int);
void ProjectVolKWayPartition(CtrlType *, GraphType *, int);
void ComputeVolKWayBoundary(CtrlType *, GraphType *, int);
void ComputeVolKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* match.c */
void Match_HEMN(CtrlType *ctrl, GraphType *graph);
void Match_RM(CtrlType *, GraphType *);
void Match_RM_NVW(CtrlType *, GraphType *);
void Match_HEM(CtrlType *, GraphType *);
void Match_SHEM(CtrlType *, GraphType *);
void Match_SHEMN(CtrlType *, GraphType *);

/* mbalance.c */
void MocBalance2Way(CtrlType *, GraphType *, float *, float);
void MocGeneral2WayBalance(CtrlType *, GraphType *, float *, float);

/* mbalance2.c */
void MocBalance2Way2(CtrlType *, GraphType *, float *, float *);
void MocGeneral2WayBalance2(CtrlType *, GraphType *, float *, float *);
void SelectQueue3(int, float *, float *, int *, int *, PQueueType [MAXNCON][2], float *);

/* mcoarsen.c */
GraphType *MCCoarsen2Way(CtrlType *, GraphType *);

/* memory.c */
void AllocateWorkSpace(CtrlType *, GraphType *, int);
void FreeWorkSpace(CtrlType *, GraphType *);
int WspaceAvail(CtrlType *);
idxtype *idxwspacemalloc(CtrlType *, int);
void idxwspacefree(CtrlType *, int);
float *fwspacemalloc(CtrlType *, int);
void fwspacefree(CtrlType *, int);
GraphType *CreateGraph(void);
void InitGraph(GraphType *);
void FreeGraph(GraphType *);

/* mesh.c */
void METIS_MeshToDual(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_MeshToNodal(int *, int *, idxtype *, int *, int *, idxtype *, idxtype *);
void GENDUALMETIS(int, int, int, idxtype *, idxtype *, idxtype *adjncy);
void TRINODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void TETNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void HEXNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);
void QUADNODALMETIS(int, int, idxtype *, idxtype *, idxtype *adjncy);

/* meshpart.c */
void METIS_PartMeshNodal(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);
void METIS_PartMeshDual(int *, int *, idxtype *, int *, int *, int *, int *, idxtype *, idxtype *);

/* mfm.c */
void MocFM_2WayEdgeRefine(CtrlType *, GraphType *, float *, int);
void SelectQueue(int, float *, float *, int *, int *, PQueueType [MAXNCON][2]);
int BetterBalance(int, float *, float *, float *);
float Compute2WayHLoadImbalance(int, float *, float *);
void Compute2WayHLoadImbalanceVec(int, float *, float *, float *);

/* mfm2.c */
void MocFM_2WayEdgeRefine2(CtrlType *, GraphType *, float *, float *, int);
void SelectQueue2(int, float *, float *, int *, int *, PQueueType [MAXNCON][2], float *);
int IsBetter2wayBalance(int, float *, float *, float *);

/* mincover.o */
void MinCover(idxtype *, idxtype *, int, int, idxtype *, int *);
int MinCover_Augment(idxtype *, idxtype *, int, idxtype *, idxtype *, idxtype *, int);
void MinCover_Decompose(idxtype *, idxtype *, int, int, idxtype *, idxtype *, int *);
void MinCover_ColDFS(idxtype *, idxtype *, int, idxtype *, idxtype *, int);
void MinCover_RowDFS(idxtype *, idxtype *, int, idxtype *, idxtype *, int);

/* minitpart.c */
void MocInit2WayPartition(CtrlType *, GraphType *, float *, float);
void MocGrowBisection(CtrlType *, GraphType *, float *, float);
void MocRandomBisection(CtrlType *, GraphType *, float *, float);
void MocInit2WayBalance(CtrlType *, GraphType *, float *);
int SelectQueueOneWay(int, float *, float *, int, PQueueType [MAXNCON][2]);

/* minitpart2.c */
void MocInit2WayPartition2(CtrlType *, GraphType *, float *, float *);
void MocGrowBisection2(CtrlType *, GraphType *, float *, float *);
void MocGrowBisectionNew2(CtrlType *, GraphType *, float *, float *);
void MocInit2WayBalance2(CtrlType *, GraphType *, float *, float *);
int SelectQueueOneWay2(int, float *, PQueueType [MAXNCON][2], float *);

/* mkmetis.c */
void METIS_mCPartGraphKway(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
int MCMlevelKWayPartitioning(CtrlType *, GraphType *, int, idxtype *, float *);

/* mkwayfmh.c */
void MCRandom_KWayEdgeRefineHorizontal(CtrlType *, GraphType *, int, float *, int);
void MCGreedy_KWayEdgeBalanceHorizontal(CtrlType *, GraphType *, int, float *, int);
int AreAllHVwgtsBelow(int, float, float *, float, float *, float *);
int AreAllHVwgtsAbove(int, float, float *, float, float *, float *);
void ComputeHKWayLoadImbalance(int, int, float *, float *);
int MocIsHBalanced(int, int, float *, float *);
int IsHBalanceBetterFT(int, int, float *, float *, float *, float *);
int IsHBalanceBetterTT(int, int, float *, float *, float *, float *);

/* mkwayrefine.c */
void MocRefineKWayHorizontal(CtrlType *, GraphType *, GraphType *, int, float *);
void MocAllocateKWayPartitionMemory(CtrlType *, GraphType *, int);
void MocComputeKWayPartitionParams(CtrlType *, GraphType *, int);
void MocProjectKWayPartition(CtrlType *, GraphType *, int);
void MocComputeKWayBalanceBoundary(CtrlType *, GraphType *, int);

/* mmatch.c */
void MCMatch_RM(CtrlType *, GraphType *);
void MCMatch_HEM(CtrlType *, GraphType *);
void MCMatch_SHEM(CtrlType *, GraphType *);
void MCMatch_SHEBM(CtrlType *, GraphType *, int);
void MCMatch_SBHEM(CtrlType *, GraphType *, int);
float BetterVBalance(int, int, float *, float *, float *);
int AreAllVwgtsBelowFast(int, float *, float *, float);

/* mmd.c */
void genmmd(int, idxtype *, idxtype *, idxtype *, idxtype *, int , idxtype *, idxtype *, idxtype *, idxtype *, int, int *);
void mmdelm(int, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int, int);
int  mmdint(int, idxtype *xadj, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *);
void mmdnum(int, idxtype *, idxtype *, idxtype *);
void mmdupd(int, int, idxtype *, idxtype *, int, int *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int, int *tag);

/* mpmetis.c */
void METIS_mCPartGraphRecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_mCHPartGraphRecursive(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_mCPartGraphRecursiveInternal(int *, int *, idxtype *, idxtype *, float *, idxtype *, int *, int *, int *, idxtype *);
void METIS_mCHPartGraphRecursiveInternal(int *, int *, idxtype *, idxtype *, float *, idxtype *, int *, float *, int *, int *, idxtype *);
int MCMlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, float, int);
int MCHMlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, float *, int);
void MCMlevelEdgeBisection(CtrlType *, GraphType *, float *, float);
void MCHMlevelEdgeBisection(CtrlType *, GraphType *, float *, float *);

/* mrefine.c */
void MocRefine2Way(CtrlType *, GraphType *, GraphType *, float *, float);
void MocAllocate2WayPartitionMemory(CtrlType *, GraphType *);
void MocCompute2WayPartitionParams(CtrlType *, GraphType *);
void MocProject2WayPartition(CtrlType *, GraphType *);

/* mrefine2.c */
void MocRefine2Way2(CtrlType *, GraphType *, GraphType *, float *, float *);

/* mutil.c */
int AreAllVwgtsBelow(int, float, float *, float, float *, float);
int AreAnyVwgtsBelow(int, float, float *, float, float *, float);
int AreAllVwgtsAbove(int, float, float *, float, float *, float);
float ComputeLoadImbalance(int, int, float *, float *);
int AreAllBelow(int, float *, float *);

/* myqsort.c */
void iidxsort(int, idxtype *);
void iintsort(int, int *);
void ikeysort(int, KeyValueType *);
void ikeyvalsort(int, KeyValueType *);

/* ometis.c */
void METIS_EdgeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_NodeWND(int *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void MlevelNestedDissection(CtrlType *, GraphType *, idxtype *, float, int);
void MlevelNestedDissectionCC(CtrlType *, GraphType *, idxtype *, float, int);
void MlevelNodeBisectionMultiple(CtrlType *, GraphType *, int *, float);
void MlevelNodeBisection(CtrlType *, GraphType *, int *, float);
void SplitGraphOrder(CtrlType *, GraphType *, GraphType *, GraphType *);
void MMDOrder(CtrlType *, GraphType *, idxtype *, int);
int SplitGraphOrderCC(CtrlType *, GraphType *, GraphType *, int, idxtype *, idxtype *);

/* parmetis.c */
void METIS_PartGraphKway2(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphKway2(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_NodeNDP(int, idxtype *, idxtype *, int, int *, idxtype *, idxtype *, idxtype *);
void MlevelNestedDissectionP(CtrlType *, GraphType *, idxtype *, int, int, int, idxtype *);
void METIS_NodeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *);
void METIS_EdgeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *);

/* pmetis.c */
void METIS_PartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_WPartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
int MlevelRecursiveBisection(CtrlType *, GraphType *, int, idxtype *, float *, float, int);
void MlevelEdgeBisection(CtrlType *, GraphType *, int *, float);
void SplitGraphPart(CtrlType *, GraphType *, GraphType *, GraphType *);
void SetUpSplitGraph(GraphType *, GraphType *, int, int);

/* pqueue.c */
void PQueueInit(CtrlType *ctrl, PQueueType *, int, int);
void PQueueReset(PQueueType *);
void PQueueFree(CtrlType *ctrl, PQueueType *);
int PQueueGetSize(PQueueType *);
int PQueueInsert(PQueueType *, int, int);
int PQueueDelete(PQueueType *, int, int);
int PQueueUpdate(PQueueType *, int, int, int);
void PQueueUpdateUp(PQueueType *, int, int, int);
int PQueueGetMax(PQueueType *);
int PQueueSeeMax(PQueueType *);
int PQueueGetKey(PQueueType *);
int CheckHeap(PQueueType *);

/* refine.c */
void Refine2Way(CtrlType *, GraphType *, GraphType *, int *, float ubfactor);
void Allocate2WayPartitionMemory(CtrlType *, GraphType *);
void Compute2WayPartitionParams(CtrlType *, GraphType *);
void Project2WayPartition(CtrlType *, GraphType *);

/* separator.c */
void ConstructSeparator(CtrlType *, GraphType *, float);
void ConstructMinCoverSeparator0(CtrlType *, GraphType *, float);
void ConstructMinCoverSeparator(CtrlType *, GraphType *, float);

/* sfm.c */
void FM_2WayNodeRefine(CtrlType *, GraphType *, float, int);
void FM_2WayNodeRefineEqWgt(CtrlType *, GraphType *, int);
void FM_2WayNodeRefine_OneSided(CtrlType *, GraphType *, float, int);
void FM_2WayNodeBalance(CtrlType *, GraphType *, float);
int ComputeMaxNodeGain(int, idxtype *, idxtype *, idxtype *);

/* srefine.c */
void Refine2WayNode(CtrlType *, GraphType *, GraphType *, float);
void Allocate2WayNodePartitionMemory(CtrlType *, GraphType *);
void Compute2WayNodePartitionParams(CtrlType *, GraphType *);
void Project2WayNodePartition(CtrlType *, GraphType *);

/* stat.c */
void ComputePartitionInfo(GraphType *, int, idxtype *);
void ComputePartitionInfoBipartite(GraphType *, int, idxtype *);
void ComputePartitionBalance(GraphType *, int, idxtype *, float *);
float ComputeElementBalance(int, int, idxtype *);

/* subdomains.c */
void Random_KWayEdgeRefineMConn(CtrlType *, GraphType *, int, float *, float, int, int);
void Greedy_KWayEdgeBalanceMConn(CtrlType *, GraphType *, int, float *, float, int);
void PrintSubDomainGraph(GraphType *, int, idxtype *);
void ComputeSubDomainGraph(GraphType *, int, idxtype *, idxtype *);
void EliminateSubDomainEdges(CtrlType *, GraphType *, int, float *);
void MoveGroupMConn(CtrlType *, GraphType *, idxtype *, idxtype *, int, int, int, idxtype *);
void EliminateComponents(CtrlType *, GraphType *, int, float *, float);
void MoveGroup(CtrlType *, GraphType *, int, int, int, idxtype *, idxtype *);

/* timing.c */
void InitTimers(CtrlType *);
void PrintTimers(CtrlType *);
double seconds(void);

/* util.c */
void print_help(char *program_name);
void clusterSize(GraphType * graph, int *clustersize);
void sparse2dense(GraphType * graph, double * dense, float *);
void extractfilename(char *path, char *name);
void errexit(char *,...);
#ifndef DMALLOC
Chains *chainmalloc(int n, char *msg);
float **f2malloc(int n, int m, char *msg);
int **i2malloc(int, int, char *);
int *imalloc(int, char *);
idxtype *idxmalloc(int, char *);
float *fmalloc(int, char *);
int *ismalloc(int, int, char *);
idxtype *idxsmalloc(int, idxtype, char *);
void *GKmalloc(int, char *);
#endif
void GKfree(void **,...);
int *iset(int n, int val, int *x);
idxtype *idxset(int n, idxtype val, idxtype *x);
float *sset(int n, float val, float *x);
int iamax(int, int *);
int idxamax(int, idxtype *);
int idxamax_strd(int, idxtype *, int);
int samax(int, float *);
int samax2(int, float *);
int idxamin(int, idxtype *);
int samin(int, float *);
int idxsum(int, idxtype *);
int idxsum_strd(int, idxtype *, int);
void idxadd(int, idxtype *, idxtype *);
int charsum(int, char *);
int isum(int, int *);
float ssum(int, float *);
float ssum_strd(int n, float *x, int);
void sscale(int n, float, float *x);
float snorm2(int, float *);
float sdot(int n, float *, float *);
void saxpy(int, float, float *, int, float *, int);
void RandomPermute(int, idxtype *, int);
void RandomInit(int n, int k, idxtype *label);
int ispow2(int);
void InitRandom(int);
int log2_metis(int);










/***************************************************************
* Programs Directory
****************************************************************/

/* io.c */
void ReadCoarsestInit(GraphType *graph, char *filename, int *wgtflag);
void WriteCoarsestGraph(GraphType *graph, char *filename, int *wgtflag);
void CreateGraph_Matlab(GraphType *graph, double* idata, double* jdata, double* edgeval, int vtx, int edges, int *wgtflag);
void ReadGraph(GraphType *, char *, int *);
void WritePartition(char *, idxtype *, int, int);
void WriteMeshPartition(char *, int, int, idxtype *, int, idxtype *);
void WritePermutation(char *, idxtype *, int);
int CheckGraph(GraphType *);
idxtype *ReadMesh(char *, int *, int *, int *);
void WriteGraph(char *, int, idxtype *, idxtype *);

/* smbfactor.c */
void ComputeFillIn(GraphType *, idxtype *);
idxtype ComputeFillIn2(GraphType *, idxtype *);
int smbfct(int, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int *, idxtype *, idxtype *, int *);


/***************************************************************
* Test Directory
****************************************************************/
void Test_PartGraph(int, idxtype *, idxtype *);
int VerifyPart(int, idxtype *, idxtype *, idxtype *, idxtype *, int, int, idxtype *);
int VerifyWPart(int, idxtype *, idxtype *, idxtype *, idxtype *, int, float *, int, idxtype *);
void Test_PartGraphV(int, idxtype *, idxtype *);
int VerifyPartV(int, idxtype *, idxtype *, idxtype *, idxtype *, int, int, idxtype *);
int VerifyWPartV(int, idxtype *, idxtype *, idxtype *, idxtype *, int, float *, int, idxtype *);
void Test_PartGraphmC(int, idxtype *, idxtype *);
int VerifyPartmC(int, int, idxtype *, idxtype *, idxtype *, idxtype *, int, float *, int, idxtype *);
void Test_ND(int, idxtype *, idxtype *);
int VerifyND(int, idxtype *, idxtype *);

