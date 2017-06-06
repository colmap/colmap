/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * defs.h
 *
 * This file contains constant definitions
 *
 * Started 8/27/94
 * George
 *
 * $Id: defs.h,v 1.1 1998/11/27 17:59:13 karypis Exp $
 *
 */

#define METISTITLE              "  METIS 4.0.1 Copyright 1998, Regents of the University of Minnesota\n\n"

/* addtion begins */
#define MLKKMTITLE              " Graclus 1.2 Copyright 2008, Brian Kulis and Yuqiang Guan \n\n"
/* addtion ends */

#define MAXLINE			1280000

#define LTERM			(void **) 0	/* List terminator for GKfree() */

#define MAXNCON			16		/* The maximum number of constrains */
#define MAXNOBJ			16		/* The maximum number of objectives */

#define PLUS_GAINSPAN   	500             /* Parameters for FM buckets */
#define NEG_GAINSPAN    	500

#define HTLENGTH		((1<<11)-1)

/* define what type of cut */
/*extern int cutType; */

#define NCUT                    0 //(default)
#define RASSO                   1
#define RCUT                    2



#define MAXITERATIONS           10

/* Meaning of various options[] parameters */
#define OPTION_PTYPE		0
#define OPTION_CTYPE		1
#define OPTION_ITYPE		2
#define OPTION_RTYPE		3
#define OPTION_DBGLVL		4
#define OPTION_OFLAGS		5
#define OPTION_PFACTOR		6
#define OPTION_NSEPS		7

#define OFLAG_COMPRESS		1	/* Try to compress the graph */
#define OFLAG_CCMP		2	/* Find and order connected components */


/* Default options for PMETIS */
#define PMETIS_CTYPE		MATCH_SHEM
#define PMETIS_ITYPE		IPART_GGPKL
#define PMETIS_RTYPE		RTYPE_FM
#define PMETIS_DBGLVL		0

/* Default options for KMETIS */
#define KMETIS_CTYPE		MATCH_SHEMN
/*#define KMETIS_CTYPE		MATCH_SHEM */
/*#define KMETIS_CTYPE		MATCH_HEMN */
#define KMETIS_ITYPE		IPART_PMETIS
#define KMETIS_RTYPE		RTYPE_KWAYRANDOM_MCONN
#define KMETIS_DBGLVL		0

/* Default options for OEMETIS */
#define OEMETIS_CTYPE		MATCH_SHEM
#define OEMETIS_ITYPE		IPART_GGPKL
#define OEMETIS_RTYPE		RTYPE_FM
#define OEMETIS_DBGLVL		0

/* Default options for ONMETIS */
#define ONMETIS_CTYPE		MATCH_SHEM
#define ONMETIS_ITYPE		IPART_GGPKL
#define ONMETIS_RTYPE		RTYPE_SEP1SIDED
#define ONMETIS_DBGLVL		0
#define ONMETIS_OFLAGS		OFLAG_COMPRESS
#define ONMETIS_PFACTOR		-1
#define ONMETIS_NSEPS		1

/* Default options for McPMETIS */
#define McPMETIS_CTYPE		MATCH_SHEBM_ONENORM
#define McPMETIS_ITYPE		IPART_RANDOM
#define McPMETIS_RTYPE		RTYPE_FM
#define McPMETIS_DBGLVL		0

/* Default options for McKMETIS */
#define McKMETIS_CTYPE		MATCH_SHEBM_ONENORM
#define McKMETIS_ITYPE		IPART_McHPMETIS
#define McKMETIS_RTYPE		RTYPE_KWAYRANDOM
#define McKMETIS_DBGLVL		0

/* Default options for KVMETIS */
#define KVMETIS_CTYPE		MATCH_SHEM
#define KVMETIS_ITYPE		IPART_PMETIS
#define KVMETIS_RTYPE		RTYPE_KWAYRANDOM
#define KVMETIS_DBGLVL		0


/* Operations supported by stand-alone code */
#define OP_PMETIS		1
#define OP_KMETIS		2
#define OP_OEMETIS		3
#define OP_ONMETIS		4
#define OP_ONWMETIS		5
#define OP_KVMETIS		6


/* Matching Schemes */
#define MATCH_RM		1
#define MATCH_HEM		2
#define MATCH_SHEM		3
#define MATCH_SHEMKWAY		4
#define MATCH_SHEBM_ONENORM	5
#define MATCH_SHEBM_INFNORM	6
#define MATCH_SBHEM_ONENORM	7
#define MATCH_SBHEM_INFNORM	8
#define MATCH_HEMN              9
#define MATCH_SHEMN             10

/* Initial partitioning schemes for PMETIS and ONMETIS */
#define IPART_GGPKL		1
#define IPART_GGPKLNODE		2
#define IPART_RANDOM		2

/* Refinement schemes for PMETIS */
#define RTYPE_FM		1

/* Initial partitioning schemes for KMETIS */
#define IPART_PMETIS		1

/* Refinement schemes for KMETIS */
#define RTYPE_KWAYRANDOM	1
#define RTYPE_KWAYGREEDY	2
#define RTYPE_KWAYRANDOM_MCONN	3

/* Refinement schemes for ONMETIS */
#define RTYPE_SEP2SIDED		1
#define RTYPE_SEP1SIDED		2

/* Initial Partitioning Schemes for McKMETIS */
#define IPART_McPMETIS		1   	/* Simple McPMETIS */
#define IPART_McHPMETIS		2	/* horizontally relaxed McPMETIS */

#define UNMATCHED		-1

#define HTABLE_EMPTY    	-1

#define NGR_PASSES		4	/* Number of greedy refinement passes */
#define NLGR_PASSES		5	/* Number of GR refinement during IPartition */

#define LARGENIPARTS		8	/* Number of random initial partitions */
#define SMALLNIPARTS		3	/* Number of random initial partitions */

#define COARSEN_FRACTION	0.75	/* Node reduction between succesive coarsening levels */
#define COARSEN_FRACTION2	0.90	/* Node reduction between succesive coarsening levels */
#define UNBALANCE_FRACTION		1.05

#define COMPRESSION_FRACTION		0.85

#define ORDER_UNBALANCE_FRACTION	1.10

#define MMDSWITCH		200

#define HORIZONTAL_IMBALANCE		1.05

/* Debug Levels */
#define DBG_TIME	1		/* Perform timing analysis */
#define DBG_OUTPUT	2
#define DBG_COARSEN   	4		/* Show the coarsening progress */
#define DBG_REFINE	8		/* Show info on communication during folding */
#define DBG_IPART	16		/* Show info on initial partition */
#define DBG_MOVEINFO	32		/* Show info on communication during folding */
#define DBG_KWAYPINFO	64		/* Show info on communication during folding */
#define DBG_SEPINFO	128		/* Show info on communication during folding */
