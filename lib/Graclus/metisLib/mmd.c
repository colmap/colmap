/*
 * mmd.c
 *
 * **************************************************************
 * The following C function was developed from a FORTRAN subroutine
 * in SPARSPAK written by Eleanor Chu, Alan George, Joseph Liu
 * and Esmond Ng.
 *
 * The FORTRAN-to-C transformation and modifications such as dynamic
 * memory allocation and deallocation were performed by Chunguang
 * Sun.
 * **************************************************************
 *
 * Taken from SMMS, George 12/13/94
 *
 * The meaning of invperm, and perm vectors is different from that
 * in genqmd_ of SparsPak
 *
 * $Id: mmd.c,v 1.1 1998/11/27 17:59:25 karypis Exp $
 */

#include "metis.h"


/*************************************************************************
*  genmmd  -- multiple minimum external degree
*  purpose -- this routine implements the minimum degree
*     algorithm. it makes use of the implicit representation
*     of elimination graphs by quotient graphs, and the notion
*     of indistinguishable nodes. It also implements the modifications
*     by multiple elimination and minimum external degree.
*     Caution -- the adjacency vector adjncy will be destroyed.
*  Input parameters --
*     neqns -- number of equations.
*     (xadj, adjncy) -- the adjacency structure.
*     delta  -- tolerance value for multiple elimination.
*     maxint -- maximum machine representable (short) integer
*               (any smaller estimate will do) for marking nodes.
*  Output parameters --
*     perm -- the minimum degree ordering.
*     invp -- the inverse of perm.
*     *ncsub -- an upper bound on the number of nonzero subscripts
*               for the compressed storage scheme.
*  Working parameters --
*     head -- vector for head of degree lists.
*     invp  -- used temporarily for degree forward link.
*     perm  -- used temporarily for degree backward link.
*     qsize -- vector for size of supernodes.
*     list -- vector for temporary linked lists.
*     marker -- a temporary marker vector.
*  Subroutines used -- mmdelm, mmdint, mmdnum, mmdupd.
**************************************************************************/
void genmmd(int neqns, idxtype *xadj, idxtype *adjncy, idxtype *invp, idxtype *perm,
     int delta, idxtype *head, idxtype *qsize, idxtype *list, idxtype *marker,
     int maxint, int *ncsub)
{
    int  ehead, i, mdeg, mdlmt, mdeg_node, nextmd, num, tag;

    if (neqns <= 0)
      return;

    /* Adjust from C to Fortran */
    xadj--; adjncy--; invp--; perm--; head--; qsize--; list--; marker--;

    /* initialization for the minimum degree algorithm. */
    *ncsub = 0;
    mmdint(neqns, xadj, adjncy, head, invp, perm, qsize, list, marker);

    /*  'num' counts the number of ordered nodes plus 1. */
    num = 1;

    /* eliminate all isolated nodes. */
    nextmd = head[1];
    while (nextmd > 0) {
      mdeg_node = nextmd;
      nextmd = invp[mdeg_node];
      marker[mdeg_node] = maxint;
      invp[mdeg_node] = -num;
      num = num + 1;
    }

    /* search for node of the minimum degree. 'mdeg' is the current */
    /* minimum degree; 'tag' is used to facilitate marking nodes.   */
    if (num > neqns)
      goto n1000;
    tag = 1;
    head[1] = 0;
    mdeg = 2;

    /* infinite loop here ! */
    while (1) {
      while (head[mdeg] <= 0)
        mdeg++;

      /* use value of 'delta' to set up 'mdlmt', which governs */
      /* when a degree update is to be performed.              */
      mdlmt = mdeg + delta;
      ehead = 0;

n500:
      mdeg_node = head[mdeg];
      while (mdeg_node <= 0) {
        mdeg++;

        if (mdeg > mdlmt)
          goto n900;
        mdeg_node = head[mdeg];
      };

      /*  remove 'mdeg_node' from the degree structure. */
      nextmd = invp[mdeg_node];
      head[mdeg] = nextmd;
      if (nextmd > 0)
        perm[nextmd] = -mdeg;
      invp[mdeg_node] = -num;
      *ncsub += mdeg + qsize[mdeg_node] - 2;
      if ((num+qsize[mdeg_node]) > neqns)
        goto n1000;

      /*  eliminate 'mdeg_node' and perform quotient graph */
      /*  transformation. reset 'tag' value if necessary.    */
      tag++;
      if (tag >= maxint) {
        tag = 1;
        for (i = 1; i <= neqns; i++)
          if (marker[i] < maxint)
            marker[i] = 0;
      };

      mmdelm(mdeg_node, xadj, adjncy, head, invp, perm, qsize, list, marker, maxint, tag);

      num += qsize[mdeg_node];
      list[mdeg_node] = ehead;
      ehead = mdeg_node;
      if (delta >= 0)
        goto n500;

 n900:
      /* update degrees of the nodes involved in the  */
      /* minimum degree nodes elimination.            */
      if (num > neqns)
        goto n1000;
      mmdupd( ehead, neqns, xadj, adjncy, delta, &mdeg, head, invp, perm, qsize, list, marker, maxint, &tag);
    }; /* end of -- while ( 1 ) -- */

n1000:
    mmdnum( neqns, perm, invp, qsize );

    /* Adjust from Fortran back to C*/
    xadj++; adjncy++; invp++; perm++; head++; qsize++; list++; marker++;
}


/**************************************************************************
*           mmdelm ...... multiple minimum degree elimination
* Purpose -- This routine eliminates the node mdeg_node of minimum degree
*     from the adjacency structure, which is stored in the quotient
*     graph format. It also transforms the quotient graph representation
*     of the elimination graph.
* Input parameters --
*     mdeg_node -- node of minimum degree.
*     maxint -- estimate of maximum representable (short) integer.
*     tag    -- tag value.
* Updated parameters --
*     (xadj, adjncy) -- updated adjacency structure.
*     (head, forward, backward) -- degree doubly linked structure.
*     qsize -- size of supernode.
*     marker -- marker vector.
*     list -- temporary linked list of eliminated nabors.
***************************************************************************/
void mmdelm(int mdeg_node, idxtype *xadj, idxtype *adjncy, idxtype *head, idxtype *forward,
     idxtype *backward, idxtype *qsize, idxtype *list, idxtype *marker, int maxint,int tag)
{
    int   element, i,   istop, istart, j,
          jstop, jstart, link,
          nabor, node, npv, nqnbrs, nxnode,
          pvnode, rlmt, rloc, rnode, xqnbr;

    /* find the reachable set of 'mdeg_node' and */
    /* place it in the data structure.           */
    marker[mdeg_node] = tag;
    istart = xadj[mdeg_node];
    istop = xadj[mdeg_node+1] - 1;

    /* 'element' points to the beginning of the list of  */
    /* eliminated nabors of 'mdeg_node', and 'rloc' gives the */
    /* storage location for the next reachable node.   */
    element = 0;
    rloc = istart;
    rlmt = istop;
    for ( i = istart; i <= istop; i++ ) {
        nabor = adjncy[i];
        if ( nabor == 0 ) break;
        if ( marker[nabor] < tag ) {
           marker[nabor] = tag;
           if ( forward[nabor] < 0 )  {
              list[nabor] = element;
              element = nabor;
           } else {
              adjncy[rloc] = nabor;
              rloc++;
           };
        }; /* end of -- if -- */
    }; /* end of -- for -- */

  /* merge with reachable nodes from generalized elements. */
  while ( element > 0 ) {
      adjncy[rlmt] = -element;
      link = element;

n400:
      jstart = xadj[link];
      jstop = xadj[link+1] - 1;
      for ( j = jstart; j <= jstop; j++ ) {
          node = adjncy[j];
          link = -node;
          if ( node < 0 )  goto n400;
          if ( node == 0 ) break;
          if ((marker[node]<tag)&&(forward[node]>=0)) {
             marker[node] = tag;
             /*use storage from eliminated nodes if necessary.*/
             while ( rloc >= rlmt ) {
                   link = -adjncy[rlmt];
                   rloc = xadj[link];
                   rlmt = xadj[link+1] - 1;
             };
             adjncy[rloc] = node;
             rloc++;
          };
      }; /* end of -- for ( j = jstart; -- */
      element = list[element];
    };  /* end of -- while ( element > 0 ) -- */
    if ( rloc <= rlmt ) adjncy[rloc] = 0;
    /* for each node in the reachable set, do the following. */
    link = mdeg_node;

n1100:
    istart = xadj[link];
    istop = xadj[link+1] - 1;
    for ( i = istart; i <= istop; i++ ) {
        rnode = adjncy[i];
        link = -rnode;
        if ( rnode < 0 ) goto n1100;
        if ( rnode == 0 ) return;

        /* 'rnode' is in the degree list structure. */
        pvnode = backward[rnode];
        if (( pvnode != 0 ) && ( pvnode != (-maxint) )) {
           /* then remove 'rnode' from the structure. */
           nxnode = forward[rnode];
           if ( nxnode > 0 ) backward[nxnode] = pvnode;
           if ( pvnode > 0 ) forward[pvnode] = nxnode;
           npv = -pvnode;
           if ( pvnode < 0 ) head[npv] = nxnode;
        };

        /* purge inactive quotient nabors of 'rnode'. */
        jstart = xadj[rnode];
        jstop = xadj[rnode+1] - 1;
        xqnbr = jstart;
        for ( j = jstart; j <= jstop; j++ ) {
            nabor = adjncy[j];
            if ( nabor == 0 ) break;
            if ( marker[nabor] < tag ) {
                adjncy[xqnbr] = nabor;
                xqnbr++;
            };
        };

        /* no active nabor after the purging. */
        nqnbrs = xqnbr - jstart;
        if ( nqnbrs <= 0 ) {
           /* merge 'rnode' with 'mdeg_node'. */
           qsize[mdeg_node] += qsize[rnode];
           qsize[rnode] = 0;
           marker[rnode] = maxint;
           forward[rnode] = -mdeg_node;
           backward[rnode] = -maxint;
        } else {
           /* flag 'rnode' for degree update, and  */
           /* add 'mdeg_node' as a nabor of 'rnode'.      */
           forward[rnode] = nqnbrs + 1;
           backward[rnode] = 0;
           adjncy[xqnbr] = mdeg_node;
           xqnbr++;
           if ( xqnbr <= jstop )  adjncy[xqnbr] = 0;
        };
      }; /* end of -- for ( i = istart; -- */
      return;
 }

/***************************************************************************
*    mmdint ---- mult minimum degree initialization
*    purpose -- this routine performs initialization for the
*       multiple elimination version of the minimum degree algorithm.
*    input parameters --
*       neqns  -- number of equations.
*       (xadj, adjncy) -- adjacency structure.
*    output parameters --
*       (head, dfrow, backward) -- degree doubly linked structure.
*       qsize -- size of supernode ( initialized to one).
*       list -- linked list.
*       marker -- marker vector.
****************************************************************************/
int  mmdint(int neqns, idxtype *xadj, idxtype *adjncy, idxtype *head, idxtype *forward,
     idxtype *backward, idxtype *qsize, idxtype *list, idxtype *marker)
{
    int  fnode, ndeg, node;

    for ( node = 1; node <= neqns; node++ ) {
        head[node] = 0;
        qsize[node] = 1;
        marker[node] = 0;
        list[node] = 0;
    };

    /* initialize the degree doubly linked lists. */
    for ( node = 1; node <= neqns; node++ ) {
        ndeg = xadj[node+1] - xadj[node]/* + 1*/;   /* george */
        if (ndeg == 0)
          ndeg = 1;
        fnode = head[ndeg];
        forward[node] = fnode;
        head[ndeg] = node;
        if ( fnode > 0 ) backward[fnode] = node;
        backward[node] = -ndeg;
    };
    return 0;
}

/****************************************************************************
* mmdnum --- multi minimum degree numbering
* purpose -- this routine performs the final step in producing
*    the permutation and inverse permutation vectors in the
*    multiple elimination version of the minimum degree
*    ordering algorithm.
* input parameters --
*     neqns -- number of equations.
*     qsize -- size of supernodes at elimination.
* updated parameters --
*     invp -- inverse permutation vector. on input,
*             if qsize[node] = 0, then node has been merged
*             into the node -invp[node]; otherwise,
*            -invp[node] is its inverse labelling.
* output parameters --
*     perm -- the permutation vector.
****************************************************************************/
void mmdnum(int neqns, idxtype *perm, idxtype *invp, idxtype *qsize)
{
  int father, nextf, node, nqsize, num, root;

  for ( node = 1; node <= neqns; node++ ) {
      nqsize = qsize[node];
      if ( nqsize <= 0 ) perm[node] = invp[node];
      if ( nqsize > 0 )  perm[node] = -invp[node];
  };

  /* for each node which has been merged, do the following. */
  for ( node = 1; node <= neqns; node++ ) {
      if ( perm[node] <= 0 )  {

	 /* trace the merged tree until one which has not */
         /* been merged, call it root.                    */
         father = node;
         while ( perm[father] <= 0 )
            father = - perm[father];

         /* number node after root. */
         root = father;
         num = perm[root] + 1;
         invp[node] = -num;
         perm[root] = num;

         /* shorten the merged tree. */
         father = node;
         nextf = - perm[father];
         while ( nextf > 0 ) {
            perm[father] = -root;
            father = nextf;
            nextf = -perm[father];
         };
      };  /* end of -- if ( perm[node] <= 0 ) -- */
  }; /* end of -- for ( node = 1; -- */

  /* ready to compute perm. */
  for ( node = 1; node <= neqns; node++ ) {
        num = -invp[node];
        invp[node] = num;
        perm[num] = node;
  };
  return;
}

/****************************************************************************
* mmdupd ---- multiple minimum degree update
* purpose -- this routine updates the degrees of nodes after a
*            multiple elimination step.
* input parameters --
*    ehead -- the beginning of the list of eliminated nodes
*             (i.e., newly formed elements).
*    neqns -- number of equations.
*    (xadj, adjncy) -- adjacency structure.
*    delta -- tolerance value for multiple elimination.
*    maxint -- maximum machine representable (short) integer.
* updated parameters --
*    mdeg -- new minimum degree after degree update.
*    (head, forward, backward) -- degree doubly linked structure.
*    qsize -- size of supernode.
*    list -- marker vector for degree update.
*    *tag   -- tag value.
****************************************************************************/
void mmdupd(int ehead, int neqns, idxtype *xadj, idxtype *adjncy, int delta, int *mdeg,
     idxtype *head, idxtype *forward, idxtype *backward, idxtype *qsize, idxtype *list,
     idxtype *marker, int maxint,int *tag)
{
 int  deg, deg0, element, enode, fnode, i, iq2, istop,
      istart, j, jstop, jstart, link, mdeg0, mtag, nabor,
      node, q2head, qxhead;

      mdeg0 = *mdeg + delta;
      element = ehead;

n100:
      if ( element <= 0 ) return;

      /* for each of the newly formed element, do the following. */
      /* reset tag value if necessary.                           */
      mtag = *tag + mdeg0;
      if ( mtag >= maxint ) {
         *tag = 1;
         for ( i = 1; i <= neqns; i++ )
             if ( marker[i] < maxint ) marker[i] = 0;
         mtag = *tag + mdeg0;
      };

      /* create two linked lists from nodes associated with 'element': */
      /* one with two nabors (q2head) in the adjacency structure, and the*/
      /* other with more than two nabors (qxhead). also compute 'deg0',*/
      /* number of nodes in this element.                              */
      q2head = 0;
      qxhead = 0;
      deg0 = 0;
      link =element;

n400:
      istart = xadj[link];
      istop = xadj[link+1] - 1;
      for ( i = istart; i <= istop; i++ ) {
          enode = adjncy[i];
          link = -enode;
          if ( enode < 0 )  goto n400;
          if ( enode == 0 ) break;
          if ( qsize[enode] != 0 ) {
             deg0 += qsize[enode];
             marker[enode] = mtag;

             /*'enode' requires a degree update*/
             if ( backward[enode] == 0 ) {
                /* place either in qxhead or q2head list. */
                if ( forward[enode] != 2 ) {
                     list[enode] = qxhead;
                     qxhead = enode;
                } else {
                     list[enode] = q2head;
                     q2head = enode;
                };
             };
          }; /* enf of -- if ( qsize[enode] != 0 ) -- */
      }; /* end of -- for ( i = istart; -- */

      /* for each node in q2 list, do the following. */
      enode = q2head;
      iq2 = 1;

n900:
      if ( enode <= 0 ) goto n1500;
      if ( backward[enode] != 0 ) goto n2200;
      (*tag)++;
      deg = deg0;

      /* identify the other adjacent element nabor. */
      istart = xadj[enode];
      nabor = adjncy[istart];
      if ( nabor == element ) nabor = adjncy[istart+1];
      link = nabor;
      if ( forward[nabor] >= 0 ) {
           /* nabor is uneliminated, increase degree count. */
           deg += qsize[nabor];
           goto n2100;
      };

       /* the nabor is eliminated. for each node in the 2nd element */
       /* do the following.                                         */
n1000:
       istart = xadj[link];
       istop = xadj[link+1] - 1;
       for ( i = istart; i <= istop; i++ ) {
           node = adjncy[i];
           link = -node;
           if ( node != enode ) {
                if ( node < 0 ) goto n1000;
                if ( node == 0 )  goto n2100;
                if ( qsize[node] != 0 ) {
                     if ( marker[node] < *tag ) {
                        /* 'node' is not yet considered. */
                        marker[node] = *tag;
                        deg += qsize[node];
                     } else {
                        if ( backward[node] == 0 ) {
                             if ( forward[node] == 2 ) {
                                /* 'node' is indistinguishable from 'enode'.*/
                                /* merge them into a new supernode.         */
                                qsize[enode] += qsize[node];
                                qsize[node] = 0;
                                marker[node] = maxint;
                                forward[node] = -enode;
                                backward[node] = -maxint;
                             } else {
                                /* 'node' is outmacthed by 'enode' */
				if (backward[node]==0) backward[node] = -maxint;
                             };
                        }; /* end of -- if ( backward[node] == 0 ) -- */
                    }; /* end of -- if ( marker[node] < *tag ) -- */
                }; /* end of -- if ( qsize[node] != 0 ) -- */
              }; /* end of -- if ( node != enode ) -- */
          }; /* end of -- for ( i = istart; -- */
          goto n2100;

n1500:
          /* for each 'enode' in the 'qx' list, do the following. */
          enode = qxhead;
          iq2 = 0;

n1600:    if ( enode <= 0 )  goto n2300;
          if ( backward[enode] != 0 )  goto n2200;
          (*tag)++;
          deg = deg0;

          /*for each unmarked nabor of 'enode', do the following.*/
          istart = xadj[enode];
          istop = xadj[enode+1] - 1;
          for ( i = istart; i <= istop; i++ ) {
                nabor = adjncy[i];
                if ( nabor == 0 ) break;
                if ( marker[nabor] < *tag ) {
                     marker[nabor] = *tag;
                     link = nabor;
                     if ( forward[nabor] >= 0 )
                          /*if uneliminated, include it in deg count.*/
                          deg += qsize[nabor];
                     else {
n1700:
                          /* if eliminated, include unmarked nodes in this*/
                          /* element into the degree count.             */
                          jstart = xadj[link];
                          jstop = xadj[link+1] - 1;
                          for ( j = jstart; j <= jstop; j++ ) {
                                node = adjncy[j];
                                link = -node;
                                if ( node < 0 ) goto n1700;
                                if ( node == 0 ) break;
                                if ( marker[node] < *tag ) {
                                    marker[node] = *tag;
                                    deg += qsize[node];
                                };
                          }; /* end of -- for ( j = jstart; -- */
                     }; /* end of -- if ( forward[nabor] >= 0 ) -- */
                  }; /* end of -- if ( marker[nabor] < *tag ) -- */
          }; /* end of -- for ( i = istart; -- */

n2100:
          /* update external degree of 'enode' in degree structure, */
          /* and '*mdeg' if necessary.                     */
          deg = deg - qsize[enode] + 1;
          fnode = head[deg];
          forward[enode] = fnode;
          backward[enode] = -deg;
          if ( fnode > 0 ) backward[fnode] = enode;
          head[deg] = enode;
          if ( deg < *mdeg ) *mdeg = deg;

n2200:
          /* get next enode in current element. */
          enode = list[enode];
          if ( iq2 == 1 ) goto n900;
          goto n1600;

n2300:
          /* get next element in the list. */
          *tag = mtag;
          element = list[element];
          goto n100;
    }
