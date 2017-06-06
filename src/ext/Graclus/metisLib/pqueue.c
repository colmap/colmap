/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * pqueue.c
 *
 * This file contains functions for manipulating the bucket list
 * representation of the gains associated with each vertex in a graph.
 * These functions are used by the refinement algorithms
 *
 * Started 9/2/94
 * George
 *
 * $Id: pqueue.c,v 1.1 1998/11/27 17:59:28 karypis Exp $
 *
 */

#include "metis.h"


/*************************************************************************
* This function initializes the data structures of the priority queue
**************************************************************************/
void PQueueInit(CtrlType *ctrl, PQueueType *queue, int maxnodes, int maxgain)
{
  int i, j, ncore;

  queue->nnodes = 0;
  queue->maxnodes = maxnodes;

  queue->buckets = NULL;
  queue->nodes = NULL;
  queue->heap = NULL;
  queue->locator = NULL;

  if (maxgain > PLUS_GAINSPAN || maxnodes < 500)
    queue->type = 2;
  else
    queue->type = 1;

  if (queue->type == 1) {
    queue->pgainspan = amin(PLUS_GAINSPAN, maxgain);
    queue->ngainspan = amin(NEG_GAINSPAN, maxgain);

    j = queue->ngainspan+queue->pgainspan+1;

    ncore = 2 + (sizeof(ListNodeType)/sizeof(idxtype))*maxnodes + (sizeof(ListNodeType *)/sizeof(idxtype))*j;

    if (WspaceAvail(ctrl) > ncore) {
      queue->nodes = (ListNodeType *)idxwspacemalloc(ctrl, (sizeof(ListNodeType)/sizeof(idxtype))*maxnodes);
      queue->buckets = (ListNodeType **)idxwspacemalloc(ctrl, (sizeof(ListNodeType *)/sizeof(idxtype))*j);
      queue->mustfree = 0;
    }
    else { /* Not enough memory in the wspace, allocate it */
      queue->nodes = (ListNodeType *)idxmalloc((sizeof(ListNodeType)/sizeof(idxtype))*maxnodes, "PQueueInit: queue->nodes");
      queue->buckets = (ListNodeType **)idxmalloc((sizeof(ListNodeType *)/sizeof(idxtype))*j, "PQueueInit: queue->buckets");
      queue->mustfree = 1;
    }

    for (i=0; i<maxnodes; i++)
      queue->nodes[i].id = i;

    for (i=0; i<j; i++)
      queue->buckets[i] = NULL;

    queue->buckets += queue->ngainspan;  /* Advance buckets by the ngainspan proper indexing */
    queue->maxgain = -queue->ngainspan;
  }
  else {
    queue->heap = (KeyValueType *)idxwspacemalloc(ctrl, (sizeof(KeyValueType)/sizeof(idxtype))*maxnodes);
    queue->locator = idxwspacemalloc(ctrl, maxnodes);
    idxset(maxnodes, -1, queue->locator);
  }

}


/*************************************************************************
* This function resets the buckets
**************************************************************************/
void PQueueReset(PQueueType *queue)
{
  int i, j;
  queue->nnodes = 0;

  if (queue->type == 1) {
    queue->maxgain = -queue->ngainspan;

    j = queue->ngainspan+queue->pgainspan+1;
    queue->buckets -= queue->ngainspan;
    for (i=0; i<j; i++)
      queue->buckets[i] = NULL;
    queue->buckets += queue->ngainspan;
  }
  else {
    idxset(queue->maxnodes, -1, queue->locator);
  }

}


/*************************************************************************
* This function frees the buckets
**************************************************************************/
void PQueueFree(CtrlType *ctrl, PQueueType *queue)
{

  if (queue->type == 1) {
    if (queue->mustfree) {
      queue->buckets -= queue->ngainspan;
      GKfree((void**) &queue->nodes, (void**) &queue->buckets, LTERM);
    }
    else {
      idxwspacefree(ctrl, sizeof(ListNodeType *)*(queue->ngainspan+queue->pgainspan+1)/sizeof(idxtype));
      idxwspacefree(ctrl, sizeof(ListNodeType)*queue->maxnodes/sizeof(idxtype));
    }
  }
  else {
    idxwspacefree(ctrl, sizeof(KeyValueType)*queue->maxnodes/sizeof(idxtype));
    idxwspacefree(ctrl, queue->maxnodes);
  }

  queue->maxnodes = 0;
}


/*************************************************************************
* This function returns the number of nodes in the queue
**************************************************************************/
int PQueueGetSize(PQueueType *queue)
{
  return queue->nnodes;
}


/*************************************************************************
* This function adds a node of certain gain into a partition
**************************************************************************/
int PQueueInsert(PQueueType *queue, int node, int gain)
{
  int i, j, k;
  idxtype *locator;
  ListNodeType *newnode;
  KeyValueType *heap;

  if (queue->type == 1) {
    ASSERT(gain >= -queue->ngainspan && gain <= queue->pgainspan);

    /* Allocate and add the node */
    queue->nnodes++;
    newnode = queue->nodes + node;

    /* Attach this node in the doubly-linked list */
    newnode->next = queue->buckets[gain];
    newnode->prev = NULL;
    if (newnode->next != NULL)
      newnode->next->prev = newnode;
    queue->buckets[gain] = newnode;

    if (queue->maxgain < gain)
      queue->maxgain = gain;
  }
  else {
    ASSERT(CheckHeap(queue));

    heap = queue->heap;
    locator = queue->locator;

    ASSERT(locator[node] == -1);

    i = queue->nnodes++;
    while (i > 0) {
      j = (i-1)/2;
      if (heap[j].key < gain) {
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else
        break;
    }
    ASSERT(i >= 0);
    heap[i].key = gain;
    heap[i].val = node;
    locator[node] = i;

    ASSERT(CheckHeap(queue));
  }

  return 0;
}


/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
int PQueueDelete(PQueueType *queue, int node, int gain)
{
  int i, j, newgain, oldgain;
  idxtype *locator;
  ListNodeType *newnode, **buckets;
  KeyValueType *heap;

  if (queue->type == 1) {
    ASSERT(gain >= -queue->ngainspan && gain <= queue->pgainspan);
    ASSERT(queue->nnodes > 0);

    buckets = queue->buckets;
    queue->nnodes--;
    newnode = queue->nodes+node;

    /* Remove newnode from the doubly-linked list */
    if (newnode->prev != NULL)
      newnode->prev->next = newnode->next;
    else
      buckets[gain] = newnode->next;
    if (newnode->next != NULL)
      newnode->next->prev = newnode->prev;

    if (buckets[gain] == NULL && gain == queue->maxgain) {
      if (queue->nnodes == 0)
        queue->maxgain = -queue->ngainspan;
      else
        for (; buckets[queue->maxgain]==NULL; queue->maxgain--);
    }
  }
  else { /* Heap Priority Queue */
    heap = queue->heap;
    locator = queue->locator;

    ASSERT(locator[node] != -1);
    ASSERT(heap[locator[node]].val == node);

    ASSERT(CheckHeap(queue));

    i = locator[node];
    locator[node] = -1;

    if (--queue->nnodes > 0 && heap[queue->nnodes].val != node) {
      node = heap[queue->nnodes].val;
      newgain = heap[queue->nnodes].key;
      oldgain = heap[i].key;

      if (oldgain < newgain) { /* Filter-up */
        while (i > 0) {
          j = (i-1)>>1;
          if (heap[j].key < newgain) {
            heap[i] = heap[j];
            locator[heap[i].val] = i;
            i = j;
          }
          else
            break;
        }
      }
      else { /* Filter down */
        while ((j=2*i+1) < queue->nnodes) {
          if (heap[j].key > newgain) {
            if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
              j = j+1;
            heap[i] = heap[j];
            locator[heap[i].val] = i;
            i = j;
          }
          else if (j+1 < queue->nnodes && heap[j+1].key > newgain) {
            j = j+1;
            heap[i] = heap[j];
            locator[heap[i].val] = i;
            i = j;
          }
          else
            break;
        }
      }

      heap[i].key = newgain;
      heap[i].val = node;
      locator[node] = i;
    }

    ASSERT(CheckHeap(queue));
  }

  return 0;
}



/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
int PQueueUpdate(PQueueType *queue, int node, int oldgain, int newgain)
{
  int i, j;
  idxtype *locator;
  ListNodeType *newnode;
  KeyValueType *heap;

  if (oldgain == newgain)
    return 0;

  if (queue->type == 1) {
    /* First delete the node and then insert it */
    PQueueDelete(queue, node, oldgain);
    return PQueueInsert(queue, node, newgain);
  }
  else { /* Heap Priority Queue */
    heap = queue->heap;
    locator = queue->locator;

    ASSERT(locator[node] != -1);
    ASSERT(heap[locator[node]].val == node);
    ASSERT(heap[locator[node]].key == oldgain);
    ASSERT(CheckHeap(queue));

    i = locator[node];

    if (oldgain < newgain) { /* Filter-up */
      while (i > 0) {
        j = (i-1)>>1;
        if (heap[j].key < newgain) {
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else
          break;
      }
    }
    else { /* Filter down */
      while ((j=2*i+1) < queue->nnodes) {
        if (heap[j].key > newgain) {
          if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
            j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else if (j+1 < queue->nnodes && heap[j+1].key > newgain) {
          j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else
          break;
      }
    }

    heap[i].key = newgain;
    heap[i].val = node;
    locator[node] = i;

    ASSERT(CheckHeap(queue));
  }

  return 0;
}



/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
void PQueueUpdateUp(PQueueType *queue, int node, int oldgain, int newgain)
{
  int i, j;
  idxtype *locator;
  ListNodeType *newnode, **buckets;
  KeyValueType *heap;

  if (oldgain == newgain)
    return;

  if (queue->type == 1) {
    ASSERT(oldgain >= -queue->ngainspan && oldgain <= queue->pgainspan);
    ASSERT(newgain >= -queue->ngainspan && newgain <= queue->pgainspan);
    ASSERT(queue->nnodes > 0);

    buckets = queue->buckets;
    newnode = queue->nodes+node;

    /* First delete the node */
    if (newnode->prev != NULL)
      newnode->prev->next = newnode->next;
    else
      buckets[oldgain] = newnode->next;
    if (newnode->next != NULL)
      newnode->next->prev = newnode->prev;

    /* Attach this node in the doubly-linked list */
    newnode->next = buckets[newgain];
    newnode->prev = NULL;
    if (newnode->next != NULL)
      newnode->next->prev = newnode;
    buckets[newgain] = newnode;

    if (queue->maxgain < newgain)
      queue->maxgain = newgain;
  }
  else { /* Heap Priority Queue */
    heap = queue->heap;
    locator = queue->locator;

    ASSERT(locator[node] != -1);
    ASSERT(heap[locator[node]].val == node);
    ASSERT(heap[locator[node]].key == oldgain);
    ASSERT(CheckHeap(queue));


    /* Here we are just filtering up since the newgain is greater than the oldgain */
    i = locator[node];
    while (i > 0) {
      j = (i-1)>>1;
      if (heap[j].key < newgain) {
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else
        break;
    }

    heap[i].key = newgain;
    heap[i].val = node;
    locator[node] = i;

    ASSERT(CheckHeap(queue));
  }

}


/*************************************************************************
* This function returns the vertex with the largest gain from a partition
* and removes the node from the bucket list
**************************************************************************/
int PQueueGetMax(PQueueType *queue)
{
  int vtx, i, j, gain, node;
  idxtype *locator;
  ListNodeType *tptr;
  KeyValueType *heap;

  if (queue->nnodes == 0)
    return -1;

  queue->nnodes--;

  if (queue->type == 1) {
    tptr = queue->buckets[queue->maxgain];
    queue->buckets[queue->maxgain] = tptr->next;
    if (tptr->next != NULL) {
      tptr->next->prev = NULL;
    }
    else {
      if (queue->nnodes == 0) {
        queue->maxgain = -queue->ngainspan;
      }
      else
        for (; queue->buckets[queue->maxgain]==NULL; queue->maxgain--);
    }

    return tptr->id;
  }
  else {
    heap = queue->heap;
    locator = queue->locator;

    vtx = heap[0].val;
    locator[vtx] = -1;

    if ((i = queue->nnodes) > 0) {
      gain = heap[i].key;
      node = heap[i].val;
      i = 0;
      while ((j=2*i+1) < queue->nnodes) {
        if (heap[j].key > gain) {
          if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
            j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else if (j+1 < queue->nnodes && heap[j+1].key > gain) {
          j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else
          break;
      }

      heap[i].key = gain;
      heap[i].val = node;
      locator[node] = i;
    }

    ASSERT(CheckHeap(queue));
    return vtx;
  }
}


/*************************************************************************
* This function returns the vertex with the largest gain from a partition
**************************************************************************/
int PQueueSeeMax(PQueueType *queue)
{
  int vtx;

  if (queue->nnodes == 0)
    return -1;

  if (queue->type == 1)
    vtx = queue->buckets[queue->maxgain]->id;
  else
    vtx = queue->heap[0].val;

  return vtx;
}


/*************************************************************************
* This function returns the vertex with the largest gain from a partition
**************************************************************************/
int PQueueGetKey(PQueueType *queue)
{
  int key;

  if (queue->nnodes == 0)
    return -1;

  if (queue->type == 1)
    key = queue->maxgain;
  else
    key = queue->heap[0].key;

  return key;
}




/*************************************************************************
* This functions checks the consistency of the heap
**************************************************************************/
int CheckHeap(PQueueType *queue)
{
  int i, j, nnodes;
  idxtype *locator;
  KeyValueType *heap;

  heap = queue->heap;
  locator = queue->locator;
  nnodes = queue->nnodes;

  if (nnodes == 0)
    return 1;

  ASSERT(locator[heap[0].val] == 0);
  for (i=1; i<nnodes; i++) {
    ASSERTP(locator[heap[i].val] == i, ("%d %d %d %d\n", nnodes, i, heap[i].val, locator[heap[i].val]));
    ASSERTP(heap[i].key <= heap[(i-1)/2].key, ("%d %d %d %d %d\n", i, (i-1)/2, nnodes, heap[i].key, heap[(i-1)/2].key));
  }
  for (i=1; i<nnodes; i++)
    ASSERT(heap[i].key <= heap[0].key);

  for (j=i=0; i<queue->maxnodes; i++) {
    if (locator[i] != -1)
      j++;
  }
  ASSERTP(j == nnodes, ("%d %d\n", j, nnodes));

  return 1;
}
