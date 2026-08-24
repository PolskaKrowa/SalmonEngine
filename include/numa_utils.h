/*
 * numa_utils.h — NUMA topology detection and thread pinning (Linux only)
 */
#pragma once
#include <stddef.h>
#define _GNU_SOURCE
#include <sched.h>

/* Count the number of NUMA nodes.  Returns 1 if NUMA is not available. */
int numa_num_nodes(void);

/* Read the CPU list for a NUMA node into a cpu_set_t.
 * Returns the number of CPUs in the set, or 0 on failure. */
int numa_node_cpus(int node, cpu_set_t *mask, size_t masksize);

/* Bind the calling thread to a NUMA node.
 * Returns 0 on success, -1 on failure. */
int numa_bind_to_node(int node);

/* Distribute `nthreads` threads across `nnodes` NUMA nodes.
 * Returns the NUMA node index for thread `thread_idx`. */
int numa_distribute_thread(int thread_idx, int nthreads, int nnodes);

/* Multi-threaded TT first-touch.
 * Each worker thread touches its slice of the TT to fault pages onto
 * its local NUMA node. */
void numa_first_touch_tt(int thread_idx, int nthreads,
                          void *tt_base, size_t tt_entries,
                          size_t entry_size);
