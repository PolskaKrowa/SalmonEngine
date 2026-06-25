/*
 * numa_utils.c — NUMA topology detection and thread pinning (Linux only)
 *
 * OPT-SMP-3: NUMA-aware thread pinning + multi-threaded TT first-touch.
 *
 * On multi-socket systems, the OS scheduler may migrate search threads
 * across sockets, causing cross-socket TT probes (high latency) and
 * cache locality loss.  This module:
 *   1. Detects NUMA topology from /sys/devices/system/node/
 *   2. Provides a function to bind the calling thread to a specific NUMA node
 *   3. Distributes threads across NUMA nodes proportionally
 *
 * On single-socket systems (the common case for consumer hardware),
 * all functions are no-ops — the OS scheduler does fine.
 *
 * Source: Stockfish `numa.h` (Daniel Infuehr); anemato.de/blog/nuca.
 */

#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* Maximum NUMA nodes we support. */
#define MAX_NUMA_NODES 64

/* ──────────────────────────────────────────────
 *  Topology detection
 * ────────────────────────────────────────────── */

/* Count the number of NUMA nodes by reading /sys/devices/system/node/online.
 * Returns 1 if NUMA is not available (single-socket or non-Linux).
 * The format is "0-3" or "0,2,4" etc. */
int numa_num_nodes(void) {
    FILE *f = fopen("/sys/devices/system/node/online", "r");
    if (!f) return 1;
    char buf[256];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 1; }
    fclose(f);

    /* Parse ranges like "0-3" or "0,2,4" or "0-1,3" */
    int max_node = 0;
    char *p = buf;
    while (*p) {
        if (*p >= '0' && *p <= '9') {
            long n = strtol(p, &p, 10);
            if (n > max_node) max_node = (int)n;
            if (*p == '-') {
                long end = strtol(p + 1, &p, 10);
                if (end > max_node) max_node = (int)end;
            }
        } else {
            p++;
        }
    }
    return max_node + 1;
}

/* Read the CPU list for a NUMA node from
 * /sys/devices/system/node/nodeN/cpulist.
 * Fills the cpu_set_t with the CPUs on that node.
 * Returns the number of CPUs in the set, or 0 on failure. */
int numa_node_cpus(int node, cpu_set_t *mask, size_t masksize) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE *f = fopen(path, "r");
    if (!f) { CPU_ZERO_S(masksize, mask); return 0; }
    char buf[512];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); CPU_ZERO_S(masksize, mask); return 0; }
    fclose(f);

    CPU_ZERO_S(masksize, mask);
    int count = 0;
    char *p = buf;
    while (*p) {
        if (*p < '0' || *p > '9') { p++; continue; }
        long a = strtol(p, &p, 10);
        long b = a;
        if (*p == '-') b = strtol(p + 1, &p, 10);
        for (long c = a; c <= b; c++) {
            if (c >= 0 && c < 1024) {
                CPU_SET_S(c, masksize, mask);
                count++;
            }
        }
        if (*p == ',') p++;
    }
    return count;
}

/* ──────────────────────────────────────────────
 *  Thread binding
 * ────────────────────────────────────────────── */

/* Bind the calling thread to the CPUs in the given cpu_set_t.
 * Returns 0 on success, -1 on failure. */
int numa_bind_to_cpuset(cpu_set_t *mask, size_t masksize) {
    return sched_setaffinity(0, masksize, mask);
}

/* Bind the calling thread to NUMA node `node`.
 * Uses the node's CPU list to set the affinity mask.
 * Returns 0 on success, -1 on failure. */
int numa_bind_to_node(int node) {
    int max_cpus = 1024;
    cpu_set_t *mask = CPU_ALLOC(max_cpus);
    if (!mask) return -1;
    size_t masksize = CPU_ALLOC_SIZE(max_cpus);
    int n = numa_node_cpus(node, mask, masksize);
    int rc = -1;
    if (n > 0) {
        rc = numa_bind_to_cpuset(mask, masksize);
    }
    CPU_FREE(mask);
    return rc;
}

/* ──────────────────────────────────────────────
 *  Thread distribution
 * ────────────────────────────────────────────── */

/* Distribute `nthreads` threads across `nnodes` NUMA nodes.
 * Returns the NUMA node index for thread `thread_idx`.
 *
 * Simple round-robin distribution.  A more sophisticated version
 * would read node sizes (cores per node) and distribute proportionally,
 * but round-robin works well for the common case of equal-size nodes.
 *
 * Source: SF `distribute_threads_among_numa_nodes` (numa.h:793-830)
 * uses a greedy fill-ratio approach; this is a simplified version. */
int numa_distribute_thread(int thread_idx, int nthreads, int nnodes) {
    if (nnodes <= 1) return 0;
    return thread_idx % nnodes;
}

/* ──────────────────────────────────────────────
 *  First-touch TT initialization
 * ────────────────────────────────────────────── */

/* OPT-SMP-3b: Multi-threaded TT first-touch.
 *
 * On NUMA systems, the Linux default memory policy is "first-touch":
 * a page is allocated on the NUMA node of the CPU that first writes
 * to it.  If the main thread does calloc()/memset() of the TT, ALL
 * pages land on socket 0 — every probe from socket 1 crosses the
 * interconnect (high latency).
 *
 * The fix: each worker thread touches (writes to) its own slice of
 * the TT, so pages are distributed across nodes.
 *
 * This function is called by each worker thread at search startup.
 * `thread_idx` is the thread's index (0..nthreads-1).
 * `nthreads` is the total number of threads.
 * `tt_base` is the base pointer of the TT array.
 * `tt_entries` is the total number of TT entries.
 * `entry_size` is sizeof(TTEntry).
 *
 * Each thread touches one byte per cache line (64 bytes) in its slice.
 * The writes force page faults on the thread's local NUMA node.
 */
void numa_first_touch_tt(int thread_idx, int nthreads,
                          void *tt_base, size_t tt_entries,
                          size_t entry_size) {
    if (nthreads <= 1 || !tt_base || tt_entries == 0) return;

    /* Each thread touches a contiguous slice.
     * Thread 0 touches [0, stride), thread 1 touches [stride, 2*stride), etc. */
    size_t stride = tt_entries / nthreads;
    size_t start  = stride * thread_idx;
    size_t end    = (thread_idx + 1 == nthreads) ? tt_entries : start + stride;

    /* Touch one byte per cache line (64 bytes) to fault in every page.
     * We write 0 to the flag byte (offset 0 in TTEntry for simplicity —
     * the actual flag field doesn't matter, we just need to write). */
    char *base = (char *)tt_base;
    size_t cache_line_entries = 64 / entry_size;
    if (cache_line_entries < 1) cache_line_entries = 1;

    for (size_t i = start; i < end; i += cache_line_entries) {
        /* Write to the first byte of this entry to force a page fault. */
        base[i * entry_size] = 0;
    }
}
