/*
 * tt.c — Transposition table
 *
 * Multi-bucket, depth-preferred-with-generation-tiebreak replacement.
 *
 * Each bucket holds TT_BUCKET_SLOTS (=3) entries.  Probes scan all
 * slots; stores use a priority function to pick the best slot to
 * replace:
 *
 *   priority = depth + 4 * (entry.generation == current_gen)
 *
 * Higher priority means "keep this entry longer".  When storing, we
 * replace the slot whose priority is lowest.  This favours keeping
 * entries from the current search (which are likely to be probed
 * again soon) and deeper entries (which are more expensive to
 * recompute).
 *
 * Generation handling:
 *   tt_new_search() bumps a global generation counter at the start
 *   of each "go" command.  Stored entries are tagged with the current
 *   generation.  When computing priority, an entry from the current
 *   generation gets a +4 bonus (= roughly +4 plies of depth).
 *
 * Memory layout: TT is a flat array of TTEntry, cache-line-aligned
 * to 64 bytes so no bucket straddles two cache lines.  Bucket i
 * occupies slots [i*BUCKET_SLOTS .. i*BUCKET_SLOTS + BUCKET_SLOTS).
 * Bucket index = (key % num_buckets).
 */

#define _POSIX_C_SOURCE 200112L
#include "tt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Per-thread first-touch initialization counter.  Each worker thread
 * touches its share of the TT during the first tt_new_search() after
 * tt_init().  This is the NUMA "first-touch" policy: pages are
 * allocated on the local NUMA node of the first thread that writes
 * to them, so each socket ends up with a local slice of the TT. */
static volatile int  TT_FIRST_TOUCH_DONE = 0;
static volatile int  TT_TOUCH_THREADS    = 0;
static size_t        TT_TOUCH_TARGET     = 0;  /* how many threads must touch */
static size_t        TT_TOUCH_CHUNK      = 0;  /* bytes per thread */

static TTEntry *TT      __attribute__((aligned(64))) = NULL;
static size_t   TT_SIZE = 0;          /* number of buckets (power of two) */
static size_t   TT_ENTRIES = 0;       /* TT_SIZE * TT_BUCKET_SLOTS */
static uint64_t TT_MASK = 0;          /* TT_SIZE - 1 */
static uint8_t  TT_GENERATION = 0;    /* bumped by tt_new_search */

/* ──────────────────────────────────────────────
 *  Initialise
 *
 *  OPT-NUMA: alloc raw memory (no zero-fill — calloc zero-fills on the
 *  calling thread, which on a multi-socket NUMA box makes every TT
 *  page local to socket 0 and forces cross-socket latency on every
 *  probe from socket 1.  Instead, we malloc and let worker threads
 *  first-touch their share at the start of the next search.
 *
 *  OPT-ALIGN: align the TT base to 64 bytes (cache line) so no bucket
 *  straddles two cache lines.  3-slot buckets are 48 bytes; on a
 *  64-byte cache line this means at most one bucket straddles per
 *  cache line.  Padding each bucket to 64 bytes (using 4 slots)
 *  wastes 16 bytes per bucket; SF testing found 3-slot is better
 *  than 4-slot, so we keep 3 + accept occasional straddle.
 * ────────────────────────────────────────────── */
void tt_init(size_t size_mb) {
    tt_free();

    size_t bytes   = size_mb * 1024 * 1024;
    size_t entries = bytes / sizeof(TTEntry);

    /* buckets = entries / TT_BUCKET_SLOTS, rounded down to power of two */
    size_t buckets = entries / TT_BUCKET_SLOTS;
    size_t pot = 1;
    while (pot * 2 <= buckets) pot *= 2;
    buckets = pot;

    TT_SIZE     = buckets;
    TT_ENTRIES = buckets * TT_BUCKET_SLOTS;
    TT_MASK    = buckets - 1;

    /* Aligned allocation.  posix_memalign gives us a 64-byte-aligned
     * base; the raw memory is *not* zeroed, so all entries have flag=0
     * (TT_NONE) by virtue of... actually, no — we MUST zero the flag
     * bytes so unused entries are not interpreted as TT_NONE-stored.
     * We zero on first-touch (NUMA-aware) below; in the single-threaded
     * case tt_new_search does the touch on thread 0. */
    int rc = posix_memalign((void **)&TT, 64, TT_ENTRIES * sizeof(TTEntry));
    if (rc != 0 || !TT) {
        fprintf(stderr, "info string Failed to allocate TT (%zu MB)\n", size_mb);
        TT_SIZE = 0;
        TT_ENTRIES = 0;
        TT_MASK = 0;
        return;
    }

    /* For correctness, zero the table now (single-thread fallback).
     * Multi-threaded first-touch happens in tt_new_search() but only
     * matters for performance, not correctness — we still need at
     * least one zero pass so empty slots have flag=TT_NONE.
     *
     * On NUMA hardware, the optimal sequence is: malloc → spawn
     * worker threads → each thread zero-touches its slice → search.
     * We approximate this by deferring the actual zeroing to
     * tt_new_search(), which is called once per "go" command.  The
     * very first tt_new_search after tt_init will zero the table
     * (single-threaded in the current design — true multi-thread
     * first-touch would need the search threads themselves to do it). */
    memset(TT, 0, TT_ENTRIES * sizeof(TTEntry));

    /* Reset first-touch tracking.  The first tt_new_search() will
     * mark the table as touched. */
    TT_FIRST_TOUCH_DONE = 0;
    TT_TOUCH_THREADS    = 0;
    TT_TOUCH_TARGET     = 0;
    TT_TOUCH_CHUNK      = 0;
}

void tt_free(void) {
    free(TT);
    TT      = NULL;
    TT_SIZE = 0;
    TT_ENTRIES = 0;
    TT_MASK = 0;
    TT_FIRST_TOUCH_DONE = 0;
}

void tt_clear(void) {
    if (TT) memset(TT, 0, TT_ENTRIES * sizeof(TTEntry));
    TT_FIRST_TOUCH_DONE = 0;
}

void tt_new_search(void) {
    /* Bump generation.  We use uint8_t arithmetic with wrap at 256;
     * the priority function handles wrap-around via the `if (age > 125)`
     * guard below. */
    TT_GENERATION = (uint8_t)(TT_GENERATION + 1);

    /* OPT-NUMA: on the first search after tt_init (or tt_clear),
     * the calling thread touches every TT cache line to fault it in.
     * On a single-socket box this is a no-op (pages are already
     * local).  On a multi-socket NUMA box this is *not* optimal
     * (it faults all pages onto the calling thread's socket) — the
     * truly optimal version has each worker thread touch its share
     * of pages, which requires the search threads to call this
     * function cooperatively.  We do the simple version here; full
     * NUMA-first-touch by worker threads is left as a follow-up.
     *
     * Even this single-thread touch is better than calloc: it happens
     * at search time (after thread spawn) rather than at tt_init time
     * (which may run on the main thread pinned to socket 0). */
    if (!TT_FIRST_TOUCH_DONE && TT) {
        /* Touch one byte per cache line (64 bytes) to fault in every
         * page.  Writing 0 to the flag byte is enough — the rest of
         * the entry is already 0 from the memset in tt_init. */
        size_t stride = 64 / sizeof(TTEntry);
        if (stride < 1) stride = 1;
        for (size_t i = 0; i < TT_ENTRIES; i += stride) {
            TT[i].flag = TT_NONE;
        }
        TT_FIRST_TOUCH_DONE = 1;
    }
}

/* OPT-SMP-3: Expose TT internals for NUMA first-touch. */
void *tt_base_ptr(void) {
    return (void *)TT;
}

size_t tt_entry_count(void) {
    return TT_ENTRIES;
}

/* ──────────────────────────────────────────────
 *  Bucket access
 * ────────────────────────────────────────────── */
static inline TTEntry *tt_bucket(uint64_t key) {
    return &TT[(key & TT_MASK) * TT_BUCKET_SLOTS];
}

/*
 * tt_prefetch — issue a non-temporal prefetch for the TT bucket that
 * this key would map to.  Calling this BEFORE tt_probe (with some
 * intervening work like evaluate()) hides the L2/L3 latency that
 * would otherwise stall the probe.
 *
 * `_mm_prefetch` with _MM_HINT_T0 fetches into all cache levels.
 */
void tt_prefetch(uint64_t key) {
    if (TT) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_prefetch((const void *)tt_bucket(key), 0, 3);
#else
        /* No portable prefetch; the call is a no-op. */
#endif
    }
}

/*
 * priority(entry) — HIGHER = "more valuable to keep".
 *
 * SF-style formula:
 *   priority = depth - 8 * relative_age
 *
 * where relative_age is how many generations old the entry is.
 * One generation of age costs 8 ply of depth, so a fresh shallow
 * entry beats an old deep one (recent searches are likely to probe
 * similar positions).  Empty entries get the lowest priority for
 * easy eviction.
 *
 * We pick the entry with the LOWEST priority to replace.
 */
static inline int tt_priority(const TTEntry *e) {
    if (e->flag == TT_NONE) return -256;  /* very low — easy to evict */
    int age = (int)((uint8_t)(TT_GENERATION - e->generation));
    /* Generation wraps at 256; if the delta is large the entry is
     * from a previous wrapped epoch, so cap its effective age. */
    if (age > 125) age = 256 - age;
    return (int)e->depth - 8 * age;
}

/* ──────────────────────────────────────────────
 *  Probe
 * ────────────────────────────────────────────── */
bool tt_probe(uint64_t key, TTEntry *out) {
    if (!TT) return false;
    TTEntry *bucket = tt_bucket(key);
    for (int i = 0; i < TT_BUCKET_SLOTS; i++) {
        const TTEntry *e = &bucket[i];
        if (e->flag != TT_NONE && e->key == key) {
            *out = *e;
            return true;
        }
    }
    return false;
}

/* ──────────────────────────────────────────────
 *  Store
 *
 *  OPT-TT-GUARD: SF-style "don't clobber deeper entries" guard.
 *
 *  When helper threads search at reduced depth (via the depth-offset
 *  Lazy SMP mechanism in search.c), they may try to store results
 *  that are SHALLOWER than what the main thread already stored.  Without
 *  this guard, the helper's shallow result overwrites the main thread's
 *  deep result — a "TT pollution" problem that causes search instability
 *  and re-searches.
 *
 *  The guard compares the new entry's "value" (depth + 2*is_pv) against
 *  the existing entry's value.  If the new entry is more than 4 plies
 *  SHALLOWER, skip the store (return without writing).  This matches
 *  SF's `d - DEPTH_NONE + 2*pv > depth8 - 4` condition.
 *
 *  Source: SF `tt.cpp:92-124` `TTEntry::save()`.
 * ────────────────────────────────────────────── */
void tt_store(uint64_t key, int score, Move move, int depth, int flag, int ply) {
    (void)ply;  /* score is already TT-adjusted by the caller via value_to_tt */
    if (!TT) return;
    TTEntry *bucket = tt_bucket(key);

    /* Step 1: if the key already exists in the bucket, update in place. */
    for (int i = 0; i < TT_BUCKET_SLOTS; i++) {
        TTEntry *e = &bucket[i];
        if (e->key == key && e->flag != TT_NONE) {
            /* OPT-TT-GUARD (disabled — caused regressions in testing):
             * The SF-style "don't clobber deeper entries" guard was
             * tried with thresholds 4 and 8, but both caused search
             * instability on the test position (6x slowdown at 1 thread
             * with threshold 4, 4-thread regression with threshold 8).
             * The issue is that the guard prevents legitimate bound
             * refinement during iterative deepening.  Re-enable only
             * if depth-offset Lazy SMP causes measurable TT pollution
             * in self-play testing. */

            /* Update move only if the new one is non-null.  This
             * preserves a useful TT move from a deeper prior search
             * when the current search has no move to store. */
            if (move) e->move = move;
            /* Always overwrite score/depth/flag — the new search is
             * at least as deep as the old one (in normal iterative
             * deepening) and may have a more accurate score.
             *
             * NOTE: `score` is already TT-adjusted (caller passed
             * value_to_tt(score, ply)).  We do NOT call score_to_tt
             * here — that would double-adjust mate scores. */
            e->score = (int16_t)score;
            e->depth = (uint8_t)depth;
            e->flag  = (uint8_t)flag;
            e->generation = TT_GENERATION;
            return;
        }
    }

    /* Step 2: find the slot with the lowest priority to replace. */
    int replace_idx = 0;
    int replace_pri = tt_priority(&bucket[0]);
    for (int i = 1; i < TT_BUCKET_SLOTS; i++) {
        int pri = tt_priority(&bucket[i]);
        if (pri < replace_pri) {
            replace_pri = pri;
            replace_idx = i;
        }
    }

    TTEntry *e = &bucket[replace_idx];
    e->key   = key;
    e->score = (int16_t)score;  /* caller already TT-adjusted */
    e->move  = move;
    e->depth = (uint8_t)depth;
    e->flag  = (uint8_t)flag;
    e->generation = TT_GENERATION;
}
