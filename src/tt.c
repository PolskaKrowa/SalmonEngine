/*
 * tt.c — Transposition table
 *
 * Multi-bucket, depth-preferred-with-generation-tiebreak replacement.
 *
 * Each bucket holds TT_BUCKET_SLOTS (=2) entries.  Probes scan all
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
 * Memory layout: TT is a flat array of TTEntry.  Bucket i occupies
 * slots [i*BUCKET_SLOTS .. i*BUCKET_SLOTS + BUCKET_SLOTS).  Bucket
 * index = (key % num_buckets).
 */

#include "tt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static TTEntry *TT      = NULL;
static size_t   TT_SIZE = 0;          /* number of buckets (power of two) */
static size_t   TT_ENTRIES = 0;       /* TT_SIZE * TT_BUCKET_SLOTS */
static uint64_t TT_MASK = 0;          /* TT_SIZE - 1 */
static uint8_t  TT_GENERATION = 0;    /* bumped by tt_new_search */

/* ──────────────────────────────────────────────
 *  Initialise
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

    TT = (TTEntry *)calloc(TT_ENTRIES, sizeof(TTEntry));
    if (!TT) {
        fprintf(stderr, "info string Failed to allocate TT (%zu MB)\n", size_mb);
        TT_SIZE = 0;
        TT_ENTRIES = 0;
        TT_MASK = 0;
    }
}

void tt_free(void) {
    free(TT);
    TT      = NULL;
    TT_SIZE = 0;
    TT_ENTRIES = 0;
    TT_MASK = 0;
}

void tt_clear(void) {
    if (TT) memset(TT, 0, TT_ENTRIES * sizeof(TTEntry));
}

void tt_new_search(void) {
    /* Wrap around at 250 to leave headroom in the uint8_t — the priority
     * arithmetic uses generation delta, which stays small as long as we
     * wrap cleanly.  250 is well below 256 so arithmetic never overflows. */
    TT_GENERATION = (uint8_t)((TT_GENERATION + 1) % 250);
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
 * priority(entry) — higher = "more valuable to keep".
 *
 *   depth           (0..127)
 *   + 4 if entry is from the current generation
 *   - 1 if entry is empty (TT_NONE) — easy to replace
 *
 * The +4 generation bonus is roughly equivalent to 4 extra plies of
 * depth, ensuring current-search entries survive at the expense of
 * stale entries from previous searches.
 */
static inline int tt_priority(const TTEntry *e) {
    if (e->flag == TT_NONE) return -1;
    int p = (int)e->depth;
    if (e->generation == TT_GENERATION) p += 4;
    return p;
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
 * ────────────────────────────────────────────── */
void tt_store(uint64_t key, int score, Move move, int depth, int flag, int ply) {
    if (!TT) return;
    TTEntry *bucket = tt_bucket(key);

    /* Step 1: if the key already exists in the bucket, update in place. */
    for (int i = 0; i < TT_BUCKET_SLOTS; i++) {
        TTEntry *e = &bucket[i];
        if (e->key == key && e->flag != TT_NONE) {
            /* Update move only if the new one is non-null.  This
             * preserves a useful TT move from a deeper prior search
             * when the current search has no move to store. */
            if (move) e->move = move;
            /* Always overwrite score/depth/flag — the new search is
             * at least as deep as the old one (in normal iterative
             * deepening) and may have a more accurate score. */
            e->score = (int16_t)score_to_tt(score, ply);
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
    e->score = (int16_t)score_to_tt(score, ply);
    e->move  = move;
    e->depth = (uint8_t)depth;
    e->flag  = (uint8_t)flag;
    e->generation = TT_GENERATION;
}
