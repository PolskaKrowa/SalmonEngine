#pragma once
#include "types.h"
#include <stddef.h>

/* ──────────────────────────────────────────────
 *  TT entry flags
 * ────────────────────────────────────────────── */
#define TT_NONE  0
#define TT_EXACT 1   /* exact score (PV node)          */
#define TT_LOWER 2   /* lower bound (fail-high / cut)  */
#define TT_UPPER 4   /* upper bound (fail-low / all)   */

/*
 * Packed TT entry — 16 bytes.
 *
 * The previous layout wasted 8 bytes per entry on alignment padding
 * (24-byte actual size despite a "16 bytes" comment).  This layout
 * packs the score into int16_t (max |score| we ever store is
 * MATE_SCORE+MAX_PLY ≈ 31256, well within int16_t range) and folds
 * the generation counter into the unused pad byte.
 *
 * With 16-byte entries and BUCKET_PER_SLOT=2 entries per bucket, each
 * bucket is exactly 32 bytes — half a typical 64-byte cache line — so
 * two buckets fit per cache line.  This is the same memory model used
 * by Stockfish / Ethereal.
 *
 * Replacement policy:
 *   • If the key already exists in the bucket → in-place update.
 *   • Else, prefer replacing the entry with the lowest priority,
 *     where priority = depth + 2*generation_delta (a depth-preferred
 *     scheme with a generation tiebreak).  Always-replace is the
 *     fallback when both slots have similar priority.
 */
typedef struct {
    uint64_t key;        /* full 64-bit Zobrist key for verification */
    int16_t  score;      /* score stored (adjusted for mate distance)  */
    Move     move;       /* best move found at this node (uint16_t)    */
    uint8_t  depth;      /* remaining depth when entry was stored      */
    uint8_t  flag;       /* TT_EXACT / TT_LOWER / TT_UPPER            */
    uint8_t  generation; /* bumped at the start of each search         */
    uint8_t  _pad;       /* explicit pad to reach 16 bytes             */
} TTEntry;               /* sizeof = 16 bytes                          */

/* Number of entries per bucket.  2 is a good default — more entries
 * per bucket improves hit rate but increases probe cost. */
#define TT_BUCKET_SLOTS 2

/* ──────────────────────────────────────────────
 *  API
 * ────────────────────────────────────────────── */

/* Allocate the TT (size_mb megabytes).  Call once at startup. */
void tt_init(size_t size_mb);

/* Free TT memory. */
void tt_free(void);

/* Clear all entries (call on ucinewgame). */
void tt_clear(void);

/* Bump generation counter.  Call at the start of each "go" iteration
 * (or each iteration of iterative deepening, depending on granularity). */
void tt_new_search(void);

/* Probe — returns true and fills *entry if a usable hit is found.
 * On a hit, *entry is the matching slot (not necessarily the first). */
bool tt_probe(uint64_t key, TTEntry *entry);

/* Prefetch the TT bucket for this key into L1.  Call before tt_probe
 * to hide the L2/L3 latency behind other work (in_check, evaluate, etc). */
void tt_prefetch(uint64_t key);

/* Store an entry. */
void tt_store(uint64_t key, int score, Move move, int depth, int flag, int ply);

/* Mate-distance adjustment helpers */
static inline int score_to_tt (int score, int ply) {
    if (score >  MATE_SCORE - MAX_PLY) return score + ply;
    if (score < -MATE_SCORE + MAX_PLY) return score - ply;
    return score;
}
static inline int score_from_tt(int score, int ply) {
    if (score >  MATE_SCORE - MAX_PLY) return score - ply;
    if (score < -MATE_SCORE + MAX_PLY) return score + ply;
    return score;
}
