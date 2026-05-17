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

/* ──────────────────────────────────────────────
 *  Transposition table entry  (16 bytes)
 * ────────────────────────────────────────────── */
typedef struct {
    uint64_t key;     /* full 64-bit Zobrist key for verification */
    int32_t  score;   /* score stored (adjusted for mate distance)  */
    Move     move;    /* best move found at this node               */
    int8_t   depth;   /* remaining depth when entry was stored      */
    uint8_t  flag;    /* TT_EXACT / TT_LOWER / TT_UPPER            */
    uint8_t  _pad[2];
} TTEntry;            /* sizeof = 16 bytes                          */

/* ──────────────────────────────────────────────
 *  API
 * ────────────────────────────────────────────── */

/* Allocate the TT (size_mb megabytes).  Call once at startup. */
void tt_init(size_t size_mb);

/* Free TT memory. */
void tt_free(void);

/* Clear all entries (call on ucinewgame). */
void tt_clear(void);

/* Probe — returns true and fills *entry if a usable hit is found. */
bool tt_probe(uint64_t key, TTEntry *entry);

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
