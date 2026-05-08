/*
 * tt.c — Transposition table
 *
 * The table is a fixed-size hash map with a power-of-two number of buckets.
 * Each bucket holds exactly one TTEntry (direct-mapped / always-replace).
 *
 * Always-replace is simple and works well for fixed-depth iterative deepening.
 * A two-bucket scheme (bucket 0 = depth-preferred, bucket 1 = always-replace)
 * is a straightforward extension left as a future improvement.
 *
 * Collision detection: we store the full 64-bit Zobrist key and compare it
 * on probe, so false positives are vanishingly rare (~2^-64 probability).
 */

#include "tt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static TTEntry *TT      = NULL;
static size_t   TT_SIZE = 0;   /* number of entries (power of two) */
static uint64_t TT_MASK = 0;   /* TT_SIZE - 1 */

/* ──────────────────────────────────────────────
 *  Initialise
 * ────────────────────────────────────────────── */
void tt_init(size_t size_mb) {
    tt_free();

    /* Round size_mb to next lower power of two entries */
    size_t bytes   = size_mb * 1024 * 1024;
    size_t entries = bytes / sizeof(TTEntry);

    /* Force power-of-two */
    size_t pot = 1;
    while (pot * 2 <= entries) pot *= 2;
    TT_SIZE = pot;
    TT_MASK = pot - 1;

    TT = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
    if (!TT) {
        fprintf(stderr, "info string Failed to allocate TT (%zu MB)\n", size_mb);
        TT_SIZE = 0;
        TT_MASK = 0;
    }
}

void tt_free(void) {
    free(TT);
    TT      = NULL;
    TT_SIZE = 0;
    TT_MASK = 0;
}

void tt_clear(void) {
    if (TT) memset(TT, 0, TT_SIZE * sizeof(TTEntry));
}

/* ──────────────────────────────────────────────
 *  Probe
 * ────────────────────────────────────────────── */
bool tt_probe(uint64_t key, TTEntry *out) {
    if (!TT) return false;
    const TTEntry *e = &TT[key & TT_MASK];
    if (e->flag == TT_NONE || e->key != key) return false;
    *out = *e;
    return true;
}

/* ──────────────────────────────────────────────
 *  Store (always-replace)
 * ────────────────────────────────────────────── */
void tt_store(uint64_t key, int score, Move move, int depth, int flag, int ply) {
    if (!TT) return;
    TTEntry *e = &TT[key & TT_MASK];

    /* Preserve a good move from an earlier search if we have no move now */
    Move best_move = move;
    if (!best_move && e->key == key) best_move = e->move;

    e->key   = key;
    e->score = (int32_t)score_to_tt(score, ply);
    e->move  = best_move;
    e->depth = (int8_t)depth;
    e->flag  = (uint8_t)flag;
}
