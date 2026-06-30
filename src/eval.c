/*
 * eval.c — NNUE-based static evaluation
 *
 * All hand-crafted heuristics (PST, mobility, pawn structure, king safety,
 * etc.) are replaced by a forward pass through the global NNUE network
 * g_nnue.
 *
 * Startup sequence (in main / engine_init):
 *   1. bitboard_init();
 *   2. board_init();
 *   3. if (nnue_load(&g_nnue, "nnue.bin") != 0) nnue_init_random(&g_nnue);
 *
 * game_phase() is retained unchanged.  Search components (LMR, futility
 * pruning, endgame scaling) use it as a lightweight game-stage estimate
 * without calling the evaluator.
 *
 * Lazy-evaluation guard: NNUE is fast enough (~1 µs) that the material-
 * proxy shortcut from the old eval is no longer necessary.  Remove it if
 * it was hoisted into the search.
 */

#include "eval.h"
#include "nnue.h"

/* ── Phase constants ─────────────────────────────────────────────── */
/*
 * Phase counts match the old hand-crafted evaluator so that any existing
 * search code that calls game_phase() continues to behave identically.
 */
static const int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };  /* P N B R Q K */
#define PHASE_MAX 24

/* Returns a game-phase value in [0, 24].
 *   24 = opening / middlegame (all pieces on board)
 *    0 = endgame              (kings + pawns only)
 */
int game_phase(const Board *b) {
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_INC[pt] * bb_popcount(b->pieces[c][pt]);
    return phase < PHASE_MAX ? phase : PHASE_MAX;
}

/* ── Main evaluation ─────────────────────────────────────────────── */
/*
 * Returns a score in centipawns from the perspective of the side to move
 * (positive = good for side to move).
 *
 * Uses the incremental NNUE accumulator stack maintained by make_move /
 * unmake_move. This skips the ~60K-FADD feature-transformer refresh that
 * the old nnue_eval() did on every call, leaving only the ~65K-FMA L1
 * matmul (now SIMD-optimised) as the per-eval cost.
 *
 * Falls back to nnue_eval() (full refresh) if the incremental stack is
 * not initialised — e.g. when evaluate() is called from outside search
 * (tuner self-play goes through search, so this is rare).
 */
int evaluate(const Board *b) {
    if (!g_nnue) return 0;
    /* nnue_eval_positional returns 0 if the stack isn't allocated, in
     * which case fall through to the full-refresh path. */
    int v = nnue_eval_positional(b);
    if (v == 0) v = nnue_eval(g_nnue, b);
    return v;
}