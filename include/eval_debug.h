/*
 * eval_debug.h — Evaluation stage debug logging
 *
 * Compile with -DEVAL_DEBUG to activate.  In a normal build this header
 * expands to nothing, adding zero overhead to eval or search.
 *
 * Usage:
 *   Compile: gcc ... -DEVAL_DEBUG ...
 *   Run:     Play/analyse normally.
 *   Output:  "eval_debug.txt" is written at the end of every search.
 *            It contains the 20 highest-scored and 20 lowest-scored
 *            positions seen during that search, each with a full
 *            per-stage MG/EG breakdown and an ASCII board.
 *
 * Design notes:
 *   • Scores are always stored from the side-to-move's perspective
 *     (matching what evaluate() returns), so that the top list
 *     represents the "engine thinks it is winning most" positions and
 *     the bottom list represents "engine thinks it is losing most".
 *   • Per-colour stage values are stored from each colour's OWN
 *     perspective (positive = good for that side).  A "Net (tapered)"
 *     column is derived at dump time so the absolute positional impact
 *     of each stage is immediately visible.
 *   • Lazy-eval positions (early return inside evaluate()) are recorded
 *     with a `lazy` flag set and all per-stage fields zeroed — useful
 *     for confirming the lazy threshold is not swallowing interesting
 *     positions.
 */

#ifndef EVAL_DEBUG_H
#define EVAL_DEBUG_H

#ifdef EVAL_DEBUG

#include "eval.h"      /* Board type  */
#include "bitboard.h"  /* Bitboard, bb_pop */
#include <stdbool.h>
#include <stdint.h>

/* Number of positions kept in each list */
#define EVAL_DEBUG_TOP_N    20
#define EVAL_DEBUG_BOTTOM_N 20

/* ─────────────────────────────────────────────────────────────────
 * EvalBreakdown — per-stage MG and EG contributions
 *
 * All per-colour arrays are indexed [0]=WHITE, [1]=BLACK.
 * Each entry is the RAW contribution from that colour's own
 * sub-evaluation (before the sign × ±1 flip that makes Black's
 * contribution negative in the global mg/eg accumulators).
 * ───────────────────────────────────────────────────────────────── */
typedef struct {

    int  phase;         /* game_phase(): 0=full endgame, 24=full midgame */
    int  side;          /* side to move: 0=WHITE, 1=BLACK                */
    bool lazy;          /* true → lazy guard fired, stages below are 0   */

    /* ── Scalar (not per-colour) ────────────────────────────── */
    int  imbalance;     /* material_imbalance(), added to mg only        */
    int  tempo;         /* taper(TEMPO_BONUS_MG, TEMPO_BONUS_EG, phase)  */
    int  initiative;    /* initiative() complexity correction             */

    /* ── Per-colour, per-stage ──────────────────────────────── */
    int  material_mg[2],      material_eg[2];
    int  pawns_mg[2],         pawns_eg[2];
    int  mobility_mg[2],      mobility_eg[2];
    int  rooks_mg[2],         rooks_eg[2];
    int  outposts_mg[2],      outposts_eg[2];
    int  bishop_pair_mg[2],   bishop_pair_eg[2];
    int  weak_queen_mg[2],    weak_queen_eg[2];
    int  king_protector_mg[2],king_protector_eg[2];
    int  minor_behind_mg[2],  minor_behind_eg[2];
    int  king_safety_mg[2],   king_safety_eg[2];
    int  threats_mg[2],       threats_eg[2];

    /* ── Final ──────────────────────────────────────────────── */
    int  final_score;   /* from side-to-move perspective (= evaluate()) */

} EvalBreakdown;

/* ─────────────────────────────────────────────────────────────────
 * Public API
 * ───────────────────────────────────────────────────────────────── */

/* Reset both lists.  Call once at the start of each search. */
void eval_debug_init(void);

/*
 * Record a position + breakdown.  Called from inside evaluate().
 * The position is deep-copied internally.
 */
void eval_debug_record(const Board *b, const EvalBreakdown *bd);

/*
 * Write sorted top/bottom lists to `filename`.
 * Call once at the end of each search (from search()).
 */
void eval_debug_dump(const char *filename);

#endif /* EVAL_DEBUG */
#endif /* EVAL_DEBUG_H */