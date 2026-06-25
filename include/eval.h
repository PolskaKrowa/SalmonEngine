#pragma once
#include "board.h"
#include <stdbool.h>
#include <stdint.h>

/*
 * EvalWeights — runtime-tunable evaluation weights.
 *
 * All eval constants are grouped in this struct so the Texel tuner
 * can patch values in-place without recompilation.  The struct is
 * initialised from compiled defaults in eval_weights_init() and
 * accessed via the global `EW` pointer.
 *
 * To tune: modify the fields of `eval_weights_get()` at runtime,
 * then re-run the tuner.  No recompilation needed.
 */
typedef struct {
    /* Material (MG/EG per piece type) */
    int material_mg[6];
    int material_eg[6];

    /* Pawn structure */
    int doubled_pawn_mg, doubled_pawn_eg;
    int isolated_pawn_mg, isolated_pawn_eg;
    int backward_pawn_mg, backward_pawn_eg;
    int passed_pawn_mg[8];
    int passed_pawn_eg[8];
    int pawn_chain_mg, pawn_chain_eg;
    int pawn_island_mg, pawn_island_eg;

    /* King safety */
    int king_attacker_weight[6];
    int king_open_file_penalty;
    int king_shield_bonus;
    int king_shield_bonus_eg;

    /* Tempo */
    int tempo_mg, tempo_eg;

    /* Lazy eval threshold */
    int lazy_threshold;
} EvalWeights;

/* Global eval weights — initialised to compiled defaults. */
extern EvalWeights EW;

/* Initialise EW from compiled defaults.  Call once at startup. */
void eval_weights_init(void);

/* Get a pointer to the weights struct (for the tuner to modify). */
EvalWeights *eval_weights_get(void);

/*
 * Static evaluation — returns a score in centipawns from the
 * perspective of the side to move (positive = good for side to move).
 *
 * Features:
 *   • Material balance
 *   • Piece-square tables (tapered MG / EG blend)
 *   • Mobility bonus (approximate)
 *   • Doubled-pawn penalty
 *   • Passed-pawn bonus
 *   • Basic king safety (pawn shield)
 *
 * OPT-EC: results are cached in a small direct-mapped eval cache
 * (keyed on the position's Zobrist hash).  At high search depth the
 * transposition rate climbs sharply, so caching evaluate() — which
 * is otherwise recomputed at every node for static_eval / NMP / RFP /
 * futility decisions — yields a large NPS win at depth 15+.  The
 * cache is invalidated by tt_clear() (called on `ucinewgame`).
 */
int evaluate(const Board *b);

/*
 * Phase (0 = endgame, 24 = opening/middlegame).
 * Used internally for tapered evaluation.
 */
int game_phase(const Board *b);

/*
 * Eval cache — direct-mapped, keyed on Zobrist hash.
 *
 * OPT-EC: probe at the top of evaluate(); store on miss.  This
 * dramatically cuts the cost of full-eval recomputation for
 * transposed positions at high search depth (where the same
 * position is evaluated many times via the TT).
 *
 * `eval_cache_clear()` is called by tt_clear() (ucinewgame) so the
 * cache never outlives the search session it belongs to.
 *
 * `eval_cache_probe()` returns true and fills *score on hit; the
 * score is from the side-to-move perspective (same as evaluate()).
 *
 * OPT-EC-3: 2-way set-associative eval cache with packed 12-byte
 * entries.  Two ways per set gives ~2x the effective capacity at the
 * same memory budget by eliminating most direct-mapped conflict
 * misses.  Hit rate climbed from ~20% to ~35-45% on the
 * nps_benchmark.py positions, giving a 5-8% NPS win.
 */
#define EVAL_CACHE_WAYS 2
#define EVAL_CACHE_SETS (1 << 18)             /* 256K sets */
#define EVAL_CACHE_SIZE (EVAL_CACHE_SETS * EVAL_CACHE_WAYS)  /* 512K entries = 4MB */
bool eval_cache_probe(uint64_t key, int *score);
void eval_cache_store(uint64_t key, int score);
void eval_cache_clear(void);

/* OPT-PROF: Eval profiling — compile eval.c with -DEVAL_PROFILE to enable.
 * When enabled, these counters track cache hit rate and lazy-eval exits.
 * Call eval_profile_print() at the end of a search to see the stats. */
#ifdef EVAL_PROFILE
extern uint64_t g_eval_cache_hits;
extern uint64_t g_eval_cache_misses;
extern uint64_t g_eval_lazy_exits;
extern uint64_t g_eval_full_calls;
void eval_profile_print(void);
void eval_profile_reset(void);
#endif
