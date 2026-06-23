#pragma once
#include "board.h"

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
 */
int evaluate(const Board *b);

/*
 * Phase (0 = endgame, 24 = opening/middlegame).
 * Used internally for tapered evaluation.
 */
int game_phase(const Board *b);
