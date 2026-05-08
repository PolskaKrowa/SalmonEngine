#pragma once
#include "board.h"

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
