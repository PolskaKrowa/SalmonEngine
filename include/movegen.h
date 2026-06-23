#pragma once
#include "types.h"
#include "board.h"

/*
 * Generate all pseudo-legal moves for the side to move.
 * The caller must filter illegal moves (those that leave the king in check)
 * before executing them.
 */
void gen_moves(const Board *b, MoveList *ml);

/*
 * Generate only captures (+ queen promotions) — used in quiescence search.
 */
void gen_captures(const Board *b, MoveList *ml);

/*
 * Validate legality: returns true if the move leaves the king safe.
 */
bool is_legal(Board *b, Move m);
