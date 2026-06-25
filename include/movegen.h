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
 *
 * This is the SLOW fallback — does a full make/unmake.  Use
 * is_legal_fast() in hot paths where pinned/checkers/ksq are already
 * computed.
 */
bool is_legal(Board *b, Move m);

/*
 * OPT-PIN: pin-aware fast legality check.
 *
 * Given precomputed `pinned` and `checkers` bitboards and the king
 * square `ksq`, test whether move `m` is legal WITHOUT a make/unmake.
 *
 * Rules:
 *  - King moves: legal iff destination is not attacked by the opponent.
 *    We test this with the king removed from occupancy (so x-rays
 *    through the king's old square are detected).  For castling, the
 *    castling generator already verified the path squares are not
 *    attacked, so we accept any castle move that gen_moves emitted.
 *
 *  - En-passant: special case — the EP capture can expose a discovered
 *    check along the rank of the captured pawn (or, more rarely, along
 *    a diagonal).  We fall back to is_legal() for EP, which is rare
 *    enough that the make/unmake cost is negligible.
 *
 *  - All other moves: legal iff
 *      (a) not in check, AND
 *      (b) the moving piece is not pinned, OR it moves along the pin ray.
 *    When in check (checkers != 0), only check-evasions are legal —
 *    we still need the existing make/unmake fallback because the
 *    evasion rules are complex (block check, capture checker, can't
 *    move a pinned piece even to block).  So when checkers != 0 we
 *    fall back to is_legal().
 *
 * Returns true if legal, false otherwise.
 *
 * The caller MUST compute `pinned` correctly (see compute_pinned()).
 */
bool is_legal_fast(Board *b, Move m, Bitboard pinned, Bitboard checkers,
                   Square ksq);

/*
 * Compute the pinned-pieces bitboard for the side to move.
 *
 * A piece is "pinned" if it sits between an enemy slider and our king
 * on a line, AND it's the only piece on that line.  Moving a pinned
 * piece (except along the pin ray) leaves the king in check.
 *
 * Algorithm:
 *   - Find enemy sliders that *could* attack our king if no pieces
 *     were in between (rook/queen on rook lines, bishop/queen on
 *     diagonals).
 *   - For each such slider, look at the squares between it and the
 *     king (BETWEEN_BB).  If exactly one of OUR pieces is on those
 *     squares, it's pinned.
 *
 * `ksq` is our king square; `occ` is the current total occupancy.
 */
Bitboard compute_pinned(const Board *b, Square ksq, Bitboard occ);

/*
 * Compute the checkers bitboard (enemy pieces giving check to our king).
 *
 * Returns 0 if not in check.
 */
Bitboard compute_checkers(const Board *b, Square ksq, Bitboard occ);

/*
 * move_gives_check — does playing `m` put the opponent in check?
 * Computed from the pre-move board state (no make/unmake needed).
 * Used by the search to avoid the post-make in_check() call.
 */
bool move_gives_check(const Board *b, Move m);
