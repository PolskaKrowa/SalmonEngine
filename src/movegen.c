/*
 * movegen.c — Pseudo-legal move generation
 *
 * All generated moves are pseudo-legal (they may leave the king in check).
 * The search calls is_legal() before executing a move, or filters after
 * make_move by testing in_check().
 *
 * Generation order (roughly MVV-LVA friendly for captures):
 *   1. Pawn moves (pushes, double-pushes, promotions, en-passant)
 *   2. Knight moves
 *   3. Bishop moves
 *   4. Rook moves
 *   5. Queen moves
 *   6. King moves (including castling)
 */

#include "movegen.h"
#include "board.h"
#include <string.h>

/* ──────────────────────────────────────────────
 *  MoveList helpers
 * ────────────────────────────────────────────── */
static inline void add_move(MoveList *ml, Move m) {
    ml->moves[ml->count++] = m;
}

static inline void add_promo_moves(MoveList *ml, Square from, Square to, bool capture) {
    MoveType base = capture ? MT_N_PROMO_CAP : MT_N_PROMO;
    add_move(ml, MAKE_MOVE(from, to, base));           /* knight */
    add_move(ml, MAKE_MOVE(from, to, base + 1));       /* bishop */
    add_move(ml, MAKE_MOVE(from, to, base + 2));       /* rook   */
    add_move(ml, MAKE_MOVE(from, to, base + 3));       /* queen  */
}

/* ──────────────────────────────────────────────
 *  Pawn moves
 * ────────────────────────────────────────────── */
static void gen_pawn_moves(const Board *b, MoveList *ml, bool captures_only) {
    Color us   = b->side;
    Color them = us ^ 1;

    Bitboard pawns   = b->pieces[us][PAWN];
    Bitboard empty   = ~b->occ[2];
    Bitboard enemies = b->occ[them];

    Bitboard start_rank = (us == WHITE) ? RANK_BB[1] : RANK_BB[6];
    int push_dir   = (us == WHITE) ?  8 : -8;
    int dpush_dir  = (us == WHITE) ? 16 : -16;

    if (!captures_only) {
        /* Single pushes */
        Bitboard single = (us == WHITE) ? (pawns << 8) : (pawns >> 8);
        single &= empty;

        /* Promotions from single push */
        Bitboard promo = single & ((us == WHITE) ? RANK_BB[7] : RANK_BB[0]);
        Bitboard normal = single & ~promo;

        while (normal) {
            int to = bb_pop(&normal);
            add_move(ml, MAKE_MOVE((Square)(to - push_dir), (Square)to, MT_QUIET));
        }
        while (promo) {
            int to = bb_pop(&promo);
            add_promo_moves(ml, (Square)(to - push_dir), (Square)to, false);
        }

        /* Double pushes (only from start rank) */
        Bitboard dpush_pawns = pawns & start_rank;
        Bitboard dpush1 = (us == WHITE) ? (dpush_pawns << 8) : (dpush_pawns >> 8);
        dpush1 &= empty;
        Bitboard dpush2 = (us == WHITE) ? (dpush1 << 8) : (dpush1 >> 8);
        dpush2 &= empty;
        while (dpush2) {
            int to = bb_pop(&dpush2);
            add_move(ml, MAKE_MOVE((Square)(to - dpush_dir), (Square)to, MT_DPUSH));
        }
    }

    /* Captures (including promotion-captures and en-passant) */
    while (pawns) {
        int from = bb_pop(&pawns);
        Bitboard atk = PAWN_ATTACKS[us][from] & enemies;

        /* Promotion captures */
        Bitboard promo_cap = atk & ((us == WHITE) ? RANK_BB[7] : RANK_BB[0]);
        Bitboard normal_cap = atk & ~promo_cap;

        while (normal_cap) {
            int to = bb_pop(&normal_cap);
            add_move(ml, MAKE_MOVE((Square)from, (Square)to, MT_CAPTURE));
        }
        while (promo_cap) {
            int to = bb_pop(&promo_cap);
            add_promo_moves(ml, (Square)from, (Square)to, true);
        }

        /* En-passant */
        if (b->ep_sq != NO_SQ &&
            (PAWN_ATTACKS[us][from] & SQUARE_BB[b->ep_sq])) {
            add_move(ml, MAKE_MOVE((Square)from, (Square)b->ep_sq, MT_EP));
        }
    }
}

/* ──────────────────────────────────────────────
 *  Generic piece move generator
 * ────────────────────────────────────────────── */
static void gen_piece_moves(const Board *b, MoveList *ml,
                             PieceType pt, Bitboard targets) {
    Color us = b->side;
    Bitboard pieces = b->pieces[us][pt];
    while (pieces) {
        int from = bb_pop(&pieces);
        Bitboard atk;
        switch (pt) {
            case KNIGHT: atk = KNIGHT_ATTACKS[from]; break;
            case BISHOP: atk = bishop_attacks((Square)from, b->occ[2]); break;
            case ROOK:   atk = rook_attacks  ((Square)from, b->occ[2]); break;
            case QUEEN:  atk = queen_attacks  ((Square)from, b->occ[2]); break;
            case KING:   atk = KING_ATTACKS[from]; break;
            default:     atk = 0; break;
        }
        atk &= targets;
        while (atk) {
            int to = bb_pop(&atk);
            MoveType mt = (b->mailbox[to] != NO_PIECE) ? MT_CAPTURE : MT_QUIET;
            add_move(ml, MAKE_MOVE((Square)from, (Square)to, mt));
        }
    }
}

/* ──────────────────────────────────────────────
 *  Castling
 * ────────────────────────────────────────────── */
static void gen_castling(const Board *b, MoveList *ml) {
    Color us = b->side;
    Bitboard occ = b->occ[2];

    if (us == WHITE) {
        /* King-side */
        if ((b->castle & CR_WK) &&
            !(occ & (SQUARE_BB[F1] | SQUARE_BB[G1])) &&
            !is_square_attacked(b, E1, BLACK) &&
            !is_square_attacked(b, F1, BLACK) &&
            !is_square_attacked(b, G1, BLACK)) {
            add_move(ml, MAKE_MOVE(E1, G1, MT_KCASTLE));
        }
        /* Queen-side */
        if ((b->castle & CR_WQ) &&
            !(occ & (SQUARE_BB[D1] | SQUARE_BB[C1] | SQUARE_BB[B1])) &&
            !is_square_attacked(b, E1, BLACK) &&
            !is_square_attacked(b, D1, BLACK) &&
            !is_square_attacked(b, C1, BLACK)) {
            add_move(ml, MAKE_MOVE(E1, C1, MT_QCASTLE));
        }
    } else {
        if ((b->castle & CR_BK) &&
            !(occ & (SQUARE_BB[F8] | SQUARE_BB[G8])) &&
            !is_square_attacked(b, E8, WHITE) &&
            !is_square_attacked(b, F8, WHITE) &&
            !is_square_attacked(b, G8, WHITE)) {
            add_move(ml, MAKE_MOVE(E8, G8, MT_KCASTLE));
        }
        if ((b->castle & CR_BQ) &&
            !(occ & (SQUARE_BB[D8] | SQUARE_BB[C8] | SQUARE_BB[B8])) &&
            !is_square_attacked(b, E8, WHITE) &&
            !is_square_attacked(b, D8, WHITE) &&
            !is_square_attacked(b, C8, WHITE)) {
            add_move(ml, MAKE_MOVE(E8, C8, MT_QCASTLE));
        }
    }
}

/* ──────────────────────────────────────────────
 *  Public interfaces
 * ────────────────────────────────────────────── */
void gen_moves(const Board *b, MoveList *ml) {
    ml->count = 0;
    Color us = b->side;
    Bitboard targets = ~b->occ[us]; /* can't capture own pieces */

    gen_pawn_moves(b, ml, false);
    gen_piece_moves(b, ml, KNIGHT, targets);
    gen_piece_moves(b, ml, BISHOP, targets);
    gen_piece_moves(b, ml, ROOK,   targets);
    gen_piece_moves(b, ml, QUEEN,  targets);
    gen_piece_moves(b, ml, KING,   targets);
    gen_castling(b, ml);
}

void gen_captures(const Board *b, MoveList *ml) {
    ml->count = 0;
    Color us   = b->side;
    Color them = us ^ 1;
    Bitboard targets = b->occ[them]; /* captures only */

    gen_pawn_moves(b, ml, true);
    gen_piece_moves(b, ml, KNIGHT, targets);
    gen_piece_moves(b, ml, BISHOP, targets);
    gen_piece_moves(b, ml, ROOK,   targets);
    gen_piece_moves(b, ml, QUEEN,  targets);
    gen_piece_moves(b, ml, KING,   targets);
    /* No castling in captures */
}

bool is_legal(Board *b, Move m) {
    make_move(b, m);
    Color mover = b->side ^ 1; /* the side that just moved */
    int king_sq = bb_lsb(b->pieces[mover][KING]);
    bool legal = !is_square_attacked(b, (Square)king_sq, b->side);
    unmake_move(b);
    return legal;
}

/* ──────────────────────────────────────────────
 *  move_gives_check  (OPT-F)
 *
 *  Determine whether playing `m` puts the side-to-move's opponent in
 *  check, WITHOUT calling make_move / in_check.  Computing this from
 *  the pre-move board state is much cheaper than the post-make
 *  in_check() call (which re-derives the king square and runs 5+
 *  slider attack lookups), and we already know the from/to/piece
 *  from the move encoding.
 *
 *  Three sources of check:
 *    1. Direct check   — the moving piece, from its destination, attacks
 *       the enemy king.  (For promotions, use the promoted piece type.)
 *       (For castling, the rook gives the check, not the king.)
 *    2. Discovered check — a slider behind the mover (on a line through
 *       `from` and the enemy king) now sees the king after `from` is
 *       vacated.  We test this by computing slider attacks from `from`
 *       with `from` itself removed from occupancy, intersected with
 *       our sliders (rook/queen on lines, bishop/queen on diagonals).
 *    3. En-passant discovered check — when an EP capture removes a
 *       pawn, that pawn might have been blocking a slider.  Test by
 *       also XOR-ing out the captured pawn's square before computing
 *       discovered attacks.
 *
 *  The discovered-check test uses occupancy with `from` removed; for
 *  EP we also remove the captured pawn square.  This is the standard
 *  Stockfish approach.
 * ────────────────────────────────────────────── */
bool move_gives_check(const Board *b, Move m) {
    Color  us   = b->side;
    Color  them = us ^ 1;
    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    int king_sq = bb_lsb(b->pieces[them][KING]);

    /* Occupancy after the move: piece removed from `from`, placed at `to`.
     * For captures the captured piece is also removed, but for discovered-
     * check purposes that doesn't matter (the captured piece was on `to`,
     * not behind the mover).  For EP we need to also remove the captured
     * pawn (handled below). */
    Bitboard occ = b->occ[2];
    occ ^= SQUARE_BB[from];
    occ |= SQUARE_BB[to];

    /* ── Direct check ── */
    switch (mt) {
        case MT_N_PROMO:
        case MT_N_PROMO_CAP:
            if (KNIGHT_ATTACKS[to] & SQUARE_BB[king_sq]) return true;
            break;
        case MT_B_PROMO:
        case MT_B_PROMO_CAP:
            if (bishop_attacks((Square)to, occ) & SQUARE_BB[king_sq]) return true;
            break;
        case MT_R_PROMO:
        case MT_R_PROMO_CAP:
            if (rook_attacks((Square)to, occ) & SQUARE_BB[king_sq]) return true;
            break;
        case MT_Q_PROMO:
        case MT_Q_PROMO_CAP:
            if ( (bishop_attacks((Square)to, occ) | rook_attacks((Square)to, occ))
                 & SQUARE_BB[king_sq]) return true;
            break;
        case MT_KCASTLE:
        case MT_QCASTLE: {
            /* The rook is the piece that can give check after castling. */
            Square rook_to = (mt == MT_KCASTLE)
                ? (us == WHITE ? F1 : F8)
                : (us == WHITE ? D1 : D8);
            if (rook_attacks(rook_to, occ) & SQUARE_BB[king_sq]) return true;
            /* King itself cannot give check from its castled square
             * (opponent's king is never adjacent in a legal position
             * before the castle).  Skip direct king check. */
            break;
        }
        default: {
            /* Normal move: use the moving piece's type. */
            PieceType pt = piece_type(b->mailbox[from]);
            Bitboard atk = 0;
            switch (pt) {
                case PAWN:
                    atk = PAWN_ATTACKS[us][to];
                    break;
                case KNIGHT:
                    atk = KNIGHT_ATTACKS[to];
                    break;
                case BISHOP:
                    atk = bishop_attacks((Square)to, occ);
                    break;
                case ROOK:
                    atk = rook_attacks((Square)to, occ);
                    break;
                case QUEEN:
                    atk = bishop_attacks((Square)to, occ)
                        | rook_attacks((Square)to, occ);
                    break;
                case KING:
                    /* A king move can only give check if it's a discovered
                     * check — kings don't deliver direct check. */
                    break;
                default: break;
            }
            if (atk & SQUARE_BB[king_sq]) return true;
            break;
        }
    }

    /* ── Discovered check ──
     *
     * A slider of ours that was behind `from` (on a line through the
     * enemy king) now sees the king.  The correct test is:
     *   rook_attacks(king_sq, occ_with_from_removed) & our_rooks
     * which asks "with `from` vacated, which of our rooks attack the
     * king?"  Symmetry of rook attacks on a line means this is the
     * same as asking which of our rooks the king would attack.
     *
     * IMPORTANT: the moving piece is still recorded at `from` in
     * b->pieces[us][...] (we have NOT called make_move yet).  When
     * the mover is itself a rook/bishop/queen, it would create a
     * false positive on the discovered-check test (the slider at
     * `from` "sees" the king through the now-empty `from` square,
     * but the slider has actually moved to `to`).  Mask `from` out
     * of the candidate slider sets to avoid this.
     *
     * For EP, we must also XOR out the captured pawn square (it was a
     * blocker that the EP capture removes). */
    if (mt == MT_EP) {
        Square cap_sq = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        occ ^= SQUARE_BB[cap_sq];
    }

    Bitboard from_bb = SQUARE_BB[from];
    Bitboard our_rooks   = b->pieces[us][ROOK]   & ~from_bb;
    Bitboard our_bishops = b->pieces[us][BISHOP] & ~from_bb;
    Bitboard our_queens  = b->pieces[us][QUEEN]  & ~from_bb;

    /* Rook/queen discovered check. */
    Bitboard rook_dc = rook_attacks((Square)king_sq, occ)
                     & (our_rooks | our_queens);
    if (rook_dc) return true;

    /* Bishop/queen discovered check. */
    Bitboard bishop_dc = bishop_attacks((Square)king_sq, occ)
                       & (our_bishops | our_queens);
    if (bishop_dc) return true;

    return false;
}
