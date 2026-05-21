/*
 * movegen.c — Pseudo-legal move generation  (optimised)
 *
 * All generated moves are pseudo-legal (they may leave the king in check).
 * The search calls is_legal() before executing a move, or filters after
 * make_move by testing in_check().
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
    add_move(ml, MAKE_MOVE(from, to, base));        /* knight */
    add_move(ml, MAKE_MOVE(from, to, base + 1));    /* bishop */
    add_move(ml, MAKE_MOVE(from, to, base + 2));    /* rook   */
    add_move(ml, MAKE_MOVE(from, to, base + 3));    /* queen  */
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

    const int push_dir  = (us == WHITE) ?  8 : -8;
    const int dpush_dir = (us == WHITE) ? 16 : -16;
    const Bitboard promo_rank  = (us == WHITE) ? RANK_BB[7] : RANK_BB[0];
    const Bitboard start_rank  = (us == WHITE) ? RANK_BB[1] : RANK_BB[6];

    if (!captures_only) {
        /* Single pushes */
        Bitboard single = (us == WHITE) ? (pawns << 8) : (pawns >> 8);
        single &= empty;

        Bitboard promo  = single & promo_rank;
        Bitboard normal = single & ~promo_rank;

        while (normal) {
            int to = bb_pop(&normal);
            add_move(ml, MAKE_MOVE((Square)(to - push_dir), (Square)to, MT_QUIET));
        }
        while (promo) {
            int to = bb_pop(&promo);
            add_promo_moves(ml, (Square)(to - push_dir), (Square)to, false);
        }

        /* Double pushes */
        Bitboard dp1 = (us == WHITE) ? ((pawns & start_rank) << 8)
                                     : ((pawns & start_rank) >> 8);
        dp1 &= empty;
        Bitboard dp2 = (us == WHITE) ? (dp1 << 8) : (dp1 >> 8);
        dp2 &= empty;
        while (dp2) {
            int to = bb_pop(&dp2);
            add_move(ml, MAKE_MOVE((Square)(to - dpush_dir), (Square)to, MT_DPUSH));
        }
    }

    {
        /* NW/SE direction (toward A-file) */
        Bitboard atk_left = (us == WHITE)
            ? ((pawns & ~FILE_BB[0]) << 7)
            : ((pawns & ~FILE_BB[7]) >> 7);
        atk_left &= enemies;

        Bitboard promo_l  = atk_left &  promo_rank;
        Bitboard normal_l = atk_left & ~promo_rank;
        int left_offset   = (us == WHITE) ? -7 : 7;  /* to - offset = from */

        while (normal_l) {
            int to = bb_pop(&normal_l);
            add_move(ml, MAKE_MOVE((Square)(to + left_offset), (Square)to, MT_CAPTURE));
        }
        while (promo_l) {
            int to = bb_pop(&promo_l);
            add_promo_moves(ml, (Square)(to + left_offset), (Square)to, true);
        }

        /* NE/SW direction (toward H-file) */
        Bitboard atk_right = (us == WHITE)
            ? ((pawns & ~FILE_BB[7]) << 9)
            : ((pawns & ~FILE_BB[0]) >> 9);
        atk_right &= enemies;

        Bitboard promo_r  = atk_right &  promo_rank;
        Bitboard normal_r = atk_right & ~promo_rank;
        int right_offset  = (us == WHITE) ? -9 : 9;

        while (normal_r) {
            int to = bb_pop(&normal_r);
            add_move(ml, MAKE_MOVE((Square)(to + right_offset), (Square)to, MT_CAPTURE));
        }
        while (promo_r) {
            int to = bb_pop(&promo_r);
            add_promo_moves(ml, (Square)(to + right_offset), (Square)to, true);
        }
    }

    /* En-passant — still per-pawn; at most two pawns can attack the ep square */
    if (b->ep_sq != NO_SQ) {
        Bitboard ep_attackers = PAWN_ATTACKS[them][b->ep_sq] & pawns;
        while (ep_attackers) {
            int from = bb_pop(&ep_attackers);
            add_move(ml, MAKE_MOVE((Square)from, (Square)b->ep_sq, MT_EP));
        }
    }
}

/* Leapers: pass the precomputed attack table for this piece type */
static void gen_leaper_moves(const Board *b, MoveList *ml,
                              const Bitboard atk_table[64], Bitboard targets) {
    Bitboard pieces = b->pieces[b->side][
        /* infer piece type from table pointer */
        (atk_table == KNIGHT_ATTACKS) ? KNIGHT : KING
    ];
    while (pieces) {
        int from = bb_pop(&pieces);
        Bitboard atk = atk_table[from] & targets;
        while (atk) {
            int to = bb_pop(&atk);
            MoveType mt = (b->mailbox[to] != NO_PIECE) ? MT_CAPTURE : MT_QUIET;
            add_move(ml, MAKE_MOVE((Square)from, (Square)to, mt));
        }
    }
}

typedef Bitboard (*AtkFn)(Square, Bitboard);

static void gen_slider_moves(const Board *b, MoveList *ml,
                              PieceType pt, AtkFn atk_fn,
                              Bitboard occ, Bitboard targets) {
    Bitboard pieces = b->pieces[b->side][pt];
    while (pieces) {
        int from = bb_pop(&pieces);
        Bitboard atk = atk_fn((Square)from, occ) & targets;
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
    Color    us  = b->side;
    Bitboard occ = b->occ[2];

    if (us == WHITE) {
        if ((b->castle & CR_WK) &&
            !(occ & (SQUARE_BB[F1] | SQUARE_BB[G1])) &&
            !is_square_attacked(b, E1, BLACK) &&
            !is_square_attacked(b, F1, BLACK) &&
            !is_square_attacked(b, G1, BLACK))
            add_move(ml, MAKE_MOVE(E1, G1, MT_KCASTLE));

        if ((b->castle & CR_WQ) &&
            !(occ & (SQUARE_BB[D1] | SQUARE_BB[C1] | SQUARE_BB[B1])) &&
            !is_square_attacked(b, E1, BLACK) &&
            !is_square_attacked(b, D1, BLACK) &&
            !is_square_attacked(b, C1, BLACK))
            add_move(ml, MAKE_MOVE(E1, C1, MT_QCASTLE));
    } else {
        if ((b->castle & CR_BK) &&
            !(occ & (SQUARE_BB[F8] | SQUARE_BB[G8])) &&
            !is_square_attacked(b, E8, WHITE) &&
            !is_square_attacked(b, F8, WHITE) &&
            !is_square_attacked(b, G8, WHITE))
            add_move(ml, MAKE_MOVE(E8, G8, MT_KCASTLE));

        if ((b->castle & CR_BQ) &&
            !(occ & (SQUARE_BB[D8] | SQUARE_BB[C8] | SQUARE_BB[B8])) &&
            !is_square_attacked(b, E8, WHITE) &&
            !is_square_attacked(b, D8, WHITE) &&
            !is_square_attacked(b, C8, WHITE))
            add_move(ml, MAKE_MOVE(E8, C8, MT_QCASTLE));
    }
}

/* ──────────────────────────────────────────────
 *  Public interfaces
 * ────────────────────────────────────────────── */
void gen_moves(const Board *b, MoveList *ml) {
    ml->count = 0;
    Color    us      = b->side;
    Bitboard occ     = b->occ[2];
    Bitboard targets = ~b->occ[us];

    gen_pawn_moves(b, ml, false);
    gen_leaper_moves(b, ml, KNIGHT_ATTACKS, targets);
    gen_slider_moves(b, ml, BISHOP, bishop_attacks, occ, targets);
    gen_slider_moves(b, ml, ROOK,   rook_attacks,   occ, targets);
    gen_slider_moves(b, ml, QUEEN,  queen_attacks,  occ, targets);
    gen_leaper_moves(b, ml, KING_ATTACKS, targets);
    gen_castling(b, ml);
}

void gen_captures(const Board *b, MoveList *ml) {
    ml->count = 0;
    Color    us      = b->side;
    Color    them    = us ^ 1;
    Bitboard occ     = b->occ[2];
    Bitboard targets = b->occ[them];

    gen_pawn_moves(b, ml, true);
    gen_leaper_moves(b, ml, KNIGHT_ATTACKS, targets);
    gen_slider_moves(b, ml, BISHOP, bishop_attacks, occ, targets);
    gen_slider_moves(b, ml, ROOK,   rook_attacks,   occ, targets);
    gen_slider_moves(b, ml, QUEEN,  queen_attacks,  occ, targets);
    gen_leaper_moves(b, ml, KING_ATTACKS, targets);
    /* No castling in captures */
}

bool is_legal(Board *b, Move m) {
    make_move(b, m);
    Color mover  = b->side ^ 1;
    int   king_sq = bb_lsb(b->pieces[mover][KING]);
    bool  legal  = !is_square_attacked(b, (Square)king_sq, b->side);
    unmake_move(b);
    return legal;
}