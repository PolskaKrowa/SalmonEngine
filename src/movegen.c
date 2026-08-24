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
 *  Pin and checker computation (OPT-PIN)
 *
 *  compute_pinned: find pieces of the side-to-move that are pinned
 *  to their king by an enemy slider.  A piece P on square `p_sq` is
 *  pinned iff:
 *    - There exists an enemy rook/queen on a rook line through ksq
 *      (or bishop/queen on a diagonal line through ksq), AND
 *    - P is the ONLY piece between the slider and the king.
 *
 *  Algorithm (standard "x-ray" pin detection):
 *    1. Find candidate pinners = enemy sliders that, with the king's
 *       square removed from occupancy, attack the king's square.
 *       (Removing the king simulates "what would attack the king if
 *       nothing were in the way" — sliders that *would* attack the
 *       king if every blocker were removed.)
 *    2. For each candidate pinner, the squares between it and the king
 *       (BETWEEN_BB) hold the potential blockers.  If exactly one of
 *       our pieces sits in that range, it's pinned.
 *
 *  Source: Peter Ellis Jones, "Generating Legal Chess Moves Efficiently";
 *  chess.stackexchange.com q25137.
 * ────────────────────────────────────────────── */
Bitboard compute_pinned(const Board *b, Square ksq, Bitboard occ) {
    Color us = b->side;
    Color them = us ^ 1;

    /* OPT-SKIP-SLIDER: if enemy has no orthogonal sliders (no rooks,
     * no queens) and no diagonal sliders (no bishops, no queens), there
     * can be no pins — return 0 without doing any slider lookups.
     * Common in KP endgames and K+minor endgames. */
    Bitboard enemy_rq = b->pieces[them][ROOK]   | b->pieces[them][QUEEN];
    Bitboard enemy_bq = b->pieces[them][BISHOP] | b->pieces[them][QUEEN];
    if (!enemy_rq && !enemy_bq) return 0;

    /* Remove ALL our pieces from occupancy — this exposes enemy sliders
     * that sit behind our pieces on lines to the king.  These are the
     * "candidate pinners" (x-ray attackers).
     *
     * Note: we do NOT remove the king itself.  The king is the origin
     * of the attack computation, not a blocker on a ray.  Removing our
     * pieces lets the slider attacks pass through where our pieces were,
     * reaching enemy sliders that would attack the king if our pieces
     * weren't there. */
    Bitboard occ_no_us = occ & ~b->occ[us];

    /* Candidate pinners: enemy sliders that attack ksq when our pieces
     * are removed from occupancy.  This includes BOTH direct attackers
     * (checkers — 0 of our pieces between) and x-ray attackers
     * (pinners — 1+ of our pieces between).  The "exactly one between"
     * check below filters to just the pinners. */
    Bitboard pinners = 0;
    if (enemy_rq) {
        pinners |= rook_attacks(ksq, occ_no_us) & enemy_rq;
    }
    if (enemy_bq) {
        pinners |= bishop_attacks(ksq, occ_no_us) & enemy_bq;
    }
    if (!pinners) return 0;

    Bitboard pinned  = 0;
    while (pinners) {
        Square pinner_sq = (Square)bb_pop(&pinners);
        /* Squares strictly between pinner and king. */
        Bitboard between = BETWEEN_BB[ksq][pinner_sq];
        /* Our pieces in the between range. */
        Bitboard our_blockers = between & b->occ[us];
        /* Pinned iff exactly one of our pieces is between (0 = direct
         * check, 2+ = double-blocked, neither is a pin). */
        if (our_blockers && !(our_blockers & (our_blockers - 1))) {
            pinned |= our_blockers;
        }
    }
    return pinned;
}

Bitboard compute_checkers(const Board *b, Square ksq, Bitboard occ) {
    Color them = b->side ^ 1;
    Bitboard checkers = 0;

    /* Pawn checks: a pawn of `them` checks `ksq` iff ksq is attacked
     * by a them-pawn.  PAWN_ATTACKS[us][ksq] gives the squares from
     * which an us-pawn would attack ksq — but we want where THEM
     * pawns sit that attack ksq, which is the same set by symmetry
     * (pawn attacks are mutual along diagonals). */
    checkers |= PAWN_ATTACKS[b->side][ksq] & b->pieces[them][PAWN];
    checkers |= KNIGHT_ATTACKS[ksq]       & b->pieces[them][KNIGHT];
    checkers |= KING_ATTACKS[ksq]         & b->pieces[them][KING];

    /* OPT-SKIP-SLIDER: skip slider lookups when no enemy sliders exist. */
    Bitboard enemy_bq = b->pieces[them][BISHOP] | b->pieces[them][QUEEN];
    if (enemy_bq) {
        checkers |= bishop_attacks(ksq, occ) & enemy_bq;
    }
    Bitboard enemy_rq = b->pieces[them][ROOK] | b->pieces[them][QUEEN];
    if (enemy_rq) {
        checkers |= rook_attacks(ksq, occ) & enemy_rq;
    }
    return checkers;
}

/* ──────────────────────────────────────────────
 *  is_legal_fast — pin-aware legality check (OPT-PIN)
 *
 *  Avoids the make/unmake of is_legal() in the common case (not in
 *  check, not a king move, not EP).  Falls back to is_legal() for
 *  the rare/complex cases.
 * ────────────────────────────────────────────── */
bool is_legal_fast(Board *b, Move m, Bitboard pinned, Bitboard checkers,
                   Square ksq) {
    Color us = b->side;
    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    /* Castling: gen_castling already verified path squares are not
     * attacked and the king isn't currently in check.  Accept any
     * castle move the generator emitted. */
    if (mt == MT_KCASTLE || mt == MT_QCASTLE) return true;

    /* King moves (non-castle): test if `to` is attacked by the
     * opponent, with the king removed from occupancy (so x-rays
     * through the king's old square are detected).  For king
     * CAPTURES, also remove the captured piece from occupancy so
     * that x-rays THROUGH the captured piece are detected (e.g.
     * enemy rook behind the captured piece would attack `to`). */
    if (from == ksq) {
        Bitboard occ_test = b->occ[2] ^ SQUARE_BB[ksq];
        /* If `to` has an enemy piece, also remove it (it's being captured). */
        if (b->mailbox[to] != NO_PIECE) {
            occ_test ^= SQUARE_BB[to];
        }
        return !is_square_attacked_with_occ(b, to, us ^ 1, occ_test);
    }

    /* En-passant: rare and tricky (can expose discovered check along
     * the rank of the captured pawn).  Fall back to make/unmake. */
    if (mt == MT_EP) {
        return is_legal(b, m);
    }

    /* In check: only check-evasions are legal.  The evasion rules
     * (block check, capture checker, can't move a non-king piece
     * that's also pinned) are complex enough that the make/unmake
     * fallback is the safe choice. */
    if (checkers) {
        /* Single check: must capture the checker, block it, or move
         * the king.  We could optimize this, but it's a rare case
         * (typically <5% of nodes) and the make/unmake cost is
         * amortized across the few evasion moves. */
        return is_legal(b, m);
    }

    /* Common case: not in check, not a king move, not EP.
     * Legal iff the moving piece is not pinned, OR it moves along
     * the pin ray (i.e., `to` lies on the line through ksq and from). */
    if (!(pinned & SQUARE_BB[from])) {
        return true;  /* not pinned — free to move anywhere */
    }
    /* Pinned: legal only if `to` is on the same line as ksq and from.
     * LINE_BB[ksq][from] is the full line through both squares. */
    return (LINE_BB[ksq][from] & SQUARE_BB[to]) != 0;
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
     * blocker that the EP capture removes).
     *
     * discovered check can ONLY happen when `from` lies on a
     * line (rank/file/diagonal/anti-diagonal) through the enemy king.
     * If `LINE_BB[king_sq][from] == 0`, there is no slider behind `from`
     * that could attack the king through the now-empty `from` — skip
     * BOTH slider lookups.  In typical middlegame positions >85% of moves
     * have `from` off any king-line, saving ~2 slider lookups per move.
     *
     * Furthermore, when `from` IS on a line through king_sq, only ONE
     * slider type can give discovered check (rook for rank/file lines,
     * bishop for diagonal lines).  Test only the relevant type. */
    if (mt == MT_EP) {
        Square cap_sq = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        occ ^= SQUARE_BB[cap_sq];
    }

    Bitboard line = LINE_BB[king_sq][from];
    if (__builtin_expect(line != 0, 0)) {
        Bitboard from_bb = SQUARE_BB[from];
        Bitboard our_queens = b->pieces[us][QUEEN] & ~from_bb;

        /* `from` shares a rank or file with king_sq → rook-type ray.
         * (Queens act as both rooks and bishops; we still only need
         *  the one relevant slider-attack test because a queen on a
         *  rook-ray only discovers via the rook ray.) */
        bool is_rook_ray = ((from & 7) == (king_sq & 7))
                        || ((from >> 3) == (king_sq >> 3));

        if (is_rook_ray) {
            Bitboard our_rooks = b->pieces[us][ROOK] & ~from_bb;
            if (rook_attacks((Square)king_sq, occ) & (our_rooks | our_queens))
                return true;
        } else {
            Bitboard our_bishops = b->pieces[us][BISHOP] & ~from_bb;
            if (bishop_attacks((Square)king_sq, occ) & (our_bishops | our_queens))
                return true;
        }
    }

    /* Special case: EP can also discover via the captured pawn's
     * square.  We already XORed cap_sq out of `occ` above; the
     * LINE_BB check above uses `from`, so it may miss a discovered
     * check revealed by removing `cap_sq`.  For correctness, fall
     * back to the full test when EP could expose a rank discovered
     * check (cap_sq on king's rank).  EP is rare enough that this
     * extra check is negligible. */
    if (mt == MT_EP) {
        /* If cap_sq is on a line with king_sq that wasn't already
         * covered by `from`, do a full slider lookup. */
        Square cap_sq = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        Bitboard cap_line = LINE_BB[king_sq][cap_sq];
        if (cap_line && !(cap_line & line)) {
            /* cap_sq is on a DIFFERENT line than `from` — its removal
             * could reveal a slider not already tested. */
            Bitboard from_bb = SQUARE_BB[from];
            Bitboard our_rooks   = b->pieces[us][ROOK]   & ~from_bb;
            Bitboard our_bishops = b->pieces[us][BISHOP] & ~from_bb;
            Bitboard our_queens  = b->pieces[us][QUEEN]  & ~from_bb;

            if (rook_attacks((Square)king_sq, occ) & (our_rooks | our_queens))
                return true;
            if (bishop_attacks((Square)king_sq, occ) & (our_bishops | our_queens))
                return true;
        } else if (!cap_line) {
            /* cap_sq not on any king-line — no extra discovery possible. */
            /* (Already covered by the `line` test if from was on a line.) */
        }
    }

    return false;
}
