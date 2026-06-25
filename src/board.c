/*
 * board.c — Board state management
 *
 * Zobrist keys are generated using xorshift64* for a uniform, fast PRNG.
 * make_move / unmake_move maintain all bitboards, the mailbox, occupancy,
 * and the Zobrist hash incrementally.
 */

#include "board.h"
#include "movegen.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

/* ──────────────────────────────────────────────
 *  Zobrist key initialisation
 * ────────────────────────────────────────────── */
ZobristKeys ZKEYS;

static uint64_t xorshift64(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

void board_init(void) {
    uint64_t seed = 0xDEADBEEFCAFEBABEULL;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            for (int sq = 0; sq < 64; sq++)
                ZKEYS.piece[c][pt][sq] = xorshift64(&seed);

    ZKEYS.black_to_move = xorshift64(&seed);

    for (int i = 0; i < 16; i++)
        ZKEYS.castle[i] = xorshift64(&seed);

    for (int i = 0; i < 8; i++)
        ZKEYS.ep[i] = xorshift64(&seed);
}

/* ──────────────────────────────────────────────
 *  Board helpers
 * ────────────────────────────────────────────── */
static void board_clear(Board *b) {
    memset(b, 0, sizeof(*b));
    b->ep_sq = NO_SQ;
    for (int i = 0; i < 64; i++) b->mailbox[i] = NO_PIECE;
}

static uint64_t compute_hash(const Board *b) {
    uint64_t h = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq = bb_pop(&bb);
                h ^= ZKEYS.piece[c][pt][sq];
            }
        }
    if (b->side == BLACK) h ^= ZKEYS.black_to_move;
    h ^= ZKEYS.castle[b->castle];
    if (b->ep_sq != NO_SQ) h ^= ZKEYS.ep[b->ep_sq & 7];
    return h;
}

/* ──────────────────────────────────────────────
 *  Start position
 * ────────────────────────────────────────────── */
void board_start_pos(Board *b) {
    board_from_fen(b, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

/* ──────────────────────────────────────────────
 *  FEN parser
 * ────────────────────────────────────────────── */
bool board_from_fen(Board *b, const char *fen) {
    board_clear(b);

    /* Piece placement */
    int rank = 7, file = 0;
    const char *p = fen;
    while (*p && *p != ' ') {
        char c = *p++;
        if (c == '/') { rank--; file = 0; }
        else if (c >= '1' && c <= '8') { file += c - '0'; }
        else {
            static const char *piece_chars = "PNBRQKpnbrqk";
            const char *pos = strchr(piece_chars, c);
            if (!pos) return false;
            int piece_idx = (int)(pos - piece_chars);
            Color col = (piece_idx >= 6) ? BLACK : WHITE;
            PieceType pt = (PieceType)(piece_idx % 6);
            Square sq = (Square)(rank * 8 + file);
            /* put_piece hashes incrementally, but hash not started yet; 
               we will compute it from scratch at the end */
            Bitboard s = SQUARE_BB[sq];
            b->pieces[col][pt] |= s;
            b->occ[col]        |= s;
            b->occ[2]          |= s;
            b->mailbox[sq]      = make_piece(col, pt);
            file++;
        }
    }

    /* Side to move */
    if (*p) p++; /* space */
    b->side = (*p == 'b') ? BLACK : WHITE;
    if (*p) p += 2; /* 'w '/' b ' */

    /* Castling rights */
    b->castle = 0;
    while (*p && *p != ' ') {
        switch (*p++) {
            case 'K': b->castle |= CR_WK; break;
            case 'Q': b->castle |= CR_WQ; break;
            case 'k': b->castle |= CR_BK; break;
            case 'q': b->castle |= CR_BQ; break;
            default: break;
        }
    }
    if (*p) p++; /* space */

    /* En passant */
    b->ep_sq = NO_SQ;
    if (*p && *p != '-') {
        int ef = *p - 'a';
        p++;
        int er = *p - '1';
        p++;
        b->ep_sq = er * 8 + ef;
    } else if (*p) { p++; }
    if (*p) p++;

    /* Half-move clock */
    b->halfmove = 0;
    if (*p) { b->halfmove = atoi(p); while (*p && *p != ' ') p++; if (*p) p++; }

    /* Full-move number */
    b->fullmove = 1;
    if (*p) b->fullmove = atoi(p);

    b->hash = compute_hash(b);
    return true;
}

/* ──────────────────────────────────────────────
 *  Board print (ASCII)
 * ────────────────────────────────────────────── */
static const char PIECE_CHARS[] = "PNBRQKpnbrqk.";

void board_print(const Board *b) {
    printf("\n  +-----------------+\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d | ", r + 1);
        for (int f = 0; f < 8; f++) {
            Piece pc = b->mailbox[r * 8 + f];
            printf("%c ", PIECE_CHARS[pc]);
        }
        printf("|\n");
    }
    printf("  +-----------------+\n");
    printf("    a b c d e f g h\n");
    printf("Side: %s  Castle: %d  EP: %d  Half: %d\n\n",
           b->side == WHITE ? "White" : "Black",
           b->castle, b->ep_sq, b->halfmove);
}

/* ──────────────────────────────────────────────
 *  Attack detection
 *
 *  OPT-SKIP-SLIDER: skip the expensive slider lookups when the attacker
 *  has no sliders of the relevant type.  In endgames (K+P, K+minor+P)
 *  this saves 2 slider lookups per is_square_attacked call — at 1.3M
 *  calls per search, that's ~2.6M saved slider lookups.
 *
 *  In the middlegame the check is essentially free (one OR + non-zero
 *  test) and the slider lookups proceed as before.
 * ────────────────────────────────────────────── */
bool is_square_attacked(const Board *b, Square sq, Color attacker) {
    Bitboard occ = b->occ[2];

    if (PAWN_ATTACKS[attacker ^ 1][sq] & b->pieces[attacker][PAWN])   return true;
    if (KNIGHT_ATTACKS[sq]             & b->pieces[attacker][KNIGHT])  return true;
    if (KING_ATTACKS[sq]               & b->pieces[attacker][KING])    return true;

    /* Slider checks — gated on enemy material so we skip the slider
     * lookup entirely when there's nothing to find. */
    Bitboard diag_attackers = b->pieces[attacker][BISHOP] | b->pieces[attacker][QUEEN];
    if (diag_attackers && (bishop_attacks(sq, occ) & diag_attackers)) return true;
    Bitboard orth_attackers = b->pieces[attacker][ROOK]   | b->pieces[attacker][QUEEN];
    if (orth_attackers && (rook_attacks(sq, occ)   & orth_attackers)) return true;
    return false;
}

/* OPT-PIN: occupancy-parameterised version for king-move legality. */
bool is_square_attacked_with_occ(const Board *b, Square sq, Color attacker,
                                  Bitboard occ) {
    if (PAWN_ATTACKS[attacker ^ 1][sq] & b->pieces[attacker][PAWN])   return true;
    if (KNIGHT_ATTACKS[sq]             & b->pieces[attacker][KNIGHT])  return true;
    if (KING_ATTACKS[sq]               & b->pieces[attacker][KING])    return true;

    Bitboard diag_attackers = b->pieces[attacker][BISHOP] | b->pieces[attacker][QUEEN];
    if (diag_attackers && (bishop_attacks(sq, occ) & diag_attackers)) return true;
    Bitboard orth_attackers = b->pieces[attacker][ROOK]   | b->pieces[attacker][QUEEN];
    if (orth_attackers && (rook_attacks(sq, occ)   & orth_attackers)) return true;
    return false;
}

bool in_check(const Board *b) {
    int king_sq = bb_lsb(b->pieces[b->side][KING]);
    return is_square_attacked(b, (Square)king_sq, (Color)(b->side ^ 1));
}

/* ──────────────────────────────────────────────
 *  Make move
 * ────────────────────────────────────────────── */
void make_move(Board *b, Move m) {
    Color  us   = b->side;
    Color  them = us ^ 1;
    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    if (b->hist_idx >= MAX_PLY) {
        /* Should never reach here after the negamax ply ceiling is in place.
           Silently clamp rather than crash so a bug report for the CPU manufacturer is still possible. */
        b->hist_idx = MAX_PLY - 1;
    }

    /* Save undo state */
    UndoInfo *u  = &b->history[b->hist_idx++];
    u->move      = m;
    u->ep_sq     = b->ep_sq;
    u->castle    = b->castle;
    u->halfmove  = b->halfmove;
    u->hash      = b->hash;
    u->captured  = NO_PIECE;

    /* Update hash for old state */
    b->hash ^= ZKEYS.castle[b->castle];
    if (b->ep_sq != NO_SQ) b->hash ^= ZKEYS.ep[b->ep_sq & 7];
    b->ep_sq = NO_SQ;

    Piece moving = b->mailbox[from];
    PieceType pt = piece_type(moving);

    b->halfmove++;

    /* Remove captured piece */
    if (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP) {
        Piece cap = b->mailbox[to];
        u->captured = cap;
        remove_piece(b, them, piece_type(cap), to);
        b->halfmove = 0;
    } else if (mt == MT_EP) {
        Square ep_cap = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        u->captured = make_piece(them, PAWN);
        remove_piece(b, them, PAWN, ep_cap);
        b->halfmove = 0;
    }

    if (pt == PAWN) b->halfmove = 0;

    /* Move the piece */
    if (mt >= MT_N_PROMO) {
        /* Promotion: remove pawn, place promoted piece */
        remove_piece(b, us, PAWN, from);
        PieceType promo = MOVE_PROMO_PT(m);
        put_piece(b, us, promo, to);
    } else {
        move_piece(b, us, pt, from, to);
    }

    /* Castling: also move the rook */
    if (mt == MT_KCASTLE) {
        Square rf = (us == WHITE) ? H1 : H8;
        Square rt = (us == WHITE) ? F1 : F8;
        move_piece(b, us, ROOK, rf, rt);
    } else if (mt == MT_QCASTLE) {
        Square rf = (us == WHITE) ? A1 : A8;
        Square rt = (us == WHITE) ? D1 : D8;
        move_piece(b, us, ROOK, rf, rt);
    }

    /* Double pawn push sets en-passant square */
    if (mt == MT_DPUSH) {
        b->ep_sq = (us == WHITE) ? (int)(to - 8) : (int)(to + 8);
        b->hash ^= ZKEYS.ep[b->ep_sq & 7];
    }

    /* Update castling rights.
     * OPT-I: fully-initialized 64-entry table where every non-castling-
     * relevant square has mask ~0 (no rights removed), so we can index
     * without a fallback branch.  The previous code relied on C's
     * default-zero initialization for unlisted indices and then branched
     * to convert 0 → ~0; this version is branchless.
     *
     * Square indices: A1=0, E1=4, H1=7, A8=56, E8=60, H8=63.
     * Mask values: ~CR_WQ = ~2 = 0xFFFFFFFD, ~(CR_WK|CR_WQ) = ~3 = 0xFFFFFFFC,
     *              ~CR_WK = ~1 = 0xFFFFFFFE, similarly for black. */
    static const int castle_mask[64] = {
        /* a1 */ ~CR_WQ,             /* b1 */ ~0, ~0, ~0,
        /* e1 */ ~(CR_WK|CR_WQ),     /* f1 */ ~0, ~0,
        /* h1 */ ~CR_WK,
        /* a2..h2 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a3..h3 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a4..h4 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a5..h5 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a6..h6 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a7..h7 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
        /* a8 */ ~CR_BQ,             /* b8 */ ~0, ~0, ~0,
        /* e8 */ ~(CR_BK|CR_BQ),     /* f8 */ ~0, ~0,
        /* h8 */ ~CR_BK
    };
    b->castle &= castle_mask[from] & castle_mask[to];

    /* Finalise hash */
    b->hash ^= ZKEYS.castle[b->castle];
    b->hash ^= ZKEYS.black_to_move;
    b->side  = them;
    b->ply++;
    if (us == BLACK) b->fullmove++;
}

/* ──────────────────────────────────────────────
 *  Unmake move
 * ────────────────────────────────────────────── */
void unmake_move(Board *b) {
    UndoInfo *u = &b->history[--b->hist_idx];
    Move m  = u->move;
    Color them = b->side;          /* after unmake, them becomes the mover */
    Color us   = them ^ 1;

    b->side      = us;
    b->ep_sq     = u->ep_sq;
    b->castle    = u->castle;
    b->halfmove  = u->halfmove;
    /* NOTE: Do NOT restore b->hash here yet.  The put_piece / move_piece /
     * remove_piece calls below each XOR the hash incrementally, which
     * would corrupt a restored hash.  Instead, we let those calls
     * re-derive the hash delta (they undo the make_move's hash delta),
     * and then we restore the saved hash at the very end.  Failing to
     * do this was the root cause of the "illegal position" / "two
     * kings" bug: the hash drifted from the actual board state,
     * causing TT entries from unrelated positions to be applied. */
    b->ply--;
    if (us == BLACK) b->fullmove--;

    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    /* Undo promotion */
    if (mt >= MT_N_PROMO) {
        PieceType promo = MOVE_PROMO_PT(m);
        remove_piece(b, us, promo, to);
        put_piece   (b, us, PAWN, from);
    } else {
        move_piece(b, us, piece_type(b->mailbox[to]), to, from);
    }

    /* Restore captured piece */
    if (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP) {
        put_piece(b, them, piece_type(u->captured), to);
    } else if (mt == MT_EP) {
        Square ep_cap = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        put_piece(b, them, PAWN, ep_cap);
    }

    /* Undo castling rook move */
    if (mt == MT_KCASTLE) {
        Square rf = (us == WHITE) ? F1 : F8;
        Square rt = (us == WHITE) ? H1 : H8;
        move_piece(b, us, ROOK, rf, rt);
    } else if (mt == MT_QCASTLE) {
        Square rf = (us == WHITE) ? D1 : D8;
        Square rt = (us == WHITE) ? A1 : A8;
        move_piece(b, us, ROOK, rf, rt);
    }

    /* Now restore the saved hash — after all piece ops are done. */
    b->hash = u->hash;
}

/* ──────────────────────────────────────────────
 *  Null move (for null-move pruning in search)
 * ────────────────────────────────────────────── */
void make_null_move(Board *b) {
    UndoInfo *u = &b->history[b->hist_idx++];
    u->move     = NULL_MOVE;
    u->ep_sq    = b->ep_sq;
    u->castle   = b->castle;
    u->halfmove = b->halfmove;
    u->hash     = b->hash;
    u->captured = NO_PIECE;

    if (b->ep_sq != NO_SQ) {
        b->hash ^= ZKEYS.ep[b->ep_sq & 7];
        b->ep_sq = NO_SQ;
    }
    b->hash ^= ZKEYS.black_to_move;
    b->side ^= 1;
    b->ply++;
    b->halfmove++;
}

void unmake_null_move(Board *b) {
    UndoInfo *u = &b->history[--b->hist_idx];
    b->side     = b->side ^ 1;
    b->ep_sq    = u->ep_sq;
    b->castle   = u->castle;
    b->halfmove = u->halfmove;
    b->hash     = u->hash;
    b->ply--;
}

/* ──────────────────────────────────────────────
 *  Perft
 * ────────────────────────────────────────────── */
uint64_t perft(Board *b, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    gen_moves(b, &ml);
    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; i++) {
        make_move(b, ml.moves[i]);
        if (!is_square_attacked(b, (Square)bb_lsb(b->pieces[b->side^1][KING]), b->side))
            nodes += perft(b, depth - 1);
        unmake_move(b);
    }
    return nodes;
}

/* ──────────────────────────────────────────────
 *  board_key_after — incremental Zobrist hash for the position after
 *  playing move `m`, without calling make_move.  Used for speculative
 *  TT prefetch (OPT-PF).
 *
 *  The hash delta must EXACTLY match what make_move XORs — otherwise
 *  the speculative prefetch points at the wrong TT bucket.  We
 *  replicate the logic of make_move's hash updates:
 *
 *    1. Remove old castle + EP keys (XOR off)
 *    2. Apply castle-rights mask (from/to squares)
 *    3. Add new castle key (XOR on)
 *    4. Remove captured piece (if any)
 *    5. Move piece (remove from, add to) OR handle promotion
 *    6. Move castling rook (if castle)
 *    7. Set new EP square (if double-push)
 *    8. Flip side-to-move
 *
 *  We need the castle_mask table here too — duplicate it (small).
 *  See make_move() for the authoritative version.
 * ────────────────────────────────────────────── */
static const int board_castle_mask[64] = {
    /* a1 */ ~CR_WQ,             /* b1 */ ~0, ~0, ~0,
    /* e1 */ ~(CR_WK|CR_WQ),     /* f1 */ ~0, ~0,
    /* h1 */ ~CR_WK,
    /* a2..h2 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a3..h3 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a4..h4 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a5..h5 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a6..h6 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a7..h7 */ ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a8 */ ~CR_BQ,             /* b8 */ ~0, ~0, ~0,
    /* e8 */ ~(CR_BK|CR_BQ),     /* f8 */ ~0, ~0,
    /* h8 */ ~CR_BK
};

uint64_t board_key_after(const Board *b, Move m) {
    Color  us   = b->side;
    Color  them = us ^ 1;
    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    uint64_t h = b->hash;

    /* 1. Remove old castle + EP keys (XOR off, since they were XORed on) */
    h ^= ZKEYS.castle[b->castle];
    if (b->ep_sq != NO_SQ) h ^= ZKEYS.ep[b->ep_sq & 7];

    /* 2-3. Castle rights change (apply mask).  We compute the new
     * castle rights and XOR on the new key.  make_move does the
     * same with `b->castle &= castle_mask[from] & castle_mask[to]`. */
    int new_castle = b->castle & board_castle_mask[from] & board_castle_mask[to];
    h ^= ZKEYS.castle[new_castle];

    /* 4. Remove captured piece (regular capture or EP) */
    if (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP) {
        Piece cap = b->mailbox[to];
        h ^= ZKEYS.piece[them][piece_type(cap)][to];
    } else if (mt == MT_EP) {
        Square ep_cap = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        h ^= ZKEYS.piece[them][PAWN][ep_cap];
    }

    /* 5. Move piece (or promotion) */
    if (mt >= MT_N_PROMO) {
        /* Promotion: remove pawn from `from`, add promoted piece at `to` */
        h ^= ZKEYS.piece[us][PAWN][from];
        PieceType promo = MOVE_PROMO_PT(m);
        h ^= ZKEYS.piece[us][promo][to];
    } else {
        /* Normal move: remove piece from `from`, add at `to` */
        Piece moving = b->mailbox[from];
        PieceType pt = piece_type(moving);
        h ^= ZKEYS.piece[us][pt][from];
        h ^= ZKEYS.piece[us][pt][to];
    }

    /* 6. Castling rook move (king-side or queen-side) */
    if (mt == MT_KCASTLE) {
        Square rf = (us == WHITE) ? H1 : H8;
        Square rt = (us == WHITE) ? F1 : F8;
        h ^= ZKEYS.piece[us][ROOK][rf];
        h ^= ZKEYS.piece[us][ROOK][rt];
    } else if (mt == MT_QCASTLE) {
        Square rf = (us == WHITE) ? A1 : A8;
        Square rt = (us == WHITE) ? D1 : D8;
        h ^= ZKEYS.piece[us][ROOK][rf];
        h ^= ZKEYS.piece[us][ROOK][rt];
    }

    /* 7. New EP square (double pawn push only) */
    if (mt == MT_DPUSH) {
        int ep_sq = (us == WHITE) ? (int)(to - 8) : (int)(to + 8);
        h ^= ZKEYS.ep[ep_sq & 7];
    }

    /* 8. Side to move */
    h ^= ZKEYS.black_to_move;

    return h;
}