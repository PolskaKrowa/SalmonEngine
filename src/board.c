/*
 * board.c — Board state management  (optimised)
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

static const int CASTLE_MASK[64] = {
    /* a1 */ ~CR_WQ, ~0, ~0, ~0,
    /* e1 */ ~(CR_WK|CR_WQ), ~0, ~0,
    /* h1 */ ~CR_WK,
    /* a2-a7 rows: all ~0 */
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    ~0, ~0, ~0, ~0, ~0, ~0, ~0, ~0,
    /* a8 */ ~CR_BQ, ~0, ~0, ~0,
    /* e8 */ ~(CR_BK|CR_BQ), ~0, ~0,
    /* h8 */ ~CR_BK,
};

/* ──────────────────────────────────────────────
 *  Board helpers
 * ────────────────────────────────────────────── */
static void board_clear(Board *b) {
    memset(b, 0, sizeof(*b));
    b->ep_sq = NO_SQ;
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
            Bitboard s = SQUARE_BB[sq];
            b->pieces[col][pt] |= s;
            b->occ[col]        |= s;
            b->occ[2]          |= s;
            b->mailbox[sq]      = make_piece(col, pt);
            file++;
        }
    }

    if (*p) p++;
    b->side = (*p == 'b') ? BLACK : WHITE;
    if (*p) p += 2;

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
    if (*p) p++;

    b->ep_sq = NO_SQ;
    if (*p && *p != '-') {
        int ef = *p - 'a'; p++;
        int er = *p - '1'; p++;
        b->ep_sq = er * 8 + ef;
    } else if (*p) { p++; }
    if (*p) p++;

    b->halfmove = 0;
    if (*p) { b->halfmove = atoi(p); while (*p && *p != ' ') p++; if (*p) p++; }

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
 * ────────────────────────────────────────────── */
bool is_square_attacked(const Board *b, Square sq, Color attacker) {
    Bitboard occ = b->occ[2];

    if (PAWN_ATTACKS[attacker ^ 1][sq] & b->pieces[attacker][PAWN])  return true;
    if (KNIGHT_ATTACKS[sq]             & b->pieces[attacker][KNIGHT]) return true;
    if (KING_ATTACKS[sq]               & b->pieces[attacker][KING])   return true;

    Bitboard queen_bb = b->pieces[attacker][QUEEN];
    if (bishop_attacks(sq, occ) & (b->pieces[attacker][BISHOP] | queen_bb)) return true;
    if (rook_attacks  (sq, occ) & (b->pieces[attacker][ROOK]   | queen_bb)) return true;

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
    Color    us   = b->side;
    Color    them = us ^ 1;
    Square   from = MOVE_FROM(m);
    Square   to   = MOVE_TO(m);
    MoveType mt   = MOVE_TYPE(m);

    if (b->hist_idx >= MAX_PLY)
        b->hist_idx = MAX_PLY - 1;

    UndoInfo *u = &b->history[b->hist_idx++];
    u->move     = m;
    u->ep_sq    = b->ep_sq;
    u->castle   = b->castle;
    u->halfmove = b->halfmove;
    u->hash     = b->hash;
    u->captured = NO_PIECE;

    b->hash ^= ZKEYS.castle[b->castle];
    if (b->ep_sq != NO_SQ) b->hash ^= ZKEYS.ep[b->ep_sq & 7];
    b->ep_sq = NO_SQ;

    Piece     moving = b->mailbox[from];
    PieceType pt     = piece_type(moving);

    b->halfmove++;

    if (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP) {
        Piece cap   = b->mailbox[to];
        u->captured = cap;
        remove_piece(b, them, piece_type(cap), to);
        b->halfmove = 0;
    } else if (mt == MT_EP) {
        Square ep_cap = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        u->captured   = make_piece(them, PAWN);
        remove_piece(b, them, PAWN, ep_cap);
        b->halfmove = 0;
    }

    if (pt == PAWN) b->halfmove = 0;

    if (mt >= MT_N_PROMO) {
        remove_piece(b, us, PAWN, from);
        put_piece(b, us, MOVE_PROMO_PT(m), to);
    } else {
        move_piece(b, us, pt, from, to);
    }

    if (mt == MT_KCASTLE) {
        Square rf = (us == WHITE) ? H1 : H8;
        Square rt = (us == WHITE) ? F1 : F8;
        move_piece(b, us, ROOK, rf, rt);
    } else if (mt == MT_QCASTLE) {
        Square rf = (us == WHITE) ? A1 : A8;
        Square rt = (us == WHITE) ? D1 : D8;
        move_piece(b, us, ROOK, rf, rt);
    }

    if (mt == MT_DPUSH) {
        b->ep_sq = (us == WHITE) ? (int)(to - 8) : (int)(to + 8);
        b->hash ^= ZKEYS.ep[b->ep_sq & 7];
    }

    b->castle &= CASTLE_MASK[from] & CASTLE_MASK[to];

    b->hash ^= ZKEYS.castle[b->castle];
    b->hash ^= ZKEYS.black_to_move;
    b->side   = them;
    b->ply++;
    if (us == BLACK) b->fullmove++;
}

/* ──────────────────────────────────────────────
 *  Unmake move
 * ────────────────────────────────────────────── */
void unmake_move(Board *b) {
    UndoInfo *u = &b->history[--b->hist_idx];
    Move     m  = u->move;
    Color    them = b->side;
    Color    us   = them ^ 1;

    b->side     = us;
    b->ep_sq    = u->ep_sq;
    b->castle   = u->castle;
    b->halfmove = u->halfmove;
    b->hash     = u->hash;
    b->ply--;
    if (us == BLACK) b->fullmove--;

    Square   from = MOVE_FROM(m);
    Square   to   = MOVE_TO(m);
    MoveType mt   = MOVE_TYPE(m);

    if (mt >= MT_N_PROMO) {
        remove_piece(b, us, MOVE_PROMO_PT(m), to);
        put_piece   (b, us, PAWN, from);
    } else {
        move_piece(b, us, piece_type(b->mailbox[to]), to, from);
    }

    if (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP) {
        put_piece(b, them, piece_type(u->captured), to);
    } else if (mt == MT_EP) {
        Square ep_cap = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
        put_piece(b, them, PAWN, ep_cap);
    }

    if (mt == MT_KCASTLE) {
        Square rf = (us == WHITE) ? F1 : F8;
        Square rt = (us == WHITE) ? H1 : H8;
        move_piece(b, us, ROOK, rf, rt);
    } else if (mt == MT_QCASTLE) {
        Square rf = (us == WHITE) ? D1 : D8;
        Square rt = (us == WHITE) ? A1 : A8;
        move_piece(b, us, ROOK, rf, rt);
    }
}

/* ──────────────────────────────────────────────
 *  Null move
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

    if (depth == 1) {
        uint64_t count = 0;
        for (int i = 0; i < ml.count; i++) {
            make_move(b, ml.moves[i]);
            if (!is_square_attacked(b,
                    (Square)bb_lsb(b->pieces[b->side ^ 1][KING]), b->side))
                count++;
            unmake_move(b);
        }
        return count;
    }

    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; i++) {
        make_move(b, ml.moves[i]);
        if (!is_square_attacked(b,
                (Square)bb_lsb(b->pieces[b->side ^ 1][KING]), b->side))
            nodes += perft(b, depth - 1);
        unmake_move(b);
    }
    return nodes;
}