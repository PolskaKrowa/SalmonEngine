#pragma once
#include "types.h"
#include "bitboard.h"

/* ──────────────────────────────────────────────
 *  Undo information saved before make_move
 * ────────────────────────────────────────────── */
typedef struct {
    Move     move;
    Piece    captured;
    int      ep_sq;           /* NO_SQ when absent */
    int      castle;          /* castling rights bitmask */
    int      halfmove;        /* 50-move counter */
    uint64_t hash;
} UndoInfo;

/* ──────────────────────────────────────────────
 *  Board representation
 *
 *  12 bitboards (one per piece), 3 occupancy BBs,
 *  a mailbox for O(1) "what is on square X?" and
 *  all position metadata.
 * ────────────────────────────────────────────── */
typedef struct {
    Bitboard pieces[2][6];    /* [color][piece_type] */
    Bitboard occ[3];          /* [WHITE][BLACK][BOTH] */
    Piece    mailbox[64];     /* piece on each square  */

    Color    side;            /* side to move          */
    int      ep_sq;           /* en-passant target sq  */
    int      castle;          /* castling rights mask  */
    int      halfmove;        /* 50-move clock         */
    int      fullmove;        /* full-move number      */
    uint64_t hash;            /* Zobrist key           */

    UndoInfo history[MAX_PLY];
    int      ply;             /* current search ply    */
    int      hist_idx;        /* history stack pointer */
} Board;

/* ──────────────────────────────────────────────
 *  Zobrist keys  (initialised in board_init)
 * ────────────────────────────────────────────── */
typedef struct {
    uint64_t piece[2][6][64]; /* [color][type][square] */
    uint64_t black_to_move;
    uint64_t castle[16];      /* one key per rights combination */
    uint64_t ep[8];           /* one key per ep file */
} ZobristKeys;

extern ZobristKeys ZKEYS;

/* ──────────────────────────────────────────────
 *  API
 * ────────────────────────────────────────────── */
void board_init(void);               /* initialise Zobrist keys etc. */
void board_start_pos(Board *b);      /* set up initial position      */
bool board_from_fen(Board *b, const char *fen);
void board_print(const Board *b);

/* Incremental Zobrist helpers */
static inline void hash_piece(Board *b, Color c, PieceType pt, Square sq) {
    b->hash ^= ZKEYS.piece[c][pt][sq];
}

/* Place / remove pieces — keep bitboards and mailbox in sync */
static inline void put_piece(Board *b, Color c, PieceType pt, Square sq) {
    Bitboard s = sq_bb(sq);
    b->pieces[c][pt]  |= s;
    b->occ[c]         |= s;
    b->occ[2]         |= s;
    b->mailbox[sq]     = make_piece(c, pt);
    hash_piece(b, c, pt, sq);
}

static inline void remove_piece(Board *b, Color c, PieceType pt, Square sq) {
    Bitboard s = sq_bb(sq);
    b->pieces[c][pt]  &= ~s;
    b->occ[c]         &= ~s;
    b->occ[2]         &= ~s;
    b->mailbox[sq]     = NO_PIECE;
    hash_piece(b, c, pt, sq);
}

static inline void move_piece(Board *b, Color c, PieceType pt, Square from, Square to) {
    Bitboard mask = sq_bb(from) | sq_bb(to);
    b->pieces[c][pt]  ^= mask;
    b->occ[c]         ^= mask;
    b->occ[2]         ^= mask;
    b->mailbox[from]   = NO_PIECE;
    b->mailbox[to]     = make_piece(c, pt);
    hash_piece(b, c, pt, from);
    hash_piece(b, c, pt, to);
}

/* Make / unmake */
void make_move  (Board *b, Move m);
void unmake_move(Board *b);

/* Null move (for null-move pruning) */
void make_null_move  (Board *b);
void unmake_null_move(Board *b);

/*
 * board_key_after — compute the Zobrist hash of the position AFTER
 * playing move `m`, WITHOUT calling make_move.  Used for speculative
 * TT prefetch (OPT-PF): the search calls this before make_move so the
 * TT bucket for the child position is on its way to L1 while the
 * parent makes the move and sets up the recursive call.
 *
 * The hash is computed incrementally by XOR-ing the same Zobrist keys
 * that make_move would XOR — moving piece (from→to), captured piece,
 * promotion, EP square, castle rights, side-to-move.
 *
 * For non-castling, non-EP, non-promotion quiet moves, this is 4 XORs.
 * For captures, +2 XORs (remove captured).  For promotion, +2 (remove
 * pawn, add promo).  For castling, +4 (also move rook).  For EP, +1
 * (remove captured pawn from its actual square).
 *
 * NOTE: this function does NOT validate the move.  It assumes `m` is
 * a pseudo-legal move in the current position.  Calling it on an
 * illegal move produces a garbage hash — harmless because the prefetch
 * is advisory, but the *real* TT probe (after make_move) uses the
 * correct hash.
 */
uint64_t board_key_after(const Board *b, Move m);

/* Utility */
bool is_square_attacked(const Board *b, Square sq, Color attacker);

/* OPT-PIN: same as is_square_attacked but with a caller-supplied
 * occupancy bitboard.  Used by is_legal_fast() to test king moves
 * with the king itself removed from occupancy (so x-rays through
 * the king's old square are detected). */
bool is_square_attacked_with_occ(const Board *b, Square sq, Color attacker,
                                  Bitboard occ);

bool in_check(const Board *b);

/* Perft (for testing correctness) */
uint64_t perft(Board *b, int depth);
