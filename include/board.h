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

/* Utility */
bool is_square_attacked(const Board *b, Square sq, Color attacker);
bool in_check(const Board *b);

/* Perft (for testing correctness) */
uint64_t perft(Board *b, int depth);
