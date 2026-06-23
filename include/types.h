#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
 *  Fundamental type aliases
 * ────────────────────────────────────────────── */
typedef uint64_t Bitboard;

/* ──────────────────────────────────────────────
 *  Square encoding  (A1 = 0, H8 = 63)
 *
 *   8  56 57 58 59 60 61 62 63
 *   7  48 49 50 51 52 53 54 55
 *   6  40 41 42 43 44 45 46 47
 *   5  32 33 34 35 36 37 38 39
 *   4  24 25 26 27 28 29 30 31
 *   3  16 17 18 19 20 21 22 23
 *   2   8  9 10 11 12 13 14 15
 *   1   0  1  2  3  4  5  6  7
 *       a  b  c  d  e  f  g  h
 * ────────────────────────────────────────────── */
typedef enum {
    A1=0,  B1,  C1,  D1,  E1,  F1,  G1,  H1,
    A2=8,  B2,  C2,  D2,  E2,  F2,  G2,  H2,
    A3=16, B3,  C3,  D3,  E3,  F3,  G3,  H3,
    A4=24, B4,  C4,  D4,  E4,  F4,  G4,  H4,
    A5=32, B5,  C5,  D5,  E5,  F5,  G5,  H5,
    A6=40, B6,  C6,  D6,  E6,  F6,  G6,  H6,
    A7=48, B7,  C7,  D7,  E7,  F7,  G7,  H7,
    A8=56, B8,  C8,  D8,  E8,  F8,  G8,  H8,
    NO_SQ = 64
} Square;

/* ──────────────────────────────────────────────
 *  Piece types and colours
 * ────────────────────────────────────────────── */
typedef enum { PAWN=0, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE=6 } PieceType;
typedef enum { WHITE=0, BLACK=1 } Color;

/* Combined piece (colour + type)
 * WP=0 .. WK=5, BP=6 .. BK=11 */
typedef enum {
    WP=0, WN, WB, WR, WQ, WK,
    BP=6, BN, BB, BR, BQ, BK,
    NO_PIECE = 12
} Piece;

static inline Color  piece_color(Piece p) { return (Color)(p / 6); }
static inline Color  piece_col(Piece p)   { return (Color)(p >= BP ? BLACK : WHITE); }
static inline PieceType piece_type(Piece p) { return (PieceType)(p % 6); }
static inline Piece  make_piece(Color c, PieceType pt) { return (Piece)(c * 6 + pt); }

/* ──────────────────────────────────────────────
 *  Move encoding  (16-bit)
 *
 *  bits  0-5  : from square
 *  bits  6-11 : to square
 *  bits 12-15 : move type
 * ────────────────────────────────────────────── */
typedef uint16_t Move;

typedef enum {
    MT_QUIET       = 0,
    MT_DPUSH       = 1,   /* double pawn push */
    MT_KCASTLE     = 2,   /* king-side castle  */
    MT_QCASTLE     = 3,   /* queen-side castle */
    MT_CAPTURE     = 4,
    MT_EP          = 5,   /* en-passant capture */
    /* 6, 7 reserved */
    MT_N_PROMO     = 8,
    MT_B_PROMO     = 9,
    MT_R_PROMO     = 10,
    MT_Q_PROMO     = 11,
    MT_N_PROMO_CAP = 12,
    MT_B_PROMO_CAP = 13,
    MT_R_PROMO_CAP = 14,
    MT_Q_PROMO_CAP = 15,
} MoveType;

#define MOVE_FROM(m)     ((Square)((m) & 0x3F))
#define MOVE_TO(m)       ((Square)(((m) >> 6) & 0x3F))
#define MOVE_TYPE(m)     ((MoveType)(((m) >> 12) & 0xF))
#define MOVE_IS_CAP(m)   (MOVE_TYPE(m) & MT_CAPTURE)
#define MOVE_IS_PROMO(m) (MOVE_TYPE(m) >= MT_N_PROMO)
#define MOVE_PROMO_PT(m) ((PieceType)(KNIGHT + ((MOVE_TYPE(m)) & 3)))

#define MAKE_MOVE(f,t,mt) ((Move)((f) | ((t) << 6) | ((mt) << 12)))
#define NULL_MOVE         ((Move)0)

/* ──────────────────────────────────────────────
 *  Castling rights bitmask
 * ────────────────────────────────────────────── */
#define CR_WK  1
#define CR_WQ  2
#define CR_BK  4
#define CR_BQ  8
#define CR_ALL 15

/* ──────────────────────────────────────────────
 *  Move list
 * ────────────────────────────────────────────── */
#define MAX_MOVES 512
#define MAX_PLY   256

typedef struct { Move moves[MAX_MOVES]; int count; } MoveList;

/* ──────────────────────────────────────────────
 *  Score constants
 * ────────────────────────────────────────────── */
#define INF       32000
#define MATE_SCORE 31000
#define DRAW_SCORE 0