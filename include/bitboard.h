#pragma once
#include "types.h"
#include <stdio.h>

/* ──────────────────────────────────────────────
 *  Compile-time feature detection
 *  The Makefile passes -march=native so GCC/Clang
 *  will define __POPCNT__, __BMI2__, __AVX2__ etc.
 *  automatically when the host CPU supports them.
 * ────────────────────────────────────────────── */
#if defined(__BMI2__)
#  include <immintrin.h>
#  define HAS_PEXT 1
#endif

#if defined(__AVX2__)
#  include <immintrin.h>
#  define HAS_AVX2 1
#endif

/* ──────────────────────────────────────────────
 *  Pre-computed tables (defined in bitboard.c)
 * ────────────────────────────────────────────── */
extern Bitboard SQUARE_BB[64];

extern Bitboard FILE_BB[8];      /* all squares on file 0..7 */
extern Bitboard RANK_BB[8];      /* all squares on rank 0..7 */
extern Bitboard FILE_MASK[64];   /* file through square (for HQ) */
extern Bitboard DIAG_MASK[64];   /* diagonal through square     */
extern Bitboard ADIAG_MASK[64];  /* anti-diagonal through square */

extern Bitboard PAWN_ATTACKS[2][64];
extern Bitboard KNIGHT_ATTACKS[64];
extern Bitboard KING_ATTACKS[64];

/* Between/ray tables for check evasion & pin detection */
extern Bitboard BETWEEN_BB[64][64];
extern Bitboard LINE_BB[64][64];

/* ──────────────────────────────────────────────
 *  Initialisation (call once at startup)
 * ────────────────────────────────────────────── */
void bitboard_init(void);

/* ──────────────────────────────────────────────
 *  Sliding piece attack lookups
 *
 *  Implementation uses:
 *   • Hyperbola Quintessence (o^(o−2r) trick)
 *     for files and diagonals — O(1), branchless.
 *   • Pre-computed first-rank attack table for ranks.
 *   • When BMI2 PEXT is available the rank attacks
 *     use _pext_u64 for a further small speedup.
 * ────────────────────────────────────────────── */
Bitboard rook_attacks  (Square sq, Bitboard occ);
Bitboard bishop_attacks(Square sq, Bitboard occ);

Bitboard queen_attacks(Square sq, Bitboard occ);

/* ──────────────────────────────────────────────
 *  Bit-manipulation helpers (all inline / intrinsic)
 * ────────────────────────────────────────────── */
static inline int bb_lsb(Bitboard bb) {
    return __builtin_ctzll(bb);
}

static inline int bb_msb(Bitboard bb) {
    return 63 ^ __builtin_clzll(bb);
}

static inline int bb_pop(Bitboard *bb) {
    int sq = __builtin_ctzll(*bb);
    *bb &= *bb - 1;
    return sq;
}

static inline int bb_popcount(Bitboard bb) {
    return __builtin_popcountll(bb);
}

static inline Bitboard sq_bb(Square sq) {
    return SQUARE_BB[sq];
}

/* Shift helpers (avoid undefined behaviour on zero) */
static inline Bitboard bb_north(Bitboard b) { return b << 8; }
static inline Bitboard bb_south(Bitboard b) { return b >> 8; }
static inline Bitboard bb_east (Bitboard b) { return (b & ~FILE_BB[7]) << 1; }
static inline Bitboard bb_west (Bitboard b) { return (b & ~FILE_BB[0]) >> 1; }

/* ──────────────────────────────────────────────
 *  Debug helper
 * ────────────────────────────────────────────── */
void bb_print(Bitboard bb);
