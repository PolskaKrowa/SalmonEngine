#pragma once
#include "types.h"
#include "config.h"
#include <stdio.h>

/* ──────────────────────────────────────────────
 *  Compile-time feature detection
 *
 *  We rely on the autotools-detected HAVE_BMI2 / HAVE_AVX2 macros from
 *  config.h (set by configure.ac). When HAVE_BMI2 is 1 we use the BMI2
 *  PEXT instruction for magic-bitboard index extraction — no magic-number
 *  search required, just a single _pext_u64 call. When unavailable we
 *  fall back to the classical "fancy magic" approach with pre-computed
 *  magic numbers.
 * ────────────────────────────────────────────── */
#if defined(HAVE_BMI2) && HAVE_BMI2
#  include <immintrin.h>
#  define BB_USE_PEXT 1
#else
#  define BB_USE_PEXT 0
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
 *  Implementation: magic bitboards.
 *   • BMI2 PEXT variant (no magic-number search needed — a single
 *     _pext_u64 extracts the index). Used when HAVE_BMI2=1.
 *   • Classical "fancy magic" variant (pre-computed magic numbers
 *     found by random search at init time). Used otherwise.
 *
 *  Both variants produce identical attack bitboards. The magic-bitboard
 *  lookup is ~1.5-2× faster than the previous Hyperbola Quintessence
 *  approach for slider-attack queries because it's a single table lookup
 *  instead of a 4-instruction bswap+subtract chain per direction.
 * ────────────────────────────────────────────── */
Bitboard rook_attacks  (Square sq, Bitboard occ);
Bitboard bishop_attacks(Square sq, Bitboard occ);

static inline Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

/* ──────────────────────────────────────────────
 *  Bit-manipulation helpers (all inline / intrinsic)
 * ────────────────────────────────────────────── */
static inline int bb_lsb(Bitboard bb) {
    if (bb == 0) return -1;
    return __builtin_ctzll(bb);
}

static inline int bb_msb(Bitboard bb) {
    if (bb == 0) return -1;
    return 63 ^ __builtin_clzll(bb);
}

static inline int bb_pop(Bitboard *bb) {
    if (*bb == 0) return -1;
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
