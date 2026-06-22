#pragma once
#include "types.h"
#include <stdio.h>

/* ──────────────────────────────────────────────
 *  Compile-time feature detection.
 *
 *  We do NOT rely on -march=native for BMI2, because that bakes in a
 *  hard requirement on the build host's CPU.  Instead we always
 *  compile the PEXT code path under __BMI2__ if the compiler supports
 *  the intrinsics, and dispatch at runtime via function pointers
 *  initialised in bitboard_init().  The fallback path (Hyperbola
 *  Quintessence + rank table) is always present.
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

/*
 * PEXT attack tables (used only when BMI2 is available at runtime).
 *
 * For each square, ATT_RELEVANT[sq] is the occupancy mask of squares
 * that actually matter for that slider's attacks (excludes the outer
 * rank/file edges, which don't change the attack set).  The attack set
 * is then looked up in a flat array indexed by _pext_u64(occ, mask).
 *
 * Total size:
 *   Rook:   102K entries × 8 bytes  = ~800 KB
 *   Bishop:  5K entries × 8 bytes   = ~40 KB
 *
 * The rook table is sizeable but fits in L2 on modern CPUs and the
 * access pattern is very regular.  Hyperbola Quintessence stays as
 * the fallback for CPUs without BMI2.
 */
extern Bitboard ROOK_ATTACKS   [0x19000];   /* 102400 entries */
extern Bitboard BISHOP_ATTACKS [0x01480];   /*   5248 entries */
extern Bitboard ROOK_RELEVANT  [64];
extern Bitboard BISHOP_RELEVANT[64];

/* ──────────────────────────────────────────────
 *  Initialisation (call once at startup)
 * ────────────────────────────────────────────── */
void bitboard_init(void);

/* ──────────────────────────────────────────────
 *  Sliding piece attack lookups
 *
 *  Two implementations are linked in:
 *    • Hyperbola Quintessence (o^(o−2r) trick) — fallback, no BMI2.
 *    • PEXT-based table lookup — used when BMI2 is detected at runtime.
 *
 *  The public rook_attacks / bishop_attacks functions dispatch via
 *  function pointers set in bitboard_init() based on a runtime CPU
 *  feature check.  This keeps a single binary portable across
 *  BMI2 and non-BMI2 hosts.
 * ────────────────────────────────────────────── */
Bitboard rook_attacks  (Square sq, Bitboard occ);
Bitboard bishop_attacks(Square sq, Bitboard occ);

/* Direct (non-dispatched) entry points — used internally and by tests. */
Bitboard rook_attacks_hq  (Square sq, Bitboard occ);
Bitboard bishop_attacks_hq(Square sq, Bitboard occ);
Bitboard rook_attacks_pext  (Square sq, Bitboard occ);
Bitboard bishop_attacks_pext(Square sq, Bitboard occ);

static inline Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

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
