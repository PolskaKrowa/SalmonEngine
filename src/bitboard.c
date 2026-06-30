/*
 * bitboard.c — Bitboard attack generation (magic bitboards)
 *
 * Sliding piece attacks use magic bitboards — a pre-computed lookup table
 * indexed by (square, occupancy-mask). Two variants are supported:
 *
 *   • BMI2 PEXT  — uses the _pext_u64 instruction to extract the relevant
 *                  occupancy bits as a contiguous index. No magic-number
 *                  search needed; the table is sized to 2^(popcount(mask))
 *                  which is the minimal possible. ~2× faster than classical
 *                  magics on BMI2 hardware.
 *
 *   • Classical  — uses pre-computed "magic" numbers found by random
 *                  search at init time. Slightly larger tables (the magic
 *                  may waste a few bits) but no hardware requirement.
 *
 * Both variants produce identical attack bitboards. Non-sliding pieces
 * (pawn, knight, king) use simple pre-computed lookup tables.
 *
 * Total table sizes:
 *   Rooks:   ~1024 + 4096 + 2048 + ... ≈ 102K entries × 8 bytes ≈ 800 KB
 *   Bishops: ~64 + 144 + 320 + ...     ≈  30K entries × 8 bytes ≈ 240 KB
 *   Combined: ~1 MB — fits in L2 cache.
 */

#include "bitboard.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ──────────────────────────────────────────────
 *  Global tables
 * ────────────────────────────────────────────── */
Bitboard SQUARE_BB[64];
Bitboard FILE_BB[8];
Bitboard RANK_BB[8];
Bitboard FILE_MASK[64];
Bitboard DIAG_MASK[64];
Bitboard ADIAG_MASK[64];

Bitboard PAWN_ATTACKS[2][64];
Bitboard KNIGHT_ATTACKS[64];
Bitboard KING_ATTACKS[64];

Bitboard BETWEEN_BB[64][64];
Bitboard LINE_BB[64][64];

/* ──────────────────────────────────────────────
 *  Magic bitboard tables
 * ────────────────────────────────────────────── */

typedef struct {
    Bitboard  mask;       /* occupancy mask: relevant squares only         */
    Bitboard  magic;      /* magic multiplier (classical only; unused PEXT) */
    Bitboard *attacks;    /* pointer into the shared attack table          */
    unsigned  shift;      /* right-shift amount: 64 - popcount(mask)        */
} MagicEntry;

static MagicEntry ROOK_MAGICS[64];
static MagicEntry BISHOP_MAGICS[64];

/* Shared attack tables — allocated once at init.
 * Total sizes (worst case classical magics):
 *   rooks  : 102400 entries  (~800 KB)
 *   bishops:  29760 entries  (~232 KB) */
static Bitboard ROOK_ATTACK_TABLE  [102400];
static Bitboard BISHOP_ATTACK_TABLE[ 29760];

/* ──────────────────────────────────────────────
 *  Slow (reference) slider — used only at init
 * ────────────────────────────────────────────── */
static Bitboard slow_slider(Square sq, Bitboard occ, int df, int dr) {
    Bitboard atk = 0;
    int f = (int)(sq & 7), r = (int)(sq >> 3);
    for (;;) {
        f += df; r += dr;
        if (f < 0 || f > 7 || r < 0 || r > 7) break;
        Bitboard s = SQUARE_BB[r * 8 + f];
        atk |= s;
        if (s & occ) break;
    }
    return atk;
}

/* Reference rook/bishop attacks — used to populate the magic table. */
static Bitboard slow_rook_attacks(Square sq, Bitboard occ) {
    return slow_slider(sq, occ, 1, 0) | slow_slider(sq, occ, -1, 0)
         | slow_slider(sq, occ, 0, 1) | slow_slider(sq, occ, 0, -1);
}

static Bitboard slow_bishop_attacks(Square sq, Bitboard occ) {
    return slow_slider(sq, occ, 1, 1) | slow_slider(sq, occ, -1, -1)
         | slow_slider(sq, occ, 1, -1) | slow_slider(sq, occ, -1, 1);
}

/* ──────────────────────────────────────────────
 *  Magic-bitboard initialisation
 *
 *  For each square, build:
 *    1. The relevant-occupancy mask (all squares the slider could pass
 *       through, excluding the edges of the board in each direction).
 *    2. The list of all possible occupancies of that mask (2^popcount
 *       permutations).
 *    3. The corresponding attack bitboard for each occupancy.
 *    4. A way to map occupancy -> attack table index.
 *
 *  PEXT path:  index = _pext_u64(occ & mask, mask). The shift is
 *              unused. Table size = 2^popcount(mask).
 *
 *  Classical:  find a magic number m such that
 *              ((occ & mask) * m) >> shift
 *              is a unique index into the attack table for each distinct
 *              occupancy of the mask. We try random 64-bit candidates
 *              with few set bits until one works. Table size = 2^shift.
 * ────────────────────────────────────────────── */

/* Compute the relevant-occupancy mask for a rook on `sq`.
 * Includes all squares the rook could attack on an empty board,
 * EXCEPT the edge squares of each ray (those can never contain a
 * blocker that changes the attack pattern). */
static Bitboard rook_mask(Square sq) {
    Bitboard m = 0;
    int f = sq & 7, r = sq >> 3;
    /* North */
    for (int i = r + 1; i <= 6; i++) m |= SQUARE_BB[i * 8 + f];
    /* South */
    for (int i = r - 1; i >= 1; i--) m |= SQUARE_BB[i * 8 + f];
    /* East */
    for (int i = f + 1; i <= 6; i++) m |= SQUARE_BB[r * 8 + i];
    /* West */
    for (int i = f - 1; i >= 1; i--) m |= SQUARE_BB[r * 8 + i];
    return m;
}

/* Compute the relevant-occupancy mask for a bishop on `sq`. */
static Bitboard bishop_mask(Square sq) {
    Bitboard m = 0;
    int f = sq & 7, r = sq >> 3;
    /* NE */
    for (int i = 1; f + i <= 6 && r + i <= 6; i++) m |= SQUARE_BB[(r + i) * 8 + (f + i)];
    /* NW */
    for (int i = 1; f - i >= 1 && r + i <= 6; i++) m |= SQUARE_BB[(r + i) * 8 + (f - i)];
    /* SE */
    for (int i = 1; f + i <= 6 && r - i >= 1; i++) m |= SQUARE_BB[(r - i) * 8 + (f + i)];
    /* SW */
    for (int i = 1; f - i >= 1 && r - i >= 1; i++) m |= SQUARE_BB[(r - i) * 8 + (f - i)];
    return m;
}

/* PRNG for magic-number search (xorshift64*). */
static uint64_t magic_rng(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * 0x2545F4914F6CDD1DULL;
}

/* Candidate magic: sparse — only ~popcount(mask) bits set on average.
 * This biases toward sparse magics which (empirically) work more often. */
static uint64_t sparse_magic(uint64_t *state) {
    return magic_rng(state) & magic_rng(state) & magic_rng(state);
}

/* Iterate over all sub-masks of `mask` (i.e. all subsets of the set bits).
 * Uses the standard "Gosper's hack" via the formula:
 *   sub = (sub - mask) & mask
 * starting from sub = 0. Visits all 2^popcount(mask) subsets in some order. */
static Bitboard next_submask(Bitboard sub, Bitboard mask) {
    return (sub - mask) & mask;
}

static void init_magics_for(MagicEntry *magics, Square sq,
                            Bitboard mask, Bitboard *table_start,
                            Bitboard (*slow_attacks)(Square, Bitboard),
                            unsigned *table_used_out) {
    int pop = __builtin_popcountll(mask);
    unsigned shift = 64 - (unsigned)pop;
    magics[sq].mask  = mask;
    magics[sq].shift = shift;
    magics[sq].attacks = table_start;

    /* Enumerate all 2^pop occupancies and their corresponding attacks. */
    unsigned n = 1u << pop;
    Bitboard *reference_attacks = (Bitboard *)malloc(n * sizeof(Bitboard));
    if (!reference_attacks) {
        fprintf(stderr, "init_magics: out of memory (n=%u)\n", n);
        exit(1);
    }

    Bitboard occ = 0;
    unsigned i = 0;
    do {
        reference_attacks[i] = slow_attacks(sq, occ);
        i++;
        occ = next_submask(occ, mask);
    } while (occ != 0 && i < n);

#if BB_USE_PEXT
    /* PEXT path: index = _pext_u64(occ & mask, mask). No magic search
     * needed; the table is laid out in PEXT order. */
    magics[sq].magic = 0;  /* unused */
    occ = 0;
    i = 0;
    do {
        /* _pext_u64(occ, mask) gives the same enumeration order as our
         * next_submask loop only if we walk sub-masks in the same order.
         * Verify by computing both indexes and asserting they match. */
        unsigned idx = (unsigned)_pext_u64(occ, mask);
        table_start[idx] = reference_attacks[i];
        i++;
        occ = next_submask(occ, mask);
    } while (occ != 0 && i < n);
    *table_used_out = n;
#else
    /* Classical path: find a magic that maps each occupancy to a unique
     * index. Try sparse-random candidates until one works. */
    uint64_t rng_state = 0x123456789ABCDEF0ULL ^ ((uint64_t)sq << 32) ^ mask;
    Bitboard *used_marker = (Bitboard *)calloc(n, sizeof(Bitboard));
    if (!used_marker) { fprintf(stderr, "init_magics: OOM\n"); exit(1); }

    for (int tries = 0; tries < 100000000; tries++) {
        Bitboard magic = sparse_magic(&rng_state);
        if (__builtin_popcountll((mask * magic) & 0xFF00000000000000ULL) < 6)
            continue;  /* magic-bits upper bits must have enough entropy */

        memset(used_marker, 0, n * sizeof(Bitboard));
        bool ok = true;
        unsigned max_idx = 0;
        occ = 0;
        i = 0;
        do {
            unsigned idx = (unsigned)(((occ & mask) * magic) >> shift);
            if (idx >= n) { ok = false; break; }
            if (used_marker[idx] && used_marker[idx] != reference_attacks[i]) {
                ok = false; break;
            }
            used_marker[idx] = reference_attacks[i];
            table_start[idx] = reference_attacks[i];
            if (idx > max_idx) max_idx = idx;
            i++;
            occ = next_submask(occ, mask);
        } while (occ != 0 && i < n);

        if (ok) {
            magics[sq].magic = magic;
            *table_used_out = max_idx + 1;
            free(used_marker);
            free(reference_attacks);
            return;
        }
    }
    fprintf(stderr, "init_magics: failed to find a magic for square %d "
            "(popcount=%d, n=%u)\n", sq, pop, n);
    exit(1);
#endif
}

static void init_magic_bitboards(void) {
    unsigned rook_offset  = 0;
    unsigned bishop_offset = 0;

    for (int sq = 0; sq < 64; sq++) {
        unsigned used = 0;
        init_magics_for(ROOK_MAGICS, (Square)sq, rook_mask((Square)sq),
                        &ROOK_ATTACK_TABLE[rook_offset],
                        slow_rook_attacks, &used);
        rook_offset += used;

        init_magics_for(BISHOP_MAGICS, (Square)sq, bishop_mask((Square)sq),
                        &BISHOP_ATTACK_TABLE[bishop_offset],
                        slow_bishop_attacks, &used);
        bishop_offset += used;
    }

    if (rook_offset > sizeof(ROOK_ATTACK_TABLE) / sizeof(Bitboard)) {
        fprintf(stderr, "init_magic_bitboards: rook table overflowed "
                "(%u > %zu)\n", rook_offset,
                sizeof(ROOK_ATTACK_TABLE) / sizeof(Bitboard));
        exit(1);
    }
    if (bishop_offset > sizeof(BISHOP_ATTACK_TABLE) / sizeof(Bitboard)) {
        fprintf(stderr, "init_magic_bitboards: bishop table overflowed "
                "(%u > %zu)\n", bishop_offset,
                sizeof(BISHOP_ATTACK_TABLE) / sizeof(Bitboard));
        exit(1);
    }
}

/* ──────────────────────────────────────────────
 *  Init helpers (non-slider tables)
 * ────────────────────────────────────────────── */
static void init_pawn_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = SQUARE_BB[sq];
        /* White pawns attack north-east and north-west */
        PAWN_ATTACKS[WHITE][sq] =
            ((b & ~FILE_BB[0]) << 7) | ((b & ~FILE_BB[7]) << 9);
        /* Black pawns attack south-east and south-west */
        PAWN_ATTACKS[BLACK][sq] =
            ((b & ~FILE_BB[7]) >> 7) | ((b & ~FILE_BB[0]) >> 9);
    }
}

static void init_knight_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = SQUARE_BB[sq];
        Bitboard nf = ~FILE_BB[0];   /* not A */
        Bitboard nG = ~FILE_BB[7];   /* not H */
        Bitboard nab = nf & ~FILE_BB[1];
        Bitboard ngh = nG & ~FILE_BB[6];
        KNIGHT_ATTACKS[sq] =
            ((b & nab) << 6)  | ((b & ngh) << 10) |
            ((b & nab) >> 10) | ((b & ngh) >> 6)  |
            ((b & nf)  << 15) | ((b & nG)  << 17) |
            ((b & nG)  >> 15) | ((b & nf)  >> 17);
    }
}

static void init_king_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = SQUARE_BB[sq];
        Bitboard nf = ~FILE_BB[0];
        Bitboard nG = ~FILE_BB[7];
        KING_ATTACKS[sq] =
            (b << 8) | (b >> 8) |
            ((b & nG) << 1) | ((b & nf) >> 1) |
            ((b & nG) << 9) | ((b & nf) << 7) |
            ((b & nf) >> 9) | ((b & nG) >> 7);
    }
}

static void init_between_line(void) {
    for (int a = 0; a < 64; a++) {
        for (int b = 0; b < 64; b++) {
            BETWEEN_BB[a][b] = 0;
            LINE_BB[a][b]    = 0;
            if (a == b) continue;

            /* Determine if a and b share a line */
            static const int dirs[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
            int fa = a & 7, ra = a >> 3;
            int fb = b & 7, rb = b >> 3;

            for (int d = 0; d < 4; d++) {
                int df = dirs[d][0], dr = dirs[d][1];
                /* Check if b lies on the ray from a in direction ±(df,dr) */
                int df2 = fb - fa, dr2 = rb - ra;
                bool same_line = false;
                if (df == 0 && dr == 0) continue;
                if (df == 0) { same_line = (df2 == 0 && dr != 0); }
                else if (dr == 0) { same_line = (dr2 == 0 && df2 != 0); }
                else { same_line = (df2 * dr == dr2 * df && df2 != 0); }

                if (!same_line) continue;

                /* Build LINE_BB */
                Bitboard line = SQUARE_BB[a] | SQUARE_BB[b];
                line |= slow_slider((Square)a, 0, df, dr);
                line |= slow_slider((Square)a, 0, -df, -dr);
                LINE_BB[a][b] = line;

                /* Build BETWEEN_BB */
                Bitboard between = 0;
                int f = fa, r = ra;
                int sf = (df2 > 0) - (df2 < 0), sr = (dr2 > 0) - (dr2 < 0);
                f += sf; r += sr;
                while (f != fb || r != rb) {
                    between |= SQUARE_BB[r * 8 + f];
                    f += sf; r += sr;
                }
                BETWEEN_BB[a][b] = between;
                break;
            }
        }
    }
}

/* ──────────────────────────────────────────────
 *  Public init
 * ────────────────────────────────────────────── */
void bitboard_init(void) {
    /* Square bitboards */
    for (int i = 0; i < 64; i++)
        SQUARE_BB[i] = (Bitboard)1 << i;

    /* File / rank masks */
    for (int i = 0; i < 8; i++) {
        FILE_BB[i] = 0x0101010101010101ULL << i;
        RANK_BB[i] = 0xFFULL << (i * 8);
    }

    /* Line masks through each square (used by init_between_line). */
    for (int sq = 0; sq < 64; sq++) {
        int f = sq & 7, r = sq >> 3;
        FILE_MASK[sq]  = FILE_BB[f];

        /* Diagonal (A1-H8 direction: delta = +9) */
        Bitboard diag = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f - r + i; /* canonical diagonal index */
            if (ff >= 0 && ff < 8) diag |= SQUARE_BB[i * 8 + ff];
        }
        DIAG_MASK[sq] = diag;

        /* Anti-diagonal (A8-H1 direction: delta = +7) */
        Bitboard adiag = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f + r - i;
            if (ff >= 0 && ff < 8) adiag |= SQUARE_BB[i * 8 + ff];
        }
        ADIAG_MASK[sq] = adiag;
    }

    init_magic_bitboards();
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_between_line();
}

/* ──────────────────────────────────────────────
 *  Public attack functions — single table lookup.
 *
 *  PEXT path:    index = _pext_u64(occ & mask, mask)
 *  Classical:    index = ((occ & mask) * magic) >> shift
 *
 *  Both produce identical attack bitboards.
 * ────────────────────────────────────────────── */
Bitboard rook_attacks(Square sq, Bitboard occ) {
    const MagicEntry *m = &ROOK_MAGICS[sq];
#if BB_USE_PEXT
    unsigned idx = (unsigned)_pext_u64(occ & m->mask, m->mask);
#else
    unsigned idx = (unsigned)(((occ & m->mask) * m->magic) >> m->shift);
#endif
    return m->attacks[idx];
}

Bitboard bishop_attacks(Square sq, Bitboard occ) {
    const MagicEntry *m = &BISHOP_MAGICS[sq];
#if BB_USE_PEXT
    unsigned idx = (unsigned)_pext_u64(occ & m->mask, m->mask);
#else
    unsigned idx = (unsigned)(((occ & m->mask) * m->magic) >> m->shift);
#endif
    return m->attacks[idx];
}

/* ──────────────────────────────────────────────
 *  Debug print
 * ────────────────────────────────────────────── */
void bb_print(Bitboard bb) {
    printf("  +-----------------+\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d | ", r + 1);
        for (int f = 0; f < 8; f++) {
            printf("%c ", (bb >> (r * 8 + f)) & 1 ? '1' : '.');
        }
        printf("|\n");
    }
    printf("  +-----------------+\n");
    printf("    a b c d e f g h\n\n");
}
