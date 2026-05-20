/*
 * bitboard.c — Bitboard attack generation  (optimised)
 *
 * Sliding piece attacks are computed using two complementary techniques:
 *
 *  1. Hyperbola Quintessence (o^(o−2r) trick)
 *     Works perfectly for files and diagonals, where bswap acts as
 *     a bit-reversal (each square on those lines sits in a distinct byte).
 *     Completely branchless and ~4 instructions per direction.
 *
 *  2. First-rank attack table (rank_atk[file][8-bit occupancy])
 *     Handles east/west attacks.  The table is 8×256 = 2 048 bytes and
 *     fits entirely in L1 cache.
 *
 * Non-sliding pieces use simple pre-computed lookup tables.
 * 
 */

#include "bitboard.h"
#include <string.h>
#include <stdio.h>

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

static uint8_t rank_atk[8][256] __attribute__((aligned(64)));

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

/* ──────────────────────────────────────────────
 *  Init helpers
 * ────────────────────────────────────────────── */
static void init_rank_table(void) {
    for (int file = 0; file < 8; file++) {
        for (int occ = 0; occ < 256; occ++) {
            int atk = 0;
            for (int f = file + 1; f < 8; f++) {
                atk |= 1 << f;
                if (occ & (1 << f)) break;
            }
            for (int f = file - 1; f >= 0; f--) {
                atk |= 1 << f;
                if (occ & (1 << f)) break;
            }
            rank_atk[file][occ] = (uint8_t)atk;
        }
    }
}

static void init_pawn_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = SQUARE_BB[sq];
        PAWN_ATTACKS[WHITE][sq] =
            ((b & ~FILE_BB[0]) << 7) | ((b & ~FILE_BB[7]) << 9);
        PAWN_ATTACKS[BLACK][sq] =
            ((b & ~FILE_BB[7]) >> 7) | ((b & ~FILE_BB[0]) >> 9);
    }
}

static void init_knight_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b   = SQUARE_BB[sq];
        Bitboard nA  = ~FILE_BB[0];
        Bitboard nH  = ~FILE_BB[7];
        Bitboard nAB = nA & ~FILE_BB[1];
        Bitboard nGH = nH & ~FILE_BB[6];
        KNIGHT_ATTACKS[sq] =
            ((b & nAB) << 6)  | ((b & nGH) << 10) |
            ((b & nAB) >> 10) | ((b & nGH) >> 6)  |
            ((b & nA)  << 15) | ((b & nH)  << 17) |
            ((b & nH)  >> 15) | ((b & nA)  >> 17);
    }
}

static void init_king_attacks(void) {
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b  = SQUARE_BB[sq];
        Bitboard nA = ~FILE_BB[0];
        Bitboard nH = ~FILE_BB[7];
        KING_ATTACKS[sq] =
            (b << 8) | (b >> 8) |
            ((b & nH) << 1) | ((b & nA) >> 1) |
            ((b & nH) << 9) | ((b & nA) << 7) |
            ((b & nA) >> 9) | ((b & nH) >> 7);
    }
}

static void init_between_line(void) {
    static const int dirs[4][2] = {{1,0},{0,1},{1,1},{1,-1}};

    for (int a = 0; a < 64; a++) {
        for (int b = 0; b < 64; b++) {
            BETWEEN_BB[a][b] = 0;
            LINE_BB[a][b]    = 0;
            if (a == b) continue;

            int fa = a & 7, ra = a >> 3;
            int fb = b & 7, rb = b >> 3;

            for (int d = 0; d < 4; d++) {
                int df = dirs[d][0], dr = dirs[d][1];
                int df2 = fb - fa, dr2 = rb - ra;
                bool same_line = false;
                if (df == 0 && dr == 0) continue;
                if      (df == 0) { same_line = (df2 == 0 && dr != 0); }
                else if (dr == 0) { same_line = (dr2 == 0 && df2 != 0); }
                else              { same_line = (df2 * dr == dr2 * df && df2 != 0); }

                if (!same_line) continue;

                Bitboard line = SQUARE_BB[a] | SQUARE_BB[b];
                line |= slow_slider((Square)a, 0,  df,  dr);
                line |= slow_slider((Square)a, 0, -df, -dr);
                LINE_BB[a][b] = line;

                Bitboard between = 0;
                int f = fa, r = ra;
                int sf = (df2 > 0) - (df2 < 0);
                int sr = (dr2 > 0) - (dr2 < 0);
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
    for (int i = 0; i < 64; i++)
        SQUARE_BB[i] = (Bitboard)1 << i;

    for (int i = 0; i < 8; i++) {
        FILE_BB[i] = 0x0101010101010101ULL << i;
        RANK_BB[i] = 0xFFULL << (i * 8);
    }

    for (int sq = 0; sq < 64; sq++) {
        int f = sq & 7, r = sq >> 3;
        FILE_MASK[sq] = FILE_BB[f];

        Bitboard diag = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f - r + i;
            if (ff >= 0 && ff < 8) diag |= SQUARE_BB[i * 8 + ff];
        }
        DIAG_MASK[sq] = diag;

        Bitboard adiag = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f + r - i;
            if (ff >= 0 && ff < 8) adiag |= SQUARE_BB[i * 8 + ff];
        }
        ADIAG_MASK[sq] = adiag;
    }

    init_rank_table();
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_between_line();
}

/* ──────────────────────────────────────────────
 *  Hyperbola Quintessence core
 *
 *    forward  = (o & mask) - 2*s
 *    backward = bswap( bswap(o & mask) - 2*bswap(s) )
 *    attacks  = (forward ^ backward) & mask
 * ────────────────────────────────────────────── */
static inline Bitboard hyp_quint(Bitboard occ, Bitboard sq_b, Bitboard mask) {
    Bitboard o = occ & mask;
    Bitboard r = __builtin_bswap64(o);
    Bitboard t = __builtin_bswap64(sq_b);
    Bitboard fwd = o - (sq_b << 1);
    Bitboard bwd = __builtin_bswap64(r - (t << 1));
    return (fwd ^ bwd) & mask;
}

static inline Bitboard rank_attacks(Square sq, Bitboard occ) {
    int shift  = (int)sq & 0x38;                       /* rank * 8        */
    uint8_t o8 = (uint8_t)(occ >> shift);              /* rank occupancy  */
    return (Bitboard)rank_atk[(int)sq & 7][o8] << shift;
}

/* ──────────────────────────────────────────────
 *  Public attack functions
 * ────────────────────────────────────────────── */

Bitboard rook_attacks(Square sq, Bitboard occ) {
    Bitboard sq_b = SQUARE_BB[sq];
    return hyp_quint(occ, sq_b, FILE_MASK[sq])
         | rank_attacks(sq, occ);
}

Bitboard bishop_attacks(Square sq, Bitboard occ) {
    Bitboard sq_b = SQUARE_BB[sq];
    return hyp_quint(occ, sq_b, DIAG_MASK[sq])
         | hyp_quint(occ, sq_b, ADIAG_MASK[sq]);
}

Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

/* ─────────────────────────────────────────────
 *  Debug print
 * ────────────────────────────────────────────── */
void bb_print(Bitboard bb) {
    printf("  +-----------------+\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d | ", r + 1);
        for (int f = 0; f < 8; f++)
            printf("%c ", (bb >> (r * 8 + f)) & 1 ? '1' : '.');
        printf("|\n");
    }
    printf("  +-----------------+\n");
    printf("    a b c d e f g h\n\n");
}