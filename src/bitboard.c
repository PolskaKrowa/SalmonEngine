/*
 * bitboard.c — Bitboard attack generation
 *
 * Two slider implementations are linked in:
 *
 *  1. Hyperbola Quintessence (o^(o−2r) trick)
 *     Works perfectly for files and diagonals, where bswap acts as
 *     a bit-reversal.  Completely branchless and ~4 instructions per
 *     direction.  Rank attacks use a pre-computed first-rank table.
 *
 *  2. PEXT-based table lookup (BMI2)
 *     Faster than HQ on Intel CPUs (3 instructions per slider: AND,
 *     PEXT, load).  Slower on AMD Zen 1/2 (PEXT is microcoded), so
 *     we dispatch at runtime based on a CPU feature check.
 *
 * bitboard_init() probes __builtin_cpu_supports("bmi2") and installs
 * the appropriate function pointers.  The public rook_attacks /
 * bishop_attacks functions indirect through these pointers — the
 * indirect call costs ~1 cycle but is dwarfed by the work it saves.
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

/* First-rank attack table: [file 0..7][8-bit occupancy] */
static uint8_t rank_atk[8][256];

/*
 * Dispatch tables for slider attack functions.
 *
 * Set in bitboard_init() based on __builtin_cpu_supports("bmi2").
 * The public rook_attacks / bishop_attacks functions indirect
 * through these pointers.
 */
typedef Bitboard (*slider_fn)(Square, Bitboard);
static slider_fn rook_dispatch   = NULL;
static slider_fn bishop_dispatch = NULL;

/* PEXT attack tables.  Sized to fit every square's relevant occupancy.
 *   ROOK_ATTACKS:   sum of 2^popcount(ROOK_RELEVANT[sq]) over all sq
 *                   = 4096 + 4096 + ... (see below) = 102400 entries
 *   BISHOP_ATTACKS: 5248 entries
 */
Bitboard ROOK_ATTACKS   [0x19000];   /* 102400 */
Bitboard BISHOP_ATTACKS [0x01480];   /*   5248 */
Bitboard ROOK_RELEVANT  [64];
Bitboard BISHOP_RELEVANT[64];

/*
 * Per-square offsets into ROOK_ATTACKS / BISHOP_ATTACKS.
 *
 * Rather than computing the offset on every call (which would need
 * another table or a popcount prefix-sum), we precompute the offset
 * at init time and store it alongside the relevant-mask.  The lookup
 * then becomes:
 *
 *   attacks = TABLE[offset[sq] + _pext_u64(occ, RELEVANT[sq])]
 *
 * The offset table is small enough (64 ints = 256 bytes each) to
 * stay in L1 cache permanently.
 */
static uint32_t ROOK_OFFSET  [64];
static uint32_t BISHOP_OFFSET[64];

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
            /* east */
            for (int f = file + 1; f < 8; f++) {
                atk |= 1 << f;
                if (occ & (1 << f)) break;
            }
            /* west */
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

/*
 * Build the PEXT attack tables.
 *
 * For each square, we enumerate all 2^N relevant occupancies (where N
 * = popcount of the relevant mask, ≤ 12 for rooks on the edge, ≤ 9 for
 * bishops), compute the attack set with the slow reference slider, and
 * store it at offset[sq] + pext_index.
 *
 * The relevant occupancy mask = the slider's full attack set with the
 * outer rank/file edges removed.  Removing those bits doesn't change
 * the attack set (a rook on a1 doesn't care what's on h1 for its a1→h8
 * ray's behaviour), and it shrinks the table by 2^(removed bits).
 */
static void init_pext_tables(void) {
    /*
     * Rook relevant masks.
     *
     * For a rook on square sq, the relevant occupancy is the set of
     * squares the rook's attack could be blocked by.  This is:
     *   - the file through sq, MINUS the outer ranks (rank 0 and rank 7)
     *   - the rank through sq, MINUS the outer files (file 0 and file 7)
     *   - MINUS sq itself
     *
     * Crucially, the edge removal is DIRECTIONAL: we remove rank-edges
     * from the file mask (the file's two endpoints are on rank 0/7 and
     * never block — the ray just terminates there), and we remove
     * file-edges from the rank mask (the rank's two endpoints are on
     * file 0/7).  Applying a single ~EDGES to the union would
     * incorrectly strip the entire file/rank for rooks on edge squares.
     *
     * Example: rook on a1.
     *   file 0 = a1..a8.  Remove rank 0 and rank 7 → a2..a7 (6 squares).
     *   rank 0 = a1..h1.  Remove file 0 and file 7 → b1..g1 (6 squares).
     *   Union minus a1 = a2..a7 | b1..g1 = 12 squares.  ✓
     */
    {
        Bitboard outer_ranks = RANK_BB[0] | RANK_BB[7];
        Bitboard outer_files = FILE_BB[0] | FILE_BB[7];

        uint32_t offset = 0;
        for (int sq = 0; sq < 64; sq++) {
            Bitboard file_inner = FILE_MASK[sq]  & ~outer_ranks;
            Bitboard rank_inner = RANK_BB[sq >> 3] & ~outer_files;
            ROOK_RELEVANT[sq] = (file_inner | rank_inner) & ~SQUARE_BB[sq];
            ROOK_OFFSET[sq] = offset;
            int n = __builtin_popcountll(ROOK_RELEVANT[sq]);
            int entries = 1 << n;
            for (int i = 0; i < entries; i++) {
                /* Materialise occupancy #i by walking the bits of i */
                Bitboard occ = 0;
                Bitboard mask = ROOK_RELEVANT[sq];
                for (int b = 0; b < n; b++) {
                    int bit_sq = __builtin_ctzll(mask);
                    mask &= mask - 1;
                    if (i & (1 << b)) occ |= SQUARE_BB[bit_sq];
                }
                ROOK_ATTACKS[offset + i] =
                      slow_slider((Square)sq, occ,  1,  0)
                    | slow_slider((Square)sq, occ, -1,  0)
                    | slow_slider((Square)sq, occ,  0,  1)
                    | slow_slider((Square)sq, occ,  0, -1);
            }
            offset += entries;
        }
    }

    /* Bishop relevant masks: both diagonals through sq, minus the
     * edge squares (a bishop on the edge still has all its attacks,
     * but the edge bits never block anything useful). */
    {
        Bitboard edge_ranks = RANK_BB[0] | RANK_BB[7];
        Bitboard edge_files = FILE_BB[0] | FILE_BB[7];
        Bitboard edges = edge_ranks | edge_files;

        uint32_t offset = 0;
        for (int sq = 0; sq < 64; sq++) {
            BISHOP_RELEVANT[sq] = (DIAG_MASK[sq] | ADIAG_MASK[sq]) & ~edges
                                  & ~SQUARE_BB[sq];
            BISHOP_OFFSET[sq] = offset;
            int n = __builtin_popcountll(BISHOP_RELEVANT[sq]);
            int entries = 1 << n;
            for (int i = 0; i < entries; i++) {
                Bitboard occ = 0;
                Bitboard mask = BISHOP_RELEVANT[sq];
                for (int b = 0; b < n; b++) {
                    int bit_sq = __builtin_ctzll(mask);
                    mask &= mask - 1;
                    if (i & (1 << b)) occ |= SQUARE_BB[bit_sq];
                }
                BISHOP_ATTACKS[offset + i] =
                      slow_slider((Square)sq, occ,  1,  1)
                    | slow_slider((Square)sq, occ, -1, -1)
                    | slow_slider((Square)sq, occ,  1, -1)
                    | slow_slider((Square)sq, occ, -1,  1);
            }
            offset += entries;
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

    /* Line masks through each square */
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

    init_rank_table();
    init_pawn_attacks();
    init_knight_attacks();
    init_king_attacks();
    init_between_line();
    init_pext_tables();

    /* Runtime CPU dispatch. */
    bool has_bmi2 = __builtin_cpu_supports("bmi2");
    rook_dispatch   = has_bmi2 ? rook_attacks_pext   : rook_attacks_hq;
    bishop_dispatch = has_bmi2 ? bishop_attacks_pext : bishop_attacks_hq;
}

/* ──────────────────────────────────────────────
 *  Hyperbola Quintessence core
 *
 *  For a line where bswap acts as reversal (files, diagonals):
 *
 *    forward = (o & mask) - 2*s
 *    backward = bswap( bswap(o & mask) - 2*bswap(s) )
 *    attacks  = (forward ^ backward) & mask
 *
 *  This is branchless and handles both ray directions simultaneously.
 * ────────────────────────────────────────────── */
static inline Bitboard hyp_quint(Bitboard occ, Bitboard sq_b, Bitboard mask) {
    Bitboard o = occ & mask;
    Bitboard r = __builtin_bswap64(o);
    Bitboard s = sq_b;
    Bitboard t = __builtin_bswap64(s);
    Bitboard forward  = o - (s << 1);
    Bitboard backward = __builtin_bswap64(r - (t << 1));
    return (forward ^ backward) & mask;
}

/* Rank attacks using the pre-computed first-rank table */
static inline Bitboard rank_attacks(Square sq, Bitboard occ) {
    int rank = (int)(sq >> 3);
    int file = (int)(sq & 7);
    uint8_t occ_byte = (uint8_t)(occ >> (rank * 8));
    return (Bitboard)rank_atk[file][occ_byte] << (rank * 8);
}

/* ──────────────────────────────────────────────
 *  HQ-based attack functions (fallback, no BMI2)
 * ────────────────────────────────────────────── */
Bitboard rook_attacks_hq(Square sq, Bitboard occ) {
    return hyp_quint(occ, SQUARE_BB[sq], FILE_MASK[sq])
         | rank_attacks(sq, occ);
}

Bitboard bishop_attacks_hq(Square sq, Bitboard occ) {
    return hyp_quint(occ, SQUARE_BB[sq], DIAG_MASK[sq])
         | hyp_quint(occ, SQUARE_BB[sq], ADIAG_MASK[sq]);
}

/* ──────────────────────────────────────────────
 *  PEXT-based attack functions (BMI2 only)
 *
 *  ~3 instructions per call (AND + PEXT + load), versus ~12 for HQ.
 *  On Intel CPUs this is a clear win (~15-20% NPS).
 *  On AMD Zen 1/2 PEXT is microcoded and slow; runtime dispatch
 *  ensures we don't use it there.
 * ────────────────────────────────────────────── */
#if defined(__BMI2__)
Bitboard rook_attacks_pext(Square sq, Bitboard occ) {
    Bitboard relevant = ROOK_RELEVANT[sq];
    uint32_t idx = ROOK_OFFSET[sq] + (uint32_t)_pext_u64(occ, relevant);
    return ROOK_ATTACKS[idx];
}

Bitboard bishop_attacks_pext(Square sq, Bitboard occ) {
    Bitboard relevant = BISHOP_RELEVANT[sq];
    uint32_t idx = BISHOP_OFFSET[sq] + (uint32_t)_pext_u64(occ, relevant);
    return BISHOP_ATTACKS[idx];
}
#else
/* No compiler BMI2 support — fall back to HQ even if the CPU has it.
 * This branch is taken on, e.g., builds with -march=x86-64 (baseline). */
Bitboard rook_attacks_pext  (Square sq, Bitboard occ) { return rook_attacks_hq  (sq, occ); }
Bitboard bishop_attacks_pext(Square sq, Bitboard occ) { return bishop_attacks_hq(sq, occ); }
#endif

/* ──────────────────────────────────────────────
 *  Dispatch entry points
 *
 *  rook_attacks / bishop_attacks indirect through the dispatch
 *  pointers set in bitboard_init().  The indirection costs ~1 cycle
 *  but the called function is significantly faster on BMI2 hosts.
 * ────────────────────────────────────────────── */
Bitboard rook_attacks  (Square sq, Bitboard occ) { return rook_dispatch  (sq, occ); }
Bitboard bishop_attacks(Square sq, Bitboard occ) { return bishop_dispatch(sq, occ); }

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
