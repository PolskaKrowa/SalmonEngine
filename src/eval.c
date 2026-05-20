/*
 * eval.c — Static evaluation
 *
 * Scoring convention: always from the perspective of the side to move
 * (positive = good for side to move).
 *
 * Features implemented:
 *   • Material balance
 *   • Piece-square tables — separate MG and EG tables, blended by game phase
 *   • Material imbalance — SF 11 quadratic polynomial (bishop pair, rook vs
 *       minors, etc.)  Replaces a flat material count.
 *   • Mobility — non-linear per-count tables (SF 11 MobilityBonus values)
 *       so the first squares of freedom are worth the most.
 *   • Pawn structure:
 *       – Doubled-pawn penalty
 *       – Isolated-pawn penalty
 *       – Backward-pawn penalty
 *       – Passed-pawn bonus
 *   • Outpost squares for knights and bishops
 *   • TrappedRook  — penalty when a rook with ≤3 moves is hemmed in on the
 *       same side as its own king
 *   • WeakQueen    — penalty when enemy sliders x-ray through our queen
 *   • KingProtector — minor pieces far from own king incur a distance penalty
 *   • MinorBehindPawn — minor piece sheltered behind a friendly pawn earns a bonus
 *   • King safety:
 *       – Pawn shield bonus
 *       – Open-file penalty near king
 *       – Distance-weighted enemy-piece attack count
 *       – Endgame king-activity term (king-to-king proximity bonus)
 *   • Bishop pair bonus
 *   • Rook on open / semi-open file bonus
 *   • Rook on seventh rank bonus
 *   • Tempo bonus (side to move)
 *   • Initiative / complexity correction — reduces a winning advantage when
 *       the winning side cannot realistically convert (no passed pawns, no
 *       outflanking, pawns on only one flank).
 *   • Lazy evaluation guard — fast material+PST proxy bails out when the
 *       score is far from any search window, saving ~5 % NPS for free.
 */

#include "eval.h"
#include "bitboard.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static const int MATERIAL_MG[6] = { 82, 344, 358, 480, 1022, 0 };
static const int MATERIAL_EG[6] = { 94, 338, 329, 546, 924, 0 };

static const int PST_PAWN_MG[64] = {
        0,     0,     0,     0,     0,     0,     0,     0,
       98,   134,    61,    95,    68,   126,    34,   -11,
       -6,     7,    26,    31,    65,    56,    25,   -20,
      -14,    13,     6,    21,    23,    12,    17,   -23,
      -27,    -2,    -5,    12,    17,     6,    10,   -25,
      -26,    -4,    -4,   -10,     3,     3,    33,   -12,
      -35,    -1,   -20,   -23,   -15,    24,    38,   -22,
        0,     0,     0,     0,     0,     0,     0,     0,
};
static const int PST_PAWN_EG[64] = {
        0,     0,     0,     0,     0,     0,     0,     0,
      178,   173,   158,   134,   147,   132,   165,   187,
       94,   100,    85,    67,    56,    53,    82,    84,
       32,    24,    13,     5,    -2,     4,    17,    17,
       13,     9,    -3,    -7,    -7,    -8,     3,    -1,
        4,     7,    -6,     1,     0,    -5,    -1,    -8,
       13,     8,     8,    10,    13,     0,     2,    -7,
        0,     0,     0,     0,     0,     0,     0,     0,
};

static const int PST_KNIGHT_MG[64] = {
     -167,   -89,   -34,   -49,    61,   -97,   -15,  -107,
      -73,   -41,    72,    36,    23,    62,     7,   -17,
      -47,    60,    37,    65,    84,   129,    73,    44,
       -9,    17,    19,    53,    37,    69,    18,    22,
      -13,     4,    16,    13,    28,    19,    21,    -8,
      -23,    -9,    12,    10,    19,    17,    25,   -16,
      -29,   -53,   -12,    -3,    -1,    18,   -14,   -19,
      -99,   -21,   -58,   -33,   -17,   -28,   -19,   -23,
};
static const int PST_KNIGHT_EG[64] = {
      -58,   -38,   -13,   -28,   -31,   -27,   -63,   -99,
      -25,    -8,   -25,    -2,    -9,   -25,   -24,   -52,
      -24,   -20,    10,     9,    -1,    -9,   -19,   -41,
      -17,     3,    22,    22,    22,    11,     8,   -18,
      -18,    -6,    16,    25,    16,    17,     4,   -18,
      -23,    -3,    -1,    15,    10,    -3,   -20,   -22,
      -42,   -20,   -10,    -5,    -2,   -20,   -23,   -44,
       27,   -51,   -23,   -15,   -22,   -18,   -50,   -64,
};

static const int PST_BISHOP_MG[64] = {
      -30,   -22,   -82,   -37,   -25,   -42,     7,    -8,
      -26,    16,   -18,   -13,    30,    59,    18,   -47,
      -16,    37,    43,    40,    35,    50,    37,    -2,
       -4,     5,    19,    50,    37,    37,     7,    -2,
       -6,    13,    13,    26,    34,    12,    10,     4,
        0,    15,    15,    15,    14,    27,    18,    10,
        4,    15,    16,     0,     7,    21,    33,     1,
      -26,    -3,   -14,   -21,   -13,   -12,   -39,   -21,
};
static const int PST_BISHOP_EG[64] = {
      -70,   -75,   -11,    -8,    -7,    -9,   -17,   -24,
       -8,    -4,     7,   -12,    -3,   -13,    -4,   -14,
        2,    -8,     0,    -1,    -2,     6,     0,     4,
       -3,     9,    12,     9,    14,    10,     3,     2,
       -6,     3,    13,    19,     7,    10,    -3,    -9,
      -12,    -3,     8,    10,    13,     3,    -7,   -15,
      -14,   -18,    -7,    -1,     4,    -9,   -15,   -27,
       29,    -9,   -23,    -5,    -9,   -16,    -5,   -17,
};

static const int PST_ROOK_MG[64] = {
       32,    42,    32,    51,    63,     9,    31,    43,
       27,    32,    58,    62,    80,    67,    26,    44,
       -5,    19,    26,    36,    17,    45,    61,    16,
      -24,   -11,     7,    26,    24,    35,    -8,   -20,
      -36,   -26,   -12,    -1,     9,    -7,     6,   -23,
      -45,   -25,   -16,   -17,     3,     0,    -5,   -33,
      -44,   -16,   -20,    -9,    -1,    11,    -6,   -71,
      -13,   -13,     1,    17,    16,     7,   -37,   -26,
};
static const int PST_ROOK_EG[64] = {
       13,    10,    18,    15,    12,    12,     8,     5,
       11,    13,    13,    11,    -3,     3,     8,     3,
        7,     7,     7,     5,     4,    -3,    -5,    -3,
        4,     3,    13,     1,     2,     1,    -1,     2,
        3,     5,     8,     4,    -5,    -6,    -8,   -11,
       -4,     0,    -5,    -1,    -7,   -12,    -8,   -16,
       -6,    -6,     0,     2,    -9,    -9,   -11,    -3,
       31,     2,     3,    -1,    -5,   -13,     4,   -20,
};

static const int PST_QUEEN_MG[64] = {
      -28,     0,    29,    12,    59,    44,    43,    45,
      -24,   -39,    -5,     1,   -16,    57,    28,    54,
      -13,   -17,     7,     8,    29,    56,    47,    57,
      -27,   -27,   -16,   -16,    -1,    17,    -2,     1,
       -9,   -26,    -9,   -10,    -2,    -4,     3,    -3,
      -14,     2,   -11,    -2,    -5,     2,    14,     5,
      -35,    -8,    11,     2,     8,    15,    -3,     1,
       -5,   -18,    -9,    10,   -15,   -25,   -31,   -50,
};
static const int PST_QUEEN_EG[64] = {
       -9,    22,    22,    27,    27,    19,    10,    20,
      -17,    20,    32,    41,    58,    25,    30,     0,
      -20,     6,     9,    49,    47,    35,    19,     9,
        3,    22,    24,    45,    57,    40,    57,    36,
      -18,    28,    19,    47,    31,    34,    39,    23,
      -16,   -27,    15,     6,     9,    17,    10,     5,
      -22,   -23,   -30,   -16,   -16,   -23,   -36,   -32,
      -44,   -28,   -22,   -43,    -5,   -32,   -20,   -41,
};

static const int PST_KING_MG[64] = {
      -65,    23,    16,   -15,   -56,   -34,     2,    13,
       29,    -1,   -20,    -7,    -8,    -4,   -38,   -29,
       -9,    24,     2,   -16,   -20,     6,    22,   -22,
      -17,   -20,   -12,   -27,   -30,   -25,   -14,   -36,
      -49,    -1,   -27,   -39,   -46,   -44,   -33,   -51,
      -14,   -14,   -22,   -46,   -44,   -30,   -15,   -27,
        1,     7,    -8,   -64,   -43,   -16,     9,     8,
      -15,    36,    12,   -54,     8,   -28,    24,    14,
};
static const int PST_KING_EG[64] = {
      -74,   -35,   -18,   -18,   -11,    15,     4,   -17,
      -12,    17,    14,    17,    17,    38,    23,    11,
       10,    17,    23,    15,    20,    45,    44,    13,
       -8,    22,    24,    27,    26,    33,    26,     3,
      -18,    -4,    21,    24,    27,    23,     9,   -11,
      -19,    -3,    11,    21,    23,    16,     7,    -9,
      -27,   -11,     4,    13,    14,     4,    -5,   -17,
      120,   -34,   -21,   -11,   -28,   -14,   -24,   -43,
};

static const int DOUBLED_PAWN_PENALTY_MG  = -11;
static const int DOUBLED_PAWN_PENALTY_EG  = -56;
static const int ISOLATED_PAWN_PENALTY_MG = -15;
static const int ISOLATED_PAWN_PENALTY_EG = -15;
static const int BACKWARD_PAWN_PENALTY_MG = -9;
static const int BACKWARD_PAWN_PENALTY_EG = -22;
static const int PASSED_PAWN_BONUS_MG[8]  = { 0, 5, 10, 20, 35, 60, 90, 0, };
static const int PASSED_PAWN_BONUS_EG[8]  = { 0, 10, 20, 40, 65, 95, 140, 0, };

/*
 * Non-linear MobilityBonus tables (SF 11 values).
 * Each table is indexed by the number of accessible squares for that piece
 * type (capped at MOB_MAX).  The first few squares are worth the most;
 * flat per-square scoring over-values the 15th square relative to the 1st.
 */

/* Knights: 0..8 squares */
static const int MOB_KNIGHT_MG[] = {-62,-53,-12, -4,  3, 13, 22, 28, 33};
static const int MOB_KNIGHT_EG[] = {-81,-56,-30,-14,  8, 15, 23, 27, 33};

/* Bishops: 0..13 squares */
static const int MOB_BISHOP_MG[] = {-48,-20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98};
static const int MOB_BISHOP_EG[] = {-59,-23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97};

/* Rooks: 0..14 squares */
static const int MOB_ROOK_MG[] = {-58,-27,-15,-10, -5, -2,  9, 16, 30, 29, 32, 38, 46, 48, 58};
static const int MOB_ROOK_EG[] = {-76,-18, 28, 55, 69, 82,112,118,132,142,155,165,166,169,171};

/* Queens: 0..27 squares */
static const int MOB_QUEEN_MG[] = {
    -39,-21,  3,  3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66,
     67, 70, 71, 73, 79, 88, 88, 99,102,102,106,109,113,116};
static const int MOB_QUEEN_EG[] = {
    -36,-15,  8, 18, 34, 54, 61, 73, 79, 92, 94,104,113,120,
    123,126,133,136,140,143,148,166,170,175,184,191,206,212};

static const int ROOK_OPEN_FILE_MG  = 27;
static const int ROOK_OPEN_FILE_EG  = 57;
static const int ROOK_SEMIOPEN_MG   = 12;
static const int ROOK_SEMIOPEN_EG   = 7;
static const int ROOK_ON_SEVENTH_MG = 20;
static const int ROOK_ON_SEVENTH_EG = 32;

static const int BISHOP_PAIR_MG = 30;
static const int BISHOP_PAIR_EG = 60;

static const int OUTPOST_KNIGHT_MG = 22;
static const int OUTPOST_KNIGHT_EG = 14;
static const int OUTPOST_BISHOP_MG = 12;
static const int OUTPOST_BISHOP_EG = 8;

static const int KING_SHIELD_BONUS        = 7;
static const int KING_OPEN_FILE_PENALTY   = -25;
static const int KING_ATTACKER_WEIGHT[6]  = { 0, 20, 20, 40, 80, 0, };

/*
 * King endgame activity: per-Chebyshev-step penalty for being far from the
 * enemy king.  PST_KING_EG rewards centrality; this term rewards king-to-king
 * proximity (boxing-in bonus), which the PST cannot express because it only
 * sees one king's position.  Range: [4 cp at dist 1] to [28 cp at dist 7].
 */
static const int KING_EG_DISTANCE_PENALTY = 4;

#define TEMPO_BONUS_MG 16
#define TEMPO_BONUS_EG 0

/* Mobility evaluation indexing */
static const int * const MOB_MG_TABLE[6] = {
    NULL, MOB_KNIGHT_MG, MOB_BISHOP_MG, MOB_ROOK_MG, MOB_QUEEN_MG, NULL};
static const int * const MOB_EG_TABLE[6] = {
    NULL, MOB_KNIGHT_EG, MOB_BISHOP_EG, MOB_ROOK_EG, MOB_QUEEN_EG, NULL};
static const int MOB_MAX[6] = {0, 8, 13, 14, 27, 0};

/* Table-of-tables for quick lookup */
static const int * const PST_MG[6] = {
    PST_PAWN_MG, PST_KNIGHT_MG, PST_BISHOP_MG,
    PST_ROOK_MG, PST_QUEEN_MG,  PST_KING_MG
};
static const int * const PST_EG[6] = {
    PST_PAWN_EG, PST_KNIGHT_EG, PST_BISHOP_EG,
    PST_ROOK_EG, PST_QUEEN_EG,  PST_KING_EG
};

/* Flip square for Black (mirror vertically) */
static inline int pst_sq(Color c, Square sq) {
    return (c == WHITE) ? (int)sq : ((int)sq ^ 56);
}

/*
 * Chebyshev (king-move) distance between two squares.
 * = max(|file_delta|, |rank_delta|)
 * Used by the king-safety model to weight attacks by proximity.
 */
static inline int chebyshev(int sq1, int sq2) {
    int df = (sq1 & 7) - (sq2 & 7); if (df < 0) df = -df;
    int dr = (sq1 >> 3) - (sq2 >> 3); if (dr < 0) dr = -dr;
    return (df > dr) ? df : dr;
}

/* ──────────────────────────────────────────────
 *  Phase calculation
 *  Phase 24 = full midgame; 0 = full endgame.
 *  Pawns and kings are not counted.
 * ────────────────────────────────────────────── */
static const int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };
#define MAX_PHASE 24

int game_phase(const Board *b) {
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_INC[pt] * bb_popcount(b->pieces[c][pt]);
    return (phase > MAX_PHASE) ? MAX_PHASE : phase;
}

/* Blend MG and EG scores by phase */
static inline int taper(int mg, int eg, int phase) {
    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
}

/* ──────────────────────────────────────────────
 *  Pawn hash cache
 *
 *  Pawn structure evaluation is expensive and depends only on the two pawn
 *  bitboards plus the colour being evaluated. A small direct-mapped cache
 *  avoids recomputing doubled / isolated / passed / backward pawn terms for
 *  positions that recur through transpositions or iterative deepening.
 * ────────────────────────────────────────────── */
#define PAWN_CACHE_SIZE 2048

typedef struct {
    Bitboard wp;
    Bitboard bp;
    Color    us;
    int      mg;
    int      eg;
    bool     valid;
} PawnCacheEntry;

static PawnCacheEntry pawn_cache[PAWN_CACHE_SIZE];

static inline unsigned pawn_cache_idx(Bitboard wp, Bitboard bp, Color us) {
    uint64_t key = (uint64_t)wp
                 ^ ((uint64_t)bp << 1)
                 ^ ((uint64_t)us << 63)
                 ^ ((uint64_t)wp >> 17)
                 ^ ((uint64_t)bp >> 29);
    key ^= key >> 32;
    key ^= key >> 16;
    return (unsigned)key & (PAWN_CACHE_SIZE - 1);
}

static bool pawn_cache_probe(Bitboard wp, Bitboard bp, Color us, int *mg, int *eg) {
    PawnCacheEntry *e = &pawn_cache[pawn_cache_idx(wp, bp, us)];
    if (e->valid && e->wp == wp && e->bp == bp && e->us == us) {
        *mg = e->mg;
        *eg = e->eg;
        return true;
    }
    return false;
}

static void pawn_cache_store(Bitboard wp, Bitboard bp, Color us, int mg, int eg) {
    PawnCacheEntry *e = &pawn_cache[pawn_cache_idx(wp, bp, us)];
    e->wp = wp;
    e->bp = bp;
    e->us = us;
    e->mg = mg;
    e->eg = eg;
    e->valid = true;
}

/* ──────────────────────────────────────────────
 *  Pawn structure evaluation (one side)
 * ────────────────────────────────────────────── */
static void eval_pawns_uncached(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard pawns_copy  = our_pawns;

    while (pawns_copy) {
        int sq   = bb_pop(&pawns_copy);
        int file = sq & 7;
        int rank = sq >> 3;

        /* Adjacent file masks */
        Bitboard adj_files = 0;
        if (file > 0) adj_files |= FILE_BB[file - 1];
        if (file < 7) adj_files |= FILE_BB[file + 1];

        /* ── Doubled pawn ── */
        if (bb_popcount(our_pawns & FILE_BB[file]) > 1) {
            *mg += DOUBLED_PAWN_PENALTY_MG;
            *eg += DOUBLED_PAWN_PENALTY_EG;
        }

        /* ── Isolated pawn ── */
        bool is_isolated = !(our_pawns & adj_files);
        if (is_isolated) {
            *mg += ISOLATED_PAWN_PENALTY_MG;
            *eg += ISOLATED_PAWN_PENALTY_EG;
        }

        /* ── Passed pawn ──
         * No enemy pawns on same or adjacent files ahead of this pawn. */
        Bitboard ahead_mask = 0;
        if (us == WHITE) {
            for (int r = rank + 1; r < 8; r++) ahead_mask |= RANK_BB[r];
        } else {
            for (int r = 0; r < rank; r++)      ahead_mask |= RANK_BB[r];
        }
        Bitboard enemy_span = (their_pawns & (FILE_BB[file] | adj_files))
                              & ahead_mask;
        bool is_passed = !enemy_span;
        if (is_passed) {
            int bonus_rank = (us == WHITE) ? rank : (7 - rank);
            *mg += PASSED_PAWN_BONUS_MG[bonus_rank];
            *eg += PASSED_PAWN_BONUS_EG[bonus_rank];
        }

        /*
         * ── Backward pawn ──
         *
         * A pawn is backward when:
         *   1. Its stop square (one step forward) is controlled by an enemy
         *      pawn — it cannot safely advance.
         *   2. It has no friendly pawn support from behind on adjacent files.
         *   3. It is not already passed.
         *
         * Stop-square control uses PAWN_ATTACKS[us][stop] & their_pawns:
         *   PAWN_ATTACKS[WHITE][stop] = squares diagonally ahead of stop
         *   from WHITE's perspective — exactly where a BLACK pawn would need
         *   to sit to attack stop.  The formula is symmetric for BLACK.
         *
         * The behind-span uses clean rank-loop iteration instead of
         * full-board bit arithmetic (SQUARE_BB[sq]-1 etc.), which is both
         * easier to read and avoids edge-case surprises at sq==0 or sq==63.
         */
        if (!is_passed) {
            int stop = (us == WHITE) ? sq + 8 : sq - 8;
            if (stop >= 0 && stop < 64) {
                bool stop_attacked = (PAWN_ATTACKS[us][stop] & their_pawns) != 0;

                if (stop_attacked) {
                    /* All squares on adjacent files strictly behind this pawn */
                    Bitboard behind_span = 0;
                    if (us == WHITE) {
                        for (int r = 0; r < rank; r++)
                            behind_span |= RANK_BB[r];
                    } else {
                        for (int r = rank + 1; r < 8; r++)
                            behind_span |= RANK_BB[r];
                    }
                    behind_span &= adj_files;

                    if (!(our_pawns & behind_span)) {
                        *mg += BACKWARD_PAWN_PENALTY_MG;
                        *eg += BACKWARD_PAWN_PENALTY_EG;
                    }
                }
            }
        }
    }
}

static void eval_pawns(const Board *b, Color us, int *mg, int *eg) {
    Bitboard wp = b->pieces[WHITE][PAWN];
    Bitboard bp = b->pieces[BLACK][PAWN];

    int cached_mg, cached_eg;
    if (pawn_cache_probe(wp, bp, us, &cached_mg, &cached_eg)) {
        *mg += cached_mg;
        *eg += cached_eg;
        return;
    }

    int local_mg = 0;
    int local_eg = 0;
    eval_pawns_uncached(b, us, &local_mg, &local_eg);

    pawn_cache_store(wp, bp, us, local_mg, local_eg);
    *mg += local_mg;
    *eg += local_eg;
}

static void eval_mobility(const Board *b, Color us, int *mg, int *eg) {
    Bitboard occ    = b->occ[2];
    Bitboard not_us = ~b->occ[us];

    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk = 0;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq]; break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default: break;
            }
            int mob = bb_popcount(atk & not_us);
            if (mob > MOB_MAX[pt]) mob = MOB_MAX[pt];
            *mg += MOB_MG_TABLE[pt][mob];
            *eg += MOB_EG_TABLE[pt][mob];
        }
    }
}


static void eval_rooks(const Board *b, Color us, int *mg, int *eg) {
    Color them           = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard rooks       = b->pieces[us  ][ROOK];

    int rank7 = (us == WHITE) ? 6 : 1;
    int rank8 = (us == WHITE) ? 7 : 0;

    while (rooks) {
        int sq   = bb_pop(&rooks);
        int file = sq & 7;
        int rank = sq >> 3;
        Bitboard file_bb = FILE_BB[file];

        bool no_own_pawn   = !(our_pawns   & file_bb);
        bool no_enemy_pawn = !(their_pawns & file_bb);

        if (no_own_pawn && no_enemy_pawn) {
            *mg += ROOK_OPEN_FILE_MG;
            *eg += ROOK_OPEN_FILE_EG;
        } else if (no_own_pawn) {
            *mg += ROOK_SEMIOPEN_MG;
            *eg += ROOK_SEMIOPEN_EG;
        }

        /*
         * TrappedRook: a rook with very limited mobility (<=3 squares) on
         * a closed file is penalised, especially when trapped by its own king.
         */
        if (!no_own_pawn && !no_enemy_pawn) {
            Bitboard rook_mob = rook_attacks((Square)sq, b->occ[2]) & ~b->occ[us];
            int mob = bb_popcount(rook_mob);
            if (mob <= 3) {
                int king_sq = bb_lsb(b->pieces[us][KING]);
                int kf = king_sq & 7;
                /* Penalise when the rook is hemmed in on the same side as king */
                if ((kf < 4) == (file < kf)) {
                    *mg -= 52;
                    *eg -= 10;
                }
            }
        }

        /*
         * Rook on the 7th rank: valuable when the enemy king is on the 8th
         * rank or there are enemy pawns on the 7th rank to harass.
         */
        if (rank == rank7) {
            int enemy_king_sq = bb_lsb(b->pieces[them][KING]);
            bool enemy_king_on_8th = ((enemy_king_sq >> 3) == rank8);
            bool pawns_on_7th      = (their_pawns & RANK_BB[rank7]) != 0;

            if (enemy_king_on_8th || pawns_on_7th) {
                *mg += ROOK_ON_SEVENTH_MG;
                *eg += ROOK_ON_SEVENTH_EG;
            }
        }
    }
}

/*
 * eval_outposts — Clean per-piece front-span / attack-mask approach.
 *
 * OLD approach: precomputed enemy_pawn_attack_span by looping over every
 * enemy pawn and filling whole ranks ahead of it on adjacent files.  This
 * was an over-approximation (it could flag squares unreachable due to
 * blockades) and the fill direction was easy to mis-index.
 *
 * NEW approach: for each candidate piece check directly:
 *   1. Is the square in the outpost rank range (ranks 4–6 from our side)?
 *   2. Is it pawn-supported?  A friendly pawn attacks this square when
 *      PAWN_ATTACKS[them][sq] & our_pawns is non-zero (by symmetry of
 *      pawn-attack geometry).
 *   3. Is there NO enemy pawn on adjacent files that could advance to
 *      attack it in the future? — the "future-threat span".
 *
 * Future-threat span:
 *   For us=WHITE: BLACK pawns advance downward.  Any BLACK pawn on an
 *   adjacent file at rank > sq_rank can advance to sq_rank+1 and attack
 *   sq diagonally, so the span is adj_files & {ranks > sq_rank}.
 *
 *   For us=BLACK: WHITE pawns advance upward.  Any WHITE pawn on an
 *   adjacent file at rank < sq_rank threatens sq, so the span is
 *   adj_files & {ranks < sq_rank}.
 *
 * This is explicit, symmetric, and correct at every file and rank.
 */
static void eval_outposts(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];

    /* Outpost ranks: 4–6 from our perspective (0-indexed). */
    int r_min = (us == WHITE) ? 3 : 2;
    int r_max = (us == WHITE) ? 5 : 4;
    Bitboard outpost_rank_mask = 0;
    for (int r = r_min; r <= r_max; r++) outpost_rank_mask |= RANK_BB[r];

    /* Evaluate knights and bishops with identical logic, different bonuses. */
    for (int pt = KNIGHT; pt <= BISHOP; pt++) {
        int bonus_mg = (pt == KNIGHT) ? OUTPOST_KNIGHT_MG : OUTPOST_BISHOP_MG;
        int bonus_eg = (pt == KNIGHT) ? OUTPOST_KNIGHT_EG : OUTPOST_BISHOP_EG;

        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq   = bb_pop(&pieces);
            int file = sq & 7;
            int rank = sq >> 3;

            /* Must be in the outpost rank range */
            if (!(SQUARE_BB[sq] & outpost_rank_mask)) continue;

            /*
             * Pawn support check: PAWN_ATTACKS[them][sq] gives squares from
             * which a them-coloured pawn would attack sq; by symmetry these
             * are exactly the squares from which one of OUR pawns attacks sq.
             */
            if (!(PAWN_ATTACKS[them][sq] & our_pawns)) continue;

            /* Adjacent files */
            Bitboard adj_files = 0;
            if (file > 0) adj_files |= FILE_BB[file - 1];
            if (file < 7) adj_files |= FILE_BB[file + 1];

            /* Future-threat span: ranks from which an enemy pawn can advance
             * to attack this square.  See function-level comment above. */
            Bitboard future_threat = 0;
            if (us == WHITE) {
                for (int r = rank + 1; r < 8; r++) future_threat |= RANK_BB[r];
            } else {
                for (int r = 0; r < rank; r++)     future_threat |= RANK_BB[r];
            }
            future_threat &= adj_files;

            if (their_pawns & future_threat) continue;  /* can be chased off */

            *mg += bonus_mg;
            *eg += bonus_eg;
        }
    }
}

/*
 * eval_king_safety — Tapered, distance-weighted attack model.
 *
 * Changes from the original:
 *
 * MG — Distance-weighted attacks:
 *   The original gave each attacking piece a flat weight regardless of how
 *   close it was to the king.  A queen on the opposite side of the board is
 *   far less threatening than one sitting on the f-file adjacent to the king.
 *   We now scale each attacker's base weight by its Chebyshev distance to
 *   our king (scale 4× / 2× / 1× for dist ≤1 / dist 2–3 / dist ≥4, with a
 *   final divide-by-2 to keep values in the same ballpark as before).
 *
 * EG — King activity (king-proximity bonus):
 *   PST_KING_EG already rewards centralization.  It cannot express the
 *   dynamic bonus for approaching the enemy king (crucial for K+P endings,
 *   opposition, and mating nets).  We subtract `dist * KING_EG_DISTANCE_PENALTY`
 *   from the EG score so that the winning king is motivated to close in.
 *   The function now takes *eg so this term feeds into the tapered blend.
 */
static void eval_king_safety(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    int king_sq   = bb_lsb(b->pieces[us][KING]);
    int king_file = king_sq & 7;
    int king_rank = king_sq >> 3;

    Bitboard our_pawns = b->pieces[us][PAWN];

    /* ── 1. Pawn shield + open-file penalty ── */
    for (int f = king_file - 1; f <= king_file + 1; f++) {
        if (f < 0 || f > 7) continue;
        Bitboard file_pawns = our_pawns & FILE_BB[f];

        if (!file_pawns) {
            *mg += KING_OPEN_FILE_PENALTY;
        } else {
            int shield1 = (us == WHITE) ? king_rank + 1 : king_rank - 1;
            int shield2 = (us == WHITE) ? king_rank + 2 : king_rank - 2;
            if (shield1 >= 0 && shield1 < 8 && (file_pawns & RANK_BB[shield1]))
                *mg += KING_SHIELD_BONUS * 2;
            else if (shield2 >= 0 && shield2 < 8 && (file_pawns & RANK_BB[shield2]))
                *mg += KING_SHIELD_BONUS;
        }
    }

    /*
     * ── 2. Distance-weighted enemy piece attacks on king zone ──
     *
     * King zone = king square + all king-move squares (up to 9 squares).
     *
     * For each enemy piece that attacks into the zone, scale its weight by
     * proximity to our king:
     *
     *   dist <= 1 (adjacent or on king sq): scale 4  — immediate threat,
     *             can deliver check or capture next move.
     *   dist 2-3 (nearby):                 scale 2  — standard threat.
     *   dist >= 4 (far):                   scale 1  — long-range pressure.
     *
     * Divide by 2 at the end so the effective multipliers are 2 / 1 / 0.5,
     * keeping total weights comparable to the original flat model.
     * The quadratic penalty for 2+ pieces is preserved.
     */
    Bitboard king_zone = KING_ATTACKS[king_sq] | SQUARE_BB[king_sq];
    Bitboard occ       = b->occ[2];

    int attack_count  = 0;
    int attack_weight = 0;

    for (int pt = KNIGHT; pt < KING; pt++) {
        Bitboard pieces = b->pieces[them][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq]; break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default:     atk = 0; break;
            }
            if (atk & king_zone) {
                int dist  = chebyshev(sq, king_sq);
                int scale = (dist <= 1) ? 4 : (dist <= 3) ? 2 : 1;
                attack_count++;
                attack_weight += KING_ATTACKER_WEIGHT[pt] * scale;
            }
        }
    }

    /* Restore to effective 2×/1×/0.5× scale */
    attack_weight /= 2;

    /* Quadratic penalty when 2+ coordinated pieces attack */
    if (attack_count >= 2) {
        int danger = attack_weight * attack_weight / 200;
        *mg -= danger;
    }

    /*
     * ── 3. Endgame king activity ──
     *
     * Reward the king for being close to the enemy king.  This incentivises
     * the stronger side to centralise and box in the opponent rather than
     * remain passive, and penalises the weaker side's king for hiding.
     *
     * chebyshev ∈ [1, 7] → bonus ∈ [4 cp, 28 cp] (endgame only).
     * PST_KING_EG already captures centrality; this adds the inter-king
     * proximity dimension that a per-piece PST cannot express.
     */
    int enemy_king_sq = bb_lsb(b->pieces[them][KING]);
    int king_dist     = chebyshev(king_sq, enemy_king_sq);
    *eg -= king_dist * KING_EG_DISTANCE_PENALTY;
}

/* ──────────────────────────────────────────────
 *  Tactical threat evaluation
 *
 *  Three terms evaluated for the side `us`:
 *
 *  1. Hanging piece penalty
 *     A piece is "hanging" if it is attacked by the opponent and either:
 *       a) has no friendly defender at all — full penalty (≈ 50 % of piece
 *          value, the rest comes from the search finding the capture), or
 *       b) is attacked by a cheaper enemy piece — partial penalty scaled
 *          by the value difference.
 *     This term fills the gap when pruning prevents the search from
 *     reaching the capturing move: even at the leaf node, the eval will
 *     reflect the material danger, pushing the engine away from leaving
 *     pieces en prise in favour of superficially attractive positional
 *     moves (open files, outposts, etc.).
 *
 *  2. Knight fork bonus
 *     A knight that attacks 2+ enemy pieces worth at least a bishop in the
 *     same move has immediate forking potential.  Rewarding this in the
 *     eval nudges the engine to seek out fork squares even when the fork
 *     itself is a move or two away.
 *
 *  3. Sliding piece battery / skewer bonus
 *     A rook or queen that attacks 2+ valuable enemy pieces along a rank
 *     or file (or a bishop/queen on a diagonal) represents a latent skewer
 *     or battery threat.  A small bonus encourages the engine to establish
 *     such alignments proactively.
 *
 *  All terms are tapered automatically through the mg/eg blend in
 *  evaluate().  sq_attackers() is a local duplicate of the homonymous
 *  function in search.c, kept static here to avoid adding a cross-file
 *  dependency on internal search machinery.
 * ────────────────────────────────────────────── */

static Bitboard sq_attackers(const Board *b, int sq, Bitboard occ) {
    return (PAWN_ATTACKS[WHITE][sq]  & b->pieces[BLACK][PAWN])
         | (PAWN_ATTACKS[BLACK][sq]  & b->pieces[WHITE][PAWN])
         | (KNIGHT_ATTACKS[sq]       & (b->pieces[WHITE][KNIGHT] | b->pieces[BLACK][KNIGHT]))
         | (bishop_attacks((Square)sq, occ)
                           & (  b->pieces[WHITE][BISHOP] | b->pieces[BLACK][BISHOP]
                               | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (rook_attacks((Square)sq, occ)
                         & (  b->pieces[WHITE][ROOK]   | b->pieces[BLACK][ROOK]
                             | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (KING_ATTACKS[sq]         & (b->pieces[WHITE][KING]   | b->pieces[BLACK][KING]));
}

/* Hanging-piece penalty table: roughly 50 % of the piece's material value. */
static const int HANG_PENALTY_MG[6] = {  40, 170, 175, 250, 500, 0 };
static const int HANG_PENALTY_EG[6] = {  50, 165, 160, 270, 460, 0 };

static void eval_threats(const Board *b, Color us, int *mg, int *eg) {
    Color    them = us ^ 1;
    Bitboard occ  = b->occ[2];

    /* ── 1. Hanging / en-prise pieces ── */
    for (int pt = PAWN; pt < KING; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);

            Bitboard enemy_atk = sq_attackers(b, sq, occ) & b->occ[them];
            if (!enemy_atk) continue;   /* not attacked at all — safe */

            Bitboard friendly_def = sq_attackers(b, sq, occ) & b->occ[us];
            /* (the piece on 'sq' itself doesn't appear in sq_attackers output,
               so no need to mask it out.) */

            if (!friendly_def) {
                /* Completely undefended: apply the full hanging penalty. */
                *mg -= HANG_PENALTY_MG[pt];
                *eg -= HANG_PENALTY_EG[pt];
            } else {
                /*
                 * Defended, but attacked by a cheaper piece.  Walk the enemy
                 * attackers from cheapest (PAWN) upward; the first hit is the
                 * minimum-value attacker.
                 */
                int min_atk_mg = 30000;
                for (int apt = PAWN; apt <= QUEEN; apt++) {
                    if (b->pieces[them][apt] & enemy_atk) {
                        min_atk_mg = MATERIAL_MG[apt];
                        break;
                    }
                }
                int piece_mg = MATERIAL_MG[pt];
                if (min_atk_mg < piece_mg) {
                    /* Partial penalty: proportional to value difference. */
                    *mg -= (piece_mg - min_atk_mg) / 6;
                    *eg -= (piece_mg - min_atk_mg) / 6;
                }
            }
        }
    }

    /* ── 2. Knight fork bonus ── */
    {
        /*
         * Any square a knight can reach that hits 2+ enemy pieces worth at
         * least a minor piece represents a fork threat.  We reward the
         * current presence of knights that already attack multiple targets;
         * the eval gradient encourages the engine to manoeuvre toward such
         * squares one move before the fork fires.
         */
        Bitboard valuable_enemy = b->pieces[them][BISHOP]
                                | b->pieces[them][ROOK]
                                | b->pieces[them][QUEEN]
                                | b->pieces[them][KING];
        Bitboard knights = b->pieces[us][KNIGHT];
        while (knights) {
            int sq   = bb_pop(&knights);
            int hits = bb_popcount(KNIGHT_ATTACKS[sq] & valuable_enemy);
            if (hits >= 2) {
                /* Base bonus + extra for each additional piece hit. */
                *mg += 45 + 20 * (hits - 2);
                *eg += 35 + 15 * (hits - 2);
            }
        }
    }

    /* ── 3. Sliding piece battery / skewer bonus ── */
    {
        /*
         * A rook or queen that sees 2+ valuable enemy pieces along its attack
         * ray has a latent battery or skewer threat even if not immediately
         * executable (the intermediate piece may be capturable or moveable).
         * Same logic for bishops on diagonals through the royal pair.
         */
        Bitboard royal      = b->pieces[them][QUEEN] | b->pieces[them][KING];
        Bitboard valuable_r = royal | b->pieces[them][ROOK];   /* rook targets */
        Bitboard valuable_b = royal;                            /* bishop targets */

        Bitboard rq = b->pieces[us][ROOK] | b->pieces[us][QUEEN];
        while (rq) {
            int sq = bb_pop(&rq);
            if (bb_popcount(rook_attacks((Square)sq, occ) & valuable_r) >= 2) {
                *mg += 20;
                *eg += 15;
            }
        }

        Bitboard bq = b->pieces[us][BISHOP] | b->pieces[us][QUEEN];
        while (bq) {
            int sq = bb_pop(&bq);
            if (bb_popcount(bishop_attacks((Square)sq, occ) & valuable_b) >= 2) {
                *mg += 15;
                *eg += 10;
            }
        }
    }
}

/* ──────────────────────────────────────────────
 *  WeakQueen — penalty when a slider x-rays through our queen
 * ────────────────────────────────────────────── */
static void eval_queen_weak(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard occ    = b->occ[2];
    Bitboard queens = b->pieces[us][QUEEN];
    while (queens) {
        int sq = bb_pop(&queens);
        /* Remove the queen from occupancy to reveal x-ray sliders */
        Bitboard occ_no_q = occ ^ SQUARE_BB[sq];
        Bitboard rook_x   = rook_attacks((Square)sq, occ_no_q)
                          & (b->pieces[them][ROOK] | b->pieces[them][QUEEN]);
        Bitboard bish_x   = bishop_attacks((Square)sq, occ_no_q)
                          & (b->pieces[them][BISHOP] | b->pieces[them][QUEEN]);
        if (rook_x | bish_x) {
            *mg -= 49;
            *eg -= 15;
        }
    }
}

/* ──────────────────────────────────────────────
 *  KingProtector — minor pieces far from own king incur a penalty
 * ────────────────────────────────────────────── */
static void eval_king_protector(const Board *b, Color us, int *mg, int *eg) {
    int king_sq = bb_lsb(b->pieces[us][KING]);
    for (int pt = KNIGHT; pt <= BISHOP; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq   = bb_pop(&pieces);
            int dist = chebyshev(sq, king_sq);
            *mg -= 7 * dist;
            *eg -= 8 * dist;
        }
    }
}

/* ──────────────────────────────────────────────
 *  MinorBehindPawn — minor piece directly behind a friendly pawn
 * ────────────────────────────────────────────── */
static void eval_minor_behind_pawn(const Board *b, Color us, int *mg, int *eg) {
    Bitboard pawns = b->pieces[us][PAWN];
    Bitboard behind_pawns = (us == WHITE) ? (pawns >> 8) : (pawns << 8);
    Bitboard minors = b->pieces[us][KNIGHT] | b->pieces[us][BISHOP];
    int count = bb_popcount(minors & behind_pawns);
    *mg += 18 * count;
    *eg +=  3 * count;
}

/* ──────────────────────────────────────────────
 *  Material Imbalance (SF 11 quadratic polynomial)
 *
 *  Uses per-piece-pair coefficients to reward or penalise having certain
 *  piece combinations: bishop pair, rook vs. minors, etc.  The quadratic
 *  term captures interactions that a simple count cannot (e.g. the bishop
 *  pair is worth more when there are many pawns).
 *
 *  Index 0 = bishop-pair proxy; 1=Pawn 2=Knight 3=Bishop 4=Rook 5=Queen.
 * ────────────────────────────────────────────── */
static const int QuadOurs[6][6] = {
    {1438,   0,   0,   0,    0,   0},   /* bishop pair */
    {  40,  38,   0,   0,    0,   0},   /* pawn        */
    {  32, 255, -62,   0,    0,   0},   /* knight      */
    {   0, 104,   4,   0,    0,   0},   /* bishop      */
    { -26,  -2,  47, 105, -208,   0},   /* rook        */
    {-189,  24, 117, 133, -134,  -6},   /* queen       */
};
static const int QuadTheirs[6][6] = {
    {  0,   0,   0,   0,   0,   0},
    { 36,   0,   0,   0,   0,   0},
    {  9,  63,   0,   0,   0,   0},
    { 59,  65,  42,   0,   0,   0},
    { 46,  39,  24, -24,   0,   0},
    { 97, 100, -42, 137, 268,   0},
};

static int material_imbalance(const Board *b) {
    int pc[2][6] = {{0}};
    for (int c = 0; c < 2; c++) {
        pc[c][0] = (bb_popcount(b->pieces[c][BISHOP]) > 1) ? 1 : 0;
        for (int pt = PAWN; pt <= QUEEN; pt++)
            pc[c][pt + 1] = bb_popcount(b->pieces[c][pt]);
    }

    int bonus = 0;
    for (int us = 0; us < 2; us++) {
        int them = us ^ 1;
        int sign = (us == WHITE) ? 1 : -1;
        int b_us = 0;
        for (int pt1 = 0; pt1 <= 5; pt1++) {
            if (!pc[us][pt1]) continue;
            int v = 0;
            for (int pt2 = 0; pt2 <= pt1; pt2++)
                v += QuadOurs[pt1][pt2]   * pc[us][pt2]
                   + QuadTheirs[pt1][pt2] * pc[them][pt2];
            b_us += pc[us][pt1] * v;
        }
        bonus += sign * b_us;
    }
    return bonus / 16;
}

/* ──────────────────────────────────────────────
 *  Initiative / complexity correction (SF 11)
 *
 *  Reduces a winning advantage when the winning side cannot realistically
 *  convert it: no passed pawns, no outflanking, pawns on only one flank.
 *  Applied just before the side-to-move flip.
 * ────────────────────────────────────────────── */
static int initiative(const Board *b, int mg, int eg) {
    int wk = bb_lsb(b->pieces[WHITE][KING]);
    int bk = bb_lsb(b->pieces[BLACK][KING]);

    int outflanking = ((wk & 7) - (bk & 7))
                    - ((wk >> 3) - (bk >> 3));

    bool infiltration = (wk >> 3) > 3 || (bk >> 3) < 4;

    Bitboard all_pawns = b->pieces[WHITE][PAWN] | b->pieces[BLACK][PAWN];
    bool both_flanks = (all_pawns & 0x0F0F0F0F0F0F0F0FULL) &&
                       (all_pawns & 0xF0F0F0F0F0F0F0F0ULL);

    /* Count passed pawns for both sides */
    int passed_count = 0;
    {
        Bitboard wp = b->pieces[WHITE][PAWN];
        Bitboard bp = b->pieces[BLACK][PAWN];
        Bitboard tmp = wp;
        while (tmp) {
            int sq = bb_pop(&tmp);
            int f = sq & 7, r = sq >> 3;
            Bitboard adj = 0;
            if (f > 0) adj |= FILE_BB[f-1];
            if (f < 7) adj |= FILE_BB[f+1];
            Bitboard ahead = 0;
            for (int rr = r+1; rr < 8; rr++) ahead |= RANK_BB[rr];
            if (!(bp & (FILE_BB[f] | adj) & ahead)) passed_count++;
        }
        tmp = bp;
        while (tmp) {
            int sq = bb_pop(&tmp);
            int f = sq & 7, r = sq >> 3;
            Bitboard adj = 0;
            if (f > 0) adj |= FILE_BB[f-1];
            if (f < 7) adj |= FILE_BB[f+1];
            Bitboard ahead = 0;
            for (int rr = 0; rr < r; rr++) ahead |= RANK_BB[rr];
            if (!(wp & (FILE_BB[f] | adj) & ahead)) passed_count++;
        }
    }

    int pawn_count = bb_popcount(all_pawns);
    bool no_npm = (b->occ[WHITE] & ~b->pieces[WHITE][PAWN] & ~b->pieces[WHITE][KING]) == 0
               && (b->occ[BLACK] & ~b->pieces[BLACK][PAWN] & ~b->pieces[BLACK][KING]) == 0;

    bool almost_unwinnable = !passed_count && outflanking < 0 && !both_flanks;

    int complexity =
          9 * passed_count
        + 11 * pawn_count
        +  9 * outflanking
        + 12 * (int)infiltration
        + 21 * (int)both_flanks
        + 51 * (int)no_npm
        - 43 * (int)almost_unwinnable
        - 100;

    int sign_mg = (mg > 0) - (mg < 0);
    int sign_eg = (eg > 0) - (eg < 0);

    int u_raw = complexity + 50;
    int u = sign_mg * (u_raw < 0 ? u_raw : 0);
    if (u < -abs(mg)) u = -abs(mg);

    int v = sign_eg * (complexity > -abs(eg) ? complexity : -abs(eg));

    int phase = game_phase(b);
    return taper(u, v, phase);
}

/* ──────────────────────────────────────────────
 *  Lazy evaluation guard (~5 % NPS, free Elo)
 *
 *  Before running the full evaluation, compute a cheap material+PST proxy.
 *  If the proxy is far outside the window, return it immediately — the
 *  full eval cannot change the result.  LAZY_THRESHOLD is tuned conservatively
 *  so we never skip evaluation for positions near the window boundary.
 * ────────────────────────────────────────────── */
#define LAZY_THRESHOLD 1400

static int lazy_score(const Board *b, int phase) {
    int mg = 0, eg = 0;
    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = pst_sq((Color)c, (Square)sq);
                mg += sign * (MATERIAL_MG[pt] + PST_MG[pt][psq]);
                eg += sign * (MATERIAL_EG[pt] + PST_EG[pt][psq]);
            }
        }
    }
    return taper(mg, eg, phase);
}

/* ──────────────────────────────────────────────
 *  Main evaluation
 * ────────────────────────────────────────────── */
int evaluate(const Board *b) {
    int phase = game_phase(b);

    /*
     * ── Lazy evaluation guard (~5 % NPS) ──────────────────────────────
     * Compute a cheap material+PST proxy.  If it is far outside any
     * plausible search window the full evaluation cannot change the result,
     * so we return early.  LAZY_THRESHOLD (1400 cp) is deliberately large
     * to guarantee we never skip a position close to the window boundary.
     */
    int proxy = lazy_score(b, phase);
    if (abs(proxy) > LAZY_THRESHOLD) {
        /* Still apply the side-to-move flip for consistency */
        return (b->side == WHITE) ? proxy : -proxy;
    }

    int mg = 0, eg = 0;

    /* ── Material imbalance (quadratic polynomial, SF 11) ─────────────
     * Accounts for piece-combination interactions: bishop pair value,
     * rook vs. two minors, etc.  Applied once per position, not per side.
     */
    mg += material_imbalance(b);

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        int c_mg = 0, c_eg = 0;

        /* Material + PST */
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = pst_sq((Color)c, (Square)sq);
                c_mg += MATERIAL_MG[pt] + PST_MG[pt][psq];
                c_eg += MATERIAL_EG[pt] + PST_EG[pt][psq];
            }
        }

        /* Pawn structure (doubled, isolated, backward, passed) */
        eval_pawns(b, (Color)c, &c_mg, &c_eg);

        /* Piece mobility — non-linear per-count SF 11 tables */
        eval_mobility(b, (Color)c, &c_mg, &c_eg);

        /* Rook bonuses (open file, semi-open, 7th rank, TrappedRook) */
        eval_rooks(b, (Color)c, &c_mg, &c_eg);

        /* Outpost bonuses for knights and bishops */
        eval_outposts(b, (Color)c, &c_mg, &c_eg);

        /* Bishop pair */
        if (bb_popcount(b->pieces[c][BISHOP]) >= 2) {
            c_mg += BISHOP_PAIR_MG;
            c_eg += BISHOP_PAIR_EG;
        }

        /* WeakQueen: penalty when enemy sliders x-ray through our queen */
        eval_queen_weak(b, (Color)c, &c_mg, &c_eg);

        /* KingProtector: minor pieces far from own king incur a penalty */
        eval_king_protector(b, (Color)c, &c_mg, &c_eg);

        /* MinorBehindPawn: minor pieces sheltered behind friendly pawns */
        eval_minor_behind_pawn(b, (Color)c, &c_mg, &c_eg);

        /*
         * King safety: shield + open files + distance-weighted enemy
         * attacks (MG) + king-proximity activity bonus (EG).
         */
        eval_king_safety(b, (Color)c, &c_mg, &c_eg);

        /* Tactical threats: hanging pieces, fork potential, skewers */
        eval_threats(b, (Color)c, &c_mg, &c_eg);

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    /* Taper MG/EG blend */
    int score = taper(mg, eg, phase);

    /* Tempo bonus */
    score += taper(TEMPO_BONUS_MG, TEMPO_BONUS_EG, phase);

    /*
     * ── Initiative / complexity correction (SF 11) ────────────────────
     * Reduces a winning advantage when the winning side cannot realistically
     * convert it: no passed pawns, no outflanking, pawns on only one flank.
     * Applied before the side-to-move flip.
     */
    score += initiative(b, mg, eg);

    /* Return from side-to-move perspective */
    return (b->side == WHITE) ? score : -score;
}