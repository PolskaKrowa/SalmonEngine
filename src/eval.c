/*
 * eval.c — Static evaluation (improved)
 *
 * Scoring convention: always from the perspective of the side to move
 * (positive = good for side to move).
 *
 * Features implemented:
 *   • Material balance
 *   • Piece-square tables — separate MG and EG tables, blended by game phase
 *   • Mobility (approximate: popcount of attack set minus own pieces)
 *   • Pawn structure:
 *       – Doubled-pawn penalty
 *       – Isolated-pawn penalty
 *       – Backward-pawn penalty
 *       – Passed-pawn bonus
 *   • Outpost squares for knights and bishops
 *   • King safety:
 *       – Pawn shield bonus
 *       – Open-file penalty near king
 *       – Weighted enemy-piece attack count
 *   • Bishop pair bonus
 *   • Rook on open / semi-open file bonus
 *   • Rook on seventh rank bonus
 *   • Tempo bonus (side to move)
 */

#include "eval.h"
#include "bitboard.h"
#include <string.h>

/* ──────────────────────────────────────────────
 *  Material values (centipawns)
 *  Separate MG and EG values for tapered eval.
 * ────────────────────────────────────────────── */
static const int MATERIAL_MG[6] = { 82, 337, 365, 477, 1025,  0 };
static const int MATERIAL_EG[6] = { 94, 281, 297, 512,  936,  0 };

/* ──────────────────────────────────────────────
 *  Piece-square tables (from White's perspective,
 *  rank 1 at index 0, so flip for Black)
 *
 *  Layout: [rank 1..8][file a..h] = index [0..63]
 *  Positive = good for the piece on that square.
 * ────────────────────────────────────────────── */

/* ---- PAWN ---- */
static const int PST_PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int PST_PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};

/* ---- KNIGHT ---- */
static const int PST_KNIGHT_MG[64] = {
    -167, -89, -34, -49,  61, -97, -15,-107,
     -73, -41,  72,  36,  23,  62,   7, -17,
     -47,  60,  37,  65,  84, 129,  73,  44,
      -9,  17,  19,  53,  37,  69,  18,  22,
     -13,   4,  16,  13,  28,  19,  21,  -8,
     -23,  -9,  12,  10,  19,  17,  25, -16,
     -29, -53, -12,  -3,  -1,  18, -14, -19,
    -105, -21, -58, -33, -17, -28, -19, -23,
};
static const int PST_KNIGHT_EG[64] = {
     -58, -38, -13, -28, -31, -27, -63, -99,
     -25,  -8, -25,  -2,  -9, -25, -24, -52,
     -24, -20,  10,   9,  -1,  -9, -19, -41,
     -17,   3,  22,  22,  22,  11,   8, -18,
     -18,  -6,  16,  25,  16,  17,   4, -18,
     -23,  -3,  -1,  15,  10,  -3, -20, -22,
     -42, -20, -10,  -5,  -2, -20, -23, -44,
     -29, -51, -23, -15, -22, -18, -50, -64,
};

/* ---- BISHOP ---- */
static const int PST_BISHOP_MG[64] = {
     -29,   4, -82, -37, -25, -42,   7,  -8,
     -26,  16, -18, -13,  30,  59,  18, -47,
     -16,  37,  43,  40,  35,  50,  37,  -2,
      -4,   5,  19,  50,  37,  37,   7,  -2,
      -6,  13,  13,  26,  34,  12,  10,   4,
       0,  15,  15,  15,  14,  27,  18,  10,
       4,  15,  16,   0,   7,  21,  33,   1,
     -33,  -3, -14, -21, -13, -12, -39, -21,
};
static const int PST_BISHOP_EG[64] = {
     -14, -21, -11,  -8,  -7,  -9, -17, -24,
      -8,  -4,   7, -12,  -3, -13,  -4, -14,
       2,  -8,   0,  -1,  -2,   6,   0,   4,
      -3,   9,  12,   9,  14,  10,   3,   2,
      -6,   3,  13,  19,   7,  10,  -3,  -9,
     -12,  -3,   8,  10,  13,   3,  -7, -15,
     -14, -18,  -7,  -1,   4,  -9, -15, -27,
     -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};

/* ---- ROOK ---- */
static const int PST_ROOK_MG[64] = {
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
};
static const int PST_ROOK_EG[64] = {
     13, 10, 18, 15, 12,  12,   8,   5,
     11, 13, 13, 11, -3,   3,   8,   3,
      7,  7,  7,  5,  4,  -3,  -5,  -3,
      4,  3, 13,  1,  2,   1,  -1,   2,
      3,  5,  8,  4, -5,  -6,  -8, -11,
     -4,  0, -5, -1, -7, -12,  -8, -16,
     -6, -6,  0,  2, -9,  -9, -11,  -3,
     -9,  2,  3, -1, -5, -13,   4, -20,
};

/* ---- QUEEN ---- */
static const int PST_QUEEN_MG[64] = {
     -28,   0,  29,  12,  59,  44,  43,  45,
     -24, -39,  -5,   1, -16,  57,  28,  54,
     -13, -17,   7,   8,  29,  56,  47,  57,
     -27, -27, -16, -16,  -1,  17,  -2,   1,
      -9, -26,  -9, -10,  -2,  -4,   3,  -3,
     -14,   2, -11,  -2,  -5,   2,  14,   5,
     -35,  -8,  11,   2,   8,  15,  -3,   1,
       -1, -18,  -9,  10, -15, -25, -31, -50,
};
static const int PST_QUEEN_EG[64] = {
      -9,  22,  22,  27,  27,  19,  10,  20,
     -17,  20,  32,  41,  58,  25,  30,   0,
     -20,   6,   9,  49,  47,  35,  19,   9,
       3,  22,  24,  45,  57,  40,  57,  36,
     -18,  28,  19,  47,  31,  34,  39,  23,
     -16, -27,  15,   6,   9,  17,  10,   5,
     -22, -23, -30, -16, -16, -23, -36, -32,
     -33, -28, -22, -43,  -5, -32, -20, -41,
};

/* ---- KING ---- */
static const int PST_KING_MG[64] = {
     -65,  23,  16, -15, -56, -34,   2,  13,
      29,  -1, -20,  -7,  -8,  -4, -38, -29,
      -9,  24,   2, -16, -20,   6,  22, -22,
     -17, -20, -12, -27, -30, -25, -14, -36,
     -49,  -1, -27, -39, -46, -44, -33, -51,
     -14, -14, -22, -46, -44, -30, -15, -27,
       1,   7,  -8, -64, -43, -16,   9,   8,
     -15,  36,  12, -54,   8, -28,  24,  14,
};
static const int PST_KING_EG[64] = {
     -74, -35, -18, -18, -11,  15,   4, -17,
     -12,  17,  14,  17,  17,  38,  23,  11,
      10,  17,  23,  15,  20,  45,  44,  13,
      -8,  22,  24,  27,  26,  33,  26,   3,
     -18,  -4,  21,  24,  27,  23,   9, -11,
     -19,  -3,  11,  21,  23,  16,   7,  -9,
     -27, -11,   4,  13,  14,   4,  -5, -17,
     -53, -34, -21, -11, -28, -14, -24, -43,
};

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
 *  Pawn structure constants
 * ────────────────────────────────────────────── */
static const int DOUBLED_PAWN_PENALTY_MG  = -11;
static const int DOUBLED_PAWN_PENALTY_EG  = -56;
static const int ISOLATED_PAWN_PENALTY_MG = -15;
static const int ISOLATED_PAWN_PENALTY_EG = -15;
static const int BACKWARD_PAWN_PENALTY_MG =  -9;  /* [NEW] */
static const int BACKWARD_PAWN_PENALTY_EG = -22;  /* [NEW] */
static const int PASSED_PAWN_BONUS_MG[8]  = { 0,  5, 10, 20, 35, 60, 90,  0 };
static const int PASSED_PAWN_BONUS_EG[8]  = { 0, 10, 20, 40, 65, 95,140,  0 };

/* ──────────────────────────────────────────────
 *  Pawn structure evaluation (one side)
 * ────────────────────────────────────────────── */
static void eval_pawns(const Board *b, Color us, int *mg, int *eg) {
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
         * No enemy pawns on same or adjacent files ahead of this pawn.        */
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
         * ── Backward pawn [NEW] ──
         *
         * A pawn is backward when:
         *   1. Its stop square (one step forward) is controlled by an enemy
         *      pawn — it cannot safely advance.
         *   2. It has no friendly pawn support from behind on adjacent files
         *      (so it cannot be "pushed" by a lever).
         *   3. It is not already passed.
         *
         * We detect enemy pawn control of the stop square by checking
         * PAWN_ATTACKS[us][stop_sq] & their_pawns:
         *   For us=WHITE, stop_sq = sq+8, PAWN_ATTACKS[WHITE][stop_sq] gives
         *   the squares diagonally ahead of stop_sq — exactly where a BLACK
         *   pawn would need to be to attack stop_sq from behind (its forward).
         *   The same formula works symmetrically for BLACK.
         */
        if (!is_passed) {
            int stop = (us == WHITE) ? sq + 8 : sq - 8;
            if (stop >= 0 && stop < 64) {
                bool stop_attacked = (PAWN_ATTACKS[us][stop] & their_pawns) != 0;

                if (stop_attacked) {
                    /* Squares on adjacent files BEHIND this pawn */
                    Bitboard below_sq   = SQUARE_BB[sq] - 1; /* bits 0..sq-1 */
                    Bitboard above_sq   = ~SQUARE_BB[sq] & ~below_sq; /* bits sq+1..63 */
                    Bitboard behind_span = ((us == WHITE) ? below_sq : above_sq)
                                          & adj_files
                                          & ~RANK_BB[rank];

                    bool has_support = (our_pawns & behind_span) != 0;
                    if (!has_support) {
                        *mg += BACKWARD_PAWN_PENALTY_MG;
                        *eg += BACKWARD_PAWN_PENALTY_EG;
                    }
                }
            }
        }
    }
}

/* ──────────────────────────────────────────────
 *  Mobility bonus (per piece, per reachable square)
 * ────────────────────────────────────────────── */
static const int MOBILITY_MG[6] = {  0,  4,  3,  2,  1, 0 };
static const int MOBILITY_EG[6] = {  0,  4,  3,  3,  2, 0 };

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
            *mg += MOBILITY_MG[pt] * mob;
            *eg += MOBILITY_EG[pt] * mob;
        }
    }
}

/* ──────────────────────────────────────────────
 *  Rook evaluation: open/semi-open files + 7th rank [IMPROVED]
 * ────────────────────────────────────────────── */
static const int ROOK_OPEN_FILE_MG    =  23;
static const int ROOK_OPEN_FILE_EG    =  15;
static const int ROOK_SEMIOPEN_MG     =  12;
static const int ROOK_SEMIOPEN_EG     =   7;
static const int ROOK_ON_SEVENTH_MG   =  20;   /* [NEW] */
static const int ROOK_ON_SEVENTH_EG   =  32;   /* [NEW] */

static void eval_rooks(const Board *b, Color us, int *mg, int *eg) {
    Color them           = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard rooks       = b->pieces[us  ][ROOK];

    /* Rank index for the "7th rank" (relative to the side being evaluated) */
    int rank7 = (us == WHITE) ? 6 : 1;   /* rank index 0-based */
    int rank8 = (us == WHITE) ? 7 : 0;   /* enemy back rank     */

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
         * Rook on the 7th rank [NEW]:
         * Valuable when the enemy king is on the 8th rank or there are
         * enemy pawns on the 7th rank to harass.
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

/* ──────────────────────────────────────────────
 *  Bishop pair bonus
 * ────────────────────────────────────────────── */
static const int BISHOP_PAIR_MG = 30;
static const int BISHOP_PAIR_EG = 60;

/* ──────────────────────────────────────────────
 *  Outpost evaluation [NEW]
 *
 *  An outpost square is:
 *    • On ranks 4–6 from the piece's own perspective (rank 3–5 from index 0)
 *    • Supported by a friendly pawn (pawn attacks that square)
 *    • Cannot be attacked by an enemy pawn (no enemy pawn on adjacent files
 *      with a rank that could advance and attack it)
 *
 *  Knights and bishops on outposts receive bonuses.
 * ────────────────────────────────────────────── */
static const int OUTPOST_KNIGHT_MG = 22;
static const int OUTPOST_KNIGHT_EG = 14;
static const int OUTPOST_BISHOP_MG = 12;
static const int OUTPOST_BISHOP_EG =  8;

static void eval_outposts(const Board *b, Color us, int *mg, int *eg) {
    Color them        = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];

    /*
     * Build "outpost mask": squares on ranks 4–6 (relative) that are
     *   - Supported by at least one friendly pawn
     *   - Not attackable by any enemy pawn now or in the future
     *     (no enemy pawn on adjacent files with a rank that could advance).
     *
     * "Pawn attack span" of the enemy: all squares an enemy pawn can
     * eventually attack, going forward.  A simplified version: for each
     * enemy pawn, the files ±1 ahead of it are forever controlled.
     *
     * We approximate this with the forward-fill of enemy pawn attacks.
     */

    /* Enemy pawn attack spans: squares on adjacent files, at ranks
       in front of each enemy pawn (towards our side).             */
    Bitboard enemy_pawn_attack_span = 0;
    {
        Bitboard ep = their_pawns;
        while (ep) {
            int sq   = bb_pop(&ep);
            int file = sq & 7;
            int rank = sq >> 3;
            Bitboard adj = 0;
            if (file > 0) adj |= FILE_BB[file - 1];
            if (file < 7) adj |= FILE_BB[file + 1];

            /* Ranks between the enemy pawn and our back rank */
            if (them == BLACK) { /* them=BLACK means their pawns go down; our back rank is rank 0 */
                for (int r = rank - 1; r >= 0; r--) enemy_pawn_attack_span |= RANK_BB[r] & adj;
            } else {
                for (int r = rank + 1; r < 8; r++) enemy_pawn_attack_span |= RANK_BB[r] & adj;
            }
        }
    }

    /* Outpost ranks: ranks 4–6 from our perspective (0-indexed) */
    int r_min = (us == WHITE) ? 3 : 2;  /* 0-based rank 4 for WHITE, rank 3 for BLACK */
    int r_max = (us == WHITE) ? 5 : 4;  /* 0-based rank 6 for WHITE, rank 5 for BLACK */
    Bitboard outpost_rank_mask = 0;
    for (int r = r_min; r <= r_max; r++) outpost_rank_mask |= RANK_BB[r];

    /* Evaluate knights on outposts */
    Bitboard knights = b->pieces[us][KNIGHT];
    while (knights) {
        int sq = bb_pop(&knights);
        if (!((SQUARE_BB[sq]) & outpost_rank_mask)) continue;
        if (enemy_pawn_attack_span & SQUARE_BB[sq])  continue; /* can be chased */
        /* Must be supported by a friendly pawn */
        if (!(PAWN_ATTACKS[them][sq] & our_pawns))   continue; /* not supported */

        *mg += OUTPOST_KNIGHT_MG;
        *eg += OUTPOST_KNIGHT_EG;
    }

    /* Evaluate bishops on outposts (less valuable — bishops can be longer-range) */
    Bitboard bishops = b->pieces[us][BISHOP];
    while (bishops) {
        int sq = bb_pop(&bishops);
        if (!((SQUARE_BB[sq]) & outpost_rank_mask)) continue;
        if (enemy_pawn_attack_span & SQUARE_BB[sq])  continue;
        if (!(PAWN_ATTACKS[them][sq] & our_pawns))   continue;

        *mg += OUTPOST_BISHOP_MG;
        *eg += OUTPOST_BISHOP_EG;
    }
}

/* ──────────────────────────────────────────────
 *  King safety [IMPROVED]
 *
 *  Combines:
 *    1. Pawn shield: pawns directly in front of the king are rewarded.
 *    2. Open-file penalty: files near the king without own pawns.
 *    3. Enemy piece attack count on the king zone [NEW]: each enemy
 *       piece type attacking squares adjacent to the king contributes
 *       a weighted penalty.  Two or more attackers trigger a quadratic
 *       scaling, modelling the danger of coordinated attacks.
 * ────────────────────────────────────────────── */
static const int KING_SHIELD_BONUS        =   7;
static const int KING_OPEN_FILE_PENALTY   = -25;

/* Per-piece-type attack weight for king danger */
static const int KING_ATTACKER_WEIGHT[6] = { 0, 20, 20, 40, 80, 0 };
/*                                            P   N   B   R   Q   K  */

static void eval_king_safety(const Board *b, Color us, int *mg) {
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
     * ── 2. Enemy piece attacks on king zone [NEW] ──
     *
     * King zone = king square + all king-move squares.
     * For each enemy piece, if any of its attacks land in the king zone,
     * add the piece's attack weight to a running total.
     * The final penalty is quadratic when 2+ pieces are involved, because
     * coordinated attacks are much more dangerous than isolated probes.
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
                attack_count++;
                attack_weight += KING_ATTACKER_WEIGHT[pt];
            }
        }
    }

    /* Quadratic penalty when 2+ pieces join the attack */
    if (attack_count >= 2) {
        int danger = attack_weight * attack_weight / 200;
        *mg -= danger;
    }
}

/* ──────────────────────────────────────────────
 *  Tempo bonus [NEW]
 *
 *  A small bonus for the side to move.  The engine already has an
 *  implicit tempo advantage through move generation, but an explicit
 *  bonus improves practical play in dynamically balanced positions.
 * ────────────────────────────────────────────── */
#define TEMPO_BONUS_MG 14
#define TEMPO_BONUS_EG  8

/* ──────────────────────────────────────────────
 *  Main evaluation
 * ────────────────────────────────────────────── */
int evaluate(const Board *b) {
    int mg = 0, eg = 0;
    int phase = game_phase(b);

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

        /* Piece mobility */
        eval_mobility(b, (Color)c, &c_mg, &c_eg);

        /* Rook bonuses */
        eval_rooks(b, (Color)c, &c_mg, &c_eg);

        /* Outpost bonuses */
        eval_outposts(b, (Color)c, &c_mg, &c_eg);

        /* Bishop pair */
        if (bb_popcount(b->pieces[c][BISHOP]) >= 2) {
            c_mg += BISHOP_PAIR_MG;
            c_eg += BISHOP_PAIR_EG;
        }

        /* King safety (shield + open files + enemy attacks) */
        eval_king_safety(b, (Color)c, &c_mg);

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    /* Taper MG/EG blend */
    int score = taper(mg, eg, phase);

    /*
     * Tempo bonus [NEW]:
     * The side to move gets a small bonus in both MG and EG.
     * Tapered here to blend naturally with the rest of the eval.
     */
    score += taper(TEMPO_BONUS_MG, TEMPO_BONUS_EG, phase);

    /* Return from side-to-move perspective */
    return (b->side == WHITE) ? score : -score;
}