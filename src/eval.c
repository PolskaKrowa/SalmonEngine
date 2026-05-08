/*
 * eval.c — Static evaluation (improved)
 *
 * Scoring convention: always from the perspective of the side to move
 * (positive = good for side to move).
 *
 */

#include "eval.h"
#include "bitboard.h"
#include <string.h>

/* ──────────────────────────────────────────────
 *  Material values (centipawns)
 * ────────────────────────────────────────────── */
static const int MATERIAL_MG[6] = { 82, 337, 365, 477, 1025,  0 };
static const int MATERIAL_EG[6] = { 94, 281, 297, 512,  936,  0 };

/* ──────────────────────────────────────────────
 *  Piece-square tables (White's perspective;
 *  flip vertically for Black via pst_sq())
 * ────────────────────────────────────────────── */

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

static const int * const PST_MG[6] = {
    PST_PAWN_MG, PST_KNIGHT_MG, PST_BISHOP_MG,
    PST_ROOK_MG, PST_QUEEN_MG,  PST_KING_MG
};
static const int * const PST_EG[6] = {
    PST_PAWN_EG, PST_KNIGHT_EG, PST_BISHOP_EG,
    PST_ROOK_EG, PST_QUEEN_EG,  PST_KING_EG
};

static inline int pst_sq(Color c, Square sq) {
    return (c == WHITE) ? (int)sq : ((int)sq ^ 56);
}

/* ──────────────────────────────────────────────
 *  Phase / taper (unchanged)
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

static inline int taper(int mg, int eg, int phase) {
    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
}

/* ──────────────────────────────────────────────
 *  Colour-square masks for bad-bishop detection
 * ────────────────────────────────────────────── */
/* Light squares: (file+rank) even  →  a1,c1,e1,g1,b2,d2,...
   Verified: a1(sq 0): 0x55 byte → bit 0 set ✓                */
static const Bitboard LIGHT_SQ = 0x55AA55AA55AA55AAULL;
static const Bitboard DARK_SQ  = 0xAA55AA55AA55AA55ULL;

/* ──────────────────────────────────────────────
 *  EvalInfo — pre-computed once per evaluate()
 * ────────────────────────────────────────────── */
typedef struct {
    Bitboard pawn_attacks[2]; /* all squares attacked by each side's pawns */
    Bitboard king_zone[2];    /* king-attack zone for each side             */
    int      king_sq[2];
} EvalInfo;

/* Bulk pawn attack map (all pawns of one colour at once) */
static inline Bitboard pawn_attacks_all(Color c, Bitboard pawns) {
    return (c == WHITE)
        ? (bb_north(bb_east(pawns)) | bb_north(bb_west(pawns)))
        : (bb_south(bb_east(pawns)) | bb_south(bb_west(pawns)));
}

static void init_eval_info(const Board *b, EvalInfo *ei) {
    for (int c = 0; c < 2; c++) {
        ei->pawn_attacks[c] = pawn_attacks_all((Color)c, b->pieces[c][PAWN]);
        ei->king_sq[c]      = bb_lsb(b->pieces[c][KING]);
        /* King zone = squares the king can step to, plus one extra rank
           toward the opponent (where attacks typically land first).      */
        Bitboard kz = KING_ATTACKS[ei->king_sq[c]]
                    | sq_bb((Square)ei->king_sq[c]);
        kz |= (c == WHITE) ? bb_north(kz) : bb_south(kz);
        ei->king_zone[c] = kz;
    }
}

/* ──────────────────────────────────────────────
 *  King danger lookup table  (Fruit / Toga style)
 *
 *  Index = cumulative attack units:
 *    Knight or bishop attacking king zone → +2 units + #attacked squares
 *    Rook                                 → +3 units + #attacked squares
 *    Queen                                → +5 units + #attacked squares
 *
 *  Penalty is applied (MG only) when ≥ 2 pieces participate.
 *  Values are centipawns; table caps at index 49 (≥ that → 400 cp).
 * ────────────────────────────────────────────── */
static const int KING_DANGER[50] = {
      0,   0,   1,   2,   3,   5,   7,   9,  12,  15,
     18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
     68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
    141, 151, 162, 172, 183, 195, 207, 219, 232, 245,
    258, 272, 287, 302, 317, 333, 349, 366, 383, 400,
};
static const int KING_ATTACK_WEIGHT[6] = { 0, 2, 2, 3, 5, 0 };

static inline int king_danger_score(int units) {
    if (units <= 0)  return 0;
    if (units >= 49) return 400;
    return KING_DANGER[units];
}

/* ──────────────────────────────────────────────
 *  Evaluation constants
 * ────────────────────────────────────────────── */

/* Pawn structure */
static const int DOUBLED_PAWN_PENALTY_MG  = -11;
static const int DOUBLED_PAWN_PENALTY_EG  = -56;
static const int ISOLATED_PAWN_PENALTY_MG = -15;
static const int ISOLATED_PAWN_PENALTY_EG = -15;
static const int BACKWARD_PAWN_PENALTY_MG =  -9;
static const int BACKWARD_PAWN_PENALTY_EG = -12;
static const int PASSED_PAWN_BONUS_MG[8]  = {  0,  5, 10, 20,  35,  60,  90, 0 };
static const int PASSED_PAWN_BONUS_EG[8]  = {  0, 10, 20, 40,  65,  95, 140, 0 };

/* Outpost / piece quality */
static const int KNIGHT_OUTPOST_MG           = 20;
static const int KNIGHT_OUTPOST_EG           = 10;
static const int KNIGHT_REACHABLE_OUTPOST_MG =  8;
static const int KNIGHT_REACHABLE_OUTPOST_EG =  4;
static const int BISHOP_OUTPOST_MG           = 10;
static const int BISHOP_OUTPOST_EG           =  5;
static const int BAD_BISHOP_MG               = -4; /* per own pawn on bishop's colour */
static const int BAD_BISHOP_EG               = -8;

/* Mobility */
static const int MOBILITY_MG[6] = { 0, 4, 3, 2, 1, 0 };
static const int MOBILITY_EG[6] = { 0, 4, 3, 3, 2, 0 };

/* Rooks */
static const int ROOK_OPEN_FILE_MG  =  23;
static const int ROOK_OPEN_FILE_EG  =  15;
static const int ROOK_SEMIOPEN_MG   =  12;
static const int ROOK_SEMIOPEN_EG   =   7;
static const int ROOK_ON_7TH_MG     =  11;
static const int ROOK_ON_7TH_EG     =  20;
static const int CONNECTED_ROOKS_MG =  10;
static const int CONNECTED_ROOKS_EG =   5;

/* Bishop pair */
static const int BISHOP_PAIR_MG = 30;
static const int BISHOP_PAIR_EG = 60;

/* King safety — pawn shield */
static const int KING_SHIELD_BONUS      =   7;
static const int KING_OPEN_FILE_PENALTY = -25;

/* Space & tempo */
static const int SPACE_BONUS = 2;  /* per safe centre square */
static const int TEMPO       = 14; /* small bonus for the side to move */

/* ──────────────────────────────────────────────
 *  Pawn structure evaluation
 * ────────────────────────────────────────────── */
static void eval_pawns(const Board *b, Color us,
                       const EvalInfo *ei, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard ep_atk      = ei->pawn_attacks[them]; /* enemy pawn attack map */
    Bitboard copy        = our_pawns;

    while (copy) {
        int sq   = bb_pop(&copy);
        int file = sq & 7;
        int rank = sq >> 3;

        /* Adjacent-file mask */
        Bitboard adj = 0;
        if (file > 0) adj |= FILE_BB[file - 1];
        if (file < 7) adj |= FILE_BB[file + 1];

        /* ── Doubled ── */
        if (bb_popcount(our_pawns & FILE_BB[file]) > 1) {
            *mg += DOUBLED_PAWN_PENALTY_MG;
            *eg += DOUBLED_PAWN_PENALTY_EG;
        }

        /* ── Isolated ── */
        bool isolated = !(our_pawns & adj);
        if (isolated) {
            *mg += ISOLATED_PAWN_PENALTY_MG;
            *eg += ISOLATED_PAWN_PENALTY_EG;
        }

        /* ── Backward (not double-counted with isolated) ──
         *  A pawn is backward when:
         *  (a) its stop square is attacked by an enemy pawn, AND
         *  (b) no friendly pawn on an adjacent file is at the same rank
         *      or behind it (so none can advance to provide support).   */
        if (!isolated) {
            /* "Support zone": adjacent files, ranks at or behind this pawn */
            Bitboard support_zone = adj;
            if (us == WHITE) {
                /* Keep ranks 0..rank; mask out ranks above */
                for (int r = rank + 1; r < 8; r++) support_zone &= ~RANK_BB[r];
            } else {
                /* Keep ranks rank..7; mask out ranks below */
                for (int r = 0; r < rank; r++) support_zone &= ~RANK_BB[r];
            }
            bool no_support = !(our_pawns & support_zone);
            int  stop       = (us == WHITE) ? sq + 8 : sq - 8;
            bool stop_atk   = (stop >= 0 && stop < 64) && ((ep_atk >> stop) & 1);
            if (no_support && stop_atk) {
                *mg += BACKWARD_PAWN_PENALTY_MG;
                *eg += BACKWARD_PAWN_PENALTY_EG;
            }
        }

        /* ── Passed ── */
        Bitboard ahead = 0;
        if (us == WHITE) { for (int r = rank+1; r < 8; r++) ahead |= RANK_BB[r]; }
        else             { for (int r = 0;      r < rank; r++) ahead |= RANK_BB[r]; }
        if (!((their_pawns & (FILE_BB[file] | adj)) & ahead)) {
            int br = (us == WHITE) ? rank : (7 - rank);
            *mg += PASSED_PAWN_BONUS_MG[br];
            *eg += PASSED_PAWN_BONUS_EG[br];
        }
    }
}

/* ──────────────────────────────────────────────
 *  Minor-piece evaluation: outpost & bad bishop
 * ────────────────────────────────────────────── */
static void eval_pieces(const Board *b, Color us,
                        const EvalInfo *ei, int *mg, int *eg) {
    Bitboard our_pawns = b->pieces[us][PAWN];
    Bitboard not_us    = ~b->occ[us];
    /* Outpost squares: defended by our pawn AND not attackable by enemy pawn */
    Bitboard outpost_sqs = ei->pawn_attacks[us] & ~ei->pawn_attacks[us ^ 1];

    /* ── Knights ── */
    Bitboard knights = b->pieces[us][KNIGHT];
    while (knights) {
        int sq   = bb_pop(&knights);
        int rank = sq >> 3;
        /* Outposts are only meaningful from rank 4 onward (0-indexed: rank ≥ 3) */
        bool good_rank = (us == WHITE) ? (rank >= 3) : (rank <= 4);

        if (good_rank && (outpost_sqs & sq_bb((Square)sq))) {
            /* Knight is already on an outpost */
            *mg += KNIGHT_OUTPOST_MG;
            *eg += KNIGHT_OUTPOST_EG;
        } else if (good_rank && (KNIGHT_ATTACKS[sq] & outpost_sqs & not_us)) {
            /* Knight can jump to an outpost next move */
            *mg += KNIGHT_REACHABLE_OUTPOST_MG;
            *eg += KNIGHT_REACHABLE_OUTPOST_EG;
        }
    }

    /* ── Bishops ── */
    Bitboard bishops = b->pieces[us][BISHOP];
    while (bishops) {
        int sq    = bb_pop(&bishops);
        int rank  = sq >> 3;
        bool good_rank = (us == WHITE) ? (rank >= 3) : (rank <= 4);

        /* Outpost */
        if (good_rank && (outpost_sqs & sq_bb((Square)sq))) {
            *mg += BISHOP_OUTPOST_MG;
            *eg += BISHOP_OUTPOST_EG;
        }

        /* Bad bishop: own pawns on the same colour squares as this bishop.
         * Each such pawn slightly devalues the bishop.                    */
        Bitboard sq_colour = (sq_bb((Square)sq) & LIGHT_SQ) ? LIGHT_SQ : DARK_SQ;
        int own_blocked = bb_popcount(our_pawns & sq_colour);
        *mg += BAD_BISHOP_MG * own_blocked;
        *eg += BAD_BISHOP_EG * own_blocked;
    }
}

/* ──────────────────────────────────────────────
 *  Mobility
 *  Minors exclude squares attacked by enemy pawns
 *  (those squares are not truly safe to occupy).
 * ────────────────────────────────────────────── */
static void eval_mobility(const Board *b, Color us,
                           const EvalInfo *ei, int *mg, int *eg) {
    Bitboard occ    = b->occ[2];
    Bitboard not_us = ~b->occ[us];
    Bitboard ep_atk = ei->pawn_attacks[us ^ 1];

    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        /* For minor pieces, subtract squares controlled by enemy pawns */
        Bitboard safe = (pt <= BISHOP) ? (not_us & ~ep_atk) : not_us;
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
            int mob = bb_popcount(atk & safe);
            *mg += MOBILITY_MG[pt] * mob;
            *eg += MOBILITY_EG[pt] * mob;
        }
    }
}

/* ──────────────────────────────────────────────
 *  Rook evaluation
 * ────────────────────────────────────────────── */
static void eval_rooks(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard rooks       = b->pieces[us][ROOK];
    Bitboard occ         = b->occ[2];

    /* Rank numbers (0-indexed) for "7th rank" and the enemy back rank */
    int seventh   = (us == WHITE) ? 6 : 1;
    int enemy_8th = (us == WHITE) ? 7 : 0;
    bool connected = false; /* count connected-rooks bonus only once */

    Bitboard copy = rooks;
    while (copy) {
        int sq   = bb_pop(&copy);
        int file = sq & 7;
        int rank = sq >> 3;

        /* ── Open / semi-open file ── */
        bool no_own_pawn   = !(our_pawns   & FILE_BB[file]);
        bool no_enemy_pawn = !(their_pawns & FILE_BB[file]);
        if (no_own_pawn && no_enemy_pawn) {
            *mg += ROOK_OPEN_FILE_MG;  *eg += ROOK_OPEN_FILE_EG;
        } else if (no_own_pawn) {
            *mg += ROOK_SEMIOPEN_MG;   *eg += ROOK_SEMIOPEN_EG;
        }

        /* ── Rook on the 7th rank ──
         *  Bonus when the enemy king is on the 8th rank or there are
         *  enemy pawns on the 7th rank to attack.                     */
        if (rank == seventh) {
            if ((b->pieces[them][PAWN] & RANK_BB[seventh]) ||
                (b->pieces[them][KING] & RANK_BB[enemy_8th])) {
                *mg += ROOK_ON_7TH_MG;
                *eg += ROOK_ON_7TH_EG;
            }
        }

        /* ── Connected rooks ──
         *  A rook "sees" another friendly rook when there are no blocking
         *  pieces between them on the same rank or file.                  */
        if (!connected) {
            Bitboard others = rooks & ~sq_bb((Square)sq);
            if (others && (rook_attacks((Square)sq, occ) & others)) {
                *mg += CONNECTED_ROOKS_MG;
                *eg += CONNECTED_ROOKS_EG;
                connected = true; /* award once per side */
            }
        }
    }
}

/* ──────────────────────────────────────────────
 *  King safety: pawn shield + attack-unit danger
 *
 *  Two complementary components:
 *  1. Pawn shield — structural bonus for pawns near the king.
 *  2. Attack units — piece-based attack count against the king zone.
 *     Penalty applied only when ≥ 2 enemy pieces participate (a lone
 *     attacker is usually not dangerous).
 * ────────────────────────────────────────────── */
static void eval_king_safety(const Board *b, Color us,
                              const EvalInfo *ei, int *mg) {
    int k_sq   = ei->king_sq[us];
    int k_file = k_sq & 7;
    int k_rank = k_sq >> 3;
    Bitboard our_pawns = b->pieces[us][PAWN];

    /* ── Pawn shield ── */
    for (int f = k_file - 1; f <= k_file + 1; f++) {
        if (f < 0 || f > 7) continue;
        Bitboard fp = our_pawns & FILE_BB[f];
        if (!fp) {
            *mg += KING_OPEN_FILE_PENALTY;
        } else {
            int s1 = (us == WHITE) ? k_rank + 1 : k_rank - 1;
            int s2 = (us == WHITE) ? k_rank + 2 : k_rank - 2;
            if (s1 >= 0 && s1 < 8 && (fp & RANK_BB[s1]))
                *mg += KING_SHIELD_BONUS * 2;
            else if (s2 >= 0 && s2 < 8 && (fp & RANK_BB[s2]))
                *mg += KING_SHIELD_BONUS;
        }
    }

    /* ── Attack units ── */
    Color them   = us ^ 1;
    Bitboard occ = b->occ[2];
    Bitboard kz  = ei->king_zone[us];
    int attackers = 0, units = 0;

    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        Bitboard pieces = b->pieces[them][pt];
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
            Bitboard zone_hit = atk & kz;
            if (zone_hit) {
                attackers++;
                /* Base weight per attacker type + squares covered in zone */
                units += KING_ATTACK_WEIGHT[pt] + bb_popcount(zone_hit);
            }
        }
    }

    /* A single stray attacker is rarely dangerous — require ≥ 2 */
    if (attackers >= 2)
        *mg -= king_danger_score(units);
}

/* ──────────────────────────────────────────────
 *  Space evaluation
 *
 *  Award a small bonus for safe squares on the four central files
 *  (c–f) in our half of the board that are:
 *   • Not occupied by our own pawns, and
 *   • Not attacked by enemy pawns.
 *
 *  This rewards controlling "elbow room" for piece manoeuvres and is a
 *  rough proxy for the Stockfish "space" term.  MG-only (space matters
 *  little once the board opens up in the endgame).
 * ────────────────────────────────────────────── */
static void eval_space(const Board *b, Color us,
                        const EvalInfo *ei, int *mg) {
    /* Center files c–f (0-indexed: 2,3,4,5) */
    static const Bitboard CENTER_FILES =
        /* computed once; FILE_BB not available as compile-time constant */
        (Bitboard)0; /* initialised in the function body below */
    (void)CENTER_FILES;

    Bitboard cf = FILE_BB[2] | FILE_BB[3] | FILE_BB[4] | FILE_BB[5];
    /* "Our half": ranks 2–4 for White (0-indexed 1–3),
                   ranks 5–7 for Black (0-indexed 4–6)      */
    Bitboard our_half = (us == WHITE)
        ? (RANK_BB[1] | RANK_BB[2] | RANK_BB[3])
        : (RANK_BB[4] | RANK_BB[5] | RANK_BB[6]);

    Bitboard safe = cf & our_half
                  & ~b->pieces[us][PAWN]      /* not blocked by own pawn   */
                  & ~ei->pawn_attacks[us ^ 1]; /* not hit by enemy pawn atk */
    *mg += SPACE_BONUS * bb_popcount(safe);
}

/* ──────────────────────────────────────────────
 *  Main evaluation
 * ────────────────────────────────────────────── */
int evaluate(const Board *b) {
    int mg = 0, eg = 0;
    int phase = game_phase(b);

    EvalInfo ei;
    init_eval_info(b, &ei);

    for (int c = 0; c < 2; c++) {
        int  sign = (c == WHITE) ? 1 : -1;
        int  c_mg = 0, c_eg = 0;
        Color us = (Color)c;

        /* ── Material + PST ── */
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = pst_sq(us, (Square)sq);
                c_mg += MATERIAL_MG[pt] + PST_MG[pt][psq];
                c_eg += MATERIAL_EG[pt] + PST_EG[pt][psq];
            }
        }

        /* ── Pawn structure ── */
        eval_pawns(b, us, &ei, &c_mg, &c_eg);

        /* ── Minor-piece quality ── */
        eval_pieces(b, us, &ei, &c_mg, &c_eg);

        /* ── Mobility ── */
        eval_mobility(b, us, &ei, &c_mg, &c_eg);

        /* ── Rooks ── */
        eval_rooks(b, us, &c_mg, &c_eg);

        /* ── Bishop pair ── */
        if (bb_popcount(b->pieces[c][BISHOP]) >= 2) {
            c_mg += BISHOP_PAIR_MG;
            c_eg += BISHOP_PAIR_EG;
        }

        /* ── King safety ── */
        eval_king_safety(b, us, &ei, &c_mg);

        /* ── Space ── */
        eval_space(b, us, &ei, &c_mg);

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    /* Taper MG/EG blend, then add tempo for the side to move */
    int score = taper(mg, eg, phase);
    int result = (b->side == WHITE) ? score : -score;
    return result + TEMPO;
}