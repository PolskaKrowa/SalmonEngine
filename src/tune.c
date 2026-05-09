/*
 * tune.c — Texel-style multithreaded evaluation tuner (SalmonEngine).
 *
 * Coordinate descent with parallel MSE computation.
 * See tune.h for full documentation.
 */

#include "tune.h"
#include "bitboard.h"
#include "board.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* ══════════════════════════════════════════════════════════════════════
 *  Default weights — must be kept in sync with eval.c.
 *  These are the values the engine ships with; tuning starts here.
 * ══════════════════════════════════════════════════════════════════════ */

static const int DEF_MATERIAL_MG[6] = { 82, 337, 365, 477, 1025, 0 };
static const int DEF_MATERIAL_EG[6] = { 94, 281, 297, 512,  936, 0 };

static const int DEF_PST_PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int DEF_PST_PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int DEF_PST_KNIGHT_MG[64] = {
    -167, -89, -34, -49,  61, -97, -15,-107,
     -73, -41,  72,  36,  23,  62,   7, -17,
     -47,  60,  37,  65,  84, 129,  73,  44,
      -9,  17,  19,  53,  37,  69,  18,  22,
     -13,   4,  16,  13,  28,  19,  21,  -8,
     -23,  -9,  12,  10,  19,  17,  25, -16,
     -29, -53, -12,  -3,  -1,  18, -14, -19,
    -105, -21, -58, -33, -17, -28, -19, -23,
};
static const int DEF_PST_KNIGHT_EG[64] = {
     -58, -38, -13, -28, -31, -27, -63, -99,
     -25,  -8, -25,  -2,  -9, -25, -24, -52,
     -24, -20,  10,   9,  -1,  -9, -19, -41,
     -17,   3,  22,  22,  22,  11,   8, -18,
     -18,  -6,  16,  25,  16,  17,   4, -18,
     -23,  -3,  -1,  15,  10,  -3, -20, -22,
     -42, -20, -10,  -5,  -2, -20, -23, -44,
     -29, -51, -23, -15, -22, -18, -50, -64,
};
static const int DEF_PST_BISHOP_MG[64] = {
     -29,   4, -82, -37, -25, -42,   7,  -8,
     -26,  16, -18, -13,  30,  59,  18, -47,
     -16,  37,  43,  40,  35,  50,  37,  -2,
      -4,   5,  19,  50,  37,  37,   7,  -2,
      -6,  13,  13,  26,  34,  12,  10,   4,
       0,  15,  15,  15,  14,  27,  18,  10,
       4,  15,  16,   0,   7,  21,  33,   1,
     -33,  -3, -14, -21, -13, -12, -39, -21,
};
static const int DEF_PST_BISHOP_EG[64] = {
     -14, -21, -11,  -8,  -7,  -9, -17, -24,
      -8,  -4,   7, -12,  -3, -13,  -4, -14,
       2,  -8,   0,  -1,  -2,   6,   0,   4,
      -3,   9,  12,   9,  14,  10,   3,   2,
      -6,   3,  13,  19,   7,  10,  -3,  -9,
     -12,  -3,   8,  10,  13,   3,  -7, -15,
     -14, -18,  -7,  -1,   4,  -9, -15, -27,
     -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
static const int DEF_PST_ROOK_MG[64] = {
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
};
static const int DEF_PST_ROOK_EG[64] = {
     13, 10, 18, 15, 12,  12,   8,   5,
     11, 13, 13, 11, -3,   3,   8,   3,
      7,  7,  7,  5,  4,  -3,  -5,  -3,
      4,  3, 13,  1,  2,   1,  -1,   2,
      3,  5,  8,  4, -5,  -6,  -8, -11,
     -4,  0, -5, -1, -7, -12,  -8, -16,
     -6, -6,  0,  2, -9,  -9, -11,  -3,
     -9,  2,  3, -1, -5, -13,   4, -20,
};
static const int DEF_PST_QUEEN_MG[64] = {
     -28,   0,  29,  12,  59,  44,  43,  45,
     -24, -39,  -5,   1, -16,  57,  28,  54,
     -13, -17,   7,   8,  29,  56,  47,  57,
     -27, -27, -16, -16,  -1,  17,  -2,   1,
      -9, -26,  -9, -10,  -2,  -4,   3,  -3,
     -14,   2, -11,  -2,  -5,   2,  14,   5,
     -35,  -8,  11,   2,   8,  15,  -3,   1,
      -1, -18,  -9,  10, -15, -25, -31, -50,
};
static const int DEF_PST_QUEEN_EG[64] = {
      -9,  22,  22,  27,  27,  19,  10,  20,
     -17,  20,  32,  41,  58,  25,  30,   0,
     -20,   6,   9,  49,  47,  35,  19,   9,
       3,  22,  24,  45,  57,  40,  57,  36,
     -18,  28,  19,  47,  31,  34,  39,  23,
     -16, -27,  15,   6,   9,  17,  10,   5,
     -22, -23, -30, -16, -16, -23, -36, -32,
     -33, -28, -22, -43,  -5, -32, -20, -41,
};
static const int DEF_PST_KING_MG[64] = {
     -65,  23,  16, -15, -56, -34,   2,  13,
      29,  -1, -20,  -7,  -8,  -4, -38, -29,
      -9,  24,   2, -16, -20,   6,  22, -22,
     -17, -20, -12, -27, -30, -25, -14, -36,
     -49,  -1, -27, -39, -46, -44, -33, -51,
     -14, -14, -22, -46, -44, -30, -15, -27,
       1,   7,  -8, -64, -43, -16,   9,   8,
     -15,  36,  12, -54,   8, -28,  24,  14,
};
static const int DEF_PST_KING_EG[64] = {
     -74, -35, -18, -18, -11,  15,   4, -17,
     -12,  17,  14,  17,  17,  38,  23,  11,
      10,  17,  23,  15,  20,  45,  44,  13,
      -8,  22,  24,  27,  26,  33,  26,   3,
     -18,  -4,  21,  24,  27,  23,   9, -11,
     -19,  -3,  11,  21,  23,  16,   7,  -9,
     -27, -11,   4,  13,  14,   4,  -5, -17,
     -53, -34, -21, -11, -28, -14, -24, -43,
};

/* ── weights_init_defaults ────────────────────────────────────────────── */
void weights_init_defaults(EvalWeights *w) {
    memcpy(w->material_mg, DEF_MATERIAL_MG, sizeof w->material_mg);
    memcpy(w->material_eg, DEF_MATERIAL_EG, sizeof w->material_eg);

    memcpy(w->pst_mg[PAWN],   DEF_PST_PAWN_MG,   64 * sizeof(int));
    memcpy(w->pst_mg[KNIGHT], DEF_PST_KNIGHT_MG, 64 * sizeof(int));
    memcpy(w->pst_mg[BISHOP], DEF_PST_BISHOP_MG, 64 * sizeof(int));
    memcpy(w->pst_mg[ROOK],   DEF_PST_ROOK_MG,   64 * sizeof(int));
    memcpy(w->pst_mg[QUEEN],  DEF_PST_QUEEN_MG,  64 * sizeof(int));
    memcpy(w->pst_mg[KING],   DEF_PST_KING_MG,   64 * sizeof(int));

    memcpy(w->pst_eg[PAWN],   DEF_PST_PAWN_EG,   64 * sizeof(int));
    memcpy(w->pst_eg[KNIGHT], DEF_PST_KNIGHT_EG, 64 * sizeof(int));
    memcpy(w->pst_eg[BISHOP], DEF_PST_BISHOP_EG, 64 * sizeof(int));
    memcpy(w->pst_eg[ROOK],   DEF_PST_ROOK_EG,   64 * sizeof(int));
    memcpy(w->pst_eg[QUEEN],  DEF_PST_QUEEN_EG,  64 * sizeof(int));
    memcpy(w->pst_eg[KING],   DEF_PST_KING_EG,   64 * sizeof(int));

    w->doubled_pawn_mg  = -11; w->doubled_pawn_eg  = -56;
    w->isolated_pawn_mg = -15; w->isolated_pawn_eg = -15;
    w->backward_pawn_mg =  -9; w->backward_pawn_eg = -22;

    static const int def_pmg[8] = {  0,  5, 10,  20,  35,  60,  90,  0 };
    static const int def_peg[8] = {  0, 10, 20,  40,  65,  95, 140,  0 };
    memcpy(w->passed_pawn_mg, def_pmg, sizeof w->passed_pawn_mg);
    memcpy(w->passed_pawn_eg, def_peg, sizeof w->passed_pawn_eg);

    static const int def_mmg[6] = { 0, 4, 3, 2, 1, 0 };
    static const int def_meg[6] = { 0, 4, 3, 3, 2, 0 };
    memcpy(w->mobility_mg, def_mmg, sizeof w->mobility_mg);
    memcpy(w->mobility_eg, def_meg, sizeof w->mobility_eg);

    w->rook_open_mg    = 23; w->rook_open_eg    = 15;
    w->rook_semi_mg    = 12; w->rook_semi_eg    =  7;
    w->rook_seventh_mg = 20; w->rook_seventh_eg = 32;

    w->bishop_pair_mg = 30; w->bishop_pair_eg = 60;

    w->outpost_knight_mg = 22; w->outpost_knight_eg = 14;
    w->outpost_bishop_mg = 12; w->outpost_bishop_eg =  8;

    w->king_shield    =   7;
    w->king_open_file = -25;

    static const int def_kaw[6] = { 0, 20, 20, 40, 80, 0 };
    memcpy(w->king_attacker_weight, def_kaw, sizeof w->king_attacker_weight);

    w->tempo_mg = 14; w->tempo_eg = 8;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Parameterised evaluator
 *
 *  Mirrors eval.c exactly, but every static constant is read from *w.
 *  This is the hot path; keep it inlined and free of allocations.
 * ══════════════════════════════════════════════════════════════════════ */

static const int PHASE_INC_W[6] = { 0, 1, 1, 2, 4, 0 };
#define MAX_PHASE_W 24

static inline int game_phase_w(const Board *b) {
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_INC_W[pt] * bb_popcount(b->pieces[c][pt]);
    return (phase > MAX_PHASE_W) ? MAX_PHASE_W : phase;
}

static inline int taper_w(int mg, int eg, int phase) {
    return (mg * phase + eg * (MAX_PHASE_W - phase)) / MAX_PHASE_W;
}

static inline int pst_sq_w(Color c, int sq) {
    return (c == WHITE) ? sq : (sq ^ 56);
}

/* ── Pawn structure ─────────────────────────────────────────────────── */
static void eval_pawns_w(const Board *b, Color us, int *mg, int *eg,
                          const EvalWeights *w) {
    Color    them        = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard copy        = our_pawns;

    while (copy) {
        int sq   = bb_pop(&copy);
        int file = sq & 7;
        int rank = sq >> 3;

        Bitboard adj = 0;
        if (file > 0) adj |= FILE_BB[file - 1];
        if (file < 7) adj |= FILE_BB[file + 1];

        /* Doubled pawn */
        if (bb_popcount(our_pawns & FILE_BB[file]) > 1) {
            *mg += w->doubled_pawn_mg;
            *eg += w->doubled_pawn_eg;
        }

        /* Isolated pawn */
        bool is_isolated = !(our_pawns & adj);
        if (is_isolated) {
            *mg += w->isolated_pawn_mg;
            *eg += w->isolated_pawn_eg;
        }

        /* Passed pawn — no enemy pawn on same or adjacent files ahead */
        Bitboard ahead = 0;
        if (us == WHITE) { for (int r = rank + 1; r < 8; r++) ahead |= RANK_BB[r]; }
        else             { for (int r = 0; r < rank; r++)      ahead |= RANK_BB[r]; }
        bool is_passed = !((their_pawns & (FILE_BB[file] | adj)) & ahead);
        if (is_passed) {
            int br = (us == WHITE) ? rank : (7 - rank);
            *mg += w->passed_pawn_mg[br];
            *eg += w->passed_pawn_eg[br];
        }

        /* Backward pawn */
        if (!is_passed) {
            int stop = (us == WHITE) ? sq + 8 : sq - 8;
            if (stop >= 0 && stop < 64) {
                bool stop_attacked = (PAWN_ATTACKS[us][stop] & their_pawns) != 0;
                if (stop_attacked) {
                    Bitboard below  = SQUARE_BB[sq] - 1;
                    Bitboard above  = ~SQUARE_BB[sq] & ~below;
                    Bitboard behind = ((us == WHITE) ? below : above)
                                      & adj & ~RANK_BB[rank];
                    if (!(our_pawns & behind)) {
                        *mg += w->backward_pawn_mg;
                        *eg += w->backward_pawn_eg;
                    }
                }
            }
        }
    }
}

/* ── Mobility ───────────────────────────────────────────────────────── */
static void eval_mobility_w(const Board *b, Color us, int *mg, int *eg,
                              const EvalWeights *w) {
    Bitboard occ    = b->occ[2];
    Bitboard not_us = ~b->occ[us];

    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk = 0;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq];              break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default: break;
            }
            int mob = bb_popcount(atk & not_us);
            *mg += w->mobility_mg[pt] * mob;
            *eg += w->mobility_eg[pt] * mob;
        }
    }
}

/* ── Rook bonuses ───────────────────────────────────────────────────── */
static void eval_rooks_w(const Board *b, Color us, int *mg, int *eg,
                           const EvalWeights *w) {
    Color    them        = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard rooks       = b->pieces[us  ][ROOK];
    int      rank7       = (us == WHITE) ? 6 : 1;
    int      rank8       = (us == WHITE) ? 7 : 0;

    while (rooks) {
        int sq   = bb_pop(&rooks);
        int file = sq & 7;
        int rank = sq >> 3;
        Bitboard fb = FILE_BB[file];

        bool no_own   = !(our_pawns   & fb);
        bool no_enemy = !(their_pawns & fb);
        if (no_own && no_enemy) { *mg += w->rook_open_mg; *eg += w->rook_open_eg; }
        else if (no_own)        { *mg += w->rook_semi_mg; *eg += w->rook_semi_eg; }

        if (rank == rank7) {
            int      ek8th = bb_lsb(b->pieces[them][KING]) >> 3;
            bool     cond  = (ek8th == rank8) || (their_pawns & RANK_BB[rank7]);
            if (cond) { *mg += w->rook_seventh_mg; *eg += w->rook_seventh_eg; }
        }
    }
}

/* ── Outposts ────────────────────────────────────────────────────────── */
static void eval_outposts_w(const Board *b, Color us, int *mg, int *eg,
                              const EvalWeights *w) {
    Color    them        = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];

    /* Build enemy pawn attack span (files ±1 ahead of each enemy pawn) */
    Bitboard enemy_span = 0;
    {
        Bitboard ep = their_pawns;
        while (ep) {
            int sq   = bb_pop(&ep);
            int file = sq & 7;
            int rank = sq >> 3;
            Bitboard adj = 0;
            if (file > 0) adj |= FILE_BB[file - 1];
            if (file < 7) adj |= FILE_BB[file + 1];
            if (them == BLACK) {
                for (int r = rank - 1; r >= 0; r--) enemy_span |= RANK_BB[r] & adj;
            } else {
                for (int r = rank + 1; r < 8;  r++) enemy_span |= RANK_BB[r] & adj;
            }
        }
    }

    /* Outpost rank mask: ranks 4–6 from our perspective */
    int      r_min       = (us == WHITE) ? 3 : 2;
    int      r_max       = (us == WHITE) ? 5 : 4;
    Bitboard outpost_msk = 0;
    for (int r = r_min; r <= r_max; r++) outpost_msk |= RANK_BB[r];

    /* Knights */
    Bitboard kn = b->pieces[us][KNIGHT];
    while (kn) {
        int sq = bb_pop(&kn);
        if (!(SQUARE_BB[sq] & outpost_msk))        continue;
        if (enemy_span & SQUARE_BB[sq])             continue;
        if (!(PAWN_ATTACKS[them][sq] & our_pawns))  continue;
        *mg += w->outpost_knight_mg;
        *eg += w->outpost_knight_eg;
    }

    /* Bishops */
    Bitboard bi = b->pieces[us][BISHOP];
    while (bi) {
        int sq = bb_pop(&bi);
        if (!(SQUARE_BB[sq] & outpost_msk))        continue;
        if (enemy_span & SQUARE_BB[sq])             continue;
        if (!(PAWN_ATTACKS[them][sq] & our_pawns))  continue;
        *mg += w->outpost_bishop_mg;
        *eg += w->outpost_bishop_eg;
    }
}

/* ── King safety ─────────────────────────────────────────────────────── */
static void eval_king_safety_w(const Board *b, Color us, int *mg,
                                 const EvalWeights *w) {
    Color    them      = us ^ 1;
    int      ksq       = bb_lsb(b->pieces[us][KING]);
    int      king_file = ksq & 7;
    int      king_rank = ksq >> 3;
    Bitboard our_pawns = b->pieces[us][PAWN];

    /* 1. Pawn shield + open-file penalty */
    for (int f = king_file - 1; f <= king_file + 1; f++) {
        if (f < 0 || f > 7) continue;
        Bitboard fp = our_pawns & FILE_BB[f];
        if (!fp) {
            *mg += w->king_open_file;
        } else {
            int s1 = (us == WHITE) ? king_rank + 1 : king_rank - 1;
            int s2 = (us == WHITE) ? king_rank + 2 : king_rank - 2;
            if (s1 >= 0 && s1 < 8 && (fp & RANK_BB[s1]))      *mg += w->king_shield * 2;
            else if (s2 >= 0 && s2 < 8 && (fp & RANK_BB[s2])) *mg += w->king_shield;
        }
    }

    /* 2. Enemy piece attacks on king zone (quadratic danger model) */
    Bitboard king_zone    = KING_ATTACKS[ksq] | SQUARE_BB[ksq];
    Bitboard occ          = b->occ[2];
    int      attack_count = 0, attack_weight = 0;

    for (int pt = KNIGHT; pt < KING; pt++) {
        Bitboard pieces = b->pieces[them][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq];              break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default:     atk = 0;                                break;
            }
            if (atk & king_zone) {
                attack_count++;
                attack_weight += w->king_attacker_weight[pt];
            }
        }
    }
    if (attack_count >= 2)
        *mg -= (attack_weight * attack_weight) / 200;
}

/* ── Main parameterised evaluation ──────────────────────────────────── */
int evaluate_w(const Board *b, const EvalWeights *w) {
    int mg = 0, eg = 0;
    int phase = game_phase_w(b);

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        int c_mg = 0, c_eg = 0;

        /* Material + PST */
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = pst_sq_w((Color)c, sq);
                c_mg += w->material_mg[pt] + w->pst_mg[pt][psq];
                c_eg += w->material_eg[pt] + w->pst_eg[pt][psq];
            }
        }

        eval_pawns_w      (b, (Color)c, &c_mg, &c_eg, w);
        eval_mobility_w   (b, (Color)c, &c_mg, &c_eg, w);
        eval_rooks_w      (b, (Color)c, &c_mg, &c_eg, w);
        eval_outposts_w   (b, (Color)c, &c_mg, &c_eg, w);

        if (bb_popcount(b->pieces[c][BISHOP]) >= 2) {
            c_mg += w->bishop_pair_mg;
            c_eg += w->bishop_pair_eg;
        }

        eval_king_safety_w(b, (Color)c, &c_mg, w);

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    int score  = taper_w(mg, eg, phase);
    score     += taper_w(w->tempo_mg, w->tempo_eg, phase);
    return (b->side == WHITE) ? score : -score;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Position corpus
 * ══════════════════════════════════════════════════════════════════════ */

#define MAX_POSITIONS 2000000

typedef struct {
    Board  board;
    double result; /* 1.0 = White win, 0.5 = draw, 0.0 = Black win */
} TunePos;

static TunePos *g_pos  = NULL;
static int      g_npos = 0;

/* Parse a result annotation → 0.0 / 0.5 / 1.0, or -1.0 on failure */
static double parse_result(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '"' || *s == '[') s++;
    if (strncmp(s, "1-0",     3) == 0) return 1.0;
    if (strncmp(s, "0-1",     3) == 0) return 0.0;
    if (strncmp(s, "1/2-1/2", 7) == 0) return 0.5;
    /* Try numeric */
    char *end;
    double v = strtod(s, &end);
    if (end != s && v >= 0.0 && v <= 1.0) return v;
    return -1.0;
}

/*
 * Skip n whitespace-delimited tokens in str.
 * Returns pointer to the start of the (n+1)-th token, or end of string.
 */
static const char *skip_tokens(const char *s, int n) {
    for (int i = 0; i < n; i++) {
        while (*s && *s != ' ' && *s != '\t') s++; /* skip token  */
        while (*s == ' ' || *s == '\t')       s++; /* skip spaces */
    }
    return s;
}

/*
 * Load annotated positions.  Returns number loaded, or -1 on I/O error.
 *
 * Supported per-line formats:
 *   <FEN 6-tokens> [<result>]
 *   <FEN 6-tokens> c9 "<result>";
 *   <FEN 6-tokens> <float>
 */
static int load_epd(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }

    g_pos = malloc(MAX_POSITIONS * sizeof *g_pos);
    if (!g_pos) { fclose(f); return -1; }

    char line[1024];
    g_npos = 0;

    while (fgets(line, sizeof line, f) && g_npos < MAX_POSITIONS) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0] || line[0] == '#') continue;

        /* The FEN is the first six tokens */
        const char *rest = skip_tokens(line, 6);
        if (!*rest) continue; /* no annotation */

        /* Copy FEN (everything before rest) */
        int fen_len = (int)(rest - line);
        while (fen_len > 0 && (line[fen_len-1] == ' ' || line[fen_len-1] == '\t'))
            fen_len--;
        if (fen_len <= 0 || fen_len >= 128) continue;

        char fen[128];
        memcpy(fen, line, (size_t)fen_len);
        fen[fen_len] = '\0';

        /* Parse result from annotation */
        double result = parse_result(rest);
        if (result < 0.0) continue;

        /* Attempt to parse FEN — requires board_from_fen() in board.c */
        if (board_from_fen(&g_pos[g_npos].board, fen) != 0) continue;
        g_pos[g_npos].result = result;
        g_npos++;
    }

    fclose(f);
    printf("[tune] Loaded %d positions from '%s'\n", g_npos, path);
    return g_npos;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Multithreaded MSE computation
 * ══════════════════════════════════════════════════════════════════════ */

/*
 * Sigmoid scaling constant K.
 * A value of 1.13 is typical for centipawn evaluations.
 * Dividing by 400 converts centipawns to the [0,1] probability domain.
 * Tune K separately with tune_find_k() if needed.
 */
static double g_K = 1.13 / 400.0;

static inline double sigmoid(double cp) {
    return 1.0 / (1.0 + exp(-g_K * cp));
}

/* Per-thread work descriptor */
typedef struct {
    const EvalWeights *w;       /* shared read-only weights         */
    int                start;   /* first position index (inclusive) */
    int                end;     /* last  position index (exclusive) */
    double             partial; /* output: partial MSE sum          */
} WorkerArg;

static void *worker_fn(void *arg) {
    WorkerArg *wa = (WorkerArg *)arg;
    double     err = 0.0;
    for (int i = wa->start; i < wa->end; i++) {
        double s = sigmoid((double)evaluate_w(&g_pos[i].board, wa->w));
        double d = s - g_pos[i].result;
        err += d * d;
    }
    wa->partial = err;
    return NULL;
}

/*
 * Compute MSE over the entire corpus using nthreads worker threads.
 * Returns the average squared error per position.
 */
static double compute_error(const EvalWeights *w, int nthreads) {
    if (g_npos == 0) return 0.0;

    /* Single-threaded fast path */
    if (nthreads <= 1) {
        WorkerArg wa = { w, 0, g_npos, 0.0 };
        worker_fn(&wa);
        return wa.partial / (double)g_npos;
    }

    /* Allocate thread handles and argument blocks */
    pthread_t *threads = malloc((size_t)nthreads * sizeof *threads);
    WorkerArg *args    = malloc((size_t)nthreads * sizeof *args);
    if (!threads || !args) {
        free(threads); free(args);
        /* Fallback: single-threaded */
        WorkerArg wa = { w, 0, g_npos, 0.0 };
        worker_fn(&wa);
        return wa.partial / (double)g_npos;
    }

    int chunk = g_npos / nthreads;
    for (int t = 0; t < nthreads; t++) {
        args[t].w       = w;
        args[t].start   = t * chunk;
        args[t].end     = (t == nthreads - 1) ? g_npos : (t + 1) * chunk;
        args[t].partial = 0.0;
        pthread_create(&threads[t], NULL, worker_fn, &args[t]);
    }

    double total = 0.0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        total += args[t].partial;
    }

    free(threads);
    free(args);
    return total / (double)g_npos;
}

/* ══════════════════════════════════════════════════════════════════════
 *  K calibration (Golden-section search)
 *
 *  Finds the K value that minimises MSE with the default weights.
 *  Run once before tune_run() to calibrate K for your corpus.
 * ══════════════════════════════════════════════════════════════════════ */
double tune_find_k(int nthreads) {
    printf("[tune] Calibrating K...\n");
    EvalWeights w;
    weights_init_defaults(&w);

    double lo = 0.5 / 400.0, hi = 4.0 / 400.0;
    const double phi = 0.6180339887; /* 1 - 1/golden ratio */
    const int    iters = 30;

    for (int i = 0; i < iters; i++) {
        double m1 = hi - phi * (hi - lo);
        double m2 = lo + phi * (hi - lo);
        g_K = m1; double e1 = compute_error(&w, nthreads);
        g_K = m2; double e2 = compute_error(&w, nthreads);
        if (e1 < e2) hi = m2; else lo = m1;
    }

    g_K = (lo + hi) / 2.0;
    printf("[tune] Optimal K = %.6f  (raw centipawn scale: %.4f)\n",
           g_K, g_K * 400.0);
    return g_K;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Parameter registry
 *
 *  Registers every tunable int in EvalWeights as a Param with its
 *  pointer, clamping bounds, and a human-readable name for diagnostics.
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int        *ptr;
    int         lo, hi;
    const char *name;
} Param;

#define MAX_PARAMS 1024
static Param g_params[MAX_PARAMS];
static int   g_nparams = 0;

static void reg(int *ptr, int lo, int hi, const char *name) {
    if (g_nparams >= MAX_PARAMS) return;
    g_params[g_nparams++] = (Param){ ptr, lo, hi, name };
}

static void register_params(EvalWeights *w) {
    g_nparams = 0;

    /* Material (skip KING — always 0 in tapered engines) */
    for (int pt = PAWN; pt < KING; pt++) {
        reg(&w->material_mg[pt], 50, 2000, "material_mg");
        reg(&w->material_eg[pt], 50, 2000, "material_eg");
    }

    /* PST — all piece types, all 64 squares.
     * King PST is tuned but with tighter bounds for stability. */
    for (int pt = 0; pt < 6; pt++) {
        int lo = (pt == KING) ? -120 : -200;
        int hi = (pt == KING) ?  120 :  200;
        for (int sq = 0; sq < 64; sq++) {
            reg(&w->pst_mg[pt][sq], lo, hi, "pst_mg");
            reg(&w->pst_eg[pt][sq], lo, hi, "pst_eg");
        }
    }

    /* Pawn structure — penalties are negative, bonuses positive */
    reg(&w->doubled_pawn_mg,  -150,   0, "doubled_pawn_mg");
    reg(&w->doubled_pawn_eg,  -150,   0, "doubled_pawn_eg");
    reg(&w->isolated_pawn_mg, -100,   0, "isolated_pawn_mg");
    reg(&w->isolated_pawn_eg, -100,   0, "isolated_pawn_eg");
    reg(&w->backward_pawn_mg, -100,   0, "backward_pawn_mg");
    reg(&w->backward_pawn_eg, -100,   0, "backward_pawn_eg");
    /* Passed pawn bonuses — skip rank 0 and 7 (always 0 by design) */
    for (int i = 1; i < 7; i++) {
        reg(&w->passed_pawn_mg[i],   0, 250, "passed_pawn_mg");
        reg(&w->passed_pawn_eg[i],   0, 350, "passed_pawn_eg");
    }

    /* Mobility — must stay non-negative */
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        reg(&w->mobility_mg[pt], 0, 20, "mobility_mg");
        reg(&w->mobility_eg[pt], 0, 20, "mobility_eg");
    }

    /* Rook bonuses */
    reg(&w->rook_open_mg,     0,  80, "rook_open_mg");
    reg(&w->rook_open_eg,     0,  80, "rook_open_eg");
    reg(&w->rook_semi_mg,     0,  50, "rook_semi_mg");
    reg(&w->rook_semi_eg,     0,  50, "rook_semi_eg");
    reg(&w->rook_seventh_mg,  0,  80, "rook_seventh_mg");
    reg(&w->rook_seventh_eg,  0,  80, "rook_seventh_eg");

    /* Bishop pair */
    reg(&w->bishop_pair_mg,   0, 100, "bishop_pair_mg");
    reg(&w->bishop_pair_eg,   0, 100, "bishop_pair_eg");

    /* Outposts */
    reg(&w->outpost_knight_mg, 0, 60, "outpost_knight_mg");
    reg(&w->outpost_knight_eg, 0, 60, "outpost_knight_eg");
    reg(&w->outpost_bishop_mg, 0, 40, "outpost_bishop_mg");
    reg(&w->outpost_bishop_eg, 0, 40, "outpost_bishop_eg");

    /* King safety */
    reg(&w->king_shield,       0,  40, "king_shield");
    reg(&w->king_open_file, -100,   0, "king_open_file");
    for (int pt = KNIGHT; pt <= QUEEN; pt++)
        reg(&w->king_attacker_weight[pt], 0, 200, "king_attacker_weight");

    /* Tempo */
    reg(&w->tempo_mg, 0, 40, "tempo_mg");
    reg(&w->tempo_eg, 0, 40, "tempo_eg");
}

/* ══════════════════════════════════════════════════════════════════════
 *  Coordinate descent
 * ══════════════════════════════════════════════════════════════════════ */

/* Fisher-Yates shuffle of an integer index array */
static void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j   = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  Weight serialiser
 *
 *  Emits the tuned weights as C source code that can be pasted directly
 *  into eval.c, replacing the existing static const declarations.
 * ══════════════════════════════════════════════════════════════════════ */
static void save_weights(const EvalWeights *w, const char *path) {
    FILE *out = path ? fopen(path, "w") : stdout;
    if (!out) { perror(path); return; }

    static const char *pt_name[6] = {
        "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"
    };

    fprintf(out,
        "/*\n"
        " * Tuned weights — generated by SalmonEngine Texel tuner.\n"
        " * Paste into eval.c, replacing the static const declarations.\n"
        " */\n\n");

    /* Material */
    fprintf(out,
        "static const int MATERIAL_MG[6] = { %d, %d, %d, %d, %d, %d };\n",
        w->material_mg[0], w->material_mg[1], w->material_mg[2],
        w->material_mg[3], w->material_mg[4], w->material_mg[5]);
    fprintf(out,
        "static const int MATERIAL_EG[6] = { %d, %d, %d, %d, %d, %d };\n\n",
        w->material_eg[0], w->material_eg[1], w->material_eg[2],
        w->material_eg[3], w->material_eg[4], w->material_eg[5]);

    /* PST tables */
    for (int pt = 0; pt < 6; pt++) {
        for (int phase = 0; phase < 2; phase++) {
            const int *tbl = phase ? w->pst_eg[pt] : w->pst_mg[pt];
            fprintf(out, "static const int PST_%s_%s[64] = {\n",
                    pt_name[pt], phase ? "EG" : "MG");
            for (int r = 0; r < 8; r++) {
                fprintf(out, "   ");
                for (int file = 0; file < 8; file++)
                    fprintf(out, " %5d,", tbl[r * 8 + file]);
                fprintf(out, "\n");
            }
            fprintf(out, "};\n");
        }
        fprintf(out, "\n");
    }

    /* Pawn structure */
    fprintf(out, "static const int DOUBLED_PAWN_PENALTY_MG  = %d;\n", w->doubled_pawn_mg);
    fprintf(out, "static const int DOUBLED_PAWN_PENALTY_EG  = %d;\n", w->doubled_pawn_eg);
    fprintf(out, "static const int ISOLATED_PAWN_PENALTY_MG = %d;\n", w->isolated_pawn_mg);
    fprintf(out, "static const int ISOLATED_PAWN_PENALTY_EG = %d;\n", w->isolated_pawn_eg);
    fprintf(out, "static const int BACKWARD_PAWN_PENALTY_MG = %d;\n", w->backward_pawn_mg);
    fprintf(out, "static const int BACKWARD_PAWN_PENALTY_EG = %d;\n", w->backward_pawn_eg);

    fprintf(out, "static const int PASSED_PAWN_BONUS_MG[8]  = {");
    for (int i = 0; i < 8; i++) fprintf(out, " %d,", w->passed_pawn_mg[i]);
    fprintf(out, " };\n");
    fprintf(out, "static const int PASSED_PAWN_BONUS_EG[8]  = {");
    for (int i = 0; i < 8; i++) fprintf(out, " %d,", w->passed_pawn_eg[i]);
    fprintf(out, " };\n\n");

    /* Mobility */
    fprintf(out, "static const int MOBILITY_MG[6] = {");
    for (int i = 0; i < 6; i++) fprintf(out, " %d,", w->mobility_mg[i]);
    fprintf(out, " };\n");
    fprintf(out, "static const int MOBILITY_EG[6] = {");
    for (int i = 0; i < 6; i++) fprintf(out, " %d,", w->mobility_eg[i]);
    fprintf(out, " };\n\n");

    /* Rook */
    fprintf(out, "static const int ROOK_OPEN_FILE_MG  = %d;\n", w->rook_open_mg);
    fprintf(out, "static const int ROOK_OPEN_FILE_EG  = %d;\n", w->rook_open_eg);
    fprintf(out, "static const int ROOK_SEMIOPEN_MG   = %d;\n", w->rook_semi_mg);
    fprintf(out, "static const int ROOK_SEMIOPEN_EG   = %d;\n", w->rook_semi_eg);
    fprintf(out, "static const int ROOK_ON_SEVENTH_MG = %d;\n", w->rook_seventh_mg);
    fprintf(out, "static const int ROOK_ON_SEVENTH_EG = %d;\n\n", w->rook_seventh_eg);

    /* Bishop */
    fprintf(out, "static const int BISHOP_PAIR_MG = %d;\n",   w->bishop_pair_mg);
    fprintf(out, "static const int BISHOP_PAIR_EG = %d;\n\n", w->bishop_pair_eg);

    /* Outposts */
    fprintf(out, "static const int OUTPOST_KNIGHT_MG = %d;\n", w->outpost_knight_mg);
    fprintf(out, "static const int OUTPOST_KNIGHT_EG = %d;\n", w->outpost_knight_eg);
    fprintf(out, "static const int OUTPOST_BISHOP_MG = %d;\n", w->outpost_bishop_mg);
    fprintf(out, "static const int OUTPOST_BISHOP_EG = %d;\n\n", w->outpost_bishop_eg);

    /* King safety */
    fprintf(out, "static const int KING_SHIELD_BONUS        = %d;\n", w->king_shield);
    fprintf(out, "static const int KING_OPEN_FILE_PENALTY   = %d;\n", w->king_open_file);
    fprintf(out, "static const int KING_ATTACKER_WEIGHT[6]  = {");
    for (int i = 0; i < 6; i++) fprintf(out, " %d,", w->king_attacker_weight[i]);
    fprintf(out, " };\n\n");

    /* Tempo */
    fprintf(out, "#define TEMPO_BONUS_MG %d\n", w->tempo_mg);
    fprintf(out, "#define TEMPO_BONUS_EG %d\n", w->tempo_eg);

    if (path) {
        fclose(out);
        printf("[tune] Weights written to '%s'\n", path);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  tune_run — main entry point
 * ══════════════════════════════════════════════════════════════════════ */
void tune_run(const char *epd_path, int nthreads, int max_iters,
              const char *out_path) {
    srand((unsigned)time(NULL));

    /* ── 1. Load corpus ── */
    if (load_epd(epd_path) <= 0) {
        fprintf(stderr, "[tune] No positions loaded — aborting.\n");
        return;
    }

    /* ── 2. Initialise weights from eval.c defaults ── */
    EvalWeights w;
    weights_init_defaults(&w);

    /* ── 3. Register all tunable parameters ── */
    register_params(&w);
    printf("[tune] %d tunable parameters registered\n", g_nparams);
    printf("[tune] K = %.6f (%.4f × 400)\n\n", g_K, g_K * 400.0);

    /* ── 4. Compute initial error ── */
    double best_err = compute_error(&w, nthreads);
    printf("[tune] Initial MSE: %.8f\n\n", best_err);

    /* ── 5. Build shuffled index array for random parameter ordering ── */
    int *idx = malloc((size_t)g_nparams * sizeof *idx);
    if (!idx) { fprintf(stderr, "[tune] OOM\n"); free(g_pos); return; }
    for (int i = 0; i < g_nparams; i++) idx[i] = i;

    /* ── 6. Coordinate descent ── */
    for (int iter = 0; iter < max_iters; iter++) {
        int improved = 0;
        time_t t0 = time(NULL);
        shuffle(idx, g_nparams);

        for (int i = 0; i < g_nparams; i++) {
            Param *p = &g_params[idx[i]];

            /* Try increment (+1) */
            if (*p->ptr < p->hi) {
                (*p->ptr)++;
                double e = compute_error(&w, nthreads);
                if (e < best_err) { best_err = e; improved++; continue; }
                (*p->ptr)--;
            }

            /* Try decrement (−1) */
            if (*p->ptr > p->lo) {
                (*p->ptr)--;
                double e = compute_error(&w, nthreads);
                if (e < best_err) { best_err = e; improved++; continue; }
                (*p->ptr)++;
            }
            /* No improvement — leave parameter unchanged */
        }

        long elapsed = (long)(time(NULL) - t0);
        printf("[tune] Iter %3d/%d  MSE=%.8f  improved=%d  time=%lds\n",
               iter + 1, max_iters, best_err, improved, elapsed);
        fflush(stdout);

        if (improved == 0) {
            printf("[tune] No improvement in full sweep — converged after %d iterations.\n",
                   iter + 1);
            break;
        }
    }

    /* ── 7. Save results ── */
    printf("\n[tune] Tuning complete.  Final MSE: %.8f\n", best_err);
    save_weights(&w, out_path);

    free(idx);
    free(g_pos);
    g_pos  = NULL;
    g_npos = 0;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Optional standalone main
 *
 *  Compile with -DTUNE_STANDALONE to build a dedicated tuner binary
 *  separate from the UCI engine:
 *
 *    gcc -O3 -march=native -pthread -lm -DTUNE_STANDALONE \
 *        tune.c bitboard.c board.c -o tuner
 * ══════════════════════════════════════════════════════════════════════ */
#ifdef TUNE_STANDALONE
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <epd_file> [threads] [max_iters] [out_file]\n"
            "\n"
            "  epd_file   Annotated FEN file (result per line)\n"
            "  threads    Worker threads  (default: 4)\n"
            "  max_iters  Max sweeps      (default: 1000)\n"
            "  out_file   Output C source (default: tuned_weights.c)\n",
            argv[0]);
        return 1;
    }

    const char *epd      = argv[1];
    int         threads  = (argc >= 3) ? atoi(argv[2]) : 4;
    int         iters    = (argc >= 4) ? atoi(argv[3]) : 1000;
    const char *out      = (argc >= 5) ? argv[4]       : "tuned_weights.c";

    if (threads < 1)  threads = 1;
    if (threads > 256) threads = 256;
    if (iters   < 1)  iters   = 1;

    /* Initialise engine subsystems */
    bitboard_init();
    board_init();

    /* Calibrate K on the corpus first */
    /* Load corpus temporarily for K calibration */
    if (load_epd(epd) > 0) {
        tune_find_k(threads);
        free(g_pos); g_pos = NULL; g_npos = 0;
    }

    /* Run the tuner */
    tune_run(epd, threads, iters, out);
    return 0;
}
#endif /* TUNE_STANDALONE */