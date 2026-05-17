/*
 * tune.c — Self-Play Reinforcement Learning Tuner (SalmonEngine)
 *
 * DISTRIBUTED-READY ARCHITECTURE:
 * This system is decoupled into two primary components:
 * 1. self_play_worker(): Plays games against itself using current weights,
 * generating a dataset of (Position, Result) pairs.
 * 2. optimize_dataset(): Runs Coordinate Descent on a generated dataset.
 *
 * To distribute this later:
 * - Have worker nodes run `self_play_worker()` and save the output to .epd files.
 * - Have the master node load the .epd files, run `optimize_dataset()`, 
 * and broadcast the new weights back to the workers.
 */

#include "tune.h"
#include "bitboard.h"
#include "board.h"
#include "movegen.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* ── Threading ───────────────────────────────────────────────────────── */
#define MAX_TUNE_THREADS 128
static int g_num_threads = 1;

/* Portable, re-entrant replacement for rand_r().
 * Each thread keeps its own seed — no shared state, no locks needed.
 * Returns a value in [0, 0x7fffffff]. */
static inline int tune_rand(unsigned int *seed) {
    *seed = *seed * 1664525u + 1013904223u;   /* Numerical Recipes LCG */
    return (int)((*seed >> 1) & 0x7fffffff);
}

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
 
    /* Material (synced to eval.c) */
    static const int D_MAT_MG[6] = { 82, 344, 358, 480, 1022, 0 };
    static const int D_MAT_EG[6] = { 94, 338, 329, 546,  924, 0 };
    memcpy(w->material_mg, D_MAT_MG, sizeof w->material_mg);
    memcpy(w->material_eg, D_MAT_EG, sizeof w->material_eg);
 
    /* PST (synced to eval.c) */
    static const int D_PAWN_MG[64] = {
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  65,  56,  25, -20,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    };
    static const int D_PAWN_EG[64] = {
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    };
    static const int D_KNIGHT_MG[64] = {
        -167, -89, -34, -49,  61, -97, -15,-107,
         -73, -41,  72,  36,  23,  62,   7, -17,
         -47,  60,  37,  65,  84, 129,  73,  44,
          -9,  17,  19,  53,  37,  69,  18,  22,
         -13,   4,  16,  13,  28,  19,  21,  -8,
         -23,  -9,  12,  10,  19,  17,  25, -16,
         -29, -53, -12,  -3,  -1,  18, -14, -19,
         -99, -21, -58, -33, -17, -28, -19, -23,
    };
    static const int D_KNIGHT_EG[64] = {
         -58, -38, -13, -28, -31, -27, -63, -99,
         -25,  -8, -25,  -2,  -9, -25, -24, -52,
         -24, -20,  10,   9,  -1,  -9, -19, -41,
         -17,   3,  22,  22,  22,  11,   8, -18,
         -18,  -6,  16,  25,  16,  17,   4, -18,
         -23,  -3,  -1,  15,  10,  -3, -20, -22,
         -42, -20, -10,  -5,  -2, -20, -23, -44,
          27, -51, -23, -15, -22, -18, -50, -64,
    };
    static const int D_BISHOP_MG[64] = {
         -30, -22, -82, -37, -25, -42,   7,  -8,
         -26,  16, -18, -13,  30,  59,  18, -47,
         -16,  37,  43,  40,  35,  50,  37,  -2,
          -4,   5,  19,  50,  37,  37,   7,  -2,
          -6,  13,  13,  26,  34,  12,  10,   4,
           0,  15,  15,  15,  14,  27,  18,  10,
           4,  15,  16,   0,   7,  21,  33,   1,
         -26,  -3, -14, -21, -13, -12, -39, -21,
    };
    static const int D_BISHOP_EG[64] = {
         -70, -75, -11,  -8,  -7,  -9, -17, -24,
          -8,  -4,   7, -12,  -3, -13,  -4, -14,
           2,  -8,   0,  -1,  -2,   6,   0,   4,
          -3,   9,  12,   9,  14,  10,   3,   2,
          -6,   3,  13,  19,   7,  10,  -3,  -9,
         -12,  -3,   8,  10,  13,   3,  -7, -15,
         -14, -18,  -7,  -1,   4,  -9, -15, -27,
          29,  -9, -23,  -5,  -9, -16,  -5, -17,
    };
    static const int D_ROOK_MG[64] = {
          32,  42,  32,  51,  63,   9,  31,  43,
          27,  32,  58,  62,  80,  67,  26,  44,
          -5,  19,  26,  36,  17,  45,  61,  16,
         -24, -11,   7,  26,  24,  35,  -8, -20,
         -36, -26, -12,  -1,   9,  -7,   6, -23,
         -45, -25, -16, -17,   3,   0,  -5, -33,
         -44, -16, -20,  -9,  -1,  11,  -6, -71,
         -13, -13,   1,  17,  16,   7, -37, -26,
    };
    static const int D_ROOK_EG[64] = {
          13,  10,  18,  15,  12,  12,   8,   5,
          11,  13,  13,  11,  -3,   3,   8,   3,
           7,   7,   7,   5,   4,  -3,  -5,  -3,
           4,   3,  13,   1,   2,   1,  -1,   2,
           3,   5,   8,   4,  -5,  -6,  -8, -11,
          -4,   0,  -5,  -1,  -7, -12,  -8, -16,
          -6,  -6,   0,   2,  -9,  -9, -11,  -3,
          31,   2,   3,  -1,  -5, -13,   4, -20,
    };
    static const int D_QUEEN_MG[64] = {
         -28,   0,  29,  12,  59,  44,  43,  45,
         -24, -39,  -5,   1, -16,  57,  28,  54,
         -13, -17,   7,   8,  29,  56,  47,  57,
         -27, -27, -16, -16,  -1,  17,  -2,   1,
          -9, -26,  -9, -10,  -2,  -4,   3,  -3,
         -14,   2, -11,  -2,  -5,   2,  14,   5,
         -35,  -8,  11,   2,   8,  15,  -3,   1,
          -5, -18,  -9,  10, -15, -25, -31, -50,
    };
    static const int D_QUEEN_EG[64] = {
          -9,  22,  22,  27,  27,  19,  10,  20,
         -17,  20,  32,  41,  58,  25,  30,   0,
         -20,   6,   9,  49,  47,  35,  19,   9,
           3,  22,  24,  45,  57,  40,  57,  36,
         -18,  28,  19,  47,  31,  34,  39,  23,
         -16, -27,  15,   6,   9,  17,  10,   5,
         -22, -23, -30, -16, -16, -23, -36, -32,
         -44, -28, -22, -43,  -5, -32, -20, -41,
    };
    static const int D_KING_MG[64] = {
         -65,  23,  16, -15, -56, -34,   2,  13,
          29,  -1, -20,  -7,  -8,  -4, -38, -29,
          -9,  24,   2, -16, -20,   6,  22, -22,
         -17, -20, -12, -27, -30, -25, -14, -36,
         -49,  -1, -27, -39, -46, -44, -33, -51,
         -14, -14, -22, -46, -44, -30, -15, -27,
           1,   7,  -8, -64, -43, -16,   9,   8,
         -15,  36,  12, -54,   8, -28,  24,  14,
    };
    static const int D_KING_EG[64] = {
         -74, -35, -18, -18, -11,  15,   4, -17,
         -12,  17,  14,  17,  17,  38,  23,  11,
          10,  17,  23,  15,  20,  45,  44,  13,
          -8,  22,  24,  27,  26,  33,  26,   3,
         -18,  -4,  21,  24,  27,  23,   9, -11,
         -19,  -3,  11,  21,  23,  16,   7,  -9,
         -27, -11,   4,  13,  14,   4,  -5, -17,
         120, -34, -21, -11, -28, -14, -24, -43,
    };
 
    memcpy(w->pst_mg[PAWN],   D_PAWN_MG,   64*sizeof(int));
    memcpy(w->pst_mg[KNIGHT], D_KNIGHT_MG, 64*sizeof(int));
    memcpy(w->pst_mg[BISHOP], D_BISHOP_MG, 64*sizeof(int));
    memcpy(w->pst_mg[ROOK],   D_ROOK_MG,   64*sizeof(int));
    memcpy(w->pst_mg[QUEEN],  D_QUEEN_MG,  64*sizeof(int));
    memcpy(w->pst_mg[KING],   D_KING_MG,   64*sizeof(int));
    memcpy(w->pst_eg[PAWN],   D_PAWN_EG,   64*sizeof(int));
    memcpy(w->pst_eg[KNIGHT], D_KNIGHT_EG, 64*sizeof(int));
    memcpy(w->pst_eg[BISHOP], D_BISHOP_EG, 64*sizeof(int));
    memcpy(w->pst_eg[ROOK],   D_ROOK_EG,   64*sizeof(int));
    memcpy(w->pst_eg[QUEEN],  D_QUEEN_EG,  64*sizeof(int));
    memcpy(w->pst_eg[KING],   D_KING_EG,   64*sizeof(int));
 
    /* Pawn structure */
    w->doubled_pawn_mg  = -11; w->doubled_pawn_eg  = -56;
    w->isolated_pawn_mg = -15; w->isolated_pawn_eg = -15;
    w->backward_pawn_mg =  -9; w->backward_pawn_eg = -22;
    static const int D_PMG[8] = {  0,  5, 10,  20,  35,  60,  90,  0 };
    static const int D_PEG[8] = {  0, 10, 20,  40,  65,  95, 140,  0 };
    memcpy(w->passed_pawn_mg, D_PMG, sizeof w->passed_pawn_mg);
    memcpy(w->passed_pawn_eg, D_PEG, sizeof w->passed_pawn_eg);
    for (int i = 0; i < 8; i++) w->passed_pawn_king_dist_mult[i] = 2;
 
    /* Non-linear mobility (from eval.c MOB_*_TABLE) */
    static const int D_MOB_KN_MG[9]  = {-62,-53,-12, -4,  3, 13, 22, 28, 33};
    static const int D_MOB_KN_EG[9]  = {-81,-56,-30,-14,  8, 15, 23, 27, 33};
    static const int D_MOB_BI_MG[14] = {-48,-20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98};
    static const int D_MOB_BI_EG[14] = {-59,-23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97};
    static const int D_MOB_RK_MG[15] = {-58,-27,-15,-10, -5, -2,  9, 16, 30, 29, 32, 38, 46, 48, 58};
    static const int D_MOB_RK_EG[15] = {-76,-18, 28, 55, 69, 82,112,118,132,142,155,165,166,169,171};
    static const int D_MOB_QN_MG[28] = {
        -39,-21,  3,  3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66,
         67, 70, 71, 73, 79, 88, 88, 99,102,102,106,109,113,116};
    static const int D_MOB_QN_EG[28] = {
        -36,-15,  8, 18, 34, 54, 61, 73, 79, 92, 94,104,113,120,
        123,126,133,136,140,143,148,166,170,175,184,191,206,212};
    memcpy(w->mob_knight_mg, D_MOB_KN_MG, sizeof w->mob_knight_mg);
    memcpy(w->mob_knight_eg, D_MOB_KN_EG, sizeof w->mob_knight_eg);
    memcpy(w->mob_bishop_mg, D_MOB_BI_MG, sizeof w->mob_bishop_mg);
    memcpy(w->mob_bishop_eg, D_MOB_BI_EG, sizeof w->mob_bishop_eg);
    memcpy(w->mob_rook_mg,   D_MOB_RK_MG, sizeof w->mob_rook_mg);
    memcpy(w->mob_rook_eg,   D_MOB_RK_EG, sizeof w->mob_rook_eg);
    memcpy(w->mob_queen_mg,  D_MOB_QN_MG, sizeof w->mob_queen_mg);
    memcpy(w->mob_queen_eg,  D_MOB_QN_EG, sizeof w->mob_queen_eg);
 
    /* Rook bonuses */
    w->rook_open_mg    = 27; w->rook_open_eg    = 57;
    w->rook_semi_mg    = 12; w->rook_semi_eg    =  7;
    w->rook_seventh_mg = 20; w->rook_seventh_eg = 32;
    w->trapped_rook_mg = 52; w->trapped_rook_eg = 10;
 
    /* Bishop pair */
    w->bishop_pair_mg = 30; w->bishop_pair_eg = 60;
    w->bishop_pair_open_bonus = 0;
 
    /* Outposts */
    w->outpost_knight_mg = 22; w->outpost_knight_eg = 14;
    w->outpost_bishop_mg = 12; w->outpost_bishop_eg =  8;
 
    /* King safety */
    w->king_shield    =   7;
    w->king_open_file = -25;
    static const int D_KAW[6] = { 0, 20, 20, 40, 80, 0 };
    memcpy(w->king_attacker_weight, D_KAW, sizeof w->king_attacker_weight);
    w->king_danger_quadratic_scale = 200;
    w->king_pawn_storm_penalty     = -10;
    w->king_eg_distance_penalty    =   4;
 
    /* WeakQueen */
    w->weak_queen_mg = 49; w->weak_queen_eg = 15;
 
    /* KingProtector */
    w->king_protector_mg = 7; w->king_protector_eg = 8;
 
    /* MinorBehindPawn */
    w->minor_behind_pawn_mg = 18; w->minor_behind_pawn_eg = 3;
 
    /* Tactical threats */
    static const int D_HANG_MG[6] = {  40, 170, 175, 250, 500, 0 };
    static const int D_HANG_EG[6] = {  50, 165, 160, 270, 460, 0 };
    memcpy(w->hang_penalty_mg, D_HANG_MG, sizeof w->hang_penalty_mg);
    memcpy(w->hang_penalty_eg, D_HANG_EG, sizeof w->hang_penalty_eg);
    w->fork_base_mg  = 45; w->fork_base_eg  = 35;
    w->fork_extra_mg = 20; w->fork_extra_eg = 15;
    w->skewer_rq_mg  = 20; w->skewer_rq_eg  = 15;
    w->skewer_bq_mg  = 15; w->skewer_bq_eg  = 10;
 
    /* Tempo */
    w->tempo_mg = 16; w->tempo_eg = 0;
 
    /* Material imbalance */
    static const int D_QO[36] = {
        1438,   0,   0,   0,    0,   0,
          40,  38,   0,   0,    0,   0,
          32, 255, -62,   0,    0,   0,
           0, 104,   4,   0,    0,   0,
         -26,  -2,  47, 105, -208,   0,
        -189,  24, 117, 133, -134,  -6,
    };
    static const int D_QT[36] = {
          0,   0,   0,   0,   0,   0,
         36,   0,   0,   0,   0,   0,
          9,  63,   0,   0,   0,   0,
         59,  65,  42,   0,   0,   0,
         46,  39,  24, -24,   0,   0,
         97, 100, -42, 137, 268,   0,
    };
    memcpy(w->quad_ours,   D_QO, sizeof w->quad_ours);
    memcpy(w->quad_theirs, D_QT, sizeof w->quad_theirs);
}

/* ══════════════════════════════════════════════════════════════════════
 * save_weights — emits tuned weights as a drop-in C source file.
 *
 * Strategy:
 *   1. Write PST arrays with a tight column-aligned formatter (fast,
 *      readable output identical in shape to the DEF_PST_* tables above).
 *   2. Write every scalar weight via the t_g_params registry that
 *      register_all_params() already built — no field enumeration needed.
 *   3. Atomic rename: write to "<path>.tmp" then rename so a crash
 *      mid-save never leaves a half-written checkpoint.
 *
 * Returns 0 on success, -1 on I/O error.
 * ══════════════════════════════════════════════════════════════════════ */

/* Helper: emit one 8×8 int table as a C array literal. */
static int emit_pst(FILE *f, const char *name, const int sq[64])
{
    if (fprintf(f, "static const int %s[64] = {\n", name) < 0) return -1;
    for (int r = 0; r < 8; r++) {
        if (fputs("    ", f) == EOF) return -1;
        for (int c = 0; c < 8; c++) {
            if (fprintf(f, "%5d%s", sq[r*8+c],
                        (r == 7 && c == 7) ? "\n" : ",") < 0) return -1;
        }
    }
    return fputs("};\n", f) == EOF ? -1 : 0;
}

/* Helper: emit a small fixed-length int array on one line. */
static int emit_arr(FILE *f, const char *name, const int *v, int n)
{
    if (fprintf(f, "static const int %s[%d] = {", name, n) < 0) return -1;
    for (int i = 0; i < n; i++)
        if (fprintf(f, " %d%s", v[i], i < n-1 ? "," : "") < 0) return -1;
    return fputs(" };\n", f) == EOF ? -1 : 0;
}

static const char *PIECE_NAMES[6] = {
    "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"
};

int save_weights(const EvalWeights *w, const char *path)
{
    /* Build a temp path alongside the target so rename() is atomic. */
    char tmp[512];
    if ((size_t)snprintf(tmp, sizeof tmp, "%s.tmp", path) >= sizeof tmp) {
        fprintf(stderr, "[save_weights] path too long\n");
        return -1;
    }

    FILE *f = fopen(tmp, "w");
    if (!f) { perror("[save_weights] fopen"); return -1; }

#define CHECK(expr) do { if ((expr) < 0) goto io_err; } while(0)

    CHECK(fprintf(f,
        "/* Auto-generated by save_weights() — do not edit by hand. */\n\n"
        "#include \"tune.h\"\n\n"
        "void weights_load_tuned(EvalWeights *w) {\n\n"));

    /* ── Material ───────────────────────────────────────────────────── */
    CHECK(emit_arr(f, "mg", w->material_mg, 6));
    CHECK(emit_arr(f, "eg", w->material_eg, 6));
    CHECK(fputs(
        "    memcpy(w->material_mg, mg, sizeof w->material_mg);\n"
        "    memcpy(w->material_eg, eg, sizeof w->material_eg);\n\n", f) == EOF ? -1 : 0);

    /* ── PST tables ─────────────────────────────────────────────────── */
    static const char *phase[2] = { "mg", "eg" };
    const int (*pst[2])[64] = { w->pst_mg, w->pst_eg };

    for (int ph = 0; ph < 2; ph++) {
        for (int pt = 0; pt < 6; pt++) {
            char name[64];
            snprintf(name, sizeof name, "pst_%s_%s",
                     PIECE_NAMES[pt], phase[ph]);
            /* lowercase the piece name */
            for (char *p = name + 4; *p && *p != '_'; p++)
                if (*p >= 'A' && *p <= 'Z') *p |= 0x20;

            CHECK(emit_pst(f, name, pst[ph][pt]));
            CHECK(fprintf(f,
                "    memcpy(w->pst_%s[%s], %s, 64*sizeof(int));\n",
                phase[ph], PIECE_NAMES[pt], name));
        }
        CHECK(fputc('\n', f) == EOF ? -1 : 0);
    }

    /* ── Scalar fields via the param registry ───────────────────────── */
    /* register_all_params() was already called by optimize_dataset();
     * if save_weights() is ever called before tuning, call it now. */
    if (g_nparams == 0)
        register_all_params((EvalWeights *)w); /* cast: we won't modify *w */

    CHECK(fputs("    /* Scalar weights (from param registry) */\n", f) == EOF ? -1 : 0);
    for (int i = 0; i < g_nparams; i++) {
        /* Reconstruct the field offset to get the field name.
         * t_g_params[i].name holds the string passed to reg(). */
        CHECK(fprintf(f, "    /* %s */ *((int*)((char*)w + %td)) = %d;\n",
                      t_g_params[i].name,
                      (char *)t_g_params[i].ptr - (char *)w,
                      *t_g_params[i].ptr));
    }

    CHECK(fputs("\n}\n", f) == EOF ? -1 : 0);

#undef CHECK

    if (fflush(f) != 0 || fclose(f) != 0) {
        perror("[save_weights] fclose");
        remove(tmp);
        return -1;
    }

    if (rename(tmp, path) != 0) {
        perror("[save_weights] rename");
        remove(tmp);
        return -1;
    }

    printf("[save_weights] Wrote %d params to '%s'\n", g_nparams, path);
    return 0;

io_err:
    perror("[save_weights] write error");
    fclose(f);
    remove(tmp);
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════
 * Parameterised Evaluator (evaluate_w)
 * ══════════════════════════════════════════════════════════════════════ */
static const int PHASE_INC_W[6] = { 0, 1, 1, 2, 4, 0 };
#define MAX_PHASE_W 24

static inline int ew_chebyshev(int sq1, int sq2) {
    int df = (sq1 & 7) - (sq2 & 7); if (df < 0) df = -df;
    int dr = (sq1 >> 3) - (sq2 >> 3); if (dr < 0) dr = -dr;
    return (df > dr) ? df : dr;
}
 
static Bitboard ew_sq_attackers(const Board *b, int sq, Bitboard occ) {
    return (PAWN_ATTACKS[WHITE][sq]  & b->pieces[BLACK][PAWN])
         | (PAWN_ATTACKS[BLACK][sq]  & b->pieces[WHITE][PAWN])
         | (KNIGHT_ATTACKS[sq] & (b->pieces[WHITE][KNIGHT] | b->pieces[BLACK][KNIGHT]))
         | (bishop_attacks((Square)sq, occ)
                & (  b->pieces[WHITE][BISHOP] | b->pieces[BLACK][BISHOP]
                   | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (rook_attacks((Square)sq, occ)
                & (  b->pieces[WHITE][ROOK]  | b->pieces[BLACK][ROOK]
                   | b->pieces[WHITE][QUEEN] | b->pieces[BLACK][QUEEN]))
         | (KING_ATTACKS[sq] & (b->pieces[WHITE][KING] | b->pieces[BLACK][KING]));
}
 
int evaluate_w(const Board *b, const EvalWeights *w) {
    static const int PH_INC[6] = { 0, 1, 1, 2, 4, 0 };
    static const int PH_MAX    = 24;
 
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PH_INC[pt] * bb_popcount(b->pieces[c][pt]);
    if (phase > PH_MAX) phase = PH_MAX;
 
    int mg = 0, eg = 0;
 
    for (int c = 0; c < 2; c++) {
        int    sign = (c == WHITE) ? 1 : -1;
        int    c_mg = 0, c_eg = 0;
        Color  them = (Color)(c ^ 1);
        Bitboard occ = b->occ[2];
 
        /* Material + PST */
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = (c == WHITE) ? sq : (sq ^ 56);
                c_mg += w->material_mg[pt] + w->pst_mg[pt][psq];
                c_eg += w->material_eg[pt] + w->pst_eg[pt][psq];
            }
        }
 
        /* Non-linear mobility */
        {
            static const int MOB_MAX_W[6] = { 0, 8, 13, 14, 27, 0 };
            Bitboard not_us = ~b->occ[c];
            for (int pt = KNIGHT; pt <= QUEEN; pt++) {
                Bitboard pieces = b->pieces[c][pt];
                while (pieces) {
                    int sq = bb_pop(&pieces);
                    Bitboard atk = 0;
                    switch (pt) {
                        case KNIGHT: atk = KNIGHT_ATTACKS[sq];              break;
                        case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                        case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                        case QUEEN:  atk = queen_attacks ((Square)sq, occ); break;
                        default: break;
                    }
                    int mob = bb_popcount(atk & not_us);
                    if (mob > MOB_MAX_W[pt]) mob = MOB_MAX_W[pt];
                    switch (pt) {
                        case KNIGHT: c_mg+=w->mob_knight_mg[mob]; c_eg+=w->mob_knight_eg[mob]; break;
                        case BISHOP: c_mg+=w->mob_bishop_mg[mob]; c_eg+=w->mob_bishop_eg[mob]; break;
                        case ROOK:   c_mg+=w->mob_rook_mg[mob];   c_eg+=w->mob_rook_eg[mob];   break;
                        case QUEEN:  c_mg+=w->mob_queen_mg[mob];  c_eg+=w->mob_queen_eg[mob];  break;
                        default: break;
                    }
                }
            }
        }
 
        /* Pawn structure */
        {
            Bitboard our_pawns   = b->pieces[c][PAWN];
            Bitboard their_pawns = b->pieces[them][PAWN];
            Bitboard tmp = our_pawns;
            while (tmp) {
                int sq   = bb_pop(&tmp);
                int file = sq & 7;
                int rank = sq >> 3;
                Bitboard adj = 0;
                if (file > 0) adj |= FILE_BB[file-1];
                if (file < 7) adj |= FILE_BB[file+1];
 
                if (bb_popcount(our_pawns & FILE_BB[file]) > 1)
                    { c_mg += w->doubled_pawn_mg;  c_eg += w->doubled_pawn_eg; }
                if (!(our_pawns & adj))
                    { c_mg += w->isolated_pawn_mg; c_eg += w->isolated_pawn_eg; }
 
                Bitboard ahead = 0;
                if (c == WHITE) { for (int r=rank+1;r<8;r++) ahead|=RANK_BB[r]; }
                else            { for (int r=0;r<rank;r++)   ahead|=RANK_BB[r]; }
 
                bool is_passed = !((their_pawns & (FILE_BB[file]|adj)) & ahead);
                if (is_passed) {
                    int br = (c == WHITE) ? rank : (7-rank);
                    c_mg += w->passed_pawn_mg[br];
                    c_eg += w->passed_pawn_eg[br];
                } else {
                    int stop = (c == WHITE) ? sq+8 : sq-8;
                    if (stop >= 0 && stop < 64 &&
                        (PAWN_ATTACKS[c][stop] & their_pawns)) {
                        Bitboard behind = 0;
                        if (c==WHITE) { for(int r=0;r<rank;r++)   behind|=RANK_BB[r]; }
                        else          { for(int r=rank+1;r<8;r++) behind|=RANK_BB[r]; }
                        if (!(our_pawns & (behind & adj)))
                            { c_mg += w->backward_pawn_mg; c_eg += w->backward_pawn_eg; }
                    }
                }
            }
        }
 
        /* Rook bonuses + TrappedRook */
        {
            Bitboard our_pawns   = b->pieces[c][PAWN];
            Bitboard their_pawns = b->pieces[them][PAWN];
            Bitboard rooks = b->pieces[c][ROOK];
            int rank7 = (c==WHITE) ? 6 : 1;
            int rank8 = (c==WHITE) ? 7 : 0;
            while (rooks) {
                int sq   = bb_pop(&rooks);
                int file = sq & 7;
                int rank = sq >> 3;
                Bitboard fbb = FILE_BB[file];
                bool no_own   = !(our_pawns   & fbb);
                bool no_enemy = !(their_pawns & fbb);
                if (no_own && no_enemy)
                    { c_mg += w->rook_open_mg; c_eg += w->rook_open_eg; }
                else if (no_own)
                    { c_mg += w->rook_semi_mg; c_eg += w->rook_semi_eg; }
 
                if (!no_own && !no_enemy) {
                    Bitboard rmob = rook_attacks((Square)sq, occ) & ~b->occ[c];
                    if (bb_popcount(rmob) <= 3) {
                        int ksq = bb_lsb(b->pieces[c][KING]);
                        int kf  = ksq & 7;
                        if ((kf < 4) == (file < kf))
                            { c_mg -= w->trapped_rook_mg; c_eg -= w->trapped_rook_eg; }
                    }
                }
 
                if (rank == rank7) {
                    int eking = bb_lsb(b->pieces[them][KING]);
                    if (((eking>>3)==rank8) || (their_pawns & RANK_BB[rank7]))
                        { c_mg += w->rook_seventh_mg; c_eg += w->rook_seventh_eg; }
                }
            }
        }
 
        /* Outposts */
        {
            Bitboard our_pawns   = b->pieces[c][PAWN];
            Bitboard their_pawns = b->pieces[them][PAWN];
            int r_min = (c==WHITE) ? 3 : 2;
            int r_max = (c==WHITE) ? 5 : 4;
            Bitboard omask = 0;
            for (int r = r_min; r <= r_max; r++) omask |= RANK_BB[r];
            for (int pt = KNIGHT; pt <= BISHOP; pt++) {
                int bmg = (pt==KNIGHT) ? w->outpost_knight_mg : w->outpost_bishop_mg;
                int beg = (pt==KNIGHT) ? w->outpost_knight_eg : w->outpost_bishop_eg;
                Bitboard pieces = b->pieces[c][pt];
                while (pieces) {
                    int sq   = bb_pop(&pieces);
                    int file = sq & 7;
                    int rank = sq >> 3;
                    if (!(SQUARE_BB[sq] & omask)) continue;
                    if (!(PAWN_ATTACKS[them][sq] & our_pawns)) continue;
                    Bitboard adj = 0;
                    if (file>0) adj|=FILE_BB[file-1];
                    if (file<7) adj|=FILE_BB[file+1];
                    Bitboard ft = 0;
                    if (c==WHITE) { for(int r=rank+1;r<8;r++) ft|=RANK_BB[r]; }
                    else          { for(int r=0;r<rank;r++)   ft|=RANK_BB[r]; }
                    if (their_pawns & (ft & adj)) continue;
                    c_mg += bmg; c_eg += beg;
                }
            }
        }
 
        /* Bishop pair */
        if (bb_popcount(b->pieces[c][BISHOP]) >= 2)
            { c_mg += w->bishop_pair_mg; c_eg += w->bishop_pair_eg; }
 
        /* WeakQueen */
        {
            Bitboard queens = b->pieces[c][QUEEN];
            while (queens) {
                int sq = bb_pop(&queens);
                Bitboard occ_nq = occ ^ SQUARE_BB[sq];
                Bitboard rx = rook_attacks((Square)sq, occ_nq)
                            & (b->pieces[them][ROOK]   | b->pieces[them][QUEEN]);
                Bitboard bx = bishop_attacks((Square)sq, occ_nq)
                            & (b->pieces[them][BISHOP] | b->pieces[them][QUEEN]);
                if (rx|bx) { c_mg -= w->weak_queen_mg; c_eg -= w->weak_queen_eg; }
            }
        }
 
        /* KingProtector */
        {
            int ksq = bb_lsb(b->pieces[c][KING]);
            for (int pt = KNIGHT; pt <= BISHOP; pt++) {
                Bitboard pieces = b->pieces[c][pt];
                while (pieces) {
                    int sq   = bb_pop(&pieces);
                    int dist = ew_chebyshev(sq, ksq);
                    c_mg -= w->king_protector_mg * dist;
                    c_eg -= w->king_protector_eg * dist;
                }
            }
        }
 
        /* MinorBehindPawn */
        {
            Bitboard pawns  = b->pieces[c][PAWN];
            Bitboard behind = (c==WHITE) ? (pawns>>8) : (pawns<<8);
            Bitboard minors = b->pieces[c][KNIGHT] | b->pieces[c][BISHOP];
            int cnt = bb_popcount(minors & behind);
            c_mg += w->minor_behind_pawn_mg * cnt;
            c_eg += w->minor_behind_pawn_eg * cnt;
        }
 
        /* King safety */
        {
            int ksq = bb_lsb(b->pieces[c][KING]);
            int kf  = ksq & 7;
            int kr  = ksq >> 3;
            Bitboard our_pawns = b->pieces[c][PAWN];
 
            for (int f = kf-1; f <= kf+1; f++) {
                if (f<0||f>7) continue;
                Bitboard fp = our_pawns & FILE_BB[f];
                if (!fp) {
                    c_mg += w->king_open_file;
                } else {
                    int s1 = (c==WHITE) ? kr+1 : kr-1;
                    int s2 = (c==WHITE) ? kr+2 : kr-2;
                    if (s1>=0&&s1<8&&(fp&RANK_BB[s1])) c_mg += w->king_shield*2;
                    else if (s2>=0&&s2<8&&(fp&RANK_BB[s2])) c_mg += w->king_shield;
                }
            }
 
            Bitboard kzone = KING_ATTACKS[ksq] | SQUARE_BB[ksq];
            int acnt = 0, awt = 0;
            for (int pt = KNIGHT; pt < KING; pt++) {
                Bitboard pieces = b->pieces[them][pt];
                while (pieces) {
                    int sq = bb_pop(&pieces);
                    Bitboard atk = 0;
                    switch(pt) {
                        case KNIGHT: atk=KNIGHT_ATTACKS[sq];              break;
                        case BISHOP: atk=bishop_attacks((Square)sq,occ); break;
                        case ROOK:   atk=rook_attacks  ((Square)sq,occ); break;
                        case QUEEN:  atk=queen_attacks ((Square)sq,occ); break;
                        default: break;
                    }
                    if (atk & kzone) {
                        int dist  = ew_chebyshev(sq, ksq);
                        int scale = (dist<=1)?4:(dist<=3)?2:1;
                        acnt++;
                        awt += w->king_attacker_weight[pt] * scale;
                    }
                }
            }
            awt /= 2;
            if (acnt >= 2) c_mg -= awt * awt / 200;
 
            int eking = bb_lsb(b->pieces[them][KING]);
            c_eg -= ew_chebyshev(ksq, eking) * w->king_eg_distance_penalty;
        }
 
        /* Tactical threats */
        {
            /* Hanging / en-prise pieces */
            for (int pt = PAWN; pt < KING; pt++) {
                Bitboard pieces = b->pieces[c][pt];
                while (pieces) {
                    int sq = bb_pop(&pieces);
                    Bitboard eatk = ew_sq_attackers(b, sq, occ) & b->occ[them];
                    if (!eatk) continue;
                    Bitboard fdef = ew_sq_attackers(b, sq, occ) & b->occ[c];
                    if (!fdef) {
                        c_mg -= w->hang_penalty_mg[pt];
                        c_eg -= w->hang_penalty_eg[pt];
                    } else {
                        int min_atk = 30000;
                        for (int apt = PAWN; apt <= QUEEN; apt++)
                            if (b->pieces[them][apt] & eatk)
                                { min_atk = w->material_mg[apt]; break; }
                        int pv = w->material_mg[pt];
                        if (min_atk < pv)
                            { c_mg -= (pv-min_atk)/6; c_eg -= (pv-min_atk)/6; }
                    }
                }
            }
 
            /* Knight forks */
            {
                Bitboard venemy = b->pieces[them][BISHOP]
                                | b->pieces[them][ROOK]
                                | b->pieces[them][QUEEN]
                                | b->pieces[them][KING];
                Bitboard kts = b->pieces[c][KNIGHT];
                while (kts) {
                    int sq   = bb_pop(&kts);
                    int hits = bb_popcount(KNIGHT_ATTACKS[sq] & venemy);
                    if (hits >= 2) {
                        c_mg += w->fork_base_mg  + w->fork_extra_mg * (hits-2);
                        c_eg += w->fork_base_eg  + w->fork_extra_eg * (hits-2);
                    }
                }
            }
 
            /* Skewers / batteries */
            {
                Bitboard royal = b->pieces[them][QUEEN] | b->pieces[them][KING];
                Bitboard val_r = royal | b->pieces[them][ROOK];
 
                Bitboard rq = b->pieces[c][ROOK] | b->pieces[c][QUEEN];
                while (rq) {
                    int sq = bb_pop(&rq);
                    if (bb_popcount(rook_attacks((Square)sq,occ) & val_r) >= 2)
                        { c_mg += w->skewer_rq_mg; c_eg += w->skewer_rq_eg; }
                }
 
                Bitboard bq = b->pieces[c][BISHOP] | b->pieces[c][QUEEN];
                while (bq) {
                    int sq = bb_pop(&bq);
                    if (bb_popcount(bishop_attacks((Square)sq,occ) & royal) >= 2)
                        { c_mg += w->skewer_bq_mg; c_eg += w->skewer_bq_eg; }
                }
            }
        }
 
        mg += sign * c_mg;
        eg += sign * c_eg;
    }
 
    int score = (mg * phase + eg * (PH_MAX - phase)) / PH_MAX;
    score += (w->tempo_mg * phase + w->tempo_eg * (PH_MAX - phase)) / PH_MAX;
    return (b->side == WHITE) ? score : -score;
}

/* ══════════════════════════════════════════════════════════════════════
 * Self-Play Minimal Searcher
 * ══════════════════════════════════════════════════════════════════════ */
#define INF 30000

static int qsearch_w(Board *b, int alpha, int beta, const EvalWeights *w, int depth) {
    int stand_pat = evaluate_w(b, w);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    if (depth <= 0) return alpha;  /* hard cap — no more captures */

    MoveList ml;
    gen_captures(b, &ml);
    /* In a real scenario, implement basic MVV-LVA ordering here */

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (!is_legal(b, m)) continue;

        make_move(b, m);
        int score = -qsearch_w(b, -beta, -alpha, w, depth - 1);
        unmake_move(b);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

static int alphabeta_w(Board *b, int depth, int alpha, int beta, const EvalWeights *w, Move *best_move) {
    if (depth == 0) return qsearch_w(b, alpha, beta, w, 8); /* cap captures at 8 plies */

    MoveList ml;
    gen_moves(b, &ml);
    int legal_count = 0;
    int best_score = -INF;
    Move local_best = NULL_MOVE;

    /* Simple move ordering (Capture first) */
    for(int i=0; i<ml.count; i++) {
        if (MOVE_IS_CAP(ml.moves[i])) {
            Move temp = ml.moves[0];
            ml.moves[0] = ml.moves[i];
            ml.moves[i] = temp;
        }
    }

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (!is_legal(b, m)) continue;
        legal_count++;

        make_move(b, m);
        int score = -alphabeta_w(b, depth - 1, -beta, -alpha, w, NULL);
        unmake_move(b);

        if (score > best_score) {
            best_score = score;
            local_best = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; /* Cutoff */
    }

    if (legal_count == 0) return in_check(b) ? -INF : 0;
    if (best_move) *best_move = local_best;
    return best_score;
}

/* ══════════════════════════════════════════════════════════════════════
 * Dataset Architecture (Distributed-Ready)
 * ══════════════════════════════════════════════════════════════════════ */
#define MAX_POSITIONS_PER_BATCH 500000

typedef struct {
    Board board;
    double result; /* 1.0 (W), 0.5 (D), 0.0 (B) */
} TunePos;

typedef struct {
    TunePos *positions;
    int count;
} PosDataset;

PosDataset* create_dataset() {
    PosDataset* ds = malloc(sizeof(PosDataset));
    ds->positions = malloc(MAX_POSITIONS_PER_BATCH * sizeof(TunePos));
    ds->count = 0;
    return ds;
}

void free_dataset(PosDataset* ds) {
    free(ds->positions);
    free(ds);
}

/* ══════════════════════════════════════════════════════════════════════
 * Game Generator (Self-Play Worker)
 * ══════════════════════════════════════════════════════════════════════ */

/* Plays one game and appends positions to the dataset.
 * seed is a per-thread tune_rand() state — never share the same pointer
 * between threads. */
static void play_game(const EvalWeights *w, PosDataset *ds, int search_depth,
                      unsigned int *seed) {
    Board b;
    board_start_pos(&b);
    
    Board *game_history = malloc(500 * sizeof(Board));
    if (!game_history) return;
    int ply = 0;

    /* Randomize first 4 plies to ensure opening variety (Temperature) */
    for (int i = 0; i < 4; i++) {
        MoveList ml;
        gen_moves(&b, &ml);
        int legal_moves[256];
        int l_count = 0;
        for (int j = 0; j < ml.count; j++) {
            if (is_legal(&b, ml.moves[j])) legal_moves[l_count++] = j;
        }
        if (l_count == 0) { free(game_history); return; }
        make_move(&b, ml.moves[legal_moves[tune_rand(seed) % l_count]]);
    }

    /* Self-Play loop */
    double game_result = 0.5; /* Default draw */
    while (ply < 400) {
        Move best_move = NULL_MOVE;
        int score = alphabeta_w(&b, search_depth, -INF, INF, w, &best_move);

        if (best_move == NULL_MOVE) {
            game_result = in_check(&b) ? ((b.side == WHITE) ? 0.0 : 1.0) : 0.5;
            break;
        }

        /* Detect easy draw */
        if (b.halfmove >= 100) { game_result = 0.5; break; }

        /* Save position */
        game_history[ply++] = b;
        make_move(&b, best_move);

        /* Adjudicate obvious wins/losses to save time */
        if (score > 1000) { game_result = (b.side == WHITE) ? 0.0 : 1.0; break; }
        if (score < -1000) { game_result = (b.side == WHITE) ? 1.0 : 0.0; break; }
    }

    /* Append to dataset */
    for (int i = 0; i < ply; i++) {
        if (ds->count >= MAX_POSITIONS_PER_BATCH) break;
        ds->positions[ds->count].board = game_history[i];
        ds->positions[ds->count].result = game_result;
        ds->count++;
    }
    free(game_history);
}

/* ── Parallel self-play ──────────────────────────────────────────────── */
typedef struct {
    const EvalWeights *w;
    int                num_games;
    int                depth;
    unsigned int       seed;
    PosDataset        *out_ds; /* private per-thread, allocated by thread */
} SelfPlayCtx;

static void *self_play_thread_fn(void *arg) {
    SelfPlayCtx *ctx = (SelfPlayCtx *)arg;
    ctx->out_ds = create_dataset();
    for (int i = 0; i < ctx->num_games; i++)
        play_game(ctx->w, ctx->out_ds, ctx->depth, &ctx->seed);
    return NULL;
}

/* Worker Entry Point: Generates N games using current weights, distributed
 * across g_num_threads worker threads.  Results are merged into *ds. */
void self_play_worker(const EvalWeights *w, int num_games, int depth, PosDataset *ds) {
    int nthreads = g_num_threads;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > num_games) nthreads = num_games;

    printf("[Worker] Starting self-play for %d games at depth %d across %d thread(s)...\n",
           num_games, depth, nthreads);

    /* Single-thread fast path */
    if (nthreads == 1) {
        unsigned int seed = (unsigned int)time(NULL);
        for (int i = 0; i < num_games; i++) {
            play_game(w, ds, depth, &seed);
            if ((i + 1) % 50 == 0) printf("  Played %d games...\n", i + 1);
        }
        printf("[Worker] Generated %d positions.\n", ds->count);
        return;
    }

    pthread_t   threads[MAX_TUNE_THREADS];
    SelfPlayCtx ctxs[MAX_TUNE_THREADS];
    int base  = num_games / nthreads;
    int extra = num_games % nthreads;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 64 * 1024 * 1024); /* 64 MB — qsearch can recurse deep */

    for (int t = 0; t < nthreads; t++) {
        ctxs[t].w         = w;
        ctxs[t].num_games = base + (t < extra ? 1 : 0);
        ctxs[t].depth     = depth;
        ctxs[t].seed      = (unsigned int)(time(NULL) ^ (unsigned int)(t * 2654435761u));
        ctxs[t].out_ds    = NULL;
        pthread_create(&threads[t], &attr, self_play_thread_fn, &ctxs[t]);
    }
    pthread_attr_destroy(&attr);

    /* Join threads and merge their private datasets into *ds */
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        PosDataset *src = ctxs[t].out_ds;
        if (!src) continue;
        int to_copy = src->count;
        if (ds->count + to_copy > MAX_POSITIONS_PER_BATCH)
            to_copy = MAX_POSITIONS_PER_BATCH - ds->count;
        if (to_copy > 0) {
            memcpy(ds->positions + ds->count, src->positions,
                   (size_t)to_copy * sizeof(TunePos));
            ds->count += to_copy;
        }
        free_dataset(src);
    }

    printf("[Worker] Generated %d positions from %d threads.\n", ds->count, nthreads);
}

/* ══════════════════════════════════════════════════════════════════════
 * Optimizer (Coordinate Descent)
 * ══════════════════════════════════════════════════════════════════════ */

static double g_K = 1.13 / 400.0;

static inline double sigmoid(double cp) {
    return 1.0 / (1.0 + exp(-g_K * cp));
}

/* ── Parallel error computation ─────────────────────────────────────── */
typedef struct {
    const EvalWeights *w;
    const TunePos     *positions;
    int                start;
    int                end;
    double             partial_err;
} ErrWorkerCtx;

static void *err_worker_fn(void *arg) {
    ErrWorkerCtx *ctx = (ErrWorkerCtx *)arg;
    double err = 0.0;
    for (int i = ctx->start; i < ctx->end; i++) {
        double s = sigmoid((double)evaluate_w(&ctx->positions[i].board, ctx->w));
        double d = s - ctx->positions[i].result;
        err += d * d;
    }
    ctx->partial_err = err;
    return NULL;
}

static double compute_error(const EvalWeights *w, PosDataset *ds) {
    if (ds->count == 0) return 0.0;

    int nthreads = g_num_threads;
    /* Fall back to single-thread if dataset is tiny or only 1 thread requested */
    if (nthreads <= 1 || ds->count < nthreads) {
        double err = 0.0;
        for (int i = 0; i < ds->count; i++) {
            double s = sigmoid((double)evaluate_w(&ds->positions[i].board, w));
            double d = s - ds->positions[i].result;
            err += d * d;
        }
        return err / (double)ds->count;
    }

    pthread_t      threads[MAX_TUNE_THREADS];
    ErrWorkerCtx   ctxs[MAX_TUNE_THREADS];
    int slice = ds->count / nthreads;

    for (int t = 0; t < nthreads; t++) {
        ctxs[t].w            = w;
        ctxs[t].positions    = ds->positions;
        ctxs[t].start        = t * slice;
        ctxs[t].end          = (t == nthreads - 1) ? ds->count : (t + 1) * slice;
        ctxs[t].partial_err  = 0.0;
        pthread_create(&threads[t], NULL, err_worker_fn, &ctxs[t]);
    }

    double total = 0.0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        total += ctxs[t].partial_err;
    }
    return total / (double)ds->count;
}

static void reg(int *ptr, int lo, int hi, const char *name) {
    if (g_nparams >= MAX_PARAMS) return;
    t_g_params[g_nparams++] = (Param){ ptr, lo, hi, name };
}

static void register_all_params(EvalWeights *w) {
    g_nparams = 0;
 
    /* Material */
    for (int pt = 0; pt < 6; pt++) {
        reg(&w->material_mg[pt],   0, 1200, "material_mg");
        reg(&w->material_eg[pt],   0, 1200, "material_eg");
    }
 
    /* PST */
    for (int pt = 0; pt < 6; pt++) {
        int lo = (pt == KING) ? -120 : -200;
        int hi = (pt == KING) ?  120 :  200;
        for (int sq = 0; sq < 64; sq++) {
            reg(&w->pst_mg[pt][sq], lo, hi, "pst_mg");
            reg(&w->pst_eg[pt][sq], lo, hi, "pst_eg");
        }
    }
 
    /* Pawn structure */
    reg(&w->doubled_pawn_mg,  -100,  0, "doubled_pawn_mg");
    reg(&w->doubled_pawn_eg,  -100,  0, "doubled_pawn_eg");
    reg(&w->isolated_pawn_mg, -100,  0, "isolated_pawn_mg");
    reg(&w->isolated_pawn_eg, -100,  0, "isolated_pawn_eg");
    reg(&w->backward_pawn_mg, -100,  0, "backward_pawn_mg");
    reg(&w->backward_pawn_eg, -100,  0, "backward_pawn_eg");
    for (int i = 1; i < 7; i++) {
        reg(&w->passed_pawn_mg[i],   0, 250, "passed_pawn_mg");
        reg(&w->passed_pawn_eg[i],   0, 350, "passed_pawn_eg");
    }
 
    /* Non-linear mobility */
    for (int i = 0; i < 9;  i++) {
        reg(&w->mob_knight_mg[i], -100,  50, "mob_knight_mg");
        reg(&w->mob_knight_eg[i], -100,  50, "mob_knight_eg");
    }
    for (int i = 0; i < 14; i++) {
        reg(&w->mob_bishop_mg[i], -100, 120, "mob_bishop_mg");
        reg(&w->mob_bishop_eg[i], -100, 120, "mob_bishop_eg");
    }
    for (int i = 0; i < 15; i++) {
        reg(&w->mob_rook_mg[i],   -100, 200, "mob_rook_mg");
        reg(&w->mob_rook_eg[i],   -100, 200, "mob_rook_eg");
    }
    for (int i = 0; i < 28; i++) {
        reg(&w->mob_queen_mg[i],   -50, 130, "mob_queen_mg");
        reg(&w->mob_queen_eg[i],   -50, 220, "mob_queen_eg");
    }
 
    /* Rook bonuses */
    reg(&w->rook_open_mg,      0, 100, "rook_open_mg");
    reg(&w->rook_open_eg,      0, 100, "rook_open_eg");
    reg(&w->rook_semi_mg,      0,  60, "rook_semi_mg");
    reg(&w->rook_semi_eg,      0,  60, "rook_semi_eg");
    reg(&w->rook_seventh_mg,   0,  80, "rook_seventh_mg");
    reg(&w->rook_seventh_eg,   0,  80, "rook_seventh_eg");
    reg(&w->trapped_rook_mg,   0, 120, "trapped_rook_mg");
    reg(&w->trapped_rook_eg,   0,  60, "trapped_rook_eg");
 
    /* Bishop pair */
    reg(&w->bishop_pair_mg,    0, 100, "bishop_pair_mg");
    reg(&w->bishop_pair_eg,    0, 100, "bishop_pair_eg");
 
    /* Outposts */
    reg(&w->outpost_knight_mg, 0,  60, "outpost_knight_mg");
    reg(&w->outpost_knight_eg, 0,  60, "outpost_knight_eg");
    reg(&w->outpost_bishop_mg, 0,  40, "outpost_bishop_mg");
    reg(&w->outpost_bishop_eg, 0,  40, "outpost_bishop_eg");
 
    /* King safety */
    reg(&w->king_shield,            0,  40, "king_shield");
    reg(&w->king_open_file,      -100,   0, "king_open_file");
    for (int pt = KNIGHT; pt <= QUEEN; pt++)
        reg(&w->king_attacker_weight[pt], 0, 200, "king_attacker_weight");
    reg(&w->king_eg_distance_penalty,    0,  20, "king_eg_distance_penalty");
 
    /* WeakQueen */
    reg(&w->weak_queen_mg,         0, 100, "weak_queen_mg");
    reg(&w->weak_queen_eg,         0,  60, "weak_queen_eg");
 
    /* KingProtector */
    reg(&w->king_protector_mg,     0,  30, "king_protector_mg");
    reg(&w->king_protector_eg,     0,  30, "king_protector_eg");
 
    /* MinorBehindPawn */
    reg(&w->minor_behind_pawn_mg,  0,  40, "minor_behind_pawn_mg");
    reg(&w->minor_behind_pawn_eg,  0,  20, "minor_behind_pawn_eg");
 
    /* Tactical threats */
    for (int pt = PAWN; pt < KING; pt++) {
        reg(&w->hang_penalty_mg[pt],   0, 700, "hang_penalty_mg");
        reg(&w->hang_penalty_eg[pt],   0, 700, "hang_penalty_eg");
    }
    reg(&w->fork_base_mg,          0, 100, "fork_base_mg");
    reg(&w->fork_base_eg,          0, 100, "fork_base_eg");
    reg(&w->fork_extra_mg,         0,  60, "fork_extra_mg");
    reg(&w->fork_extra_eg,         0,  60, "fork_extra_eg");
    reg(&w->skewer_rq_mg,          0,  60, "skewer_rq_mg");
    reg(&w->skewer_rq_eg,          0,  60, "skewer_rq_eg");
    reg(&w->skewer_bq_mg,          0,  50, "skewer_bq_mg");
    reg(&w->skewer_bq_eg,          0,  50, "skewer_bq_eg");
 
    /* Tempo */
    reg(&w->tempo_mg,              0,  40, "tempo_mg");
    reg(&w->tempo_eg,              0,  20, "tempo_eg");
 
    /* Material imbalance -- tight bounds, these are structural */
    for (int i = 0; i < 36; i++) {
        reg(&w->quad_ours[i],   -400, 1600, "quad_ours");
        reg(&w->quad_theirs[i], -100,  400, "quad_theirs");
    }
}
 
void optimize_dataset(EvalWeights *w, PosDataset *ds, int max_iters) {
    register_all_params(w);
    double best_err = compute_error(w, ds);
    printf("[Optimizer] Starting Error: %.8f over %d parameters\n",
           best_err, g_nparams);
 
    for (int iter = 0; iter < max_iters; iter++) {
        int improved = 0;
 
        for (int i = 0; i < g_nparams; i++) {
            Param *p = &t_g_params[i];
 
            /* Try +1, clamped to hi */
            if (*p->ptr < p->hi) {
                (*p->ptr)++;
                double e = compute_error(w, ds);
                if (e < best_err) { best_err = e; improved++; continue; }
                (*p->ptr)--;
            }
 
            /* Try -1, clamped to lo */
            if (*p->ptr > p->lo) {
                (*p->ptr)--;
                double e = compute_error(w, ds);
                if (e < best_err) { best_err = e; improved++; continue; }
                (*p->ptr)++;
            }
        }
 
        printf("[Optimizer] Iter %d: MSE=%.8f (Improved %d params)\n",
               iter + 1, best_err, improved);
        if (improved == 0) break;
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Main Reinforcement Learning Loop
 * ══════════════════════════════════════════════════════════════════════ */

void tune_self_play_loop(int iterations, int games_per_iter, int search_depth, int num_threads) {
    srand((unsigned)time(NULL));

    /* Clamp and publish thread count for compute_error + self_play_worker */
    if (num_threads < 1)               num_threads = 1;
    if (num_threads > MAX_TUNE_THREADS) num_threads = MAX_TUNE_THREADS;
    g_num_threads = num_threads;

    printf("[Tuner] Using %d thread(s).\n", g_num_threads);

    EvalWeights w;
    weights_init_defaults(&w);

    for (int iter = 1; iter <= iterations; iter++) {
        printf("\n======================================================\n");
        printf(" RL Iteration %d / %d\n", iter, iterations);
        printf("======================================================\n");

        PosDataset *ds = create_dataset();

        /* Step 1: Self-Play (Data Generation Phase) 
         * In a distributed setup, you would dispatch a network request here */
        self_play_worker(&w, games_per_iter, search_depth, ds);

        /* Step 2: Optimization Phase 
         * In a distributed setup, workers would upload .epd files, and this 
         * process would aggregate them before calling optimize_dataset */
        optimize_dataset(&w, ds, 50); /* Max 50 coordinate descent sweeps per RL iteration */

        free_dataset(ds);
        
        /* Save progress checkpoint */
        printf("[Master] Checkpointing weights...\n");
        save_weights(&w, "tuned_weights_checkpoint.c");
    }
}

#ifdef TUNE_STANDALONE
int main(int argc, char **argv) {
    bitboard_init();
    board_init();

    int rl_iters      = 50;   /* optimiser sweeps per RL iteration             */
    int games_per_iter = 100; /* scale up heavily for distributed runs         */
    int depth          = 3;   /* keep low for throughput; rely on eval quality */
    int num_threads    = 12;  /* default: 12 threads                           */

    /* Optional positional args: [num_threads] [games_per_iter] [rl_iters] [depth] */
    if (argc > 1) num_threads    = atoi(argv[1]);
    if (argc > 2) games_per_iter = atoi(argv[2]);
    if (argc > 3) rl_iters       = atoi(argv[3]);
    if (argc > 4) depth          = atoi(argv[4]);

    printf("Starting Distributed-Ready Reinforcement Tuning System\n");
    printf("  rl_iters=%d  games_per_iter=%d  depth=%d  threads=%d\n",
           rl_iters, games_per_iter, depth, num_threads);

    tune_self_play_loop(rl_iters, games_per_iter, depth, num_threads);

    return 0;
}
#endif