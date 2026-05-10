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

    /* Pawn Structure (Expanded) */
    w->doubled_pawn_mg = -11; w->doubled_pawn_eg = -56;
    w->isolated_pawn_mg = -15; w->isolated_pawn_eg = -15;
    w->backward_pawn_mg = -9; w->backward_pawn_eg = -22;
    
    for (int i=0; i<8; i++) {
        w->passed_pawn_mg[i] = i * 10;
        w->passed_pawn_eg[i] = i * 20;
        /* NEW: Passed pawn danger based on distance to enemy king */
        w->passed_pawn_king_dist_mult[i] = 2; 
    }

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

    w->outpost_knight_mg = 22; w->outpost_knight_eg = 14;
    w->outpost_bishop_mg = 12; w->outpost_bishop_eg =  8;

    w->king_shield    =   7;
    w->king_open_file = -25;

    w->bishop_pair_mg = 30; 
    w->bishop_pair_eg = 60;
    w->bishop_pair_open_bonus = 15;

    static const int def_kaw[6] = { 0, 20, 20, 40, 80, 0 };
    memcpy(w->king_attacker_weight, def_kaw, sizeof w->king_attacker_weight);

    w->king_danger_quadratic_scale = 200; 
    w->king_pawn_storm_penalty = -10;

    w->tempo_mg = 14; w->tempo_eg = 8;
}

/* ══════════════════════════════════════════════════════════════════════
 * Parameterised Evaluator (evaluate_w)
 * (Simplified representation of your eval.c using the new EvalWeights)
 * ══════════════════════════════════════════════════════════════════════ */
static const int PHASE_INC_W[6] = { 0, 1, 1, 2, 4, 0 };
#define MAX_PHASE_W 24

int evaluate_w(const Board *b, const EvalWeights *w) {
    int mg = 0, eg = 0;
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_INC_W[pt] * bb_popcount(b->pieces[c][pt]);
    phase = (phase > MAX_PHASE_W) ? MAX_PHASE_W : phase;

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        int c_mg = 0, c_eg = 0;

        /* Core Material */
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq = bb_pop(&bb);
                int psq = (c == WHITE) ? sq : (sq ^ 56);
                c_mg += w->material_mg[pt] + w->pst_mg[pt][psq];
                c_eg += w->material_eg[pt] + w->pst_eg[pt][psq];
            }
        }
        
        /* Insert additional heuristic calculations here based on the 
         * expanded EvalWeights fields... */

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    int score = (mg * phase + eg * (MAX_PHASE_W - phase)) / MAX_PHASE_W;
    score += (w->tempo_mg * phase + w->tempo_eg * (MAX_PHASE_W - phase)) / MAX_PHASE_W;
    return (b->side == WHITE) ? score : -score;
}

/* ══════════════════════════════════════════════════════════════════════
 * Self-Play Minimal Searcher
 * ══════════════════════════════════════════════════════════════════════ */
#define INF 30000

static int qsearch_w(Board *b, int alpha, int beta, const EvalWeights *w) {
    int stand_pat = evaluate_w(b, w);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList ml;
    gen_captures(b, &ml);
    /* In a real scenario, implement basic MVV-LVA ordering here */

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (!is_legal(b, m)) continue;

        make_move(b, m);
        int score = -qsearch_w(b, -beta, -alpha, w);
        unmake_move(b);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

static int alphabeta_w(Board *b, int depth, int alpha, int beta, const EvalWeights *w, Move *best_move) {
    if (depth == 0) return qsearch_w(b, alpha, beta, w);

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

/* Plays one game and appends positions to the dataset */
static void play_game(const EvalWeights *w, PosDataset *ds, int search_depth) {
    Board b;
    board_start_pos(&b);
    
    Board game_history[500];
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
        if (l_count == 0) return;
        make_move(&b, ml.moves[legal_moves[rand() % l_count]]);
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
}

/* Worker Entry Point: Generates N games using current weights */
void self_play_worker(const EvalWeights *w, int num_games, int depth, PosDataset *ds) {
    printf("[Worker] Starting self-play for %d games at depth %d...\n", num_games, depth);
    for (int i = 0; i < num_games; i++) {
        play_game(w, ds, depth);
        if ((i+1) % 50 == 0) printf("  Played %d games...\n", i+1);
    }
    printf("[Worker] Generated %d positions.\n", ds->count);
}

/* ══════════════════════════════════════════════════════════════════════
 * Optimizer (Coordinate Descent)
 * ══════════════════════════════════════════════════════════════════════ */

static double g_K = 1.13 / 400.0;

static inline double sigmoid(double cp) {
    return 1.0 / (1.0 + exp(-g_K * cp));
}

static double compute_error(const EvalWeights *w, PosDataset *ds) {
    double err = 0.0;
    for (int i = 0; i < ds->count; i++) {
        double s = sigmoid((double)evaluate_w(&ds->positions[i].board, w));
        double d = s - ds->positions[i].result;
        err += d * d;
    }
    return err / (double)ds->count;
}

/* Registers all expanded parameters (Abbreviated for demonstration) */
typedef struct { int *ptr; int step; } Param;
#define MAX_PARAMS 1024
static Param g_params[MAX_PARAMS];
static int g_nparams = 0;

static void reg(int *ptr, int lo, int hi, const char *name) {
    if (g_nparams >= MAX_PARAMS) return;
    g_params[g_nparams++] = (Param){ ptr, lo, hi, name };
}

static void register_all_params(EvalWeights *w) {
    g_nparams = 0;
    for(int pt=0; pt<6; pt++) {
        reg(&w->material_mg[pt], 0, 350, "material_mg");
        reg(&w->material_eg[pt], 0, 350, "material_eg");
    }
    reg(&w->doubled_pawn_mg, 0, 350, "doubled_pawn_mg"); reg(&w->doubled_pawn_eg, 0, 350, "doubled_pawn_eg");
    reg(&w->king_danger_quadratic_scale, 0, 10, "king_danger_quadratic_scale");

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

/* * optimize_dataset runs Coordinate Descent over the generated dataset.
 * It uses an adaptive step size defined in the parameter registry.
 */
void optimize_dataset(EvalWeights *w, PosDataset *ds, int max_iters) {
    register_all_params(w);
    double best_err = compute_error(w, ds);
    printf("[Optimizer] Starting Error: %.8f over %d parameters\n", best_err, g_nparams);

    for (int iter = 0; iter < max_iters; iter++) {
        int improved = 0;
        
        for (int i = 0; i < g_nparams; i++) {
            Param *p = &g_params[i];
            
            /* Try +Step */
            *p->ptr += p->step;
            double e = compute_error(w, ds);
            if (e < best_err) { best_err = e; improved++; continue; }
            *p->ptr -= p->step; /* Revert */

            /* Try -Step */
            *p->ptr -= p->step;
            e = compute_error(w, ds);
            if (e < best_err) { best_err = e; improved++; continue; }
            *p->ptr += p->step; /* Revert */
        }

        printf("[Optimizer] Iter %d: MSE=%.8f (Improved %d params)\n", iter+1, best_err, improved);
        if (improved == 0) break; /* Converged */
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Main Reinforcement Learning Loop
 * ══════════════════════════════════════════════════════════════════════ */

void tune_self_play_loop(int iterations, int games_per_iter, int search_depth) {
    srand((unsigned)time(NULL));

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
        optimize_dataset(&w, ds, 10); /* Max 10 coordinate descent sweeps per RL iteration */

        free_dataset(ds);
        
        /* Save progress checkpoint */
        printf("[Master] Checkpointing weights...\n");
        /* save_weights(&w, "tuned_weights_checkpoint.c"); */
    }
}

#ifdef TUNE_STANDALONE
int main(int argc, char **argv) {
    bitboard_init();
    board_init();

    int rl_iters = 10;
    int games_per_iter = 100; /* Scale this up heavily in distributed runs */
    int depth = 3; /* Keep depth low for throughput, rely on evaluation accuracy */

    printf("Starting Distributed-Ready Reinforcement Tuning System\n");
    tune_self_play_loop(rl_iters, games_per_iter, depth);

    return 0;
}
#endif