#pragma once
/*
 * tune.h — Texel-style multithreaded parameter tuner for SalmonEngine.
 *
 * Algorithm overview
 * ──────────────────
 * 1. Load a corpus of quiet positions with known game outcomes (EPD / annotated FEN).
 * 2. For each parameter in a random permutation, try ±1 and keep whichever direction
 *    reduces the mean-squared error (MSE) between sigmoid(eval) and the true result.
 * 3. Repeat until a full sweep produces no improvement, or max_iters is reached.
 * 4. Emit the tuned parameters as ready-to-paste C source code.
 *
 * MSE cost function
 * ─────────────────
 *   E = (1/N) Σ  ( sigmoid(K · cp_eval(pos)) − result )²
 *
 *   K      : scaling constant (default 1.13/400 ≈ 0.002825, matches centipawn scale)
 *   result : 1.0 = White win, 0.5 = draw, 0.0 = Black win
 *
 * Threading model
 * ───────────────
 * For each parameter perturbation the position corpus is partitioned into
 * num_threads contiguous slices.  Each slice is processed by a worker thread
 * that accumulates a partial MSE.  The main thread joins all workers, sums
 * the partial errors, and decides whether to keep the perturbation.
 *
 * Compilation
 * ───────────
 *   gcc -O3 -march=native -pthread -lm \
 *       tune.c eval.c bitboard.c board.c search.c -o tuner
 *
 * Usage
 * ─────
 *   tuner <epd_file> [num_threads] [max_iters] [out_file]
 *
 * EPD file format (any of these per line)
 * ────────────────────────────────────────
 *   <FEN> [1.0]                   (float result, 0.0 / 0.5 / 1.0)
 *   <FEN> [1/2-1/2]               (result string in brackets)
 *   <FEN> c9 "1-0";               (standard EPD opcode)
 *   Lines beginning with '#' are comments and are skipped.
 *
 * Assumptions
 * ───────────
 *   • board.h exposes: board_from_fen(Board *, const char *) → 0 on success.
 *   • bitboard.h exposes all attack tables and bb_* functions used by eval.c.
 *   • board_init() / bitboard_init() are called before tune_run().
 */

#include "board.h"

/* ══════════════════════════════════════════════════════════════════════
 *  EvalWeights — every tunable constant from eval.c in one flat struct.
 *
 *  Index convention follows the PieceType enum from board.h:
 *    PAWN=0  KNIGHT=1  BISHOP=2  ROOK=3  QUEEN=4  KING=5
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Material */
    int material_mg[6];
    int material_eg[6];
 
    /* Piece-square tables */
    int pst_mg[6][64];
    int pst_eg[6][64];
 
    /* Pawn structure */
    int doubled_pawn_mg,  doubled_pawn_eg;
    int isolated_pawn_mg, isolated_pawn_eg;
    int backward_pawn_mg, backward_pawn_eg;
    int passed_pawn_mg[8];
    int passed_pawn_eg[8];
    int passed_pawn_king_dist_mult[8]; /* kept for compat; unused by eval.c */
 
    /* Non-linear mobility tables (indexed by square count, capped at MOB_MAX).
     * Knight: 9 entries (0-8), Bishop: 14 (0-13),
     * Rook:  15 (0-14),        Queen:  28 (0-27). */
    int mob_knight_mg[9],  mob_knight_eg[9];
    int mob_bishop_mg[14], mob_bishop_eg[14];
    int mob_rook_mg[15],   mob_rook_eg[15];
    int mob_queen_mg[28],  mob_queen_eg[28];
 
    /* Rook bonuses */
    int rook_open_mg,    rook_open_eg;
    int rook_semi_mg,    rook_semi_eg;
    int rook_seventh_mg, rook_seventh_eg;
    /* TrappedRook: penalty when rook has <=3 moves hemmed in by own king */
    int trapped_rook_mg, trapped_rook_eg;
 
    /* Bishop pair */
    int bishop_pair_mg,         bishop_pair_eg;
    int bishop_pair_open_bonus; /* kept for compat; unused by eval.c */
 
    /* Outposts */
    int outpost_knight_mg, outpost_knight_eg;
    int outpost_bishop_mg, outpost_bishop_eg;
 
    /* King safety */
    int king_shield;
    int king_open_file;
    int king_attacker_weight[6];
    int king_danger_quadratic_scale;
    int king_pawn_storm_penalty;
    int king_eg_distance_penalty; /* EG: per Chebyshev step between kings */
 
    /* WeakQueen: penalty when an enemy slider x-rays through our queen */
    int weak_queen_mg, weak_queen_eg;
 
    /* KingProtector: per-step penalty for minors far from own king */
    int king_protector_mg, king_protector_eg;
 
    /* MinorBehindPawn: bonus per minor sheltered behind a friendly pawn */
    int minor_behind_pawn_mg, minor_behind_pawn_eg;
 
    /* Tactical threats */
    int hang_penalty_mg[6];
    int hang_penalty_eg[6];
    int fork_base_mg,  fork_base_eg;
    int fork_extra_mg, fork_extra_eg;
    int skewer_rq_mg,  skewer_rq_eg;
    int skewer_bq_mg,  skewer_bq_eg;
 
    /* Tempo */
    int tempo_mg, tempo_eg;
 
    /* Material imbalance (SF 11 quadratic, flat [6*6] row-major) */
    int quad_ours[36];
    int quad_theirs[36];
 
} EvalWeights;

/* ── Public API ─────────────────────────────────────────────────────── */

/*
 * Initialise *w from the hard-coded defaults in eval.c.
 * Always call this before tuning so you start from a known-good baseline.
 */
void weights_init_defaults(EvalWeights *w);

/*
 * Evaluate *b using the weights in *w instead of eval.c's static constants.
 * Returns a score in centipawns from the side-to-move's perspective.
 * This is the core function called by every worker thread.
 */
int evaluate_w(const Board *b, const EvalWeights *w);

/*
 * Find the optimal sigmoid scaling constant K by a Golden-section search
 * over the position corpus.  Call once before tune_run() if you want to
 * calibrate K rather than use the default 1.13/400.
 * Returns the optimal K value; pass it (× 400) to tune_run if desired.
 */
double tune_find_k(int num_threads);

/*
 * Run the coordinate-descent tuner.
 *
 *   epd_path   : path to annotated position file (see format above)
 *   num_threads: worker threads (recommended: physical core count)
 *   max_iters  : maximum full sweeps; stops early on convergence
 *   out_path   : output file for tuned weights as C source (NULL → stdout)
 */
void tune_run(const char *epd_path, int num_threads, int max_iters,
              const char *out_path);


/* fixes a compiler error */
static void register_all_params(EvalWeights *w);

typedef struct {
    int        *ptr;
    int         lo;
    int         hi;
    const char *name;
} Param;

#define MAX_PARAMS 2048
static Param t_g_params[MAX_PARAMS];
static int g_nparams = 0;