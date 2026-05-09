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
    /* ── Material (centipawns) ── */
    int material_mg[6];
    int material_eg[6];

    /* ── Piece-square tables [piece_type][square 0..63] ──
     * White perspective; Black squares are mirrored (sq ^ 56) inside
     * evaluate_w(), matching the convention in eval.c.               */
    int pst_mg[6][64];
    int pst_eg[6][64];

    /* ── Pawn structure ── */
    int doubled_pawn_mg,  doubled_pawn_eg;
    int isolated_pawn_mg, isolated_pawn_eg;
    int backward_pawn_mg, backward_pawn_eg;
    int passed_pawn_mg[8];   /* indexed by bonus_rank (0=back rank, 7=promo) */
    int passed_pawn_eg[8];

    /* ── Mobility (per reachable square) ── */
    int mobility_mg[6];
    int mobility_eg[6];

    /* ── Rook bonuses ── */
    int rook_open_mg,    rook_open_eg;
    int rook_semi_mg,    rook_semi_eg;
    int rook_seventh_mg, rook_seventh_eg;

    /* ── Bishop pair ── */
    int bishop_pair_mg, bishop_pair_eg;

    /* ── Outposts ── */
    int outpost_knight_mg, outpost_knight_eg;
    int outpost_bishop_mg, outpost_bishop_eg;

    /* ── King safety ── */
    int king_shield;            /* pawn-shield bonus per pawn      */
    int king_open_file;         /* open-file penalty (negative)    */
    int king_attacker_weight[6];/* indexed by piece type           */

    /* ── Tempo ── */
    int tempo_mg, tempo_eg;
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