/*
 * tune_internal.h — Internal types and function declarations from tune.c
 *
 * Purpose
 * ───────
 * tune.c keeps its training primitives (ReplayBuffer, AdamState, dataset
 * helpers, the training loop) as translation-unit-private symbols to avoid
 * polluting the global namespace.  The distributed layer (dist_tune.c) needs
 * to drive those same primitives.
 *
 * Usage
 * ─────
 * 1. Include this header in tune.c (before the static definitions).
 * 2. Remove the `static` qualifier from every function listed in the
 *    "Internal function declarations" section below.
 * 3. Include this header in dist_tune.c.
 *
 * The types are defined HERE and included by both translation units so that
 * sizeof() and layout always agree.
 *
 * Constants that govern memory usage are also centralised here so that both
 * tune.c and dist_tune.c see the same values.
 */

#pragma once

#include "nnue.h"   /* NNUENet, NNUEGrad, NNUE_TOTAL_PARAMS */
#include "board.h"  /* Board */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Shared constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef REPLAY_BUFFER_CAP
#  define REPLAY_BUFFER_CAP  1000000
#endif

#ifndef WDL_LAMBDA
#  define WDL_LAMBDA  0.5f
#endif

#ifndef GRAD_CLIP_NORM
#  define GRAD_CLIP_NORM  1.0f
#endif

#ifndef FILTER_EVAL_CP
#  define FILTER_EVAL_CP  2000
#endif

#ifndef ADAM_LR
#  define ADAM_LR   1e-3f
#  define ADAM_B1   0.9f
#  define ADAM_B2   0.999f
#  define ADAM_EPS  1e-8f
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Types
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * One position stored in the replay buffer.
 * score_cp is the STM search evaluation at the ply the position was recorded;
 * result is the final game outcome, white-relative (1.0 / 0.5 / 0.0).
 */
typedef struct {
    Board  board;
    double result;
    int    score_cp;
} TunePos;

/*
 * Adam optimiser moment state.
 * Must NOT be reset between RL iterations — resetting m/v/t forces Adam
 * back into its high-variance warm-up phase on every cycle.
 */
typedef struct {
    NNUEGrad m;   /* first  moment (EMA of gradient)   */
    NNUEGrad v;   /* second moment (EMA of gradient²)  */
    int      t;   /* global step counter               */
} AdamState;

/* Lightweight per-iteration dataset; freed after positions are pushed to
 * the replay buffer. */
typedef struct {
    TunePos *positions;
    int      count;
    int      cap;
} PosDataset;

/* Circular ring buffer — AlphaZero experience window. */
typedef struct {
    TunePos     *buf;
    int          cap;
    int          count;
    int          head;
    unsigned int seed;
} ReplayBuffer;

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  Internal function declarations
 *     (remove `static` from the matching definitions in tune.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Dataset helpers */
PosDataset  *create_dataset(void);
void         free_dataset(PosDataset *ds);

/* Replay buffer */
ReplayBuffer *replay_create(int cap);
void          replay_free(ReplayBuffer *rb);
void          replay_push(ReplayBuffer *rb, const TunePos *pos);

/* Self-play */
void self_play_worker(int num_games, int depth, PosDataset *ds);

/* Training */
void train_on_dataset(NNUENet *net, ReplayBuffer *rb,
                      int max_epochs, int batch_size,
                      float lr, float wdl_lambda,
                      int nthreads, AdamState *adam);

/* Learning-rate schedule */
float cosine_lr(float lr_max, float lr_min, int iter, int total_iters);

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  Shared globals
 *
 *  g_nnue        — defined in nnue.c; used by tune.c and dist_tune.c
 *  g_num_threads — defined in tune.c (non-static); written by dist_tune.c
 *                  before the training loop, read inside tune.c workers
 * ═══════════════════════════════════════════════════════════════════════════ */

extern NNUENet *g_nnue;        /* owned by nnue.c          */
extern int      g_num_threads; /* owned by tune.c          */

#ifdef __cplusplus
}
#endif