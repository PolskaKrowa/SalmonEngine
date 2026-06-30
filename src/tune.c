/*
 * tune.c — NNUE training via self-play reinforcement learning (AlphaZero-style)
 *
 * ════════════════════════════════════════════════════════════════════════
 * MEMORY BUDGET
 * ════════════════════════════════════════════════════════════════════════
 *
 *   NNUENet            ~185 MB
 *   AdamState          ~370 MB  (two NNUEGrad: m + v)
 *   Replay buffer      ~varies  (REPLAY_BUFFER_CAP × sizeof(TunePos))
 *   Per-thread grads   ~185 MB × g_num_threads  (during training)
 *
 *   Reduce g_num_threads or REPLAY_BUFFER_CAP on memory-constrained machines.
 *
 * ════════════════════════════════════════════════════════════════════════
 * DISTRIBUTED TRAINING HOOK
 * ════════════════════════════════════════════════════════════════════════
 *
 *   dist_allreduce_gradients() is a no-op.  Replace with MPI_Allreduce,
 *   NCCL AllReduce, or similar before the Adam step.
 *
 *   Example (MPI):
 *     MPI_Allreduce(MPI_IN_PLACE, (float*)g,
 *                   NNUE_TOTAL_PARAMS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
 */

#include "nnue.h"
#include "bitboard.h"
#include "board.h"
#include "movegen.h"

#include <openblas/cblas.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#ifndef M_PI
#  define M_PI 3.141592653589793238462643383279502884
#endif

/* ── Threading ───────────────────────────────────────────────────────── */
#define MAX_TUNE_THREADS 128
int g_num_threads = 1;

/* Portable, lock-free per-thread LCG PRNG. */
static inline int tune_rand(unsigned int *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (int)((*seed >> 1) & 0x7fffffff);
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 1 — Training hyper-parameters
 * ════════════════════════════════════════════════════════════════════════ */

#define ADAM_LR    1e-3f
#define ADAM_B1    0.9f
#define ADAM_B2    0.999f
#define ADAM_EPS   1e-8f

/*
 * Replay buffer capacity.  Acts as AlphaZero's "window of recent games".
 * Older positions are evicted as new ones arrive (ring buffer).
 *
 * Tune to your available RAM:
 *   1 M positions × ~300 B per TunePos  ≈  ~300 MB
 */
#define REPLAY_BUFFER_CAP  1000000

/*
 * WDL mixing coefficient λ ∈ [0, 1]:
 *   target = λ · σ(K · stored_search_score)  +  (1−λ) · game_result
 *
 *   λ = 0.0 → pure game result (classic AlphaZero)
 *   λ = 1.0 → pure search-score target (ignores outcomes entirely)
 *   λ = 0.5 → balanced blend (recommended; matches Lc0 / SF practice)
 *
 * Using the stored search score (not the network's live eval) gives a
 * fixed, lower-variance target that stabilises training.
 */
#define WDL_LAMBDA         0.5f

/*
 * Global L2-norm gradient clipping threshold.
 * Prevents exploding gradients during policy oscillations.
 */
#define GRAD_CLIP_NORM     1.0f

/*
 * Position filter: discard positions with |eval_white_cp| above this
 * threshold.  Extreme evaluations carry little information beyond the
 * game result; keeping them adds noise.
 *   2000 cp ≈ a clean extra queen — clearly decided.
 */
#define FILTER_EVAL_CP     2000

/* Sigmoid scaling constant — must match loss computation. */
static const double g_K = 1.13 / 400.0;

/* ════════════════════════════════════════════════════════════════════════
 * Section 2 — Adam optimiser state
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    NNUEGrad m;   /* first  moment (EMA of gradient)   */
    NNUEGrad v;   /* second moment (EMA of gradient²)  */
    int      t;   /* global step counter — MUST NOT reset between RL iters */
} AdamState;

/* ════════════════════════════════════════════════════════════════════════
 * Section 3 — Forward pass (activations cached for backprop)
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * Naming convention:
 *   pre_*  — pre-activation (before SCReLU)
 *   h*     — post-activation (after SCReLU)
 */
typedef struct {
    float pre_stm[NNUE_L1];
    float pre_opp[NNUE_L1];
    float x[2 * NNUE_L1];    /* SCReLU(pre_stm) ‖ SCReLU(pre_opp) */
    float pre_h1[NNUE_L2];
    float h1    [NNUE_L2];
    float pre_h2[NNUE_L3];
    float h2    [NNUE_L3];
    float out;

    int stm_feats[32];
    int opp_feats[32];
    int n_stm, n_opp;
    int stm;
} NNUEFwdCache;

static inline float screlu_f(float x) {
    float c = x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
    return c * c;
}

/*
 * Derivative of SCReLU w.r.t. the pre-activation z:
 *   d/dz [clamp(z,0,1)²] = 2z   for z ∈ (0, 1), else 0.
 */
static inline float dscrelu(float z) {
    return (z > 0.0f && z < 1.0f) ? 2.0f * z : 0.0f;
}

/*
 * Forward pass storing all intermediate activations for backprop.
 *
 * The feature transformer (sparse) is hand-rolled — there are only ~30
 * active rows per perspective so BLAS overhead would dominate.
 *
 * The dense hidden layers use cblas_sgemv:
 *   Weight layout: net->l1_weight[in_idx][out_idx]  →  (2·L1 × L2) row-major.
 *   We compute:  pre_h = bias + W^T · x
 *   In BLAS terms: CblasTrans sgemv with M=2·L1, N=L2.
 */
static float nnue_forward_train(const NNUENet *net, const Board *b,
                                NNUEFwdCache *c) {
    c->stm = b->side;

    c->n_stm = nnue_get_features(b, c->stm,     c->stm_feats);
    c->n_opp = nnue_get_features(b, c->stm ^ 1, c->opp_feats);

    /* Feature transformer — sparse accumulation */
    memcpy(c->pre_stm, net->ft_bias, NNUE_L1 * sizeof(float));
    memcpy(c->pre_opp, net->ft_bias, NNUE_L1 * sizeof(float));
    for (int i = 0; i < c->n_stm; i++) {
        const float *row = net->ft_weight[c->stm_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) c->pre_stm[k] += row[k];
    }
    for (int i = 0; i < c->n_opp; i++) {
        const float *row = net->ft_weight[c->opp_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) c->pre_opp[k] += row[k];
    }

    /* SCReLU + concatenate → x */
    for (int k = 0; k < NNUE_L1; k++) {
        c->x[k]           = screlu_f(c->pre_stm[k]);
        c->x[NNUE_L1 + k] = screlu_f(c->pre_opp[k]);
    }

    /*
     * Hidden layer 1: pre_h1 = l1_bias + l1_weight^T · x
     *
     * l1_weight is (2·NNUE_L1 × NNUE_L2) row-major (outer dim = input).
     * CblasTrans gives us the matrix-vector product transposed:
     *   y[j] += Σ_k  A[k][j] · x[k]    (j: output, k: input)
     */
    memcpy(c->pre_h1, net->l1_bias, NNUE_L2 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans,
                2 * NNUE_L1, NNUE_L2,
                1.0f, (const float *)net->l1_weight, NNUE_L2,
                c->x, 1,
                1.0f, c->pre_h1, 1);
    for (int j = 0; j < NNUE_L2; j++) c->h1[j] = screlu_f(c->pre_h1[j]);

    /* Hidden layer 2: pre_h2 = l2_bias + l2_weight^T · h1 */
    memcpy(c->pre_h2, net->l2_bias, NNUE_L3 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans,
                NNUE_L2, NNUE_L3,
                1.0f, (const float *)net->l2_weight, NNUE_L3,
                c->h1, 1,
                1.0f, c->pre_h2, 1);
    for (int j = 0; j < NNUE_L3; j++) c->h2[j] = screlu_f(c->pre_h2[j]);

    /* Output layer: simple dot product (L3 = 32; BLAS overhead not worth it) */
    float out = net->l3_bias;
    for (int j = 0; j < NNUE_L3; j++) out += net->l3_weight[j] * c->h2[j];
    c->out = out;
    return out;
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 4 — Backward pass (BLAS-accelerated dense layers)
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * Accumulates gradients into *grad (does NOT zero it; caller manages zeroing).
 * dL_d_out: ∂L/∂(raw_network_output), STM perspective, sign-adjusted by caller.
 *
 * Weight gradient pattern (cblas_sger outer product):
 *   ∂L/∂W[k][j] = ∂L/∂(pre_h)[j] · input[k]
 *   → sger: A += α · x · y^T  (A is M×N, x length M, y length N)
 *
 * Input gradient pattern (cblas_sgemv no-trans):
 *   ∂L/∂(input[k]) = Σ_j  W[k][j] · ∂L/∂(pre_h)[j]
 *   → sgemv CblasNoTrans: y = A·x  (A is M×N, x length N, y length M)
 */
static void nnue_backward_train(const NNUENet *net, const NNUEFwdCache *c,
                                float dL_d_out, NNUEGrad *grad) {

    /* ── Output layer ─────────────────────────────────────────────────── */
    grad->l3_bias += dL_d_out;
    for (int j = 0; j < NNUE_L3; j++)
        grad->l3_weight[j] += dL_d_out * c->h2[j];

    /* ── Layer 2 backward ────────────────────────────────────────────── */
    /* d(pre_h2[j]) = dL_d_out · l3_weight[j] · dscrelu(pre_h2[j]) */
    float d_pre_h2[NNUE_L3];
    for (int j = 0; j < NNUE_L3; j++)
        d_pre_h2[j] = dL_d_out * net->l3_weight[j] * dscrelu(c->pre_h2[j]);

    /* L2 bias gradient */
    for (int j = 0; j < NNUE_L3; j++) grad->l2_bias[j] += d_pre_h2[j];

    /* L2 weight gradient: grad->l2_weight[k][j] += d_pre_h2[j] · h1[k]
     * = outer product h1 ⊗ d_pre_h2  (sger: A[k][j] += x[k]·y[j]) */
    cblas_sger(CblasRowMajor,
               NNUE_L2, NNUE_L3,
               1.0f, c->h1, 1, d_pre_h2, 1,
               (float *)grad->l2_weight, NNUE_L3);

    /* d_h1[k] = Σ_j  d_pre_h2[j] · l2_weight[k][j]
     * = l2_weight (no-trans) · d_pre_h2  (sgemv CblasNoTrans) */
    float d_h1[NNUE_L2];
    memset(d_h1, 0, sizeof d_h1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                NNUE_L2, NNUE_L3,
                1.0f, (const float *)net->l2_weight, NNUE_L3,
                d_pre_h2, 1,
                1.0f, d_h1, 1);

    /* ── Layer 1 backward ────────────────────────────────────────────── */
    float d_pre_h1[NNUE_L2];
    for (int j = 0; j < NNUE_L2; j++)
        d_pre_h1[j] = d_h1[j] * dscrelu(c->pre_h1[j]);

    /* L1 bias gradient */
    for (int j = 0; j < NNUE_L2; j++) grad->l1_bias[j] += d_pre_h1[j];

    /* L1 weight gradient: sger outer product */
    cblas_sger(CblasRowMajor,
               2 * NNUE_L1, NNUE_L2,
               1.0f, c->x, 1, d_pre_h1, 1,
               (float *)grad->l1_weight, NNUE_L2);

    /* d_x[k] = Σ_j  d_pre_h1[j] · l1_weight[k][j]
     * = l1_weight (no-trans) · d_pre_h1 */
    float d_x[2 * NNUE_L1];
    memset(d_x, 0, sizeof d_x);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                2 * NNUE_L1, NNUE_L2,
                1.0f, (const float *)net->l1_weight, NNUE_L2,
                d_pre_h1, 1,
                1.0f, d_x, 1);

    /* ── Feature transformer — STM accumulator ────────────────────────── */
    for (int k = 0; k < NNUE_L1; k++) {
        float d_acc_stm = d_x[k] * dscrelu(c->pre_stm[k]);
        grad->ft_bias[k] += d_acc_stm;
        for (int i = 0; i < c->n_stm; i++)
            grad->ft_weight[c->stm_feats[i]][k] += d_acc_stm;
    }

    /* ── Feature transformer — OPP accumulator ────────────────────────── */
    for (int k = 0; k < NNUE_L1; k++) {
        float d_acc_opp = d_x[NNUE_L1 + k] * dscrelu(c->pre_opp[k]);
        grad->ft_bias[k] += d_acc_opp;
        for (int i = 0; i < c->n_opp; i++)
            grad->ft_weight[c->opp_feats[i]][k] += d_acc_opp;
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 5 — Gradient utilities
 * ════════════════════════════════════════════════════════════════════════ */

static inline void zero_grad(NNUEGrad *g) { memset(g, 0, sizeof *g); }

static void add_grad(NNUEGrad *dst, const NNUEGrad *src) {
    float       *d = (float *)dst;
    const float *s = (const float *)src;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) d[i] += s[i];
}

static void scale_grad(NNUEGrad *g, float scale) {
    float *p = (float *)g;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) p[i] *= scale;
}

/*
 * Global L2 norm of the gradient vector.
 * Computed in double to avoid catastrophic cancellation with large flat32 vectors.
 */
static float grad_l2_norm(const NNUEGrad *g) {
    const float *p = (const float *)g;
    double sum = 0.0;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++)
        sum += (double)p[i] * (double)p[i];
    return (float)sqrt(sum);
}

/*
 * Clip gradient to a maximum global L2 norm.
 * If the norm exceeds max_norm, all elements are scaled uniformly.
 * This preserves gradient direction while bounding its magnitude.
 */
static void clip_grad(NNUEGrad *g, float max_norm) {
    float norm = grad_l2_norm(g);
    if (norm > max_norm)
        scale_grad(g, max_norm / (norm + 1e-8f));
}

/* ── Adam update ─────────────────────────────────────────────────────── */

static void adam_update_array(float *w, const float *g, float *m, float *v,
                               int n, float lr, float b1, float b2,
                               float eps, float bc1, float bc2) {
    for (int i = 0; i < n; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

static void adam_update(NNUENet *net, const NNUEGrad *grad, AdamState *state,
                        float lr, float b1, float b2, float eps) {
    state->t++;
    float bc1 = 1.0f - powf(b1, (float)state->t);
    float bc2 = 1.0f - powf(b2, (float)state->t);

#define ADAM(field, n) \
    adam_update_array( \
        (float *)&net->field,     (const float *)&grad->field, \
        (float *)&state->m.field, (float *)&state->v.field,    \
        (n), lr, b1, b2, eps, bc1, bc2)

    ADAM(ft_weight, NNUE_FT_IN * NNUE_L1);
    ADAM(ft_bias,   NNUE_L1);
    ADAM(l1_weight, 2 * NNUE_L1 * NNUE_L2);
    ADAM(l1_bias,   NNUE_L2);
    ADAM(l2_weight, NNUE_L2 * NNUE_L3);
    ADAM(l2_bias,   NNUE_L3);
    ADAM(l3_weight, NNUE_L3);
    ADAM(l3_bias,   1);
#undef ADAM

    /*
     * FT weight clipping: bounds follow common NNUE practice.
     * Prevents accumulators growing unboundedly in early training.
     */
    float *ft = (float *)net->ft_weight;
    for (int i = 0; i < NNUE_FT_IN * NNUE_L1; i++) {
        if (ft[i] >  2.0f) ft[i] =  2.0f;
        if (ft[i] < -2.0f) ft[i] = -2.0f;
    }
}

/*
 * Cosine learning-rate schedule.
 *   iter: 0-indexed (0 .. total_iters-1)
 *   Returns a value in [lr_min, lr_max].
 */
float cosine_lr(float lr_max, float lr_min, int iter, int total_iters) {
    if (total_iters <= 1) return lr_max;
    float t = (float)iter / (float)(total_iters - 1);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf((float)M_PI * t));
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 6 — Multithreaded gradient accumulation
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * TunePos stores the board, the game outcome, AND the search score at the
 * time the position was generated.  The stored score is used as the "eval"
 * component of the WDL-mixed target, giving a stable reference that does
 * not change as the network evolves during training.
 */
typedef struct {
    Board  board;
    double result;    /* game outcome, white-relative: 1.0 / 0.5 / 0.0    */
    int    score_cp;  /* STM evaluation at search depth, in centipawns     */
} TunePos;

typedef struct {
    const NNUENet *net;
    const TunePos *positions;
    int            start;
    int            end;
    NNUEGrad      *grad;
    double         partial_err;
    float          wdl_lambda;
} GradWorkerCtx;

static double g_sigmoid_K = 1.13 / 400.0;

static inline double sigmoid_scaled(double cp) {
    return 1.0 / (1.0 + exp(-g_sigmoid_K * cp));
}

static void *grad_worker_fn(void *arg) {
    GradWorkerCtx *ctx     = (GradWorkerCtx *)arg;
    const TunePos *pos_arr = ctx->positions;
    double         err     = 0.0;
    float          lambda  = ctx->wdl_lambda;

    for (int i = ctx->start; i < ctx->end; i++) {
        const TunePos *pos = &pos_arr[i];

        NNUEFwdCache cache;
        float raw_eval_stm = nnue_forward_train(ctx->net, &pos->board, &cache);

        double eval_stm_scaled = (double)raw_eval_stm * NNUE_OUTPUT_SCALE;
        double eval_white_cp   = (pos->board.side == WHITE)
                                     ?  eval_stm_scaled
                                     : -eval_stm_scaled;

        /* Network's current prediction (white-relative sigmoid) */
        double sigma = sigmoid_scaled(eval_white_cp);

        /*
         * True WDL-mixed target using the STORED search score.
         *
         * Unlike the original code (which mixed with sigma, the network's
         * own live output), we mix with the sigmoid of the search score
         * recorded at self-play time.  This provides a fixed, stable target
         * that is independent of the current network weights.
         *
         * stored_eval_white: the search score in centipawns, white-relative.
         * sigma_stored:      sigmoid(K · stored_eval_white) — fixed per position.
         *
         *   target = λ · sigma_stored  +  (1−λ) · game_result
         *
         * At λ=0.5: half the signal comes from the (fixed) search score that
         * generated the position; half from the actual game outcome.
         */
        double stored_eval_white = (pos->board.side == WHITE)
                                       ?  (double)pos->score_cp
                                       : -(double)pos->score_cp;
        double sigma_stored = sigmoid_scaled(stored_eval_white);
        double target       = (double)lambda * sigma_stored
                            + (1.0 - (double)lambda) * pos->result;

        double d   = sigma - target;
        err       += d * d;

        /* ∂L/∂(eval_white_cp) */
        double dL_d_eval_white = 2.0 * d * sigma * (1.0 - sigma) * g_sigmoid_K;

        /* Chain back through the STM perspective sign and output scale */
        float dL_d_raw_stm = (float)(
            (pos->board.side == WHITE ? dL_d_eval_white : -dL_d_eval_white)
            * NNUE_OUTPUT_SCALE);

        nnue_backward_train(ctx->net, &cache, dL_d_raw_stm, ctx->grad);
    }

    ctx->partial_err = err;
    return NULL;
}

/*
 * ── Distributed training hook ─────────────────────────────────────────
 * No-op in single-machine mode.  Replace with an allreduce for multi-node:
 *   MPI:   MPI_Allreduce(MPI_IN_PLACE, (float*)g, NNUE_TOTAL_PARAMS,
 *                        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
 *   NCCL:  ncclAllReduce((float*)g, (float*)g, NNUE_TOTAL_PARAMS,
 *                        ncclFloat, ncclSum, comm, stream);
 */
static void dist_allreduce_gradients(NNUEGrad *g) { (void)g; }

/*
 * Compute gradients over positions[0..count) in parallel.
 * Returns mean squared error.  Gradient is the SUM (caller divides by batch).
 */
static double compute_gradients_mt(const NNUENet *net, const TunePos *positions,
                                   int count, NNUEGrad *grad_out,
                                   int nthreads, float wdl_lambda) {
    if (count <= 0) return 0.0;
    if (nthreads > count) nthreads = count;
    if (nthreads < 1)     nthreads = 1;

    NNUEGrad     **tgrads  = malloc((size_t)nthreads * sizeof(NNUEGrad *));
    GradWorkerCtx *ctxs    = malloc((size_t)nthreads * sizeof(GradWorkerCtx));
    pthread_t     *threads = malloc((size_t)nthreads * sizeof(pthread_t));
    if (!tgrads || !ctxs || !threads) { perror("compute_gradients_mt"); exit(1); }

    int slice = count / nthreads;
    for (int t = 0; t < nthreads; t++) {
        tgrads[t] = calloc(1, sizeof(NNUEGrad));
        if (!tgrads[t]) { perror("calloc grad"); exit(1); }

        ctxs[t].net         = net;
        ctxs[t].positions   = positions;
        ctxs[t].start       = t * slice;
        ctxs[t].end         = (t == nthreads - 1) ? count : (t + 1) * slice;
        ctxs[t].grad        = tgrads[t];
        ctxs[t].partial_err = 0.0;
        ctxs[t].wdl_lambda  = wdl_lambda;

        pthread_create(&threads[t], NULL, grad_worker_fn, &ctxs[t]);
    }

    zero_grad(grad_out);
    double total_err = 0.0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        add_grad(grad_out, tgrads[t]);
        total_err += ctxs[t].partial_err;
        free(tgrads[t]);
    }
    free(threads); free(ctxs); free(tgrads);

    dist_allreduce_gradients(grad_out);

    return total_err / (double)count;
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 7 — Replay buffer (AlphaZero-style experience window)
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    TunePos     *buf;    /* circular ring                          */
    int          cap;    /* allocated capacity (fixed)             */
    int          count;  /* live positions: 0 .. cap               */
    int          head;   /* next write index                       */
    unsigned int seed;   /* PRNG state for sampling                */
} ReplayBuffer;

ReplayBuffer *replay_create(int cap) {
    ReplayBuffer *rb = malloc(sizeof(ReplayBuffer));
    if (!rb) return NULL;
    rb->buf   = malloc((size_t)cap * sizeof(TunePos));
    if (!rb->buf) { free(rb); return NULL; }
    rb->cap   = cap;
    rb->count = 0;
    rb->head  = 0;
    rb->seed  = (unsigned int)time(NULL);
    return rb;
}

void replay_free(ReplayBuffer *rb) {
    if (rb) { free(rb->buf); free(rb); }
}

/*
 * Push one position into the buffer.
 * If full, the oldest position is silently evicted (ring behaviour).
 */
void replay_push(ReplayBuffer *rb, const TunePos *pos) {
    rb->buf[rb->head] = *pos;
    rb->head = (rb->head + 1) % rb->cap;
    if (rb->count < rb->cap) rb->count++;
}

/*
 * Sample n positions uniformly at random (with replacement) into out[].
 * Matches AlphaZero's uniform sampling from the experience window.
 */
static void replay_sample(ReplayBuffer *rb, TunePos *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = rb->buf[(unsigned)tune_rand(&rb->seed) % (unsigned)rb->count];
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 8 — Training loop (AdamState externally owned)
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * Run one epoch of minibatch Adam on the replay buffer.
 *
 * We sample uniformly from the full buffer rather than iterating
 * sequentially.  This breaks temporal correlations between same-game
 * positions and ensures every step sees a balanced mix of old and new data.
 *
 * Returns the mean squared error over the epoch.
 */
static double train_one_epoch(NNUENet *net, ReplayBuffer *rb,
                              int batch_size, float lr, float wdl_lambda,
                              int nthreads, AdamState *adam) {
    if (rb->count < batch_size) return 0.0;

    NNUEGrad *grad  = malloc(sizeof(NNUEGrad));
    TunePos  *batch = malloc((size_t)batch_size * sizeof(TunePos));
    if (!grad || !batch) { perror("train_one_epoch: malloc"); exit(1); }

    double total_err = 0.0;
    int    n_batches = 0;

    /* Steps per epoch: one pass over the live replay buffer */
    int steps = rb->count / batch_size;
    if (steps < 1) steps = 1;

    for (int s = 0; s < steps; s++) {
        replay_sample(rb, batch, batch_size);

        double err = compute_gradients_mt(net, batch, batch_size, grad,
                                          nthreads, wdl_lambda);

        scale_grad(grad, 1.0f / (float)batch_size);  /* mean gradient    */
        clip_grad(grad, GRAD_CLIP_NORM);              /* L2 norm clipping */

        adam_update(net, grad, adam, lr, ADAM_B1, ADAM_B2, ADAM_EPS);

        total_err += err;
        n_batches++;
    }

    free(grad);
    free(batch);
    return n_batches > 0 ? total_err / n_batches : 0.0;
}

/*
 * Full training run over the replay buffer for up to max_epochs.
 *
 * KEY INVARIANT: adam is owned by the caller (tune_self_play_loop) and
 * must NOT be allocated or freed here.  The moment estimates in adam
 * must persist across all RL iterations; resetting them was the root
 * cause of the quality degradation in the original code.
 */
void train_on_dataset(NNUENet *net, ReplayBuffer *rb,
                             int max_epochs, int batch_size,
                             float lr, float wdl_lambda,
                             int nthreads, AdamState *adam) {
    printf("[Trainer] replay=%d pos, batch=%d, lr=%.2e, λ=%.2f, "
           "threads=%d, Adam step=%d\n",
           rb->count, batch_size, (double)lr, (double)wdl_lambda,
           nthreads, adam->t);

    double prev_err = 1e18;
    for (int ep = 1; ep <= max_epochs; ep++) {
        double mse = train_one_epoch(net, rb, batch_size, lr, wdl_lambda,
                                     nthreads, adam);
        printf("[Trainer] Epoch %3d / %d  MSE = %.8f  (Adam step %d)\r",
               ep, max_epochs, mse, adam->t);
        fflush(stdout);

        /* Early stopping: quit if epoch-over-epoch improvement is negligible.
         * 1e-7 is loose enough to survive floating-point noise. */
        if (fabs(prev_err - mse) < 1e-7) break;
        prev_err = mse;
    }
    printf("\n");
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 9 — Self-play data generation
 * ════════════════════════════════════════════════════════════════════════ */

#define INF 30000

/* Lightweight per-iteration dataset — only lives until positions are pushed
 * into the replay buffer, then freed. */
typedef struct {
    TunePos *positions;
    int      count;
    int      cap;
} PosDataset;

PosDataset *create_dataset(void) {
    PosDataset *ds = malloc(sizeof(PosDataset));
    ds->cap        = 500000;
    ds->positions  = malloc((size_t)ds->cap * sizeof(TunePos));
    ds->count      = 0;
    return ds;
}

void free_dataset(PosDataset *ds) {
    free(ds->positions);
    free(ds);
}

static int qsearch_nnue(Board *b, int alpha, int beta, int depth) {
    int stand_pat = nnue_eval(g_nnue, b);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    if (depth <= 0)        return alpha;

    MoveList ml;
    gen_captures(b, &ml);

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (!is_legal(b, m)) continue;
        make_move(b, m);
        int score = -qsearch_nnue(b, -beta, -alpha, depth - 1);
        unmake_move(b);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

static int alphabeta_nnue(Board *b, int depth, int alpha, int beta,
                          Move *best_move) {
    if (depth == 0) return qsearch_nnue(b, alpha, beta, 6);

    MoveList ml;
    gen_moves(b, &ml);

    /* Capture-first move ordering */
    for (int i = 0; i < ml.count; i++)
        if (MOVE_IS_CAP(ml.moves[i])) {
            Move t = ml.moves[0]; ml.moves[0] = ml.moves[i]; ml.moves[i] = t;
        }

    int  best_score = -INF;
    Move local_best = NULL_MOVE;
    int  legal      = 0;

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (!is_legal(b, m)) continue;
        legal++;
        make_move(b, m);
        int score = -alphabeta_nnue(b, depth - 1, -beta, -alpha, NULL);
        unmake_move(b);
        if (score > best_score) { best_score = score; local_best = m; }
        if (score > alpha)  alpha = score;
        if (alpha >= beta)  break;
    }

    if (legal == 0) return in_check(b) ? -INF : 0;
    if (best_move) *best_move = local_best;
    return best_score;
}

/*
 * Position quality filter.
 *
 * Returns 1 (keep) or 0 (discard).  We discard:
 *   • Positions in check — the search result is unreliable because the
 *     next move is forced; the eval may reflect forced material loss rather
 *     than positional assessment.
 *   • Positions with |eval_white| > FILTER_EVAL_CP — the game is already
 *     decided; keeping them adds high-variance noise with little signal
 *     beyond the game result.
 */
static int position_is_quiet(const Board *b, int eval_white_cp) {
    if (in_check(b)) return 0;
    if (eval_white_cp >  FILTER_EVAL_CP) return 0;
    if (eval_white_cp < -FILTER_EVAL_CP) return 0;
    return 1;
}

/*
 * Play one self-play game and append qualifying positions to *ds.
 *
 * Changes from the original:
 *   • TunePos now stores score_cp (the search score at that ply) in
 *     addition to the game result, enabling true WDL mixing at training time.
 *   • position_is_quiet() filters positions in check or with extreme evals.
 *   • history_scores[] is allocated alongside history[] to track per-ply evals.
 */
static void play_game(PosDataset *ds, int search_depth, unsigned int *seed) {
#define MAX_GAME_PLY 500
    Board b;
    board_start_pos(&b);

    Board *history        = malloc(MAX_GAME_PLY * sizeof(Board));
    int   *history_scores = malloc(MAX_GAME_PLY * sizeof(int));
    if (!history || !history_scores) {
        free(history); free(history_scores);
        return;
    }
    int ply = 0;

    /* Randomise the first 4 half-moves for opening variety.
     * These positions are NOT added to the dataset (opening book noise). */
    for (int i = 0; i < 4; i++) {
        MoveList ml; gen_moves(&b, &ml);
        int legal[256], n = 0;
        for (int j = 0; j < ml.count; j++)
            if (is_legal(&b, ml.moves[j])) legal[n++] = j;
        if (n == 0) goto cleanup;
        make_move(&b, ml.moves[legal[tune_rand(seed) % n]]);
    }

    double result = 0.5;
    while (ply < MAX_GAME_PLY) {
        Move best  = NULL_MOVE;
        int  score = alphabeta_nnue(&b, search_depth, -INF, INF, &best);

        if (best == NULL_MOVE) {
            result = in_check(&b) ? ((b.side == WHITE) ? 0.0 : 1.0) : 0.5;
            break;
        }
        if (b.halfmove >= 100) { result = 0.5; break; }

        /* Record the board and the STM search score BEFORE making the move */
        history[ply]        = b;
        history_scores[ply] = score;   /* centipawns, STM perspective */
        ply++;
        make_move(&b, best);

        /* Adjudication: avoid playing out obvious resignations */
        if (score >  1000) { result = (b.side == WHITE) ? 0.0 : 1.0; break; }
        if (score < -1000) { result = (b.side == WHITE) ? 1.0 : 0.0; break; }
    }

    /* Append filtered positions to dataset (skip random opening: start i=0) */
    for (int i = 0; i < ply && ds->count < ds->cap; i++) {
        int score_stm       = history_scores[i];
        int eval_white_cp   = (history[i].side == WHITE) ? score_stm : -score_stm;

        if (!position_is_quiet(&history[i], eval_white_cp)) continue;

        TunePos *tp   = &ds->positions[ds->count++];
        tp->board     = history[i];
        tp->result    = result;
        tp->score_cp  = score_stm;   /* STM-relative; grad_worker converts */
    }

cleanup:
    free(history);
    free(history_scores);
#undef MAX_GAME_PLY
}

/* ── Parallel self-play ─────────────────────────────────────────────── */

typedef struct {
    int          num_games;
    int          depth;
    unsigned int seed;
    PosDataset  *out_ds;
} SelfPlayCtx;

static void *self_play_thread_fn(void *arg) {
    SelfPlayCtx *ctx = (SelfPlayCtx *)arg;
    ctx->out_ds = create_dataset();
    for (int i = 0; i < ctx->num_games; i++)
        play_game(ctx->out_ds, ctx->depth, &ctx->seed);
    return NULL;
}

void self_play_worker(int num_games, int depth, PosDataset *ds) {
    int nt = g_num_threads;
    if (nt < 1) nt = 1;
    if (nt > num_games) nt = num_games;

    printf("[Self-play] %d games at depth %d across %d thread(s)...\n",
           num_games, depth, nt);

    if (nt == 1) {
        unsigned int seed = (unsigned int)time(NULL);
        for (int i = 0; i < num_games; i++) {
            play_game(ds, depth, &seed);
            if ((i + 1) % 50 == 0)
                printf("  played %d games (%d positions)\n", i+1, ds->count);
        }
        printf("[Self-play] Done — %d positions.\n", ds->count);
        return;
    }

    pthread_t      threads[MAX_TUNE_THREADS];
    SelfPlayCtx    ctxs   [MAX_TUNE_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 32 * 1024 * 1024);

    int base = num_games / nt, extra = num_games % nt;
    for (int t = 0; t < nt; t++) {
        ctxs[t].num_games = base + (t < extra ? 1 : 0);
        ctxs[t].depth     = depth;
        ctxs[t].seed      = (unsigned int)(time(NULL) ^ (t * 2654435761u));
        ctxs[t].out_ds    = NULL;
        pthread_create(&threads[t], &attr, self_play_thread_fn, &ctxs[t]);
    }
    pthread_attr_destroy(&attr);

    for (int t = 0; t < nt; t++) {
        pthread_join(threads[t], NULL);
        PosDataset *src = ctxs[t].out_ds;
        if (!src) continue;
        int to_copy = src->count;
        if (ds->count + to_copy > ds->cap) to_copy = ds->cap - ds->count;
        if (to_copy > 0) {
            memcpy(ds->positions + ds->count, src->positions,
                   (size_t)to_copy * sizeof(TunePos));
            ds->count += to_copy;
        }
        free_dataset(src);
    }

    printf("[Self-play] Done — %d positions from %d threads.\n", ds->count, nt);
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 10 — Main reinforcement-learning loop
 * ════════════════════════════════════════════════════════════════════════ */

/*
 * AlphaZero-style RL loop:
 *
 *   ONCE:  allocate AdamState (persistent for the full run)
 *          allocate ReplayBuffer (REPLAY_BUFFER_CAP positions)
 *          load checkpoint or randomly initialise
 *
 *   for each iteration:
 *     LR = cosine_lr(lr_max, lr_min, iter, total_iters)
 *     1. Self-play  — generate games with the current network
 *     2. Push       — add new positions to the replay buffer
 *     3. Train      — sample uniformly from the buffer, run Adam
 *     4. Checkpoint — save network weights
 *
 * The loop is terminated externally (by iterations count); training
 * does not stop early based on loss (unlike the original).
 */
void tune_self_play_loop(int iterations, int games_per_iter, int search_depth,
                         int num_threads, int batch_size, float lr_max,
                         int train_epochs) {
    if (num_threads < 1)                num_threads = 1;
    if (num_threads > MAX_TUNE_THREADS) num_threads = MAX_TUNE_THREADS;
    g_num_threads = num_threads;

    float lr_min = lr_max * 0.01f;

    printf("[Tuner] %d iterations, %d games/iter, depth %d, %d thread(s)\n",
           iterations, games_per_iter, search_depth, g_num_threads);
    printf("[Tuner] Adam: lr %.2e → %.2e (cosine)  batch=%d  epochs/iter=%d\n",
           (double)lr_max, (double)lr_min, batch_size, train_epochs);
    printf("[Tuner] WDL λ=%.2f  grad_clip=%.2f  filter=±%d cp  "
           "replay_cap=%d\n",
           (double)WDL_LAMBDA, (double)GRAD_CLIP_NORM,
           FILTER_EVAL_CP, REPLAY_BUFFER_CAP);

    /* Load or randomly initialise the network */
    if (nnue_load(g_nnue, "nnue_checkpoint.bin") != 0) {
        printf("[Tuner] No checkpoint found — random initialisation.\n");
        nnue_init_random(g_nnue);
    }

    /*
     * Persistent AdamState — allocated ONCE for the full training run.
     *
     * ROOT CAUSE FIX: the original code allocated AdamState inside
     * train_on_dataset() and freed it before returning.  Every RL
     * iteration therefore reset m, v, and t to zero, forcing Adam into
     * its high-variance warm-up phase on every cycle.  This caused the
     * quality regression that worsened with training time.
     */
    AdamState *adam = calloc(1, sizeof(AdamState));
    if (!adam) { perror("calloc AdamState (~370 MB)"); return; }
    printf("[Tuner] AdamState allocated (~370 MB).  "
           "Moments will persist across all iterations.\n");

    /*
     * Replay buffer — circular ring of the most recent REPLAY_BUFFER_CAP
     * positions.  Provides data diversity across iterations and prevents
     * the network from catastrophically forgetting earlier learning.
     */
    ReplayBuffer *replay = replay_create(REPLAY_BUFFER_CAP);
    if (!replay) { perror("replay_create"); free(adam); return; }
    printf("[Tuner] Replay buffer allocated (%d slots).\n", REPLAY_BUFFER_CAP);

    for (int iter = 1; iter <= iterations; iter++) {
        printf("\n══════════════════════════════════════════════════════\n");
        printf(" RL Iteration %d / %d   (Adam step %d)\n",
               iter, iterations, adam->t);
        printf("══════════════════════════════════════════════════════\n");

        /* Cosine-scheduled LR for this iteration */
        float lr = cosine_lr(lr_max, lr_min, iter - 1, iterations);
        printf("[Tuner] LR this iteration: %.3e\n", (double)lr);

        /* ── Step 1: Self-play ──────────────────────────────────────── */
        PosDataset *ds = create_dataset();
        self_play_worker(games_per_iter, search_depth, ds);

        if (ds->count == 0) {
            printf("[Tuner] Warning: empty dataset; skipping this iteration.\n");
            free_dataset(ds);
            continue;
        }

        /* ── Step 2: Push new positions into replay buffer ──────────── */
        for (int i = 0; i < ds->count; i++)
            replay_push(replay, &ds->positions[i]);
        free_dataset(ds);

        printf("[Tuner] Replay buffer: %d / %d positions (%.1f%% full)\n",
               replay->count, replay->cap,
               100.0 * replay->count / replay->cap);

        /* ── Step 3: Train from replay buffer ───────────────────────── */
        train_on_dataset(g_nnue, replay, train_epochs, batch_size,
                         lr, WDL_LAMBDA, g_num_threads, adam);

        /* ── Step 4: Checkpoint ─────────────────────────────────────── */
        printf("[Master] Checkpointing (Adam step %d)...\n", adam->t);
        if (nnue_save(g_nnue, "nnue_checkpoint.bin") == 0)
            printf("[Master] Saved to nnue_checkpoint.bin\n");
        else
            fprintf(stderr, "[Master] Checkpoint save failed!\n");
    }

    replay_free(replay);
    free(adam);
    printf("[Tuner] Training complete.\n");
}

/* ════════════════════════════════════════════════════════════════════════
 * Section 11 — Entry point
 * ════════════════════════════════════════════════════════════════════════ */

#ifdef TUNE_STANDALONE
int main(int argc, char **argv) {
    bitboard_init();
    board_init();

    g_nnue = nnue_alloc();
    if (!g_nnue) {
        fprintf(stderr, "Fatal: could not allocate NNUENet (~185 MB)\n");
        return 1;
    }

    /* Defaults */
    int   iterations     = 50;
    int   games_per_iter = 1000;
    int   search_depth   = 3;
    int   num_threads    = 12;
    int   batch_size     = 1024;
    float lr             = ADAM_LR;
    /*
     * Reduced default epochs from 1000 → 10.
     *
     * With the replay buffer, we draw diverse samples from up to
     * REPLAY_BUFFER_CAP positions on every epoch.  A single epoch of
     * (replay_count / batch_size) steps already covers the full buffer
     * once, so many epochs per iteration are not only unnecessary but
     * risk overfitting to the current buffer contents.
     * Increase if the buffer fills slowly in the first few iterations.
     */
    int   train_epochs   = 10;

    /* Optional positional args: threads games_per_iter iters depth batch epochs lr */
    if (argc > 1) num_threads    = atoi(argv[1]);
    if (argc > 2) games_per_iter = atoi(argv[2]);
    if (argc > 3) iterations     = atoi(argv[3]);
    if (argc > 4) search_depth   = atoi(argv[4]);
    if (argc > 5) batch_size     = atoi(argv[5]);
    if (argc > 6) train_epochs   = atoi(argv[6]);
    if (argc > 7) lr             = (float)atof(argv[7]);

    printf("NNUE Self-Play RL Tuner — AlphaZero-style (HalfKAv2_hm)\n");
    printf("  Architecture: %d → %d → %d → %d → 1  (%d params)\n",
           NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_L3, NNUE_TOTAL_PARAMS);
    printf("  Activation: SCReLU  |  Output scale: %d\n", NNUE_OUTPUT_SCALE);
    printf("  iters=%d  games/iter=%d  depth=%d  threads=%d\n",
           iterations, games_per_iter, search_depth, num_threads);
    printf("  Dense layers via OpenBLAS cblas_sgemv / cblas_sger\n");

    tune_self_play_loop(iterations, games_per_iter, search_depth,
                        num_threads, batch_size, lr, train_epochs);

    nnue_free(g_nnue);
    g_nnue = NULL;
    return 0;
}
#endif