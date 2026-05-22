/*
 * tune.c — NNUE training via self-play reinforcement learning
 *
 * ════════════════════════════════════════════════════════════════════════
 * DISTRIBUTED-READY ARCHITECTURE
 * ════════════════════════════════════════════════════════════════════════
 *
 * The system is split into two decoupled phases:
 *
 *   self_play_worker()  — plays games using the current NNUE weights,
 *                         producing a (Board, white-relative result) dataset.
 *
 *   train_on_dataset()  — runs minibatch Adam SGD on the dataset, computing
 *                         gradients in parallel across g_num_threads threads
 *                         and applying them on the master thread.
 *
 * To distribute:
 *   Workers:  call self_play_worker() independently; serialise their
 *             PosDataset to disk via nnue_pos_save() or a network call.
 *   Master:   load / merge all worker datasets, call train_on_dataset(),
 *             then broadcast the new nnue.bin to all workers via nnue_save().
 *
 * Gradient allreduce hook:
 *   dist_allreduce_gradients() is a no-op here.  Replace the body with
 *   MPI_Allreduce, NCCL AllReduce, or your distributed framework's equivalent
 *   to sum gradients across nodes before the Adam step.
 *
 *   Example (MPI):
 *     MPI_Allreduce(MPI_IN_PLACE, (float*)g,
 *                   NNUE_TOTAL_PARAMS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
 *
 * ════════════════════════════════════════════════════════════════════════
 */

#include "nnue.h"
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

/* ── Threading ───────────────────────────────────────────────────── */
#define MAX_TUNE_THREADS 128
static int g_num_threads = 1;

/* Portable re-entrant LCG — each thread keeps its own seed, no locks. */
static inline int tune_rand(unsigned int *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return (int)((*seed >> 1) & 0x7fffffff);
}

/* ════════════════════════════════════════════════════════════════════
 * Section 1 — Adam optimiser state
 * ════════════════════════════════════════════════════════════════════ */

typedef struct {
    NNUEGrad m;  /* first  moment (exponential moving avg of gradient)  */
    NNUEGrad v;  /* second moment (exponential moving avg of gradient²) */
    int      t;  /* step counter (for bias-correction denominators)     */
} AdamState;

/* Default hyper-parameters — override via train_on_dataset() args. */
#define ADAM_LR    1e-3f
#define ADAM_B1    0.9f
#define ADAM_B2    0.999f
#define ADAM_EPS   1e-8f

/* Sigmoid scaling constant (must match loss computation below). */
static const double g_K = 1.13 / 400.0;

/* ════════════════════════════════════════════════════════════════════
 * Section 2 — Forward pass with activations cached for backprop
 * ════════════════════════════════════════════════════════════════════ */

/*
 * All intermediate activations needed for backpropagation.
 * The struct is stack-allocated per position — ~2 KB.
 */
typedef struct {
    /* Raw (pre-CReLU) accumulator outputs for both perspectives */
    float pre_stm[NNUE_L1];
    float pre_opp[NNUE_L1];

    /* Post-CReLU concat input to hidden layer 1: [stm | opp] */
    float x[2 * NNUE_L1];

    /* Raw (pre-CReLU) hidden-layer-1 activations */
    float pre_h[NNUE_L2];

    /* Post-CReLU hidden-layer-1 activations */
    float h[NNUE_L2];

    /* Network output (centipawns, side-to-move perspective) */
    float out;

    /* Active feature indices for both perspectives */
    int stm_feats[32];
    int opp_feats[32];
    int n_stm, n_opp;

    /* Side to move for this position (WHITE=0 / BLACK=1) */
    int stm;
} NNUEFwdCache;

/* CReLU derivative: 1 iff 0 < x < 1, else 0. */
static inline float dcrelu(float x) {
    return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
}
static inline float crelu_f(float x) {
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}

/*
 * Forward pass that stores all intermediate activations.
 * Returns the raw network output (centipawns, STM perspective).
 */
static float nnue_forward_train(const NNUENet *net, const Board *b,
                                NNUEFwdCache *c) {
    c->stm = b->side;

    /* Enumerate features for each perspective */
    c->n_stm = nnue_get_features(b, c->stm,      c->stm_feats);
    c->n_opp = nnue_get_features(b, c->stm ^ 1,  c->opp_feats);

    /* Feature transformer — accumulate into pre_stm / pre_opp */
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

    /* CReLU + concat */
    for (int k = 0; k < NNUE_L1; k++) {
        c->x[k]           = crelu_f(c->pre_stm[k]);
        c->x[NNUE_L1 + k] = crelu_f(c->pre_opp[k]);
    }

    /* Hidden layer 1 */
    for (int j = 0; j < NNUE_L2; j++) {
        float s = net->l1_bias[j];
        for (int k = 0; k < 2 * NNUE_L1; k++)
            s += net->l1_weight[k][j] * c->x[k];
        c->pre_h[j] = s;
        c->h[j]     = crelu_f(s);
    }

    /* Output */
    float out = net->l2_bias;
    for (int j = 0; j < NNUE_L2; j++)
        out += net->l2_weight[j] * c->h[j];
    c->out = out;
    return out;
}

/*
 * Backward pass — accumulates gradients into *grad.
 * grad must have been zeroed (or accumulated from previous positions)
 * before this call; gradients are summed in, not overwritten.
 *
 * dL_d_out: dLoss/d(network_output), already converted to STM perspective
 *           (see compute_grad_slice for the sign-flip logic).
 */
static void nnue_backward_train(const NNUENet *net, const NNUEFwdCache *c,
                                float dL_d_out, NNUEGrad *grad) {
    /* ── Output layer ──────────────────────────────────────────────── */
    grad->l2_bias += dL_d_out;
    for (int j = 0; j < NNUE_L2; j++)
        grad->l2_weight[j] += dL_d_out * c->h[j];

    /* ── Hidden layer 1 ────────────────────────────────────────────── */
    float d_x[2 * NNUE_L1];
    memset(d_x, 0, sizeof d_x);

    for (int j = 0; j < NNUE_L2; j++) {
        float d_pre_h = dL_d_out * net->l2_weight[j] * dcrelu(c->pre_h[j]);

        grad->l1_bias[j] += d_pre_h;
        for (int k = 0; k < 2 * NNUE_L1; k++) {
            grad->l1_weight[k][j] += d_pre_h * c->x[k];
            d_x[k]                += d_pre_h * net->l1_weight[k][j];
        }
    }

    /* ── Feature transformer — STM accumulator ─────────────────────── */
    for (int k = 0; k < NNUE_L1; k++) {
        float d_acc = d_x[k] * dcrelu(c->pre_stm[k]);
        grad->ft_bias[k] += d_acc;
        for (int i = 0; i < c->n_stm; i++)
            grad->ft_weight[c->stm_feats[i]][k] += d_acc;
    }

    /* ── Feature transformer — OPP accumulator ─────────────────────── */
    for (int k = 0; k < NNUE_L1; k++) {
        float d_acc = d_x[NNUE_L1 + k] * dcrelu(c->pre_opp[k]);
        grad->ft_bias[k] += d_acc;
        for (int i = 0; i < c->n_opp; i++)
            grad->ft_weight[c->opp_feats[i]][k] += d_acc;
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Section 3 — Gradient utilities
 * ════════════════════════════════════════════════════════════════════ */

static inline void zero_grad(NNUEGrad *g) {
    memset(g, 0, sizeof *g);
}

/* Add src into dst (element-wise), treating both as flat float arrays. */
static void add_grad(NNUEGrad *dst, const NNUEGrad *src) {
    float       *d = (float *)dst;
    const float *s = (const float *)src;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) d[i] += s[i];
}

/* Divide every element of g by scalar (to convert sum→mean). */
static void scale_grad(NNUEGrad *g, float inv_scale) {
    float *p = (float *)g;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) p[i] *= inv_scale;
}

/*
 * Adam update for a single contiguous float array.
 *   bc1, bc2: pre-computed bias-correction terms (1 - β1^t), (1 - β2^t).
 */
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

/*
 * Apply one Adam step to every parameter in net using the gradient in
 * grad and the moment estimates in state.
 */
static void adam_update(NNUENet *net, const NNUEGrad *grad, AdamState *state,
                        float lr, float b1, float b2, float eps) {
    state->t++;
    float bc1 = 1.0f - powf(b1, (float)state->t);
    float bc2 = 1.0f - powf(b2, (float)state->t);

#define ADAM(field, n) \
    adam_update_array( \
        (float *)&net->field, (const float *)&grad->field, \
        (float *)&state->m.field, (float *)&state->v.field, \
        (n), lr, b1, b2, eps, bc1, bc2)

    ADAM(ft_weight,  NNUE_FT_IN * NNUE_L1);
    ADAM(ft_bias,    NNUE_L1);
    ADAM(l1_weight,  2 * NNUE_L1 * NNUE_L2);
    ADAM(l1_bias,    NNUE_L2);
    ADAM(l2_weight,  NNUE_L2);
    ADAM(l2_bias,   1);
#undef ADAM

    /*
     * Weight clipping on the feature transformer prevents the
     * accumulators from growing without bound during early training.
     * Tight bounds here follow common NNUE practice; relax once the
     * network has stabilised.
     */
    float *ft = (float *)net->ft_weight;
    for (int i = 0; i < NNUE_FT_IN * NNUE_L1; i++) {
        if (ft[i] >  2.0f) ft[i] =  2.0f;
        if (ft[i] < -2.0f) ft[i] = -2.0f;
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Section 4 — Multithreaded gradient accumulation
 * ════════════════════════════════════════════════════════════════════ */

typedef struct {
    const NNUENet *net;
    const void    *positions;   /* TunePos array — forward declaration below */
    int            start;
    int            end;
    NNUEGrad      *grad;        /* caller-allocated; zeroed before thread launch */
    double         partial_err;
} GradWorkerCtx;

/* Forward declaration (TunePos defined in Section 6) */
typedef struct { Board board; double result; } TunePos;

static double g_sigmoid_K = 1.13 / 400.0;

static inline double sigmoid_scaled(double cp) {
    return 1.0 / (1.0 + exp(-g_sigmoid_K * cp));
}

static void *grad_worker_fn(void *arg) {
    GradWorkerCtx *ctx     = (GradWorkerCtx *)arg;
    const TunePos *pos_arr = (const TunePos *)ctx->positions;
    double         err     = 0.0;

    for (int i = ctx->start; i < ctx->end; i++) {
        const TunePos *pos = &pos_arr[i];

        NNUEFwdCache cache;
        float eval_stm = nnue_forward_train(ctx->net, &pos->board, &cache);

        /*
         * Loss is MSE between sigmoid(eval_white / K) and result (white-
         * relative: 1.0=white wins, 0.5=draw, 0.0=black wins).
         *
         * Convert STM eval → white-relative eval:
         *   eval_white = (stm==WHITE) ? eval_stm : -eval_stm
         *
         * dL/d(eval_white) = 2·(σ - result)·σ·(1−σ)·K
         *
         * Chain rule back to STM perspective (the direction the network sees):
         *   dL/d(eval_stm) = dL/d(eval_white) · (stm==WHITE ? 1 : -1)
         */
        double eval_white = (pos->board.side == WHITE) ? eval_stm : -(double)eval_stm;
        double sigma      = sigmoid_scaled(eval_white);
        double d          = sigma - pos->result;
        err              += d * d;

        double dL_d_eval_white = 2.0 * d * sigma * (1.0 - sigma) * g_sigmoid_K;
        float  dL_d_eval_stm   = (float)((pos->board.side == WHITE)
                                         ?  dL_d_eval_white
                                         : -dL_d_eval_white);

        nnue_backward_train(ctx->net, &cache, dL_d_eval_stm, ctx->grad);
    }

    ctx->partial_err = err;
    return NULL;
}

/*
 * ── DISTRIBUTED TRAINING HOOK ──────────────────────────────────────
 *
 * This function is called once per Adam step, after all local threads
 * have reduced their gradients into *g, and before the Adam update.
 *
 * In a single-machine run this is a no-op.
 *
 * For multi-machine distributed training, replace the body with an
 * allreduce over all worker nodes.  The result should be the *sum*
 * of gradients (not the mean) — train_on_dataset() divides by batch
 * size afterwards, so both local and remote positions are included.
 *
 * MPI example:
 *   MPI_Allreduce(MPI_IN_PLACE, (float*)g,
 *                 NNUE_TOTAL_PARAMS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
 *
 * NCCL example (GPU):
 *   ncclAllReduce((float*)g, (float*)g, NNUE_TOTAL_PARAMS,
 *                 ncclFloat, ncclSum, comm, stream);
 *   cudaStreamSynchronize(stream);
 */
static void dist_allreduce_gradients(NNUEGrad *g) {
    (void)g;  /* single-machine: nothing to do */
}

/*
 * Compute gradients over positions[0..count) in parallel, writing the
 * summed gradient (NOT divided by count) into *grad_out.
 * Returns the mean squared error over the batch.
 */
static double compute_gradients_mt(const NNUENet *net, const TunePos *positions,
                                   int count, NNUEGrad *grad_out, int nthreads) {
    if (count <= 0) return 0.0;

    /* Clamp threads to the batch size */
    if (nthreads > count) nthreads = count;
    if (nthreads < 1)     nthreads = 1;

    /* Allocate one NNUEGrad per thread (~853 KB each, heap-allocated) */
    NNUEGrad     **tgrads  = malloc((size_t)nthreads * sizeof(NNUEGrad *));
    GradWorkerCtx *ctxs    = malloc((size_t)nthreads * sizeof(GradWorkerCtx));
    pthread_t     *threads = malloc((size_t)nthreads * sizeof(pthread_t));
    if (!tgrads || !ctxs || !threads) { perror("compute_gradients_mt: malloc"); exit(1); }

    int slice = count / nthreads;
    for (int t = 0; t < nthreads; t++) {
        tgrads[t] = calloc(1, sizeof(NNUEGrad));
        if (!tgrads[t]) { perror("compute_gradients_mt: calloc grad"); exit(1); }

        ctxs[t].net          = net;
        ctxs[t].positions    = positions;
        ctxs[t].start        = t * slice;
        ctxs[t].end          = (t == nthreads - 1) ? count : (t + 1) * slice;
        ctxs[t].grad         = tgrads[t];
        ctxs[t].partial_err  = 0.0;

        pthread_create(&threads[t], NULL, grad_worker_fn, &ctxs[t]);
    }

    /* Join and reduce */
    zero_grad(grad_out);
    double total_err = 0.0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        add_grad(grad_out, tgrads[t]);
        total_err += ctxs[t].partial_err;
        free(tgrads[t]);
    }

    free(threads); free(ctxs); free(tgrads);

    /*
     * Distributed hook: sum gradients from all nodes before the Adam step.
     * If running distributed, total_err should also be allreduced here.
     */
    dist_allreduce_gradients(grad_out);

    return total_err / (double)count;
}

/* ════════════════════════════════════════════════════════════════════
 * Section 5 — Adam training loop
 * ════════════════════════════════════════════════════════════════════ */

typedef struct {
    TunePos *positions;
    int      count;
} PosDataset;

static PosDataset *create_dataset(void) {
    PosDataset *ds = malloc(sizeof(PosDataset));
    ds->positions  = malloc(500000 * sizeof(TunePos));
    ds->count      = 0;
    return ds;
}

static void free_dataset(PosDataset *ds) {
    free(ds->positions);
    free(ds);
}

/*
 * Shuffle the dataset in-place (Fisher-Yates).
 * Helps break correlations between consecutive positions from the same game.
 */
static void shuffle_dataset(PosDataset *ds, unsigned int *seed) {
    for (int i = ds->count - 1; i > 0; i--) {
        int j = tune_rand(seed) % (i + 1);
        TunePos tmp      = ds->positions[i];
        ds->positions[i] = ds->positions[j];
        ds->positions[j] = tmp;
    }
}

/*
 * Train on ds for one epoch using minibatch Adam SGD.
 *
 *   net        : network to train (modified in-place, i.e. g_nnue)
 *   ds         : dataset of (Board, white-relative result) pairs
 *   batch_size : positions per Adam step (default: 1024)
 *   lr         : learning rate (default: ADAM_LR = 1e-3)
 *   nthreads   : threads to use for gradient accumulation
 *   state      : persistent Adam moment state (pass the same struct across
 *                epochs; zero-initialise once before the first call)
 */
static double train_one_epoch(NNUENet *net, PosDataset *ds,
                              int batch_size, float lr, int nthreads,
                              AdamState *state) {
    NNUEGrad *grad = malloc(sizeof(NNUEGrad));
    if (!grad) { perror("train_one_epoch: malloc grad"); exit(1); }

    unsigned int seed     = (unsigned int)time(NULL) ^ (unsigned int)state->t;
    double       total_err = 0.0;
    int          n_batches = 0;

    shuffle_dataset(ds, &seed);

    for (int start = 0; start < ds->count; start += batch_size) {
        int end   = start + batch_size;
        if (end > ds->count) end = ds->count;
        int bsz   = end - start;

        double err = compute_gradients_mt(net, ds->positions + start,
                                          bsz, grad, nthreads);
        scale_grad(grad, 1.0f / (float)bsz);  /* mean gradient */

        adam_update(net, grad, state, lr, ADAM_B1, ADAM_B2, ADAM_EPS);

        total_err += err;
        n_batches++;
    }

    free(grad);
    return n_batches > 0 ? total_err / n_batches : 0.0;
}

/*
 * Full training run on ds — multiple epochs until convergence or max_epochs.
 * Prints per-epoch MSE.  Calls train_one_epoch() internally; state is
 * allocated here so moments accumulate across epochs (correct for Adam).
 */
static void train_on_dataset(NNUENet *net, PosDataset *ds,
                             int max_epochs, int batch_size,
                             float lr, int nthreads) {
    AdamState *state = calloc(1, sizeof(AdamState));
    if (!state) { perror("train_on_dataset: calloc"); exit(1); }

    printf("[Trainer] %d positions, batch=%d, lr=%.1e, threads=%d\n",
           ds->count, batch_size, (double)lr, nthreads);

    double prev_err = 1e18;
    for (int ep = 1; ep <= max_epochs; ep++) {
        double mse = train_one_epoch(net, ds, batch_size, lr, nthreads, state);
        printf("[Trainer] Epoch %3d / %d  MSE = %.8f\r", ep, max_epochs, mse);
        fflush(stdout);
        /* Early stopping: stop if improvement is negligible */
        if (fabs(prev_err - mse) < 1e-9) break;
        prev_err = mse;
    }
    printf("\n");

    free(state);
}

/* ════════════════════════════════════════════════════════════════════
 * Section 6 — Self-play data generation
 * ════════════════════════════════════════════════════════════════════ */

#define INF 30000

static int qsearch_nnue(Board *b, int alpha, int beta, int depth) {
    int stand_pat = nnue_eval(&g_nnue, b);
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
    for (int i = 0; i < ml.count; i++) {
        if (MOVE_IS_CAP(ml.moves[i])) {
            Move t = ml.moves[0]; ml.moves[0] = ml.moves[i]; ml.moves[i] = t;
        }
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
 * Play one self-play game and append all non-opening positions to ds.
 * game_result is stored white-relative (1.0 / 0.5 / 0.0).
 */
static void play_game(PosDataset *ds, int search_depth, unsigned int *seed) {
#define MAX_GAME_PLY 500
    Board b;
    board_start_pos(&b);

    Board *history = malloc(MAX_GAME_PLY * sizeof(Board));
    if (!history) return;
    int ply = 0;

    /* Randomise the first 4 half-moves for opening variety */
    for (int i = 0; i < 4; i++) {
        MoveList ml; gen_moves(&b, &ml);
        int legal[256], n = 0;
        for (int j = 0; j < ml.count; j++)
            if (is_legal(&b, ml.moves[j])) legal[n++] = j;
        if (n == 0) { free(history); return; }
        make_move(&b, ml.moves[legal[tune_rand(seed) % n]]);
    }

    double result = 0.5;
    while (ply < MAX_GAME_PLY) {
        Move best = NULL_MOVE;
        int  score = alphabeta_nnue(&b, search_depth, -INF, INF, &best);

        if (best == NULL_MOVE) {
            result = in_check(&b) ? ((b.side == WHITE) ? 0.0 : 1.0) : 0.5;
            break;
        }
        if (b.halfmove >= 100) { result = 0.5; break; }

        history[ply++] = b;
        make_move(&b, best);

        /* Adjudicate: avoid playing out obvious resignations */
        if (score >  1000) { result = (b.side == WHITE) ? 0.0 : 1.0; break; }
        if (score < -1000) { result = (b.side == WHITE) ? 1.0 : 0.0; break; }
    }

    /* Append positions to dataset (skip the randomised opening ply = 0..3) */
    for (int i = 0; i < ply; i++) {
        if (ds->count >= 500000) break;
        ds->positions[ds->count].board  = history[i];
        ds->positions[ds->count].result = result;
        ds->count++;
    }
    free(history);
#undef MAX_GAME_PLY
}

/* ── Parallel self-play ──────────────────────────────────────────── */
typedef struct {
    int          num_games;
    int          depth;
    unsigned int seed;
    PosDataset  *out_ds;  /* private per-thread, allocated by thread */
} SelfPlayCtx;

static void *self_play_thread_fn(void *arg) {
    SelfPlayCtx *ctx = (SelfPlayCtx *)arg;
    ctx->out_ds = create_dataset();
    for (int i = 0; i < ctx->num_games; i++)
        play_game(ctx->out_ds, ctx->depth, &ctx->seed);
    return NULL;
}

/*
 * Generate num_games self-play games and merge all positions into *ds.
 * Uses g_num_threads worker threads.
 */
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

    pthread_t   threads[MAX_TUNE_THREADS];
    SelfPlayCtx ctxs   [MAX_TUNE_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 32 * 1024 * 1024); /* 32 MB for qsearch */

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
        if (ds->count + to_copy > 500000) to_copy = 500000 - ds->count;
        if (to_copy > 0) {
            memcpy(ds->positions + ds->count, src->positions,
                   (size_t)to_copy * sizeof(TunePos));
            ds->count += to_copy;
        }
        free_dataset(src);
    }

    printf("[Self-play] Done — %d positions from %d threads.\n", ds->count, nt);
}

/* ════════════════════════════════════════════════════════════════════
 * Section 7 — Main reinforcement-learning loop
 * ════════════════════════════════════════════════════════════════════ */

/*
 * Outer loop: alternate between self-play data generation and NNUE training.
 *
 *   iterations     : number of generate→train cycles
 *   games_per_iter : self-play games per cycle (scale up for distributed)
 *   search_depth   : minimax depth used during self-play
 *   num_threads    : worker threads (shared by self-play and gradient comp.)
 *   batch_size     : Adam minibatch size
 *   lr             : Adam learning rate
 *   train_epochs   : Adam epochs per cycle
 *
 * Checkpoint: after each cycle, weights are saved to "nnue_checkpoint.bin".
 */
void tune_self_play_loop(int iterations, int games_per_iter, int search_depth,
                         int num_threads, int batch_size, float lr,
                         int train_epochs) {
    if (num_threads < 1)               num_threads = 1;
    if (num_threads > MAX_TUNE_THREADS) num_threads = MAX_TUNE_THREADS;
    g_num_threads = num_threads;

    printf("[Tuner] %d iterations, %d games/iter, depth %d, %d thread(s)\n",
           iterations, games_per_iter, search_depth, g_num_threads);
    printf("[Tuner] Adam: lr=%.1e  batch=%d  epochs/iter=%d\n",
           (double)lr, batch_size, train_epochs);

    /* Try to resume from a checkpoint; fall back to random init. */
    if (nnue_load(&g_nnue, "nnue_checkpoint.bin") != 0) {
        printf("[Tuner] No checkpoint found — random initialisation.\n");
        nnue_init_random(&g_nnue);
    }

    for (int iter = 1; iter <= iterations; iter++) {
        printf("\n══════════════════════════════════════════════════════\n");
        printf(" RL Iteration %d / %d\n", iter, iterations);
        printf("══════════════════════════════════════════════════════\n");

        /* ── Step 1: Data generation ─────────────────────────────────
         * DISTRIBUTED: in a multi-node setup, dispatch this call to each
         * worker node.  Workers run self_play_worker() independently and
         * upload their position files.  The master then loads and merges
         * them before Step 2. */
        PosDataset *ds = create_dataset();
        self_play_worker(games_per_iter, search_depth, ds);

        if (ds->count == 0) {
            printf("[Tuner] Warning: empty dataset; skipping training.\n");
            free_dataset(ds);
            continue;
        }

        /* ── Step 2: Training ────────────────────────────────────────
         * DISTRIBUTED: the master node loads all worker datasets here,
         * merges them into a single PosDataset, then calls train_on_dataset.
         * dist_allreduce_gradients() inside compute_gradients_mt handles
         * the cross-node gradient synchronisation. */
        train_on_dataset(&g_nnue, ds, train_epochs, batch_size, lr, g_num_threads);

        free_dataset(ds);

        /* ── Checkpoint ──────────────────────────────────────────────
         * DISTRIBUTED: after nnue_save(), broadcast "nnue_checkpoint.bin"
         * to all worker nodes (e.g. via rsync, S3 push, or MPI_Bcast). */
        printf("[Master] Checkpointing...\n");
        if (nnue_save(&g_nnue, "nnue_checkpoint.bin") == 0)
            printf("[Master] Saved to nnue_checkpoint.bin\n");
        else
            fprintf(stderr, "[Master] Checkpoint save failed!\n");
    }

    printf("[Tuner] Training complete.\n");
}

/* ════════════════════════════════════════════════════════════════════
 * Section 8 — Entry point
 * ════════════════════════════════════════════════════════════════════ */

#ifdef TUNE_STANDALONE
int main(int argc, char **argv) {
    bitboard_init();
    board_init();

    /* Defaults */
    int   iterations     = 50;
    int   games_per_iter = 1000;  /* scale up heavily for distributed */
    int   search_depth   = 3;    /* keep low; rely on eval quality   */
    int   num_threads    = 12;
    int   batch_size     = 1024;
    float lr             = ADAM_LR;
    int   train_epochs   = 1000;

    /* Optional positional args: threads games_per_iter iters depth */
    if (argc > 1) num_threads    = atoi(argv[1]);
    if (argc > 2) games_per_iter = atoi(argv[2]);
    if (argc > 3) iterations     = atoi(argv[3]);
    if (argc > 4) search_depth   = atoi(argv[4]);
    if (argc > 5) batch_size     = atoi(argv[5]);
    if (argc > 6) train_epochs   = atoi(argv[6]);

    printf("NNUE Self-Play Reinforcement Tuner\n");
    printf("  Architecture: %d → %d → %d → 1  (%d params)\n",
           NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_TOTAL_PARAMS);
    printf("  iters=%d  games/iter=%d  depth=%d  threads=%d\n",
           iterations, games_per_iter, search_depth, num_threads);

    tune_self_play_loop(iterations, games_per_iter, search_depth,
                        num_threads, batch_size, lr, train_epochs);
    return 0;
}
#endif