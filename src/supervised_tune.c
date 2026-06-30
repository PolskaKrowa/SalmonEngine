/*
 * supervised_tune.c — Supervised NNUE fine-tuning from a CSV dataset
 *
 * Training objective: pairwise margin ranking loss.
 *
 * For each example (FEN position P, annotated best move m*):
 *   • Apply m*  → position P*  (opponent to move, expected to be losing)
 *   • Sample N_NEG random legal alternatives m₁..mN → P₁..PN
 *   • Evaluate each with the NNUE:
 *       o*  = nnue_raw(P*)       — should be LOW  (opponent is losing)
 *       oᵢ  = nnue_raw(Pᵢ)      — should be HIGH (opponent not losing)
 *   • Loss = (1/N) Σᵢ  max(0,  o* − oᵢ + MARGIN)
 *
 * Gradients:
 *   Let k = number of i where the margin is violated (loss > 0).
 *   ∂L/∂o*  =  k / N              (push o* down)
 *   ∂L/∂oᵢ  = −1/N  if violated  (push oᵢ up)
 *
 * Optimiser: Adam with cosine LR schedule and global grad-norm clipping.
 *
 * ── Engine API assumptions ────────────────────────────────────────────────
 *
 *   This file links against the Salmon engine sources (bitboard.c, board.c,
 *   movegen.c, nnue.c) and OpenBLAS.  It relies on three engine functions
 *   whose signatures are assumed — adjust names in §2 if they differ:
 *
 *     board_from_fen(Board *b, const char *fen)
 *         Initialise *b from the given FEN string.
 *
 *     move_from_uci(const Board *b, const char *uci, Move *out)
 *         Parse a UCI move string (e.g. "e2e4", "e7e8q") into *out.
 *         Returns 1 on success, 0 if the move string is invalid or
 *         illegal in position *b.
 *         NOTE: This function is called by the UCI input handler in uci.c;
 *         it almost certainly exists — check uci.c or board.c for the
 *         exact name and adjust SUP_BOARD_FROM_FEN / SUP_MOVE_FROM_UCI
 *         below if needed.
 *
 * ── Build ─────────────────────────────────────────────────────────────────
 *
 *   Added to Makefile.am under the ENABLE_TUNER conditional:
 *     suptuna target (see patch at the bottom of this file)
 *
 *   Compile with -DSUPERVISED_STANDALONE for the standalone binary.
 *
 * ── CLI ───────────────────────────────────────────────────────────────────
 *
 *   suptuna --csv <path>  [options]
 *     --csv      path     input CSV (fen,best_move columns required)
 *     --ckpt-in  path     load initial weights                [none]
 *     --ckpt-out path     checkpoint output path              [nnue_sup.bin]
 *     --epochs   n        training epochs                     [50]
 *     --batch    n        minibatch size                      [64]
 *     --lr       f        peak Adam learning rate             [1e-3]
 *     --lr-min   f        cosine-schedule floor LR            [1e-5]
 *     --n-neg    n        random alternatives per example     [7]
 *     --margin   f        hinge margin in raw NNUE units      [0.05]
 *     --threads  n        parallel gradient threads           [1]
 *     --seed     n        PRNG seed                           [42]
 *     --log-every n       print progress every n steps       [50]
 */

#include "nnue.h"
#include "bitboard.h"
#include "board.h"
#include "movegen.h"
#include "uci.h"

#include <openblas/cblas.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * §1  Engine API shims
 *     Adjust these names to match your board.h / uci.c signatures.
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * Name of the FEN-loading function in board.h / board.c.
 */
#define SUP_BOARD_FROM_FEN(b, fen)  board_from_fen((b), (fen))

#define SUP_MOVE_FROM_UCI(b, uci) str_to_move((b), (uci))

/* ══════════════════════════════════════════════════════════════════════════
 * §2  Field limits / defaults
 * ══════════════════════════════════════════════════════════════════════════ */

#define SUP_MAX_CSV_ROWS  100000
#define SUP_FEN_LEN       128
#define SUP_MOVE_LEN      8

#define SUP_ADAM_B1     0.9f
#define SUP_ADAM_B2     0.999f
#define SUP_ADAM_EPS    1e-8f
#define SUP_GRAD_CLIP   1.0f

/* Default hyper-parameters; overridden by CLI args */
static int          g_epochs    = 50;
static int          g_batch     = 64;
static float        g_lr_max    = 1e-3f;
static float        g_lr_min    = 1e-5f;
static int          g_n_neg     = 7;
static float        g_margin    = 0.05f;   /* raw NNUE units ≈ margin*NNUE_OUTPUT_SCALE cp */
static int          g_threads   = 1;
static unsigned int g_seed      = 42u;
static int          g_log_every = 50;
static const char  *g_csv_path  = NULL;
static const char  *g_ckpt_in   = NULL;
static const char  *g_ckpt_out  = "nnue_sup.bin";

/* ══════════════════════════════════════════════════════════════════════════
 * §3  CSV dataset
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char fen      [SUP_FEN_LEN];
    char best_move[SUP_MOVE_LEN];
} SupExample;

typedef struct {
    SupExample *rows;
    int         count;
    int         cap;
} SupDataset;

/*
 * Trim leading/trailing whitespace in-place.
 * Returns a pointer to the first non-space character.
 */
static char *sup_trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    char *end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' ||
                       end[-1] == '\r' || end[-1] == '\n'))
        *--end = '\0';
    return s;
}

/*
 * Parse one CSV line into at most max_fields fields.
 * Fields are delimited by commas; quoted fields (double-quoted) are
 * supported so labels containing commas work correctly.
 * Returns the number of fields written into out[].
 */
static int sup_csv_split(char *line, char **out, int max_fields) {
    int n = 0;
    char *p = line;
    while (*p && n < max_fields) {
        if (*p == '"') {
            /* Quoted field: advance past opening quote */
            p++;
            out[n++] = p;
            while (*p && *p != '"') p++;
            if (*p == '"') *p++ = '\0';
        } else {
            out[n++] = p;
            while (*p && *p != ',') p++;
        }
        if (*p == ',') *p++ = '\0';
    }
    return n;
}

/*
 * Load a CSV file into a SupDataset.
 * The first row must be a header containing "fen" and "best_move" columns.
 * Extra columns (e.g. "label") are silently ignored.
 * Returns NULL on fatal error.
 */
static SupDataset *sup_csv_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("sup_csv_load: fopen"); return NULL; }

    SupDataset *ds = malloc(sizeof(SupDataset));
    if (!ds) { fclose(f); return NULL; }
    ds->cap   = 4096;
    ds->count = 0;
    ds->rows  = malloc((size_t)ds->cap * sizeof(SupExample));
    if (!ds->rows) { free(ds); fclose(f); return NULL; }

    char line[512];
    /* ── Parse header row to find column indices ── */
    if (!fgets(line, sizeof line, f)) {
        fprintf(stderr, "sup_csv_load: empty file\n");
        free(ds->rows); free(ds); fclose(f); return NULL;
    }
    char *hdr_fields[16];
    int   n_hdr = sup_csv_split(line, hdr_fields, 16);
    int   col_fen = -1, col_move = -1;
    for (int i = 0; i < n_hdr; i++) {
        char *h = sup_trim(hdr_fields[i]);
        if (strcmp(h, "fen")       == 0) col_fen  = i;
        if (strcmp(h, "best_move") == 0) col_move = i;
    }
    if (col_fen < 0 || col_move < 0) {
        fprintf(stderr,
                "sup_csv_load: header must contain 'fen' and 'best_move' columns\n"
                "  found: ");
        for (int i = 0; i < n_hdr; i++) fprintf(stderr, "'%s' ", hdr_fields[i]);
        fprintf(stderr, "\n");
        free(ds->rows); free(ds); fclose(f); return NULL;
    }

    int row_num = 1;
    while (fgets(line, sizeof line, f)) {
        row_num++;
        /* Grow buffer if needed */
        if (ds->count >= ds->cap) {
            int new_cap = ds->cap * 2;
            if (new_cap > SUP_MAX_CSV_ROWS) new_cap = SUP_MAX_CSV_ROWS;
            if (ds->count >= new_cap) {
                fprintf(stderr,
                        "sup_csv_load: row limit %d reached; truncating\n",
                        SUP_MAX_CSV_ROWS);
                break;
            }
            SupExample *tmp = realloc(ds->rows,
                                      (size_t)new_cap * sizeof(SupExample));
            if (!tmp) {
                fprintf(stderr, "sup_csv_load: realloc failed at row %d\n", row_num);
                break;
            }
            ds->rows = tmp;
            ds->cap  = new_cap;
        }
        /* Split the row */
        char *fields[16];
        int n_f = sup_csv_split(line, fields, 16);
        int max_needed = (col_fen > col_move ? col_fen : col_move) + 1;
        if (n_f < max_needed) {
            fprintf(stderr, "sup_csv_load: row %d has too few fields (%d); skipping\n",
                    row_num, n_f);
            continue;
        }

        SupExample *ex = &ds->rows[ds->count];
        char *fen  = sup_trim(fields[col_fen]);
        char *move = sup_trim(fields[col_move]);

        if (strlen(fen)  == 0) continue;
        if (strlen(move) == 0) continue;

        snprintf(ex->fen,       SUP_FEN_LEN,  "%s", fen);
        snprintf(ex->best_move, SUP_MOVE_LEN, "%s", move);
        ds->count++;
    }

    fclose(f);
    fprintf(stderr, "[CSV] Loaded %d examples from '%s'\n", ds->count, path);
    return ds;
}

static void sup_csv_free(SupDataset *ds) {
    if (ds) { free(ds->rows); free(ds); }
}

/* Fisher-Yates shuffle of the dataset */
static void sup_shuffle(SupDataset *ds, unsigned int *seed) {
    for (int i = ds->count - 1; i > 0; i--) {
        *seed = *seed * 1664525u + 1013904223u;
        int j = (int)((*seed >> 1) % (unsigned int)(i + 1));
        SupExample tmp = ds->rows[i];
        ds->rows[i]    = ds->rows[j];
        ds->rows[j]    = tmp;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §4  Forward pass with cached activations (for backprop)
 *
 *     Copied from tune.c (where it is static).  Only the name is changed
 *     (sup_forward) and the cblas layout comments are preserved verbatim.
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float pre_stm[NNUE_L1];
    float pre_opp[NNUE_L1];
    float x[2 * NNUE_L1];      /* SCReLU(pre_stm) ‖ SCReLU(pre_opp) */
    float pre_h1[NNUE_L2];
    float h1    [NNUE_L2];
    float pre_h2[NNUE_L3];
    float h2    [NNUE_L3];
    float out;

    int stm_feats[32];
    int opp_feats[32];
    int n_stm, n_opp;
    int stm;
} SupFwdCache;

static inline float sup_screlu(float x) {
    float c = x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
    return c * c;
}

static inline float sup_dscrelu(float z) {
    return (z > 0.0f && z < 1.0f) ? 2.0f * z : 0.0f;
}

static float sup_forward(const NNUENet *net, const Board *b, SupFwdCache *c) {
    c->stm   = b->side;
    c->n_stm = nnue_get_features(b, c->stm,     c->stm_feats);
    c->n_opp = nnue_get_features(b, c->stm ^ 1, c->opp_feats);

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

    for (int k = 0; k < NNUE_L1; k++) {
        c->x[k]           = sup_screlu(c->pre_stm[k]);
        c->x[NNUE_L1 + k] = sup_screlu(c->pre_opp[k]);
    }

    /* L1: pre_h1 = l1_bias + l1_weight^T · x  (BLAS CblasTrans sgemv) */
    memcpy(c->pre_h1, net->l1_bias, NNUE_L2 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans,
                2 * NNUE_L1, NNUE_L2,
                1.0f, (const float *)net->l1_weight, NNUE_L2,
                c->x, 1,
                1.0f, c->pre_h1, 1);
    for (int j = 0; j < NNUE_L2; j++) c->h1[j] = sup_screlu(c->pre_h1[j]);

    /* L2: pre_h2 = l2_bias + l2_weight^T · h1 */
    memcpy(c->pre_h2, net->l2_bias, NNUE_L3 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans,
                NNUE_L2, NNUE_L3,
                1.0f, (const float *)net->l2_weight, NNUE_L3,
                c->h1, 1,
                1.0f, c->pre_h2, 1);
    for (int j = 0; j < NNUE_L3; j++) c->h2[j] = sup_screlu(c->pre_h2[j]);

    float out = net->l3_bias;
    for (int j = 0; j < NNUE_L3; j++) out += net->l3_weight[j] * c->h2[j];
    c->out = out;
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §5  Backward pass  (also copied from tune.c, renamed sup_backward)
 * ══════════════════════════════════════════════════════════════════════════ */

static void sup_backward(const NNUENet *net, const SupFwdCache *c,
                         float dL_d_out, NNUEGrad *grad) {
    /* Output layer */
    grad->l3_bias += dL_d_out;
    for (int j = 0; j < NNUE_L3; j++)
        grad->l3_weight[j] += dL_d_out * c->h2[j];

    /* L2 backward */
    float d_pre_h2[NNUE_L3];
    for (int j = 0; j < NNUE_L3; j++)
        d_pre_h2[j] = dL_d_out * net->l3_weight[j] * sup_dscrelu(c->pre_h2[j]);

    for (int j = 0; j < NNUE_L3; j++) grad->l2_bias[j] += d_pre_h2[j];

    cblas_sger(CblasRowMajor,
               NNUE_L2, NNUE_L3,
               1.0f, c->h1, 1, d_pre_h2, 1,
               (float *)grad->l2_weight, NNUE_L3);

    float d_h1[NNUE_L2];
    memset(d_h1, 0, sizeof d_h1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                NNUE_L2, NNUE_L3,
                1.0f, (const float *)net->l2_weight, NNUE_L3,
                d_pre_h2, 1,
                1.0f, d_h1, 1);

    /* L1 backward */
    float d_pre_h1[NNUE_L2];
    for (int j = 0; j < NNUE_L2; j++)
        d_pre_h1[j] = d_h1[j] * sup_dscrelu(c->pre_h1[j]);

    for (int j = 0; j < NNUE_L2; j++) grad->l1_bias[j] += d_pre_h1[j];

    cblas_sger(CblasRowMajor,
               2 * NNUE_L1, NNUE_L2,
               1.0f, c->x, 1, d_pre_h1, 1,
               (float *)grad->l1_weight, NNUE_L2);

    float d_x[2 * NNUE_L1];
    memset(d_x, 0, sizeof d_x);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                2 * NNUE_L1, NNUE_L2,
                1.0f, (const float *)net->l1_weight, NNUE_L2,
                d_pre_h1, 1,
                1.0f, d_x, 1);

    /* FT — STM accumulator */
    for (int k = 0; k < NNUE_L1; k++) {
        float da = d_x[k] * sup_dscrelu(c->pre_stm[k]);
        grad->ft_bias[k] += da;
        for (int i = 0; i < c->n_stm; i++)
            grad->ft_weight[c->stm_feats[i]][k] += da;
    }

    /* FT — OPP accumulator */
    for (int k = 0; k < NNUE_L1; k++) {
        float da = d_x[NNUE_L1 + k] * sup_dscrelu(c->pre_opp[k]);
        grad->ft_bias[k] += da;
        for (int i = 0; i < c->n_opp; i++)
            grad->ft_weight[c->opp_feats[i]][k] += da;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * §6  Gradient utilities  (same logic as tune.c §5, prefixed sup_)
 * ══════════════════════════════════════════════════════════════════════════ */

static inline void sup_grad_zero(NNUEGrad *g)  { memset(g, 0, sizeof *g); }

static void sup_grad_add(NNUEGrad *dst, const NNUEGrad *src) {
    float       *d = (float *)dst;
    const float *s = (const float *)src;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) d[i] += s[i];
}

static void sup_grad_scale(NNUEGrad *g, float scale) {
    float *p = (float *)g;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) p[i] *= scale;
}

static float sup_grad_norm(const NNUEGrad *g) {
    const float *p = (const float *)g;
    double acc = 0.0;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) acc += (double)p[i] * (double)p[i];
    return (float)sqrt(acc);
}

static void sup_grad_clip(NNUEGrad *g, float max_norm) {
    float norm = sup_grad_norm(g);
    if (norm > max_norm) sup_grad_scale(g, max_norm / (norm + 1e-8f));
}

/* ══════════════════════════════════════════════════════════════════════════
 * §7  Adam optimiser
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    NNUEGrad m;   /* first  moment */
    NNUEGrad v;   /* second moment */
    int      t;   /* step counter  */
} SupAdamState;

static void sup_adam_update(NNUENet *net, const NNUEGrad *grad,
                            SupAdamState *st, float lr) {
    st->t++;
    float bc1 = 1.0f - powf(SUP_ADAM_B1, (float)st->t);
    float bc2 = 1.0f - powf(SUP_ADAM_B2, (float)st->t);

    float       *w  = (float *)net;
    const float *g  = (const float *)grad;
    float       *m  = (float *)&st->m;
    float       *v  = (float *)&st->v;

    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) {
        m[i] = SUP_ADAM_B1 * m[i] + (1.0f - SUP_ADAM_B1) * g[i];
        v[i] = SUP_ADAM_B2 * v[i] + (1.0f - SUP_ADAM_B2) * g[i] * g[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + SUP_ADAM_EPS);
    }

    /* FT weight clipping: keeps accumulators bounded, matches tune.c §5 */
    float *ft = (float *)net->ft_weight;
    for (int i = 0; i < NNUE_FT_IN * NNUE_L1; i++) {
        if (ft[i] >  2.0f) ft[i] =  2.0f;
        if (ft[i] < -2.0f) ft[i] = -2.0f;
    }
}

/* Cosine LR schedule (matches tune.c) */
static float sup_cosine_lr(float lr_max, float lr_min, int step, int total) {
    if (total <= 1) return lr_max;
    float t = (float)step / (float)(total - 1);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf((float)M_PI * t));
}

/* ══════════════════════════════════════════════════════════════════════════
 * §8  Per-example gradient computation  (margin ranking loss)
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * Compute the margin ranking loss and accumulate gradients for one example.
 *
 * b        — position to evaluate (will be mutated then restored)
 * best_move— annotated best move
 * grad     — gradient accumulator (caller initialises and owns)
 * n_neg    — number of random alternative moves to sample
 * margin   — hinge margin (raw NNUE units; e.g. 0.05 ≈ 20 cp if scale=400)
 * rng      — per-thread PRNG state (modified in place)
 *
 * Returns the loss contribution from this example (≥ 0).
 */
static double sup_example_gradient(const NNUENet *net, Board *b,
                                    Move best_move,
                                    NNUEGrad *grad,
                                    int n_neg, float margin,
                                    unsigned int *rng) {
    /* ── Evaluate the best-move resulting position ─────────────────────── */
    SupFwdCache cache_best;
    make_move(b, best_move);
    float o_best = sup_forward(net, b, &cache_best);
    unmake_move(b);

    /* ── Sample random alternative moves from the current position ─────── */
    MoveList ml;
    gen_moves(b, &ml);

    /*
     * Build an array of legal moves that differ from best_move.
     * We use a two-pass approach: first count legals, then sample.
     */
    Move legal_alts[256];
    int  n_legal = 0;
    for (int i = 0; i < ml.count && n_legal < 256; i++) {
        if (ml.moves[i] == best_move) continue;
        if (!is_legal(b, ml.moves[i])) continue;
        legal_alts[n_legal++] = ml.moves[i];
    }
    if (n_legal == 0) {
        /* No alternatives available — skip this example */
        return 0.0;
    }

    int n_sample = n_neg < n_legal ? n_neg : n_legal;

    /* Fisher-Yates to select n_sample without replacement */
    for (int i = 0; i < n_sample; i++) {
        *rng = *rng * 1664525u + 1013904223u;
        int j = i + (int)((*rng >> 1) % (unsigned)(n_legal - i));
        Move tmp       = legal_alts[i];
        legal_alts[i]  = legal_alts[j];
        legal_alts[j]  = tmp;
    }

    /* ── Forward pass for each alternative ─────────────────────────────── */
    SupFwdCache *cache_alts = malloc((size_t)n_sample * sizeof(SupFwdCache));
    if (!cache_alts) return 0.0;

    float *o_alts = malloc((size_t)n_sample * sizeof(float));
    if (!o_alts) { free(cache_alts); return 0.0; }

    for (int i = 0; i < n_sample; i++) {
        make_move(b, legal_alts[i]);
        o_alts[i] = sup_forward(net, b, &cache_alts[i]);
        unmake_move(b);
    }

    /* ── Compute margin loss and gradients ──────────────────────────────── */
    /*
     * L = (1/N) Σᵢ  max(0,  o* − oᵢ + margin)
     *
     * We want o* < oᵢ (best-move result is BAD for the opponent).
     * When o* − oᵢ + margin > 0, the constraint is violated:
     *   ∂L/∂o*   += 1/N   (push o* down)
     *   ∂L/∂oᵢ  -= 1/N   (push oᵢ up)
     */
    double loss        = 0.0;
    int    n_violated  = 0;
    int   *violated    = malloc((size_t)n_sample * sizeof(int));
    if (!violated) { free(o_alts); free(cache_alts); return 0.0; }

    for (int i = 0; i < n_sample; i++) {
        float hinge = o_best - o_alts[i] + margin;
        if (hinge > 0.0f) {
            loss        += (double)hinge;
            violated[n_violated++] = i;
        }
    }
    loss /= (double)n_sample;

    if (n_violated > 0) {
        float dL_best = (float)n_violated / (float)n_sample;
        sup_backward(net, &cache_best, dL_best, grad);

        float dL_alt = -1.0f / (float)n_sample;
        for (int vi = 0; vi < n_violated; vi++) {
            int i = violated[vi];
            sup_backward(net, &cache_alts[i], dL_alt, grad);
        }
    }

    free(violated);
    free(o_alts);
    free(cache_alts);
    return loss;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §9  Parallel gradient worker
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const NNUENet  *net;
    const SupExample *examples;
    int              start, end;
    NNUEGrad        *grad;
    double           partial_loss;
    int              n_processed;   /* Bug 2 fix: track examples that weren't skipped */
    int              n_neg;
    float            margin;
    unsigned int     rng;
} SupGradWorker;

static void *sup_grad_worker_fn(void *arg) {
    SupGradWorker *w = (SupGradWorker *)arg;
    double loss = 0.0;
    int    n_processed = 0;
    int    n_skip_fen = 0, n_skip_move = 0, n_skip_legal = 0;
    Board b;

    for (int i = w->start; i < w->end; i++) {
        if (SUP_BOARD_FROM_FEN(&b, w->examples[i].fen) == 0) {
            /* board_from_fen returns non-zero on success in this engine */
            if (++n_skip_fen <= 3)
                fprintf(stderr, "[SKIP][thread] board_from_fen failed: '%s'\n",
                        w->examples[i].fen);
            continue;
        }
        Move best = SUP_MOVE_FROM_UCI(&b, w->examples[i].best_move);
        if (!best) {
            if (++n_skip_move <= 3)
                fprintf(stderr, "[SKIP][thread] str_to_move failed: move='%s' fen='%s'\n",
                        w->examples[i].best_move, w->examples[i].fen);
            continue;
        }
        if (!is_legal(&b, best)) {
            if (++n_skip_legal <= 3)
                fprintf(stderr, "[SKIP][thread] is_legal rejected: move='%s' fen='%s'\n",
                        w->examples[i].best_move, w->examples[i].fen);
            continue;
        }

        n_processed++;
        loss += sup_example_gradient(w->net, &b, best,
                                      w->grad, w->n_neg, w->margin, &w->rng);
    }

    /* Bug 2 fix: report skip counts once per thread if anything was dropped */
    int n_total = w->end - w->start;
    if (n_skip_fen + n_skip_move + n_skip_legal > 0)
        fprintf(stderr,
                "[BATCH][thread] processed=%d/%d  "
                "skip_fen=%d  skip_move=%d  skip_legal=%d\n",
                n_processed, n_total,
                n_skip_fen, n_skip_move, n_skip_legal);

    w->partial_loss  = loss;
    w->n_processed   = n_processed;
    return NULL;
}

/*
 * Accumulate gradients for examples[0..count) in parallel across nthreads.
 * Returns the mean loss over the batch.
 */
static double sup_batch_gradient(const NNUENet *net,
                                  const SupExample *examples, int count,
                                  NNUEGrad *grad_out,
                                  int nthreads, int n_neg, float margin,
                                  unsigned int *rng) {
    if (count <= 0) return 0.0;
    if (nthreads > count) nthreads = count;
    if (nthreads < 1)     nthreads = 1;

    sup_grad_zero(grad_out);

    if (nthreads == 1) {
        /* Fast path: no thread overhead */
        double loss = 0.0;
        int    n_processed = 0;
        int    n_skip_fen = 0, n_skip_move = 0, n_skip_legal = 0;
        Board b;
        for (int i = 0; i < count; i++) {
            if (SUP_BOARD_FROM_FEN(&b, examples[i].fen) == 0) {
                /* board_from_fen returns non-zero on success in this engine;
                 * flip to == 0 to catch genuine failures. */
                if (++n_skip_fen <= 3)
                    fprintf(stderr,
                            "[SKIP] board_from_fen failed: '%s'\n",
                            examples[i].fen);
                continue;
            }
            Move best = SUP_MOVE_FROM_UCI(&b, examples[i].best_move);
            if (!best) {
                if (++n_skip_move <= 3)
                    fprintf(stderr,
                            "[SKIP] str_to_move failed: move='%s' fen='%s'\n",
                            examples[i].best_move, examples[i].fen);
                continue;
            }
            if (!is_legal(&b, best)) {
                if (++n_skip_legal <= 3)
                    fprintf(stderr,
                            "[SKIP] is_legal rejected: move='%s' fen='%s'\n",
                            examples[i].best_move, examples[i].fen);
                continue;
            }
            n_processed++;
            loss += sup_example_gradient(net, &b, best,
                                          grad_out, n_neg, margin, rng);
        }
        /* Bug 3 fix: report skip summary and normalise by n_processed, not
         * count, so the loss isn't diluted by silently dropped examples. */
        if (n_skip_fen + n_skip_move + n_skip_legal > 0)
            fprintf(stderr,
                    "[BATCH] processed=%d/%d  "
                    "skip_fen=%d  skip_move=%d  skip_legal=%d\n",
                    n_processed, count,
                    n_skip_fen, n_skip_move, n_skip_legal);
        return n_processed > 0 ? loss / (double)n_processed : 0.0;
    }

    /* Multi-threaded path */
    NNUEGrad     **tgrads  = malloc((size_t)nthreads * sizeof(NNUEGrad *));
    SupGradWorker *ctxs    = malloc((size_t)nthreads * sizeof(SupGradWorker));
    pthread_t     *threads = malloc((size_t)nthreads * sizeof(pthread_t));
    if (!tgrads || !ctxs || !threads) {
        perror("sup_batch_gradient: malloc"); exit(1);
    }

    int slice = count / nthreads;
    for (int t = 0; t < nthreads; t++) {
        tgrads[t] = calloc(1, sizeof(NNUEGrad));
        if (!tgrads[t]) { perror("calloc"); exit(1); }
        ctxs[t].net          = net;
        ctxs[t].examples     = examples;
        ctxs[t].start        = t * slice;
        ctxs[t].end          = (t == nthreads - 1) ? count : (t + 1) * slice;
        ctxs[t].grad         = tgrads[t];
        ctxs[t].partial_loss = 0.0;
        ctxs[t].n_processed  = 0;
        ctxs[t].n_neg        = n_neg;
        ctxs[t].margin       = margin;
        /* Give each thread a distinct PRNG stream */
        *rng = *rng * 1664525u + 1013904223u;
        ctxs[t].rng = *rng ^ (unsigned int)(t * 2654435761u);
        pthread_create(&threads[t], NULL, sup_grad_worker_fn, &ctxs[t]);
    }

    double total_loss      = 0.0;
    int    total_processed = 0;
    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        sup_grad_add(grad_out, tgrads[t]);
        total_loss      += ctxs[t].partial_loss;
        total_processed += ctxs[t].n_processed;
        free(tgrads[t]);
    }
    free(threads); free(ctxs); free(tgrads);

    /* Bug 3 fix: normalise by examples that were actually processed */
    return total_processed > 0 ? total_loss / (double)total_processed : 0.0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * §10  Training loop
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * Run the full supervised fine-tuning procedure.
 *
 * Emits machine-readable progress lines to stdout in the format:
 *   PROGRESS step=N epoch=E loss=F lr=F
 *   EPOCH_DONE epoch=E avg_loss=F
 *   DONE total_steps=N final_loss=F
 *
 * These are parsed by supervised_tune.py.
 */
void sup_train_loop(NNUENet *net, SupDataset *ds,
                    int epochs, int batch_size,
                    float lr_max, float lr_min,
                    int n_neg, float margin,
                    int nthreads, unsigned int seed,
                    int log_every) {
    int total_steps = 0;
    double last_loss = 0.0;

    SupAdamState *adam = calloc(1, sizeof(SupAdamState));
    NNUEGrad     *grad = malloc(sizeof(NNUEGrad));
    if (!adam || !grad) { perror("sup_train_loop: alloc"); return; }

    int total_epochs_steps = epochs * ((ds->count + batch_size - 1) / batch_size);

    fprintf(stderr,
            "[Supervised] dataset=%d  epochs=%d  batch=%d  lr=%.1e→%.1e\n"
            "[Supervised] n_neg=%d  margin=%.3f  threads=%d  seed=%u\n",
            ds->count, epochs, batch_size,
            (double)lr_max, (double)lr_min,
            n_neg, (double)margin, nthreads, seed);

    for (int ep = 1; ep <= epochs; ep++) {
        sup_shuffle(ds, &seed);

        double epoch_loss = 0.0;
        int    n_batches  = 0;

        float lr = sup_cosine_lr(lr_max, lr_min, ep - 1, epochs);

        int n_steps = (ds->count + batch_size - 1) / batch_size;
        for (int s = 0; s < n_steps; s++) {
            int batch_start = s * batch_size;
            int batch_end   = batch_start + batch_size;
            if (batch_end > ds->count) batch_end = ds->count;
            int cur_batch   = batch_end - batch_start;

            double loss = sup_batch_gradient(net,
                              ds->rows + batch_start, cur_batch,
                              grad, nthreads, n_neg, margin, &seed);

            /* Mean gradient, then clip */
            sup_grad_scale(grad, 1.0f / (float)cur_batch);
            sup_grad_clip(grad, SUP_GRAD_CLIP);

            sup_adam_update(net, grad, adam, lr);

            epoch_loss += loss;
            n_batches++;
            total_steps++;

            if (total_steps % log_every == 0) {
                printf("PROGRESS step=%d epoch=%d loss=%.8f lr=%.6e\n",
                       total_steps, ep, loss, (double)lr);
                fflush(stdout);
            }
        }

        double avg = n_batches > 0 ? epoch_loss / n_batches : 0.0;
        last_loss  = avg;
        printf("EPOCH_DONE epoch=%d avg_loss=%.8f\n", ep, avg);
        fflush(stdout);

        fprintf(stderr, "[Supervised] Epoch %3d/%d  avg_loss=%.8f  lr=%.3e\n",
                ep, epochs, avg, (double)lr);

        /* Save checkpoint after every epoch */
        if (nnue_save(net, g_ckpt_out) == 0)
            fprintf(stderr, "[Supervised] Checkpoint saved → %s\n", g_ckpt_out);

        (void)total_epochs_steps; /* suppress warning */
    }

    printf("DONE total_steps=%d final_loss=%.8f\n", total_steps, last_loss);
    fflush(stdout);

    free(adam);
    free(grad);
}

/* ══════════════════════════════════════════════════════════════════════════
 * §11  Entry point
 * ══════════════════════════════════════════════════════════════════════════ */

#ifdef SUPERVISED_STANDALONE

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --csv <path> [options]\n\n"
        "Options:\n"
        "  --csv      <path>   input CSV (fen,best_move columns required)\n"
        "  --ckpt-in  <path>   load initial weights from this file\n"
        "  --ckpt-out <path>   save weights here [nnue_sup.bin]\n"
        "  --epochs   <n>      training epochs [50]\n"
        "  --batch    <n>      minibatch size [64]\n"
        "  --lr       <f>      peak Adam LR [1e-3]\n"
        "  --lr-min   <f>      cosine-schedule floor LR [1e-5]\n"
        "  --n-neg    <n>      random alternatives per example [7]\n"
        "  --margin   <f>      hinge margin in raw NNUE units [0.05]\n"
        "  --threads  <n>      gradient worker threads [1]\n"
        "  --seed     <n>      PRNG seed [42]\n"
        "  --log-every <n>     print progress every n steps [50]\n",
        prog);
}

int main(int argc, char **argv) {
    /* ── Parse CLI ──────────────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
#define NEEDS_ARG \
    if (i + 1 >= argc) { \
        fprintf(stderr, "error: %s requires an argument\n", argv[i]); \
        return 1; \
    }
        if      (!strcmp(argv[i], "--csv"))       { NEEDS_ARG; g_csv_path  = argv[++i]; }
        else if (!strcmp(argv[i], "--ckpt-in"))   { NEEDS_ARG; g_ckpt_in   = argv[++i]; }
        else if (!strcmp(argv[i], "--ckpt-out"))  { NEEDS_ARG; g_ckpt_out  = argv[++i]; }
        else if (!strcmp(argv[i], "--epochs"))    { NEEDS_ARG; g_epochs    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--batch"))     { NEEDS_ARG; g_batch     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--lr"))        { NEEDS_ARG; g_lr_max    = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--lr-min"))    { NEEDS_ARG; g_lr_min    = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--n-neg"))     { NEEDS_ARG; g_n_neg     = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--margin"))    { NEEDS_ARG; g_margin    = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--threads"))   { NEEDS_ARG; g_threads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--seed"))      { NEEDS_ARG; g_seed      = (unsigned)atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--log-every")) { NEEDS_ARG; g_log_every = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
#undef NEEDS_ARG
    }

    if (!g_csv_path) {
        fprintf(stderr, "error: --csv is required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Engine initialisation ──────────────────────────────────────────── */
    bitboard_init();
    board_init();

    g_nnue = nnue_alloc();
    if (!g_nnue) {
        fprintf(stderr, "fatal: cannot allocate NNUENet (~%.0f MB)\n",
                (double)sizeof(NNUENet) / (1024.0 * 1024.0));
        return 1;
    }

    if (g_ckpt_in) {
        if (nnue_load(g_nnue, g_ckpt_in) == 0) {
            fprintf(stderr, "[Supervised] Loaded weights from '%s'\n", g_ckpt_in);
        } else {
            fprintf(stderr, "[Supervised] Could not load '%s' — using random init\n",
                    g_ckpt_in);
            nnue_init_random(g_nnue);
        }
    } else {
        fprintf(stderr, "[Supervised] No --ckpt-in; random weight initialisation\n");
        nnue_init_random(g_nnue);
    }

    /* ── Load dataset ───────────────────────────────────────────────────── */
    SupDataset *ds = sup_csv_load(g_csv_path);
    if (!ds || ds->count == 0) {
        fprintf(stderr, "fatal: empty or unreadable CSV '%s'\n", g_csv_path);
        nnue_free(g_nnue);
        return 1;
    }

    fprintf(stderr,
            "\nSalmon NNUE — Supervised Fine-Tuner\n"
            "  Architecture : FT %d→%d  L1→L2 %d→%d  output %d→1\n"
            "  Parameters   : %d\n"
            "  Output scale : %d\n"
            "  Margin (cp)  : ~%.0f\n\n",
            NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_L3,
            NNUE_L3, NNUE_TOTAL_PARAMS, NNUE_OUTPUT_SCALE,
            (double)g_margin * NNUE_OUTPUT_SCALE);

    /* ── Train ──────────────────────────────────────────────────────────── */
    sup_train_loop(g_nnue, ds,
                   g_epochs, g_batch,
                   g_lr_max, g_lr_min,
                   g_n_neg, g_margin,
                   g_threads, g_seed,
                   g_log_every);

    sup_csv_free(ds);
    nnue_free(g_nnue);
    g_nnue = NULL;
    return 0;
}

#endif /* SUPERVISED_STANDALONE */