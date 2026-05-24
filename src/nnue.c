/*
 * nnue.c — HalfKAv2_hm NNUE forward pass and weight I/O
 *
 * All activations use SCReLU: f(x) = clamp(x, 0, 1)².
 *
 * Architecture overview:
 *   FT    : 45,056 → 1,024  (per side, bias + active feature rows)
 *   L1    : 2,048  →    32  (SCReLU, STM‖OPP concatenation as input)
 *   L2    :    32  →    32  (SCReLU)
 *   output:    32  →     1  (linear, × NNUE_OUTPUT_SCALE → centipawns)
 *
 * King moves invalidate the relevant accumulator half — the caller must
 * set acc->dirty[pov] = true or call nnue_accum_refresh() explicitly.
 *
 * Thread safety: g_nnue is read-only during search (many concurrent readers).
 * Writes occur only in the training loop while all search threads are quiesced.
 */

#include "nnue.h"
#include "bitboard.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Verify that NNUENet is a flat, padding-free array of floats.
 * If this fires the tuner's flat-pointer iteration is broken.
 */
_Static_assert(sizeof(NNUENet) == NNUE_TOTAL_PARAMS * sizeof(float),
               "NNUENet must be a contiguous float array with no padding");

/* ── Global network instance ─────────────────────────────────────── */
NNUENet *g_nnue = NULL;

/* ── Memory management ───────────────────────────────────────────── */
NNUENet *nnue_alloc(void) {
    NNUENet *net = calloc(1, sizeof(NNUENet));
    if (!net)
        fprintf(stderr, "nnue_alloc: failed to allocate %.1f MB\n",
                (double)sizeof(NNUENet) / (1024.0 * 1024.0));
    return net;
}

void nnue_free(NNUENet *net) { free(net); }

/* ── Activation functions ────────────────────────────────────────── */

/*
 * CReLU: clamp(x, 0, 1).
 * Used only as a building block for SCReLU.
 */
static inline float crelu(float x) {
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}

/*
 * SCReLU: squared clipped ReLU — Stockfish's hidden-layer activation.
 * f(x) = clamp(x, 0, 1)²
 *
 * Properties:
 *   • Range [0, 1], gradient = 2·clamp(x,0,1) for x ∈ (0,1), else 0.
 *   • Smooth at x=1 (gradient approaches 2), sharp at x=0 (gradient → 0⁺).
 *   • Compared to CReLU: stronger gradient signal for well-activated neurons;
 *     implicit L2-like pressure keeps weights from saturating at 1.
 */
static inline float screlu(float x) {
    float c = crelu(x);
    return c * c;
}

/* ── Helper: find king square for one side ───────────────────────── */
/*
 * The king bitboard is always a single set bit.  bb_pop() on a copy
 * returns that square without modifying the board state.
 */
static inline int king_square(const Board *b, int color) {
    Bitboard kb = b->pieces[color][5]; /* KING = 5 */
    return bb_pop(&kb);
}

/* ── Feature enumeration (HalfKAv2_hm) ──────────────────────────── */
int nnue_get_features(const Board *b, int perspective, int *features) {
    int n    = 0;
    int ksq  = king_square(b, perspective);

    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int idx = nnue_feature_idx(perspective, ksq, c, pt, sq);
                if (idx >= 0) features[n++] = idx;  /* –1 = own king, skip */
            }
        }
    }
    return n;
}

/* ── Accumulator refresh ─────────────────────────────────────────── */
void nnue_accum_refresh(const NNUENet *net, const Board *b, NNUEAccum *acc) {
    if (!net || !b || !acc)
        return;

    for (int pov = 0; pov < 2; pov++) {
        int ksq = king_square(b, pov);
        acc->king_sq[pov] = ksq;

        /* Initialise from bias vector */
        memcpy(acc->acc[pov], net->ft_bias, NNUE_L1 * sizeof(float));

        /* Sum in the FT row for each active feature */
        int features[32];
        int n = nnue_get_features(b, pov, features);
        for (int i = 0; i < n; i++) {
            const float *row = net->ft_weight[features[i]];
            for (int k = 0; k < NNUE_L1; k++)
                acc->acc[pov][k] += row[k];
        }
        acc->dirty[pov] = false;
    }
}

/* ── Incremental accumulator update ─────────────────────────────── */
void nnue_accum_update(const NNUENet *net, NNUEAccum *acc, int pov,
                       const int *add_feats, int n_add,
                       const int *rem_feats, int n_rem) {
    float *a = acc->acc[pov];
    for (int i = 0; i < n_add; i++) {
        const float *row = net->ft_weight[add_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) a[k] += row[k];
    }
    for (int i = 0; i < n_rem; i++) {
        const float *row = net->ft_weight[rem_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) a[k] -= row[k];
    }
    acc->dirty[pov] = false;
}

/* ── Forward pass (L1 + L2 + output) from accumulator ───────────── */
int nnue_eval_from_accum(const NNUENet *net, const NNUEAccum *acc, int stm) {
    /* Defensive checks to avoid null-pointer dereference reported by sanitizer */
    if (net == NULL || acc == NULL) return 0;
    /*
     * Step 1: SCReLU the two accumulator halves and concatenate.
     *   x[0..L1)      = SCReLU(acc[stm])
     *   x[L1..2·L1)   = SCReLU(acc[stm^1])
     */
    float x[2 * NNUE_L1];
    const float *acc_stm = acc->acc[stm];
    const float *acc_opp = acc->acc[stm ^ 1];
    for (int k = 0; k < NNUE_L1; k++) {
        x[k]           = screlu(acc_stm[k]);
        x[NNUE_L1 + k] = screlu(acc_opp[k]);
    }

    /* Step 2: Hidden layer 1 — (2·L1) → L2, SCReLU */
    float h1[NNUE_L2];
    for (int j = 0; j < NNUE_L2; j++) {
        float s = net->l1_bias[j];
        for (int k = 0; k < 2 * NNUE_L1; k++)
            s += net->l1_weight[k][j] * x[k];
        h1[j] = screlu(s);
    }

    /* Step 3: Hidden layer 2 — L2 → L3, SCReLU */
    float h2[NNUE_L3];
    for (int j = 0; j < NNUE_L3; j++) {
        float s = net->l2_bias[j];
        for (int k = 0; k < NNUE_L2; k++)
            s += net->l2_weight[k][j] * h1[k];
        h2[j] = screlu(s);
    }

    /* Step 4: Output layer — L3 → 1, linear, scale to centipawns */
    float out = net->l3_bias;
    for (int j = 0; j < NNUE_L3; j++)
        out += net->l3_weight[j] * h2[j];

    return (int)(out * NNUE_OUTPUT_SCALE);
}

/* ── High-level evaluation ───────────────────────────────────────── */
int nnue_eval(const NNUENet *net, const Board *b) {
    NNUEAccum acc;
    nnue_accum_refresh(net, b, &acc);
    return nnue_eval_from_accum(net, &acc, b->side);
}

/* ── Random initialisation (Kaiming-uniform) ─────────────────────── */
static float rand_uniform(unsigned int *seed, float lo, float hi) {
    *seed = *seed * 1664525u + 1013904223u;
    float t = (float)((*seed >> 1) & 0x7fffffffu) * (1.0f / 2147483648.0f);
    return lo + t * (hi - lo);
}

void nnue_init_random(NNUENet *net) {
    unsigned int seed = (unsigned int)time(NULL);
    float a;

    /* Feature transformer: Kaiming-uniform, fan_in = NNUE_FT_IN */
    a = sqrtf(6.0f / NNUE_FT_IN);
    for (int i = 0; i < NNUE_FT_IN; i++)
        for (int k = 0; k < NNUE_L1; k++)
            net->ft_weight[i][k] = rand_uniform(&seed, -a, a);
    memset(net->ft_bias, 0, sizeof net->ft_bias);

    /* Hidden layer 1: Kaiming-uniform, fan_in = 2·NNUE_L1 */
    a = sqrtf(6.0f / (2 * NNUE_L1));
    for (int k = 0; k < 2 * NNUE_L1; k++)
        for (int j = 0; j < NNUE_L2; j++)
            net->l1_weight[k][j] = rand_uniform(&seed, -a, a);
    memset(net->l1_bias, 0, sizeof net->l1_bias);

    /* Hidden layer 2: Kaiming-uniform, fan_in = NNUE_L2 */
    a = sqrtf(6.0f / NNUE_L2);
    for (int k = 0; k < NNUE_L2; k++)
        for (int j = 0; j < NNUE_L3; j++)
            net->l2_weight[k][j] = rand_uniform(&seed, -a, a);
    memset(net->l2_bias, 0, sizeof net->l2_bias);

    /*
     * Output layer: small uniform initialisation so the network emits ≈ 0 cp
     * for all positions at the start of training, avoiding early instability.
     * ±(1/L3) gives a max raw output of ≈ ±1 before the output scale.
     */
    a = 1.0f / NNUE_L3;
    for (int j = 0; j < NNUE_L3; j++)
        net->l3_weight[j] = rand_uniform(&seed, -a, a);
    net->l3_bias = 0.0f;
}

/* ── Binary I/O ──────────────────────────────────────────────────── */
#define NNUE_MAGIC    0x4E4E5545u  /* "NNUE" */
#define NNUE_VERSION  2u           /* v2: HalfKAv2_hm + L3 + SCReLU        */

int nnue_save(const NNUENet *net, const char *path) {
    /* Write to a temp file first, then rename — atomic on POSIX */
    char tmp[256];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { perror("nnue_save: fopen"); return -1; }

    /* Header: magic, version, FT_IN, L1, L2, L3 */
    uint32_t hdr[6] = { NNUE_MAGIC, NNUE_VERSION,
                        NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_L3 };
    if (fwrite(hdr, sizeof hdr, 1, f) != 1) goto err;

#define WF(arr, n) \
    if (fwrite((arr), sizeof(float), (n), f) != (size_t)(n)) goto err
    WF(net->ft_weight,  NNUE_FT_IN * NNUE_L1);
    WF(net->ft_bias,    NNUE_L1);
    WF(net->l1_weight,  2 * NNUE_L1 * NNUE_L2);
    WF(net->l1_bias,    NNUE_L2);
    WF(net->l2_weight,  NNUE_L2 * NNUE_L3);
    WF(net->l2_bias,    NNUE_L3);
    WF(net->l3_weight,  NNUE_L3);
    WF(&net->l3_bias,   1);
#undef WF

    if (fflush(f) || fclose(f)) { perror("nnue_save: fclose"); remove(tmp); return -1; }
    if (rename(tmp, path))       { perror("nnue_save: rename"); remove(tmp); return -1; }
    return 0;
err:
    perror("nnue_save: write");
    fclose(f); remove(tmp);
    return -1;
}

int nnue_load(NNUENet *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("nnue_load: fopen"); return -1; }

    uint32_t hdr[6];
    if (fread(hdr, sizeof hdr, 1, f) != 1) goto err;
    if (hdr[0] != NNUE_MAGIC || hdr[1] != NNUE_VERSION ||
        hdr[2] != NNUE_FT_IN || hdr[3] != NNUE_L1 ||
        hdr[4] != NNUE_L2    || hdr[5] != NNUE_L3) {
        fprintf(stderr,
                "nnue_load: architecture mismatch in '%s'\n"
                "  expected FT=%d L1=%d L2=%d L3=%d (version %u)\n"
                "  file has  FT=%u L1=%u L2=%u L3=%u (version %u)\n",
                path, NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_L3, NNUE_VERSION,
                hdr[2], hdr[3], hdr[4], hdr[5], hdr[1]);
        fclose(f); return -1;
    }

#define RF(arr, n) \
    if (fread((arr), sizeof(float), (n), f) != (size_t)(n)) goto err
    RF(net->ft_weight,  NNUE_FT_IN * NNUE_L1);
    RF(net->ft_bias,    NNUE_L1);
    RF(net->l1_weight,  2 * NNUE_L1 * NNUE_L2);
    RF(net->l1_bias,    NNUE_L2);
    RF(net->l2_weight,  NNUE_L2 * NNUE_L3);
    RF(net->l2_bias,    NNUE_L3);
    RF(net->l3_weight,  NNUE_L3);
    RF(&net->l3_bias,   1);
#undef RF

    fclose(f);
    return 0;
err:
    perror("nnue_load: read");
    fclose(f); return -1;
}