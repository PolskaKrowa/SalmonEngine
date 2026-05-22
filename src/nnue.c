/*
 * nnue.c — NNUE forward pass and weight I/O
 *
 * Float32 reference implementation.  All activation bounds are [0, 1]
 * (CReLU), matching the quantised int16 range [0, QA] so that the two
 * implementations are numerically equivalent up to rounding.
 *
 * Quantisation hook: to build a production int16 inference path, add
 * nnue_simd.c that provides the same nnue_eval* signatures and disable
 * this file's definitions with #ifdef NNUE_FLOAT_REFERENCE.
 *
 * Thread safety: g_nnue is read-only during search (many concurrent
 * readers).  Writes happen only in the training loop, which holds an
 * exclusive phase with all search threads quiesced.
 */

#include "nnue.h"
#include "bitboard.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Verify that NNUENet is a flat array of floats with no hidden padding.
 * If this fires, the tuner's flat-pointer iteration will be wrong. */
_Static_assert(sizeof(NNUENet) == NNUE_TOTAL_PARAMS * sizeof(float),
               "NNUENet must be a contiguous float array (no padding)");

/* ── Global network instance ─────────────────────────────────────── */
NNUENet g_nnue;

/* ── Activation function ─────────────────────────────────────────── */
/*
 * CReLU: clamp to [0, 1].
 * In quantised inference this maps to clamp(x, 0, QA) with int16 arithmetic.
 */
static inline float crelu(float x) {
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}

/* ── Feature enumeration ─────────────────────────────────────────── */
int nnue_get_features(const Board *b, int perspective, int *features) {
    int n = 0;
    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq = bb_pop(&bb);
                features[n++] = nnue_feature_idx(perspective, c, pt, sq);
            }
        }
    }
    return n;
}

/* ── Accumulator refresh ─────────────────────────────────────────── */
void nnue_accum_refresh(const NNUENet *net, const Board *b, NNUEAccum *acc) {
    for (int pov = 0; pov < 2; pov++) {
        /* Start from bias vector */
        memcpy(acc->acc[pov], net->ft_bias, NNUE_L1 * sizeof(float));

        int features[32];
        int n = nnue_get_features(b, pov, features);

        /* Sum in the feature-transformer row for each active feature */
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

/* ── Forward pass (L1 + output) from accumulator ────────────────── */
int nnue_eval_from_accum(const NNUENet *net, const NNUEAccum *acc, int stm) {
    /* Concatenate STM half, then OPP half — both after CReLU */
    float x[2 * NNUE_L1];
    const float *acc_stm = acc->acc[stm];
    const float *acc_opp = acc->acc[stm ^ 1];
    for (int k = 0; k < NNUE_L1; k++) {
        x[k]            = crelu(acc_stm[k]);
        x[NNUE_L1 + k]  = crelu(acc_opp[k]);
    }

    /* Hidden layer 1: (2·L1) → L2, CReLU */
    float h[NNUE_L2];
    for (int j = 0; j < NNUE_L2; j++) {
        float s = net->l1_bias[j];
        for (int k = 0; k < 2 * NNUE_L1; k++)
            s += net->l1_weight[k][j] * x[k];
        h[j] = crelu(s);
    }

    /* Output layer: L2 → 1, linear (centipawns) */
    float out = net->l2_bias;
    for (int j = 0; j < NNUE_L2; j++)
        out += net->l2_weight[j] * h[j];

    return (int)out;
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

    /* Feature transformer: Kaiming uniform, fan_in = NNUE_FT_IN */
    a = sqrtf(6.0f / NNUE_FT_IN);
    for (int i = 0; i < NNUE_FT_IN; i++)
        for (int k = 0; k < NNUE_L1; k++)
            net->ft_weight[i][k] = rand_uniform(&seed, -a, a);
    memset(net->ft_bias, 0, sizeof net->ft_bias);

    /* Hidden layer 1: Kaiming uniform, fan_in = 2·NNUE_L1 */
    a = sqrtf(6.0f / (2 * NNUE_L1));
    for (int k = 0; k < 2 * NNUE_L1; k++)
        for (int j = 0; j < NNUE_L2; j++)
            net->l1_weight[k][j] = rand_uniform(&seed, -a, a);
    memset(net->l1_bias, 0, sizeof net->l1_bias);

    /*
     * Output layer: small uniform so the network initially outputs
     * ≈ 0 cp for all positions (avoids instability at the start of
     * training before the FT has learned meaningful features).
     */
    a = 1.0f / NNUE_L2;  /* ±(1/32) — max initial output ≈ ±1 cp */
    for (int j = 0; j < NNUE_L2; j++)
        net->l2_weight[j] = rand_uniform(&seed, -a, a);
    net->l2_bias = 0.0f;
}

/* ── Binary I/O ──────────────────────────────────────────────────── */
#define NNUE_MAGIC   0x4E4E5545u  /* "NNUE" */
#define NNUE_VERSION 1u

int nnue_save(const NNUENet *net, const char *path) {
    /* Write to a temp file, then rename — atomic on POSIX */
    char tmp[256];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { perror("nnue_save: fopen"); return -1; }

    uint32_t hdr[5] = { NNUE_MAGIC, NNUE_VERSION,
                        NNUE_FT_IN, NNUE_L1, NNUE_L2 };
    if (fwrite(hdr, sizeof hdr, 1, f) != 1) goto err;

#define WF(arr, n) \
    if (fwrite((arr), sizeof(float), (n), f) != (size_t)(n)) goto err
    WF(net->ft_weight,  NNUE_FT_IN * NNUE_L1);
    WF(net->ft_bias,    NNUE_L1);
    WF(net->l1_weight,  2 * NNUE_L1 * NNUE_L2);
    WF(net->l1_bias,    NNUE_L2);
    WF(net->l2_weight,  NNUE_L2);
    WF(&net->l2_bias,   1);
#undef WF

    if (fflush(f) || fclose(f)) { perror("nnue_save: fclose"); remove(tmp); return -1; }
    if (rename(tmp, path))      { perror("nnue_save: rename"); remove(tmp); return -1; }
    return 0;
err:
    perror("nnue_save: write");
    fclose(f); remove(tmp);
    return -1;
}

int nnue_load(NNUENet *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("nnue_load: fopen"); return -1; }

    uint32_t hdr[5];
    if (fread(hdr, sizeof hdr, 1, f) != 1) goto err;
    if (hdr[0] != NNUE_MAGIC || hdr[1] != NNUE_VERSION ||
        hdr[2] != NNUE_FT_IN || hdr[3] != NNUE_L1 || hdr[4] != NNUE_L2) {
        fprintf(stderr, "nnue_load: architecture mismatch in '%s' "
                "(expected FT=%d L1=%d L2=%d, got %u %u %u)\n",
                path, NNUE_FT_IN, NNUE_L1, NNUE_L2, hdr[2], hdr[3], hdr[4]);
        fclose(f); return -1;
    }

#define RF(arr, n) \
    if (fread((arr), sizeof(float), (n), f) != (size_t)(n)) goto err
    RF(net->ft_weight,  NNUE_FT_IN * NNUE_L1);
    RF(net->ft_bias,    NNUE_L1);
    RF(net->l1_weight,  2 * NNUE_L1 * NNUE_L2);
    RF(net->l1_bias,    NNUE_L2);
    RF(net->l2_weight,  NNUE_L2);
    RF(&net->l2_bias,   1);
#undef RF

    fclose(f);
    return 0;
err:
    perror("nnue_load: read");
    fclose(f); return -1;
}