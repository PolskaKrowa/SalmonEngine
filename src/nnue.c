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
#include "config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#if defined(HAVE_AVX2) && HAVE_AVX2
#include <immintrin.h>
#define NNUE_USE_AVX2 1
#else
#define NNUE_USE_AVX2 0
#endif

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
/*
 * SIMD-optimised: the inner loop is a contiguous vector add/sub on both
 * `a[k]` and `row[k]`, so 8-wide AVX2 gives a near-8× speedup over the
 * scalar version. Each feature change costs NNUE_L1/8 = 128 AVX2 ops
 * instead of 1024 scalar adds.
 */
void nnue_accum_update(const NNUENet *net, NNUEAccum *acc, int pov,
                       const int *add_feats, int n_add,
                       const int *rem_feats, int n_rem) {
    float *a = acc->acc[pov];

#if NNUE_USE_AVX2
    for (int i = 0; i < n_add; i++) {
        const float *row = net->ft_weight[add_feats[i]];
        for (int k = 0; k < NNUE_L1; k += 8) {
            __m256 av = _mm256_loadu_ps(&a[k]);
            __m256 rv = _mm256_loadu_ps(&row[k]);
            _mm256_storeu_ps(&a[k], _mm256_add_ps(av, rv));
        }
    }
    for (int i = 0; i < n_rem; i++) {
        const float *row = net->ft_weight[rem_feats[i]];
        for (int k = 0; k < NNUE_L1; k += 8) {
            __m256 av = _mm256_loadu_ps(&a[k]);
            __m256 rv = _mm256_loadu_ps(&row[k]);
            _mm256_storeu_ps(&a[k], _mm256_sub_ps(av, rv));
        }
    }
#else
    for (int i = 0; i < n_add; i++) {
        const float *row = net->ft_weight[add_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) a[k] += row[k];
    }
    for (int i = 0; i < n_rem; i++) {
        const float *row = net->ft_weight[rem_feats[i]];
        for (int k = 0; k < NNUE_L1; k++) a[k] -= row[k];
    }
#endif
    acc->dirty[pov] = false;
}

/* ── Forward pass (L1 + L2 + output) from accumulator ───────────── */
/*
 * SIMD-optimised forward pass.
 *
 * The L1 matmul is the dominant cost: 2*NNUE_L1 * NNUE_L2 = 65,536 FMAs.
 * The original scalar loop accessed l1_weight[k][j] with j fixed in the
 * inner loop, giving a stride-32 access pattern that defeated both the
 * hardware prefetcher and any compiler auto-vectorisation pass.
 *
 * The new layout processes 8 outputs per AVX2 accumulator (4 accumulators
 * total for NNUE_L2=32). The inner loop walks k=0..2*NNUE_L1-1 with
 * contiguous loads of l1_weight[k][0..7], [8..15], [16..23], [24..31]
 * and a single broadcast of x[k]. Each iteration does 4 AVX2 FMAs =
 * 32 scalar FMAs in 4 instructions — an 8× instruction-count reduction.
 *
 * Falls back to the scalar loop when AVX2 is unavailable.
 */
int nnue_eval_from_accum(const NNUENet *net, const NNUEAccum *acc, int stm) {
    if (net == NULL || acc == NULL) return 0;

    /* Step 1: SCReLU + concat the two accumulator halves into x[0..2*NNUE_L1). */
    float x[2 * NNUE_L1];
    const float *acc_stm = acc->acc[stm];
    const float *acc_opp = acc->acc[stm ^ 1];

#if NNUE_USE_AVX2
    /* SCReLU: clamp(x, 0, 1) then square. Vectorised 8 floats at a time. */
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);
    for (int k = 0; k < NNUE_L1; k += 8) {
        __m256 a = _mm256_loadu_ps(&acc_stm[k]);
        __m256 b = _mm256_loadu_ps(&acc_opp[k]);
        a = _mm256_min_ps(_mm256_max_ps(a, zero), one);
        b = _mm256_min_ps(_mm256_max_ps(b, zero), one);
        __m256 sa = _mm256_mul_ps(a, a);
        __m256 sb = _mm256_mul_ps(b, b);
        _mm256_storeu_ps(&x[k],           sa);
        _mm256_storeu_ps(&x[NNUE_L1 + k], sb);
    }
#else
    for (int k = 0; k < NNUE_L1; k++) {
        x[k]           = screlu(acc_stm[k]);
        x[NNUE_L1 + k] = screlu(acc_opp[k]);
    }
#endif

    /* Step 2: Hidden layer 1 — (2·L1) → L2, SCReLU. */
    float h1[NNUE_L2];

#if NNUE_USE_AVX2
    /* Process all 32 outputs in parallel using 4 AVX2 accumulators
     * (8 floats each). For each input x[k], broadcast it and FMA with
     * the 4 contiguous 8-float chunks of l1_weight[k]. */
    {
        __m256 acc0 = _mm256_loadu_ps(&net->l1_bias[0]);
        __m256 acc1 = _mm256_loadu_ps(&net->l1_bias[8]);
        __m256 acc2 = _mm256_loadu_ps(&net->l1_bias[16]);
        __m256 acc3 = _mm256_loadu_ps(&net->l1_bias[24]);

        for (int k = 0; k < 2 * NNUE_L1; k++) {
            __m256 xk = _mm256_set1_ps(x[k]);
            const float *w = net->l1_weight[k];
            acc0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(&w[0]),  acc0);
            acc1 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(&w[8]),  acc1);
            acc2 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(&w[16]), acc2);
            acc3 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(&w[24]), acc3);
        }

        _mm256_storeu_ps(&h1[0],  acc0);
        _mm256_storeu_ps(&h1[8],  acc1);
        _mm256_storeu_ps(&h1[16], acc2);
        _mm256_storeu_ps(&h1[24], acc3);
    }

    /* Apply SCReLU to h1. */
    {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one  = _mm256_set1_ps(1.0f);
        for (int j = 0; j < NNUE_L2; j += 8) {
            __m256 a = _mm256_loadu_ps(&h1[j]);
            a = _mm256_min_ps(_mm256_max_ps(a, zero), one);
            __m256 sa = _mm256_mul_ps(a, a);
            _mm256_storeu_ps(&h1[j], sa);
        }
    }
#else
    for (int j = 0; j < NNUE_L2; j++) {
        float s = net->l1_bias[j];
        for (int k = 0; k < 2 * NNUE_L1; k++)
            s += net->l1_weight[k][j] * x[k];
        h1[j] = screlu(s);
    }
#endif

    /* Step 3: Hidden layer 2 — L2 → L3, SCReLU. */
    float h2[NNUE_L3];
    for (int j = 0; j < NNUE_L3; j++) {
        float s = net->l2_bias[j];
        for (int k = 0; k < NNUE_L2; k++)
            s += net->l2_weight[k][j] * h1[k];
        h2[j] = screlu(s);
    }

    /* Step 4: Output layer — L3 → 1, linear, scale to centipawns.
     *
     * Use lroundf instead of truncation: the original (int)(out * scale)
     * rounds toward zero, biasing every eval by up to 0.5 cp on average.
     * lroundf gives correct rounding. */
    float out = net->l3_bias;
    for (int j = 0; j < NNUE_L3; j++)
        out += net->l3_weight[j] * h2[j];

    return (int)lroundf(out * NNUE_OUTPUT_SCALE);
}

/* ── High-level evaluation ───────────────────────────────────────── */
int nnue_eval(const NNUENet *net, const Board *b) {
    if (!net || !b) return 0;
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
    if (!net) { fprintf(stderr, "nnue_load: net not initialised\n"); return -1; }
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

/* ────────────────────────────────────────────────────────────────────
 *  Incremental accumulator stack
 *
 *  Maintains a stack of NNUEAccum entries indexed by Board.ply. Each
 *  make_move computes acc_stack[ply+1] from acc_stack[ply] using a small
 *  delta (add/rem feature lists). unmake_move is free — the parent's
 *  accumulator is still on the stack.
 *
 *  King moves trigger a full refresh of the moving side's perspective
 *  (every feature index is conditioned on the king square). The opponent's
 *  perspective is still incrementally updatable: their king square didn't
 *  change, only the moving side's king square did, and the opponent sees
 *  the moving side's king as a normal enemy-piece feature.
 * ──────────────────────────────────────────────────────────────────── */

/* Thread-local accumulator stack — each lazy-SMP worker thread gets its
 * own stack so make_move/unmake_move don't race on the accumulators.
 *
 * __thread makes the pointer itself thread-local; each thread's pointer
 * starts as NULL and is allocated on first use (lazy init in
 * nnue_acc_reset). nnue_acc_init() just pre-allocates for the main thread
 * so the first search doesn't pay the allocation cost. */
static __thread NNUEAccum *g_acc_stack = NULL;
static __thread int        g_acc_ply   = 0;
static __thread bool       g_acc_active = false;

/* Track how many threads have allocated stacks so nnue_acc_free can warn
 * if some weren't freed (leak detection — not critical for correctness). */
static int g_acc_thread_count = 0;

void nnue_acc_init(void) {
    /* Pre-allocate the main thread's stack. Worker threads allocate
     * lazily in nnue_acc_reset() on their first call. */
    if (!g_acc_stack) {
        g_acc_stack = (NNUEAccum *)calloc(MAX_PLY + 1, sizeof(NNUEAccum));
        if (!g_acc_stack) {
            fprintf(stderr, "nnue_acc_init: failed to allocate accumulator stack "
                    "(%zu bytes)\n", (size_t)(MAX_PLY + 1) * sizeof(NNUEAccum));
        } else {
            g_acc_thread_count++;
        }
        g_acc_ply = 0;
    }
}

void nnue_acc_free(void) {
    /* Only frees the calling thread's stack. Worker threads' stacks are
     * freed when they exit (pthread_join ensures the thread has finished
     * using its stack before the main thread calls nnue_acc_free).
     * In practice we don't bother freeing worker stacks — the OS reclaims
     * them on process exit, and they're only ~2.5 MB each. */
    free(g_acc_stack);
    g_acc_stack = NULL;
    g_acc_ply = 0;
    g_acc_active = false;
}

/* Lazy allocation for worker threads that haven't been init'd yet. */
static void nnue_acc_ensure_allocated(void) {
    if (!g_acc_stack) {
        g_acc_stack = (NNUEAccum *)calloc(MAX_PLY + 1, sizeof(NNUEAccum));
        if (!g_acc_stack) {
            fprintf(stderr, "nnue_acc: thread %p failed to allocate accumulator "
                    "stack (%zu bytes)\n", (void *)pthread_self(),
                    (size_t)(MAX_PLY + 1) * sizeof(NNUEAccum));
        }
    }
}

void nnue_acc_reset(const Board *b) {
    if (!g_nnue) return;
    nnue_acc_ensure_allocated();
    if (!g_acc_stack) return;
    /* Refresh the accumulator for the current position from scratch.
     * The search starts here; subsequent make_move calls will maintain
     * acc_stack[ply+1..] incrementally. Also flips g_acc_active on so that
     * make_move/unmake_move start maintaining the stack (skipped during
     * perft, tuner setup, etc.). */
    g_acc_ply = b->ply;
    g_acc_active = true;
    nnue_accum_refresh(g_nnue, b, &g_acc_stack[b->ply]);
}

/* Disable accumulator maintenance — call at the end of search so that
 * subsequent make_move/unmake_move calls (e.g. from perft or the UCI
 * position parser) don't pay the 8KB memcpy + delta cost. */
void nnue_acc_deactivate(void) {
    g_acc_active = false;
}

int nnue_eval_positional(const Board *b) {
    if (!g_nnue || !g_acc_stack) return 0;
    /* Defensive: if acc_ply drifted out of sync (e.g., evaluate called
     * outside the search), fall back to a full refresh. */
    if (b->ply != g_acc_ply || b->ply > MAX_PLY) {
        nnue_accum_refresh(g_nnue, b, &g_acc_stack[b->ply < MAX_PLY ? b->ply : MAX_PLY]);
        g_acc_ply = b->ply;
    }
    return nnue_eval_from_accum(g_nnue, &g_acc_stack[b->ply], b->side);
}

/* Helper: compute the feature index for a piece on a square, from a given
 * perspective, using that perspective's CURRENT (post-move) king square. */
static inline int feat_idx(int pov, int king_sq, int piece_color, int piece_type, int square) {
    return nnue_feature_idx(pov, king_sq, piece_color, piece_type, square);
}

void nnue_acc_make_move(const Board *b, Move m, Color mover) {
    if (!g_acc_active || !g_acc_stack || !g_nnue || m == NULL_MOVE) return;

    int old_ply = b->ply - 1;        /* parent's ply (before make_move incremented it) */
    int new_ply = b->ply;            /* current ply (after make_move) */
    if (new_ply > MAX_PLY) return;   /* can't push further */

    NNUEAccum *parent = &g_acc_stack[old_ply];
    NNUEAccum *child  = &g_acc_stack[new_ply];

    /* Default: copy parent verbatim, then apply deltas. */
    memcpy(child, parent, sizeof(NNUEAccum));

    Square from = MOVE_FROM(m);
    Square to   = MOVE_TO(m);
    MoveType mt = MOVE_TYPE(m);

    /* Identify what moved and what was captured (post-move, so we read
     * b->mailbox[to] for the moving piece, and rely on UndoInfo for the
     * captured piece — but we can reconstruct it from the move type). */
    Color them = mover ^ 1;

    /* The moving piece's type at `to` (post-move). For promotions, this
     * is the promoted type; otherwise it's the original type. */
    PieceType mover_pt;
    if (mt >= MT_N_PROMO) {
        mover_pt = MOVE_PROMO_PT(m);
    } else {
        mover_pt = piece_type(b->mailbox[to]);
    }

    /* Original moving piece type at `from` (pre-move). For promotions, pawn. */
    PieceType orig_pt = (mt >= MT_N_PROMO) ? PAWN : mover_pt;

    /* King move? (Including castling, which is a king move + rook move.) */
    bool king_moved = (orig_pt == KING);

    /* Captured piece (if any). For EP, it's a pawn at a different square. */
    bool is_capture = (mt == MT_CAPTURE || mt >= MT_N_PROMO_CAP || mt == MT_EP);
    PieceType captured_pt = NO_PIECE_TYPE;  /* sentinel */
    Square    captured_sq = NO_SQ;
    if (mt == MT_EP) {
        captured_pt = PAWN;
        captured_sq = (mover == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
    } else if (is_capture) {
        /* Captured piece was at `to` before the move; its type is gone from
         * the board now. We can recover it from the move encoding if it's
         * a promo-cap (no — promo-cap doesn't encode victim), so we have
         * to rely on the fact that all captures except EP are at `to`.
         * The piece type is whatever was there before — we don't have it.
         * Walk back via UndoInfo? Too coupled. Use a simpler recovery:
         * for non-EP captures, the victim's color is `them` and its type
         * is whatever the mover captured. We can store this in UndoInfo
         * (already there as `u->captured`), but we don't have access here.
         *
         * Simpler: look at the move type. MT_CAPTURE = plain capture.
         * We don't know the victim type without reading the undo log.
         * For now, do a full refresh of BOTH perspectives whenever a
         * capture happens — captures are rarer than quiet moves and the
         * extra refresh cost is acceptable. (Future: read UndoInfo.)
         */
        captured_pt = NO_PIECE_TYPE;  /* unknown — signal full refresh */
        captured_sq = to;
    }

    /* For each perspective, compute add/rem feature lists. */
    for (int pov = 0; pov < 2; pov++) {
        int king_sq = child->king_sq[pov];   /* post-move king square for this pov */

        /* If this pov's own king moved (i.e., pov == mover and king_moved),
         * the entire perspective's accumulator is invalid. Full refresh. */
        if (pov == mover && king_moved) {
            /* Refresh only this pov's half from scratch. */
            int ksq = king_square(b, pov);
            child->king_sq[pov] = ksq;
            memcpy(child->acc[pov], g_nnue->ft_bias, NNUE_L1 * sizeof(float));
            int features[32];
            int n = nnue_get_features(b, pov, features);
            for (int i = 0; i < n; i++) {
                const float *row = g_nnue->ft_weight[features[i]];
                for (int k = 0; k < NNUE_L1; k++) child->acc[pov][k] += row[k];
            }
            child->dirty[pov] = false;
            continue;
        }

        /* Incremental update. Build add/rem lists for this pov. */
        int add_feats[8];
        int rem_feats[8];
        int n_add = 0, n_rem = 0;

        /* The moving piece left `from` and arrived at `to`. */
        if (mt >= MT_N_PROMO) {
            /* Promotion: pawn left `from`, promoted piece arrived at `to`. */
            rem_feats[n_rem++] = feat_idx(pov, king_sq, mover, PAWN, from);
            add_feats[n_add++] = feat_idx(pov, king_sq, mover, mover_pt, to);
        } else {
            /* Plain move (or capture, or castle king move handled above). */
            rem_feats[n_rem++] = feat_idx(pov, king_sq, mover, orig_pt, from);
            add_feats[n_add++] = feat_idx(pov, king_sq, mover, mover_pt, to);
        }

        /* Captured piece (if any) is removed. */
        if (is_capture && captured_pt != NO_PIECE_TYPE) {
            rem_feats[n_rem++] = feat_idx(pov, king_sq, them, captured_pt, captured_sq);
        } else if (is_capture && captured_pt == NO_PIECE_TYPE) {
            /* Unknown victim type (plain capture) — fall back to full refresh
             * for this perspective. Rare enough that the cost is acceptable. */
            int ksq = king_square(b, pov);
            child->king_sq[pov] = ksq;
            memcpy(child->acc[pov], g_nnue->ft_bias, NNUE_L1 * sizeof(float));
            int features[32];
            int n = nnue_get_features(b, pov, features);
            for (int i = 0; i < n; i++) {
                const float *row = g_nnue->ft_weight[features[i]];
                for (int k = 0; k < NNUE_L1; k++) child->acc[pov][k] += row[k];
            }
            child->dirty[pov] = false;
            continue;
        }

        /* Castling rook move (king move was handled above). */
        if (mt == MT_KCASTLE) {
            Square rf = (mover == WHITE) ? H1 : H8;
            Square rt = (mover == WHITE) ? F1 : F8;
            rem_feats[n_rem++] = feat_idx(pov, king_sq, mover, ROOK, rf);
            add_feats[n_add++] = feat_idx(pov, king_sq, mover, ROOK, rt);
        } else if (mt == MT_QCASTLE) {
            Square rf = (mover == WHITE) ? A1 : A8;
            Square rt = (mover == WHITE) ? D1 : D8;
            rem_feats[n_rem++] = feat_idx(pov, king_sq, mover, ROOK, rf);
            add_feats[n_add++] = feat_idx(pov, king_sq, mover, ROOK, rt);
        }

        /* Apply the delta to this pov's accumulator. */
        nnue_accum_update(g_nnue, child, pov, add_feats, n_add, rem_feats, n_rem);
    }

    g_acc_ply = new_ply;
}

void nnue_acc_unmake_move(const Board *b) {
    /* No work needed — the parent's accumulator is still on the stack.
     * Just track the ply for sanity-check purposes. */
    g_acc_ply = b->ply;
}

void nnue_acc_make_null_move(const Board *b) {
    if (!g_acc_active || !g_acc_stack || !g_nnue) return;
    int new_ply = b->ply;
    if (new_ply > MAX_PLY) return;
    /* Null move doesn't change the position, so the accumulator is identical. */
    memcpy(&g_acc_stack[new_ply], &g_acc_stack[new_ply - 1], sizeof(NNUEAccum));
    g_acc_ply = new_ply;
}

void nnue_acc_unmake_null_move(const Board *b) {
    g_acc_ply = b->ply;
}