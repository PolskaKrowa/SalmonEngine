#pragma once
/*
 * nnue.h — Efficiently Updatable Neural Network Evaluation
 *
 * Architecture (HalfKA — piece-on-square, both perspectives):
 *
 *   Feature transformer : NNUE_FT_IN (768) → NNUE_L1 (256)
 *   Hidden layer 1      : 2·NNUE_L1  (512) → NNUE_L2 (32)    [CReLU]
 *   Output              : NNUE_L2    (32)  → 1                 [linear]
 *
 * Two independent accumulators are maintained — one for each side's
 * perspective — and concatenated (STM first) before the hidden layers.
 * This lets incremental make/unmake updates touch only the changed rows.
 *
 * Quantisation notes (for future production builds):
 *   • Replace float32 weights with int16_t.
 *   • Scale ft_weight by QA = 255, other layers by QB = 64.
 *   • Use SIMD pmaddubsw / dpbusd for the accumulator dot-product.
 *   • Add an nnue_simd.c that provides the same API below.
 *
 * Binary file format: see nnue_save / nnue_load.
 */

#include "board.h"
#include <stdbool.h>
#include <stdint.h>

/* ── Network dimensions ──────────────────────────────────────────── */
#define NNUE_FT_IN  768     /* 2 colours × 6 piece types × 64 squares   */
#define NNUE_L1     256     /* feature-transformer width / accumulator   */
#define NNUE_L2      32     /* hidden-layer-1 width                      */

/*
 * Total parameter count (compile-time constant, useful for Adam state
 * allocation and flat-array iteration):
 *
 *   768·256 + 256 + 512·32 + 32 + 32 + 1  =  213,313
 */
#define NNUE_TOTAL_PARAMS \
    (NNUE_FT_IN*NNUE_L1 + NNUE_L1 + 2*NNUE_L1*NNUE_L2 + NNUE_L2 + NNUE_L2 + 1)

/* ── Float32 network parameters ──────────────────────────────────── */
/*
 * Memory layout is a flat sequence of float arrays with no padding
 * (verified by the _Static_assert in nnue.c).  The tuner uses this to
 * iterate over all weights as a single float[NNUE_TOTAL_PARAMS] array.
 */
typedef struct {
    float ft_weight[NNUE_FT_IN][NNUE_L1];   /* feature transformer weights */
    float ft_bias  [NNUE_L1];               /* feature transformer biases  */
    float l1_weight[2 * NNUE_L1][NNUE_L2];  /* hidden-layer-1 weights      */
    float l1_bias  [NNUE_L2];               /* hidden-layer-1 biases       */
    float l2_weight[NNUE_L2];               /* output weights              */
    float l2_bias;                          /* output bias                 */
} NNUENet;

/*
 * Gradient buffer — same memory layout as NNUENet.
 * Tuner threads accumulate into private NNUEGrad structs; the master
 * thread sums them and applies the Adam update.
 */
typedef NNUENet NNUEGrad;

/* ── Per-position incremental accumulator ────────────────────────── */
typedef struct {
    float acc[2][NNUE_L1];  /* [0] = white POV, [1] = black POV         */
    bool  dirty[2];         /* set true when a refresh is needed         */
} NNUEAccum;

/* ── Global network (definition in nnue.c) ───────────────────────── */
extern NNUENet g_nnue;

/* ── Feature indexing ────────────────────────────────────────────── */
/*
 * Returns the 0..767 feature index for a piece, from a given perspective.
 *
 *   perspective : 0 = white, 1 = black
 *   piece_color : 0 = white, 1 = black (board.h WHITE / BLACK)
 *   piece_type  : 0..5 (board.h PAWN..KING)
 *   square      : 0..63, little-endian (A1=0, H8=63)
 *
 * Black's perspective mirrors the board vertically (square ^= 56) and
 * swaps colours, so the network sees "own piece" vs "enemy piece"
 * independently of which side is actually White.
 */
static inline int nnue_feature_idx(int perspective, int piece_color,
                                   int piece_type, int square) {
    if (perspective == 1) { square ^= 56; piece_color ^= 1; }
    return (piece_color * 6 + piece_type) * 64 + square;
}

/* ── API ─────────────────────────────────────────────────────────── */

/*
 * Enumerate active feature indices for position b from the given
 * perspective.  features[] must hold at least 32 entries.
 * Returns the number of active features (≤ 32 — max pieces on board).
 */
int nnue_get_features(const Board *b, int perspective, int *features);

/*
 * Kaiming-uniform random initialisation.
 * Call once at startup when no saved weights are available.
 */
void nnue_init_random(NNUENet *net);

/*
 * Load / save a binary weight file.
 * The header stores the magic "NNUE" and dimension constants so that
 * a load with mismatched architecture is detected immediately.
 * Returns 0 on success, –1 on failure.
 */
int nnue_load(NNUENet *net, const char *path);
int nnue_save(const NNUENet *net, const char *path);

/*
 * Fully recompute both halves of acc for position b.
 * Call when switching to a position with no accumulated history.
 */
void nnue_accum_refresh(const NNUENet *net, const Board *b, NNUEAccum *acc);

/*
 * Incrementally update one half of acc after a make/unmake move.
 *
 *   pov      : which perspective to update (0 = white, 1 = black)
 *   add_feats / n_add : features to add (newly placed / revealed pieces)
 *   rem_feats / n_rem : features to remove (captured / moving pieces)
 *
 * Incremental updates are O(changed_features·NNUE_L1) vs the O(32·NNUE_L1)
 * full refresh.  Implement make_move to call this with the delta.
 */
void nnue_accum_update(const NNUENet *net, NNUEAccum *acc, int pov,
                       const int *add_feats, int n_add,
                       const int *rem_feats, int n_rem);

/*
 * Forward pass (L1 + output) from a pre-populated accumulator.
 *   stm : side to move — WHITE(0) or BLACK(1).
 * Returns centipawns from side-to-move perspective.
 */
int nnue_eval_from_accum(const NNUENet *net, const NNUEAccum *acc, int stm);

/*
 * Full evaluation: refresh accumulator, then forward pass.
 * Returns centipawns from side-to-move perspective.
 */
int nnue_eval(const NNUENet *net, const Board *b);