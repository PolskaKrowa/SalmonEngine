#pragma once
/*
 * nnue.h — HalfKAv2_hm NNUE, Stockfish-compatible architecture
 *
 * Architecture matches Stockfish SF15/SF16:
 *
 *   Feature transformer : NNUE_FT_IN (45,056) → NNUE_L1 (1024)  [per side]
 *   Hidden layer 1      : 2·NNUE_L1  (2048)   → NNUE_L2  (32)   [SCReLU]
 *   Hidden layer 2      : NNUE_L2    (32)      → NNUE_L3  (32)   [SCReLU]
 *   Output              : NNUE_L3    (32)      → 1               [linear]
 *
 * Feature set: HalfKAv2_hm
 * ─────────────────────────
 *   index = king_sq * 704 + piece_idx * 64 + piece_sq
 *
 *   704 = 11 piece-types × 64 squares
 *   11  = {own P, N, B, R, Q}   (5 own, king excluded)
 *       + {enemy P, N, B, R, Q, K}  (6 enemy, including king)
 *
 *   Horizontal mirroring (the "hm"):
 *     When the perspective king is on the right half of the board (file ≥ 4),
 *     both the king square and every piece square are mirrored (col ^= 7) so
 *     that the king always indexes into files A–D (0–3).  This halves the
 *     effective king-bucket count from 64 to 32 without reducing table size.
 *
 *   Perspective flip (black's view):
 *     The board is flipped vertically (sq ^= 56) and colours swapped, so the
 *     network always sees "own piece" vs "enemy piece" regardless of which
 *     side is actually White.
 *
 * Activation: SCReLU — squared clipped ReLU
 * ──────────────────────────────────────────
 *   f(x) = clamp(x, 0, 1)²
 *   Matches Stockfish's hidden-layer activation from SF15 onward.
 *   The feature transformer still feeds into SCReLU before concatenation.
 *
 * King moves
 * ──────────
 *   Because every feature is conditioned on the perspective king's square,
 *   any king move (including castling) invalidates that side's accumulator.
 *   The make-move logic must set acc->dirty[pov] = true before calling
 *   nnue_accum_update; or call nnue_accum_refresh directly.
 *   acc->king_sq[pov] is kept in sync by nnue_accum_refresh() and can be
 *   compared against b->pieces[pov][5] to detect stale accumulators.
 *
 * Memory
 * ──────
 *   NNUENet is ~185 MB as float32.  NEVER place on the stack.
 *   Always allocate with nnue_alloc() and free with nnue_free().
 *   NNUEGrad (alias of NNUENet) is equally large; compute_gradients_mt()
 *   allocates one per worker thread on the heap.
 *
 * Quantisation path (future production build)
 * ────────────────────────────────────────────
 *   Replace float32 with int16_t, scale ft_weight by QA=255, others by
 *   QB=64.  The SCReLU output range is [0, QA²] for the FT and [0, QB²]
 *   for hidden layers.  Provide nnue_simd.c with the same API signatures.
 *
 * Binary file format: custom float32 (see nnue_save / nnue_load, version 2).
 *   Not byte-compatible with Stockfish's int8/int16-quantised .nnue files;
 *   a quantisation pass is required for production use.
 */

#include "board.h"
#include <stdbool.h>
#include <stdint.h>

/* ── Network dimensions ──────────────────────────────────────────── */

/*
 * HalfKAv2_hm input size:
 *   64 king-squares × 11 piece-types × 64 squares = 45,056
 */
#define NNUE_FT_IN   45056  /* feature-transformer input width              */
#define NNUE_L1       1024  /* accumulator width per side                   */
#define NNUE_L2         32  /* hidden layer 1 width                         */
#define NNUE_L3         32  /* hidden layer 2 width                         */

/*
 * Output scale: the network's raw float output is multiplied by this to
 * yield centipawns.  400 matches the standard logistic WDL scale where
 * 100 cp ≈ 1 pawn advantage and sigmoid(400/400) ≈ 0.731 win probability.
 */
#define NNUE_OUTPUT_SCALE  400

/*
 * Total parameter count (compile-time constant for Adam state allocation
 * and flat-array iteration):
 *
 *   45056·1024 + 1024 + 2·1024·32 + 32 + 32·32 + 32 + 32 + 1
 *   = 46,205,025
 */
#define NNUE_TOTAL_PARAMS                                        \
    ( NNUE_FT_IN * NNUE_L1   + NNUE_L1                         \
    + 2*NNUE_L1  * NNUE_L2   + NNUE_L2                         \
    + NNUE_L2    * NNUE_L3   + NNUE_L3                         \
    + NNUE_L3 + 1 )

/* ── Float32 network parameters ──────────────────────────────────── */
/*
 * Memory layout is a flat, padding-free sequence of float arrays
 * (verified by the _Static_assert in nnue.c).  The tuner uses this to
 * iterate every weight as a single float[NNUE_TOTAL_PARAMS] array via
 * the (float *)net cast.
 *
 * Due to the ~185 MB size, NNUENet must ALWAYS be heap-allocated.
 * Use nnue_alloc() / nnue_free().
 */
typedef struct {
    float ft_weight[NNUE_FT_IN][NNUE_L1];   /* feature-transformer weights */
    float ft_bias  [NNUE_L1];               /* feature-transformer biases  */
    float l1_weight[2*NNUE_L1][NNUE_L2];    /* hidden-layer-1 weights      */
    float l1_bias  [NNUE_L2];               /* hidden-layer-1 biases       */
    float l2_weight[NNUE_L2][NNUE_L3];      /* hidden-layer-2 weights      */
    float l2_bias  [NNUE_L3];               /* hidden-layer-2 biases       */
    float l3_weight[NNUE_L3];               /* output-layer weights        */
    float l3_bias;                          /* output-layer bias           */
} NNUENet;

/*
 * Gradient buffer — identical layout to NNUENet.
 * Tuner worker threads accumulate into private NNUEGrad structs; the
 * master thread sums them via add_grad() then applies the Adam step.
 */
typedef NNUENet NNUEGrad;

/* ── Per-position incremental accumulator ────────────────────────── */
typedef struct {
    float acc    [2][NNUE_L1]; /* [0] = white POV, [1] = black POV         */
    bool  dirty  [2];          /* true → full refresh needed before eval    */
    int   king_sq[2];          /* cached king squares; used to detect moves */
} NNUEAccum;

/* ── Global network instance ─────────────────────────────────────── */
/*
 * Pointer — not a value — because NNUENet is ~185 MB.
 * Initialise with:   g_nnue = nnue_alloc();
 * All other calls pass g_nnue (not &g_nnue) since it is already a pointer.
 */
extern NNUENet *g_nnue;

/* ── Feature indexing (HalfKAv2_hm) ─────────────────────────────── */
/*
 * Returns the feature index in [0, NNUE_FT_IN) for a single piece as
 * seen from 'perspective'.
 *
 *   perspective  : 0 = white, 1 = black
 *   king_sq      : the perspective side's king square (0–63, A1=0)
 *   piece_color  : 0 = white piece, 1 = black piece
 *   piece_type   : 0=P  1=N  2=B  3=R  4=Q  5=K  (board.h PAWN…KING)
 *   square       : piece square (0–63, little-endian)
 *
 * Returns –1 if the piece should be skipped (the perspective's own king).
 *
 * Transformation pipeline:
 *   1. Perspective flip  (black only): sq ^= 56, piece_color ^= 1
 *   2. Horizontal mirror (when king's file ≥ 4): king_sq ^= 7, sq ^= 7
 *   3. Piece index mapping:
 *        own   P/N/B/R/Q → 0–4   (own king skipped → return -1)
 *        enemy P/N/B/R/Q/K → 5–10
 *   4. index = king_sq * 704 + piece_idx * 64 + sq
 *      max   = 63 * 704 + 10 * 64 + 63 = 45,055  ✓
 */
static inline int nnue_feature_idx(int perspective, int king_sq,
                                   int piece_color, int piece_type,
                                   int square) {
    /* Step 1: vertical flip + colour swap for black's perspective */
    if (perspective == 1) {
        king_sq     ^= 56;
        square      ^= 56;
        piece_color ^=  1;
    }

    /* Step 2: horizontal mirror when king is on files E–H */
    if ((king_sq & 7) >= 4) {
        king_sq ^= 7;
        square  ^= 7;
    }

    /* Step 3: piece index (own king is never a feature → return –1) */
    int piece_idx;
    if (piece_color == 0) {           /* own piece */
        if (piece_type == 5) return -1;   /* skip own king */
        piece_idx = piece_type;           /* 0–4 */
    } else {                          /* enemy piece (all 6 types included) */
        piece_idx = 5 + piece_type;       /* 5–10 */
    }

    return king_sq * (11 * 64) + piece_idx * 64 + square;
}

/* ── Memory management ───────────────────────────────────────────── */

/*
 * Allocate and zero-initialise an NNUENet on the heap (~185 MB).
 * Returns NULL on allocation failure (prints a message to stderr).
 */
NNUENet *nnue_alloc(void);

/*
 * Free a network allocated by nnue_alloc().  Safe to call with NULL.
 */
void nnue_free(NNUENet *net);

/* ── API ─────────────────────────────────────────────────────────── */

/*
 * Enumerate active HalfKAv2_hm feature indices for position b from the
 * given perspective.  features[] must hold at least 32 entries
 * (max 31 non-own-king pieces, but 32 gives one spare slot).
 * Returns the number of active features.
 */
int nnue_get_features(const Board *b, int perspective, int *features);

/*
 * Kaiming-uniform random initialisation.
 * Call once at startup when no saved weights are available.
 */
void nnue_init_random(NNUENet *net);

/*
 * Save / load a binary weight file.
 * The 6-word header stores the magic number, version, and all four
 * dimension constants so that a mismatched architecture is caught
 * immediately on load.
 * Returns 0 on success, –1 on failure.
 */
int nnue_save(const NNUENet *net, const char *path);
int nnue_load(NNUENet *net, const char *path);

/*
 * Fully recompute both accumulator halves for position b.
 * Also caches king squares in acc->king_sq[].
 * Call when switching to a position with no prior accumulated history,
 * or after any king move.
 */
void nnue_accum_refresh(const NNUENet *net, const Board *b, NNUEAccum *acc);

/*
 * Incrementally update one half of acc after a make/unmake move.
 *
 *   pov      : perspective to update (0 = white, 1 = black)
 *   add_feats / n_add : features to add   (newly placed / revealed pieces)
 *   rem_feats / n_rem : features to remove (captured / moving pieces)
 *
 * IMPORTANT: Do NOT call for king moves.  Instead set acc->dirty[pov] = true
 * so that nnue_eval_from_accum triggers a lazy refresh, or call
 * nnue_accum_refresh() directly.  King moves invalidate all features
 * because every index is conditioned on the king square.
 *
 * Cost: O(changed_features · NNUE_L1) vs O(32 · NNUE_L1) for a full refresh.
 */
void nnue_accum_update(const NNUENet *net, NNUEAccum *acc, int pov,
                       const int *add_feats, int n_add,
                       const int *rem_feats, int n_rem);

/*
 * Forward pass (L1 → L2 → L3 → output) from a pre-populated accumulator.
 *   stm : side to move — WHITE(0) or BLACK(1).
 * Returns centipawns from the side-to-move's perspective.
 */
int nnue_eval_from_accum(const NNUENet *net, const NNUEAccum *acc, int stm);

/*
 * Full evaluation: refresh the accumulator then run the forward pass.
 * Returns centipawns from the side-to-move's perspective.
 * Use for positions where no incremental accumulator is maintained.
 */
int nnue_eval(const NNUENet *net, const Board *b);

/* ── Incremental accumulator stack ──────────────────────────────────
 *
 * The search maintains a stack of NNUEAccum entries, one per ply, so that
 * make_move can compute the child's accumulator from the parent's via a
 * small delta (add/rem feature lists) instead of a full ~60K-FADD refresh.
 * unmake_move is free — the parent's accumulator is still on the stack.
 *
 * King moves (including castling) invalidate the moving side's perspective
 * and trigger a full refresh of that half only; the opponent's perspective
 * is still incrementally updatable because the king's *own* square doesn't
 * appear in the opponent's feature indices (it appears as an enemy piece,
 * which is just a normal feature).
 *
 * Call sequence:
 *   nnue_acc_init()                  — once at program startup
 *   nnue_acc_reset(b)                — at the start of every search()
 *   nnue_acc_make_move(b, m, mover)  — inside make_move, AFTER b is updated
 *   nnue_acc_unmake_move(b)          — inside unmake_move (no-op besides accounting)
 *   nnue_acc_make_null_move(b)       — inside make_null_move
 *   nnue_acc_unmake_null_move(b)     — inside unmake_null_move
 *   nnue_eval_positional(b)          — replaces nnue_eval() in the search path
 */
void nnue_acc_init(void);
void nnue_acc_free(void);
void nnue_acc_reset(const Board *b);
void nnue_acc_deactivate(void);
void nnue_acc_make_move(const Board *b, Move m, Color mover);
void nnue_acc_unmake_move(const Board *b);
void nnue_acc_make_null_move(const Board *b);
void nnue_acc_unmake_null_move(const Board *b);

/*
 * Evaluate the current position using the incremental accumulator stack.
 * Equivalent to nnue_eval() but skips the FT refresh (which is maintained
 * incrementally by nnue_acc_make_move). Returns centipawns from STM's POV.
 */
int nnue_eval_positional(const Board *b);