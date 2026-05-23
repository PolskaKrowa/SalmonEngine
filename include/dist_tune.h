/*
 * dist_tune.h — Distributed NNUE training interface
 *
 * ════════════════════════════════════════════════════════════════════════
 * Architecture
 * ════════════════════════════════════════════════════════════════════════
 *
 *   Data-parallel AllReduce (OpenMPI):
 *
 *     ┌──────────────────────────────────────────────────────────┐
 *     │  Iteration N                                              │
 *     │                                                           │
 *     │  Each rank:                                               │
 *     │    1. Self-play  local_games (= total / world_size)       │
 *     │    2. Push positions → local replay buffer                │
 *     │    3. For each minibatch:                                 │
 *     │         a. Compute local gradients  (pthreads)            │
 *     │         b. MPI_Allreduce  → global mean gradient          │
 *     │         c. Adam step                                      │
 *     │    4. Rank 0: save checkpoint with HMAC-SHA256            │
 *     │    5. MPI_Bcast: broadcast updated weights to all ranks   │
 *     └──────────────────────────────────────────────────────────┘
 *
 *   Because every rank runs the same Adam step on the same mean gradient
 *   (AllReduce result), all rank-local network copies remain identical
 *   after each step — no additional broadcast is needed mid-epoch.
 *   The broadcast at step 5 covers the checkpoint round-trip.
 *
 * ════════════════════════════════════════════════════════════════════════
 * External libraries
 * ════════════════════════════════════════════════════════════════════════
 *
 *   OpenMPI  4.x  (mpi.h)          —  gradient allreduce, weight broadcast
 *   OpenSSL  3.x  (openssl/evp.h)  —  HMAC-SHA256 checkpoint integrity
 *
 * ════════════════════════════════════════════════════════════════════════
 * Quick-start integration
 * ════════════════════════════════════════════════════════════════════════
 *
 *   In tune.c:
 *     1. Add  #include "tune_internal.h"
 *     2. Remove `static` from the functions listed in tune_internal.h §3.
 *     3. Remove the existing no-op dist_allreduce_gradients() — dist_tune.c
 *        provides the real implementation with the same signature.
 *
 *   In your main translation unit:
 *     DistCtx *ctx = dist_init(&argc, &argv);
 *     // ... derive or load an HMAC key ...
 *     dist_tune_loop(ctx, hmac_key, ...);
 *     dist_teardown(ctx);
 *
 *   Build:
 *     mpicc -O3 tune.c dist_tune.c dist_tune_main.c \
 *           -lopenblas -lssl -lcrypto -lm -lpthread -o dist_tuner
 *     mpirun -np 4 ./dist_tuner [options]
 */

#pragma once

#include "tune_internal.h"   /* NNUENet, NNUEGrad, NNUE_TOTAL_PARAMS, types */
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Opaque context
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct DistCtx DistCtx;

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Checkpoint security
 * ═══════════════════════════════════════════════════════════════════════════ */

/* HMAC-SHA256 key length (bytes).  256-bit key — do not reduce. */
#define DIST_HMAC_KEY_LEN  32

/* HMAC-SHA256 tag length appended to every checkpoint file (bytes). */
#define DIST_HMAC_LEN      32

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Initialise MPI with MPI_THREAD_FUNNELED support.
 * Must be the first dist_ call.  Pass the same argc/argv you received in
 * main(); MPI may consume its own flags from the argument list.
 *
 * Returns NULL on failure (error printed to stderr).
 *
 * Side-effect: sets the module-level g_dist pointer so that
 * dist_allreduce_gradients() (called from within tune.c) can reach the
 * MPI context without a parameter change.
 */
DistCtx *dist_init(int *argc, char ***argv);

/*
 * Flush all pending MPI operations, call MPI_Finalize, and free the context.
 * No dist_ function may be called after this.
 */
void dist_teardown(DistCtx *ctx);

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  Topology queries
 * ═══════════════════════════════════════════════════════════════════════════ */

int  dist_rank(const DistCtx *ctx);         /* 0 .. world_size-1             */
int  dist_size(const DistCtx *ctx);         /* number of MPI ranks           */
bool dist_is_master(const DistCtx *ctx);    /* true iff rank == 0            */

/*
 * How many self-play games should this rank run this iteration?
 * Distributes total_games as evenly as possible; lower ranks get +1 when
 * total_games is not divisible by world_size.
 */
int  dist_local_games(const DistCtx *ctx, int total_games);

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  Synchronisation
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Global barrier — all ranks block until all have called. */
void dist_barrier(DistCtx *ctx);

/*
 * Drop-in replacement for the no-op dist_allreduce_gradients(NNUEGrad *)
 * declared in tune.c.  Uses the module-level context set by dist_init().
 *
 * Computes the MEAN gradient across all ranks:
 *
 *   result[i] = (Σ_rank  g_rank[i]) / world_size
 *
 * Why mean, not sum?  tune.c already calls scale_grad(grad, 1/batch_size)
 * after compute_gradients_mt returns.  With W ranks each contributing a
 * batch_size-sized sum, AllReduce SUM gives W × sum_per_rank.  Dividing
 * by W here restores mean-gradient semantics, giving an effective global
 * batch size of W × batch_size with the correct gradient scale.
 *
 * Single-rank behaviour: returns immediately without any MPI call.
 */
void dist_allreduce_gradients(NNUEGrad *g);

/*
 * Broadcast the full NNUENet weight blob from rank 0 to all other ranks.
 * Call at the start of each RL iteration to guarantee all ranks start
 * from identical weights (important after a checkpoint round-trip).
 */
void dist_broadcast_net(DistCtx *ctx, NNUENet *net);

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  Checkpoint I/O with HMAC-SHA256 integrity
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Save net to path with a 32-byte HMAC-SHA256 integrity tag appended.
 *
 *   File layout:  [ NNUENet payload (sizeof NNUENet bytes) ]
 *                 [ HMAC-SHA256 tag (32 bytes)             ]
 *
 * Only rank 0 writes; a barrier synchronises non-masters so they do not
 * proceed until the file is fully written.
 *
 * key  — DIST_HMAC_KEY_LEN-byte secret, identical on all ranks.
 *         Distribute out-of-band (env var, secrets manager, etc.) — never
 *         send it over MPI.
 *
 * Returns 0 on success, -1 on I/O or crypto error (message to stderr).
 */
int dist_checkpoint_save(DistCtx *ctx, const NNUENet *net, const char *path,
                          const unsigned char key[DIST_HMAC_KEY_LEN]);

/*
 * Load net from path and verify its HMAC-SHA256 tag.
 * All ranks call this independently (shared filesystem assumed).
 *
 * Returns:
 *   0   success
 *  -1   I/O error (file missing, short read, etc.)
 *  -2   HMAC mismatch — file is corrupted or tampered; net zeroed for safety
 */
int dist_checkpoint_load(DistCtx *ctx, NNUENet *net, const char *path,
                          const unsigned char key[DIST_HMAC_KEY_LEN]);

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  Main distributed training loop
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * AlphaZero-style RL loop, distributed across MPI ranks.
 *
 * Replaces tune_self_play_loop() from tune.c.  The self-play workload is
 * split evenly across ranks; Adam steps are synchronised via AllReduce.
 * Checkpoints are saved by rank 0 and broadcast to all ranks.
 *
 * Parameters
 * ──────────
 * ctx                   distributed context from dist_init()
 * hmac_key              DIST_HMAC_KEY_LEN-byte key for checkpoint integrity;
 *                       all ranks must supply the same key
 * iterations            number of RL cycles (self-play → train → checkpoint)
 * games_per_iter        TOTAL self-play games across all ranks per iteration
 * search_depth          alpha-beta search depth for self-play
 * num_threads_per_rank  pthreads to use for gradient computation on each rank
 * batch_size            minibatch size per rank per Adam step
 * lr_max                peak learning rate (cosine schedule: lr_max → lr_min)
 * train_epochs          Adam epochs over the replay buffer per iteration
 */
void dist_tune_loop(DistCtx *ctx,
                    const unsigned char hmac_key[DIST_HMAC_KEY_LEN],
                    int iterations,
                    int games_per_iter,
                    int search_depth,
                    int num_threads_per_rank,
                    int batch_size,
                    float lr_max,
                    int train_epochs);

#ifdef __cplusplus
}
#endif