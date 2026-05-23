/*
 * dist_tune.c — Distributed NNUE training implementation
 *
 * Transport:  OpenMPI 4.x  (MPI_Allreduce, MPI_Bcast)
 * Security:   OpenSSL 3.x  EVP HMAC-SHA256 checkpoint integrity
 *
 * See dist_tune.h for the architecture overview and integration guide.
 */

#include "dist_tune.h"

#include <mpi.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

/* ── Module-level context ────────────────────────────────────────────────
 *
 * Stored here so that dist_allreduce_gradients(NNUEGrad *) can remain
 * signature-compatible with tune.c's no-op without requiring a context
 * parameter to be threaded through the call stack.
 */
static DistCtx *g_dist = NULL;

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Context struct (opaque in the header)
 * ═══════════════════════════════════════════════════════════════════════════ */

struct DistCtx {
    int      rank;
    int      size;
    MPI_Comm comm;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * §2  Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

DistCtx *dist_init(int *argc, char ***argv) {
    int provided = MPI_THREAD_SINGLE;

    int rc = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "[dist] MPI_Init_thread failed (rc=%d)\n", rc);
        return NULL;
    }

    /*
     * MPI_THREAD_FUNNELED guarantees that only the main thread calls MPI.
     * Our pthreads accumulate gradients locally; only the main thread calls
     * dist_allreduce_gradients() after pthread_join().  A lower level means
     * we still function correctly but the warning helps catch mis-use.
     */
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr,
            "[dist] Warning: MPI thread support level %d < FUNNELED (%d).\n"
            "  MPI calls must not overlap with pthread gradient workers.\n"
            "  Correctness is maintained if you call dist_allreduce_gradients()\n"
            "  only after all gradient threads have joined.\n",
            provided, MPI_THREAD_FUNNELED);
    }

    DistCtx *ctx = malloc(sizeof *ctx);
    if (!ctx) {
        MPI_Finalize();
        return NULL;
    }

    ctx->comm = MPI_COMM_WORLD;
    MPI_Comm_rank(ctx->comm, &ctx->rank);
    MPI_Comm_size(ctx->comm, &ctx->size);

    g_dist = ctx;   /* expose to dist_allreduce_gradients() */

    if (ctx->rank == 0) {
        printf("[dist] Initialised: %d rank(s), MPI thread level %d/%d\n",
               ctx->size, provided, MPI_THREAD_FUNNELED);
    }
    return ctx;
}

void dist_teardown(DistCtx *ctx) {
    if (!ctx) return;
    if (g_dist == ctx) g_dist = NULL;
    MPI_Finalize();
    free(ctx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3  Topology helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

int  dist_rank(const DistCtx *ctx)       { return ctx->rank; }
int  dist_size(const DistCtx *ctx)       { return ctx->size; }
bool dist_is_master(const DistCtx *ctx)  { return ctx->rank == 0; }

int dist_local_games(const DistCtx *ctx, int total_games) {
    /* Round-robin assignment: lower ranks absorb the remainder. */
    int base  = total_games / ctx->size;
    int extra = total_games % ctx->size;
    return base + (ctx->rank < extra ? 1 : 0);
}

void dist_barrier(DistCtx *ctx) {
    MPI_Barrier(ctx->comm);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4  Gradient AllReduce  — drop-in for tune.c's no-op
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Called from inside compute_gradients_mt() in tune.c after all gradient
 * worker threads have joined.  The gradient at this point is the SUM over
 * the local minibatch (scale_grad(1/batch_size) happens in train_one_epoch
 * after we return).
 *
 * AllReduce semantics:
 *
 *   Before: each rank holds  local_sum_i  (sum over its batch_size positions)
 *   After:  every rank holds  Σ local_sum_i   [MPI_SUM]
 *   Then we divide by world_size → mean gradient
 *
 *   Combined with tune.c's 1/batch_size scaling:
 *     final = (Σ local_sum_i / world_size) / batch_size
 *           = global_mean_gradient
 *
 *   Effective global batch = world_size × batch_size.
 */
void dist_allreduce_gradients(NNUEGrad *g) {
    if (!g_dist || g_dist->size <= 1) return;  /* single node: nothing to do */

    int rc = MPI_Allreduce(MPI_IN_PLACE,
                            (float *)g,
                            NNUE_TOTAL_PARAMS,
                            MPI_FLOAT,
                            MPI_SUM,
                            g_dist->comm);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr,
            "[dist rank %d] MPI_Allreduce failed (rc=%d) — aborting\n",
            g_dist->rank, rc);
        MPI_Abort(g_dist->comm, rc);
    }

    /* SUM → MEAN.  All ranks apply the same scale so they remain identical. */
    float  scale = 1.0f / (float)g_dist->size;
    float *p     = (float *)g;
    for (int i = 0; i < NNUE_TOTAL_PARAMS; i++) p[i] *= scale;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5  Network broadcast
 * ═══════════════════════════════════════════════════════════════════════════ */

void dist_broadcast_net(DistCtx *ctx, NNUENet *net) {
    /*
     * Broadcast raw bytes.  NNUENet is a plain-old-data struct (all floats,
     * no pointers) so a byte-level bcast is safe and layout-stable within
     * a homogeneous cluster.  If you ever run across architectures with
     * different float endianness you will need to serialise properly — but
     * that situation is vanishingly rare on modern x86/ARM clusters.
     */
    int rc = MPI_Bcast(net, (int)sizeof *net, MPI_BYTE, 0, ctx->comm);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr,
            "[dist rank %d] MPI_Bcast (net) failed (rc=%d) — aborting\n",
            ctx->rank, rc);
        MPI_Abort(ctx->comm, rc);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6  HMAC-SHA256 checkpoint I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Compute HMAC-SHA256 over `len` bytes of `data` using `key`.
 * Writes exactly DIST_HMAC_LEN (32) bytes to `out`.
 *
 * Uses the EVP API (OpenSSL 3.x preferred path; not the deprecated HMAC()).
 * Returns 0 on success, -1 on allocation or EVP failure.
 */
static int compute_hmac_sha256(const unsigned char key[DIST_HMAC_KEY_LEN],
                                const void         *data,
                                size_t              len,
                                unsigned char       out[DIST_HMAC_LEN]) {
    int ok = 0;

    EVP_MAC      *mac = EVP_MAC_fetch(NULL, "HMAC", NULL);
    EVP_MAC_CTX  *ctx = mac ? EVP_MAC_CTX_new(mac) : NULL;
    if (!ctx) goto done;

    /* Build parameter list: digest = sha256 */
    OSSL_PARAM params[] = {
        OSSL_PARAM_construct_utf8_string("digest", "SHA256", 0),
        OSSL_PARAM_construct_end()
    };

    if (EVP_MAC_init(ctx, key, DIST_HMAC_KEY_LEN, params) != 1) goto done;
    if (EVP_MAC_update(ctx, (const unsigned char *)data, len) != 1) goto done;

    size_t out_len = DIST_HMAC_LEN;
    if (EVP_MAC_final(ctx, out, &out_len, DIST_HMAC_LEN) != 1) goto done;
    if (out_len != DIST_HMAC_LEN) goto done;

    ok = 1;

done:
    EVP_MAC_CTX_free(ctx);
    EVP_MAC_free(mac);
    return ok ? 0 : -1;
}

/*
 * Constant-time comparison of two HMAC tags.
 *
 * A branch-based byte comparison leaks whether, and at which byte, two tags
 * diverge — enough for a timing oracle attack if an adversary can measure
 * load times.  XOR-accumulate avoids any early exit.
 *
 * Returns 1 (equal) or 0 (not equal).
 */
static int hmac_eq_ct(const unsigned char a[DIST_HMAC_LEN],
                       const unsigned char b[DIST_HMAC_LEN]) {
    unsigned char diff = 0;
    for (int i = 0; i < DIST_HMAC_LEN; i++) diff |= a[i] ^ b[i];
    return diff == 0;
}

int dist_checkpoint_save(DistCtx *ctx, const NNUENet *net, const char *path,
                          const unsigned char key[DIST_HMAC_KEY_LEN]) {
    int result = 0;

    if (!dist_is_master(ctx)) {
        /* Non-masters wait for rank 0 to finish before proceeding. */
        MPI_Barrier(ctx->comm);
        return 0;
    }

    /* ── Rank 0: write payload then HMAC tag ─────────────────────────── */

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[dist master] Cannot open checkpoint for writing: "
                        "%s — %s\n", path, strerror(errno));
        result = -1;
        goto barrier;
    }

    if (fwrite(net, sizeof *net, 1, f) != 1) {
        fprintf(stderr, "[dist master] Short write to checkpoint: %s\n", path);
        result = -1;
        goto close;
    }

    unsigned char tag[DIST_HMAC_LEN];
    if (compute_hmac_sha256(key, net, sizeof *net, tag) != 0) {
        fprintf(stderr, "[dist master] HMAC-SHA256 computation failed\n");
        result = -1;
        goto close;
    }

    if (fwrite(tag, DIST_HMAC_LEN, 1, f) != 1) {
        fprintf(stderr, "[dist master] Short write of HMAC tag: %s\n", path);
        result = -1;
        goto close;
    }

    /* fflush + fclose ensures the file is on disk before the barrier. */
    fflush(f);

close:
    fclose(f);
barrier:
    MPI_Barrier(ctx->comm);  /* signal non-masters that the file is ready */
    return result;
}

int dist_checkpoint_load(DistCtx *ctx, NNUENet *net, const char *path,
                          const unsigned char key[DIST_HMAC_KEY_LEN]) {
    int load_err = 0;
    unsigned char stored_tag[DIST_HMAC_LEN] = {0};

    /* Rank 0 reads, everyone else receives. */
    if (ctx->rank == 0) {
        FILE *f = fopen(path, "rb");
        if (!f) {
            load_err = -1;
        } else {
            if (fread(net, sizeof *net, 1, f) != 1)
                load_err = -1;
            else if (fread(stored_tag, DIST_HMAC_LEN, 1, f) != 1)
                load_err = -1;
            fclose(f);
        }
    }

    MPI_Bcast(&load_err, 1, MPI_INT, 0, ctx->comm);
    if (load_err != 0)
        return -1;

    MPI_Bcast(net, (int)sizeof *net, MPI_BYTE, 0, ctx->comm);
    MPI_Bcast(stored_tag, DIST_HMAC_LEN, MPI_BYTE, 0, ctx->comm);

    unsigned char computed_tag[DIST_HMAC_LEN];
    if (compute_hmac_sha256(key, net, sizeof *net, computed_tag) != 0)
        return -1;

    if (!hmac_eq_ct(stored_tag, computed_tag)) {
        fprintf(stderr,
            "[dist rank %d] HMAC-SHA256 verification FAILED for '%s'.\n"
            "  The checkpoint may be corrupted or was written with a\n"
            "  different key.  Refusing to load — net zeroed for safety.\n",
            ctx->rank, path);
        /* Zero the network so the caller can detect a bad load and reinit. */
        memset(net, 0, sizeof *net);
        return -2;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §7  Distributed RL loop
 * ═══════════════════════════════════════════════════════════════════════════ */

void dist_tune_loop(DistCtx *ctx,
                    const unsigned char hmac_key[DIST_HMAC_KEY_LEN],
                    int   iterations,
                    int   games_per_iter,
                    int   search_depth,
                    int   num_threads_per_rank,
                    int   batch_size,
                    float lr_max,
                    int   train_epochs) {

    /* ── Validate / clamp ────────────────────────────────────────────── */
    if (num_threads_per_rank < 1) num_threads_per_rank = 1;
    g_num_threads = num_threads_per_rank;

    float lr_min = lr_max * 0.01f;

    /* ── Rank-0 banner ───────────────────────────────────────────────── */
    if (dist_is_master(ctx)) {
        printf("[dist] ════════════════════════════════════════════════\n");
        printf("[dist] Distributed NNUE RL Tuner — %d rank(s)\n", ctx->size);
        printf("[dist] Effective batch per step: %d × %d = %d\n",
               ctx->size, batch_size, ctx->size * batch_size);
        printf("[dist] Iterations: %d  |  games/iter (total): %d  "
               "|  depth: %d\n",
               iterations, games_per_iter, search_depth);
        printf("[dist] LR: %.2e → %.2e (cosine)  |  "
               "epochs/iter: %d  |  threads/rank: %d\n",
               (double)lr_max, (double)lr_min,
               train_epochs, num_threads_per_rank);
        printf("[dist] Checkpoint security: HMAC-SHA256 (%d-byte key)\n",
               DIST_HMAC_KEY_LEN);
        printf("[dist] ════════════════════════════════════════════════\n");
    }

    /* ── Load or initialise network ──────────────────────────────────── */
    int load_rc = dist_checkpoint_load(ctx, g_nnue, "nnue_checkpoint.bin",
                                        hmac_key);
    if (load_rc == 0) {
        if (dist_is_master(ctx))
            printf("[dist] Loaded checkpoint 'nnue_checkpoint.bin'.\n");
    } else {
        if (dist_is_master(ctx)) {
            if (load_rc == -2)
                fprintf(stderr, "[dist] HMAC check failed — random init.\n");
            else
                printf("[dist] No checkpoint found — random initialisation.\n");
        }
        nnue_init_random(g_nnue);
        /* Broadcast the freshly initialised weights so all ranks agree. */
        dist_broadcast_net(ctx, g_nnue);
    }

    /* ── Persistent Adam state ───────────────────────────────────────── */
    /*
     * Each rank runs an identical Adam step on the mean gradient (thanks to
     * AllReduce) so all rank-local AdamState structs stay in sync — we do not
     * need to broadcast them.  The moment vectors must persist across
     * iterations; re-allocating them every iteration would reset the warm-up.
     */
    AdamState *adam = calloc(1, sizeof *adam);
    if (!adam) {
        perror("[dist] calloc AdamState");
        MPI_Abort(ctx->comm, 1);
    }

    /* ── Per-rank replay buffer ──────────────────────────────────────── */
    /*
     * Each rank maintains its own independent replay buffer.  Sharing a
     * single buffer across ranks would require additional synchronisation and
     * provide negligible benefit — the AllReduce already aggregates learning
     * signal across all ranks' data.
     */
    ReplayBuffer *replay = replay_create(REPLAY_BUFFER_CAP);
    if (!replay) {
        perror("[dist] replay_create");
        free(adam);
        MPI_Abort(ctx->comm, 1);
    }

    /* ════════════════════════════════════════════════════════════════════
     * Main RL loop
     * ════════════════════════════════════════════════════════════════════ */
    for (int iter = 1; iter <= iterations; iter++) {

        dist_barrier(ctx);   /* align all ranks at the top of each iteration */

        if (dist_is_master(ctx)) {
            printf("\n[dist] ─── Iteration %d / %d  (Adam step %d) ───\n",
                   iter, iterations, adam->t);
        }

        /* Cosine-scheduled learning rate for this iteration */
        float lr = cosine_lr(lr_max, lr_min, iter - 1, iterations);
        if (dist_is_master(ctx))
            printf("[dist] LR = %.4e\n", (double)lr);

        /* ── Step 1: Rank-local self-play ────────────────────────────── */
        int local_games = dist_local_games(ctx, games_per_iter);

        printf("[dist rank %d] Self-play: %d game(s) at depth %d ...\n",
               ctx->rank, local_games, search_depth);

        PosDataset *ds = create_dataset();
        if (!ds) {
            fprintf(stderr, "[dist rank %d] create_dataset failed\n",
                    ctx->rank);
            MPI_Abort(ctx->comm, 1);
        }

        self_play_worker(local_games, search_depth, ds);

        printf("[dist rank %d] Self-play done: %d position(s)\n",
               ctx->rank, ds->count);

        if (ds->count == 0) {
            fprintf(stderr, "[dist rank %d] Warning: empty dataset this iter\n",
                    ctx->rank);
            free_dataset(ds);
            continue;
        }

        /* ── Step 2: Push positions → local replay buffer ────────────── */
        for (int i = 0; i < ds->count; i++)
            replay_push(replay, &ds->positions[i]);
        free_dataset(ds);

        if (dist_is_master(ctx)) {
            printf("[dist master] Replay: %d / %d positions (%.1f%% full)\n",
                   replay->count, replay->cap,
                   100.0 * replay->count / replay->cap);
        }

        /* ── Step 3: Train (allreduce inside compute_gradients_mt) ────── */
        /*
         * dist_allreduce_gradients() — called from within tune.c's
         * compute_gradients_mt() — averages gradients across all ranks
         * before the Adam step.  No changes to train_on_dataset() needed.
         */
        train_on_dataset(g_nnue, replay,
                          train_epochs, batch_size,
                          lr, WDL_LAMBDA,
                          num_threads_per_rank, adam);

        /* ── Step 4: Checkpoint ──────────────────────────────────────── */
        /*
         * Rank 0 writes; the barrier inside dist_checkpoint_save() ensures
         * all ranks have a coherent view of the file before the next
         * dist_broadcast_net() (step 5).
         */
        if (dist_is_master(ctx)) {
            printf("[dist master] Saving checkpoint (Adam step %d)...\n",
                   adam->t);
        }

        int save_rc = dist_checkpoint_save(ctx, g_nnue,
                                            "nnue_checkpoint.bin", hmac_key);
        if (save_rc != 0 && dist_is_master(ctx)) {
            fprintf(stderr, "[dist master] Checkpoint save failed!\n");
        } else if (dist_is_master(ctx)) {
            printf("[dist master] Checkpoint saved.\n");
        }

        /* ── Step 5: Broadcast updated weights to all ranks ──────────── */
        /*
         * Because every rank performed the same Adam step on the same mean
         * gradient, all rank-local g_nnue copies should already be identical.
         * The broadcast is a safety net: it ensures any floating-point
         * rounding divergence (very unlikely but possible with non-IEEE MPI
         * reductions) is corrected by anchoring everyone to rank 0's copy.
         */
        dist_broadcast_net(ctx, g_nnue);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    replay_free(replay);
    free(adam);

    if (dist_is_master(ctx))
        printf("[dist] Training complete.\n");
}