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

    _Static_assert(sizeof(NNUEGrad) == sizeof(float) * NNUE_TOTAL_PARAMS,
                "NNUEGrad must be a packed float array for MPI_Allreduce; "
                "update dist_tune.c if the struct layout changes.");

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

    /* ── Rank 0 only ─────────────────────────────────────────────────────
     *
     * Strategy: delegate serialisation entirely to nnue_save() so the on-disk
     * format is identical to what the engine reads natively (magic, version,
     * architecture header, then weight arrays).  The HMAC tag is appended
     * after the NNUE payload; dist_checkpoint_load() knows to strip it.
     *
     * Steps:
     *   1. nnue_save() → a uniquely named temp file  (handles its own atomic
     *      rename internally, so we get a fully flushed, valid NNUE file)
     *   2. Read the temp file back into a heap buffer
     *   3. Compute HMAC-SHA256 over those bytes
     *   4. Write [nnue_payload][hmac_tag] to a second temp file, then rename
     *      to `path` (atomic on POSIX)
     */

    char nnue_tmp[512];
    snprintf(nnue_tmp, sizeof nnue_tmp, "%s.nnue_tmp", path);

    if (nnue_save(net, nnue_tmp) != 0) {
        fprintf(stderr, "[dist master] nnue_save failed while saving "
                        "checkpoint '%s'\n", path);
        result = -1;
        goto barrier;
    }

    {
        /* ── Read the nnue_save output back as raw bytes ─────────────── */
        FILE *f = fopen(nnue_tmp, "rb");
        if (!f) {
            fprintf(stderr, "[dist master] Cannot re-open temp NNUE file "
                            "'%s': %s\n", nnue_tmp, strerror(errno));
            remove(nnue_tmp);
            result = -1;
            goto barrier;
        }

        if (fseek(f, 0, SEEK_END) != 0) {
            fprintf(stderr, "[dist master] fseek failed on '%s'\n", nnue_tmp);
            fclose(f); remove(nnue_tmp);
            result = -1;
            goto barrier;
        }
        long nnue_size = ftell(f);
        rewind(f);

        if (nnue_size <= 0) {
            fprintf(stderr, "[dist master] Unexpected size (%ld) for '%s'\n",
                    nnue_size, nnue_tmp);
            fclose(f); remove(nnue_tmp);
            result = -1;
            goto barrier;
        }

        unsigned char *buf = malloc((size_t)nnue_size);
        if (!buf) {
            fprintf(stderr, "[dist master] OOM reading NNUE temp file\n");
            fclose(f); remove(nnue_tmp);
            result = -1;
            goto barrier;
        }

        if ((long)fread(buf, 1, (size_t)nnue_size, f) != nnue_size) {
            fprintf(stderr, "[dist master] Short read from '%s'\n", nnue_tmp);
            free(buf); fclose(f); remove(nnue_tmp);
            result = -1;
            goto barrier;
        }
        fclose(f);
        remove(nnue_tmp);

        /* ── Compute HMAC-SHA256 over the NNUE file bytes ────────────── */
        unsigned char tag[DIST_HMAC_LEN];
        if (compute_hmac_sha256(key, buf, (size_t)nnue_size, tag) != 0) {
            fprintf(stderr, "[dist master] HMAC-SHA256 computation failed\n");
            free(buf);
            result = -1;
            goto barrier;
        }

        /* ── Atomically write [nnue_payload][hmac_tag] ───────────────── */
        char cp_tmp[512];
        snprintf(cp_tmp, sizeof cp_tmp, "%s.tmp", path);

        FILE *out = fopen(cp_tmp, "wb");
        if (!out) {
            fprintf(stderr, "[dist master] Cannot open checkpoint for writing: "
                            "%s — %s\n", cp_tmp, strerror(errno));
            free(buf);
            result = -1;
            goto barrier;
        }

        bool write_ok = (fwrite(buf, 1, (size_t)nnue_size, out)
                                == (size_t)nnue_size)
                     && (fwrite(tag, DIST_HMAC_LEN, 1, out) == 1);
        free(buf);
        fflush(out);
        fclose(out);

        if (!write_ok) {
            fprintf(stderr, "[dist master] Short write to checkpoint '%s'\n",
                    cp_tmp);
            remove(cp_tmp);
            result = -1;
            goto barrier;
        }

        if (rename(cp_tmp, path) != 0) {
            fprintf(stderr, "[dist master] rename('%s' → '%s') failed: %s\n",
                    cp_tmp, path, strerror(errno));
            remove(cp_tmp);
            result = -1;
        }
    }

barrier:
    MPI_Barrier(ctx->comm);  /* signal non-masters that the file is ready */
    return result;
}

int dist_checkpoint_load(DistCtx *ctx, NNUENet *net, const char *path,
                          const unsigned char key[DIST_HMAC_KEY_LEN]) {
    int load_err = 0;

    /*
     * Rank 0 reads and verifies; all other ranks wait, then receive the
     * parsed network via MPI_Bcast.
     *
     * Strategy: mirror what dist_checkpoint_save() wrote —
     *   [nnue_payload (nnue_save() format)][DIST_HMAC_LEN bytes of HMAC tag]
     *
     * Steps (rank 0 only):
     *   1. Read the whole file into a heap buffer
     *   2. Split at [total_size − DIST_HMAC_LEN]
     *   3. Verify HMAC-SHA256 over the NNUE payload bytes
     *   4. Write the payload to a temp file, call nnue_load() on it
     *      (nnue_load validates magic, version, and architecture dimensions)
     *   5. Remove the temp file
     */

    if (ctx->rank == 0) {
        FILE *f = fopen(path, "rb");
        if (!f) {
            load_err = -1;
            goto bcast_err;
        }

        if (fseek(f, 0, SEEK_END) != 0) {
            fclose(f);
            load_err = -1;
            goto bcast_err;
        }
        long total = ftell(f);
        rewind(f);

        long nnue_size = total - (long)DIST_HMAC_LEN;
        if (nnue_size <= 0) {
            fprintf(stderr,
                "[dist master] Checkpoint '%s' is too small to contain a "
                "valid NNUE payload + HMAC tag (total=%ld bytes)\n",
                path, total);
            fclose(f);
            load_err = -1;
            goto bcast_err;
        }

        unsigned char *buf = malloc((size_t)total);
        if (!buf) {
            perror("[dist master] malloc in dist_checkpoint_load");
            fclose(f);
            load_err = -1;
            goto bcast_err;
        }

        if ((long)fread(buf, 1, (size_t)total, f) != total) {
            fprintf(stderr, "[dist master] Short read from checkpoint '%s'\n",
                    path);
            free(buf); fclose(f);
            load_err = -1;
            goto bcast_err;
        }
        fclose(f);

        /* ── HMAC verification over the NNUE payload bytes ───────────── */
        const unsigned char *stored_tag  = buf + nnue_size;
        unsigned char        computed_tag[DIST_HMAC_LEN];

        if (compute_hmac_sha256(key, buf, (size_t)nnue_size, computed_tag) != 0) {
            fprintf(stderr, "[dist master] HMAC-SHA256 computation failed "
                            "during load of '%s'\n", path);
            free(buf);
            load_err = -1;
            goto bcast_err;
        }

        if (!hmac_eq_ct(stored_tag, computed_tag)) {
            fprintf(stderr,
                "[dist master] HMAC-SHA256 verification FAILED for '%s'.\n"
                "  The checkpoint may be corrupted or was written with a\n"
                "  different key.  Refusing to load — net zeroed for safety.\n",
                path);
            free(buf);
            memset(net, 0, sizeof *net);
            load_err = -2;
            goto bcast_err;
        }

        /* ── Write NNUE payload to a temp file for nnue_load() ───────── */
        /*
         * nnue_load() validates magic, version, and architecture dimensions,
         * giving us a second layer of structural integrity on top of the HMAC.
         */
        char tmp[512];
        snprintf(tmp, sizeof tmp, "%s.nnue_tmp", path);

        FILE *tf = fopen(tmp, "wb");
        if (!tf ||
            fwrite(buf, 1, (size_t)nnue_size, tf) != (size_t)nnue_size) {
            fprintf(stderr, "[dist master] Failed to write temp NNUE file "
                            "'%s': %s\n", tmp, strerror(errno));
            if (tf) { fclose(tf); remove(tmp); }
            free(buf);
            load_err = -1;
            goto bcast_err;
        }
        fclose(tf);
        free(buf);

        if (nnue_load(net, tmp) != 0) {
            fprintf(stderr, "[dist master] nnue_load failed for checkpoint "
                            "'%s' (NNUE payload from '%s')\n", path, tmp);
            remove(tmp);
            load_err = -1;
            goto bcast_err;
        }
        remove(tmp);
    }

bcast_err:
    MPI_Bcast(&load_err, 1, MPI_INT, 0, ctx->comm);
    if (load_err != 0)
        return load_err;

    MPI_Bcast(net, (int)sizeof *net, MPI_BYTE, 0, ctx->comm);
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