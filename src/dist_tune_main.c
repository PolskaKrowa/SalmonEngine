/*
 * dist_tune_main.c — Entry point for the distributed NNUE tuner
 *
 * Replaces the `#ifdef TUNE_STANDALONE` main() block in tune.c.
 * Compile tune.c WITHOUT -DTUNE_STANDALONE when linking this file.
 *
 * Build:
 *   mpicc -O3 -march=native \
 *         tune.c dist_tune.c dist_tune_main.c \
 *         -lopenblas -lssl -lcrypto -lm -lpthread \
 *         -o dist_tuner
 *
 * Run (4 processes, 2 threads each):
 *   mpirun -np 4 ./dist_tuner --threads 2 --games 2000 --iters 100
 *
 * The HMAC key is read from the environment variable TUNE_HMAC_KEY_HEX,
 * which must be exactly 64 hex characters (= 32 bytes).
 *
 * Example:
 *   export TUNE_HMAC_KEY_HEX=$(openssl rand -hex 32)
 *   mpirun -np 4 ./dist_tuner
 *
 * Keep the key consistent across all nodes and all sessions that share
 * checkpoints — changing the key invalidates existing checkpoint files.
 */

#include "dist_tune.h"
#include "nnue.h"
#include "bitboard.h"
#include "board.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ── Global network pointer (referenced by tune.c and dist_tune.c) ──────── */
NNUENet *g_nnue       = NULL;
int      g_num_threads = 1;

/* ── Defaults ────────────────────────────────────────────────────────────── */
#define DEFAULT_ITERATIONS      50
#define DEFAULT_GAMES_PER_ITER  1000
#define DEFAULT_SEARCH_DEPTH    3
#define DEFAULT_THREADS         4
#define DEFAULT_BATCH_SIZE      1024
#define DEFAULT_TRAIN_EPOCHS    10
#define DEFAULT_LR              1e-3f

/* ─────────────────────────────────────────────────────────────────────────
 * HMAC key loading
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Reads TUNE_HMAC_KEY_HEX from the environment.  This is a simple,
 * deployment-friendly approach that avoids key material appearing in
 * command-line arguments (which are visible in ps / /proc).
 *
 * For production use consider replacing this with a proper secrets manager
 * (HashiCorp Vault, AWS Secrets Manager, etc.).
 */
static int load_hmac_key(unsigned char key[DIST_HMAC_KEY_LEN], int rank) {
    const char *hex = getenv("TUNE_HMAC_KEY_HEX");
    if (!hex) {
        if (rank == 0) {
            fprintf(stderr,
                "[dist] TUNE_HMAC_KEY_HEX not set.\n"
                "  Generate one with:  openssl rand -hex 32\n"
                "  Then export it:     export TUNE_HMAC_KEY_HEX=<value>\n"
                "  All ranks must use the same key.\n");
        }
        return -1;
    }

    int hex_len = (int)strlen(hex);
    if (hex_len != 2 * DIST_HMAC_KEY_LEN) {
        if (rank == 0)
            fprintf(stderr,
                "[dist] TUNE_HMAC_KEY_HEX must be exactly %d hex chars "
                "(got %d)\n", 2 * DIST_HMAC_KEY_LEN, hex_len);
        return -1;
    }

    for (int i = 0; i < DIST_HMAC_KEY_LEN; i++) {
        char hi = tolower((unsigned char)hex[2 * i]);
        char lo = tolower((unsigned char)hex[2 * i + 1]);

        int hv = (hi >= '0' && hi <= '9') ? hi - '0'
               : (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10 : -1;
        int lv = (lo >= '0' && lo <= '9') ? lo - '0'
               : (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10 : -1;

        if (hv < 0 || lv < 0) {
            if (rank == 0)
                fprintf(stderr, "[dist] Invalid hex character in "
                                "TUNE_HMAC_KEY_HEX at position %d\n", 2 * i);
            return -1;
        }
        key[i] = (unsigned char)((hv << 4) | lv);
    }
    return 0;
}

/* ── Argument parsing ────────────────────────────────────────────────────── */

static void print_usage(const char *prog, int rank) {
    if (rank != 0) return;
    fprintf(stderr,
        "Usage: mpirun -np N %s [options]\n"
        "\n"
        "  --iters    INT    RL iterations          (default: %d)\n"
        "  --games    INT    Total self-play games   (default: %d)\n"
        "  --depth    INT    Search depth            (default: %d)\n"
        "  --threads  INT    pthreads per rank       (default: %d)\n"
        "  --batch    INT    Minibatch size per rank (default: %d)\n"
        "  --epochs   INT    Train epochs per iter   (default: %d)\n"
        "  --lr       FLOAT  Peak learning rate      (default: %.2e)\n"
        "  --help            Print this message\n"
        "\n"
        "Environment:\n"
        "  TUNE_HMAC_KEY_HEX  64-char hex key for checkpoint HMAC-SHA256\n"
        "                     (required; generate with: openssl rand -hex 32)\n",
        prog,
        DEFAULT_ITERATIONS, DEFAULT_GAMES_PER_ITER, DEFAULT_SEARCH_DEPTH,
        DEFAULT_THREADS, DEFAULT_BATCH_SIZE, DEFAULT_TRAIN_EPOCHS,
        (double)DEFAULT_LR);
}

typedef struct {
    int   iterations;
    int   games_per_iter;
    int   search_depth;
    int   threads;
    int   batch_size;
    int   train_epochs;
    float lr;
} Args;

static int parse_args(int argc, char **argv, Args *a, int rank) {
    a->iterations    = DEFAULT_ITERATIONS;
    a->games_per_iter = DEFAULT_GAMES_PER_ITER;
    a->search_depth  = DEFAULT_SEARCH_DEPTH;
    a->threads       = DEFAULT_THREADS;
    a->batch_size    = DEFAULT_BATCH_SIZE;
    a->train_epochs  = DEFAULT_TRAIN_EPOCHS;
    a->lr            = DEFAULT_LR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0], rank);
            return -1;
        }
#define INTARG(flag, field) \
        if (strcmp(argv[i], flag) == 0) { \
            if (++i >= argc) { \
                if (rank == 0) \
                    fprintf(stderr, "[dist] Missing value for %s\n", flag); \
                return -1; \
            } \
            a->field = atoi(argv[i]); continue; \
        }
#define FLTARG(flag, field) \
        if (strcmp(argv[i], flag) == 0) { \
            if (++i >= argc) { \
                if (rank == 0) \
                    fprintf(stderr, "[dist] Missing value for %s\n", flag); \
                return -1; \
            } \
            a->field = (float)atof(argv[i]); continue; \
        }
        INTARG("--iters",   iterations)
        INTARG("--games",   games_per_iter)
        INTARG("--depth",   search_depth)
        INTARG("--threads", threads)
        INTARG("--batch",   batch_size)
        INTARG("--epochs",  train_epochs)
        FLTARG("--lr",      lr)
#undef INTARG
#undef FLTARG
        if (rank == 0)
            fprintf(stderr, "[dist] Unknown argument: %s\n", argv[i]);
        return -1;
    }
    return 0;
}

/* ── Entry point ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {

    /* ── Init MPI first; we need rank for error messages ────────────── */
    DistCtx *ctx = dist_init(&argc, &argv);
    if (!ctx) return 1;

    int rank = dist_rank(ctx);

    /* ── Parse arguments ─────────────────────────────────────────────── */
    Args args;
    if (parse_args(argc, argv, &args, rank) != 0) {
        dist_teardown(ctx);
        return 1;
    }

    /* ── Load HMAC key from environment ──────────────────────────────── */
    unsigned char hmac_key[DIST_HMAC_KEY_LEN];
    if (load_hmac_key(hmac_key, rank) != 0) {
        dist_teardown(ctx);
        return 1;
    }

    /* ── Engine init (all ranks) ─────────────────────────────────────── */
    bitboard_init();
    board_init();

    g_nnue = nnue_alloc();
    if (!g_nnue) {
        fprintf(stderr, "[dist rank %d] Fatal: nnue_alloc failed (~185 MB)\n",
                rank);
        dist_teardown(ctx);
        return 1;
    }

    /* ── Banner (rank 0 only) ────────────────────────────────────────── */
    if (rank == 0) {
        printf("NNUE Distributed Self-Play RL Tuner\n");
        printf("  Architecture: %d → %d → %d → %d → 1  (%d params)\n",
               NNUE_FT_IN, NNUE_L1, NNUE_L2, NNUE_L3, NNUE_TOTAL_PARAMS);
        printf("  Activation: SCReLU  |  Output scale: %d\n",
               NNUE_OUTPUT_SCALE);
        printf("  MPI ranks: %d  |  threads/rank: %d\n",
               dist_size(ctx), args.threads);
        printf("  Effective batch/step: %d × %d = %d\n",
               dist_size(ctx), args.batch_size,
               dist_size(ctx) * args.batch_size);
    }

    /* ── Run distributed training ────────────────────────────────────── */
    dist_tune_loop(ctx, hmac_key,
                    args.iterations,
                    args.games_per_iter,
                    args.search_depth,
                    args.threads,
                    args.batch_size,
                    args.lr,
                    args.train_epochs);

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    nnue_free(g_nnue);
    g_nnue = NULL;

    /* Zero the key from the stack before returning */
    memset(hmac_key, 0, sizeof hmac_key);

    dist_teardown(ctx);
    return 0;
}