#define _GNU_SOURCE
#include "uci.h"
#include "movegen.h"
#include "tt.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>   /* strncasecmp */
#include <pthread.h>

/* Default TT size in MB */
#define DEFAULT_HASH_MB 64
/* Default thread count — 1 means single-threaded (no helper threads). */
#define DEFAULT_THREADS 1

/* Runtime-configurable options. */
static int g_hash_mb   = DEFAULT_HASH_MB;
static int g_threads   = DEFAULT_THREADS;

/* ──────────────────────────────────────────────
 *  Move string helpers
 * ────────────────────────────────────────────── */
static const char FILE_CHARS[] = "abcdefgh";
static const char RANK_CHARS[] = "12345678";

void move_to_str(Move m, char *out) {
    if (!m) { strcpy(out, "0000"); return; }
    int from = MOVE_FROM(m);
    int to   = MOVE_TO(m);
    out[0] = FILE_CHARS[from & 7];
    out[1] = RANK_CHARS[from >> 3];
    out[2] = FILE_CHARS[to   & 7];
    out[3] = RANK_CHARS[to   >> 3];
    out[4] = '\0';
    if (MOVE_IS_PROMO(m)) {
        /* MOVE_PROMO_PT returns KNIGHT..QUEEN (1..4) */
        static const char promo_ch[] = "nbrq";
        out[4] = promo_ch[MOVE_PROMO_PT(m) - KNIGHT];
        out[5] = '\0';
    }
}

Move str_to_move(Board *b, const char *s) {
    if (!s || strlen(s) < 4) return NULL_MOVE;

    int from_file = s[0] - 'a';
    int from_rank = s[1] - '1';
    int to_file   = s[2] - 'a';
    int to_rank   = s[3] - '1';

    if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7 ||
        to_file   < 0 || to_file   > 7 || to_rank   < 0 || to_rank   > 7)
        return NULL_MOVE;

    Square from = (Square)(from_rank * 8 + from_file);
    Square to   = (Square)(to_rank   * 8 + to_file);

    /* Determine promotion piece type */
    PieceType promo_pt = NO_PIECE_TYPE;
    if (s[4]) {
        switch (tolower((unsigned char)s[4])) {
            case 'n': promo_pt = KNIGHT; break;
            case 'b': promo_pt = BISHOP; break;
            case 'r': promo_pt = ROOK;   break;
            case 'q': promo_pt = QUEEN;  break;
            default:  break;
        }
    }

    /* Match against generated move list to get the correct MoveType */
    MoveList ml;
    gen_moves(b, &ml);
    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (MOVE_FROM(m) != from || MOVE_TO(m) != to) continue;
        if (promo_pt != NO_PIECE_TYPE) {
            if (!MOVE_IS_PROMO(m)) continue;
            if (MOVE_PROMO_PT(m) != promo_pt) continue;
        } else {
            if (MOVE_IS_PROMO(m)) continue;
        }
        return m;
    }
    return NULL_MOVE;
}

/* ──────────────────────────────────────────────
 *  Command handlers
 * ────────────────────────────────────────────── */
static void handle_uci(void) {
    printf("id name %s %s\n", ENGINE_NAME, ENGINE_VERSION);
    printf("id author %s\n",  ENGINE_AUTHOR);
    printf("option name Hash type spin default %d min 1 max 2048\n",
           DEFAULT_HASH_MB);
    printf("option name Threads type spin default %d min 1 max 256\n",
           DEFAULT_THREADS);
    printf("uciok\n");
    fflush(stdout);
}

/* Parse a "position" command:
 *   position startpos [moves m1 m2 ...]
 *   position fen <fen> [moves m1 m2 ...]
 */
static void handle_position(Board *b, const char *line) {
    const char *p = line + 8; /* skip "position" */
    while (*p == ' ') p++;

    if (strncmp(p, "startpos", 8) == 0) {
        board_start_pos(b);
        p += 8;
    } else if (strncmp(p, "fen", 3) == 0) {
        p += 3;
        while (*p == ' ') p++;
        board_from_fen(b, p);
        /* Skip past the FEN fields */
        int spaces = 0;
        while (*p && spaces < 6) {
            if (*p == ' ') spaces++;
            p++;
        }
    } else {
        return;
    }

    /* Skip to "moves" token */
    p = strstr(p, "moves");
    if (!p) return;
    p += 5;

    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        char token[8] = {0};
        int  i = 0;
        while (*p && *p != ' ' && i < 7) token[i++] = *p++;
        Move m = str_to_move(b, token);
        if (m) make_move(b, m);
    }
}

/* Parse a "go" command */
static SearchLimits g_limits;  /* allocated once, avoids stack VLA */

/* Worker thread state.
 *
 * The main UCI loop spawns a worker thread to run search(); while the
 * worker is running, the main loop continues reading stdin so it can
 * receive `stop` (or `quit`). `search_running` is set while the worker
 * is active; the main loop joins it before processing any new command. */
static pthread_t g_search_thread;
static bool      g_search_running = false;
static Board    *g_search_board   = NULL;

static void *search_worker(void *arg) {
    (void)arg;
    search(g_search_board, &g_limits);
    g_search_running = false;  /* signal completion to the main loop */
    return NULL;
}

/* If a search is running, wait for it to finish. Called before processing
 * any non-`stop` command (and before `quit`) so we don't race on the board
 * state. */
static void wait_for_search_done(void) {
    if (g_search_running) {
        /* Setting stop here is conservative — most callers already set it
         * via the `stop` command, but if a `position`/`go` arrives mid-
         * search we need to abort the in-flight search before touching
         * the board. */
        g_limits.stop = true;
        pthread_join(g_search_thread, NULL);
        g_search_running = false;
    }
}

static void handle_go(Board *b, const char *line) {
    /* If a search is somehow already running (shouldn't happen in a
     * well-behaved GUI, but some test harnesses do this), abort it first. */
    wait_for_search_done();

    memset(&g_limits, 0, sizeof(g_limits));

    g_limits.depth = 32; // default search limit to prevent overcomputation
    g_limits.threads = g_threads;

    const char *p = line;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;

        if      (strncmp(p, "depth",    5) == 0) { p += 5; g_limits.depth    = atoi(p); }
        else if (strncmp(p, "movetime", 8) == 0) { p += 8; g_limits.movetime = atoi(p); }
        else if (strncmp(p, "wtime",    5) == 0) { p += 5; g_limits.wtime    = atoi(p); }
        else if (strncmp(p, "btime",    5) == 0) { p += 5; g_limits.btime    = atoi(p); }
        else if (strncmp(p, "winc",     4) == 0) { p += 4; g_limits.winc     = atoi(p); }
        else if (strncmp(p, "binc",     4) == 0) { p += 4; g_limits.binc     = atoi(p); }
        else if (strncmp(p, "infinite", 8) == 0) { p += 8; g_limits.infinite = true; }
        else { while (*p && *p != ' ') p++; continue; }

        while (*p && *p != ' ') p++;
    }

    /* If infinite mode is requested, ignore the default depth=32 so the
     * search truly runs until `stop` arrives. */
    if (g_limits.infinite) g_limits.depth = 0;
    /* If movetime is set, ignore depth so time is the only limit. */
    if (g_limits.movetime > 0) g_limits.depth = 0;

    /* Spawn the search in a worker thread so the main loop can keep
     * reading stdin and process `stop` / `quit` while search runs.
     * This fixes the long-standing UCI `stop` bug where the engine
     * couldn't be interrupted mid-search — the main thread was blocked
     * inside search() and couldn't process incoming commands. */
    g_search_board = b;
    g_search_running = true;
    if (pthread_create(&g_search_thread, NULL, search_worker, NULL) != 0) {
        /* Spawn failed — fall back to blocking search. */
        g_search_running = false;
        search(b, &g_limits);
    }
    /* NOTE: we do NOT join here. The main loop continues; it will join
     * when stop arrives or before the next command. */
}

static void handle_setoption(const char *line) {
    /* Supported options: Hash, Threads */
    const char *name = strstr(line, "name");
    const char *val  = strstr(line, "value");
    if (!name || !val) return;
    name += 4; while (*name == ' ') name++;
    val  += 5; while (*val  == ' ') val++;

    if (strncasecmp(name, "hash", 4) == 0) {
        int mb = atoi(val);
        if (mb < 1) mb = 1;
        if (mb > 2048) mb = 2048;
        g_hash_mb = mb;
        tt_init((size_t)mb);
    } else if (strncasecmp(name, "threads", 7) == 0) {
        int n = atoi(val);
        if (n < 1) n = 1;
        if (n > 256) n = 256;
        g_threads = n;
    }
}

static void handle_perft(Board *b, const char *line) {
    int depth = atoi(line + 5);
    if (depth < 1) depth = 1;
    for (int d = 1; d <= depth; d++) {
        uint64_t nodes = perft(b, d);
        printf("depth %d: %llu nodes\n", d, (unsigned long long)nodes);
        fflush(stdout);
    }
}

/* ──────────────────────────────────────────────
 *  Main UCI loop
 *
 *  While a search is running (in a worker thread), the main loop continues
 *  to read stdin. Commands that would mutate the board (position, perft,
 *  setoption, go, quit) first call wait_for_search_done() to abort and
 *  join the worker. `stop` is handled immediately by setting the
 *  SearchLimits.stop flag — the worker polls this flag every 2048 nodes
 *  and will exit on the next check.
 * ────────────────────────────────────────────── */
void uci_loop(Board *b) {
    tt_init(DEFAULT_HASH_MB);
    board_start_pos(b);

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[len-1] = '\0';

        if (strcmp(line, "uci") == 0) {
            handle_uci();
        } else if (strcmp(line, "isready") == 0) {
            printf("readyok\n");
            fflush(stdout);
        } else if (strcmp(line, "ucinewgame") == 0) {
            wait_for_search_done();
            tt_clear();
            board_start_pos(b);
        } else if (strncmp(line, "position", 8) == 0) {
            wait_for_search_done();
            handle_position(b, line);
        } else if (strncmp(line, "go", 2) == 0) {
            handle_go(b, line);
        } else if (strcmp(line, "stop") == 0) {
            /* Set the stop flag and wait for the search to finish. UCI
             * semantics require that `bestmove` is printed before any
             * subsequent command is processed, so we must join here. */
            g_limits.stop = true;
            wait_for_search_done();
        } else if (strncmp(line, "setoption", 9) == 0) {
            wait_for_search_done();
            handle_setoption(line);
        } else if (strncmp(line, "perft", 5) == 0) {
            wait_for_search_done();
            handle_perft(b, line);
        } else if (strncmp(line, "d", 1) == 0) {
            wait_for_search_done();
            board_print(b);
        } else if (strcmp(line, "quit") == 0) {
            wait_for_search_done();
            tt_free();
            return;
        }
    }

    wait_for_search_done();
    tt_free();
}
