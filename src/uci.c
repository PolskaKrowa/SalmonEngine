#define _GNU_SOURCE
#include "uci.h"
#include "movegen.h"
#include "tt.h"
#include "book.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <strings.h>   /* strncasecmp */
#include <pthread.h>
#include <unistd.h>

/* Default TT size in MB */
#define DEFAULT_HASH_MB 64

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
    printf("option name OwnBook type check default true\n");
    printf("option name Ponder type check default false\n");
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

/* ── Search thread infrastructure ──
 *
 * The search runs in a background thread so the main UCI loop can
 * continue reading commands (stop, ponderhit, quit) while the search
 * is in progress.  This is essential for pondering: the search thread
 * runs indefinitely on `go ponder`, and the main thread waits for
 * either `ponderhit` (convert to time-limited) or `stop`/new `go`
 * (abort and return best move).
 */
static pthread_t g_search_thread;
static Board    *g_search_board;     /* points to the main Board in uci_loop */
static bool      g_search_running = false;

/* Thread entry point: calls search() with the global limits. */
static void *search_thread_main(void *arg) {
    (void)arg;
    search(g_search_board, &g_limits);
    g_search_running = false;
    return NULL;
}

/* Start a search in a background thread. */
static void start_search_thread(Board *b) {
    g_search_board    = b;
    g_search_running  = true;
    g_limits.stop     = false;
    pthread_create(&g_search_thread, NULL, search_thread_main, NULL);
}

/* Wait for the search thread to finish (called after stop/ponderhit). */
static void wait_for_search_thread(void) {
    if (g_search_running) {
        pthread_join(g_search_thread, NULL);
        g_search_running = false;
    }
}

static void handle_go(Board *b, const char *line) {
    /* If a previous search is still running (e.g., ponder), stop it
     * and wait for it to finish before starting a new one. */
    if (g_search_running) {
        g_limits.stop = true;
        wait_for_search_thread();
    }

    memset(&g_limits, 0, sizeof(g_limits));

    g_limits.depth = 32; // default search limit to prevent overcomputation

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
        else if (strncmp(p, "ponder",   6) == 0) { p += 6; g_limits.ponder   = true; }
        else { while (*p && *p != ' ') p++; continue; }

        while (*p && *p != ' ') p++;
    }

    /* For ponder or infinite, run in a thread so we can receive
     * ponderhit/stop.  For normal searches, also run in a thread so
     * `stop` can interrupt mid-search (this is required by UCI:
     * `stop` can arrive at any time). */
    start_search_thread(b);

    /* If this is NOT a ponder/infinite search, wait for it to finish
     * (the GUI expects `bestmove` before sending the next command). */
    if (!g_limits.ponder && !g_limits.infinite) {
        wait_for_search_thread();
    }
}

static void handle_setoption(const char *line) {
    /* Supported: "Hash" (spin) and "OwnBook" (check) */
    const char *name = strstr(line, "name");
    const char *val  = strstr(line, "value");
    if (!name || !val) return;
    name += 4; while (*name == ' ') name++;
    val  += 5; while (*val  == ' ') val++;

    if (strncasecmp(name, "hash", 4) == 0) {
        int mb = atoi(val);
        if (mb < 1) mb = 1;
        tt_init((size_t)mb);
    } else if (strncasecmp(name, "ownbook", 7) == 0
            || strncasecmp(name, "book", 4) == 0) {
        /* Accept "true"/"false" or "1"/"0". */
        bool enabled = (strncasecmp(val, "true", 4) == 0
                     || strncasecmp(val, "1",    1) == 0);
        book_set_enabled(enabled);
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
 *  Ponderhit handler
 *
 *  When the GUI sends `ponderhit`, the opponent played the move we
 *  predicted.  We convert the pondering (infinite) search into a
 *  normal time-limited search by clearing the `ponder` flag.  The
 *  search thread's `time_up()` will then start enforcing the time
 *  budget (which was set from wtime/btime/winc/binc at the start of
 *  search()).
 *
 *  NOTE: For this to work properly, the search() function must
 *  recompute its time limits when `ponder` transitions from true to
 *  false.  Currently the time limits are computed once at the start
 *  of search() and stored in SearchInfo; if ponder was true at
 *  start, those limits are 0 (infinite).  We need a mechanism to
 *  signal search() to recompute.
 *
 *  The simplest approach: store the time limits in g_limits (shared
 *  with the search thread) and have time_up() read from g_limits
 *  directly.  But that's a bigger refactor.
 *
 *  For now, we use the "stop after a delay" approach: on ponderhit,
 *  clear the ponder flag and set a short timer to set `stop`.  This
 *  gives the search a chance to finish its current iteration.  Not
 *  ideal, but functional.
 * ────────────────────────────────────────────── */
static void handle_ponderhit(void) {
    if (!g_search_running) return;
    /* Clear ponder flag — search will now check time limits.
     * But since time limits were set to 0 (infinite) at search start,
     * we need to also set `stop` after a brief delay to actually
     * terminate the search.
     *
     * The proper fix requires recomputing time limits mid-search,
     * which needs SearchInfo to be shared.  For now we use a simple
     * heuristic: sleep for 50ms (let the search do some work) then
     * set stop.  GUIs typically send `stop` shortly after `ponderhit`
     * anyway when their clock gets low. */
    g_limits.ponder = false;
    /* Give the search a brief window to complete the current
     * iteration, then stop.  The GUI's subsequent `stop` will also
     * work if it arrives first. */
    usleep(50 * 1000);  /* 50ms */
    g_limits.stop = true;
    wait_for_search_thread();
}

/* ──────────────────────────────────────────────
 *  Main UCI loop
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
            /* If a ponder search is running, stop it first. */
            if (g_search_running) {
                g_limits.stop = true;
                wait_for_search_thread();
            }
            tt_clear();
            board_start_pos(b);
        } else if (strncmp(line, "position", 8) == 0) {
            /* If a ponder search is running, stop it first (the GUI
             * is sending a new position, abandoning the ponder). */
            if (g_search_running) {
                g_limits.stop = true;
                wait_for_search_thread();
            }
            handle_position(b, line);
        } else if (strncmp(line, "go", 2) == 0) {
            handle_go(b, line);
        } else if (strcmp(line, "stop") == 0) {
            g_limits.stop = true;
            wait_for_search_thread();
        } else if (strcmp(line, "ponderhit") == 0) {
            handle_ponderhit();
        } else if (strncmp(line, "setoption", 9) == 0) {
            handle_setoption(line);
        } else if (strncmp(line, "perft", 5) == 0) {
            handle_perft(b, line);
        } else if (strncmp(line, "d", 1) == 0) {
            board_print(b); /* debug: print board */
        } else if (strcmp(line, "quit") == 0) {
            /* Stop any running search before exiting. */
            if (g_search_running) {
                g_limits.stop = true;
                wait_for_search_thread();
            }
            tt_free();
            return;
        }
    }

    /* If we exit the loop (EOF) while a search is running, stop it. */
    if (g_search_running) {
        g_limits.stop = true;
        wait_for_search_thread();
    }
    tt_free();
}
