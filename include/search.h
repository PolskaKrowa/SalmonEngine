/*
 * search.h — Search types and interface
 *
 * Refactored:
 *  • cont_hist[7][64] added to SearchInfo for continuation history (~30 Elo)
 *  • score_from_tt removed; replaced by value_from_tt / value_to_tt in search.c
 */

#pragma once
#include "board.h"
#include "tt.h"
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

/* ── Score constants ── */
#define INF          32000
#define MATE_SCORE   31000
#define DRAW_SCORE       0
#define NO_SCORE    -32001

/* ── Search limits (UCI) ── */
typedef struct {
    int  wtime, btime, winc, binc;
    int  movetime;
    int  depth;
    int  threads;       /* number of helper threads (lazy-SMP); 0 = single-threaded */
    bool infinite;
    bool stop;
} SearchLimits;

/* ── Per-search state ── */
typedef struct {
    SearchLimits *limits;
    /* Wall-clock start time (CLOCK_MONOTONIC). Using clock() would be wrong
     * in multi-threaded mode because clock() sums CPU time across threads. */
    struct timespec start_time_ts;
    /* Backwards-compat: legacy clock() value, still used by perft/test paths
     * that don't initialise start_time_ts. */
    clock_t       start_time;
    int           allotted_ms;

    uint64_t      nodes;
    int           seldepth;

    /* Heuristic tables */
    int  history   [2][64][64];   /* butterfly history [side][from][to]  */
    int  cont_hist [7][64];       /* continuation history [piece_type][to]
                                     Lightweight port of SF's continuation
                                     history: captures the (piece,to) affinity
                                     at ply-1 with ~30 Elo benefit.          */
    Move killers   [MAX_PLY][2];
    Move countermove[64][64];

    /* Per-ply stacks */
    int  eval_stack[MAX_PLY];
    Move move_stack[MAX_PLY];
} SearchInfo;

/* ── Public API ── */
void search_init (void);
void search      (Board *b, SearchLimits *lim);
bool time_up     (SearchInfo *si);
int  see         (const Board *b, Move m);

/* negamax with optional `excluded` move — used for singular-extension
 * verification. When excluded != NULL_MOVE, the TT probe uses an
 * exclusion-keyed slot (b->hash ^ ZKEYS.exclusion[from][to]) so the
 * bound stored for the parent's TT move doesn't short-circuit the
 * singularity test. Callers passing excluded=NULL_MOVE get normal
 * behaviour. */
int  negamax     (Board *b, int depth, int alpha, int beta,
                  int ply, SearchInfo *si, Move excluded);
int  quiesce     (Board *b, int alpha, int beta,
                  int ply, SearchInfo *si);
