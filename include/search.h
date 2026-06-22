/*
 * search.h — Search types and interface
 *
 * Refactored:
 *  • cont_hist now has two tables (ply-1 and ply-2) for stronger move ordering.
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
    bool infinite;
    bool stop;
} SearchLimits;

/*
 * Continuation-history tables.
 *
 * Indexed by [piece_type][to_square], where `piece_type` covers the six
 * piece types PAWN..KING.  Slot 0 (PAWN) is rarely used but kept for
 * indexing simplicity — indexing by `piece_type` directly avoids a
 * subtract that would otherwise be needed everywhere.
 *
 * cont_hist_p1 is updated/read using the *previous* ply's (piece, to).
 * cont_hist_p2 is updated/read using the *ply-2*  (piece, to).
 *
 * Both tables are int16_t (~32 KB total) and use SF-style saturating
 * arithmetic via the stat_bonus / stat_hat_penalty helpers in search.c.
 * Compared to the previous 7*64 int table this halves the memory
 * footprint and lets the second continuation history ride for free.
 */
#define CONT_HIST_PIECES 6
#define CONT_HIST_MAX    16384

/* ── Per-search state ── */
typedef struct {
    SearchLimits *limits;
    clock_t       start_time;
    int           allotted_ms;

    uint64_t      nodes;
    int           seldepth;

    /* Heuristic tables */
    int  history   [2][64][64];        /* butterfly history [side][from][to]  */
    int16_t cont_hist_p1 [CONT_HIST_PIECES][64]; /* ply-1 continuation history */
    int16_t cont_hist_p2 [CONT_HIST_PIECES][64]; /* ply-2 continuation history */
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
int  negamax     (Board *b, int depth, int alpha, int beta,
                  int ply, SearchInfo *si);
int  quiesce     (Board *b, int alpha, int beta,
                  int ply, SearchInfo *si);
