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

/*
 * Countermove history: [piece_type][to_square] → best counter-move score.
 *
 * The plain countermove heuristic stores a single best reply for each
 * (piece, to) pair.  The 3D countermove history additionally stores a
 * SCORE for every potential counter, indexed by the counter's own
 * (piece, to).  This gives the move ordering a much richer signal:
 * instead of "this move was the previous cutoff reply", it's "this
 * move has historically been good in this context".
 *
 * Total size: 6 * 64 * 6 * 64 * 2 bytes = 294,912 bytes (~288 KB).
 * The access pattern is regular enough that L1/L2 hit rates stay high.
 *
 * We use int16_t to keep the table compact.  Saturating arithmetic in
 * the update prevents overflow — same formula as the butterfly history.
 */
static const int COUNTER_HIST_MAX = 16384;  /* |score| clamp before store */

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
    /*
     * counter_hist: [prev_piece][prev_to][cur_piece][cur_to].
     * Replaces the single-move countermove table for *scoring*.
     * The plain `countermove[from][to]` table stays for the killer-style
     * exact-match bonus, but counter_hist provides the per-move *score*
     * that lets us rank multiple counter candidates.
     */
    int16_t counter_hist [CONT_HIST_PIECES][64][CONT_HIST_PIECES][64];
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
