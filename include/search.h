#pragma once
#include "board.h"
#include "tt.h"
#include <time.h>
#include <stdint.h>

/* ──────────────────────────────────────────────
 *  Search limits passed in from UCI "go" command
 * ────────────────────────────────────────────── */
typedef struct {
    int   depth;          /* maximum depth (0 = unlimited) */
    int   movetime;       /* fixed time per move in ms (0 = use wtime/btime) */
    int   wtime, btime;   /* remaining time for each side in ms */
    int   winc,  binc;    /* increment per move in ms */
    bool  infinite;       /* ignore time limits */
    bool  stop;           /* set to true to abort search */
} SearchLimits;

/* ──────────────────────────────────────────────
 *  Search state / statistics
 * ────────────────────────────────────────────── */
#define MAX_KILLERS 2

typedef struct {
    uint64_t nodes;                         /* nodes visited this search              */
    int      seldepth;                      /* max selective depth reached            */
    Move     killers[MAX_PLY][MAX_KILLERS]; /* killer moves                           */
    int      history[2][64][64];            /* [color][from][to] history              */
    clock_t  start_time;                    /* when search began                      */
    int      allotted_ms;                   /* time budget in ms                      */
    SearchLimits *limits;
    int   eval_stack[MAX_PLY];              /* cached static eval at each ply         */
    Move  countermove[64][64];              /* countermove[from][to] refutation table */
    Move  move_stack[MAX_PLY];              /* the move played to reach ply N         */
} SearchInfo;

/* ──────────────────────────────────────────────
 *  MVV-LVA capture ordering table
 *  Indexed [attacker_type][victim_type] → score
 * ────────────────────────────────────────────── */
extern int MVV_LVA[6][6];

/* ──────────────────────────────────────────────
 *  API
 * ────────────────────────────────────────────── */
void search_init(void);

/*
 * Main search entry point.
 * Performs iterative deepening up to limits->depth (or until time runs out).
 * Prints UCI info lines and finishes with "bestmove <move>".
 */
void search(Board *b, SearchLimits *lim);

/* Exposed for internal use */
int  negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si);
int  quiesce(Board *b, int alpha, int beta, int ply, SearchInfo *si);

/* Time check (checked every 2048 nodes) */
bool time_up(SearchInfo *si);

int see(const Board *b, Move m);
