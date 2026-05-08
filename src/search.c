/*
 * search.c — Move searching
 *
 * Architecture:
 *  • Iterative deepening (ID) driver in search()
 *  • negamax() — principal variation search (PVS) with:
 *      – Transposition table cutoffs
 *      – Null-move pruning (NMP, R=3)
 *      – Late move reductions (LMR, simple formula)
 *      – Killer move heuristic (2 killers per ply)
 *      – History heuristic
 *      – MVV-LVA capture ordering
 *      – TT move first
 *  • quiesce() — quiescence search on captures + queen-promos only
 *  • Time management: check every 2048 nodes; stop when budget exceeded
 */

#include "search.h"
#include "movegen.h"
#include "eval.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ──────────────────────────────────────────────
 *  MVV-LVA table  [victim][attacker]
 * ────────────────────────────────────────────── */
int MVV_LVA[6][6];

static void init_mvv_lva(void) {
    /* Victim values: P=10, N=20, B=30, R=40, Q=50, K=60
       Attacker penalty subtracted so that, e.g. PxQ > BxQ  */
    static const int victim_val[6]   = { 10, 20, 30, 40, 50, 60 };
    static const int attacker_val[6] = {  1,  2,  3,  4,  5,  6 };
    for (int v = 0; v < 6; v++)
        for (int a = 0; a < 6; a++)
            MVV_LVA[a][v] = victim_val[v] * 10 - attacker_val[a];
}

/* ──────────────────────────────────────────────
 *  Move scoring for ordering
 * ────────────────────────────────────────────── */
#define SCORE_TT_MOVE   1000000
#define SCORE_GOOD_CAP   900000
#define SCORE_KILLER1    800000
#define SCORE_KILLER2    790000
#define SCORE_QUIET_BASE       0  /* + history score */

static int score_move(const Board *b, Move m, Move tt_move,
                      const Move *killers, int hist[64][64]) {
    if (m == tt_move) return SCORE_TT_MOVE;

    if (MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP) {
        int from = MOVE_FROM(m), to = MOVE_TO(m);
        int attacker = (int)piece_type(b->mailbox[from]);
        int victim   = (MOVE_TYPE(m) == MT_EP)
                       ? (int)PAWN
                       : (int)piece_type(b->mailbox[to]);
        return SCORE_GOOD_CAP + MVV_LVA[attacker][victim];
    }

    if (MOVE_IS_PROMO(m)) return SCORE_GOOD_CAP - 1;

    if (m == killers[0]) return SCORE_KILLER1;
    if (m == killers[1]) return SCORE_KILLER2;

    return SCORE_QUIET_BASE + hist[MOVE_FROM(m)][MOVE_TO(m)];
}

/* Partial insertion sort: bring the best move to position [idx] */
static Move pick_move(Move *moves, int *scores, int count, int idx) {
    int best = idx;
    for (int i = idx + 1; i < count; i++)
        if (scores[i] > scores[best]) best = i;
    /* Swap */
    Move tm = moves[idx]; moves[idx] = moves[best]; moves[best] = tm;
    int  ts = scores[idx]; scores[idx] = scores[best]; scores[best] = ts;
    return moves[idx];
}

/* ──────────────────────────────────────────────
 *  Time management
 * ────────────────────────────────────────────── */
bool time_up(SearchInfo *si) {
    if (si->limits->infinite || si->limits->stop) return si->limits->stop;
    if (si->allotted_ms <= 0) return false;
    clock_t elapsed = (clock() - si->start_time) * 1000 / CLOCKS_PER_SEC;
    return (int)elapsed >= si->allotted_ms;
}

/* ──────────────────────────────────────────────
 *  Quiescence search
 * ────────────────────────────────────────────── */
int quiesce(Board *b, int alpha, int beta, int ply, SearchInfo *si) {
    si->nodes++;
    if (ply > si->seldepth) si->seldepth = ply;

    int stand_pat = evaluate(b);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList ml;
    gen_captures(b, &ml);

    /* Score moves */
    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply];
    int (*hist_side)[64] = si->history[b->side];
    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], NULL_MOVE,
                                killers_ply, hist_side);

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);
        if (!is_legal(b, m)) continue;

        make_move(b, m);
        int score = -quiesce(b, -beta, -alpha, ply + 1, si);
        unmake_move(b);

        if (si->limits->stop) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

/* ──────────────────────────────────────────────
 *  Negamax alpha-beta (PVS)
 * ────────────────────────────────────────────── */
int negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si) {
    if (si->nodes % 2048 == 0 && time_up(si)) {
        si->limits->stop = true;
        return 0;
    }

    si->nodes++;
    bool root = (ply == 0);
    bool pv   = (beta - alpha > 1);

    /* Draw detection */
    if (!root) {
        if (b->halfmove >= 100) return DRAW_SCORE;
        /* Simple repetition check: scan history for same hash */
        for (int i = b->hist_idx - 2; i >= 0 && i >= b->hist_idx - b->halfmove; i -= 2) {
            if (b->history[i].hash == b->hash) return DRAW_SCORE;
        }
    }

    /* TT probe */
    TTEntry tte;
    Move    tt_move = NULL_MOVE;
    if (tt_probe(b->hash, &tte)) {
        tt_move = tte.move;
        if (!pv && tte.depth >= depth) {
            int tt_score = score_from_tt(tte.score, ply);
            if (tte.flag == TT_EXACT) return tt_score;
            if (tte.flag == TT_LOWER && tt_score >= beta)  return tt_score;
            if (tte.flag == TT_UPPER && tt_score <= alpha) return tt_score;
        }
    }

    if (depth <= 0) return quiesce(b, alpha, beta, ply, si);

    bool in_chk = in_check(b);

    /* Null-move pruning (skip if in check or in endgame) */
    if (!pv && !in_chk && depth >= 3 &&
        bb_popcount(b->occ[b->side] & ~b->pieces[b->side][PAWN]
                                    & ~b->pieces[b->side][KING]) > 0) {
        make_null_move(b);
        int null_score = -negamax(b, depth - 3, -beta, -beta + 1, ply + 1, si);
        unmake_null_move(b);
        if (null_score >= beta) return beta;
    }

    /* Generate and score moves */
    MoveList ml;
    gen_moves(b, &ml);

    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply];
    int (*hist_side)[64] = si->history[b->side];
    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], tt_move,
                                killers_ply, hist_side);

    int  best_score  = -INF;
    Move best_move   = NULL_MOVE;
    int  legal_count = 0;
    int  tt_flag     = TT_UPPER;

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);
        if (!is_legal(b, m)) continue;

        legal_count++;
        make_move(b, m);

        int score;
        int new_depth = depth - 1;

        /* Check extension */
        if (in_check(b)) new_depth++;

        /* Late move reductions (LMR) */
        bool do_lmr = legal_count > 3 && depth >= 3
                   && !MOVE_IS_CAP(m) && !MOVE_IS_PROMO(m) && !in_chk;
        int reduction = 0;
        if (do_lmr) {
            reduction = 1 + (depth > 6 ? 1 : 0) + (legal_count > 10 ? 1 : 0);
        }

        if (legal_count == 1) {
            /* Full-window search for the first (presumably best) move */
            score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si);
        } else {
            /* Reduced null-window search */
            score = -negamax(b, new_depth - reduction, -alpha - 1, -alpha, ply + 1, si);
            /* Re-search at full depth if it beat alpha */
            if (score > alpha && (reduction > 0 || pv)) {
                score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si);
            }
        }

        unmake_move(b);

        if (si->limits->stop) return 0;

        if (score > best_score) {
            best_score = score;
            best_move  = m;
        }
        if (score > alpha) {
            alpha    = score;
            tt_flag  = TT_EXACT;
            if (score >= beta) {
                /* Beta cutoff — update killer and history */
                if (!MOVE_IS_CAP(m)) {
                    si->killers[ply][1] = si->killers[ply][0];
                    si->killers[ply][0] = m;
                    si->history[b->side][MOVE_FROM(m)][MOVE_TO(m)] +=
                        depth * depth;
                }
                tt_store(b->hash, beta, best_move, depth, TT_LOWER, ply);
                return beta;
            }
        }
    }

    /* No legal moves: checkmate or stalemate */
    if (legal_count == 0) {
        return in_chk ? (-MATE_SCORE + ply) : DRAW_SCORE;
    }

    tt_store(b->hash, best_score, best_move, depth, tt_flag, ply);
    return best_score;
}

/* ──────────────────────────────────────────────
 *  Iterative deepening driver
 * ────────────────────────────────────────────── */
extern void move_to_str(Move m, char *out);

void search(Board *b, SearchLimits *lim) {
    SearchInfo si;
    memset(&si, 0, sizeof(si));
    si.limits     = lim;
    si.start_time = clock();

    /* Time management: use 1/20 of remaining time + half the increment */
    if (!lim->infinite && lim->movetime == 0) {
        int time_left = (b->side == WHITE) ? lim->wtime : lim->btime;
        int inc       = (b->side == WHITE) ? lim->winc  : lim->binc;
        si.allotted_ms = (time_left / 20) + (inc / 2);
        if (si.allotted_ms < 50) si.allotted_ms = 50;
    } else if (lim->movetime > 0) {
        si.allotted_ms = lim->movetime - 50; /* small buffer */
    }

    int  max_depth = (lim->depth > 0) ? lim->depth : 100;
    Move best_move = NULL_MOVE;

    for (int depth = 1; depth <= max_depth; depth++) {
        si.seldepth = 0;
        si.nodes    = 0;
        memset(si.killers, 0, sizeof(si.killers));
        /* Note: history persists across iterations intentionally */

        int score = negamax(b, depth, -INF, INF, 0, &si);

        if (lim->stop) break;

        /* Find best move from TT */
        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) best_move = tte.move;

        /* UCI info line */
        char mv_str[6];
        move_to_str(best_move, mv_str);
        int elapsed_ms = (int)((clock() - si.start_time) * 1000 / CLOCKS_PER_SEC);
        long long nps = elapsed_ms > 0 ? (long long)(si.nodes * 1000ULL / (unsigned)elapsed_ms) : 0;

        if (abs(score) >= MATE_SCORE - MAX_PLY) {
            int mate_in = (score > 0)
                ? (MATE_SCORE - score + 1) / 2
                : -(MATE_SCORE + score + 1) / 2;
            printf("info depth %d seldepth %d score mate %d nodes %llu "
                   "nps %lld time %d pv %s\n",
                   depth, si.seldepth, mate_in,
                   (unsigned long long)si.nodes, nps, elapsed_ms, mv_str);
        } else {
            printf("info depth %d seldepth %d score cp %d nodes %llu "
                   "nps %lld time %d pv %s\n",
                   depth, si.seldepth, score,
                   (unsigned long long)si.nodes, nps, elapsed_ms, mv_str);
        }
        fflush(stdout);

        if (time_up(&si)) break;
    }

    char mv_str[6];
    move_to_str(best_move, mv_str);
    printf("bestmove %s\n", mv_str);
    fflush(stdout);
}

void search_init(void) {
    init_mvv_lva();
}
