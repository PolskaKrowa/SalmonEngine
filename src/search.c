/* clock_gettime + CLOCK_MONOTONIC require _POSIX_C_SOURCE >= 199309L.
 * The Makefile builds with -std=c2x which defines strict ANSI mode,
 * so we have to opt in explicitly. */
#define _POSIX_C_SOURCE 199309L

/*
 * search.c — Move searching (SF 11 refactored, optimised)
 *
 * Architecture:
 *  • Iterative deepening (ID) driver in search()
 *  • negamax() — principal variation search (PVS) with:
 *      – Transposition table cutoffs
 *      – TT mate-score adjustment: value_to_tt / value_from_tt
 *      – TT score refinement of static eval
 *      – Aspiration windows
 *      – Reverse futility pruning / static NMP
 *      – Razoring
 *      – Null-move pruning (NMP, adaptive R)
 *      – ProbCut (depth ≥ 5, raised-beta capture search)
 *      – Futility pruning: improving-adjusted margin via futility_margin()
 *      – Late move pruning (LMP): closed-form futility_move_count()
 *      – Late move reductions (LMR): product-of-logs Reductions table (SF 11)
 *      – Singular extensions (depth ≥ 6, TT-move uniqueness test)
 *      – Improving heuristic
 *      – Killer move heuristic (2 killers per ply)
 *      – Countermove heuristic
 *      – History heuristic: SF-calibrated stat_bonus formula + gravity/decay
 *      – Continuation history: (piece,to) affinity at ply-1
 *      – MVV-LVA + SEE capture ordering
 *      – TT move first
 *  • quiesce() — quiescence search, SEE pruning of losing caps
 *  • SEE (static exchange evaluation)
 *  • Time management: check every 2048 nodes; stop when budget exceeded
 */

#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "nnue.h"
#include "book.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

/* ──────────────────────────────────────────────
 *  sentinel for eval_stack when in check.
 *  Must be outside the normal eval range so the improving heuristic
 *  at ply+2 can detect that ply was an in-check node and avoid a
 *  spurious improving=true.  Choose INT_MIN/2 to stay safe from
 *  any accidental arithmetic that might wrap a plain INT_MIN.
 * ────────────────────────────────────────────── */
#define EVAL_NONE (INT_MIN / 2)

static Move s_root_best_move = NULL_MOVE;

/* ──────────────────────────────────────────────
 *  LMR reduction table — product-of-logs (SF 11 style)
 * ────────────────────────────────────────────── */
static int Reductions[MAX_MOVES];

static void init_reductions(void) {
    Reductions[0] = 0;
    for (int i = 1; i < MAX_MOVES; i++)
        Reductions[i] = (int)(24.8 * log((double)i));
}

/* __attribute__((pure)) — reads Reductions[] global, no side effects */
static inline int lmr_reduction(bool improving, int depth, int move_count)
    __attribute__((pure));

static inline int lmr_reduction(bool improving, int depth, int move_count) {
    int d = (depth     < MAX_MOVES) ? depth      : MAX_MOVES - 1;
    int m = (move_count < MAX_MOVES) ? move_count : MAX_MOVES - 1;
    int r = Reductions[d] * Reductions[m];
    return (r + 511) / 1024 + (!improving && r > 1007);
}

/* __attribute__((const)) — result depends only on arguments */
static inline int stat_bonus(int depth)
    __attribute__((const));
static inline int stat_bonus(int depth) {
    return depth > 15 ? -8 : 19 * depth * depth + 155 * depth - 132;
}

static inline int futility_margin(int depth, bool improving)
    __attribute__((const));
static inline int futility_margin(int depth, bool improving) {
    return 217 * (depth - (int)improving);
}

static inline int futility_move_count(bool improving, int depth)
    __attribute__((const));
static inline int futility_move_count(bool improving, int depth) {
    return (5 + depth * depth) * (1 + (int)improving) / 2 - 1;
}

static inline int value_to_tt(int v, int ply)
    __attribute__((const));
static inline int value_to_tt(int v, int ply) {
    if (v >= MATE_SCORE - MAX_PLY) return v + ply;
    if (v <= -MATE_SCORE + MAX_PLY) return v - ply;
    return v;
}

static inline int value_from_tt(int v, int ply)
    __attribute__((const));
static inline int value_from_tt(int v, int ply) {
    if (v == NO_SCORE)              return NO_SCORE;
    if (v >= MATE_SCORE - MAX_PLY)  return v - ply;
    if (v <= -MATE_SCORE + MAX_PLY) return v + ply;
    return v;
}

/* ──────────────────────────────────────────────
 *  MVV-LVA table  [attacker][victim]
 * ────────────────────────────────────────────── */
int MVV_LVA[6][6];

static void init_mvv_lva(void) {
    static const int victim_val[6]   = { 10, 20, 30, 40, 50, 60 };
    static const int attacker_val[6] = {  1,  2,  3,  4,  5,  6 };
    for (int v = 0; v < 6; v++)
        for (int a = 0; a < 6; a++)
            MVV_LVA[a][v] = victim_val[v] * 10 - attacker_val[a];
}

/* ──────────────────────────────────────────────
 *  Static Exchange Evaluation (SEE)
 *
 *  Returns the material result (in centipawns, from the moving side's
 *  perspective) of all captures that can occur on the target square,
 *  assuming both sides capture with their least-valuable piece first.
 *  Both sides may choose not to recapture if it would lose material.
 *
 *  Based on the "swap algorithm" by Koord/Hyatt (CPW wiki).
 * ────────────────────────────────────────────── */

static const int SEE_VAL[7] = { 100, 320, 330, 500, 900, 20000, 0 };

static Bitboard attacks_to_sq(const Board *b, Square sq, Bitboard occ) {
    return (PAWN_ATTACKS[WHITE][sq]  & b->pieces[BLACK][PAWN])
         | (PAWN_ATTACKS[BLACK][sq]  & b->pieces[WHITE][PAWN])
         | (KNIGHT_ATTACKS[sq]       & (b->pieces[WHITE][KNIGHT] | b->pieces[BLACK][KNIGHT]))
         | (bishop_attacks(sq, occ)  & (b->pieces[WHITE][BISHOP] | b->pieces[BLACK][BISHOP]
                                       | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (rook_attacks(sq, occ)    & (b->pieces[WHITE][ROOK]   | b->pieces[BLACK][ROOK]
                                       | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (KING_ATTACKS[sq]         & (b->pieces[WHITE][KING]   | b->pieces[BLACK][KING]));
}

int see(const Board *b, Move m) {
    Square    from = (Square)MOVE_FROM(m);
    Square    to   = (Square)MOVE_TO(m);
    MoveType  mt   = MOVE_TYPE(m);

    int captured;
    if (mt == MT_EP)
        captured = PAWN;
    else if (b->mailbox[to] != NO_PIECE)
        captured = (int)piece_type(b->mailbox[to]);
    else
        return 0;

    int gain[32];
    int d = 0;
    gain[0] = SEE_VAL[captured];

    Bitboard occ = b->occ[2] ^ SQUARE_BB[from];

    if (mt == MT_EP)
        occ ^= SQUARE_BB[b->ep_sq];

    int   attacker_type = (int)piece_type(b->mailbox[from]);
    Color side = b->side ^ 1;

    /* Optimisation: precompute the set of attackers whose existence doesn't
     * depend on occupancy (pawns, knights, kings). These don't change as
     * pieces are removed from `occ` — they're either attackers or they
     * aren't, period. Only sliders (bishop, rook, queen) need to be
     * recomputed each iteration against the running `occ`.
     *
     * The "behind" technique: when a slider is removed from `occ`, the
     * slider that was previously hidden behind it may now be exposed. We
     * detect this by recomputing slider attacks against the new `occ` and
     * ANDing with the relevant piece bitboards — which is what the original
     * code did. The win here is avoiding the pawn/knight/king recomputation
     * (which the original did inside attacks_to_sq on every iteration).
     */
    Bitboard non_slider_atk =
          (PAWN_ATTACKS[WHITE][to] & b->pieces[BLACK][PAWN])
        | (PAWN_ATTACKS[BLACK][to] & b->pieces[WHITE][PAWN])
        | (KNIGHT_ATTACKS[to] & (b->pieces[WHITE][KNIGHT] | b->pieces[BLACK][KNIGHT]))
        | (KING_ATTACKS[to]   & (b->pieces[WHITE][KING]   | b->pieces[BLACK][KING]));

    /* All piece bitboards combined, for fast lookup of "least valuable
     * attacker of a given color". */
    Bitboard all_pieces[2] = { b->occ[WHITE], b->occ[BLACK] };

    while (true) {
        d++;
        gain[d] = SEE_VAL[attacker_type] - gain[d - 1];

        /* Slider attackers depend on the current occupancy. */
        Bitboard slider_atk =
              (bishop_attacks(to, occ) & (b->pieces[WHITE][BISHOP] | b->pieces[BLACK][BISHOP]
                                          | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
            | (rook_attacks(to, occ)   & (b->pieces[WHITE][ROOK]   | b->pieces[BLACK][ROOK]
                                          | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]));

        Bitboard atk = (non_slider_atk | slider_atk) & occ & all_pieces[side];
        if (!atk) break;

        int pt;
        Bitboard piece_bb = 0;
        for (pt = PAWN; pt <= KING; pt++) {
            piece_bb = b->pieces[side][pt] & atk;
            if (piece_bb) break;
        }
        if (pt > KING) break;

        occ ^= (piece_bb & -piece_bb);
        attacker_type = pt;
        side ^= 1;
    }

    while (--d)
        gain[d - 1] = -((-gain[d - 1] > gain[d]) ? -gain[d - 1] : gain[d]);

    return gain[0];
}

/* ──────────────────────────────────────────────
 *  Move scoring for ordering
 * ────────────────────────────────────────────── */
#define SCORE_TACTICAL_CAP 1050000
#define SCORE_TT_MOVE      1000000
#define SCORE_GOOD_CAP      900000
#define SCORE_KILLER1       800000
#define SCORE_KILLER2       790000
#define SCORE_COUNTER       780000
#define SCORE_QUIET_BASE          0

/*
 * score_move
 *
 *   losing capture  → return see_score           (raw negative value)
 *   even capture    → return SCORE_GOOD_CAP + …   (≥ 900 000)
 *   winning capture → return SCORE_TACTICAL_CAP + … (≥ 1 050 000)
 *
 * This means for captures, scores[i] == see_score when the capture is losing,
 * and scores[i] >> 0 when it is even or winning.  The capture-SEE pruning
 * threshold (-20 * depth * depth, at most -2000 for depth ≤ 10) can never be
 * satisfied by a score ≥ 900 000, so replacing `see(b,m)` with `scores[i]` in
 * that pruning check is semantically identical.  See negamax() below.
 */
static int score_move(const Board *b, Move m, Move tt_move,
                      const Move *killers, Move countermove,
                      int hist[64][64], const int cont_hist[7][64],
                      Move prev_move) {
    if (m == tt_move)    return SCORE_TT_MOVE;

    if (MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP) {
        int from     = MOVE_FROM(m), to = MOVE_TO(m);
        int attacker = (int)piece_type(b->mailbox[from]);
        int victim   = (MOVE_TYPE(m) == MT_EP)
                       ? (int)PAWN
                       : (int)piece_type(b->mailbox[to]);

        int see_score = see(b, m);
        if (see_score >= 150)
            return SCORE_TACTICAL_CAP + MVV_LVA[attacker][victim];
        else if (see_score >= 0)
            return SCORE_GOOD_CAP + MVV_LVA[attacker][victim];
        else
            return see_score; /* losing capture — raw SEE as score (OPT-3 invariant) */
    }

    if (MOVE_IS_PROMO(m)) return SCORE_GOOD_CAP - 1;

    if (m == killers[0])  return SCORE_KILLER1;
    if (m == killers[1])  return SCORE_KILLER2;
    if (m == countermove) return SCORE_COUNTER;

    int h = hist[MOVE_FROM(m)][MOVE_TO(m)];
    if (cont_hist && prev_move != NULL_MOVE) {
        int pt = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
        h += 2 * cont_hist[pt][MOVE_TO(m)];
    }
    return SCORE_QUIET_BASE + h;
}

/* Partial insertion sort: bring the best move to position [idx] */
static Move pick_move(Move *moves, int *scores, int count, int idx) {
    int best = idx;
    for (int i = idx + 1; i < count; i++)
        if (scores[i] > scores[best]) best = i;
    Move tm = moves[idx]; moves[idx] = moves[best]; moves[best] = tm;
    int  ts = scores[idx]; scores[idx] = scores[best]; scores[best] = ts;
    return moves[idx];
}

/* ──────────────────────────────────────────────
 *  History gravity (age / decay the history table)
 * ────────────────────────────────────────────── */
static void decay_history(SearchInfo *si) {
    for (int c = 0; c < 2; c++)
        for (int f = 0; f < 64; f++)
            for (int t = 0; t < 64; t++)
                si->history[c][f][t] /= 2;
}

/* ──────────────────────────────────────────────
 *  Time management
 *
 *  Uses CLOCK_MONOTONIC (wall-clock) instead of clock() (CPU time).
 *  clock() sums across threads, so in lazy-SMP mode N threads would
 *  trigger the time check N× too early. Wall-clock is also what UCI
 *  GUIs expect when they send `wtime`/`btime` in milliseconds.
 * ────────────────────────────────────────────── */
static int64_t monotonic_ms(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t ms = (int64_t)(now.tv_sec - start->tv_sec) * 1000
               + (int64_t)(now.tv_nsec - start->tv_nsec) / 1000000;
    return ms < 0 ? 0 : ms;
}

bool time_up(SearchInfo *si) {
    if (si->limits->infinite || si->limits->stop) return si->limits->stop;
    if (si->allotted_ms <= 0) return false;
    int64_t elapsed = monotonic_ms(&si->start_time_ts);
    return elapsed >= si->allotted_ms;
}

/* ──────────────────────────────────────────────
 *  Quiescence search
 * ────────────────────────────────────────────── */
/* mark quiesce as hot */
__attribute__((hot))
int quiesce(Board *b, int alpha, int beta, int ply, SearchInfo *si) {
    si->nodes++;
    if (ply > si->seldepth) si->seldepth = ply;

    /* Hard ceiling: prevent move_stack overflow and infinite recursion */
    if (ply >= MAX_PLY)
        return evaluate(b);

    bool in_chk = in_check(b);

    int stand_pat = evaluate(b);
    if (!in_chk) {
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MoveList ml;
    if (in_chk)
        gen_moves(b, &ml);
    else
        gen_captures(b, &ml);

    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply < MAX_PLY ? ply : MAX_PLY - 1];
    int (*hist_side)[64]    = si->history[b->side];
    Move cm = (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE)
              ? si->countermove[MOVE_FROM(si->move_stack[ply - 1])]
                               [MOVE_TO  (si->move_stack[ply - 1])]
              : NULL_MOVE;

    Move prev_qs = (ply > 0) ? si->move_stack[ply - 1] : NULL_MOVE;
    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], NULL_MOVE,
                                killers_ply, cm, hist_side,
                                si->cont_hist, prev_qs);

    int legal_count = 0;

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);

        if (!in_chk) {
            if (scores[i] < 0)
                break;

            int captured_val = (MOVE_TYPE(m) == MT_EP)
                               ? SEE_VAL[PAWN]
                               : (b->mailbox[MOVE_TO(m)] != NO_PIECE
                                  ? SEE_VAL[(int)piece_type(b->mailbox[MOVE_TO(m)])]
                                  : 0);
            if (stand_pat + captured_val + 200 < alpha)
                continue;
        }

        /* Perft-style legality: make the move, then check if the side that
         * just moved left its king in check. This replaces the old
         * is_legal(b, m) which did a redundant make+unmake before the real
         * make_move call below — saving one full make/unmake per legal move. */
        si->move_stack[ply] = m;
        make_move(b, m);
        Color mover = b->side ^ 1;
        if (is_square_attacked(b, (Square)bb_lsb(b->pieces[mover][KING]), b->side)) {
            unmake_move(b);
            continue;
        }
        legal_count++;

        int score = -quiesce(b, -beta, -alpha, ply + 1, si);
        unmake_move(b);

        /* stop flag is rarely set in the hot path */
        if (__builtin_expect(si->limits->stop, 0)) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    if (in_chk && legal_count == 0)
        return -MATE_SCORE + ply;

    return alpha;
}

/* ──────────────────────────────────────────────
 *  Negamax alpha-beta (PVS) — the main search
 * ────────────────────────────────────────────── */
/* mark negamax as hot */
__attribute__((hot))
int negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si, Move excluded) {
    /* power-of-2 modulo → bitwise AND.
     * GCC already performs this transform at -O3, but the explicit form
     * documents intent and removes any ambiguity for readers and tools. */
    /* ── Mate-distance pruning ──
     *
     *  Cheap pre-step: if alpha is already at a mate score closer than any
     *  descendant could deliver, no point searching further. Symmetric for
     *  beta. Cuts entire subtrees in mate sequences.
     */
    int alpha0 = alpha;
    int beta0  = beta;
    if (alpha < -MATE_SCORE + MAX_PLY) alpha = -MATE_SCORE + ply;
    if (beta  >  MATE_SCORE - MAX_PLY) beta  =  MATE_SCORE - ply - 1;
    if (alpha >= beta) return alpha;

    if ((si->nodes & 2047) == 0 && time_up(si)) {
        si->limits->stop = true;
        return 0;
    }

    si->nodes++;
    bool root = (ply == 0);
    bool pv   = (beta0 - alpha0 > 1);

    if (ply >= MAX_PLY - 1)
        return quiesce(b, alpha, beta, ply, si);

    /* ── Draw detection ── */
    if (!root) {
        if (b->halfmove >= 100) return DRAW_SCORE;

        for (int i = b->hist_idx - 2; i >= 0; i -= 2) {
            if (b->history[i].hash == b->hash) return DRAW_SCORE;
            if (b->history[i].halfmove == 0) break;
        }
    }

    /* ── TT probe ──
     *
     *  When `excluded` is set (singular-extension verification search),
     *  we XOR the hash with an exclusion key so the probe lands in a
     *  distinct slot from the parent's. This prevents the parent's TT
     *  entry (which holds a bound FOR the excluded move) from short-
     *  circuiting the singularity test. Storage is also keyed this way
     *  so the excluded-search bound doesn't pollute the main TT slot.
     */
    uint64_t probe_key = (excluded != NULL_MOVE)
        ? (b->hash ^ ZKEYS.exclusion[MOVE_FROM(excluded)][MOVE_TO(excluded)])
        : b->hash;

    TTEntry tte;
    bool    tt_hit  = tt_probe(probe_key, &tte);
    Move    tt_move = tt_hit ? tte.move : NULL_MOVE;

    /* Don't return the excluded move as a TT move — by definition we are
     * testing whether the position without it is singular. */
    if (excluded != NULL_MOVE && tt_move == excluded)
        tt_move = NULL_MOVE;

    if (tt_hit && !pv && tte.depth >= depth) {
        int tt_score = value_from_tt(tte.score, ply);
        if (tte.flag == TT_EXACT) return tt_score;
        if (tte.flag == TT_LOWER && tt_score >= beta)  return tt_score;
        if (tte.flag == TT_UPPER && tt_score <= alpha) return tt_score;
    }

    if (root && tt_move == NULL_MOVE && s_root_best_move != NULL_MOVE)
        tt_move = s_root_best_move;

    if (depth <= 0)
        return quiesce(b, alpha, beta, ply, si);

    /* ── In-check detection — computed BEFORE evaluate() ──
     *
     * when the side to move is in check, every pruning path that uses
     * static_eval is already gated on !in_chk.  Calling evaluate() at in-check
     * nodes is therefore wasted work.  We skip it and store EVAL_NONE in
     * eval_stack so the improving heuristic at ply+2 correctly treats the
     * intervening in-check node as "not comparable" rather than "clearly worse."
     */
    bool in_chk = in_check(b);

    /* ── Static evaluation ── */
    int static_eval;
    if (__builtin_expect(!in_chk, 1)) {
        static_eval = evaluate(b);

        /* TT score refinement.
         *
         * The raw static eval is a heuristic estimate; the TT score is a
         * search-verified bound for this exact position.  When the TT entry
         * provides a bound that is more accurate than the static eval (i.e.
         * when the score lies outside what the static eval would suggest),
         * use the TT score as the working eval for pruning decisions.  This
         * improves the accuracy of RFP, razoring, and NMP at no extra cost.
         */
        if (tt_hit) {
            int tt_score = value_from_tt(tte.score, ply);
            if (tt_score != NO_SCORE) {
                if ( tte.flag == TT_EXACT
                  || (tte.flag == TT_LOWER && tt_score > static_eval)
                  || (tte.flag == TT_UPPER && tt_score < static_eval))
                    static_eval = tt_score;
            }
        }

        si->eval_stack[ply] = static_eval;
    } else {
        /* In check: store EVAL_NONE so ply+2's improving check is clean. */
        static_eval = EVAL_NONE;
        si->eval_stack[ply] = EVAL_NONE;
    }

    /* ── Improving heuristic ──
     *
     * follow-up: guard against EVAL_NONE in eval_stack[ply-2].
     * If that ply was an in-check node we stored EVAL_NONE; comparing
     * against it would give a spurious improving=true result.
     */
    bool improving = (!in_chk && ply >= 2
                      && si->eval_stack[ply - 2] != EVAL_NONE
                      && static_eval > si->eval_stack[ply - 2]);

    if (!in_chk && !pv) {
        /* ── Reverse Futility Pruning (Static Null-Move Pruning) ── */
        if (depth <= 8 && abs(beta) < MATE_SCORE - MAX_PLY) {
            int rfp_margin = (improving ? 60 : 80) * depth;
            if (static_eval - rfp_margin >= beta)
                return (static_eval + beta) / 2;
        }

        /* ── Razoring ── */
        if (depth <= 3 && abs(alpha) < MATE_SCORE - MAX_PLY) {
            int razor_margin = 300 + 60 * depth;
            if (static_eval + razor_margin <= alpha) {
                int q = quiesce(b, alpha, beta, ply, si);
                if (q <= alpha) return q;
            }
        }
    }

    /* pre-compute whether the previous ply has a real move.
     * This expression appeared five times in the original negamax body.
     * Computing it once saves repeated ply comparisons and NULL_MOVE checks. */
    bool prev_move_exists = (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE);

    /* ── Null-Move Pruning (NMP) ── */
    bool do_nmp = !pv && !in_chk && depth >= 3
               && static_eval >= beta
               && bb_popcount(b->occ[b->side]
                              & ~b->pieces[b->side][PAWN]
                              & ~b->pieces[b->side][KING]) > 0
               && bb_popcount(b->occ[b->side ^ 1]
                              & ~b->pieces[b->side ^ 1][PAWN]
                              & ~b->pieces[b->side ^ 1][KING]) > 0;

    if (do_nmp) {
        int R = 3 + depth / 6
                  + (((static_eval - beta) / 200) < 3
                     ? (static_eval - beta) / 200 : 3);

        make_null_move(b);
        si->move_stack[ply] = NULL_MOVE;
        int null_score = -negamax(b, depth - R, -beta, -beta + 1, ply + 1, si, NULL_MOVE);
        unmake_null_move(b);

        if (null_score >= beta) {
            if (null_score >= MATE_SCORE - MAX_PLY) null_score = beta;

            if (depth >= 12) {
                int verify = negamax(b, depth - R, beta - 1, beta, ply, si, NULL_MOVE);
                if (verify < beta) goto after_nmp;
            }
            return null_score;
        }
    }
    after_nmp:;

    /* ── ProbCut ── */
    if (!pv && !in_chk && depth >= 5
        && abs(beta) < MATE_SCORE - MAX_PLY)
    {
        int raised_beta = beta + 189 - 45 * (int)improving;
        if (raised_beta > INF) raised_beta = INF;

        MoveList pc_ml;
        gen_captures(b, &pc_ml);

        int pc_scores[MAX_MOVES];
        for (int i = 0; i < pc_ml.count; i++)
            pc_scores[i] = see(b, pc_ml.moves[i]);

        int pc_count = 0;
        for (int i = 0; i < pc_ml.count && pc_count < 2; i++) {
            Move pcm = pick_move(pc_ml.moves, pc_scores, pc_ml.count, i);

            if (!is_legal(b, pcm)) continue;
            if (see(b, pcm) < raised_beta - static_eval) continue;

            si->move_stack[ply] = pcm;
            make_move(b, pcm);

            int pc_val = -quiesce(b, -(raised_beta), -(raised_beta) + 1,
                                  ply + 1, si);

            if (pc_val >= raised_beta)
                pc_val = -negamax(b, depth - 4, -(raised_beta), -(raised_beta) + 1,
                                  ply + 1, si, NULL_MOVE);

            unmake_move(b);

            /* OPT-6 */
            if (__builtin_expect(si->limits->stop, 0)) return 0;
            if (pc_val >= raised_beta) return pc_val;
            pc_count++;
        }
    }

    /* ── Generate and score all moves ── */
    MoveList ml;
    gen_moves(b, &ml);

    /* use pre-computed prev_move_exists */
    Move cm = prev_move_exists
              ? si->countermove[MOVE_FROM(si->move_stack[ply - 1])]
                               [MOVE_TO  (si->move_stack[ply - 1])]
              : NULL_MOVE;

    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply < MAX_PLY ? ply : MAX_PLY - 1];
    int (*hist_side)[64]    = si->history[b->side];
    Move prev_neg = prev_move_exists ? si->move_stack[ply - 1] : NULL_MOVE;

    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], tt_move,
                                killers_ply, cm, hist_side,
                                si->cont_hist, prev_neg);

    int  best_score  = -INF;
    Move best_move   = NULL_MOVE;
    int  legal_count = 0;
    int  quiet_count = 0;
    int  tt_flag     = TT_UPPER;

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);

        bool is_cap   = MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP;
        bool is_promo = MOVE_IS_PROMO(m);
        bool is_quiet = !is_cap && !is_promo;

        /* ── Pruning for non-PV, non-check quiet moves ──
         *
         *  These pruning decisions are evaluated BEFORE make_move, using
         *  pseudo-legal move info. The legality check is deferred to after
         *  make_move (perft-style), which saves one full make+unmake per
         *  move compared to the old is_legal(b,m) pre-check.
         *
         *  quiet_count is incremented only after a LEGAL quiet move is
         *  confirmed, so the threshold semantics are preserved: the (K+1)th
         *  legal quiet move is pruned when quiet_count >= K at this point.
         */
        if (!pv && !in_chk && is_quiet && best_score > -MATE_SCORE + MAX_PLY) {

            if (depth <= 8) {
                /* Late Move Pruning */
                if (quiet_count >= futility_move_count(improving, depth))
                    continue;

                /* Futility Pruning */
                int fp_margin = futility_margin(depth, improving)
                              + hist_side[MOVE_FROM(m)][MOVE_TO(m)] / 128;
                if (static_eval + fp_margin <= alpha)
                    continue;
            }

            /* SEE-based quiet move pruning (depth <= 5 only).
             *
             *  The previous version called see(b, m), but see() returns 0
             *  for any non-capture (its swap-algorithm loop is entered with
             *  `gain[0]=0` and never finds a positive value), so the check
             *  `see(b, m) < -40 * depth` was always false and this pruning
             *  was dead code.
             *
             *  Replaced with a cheap hanging-piece test: if the destination
             *  square is attacked by the enemy and the moving piece is not
             *  defended there (or is less valuable than the least valuable
             *  attacker), the move hangs material and can be pruned at low
             *  depth.
             */
            if (depth <= 5) {
                Square to = MOVE_TO(m);
                Bitboard attackers = attacks_to_sq(b, to, b->occ[2]);
                Bitboard enemy_atk = attackers & b->occ[b->side ^ 1];
                if (enemy_atk) {
                    Bitboard occ_without = b->occ[2] ^ SQUARE_BB[MOVE_FROM(m)];
                    Bitboard defenders = attacks_to_sq(b, to, occ_without)
                                         & b->occ[b->side];
                    int mover_val = SEE_VAL[(int)piece_type(b->mailbox[MOVE_FROM(m)])];
                    int lv_enemy = 6;
                    for (int pt = PAWN; pt <= KING; pt++) {
                        if (b->pieces[b->side ^ 1][pt] & enemy_atk) {
                            lv_enemy = pt; break;
                        }
                    }
                    if (!defenders && SEE_VAL[lv_enemy] < mover_val
                        && SEE_VAL[lv_enemy] - mover_val < -40 * depth)
                        continue;
                }
            }
        }

        /* ── SEE pruning for captures ──
         *
         *  scores[i] for losing captures already equals the raw SEE value
         *  (score_move returns it verbatim).  For even/winning captures
         *  scores[i] >= 900 000, so the threshold (at most -2000 for
         *  depth <= 10) is never satisfied — identical semantics, zero
         *  extra SEE calls.
         */
        if (!pv && is_cap && depth <= 10
            && scores[i] < -20 * depth * depth)
            continue;

        /* ── Singular Extension ──
         *
         *  Bug fix: the previous implementation set `tt_move = NULL_MOVE`
         *  locally, but the recursive negamax call re-probed the TT with
         *  the same key (b->hash) and found the same TT entry — so the
         *  excluded move was searched first and the singularity test was
         *  effectively a no-op.
         *
         *  Now we pass `excluded = m` to the recursive call, which causes
         *  it to probe a distinct TT slot keyed by b->hash ^ exclusion[m].
         *  The bound stored at that slot represents the position WITHOUT
         *  the excluded move, which is what singularity testing requires.
         */
        int extension = 0;
        if (depth >= 6
            && m == tt_move
            && !root
            && tt_move != NULL_MOVE
            && tte.move != NULL_MOVE
            && abs(tte.score) < MATE_SCORE - MAX_PLY
            && (tte.flag & TT_LOWER)
            && tte.depth >= depth - 3
            && excluded == NULL_MOVE)   /* don't recurse into singularity */
        {
            int sing_beta  = tte.score - 2 * depth;
            int half_depth = depth / 2;

            int sing_val = negamax(b, half_depth, sing_beta - 1, sing_beta,
                                   ply, si, m);

            if (__builtin_expect(si->limits->stop, 0)) return 0;

            if (sing_val < sing_beta) {
                extension = 1;
            } else if (sing_val >= beta) {
                /* Multicut: the excluded move is NOT singular AND even
                 * without it we fail high — a strong cutoff signal. */
                return sing_beta;
            }
        }

        /* ── Make the move ── */
        si->move_stack[ply] = m;
        make_move(b, m);

        /* ── Perft-style legality check ──
         *
         *  After make_move, the side that just moved is b->side ^ 1.
         *  If their king is attacked by the new side to move (b->side),
         *  the move was illegal — unmake and skip. This eliminates the
         *  separate is_legal(b, m) make+unmake that the old code did
         *  BEFORE the real make_move, halving make/unmake work per move.
         */
        Color mover = b->side ^ 1;
        if (is_square_attacked(b, (Square)bb_lsb(b->pieces[mover][KING]), b->side)) {
            unmake_move(b);
            continue;
        }

        legal_count++;
        if (is_quiet) quiet_count++;

        int score;
        int new_depth = depth - 1 + extension;

        if (in_check(b) && extension == 0) new_depth++;

        /* ── TT prefetch for the child node ──
         *
         *  Issue a non-temporal prefetch hint for the child position's TT
         *  slot before the recursive call. The TT is the largest random-
         *  access structure in the engine; prefetching hides its memory
         *  latency on deep searches. b->hash is updated by make_move above;
         *  the child's hash is the post-move hash (before the next move
         *  flips the side-to-move bit, which negamax's first move will do).
         *  Actually, the child's hash IS b->hash right now (make_move has
         *  finalized it). Prefetch the slot it will probe.
         */
        tt_prefetch(b->hash);

        /* ── Late Move Reductions (LMR) ── */
        bool do_lmr = legal_count > 3
                   && depth >= 3
                   && is_quiet
                   && !in_chk
                   && m != tt_move;

        int reduction = 0;
        if (do_lmr) {
            reduction = lmr_reduction(improving, depth, legal_count);

            if (!improving)          reduction++;
            if (pv)                  reduction--;
            if (m == cm)             reduction--;
            if (m == killers_ply[0]
             || m == killers_ply[1]) reduction--;

            /* use pre-computed prev_move_exists
             *
             *  Bug fix: this code runs AFTER make_move(b, m), so the moving
             *  piece is now on MOVE_TO(m) — MOVE_FROM(m) is empty (NO_PIECE).
             *  Previously piece_type(NO_PIECE) returned PAWN (since NO_PIECE
             *  mod 6 == 0), so cont_hist was effectively only ever indexed
             *  by [PAWN][prev_to], and the (piece_type,to) affinity signal
             *  was lost. Use MOVE_TO(m) instead.
             */
            if (prev_move_exists) {
                int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                int cur_pt  = (int)piece_type(b->mailbox[MOVE_TO(m)]);
                int ch = si->cont_hist[cur_pt][prev_to];
                if (ch > 0) reduction--;
                else if (ch < 0) reduction++;
            }

            if (reduction < 1)          reduction = 1;
            if (reduction > new_depth)  reduction = new_depth;
        }

        /* ── PVS / LMR search ── */
        if (legal_count == 1) {
            score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si, NULL_MOVE);
        } else {
            score = -negamax(b, new_depth - reduction,
                             -alpha - 1, -alpha, ply + 1, si, NULL_MOVE);

            if (score > alpha && (reduction > 0 || pv)) {
                score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si, NULL_MOVE);
            }
        }

        unmake_move(b);

        if (__builtin_expect(si->limits->stop, 0)) return 0;

        if (score > best_score) {
            best_score = score;
            best_move  = m;
        }
        if (score > alpha) {
            alpha   = score;
            tt_flag = TT_EXACT;

            if (score >= beta) {
                if (is_quiet) {
                    /* Killers */
                    if (m != killers_ply[0]) {
                        si->killers[ply][1] = si->killers[ply][0];
                        si->killers[ply][0] = m;
                    }

                    /* Countermove — use pre-computed prev_move_exists */
                    if (prev_move_exists) {
                        si->countermove
                            [MOVE_FROM(si->move_stack[ply - 1])]
                            [MOVE_TO  (si->move_stack[ply - 1])] = m;
                    }

                    /* History bonus */
                    int bonus = stat_bonus(depth);
                    int *h = &si->history[b->side][MOVE_FROM(m)][MOVE_TO(m)];
                    *h += bonus - (*h) * bonus / 16384;

                    /* Continuation history update.
                     *
                     *  Bug fix: this code runs AFTER unmake_move(b), so the
                     *  moving piece is back on MOVE_FROM(m) — MOVE_TO(m) is
                     *  empty (NO_PIECE). Previously piece_type(NO_PIECE)
                     *  returned PAWN, so every cutoff landed in
                     *  cont_hist[PAWN][prev_to] and the per-piece-type signal
                     *  was lost. Use MOVE_FROM(m) here.
                     */
                    if (prev_move_exists) {
                        int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                        int cur_pt  = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
                        si->cont_hist[cur_pt][prev_to] += bonus
                            - si->cont_hist[cur_pt][prev_to] * bonus / 16384;
                    }
                }

                tt_store(probe_key, value_to_tt(beta, ply), best_move, depth, TT_LOWER, ply);
                return beta;
            }
        }
    }

    /* ── Terminal node detection ── */
    if (legal_count == 0)
        return in_chk ? (-MATE_SCORE + ply) : DRAW_SCORE;

    tt_store(probe_key, value_to_tt(best_score, ply), best_move, depth, tt_flag, ply);
    return best_score;
}

/* ──────────────────────────────────────────────
 *  Iterative deepening driver with aspiration windows (single thread)
 *
 *  Each thread in a lazy-SMP pool runs this independently. They share:
 *    • the TT (global, lock-free for our read-then-write pattern)
 *    • the SearchLimits (read-only flags like `stop`)
 *  Each thread has its OWN SearchInfo (history, killers, node count).
 *
 *  Lazy-SMP works because alpha-beta is unstable: tiny differences in
 *  move ordering (caused by race conditions on TT writes) lead threads
 *  to search different subtrees. When one thread finds a cutoff and
 *  writes it to the TT, the other threads benefit on their next probe.
 *
 *  Thread 0 (the main thread) is the "printer" — only it emits `info`
 *  and `bestmove` lines. Helper threads run silently and contribute
 *  through the shared TT.
 *
 *  `thread_idx` is used to slightly perturb each thread's search depth
 *  so they don't all finish the same iteration at the same instant.
 *  Concretely: helper threads search at depth + (thread_idx & 1) to
 *  give some threads a head start on the next iteration while others
 *  complete the current one.
 * ────────────────────────────────────────────── */
extern void move_to_str(Move m, char *out);

/* Shared between threads: the best move found so far across all threads. */
static volatile Move g_shared_best_move = NULL_MOVE;
static volatile int  g_shared_best_score = 0;

static void search_single_thread(Board *b, SearchLimits *lim, int thread_idx) {
    SearchInfo si;
    memset(&si, 0, sizeof(si));
    si.limits = lim;
    clock_gettime(CLOCK_MONOTONIC, &si.start_time_ts);
    si.start_time = clock();  /* backwards-compat for any legacy callers */

    /* Each thread initialises its own thread-local NNUE accumulator stack.
     * The stack is allocated lazily on first call to nnue_acc_reset. */
    nnue_acc_reset(b);

    /* Time budget allocation. */
    if (lim->infinite || lim->depth > 0) {
        si.allotted_ms = 0;
    } else if (lim->movetime > 0) {
        si.allotted_ms = lim->movetime - 50;
        if (si.allotted_ms < 1) si.allotted_ms = 1;
    } else {
        int time_left = (b->side == WHITE) ? lim->wtime : lim->btime;
        int inc       = (b->side == WHITE) ? lim->winc  : lim->binc;
        si.allotted_ms = (time_left / 20) + (inc / 2);
        if (si.allotted_ms < 50) si.allotted_ms = 50;
    }

    int  max_depth = (lim->depth > 0) ? lim->depth : 100;
    Move best_move = NULL_MOVE;
    int  prev_score = 0;

    /* Book probe — only thread 0 does this (it returns immediately and
     * emits the bestmove; helper threads also return immediately so they
     * don't fight over the TT). */
    if (thread_idx == 0) {
        Move bm = book_probe(b);
        if (bm != NULL_MOVE) {
            char mv_str[6];
            move_to_str(bm, mv_str);
            printf("info depth 0 seldepth 0 score cp 30 nodes 0 nps 0 time 0 pv %s\n",
                   mv_str);
            printf("bestmove %s\n", mv_str);
            fflush(stdout);
            g_shared_best_move = bm;
            return;
        }
    } else {
        /* Helper threads skip the book (thread 0 will return early if book hit). */
        Move bm = book_probe(b);
        if (bm != NULL_MOVE) {
            g_shared_best_move = bm;
            return;
        }
    }

    if (thread_idx == 0) s_root_best_move = NULL_MOVE;

    for (int depth = 1; depth <= max_depth; depth++) {
        /* All threads search at the same nominal depth. Lazy-SMP gets its
         * speedup from TT sharing and the natural non-determinism of
         * alpha-beta (tiny differences in move ordering cause threads to
         * explore different subtrees). Depth perturbation was tried but
         * actually hurt — helpers searching deeper iterations took too
         * long to contribute useful TT entries back to thread 0. */
        int eff_depth = depth;

        si.seldepth = 0;
        struct timespec iter_start_ts;
        clock_gettime(CLOCK_MONOTONIC, &iter_start_ts);
        uint64_t nodes_at_iter_start = si.nodes;
        if (depth > 1) decay_history(&si);

        int score;

        if (eff_depth >= 4) {
            int delta = 25;
            int asp_alpha = prev_score - delta;
            int asp_beta  = prev_score + delta;

            while (true) {
                score = negamax(b, eff_depth, asp_alpha, asp_beta, 0, &si, NULL_MOVE);

                if (lim->stop) goto done;

                if (score <= asp_alpha) {
                    asp_alpha = (score - delta < -INF) ? -INF : score - delta;
                    delta    *= 2;
                } else if (score >= asp_beta) {
                    asp_beta  = (score + delta >  INF) ?  INF : score + delta;
                    delta    *= 2;
                } else {
                    break;
                }

                if (delta > 100) {
                    score = negamax(b, eff_depth, -INF, INF, 0, &si, NULL_MOVE);
                    break;
                }
            }
        } else {
            score = negamax(b, eff_depth, -INF, INF, 0, &si, NULL_MOVE);
        }

        if (lim->stop) break;
        prev_score = score;

        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) best_move = tte.move;

        if (best_move != NULL_MOVE) {
            if (thread_idx == 0) s_root_best_move = best_move;
            /* Publish to the shared best-move slot so other threads can
             * observe our progress (and so the main thread can fall back
             * to a helper's best move if thread 0's iteration was incomplete). */
            g_shared_best_move = best_move;
            g_shared_best_score = score;
        }

        /* Only thread 0 prints progress. Helper threads run silent. */
        if (thread_idx == 0) {
            char mv_str[6];
            move_to_str(best_move, mv_str);
            int elapsed_ms = (int)monotonic_ms(&si.start_time_ts);
            int iter_ms    = (int)monotonic_ms(&iter_start_ts);
            long long nps  = iter_ms > 0
                             ? (long long)((si.nodes - nodes_at_iter_start) * 1000ULL / (unsigned)iter_ms)
                             : 0;

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
        }

        if (time_up(&si)) {
            lim->stop = true;  /* tell helper threads to stop too */
            break;
        }
    }

done:
    /* Deactivate this thread's accumulator maintenance. Each thread has
     * its own thread-local g_acc_active flag, so this is safe. */
    nnue_acc_deactivate();
    /* Helper threads just return; thread 0 will print bestmove below. */
}

/* ──────────────────────────────────────────────
 *  Lazy-SMP wrapper
 * ────────────────────────────────────────────── */
#include <pthread.h>

typedef struct {
    Board        *b;
    SearchLimits *lim;
    int           thread_idx;
} ThreadArg;

static void *worker_thread_main(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;
    /* Each worker gets its own board copy so make/unmake don't race. */
    Board local_board = *ta->b;
    search_single_thread(&local_board, ta->lim, ta->thread_idx);
    return NULL;
}

void search(Board *b, SearchLimits *lim) {
    int n_threads = lim->threads;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 256) n_threads = 256;

    g_shared_best_move = NULL_MOVE;
    g_shared_best_score = 0;

    if (n_threads == 1) {
        search_single_thread(b, lim, 0);
    } else {
        /* Spawn n_threads-1 helpers; main thread runs as thread 0 in-place
         * so it can print info lines without going through pthread_join. */
        pthread_t *tids = (pthread_t *)calloc((size_t)n_threads - 1, sizeof(pthread_t));
        ThreadArg *args = (ThreadArg *)calloc((size_t)n_threads - 1, sizeof(ThreadArg));
        if (!tids || !args) {
            free(tids); free(args);
            search_single_thread(b, lim, 0);
            return;
        }

        for (int i = 1; i < n_threads; i++) {
            args[i - 1].b = b;
            args[i - 1].lim = lim;
            args[i - 1].thread_idx = i;
            if (pthread_create(&tids[i - 1], NULL, worker_thread_main, &args[i - 1]) != 0) {
                /* Spawn failed — run that thread's work inline by simply
                 * running with fewer threads. Mark the slot so join skips it. */
                tids[i - 1] = 0;
            }
        }

        /* Main thread runs as thread 0 in-place. */
        search_single_thread(b, lim, 0);

        /* Signal helpers to stop (in case they haven't noticed yet) and join. */
        lim->stop = true;
        for (int i = 1; i < n_threads; i++) {
            if (tids[i - 1] != 0) pthread_join(tids[i - 1], NULL);
        }
        free(tids);
        free(args);
    }

    /* Print bestmove (only the main thread reaches here). If thread 0's
     * search produced no best move (e.g., it was interrupted very early),
     * fall back to whatever a helper thread published. */
    Move final_move = g_shared_best_move;
    if (final_move == NULL_MOVE) {
        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) final_move = tte.move;
    }
    {
        char mv_str[6];
        move_to_str(final_move, mv_str);
        printf("bestmove %s\n", mv_str);
        fflush(stdout);
    }
}

void search_init(void) {
    init_mvv_lva();
    init_reductions();
    book_init();
}