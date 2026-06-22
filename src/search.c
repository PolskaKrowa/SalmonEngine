/*
 * search.c — Move searching (SF 11 base, SF 14+ augmentations)
 *
 * Architecture:
 *  • Iterative deepening (ID) driver in search()
 *  • negamax() — principal variation search (PVS) with:
 *      – Transposition table cutoffs
 *      – TT mate-score adjustment: value_to_tt / value_from_tt
 *      – TT score refinement of static eval
 *      – Aspiration windows
 *      – Internal Iterative Reductions (IIR): depth-1 when no TT move
 *      – Reverse futility pruning / static NMP
 *      – Razoring
 *      – Null-move pruning (NMP, adaptive R, improving-aware)
 *      – ProbCut (depth ≥ 5, raised-beta capture search)
 *      – Futility pruning: improving-adjusted margin via futility_margin()
 *      – Late move pruning (LMP): closed-form futility_move_count()
 *      – Late move reductions (LMR): product-of-logs Reductions table (SF 11)
 *      – Singular extensions (depth ≥ 6, TT-move uniqueness test)
 *      – Improving heuristic
 *      – Killer move heuristic (2 killers per ply)
 *      – Countermove heuristic
 *      – History heuristic: SF-calibrated stat_bonus formula + gravity/decay
 *      – History malus: stat_hat_penalty applied to non-cutoff quiet moves
 *        (SF 14+).  Each quiet move that fails to produce a cutoff at a
 *        cut-node receives a negative bonus, complementing the positive
 *        bonus given to the move that does cut off.
 *      – Continuation history: two tables, (piece,to) at ply-1 AND ply-2
 *      – MVV-LVA + SEE capture ordering
 *      – TT move first
 *  • quiesce() — quiescence search, SEE pruning of losing caps
 *  • SEE (static exchange evaluation)
 *  • Time management: check every 2048 nodes; stop when budget exceeded
 */

#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "book.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#ifdef EVAL_DEBUG
#  include "eval_debug.h"
#endif

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

/*
 * stat_hat_penalty — magnitude of the *negative* bonus applied to quiet
 * moves that failed to produce a cutoff at a cut-node (SF 14+ "history
 * malus").  Smaller in absolute value than stat_bonus(depth) so the
 * signal stays softer than the positive signal — the goal is to gently
 * demote moves that have repeatedly failed, not to obliterate them.
 *
 * SF's formula: `-(depth >= 12 ? 24*depth - 176 : 32*depth - 16)`.
 * We clamp the lower end so depth-1 doesn't apply a near-zero malus
 * that just adds noise.
 */
static inline int stat_hat_penalty(int depth)
    __attribute__((const));
static inline int stat_hat_penalty(int depth) {
    if (depth < 2) return -8;
    return depth >= 12 ? -(24 * depth - 176) : -(32 * depth - 16);
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

    while (true) {
        d++;
        gain[d] = SEE_VAL[attacker_type] - gain[d - 1];

        /* removed dead stub `if (…) {};` that was here.
         * The backward pass below already implements the correct
         * "don't recapture if it loses material" logic. */

        Bitboard atk = attacks_to_sq(b, to, occ) & occ & b->occ[side];
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
                      int hist[64][64],
                      const int16_t cont_hist_p1[CONT_HIST_PIECES][64],
                      const int16_t cont_hist_p2[CONT_HIST_PIECES][64],
                      const Move *prev_moves, int n_prev) {
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

    /* Sum:
     *   - butterfly history (from,to) for our side
     *   - ply-1 continuation history (cur_piece_type, prev_to)    -- corrected
     *   - ply-2 continuation history (cur_piece_type, prev_prev_to)
     *
     * The previous cont_hist lookup was bugged (it used the wrong mailbox
     * index after make_move).  We compute the moving piece type here,
     * from the pre-make mailbox, which is correct.
     */
    int h = hist[MOVE_FROM(m)][MOVE_TO(m)];

    int cur_pt = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
    if (cur_pt < 0 || cur_pt >= CONT_HIST_PIECES) cur_pt = PAWN;

    if (n_prev >= 1 && prev_moves[0] != NULL_MOVE) {
        int prev_to = MOVE_TO(prev_moves[0]);
        h += 2 * cont_hist_p1[cur_pt][prev_to];
    }
    if (n_prev >= 2 && prev_moves[1] != NULL_MOVE) {
        int pp_to = MOVE_TO(prev_moves[1]);
        h += 1 * cont_hist_p2[cur_pt][pp_to];
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

    /* Quiescence only uses ply-1 continuation history (the search depth
     * here is so shallow that ply-2 adds noise without signal). */
    Move prev_moves[1];
    int  n_prev = 0;
    if (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE) {
        prev_moves[0] = si->move_stack[ply - 1];
        n_prev = 1;
    }
    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], NULL_MOVE,
                                killers_ply, cm, hist_side,
                                si->cont_hist_p1, si->cont_hist_p2,
                                prev_moves, n_prev);

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

        if (!is_legal(b, m)) continue;
        legal_count++;

        si->move_stack[ply] = m;
        make_move(b, m);
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
int negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si) {
    /* power-of-2 modulo → bitwise AND.
     * GCC already performs this transform at -O3, but the explicit form
     * documents intent and removes any ambiguity for readers and tools. */
    if ((si->nodes & 2047) == 0 && time_up(si)) {
        si->limits->stop = true;
        return 0;
    }

    si->nodes++;
    bool root = (ply == 0);
    bool pv   = (beta - alpha > 1);

    if (ply >= MAX_PLY - 1)
        return quiesce(b, alpha, beta, ply, si);

    /*
     * Prefetch the TT bucket for this position.  The actual tt_probe
     * happens later, after draw detection — the intervening work
     * (especially the in_check() call and history scan) hides the
     * memory latency.
     */
    tt_prefetch(b->hash);

    /* ── Draw detection ── */
    if (!root) {
        if (b->halfmove >= 100) return DRAW_SCORE;

        for (int i = b->hist_idx - 2; i >= 0; i -= 2) {
            if (b->history[i].hash == b->hash) return DRAW_SCORE;
            if (b->history[i].halfmove == 0) break;
        }
    }

    /* ── TT probe ── */
    TTEntry tte;
    bool    tt_hit  = tt_probe(b->hash, &tte);
    Move    tt_move = tt_hit ? tte.move : NULL_MOVE;

    if (tt_hit && !pv && tte.depth >= depth) {
        int tt_score = value_from_tt(tte.score, ply);
        if (tte.flag == TT_EXACT) return tt_score;
        if (tte.flag == TT_LOWER && tt_score >= beta)  return tt_score;
        if (tte.flag == TT_UPPER && tt_score <= alpha) return tt_score;
    }

    if (root && tt_move == NULL_MOVE && s_root_best_move != NULL_MOVE)
        tt_move = s_root_best_move;

    /*
     * ── In-check detection ──────────────────────────────────────────
     *
     * Computed early because IIR, the static-eval gate, and the
     * improving heuristic all need it.  Calling in_check() once here
     * is cheaper than recomputing it for each consumer.
     */
    bool in_chk = in_check(b);

    /*
     * ── Internal Iterative Reductions (IIR) ──────────────────────────
     *
     * SF 14+.  When there is no TT move at an interior (non-root) node,
     * the position is unlikely to be in the PV and we have no good
     * ordering hint.  Reduce depth by 1 instead of doing a full IID
     * search — the next iteration will likely fill the TT anyway.
     *
     * Conditions: depth >= 3, not root, not in check (check extensions
     * dominate here), no TT move.  We keep depth>=3 because at depth 1
     * or 2 the search is so shallow that the reduction would degrade
     * quality without saving meaningful work.
     */
    if (!root && !in_chk && depth >= 3 && tt_move == NULL_MOVE) {
        depth--;
    }

    if (depth <= 0)
        return quiesce(b, alpha, beta, ply, si);

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
        /*
         * Adaptive R (SF 14+ tuning).
         *
         *   base      = 4 + depth/6
         *   eval_term = (static_eval - beta) / 256     (clamped to [0, 3])
         *   non-improving nodes get +1 to R (more aggressive prune)
         *
         * The previous formula used /200 and a hard ceiling of 3, which
         * was a bit too eager in sharp tactical positions.  The new
         * formula matches modern SF defaults and is slightly less
         * aggressive on improving nodes (more reliable).
         */
        int eval_term = (static_eval - beta) / 256;
        if (eval_term < 0) eval_term = 0;
        if (eval_term > 3) eval_term = 3;
        int R = 4 + depth / 6 + eval_term + (int)!improving;
        if (R > depth - 1) R = depth - 1;  /* never reduce to <= 0 */

        make_null_move(b);
        si->move_stack[ply] = NULL_MOVE;
        int null_score = -negamax(b, depth - R, -beta, -beta + 1, ply + 1, si);
        unmake_null_move(b);

        if (null_score >= beta) {
            if (null_score >= MATE_SCORE - MAX_PLY) null_score = beta;

            /* Verification at deep nodes: prevents zugzwang false-positives. */
            if (depth - R >= 6) {
                int verify = negamax(b, depth - R, beta - 1, beta, ply, si);
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
                                  ply + 1, si);

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

    /* Provide up to two previous moves to score_move for continuation
     * history.  prev_moves[0] = ply-1, prev_moves[1] = ply-2. */
    Move prev_moves[2];
    int  n_prev = 0;
    if (ply >= 1 && si->move_stack[ply - 1] != NULL_MOVE) {
        prev_moves[0] = si->move_stack[ply - 1];
        n_prev = 1;
        if (ply >= 2 && si->move_stack[ply - 2] != NULL_MOVE) {
            prev_moves[1] = si->move_stack[ply - 2];
            n_prev = 2;
        }
    }

    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], tt_move,
                                killers_ply, cm, hist_side,
                                si->cont_hist_p1, si->cont_hist_p2,
                                prev_moves, n_prev);

    int  best_score  = -INF;
    Move best_move   = NULL_MOVE;
    int  legal_count = 0;
    int  quiet_count = 0;
    int  tt_flag     = TT_UPPER;

    /*
     * History-malus bookkeeping.
     *
     * For each quiet move we searched at this node, remember the (move,
     * piece-type, from, to) tuple so that, when a *different* move
     * eventually produces a beta cutoff, we can apply a negative bonus
     * to every searched-but-non-cutoff quiet move (SF 14+ "history
     * malus").  This is the dual of the positive history bonus and
     * substantially speeds up move ordering at recurring cut-nodes.
     *
     * 64 entries is more than enough — we cap the searched-quiet list
     * at futility_move_count(improving, depth) entries via LMP anyway.
     */
    Move  quiets_tried[MAX_MOVES];
    int   quiet_pts  [MAX_MOVES];   /* piece type of the mover, for cont_hist */
    int   n_quiets_tried = 0;

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);
        if (!is_legal(b, m)) continue;

        bool is_cap   = MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP;
        bool is_promo = MOVE_IS_PROMO(m);
        bool is_quiet = !is_cap && !is_promo;

        legal_count++;
        if (is_quiet) quiet_count++;

        /* ── Pruning for non-PV, non-check quiet moves ──
         *
         * LMP and futility pruning shared depth <= 8 but were
         * in separate if-blocks, each re-evaluating the guard.  They are
         * now merged into a single block that evaluates `depth <= 8` once.
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

            /* SEE-based quiet move pruning (depth <= 5 only) */
            if (depth <= 5 && see(b, m) < -40 * depth)
                continue;
        }

        /* ── SEE pruning for captures ──
         *
         * the original code re-called see(b, m) here.  We instead
         * use scores[i], which for losing captures already equals the raw
         * SEE value (score_move returns it verbatim).  For even/winning
         * captures scores[i] >= 900 000, so the threshold (at most -2000
         * for depth <= 10) is never satisfied — identical semantics, zero
         * extra SEE calls.
         */
        if (!pv && is_cap && depth <= 10
            && scores[i] < -20 * depth * depth)
            continue;

        /* ── Singular Extension ── */
        int extension = 0;
        if (depth >= 6
            && m == tt_move
            && !root
            && tt_move != NULL_MOVE
            && tte.move != NULL_MOVE
            && abs(tte.score) < MATE_SCORE - MAX_PLY
            && (tte.flag & TT_LOWER)
            && tte.depth >= depth - 3)
        {
            int sing_beta  = tte.score - 2 * depth;
            int half_depth = depth / 2;

            Move saved_tt = tt_move;
            tt_move = NULL_MOVE;

            int sing_val = negamax(b, half_depth, sing_beta - 1, sing_beta,
                                   ply, si);

            tt_move = saved_tt;

            if (__builtin_expect(si->limits->stop, 0)) return 0;

            if (sing_val < sing_beta) {
                extension = 1;
            } else if (sing_beta >= beta) {
                return sing_beta;
            }
        }

        /*
         * Record the moving piece's type BEFORE make_move so we can
         * update continuation history correctly later (after unmake,
         * mailbox[from] is back to the moving piece, but capturing
         * cur_pt once here is clearer and avoids the post-unmake
         * confusion that previously led to a NO_PIECE→PAWN bug).
         */
        int cur_pt = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
        if (cur_pt < 0 || cur_pt >= CONT_HIST_PIECES) cur_pt = PAWN;

        /* Track for history-malus application later. */
        if (is_quiet && n_quiets_tried < MAX_MOVES) {
            quiets_tried[n_quiets_tried] = m;
            quiet_pts  [n_quiets_tried] = cur_pt;
            n_quiets_tried++;
        }

        /* ── Make the move ── */
        si->move_stack[ply] = m;
        make_move(b, m);

        int score;
        int new_depth = depth - 1 + extension;

        if (in_check(b) && extension == 0) new_depth++;

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

            /*
             * Continuation-history adjustment.  Uses BOTH ply-1 and
             * ply-2 cont_hist tables.  Pre-computed cur_pt avoids the
             * post-make mailbox bug that the previous code had.
             */
            if (prev_move_exists) {
                int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                int ch1 = si->cont_hist_p1[cur_pt][prev_to];
                if (ch1 > 0) reduction--;
                else if (ch1 < 0) reduction++;
            }
            if (n_prev >= 2 && prev_moves[1] != NULL_MOVE) {
                int pp_to = MOVE_TO(prev_moves[1]);
                int ch2 = si->cont_hist_p2[cur_pt][pp_to];
                if (ch2 > 0) reduction--;
                else if (ch2 < 0) reduction++;
            }

            if (reduction < 1)          reduction = 1;
            if (reduction > new_depth)  reduction = new_depth;
        }

        /* ── PVS / LMR search ── */
        if (legal_count == 1) {
            score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si);
        } else {
            score = -negamax(b, new_depth - reduction,
                             -alpha - 1, -alpha, ply + 1, si);

            if (score > alpha && (reduction > 0 || pv)) {
                score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si);
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

                    /* History bonus (positive) for the cutoff move. */
                    int bonus = stat_bonus(depth);
                    int *h = &si->history[b->side][MOVE_FROM(m)][MOVE_TO(m)];
                    *h += bonus - (*h) * bonus / 16384;

                    /*
                     * History malus (negative bonus) for every other
                     * quiet move that was searched at this node but did
                     * NOT cause a cutoff.  SF 14+ — speeds up ordering
                     * at recurring cut-nodes by demoting moves that
                     * have repeatedly failed to cut off.
                     */
                    int malus = stat_hat_penalty(depth);
                    for (int j = 0; j < n_quiets_tried; j++) {
                        Move qm = quiets_tried[j];
                        if (qm == m) continue;   /* don't malus the cutoff */
                        int *qh = &si->history[b->side]
                                            [MOVE_FROM(qm)][MOVE_TO(qm)];
                        *qh += malus - (*qh) * malus / 16384;
                    }

                    /*
                     * Continuation-history updates.
                     *
                     * ply-1: bonus for (cur_pt, prev_to)
                     * ply-2: bonus for (cur_pt, prev_prev_to)
                     *
                     * Use the pre-computed cur_pt — avoids the
                     * post-unmake mailbox bug.
                     */
                    if (prev_move_exists) {
                        int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                        int16_t *ch = &si->cont_hist_p1[cur_pt][prev_to];
                        *ch += (int16_t)(bonus - (*ch) * bonus / 16384);
                    }
                    if (n_prev >= 2 && prev_moves[1] != NULL_MOVE) {
                        int pp_to = MOVE_TO(prev_moves[1]);
                        int16_t *ch = &si->cont_hist_p2[cur_pt][pp_to];
                        *ch += (int16_t)(bonus - (*ch) * bonus / 16384);
                    }
                }

                tt_store(b->hash, value_to_tt(beta, ply), best_move, depth, TT_LOWER, ply);
                return beta;
            }
        }
    }

    /* ── Terminal node detection ── */
    if (legal_count == 0)
        return in_chk ? (-MATE_SCORE + ply) : DRAW_SCORE;

    tt_store(b->hash, value_to_tt(best_score, ply), best_move, depth, tt_flag, ply);
    return best_score;
}

/* ──────────────────────────────────────────────
 *  Iterative deepening driver with aspiration windows
 * ────────────────────────────────────────────── */
extern void move_to_str(Move m, char *out);

void search(Board *b, SearchLimits *lim) {
    SearchInfo si;
    memset(&si, 0, sizeof(si));
    si.limits     = lim;
    si.start_time = clock();

    /* Bump TT generation so this search's entries outcompete stale
     * entries from previous searches during replacement. */
    tt_new_search();

#ifdef EVAL_DEBUG
    /* Reset top/bottom lists so each search produces a fresh dump. */
    eval_debug_init();
#endif

    if (!lim->infinite && lim->movetime == 0) {
        int time_left = (b->side == WHITE) ? lim->wtime : lim->btime;
        int inc       = (b->side == WHITE) ? lim->winc  : lim->binc;
        si.allotted_ms = (time_left / 20) + (inc / 2);
        if (si.allotted_ms < 50) si.allotted_ms = 50;
    } else if (lim->movetime > 0) {
        si.allotted_ms = lim->movetime - 50;
    }

    int  max_depth = (lim->depth > 0) ? lim->depth : 100;
    Move best_move = NULL_MOVE;
    int  prev_score = 0;

    {
        Move bm = book_probe(b);
        if (bm != NULL_MOVE) {
            char mv_str[6];
            move_to_str(bm, mv_str);
            printf("info depth 0 seldepth 0 score cp 30 nodes 0 nps 0 time 0 pv %s\n",
                   mv_str);
            printf("bestmove %s\n", mv_str);
            fflush(stdout);
            return;
        }
    }

    s_root_best_move = NULL_MOVE;

    for (int depth = 1; depth <= max_depth; depth++) {
        si.seldepth = 0;
        si.nodes    = 0;
        if (depth > 1) decay_history(&si);

        int score;

        if (depth >= 4) {
            int delta = 25;
            int asp_alpha = prev_score - delta;
            int asp_beta  = prev_score + delta;

            while (true) {
                score = negamax(b, depth, asp_alpha, asp_beta, 0, &si);

                if (lim->stop) goto done;

                if (score <= asp_alpha) {
                    asp_alpha = (asp_alpha - delta < -INF) ? -INF : asp_alpha - delta;
                    delta    *= 2;
                } else if (score >= asp_beta) {
                    asp_beta  = (asp_beta + delta >  INF) ?  INF : asp_beta + delta;
                    delta    *= 2;
                } else {
                    break;
                }

                if (delta > 500) {
                    score = negamax(b, depth, -INF, INF, 0, &si);
                    break;
                }
            }
        } else {
            score = negamax(b, depth, -INF, INF, 0, &si);
        }

        if (lim->stop) break;
        prev_score = score;

        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) best_move = tte.move;

        if (best_move != NULL_MOVE) s_root_best_move = best_move;

        char mv_str[6];
        move_to_str(best_move, mv_str);
        int elapsed_ms = (int)((clock() - si.start_time) * 1000 / CLOCKS_PER_SEC);
        long long nps  = elapsed_ms > 0
                         ? (long long)(si.nodes * 1000ULL / (unsigned)elapsed_ms)
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

        if (time_up(&si)) break;
    }

done:
    {
        char mv_str[6];
        move_to_str(best_move, mv_str);
        printf("bestmove %s\n", mv_str);
        fflush(stdout);
    }

#ifdef EVAL_DEBUG
    eval_debug_dump("eval_debug.txt");
#endif
}

void search_init(void) {
    init_mvv_lva();
    init_reductions();
    book_init();
}