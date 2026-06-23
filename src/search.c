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
#include <time.h>      /* clock_gettime, CLOCK_MONOTONIC */

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
    /* Was: depth > 15 ? -8 : ...  (NEGATIVE bonus for deep cutoffs — wrong sign!)
     * SF 15+: small POSITIVE cap for deep cutoffs. Deep cutoffs are noisier
     * but still worth rewarding.  1594 matches SF 15's stat_bonus cap. */
    return depth > 13 ? 1594 : 19 * depth * depth + 155 * depth - 132;
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

        Bitboard atk = attacks_to_sq(b, to, occ) & occ & b->occ[side];
        if (!atk) break;

        int pt;
        Bitboard piece_bb = 0;
        for (pt = PAWN; pt <= KING; pt++) {
            piece_bb = b->pieces[side][pt] & atk;
            if (piece_bb) break;
        }
        if (pt > KING) break;

        /* King-safety check (SF see_ge pattern):
         *
         * If the king is the next recapturer AND the opponent still has
         * an attacker on the target square after the king "captures",
         * the king's recapture is illegal (the king would be moving
         * into check, typically because an enemy slider was revealed
         * by the previous capture's x-ray).  In that case we stop the
         * swap loop — the king cannot actually recapture, so the
         * previous capture stands. */
        if (pt == KING) {
            Bitboard occ_after_king = occ ^ (piece_bb & -piece_bb);
            Bitboard remaining = attacks_to_sq(b, to, occ_after_king)
                                 & occ_after_king & b->occ[side ^ 1];
            if (remaining) {
                /* King can't recapture safely — current side keeps
                 * the gain from the previous capture.  Don't commit
                 * the king's recapture (don't update occ/attacker). */
                break;
            }
        }

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
 *
 * For quiet moves, the score is the sum of:
 *   - butterfly history [side][from][to]
 *   - 2 × ply-1 continuation history [cur_pt][prev_to]
 *   - 1 × ply-2 continuation history [cur_pt][prev_prev_to]
 *   - 2 × countermove history [prev_pt][prev_to][cur_pt][cur_to]   (NEW)
 *
 * The countermove-history term lets us rank counter candidates beyond
 * the single hard-coded "the previous cutoff reply" that the plain
 * countermove heuristic stores.  It uses the same [piece][to] indexing
 * as continuation history but is keyed on the *previous* move's
 * (piece, to) rather than the previous ply's.
 */
static int score_move(const Board *b, Move m, Move tt_move,
                      const Move *killers, Move countermove,
                      int hist[64][64],
                      const int16_t cont_hist_p1[CONT_HIST_PIECES][64],
                      const int16_t cont_hist_p2[CONT_HIST_PIECES][64],
                      const int16_t counter_hist[CONT_HIST_PIECES][64][CONT_HIST_PIECES][64],
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

    int h = hist[MOVE_FROM(m)][MOVE_TO(m)];

    int cur_pt = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
    if (cur_pt < 0 || cur_pt >= CONT_HIST_PIECES) cur_pt = PAWN;
    int cur_to = MOVE_TO(m);

    if (n_prev >= 1 && prev_moves[0] != NULL_MOVE) {
        /* Decode the previous move's (piece, to).  We need the piece
         * type — but the previous move was already unmade by the time
         * we get here (we're scoring moves at the CURRENT ply).  So
         * we derive the piece type from the previous move's from-square
         * in the current mailbox, which is correct because the previous
         * move has been unmade (the piece is back on its from-square).
         *
         * For null moves (NULL_MOVE), we skip — counter_hist is only
         * updated on real cutoffs. */
        int prev_from = MOVE_FROM(prev_moves[0]);
        int prev_to   = MOVE_TO(prev_moves[0]);
        Piece prev_pc = b->mailbox[prev_from];
        if (prev_pc != NO_PIECE) {
            int prev_pt = (int)piece_type(prev_pc);
            if (prev_pt >= 0 && prev_pt < CONT_HIST_PIECES) {
                h += 2 * cont_hist_p1[cur_pt][prev_to];
                h += 2 * counter_hist[prev_pt][prev_to][cur_pt][cur_to];
            }
        }
    }
    if (n_prev >= 2 && prev_moves[1] != NULL_MOVE) {
        int pp_from = MOVE_FROM(prev_moves[1]);
        int pp_to   = MOVE_TO(prev_moves[1]);
        /* ply-2 prev piece: similarly derived from current mailbox.
         * But after a null move at ply-2, the piece isn't on the
         * from-square — skip in that case. */
        Piece pp_pc = b->mailbox[pp_from];
        /* For ply-2 we can't easily verify the piece is still there
         * (ply-1's move may have moved it).  Just use the ply-2 cont
         * hist table without the counter-hist cross term — the
         * ply-2 counter-hist signal is weak anyway. */
        (void)pp_pc;
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
 *
 *  Wall-clock time via CLOCK_MONOTONIC (not clock() which is CPU time).
 *  Two thresholds:
 *    • allotted_ms (soft): when we'd LIKE to stop.  Used between iterations.
 *    • max_time_ms (hard): when we MUST stop.  Used inside negamax.
 *  When only `depth` is requested, both are 0 → search runs to depth.
 * ────────────────────────────────────────────── */
static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

bool time_up(SearchInfo *si) {
    /* Pondering: search indefinitely until `stop` is set (by `ponderhit`
     * with time conversion, or by a new `position`/`go`/`stop` command). */
    if (si->limits->ponder) return si->limits->stop;
    if (si->limits->infinite) return si->limits->stop;
    if (si->limits->stop)     return true;
    if (si->max_time_ms <= 0) return false;
    long elapsed = now_ms() - si->start_time_ms;
    return elapsed >= si->max_time_ms;
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

    /* Quiescence move scoring.
     *
     * We use the full score_move() for consistency with the main search's
     * move ordering.  Although quiescence is capture-dominated (and a
     * lighter MVV-LVA-only scorer would be faster per-move), the full
     * scorer's SEE-based capture classification (winning/even/losing)
     * produces better move ordering, which reduces the number of nodes
     * searched — a net win in NPS despite the higher per-move cost.
     *
     * The history/killer/countermove tables are passed for the in-check
     * case (where quiet moves are generated and need ordering). */
    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply < MAX_PLY ? ply : MAX_PLY - 1];
    int (*hist_side)[64]    = si->history[b->side];
    Move cm = (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE)
              ? si->countermove[MOVE_FROM(si->move_stack[ply - 1])]
                               [MOVE_TO  (si->move_stack[ply - 1])]
              : NULL_MOVE;

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
                                si->counter_hist,
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
int negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si,
            Move excluded) {
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

    /* ── TT probe ──
     *
     * When `excluded` is set (we are inside a singular-extension
     * verification search), the TT entry was computed WITH the
     * excluded move available, so its bound does not apply to the
     * excluded sub-search.  Disable TT cutoffs and TT move hint in
     * that case (we still probe so we can read the move for ordering
     * of the non-excluded moves, but we don't trust the score).
     */
    TTEntry tte;
    bool    tt_hit  = excluded ? false : tt_probe(b->hash, &tte);
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
        int null_score = -negamax(b, depth - R, -beta, -beta + 1, ply + 1, si, NULL_MOVE);
        unmake_null_move(b);

        if (null_score >= beta) {
            if (null_score >= MATE_SCORE - MAX_PLY) null_score = beta;

            /* Verification at deep nodes: prevents zugzwang false-positives. */
            if (depth - R >= 6) {
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
                                si->counter_hist,
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

        /* Skip the excluded move (singular extension verification). */
        if (m == excluded) continue;

        if (!is_legal(b, m)) continue;

        bool is_cap   = MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP;
        bool is_promo = MOVE_IS_PROMO(m);
        bool is_quiet = !is_cap && !is_promo;

        legal_count++;
        if (is_quiet) quiet_count++;

        /*
         * cur_pt and cur_to: piece type and destination of the move
         * currently being searched.  Computed once here from the
         * pre-make mailbox (so they're available for both pruning
         * decisions and the later LMR/history updates).
         */
        int cur_pt = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
        if (cur_pt < 0 || cur_pt >= CONT_HIST_PIECES) cur_pt = PAWN;
        int cur_to = MOVE_TO(m);

        /* ── Pruning for non-PV, non-check quiet moves ──
         *
         * Two pruning techniques are applied:
         *
         *   1. Late Move Pruning (LMP): prune quiet moves that appear
         *      "late enough" in the move ordering.  The cutoff count
         *      is history-based — a move with a strong history score
         *      is searched even when it appears late, while a move
         *      with a weak score is pruned earlier.  SF 14+.
         *
         *   2. Futility Pruning: prune quiet moves whose static eval
         *      plus margin (modulated by history) is still below alpha.
         *
         * Both techniques are gated on `depth <= 8` and on the node
         * being non-PV, non-check, with no best score yet (so we keep
         * at least one move).
         */
        if (!pv && !in_chk && is_quiet && best_score > -MATE_SCORE + MAX_PLY) {

            if (depth <= 8) {
                /*
                 * History-based LMP.  Combine the move's score across
                 * all available history tables (butterfly + cont_hist
                 * + counter_hist).  A high combined score means the
                 * move is likely good → search it even if it appears
                 * late.  A low score means it's unlikely to cut off →
                 * prune it earlier.
                 *
                 * The threshold grows with depth (more moves searched
                 * at deeper plies) and is tighter when improving
                 * (we're in a good position, so bad moves can be
                 * pruned more aggressively).
                 */
                int move_hist_score = hist_side[MOVE_FROM(m)][MOVE_TO(m)];
                if (prev_move_exists) {
                    int prev_from = MOVE_FROM(si->move_stack[ply - 1]);
                    int prev_to   = MOVE_TO(si->move_stack[ply - 1]);
                    Piece prev_pc = b->mailbox[prev_from];
                    if (prev_pc != NO_PIECE) {
                        int prev_pt = (int)piece_type(prev_pc);
                        if (prev_pt >= 0 && prev_pt < CONT_HIST_PIECES) {
                            move_hist_score += si->cont_hist_p1[cur_pt][prev_to]
                                             + si->counter_hist[prev_pt][prev_to][cur_pt][cur_to];
                        }
                    }
                }

                /*
                 * LMP threshold.  Keep the existing futility_move_count
                 * formula as the base, then EXTEND the threshold for
                 * moves with strong history (search them later than
                 * they would otherwise be pruned).  This is the safer
                 * direction — pruning MORE would risk Elo loss; pruning
                 * LESS (extending good moves) only costs a little time.
                 */
                int base_count = futility_move_count(improving, depth);
                /* Each 1024 of history score lets the move survive
                 * extra late-move slots.  Capped so one super-good
                 * move doesn't blow up the count. */
                int hist_extension = move_hist_score / 1024;
                if (hist_extension > 12) hist_extension = 12;
                int lmp_threshold = base_count + hist_extension;
                if (quiet_count > lmp_threshold)
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

        /* ── Singular Extension ──
         *
         * A move is "singular" if it is clearly better than all other
         * moves at this node.  We test this by running a reduced-depth
         * search with the TT move EXCLUDED — if that search fails low
         * (no other move can reach sing_beta = tt_score - 2*depth),
         * the TT move is singular and deserves an extension.
         *
         * The `excluded` parameter is threaded through negamax so the
         * recursive call skips the excluded move in its move loop and
         * disables TT cutoffs (which would be wrong since the TT value
         * was computed with the excluded move available).
         *
         * Multi-cut: if the excluded search FAILS HIGH over sing_beta
         * AND sing_beta >= beta, multiple moves are good enough — we
         * can prune the entire node.  This folds the classic
         * Björnsson-Marsland MC-pruning into the singular verification.
         */
        int extension = 0;
        if (depth >= 6
            && m == tt_move
            && !root
            && tt_move != NULL_MOVE
            && excluded == NULL_MOVE
            && tte.move != NULL_MOVE
            && abs(tte.score) < MATE_SCORE - MAX_PLY
            && (tte.flag & TT_LOWER)
            && tte.depth >= depth - 3)
        {
            int sing_beta  = value_from_tt(tte.score, ply) - 2 * depth;
            int half_depth = depth / 2;

            int sing_val = negamax(b, half_depth, sing_beta - 1, sing_beta,
                                   ply, si, m);

            if (__builtin_expect(si->limits->stop, 0)) return 0;

            if (sing_val < sing_beta) {
                extension = 1;
            } else if (sing_val >= beta) {
                /* Multi-cut: several moves are good enough, prune. */
                return sing_val;
            }
        }

        /*
         * cur_pt and cur_to are already computed above (before the
         * pruning decisions) from the pre-make mailbox.  We reuse
         * them for the LMR and history updates below — capturing them
         * once avoids the post-unmake mailbox bug that the previous
         * code had (it looked up mailbox[from] AFTER unmake_move,
         * which returned NO_PIECE for quiet moves, causing
         * piece_type to always return PAWN).
         */

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
             * Continuation-history + countermove-history adjustment.
             * Uses BOTH ply-1 and ply-2 cont_hist tables, plus the
             * new 3D counter_hist [prev_pt][prev_to][cur_pt][cur_to].
             * Pre-computed cur_pt avoids the post-make mailbox bug.
             */
            if (prev_move_exists) {
                int prev_from = MOVE_FROM(si->move_stack[ply - 1]);
                int prev_to   = MOVE_TO(si->move_stack[ply - 1]);
                int ch1 = si->cont_hist_p1[cur_pt][prev_to];
                if (ch1 > 0) reduction--;
                else if (ch1 < 0) reduction++;

                /* counter_hist: derive prev_pt from the current mailbox
                 * (the previous move has been unmade by the time we're
                 * scoring here, so prev_from is back to the previous
                 * mover's piece). */
                Piece prev_pc = b->mailbox[prev_from];
                if (prev_pc != NO_PIECE) {
                    int prev_pt = (int)piece_type(prev_pc);
                    if (prev_pt >= 0 && prev_pt < CONT_HIST_PIECES) {
                        int ch_cm = si->counter_hist[prev_pt][prev_to][cur_pt][cur_to];
                        if (ch_cm > 0) reduction--;
                        else if (ch_cm < 0) reduction++;
                    }
                }
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

        /* ── PVS / LMR search ──
         *
         * Propagate `excluded` to children so a singular-verify search
         * keeps the excluded move excluded throughout the sub-tree.
         * (The verify search itself sets excluded=m for its top-level
         * call; deeper calls inherit it.) */
        if (legal_count == 1) {
            score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si, excluded);
        } else {
            score = -negamax(b, new_depth - reduction,
                             -alpha - 1, -alpha, ply + 1, si, excluded);

            if (score > alpha && (reduction > 0 || pv)) {
                score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si, excluded);
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
                     *
                     * The malus is applied to:
                     *   - butterfly history [side][from][to]
                     *   - countermove history [prev_pt][prev_to][qm_pt][qm_to]
                     *     (when a prev move exists) — same dual as the
                     *     positive bonus below.
                     */
                    int malus = stat_hat_penalty(depth);
                    for (int j = 0; j < n_quiets_tried; j++) {
                        Move qm = quiets_tried[j];
                        if (qm == m) continue;   /* don't malus the cutoff */
                        int *qh = &si->history[b->side]
                                            [MOVE_FROM(qm)][MOVE_TO(qm)];
                        *qh += malus - (*qh) * malus / 16384;

                        /* counter_hist malus */
                        if (prev_move_exists) {
                            int prev_from = MOVE_FROM(si->move_stack[ply - 1]);
                            int prev_to   = MOVE_TO(si->move_stack[ply - 1]);
                            Piece prev_pc = b->mailbox[prev_from];
                            if (prev_pc != NO_PIECE) {
                                int prev_pt = (int)piece_type(prev_pc);
                                int qm_pt   = quiet_pts[j];
                                if (prev_pt >= 0 && prev_pt < CONT_HIST_PIECES
                                 && qm_pt   >= 0 && qm_pt   < CONT_HIST_PIECES) {
                                    int16_t *chm = &si->counter_hist
                                        [prev_pt][prev_to][qm_pt][MOVE_TO(qm)];
                                    *chm += (int16_t)(malus
                                        - (*chm) * malus / 16384);
                                }
                            }
                        }
                    }

                    /*
                     * Continuation-history + countermove-history updates
                     * (positive bonus for the cutoff move).
                     *
                     * ply-1 cont_hist: bonus for (cur_pt, prev_to)
                     * ply-2 cont_hist: bonus for (cur_pt, prev_prev_to)
                     * counter_hist:    bonus for (prev_pt, prev_to, cur_pt, cur_to)
                     *
                     * Use the pre-computed cur_pt — avoids the
                     * post-unmake mailbox bug.
                     */
                    if (prev_move_exists) {
                        int prev_from = MOVE_FROM(si->move_stack[ply - 1]);
                        int prev_to   = MOVE_TO(si->move_stack[ply - 1]);
                        int16_t *ch = &si->cont_hist_p1[cur_pt][prev_to];
                        *ch += (int16_t)(bonus - (*ch) * bonus / 16384);

                        /* counter_hist bonus */
                        Piece prev_pc = b->mailbox[prev_from];
                        if (prev_pc != NO_PIECE) {
                            int prev_pt = (int)piece_type(prev_pc);
                            if (prev_pt >= 0 && prev_pt < CONT_HIST_PIECES) {
                                int16_t *chp = &si->counter_hist
                                    [prev_pt][prev_to][cur_pt][cur_to];
                                *chp += (int16_t)(bonus
                                    - (*chp) * bonus / 16384);
                            }
                        }
                    }
                    if (n_prev >= 2 && prev_moves[1] != NULL_MOVE) {
                        int pp_to = MOVE_TO(prev_moves[1]);
                        int16_t *ch = &si->cont_hist_p2[cur_pt][pp_to];
                        *ch += (int16_t)(bonus - (*ch) * bonus / 16384);
                    }
                }

                /* Don't store to TT when searching with an excluded move —
                 * the score is only valid for the excluded sub-position. */
                if (excluded == NULL_MOVE)
                    tt_store(b->hash, value_to_tt(beta, ply), best_move, depth, TT_LOWER, ply);
                return beta;
            }
        }
    }

    /* ── Terminal node detection ── */
    if (legal_count == 0) {
        /* When excluded is set, the "no legal moves" case is the
         * excluded move being the only legal move — return alpha
         * (fail-low) rather than a mate score, since the position
         * with the excluded move added is not mate. */
        if (excluded != NULL_MOVE) return alpha;
        return in_chk ? (-MATE_SCORE + ply) : DRAW_SCORE;
    }

    if (excluded == NULL_MOVE)
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
    si.limits        = lim;
    si.start_time_ms = now_ms();
    si.allotted_ms   = 0;
    si.max_time_ms   = 0;

    /* Bump TT generation so this search's entries outcompete stale
     * entries from previous searches during replacement. */
    tt_new_search();

#ifdef EVAL_DEBUG
    /* Reset top/bottom lists so each search produces a fresh dump. */
    eval_debug_init();
#endif

    /* ── Time budget ──
     *
     * Three modes:
     *   (1) movetime > 0   — explicit per-move budget.
     *   (2) wtime/btime > 0 — game clock.
     *   (3) depth > 0 OR infinite — no time limit (search runs to depth).
     *
     * The original code had a bug where mode (3) was unreachable: when
     * only `depth` was set (wtime=0, movetime=0), the condition
     * `!infinite && movetime == 0` was TRUE, so allotted_ms was set
     * to max(0+0, 50) = 50ms — depth-only searches were silently
     * capped at 50ms.  We now check depth-only BEFORE the game-clock
     * fallback.
     */
    if (lim->infinite || lim->ponder) {
        /* Infinite or ponder: no time limit.  Ponder searches until
         * `stop` is set (by `ponderhit` converting to time-limited
         * mode, or by a new command). */
        si.allotted_ms = 0;
        si.max_time_ms = 0;
    } else if (lim->movetime > 0) {
        si.allotted_ms = lim->movetime * 95 / 100;
        si.max_time_ms = lim->movetime;
        if (si.allotted_ms < 1) si.allotted_ms = 1;
    } else if (lim->depth > 0) {
        /* depth-only: no time limit. */
        si.allotted_ms = 0;
        si.max_time_ms = 0;
    } else {
        int time_left = (b->side == WHITE) ? lim->wtime : lim->btime;
        int inc       = (b->side == WHITE) ? lim->winc  : lim->binc;
        if (time_left <= 0 && inc <= 0) {
            /* No usable time control — fall back to a small default. */
            si.allotted_ms = 1000;
            si.max_time_ms = 2000;
        } else {
            /* opt: assume ~30 moves left; budget = time_left/30 + inc/2.
             * max: protect against clock blow-out — never exceed time_left*0.8. */
            int opt = time_left / 30 + inc / 2;
            int max = time_left / 4 + inc;
            if (max > time_left * 8 / 10) max = time_left * 8 / 10;
            if (max < opt) max = opt;
            if (opt < 10)  opt = 10;
            if (max < 20)  max = 20;
            si.allotted_ms = opt;
            si.max_time_ms = max;
        }
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

        /*
         * Aspiration windows (SF 14+ tuning).
         * Starts at depth 4 (matches original engine behavior).
         */
        if (depth >= 4) {
            int delta = 18 + depth;
            if (delta > 50) delta = 50;
            int asp_alpha = prev_score - delta;
            int asp_beta  = prev_score + delta;
            int failed_low_count = 0, failed_high_count = 0;

            while (true) {
                score = negamax(b, depth, asp_alpha, asp_beta, 0, &si, NULL_MOVE);

                if (lim->stop) goto done;

                if (score <= asp_alpha) {
                    failed_low_count++;
                    asp_alpha = (asp_alpha - delta < -INF) ? -INF : asp_alpha - delta;
                    delta = failed_low_count == 1 ? delta * 2 : delta * 3 / 2;
                } else if (score >= asp_beta) {
                    failed_high_count++;
                    asp_beta  = (asp_beta + delta >  INF) ?  INF : asp_beta + delta;
                    delta = failed_high_count == 1 ? delta * 2 : delta * 3 / 2;
                } else {
                    break;
                }

                if (delta > 500 || failed_low_count + failed_high_count >= 3) {
                    score = negamax(b, depth, -INF, INF, 0, &si, NULL_MOVE);
                    break;
                }
            }
        } else {
            score = negamax(b, depth, -INF, INF, 0, &si, NULL_MOVE);
        }

        if (lim->stop) break;
        prev_score = score;

        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) best_move = tte.move;

        if (best_move != NULL_MOVE) s_root_best_move = best_move;

        char mv_str[6];
        move_to_str(best_move, mv_str);
        int elapsed_ms = (int)(now_ms() - si.start_time_ms);
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

        /* Stop conditions:
         *   - Hard time limit reached.
         *   - Soft time limit: if we're over the optimum budget, don't
         *     start the next (deeper) iteration.
         *
         * NOTE: We deliberately do NOT break on finding a mate.  The
         * baseline engine continues searching after finding a mate,
         * which lets it discover SHORTER mates (e.g., mate-in-1 vs
         * mate-in-3) at deeper iterations.  Breaking early would
         * cause the engine to return a longer mate when a shorter
         * one exists. */
        if (time_up(&si)) break;

        /* Soft time limit: if we're over the optimum budget, don't
         * start the next (deeper) iteration. */
        if (si.allotted_ms > 0) {
            long elapsed = now_ms() - si.start_time_ms;
            if (elapsed >= si.allotted_ms) break;
        }
    }

done:
    {
        char mv_str[6];
        move_to_str(best_move, mv_str);

        /* Extract ponder move (predicted opponent reply) for `bestmove X ponder Y`. */
        Move ponder_move = search_extract_ponder_move(b, best_move);
        if (ponder_move != NULL_MOVE) {
            char ponder_str[6];
            move_to_str(ponder_move, ponder_str);
            printf("bestmove %s ponder %s\n", mv_str, ponder_str);
        } else {
            printf("bestmove %s\n", mv_str);
        }
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

/* ──────────────────────────────────────────────
 *  Ponder move extraction
 *
 *  After a search completes, we predict the opponent's reply by
 *  probing the TT for the position AFTER our best move.  The TT
 *  entry's stored move (the opponent's best move from their
 *  perspective) is our ponder move.
 *
 *  This is the standard Stockfish-style approach: the PV is implicit
 *  in the TT, so we don't need to track it explicitly during search.
 * ────────────────────────────────────────────── */
Move search_extract_ponder_move(Board *b, Move best_move) {
    if (best_move == NULL_MOVE) return NULL_MOVE;

    /* Make our move, probe the TT, unmake. */
    make_move(b, best_move);
    TTEntry tte;
    bool hit = tt_probe(b->hash, &tte);
    Move ponder = hit ? tte.move : NULL_MOVE;

    /* Verify the ponder move is legal in the position after our move.
     * The TT might contain a stale or corrupted move (e.g., from a
     * collision or a position that was never fully searched), so we
     * must validate before returning it.  An illegal ponder move
     * would cause the GUI to reject our `bestmove X ponder Y` line. */
    if (ponder != NULL_MOVE) {
        /* Generate legal moves and check if ponder is among them.
         * We use is_legal() which does make/unmake internally. */
        MoveList ml;
        gen_moves(b, &ml);
        bool found_legal = false;
        for (int i = 0; i < ml.count; i++) {
            if (ml.moves[i] == ponder) {
                if (is_legal(b, ponder)) {
                    found_legal = true;
                    break;
                }
            }
        }
        if (!found_legal) ponder = NULL_MOVE;
    }

    unmake_move(b);
    return ponder;
}