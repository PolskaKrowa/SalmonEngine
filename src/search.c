/*
 * search.c — Move searching (SF 11 refactored)
 *
 * Architecture:
 *  • Iterative deepening (ID) driver in search()
 *  • negamax() — principal variation search (PVS) with:
 *      – Transposition table cutoffs
 *      – TT mate-score adjustment: value_to_tt / value_from_tt (correctness fix)
 *      – Aspiration windows
 *      – Reverse futility pruning / static NMP
 *      – Razoring
 *      – Null-move pruning (NMP, adaptive R)
 *      – ProbCut (depth ≥ 5, raised-beta capture search, ~10 Elo)
 *      – Futility pruning: improving-adjusted margin via futility_margin()
 *      – Late move pruning (LMP): closed-form futility_move_count()
 *      – Late move reductions (LMR): product-of-logs Reductions table (SF 11)
 *      – Singular extensions (depth ≥ 6, TT-move uniqueness test, ~70 Elo)
 *      – Improving heuristic
 *      – Killer move heuristic (2 killers per ply)
 *      – Countermove heuristic
 *      – History heuristic: SF-calibrated stat_bonus formula + gravity/decay
 *      – Continuation history: (piece,to) affinity at ply-1 (~30 Elo)
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

/* ──────────────────────────────────────────────
 *  Previous-iteration root best move
 *
 *  Stored here (file scope) rather than in SearchInfo so that the
 *  public header doesn't need touching.  Only one search runs at a
 *  time, so a single static is safe.  Reset to NULL_MOVE at the
 *  start of every search() call.
 * ────────────────────────────────────────────── */
static Move s_root_best_move = NULL_MOVE;

/* ──────────────────────────────────────────────
 *  LMR reduction table — product-of-logs (SF 11 style)
 *
 *  SF 11 uses two independent log tables multiplied together:
 *    R = (Reductions[depth] * Reductions[move_count] + 511) / 1024
 *  This captures the interaction between depth and move count better
 *  than a single-log formula: at depth 1 all moves get R=0; at depth 10
 *  move 20 gets a larger reduction than the old table gave.
 * ────────────────────────────────────────────── */
static int Reductions[MAX_MOVES];   /* one table, indexed by depth or move count */

static void init_reductions(void) {
    Reductions[0] = 0;
    for (int i = 1; i < MAX_MOVES; i++)
        Reductions[i] = (int)(24.8 * log((double)i));
}

static inline int lmr_reduction(bool improving, int depth, int move_count) {
    int d = (depth     < MAX_MOVES) ? depth      : MAX_MOVES - 1;
    int m = (move_count < MAX_MOVES) ? move_count : MAX_MOVES - 1;
    int r = Reductions[d] * Reductions[m];
    return (r + 511) / 1024 + (!improving && r > 1007);
}

/* ──────────────────────────────────────────────
 *  stat_bonus — SF 11 calibrated history bonus.
 *  Returns negative for depth > 15 to prevent
 *  history-table saturation at high depths.
 * ────────────────────────────────────────────── */
static inline int stat_bonus(int depth) {
    return depth > 15 ? -8 : 19 * depth * depth + 155 * depth - 132;
}

/* ──────────────────────────────────────────────
 *  futility_margin — improving-adjusted futility
 * ────────────────────────────────────────────── */
static inline int futility_margin(int depth, bool improving) {
    return 217 * (depth - (int)improving);
}

/* ──────────────────────────────────────────────
 *  futility_move_count — closed-form LMP threshold
 *  Replaces the LMP_THRESHOLD lookup table.
 * ────────────────────────────────────────────── */
static inline int futility_move_count(bool improving, int depth) {
    return (5 + depth * depth) * (1 + (int)improving) / 2 - 1;
}

/* ──────────────────────────────────────────────
 *  TT score adjustment for mate distances
 *
 *  Mate scores are relative to the current ply.  Storing a raw mate
 *  score in the TT would corrupt the distance when the same position
 *  is retrieved at a different ply.  value_to_tt() makes the score
 *  root-relative before storing; value_from_tt() restores it to
 *  ply-relative when reading back.
 * ────────────────────────────────────────────── */
static inline int value_to_tt(int v, int ply) {
    if (v >= MATE_SCORE - MAX_PLY) return v + ply;
    if (v <= -MATE_SCORE + MAX_PLY) return v - ply;
    return v;
}

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

/* Material values for SEE only — don't need to match eval exactly */
static const int SEE_VAL[7] = { 100, 320, 330, 500, 900, 20000, 0 };
/*                               P    N    B    R    Q    K   NO_PIECE */

/* All pieces attacking 'sq' given occupancy 'occ' (both colours) */
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

    /* Determine the piece initially captured */
    int captured;
    if (mt == MT_EP)
        captured = PAWN;
    else if (b->mailbox[to] != NO_PIECE)
        captured = (int)piece_type(b->mailbox[to]);
    else
        return 0; /* quiet move — SEE = 0 */

    int gain[32];
    int d = 0;
    gain[0] = SEE_VAL[captured];

    /* Remove the moving piece from occupancy */
    Bitboard occ = b->occ[2] ^ SQUARE_BB[from];

    /* For en-passant, also remove the captured pawn */
    if (mt == MT_EP)
        occ ^= SQUARE_BB[b->ep_sq];

    int  attacker_type = (int)piece_type(b->mailbox[from]);
    Color side = b->side ^ 1; /* side to recapture */

    while (true) {
        d++;
        gain[d] = SEE_VAL[attacker_type] - gain[d - 1];

        /* Pruning: even in the best case this branch can't change the sign */
        if ( (gain[d - 1] < 0 ? -gain[d - 1] : gain[d - 1]) >
             (gain[d]     < 0 ? -gain[d]     : gain[d]    ) )
            {}; /* keep going — no pruning here, let the loop handle it */

        /* Find all remaining attackers of 'to' */
        Bitboard atk = attacks_to_sq(b, to, occ) & occ & b->occ[side];
        if (!atk) break;

        /* Pick the least-valuable attacker for this side */
        int pt;
        Bitboard piece_bb = 0;
        for (pt = PAWN; pt <= KING; pt++) {
            piece_bb = b->pieces[side][pt] & atk;
            if (piece_bb) break;
        }
        if (pt > KING) break;

        /* Remove the attacker (LSB) from occupancy — reveals x-ray pieces */
        occ ^= (piece_bb & -piece_bb);
        attacker_type = pt;
        side ^= 1;
    }

    /*
     * Backward pass: each side may choose NOT to recapture.
     * gain[d-1] = -max(-gain[d-1], gain[d])
     *           =  min( gain[d-1], -gain[d] )
     */
    while (--d)
        gain[d - 1] = -((-gain[d - 1] > gain[d]) ? -gain[d - 1] : gain[d]);

    return gain[0];
}

/* ──────────────────────────────────────────────
 *  Move scoring for ordering
 * ────────────────────────────────────────────── */
/*
 * Move ordering score tiers (highest = searched first).
 *
 * SCORE_TACTICAL_CAP is intentionally placed ABOVE SCORE_TT_MOVE.
 * This guarantees that a clearly winning capture (SEE >= 150 cp —
 * roughly "gain at least a pawn free") is always tried before even
 * the TT move.  The TT move comes from a previous, possibly shallower
 * iteration that may not have seen the hanging piece; putting winning
 * captures first ensures we never miss a free piece because an old
 * quiet TT move happened to cause a beta cutoff first.
 */
#define SCORE_TACTICAL_CAP 1050000   /* SEE >= 150 — above TT move          */
#define SCORE_TT_MOVE      1000000
#define SCORE_GOOD_CAP      900000   /* 0 <= SEE < 150 — even / slight gain */
#define SCORE_KILLER1       800000
#define SCORE_KILLER2       790000
#define SCORE_COUNTER       780000   /* countermove bonus */
#define SCORE_QUIET_BASE          0  /* + history score   */

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

        /* Separate captures into three tiers by SEE:
         *
         *  >= 150 cp  → SCORE_TACTICAL_CAP  (clearly winning — above TT move)
         *     0..149  → SCORE_GOOD_CAP      (even or slight gain)
         *  < 0        → raw SEE score        (losing capture, tried last)
         *
         * The 150 cp threshold catches any capture of a piece that is either
         * completely undefended or defended only by a more-expensive piece.
         * Examples that go to SCORE_TACTICAL_CAP:
         *   PxN undefended (SEE=320), PxR undefended (SEE=500),
         *   NxR undefended (SEE=180), PxB undefended (SEE=230).
         */
        int see_score = see(b, m);
        if (see_score >= 150)
            return SCORE_TACTICAL_CAP + MVV_LVA[attacker][victim];
        else if (see_score >= 0)
            return SCORE_GOOD_CAP + MVV_LVA[attacker][victim];
        else
            return see_score; /* losing capture — below quiet moves */
    }

    if (MOVE_IS_PROMO(m)) return SCORE_GOOD_CAP - 1;

    if (m == killers[0])  return SCORE_KILLER1;
    if (m == killers[1])  return SCORE_KILLER2;
    if (m == countermove) return SCORE_COUNTER;

    /* Quiet move: butterfly history + continuation history blend.
     * The cont_hist bonus captures (piece,to) affinity from the previous
     * move, which is the single-ply port of SF's continuation history. */
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
 *  Called between ID iterations to avoid stale bonuses
 *  dominating early in the new iteration.
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

/* LMP_THRESHOLD replaced by the futility_move_count() formula above. */

/* ──────────────────────────────────────────────
 *  Quiescence search
 * ────────────────────────────────────────────── */
int quiesce(Board *b, int alpha, int beta, int ply, SearchInfo *si) {
    si->nodes++;
    if (ply > si->seldepth) si->seldepth = ply;

    bool in_chk = in_check(b);

    /*
     * Stand-pat: our lower bound assuming we can "do nothing".
     * Not valid when in check — we must find a legal evasion.
     */
    int stand_pat = evaluate(b);
    if (!in_chk) {
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MoveList ml;
    /*
     * When in check, generate ALL legal moves (evasions).
     * A check in qsearch that has no evasion is checkmate;
     * a stalemate (no legal move, not in check) is a draw.
     * When not in check, generate captures only — the normal qsearch path.
     */
    if (in_chk)
        gen_moves(b, &ml);
    else
        gen_captures(b, &ml);

    /* Score moves */
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
            /*
             * SEE pruning in qsearch (captures only path):
             * Skip clearly losing captures (negative SEE score) and apply
             * delta pruning.  Neither pruning applies when in check — we
             * must search every evasion.
             */
            if (scores[i] < 0)      /* negative SEE score */
                break;              /* remaining moves have equal or worse SEE */

            /* Delta pruning: stand_pat + captured value + margin can't reach alpha */
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

        if (si->limits->stop) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    /* In check with no legal evasion: checkmate */
    if (in_chk && legal_count == 0)
        return -MATE_SCORE + ply;

    return alpha;
}

/* ──────────────────────────────────────────────
 *  Negamax alpha-beta (PVS) — the main search
 * ────────────────────────────────────────────── */
int negamax(Board *b, int depth, int alpha, int beta, int ply, SearchInfo *si) {
    if (si->nodes % 2048 == 0 && time_up(si)) {
        si->limits->stop = true;
        return 0;
    }

    si->nodes++;
    bool root = (ply == 0);
    bool pv   = (beta - alpha > 1);

    if (ply >= MAX_PLY - 1)
        return quiesce(b, alpha, beta, ply, si);

    /* ── Draw detection ── */
    if (!root) {
        if (b->halfmove >= 100) return DRAW_SCORE;

        for (int i = b->hist_idx - 2; i >= 0; i -= 2) {
            if (b->history[i].hash == b->hash) return DRAW_SCORE;
            /* The halfmove stored in history[i] is the clock BEFORE that move
               was made.  If it was 0, the move at i+1 was irreversible — no
               repetition can cross it. */
            if (b->history[i].halfmove == 0) break;
        }
    }

    /* ── TT probe ── */
    TTEntry tte;
    Move    tt_move = NULL_MOVE;
    if (tt_probe(b->hash, &tte)) {
        tt_move = tte.move;
        if (!pv && tte.depth >= depth) {
            int tt_score = value_from_tt(tte.score, ply);
            if (tte.flag == TT_EXACT) return tt_score;
            if (tte.flag == TT_LOWER && tt_score >= beta)  return tt_score;
            if (tte.flag == TT_UPPER && tt_score <= alpha) return tt_score;
        }
    }

    /*
     * Root move ordering: if the TT had no move (cold start or replacement
     * collision), seed tt_move with the best move found in the previous
     * iteration.  This guarantees the strongest known candidate is always
     * searched first at the root, giving the ID framework a reliable anchor
     * regardless of TT occupancy.
     */
    if (root && tt_move == NULL_MOVE && s_root_best_move != NULL_MOVE)
        tt_move = s_root_best_move;

    /* ── Drop into quiescence ── */
    if (depth <= 0)
        return quiesce(b, alpha, beta, ply, si);

    bool in_chk = in_check(b);

    /* ── Static evaluation (cached for pruning heuristics) ── */
    int static_eval = evaluate(b);
    si->eval_stack[ply] = static_eval;

    /*
     * Improving heuristic:
     * The position is "improving" when the side-to-move is better off than
     * they were two plies ago.  An improving position allows more aggressive
     * pruning (LMR, LMP), while a worsening one is treated conservatively.
     */
    bool improving = (!in_chk && ply >= 2
                      && static_eval > si->eval_stack[ply - 2]);

    if (!in_chk && !pv) {
        /*
         * ── Reverse Futility Pruning (Static Null-Move Pruning) ──
         *
         * If the static eval already exceeds beta by a depth-scaled margin,
         * the position is probably too good and will cause a cutoff anyway.
         * Return a blended value to avoid discontinuities at the boundary.
         *
         * Conditions: not in check, not PV, depth <= 8, no mate scores.
         */
        if (depth <= 8 && abs(beta) < MATE_SCORE - MAX_PLY) {
            int rfp_margin = (improving ? 60 : 80) * depth;
            if (static_eval - rfp_margin >= beta)
                return (static_eval + beta) / 2;
        }

        /*
         * ── Razoring ──
         *
         * If the static eval is far below alpha, drop into quiescence.
         * Only useful at depth <= 3; deeper, NMP / RFP handle the work.
         */
        if (depth <= 3 && abs(alpha) < MATE_SCORE - MAX_PLY) {
            int razor_margin = 300 + 60 * depth;
            if (static_eval + razor_margin <= alpha) {
                int q = quiesce(b, alpha, beta, ply, si);
                if (q <= alpha) return q;
            }
        }
    }

    /*
     * ── Null-Move Pruning (NMP) ──
     *
     * Adaptive reduction R grows with depth and how far eval exceeds beta,
     * inspired by Stockfish / Weiss.  Skip if in check, PV node, zugzwang-
     * likely positions (only pawns + king), or we're already in a null move.
     *
     * Extra guards added:
     *  • If the *opponent* also has only pawns + king, a pawn race or
     *    blocked ending is likely — NMP is unreliable there too.
     *  • At high depths (>= 12), a verification search confirms the cutoff
     *    before we trust it, preventing tactical blunders caused by NMP.
     */
    bool do_nmp = !pv && !in_chk && depth >= 3
               && static_eval >= beta
               /* We must have at least one non-pawn/king piece */
               && bb_popcount(b->occ[b->side]
                              & ~b->pieces[b->side][PAWN]
                              & ~b->pieces[b->side][KING]) > 0
               /* Skip when opponent is pawn-only — zugzwang / pawn-race risk */
               && bb_popcount(b->occ[b->side ^ 1]
                              & ~b->pieces[b->side ^ 1][PAWN]
                              & ~b->pieces[b->side ^ 1][KING]) > 0;

    if (do_nmp) {
        int R = 3 + depth / 6
                  + (((static_eval - beta) / 200) < 3
                     ? (static_eval - beta) / 200 : 3);

        make_null_move(b);
        si->move_stack[ply] = NULL_MOVE;
        int null_score = -negamax(b, depth - R, -beta, -beta + 1, ply + 1, si);
        unmake_null_move(b);

        if (null_score >= beta) {
            /* Don't return unverified mate scores */
            if (null_score >= MATE_SCORE - MAX_PLY) null_score = beta;

            /*
             * Verification search: at high depths, run a reduced search
             * (same depth - R) with the real window to confirm the cutoff
             * is genuine and not a horizon artefact.
             */
            if (depth >= 12) {
                int verify = negamax(b, depth - R, beta - 1, beta, ply, si);
                if (verify < beta) goto after_nmp; /* cutoff not confirmed */
            }
            return null_score;
        }
    }
    after_nmp:;

    /*
     * ── ProbCut ──
     *
     * If we have a capture whose SEE already beats a raised beta threshold,
     * and a reduced search confirms it scores above that threshold, we can
     * safely prune the rest of the tree.
     * Conditions: non-PV, not in check, depth >= 5, score not near mate.
     */
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

            /* Quick qsearch confirmation */
            int pc_val = -quiesce(b, -(raised_beta), -(raised_beta) + 1,
                                  ply + 1, si);

            /* Full reduced search if qsearch held */
            if (pc_val >= raised_beta)
                pc_val = -negamax(b, depth - 4, -(raised_beta), -(raised_beta) + 1,
                                  ply + 1, si);

            unmake_move(b);

            if (si->limits->stop) return 0;
            if (pc_val >= raised_beta) return pc_val;
            pc_count++;
        }
    }

    /* ── Generate and score all moves ── */
    MoveList ml;
    gen_moves(b, &ml);

    /* Look up the countermove for the move played at the previous ply */
    Move cm = (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE)
              ? si->countermove[MOVE_FROM(si->move_stack[ply - 1])]
                               [MOVE_TO  (si->move_stack[ply - 1])]
              : NULL_MOVE;

    int scores[MAX_MOVES];
    const Move *killers_ply = si->killers[ply < MAX_PLY ? ply : MAX_PLY - 1];
    int (*hist_side)[64]    = si->history[b->side];
    Move prev_neg = (ply > 0) ? si->move_stack[ply - 1] : NULL_MOVE;

    for (int i = 0; i < ml.count; i++)
        scores[i] = score_move(b, ml.moves[i], tt_move,
                                killers_ply, cm, hist_side,
                                si->cont_hist, prev_neg);

    int  best_score  = -INF;
    Move best_move   = NULL_MOVE;
    int  legal_count = 0;
    int  quiet_count = 0;   /* quiet moves searched so far (for LMP) */
    int  tt_flag     = TT_UPPER;

    for (int i = 0; i < ml.count; i++) {
        Move m = pick_move(ml.moves, scores, ml.count, i);
        if (!is_legal(b, m)) continue;

        bool is_cap   = MOVE_IS_CAP(m) || MOVE_TYPE(m) == MT_EP;
        bool is_promo = MOVE_IS_PROMO(m);
        bool is_quiet = !is_cap && !is_promo;

        legal_count++;
        if (is_quiet) quiet_count++;

        /* ── Pruning for non-PV, non-check quiet moves ── */
        if (!pv && !in_chk && is_quiet && best_score > -MATE_SCORE + MAX_PLY) {

            /*
             * Late Move Pruning (Move Count Pruning):
             * After searching enough quiet moves at low depth, the rest are
             * unlikely to beat alpha — prune them.
             */
            if (depth <= 8
                && quiet_count >= futility_move_count(improving, depth))
                continue;

            /*
             * Futility Pruning:
             * If static eval + an improving-adjusted margin is still below
             * alpha, quiet moves at this node are unlikely to improve alpha.
             */
            if (depth <= 8) {
                int fp_margin = futility_margin(depth, improving)
                              + hist_side[MOVE_FROM(m)][MOVE_TO(m)] / 128;
                if (static_eval + fp_margin <= alpha)
                    continue;
            }

            /*
             * SEE-based quiet move pruning:
             * A quiet move with a deeply negative SEE (e.g. walking into a
             * pawn fork) is pruned at low depth.
             */
            if (depth <= 5 && see(b, m) < -40 * depth)
                continue;
        }

        /*
         * SEE pruning for captures:
         * Losing captures (SEE < threshold) are skipped at higher depths.
         * The threshold tightens with depth.
         */
        if (!pv && is_cap && depth <= 10
            && see(b, m) < -20 * depth * depth)
            continue;

        /*
         * ── Singular Extension ──
         *
         * If the TT move is the only move that fails high in a reduced
         * search with a narrowed beta, it is "singular" — extend it by
         * one ply.  If multiple moves fail high we get a multi-cut signal
         * and can prune the node early.
         *
         * Conditions: depth >= 6, move == tt_move, not root, TT entry has
         * a lower-bound score that isn't near mate, and was searched deeply
         * enough to be reliable.
         */
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

            /* Temporarily mask out the TT move by setting it to NULL so
               the recursive call won't give it a TT-move bonus in ordering. */
            Move saved_tt = tt_move;
            tt_move = NULL_MOVE;

            int sing_val = negamax(b, half_depth, sing_beta - 1, sing_beta,
                                   ply, si);

            tt_move = saved_tt;

            if (si->limits->stop) return 0;

            if (sing_val < sing_beta) {
                extension = 1;  /* TT move is singular → extend */
            } else if (sing_beta >= beta) {
                /* Multi-cut: more than one move fails high → prune */
                return sing_beta;
            }
        }

        /* ── Make the move ── */
        si->move_stack[ply] = m;
        make_move(b, m);

        int score;
        int new_depth = depth - 1 + extension;

        /*
         * Only extend for check when we have not already applied a singular
         * extension.  Stacking both extensions would give new_depth >= depth,
         * which — in a chain of checking replies — prevents depth from ever
         * reaching 0 and causes unbounded recursion.  The hard ply ceiling
         * above is the primary guard; this keeps extensions individually sane.
         */
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

            /* Adjust reduction with heuristics */
            if (!improving)          reduction++;   /* losing ground → reduce more  */
            if (pv)                  reduction--;   /* PV node → reduce less        */
            if (m == cm)             reduction--;   /* countermove → reduce less    */
            if (m == killers_ply[0]
             || m == killers_ply[1]) reduction--;   /* killer → reduce less         */

            /* Continuation history adjustment */
            if (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE) {
                int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                int cur_pt  = (int)piece_type(b->mailbox[MOVE_FROM(m)]);
                int ch = si->cont_hist[cur_pt][prev_to];
                if (ch > 0) reduction--;   /* good continuation → reduce less */
                else if (ch < 0) reduction++; /* bad continuation → reduce more */
            }

            /* Clamp: never drop below 1, never below depth-1 */
            if (reduction < 1)          reduction = 1;
            if (reduction > new_depth)  reduction = new_depth;
        }

        /* ── PVS / LMR search ── */
        if (legal_count == 1) {
            /* Full-window search for the first (presumably best) move */
            score = -negamax(b, new_depth, -beta, -alpha, ply + 1, si);
        } else {
            /* Null-window search (possibly reduced) */
            score = -negamax(b, new_depth - reduction,
                             -alpha - 1, -alpha, ply + 1, si);

            /* Re-search at full depth if it beat alpha and was reduced or PV */
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
            alpha   = score;
            tt_flag = TT_EXACT;

            if (score >= beta) {
                /* Beta cutoff — update killer, countermove and history */
                if (is_quiet) {
                    /* Killers */
                    if (m != killers_ply[0]) {
                        si->killers[ply][1] = si->killers[ply][0];
                        si->killers[ply][0] = m;
                    }

                    /* Countermove: this move refutes the opponent's last move */
                    if (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE) {
                        si->countermove
                            [MOVE_FROM(si->move_stack[ply - 1])]
                            [MOVE_TO  (si->move_stack[ply - 1])] = m;
                    }

                    /* History bonus — SF-calibrated stat_bonus formula */
                    int bonus = stat_bonus(depth);
                    int *h = &si->history[b->side][MOVE_FROM(m)][MOVE_TO(m)];
                    *h += bonus - (*h) * bonus / 16384;

                    /* Continuation history update */
                    if (ply > 0 && si->move_stack[ply - 1] != NULL_MOVE) {
                        int prev_to = MOVE_TO(si->move_stack[ply - 1]);
                        int cur_pt  = (int)piece_type(b->mailbox[MOVE_TO(m)]);
                        si->cont_hist[cur_pt][prev_to] += bonus
                            - si->cont_hist[cur_pt][prev_to] * bonus / 16384;
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

    /* Time management */
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

    /* ── Opening book probe ──
     *
     * If the current position has a book entry, play it immediately
     * without searching.  This saves the full search budget for the
     * middlegame and guarantees solid, well-tested opening moves.
     * book_probe() validates legality internally, so a hash collision
     * cannot cause an illegal move to be returned.
     */
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

    /* Reset root-move seed for this search */
    s_root_best_move = NULL_MOVE;

    for (int depth = 1; depth <= max_depth; depth++) {
        si.seldepth = 0;
        si.nodes    = 0;
        /* Keep killer moves across iterations so iterative deepening can
           reuse the strongest refutations found so far. */
        if (depth > 1) decay_history(&si);

        int score;

        /*
         * ── Aspiration Windows ──
         *
         * For depth >= 4, start the search with a narrow window around
         * the score from the previous iteration.  On a fail, widen
         * exponentially and re-search.  This reduces the search space
         * significantly when the score is stable.
         */
        if (depth >= 4) {
            int delta = 25;
            int asp_alpha = prev_score - delta;
            int asp_beta  = prev_score + delta;

            while (true) {
                score = negamax(b, depth, asp_alpha, asp_beta, 0, &si);

                if (lim->stop) goto done;

                if (score <= asp_alpha) {
                    /* Fail low: widen lower bound, keep upper tight */
                    asp_alpha = (asp_alpha - delta < -INF) ? -INF : asp_alpha - delta;
                    delta    *= 2;
                } else if (score >= asp_beta) {
                    /* Fail high: widen upper bound, keep lower tight */
                    asp_beta  = (asp_beta + delta >  INF) ?  INF : asp_beta + delta;
                    delta    *= 2;
                } else {
                    break; /* Exact score within window */
                }

                /* Safety: fall back to full window after too many fails */
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

        /* Retrieve best move from TT */
        TTEntry tte;
        if (tt_probe(b->hash, &tte) && tte.move) best_move = tte.move;

        /* Carry the best move forward as the anchor for next iteration's
         * root move ordering, in case the TT entry gets evicted.         */
        if (best_move != NULL_MOVE) s_root_best_move = best_move;

        /* UCI info output */
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
}

void search_init(void) {
    init_mvv_lva();
    init_reductions();
    book_init();
}