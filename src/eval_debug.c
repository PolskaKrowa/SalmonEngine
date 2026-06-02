/*
 * eval_debug.c — Implementation of eval stage debug logging.
 * Only compiled when EVAL_DEBUG is defined.
 */

#ifdef EVAL_DEBUG

#include "eval_debug.h"
#include "bitboard.h"    /* Bitboard, bb_pop */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

/* ─────────────────────────────────────────────────────────────────
 * Internal storage
 * ───────────────────────────────────────────────────────────────── */

typedef struct {
    EvalBreakdown bd;
    char          sq_char[64];  /* piece char per square; '.' = empty */
} EvalDebugEntry;

static EvalDebugEntry top_entries[EVAL_DEBUG_TOP_N];
static int            top_count;

static EvalDebugEntry bottom_entries[EVAL_DEBUG_BOTTOM_N];
static int            bottom_count;

/* ─────────────────────────────────────────────────────────────────
 * Board snapshot helpers
 * ───────────────────────────────────────────────────────────────── */

static void fill_sq_chars(const Board *b, char sq_char[64]) {
    /* Upper-case = White, lower-case = Black, '.' = empty */
    static const char PC[2][6] = {
        { 'P', 'N', 'B', 'R', 'Q', 'K' },   /* WHITE = 0 */
        { 'p', 'n', 'b', 'r', 'q', 'k' },   /* BLACK = 1 */
    };
    for (int i = 0; i < 64; i++) sq_char[i] = '.';
    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt]; /* local copy — bb_pop consumes it */
            while (bb) {
                int sq = bb_pop(&bb);
                sq_char[sq] = PC[c][pt];
            }
        }
    }
}

static void make_entry(EvalDebugEntry *e, const Board *b, const EvalBreakdown *bd) {
    e->bd = *bd;
    fill_sq_chars(b, e->sq_char);
}

/* ─────────────────────────────────────────────────────────────────
 * List management — simple linear-scan (debug code; speed irrelevant)
 * ───────────────────────────────────────────────────────────────── */

static int find_min_idx(const EvalDebugEntry *arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i].bd.final_score < arr[best].bd.final_score) best = i;
    return best;
}

static int find_max_idx(const EvalDebugEntry *arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i].bd.final_score > arr[best].bd.final_score) best = i;
    return best;
}

/* ─────────────────────────────────────────────────────────────────
 * Public — init / record
 * ───────────────────────────────────────────────────────────────── */

void eval_debug_init(void) {
    top_count    = 0;
    bottom_count = 0;
    /* Sentinel scores: any real chess score will beat these */
    for (int i = 0; i < EVAL_DEBUG_TOP_N;    i++) top_entries[i].bd.final_score    = INT_MIN;
    for (int i = 0; i < EVAL_DEBUG_BOTTOM_N; i++) bottom_entries[i].bd.final_score = INT_MAX;
}

void eval_debug_record(const Board *b, const EvalBreakdown *bd) {
    int score = bd->final_score;

    /* ── Update top (highest-score) list ── */
    if (top_count < EVAL_DEBUG_TOP_N) {
        make_entry(&top_entries[top_count++], b, bd);
    } else {
        int idx = find_min_idx(top_entries, EVAL_DEBUG_TOP_N);
        if (score > top_entries[idx].bd.final_score)
            make_entry(&top_entries[idx], b, bd);
    }

    /* ── Update bottom (lowest-score) list ── */
    if (bottom_count < EVAL_DEBUG_BOTTOM_N) {
        make_entry(&bottom_entries[bottom_count++], b, bd);
    } else {
        int idx = find_max_idx(bottom_entries, EVAL_DEBUG_BOTTOM_N);
        if (score < bottom_entries[idx].bd.final_score)
            make_entry(&bottom_entries[idx], b, bd);
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Dump helpers
 * ───────────────────────────────────────────────────────────────── */

static const char *phase_label(int phase) {
    if (phase >= 18) return "Middlegame";
    if (phase >=  8) return "Mixed     ";
    return                  "Endgame   ";
}

/* Linear taper: matches taper() in eval.c */
static int dbg_taper(int mg, int eg, int phase) {
    return (mg * phase + eg * (24 - phase)) / 24;
}

/* Net tapered contribution of one stage (from White's perspective) */
static int net_tap(int wmg, int weg, int bmg, int beg, int phase) {
    return dbg_taper(wmg - bmg, weg - beg, phase);
}

static void print_board(FILE *f, const char sq[64], int side) {
    fprintf(f, "  Side to move: %s\n\n", side == 0 ? "White" : "Black");
    for (int rank = 7; rank >= 0; rank--) {
        fprintf(f, "  %d |", rank + 1);
        for (int file = 0; file < 8; file++)
            fprintf(f, " %c", sq[rank * 8 + file]);
        fprintf(f, "\n");
    }
    fprintf(f, "      a b c d e f g h\n");
}

/*
 * print_breakdown — the main debug table.
 *
 * Columns:
 *   W-MG / W-EG  — White's own contribution (MG and EG) for this stage
 *   B-MG / B-EG  — Black's own contribution (MG and EG) for this stage
 *   Net (tap)    — taper(W_MG - B_MG, W_EG - B_EG, phase)
 *                  This is the stage's actual impact on the score from
 *                  White's perspective, after tapering.  Spot an
 *                  unreasonably large number here to find the culprit.
 */
static void print_breakdown(FILE *f, const EvalBreakdown *bd) {
    int ph = bd->phase;

    fprintf(f, "  Phase: %d/24  (%s)\n\n", ph, phase_label(ph));

    if (bd->lazy) {
        fprintf(f, "  ** Lazy-eval guard triggered: full breakdown unavailable. **\n");
        fprintf(f, "  ** (material+PST proxy was outside the lazy threshold)    **\n\n");
        fprintf(f, "  Score (side-to-move): %+d cp\n", bd->final_score);
        return;
    }

    /* Table header */
    fprintf(f,
        "  +------------------------------+---------+---------+"
        "---------+---------+-----------+\n");
    fprintf(f,
        "  | %-28s |  W-MG   |  W-EG   |"
        "  B-MG   |  B-EG   | Net (tap) |\n", "Stage");
    fprintf(f,
        "  +------------------------------+---------+---------+"
        "---------+---------+-----------+\n");

#define ROW(label, field)                                                    \
    fprintf(f,                                                               \
        "  | %-28s | %+7d | %+7d | %+7d | %+7d | %+9d |\n",                \
        (label),                                                             \
        bd->field##_mg[0], bd->field##_eg[0],                               \
        bd->field##_mg[1], bd->field##_eg[1],                               \
        net_tap(bd->field##_mg[0], bd->field##_eg[0],                       \
                bd->field##_mg[1], bd->field##_eg[1], ph))

    ROW("Material + PST",       material);
    ROW("Pawn structure",       pawns);
    ROW("Mobility",             mobility);
    ROW("Rooks",                rooks);
    ROW("Outposts",             outposts);
    ROW("Bishop pair",          bishop_pair);
    ROW("Weak queen",           weak_queen);
    ROW("King protector",       king_protector);
    ROW("Minor behind pawn",    minor_behind);
    ROW("King safety",          king_safety);
    ROW("Threats",              threats);
#undef ROW

    /* Divider before scalar terms */
    fprintf(f,
        "  +------------------------------+---------+---------+"
        "---------+---------+-----------+\n");

    /* Imbalance: added to mg only, so Net = taper(imbalance, 0, phase) */
    fprintf(f,
        "  | %-28s | %+7d |       - |       - |       - | %+9d |\n",
        "Imbalance (mg only)", bd->imbalance,
        dbg_taper(bd->imbalance, 0, ph));

    /* Tempo: already stored tapered */
    fprintf(f,
        "  | %-28s | %+7d |       - |       - |       - | %+9d |\n",
        "Tempo (tapered)", bd->tempo, bd->tempo);

    /* Initiative: already a single tapered correction */
    fprintf(f,
        "  | %-28s | %+7d |       - |       - |       - | %+9d |\n",
        "Initiative", bd->initiative, bd->initiative);

    fprintf(f,
        "  +------------------------------+---------+---------+"
        "---------+---------+-----------+\n");

    /*
     * Final scores — shown from both White's perspective and side-to-move's.
     * If these disagree with what you expect, a sign error in one of the
     * stage functions is likely.
     */
    int score_white_pov = (bd->side == 0) ? bd->final_score : -bd->final_score;
    fprintf(f,
        "  | %-28s |                                           | %+9d |\n",
        "Score (White's perspective)", score_white_pov);
    fprintf(f,
        "  | %-28s |                                           | %+9d |\n",
        "Score (side-to-move)", bd->final_score);
    fprintf(f,
        "  +------------------------------+---------+---------+"
        "---------+---------+-----------+\n");
}

/* ─────────────────────────────────────────────────────────────────
 * qsort comparators — overflow-safe branchless form
 * ───────────────────────────────────────────────────────────────── */

static int cmp_desc(const void *a, const void *b) {
    int sa = ((const EvalDebugEntry *)a)->bd.final_score;
    int sb = ((const EvalDebugEntry *)b)->bd.final_score;
    return (sb > sa) - (sb < sa);   /* descending */
}

static int cmp_asc(const void *a, const void *b) {
    int sa = ((const EvalDebugEntry *)a)->bd.final_score;
    int sb = ((const EvalDebugEntry *)b)->bd.final_score;
    return (sa > sb) - (sa < sb);   /* ascending  */
}

/* ─────────────────────────────────────────────────────────────────
 * Public — dump
 * ───────────────────────────────────────────────────────────────── */

void eval_debug_dump(const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "[eval_debug] Cannot open '%s' for writing\n", filename);
        return;
    }

    /* Sort before printing */
    if (top_count > 0)
        qsort(top_entries,    (size_t)top_count,    sizeof(EvalDebugEntry), cmp_desc);
    if (bottom_count > 0)
        qsort(bottom_entries, (size_t)bottom_count, sizeof(EvalDebugEntry), cmp_asc);

    fprintf(f,
        "================================================================\n"
        "  EVAL DEBUG DUMP\n"
        "\n"
        "  Scores are from the SIDE-TO-MOVE's perspective.\n"
        "  Per-stage W/B values are from each colour's OWN perspective\n"
        "  (positive = good for that colour before the sign flip).\n"
        "  'Net (tap)' = taper(W_MG - B_MG,  W_EG - B_EG, phase),\n"
        "               giving the stage's actual contribution to the\n"
        "               final score from White's perspective.\n"
        "================================================================\n\n");

    /* ── Top N ── */
    fprintf(f, "TOP %d  (highest scores — engine thinks it is winning most)\n",
            top_count);
    fprintf(f, "----------------------------------------------------------------\n\n");
    for (int i = 0; i < top_count; i++) {
        fprintf(f, ">>> Top #%d   Score: %+d cp\n\n",
                i + 1, top_entries[i].bd.final_score);
        print_board(f, top_entries[i].sq_char, top_entries[i].bd.side);
        fprintf(f, "\n");
        print_breakdown(f, &top_entries[i].bd);
        fprintf(f, "\n");
    }

    /* ── Bottom N ── */
    fprintf(f, "\nBOTTOM %d  (lowest scores — engine thinks it is losing most)\n",
            bottom_count);
    fprintf(f, "----------------------------------------------------------------\n\n");
    for (int i = 0; i < bottom_count; i++) {
        fprintf(f, ">>> Bottom #%d   Score: %+d cp\n\n",
                i + 1, bottom_entries[i].bd.final_score);
        print_board(f, bottom_entries[i].sq_char, bottom_entries[i].bd.side);
        fprintf(f, "\n");
        print_breakdown(f, &bottom_entries[i].bd);
        fprintf(f, "\n");
    }

    fclose(f);
    fprintf(stderr, "[eval_debug] Written to '%s'\n", filename);
}

#endif /* EVAL_DEBUG */