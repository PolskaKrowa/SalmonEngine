/*
 * eval.c — Static evaluation
 *
 * Scoring convention: always from the perspective of the side to move
 * (positive = good for side to move).
 *
 * Features implemented:
 *   • Material balance
 *   • Piece-square tables — separate MG and EG tables, blended by game phase
 *   • Material imbalance — SF 11 quadratic polynomial (bishop pair, rook vs
 *       minors, etc.)  Replaces a flat material count.
 *   • Mobility — non-linear per-count tables (SF 11 MobilityBonus values)
 *       so the first squares of freedom are worth the most.
 *   • Pawn structure:
 *       – Doubled-pawn penalty
 *       – Isolated-pawn penalty
 *       – Backward-pawn penalty
 *       – Passed-pawn bonus
 *   • Outpost squares for knights and bishops
 *   • TrappedRook  — penalty when a rook with ≤3 moves is hemmed in on the
 *       same side as its own king
 *   • WeakQueen    — penalty when enemy sliders x-ray through our queen
 *   • KingProtector — minor pieces far from own king incur a distance penalty
 *   • MinorBehindPawn — minor piece sheltered behind a friendly pawn earns a bonus
 *   • King safety:
 *       – Pawn shield bonus
 *       – Open-file penalty near king
 *       – Distance-weighted enemy-piece attack count
 *       – Endgame king-activity term (king-to-king proximity bonus)
 *   • Bishop pair bonus
 *   • Rook on open / semi-open file bonus
 *   • Rook on seventh rank bonus
 *   • Tempo bonus (side to move)
 *   • Initiative / complexity correction — reduces a winning advantage when
 *       the winning side cannot realistically convert (no passed pawns, no
 *       outflanking, pawns on only one flank).
 *   • Lazy evaluation guard — fast material+PST proxy bails out when the
 *       score is far from any search window, saving ~5 % NPS for free.
 */

#include "eval.h"
#include "bitboard.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#ifdef EVAL_DEBUG
#  include "eval_debug.h"
#endif

/* ──────────────────────────────────────────────
 *  Runtime-tunable eval weights (EvalWeights struct).
 *
 *  The compiled defaults below are loaded into the global `EW` struct
 *  by eval_weights_init().  The tuner can modify EW at runtime to
 *  test different values without recompilation.
 * ────────────────────────────────────────────── */
EvalWeights EW;

void eval_weights_init(void) {
    /* Material */
    EW.material_mg[PAWN]   = 82;   EW.material_eg[PAWN]   = 94;
    EW.material_mg[KNIGHT] = 344;  EW.material_eg[KNIGHT] = 338;
    EW.material_mg[BISHOP] = 358;  EW.material_eg[BISHOP] = 329;
    EW.material_mg[ROOK]   = 480;  EW.material_eg[ROOK]   = 546;
    EW.material_mg[QUEEN]  = 1022; EW.material_eg[QUEEN]  = 924;
    EW.material_mg[KING]   = 0;    EW.material_eg[KING]   = 0;

    /* Pawn structure */
    EW.doubled_pawn_mg  = -11;  EW.doubled_pawn_eg  = -56;
    EW.isolated_pawn_mg = -15;  EW.isolated_pawn_eg = -15;
    EW.backward_pawn_mg = -9;   EW.backward_pawn_eg = -22;
    EW.passed_pawn_mg[0] = 0;  EW.passed_pawn_eg[0] = 0;
    EW.passed_pawn_mg[1] = 5;  EW.passed_pawn_eg[1] = 10;
    EW.passed_pawn_mg[2] = 10; EW.passed_pawn_eg[2] = 20;
    EW.passed_pawn_mg[3] = 20; EW.passed_pawn_eg[3] = 40;
    EW.passed_pawn_mg[4] = 35; EW.passed_pawn_eg[4] = 65;
    EW.passed_pawn_mg[5] = 60; EW.passed_pawn_eg[5] = 95;
    EW.passed_pawn_mg[6] = 90; EW.passed_pawn_eg[6] = 140;
    EW.passed_pawn_mg[7] = 0;  EW.passed_pawn_eg[7] = 0;
    EW.pawn_chain_mg  = 11;   EW.pawn_chain_eg  = 4;
    EW.pawn_island_mg = -3;   EW.pawn_island_eg = -8;

    /* King safety */
    EW.king_attacker_weight[PAWN]   = 0;
    EW.king_attacker_weight[KNIGHT] = 20;
    EW.king_attacker_weight[BISHOP] = 20;
    EW.king_attacker_weight[ROOK]   = 40;
    EW.king_attacker_weight[QUEEN]  = 80;
    EW.king_attacker_weight[KING]   = 0;
    EW.king_open_file_penalty  = -25;
    EW.king_shield_bonus       = 7;
    EW.king_shield_bonus_eg    = 2;

    /* Tempo */
    EW.tempo_mg = 16;
    EW.tempo_eg = 10;

    /* Lazy eval */
    EW.lazy_threshold = 1000;
}

EvalWeights *eval_weights_get(void) { return &EW; }

/* Keep the static const arrays for PST tables (too large for the struct
 * and rarely tuned).  Material values are duplicated in EW for the
 * tuner, but the eval code reads from EW for consistency. */

static const int PST_PAWN_MG[64] = {
        0,     0,     0,     0,     0,     0,     0,     0,
       98,   134,    61,    95,    68,   126,    34,   -11,
       -6,     7,    26,    31,    65,    56,    25,   -20,
      -14,    13,     6,    21,    23,    12,    17,   -23,
      -27,    -2,    -5,    12,    17,     6,    10,   -25,
      -26,    -4,    -4,   -10,     3,     3,    33,   -12,
      -35,    -1,   -20,   -23,   -15,    24,    38,   -22,
        0,     0,     0,     0,     0,     0,     0,     0,
};
static const int PST_PAWN_EG[64] = {
        0,     0,     0,     0,     0,     0,     0,     0,
      178,   173,   158,   134,   147,   132,   165,   187,
       94,   100,    85,    67,    56,    53,    82,    84,
       32,    24,    13,     5,    -2,     4,    17,    17,
       13,     9,    -3,    -7,    -7,    -8,     3,    -1,
        4,     7,    -6,     1,     0,    -5,    -1,    -8,
       13,     8,     8,    10,    13,     0,     2,    -7,
        0,     0,     0,     0,     0,     0,     0,     0,
};

static const int PST_KNIGHT_MG[64] = {
     -167,   -89,   -34,   -49,    61,   -97,   -15,  -107,
      -73,   -41,    72,    36,    23,    62,     7,   -17,
      -47,    60,    37,    65,    84,   129,    73,    44,
       -9,    17,    19,    53,    37,    69,    18,    22,
      -13,     4,    16,    13,    28,    19,    21,    -8,
      -23,    -9,    12,    10,    19,    17,    25,   -16,
      -29,   -53,   -12,    -3,    -1,    18,   -14,   -19,
      -99,   -21,   -58,   -33,   -17,   -28,   -19,   -23,
};
static const int PST_KNIGHT_EG[64] = {
      -58,   -38,   -13,   -28,   -31,   -27,   -63,   -99,
      -25,    -8,   -25,    -2,    -9,   -25,   -24,   -52,
      -24,   -20,    10,     9,    -1,    -9,   -19,   -41,
      -17,     3,    22,    22,    22,    11,     8,   -18,
      -18,    -6,    16,    25,    16,    17,     4,   -18,
      -23,    -3,    -1,    15,    10,    -3,   -20,   -22,
      -42,   -20,   -10,    -5,    -2,   -20,   -23,   -44,
       27,   -51,   -23,   -15,   -22,   -18,   -50,   -64,
};

static const int PST_BISHOP_MG[64] = {
      -30,   -22,   -82,   -37,   -25,   -42,     7,    -8,
      -26,    16,   -18,   -13,    30,    59,    18,   -47,
      -16,    37,    43,    40,    35,    50,    37,    -2,
       -4,     5,    19,    50,    37,    37,     7,    -2,
       -6,    13,    13,    26,    34,    12,    10,     4,
        0,    15,    15,    15,    14,    27,    18,    10,
        4,    15,    16,     0,     7,    21,    33,     1,
      -26,    -3,   -14,   -21,   -13,   -12,   -39,   -21,
};
static const int PST_BISHOP_EG[64] = {
      -70,   -75,   -11,    -8,    -7,    -9,   -17,   -24,
       -8,    -4,     7,   -12,    -3,   -13,    -4,   -14,
        2,    -8,     0,    -1,    -2,     6,     0,     4,
       -3,     9,    12,     9,    14,    10,     3,     2,
       -6,     3,    13,    19,     7,    10,    -3,    -9,
      -12,    -3,     8,    10,    13,     3,    -7,   -15,
      -14,   -18,    -7,    -1,     4,    -9,   -15,   -27,
       29,    -9,   -23,    -5,    -9,   -16,    -5,   -17,
};

static const int PST_ROOK_MG[64] = {
       32,    42,    32,    51,    63,     9,    31,    43,
       27,    32,    58,    62,    80,    67,    26,    44,
       -5,    19,    26,    36,    17,    45,    61,    16,
      -24,   -11,     7,    26,    24,    35,    -8,   -20,
      -36,   -26,   -12,    -1,     9,    -7,     6,   -23,
      -45,   -25,   -16,   -17,     3,     0,    -5,   -33,
      -44,   -16,   -20,    -9,    -1,    11,    -6,   -71,
      -13,   -13,     1,    17,    16,     7,   -37,   -26,
};
static const int PST_ROOK_EG[64] = {
       13,    10,    18,    15,    12,    12,     8,     5,
       11,    13,    13,    11,    -3,     3,     8,     3,
        7,     7,     7,     5,     4,    -3,    -5,    -3,
        4,     3,    13,     1,     2,     1,    -1,     2,
        3,     5,     8,     4,    -5,    -6,    -8,   -11,
       -4,     0,    -5,    -1,    -7,   -12,    -8,   -16,
       -6,    -6,     0,     2,    -9,    -9,   -11,    -3,
       31,     2,     3,    -1,    -5,   -13,     4,   -20,
};

static const int PST_QUEEN_MG[64] = {
      -28,     0,    29,    12,    59,    44,    43,    45,
      -24,   -39,    -5,     1,   -16,    57,    28,    54,
      -13,   -17,     7,     8,    29,    56,    47,    57,
      -27,   -27,   -16,   -16,    -1,    17,    -2,     1,
       -9,   -26,    -9,   -10,    -2,    -4,     3,    -3,
      -14,     2,   -11,    -2,    -5,     2,    14,     5,
      -35,    -8,    11,     2,     8,    15,    -3,     1,
       -5,   -18,    -9,    10,   -15,   -25,   -31,   -50,
};
static const int PST_QUEEN_EG[64] = {
       -9,    22,    22,    27,    27,    19,    10,    20,
      -17,    20,    32,    41,    58,    25,    30,     0,
      -20,     6,     9,    49,    47,    35,    19,     9,
        3,    22,    24,    45,    57,    40,    57,    36,
      -18,    28,    19,    47,    31,    34,    39,    23,
      -16,   -27,    15,     6,     9,    17,    10,     5,
      -22,   -23,   -30,   -16,   -16,   -23,   -36,   -32,
      -44,   -28,   -22,   -43,    -5,   -32,   -20,   -41,
};

static const int PST_KING_MG[64] = {
      -65,    23,    16,   -15,   -56,   -34,     2,    13,
       29,    -1,   -20,    -7,    -8,    -4,   -38,   -29,
       -9,    24,     2,   -16,   -20,     6,    22,   -22,
      -17,   -20,   -12,   -27,   -30,   -25,   -14,   -36,
      -49,    -1,   -27,   -39,   -46,   -44,   -33,   -51,
      -14,   -14,   -22,   -46,   -44,   -30,   -15,   -27,
        1,     7,    -8,   -64,   -43,   -16,     9,     8,
      -15,    36,    12,   -54,     8,   -28,    24,    14,
};
static const int PST_KING_EG[64] = {
      -74,   -35,   -18,   -18,   -11,    15,     4,   -17,
      -12,    17,    14,    17,    17,    38,    23,    11,
       10,    17,    23,    15,    20,    45,    44,    13,
       -8,    22,    24,    27,    26,    33,    26,     3,
      -18,    -4,    21,    24,    27,    23,     9,   -11,
      -19,    -3,    11,    21,    23,    16,     7,    -9,
      -27,   -11,     4,    13,    14,     4,    -5,   -17,
      -53,   -34,   -21,   -11,   -28,   -14,   -24,   -43,
};

static const int DOUBLED_PAWN_PENALTY_MG  = -11;
static const int DOUBLED_PAWN_PENALTY_EG  = -56;
static const int ISOLATED_PAWN_PENALTY_MG = -15;
static const int ISOLATED_PAWN_PENALTY_EG = -15;
static const int BACKWARD_PAWN_PENALTY_MG = -9;
static const int BACKWARD_PAWN_PENALTY_EG = -22;
static const int PASSED_PAWN_BONUS_MG[8]  = { 0, 5, 10, 20, 35, 60, 90, 0, };
static const int PASSED_PAWN_BONUS_EG[8]  = { 0, 10, 20, 40, 65, 95, 140, 0, };

/* NEW: Pawn chain — bonus for a pawn that is defended by another friendly
 * pawn (i.e., a friendly pawn sits on one of the two squares diagonally
 * behind it).  Pawn chains are harder to break and provide solid structure.
 * Values from SF 11's connected-pawn table (approximate). */
static const int PAWN_CHAIN_BONUS_MG = 11;
static const int PAWN_CHAIN_BONUS_EG = 4;

/* NEW: Pawn islands — penalty for each additional group of contiguous
 * pawns files (an "island").  More islands = harder to defend, weaker
 * structure.  Penalty per island AFTER the first. */
static const int PAWN_ISLAND_PENALTY_MG = -3;
static const int PAWN_ISLAND_PENALTY_EG = -8;

/*
 * Non-linear MobilityBonus tables (SF 11 values).
 * Each table is indexed by the number of accessible squares for that piece
 * type (capped at MOB_MAX).  The first few squares are worth the most;
 * flat per-square scoring over-values the 15th square relative to the 1st.
 */

/* Knights: 0..8 squares */
static const int MOB_KNIGHT_MG[] = {-62,-53,-12, -4,  3, 13, 22, 28, 33};
static const int MOB_KNIGHT_EG[] = {-81,-56,-30,-14,  8, 15, 23, 27, 33};

/* Bishops: 0..13 squares */
static const int MOB_BISHOP_MG[] = {-48,-20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98};
static const int MOB_BISHOP_EG[] = {-59,-23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97};

/* Rooks: 0..14 squares */
static const int MOB_ROOK_MG[] = {-58,-27,-15,-10, -5, -2,  9, 16, 30, 29, 32, 38, 46, 48, 58};
static const int MOB_ROOK_EG[] = {-76,-18, 28, 55, 69, 82,112,118,132,142,155,165,166,169,171};

/* Queens: 0..27 squares */
static const int MOB_QUEEN_MG[] = {
    -39,-21,  3,  3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66,
     67, 70, 71, 73, 79, 88, 88, 99,102,102,106,109,113,116};
static const int MOB_QUEEN_EG[] = {
    -36,-15,  8, 18, 34, 54, 61, 73, 79, 92, 94,104,113,120,
    123,126,133,136,140,143,148,166,170,175,184,191,206,212};

static const int ROOK_OPEN_FILE_MG  = 27;
static const int ROOK_OPEN_FILE_EG  = 57;
static const int ROOK_SEMIOPEN_MG   = 12;
static const int ROOK_SEMIOPEN_EG   = 7;
static const int ROOK_ON_SEVENTH_MG = 20;
static const int ROOK_ON_SEVENTH_EG = 32;
/* Rook on a file containing the enemy queen — small bonus for the latent
 * battery / pinning threat.  Classical Stockfish term ("Rook on queen file"). */
static const int ROOK_ON_QUEEN_FILE_MG = 7;
static const int ROOK_ON_QUEEN_FILE_EG = 5;

/* The MG bishop-pair value is intentionally not declared here — it is
 * already encoded in material_imbalance() via QuadOurs[0][0] = 1438
 * (≈ 90 cp MG after /16).  Declaring it as a separate constant would
 * generate an unused-constant warning. */
static const int BISHOP_PAIR_EG = 60;

/*
 * Bishop pawns ("bad bishop") — penalty for a bishop whose own pawns sit on
 * the same color complex as the bishop.  A bishop on a dark square with many
 * own pawns on dark squares loses mobility and gets walled in.  Classical
 * Stockfish term ("Bishop pawns").  Penalty per same-color own pawn.
 * Tuned conservative values; more in MG than EG (the bishop has more to do
 * in the middlegame).
 */
static const int BISHOP_PAWNS_MG = 5;
static const int BISHOP_PAWNS_EG = 3;
/* Bonus that scales the penalty when the bishop has more than 3 same-color
 * pawns (extra-bad case — the bishop is effectively neutralised). */
static const int BISHOP_PAWNS_HEAVY_MG = 4;
static const int BISHOP_PAWNS_HEAVY_THRESHOLD = 3;

/*
 * Long-diagonal bishop — bonus for a bishop sitting on one of the two long
 * diagonals (a1-h8 or a8-h1) when it sees the central squares.  A long-
 * diagonal bishop exerts pressure across the whole board for very little
 * material investment.  Classical Stockfish term ("Long diagonal bishop").
 */
static const int LONG_DIAGONAL_BISHOP_MG = 30;
static const int LONG_DIAGONAL_BISHOP_EG = 12;

/*
 * Queen infiltration — bonus for a queen that has pushed into enemy territory
 * (ranks 5-7 from our perspective).  A queen on the 6th or 7th harasses
 * pawns, threats mate, and pins pieces.  Classical Stockfish term
 * ("Queen infiltration").  Tiered by depth.
 */
static const int QUEEN_INFILTRATION_MG[4] = { 0, 10, 25, 40 }; /* ranks 5/6/7/8 */
static const int QUEEN_INFILTRATION_EG[4] = { 0, 15, 30, 45 };

/*
 * Pawnless flank — penalty when our king sits on a flank (queen-side or
 * king-side) with no friendly pawns at all.  Such a king is structurally
 * exposed even without enemy pieces nearby: every exchange opens a file
 * toward it.  Classical Stockfish term ("Pawnless flank").
 */
static const int PAWNLESS_FLANK_MG = 35;
static const int PAWNLESS_FLANK_EG = 8;

/*
 * Phalanx — two friendly pawns side-by-side on the same rank.  A phalanx
 * controls the two squares in front of it mutually and is a strong passed-
 * pawn candidate.  Classical Stockfish term ("Phalanx").  Applied per pawn
 * that has a friendly neighbour on the adjacent file at the same rank.
 */
static const int PHALANX_MG = 12;
static const int PHALANX_EG = 2;

/*
 * Candidate passed pawn — a pawn that is not yet passed but whose advance is
 * hard to stop because the enemy pawns that could block it are on the same
 * file (opposed) and cannot easily get support to the adjacent files.
 * Worth ~half of a true passed pawn, scaled by rank.  Classical Stockfish
 * term ("Candidate passed").
 */
static const int CANDIDATE_PASSED_MG[8] = { 0, 5,  9, 13, 20, 35, 0, 0 };
static const int CANDIDATE_PASSED_EG[8] = { 0, 7, 11, 18, 30, 55, 0, 0 };

/*
 * Hanging pawns — the specific pawn structure: two friendly pawns on c4+d4
 * (or mirrored) with no friendly pawns on the b/e files and no enemy pawns
 * on the c/d files.  Strong centrally but structurally weak (cannot be
 * defended by other pawns).  Classical Stockfish term ("Hanging").  Net
 * effect is a small penalty in MG (weakness) and small bonus in EG
 * (centralisation pays off when pieces come off).
 */
static const int HANGING_PAWNS_MG = -14;
static const int HANGING_PAWNS_EG =  8;

/*
 * Endgame shelter — small EG bonus for retaining pawn shelter in front of
 * the king.  Even in endgames the king wants a pawn shield; the effect is
 * much smaller than in the middlegame.  Classical Stockfish term
 * ("Endgame shelter").
 */
static const int KING_SHIELD_BONUS_EG = 2;

/*
 * Space — bonus proportional to the number of squares behind our pawns that
 * we control (not attacked by enemy pawns).  More space means more room to
 * manoeuvre and harder for the opponent to defend.  Classical Stockfish term
 * ("Space").  MG-only — in the endgame the kings walk out and space matters
 * far less.  The raw count is divided by a scaling factor (here ~12) so the
 * bonus stays in a sensible range.
 */
static const int SPACE_SCALE = 12;
static const int SPACE_MG_PER_SQUARE = 7;

/*
 * Knight on queen — bonus for a knight that attacks the enemy queen.  Even
 * without an immediate tactical threat, a knight annoying the queen exerts
 * a strong positional pull.  Classical Stockfish term ("Knight on queen").
 */
static const int KNIGHT_ON_QUEEN_MG = 12;
static const int KNIGHT_ON_QUEEN_EG = 6;

/*
 * Slider on queen — bonus for a bishop/rook that attacks the enemy queen
 * along a ray.  Pin or skewer potential.  Classical Stockfish term
 * ("Slider on queen").  Bishop and rook get slightly different weights.
 */
static const int BISHOP_ON_QUEEN_MG = 8;
static const int BISHOP_ON_QUEEN_EG = 4;
static const int ROOK_ON_QUEEN_ATTACK_MG = 6;
static const int ROOK_ON_QUEEN_ATTACK_EG = 3;

static const int OUTPOST_KNIGHT_MG = 22;
static const int OUTPOST_KNIGHT_EG = 14;
static const int OUTPOST_BISHOP_MG = 12;
static const int OUTPOST_BISHOP_EG = 8;

static const int KING_SHIELD_BONUS        = 7;
static const int KING_OPEN_FILE_PENALTY   = -25;
static const int KING_ATTACKER_WEIGHT[6]  = { 0, 20, 20, 40, 80, 0, };

/*
 * King endgame activity: per-Chebyshev-step penalty for being far from the
 * enemy king.  PST_KING_EG rewards centrality; this term rewards king-to-king
 * proximity (boxing-in bonus), which the PST cannot express because it only
 * sees one king's position.  Range: [4 cp at dist 1] to [28 cp at dist 7].
 */
static const int KING_EG_DISTANCE_PENALTY = 4;

#define TEMPO_BONUS_MG 16
/*
 * Tempo EG — a small tempo bonus for the side to move in the endgame.
 * The original engine set this to 0, but in endgames with passed-pawn races
 * and opposition battles the side-to-move advantage is real (the player who
 * just moved must commit first).  Modern engines typically use 8-15 cp EG.
 */
#define TEMPO_BONUS_EG 10

/* Mobility evaluation indexing */
static const int * const MOB_MG_TABLE[6] = {
    NULL, MOB_KNIGHT_MG, MOB_BISHOP_MG, MOB_ROOK_MG, MOB_QUEEN_MG, NULL};
static const int * const MOB_EG_TABLE[6] = {
    NULL, MOB_KNIGHT_EG, MOB_BISHOP_EG, MOB_ROOK_EG, MOB_QUEEN_EG, NULL};
static const int MOB_MAX[6] = {0, 8, 13, 14, 27, 0};

/* Table-of-tables for quick lookup */
static const int * const PST_MG[6] = {
    PST_PAWN_MG, PST_KNIGHT_MG, PST_BISHOP_MG,
    PST_ROOK_MG, PST_QUEEN_MG,  PST_KING_MG
};
static const int * const PST_EG[6] = {
    PST_PAWN_EG, PST_KNIGHT_EG, PST_BISHOP_EG,
    PST_ROOK_EG, PST_QUEEN_EG,  PST_KING_EG
};

/* Flip square for Black (mirror vertically) */
static inline int pst_sq(Color c, Square sq) {
    return (c == WHITE) ? (int)sq : ((int)sq ^ 56);
}

/*
 * Chebyshev (king-move) distance between two squares.
 * = max(|file_delta|, |rank_delta|)
 * Used by the king-safety model to weight attacks by proximity.
 */
static inline int chebyshev(int sq1, int sq2) {
    int df = (sq1 & 7) - (sq2 & 7); if (df < 0) df = -df;
    int dr = (sq1 >> 3) - (sq2 >> 3); if (dr < 0) dr = -dr;
    return (df > dr) ? df : dr;
}

/* ──────────────────────────────────────────────
 *  Phase calculation
 *  Phase 24 = full midgame; 0 = full endgame.
 *  Pawns and kings are not counted.
 * ────────────────────────────────────────────── */
static const int PHASE_INC[6] = { 0, 1, 1, 2, 4, 0 };
#define MAX_PHASE 24

int game_phase(const Board *b) {
    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_INC[pt] * bb_popcount(b->pieces[c][pt]);
    return (phase > MAX_PHASE) ? MAX_PHASE : phase;
}

/* Blend MG and EG scores by phase */
static inline int taper(int mg, int eg, int phase) {
    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
}

/* ──────────────────────────────────────────────
 *  Pawn hash cache
 *
 *  Pawn structure evaluation is expensive and depends only on the two pawn
 *  bitboards plus the colour being evaluated. A small direct-mapped cache
 *  avoids recomputing doubled / isolated / passed / backward pawn terms for
 *  positions that recur through transpositions or iterative deepening.
 * ────────────────────────────────────────────── */
#define PAWN_CACHE_SIZE 2048

typedef struct {
    Bitboard wp;
    Bitboard bp;
    Color    us;
    int      mg;
    int      eg;
    bool     valid;
} PawnCacheEntry;

static PawnCacheEntry pawn_cache[PAWN_CACHE_SIZE];

static inline unsigned pawn_cache_idx(Bitboard wp, Bitboard bp, Color us) {
    uint64_t key = (uint64_t)wp
                 ^ ((uint64_t)bp << 1)
                 ^ ((uint64_t)us << 63)
                 ^ ((uint64_t)wp >> 17)
                 ^ ((uint64_t)bp >> 29);
    key ^= key >> 32;
    key ^= key >> 16;
    return (unsigned)key & (PAWN_CACHE_SIZE - 1);
}

static bool pawn_cache_probe(Bitboard wp, Bitboard bp, Color us, int *mg, int *eg) {
    PawnCacheEntry *e = &pawn_cache[pawn_cache_idx(wp, bp, us)];
    if (e->valid && e->wp == wp && e->bp == bp && e->us == us) {
        *mg = e->mg;
        *eg = e->eg;
        return true;
    }
    return false;
}

static void pawn_cache_store(Bitboard wp, Bitboard bp, Color us, int mg, int eg) {
    PawnCacheEntry *e = &pawn_cache[pawn_cache_idx(wp, bp, us)];
    e->wp = wp;
    e->bp = bp;
    e->us = us;
    e->mg = mg;
    e->eg = eg;
    e->valid = true;
}

/* ──────────────────────────────────────────────
 *  Pawn structure evaluation (one side)
 * ────────────────────────────────────────────── */
static void eval_pawns_uncached(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard pawns_copy  = our_pawns;
    Bitboard passed_bb   = 0;  /* track passed pawns for connected-passer bonus */

    while (pawns_copy) {
        int sq   = bb_pop(&pawns_copy);
        int file = sq & 7;
        int rank = sq >> 3;

        /* Adjacent file masks */
        Bitboard adj_files = 0;
        if (file > 0) adj_files |= FILE_BB[file - 1];
        if (file < 7) adj_files |= FILE_BB[file + 1];

        /* ── Doubled pawn ── */
        if (bb_popcount(our_pawns & FILE_BB[file]) > 1) {
            *mg += DOUBLED_PAWN_PENALTY_MG;
            *eg += DOUBLED_PAWN_PENALTY_EG;
        }

        /* ── Isolated pawn ── */
        bool is_isolated = !(our_pawns & adj_files);
        if (is_isolated) {
            *mg += ISOLATED_PAWN_PENALTY_MG;
            *eg += ISOLATED_PAWN_PENALTY_EG;
        }

        /* ── Passed pawn ──
         * No enemy pawns on same or adjacent files ahead of this pawn. */
        Bitboard ahead_mask = 0;
        if (us == WHITE) {
            for (int r = rank + 1; r < 8; r++) ahead_mask |= RANK_BB[r];
        } else {
            for (int r = 0; r < rank; r++)      ahead_mask |= RANK_BB[r];
        }
        Bitboard enemy_span = (their_pawns & (FILE_BB[file] | adj_files))
                              & ahead_mask;
        bool is_passed = !enemy_span;
        if (is_passed) {
            int bonus_rank = (us == WHITE) ? rank : (7 - rank);
            *mg += PASSED_PAWN_BONUS_MG[bonus_rank];
            *eg += PASSED_PAWN_BONUS_EG[bonus_rank];
            passed_bb |= SQUARE_BB[sq];  /* track for connected-passer check */

            /*
             * ── King-distance term (EG only — MG tactics handled by search) ──
             *
             * A passed pawn is more valuable when:
             *   - our own king is close to the promotion square (defends it)
             *   - the enemy king is far from the promotion square (can't catch it)
             * The metric is the difference in Chebyshev distances to the
             * promotion square.  Each unit of difference is ~5 cp EG.
             *
             * Promotion square = the back-rank square on the pawn's file.
             */
            int promo_sq = (us == WHITE) ? (file | 56) : file;
            int our_king = bb_lsb(b->pieces[us  ][KING]);
            int opp_king = bb_lsb(b->pieces[them][KING]);
            int d_our = chebyshev(our_king, promo_sq);
            int d_opp = chebyshev(opp_king, promo_sq);
            /* Closer = smaller d.  Each unit of (d_opp - d_our) helps
             * by ~5 cp.  Clamp to avoid runaway scores on far-apart
             * kings. */
            int king_dist_bonus = 5 * (d_opp - d_our);
            if (king_dist_bonus >  60) king_dist_bonus =  60;
            if (king_dist_bonus < -60) king_dist_bonus = -60;
            *eg += king_dist_bonus;

            /*
             * ── Blockage penalty ──
             *
             * An enemy non-pawn piece sitting on the square directly
             * in front of the passer suppresses its advance.  An enemy
             * KING blockading is the worst (it can never be chased
             * away by the pawn itself).
             */
            int block_sq = (us == WHITE) ? sq + 8 : sq - 8;
            if (block_sq >= 0 && block_sq < 64) {
                Bitboard blocker = b->occ[them] & SQUARE_BB[block_sq];
                if (blocker
                    && !(b->pieces[them][PAWN] & SQUARE_BB[block_sq])) {
                    *mg -= 12;
                    *eg -= 24;
                    if (b->pieces[them][KING] & SQUARE_BB[block_sq])
                        *eg -= 18;  /* enemy king blockading = ~dead passer */
                }
            }
        }

        /*
         * ── Backward pawn ──
         *
         * A pawn is backward when:
         *   1. Its stop square (one step forward) is controlled by an enemy
         *      pawn — it cannot safely advance.
         *   2. It has no friendly pawn support from behind on adjacent files.
         *   3. It is not already passed.
         *
         * Stop-square control uses PAWN_ATTACKS[us][stop] & their_pawns:
         *   PAWN_ATTACKS[WHITE][stop] = squares diagonally ahead of stop
         *   from WHITE's perspective — exactly where a BLACK pawn would need
         *   to sit to attack stop.  The formula is symmetric for BLACK.
         *
         * The behind-span uses clean rank-loop iteration instead of
         * full-board bit arithmetic (SQUARE_BB[sq]-1 etc.), which is both
         * easier to read and avoids edge-case surprises at sq==0 or sq==63.
         */
        if (!is_passed) {
            int stop = (us == WHITE) ? sq + 8 : sq - 8;
            if (stop >= 0 && stop < 64) {
                bool stop_attacked = (PAWN_ATTACKS[us][stop] & their_pawns) != 0;

                if (stop_attacked) {
                    /* All squares on adjacent files strictly behind this pawn */
                    Bitboard behind_span = 0;
                    if (us == WHITE) {
                        for (int r = 0; r < rank; r++)
                            behind_span |= RANK_BB[r];
                    } else {
                        for (int r = rank + 1; r < 8; r++)
                            behind_span |= RANK_BB[r];
                    }
                    behind_span &= adj_files;

                    if (!(our_pawns & behind_span)) {
                        *mg += BACKWARD_PAWN_PENALTY_MG;
                        *eg += BACKWARD_PAWN_PENALTY_EG;
                    }
                }
            }

            /*
             * ── Candidate passed pawn ──
             *
             * A candidate is a pawn that is not yet passed but whose advance
             * is hard to stop: there is at most one enemy pawn in its path
             * (on the same file or adjacent files ahead), and we have
             * enough own-pawn support that the enemy pawn will be traded
             * away rather than block us permanently.  We approximate the
             * SF definition: the pawn is opposed by exactly one enemy pawn
             * on the same file, and no enemy pawns on the adjacent files
             * ahead.  Worth ~half of a true passed pawn.
             */
            {
                Bitboard same_file_ahead = their_pawns & FILE_BB[file] & ahead_mask;
                Bitboard adj_file_ahead  = their_pawns & adj_files   & ahead_mask;
                if (same_file_ahead && !adj_file_ahead
                    && bb_popcount(same_file_ahead) == 1) {
                    int bonus_rank = (us == WHITE) ? rank : (7 - rank);
                    if (bonus_rank >= 1 && bonus_rank <= 6) {
                        *mg += CANDIDATE_PASSED_MG[bonus_rank];
                        *eg += CANDIDATE_PASSED_EG[bonus_rank];
                    }
                }
            }
        }

        /*
         * ── Phalanx ──
         *
         * Two friendly pawns side-by-side on the same rank form a "phalanx" —
         * they defend the two squares in front of them mutually and are a
         * strong passed-pawn / centre-formation building block.  We award a
         * per-pawn bonus when this pawn has a friendly neighbour on an
         * adjacent file at the same rank.  (Each phalanx pair gets counted
         * twice — once per pawn — which matches the SF convention.)
         */
        if (file > 0) {
            int left_sq = sq - 1;
            if (our_pawns & SQUARE_BB[left_sq]) {
                *mg += PHALANX_MG;
                *eg += PHALANX_EG;
            }
        }

        /*
         * NEW: ── Pawn chain (defended pawn bonus) ──
         *
         * A pawn is "in a chain" if a friendly pawn defends it — i.e.,
         * a friendly pawn sits on one of the two squares diagonally
         * behind it (from our perspective).  Such pawns are harder to
         * attack and form a solid structure.  SF 11 gives a small
         * per-defender bonus; we use a flat bonus per defended pawn
         * (capped at +1 even if both defenders exist, since double-
         * defense isn't much better than single).
         */
        {
            int behind1, behind2;
            if (us == WHITE) {
                behind1 = sq - 9;  /* down-left */
                behind2 = sq - 7;  /* down-right */
            } else {
                behind1 = sq + 7;  /* up-left */
                behind2 = sq + 9;  /* up-right */
            }
            bool defended = false;
            if (behind1 >= 0 && behind1 < 64 && file > 0
                && (our_pawns & SQUARE_BB[behind1]))
                defended = true;
            if (!defended && behind2 >= 0 && behind2 < 64 && file < 7
                && (our_pawns & SQUARE_BB[behind2]))
                defended = true;
            if (defended) {
                *mg += PAWN_CHAIN_BONUS_MG;
                *eg += PAWN_CHAIN_BONUS_EG;
            }
        }
    }

    /*
     * ── Hanging pawns (structural term, evaluated once per side) ──
     *
     * The "hanging pawns" structure is two friendly pawns side-by-side on
     * the c/d files (or e/f for the mirror), with NO friendly pawns on
     * either adjacent outer file (b/f or a/d) and NO enemy pawns on the
     * same two files.  They form a strong centre but cannot be defended by
     * other pawns, so the side must defend them with pieces.
     *
     * Detection: find friendly pawns on adjacent files c+d (or e+f, or
     * b+c, etc. — actually SF checks all adjacent-file pairs).  We use a
     * simpler, more lenient version: any pair of friendly pawns on
     * adjacent files at the same rank, where (a) no friendly pawns exist
     * on the next file outward, and (b) no enemy pawns exist on either of
     * the two pawn files.  This is a per-pair bonus awarded once.
     */
    {
        Bitboard p = our_pawns;
        while (p) {
            int sq1 = bb_pop(&p);
            int f1 = sq1 & 7, r1 = sq1 >> 3;
            int f2 = f1 + 1;
            if (f2 > 7) continue;
            /* Same-rank friendly pawn on the next file? */
            int sq2 = r1 * 8 + f2;
            if (!(our_pawns & SQUARE_BB[sq2])) continue;
            /* No friendly pawns on the next-outward files. */
            int f_left  = f1 - 1;
            int f_right = f2 + 1;
            Bitboard outer = 0;
            if (f_left  >= 0) outer |= FILE_BB[f_left];
            if (f_right <= 7) outer |= FILE_BB[f_right];
            if (our_pawns & outer) continue;
            /* No enemy pawns on either of the two pawn files. */
            if (their_pawns & (FILE_BB[f1] | FILE_BB[f2])) continue;
            /* Confirmed hanging pair — apply once. */
            *mg += HANGING_PAWNS_MG;
            *eg += HANGING_PAWNS_EG;
            break;  /* only one bonus per side */
        }
    }

    /*
     * NEW: ── Pawn islands ──
     *
     * An "island" is a maximal run of contiguous files that contain at
     * least one friendly pawn.  E.g. pawns on a, b, d, g, h → 3 islands
     * ({a,b}, {d}, {g,h}).  More islands = weaker structure (each island
     * must be defended separately; no mutual support across gaps).
     *
     * We apply a small per-island penalty AFTER the first island — i.e.,
     * 0 islands = 0 penalty, 1 island = 0 penalty, 2 islands = 1×penalty,
     * 3 islands = 2×penalty, etc.  This avoids penalising a single
     * connected pawn group and focuses on the fragmentation itself.
     */
    {
        int islands = 0;
        bool in_island = false;
        for (int f = 0; f < 8; f++) {
            bool has_pawn = (our_pawns & FILE_BB[f]) != 0;
            if (has_pawn && !in_island) {
                islands++;
                in_island = true;
            } else if (!has_pawn) {
                in_island = false;
            }
        }
        if (islands > 1) {
            int extra = islands - 1;
            *mg += PAWN_ISLAND_PENALTY_MG * extra;
            *eg += PAWN_ISLAND_PENALTY_EG * extra;
        }
    }

    /*
     * NEW: ── Connected passed pawns ──
     *
     * Two passed pawns on adjacent files, especially on the same or
     * nearby ranks, are far more dangerous than the sum of their
     * individual values.  They defend each other's advance and create
     * an unstoppable "phalanx" that the opponent cannot blockade with
     * a single piece.  We award a bonus per connected pair (counted
     * once per pawn, matching the phalanx convention).
     *
     * Values scale with rank — connected passers on the 6th/7th rank
     * are nearly always winning.
     */
    if (passed_bb) {
        Bitboard pp = passed_bb;
        while (pp) {
            int sq = bb_pop(&pp);
            int file = sq & 7;
            int rank = sq >> 3;
            int bonus_rank = (us == WHITE) ? rank : (7 - rank);

            /* Check adjacent files for another passed pawn */
            Bitboard adj_files_bb = 0;
            if (file > 0) adj_files_bb |= FILE_BB[file - 1];
            if (file < 7) adj_files_bb |= FILE_BB[file + 1];

            /* Same rank or one rank apart (mutual support range) */
            Bitboard support_zone = adj_files_bb
                                  & (RANK_BB[rank]
                                     | (rank > 0 ? RANK_BB[rank - 1] : 0)
                                     | (rank < 7 ? RANK_BB[rank + 1] : 0));

            if (passed_bb & support_zone) {
                /* Bonus scales with rank: +10 MG / +20 EG at rank 2,
                 * up to +30 MG / +60 EG at rank 6. */
                int conn_mg = 5 + 5 * bonus_rank;
                int conn_eg = 10 + 10 * bonus_rank;
                if (conn_mg > 30) conn_mg = 30;
                if (conn_eg > 60) conn_eg = 60;
                *mg += conn_mg;
                *eg += conn_eg;
            }
        }
    }
}

static void eval_pawns(const Board *b, Color us, int *mg, int *eg) {
    Bitboard wp = b->pieces[WHITE][PAWN];
    Bitboard bp = b->pieces[BLACK][PAWN];

    int cached_mg, cached_eg;
    if (pawn_cache_probe(wp, bp, us, &cached_mg, &cached_eg)) {
        *mg += cached_mg;
        *eg += cached_eg;
        return;
    }

    int local_mg = 0;
    int local_eg = 0;
    eval_pawns_uncached(b, us, &local_mg, &local_eg);

    pawn_cache_store(wp, bp, us, local_mg, local_eg);
    *mg += local_mg;
    *eg += local_eg;
}

static void eval_mobility(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard occ    = b->occ[2];
    Bitboard not_us = ~b->occ[us];

    /*
     * Squares attacked by enemy pawns are unsafe — moving a piece
     * there would just give it away.  SF/Ethereal's mobility excludes
     * these squares; the original code counted them, overvaluing
     * pieces that hover in front of the enemy pawn chain (especially
     * knights and bishops on the rim).
     *
     * Compute the enemy-pawn attack set once:
     *   - WHITE pawns attack NE (+9) and NW (+7), with file-edge masks.
     *   - BLACK pawns attack SE (-7) and SW (-9), symmetric.
     */
    Bitboard ep = b->pieces[them][PAWN];
    Bitboard epa;
    if (them == WHITE) {
        epa = ((ep & ~FILE_BB[0]) << 7) | ((ep & ~FILE_BB[7]) << 9);
    } else {
        epa = ((ep & ~FILE_BB[7]) >> 7) | ((ep & ~FILE_BB[0]) >> 9);
    }
    Bitboard safe = not_us & ~epa;

    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk = 0;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq]; break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default: break;
            }
            /* Use SAFE squares (excluding enemy-pawn-attacked) for the
             * mobility count.  This is the SF/Ethereal convention. */
            int mob = bb_popcount(atk & safe);
            if (mob > MOB_MAX[pt]) mob = MOB_MAX[pt];
            *mg += MOB_MG_TABLE[pt][mob];
            *eg += MOB_EG_TABLE[pt][mob];
        }
    }
}


static void eval_rooks(const Board *b, Color us, int *mg, int *eg) {
    Color them           = us ^ 1;
    Bitboard our_pawns   = b->pieces[us  ][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];
    Bitboard rooks       = b->pieces[us  ][ROOK];

    int rank7 = (us == WHITE) ? 6 : 1;
    int rank8 = (us == WHITE) ? 7 : 0;

    while (rooks) {
        int sq   = bb_pop(&rooks);
        int file = sq & 7;
        int rank = sq >> 3;
        Bitboard file_bb = FILE_BB[file];

        bool no_own_pawn   = !(our_pawns   & file_bb);
        bool no_enemy_pawn = !(their_pawns & file_bb);

        if (no_own_pawn && no_enemy_pawn) {
            *mg += ROOK_OPEN_FILE_MG;
            *eg += ROOK_OPEN_FILE_EG;
        } else if (no_own_pawn) {
            *mg += ROOK_SEMIOPEN_MG;
            *eg += ROOK_SEMIOPEN_EG;
        }

        /*
         * TrappedRook: a rook with very limited mobility (<=3 squares) on
         * a closed file is penalised, especially when trapped by its own king.
         */
        if (!no_own_pawn && !no_enemy_pawn) {
            Bitboard rook_mob = rook_attacks((Square)sq, b->occ[2]) & ~b->occ[us];
            int mob = bb_popcount(rook_mob);
            if (mob <= 3) {
                int king_sq = bb_lsb(b->pieces[us][KING]);
                int kf = king_sq & 7;
                /* Penalise when the rook is hemmed in on the same
                 * flank as its own king.  The previous test
                 * `(kf < 4) == (file < kf)` was wrong — it tested
                 * "rook strictly left of king" rather than "same flank".
                 * Correct test: both on queenside (file < 4 and kf < 4)
                 * or both on kingside (file >= 4 and kf >= 4). */
                if ((kf < 4) == (file < 4)) {
                    *mg -= 52;
                    *eg -= 10;
                }
            }
        }

        /*
         * Rook on the 7th rank: valuable when the enemy king is on the 8th
         * rank or there are enemy pawns on the 7th rank to harass.
         */
        if (rank == rank7) {
            int enemy_king_sq = bb_lsb(b->pieces[them][KING]);
            bool enemy_king_on_8th = ((enemy_king_sq >> 3) == rank8);
            bool pawns_on_7th      = (their_pawns & RANK_BB[rank7]) != 0;

            if (enemy_king_on_8th || pawns_on_7th) {
                *mg += ROOK_ON_SEVENTH_MG;
                *eg += ROOK_ON_SEVENTH_EG;
            }
        }

        /*
         * Rook on queen file: small bonus when the rook sits on a file
         * that contains an enemy queen.  This represents a latent
         * battery / pinning threat along the file, separate from the
         * direct "rook attacks queen" tactical bonus awarded elsewhere.
         * Only credit when the file is not already open or semi-open
         * (those cases get the larger ROOK_OPEN_FILE bonus instead, and
         * crediting again would double-count).
         */
        if (!(no_own_pawn && no_enemy_pawn) && !no_own_pawn) {
            Bitboard enemy_queen_file = 0;
            Bitboard eqs = b->pieces[them][QUEEN];
            while (eqs) {
                int eqs_sq = bb_pop(&eqs);
                enemy_queen_file |= FILE_BB[eqs_sq & 7];
            }
            if (enemy_queen_file & file_bb) {
                *mg += ROOK_ON_QUEEN_FILE_MG;
                *eg += ROOK_ON_QUEEN_FILE_EG;
            }
        }
    }
}

/*
 * eval_outposts — Clean per-piece front-span / attack-mask approach.
 *
 * OLD approach: precomputed enemy_pawn_attack_span by looping over every
 * enemy pawn and filling whole ranks ahead of it on adjacent files.  This
 * was an over-approximation (it could flag squares unreachable due to
 * blockades) and the fill direction was easy to mis-index.
 *
 * NEW approach: for each candidate piece check directly:
 *   1. Is the square in the outpost rank range (ranks 4–6 from our side)?
 *   2. Is it pawn-supported?  A friendly pawn attacks this square when
 *      PAWN_ATTACKS[them][sq] & our_pawns is non-zero (by symmetry of
 *      pawn-attack geometry).
 *   3. Is there NO enemy pawn on adjacent files that could advance to
 *      attack it in the future? — the "future-threat span".
 *
 * Future-threat span:
 *   For us=WHITE: BLACK pawns advance downward.  Any BLACK pawn on an
 *   adjacent file at rank > sq_rank can advance to sq_rank+1 and attack
 *   sq diagonally, so the span is adj_files & {ranks > sq_rank}.
 *
 *   For us=BLACK: WHITE pawns advance upward.  Any WHITE pawn on an
 *   adjacent file at rank < sq_rank threatens sq, so the span is
 *   adj_files & {ranks < sq_rank}.
 *
 * This is explicit, symmetric, and correct at every file and rank.
 */
static void eval_outposts(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard our_pawns   = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];

    /* Outpost ranks: 4–6 from our perspective (0-indexed). */
    int r_min = (us == WHITE) ? 3 : 2;
    int r_max = (us == WHITE) ? 5 : 4;
    Bitboard outpost_rank_mask = 0;
    for (int r = r_min; r <= r_max; r++) outpost_rank_mask |= RANK_BB[r];

    /* Evaluate knights and bishops with identical logic, different bonuses. */
    for (int pt = KNIGHT; pt <= BISHOP; pt++) {
        int bonus_mg = (pt == KNIGHT) ? OUTPOST_KNIGHT_MG : OUTPOST_BISHOP_MG;
        int bonus_eg = (pt == KNIGHT) ? OUTPOST_KNIGHT_EG : OUTPOST_BISHOP_EG;

        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq   = bb_pop(&pieces);
            int file = sq & 7;
            int rank = sq >> 3;

            /* Must be in the outpost rank range */
            if (!(SQUARE_BB[sq] & outpost_rank_mask)) continue;

            /*
             * Pawn support check: PAWN_ATTACKS[them][sq] gives squares from
             * which a them-coloured pawn would attack sq; by symmetry these
             * are exactly the squares from which one of OUR pawns attacks sq.
             */
            if (!(PAWN_ATTACKS[them][sq] & our_pawns)) continue;

            /* Adjacent files */
            Bitboard adj_files = 0;
            if (file > 0) adj_files |= FILE_BB[file - 1];
            if (file < 7) adj_files |= FILE_BB[file + 1];

            /* Future-threat span: ranks from which an enemy pawn can advance
             * to attack this square.  See function-level comment above. */
            Bitboard future_threat = 0;
            if (us == WHITE) {
                for (int r = rank + 1; r < 8; r++) future_threat |= RANK_BB[r];
            } else {
                for (int r = 0; r < rank; r++)     future_threat |= RANK_BB[r];
            }
            future_threat &= adj_files;

            if (their_pawns & future_threat) continue;  /* can be chased off */

            *mg += bonus_mg;
            *eg += bonus_eg;
        }
    }
}

/*
 * eval_king_safety — Tapered, distance-weighted attack model.
 *
 * Changes from the original:
 *
 * MG — Distance-weighted attacks:
 *   The original gave each attacking piece a flat weight regardless of how
 *   close it was to the king.  A queen on the opposite side of the board is
 *   far less threatening than one sitting on the f-file adjacent to the king.
 *   We now scale each attacker's base weight by its Chebyshev distance to
 *   our king (scale 4× / 2× / 1× for dist ≤1 / dist 2–3 / dist ≥4, with a
 *   final divide-by-2 to keep values in the same ballpark as before).
 *
 * EG — King activity (king-proximity bonus):
 *   PST_KING_EG already rewards centralization.  It cannot express the
 *   dynamic bonus for approaching the enemy king (crucial for K+P endings,
 *   opposition, and mating nets).  We subtract `dist * KING_EG_DISTANCE_PENALTY`
 *   from the EG score so that the winning king is motivated to close in.
 *   The function now takes *eg so this term feeds into the tapered blend.
 */
static void eval_king_safety(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    int king_sq   = bb_lsb(b->pieces[us][KING]);
    int king_file = king_sq & 7;
    int king_rank = king_sq >> 3;

    Bitboard our_pawns = b->pieces[us][PAWN];

    /* ── 1. Pawn shield + open-file penalty ── */
    for (int f = king_file - 1; f <= king_file + 1; f++) {
        if (f < 0 || f > 7) continue;
        Bitboard file_pawns = our_pawns & FILE_BB[f];

        if (!file_pawns) {
            *mg += KING_OPEN_FILE_PENALTY;
        } else {
            int shield1 = (us == WHITE) ? king_rank + 1 : king_rank - 1;
            int shield2 = (us == WHITE) ? king_rank + 2 : king_rank - 2;
            if (shield1 >= 0 && shield1 < 8 && (file_pawns & RANK_BB[shield1])) {
                *mg += KING_SHIELD_BONUS * 2;
                *eg += KING_SHIELD_BONUS_EG;
            } else if (shield2 >= 0 && shield2 < 8 && (file_pawns & RANK_BB[shield2])) {
                *mg += KING_SHIELD_BONUS;
            }
        }
    }

    /*
     * ── 1a. Pawnless flank ──
     *
     * If our king is on a flank (king-side files fgh or queen-side files
     * abc) with NO friendly pawns on that entire flank, that's a structural
     * weakness even without enemy piece pressure: any exchange opens a file
     * directly toward the king.  Apply a flat penalty, larger in MG where
     * the king is committed to its corner, smaller in EG where the king
     * can walk away.  Classical SF term ("Pawnless flank").
     */
    {
        Bitboard flank_files;
        if (king_file <= 2) {
            flank_files = FILE_BB[0] | FILE_BB[1] | FILE_BB[2];
        } else if (king_file >= 5) {
            flank_files = FILE_BB[5] | FILE_BB[6] | FILE_BB[7];
        } else {
            flank_files = 0;  /* king on central file — no flank */
        }
        if (flank_files && !(our_pawns & flank_files)) {
            *mg += PAWNLESS_FLANK_MG;
            *eg += PAWNLESS_FLANK_EG;
        }
    }

    /*
     * ── 1b. Pawn storm: enemy pawns advancing on the king's files ──
     *
     * Pawns are the single most important danger signal in classical
     * king safety — a pawn storm opens files toward the king and is
     * the precursor to most mating attacks.  We scan the three files
     * around the king (king's file ± 1) for enemy pawns and credit a
     * penalty proportional to how advanced each pawn is toward our
     * king.  A pawn on the king's own file (no friendly pawn shield
     * possible) is extra-dangerous.
     *
     * This term is MG-only: in the endgame the kings walk out and
     * pawn storms are no longer mating threats.
     */
    {
        Bitboard storm_files = FILE_BB[king_file];
        if (king_file > 0) storm_files |= FILE_BB[king_file - 1];
        if (king_file < 7) storm_files |= FILE_BB[king_file + 1];

        Bitboard storm = b->pieces[them][PAWN] & storm_files;
        while (storm) {
            int s = bb_pop(&storm);
            int r = s >> 3;
            /* How many ranks has the stormer advanced toward our king?
             * For us=WHITE the stormer comes from rank 6 down (BLACK pawns
             * advance downward); we measure (king_rank - r) so a pawn
             * adjacent to the king on rank 5 (king on rank 1) gives 4. */
            int advanced = (us == WHITE) ? (king_rank - r) : (r - king_rank);
            if (advanced >= 0) {
                bool on_king_file = ((s & 7) == king_file);
                *mg -= 6 * advanced + (on_king_file ? 8 : 0);
            }
        }
    }

    /*
     * ── 2. Distance-weighted enemy piece attacks on king zone ──
     *
     * King zone = king square + all king-move squares (up to 9 squares).
     *
     * For each enemy piece that attacks into the zone, scale its weight by
     * proximity to our king:
     *
     *   dist <= 1 (adjacent or on king sq): scale 4  — immediate threat,
     *             can deliver check or capture next move.
     *   dist 2-3 (nearby):                 scale 2  — standard threat.
     *   dist >= 4 (far):                   scale 1  — long-range pressure.
     *
     * Divide by 2 at the end so the effective multipliers are 2 / 1 / 0.5,
     * keeping total weights comparable to the original flat model.
     * The quadratic penalty for 2+ pieces is preserved.
     */
    Bitboard king_zone = KING_ATTACKS[king_sq] | SQUARE_BB[king_sq];
    Bitboard occ       = b->occ[2];

    int attack_count  = 0;
    int attack_weight = 0;

    for (int pt = KNIGHT; pt < KING; pt++) {
        Bitboard pieces = b->pieces[them][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);
            Bitboard atk;
            switch (pt) {
                case KNIGHT: atk = KNIGHT_ATTACKS[sq]; break;
                case BISHOP: atk = bishop_attacks((Square)sq, occ); break;
                case ROOK:   atk = rook_attacks  ((Square)sq, occ); break;
                case QUEEN:  atk = queen_attacks  ((Square)sq, occ); break;
                default:     atk = 0; break;
            }
            if (atk & king_zone) {
                int dist  = chebyshev(sq, king_sq);
                int scale = (dist <= 1) ? 4 : (dist <= 3) ? 2 : 1;
                attack_count++;
                attack_weight += KING_ATTACKER_WEIGHT[pt] * scale;

                /* NEW: Per-square attack table — count how many king-zone
                 * squares this piece attacks.  More zone coverage = more
                 * dangerous (the king has fewer escape squares).  This is
                 * the SF-style "kingAttacksCount" approach. */
                int zone_squares = bb_popcount(atk & king_zone);
                /* Each zone square attacked adds a small bonus, scaled
                 * by piece type (queen touching 5 zone squares is far
                 * worse than a knight touching 2). */
                int zone_bonus = zone_squares * KING_ATTACKER_WEIGHT[pt] / 20;
                attack_weight += zone_bonus;
            }
        }
    }

    /* Restore to effective 2×/1×/0.5× scale */
    attack_weight /= 2;

    /*
     * Scale danger by the attacker's offensive potential.  Without a
     * queen, mate is essentially impossible — a lone knight or bishop
     * reaching the king zone is not the same kind of threat as a
     * queen-led attack.  We:
     *   • Zero out the danger when the attacker has no queen AND no
     *     rook (you need at least a major piece to mate).
     *   • Quarter the danger when the attacker has rooks but no queen.
     * This matches the SF convention and prevents false king-safety
     * penalties in minor-piece endgames.
     */
    bool enemy_has_queen = (b->pieces[them][QUEEN] != 0);
    if (!enemy_has_queen) {
        if (b->pieces[them][ROOK] == 0) {
            attack_count  = 0;
            attack_weight = 0;
        } else {
            attack_weight /= 4;
        }
    }

    /* Quadratic penalty when 2+ coordinated pieces attack */
    if (attack_count >= 2) {
        int danger = attack_weight * attack_weight / 200;
        *mg -= danger;
    }

    /*
     * NEW: ── Safe check bonus ──
     *
     * A "safe check" is a check from a square where the checking piece
     * is NOT attacked by any enemy pawn.  Safe checks are powerful
     * because they force the king to move or block, creating tempo for
     * building a mating net.
     *
     * Bonus per safe check: knight +20, bishop +20, rook +40, queen +20.
     * Multiple safe checks get exponential bonuses.
     *
     * PERFORMANCE: This is gated behind a material check (only compute
     * when we have major/minor pieces that could give checks) and
     * caches the bishop/rook attack lookups to avoid redundant slider
     * computations (the main cause of the 300Knps-1Mnps variance).
     */
    {
        /* Gate: skip if we have no major pieces (queen/rook) — without
         * them, safe checks can't lead to mate, so the bonus is wasted.
         * This skips the expensive slider lookups in minor-only endgames. */
        Bitboard our_majors = b->pieces[us][ROOK] | b->pieces[us][QUEEN];
        if (our_majors) {
            Bitboard occ_local = b->occ[2];
            int ek_sq = bb_lsb(b->pieces[them][KING]);

            /* Compute slider attacks from the enemy king ONCE, then
             * reuse for both bishop and queen checks (and rook/queen). */
            Bitboard b_atk_from_king = bishop_attacks((Square)ek_sq, occ_local);
            Bitboard r_atk_from_king = rook_attacks  ((Square)ek_sq, occ_local);

            /* Our pieces that would check the enemy king. */
            Bitboard n_checks = KNIGHT_ATTACKS[ek_sq] & b->pieces[us][KNIGHT];
            Bitboard b_checks = b_atk_from_king & b->pieces[us][BISHOP];
            Bitboard r_checks = r_atk_from_king & b->pieces[us][ROOK];
            Bitboard q_checks = (b_atk_from_king | r_atk_from_king)
                              & b->pieces[us][QUEEN];

            /* Compute enemy pawn attacks ONCE (was per-piece before). */
            Bitboard safe_mask = ~0;
            Bitboard ep = b->pieces[them][PAWN];
            while (ep) {
                int s = bb_pop(&ep);
                safe_mask &= ~PAWN_ATTACKS[them][s];
            }

            /* Count safe checks (piece on a non-pawn-attacked square). */
            int safe_checks = 0;
            if (n_checks & safe_mask) { *mg += 20; safe_checks++; }
            if (b_checks & safe_mask) { *mg += 20; safe_checks++; }
            if (r_checks & safe_mask) { *mg += 40; safe_checks++; }
            if (q_checks & safe_mask) { *mg += 20; safe_checks++; }

            /* Multiple safe checks are exponentially dangerous */
            if (safe_checks >= 2) *mg += 30;
            if (safe_checks >= 3) *mg += 30;
        }
    }

    /*
     * ── 3. Endgame king activity ──
     *
     * Reward the king for being close to the enemy king.  This incentivises
     * the stronger side to centralise and box in the opponent rather than
     * remain passive, and penalises the weaker side's king for hiding.
     *
     * chebyshev ∈ [1, 7] → bonus ∈ [4 cp, 28 cp] (endgame only).
     * PST_KING_EG already captures centrality; this adds the inter-king
     * proximity dimension that a per-piece PST cannot express.
     */
    int enemy_king_sq = bb_lsb(b->pieces[them][KING]);
    int king_dist     = chebyshev(king_sq, enemy_king_sq);
    *eg -= king_dist * KING_EG_DISTANCE_PENALTY;
}

/* ──────────────────────────────────────────────
 *  Tactical threat evaluation
 *
 *  Three terms evaluated for the side `us`:
 *
 *  1. Hanging piece penalty
 *     A piece is "hanging" if it is attacked by the opponent and either:
 *       a) has no friendly defender at all — full penalty (≈ 50 % of piece
 *          value, the rest comes from the search finding the capture), or
 *       b) is attacked by a cheaper enemy piece — partial penalty scaled
 *          by the value difference.
 *     This term fills the gap when pruning prevents the search from
 *     reaching the capturing move: even at the leaf node, the eval will
 *     reflect the material danger, pushing the engine away from leaving
 *     pieces en prise in favour of superficially attractive positional
 *     moves (open files, outposts, etc.).
 *
 *  2. Knight fork bonus
 *     A knight that attacks 2+ enemy pieces worth at least a bishop in the
 *     same move has immediate forking potential.  Rewarding this in the
 *     eval nudges the engine to seek out fork squares even when the fork
 *     itself is a move or two away.
 *
 *  3. Sliding piece battery / skewer bonus
 *     A rook or queen that attacks 2+ valuable enemy pieces along a rank
 *     or file (or a bishop/queen on a diagonal) represents a latent skewer
 *     or battery threat.  A small bonus encourages the engine to establish
 *     such alignments proactively.
 *
 *  All terms are tapered automatically through the mg/eg blend in
 *  evaluate().  sq_attackers() is a local duplicate of the homonymous
 *  function in search.c, kept static here to avoid adding a cross-file
 *  dependency on internal search machinery.
 * ────────────────────────────────────────────── */

static Bitboard sq_attackers(const Board *b, int sq, Bitboard occ) {
    return (PAWN_ATTACKS[WHITE][sq]  & b->pieces[BLACK][PAWN])
         | (PAWN_ATTACKS[BLACK][sq]  & b->pieces[WHITE][PAWN])
         | (KNIGHT_ATTACKS[sq]       & (b->pieces[WHITE][KNIGHT] | b->pieces[BLACK][KNIGHT]))
         | (bishop_attacks((Square)sq, occ)
                           & (  b->pieces[WHITE][BISHOP] | b->pieces[BLACK][BISHOP]
                               | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (rook_attacks((Square)sq, occ)
                         & (  b->pieces[WHITE][ROOK]   | b->pieces[BLACK][ROOK]
                             | b->pieces[WHITE][QUEEN]  | b->pieces[BLACK][QUEEN]))
         | (KING_ATTACKS[sq]         & (b->pieces[WHITE][KING]   | b->pieces[BLACK][KING]));
}

/* Hanging-piece penalty table: roughly 50 % of the piece's material value. */
static const int HANG_PENALTY_MG[6] = {  40, 170, 175, 250, 500, 0 };
static const int HANG_PENALTY_EG[6] = {  50, 165, 160, 270, 460, 0 };

static void eval_threats(const Board *b, Color us, int *mg, int *eg) {
    Color    them = us ^ 1;
    Bitboard occ  = b->occ[2];

    /* ── 1. Hanging / en-prise pieces ── */
    for (int pt = PAWN; pt < KING; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq = bb_pop(&pieces);

            /*
             * Compute attackers ONCE.  The previous code called
             * sq_attackers twice (once for enemy_atk, once for
             * friendly_def), which is the most expensive function
             * in the file — calling it twice per piece was pure
             * waste.
             */
            Bitboard all_atk = sq_attackers(b, sq, occ);
            Bitboard enemy_atk = all_atk & b->occ[them];
            if (!enemy_atk) continue;   /* not attacked at all — safe */

            Bitboard friendly_def = all_atk & b->occ[us];
            /* (the piece on 'sq' itself doesn't appear in sq_attackers output,
               so no need to mask it out.) */

            if (!friendly_def) {
                /* Completely undefended: apply the full hanging penalty. */
                *mg -= HANG_PENALTY_MG[pt];
                *eg -= HANG_PENALTY_EG[pt];
            } else {
                /*
                 * Defended, but attacked by a cheaper piece.  Walk the enemy
                 * attackers from cheapest (PAWN) upward; the first hit is the
                 * minimum-value attacker.  Track BOTH the MG and EG minimums
                 * — the EG material values differ enough from MG that using
                 * MG-only understates the EG danger (e.g. a pawn attacking a
                 * rook is a 546-94=452 cp EG swing, not 480-82=398 cp MG).
                 */
                int min_atk_mg = 30000, min_atk_eg = 30000;
                for (int apt = PAWN; apt <= QUEEN; apt++) {
                    if (b->pieces[them][apt] & enemy_atk) {
                        min_atk_mg = EW.material_mg[apt];
                        min_atk_eg = EW.material_eg[apt];
                        break;
                    }
                }
                int piece_mg = EW.material_mg[pt];
                int piece_eg = EW.material_eg[pt];
                if (min_atk_mg < piece_mg) {
                    /* Partial penalty: proportional to value difference. */
                    *mg -= (piece_mg - min_atk_mg) / 6;
                    *eg -= (piece_eg - min_atk_eg) / 6;
                }
            }
        }
    }

    /* ── 2. Knight fork bonus ── */
    {
        /*
         * Any square a knight can reach that hits 2+ enemy pieces worth at
         * least a minor piece represents a fork threat.  We reward the
         * current presence of knights that already attack multiple targets;
         * the eval gradient encourages the engine to manoeuvre toward such
         * squares one move before the fork fires.
         */
        Bitboard valuable_enemy = b->pieces[them][BISHOP]
                                | b->pieces[them][ROOK]
                                | b->pieces[them][QUEEN]
                                | b->pieces[them][KING];
        Bitboard knights = b->pieces[us][KNIGHT];
        while (knights) {
            int sq   = bb_pop(&knights);
            int hits = bb_popcount(KNIGHT_ATTACKS[sq] & valuable_enemy);
            if (hits >= 2) {
                /* Base bonus + extra for each additional piece hit. */
                *mg += 45 + 20 * (hits - 2);
                *eg += 35 + 15 * (hits - 2);
            }
        }
    }

    /* ── 3. Sliding piece battery / skewer bonus ── */
    {
        /*
         * A rook or queen that sees 2+ valuable enemy pieces along its attack
         * ray has a latent battery or skewer threat even if not immediately
         * executable (the intermediate piece may be capturable or moveable).
         * Same logic for bishops on diagonals through the royal pair.
         */
        Bitboard royal      = b->pieces[them][QUEEN] | b->pieces[them][KING];
        Bitboard valuable_r = royal | b->pieces[them][ROOK];   /* rook targets */
        Bitboard valuable_b = royal;                            /* bishop targets */

        Bitboard rq = b->pieces[us][ROOK] | b->pieces[us][QUEEN];
        while (rq) {
            int sq = bb_pop(&rq);
            if (bb_popcount(rook_attacks((Square)sq, occ) & valuable_r) >= 2) {
                *mg += 20;
                *eg += 15;
            }
        }

        Bitboard bq = b->pieces[us][BISHOP] | b->pieces[us][QUEEN];
        while (bq) {
            int sq = bb_pop(&bq);
            if (bb_popcount(bishop_attacks((Square)sq, occ) & valuable_b) >= 2) {
                *mg += 15;
                *eg += 10;
            }
        }
    }
}

/* ──────────────────────────────────────────────
 *  WeakQueen — penalty when a slider x-rays through our queen
 * ────────────────────────────────────────────── */
static void eval_queen_weak(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard occ    = b->occ[2];
    Bitboard queens = b->pieces[us][QUEEN];
    while (queens) {
        int sq = bb_pop(&queens);
        /* Remove the queen from occupancy to reveal x-ray sliders */
        Bitboard occ_no_q = occ ^ SQUARE_BB[sq];
        Bitboard rook_x   = rook_attacks((Square)sq, occ_no_q)
                          & (b->pieces[them][ROOK] | b->pieces[them][QUEEN]);
        Bitboard bish_x   = bishop_attacks((Square)sq, occ_no_q)
                          & (b->pieces[them][BISHOP] | b->pieces[them][QUEEN]);
        if (rook_x | bish_x) {
            *mg -= 49;
            *eg -= 15;
        }
    }
}

/* ──────────────────────────────────────────────
 *  KingProtector — minor pieces far from own king incur a penalty
 *
 *  Tuning: knights need to stay close to the king (they're slow and
 *  can't help defend from afar); bishops are less tied to the king
 *  (they can defend diagonally from a distance).  We use SF-style
 *  per-piece-type weights:
 *    KNIGHT: 4 cp MG, 3 cp EG per unit of Chebyshev distance.
 *    BISHOP: 2 cp MG, 2 cp EG per unit.
 *  These are about half the previous values, which gave a 105 cp
 *  penalty for a single minor piece on the other side of the board —
 *  enough to swamp most positional terms.
 * ────────────────────────────────────────────── */
static const int KP_MG[2] = { 4, 2 };   /* knight, bishop */
static const int KP_EG[2] = { 3, 2 };

static void eval_king_protector(const Board *b, Color us, int *mg, int *eg) {
    int king_sq = bb_lsb(b->pieces[us][KING]);
    for (int pt = KNIGHT; pt <= BISHOP; pt++) {
        Bitboard pieces = b->pieces[us][pt];
        while (pieces) {
            int sq   = bb_pop(&pieces);
            int dist = chebyshev(sq, king_sq);
            *mg -= KP_MG[pt - KNIGHT] * dist;
            *eg -= KP_EG[pt - KNIGHT] * dist;
        }
    }
}

/* ──────────────────────────────────────────────
 *  MinorBehindPawn — minor piece directly behind a friendly pawn
 * ────────────────────────────────────────────── */
static void eval_minor_behind_pawn(const Board *b, Color us, int *mg, int *eg) {
    Bitboard pawns = b->pieces[us][PAWN];
    Bitboard behind_pawns = (us == WHITE) ? (pawns >> 8) : (pawns << 8);
    Bitboard minors = b->pieces[us][KNIGHT] | b->pieces[us][BISHOP];
    int count = bb_popcount(minors & behind_pawns);
    *mg += 18 * count;
    *eg +=  3 * count;
}

/*
 * ──────────────────────────────────────────────
 *  Bishop pawns ("bad bishop")
 *
 *  A bishop whose own pawns are predominantly on the same color complex as
 *  the bishop itself loses mobility and is effectively a "big pawn".  We
 *  count own pawns on the bishop's color complex and apply a per-pawn
 *  penalty.  When the count exceeds the heavy threshold (3), an additional
 *  flat penalty is added because the bishop is essentially neutralised.
 *
 *  Color complex of a square s: dark squares have (s ^ (s>>3)) & 1 == 0
 *  (sum of file+rank is even).  Same-color squares share this bit.
 *  We test (s ^ color_rep_sq) & 1 == 0 to detect same color.
 *  Color representative: sq 0 (a1) is dark.
 * ────────────────────────────────────────────── */
static void eval_bishop_pawns(const Board *b, Color us, int *mg, int *eg) {
    Bitboard bishops = b->pieces[us][BISHOP];
    Bitboard our_pawns = b->pieces[us][PAWN];

    while (bishops) {
        int sq = bb_pop(&bishops);
        /* Determine the color complex of this bishop.
         * (file + rank) parity: dark squares have even parity (a1=0+0=0). */
        int bishop_parity = ((sq & 7) + (sq >> 3)) & 1;

        /* Count own pawns on the same color complex. */
        int same_color_pawns = 0;
        Bitboard p = our_pawns;
        while (p) {
            int psq = bb_pop(&p);
            int pawn_parity = ((psq & 7) + (psq >> 3)) & 1;
            if (pawn_parity == bishop_parity) same_color_pawns++;
        }

        if (same_color_pawns > 0) {
            *mg -= BISHOP_PAWNS_MG * same_color_pawns;
            *eg -= BISHOP_PAWNS_EG * same_color_pawns;
            if (same_color_pawns > BISHOP_PAWNS_HEAVY_THRESHOLD) {
                *mg -= BISHOP_PAWNS_HEAVY_MG;
            }
        }
    }
}

/*
 * ──────────────────────────────────────────────
 *  Long diagonal bishop
 *
 *  A bishop on one of the two long diagonals (a1-h8 or a8-h1) sees the
 *  central squares d4/e4/d5/e5 and exerts board-wide pressure.  We award
 *  a small bonus for each such bishop that actually has line-of-sight to
 *  the centre (i.e. the diagonal is not blocked by another own piece).
 *
 *  We use a simple heuristic: if the bishop's attacks from its current
 *  square include at least one of the four central squares, we count it
 *  as a long-diagonal bishop when the bishop is itself on one of the two
 *  long diagonals.  This avoids rewarding a bishop on a1 that's walled in
 *  by its own pawns on b2/c3.
 * ────────────────────────────────────────────── */
static const Bitboard LONG_DIAGONAL_A1H8 =
      (1ULL <<  0) | (1ULL <<  9) | (1ULL << 18) | (1ULL << 27)
    | (1ULL << 36) | (1ULL << 45) | (1ULL << 54) | (1ULL << 63);
static const Bitboard LONG_DIAGONAL_A8H1 =
      (1ULL << 56) | (1ULL << 49) | (1ULL << 42) | (1ULL << 35)
    | (1ULL << 28) | (1ULL << 21) | (1ULL << 14) | (1ULL <<  7);

static void eval_long_diagonal_bishop(const Board *b, Color us, int *mg, int *eg) {
    Bitboard bishops = b->pieces[us][BISHOP];
    Bitboard occ = b->occ[2];

    while (bishops) {
        int sq = bb_pop(&bishops);
        Bitboard sqbb = SQUARE_BB[sq];
        bool on_long = (sqbb & (LONG_DIAGONAL_A1H8 | LONG_DIAGONAL_A8H1)) != 0;
        if (!on_long) continue;

        /* Check the bishop actually sees at least one central square. */
        Bitboard atk = bishop_attacks((Square)sq, occ);
        Bitboard centre = SQUARE_BB[27] | SQUARE_BB[28]
                        | SQUARE_BB[35] | SQUARE_BB[36];
        if (atk & centre) {
            *mg += LONG_DIAGONAL_BISHOP_MG;
            *eg += LONG_DIAGONAL_BISHOP_EG;
        }
    }
}

/*
 * ──────────────────────────────────────────────
 *  Queen infiltration
 *
 *  A queen on ranks 5/6/7 from our perspective is actively invading — it
 *  threatens pawns, mate, and pins.  We award a tiered bonus based on the
 *  rank from our side.  Rank 8 (deeply invading or behind enemy lines) gets
 *  the largest bonus.  No bonus for queens still on our half of the board.
 * ────────────────────────────────────────────── */
static void eval_queen_infiltration(const Board *b, Color us, int *mg, int *eg) {
    Bitboard queens = b->pieces[us][QUEEN];
    while (queens) {
        int sq = bb_pop(&queens);
        int rank = sq >> 3;
        /* Rank from our perspective: WHITE ranks 0..7, BLACK inverted. */
        int our_rank = (us == WHITE) ? rank : (7 - rank);
        /* our_rank 0..3 = our half → no bonus.
         * our_rank 4    = rank 5 → small bonus (index 1).
         * our_rank 5    = rank 6 → medium bonus (index 2).
         * our_rank 6..7 = ranks 7/8 → large bonus (index 3). */
        int idx;
        if      (our_rank <= 4) idx = (our_rank == 4) ? 1 : 0;
        else if (our_rank == 5) idx = 2;
        else                    idx = 3;
        *mg += QUEEN_INFILTRATION_MG[idx];
        *eg += QUEEN_INFILTRATION_EG[idx];
    }
}

/*
 * ──────────────────────────────────────────────
 *  Knight on queen / Slider on queen
 *
 *  Award a small bonus for our pieces that attack the enemy queen.  Knights
 *  annoying the queen are tactically dangerous (fork potential); bishops
 *  and rooks attacking the queen along a ray represent latent pin/skewer
 *  threats.  The bonus is small to avoid double-counting with the existing
 *  fork and battery terms — this is a positional bonus, not a tactical one.
 * ────────────────────────────────────────────── */
static void eval_pieces_on_queen(const Board *b, Color us, int *mg, int *eg) {
    Color them = us ^ 1;
    Bitboard enemy_queens = b->pieces[them][QUEEN];
    if (!enemy_queens) return;

    Bitboard occ = b->occ[2];

    /* Knight on queen: knight attacks any enemy-queen square. */
    Bitboard knights = b->pieces[us][KNIGHT];
    while (knights) {
        int sq = bb_pop(&knights);
        if (KNIGHT_ATTACKS[sq] & enemy_queens) {
            *mg += KNIGHT_ON_QUEEN_MG;
            *eg += KNIGHT_ON_QUEEN_EG;
        }
    }

    /* Bishop on queen: bishop x-rays any enemy-queen square. */
    Bitboard bishops = b->pieces[us][BISHOP];
    while (bishops) {
        int sq = bb_pop(&bishops);
        if (bishop_attacks((Square)sq, occ) & enemy_queens) {
            *mg += BISHOP_ON_QUEEN_MG;
            *eg += BISHOP_ON_QUEEN_EG;
        }
    }

    /* Rook on queen: rook x-rays any enemy-queen square (and we're not
     * already crediting this through "rook on queen file" — that term
     * covers the file-based battery, this covers the direct attack). */
    Bitboard rooks = b->pieces[us][ROOK];
    while (rooks) {
        int sq = bb_pop(&rooks);
        if (rook_attacks((Square)sq, occ) & enemy_queens) {
            *mg += ROOK_ON_QUEEN_ATTACK_MG;
            *eg += ROOK_ON_QUEEN_ATTACK_EG;
        }
    }
}

/*
 * ──────────────────────────────────────────────
 *  Space evaluation
 *
 *  Reward squares behind our pawns (up to rank 5 from our perspective) that
 *  we control (i.e. the square is empty or holds our own piece, and is not
 *  attacked by an enemy pawn).  More space = more manoeuvring room.
 *
 *  The count is divided by SPACE_SCALE to keep the bonus in a sensible range
 *  (~5-15 cp typical).  MG-only — space matters far less in the endgame
 *  where the kings walk out and pawns are passed rather than blocked.
 * ────────────────────────────────────────────── */
static void eval_space(const Board *b, Color us, int *mg, int *eg) {
    (void)eg;  /* space is MG-only */
    Color them = us ^ 1;
    Bitboard our_pawns = b->pieces[us][PAWN];
    Bitboard their_pawns = b->pieces[them][PAWN];

    /* Enemy-pawn attack set: squares the enemy pawns control. */
    Bitboard epa;
    if (them == WHITE) {
        epa = ((their_pawns & ~FILE_BB[0]) << 7) | ((their_pawns & ~FILE_BB[7]) << 9);
    } else {
        epa = ((their_pawns & ~FILE_BB[7]) >> 7) | ((their_pawns & ~FILE_BB[0]) >> 9);
    }

    /* Build the "behind-our-pawns" span: for each of our pawns, every square
     * on the same file with rank strictly less than the pawn's rank (from
     * our perspective).  For WHITE this means ranks 0..pawn_rank-1; for
     * BLACK the symmetric.  We only count up to our rank 5 (rank index 4)
     * — deep behind isn't more valuable than shallow behind, and clamping
     * to rank 5 keeps the bonus tied to the actual pawn front. */
    Bitboard behind = 0;
    Bitboard p = our_pawns;
    while (p) {
        int sq = bb_pop(&p);
        int file = sq & 7;
        int rank = sq >> 3;
        if (us == WHITE) {
            for (int r = 0; r < rank && r <= 4; r++)
                behind |= SQUARE_BB[r * 8 + file];
        } else {
            for (int r = 7; r > rank && r >= 3; r--)
                behind |= SQUARE_BB[r * 8 + file];
        }
    }

    /* A square counts toward our space if:
     *   - it's in the `behind` span
     *   - it's not attacked by an enemy pawn
     *   - it's not occupied by the enemy king (kings don't take space)
     * Own pieces and empty squares both count — having pieces in your space
     * is the whole point of having space. */
    Bitboard blocked = epa | b->pieces[them][KING];
    int count = bb_popcount(behind & ~blocked);
    *mg += (count * SPACE_MG_PER_SQUARE) / SPACE_SCALE;
}

/* ──────────────────────────────────────────────
 *  Material Imbalance (SF 11 quadratic polynomial)
 *
 *  Uses per-piece-pair coefficients to reward or penalise having certain
 *  piece combinations: bishop pair, rook vs. minors, etc.  The quadratic
 *  term captures interactions that a simple count cannot (e.g. the bishop
 *  pair is worth more when there are many pawns).
 *
 *  Index 0 = bishop-pair proxy; 1=Pawn 2=Knight 3=Bishop 4=Rook 5=Queen.
 * ────────────────────────────────────────────── */
static const int QuadOurs[6][6] = {
    {1438,   0,   0,   0,    0,   0},   /* bishop pair */
    {  40,  38,   0,   0,    0,   0},   /* pawn        */
    {  32, 255, -62,   0,    0,   0},   /* knight      */
    {   0, 104,   4,   0,    0,   0},   /* bishop      */
    { -26,  -2,  47, 105, -208,   0},   /* rook        */
    {-189,  24, 117, 133, -134,  -6},   /* queen       */
};
static const int QuadTheirs[6][6] = {
    {  0,   0,   0,   0,   0,   0},
    { 36,   0,   0,   0,   0,   0},
    {  9,  63,   0,   0,   0,   0},
    { 59,  65,  42,   0,   0,   0},
    { 46,  39,  24, -24,   0,   0},
    { 97, 100, -42, 137, 268,   0},
};

static int material_imbalance(const Board *b) {
    int pc[2][6] = {{0}};
    for (int c = 0; c < 2; c++) {
        pc[c][0] = (bb_popcount(b->pieces[c][BISHOP]) > 1) ? 1 : 0;
        for (int pt = PAWN; pt <= QUEEN; pt++)
            pc[c][pt + 1] = bb_popcount(b->pieces[c][pt]);
    }

    int bonus = 0;
    for (int us = 0; us < 2; us++) {
        int them = us ^ 1;
        int sign = (us == WHITE) ? 1 : -1;
        int b_us = 0;
        for (int pt1 = 0; pt1 <= 5; pt1++) {
            if (!pc[us][pt1]) continue;
            int v = 0;
            for (int pt2 = 0; pt2 <= pt1; pt2++)
                v += QuadOurs[pt1][pt2]   * pc[us][pt2]
                   + QuadTheirs[pt1][pt2] * pc[them][pt2];
            b_us += pc[us][pt1] * v;
        }
        bonus += sign * b_us;
    }
    return bonus / 16;
}

/* ──────────────────────────────────────────────
 *  Initiative / complexity correction (SF 11)
 *
 *  Reduces a winning advantage when the winning side cannot realistically
 *  convert it: no passed pawns, no outflanking, pawns on only one flank.
 *  Applied just before the side-to-move flip.
 * ────────────────────────────────────────────── */
static int initiative(const Board *b, int mg, int eg) {
    int wk = bb_lsb(b->pieces[WHITE][KING]);
    int bk = bb_lsb(b->pieces[BLACK][KING]);

    /*
     * Outflanking = |Δfile| - |Δrank|.
     *
     * The original code used a signed Δfile - Δrank, which gave
     * nonsense values for kings far apart on one axis.  The correct
     * definition is the absolute version: a high outflanking means
     * the kings are far apart on one axis and close on the other
     * (the winning side has the opposition).
     */
    int df = (wk & 7) - (bk & 7); if (df < 0) df = -df;
    int dr = (wk >> 3) - (bk >> 3); if (dr < 0) dr = -dr;
    int outflanking = df - dr;

    bool infiltration = (wk >> 3) > 3 || (bk >> 3) < 4;

    Bitboard all_pawns = b->pieces[WHITE][PAWN] | b->pieces[BLACK][PAWN];
    bool both_flanks = (all_pawns & 0x0F0F0F0F0F0F0F0FULL) &&
                       (all_pawns & 0xF0F0F0F0F0F0F0F0ULL);

    /* Count passed pawns for both sides */
    int passed_count = 0;
    {
        Bitboard wp = b->pieces[WHITE][PAWN];
        Bitboard bp = b->pieces[BLACK][PAWN];
        Bitboard tmp = wp;
        while (tmp) {
            int sq = bb_pop(&tmp);
            int f = sq & 7, r = sq >> 3;
            Bitboard adj = 0;
            if (f > 0) adj |= FILE_BB[f-1];
            if (f < 7) adj |= FILE_BB[f+1];
            Bitboard ahead = 0;
            for (int rr = r+1; rr < 8; rr++) ahead |= RANK_BB[rr];
            if (!(bp & (FILE_BB[f] | adj) & ahead)) passed_count++;
        }
        tmp = bp;
        while (tmp) {
            int sq = bb_pop(&tmp);
            int f = sq & 7, r = sq >> 3;
            Bitboard adj = 0;
            if (f > 0) adj |= FILE_BB[f-1];
            if (f < 7) adj |= FILE_BB[f+1];
            Bitboard ahead = 0;
            for (int rr = 0; rr < r; rr++) ahead |= RANK_BB[rr];
            if (!(wp & (FILE_BB[f] | adj) & ahead)) passed_count++;
        }
    }

    int pawn_count = bb_popcount(all_pawns);
    bool no_npm = (b->occ[WHITE] & ~b->pieces[WHITE][PAWN] & ~b->pieces[WHITE][KING]) == 0
               && (b->occ[BLACK] & ~b->pieces[BLACK][PAWN] & ~b->pieces[BLACK][KING]) == 0;

    bool almost_unwinnable = !passed_count && outflanking < 0 && !both_flanks;

    int complexity =
          9 * passed_count
        + 11 * pawn_count
        +  9 * outflanking
        + 12 * (int)infiltration
        + 21 * (int)both_flanks
        + 51 * (int)no_npm
        - 43 * (int)almost_unwinnable
        - 100;

    int sign_mg = (mg > 0) - (mg < 0);
    int sign_eg = (eg > 0) - (eg < 0);

    int u_raw = complexity + 50;
    int u = sign_mg * (u_raw < 0 ? u_raw : 0);
    if (u < -abs(mg)) u = -abs(mg);
    if (u >  abs(mg)) u =  abs(mg);   /* don't let initiative flip mg's sign */

    int v = sign_eg * (complexity > -abs(eg) ? complexity : -abs(eg));

    int phase = game_phase(b);
    return taper(u, v, phase);
}

/* ──────────────────────────────────────────────
 *  Lazy evaluation guard (~10-15 % NPS, free Elo)
 *
 *  Before running the full evaluation, compute a cheap material+PST proxy.
 *  If the proxy is far outside the window, return it immediately — the
 *  full eval cannot change the result.  LAZY_THRESHOLD is tuned to be
 *  safe — the full eval rarely swings more than ~300cp from the proxy,
 *  so 600cp leaves a comfortable margin.
 *
 *  The proxy is computed ONCE and reused: if the lazy guard doesn't
 *  fire, the same material+PST values are added to the running mg/eg
 *  totals (no duplicate computation).
 * ────────────────────────────────────────────── */
#define LAZY_THRESHOLD 1000

/* Compute material+PST for both sides, returning the tapered score.
 * Also populates *mg_out and *eg_out so the caller can reuse the
 * per-phase values without re-computing them. */
static int lazy_score(const Board *b, int phase, int *mg_out, int *eg_out) {
    int mg = 0, eg = 0;
    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        for (int pt = 0; pt < 6; pt++) {
            Bitboard bb = b->pieces[c][pt];
            while (bb) {
                int sq  = bb_pop(&bb);
                int psq = pst_sq((Color)c, (Square)sq);
                mg += sign * (EW.material_mg[pt] + PST_MG[pt][psq]);
                eg += sign * (EW.material_eg[pt] + PST_EG[pt][psq]);
            }
        }
    }
    *mg_out = mg;
    *eg_out = eg;
    return taper(mg, eg, phase);
}

/* ──────────────────────────────────────────────
 *  Main evaluation
 * ────────────────────────────────────────────── */
int evaluate(const Board *b) {
    int phase = game_phase(b);

    /*
     * ── Lazy evaluation guard ─────────────────────────────────────────
     * Compute a cheap material+PST proxy.  If it is far outside any
     * plausible search window the full evaluation cannot change the result,
     * so we return early.  LAZY_THRESHOLD (600 cp) is tuned to be safe:
     * the full eval rarely swings more than ~300cp from the proxy.
     *
     * The proxy computation also populates lazy_mg/lazy_eg so the main
     * eval can reuse them (no duplicate material+PST iteration).
     */
    int lazy_mg, lazy_eg;
    int proxy = lazy_score(b, phase, &lazy_mg, &lazy_eg);
    if (abs(proxy) > EW.lazy_threshold) {
        /* Still apply the side-to-move flip for consistency */
        int lazy_ret = (b->side == WHITE) ? proxy : -proxy;

#ifdef EVAL_DEBUG
        /* Record the lazy position so we can spot if it skews the lists */
        EvalBreakdown __dbg_lazy = {0};
        __dbg_lazy.phase       = phase;
        __dbg_lazy.side        = (int)b->side;
        __dbg_lazy.lazy        = true;
        __dbg_lazy.final_score = lazy_ret;
        eval_debug_record(b, &__dbg_lazy);
#endif
        return lazy_ret;
    }

    int mg = 0, eg = 0;

    /* ── Debug scaffolding ──────────────────────────────────────────────
     * __dbg accumulates per-stage contributions.  __snap_mg / __snap_eg
     * are "before" snapshots used to compute each stage's delta.
     * All of this compiles away completely without -DEVAL_DEBUG.
     */
#ifdef EVAL_DEBUG
    EvalBreakdown __dbg = {0};
    __dbg.phase = phase;
    __dbg.side  = (int)b->side;
    int __snap_mg, __snap_eg;
#endif

    /* ── Material imbalance (quadratic polynomial, SF 11) ─────────────
     * Accounts for piece-combination interactions: bishop pair value,
     * rook vs. two minors, etc.  Applied once per position, not per side.
     */
#ifdef EVAL_DEBUG
    __snap_mg = mg;
#endif
    mg += material_imbalance(b);
#ifdef EVAL_DEBUG
    __dbg.imbalance = mg - __snap_mg;
#endif

    /* Reuse the lazy_score's material+PST computation — no duplicate work. */
    mg += lazy_mg;
    eg += lazy_eg;

    /* Per-side breakdown for debug.  We need to split mg/eg by color
     * for the debug struct, so re-derive here.  In non-debug builds
     * this entire block compiles away. */
#ifdef EVAL_DEBUG
    {
        int dbg_mg_per_side[2] = {0, 0};
        int dbg_eg_per_side[2] = {0, 0};
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 6; pt++) {
                Bitboard bb = b->pieces[c][pt];
                while (bb) {
                    int sq  = bb_pop(&bb);
                    int psq = pst_sq((Color)c, (Square)sq);
                    dbg_mg_per_side[c] += EW.material_mg[pt] + PST_MG[pt][psq];
                    dbg_eg_per_side[c] += EW.material_eg[pt] + PST_EG[pt][psq];
                }
            }
        }
        __dbg.material_mg[WHITE] = dbg_mg_per_side[WHITE];
        __dbg.material_mg[BLACK] = -dbg_mg_per_side[BLACK];
        __dbg.material_eg[WHITE] = dbg_eg_per_side[WHITE];
        __dbg.material_eg[BLACK] = -dbg_eg_per_side[BLACK];
    }
#endif

    for (int c = 0; c < 2; c++) {
        int sign = (c == WHITE) ? 1 : -1;
        int c_mg = 0, c_eg = 0;  /* per-side accumulators for the remaining terms */

#ifdef EVAL_DEBUG
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── Pawn structure (doubled, isolated, backward, passed) ── */
        eval_pawns(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.pawns_mg[c] = c_mg - __snap_mg;
        __dbg.pawns_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── Piece mobility — non-linear per-count SF 11 tables ── */
        eval_mobility(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.mobility_mg[c] = c_mg - __snap_mg;
        __dbg.mobility_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── Rook bonuses (open file, semi-open, 7th rank, TrappedRook) ── */
        eval_rooks(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.rooks_mg[c] = c_mg - __snap_mg;
        __dbg.rooks_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── Outpost bonuses for knights and bishops ── */
        eval_outposts(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.outposts_mg[c] = c_mg - __snap_mg;
        __dbg.outposts_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /*
         * ── Bishop pair ──
         *
         * The MG bishop-pair value is already encoded in
         * material_imbalance() via QuadOurs[0][0] = 1438 (≈ 90 cp MG
         * after /16).  Adding BISHOP_PAIR_MG on top would double-count
         * the MG term.  We apply only the EG bonus here, which the
         * imbalance table does not cover (it's MG-only by design).
         */
        if (bb_popcount(b->pieces[c][BISHOP]) >= 2) {
            c_eg += BISHOP_PAIR_EG;
        }
#ifdef EVAL_DEBUG
        __dbg.bishop_pair_mg[c] = c_mg - __snap_mg;
        __dbg.bishop_pair_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── WeakQueen: penalty when enemy sliders x-ray through our queen ── */
        eval_queen_weak(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.weak_queen_mg[c] = c_mg - __snap_mg;
        __dbg.weak_queen_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── KingProtector: minor pieces far from own king incur a penalty ── */
        eval_king_protector(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.king_protector_mg[c] = c_mg - __snap_mg;
        __dbg.king_protector_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── MinorBehindPawn: minor pieces sheltered behind friendly pawns ── */
        eval_minor_behind_pawn(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.minor_behind_mg[c] = c_mg - __snap_mg;
        __dbg.minor_behind_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── BishopPawns ("bad bishop"): penalty for own pawns on bishop's colour ── */
        eval_bishop_pawns(b, (Color)c, &c_mg, &c_eg);

        /* ── LongDiagonalBishop: bonus for bishop on a1-h8 / a8-h1 long diagonal ── */
        eval_long_diagonal_bishop(b, (Color)c, &c_mg, &c_eg);

        /* ── QueenInfiltration: bonus for queen deep in enemy territory ── */
        eval_queen_infiltration(b, (Color)c, &c_mg, &c_eg);

        /* ── PiecesOnQueen: knight/slider x-raying the enemy queen ── */
        eval_pieces_on_queen(b, (Color)c, &c_mg, &c_eg);

        /* ── Space: reward for squares behind our pawns we control ── */
        eval_space(b, (Color)c, &c_mg, &c_eg);

        /*
         * ── King safety: shield + open files + distance-weighted enemy
         *    attacks (MG) + king-proximity activity bonus (EG). ──
         */
        eval_king_safety(b, (Color)c, &c_mg, &c_eg);
#ifdef EVAL_DEBUG
        __dbg.king_safety_mg[c] = c_mg - __snap_mg;
        __dbg.king_safety_eg[c] = c_eg - __snap_eg;
        __snap_mg = c_mg;  __snap_eg = c_eg;
#endif

        /* ── Tactical threats: hanging pieces, fork potential, skewers ──
         *
         * Gate: skip eval_threats when there are no minor/major pieces
         * on the board (pawn-only endgames have no "hanging pieces" in
         * the traditional sense — pawn captures are handled by the
         * search).  This saves the expensive sq_attackers() calls in
         * KP endgames where the function would return 0 anyway. */
        if (b->occ[2] & ~(b->pieces[WHITE][PAWN] | b->pieces[BLACK][PAWN]
                         | b->pieces[WHITE][KING] | b->pieces[BLACK][KING])) {
            eval_threats(b, (Color)c, &c_mg, &c_eg);
        }
#ifdef EVAL_DEBUG
        __dbg.threats_mg[c] = c_mg - __snap_mg;
        __dbg.threats_eg[c] = c_eg - __snap_eg;
#endif

        mg += sign * c_mg;
        eg += sign * c_eg;
    }

    /* Taper MG/EG blend */
    int score = taper(mg, eg, phase);

    /* Tempo bonus */
    score += taper(EW.tempo_mg, EW.tempo_eg, phase);

    /*
     * ── Opposite-colored-bishops drawishness ──────────────────────
     *
     * Endgames with bishops of opposite colors are notoriously drawish:
     * the side with the advantage cannot break through because the
     * defender's bishop controls all the squares of the attacker's
     * bishop's color.  When the position is materially simple enough
     * that the bishops are the only non-pawn non-king pieces, we shrink
     * the EG score by 25% toward zero.
     *
     * The OCB detection requires:
     *   - exactly one bishop per side
     *   - the two bishops are on opposite color complexes
     *   - no knights / rooks / queens remain
     *   - we're in the endgame phase (low non-pawn material)
     *
     * We apply this AFTER the taper so the reduction only fires when
     * the position is sufficiently endgame-ish.  The taper already
     * weighted the EG component heavily.
     */
    {
        int wb = bb_popcount(b->pieces[WHITE][BISHOP]);
        int bb_n = bb_popcount(b->pieces[BLACK][BISHOP]);
        if (wb == 1 && bb_n == 1) {
            /* Color complex check: a1 (sq 0) is dark; same-color squares
             * have (sq ^ sq2) & 1 == 0.  Bishops on opposite colors
             * have (sq_w ^ sq_b) & 1 == 1. */
            int sq_w = bb_lsb(b->pieces[WHITE][BISHOP]);
            int sq_b = bb_lsb(b->pieces[BLACK][BISHOP]);
            bool ocb = ((sq_w ^ sq_b) & 1) != 0;

            if (ocb) {
                /* Only flag as OCB endgame when no other non-pawn
                 * non-king pieces remain (otherwise it's a middlegame
                 * with OCB, which is a different beast). */
                Bitboard non_bishop_npk = b->occ[2]
                    & ~b->pieces[WHITE][PAWN]   & ~b->pieces[BLACK][PAWN]
                    & ~b->pieces[WHITE][KING]   & ~b->pieces[BLACK][KING]
                    & ~b->pieces[WHITE][BISHOP] & ~b->pieces[BLACK][BISHOP];
                if (non_bishop_npk == 0) {
                    /* Shrink the absolute score by 25%.  We do this
                     * on the *tapered* score so the effect scales
                     * naturally with how endgame-ish the position is. */
                    int sgn = (score > 0) - (score < 0);
                    score -= sgn * (abs(score) / 4);
                }
            }
        }
    }

    /*
     * NEW: ── Wrong rook pawn + wrong-color bishop drawishness ─────────
     *
     * KRP vs KB is a classic draw when:
     *   - The defender has only a bishop (no pawns, no other pieces)
     *   - The attacker's only pawn is a rook pawn (a- or h-file)
     *   - The defender's bishop is on the "wrong" color complex (i.e.,
     *     NOT the color of the promotion square)
     *   - The defender's king can reach the corner (the promotion
     *     square's color complex is the bishop's, so the king can
     *     blockade on the bishop's color and never be driven away)
     *
     * The promotion square of a rook pawn is always the same color
     * as the pawn's starting square's corner: a8 (sq 56) is light,
     * h8 (sq 63) is dark.  If the defender's bishop is on the opposite
     * color from the promotion square, the pawn cannot be supported
     * by the bishop to promote, and the king can always blockade.
     *
     * We detect this and reduce the winning side's score by 75%.
     */
    {
        /* Only applies when each side has exactly one bishop and the
         * only other pieces are kings and pawns. */
        int wb = bb_popcount(b->pieces[WHITE][BISHOP]);
        int bb_n = bb_popcount(b->pieces[BLACK][BISHOP]);

        /* Check both directions: White attacking with rook pawn + bishop,
         * Black defending with bishop; and vice versa. */
        for (int attacker = WHITE; attacker <= BLACK; attacker++) {
            int defender = attacker ^ 1;
            int atk_bishops = (attacker == WHITE) ? wb : bb_n;
            int def_bishops = (defender == WHITE) ? wb : bb_n;

            if (atk_bishops != 1 || def_bishops != 1) continue;

            /* No non-pawn non-king non-bishop pieces allowed */
            Bitboard other_pieces = b->occ[2]
                & ~b->pieces[WHITE][PAWN]   & ~b->pieces[BLACK][PAWN]
                & ~b->pieces[WHITE][KING]   & ~b->pieces[BLACK][KING]
                & ~b->pieces[WHITE][BISHOP] & ~b->pieces[BLACK][BISHOP];
            if (other_pieces != 0) continue;

            /* Attacker must have exactly one pawn, and it must be a
             * rook pawn (a- or h-file). */
            Bitboard atk_pawns = b->pieces[attacker][PAWN];
            if (bb_popcount(atk_pawns) != 1) continue;
            int pawn_sq = bb_lsb(atk_pawns);
            int pawn_file = pawn_sq & 7;
            if (pawn_file != 0 && pawn_file != 7) continue;

            /* Promotion square color: a8 (sq 56) is light (sq & 1 == 0),
             * h8 (sq 63) is dark (sq & 1 == 1).
             * For white attacker: promo = a8 or h8.
             * For black attacker: promo = a1 or h1. */
            int promo_sq = (attacker == WHITE)
                         ? (pawn_file == 0 ? 56 : 63)   /* a8 or h8 */
                         : (pawn_file == 0 ? 0  : 7);    /* a1 or h1 */
            int promo_color = promo_sq & 1;  /* 0 = light, 1 = dark */

            /* Defender's bishop color */
            int def_bishop_sq = bb_lsb(b->pieces[defender][BISHOP]);
            int def_bishop_color = def_bishop_sq & 1;

            /* "Wrong color" = bishop is on opposite color from promo sq */
            if (def_bishop_color != promo_color) {
                /* This is the drawish case — reduce the attacker's
                 * advantage by 75%.  Only apply if the attacker is
                 * winning (score > 0 for white attacker, < 0 for black). */
                int sgn = (attacker == WHITE) ? 1 : -1;
                if (sgn * score > 0) {
                    score -= sgn * (abs(score) * 3 / 4);
                }
            }
        }
    }

    /*
     * ── Initiative / complexity correction (SF 11) ────────────────────
     * Reduces a winning advantage when the winning side cannot realistically
     * convert it: no passed pawns, no outflanking, pawns on only one flank.
     * Applied before the side-to-move flip.
     */
    score += initiative(b, mg, eg);

#ifdef WORST_ENGINE
    score = -score;
#endif

    /* Return from side-to-move perspective */
    int final_score = (b->side == WHITE) ? score : -score;

#ifdef EVAL_DEBUG
    /*
     * taper() and initiative() are pure functions — calling them again
     * for debug capture is safe and identical to the calls above.
     */
    __dbg.tempo       = taper(EW.tempo_mg, EW.tempo_eg, phase);
    __dbg.initiative  = initiative(b, mg, eg);
    __dbg.final_score = final_score;
    eval_debug_record(b, &__dbg);
#endif

    return final_score;
}
