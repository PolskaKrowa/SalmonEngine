/*
 * book.c — compact opening book
 *
 * Design
 * ------
 * Opening lines are stored as arrays of UCI move strings (e.g. "e2e4").
 * At startup, book_init() replays every line from the initial position
 * using the engine's own make_move / Zobrist hash machinery.  For every
 * position reached in a line, the next move in that line is stored in a
 * fixed-size direct-mapped hash table keyed by the Zobrist hash.
 *
 * This approach is independent of any particular Zobrist implementation:
 * hashes are computed on-the-fly by the engine itself rather than
 * hard-coded, so the book is always consistent with the board code.
 *
 * To find a move, book_probe() looks up the current board's Zobrist hash
 * and then verifies the stored move against the legal-move list, so a hash
 * collision can never produce an illegal move.
 *
 * Coverage
 * --------
 * ~30 lines covering e4, d4, c4, and Nf3 openings to a depth of 6-12
 * half-moves.  Both sides' responses are stored (every position inside a
 * line gets an entry), so the book works regardless of which colour the
 * engine plays.
 *
 * Extending the book
 * ------------------
 * Add entries to BOOK_LINES[] below.  Each string is a space-separated
 * sequence of UCI moves starting from the initial position.  Longer lines
 * take precedence for positions near the end because they are stored last
 * and the table uses first-write-wins, so keep the most important lines at
 * the top.
 */

#include "book.h"
#include "movegen.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* ── Hash table ─────────────────────────────────────────────────────── */
#define BOOK_SIZE 1024   /* must be a power of 2; easily holds 300+ entries */

typedef struct {
    uint64_t hash;
    Move     move;
    bool     valid;
} BookEntry;

static BookEntry book_table[BOOK_SIZE];
static bool      book_active = false;

/* ── Engine functions we need ───────────────────────────────────────── */
extern void board_init(void);          /* reset to starting position   */
extern void move_to_str(Move m, char *out);/* convert move to UCI string   */

/* ── Internal helpers ───────────────────────────────────────────────── */

/*
 * parse_uci_move — find the legal move that matches a UCI string.
 *
 * Generates all pseudo-legal moves, stringifies each one with
 * move_to_str(), and returns the first that matches `uci`.
 * Returns NULL_MOVE if no match is found (bad line or illegal position).
 */
static Move parse_uci_move(Board *b, const char *uci) {
    MoveList ml;
    gen_moves(b, &ml);
    char buf[8];
    for (int i = 0; i < ml.count; i++) {
        if (!is_legal(b, ml.moves[i])) continue;
        move_to_str(ml.moves[i], buf);
        if (strncmp(buf, uci, 5) == 0)
            return ml.moves[i];
    }
    return NULL_MOVE;
}

/*
 * book_store — insert one (hash, move) pair.
 * First-write-wins: the first move stored for a position is kept so that
 * lines listed first in BOOK_LINES[] take priority.
 */
static void book_store(uint64_t hash, Move m) {
    unsigned idx = (unsigned)(hash & (BOOK_SIZE - 1));
    if (!book_table[idx].valid) {
        book_table[idx].hash  = hash;
        book_table[idx].move  = m;
        book_table[idx].valid = true;
    }
}

/* ── Opening lines ──────────────────────────────────────────────────── */
/*
 * Each string is a sequence of UCI half-moves from the starting position.
 * Every position in the sequence (including the start) gets a book entry
 * for the next move, covering both White and Black automatically.
 *
 * Ordering: put the most important/principled lines first so first-write-
 * wins keeps the best move in case of hash-table slot conflicts.
 */
static const char * const BOOK_LINES[] = {

    /* ── Open games: 1.e4 ── */

    /* Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 */
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4",

    /* Italian Four Knights: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Nc3 */
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 b1c3 f8c5 d2d3",

    /* Ruy Lopez (Berlin): 1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6 4.O-O */
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 g8e4 d2d4",

    /* Ruy Lopez (Morphy): 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O */
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1",

    /* King's Gambit: 1.e4 e5 2.f4 exf4 3.Nf3 d5 */
    "e2e4 e7e5 f2f4 e5f4 g1f3 d7d5",

    /* Sicilian Najdorf: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Bg5 */
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5",

    /* Sicilian Dragon: 1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 g6 5.Nc3 Bg7 */
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 b1c3 f8g7",

    /* Sicilian Scheveningen: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e6 */
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e6",

    /* Closed Sicilian: 1.e4 c5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7 5.d3 */
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 d2d3",

    /* French Classical: 1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 */
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5",

    /* French Tarrasch: 1.e4 e6 2.d4 d5 3.Nd2 Nf6 4.e5 Nfd7 5.Bd3 */
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3",

    /* French Advance: 1.e4 e6 2.d4 d5 3.e5 c5 4.c3 Nc6 5.Nf3 */
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3",

    /* Caro-Kann Classical: 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 */
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5",

    /* Caro-Kann Advance: 1.e4 c6 2.d4 d5 3.e5 Bf5 4.Nf3 e6 */
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6",

    /* Pirc: 1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.f4 Bg7 */
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7",

    /* Scandinavian: 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.d4 Nf6 */
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6",

    /* ── Semi-open: 1.e4 (Black options when White plays e4) ── */

    /* Alekhine: 1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3 */
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3",

    /* ── Closed games: 1.d4 ── */

    /* Queen's Gambit Declined Orthodox: 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 */
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3",

    /* Queen's Gambit Accepted: 1.d4 d5 2.c4 dxc4 3.e4 e5 4.Nf3 */
    "d2d4 d7d5 c2c4 d5c4 e2e4 e7e5 g1f3",

    /* Slav Defence: 1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.e3 e6 */
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 e2e3 e7e6",

    /* London System: 1.d4 d5 2.Nf3 Nf6 3.Bf4 e6 4.e3 */
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3",

    /* King's Indian Classical: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 */
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2",

    /* King's Indian Sämisch: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f3 */
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3",

    /* Nimzo-Indian: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O 5.Bd3 */
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3",

    /* Queen's Indian: 1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.g3 Bb7 5.Bg2 Be7 6.O-O */
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7 e1g1",

    /* Grünfeld: 1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5 5.e4 Nxc3 6.bxc3 Bg7 */
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7",

    /* Catalan: 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 */
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3",

    /* ── Flank openings ── */

    /* English vs KP: 1.c4 e5 2.Nc3 Nf6 3.g3 d5 4.cxd5 Nxd5 5.Bg2 */
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5 f1g2",

    /* English Symmetrical: 1.c4 c5 2.Nc3 Nf6 3.g3 g6 4.Bg2 Bg7 5.Nf3 */
    "c2c4 c7c5 b1c3 g8f6 g2g3 g7g6 f1g2 f8g7 g1f3",

    /* Réti vs d5: 1.Nf3 d5 2.g3 Nf6 3.Bg2 e6 4.O-O Be7 5.d3 */
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d3",

    /* Réti KID setup: 1.Nf3 Nf6 2.c4 g6 3.Nc3 Bg7 4.g3 O-O 5.Bg2 d6 6.O-O */
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 g2g3 e8g8 f1g2 d7d6 e1g1",
};

#define N_BOOK_LINES ((int)(sizeof(BOOK_LINES) / sizeof(BOOK_LINES[0])))

/* ── Public API ─────────────────────────────────────────────────────── */

void book_init(void) {
    memset(book_table, 0, sizeof(book_table));

    Board b;

    for (int li = 0; li < N_BOOK_LINES; li++) {
        board_start_pos(&b);

        const char *p = BOOK_LINES[li];

        while (true) {
            /* Skip whitespace */
            while (*p == ' ') p++;
            if (*p == '\0') break;

            /* Read the next space-delimited token */
            char tok[8] = {0};
            int  tlen   = 0;
            while (*p && *p != ' ' && tlen < 7)
                tok[tlen++] = *p++;

            /* Convert to a Move */
            Move m = parse_uci_move(&b, tok);
            if (m == NULL_MOVE) break;   /* malformed line — abort this line */

            /* Record: from the current position, play 'm'. */
            book_store(b.hash, m);

            /* Advance the board */
            make_move(&b, m);
        }
    }

    book_active = true;
}

Move book_probe(Board *b) {
    if (!book_active) return NULL_MOVE;

    unsigned  idx = (unsigned)(b->hash & (BOOK_SIZE - 1));
    BookEntry *e  = &book_table[idx];

    if (!e->valid || e->hash != b->hash)
        return NULL_MOVE;

    /*
     * Validate legality to guard against hash collisions.
     * Generate all moves and verify the stored move is among the legal ones.
     */
    MoveList ml;
    gen_moves(b, &ml);
    for (int i = 0; i < ml.count; i++) {
        if (ml.moves[i] == e->move && is_legal(b, ml.moves[i]))
            return e->move;
    }

    return NULL_MOVE;   /* collision — don't play an illegal move */
}