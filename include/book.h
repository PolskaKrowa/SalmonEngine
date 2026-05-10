/*
 * book.h — opening book interface
 *
 * call book_init() once at startup (already done by search_init).
 * call book_probe(b) before every search; returns NULL_MOVE when out of book.
 */
#pragma once

#include "movegen.h"   /* Board, Move, gen_moves, is_legal */

/* Build the hash table from the built-in opening lines.
 * Called automatically by search_init(). */
void book_init(void);

/* Return the book move for the current position, or NULL_MOVE.
 * The returned move is guaranteed legal (hash collisions are filtered). */
Move book_probe(Board *b);