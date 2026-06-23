/*
 * book.h — opening book interface
 *
 * call book_init() once at startup (already done by search_init).
 * call book_probe(b) before every search; returns NULL_MOVE when out of book.
 */
#pragma once

#include "movegen.h"   /* Board, Move, gen_moves, is_legal */
#include <stdbool.h>

/* Build the hash table from the built-in opening lines.
 * Called automatically by search_init(). */
void book_init(void);

/* Return the book move for the current position, or NULL_MOVE.
 * The returned move is guaranteed legal (hash collisions are filtered).
 * Returns NULL_MOVE when the book is disabled via book_set_enabled(false). */
Move book_probe(Board *b);

/* Enable/disable the opening book at runtime.  Useful for benchmarking
 * (cutechess games typically want the book ON, NPS benchmarks want it
 * OFF so the engine actually searches every position). */
void book_set_enabled(bool enabled);
bool book_is_enabled(void);