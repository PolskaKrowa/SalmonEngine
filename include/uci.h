#pragma once
#include "board.h"
#include "search.h"

/* Engine identity */
#define ENGINE_NAME    "SalmonEngine"
#define ENGINE_AUTHOR  "Steve"
#define ENGINE_VERSION "1.0"

/*
 * Run the UCI command loop.
 * Reads commands from stdin and writes responses to stdout.
 * Blocks until "quit" is received.
 */
void uci_loop(Board *b);

/* Helper: convert a Move to algebraic UCI string (e.g. "e2e4", "e7e8q") */
void move_to_str(Move m, char *out);

/* Helper: parse a UCI algebraic string into a Move on the given board */
Move str_to_move(Board *b, const char *str);
