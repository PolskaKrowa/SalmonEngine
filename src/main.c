/*
 * SalmonEngine - (Codename: OpenStessine)
 * 
 * An Open Source attempt at making a good chess engine without using NNUE
 * -----------------------------------------------------------------------
 *
 * main.c — Entry point
 *
 * Initialises all subsystems then hands control to the UCI loop.
 */

#include "bitboard.h"
#include "board.h"
#include "search.h"
#include "uci.h"
#include <stdio.h>

int main(void) {
    /* Turn off I/O buffering — UCI requires line-by-line communication */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin,  NULL, _IONBF, 0);

    /* One-time initialisation */
    bitboard_init();
    board_init();
    search_init();

    /* Enter UCI command loop */
    Board b;
    uci_loop(&b);

    return 0;
}
