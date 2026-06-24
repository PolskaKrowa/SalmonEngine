/* Test move_gives_check against the make_move + in_check reference. */
#include <stdio.h>
#include <string.h>
#include "bitboard.h"
#include "board.h"
#include "movegen.h"

static int test_count = 0;
static int test_fail  = 0;

static void test_position(const char *fen) {
    Board b;
    board_from_fen(&b, fen);

    MoveList ml;
    gen_moves(&b, &ml);

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        test_count++;

        /* Reference: make move, check if OPPONENT (now side-to-move)
         * is in check, unmake.  After make_move, b->side is the
         * opponent; their king is at b->pieces[b->side][KING].
         * We ask: does the mover (b->side ^ 1) attack the opponent's king? */
        make_move(&b, m);
        int opp_king_sq = bb_lsb(b.pieces[b.side][KING]);
        bool ref = is_square_attacked(&b, (Square)opp_king_sq, (Color)(b.side ^ 1));
        unmake_move(&b);

        /* Optimized: move_gives_check. */
        bool opt = move_gives_check(&b, m);

        if (ref != opt) {
            test_fail++;
            char mv_str[6];
            /* format move manually */
            int from = MOVE_FROM(m), to = MOVE_TO(m);
            mv_str[0] = 'a' + (from & 7);
            mv_str[1] = '1' + (from >> 3);
            mv_str[2] = 'a' + (to & 7);
            mv_str[3] = '1' + (to >> 3);
            mv_str[4] = '\0';
            fprintf(stderr, "MISMATCH fen=%s move=%s ref=%d opt=%d\n",
                    fen, mv_str, ref, opt);
        }
    }
}

int main(void) {
    bitboard_init();
    board_init();

    test_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    test_position("r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 1");
    test_position("5k2/8/3pB3/2pP1N2/p2n4/P1K2b2/1P5q/5Q2 b - - 23 53");
    test_position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    test_position("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    test_position("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    test_position("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");

    printf("Tests: %d passed, %d failed\n", test_count - test_fail, test_fail);
    return test_fail > 0 ? 1 : 0;
}
