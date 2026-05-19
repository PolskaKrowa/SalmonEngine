AM_CFLAGS = -I$(top_srcdir)/include -Wall -Wextra -Wpedantic \
            -Wshadow -Wconversion -Wno-unused-parameter

# ── Main engine binary ────────────────────────────────────────
bin_PROGRAMS = engine

engine_SOURCES = \
    bitboard.c \
    board.c \
    eval.c \
    main.c \
    movegen.c \
    search.c \
    tt.c \
    uci.c

engine_LDADD = -lm

# ── Tuner binary (only built when --enable-tuner is passed) ───
if ENABLE_TUNER
bin_PROGRAMS += tuna

tuna_SOURCES = \
    tune.c \
    bitboard.c \
    board.c \
    movegen.c

tuna_CFLAGS  = $(AM_CFLAGS) -DTUNE_STANDALONE
tuna_LDADD   = -lm
endif