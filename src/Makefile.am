# src/Makefile.am — Salmon Chess Engine
#
# Targets
# ───────
#   engine   always built            — the UCI chess engine
#   tuna     --enable-tuner          — single-node NNUE RL tuner
#   tunad    --enable-dist-tuner     — distributed MPI NNUE RL tuner
#
# Source sharing
# ──────────────
# tune.c is compiled twice when both tuna and tunad are enabled:
#
#   tuna_tune.o   — compiled with -DTUNE_STANDALONE (owns main())
#   tunad_tune.o  — compiled without it (dist_tune_main.c owns main())
#
# Automake's subdir-objects mode (set in configure.ac) generates distinct
# object file names per target automatically, so the two compilations of
# tune.c do not collide.
#
# Per-target _CFLAGS must always include AM_CFLAGS explicitly; Automake
# does not merge AM_CFLAGS into a per-target _CFLAGS automatically.

AM_CFLAGS = \
    -I$(top_srcdir)/include \
    -Wall -Wextra -Wpedantic \
    -Wshadow -Wconversion -Wno-unused-parameter

# ── Headers ───────────────────────────────────────────────────────────────
# All headers in include/ are listed here so `make dist` includes them in
# the distribution tarball. Without this, the tarball would contain only
# the .c files and fail to compile for anyone who unpacks it.
noinst_HEADERS = \
    $(top_srcdir)/include/bitboard.h \
    $(top_srcdir)/include/board.h \
    $(top_srcdir)/include/book.h \
    $(top_srcdir)/include/config.h \
    $(top_srcdir)/include/dist_tune.h \
    $(top_srcdir)/include/eval.h \
    $(top_srcdir)/include/movegen.h \
    $(top_srcdir)/include/nnue.h \
    $(top_srcdir)/include/search.h \
    $(top_srcdir)/include/tt.h \
    $(top_srcdir)/include/tune.h \
    $(top_srcdir)/include/tune_internal.h \
    $(top_srcdir)/include/types.h \
    $(top_srcdir)/include/uci.h

# ── Shared engine sources ─────────────────────────────────────────────────
#
# Collected in a variable so each target below can reference it without
# repeating the list (and risking the lists drifting out of sync).

ENGINE_SOURCES = \
    bitboard.c \
    board.c \
    movegen.c \
    nnue.c

# ── Main engine binary ────────────────────────────────────────────────────

bin_PROGRAMS = engine

engine_SOURCES = \
    $(ENGINE_SOURCES) \
    book.c \
    eval.c \
    main.c \
    search.c \
    tt.c \
    uci.c

engine_LDADD = -lm

# ── Single-node tuner (bin/tuna) ──────────────────────────────────────────
#   Enabled by: ./configure --enable-tuner
#               ./configure --enable-dist-tuner  (implied)
#
#   tune.c compiled with -DTUNE_STANDALONE so it defines main().

if ENABLE_TUNER
bin_PROGRAMS += tuna

tuna_SOURCES = \
    $(ENGINE_SOURCES) \
    tune.c

tuna_CFLAGS  = $(AM_CFLAGS) -DTUNE_STANDALONE
tuna_LDADD   = -lm $(OPENBLAS_LIBS) -lpthread

bin_PROGRAMS += suptuna
suptuna_SOURCES = $(ENGINE_SOURCES) search.c eval.c tt.c uci.c book.c supervised_tune.c
suptuna_CFLAGS  = $(AM_CFLAGS) -DSUPERVISED_STANDALONE
suptuna_LDADD   = -lm $(OPENBLAS_LIBS) -lpthread
endif

# ── Distributed tuner (bin/tunad) ────────────────────────────────────────
#   Enabled by: ./configure --enable-dist-tuner
#
#   Three extra source files relative to tuna:
#     tune_internal.h   — exposes tune.c internals to dist_tune.c
#     dist_tune.c/.h    — AllReduce gradient sync + HMAC checkpoint I/O
#     dist_tune_main.c  — distributed main(); parses args, loads HMAC key
#
#   tune.c is compiled here WITHOUT -DTUNE_STANDALONE so dist_tune_main.c
#   owns main().  Automake generates a separate tunad_tune.o for this
#   compilation because tunad_CFLAGS differs from tuna_CFLAGS.
#
#   MPI_CFLAGS adds the OpenMPI include path (-I/usr/.../openmpi/include).
#   OPENSSL_CFLAGS adds the OpenSSL include path (usually nothing on
#   standard installs, but correct for non-default prefixes).

if ENABLE_DIST_TUNER
bin_PROGRAMS += tunad

tunad_SOURCES = \
    $(ENGINE_SOURCES) \
    tune.c \
    dist_tune.c \
    dist_tune_main.c

tunad_CFLAGS  = $(AM_CFLAGS) $(MPI_CFLAGS) $(OPENSSL_CFLAGS)
tunad_LDADD   = -lm $(OPENBLAS_LIBS) -lpthread $(MPI_LIBS) $(OPENSSL_LIBS)
endif