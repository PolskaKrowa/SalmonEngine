AM_CFLAGS = -I$(top_srcdir)/include -Wall -Wextra -Wpedantic \
            -Wshadow -Wconversion -Wno-unused-parameter

# ── Main engine binary ────────────────────────────────────────
bin_PROGRAMS = engine

engine_SOURCES = \
    bitboard.c \
    board.c \
    book.c \
    eval.c \
    main.c \
    movegen.c \
    search.c \
    tt.c \
    uci.c

if ENABLE_EVAL_DEBUG
engine_SOURCES += eval_debug.c
endif

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

# ─────────────────────────────────────────────────────────────
#  Superoptimisation: Souper + Minotaur
#
#  Enabled only when ./configure was run with --enable-superopt.
#  Each .c file goes through this pipeline:
#
#    foo.c  ──clang -emit-llvm -c──►  foo.bc           (bitcode emit)
#    foo.bc ──souper──────────────►  foo.souper.bc     (Souper pass)
#    foo.souper.bc ──minotaur─────►  foo.minotaur.bc   (Minotaur pass)
#    foo.minotaur.bc ──llc + as───►  foo.o             (codegen)
#
#  Per-stage fallback (build-aux/superopt-stage.sh):
#    • If Souper fails on foo.bc, foo.souper.bc is a COPY of foo.bc
#      and Minotaur still runs on it.
#    • If Minotaur fails on foo.souper.bc, foo.minotaur.bc is a COPY
#      of foo.souper.bc and codegen still runs.
#    • Both stages log their result to stderr so you can see which
#      files were actually optimised vs. fell back.
#
#  When ENABLE_SUPEROPT is NOT set, automake's default .c.o rule
#  (compiled with $(CC) = gcc) is used — the rules below are not
#  emitted at all.
#
#  NOTE: We use explicit per-file rules (rather than pattern rules or
#  foreach/eval, both of which are GNU make extensions and trigger
#  automake -Wportability warnings).  The list of source files is
#  short and stable, so writing them out explicitly is fine.
# ─────────────────────────────────────────────────────────────
if ENABLE_SUPEROPT

# Tools (substituted by configure.ac).
CLANG      = @CLANG_BIN@
LLC        = @LLC_BIN@
SOUPER     = @SOUPER_BIN@
MINOTAUR   = @MINOTAUR_BIN@
SUPEROPT_WRAPPER = $(top_srcdir)/build-aux/superopt-stage.sh

# Flags for the initial clang bitcode emission.  We use the same
# -O/-march flags as the regular build, plus -emit-llvm.  We drop
# -flto (no point for bitcode) and add -Xclang -disable-llvm-passes
# so clang doesn't run its own optimisation pipeline before we get
# a chance to — we want Souper and Minotaur to see the unoptimised IR.
BC_CFLAGS = $(AM_CFLAGS) -DNDEBUG -fomit-frame-pointer \
            @MARCH_NATIVE@ -emit-llvm -Xclang -disable-llvm-passes

# Flags for llc when converting the final optimised bitcode to assembly.
# We use -O3 to enable LLVM's codegen optimisations (instruction selection,
# register allocation, peephole).  The file-level .bc is already optimised
# by Souper + Minotaur; this is just machine-specific lowering.
LLC_FLAGS = -O3

# ── Pipeline rules for each engine source file ──────────────
# Each file gets four rules: .bc, .souper.bc, .minotaur.bc, .o.
# The .o rule's .minotaur.bc prerequisite overrides automake's
# default .c.o rule (which would compile with gcc) — when ENABLE_SUPEROPT
# is set, the .o is rebuilt via the bitcode pipeline instead.

# bitboard.c
bitboard.bc: bitboard.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
bitboard.souper.bc: bitboard.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
bitboard.minotaur.bc: bitboard.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
bitboard.o: bitboard.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# board.c
board.bc: board.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
board.souper.bc: board.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
board.minotaur.bc: board.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
board.o: board.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# book.c
book.bc: book.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
book.souper.bc: book.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
book.minotaur.bc: book.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
book.o: book.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# eval.c
eval.bc: eval.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
eval.souper.bc: eval.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
eval.minotaur.bc: eval.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
eval.o: eval.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# main.c
main.bc: main.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
main.souper.bc: main.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
main.minotaur.bc: main.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
main.o: main.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# movegen.c
movegen.bc: movegen.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
movegen.souper.bc: movegen.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
movegen.minotaur.bc: movegen.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
movegen.o: movegen.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# search.c
search.bc: search.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
search.souper.bc: search.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
search.minotaur.bc: search.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
search.o: search.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# tt.c
tt.bc: tt.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
tt.souper.bc: tt.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
tt.minotaur.bc: tt.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
tt.o: tt.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# uci.c
uci.bc: uci.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
uci.souper.bc: uci.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
uci.minotaur.bc: uci.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
uci.o: uci.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s

# eval_debug.c (only when ENABLE_EVAL_DEBUG)
if ENABLE_EVAL_DEBUG
eval_debug.bc: eval_debug.c
	@echo "[superopt:1-clang] $< -> $@"
	$(CLANG) $(BC_CFLAGS) -c $< -o $@
eval_debug.souper.bc: eval_debug.bc
	@echo "[superopt:2-souper] $< -> $@"
	$(SUPEROPT_WRAPPER) souper $(SOUPER) $< $@
eval_debug.minotaur.bc: eval_debug.souper.bc
	@echo "[superopt:3-minotaur] $< -> $@"
	$(SUPEROPT_WRAPPER) minotaur $(MINOTAUR) $< $@
eval_debug.o: eval_debug.minotaur.bc
	@echo "[superopt:4-codegen] $< -> $@"
	$(LLC) $(LLC_FLAGS) $< -o $@.s
	$(CC) -c $@.s -o $@
	@rm -f $@.s
endif

# Clean up the intermediate bitcode files on `make clean`.
clean-superopt:
	@echo "Cleaning superopt bitcode intermediates"
	@rm -f *.bc *.souper.bc *.minotaur.bc *.bc.log *.souper.bc.log *.minotaur.bc.log *.s
clean-local: clean-superopt

endif  # ENABLE_SUPEROPT
