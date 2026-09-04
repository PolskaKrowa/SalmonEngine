AM_CFLAGS = -I$(top_srcdir)/include -Wall -Wextra -Wpedantic \
            -Wshadow -Wconversion -Wno-unused-parameter

# ── Main engine binary ──────────────────────────────────────
bin_PROGRAMS = engine

engine_SOURCES = \
    bitboard.c \
    board.c \
    book.c \
    eval.c \
    main.c \
    movegen.c \
    numa_utils.c \
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
#    foo.bc ──opt -load-pass-plugin libSouperPass.so -souper──►
#            foo.souper.bc                              (Souper pass)
#    foo.souper.bc ──opt -load-pass-plugin libMinotaurPass.so -minotaur-pass──►
#            foo.minotaur.bc                            (Minotaur pass)
#    foo.minotaur.bc ──llc + as───►  foo.o             (codegen)
#
#  Souper and Minotaur are LLVM opt pass plugins, NOT standalone
#  binaries.  The Souper pass shells out to the `souper` binary at
#  runtime to extract candidates and run the SMT solver; we expose
#  the souper binary's path to the pass via the SOUPER_BIN env var.
#
#  Per-stage fallback (build-aux/superopt-stage.sh):
#    • If the Souper pass fails on foo.bc (e.g. SMT solver timeout,
#      plugin load error, ABI mismatch with opt), foo.souper.bc is
#      a copy of foo.bc and the Minotaur pass still runs on it.
#    • If the Minotaur pass fails on foo.souper.bc, foo.minotaur.bc
#      is a copy of foo.souper.bc and codegen still runs.
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

# Tools (substituted by configure.ac via AC_SUBST).  The CLANG_BIN /
# OPT_BIN / LLC_BIN / SOUPER_PASS / MINOTAUR_PASS / SOUPER_BIN
# variables are auto-substituted into the Makefile by configure, so
# the per-file rules below reference them directly.  We define short
# aliases here ONLY for the ones that aren't AC_SUBST'd (none in this
# case) and for the derived *_CMD variables that combine opt + plugin.
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

# ── Per-stage opt invocations ────────────────────────────────
#
# Souper and Minotaur are not standalone binaries in the pipeline —
# they ship as LLVM opt pass plugins (libSouperPass.so /
# libMinotaurPass.so) loaded by the standard LLVM `opt` driver via
# -load-pass-plugin.
#
# The Souper pass shells out to the `souper` binary at runtime to
# extract candidates and run the SMT solver.  It finds `souper` via
# the SOUPER_BIN environment variable (falling back to PATH if unset),
# so we export SOUPER_BIN when invoking the Souper stage.
#
# The superopt-stage.sh wrapper signature is:
#   superopt-stage.sh <stage_name> <tool...> <input.bc> <output.bc>
# and it runs `<tool...> <input.bc> -o <tmp_output>`.  So the *_CMD
# variables below are passed as the leading tool arguments; the
# wrapper appends the input bitcode and `-o <tmp_output>`.  The
# resulting command for the Souper stage is:
#   opt -load-pass-plugin libSouperPass.so -souper <input.bc> -o <tmp>
# which is the form opt expects.
SOUPER_CMD_ENV = SOUPER_BIN="$(SOUPER_BIN)"
SOUPER_CMD = $(OPT_BIN) -load-pass-plugin $(SOUPER_PASS) -souper
MINOTAUR_CMD = $(OPT_BIN) -load-pass-plugin $(MINOTAUR_PASS) -minotaur-pass

# ── Pipeline rules for each engine source file ──────────────
# Each file gets four rules: .bc, .souper.bc, .minotaur.bc, .o.
# The .o rule's .minotaur.bc prerequisite overrides automake's
# default .c.o rule (which would compile with gcc) — when ENABLE_SUPEROPT
# is set, the .o is rebuilt via the bitcode pipeline instead.

# bitboard.c
bitboard.bc: bitboard.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
bitboard.souper.bc: bitboard.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
bitboard.minotaur.bc: bitboard.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
bitboard.o: bitboard.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# board.c
board.bc: board.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
board.souper.bc: board.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
board.minotaur.bc: board.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
board.o: board.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# book.c
book.bc: book.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
book.souper.bc: book.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
book.minotaur.bc: book.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
book.o: book.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# eval.c
eval.bc: eval.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
eval.souper.bc: eval.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
eval.minotaur.bc: eval.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
eval.o: eval.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# main.c
main.bc: main.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
main.souper.bc: main.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
main.minotaur.bc: main.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
main.o: main.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# movegen.c
movegen.bc: movegen.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
movegen.souper.bc: movegen.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
movegen.minotaur.bc: movegen.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
movegen.o: movegen.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# search.c
search.bc: search.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
search.souper.bc: search.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
search.minotaur.bc: search.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
search.o: search.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# tt.c
tt.bc: tt.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
tt.souper.bc: tt.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
tt.minotaur.bc: tt.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
tt.o: tt.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# uci.c
uci.bc: uci.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
uci.souper.bc: uci.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
uci.minotaur.bc: uci.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
uci.o: uci.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s

# eval_debug.c (only when ENABLE_EVAL_DEBUG)
if ENABLE_EVAL_DEBUG
eval_debug.bc: eval_debug.c
        @echo "[superopt:1-clang] $< -> $@"
        $(CLANG_BIN) $(BC_CFLAGS) -c $< -o $@
eval_debug.souper.bc: eval_debug.bc
        @echo "[superopt:2-souper] $< -> $@"
        $(SOUPER_CMD_ENV) $(SUPEROPT_WRAPPER) souper $(SOUPER_CMD) $< $@
eval_debug.minotaur.bc: eval_debug.souper.bc
        @echo "[superopt:3-minotaur] $< -> $@"
        $(SUPEROPT_WRAPPER) minotaur $(MINOTAUR_CMD) $< $@
eval_debug.o: eval_debug.minotaur.bc
        @echo "[superopt:4-codegen] $< -> $@"
        $(LLC_BIN) $(LLC_FLAGS) $< -o $@.s
        $(CC) -c $@.s -o $@
        @rm -f $@.s
endif

# Clean up the intermediate bitcode files on `make clean`.
clean-superopt:
        @echo "Cleaning superopt bitcode intermediates"
        @rm -f *.bc *.souper.bc *.minotaur.bc *.bc.log *.souper.bc.log *.minotaur.bc.log *.s
clean-local: clean-superopt

endif  # ENABLE_SUPEROPT
