# Contributing to SalmonEngine

Thanks for your interest in contributing. This document covers how to get set up, what kinds of contributions are welcome, and how to submit them.

---

## What you can contribute

Anything is fair game, but here are areas where help is especially welcome:

- **Evaluation** — new terms, better weights, refactor of existing scoring. The evaluation is fully tunable via the `EvalWeights` struct in `include/eval.h`; the `tuna` tuner can optimise any of those parameters.
- **Search** — pruning tweaks, new reductions, better move ordering. Search changes are the highest-variance area; see the testing guidance below.
- **Move generation** — speedups, correctness fixes. Perft is the source of truth here.
- **UCI surface** — new runtime options, better pondering, time-management improvements.
- **Tooling** — the build system, the superopt pipeline, the NPS benchmark, the Elo match runner.
- **Documentation** — explaining how the code works, fixing typos, adding examples.

If you're unsure whether an idea fits, open an issue first and describe what you're thinking.

---

## Getting set up

### Prerequisites

```bash
# Debian/Ubuntu
sudo apt-get install autoconf automake gcc

# macOS
brew install autoconf automake
```

GCC 9+ or Clang 9+ is required (the code uses C2x). The `ax_check_compile_flag` macro is bundled in `m4/`, so `autoconf-archive` is not needed.

### Build the engine

```bash
git clone https://github.com/PolskaKrowa/SalmonEngine.git
cd SalmonEngine

autoreconf --install
./configure
make -j$(nproc)
```

The engine binary is `src/engine`.

### Recommended: build with sanitizers

For development work, build with AddressSanitizer + UBSan so memory bugs and undefined behaviour surface immediately:

```bash
./configure --enable-sanitize
make -j$(nproc)
```

This is slower than a release build but catches a large class of bugs that would otherwise manifest as silent corruption hours later.

### Verify the baseline

Before making any changes, confirm the baseline passes:

```bash
make perft       # perft smoke test on a known position
```

And the mate-test suite:

```bash
python3 tests/test_mate.py --engine src/engine --depth 14
```

After your changes, run both again. If perft node counts change, you've broken move generation. If the mate-test pass rate drops, you've likely weakened the search.

---

## Testing your changes

### Perft (movegen correctness)

Perft counts the number of leaf nodes at a given depth from a position. Any change to `make_move`, `unmake_move`, `movegen.c`, or `bitboard.c` should be checked against known perft counts. The `make perft` target runs a quick smoke test; for thorough verification, run perft to depth 5 or 6 on the standard test positions (startpos, Kiwipete, and the CPW "Position 3/4/5") and compare against published reference counts.

### NPS (regression detection)

```bash
python3 scripts/nps_benchmark.py --engine src/engine --label my-change
```

This runs a CPU-normalised NPS measurement across a fixed set of positions. The CPU normalisation (a deterministic integer workload whose runtime correlates with raw CPU throughput) makes the result roughly comparable across machines, so a regression on your laptop is meaningful when reviewed on CI.

### Strength (Elo)

For evaluation or search changes that should affect playing strength, run a cutechess-cli match:

```bash
python3 scripts/elo_match.py \
    --engine-a src/engine \
    --engine-b /tmp/engine-baseline \
    --name-a my-change --name-b baseline \
    --time-control 10+0.1 --rounds 200
```

Two hundred games at 10+0.1 is a reasonable minimum for spotting medium-size effects; smaller effects need more games to distinguish from noise.

### Superopt builds (if you touched the build system)

If you've changed `configure.ac`, `Makefile.am`, or `build-aux/superopt-stage.sh`, verify the superopt pipeline still works end-to-end. The pipeline uses `clang` to emit bitcode, then runs it through two LLVM opt pass plugins (`libSouperPass.so`, `libMinotaurPass.so`) via the standard `opt` driver, then codegens with `llc`. If you don't have all the tools installed, the wrapper's per-stage fallback will copy bitcode through unchanged and the build should still complete — that's a useful smoke test of the pipeline itself.

```bash
./configure --enable-superopt \
            --with-clang=$(which clang) \
            --with-opt=$(which opt) \
            --with-llc=$(which llc) \
            --with-souper-pass=/path/to/libSouperPass.so \
            --with-minotaur-pass=/path/to/libMinotaurPass.so \
            --with-souper=$(which souper)
make clean && make
```

Watch the `[souper] OK/FAIL` and `[minotaur] OK/FAIL` lines per source file — they tell you which files were actually optimised and which fell back.

---

## Making changes

### Branching

Work on a feature branch, not directly on `main`:

```bash
git checkout -b my-feature
```

### Code style

SalmonEngine is plain C (C2x). Keep new code consistent with what's already there:

- **4-space indentation, no tabs.**
- **Snake case** for functions and variables (`make_move`, `piece_value`).
- **All-caps** for macros and constants (`MAX_PLY`, `RANK_MASK`).
- **Keep functions focused** — if a function is getting long, it probably does too much.
- **Comment *why*, not *what*.** The code says what; comments explain intent. The search module is heavily commented because the pruning rules interact in non-obvious ways; match that bar for new search code.
- **Mark hot functions** with `__attribute__((hot))` and pure/const with the corresponding attributes — the existing code does this and it helps the compiler.

### Evaluation changes

If you're touching `eval.c`:

1. Run perft before and after to confirm movegen is untouched.
2. Run the NPS benchmark — evaluation is on the hot path, so a 5% NPS regression matters.
3. Run an Elo match if the change should affect strength. Even informal "tested at 10+0.1, +12 ± 18 Elo over 300 games" is useful context.
4. If your change adds new parameters, add them to `EvalWeights` in `include/eval.h` so the tuner can pick them up.

### Search changes

Search changes are the trickiest to evaluate. Try to:

- **Isolate one change at a time.** Stacking pruning tweaks makes regressions invisible.
- **Test at short time controls first** (5+0.05 or 10+0.1) — long time controls amplify noise and take forever.
- **Run enough games.** 200 is a minimum for medium effects; 500+ for small ones. The Elo match runner reports a confidence interval — if it crosses zero, you don't have evidence of improvement.
- **Check NPS.** A more aggressive pruner may have lower NPS (each node is "richer") but reach the target depth faster. Both numbers matter.
- **Watch for tactical regressions.** Run the mate-test suite — a higher pass rate is a good sign; a lower one is a red flag.

### Tuner changes

If you're touching `tune.c` or the `EvalWeights` struct:

- Run the tuner end-to-end on a small corpus to confirm it still converges.
- If you add parameters to `EvalWeights`, make sure the tuner's parameter-iteration code picks them up (it iterates over named offsets in the struct).
- Tuner output is C source — confirm the emitted weights compile and produce a working engine.

---

## Submitting a pull request

1. **Make sure `make perft` still passes** with the correct node counts.
2. **Run the mate-test suite** — pass rate should not drop.
3. **If you added new source files**, list them in `src/Makefile.am` (and in `src/Makefile.am`'s superopt block if they should go through the Souper/Minotaur pipeline).
4. **Push your branch** and open a pull request against `main`.
5. **Fill in the PR description** — what changed, why, and how you tested it. Include perft counts, NPS numbers, and Elo results where relevant.

CI will automatically build across Linux and macOS on multiple architectures, run the perft regression suite, and report any compile errors or sanitizer findings.

---

## Reporting bugs

Open an issue and include:

- What you did (commands typed, position sent).
- What you expected to happen.
- What actually happened (engine output, crash, illegal move, etc.).
- The FEN of any relevant position.
- Your OS and compiler version (`gcc --version` or `clang --version`).
- The engine's build configuration (`--enable-*` flags passed to `./configure`).

For move-generation bugs, a failing perft position and depth is the most useful thing you can provide. For search bugs (wrong move, missed mate, etc.), the FEN, the search depth, and the engine's `info` lines up to the wrong move are essential.

---

## Questions

Open an issue tagged `question` and ask away. Chess engine development has a steep learning curve and there are no dumb questions.

We support a growing community of keen users of this software and strongly discourage negativity. People have different ways of understanding or taking in information, and a user reporting an issue — even when it turns out to be user error — should be helped without conflict or discrimination. Be respectful, be detailed, and assume good faith.
