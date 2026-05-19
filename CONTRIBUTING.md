# Contributing to SalmonEngine

Thanks for your interest in contributing! This document covers how to get set up, what kinds of contributions are welcome, and how to submit them.

---

## What You Can Contribute

Anything is fair game, but here are areas where help is especially welcome:

- **Evaluation improvements** — better piece-square tables, mobility terms, king safety, pawn structure
- **Search enhancements** — null move pruning, late move reductions, aspiration windows, etc.
- **Bug fixes** — especially movegen correctness (perft regressions are your friend)
- **UCI options** — exposing eval/search parameters at runtime
- **Documentation** — explaining how the code works, fixing typos, adding comments
- **CI/tooling** — build system, workflow improvements

If you're unsure whether an idea fits, open an issue first and describe what you're thinking.

---

## Getting Set Up

### Prerequisites

```bash
# Debian/Ubuntu
sudo apt-get install autoconf automake autoconf-archive gcc

# macOS
brew install autoconf automake autoconf-archive
```

### Build the Engine

```bash
git clone https://github.com/yourusername/SalmonEngine.git
cd SalmonEngine

autoreconf --install
./configure
make -j$(nproc)
```

### Recommended: Build with Sanitizers

For development work, build with sanitizers enabled so memory bugs and undefined behaviour surface immediately:

```bash
./configure --enable-sanitize
make -j$(nproc)
```

### Run the Perft Suite

Before making any changes, confirm the baseline passes:

```bash
make perft
```

After your changes, run it again. If the node counts change, you've broken movegen.

---

## Making Changes

### Branching

Work on a feature branch, not directly on `main`:

```bash
git checkout -b my-feature
```

### Code Style

SalmonEngine is plain C (C2x). Keep new code consistent with what's already there:

- 4-space indentation, no tabs
- Snake case for functions and variables (`make_move`, `piece_value`)
- All-caps for macros and constants (`MAX_DEPTH`, `RANK_MASK`)
- Keep functions focused — if a function is getting long, it probably does too much
- Comment *why*, not *what* — the code says what, comments explain intent

### Evaluation Changes

If you're tweaking `eval.c`, include perft results before and after to confirm movegen is untouched, and if you have self-play Elo data (via cutechess or similar) include that too. Even informal "tested at 10+0.1, +15 ± 20 Elo" is useful context.
There is a github workflow that *should* test your changes against the main branch engine and report results for you, This is simply to provide extra information if you fail to provide your own results.

### Search Changes

Search changes are the trickiest to evaluate. Try to:

- Isolate one change at a time
- Test at short time controls (5+0.05 or 10+0.1)
- Run enough games that the result is meaningful (200+ games minimum)

Describing your test setup in the PR helps a lot.

---

## Submitting a Pull Request

1. Make sure `make perft` still passes with the correct node counts
2. If you added new source files, make sure they're listed in `src/Makefile.am`
3. Push your branch and open a pull request against `main`
4. Fill in the PR description — what changed, why, and how you tested it

The CI will automatically:
- Build on Linux and macOS across multiple architectures
- Run the perft regression suite
- Report any compile errors or sanitizer findings

---

## Reporting Bugs

Open an issue and include:

- What you did
- What you expected to happen
- What actually happened
- The FEN of any relevant position
- Your OS and compiler version (`gcc --version` or `clang --version`)

For movegen bugs, a failing perft position and depth is the most useful thing you can provide.

---

## Questions

Open an issue tagged `question` and ask away. Chess engine development has a steep learning curve and there are no dumb questions.

Although there's no good moderation system in place for issue discussions, We do still advise all users to be respectful to eachother and not skip out on detail. If a user is confused or is reporting an issue, even if it's user error, the problem should still be resolved without conflict nor (whether expressed or otherwise) discrimination towards anyone.
We support a growing community of keen users of this software, and strongly discourage negativity. People simply have different ways of understanding or taking in information.

