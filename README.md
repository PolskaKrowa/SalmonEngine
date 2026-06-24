# SalmonEngine 🐟

A classical chess engine in C. No neural networks, no learned weights — just bitboards, alpha-beta search, and a hand-written evaluation function you can read end-to-end.

The goal is an engine that is **transparent by construction**: every positional judgement it makes is traceable to a named term in `eval.c`, every search decision to a named pruning rule in `search.c`. The code is meant to be read, modified, and learned from.

[![CI](https://github.com/PolskaKrowa/SalmonEngine/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/PolskaKrowa/SalmonEngine/actions)
[![Release](https://img.shields.io/github/v/release/PolskaKrowa/SalmonEngine)](https://github.com/PolskaKrowa/SalmonEngine/releases)
[![License](https://img.shields.io/github/license/PolskaKrowa/SalmonEngine)](LICENSE)

---

## Features

### Board & move generation
- **Bitboard representation** — 12 piece bitboards + 3 occupancy bitboards + mailbox, all kept in sync incrementally by `make_move` / `unmake_move`.
- **Pseudo-legal move generation** with full pawn promotion, en-passant, and castling handling.
- **Slider attacks** via runtime-dispatched BMI2/PEXT tables on x86, with a Hyperbola-Quintessence fallback for everywhere else. A single binary runs unmodified across BMI2 and non-BMI2 hosts.

### Search
- **Iterative deepening** with aspiration windows.
- **Principal variation search** (PVS / null-window) at the core.
- **Transposition table** — 16-byte entries, 3-slot buckets, depth-preferred-with-generation replacement, mate-distance-adjusted scores, prefetch to hide L2/L3 latency.
- **Pruning**: null-move pruning (adaptive R, improving-aware), reverse futility pruning, razoring, ProbCut, futility pruning, late move pruning, SEE-based quiet and capture pruning.
- **Reductions & extensions**: late move reductions (product-of-logs table), singular extensions with multi-cut, check extensions, internal iterative reductions.
- **Move ordering**: TT move first, then MVV-LVA + SEE-classified captures, killers, countermove, countermove history, butterfly history, two-ply continuation history, all combined via SF-calibrated `stat_bonus` / history-malus updates.
- **Quiescence search** with stand-pat, delta pruning, and SEE pruning of losing captures.
- **Lazy SMP** multi-threading (1–64 threads), with per-thread SearchInfo and a shared TT.
- **Time management** with separate soft (allotted) and hard (max) limits, checked every 2048 nodes.

### Evaluation
A tapered (midgame/endgame) classical evaluation with:
- Material + piece-square tables, with a lazy-eval fast path that bails out when the position is lopsided.
- Material imbalance (quadratic polynomial — bishop pair, rook-vs-minors, etc.).
- Pawn structure: doubled, isolated, backward, passed, pawn chains, pawn islands.
- Piece-specific terms: knight/bishop outposts, trapped rook, weak queen, king-protector distance, minor-behind-pawn.
- King safety: pawn shield, open-file penalty, weighted attacker count, endgame king activity.
- Rook on open/semi-open file, rook on the seventh.
- Tempo, bishop pair, initiative/complexity correction.
- A pawn-structure cache so repeated pawn shapes are scored once per search.

### Protocol & I/O
- **UCI** with `position`, `go` (depth / movetime / wtime+btime+inc / infinite / ponder), `stop`, `ponderhit`, `setoption`.
- **UCI options**: `Hash` (1–2048 MB, auto-scaled with thread count), `Threads` (1–64), `OwnBook` (opening book on/off), `Ponder`.
- **Opening book** — built-in, probed before search.
- **Pondering** — the engine predicts the opponent's reply via a TT probe after its best move and emits `bestmove X ponder Y`.

### Tooling
- **Texel-tuning / self-play tuner** (`tuna`, built with `--enable-tuner`) — coordinate-descent parameter optimisation with a self-play game generator. Distributed-ready: workers generate `(position, result)` pairs; a master node runs the optimiser and broadcasts new weights.
- **NPS benchmark** (`scripts/nps_benchmark.py`) — CPU-normalised nodes-per-second measurement for stable cross-machine regression detection.
- **Elo match runner** (`scripts/elo_match.py`) — drives cutechess-cli matches and appends results to a JSONL history file for tracking strength changes over time.
- **Perft** smoke test (`make perft`).
- **Mate test suite** (`tests/mate_positions.csv` + `tests/test_mate.py`) — 1,798 mate-in-N positions scraped from real games.

---

## Getting started

### Prerequisites

A C compiler supporting C2x (GCC 9+ or Clang 9+) and the autotools build system:

```bash
# Debian/Ubuntu
sudo apt-get install autoconf automake gcc

# macOS
brew install autoconf automake
```

The `ax_check_compile_flag` autoconf macro is bundled in `m4/`, so `autoconf-archive` is not required.

### Build

```bash
git clone https://github.com/PolskaKrowa/SalmonEngine.git
cd SalmonEngine

autoreconf --install   # generate the configure script (only needed once)
./configure            # probe the system and generate Makefiles
make -j$(nproc)        # compile
```

The engine binary is `src/engine`.

### Build options

| Flag | Effect |
|---|---|
| `--enable-debug` | `-O0 -g3 -DENGINE_DEBUG` |
| `--enable-sanitize` | AddressSanitizer + UBSan, no optimisation |
| `--enable-tuner` | also build the parameter tuner binary `src/tuna` |
| `--enable-eval-debug` | compile in per-position evaluation breakdown |
| `--enable-superopt` | run each source file through Souper + Minotaur (see below) |

Example:

```bash
./configure --enable-sanitize       # development build
./configure --enable-tuner          # also build the tuner
```

### Superoptimisation (Souper + Minotaur)

For experimental builds the engine can be passed through two LLVM-based superoptimisers — [Souper](https://github.com/google/souper) (optimistic substitution with SMT solving) and [Minotaur](https://github.com/niklasso/minotaur) (a custom LLVM opt pass pipeline). This is **off by default**.

Both tools ship as **LLVM opt pass plugins** (`libSouperPass.so`, `libMinotaurPass.so`) rather than standalone binaries. The pipeline loads them via the standard LLVM `opt` driver:

```bash
./configure --enable-superopt \
            --with-clang=/path/to/clang \
            --with-opt=/path/to/opt \
            --with-llc=/path/to/llc \
            --with-souper-pass=/path/to/libSouperPass.so \
            --with-minotaur-pass=/path/to/libMinotaurPass.so \
            --with-souper=/path/to/souper
make
```

If `clang`, `opt`, `llc`, and `souper` are on `PATH`, the corresponding `--with-*` flags can be omitted and configure will auto-detect them. The pass-plugin paths default to `<opt-prefix>/../lib/libSouperPass.so` and `/usr/lib`, `/usr/local/lib` — override with `--with-souper-pass` / `--with-minotaur-pass` if they live elsewhere.

The `souper` binary is not invoked directly by the build — it is invoked at runtime by the Souper pass (to extract candidates and run the SMT solver). The build exposes its path to the pass via the `SOUPER_BIN` environment variable; if `souper` is not on `PATH` and `--with-souper` is not given, configure prints a warning and the pass will need to find it at runtime.

Each `.c` file flows through a four-stage pipeline:

```
foo.c            ──clang -emit-llvm -c──►  foo.bc
foo.bc           ──opt -load-pass-plugin libSouperPass.so -souper──►
                                              foo.souper.bc
foo.souper.bc    ──opt -load-pass-plugin libMinotaurPass.so -minotaur-pass──►
                                              foo.minotaur.bc
foo.minotaur.bc  ──llc + as────────────►  foo.o
```

**Per-stage fallback.** If the Souper pass fails on a particular file (SMT solver timeout, plugin load error, ABI mismatch with `opt`, etc.), that file's `.souper.bc` is set to a copy of its input `.bc` and the Minotaur pass still runs on it. If Minotaur fails, the `.minotaur.bc` is set to a copy of the Souper output and codegen still runs. Both stages log `OK` or `FAIL ... falling back to input` to stderr, so you can see which files were actually optimised. The wrapper is [`build-aux/superopt-stage.sh`](build-aux/superopt-stage.sh).

When superopt is disabled, automake's default `.c.o` rule (compiled with the system C compiler) is used and the superopt rules are not emitted at all.

---

## Usage

### With a chess GUI

Point any UCI-compatible GUI (Arena, Cutechess, BanksiaGUI, ChessBase, etc.) at the `engine` binary.

### From the terminal

```bash
./src/engine
```

Then type UCI commands:

```
uci                                            # engine responds with id and uciok
position startpos                              # set the starting position
position fen <fen>                             # set a position from FEN
go depth 10                                    # search to depth 10
go movetime 5000                               # search for 5 seconds
go wtime 60000 btime 60000 winc 1000 binc 1000 # time control
stop                                           # stop searching
quit                                           # exit
```

`setoption name Hash value 256` and `setoption name Threads value 4` adjust the transposition table and search threads at runtime.

---

## Project layout

```
SalmonEngine/
├── src/
│   ├── bitboard.c    Slider attacks (PEXT/HQ), square/rank/file tables
│   ├── board.c       Board state, FEN parsing, make/unmake, Zobrist hashing
│   ├── book.c        Built-in opening book
│   ├── eval.c        Classical tapered evaluation
│   ├── eval_debug.c  Per-position evaluation breakdown (optional)
│   ├── main.c        Entry point
│   ├── movegen.c     Pseudo-legal move generation + move_gives_check
│   ├── search.c      Iterative deepening, PVS, pruning, quiescence, SEE
│   ├── tt.c          Transposition table
│   ├── tune.c        Texel-tuning + self-play parameter optimiser
│   └── uci.c         UCI protocol handler
├── include/          Public headers
├── tests/            Mate-position test suite
├── scripts/          NPS benchmark, Elo match runner, helper tools
├── build-aux/        Autotools helpers + superopt stage wrapper
├── m4/               Bundled autoconf macros
├── configure.ac      Autoconf spec
├── Makefile.am       Top-level automake rules
└── src/Makefile.am   Per-source automake rules (incl. superopt pipeline)
```

---

## Design notes

### Why no NNUE?

NNUE produces strong moves but offers little insight into *why* a move was chosen. SalmonEngine stays with a hand-written evaluation so that every positional judgement is a named, tunable term you can read, understand, and modify. This makes the engine useful for learning the classical foundations before moving on to neural approaches.

### Why runtime CPU dispatch for sliders?

PEXT-based slider attack tables are significantly faster than Hyperbola Quintessence on Intel CPUs that have native PEXT, but microcoded and slow on some AMD Zen parts. The engine always compiles both code paths and dispatches at runtime via function pointers set in `bitboard_init()`, so a single binary runs well everywhere.

### Why Texel tuning + self-play?

Hand-written evaluation terms have many parameters (PST values, piece values, pawn-structure penalties, king-safety weights, …). The bundled `tuna` tuner runs coordinate descent against a corpus of labelled positions, with optional self-play generation of fresh training data. This lets the eval improve over time without sacrificing its readability — the structure stays in C, only the constants move.

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for build setup, code style, testing expectations, and the pull-request workflow.

## Resources

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [Bitboards on CPW](https://www.chessprogramming.org/Bitboards)
- [Souper](https://github.com/google/souper)
- [Minotaur](https://github.com/niklasso/minotaur)

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
