> [!CAUTION]
> SalmonEngine's NNUE branch ***requires*** the existance of a .bin file which contains the model weights, and does not (and will not) come with hard-coded weights inside of the program binary. Currently the bot will initialise random weights on program start, which may feel like the engine isn't good at chess. In the not too distant future, i will provide pretuned model weights in the repository source and releases page.

# SalmonEngine 🐟

A ~~traditional~~ chess engine built in C, designed to be ~~**understandable, transparent, and easy to learn from**. No neural networks, no black boxes — just pure chess logic you can read, follow, and modify without a PhD in machine learning.~~ good.

> okay, the search module may look a bit intimidating. I promise it's GCC's fault and not mine.

[![CI](https://github.com/PolskaKrowa/SalmonEngine/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/PolskaKrowa/SalmonEngine/actions)
[![Release](https://img.shields.io/github/v/release/PolskaKrowa/SalmonEngine)](https://github.com/PolskaKrowa/SalmonEngine/releases)
[![License](https://img.shields.io/github/license/PolskaKrowa/SalmonEngine)](LICENSE)

---

## Features

- ~~**Classical evaluation** — Transparent material and positional scoring you can actually follow~~
- **Bitboard move generation** — Fast 64-bit board representation using `uint64_t`
- **UCI protocol** — Works out of the box with Arena, ChessBase, [Lichess](https://lichess.org/@/SteveParker), and other GUIs
- **Transposition table** — Hash-based caching for improved search efficiency
- ~~**No NNUE, no magic** — No machine learning, no precomputed syzygy tables. The engine thinks for itself~~ This branch uses an NNUE evaluator (purely for experimental purposes.)
- **Opening book** — because even salmon know where they're going *(later stages of the game are on its own)*

---

## Getting Started

### Prerequisites

You'll need a C compiler (GCC 9+ or Clang 9+) and the autotools build system:

```bash
# Debian/Ubuntu
sudo apt-get install autoconf automake autoconf-archive gcc

# macOS
brew install autoconf automake autoconf-archive
```

### Building from Source

```bash
git clone https://github.com/PolskaKrowa/SalmonEngine.git
cd SalmonEngine

autoreconf --install   # generate configure files (only needed once)
./configure            # probe your system and generate Makefile
make -j$(nproc)        # compile using all available CPU cores

make perft             # optional: run a quick smoke test
```

The engine binary will be at `src/engine`.

### Pre-built Binaries

Pre-compiled binaries for Linux, macOS, and Windows across x86-64 and ARM64 are available on the [Releases](https://github.com/PolskaKrowa/SalmonEngine/releases) page (from V2.1 onwards).

| Platform       | Architecture | AVX Support  |
|----------------|-------------|--------------|
| Linux          | x86-64      | baseline     |
| Linux          | x86-64-v3   | AVX2         |
| Linux          | x86-64-v4   | AVX-512      |
| Linux          | ARM64       | —            |
| macOS          | ARM64       | Apple Silicon|

> If you're not sure which to pick, use the `x86_64` build. Use `x86_64_v3` if your CPU is from 2013 or later and you want better performance.

### Build Options

```bash
./configure --enable-debug      # debug build (-O0 -g3 -DENGINE_DEBUG)
./configure --enable-sanitize   # AddressSanitizer + UBSan
./configure --enable-tuner      # also build the evaluation tuner (bin/tuna)
```

---

## Usage

### With a Chess GUI

Point any UCI-compatible GUI (Arena, Cutechess, BanksiaGUI, etc.) at the `engine` binary. It speaks standard UCI.

### From the Terminal

```bash
./src/engine
```

Then type UCI commands directly:

```
uci                                            # initialise, engine responds with id and uciok
position startpos                              # set starting position
position fen <fen>                             # set position from FEN string
go depth 10                                    # search to depth 10
go movetime 5000                               # search for 5 seconds
go wtime 60000 btime 60000 winc 1000 binc 1000 # time control
stop                                           # stop searching
quit                                           # exit
```

---

## Project Structure

```
SalmonEngine/
├── src/
│   ├── bitboard.c   — Bitboard utilities and move representation
│   ├── board.c      — Board state, FEN parsing, make/unmake move
│   ├── eval.c       — Evaluation engine
│   ├── nnue.c       — My worst nightmare
│   ├── main.c       — Entry point
│   ├── movegen.c    — Legal move generation
│   ├── search.c     — Minimax / alpha-beta search
│   ├── tt.c         — Transposition table
│   ├── tune.c       — Multithreaded evaluation tuner
│   └── uci.c        — UCI protocol handler
├── include/         — Header files
├── configure.ac     — Autoconf spec
├── Makefile.am      — Automake rules
└── README.md
```

---

## Design Philosophy

### Why No NNUE? (on the main branch)

NNUE (Efficiently Updatable Neural Networks) is powerful, but it creates a black box — it's hard to understand *why* a move was chosen, let alone improve it. SalmonEngine prioritises **clarity over raw strength**, making it ideal for:

- Learning how chess engines work from the ground up
- Experimenting with evaluation tweaks and seeing exactly what changed
- Understanding classical techniques before diving into neural approaches

### Why an NNUE on a separate branch?

NNUE is... i'm not going to give the whole explanation again.

This branch is purely for experimenting with using an NNUE engine for board evaluation, with release binaries from this branch being marked with "NNUE".

### Name Origin

~~Like salmon returning to their birthplace, this engine returns to the foundational principles of chess programming — the era before NNUE dominated the field.~~

I like salmon.

---

## Tuning

### Tuna

This branch uses an Adam tuner (named Tuna for an obvious reason: Fish.) and can be compiled by passing `--enable-tuner` to the `./configure` command.

Tuna requires the "OpenBLAS" library and development headers to be installed.

### Tunad

This isn't a mis-spelling of "Tuna", it's a local-cluster distributed tuning program that uses OpenMPI to distribute the computational workload across multiple machines on the same network. (Linux/MacOS only. Sorry windows users.)

To use Tunad properly, you must configure your OpenMPI cluster network to allow Tunad to share data across your compute nodes and allow for all processes to run properly.

A full guide on using Tunad can be found in [this document](TUNAD.md)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

## Resources

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [Bitboards on CPW](https://www.chessprogramming.org/Bitboards)

## License

See [LICENSE](LICENSE) for details.
This program is licensed under the Apache 2.0 Licence:

Copyright 2026 Steve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
