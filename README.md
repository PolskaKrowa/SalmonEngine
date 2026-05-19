# SalmonEngine

A traditional chess engine built in C, designed to be **understandable, transparent, and easy to learn from**. No neural networks, no black boxes, no "Unified Theory of Slightly Better Bishops heuristic" bullsh\*\*tery. Just pure chess logic you can understand and modify without being a chess GM.

okay, maybe the search module may look a bit intimidating... I promise it's GCC's fault and not mine.

## Features

- **Classical evaluation** - Transparent material and positional evaluation you can follow
- **Efficient move generation** - Bitboard-based move generation for performance
- **UCI protocol support** - Compatible with UCI chess GUIs (Arena, ChessBase, [Lichess](https://lichess.org/@/SteveParker), etc.)
- **Transposition table** - Hash-based caching for improved search efficiency
- **No NNUE dependency** - Easy to build and understand; no machine learning complexity
- **Pure engine thinking** - No precomputed move tables, syzygy tables, ~~opening books~~, etc. The engine does all the thinking for itself during the ~~whole~~ later stages of the game.

## Project Structure

```
src/
├── bitboard.c      - Bitboard utilities and move representation
├── board.c         - Chess board state management
├── eval.c          - Position evaluation and scoring
├── main.c          - Entry point
├── movegen.c       - Legal move generation
├── search.c        - Search algorithm (minimax/alpha-beta)
├── tt.c            - Transposition table (hash table)
├── tune.c          - Multithreaded evaluation tuner
└── uci.c           - UCI protocol implementation
```

## Building

To build the engine binary:
```bash
# you must have autoconf, automake and autoconf-archive installed.
autoreconf --install     # creates the necessary configure files (you run this so that the repo stays clean)
./configure              # generates a makefile and checks stuff from the system
make -j$(nproc)          # compiles and links the source code into a `bin/engine` binary using multiple threads

# optionally run a quick smoke-test
make perft
```

You can also install a pre-compiled binary that is provided in the Releases page. (after V2.1)

## Usage

### UCI Mode

```bash
./salmon
```

The engine will accept UCI commands from compatible chess GUIs.

### Basic UCI Commands

- `uci` - Initialize the engine
- `setuservalue name` - Configure engine options
- `position fen <fen>` - Set board position
- `go depth 20` - Search to a specific depth
- `quit` - Exit

## Design Philosophy

### Why No NNUE?

While neural network evaluation (NNUE) is powerful, it creates a "black box" effect—it becomes difficult to understand *why* a move was chosen. SalmonEngine prioritizes **clarity and education**. The evaluation function is straightforward, making it ideal for:

- Learning how chess engines work
- Debugging and experimenting with evaluation changes
- Understanding the fundamentals before diving into advanced techniques

### Name Origin

Like salmon returning to their birthplace to spawn, this engine returns to the foundational principles of chess programming—the era before NNUE dominated.

## Contributing

Contributions are welcome! Whether improving evaluation, optimizing search, or enhancing UCI support, feel free to submit improvements.

## License

See LICENSE file for details.

## Resources

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [Bitboard Chess Engines](https://www.chessprogramming.org/Bitboards)
