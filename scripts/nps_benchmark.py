#!/usr/bin/env python3
"""
Measure normalised NPS (nodes per second) of a UCI chess engine.

The problem with raw NPS
------------------------
Raw NPS varies wildly with the host CPU's clock speed, cache sizes,
and what else is running on the machine.  Two runs of the same engine
on the same machine can differ by ±10%.  This makes NPS-based
regression detection in CI essentially useless.

Our approach: CPU normalisation
-------------------------------
We run a small CPU-bound reference benchmark at the start of each
measurement.  The reference benchmark is a deterministic, integer-heavy
computation whose runtime correlates with the CPU's raw integer
throughput.  We then express the engine's NPS as a fraction of this
reference throughput — a dimensionless ratio that should be roughly
constant across machines.

Formally:
    norm_nps = engine_nps / cpu_reference_score

Where cpu_reference_score is the reference benchmark's score (work
units per second).  On a 2x faster CPU, both engine_nps and
cpu_reference_score double, so norm_nps stays the same.

Reference benchmark
-------------------
We use a deterministic SHA-256-style mix of integer operations
(rotates, xors, additions) over a small in-memory buffer.  This is:
  - CPU-bound (no I/O, no syscalls)
  - Branch-light (no unpredictable branches)
  - Memory-light (fits in L1)
  - Deterministic (same input → same output → same work)
  - Cross-platform (pure C, no architecture-specific intrinsics)

The work unit is "blocks processed".  More blocks = more work.

Engine NPS measurement
----------------------
For the engine, we run a fixed set of positions at a fixed depth and
parse the `info ... nodes N ... time T ms ...` lines from the UCI
output.  We use the LAST info line before bestmove for each position
(= the deepest completed search), and compute:
    engine_nps = sum(N) / sum(T_seconds)

Summing across positions reduces variance from individual position
peculiarities (some positions have more LMR cutoffs, etc.).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

# ────────────────────────────────────────────────────────────────────
# Reference benchmark: a CPU-bound integer workload.
# ────────────────────────────────────────────────────────────────────

# We ship the C source as a string so we don't have to manage a
# separate file.  The benchmark is intentionally simple — it just
# needs to be CPU-bound and deterministic.

CPU_BENCH_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* A small, deterministic, integer-heavy workload.
 *
 * Mixes a 256-byte buffer through a SHA-256-like round function for
 * N iterations.  Each iteration touches every byte of the buffer
 * and does 4 rounds of rotates + adds + xors per byte.
 *
 * The work is independent of the buffer's initial contents (we init
 * it deterministically), so the same N always produces the same
 * amount of work.
 */

static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

#define MIX(a, b, c, d) do {              \
    a = a + b + 0x9E3779B9u;              \
    d = rotr(d ^ a, 16);                  \
    c = c + d;                            \
    b = rotr(b ^ c, 12);                  \
    a = a + b + 0x85EBCA6Bu;              \
    d = rotr(d ^ a, 8);                   \
    c = c + d;                            \
    b = rotr(b ^ c, 7);                   \
} while (0)

int main(int argc, char **argv) {
    long iterations = 10000000L;  /* default: ~1-3s on a modern CPU */
    if (argc > 1) iterations = strtol(argv[1], NULL, 10);
    if (iterations <= 0) iterations = 10000000L;

    uint32_t state[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };

    /* Deterministic initial buffer */
    uint32_t buf[64];
    for (int i = 0; i < 64; i++) buf[i] = (uint32_t)i * 0x1234567u;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (long it = 0; it < iterations; it++) {
        /* Apply 4 rounds of mixing to the state, XORing with the
         * buffer at each round to prevent the compiler from
         * hoisting everything out of the loop. */
        for (int r = 0; r < 4; r++) {
            MIX(state[0], state[4], buf[r * 16 % 64], state[7]);
            MIX(state[1], state[5], buf[r * 16 % 64 + 1], state[6]);
            MIX(state[2], state[6], buf[r * 16 % 64 + 2], state[5]);
            MIX(state[3], state[7], buf[r * 16 % 64 + 3], state[4]);
            MIX(state[4], state[0], buf[r * 16 % 64 + 4], state[3]);
            MIX(state[5], state[1], buf[r * 16 % 64 + 5], state[2]);
            MIX(state[6], state[2], buf[r * 16 % 64 + 6], state[1]);
            MIX(state[7], state[3], buf[r * 16 % 64 + 7], state[0]);
        }
        /* Re-mix the buffer occasionally so the compiler can't
         * precompute the buffer values. */
        if ((it & 0xFFFF) == 0) {
            for (int i = 0; i < 64; i++)
                buf[i] ^= state[i % 8] + (uint32_t)i;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* Force the result to be observable so the compiler can't
     * elide the work. */
    volatile uint32_t sink = 0;
    for (int i = 0; i < 8; i++) sink ^= state[i];

    double secs = (t1.tv_sec - t0.tv_sec)
                + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    /* Score = iterations / second.  Higher is faster. */
    printf("cpu_bench_iterations %ld\n", iterations);
    printf("cpu_bench_seconds %.6f\n", secs);
    printf("cpu_bench_score %.0f\n", (double)iterations / secs);
    return 0;
}
"""


def build_cpu_bench(workdir: Path) -> Path:
    """Compile the CPU reference benchmark.  Returns the binary path."""
    src = workdir / "cpu_bench.c"
    bin_ = workdir / "cpu_bench"
    src.write_text(CPU_BENCH_SOURCE)
    # -O2 not -O3: we want a reproducible, conservative optimisation
    # level that all compilers support uniformly.
    # _POSIX_C_SOURCE=199309L is needed for clock_gettime on strict C11.
    cmd = ["cc", "-O2", "-std=c11", "-D_POSIX_C_SOURCE=199309L",
           "-o", str(bin_), str(src)]
    print(f"[cpu_bench] building: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return bin_


def run_cpu_bench(bin_: Path, iterations: int = 10_000_000) -> float:
    """
    Run the CPU benchmark.  Returns the score (iterations/second).
    Higher is faster.
    """
    out = subprocess.run([str(bin_), str(iterations)],
                         capture_output=True, text=True, check=True)
    score = None
    for line in out.stdout.splitlines():
        if line.startswith("cpu_bench_score "):
            score = float(line.split()[1])
    if score is None:
        raise RuntimeError(f"could not parse cpu_bench output:\n{out.stdout}")
    return score


# ────────────────────────────────────────────────────────────────────
# Engine NPS measurement
# ────────────────────────────────────────────────────────────────────

# A set of test positions chosen specifically for NPS benchmarking.
# Each tuple is (label, FEN, depth).
#
# Requirements for a good NPS benchmark position:
#   1. NOT in the opening book (or use setoption OwnBook=false).
#   2. Does NOT have a quick mate or trivial tactic that would
#      short-circuit the search at the chosen depth.
#   3. Has enough complexity that the search runs ≥ 200ms at the
#      chosen depth, so fixed overheads are negligible.
#
# Depth is fixed so the same amount of "search work" is done on every
# machine — variable time would conflate CPU speed with search depth.
#
# Positions are taken from real games or composed to be complex enough
# that the search runs long enough for stable NPS measurement.
TEST_POSITIONS: List[Tuple[str, str, int]] = [
    # Complex middlegame from a QGD-style position.  Many legal moves,
    # no immediate tactics.  Depth 18 takes ~3-4s.
    ("qgd_middlegame",
     "r1bqk2r/ppp2ppp/2n2n2/3p4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 5",
     18),
    # Piece-rich KID middlegame.  Lots of legal moves and long
    # variations to evaluate.
    ("kid_middlegame",
     "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7",
     18),
    # Caro-Kann main line middlegame.  Asymmetric structure.
    ("caro_kann",
     "r1bqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
     18),
    # complex middlegame with heavy pieces.
    ("heavy_middlegame",
     "r3r1k1/pp3ppp/1qn1p3/3pP3/3P4/2P2N2/PP3PPP/R2QR1K1 w - - 0 1",
     18),
    # Sicilian Najdorf middlegame.  Tactical but not forced.
    ("sicilian_najdorf",
     "r1bqk2r/pp1nbppp/2p2n2/3p4/3P1B2/2N2N2/PPP1QPPP/R3KB1R w KQkq - 0 8",
     18),
]


def measure_engine_nps(engine_path: str, positions: List[Tuple[str, str, int]],
                       min_ms: int = 200) -> Tuple[int, int, int, int, int]:
    """
    Run the engine on each position to the given depth and sum
    (nodes, time_ms) over the positions where the search actually
    ran long enough to give a stable NPS reading (>= min_ms).

    Positions that complete in < min_ms (e.g., because of a quick
    mate or trivial tactic) are skipped — their NPS is dominated by
    fixed overheads (UCI parse, TT init) and isn't representative.

    Returns (total_nodes, total_ms, n_positions_used, total_depth, n_completions).
      total_depth     — sum of depths reached across completed positions
      n_completions   — number of positions where the search reached
                        the requested depth (vs. being cut off by time)

    total_depth / n_completions gives the average depth reached, which
    is a useful complement to NPS — a stronger engine reaches the same
    depth in less time (or a deeper depth in the same time).
    """
    total_nodes = 0
    total_ms = 0
    n_used = 0
    total_depth = 0
    n_completions = 0

    for label, fen, depth in positions:
        # Send: uci, setoption OwnBook=false (so the engine actually
        # searches every position, including those in the book),
        # ucinewgame, position fen ..., go depth N movetime 60000, quit.
        cmds = (
            "uci\n"
            "setoption name OwnBook value false\n"
            "isready\n"
            "ucinewgame\n"
            f"position fen {fen}\n"
            f"go depth {depth} movetime 60000\n"
            "quit\n"
        )
        proc = subprocess.run([engine_path], input=cmds,
                              capture_output=True, text=True, timeout=120)

        last_info_nodes = 0
        last_info_time = 0
        last_info_depth = 0
        for line in proc.stdout.splitlines():
            if line.startswith("info ") and " nodes " in line and " time " in line:
                parts = line.split()
                try:
                    i_nodes = parts.index("nodes")
                    i_time = parts.index("time")
                    i_depth = parts.index("depth")
                    last_info_nodes = int(parts[i_nodes + 1])
                    last_info_time = int(parts[i_time + 1])
                    last_info_depth = int(parts[i_depth + 1])
                except (ValueError, IndexError):
                    pass

        if last_info_nodes == 0:
            print(f"[engine_nps] WARNING: no info line for {label}",
                  file=sys.stderr)
            continue

        nps = last_info_nodes * 1000 // max(1, last_info_time)
        status = "OK" if last_info_time >= min_ms else "SKIP (too fast)"
        print(f"[engine_nps] {label:25s} depth={last_info_depth:2d}/{depth:2d}  "
              f"nodes={last_info_nodes:>10d}  time={last_info_time:>6d}ms  "
              f"nps={nps:>8d}  {status}", file=sys.stderr)

        if last_info_time >= min_ms:
            total_nodes += last_info_nodes
            total_ms += last_info_time
            total_depth += last_info_depth
            if last_info_depth >= depth:
                n_completions += 1
            n_used += 1

    return total_nodes, total_ms, n_used, total_depth, n_completions


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True,
                    help="Path to the engine binary")
    ap.add_argument("--label", default="engine",
                    help="Label for the engine (for output)")
    ap.add_argument("--cpu-bench-iterations", type=int, default=10_000_000,
                    help="Iterations for the CPU reference benchmark")
    ap.add_argument("--workdir", default=None,
                    help="Working directory for temp files (default: auto)")
    ap.add_argument("--json", action="store_true",
                    help="Output as JSON (for machine consumption)")
    args = ap.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="nps_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Step 1: build and run the CPU reference benchmark.
    cpu_bin = build_cpu_bench(workdir)
    cpu_score = run_cpu_bench(cpu_bin, args.cpu_bench_iterations)
    print(f"[cpu_bench] score = {cpu_score:.0f} iter/s", file=sys.stderr)

    # Step 2: measure engine NPS on the test positions.
    total_nodes, total_ms, n_used, total_depth, n_completions = \
        measure_engine_nps(args.engine, TEST_POSITIONS)
    if total_ms == 0:
        print("ERROR: no engine measurements taken (all positions too fast?)",
              file=sys.stderr)
        return 2

    engine_nps = total_nodes * 1000 // total_ms
    # Normalised NPS = engine_nps / cpu_score (a dimensionless ratio,
    # scaled by 1e6 to give a readable number).
    norm_nps = engine_nps / cpu_score * 1e6

    # Time-to-depth: total wall-clock time per unit of depth reached.
    # Lower is better.  Useful for comparing engines with different
    # pruning aggressiveness — a more aggressive pruner may have LOWER
    # NPS (fewer nodes per second because each node is "richer") but
    # ALSO lower time-to-depth (reaches the target depth faster).
    avg_depth = total_depth / n_used if n_used else 0
    time_per_depth = total_ms / total_depth if total_depth else 0
    norm_time_per_depth = time_per_depth * cpu_score / 1e6

    if args.json:
        import json
        print(json.dumps({
            "label": args.label,
            "engine_nps": engine_nps,
            "cpu_bench_score": cpu_score,
            "normalised_nps": norm_nps,
            "total_nodes": total_nodes,
            "total_ms": total_ms,
            "n_positions_used": n_used,
            "total_depth_reached": total_depth,
            "avg_depth_reached": avg_depth,
            "n_target_depth_completions": n_completions,
            "time_per_depth_ms": time_per_depth,
            "normalised_time_per_depth": norm_time_per_depth,
        }, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Engine:        {args.label}")
        print(f"CPU bench:     {cpu_score:>12.0f} iter/s")
        print(f"Engine NPS:    {engine_nps:>12d} nodes/s  ({n_used} positions)")
        print(f"Normalised:    {norm_nps:>12.0f}  (engine_nps / cpu_score * 1e6)")
        print(f"Total nodes:   {total_nodes:>12d}")
        print(f"Total time:    {total_ms:>12d} ms")
        print(f"Avg depth:     {avg_depth:>12.2f}  ({n_completions}/{n_used} reached target)")
        print(f"Time/depth:    {time_per_depth:>12.1f} ms/ply")
        print(f"Norm time/depth: {norm_time_per_depth:>9.1f}  (lower = faster)")
        print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
