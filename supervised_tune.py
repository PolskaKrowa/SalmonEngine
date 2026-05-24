#!/usr/bin/env python3
"""
supervised_tune.py — Python driver for supervised NNUE fine-tuning.

Given a CSV file of chess positions with Stockfish best moves, this script:
  1. Validates the CSV (checks required columns, reports dataset statistics)
  2. Optionally builds the `suptuna` binary if it is not already compiled
  3. Invokes `suptuna` and streams its progress in real time
  4. Plots training loss curves with matplotlib
  5. Optionally benchmarks move accuracy before and after training

Usage:
  python supervised_tune.py --csv tests/mate_positions.csv --engine ./src/engine
  python supervised_tune.py --csv positions.csv --epochs 100 --lr 5e-4 --plot

Full argument list:  python supervised_tune.py --help
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# ── Optional dependencies ──────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" if you want a window
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ══════════════════════════════════════════════════════════════════════════════
# §1  CSV validation
# ══════════════════════════════════════════════════════════════════════════════

# Basic sanity-check pattern for FEN strings
_FEN_RE = re.compile(
    r"[rnbqkpRNBQKP1-8]+(/[rnbqkpRNBQKP1-8]+){7}"   # piece placement (8 ranks)
    r"\s+[wb]"                                          # side to move
)

# UCI move: two squares + optional promotion letter
_UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrqNBRQ]?$")


def validate_csv(csv_path: str) -> tuple[list[dict], list[str]]:
    """
    Load and validate the CSV.

    Returns (rows, warnings) where `rows` is a list of dicts with at least
    the keys 'fen' and 'best_move', and `warnings` is a list of human-readable
    issues found (non-fatal).

    Raises SystemExit on fatal errors (missing columns, empty file, etc.)
    """
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"[validate_csv] File not found: {csv_path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader(text.splitlines()))

    if not rows:
        sys.exit(f"[validate_csv] CSV is empty: {csv_path}")

    cols = set(rows[0].keys())
    missing = {"fen", "best_move"} - cols
    if missing:
        sys.exit(
            f"[validate_csv] Required column(s) missing: {missing}\n"
            f"  Found columns: {sorted(cols)}"
        )

    warnings: list[str] = []
    bad_fen = bad_move = dup = 0
    seen_fens: set[str] = set()

    for i, row in enumerate(rows, start=2):           # row 1 = header
        fen  = row["fen"].strip()
        move = row["best_move"].strip()

        if not _FEN_RE.search(fen):
            bad_fen += 1
            if bad_fen <= 3:
                warnings.append(f"  Row {i}: suspicious FEN — '{fen[:60]}'")

        if not _UCI_MOVE_RE.match(move):
            bad_move += 1
            if bad_move <= 3:
                warnings.append(f"  Row {i}: invalid UCI move — '{move}'")

        if fen in seen_fens:
            dup += 1
        else:
            seen_fens.add(fen)

    if bad_fen  > 3:  warnings.append(f"  … and {bad_fen  - 3} more suspicious FENs")
    if bad_move > 3:  warnings.append(f"  … and {bad_move - 3} more invalid UCI moves")
    if dup > 0:       warnings.append(f"  {dup} duplicate FEN(s) detected")

    return rows, warnings


def print_dataset_stats(rows: list[dict], warnings: list[str]) -> None:
    """Print a compact summary of the dataset to stderr."""
    print(f"\n[Dataset] {len(rows)} examples loaded", file=sys.stderr)

    # Count unique side-to-move (extracted from FEN field 2)
    whites = blacks = unknown = 0
    for row in rows:
        parts = row["fen"].strip().split()
        if len(parts) >= 2:
            if parts[1] == "w": whites += 1
            elif parts[1] == "b": blacks += 1
            else: unknown += 1

    print(f"[Dataset] Side to move: {whites} white, {blacks} black"
          + (f", {unknown} unknown" if unknown else ""),
          file=sys.stderr)

    # Promotion move count
    promotions = sum(1 for r in rows if len(r["best_move"].strip()) == 5)
    if promotions:
        print(f"[Dataset] {promotions} promotion move(s)", file=sys.stderr)

    if warnings:
        print(f"\n[Dataset] Warnings ({len(warnings)}):", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)
    else:
        print("[Dataset] All rows passed basic validation.", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# §2  Build the suptuna binary
# ══════════════════════════════════════════════════════════════════════════════

def find_suptuna(build_dir: Optional[str] = None) -> Optional[str]:
    """Return the path to suptuna if it is already compiled, else None."""
    candidates = []
    if build_dir:
        candidates.append(str(Path(build_dir) / "suptuna"))
    candidates += [
        "./src/suptuna",
        "src/suptuna",
        "./suptuna",
    ]
    # Also search PATH
    found = shutil.which("suptuna")
    if found:
        candidates.append(found)

    for c in candidates:
        if Path(c).is_file() and Path(c).stat().st_mode & 0o111:
            return c
    return None


def build_suptuna(src_dir: str = "src") -> str:
    """
    Attempt to build suptuna by running `make suptuna` in `src_dir`.

    Returns the path to the compiled binary, or raises RuntimeError.
    """
    print("[Build] Running `make suptuna` ...", file=sys.stderr)
    result = subprocess.run(
        ["make", "suptuna"],
        cwd=src_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "make suptuna failed.  "
            "Check that --enable-tuner was passed to ./configure "
            "and that supervised_tune.c is in the src/ directory."
        )
    binary = str(Path(src_dir) / "suptuna")
    if not Path(binary).exists():
        raise RuntimeError(f"make succeeded but binary not found at: {binary}")
    print(f"[Build] Built {binary}", file=sys.stderr)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# §3  Run training and stream progress
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    binary: str,
    csv_path: str,
    ckpt_in:   Optional[str],
    ckpt_out:  str,
    epochs:    int,
    batch:     int,
    lr:        float,
    lr_min:    float,
    n_neg:     int,
    margin:    float,
    threads:   int,
    seed:      int,
    log_every: int,
) -> tuple[list[float], list[float]]:
    """
    Invoke the suptuna binary and stream its output.

    Parses PROGRESS and EPOCH_DONE lines emitted by supervised_tune.c §10.

    Returns (step_losses, epoch_losses) — two lists suitable for plotting.
    """
    cmd = [
        binary,
        "--csv",       csv_path,
        "--ckpt-out",  ckpt_out,
        "--epochs",    str(epochs),
        "--batch",     str(batch),
        "--lr",        str(lr),
        "--lr-min",    str(lr_min),
        "--n-neg",     str(n_neg),
        "--margin",    str(margin),
        "--threads",   str(threads),
        "--seed",      str(seed),
        "--log-every", str(log_every),
    ]
    if ckpt_in:
        cmd += ["--ckpt-in", ckpt_in]

    print(f"\n[Training] Command: {' '.join(cmd)}", file=sys.stderr)
    print(f"[Training] Checkpoint output: {ckpt_out}\n", file=sys.stderr)

    step_losses:  list[float] = []
    epoch_losses: list[float] = []

    t_start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,          # let stderr (human-readable) pass through
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            # ── PROGRESS step=N epoch=E loss=F lr=F ──────────────────────
            if line.startswith("PROGRESS"):
                kv = _parse_kv(line)
                step = int(kv.get("step", 0))
                loss = float(kv.get("loss", 0.0))
                ep   = int(kv.get("epoch", 0))
                lr_  = float(kv.get("lr", 0.0))
                step_losses.append(loss)
                print(
                    f"  step {step:6d}  epoch {ep:3d}  "
                    f"loss {loss:.6f}  lr {lr_:.3e}",
                    end="\r",
                    flush=True,
                )

            # ── EPOCH_DONE epoch=E avg_loss=F ────────────────────────────
            elif line.startswith("EPOCH_DONE"):
                kv = int_kv = _parse_kv(line)
                ep       = int(kv.get("epoch", 0))
                avg_loss = float(kv.get("avg_loss", 0.0))
                epoch_losses.append(avg_loss)
                elapsed = time.time() - t_start
                print(
                    f"\n  ✓ Epoch {ep:3d}/{epochs}  "
                    f"avg_loss={avg_loss:.6f}  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            # ── DONE total_steps=N final_loss=F ─────────────────────────
            elif line.startswith("DONE"):
                kv         = _parse_kv(line)
                total_steps = int(kv.get("total_steps", 0))
                final_loss  = float(kv.get("final_loss", 0.0))
                elapsed     = time.time() - t_start
                print(
                    f"\n[Training] Complete — {total_steps} steps  "
                    f"final_loss={final_loss:.6f}  "
                    f"wall time={elapsed:.1f}s"
                )

        proc.wait()

    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user — checkpoint may be incomplete.",
              file=sys.stderr)
        if "proc" in dir() and proc.poll() is None:
            proc.terminate()

    return step_losses, epoch_losses


def _parse_kv(line: str) -> dict[str, str]:
    """Parse "KEY token=value token=value …" into a dict of str→str."""
    result: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            k, _, v = token.partition("=")
            result[k] = v
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §4  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_losses(
    step_losses:  list[float],
    epoch_losses: list[float],
    out_path:     str = "training_loss.png",
    log_every:    int = 50,
) -> None:
    """Save a two-panel loss plot (per-step and per-epoch) to out_path."""
    if not HAS_MATPLOTLIB:
        print("[Plot] matplotlib not installed — skipping plot.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Supervised NNUE Fine-Tuning — Training Loss", fontsize=13)

    # ── Left panel: per-step loss ──────────────────────────────────────────
    ax = axes[0]
    if step_losses:
        xs = [i * log_every for i in range(len(step_losses))]
        ax.plot(xs, step_losses, linewidth=0.8, color="#2980b9", alpha=0.85)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Margin ranking loss")
        ax.set_title("Per-step loss")
        ax.grid(True, alpha=0.3)
        # Smooth overlay
        if len(step_losses) > 20:
            window = max(1, len(step_losses) // 20)
            smoothed = _moving_average(step_losses, window)
            xs_s = [i * log_every for i in range(len(smoothed))]
            ax.plot(xs_s, smoothed, linewidth=2.0, color="#e74c3c",
                    label=f"MA({window})", alpha=0.9)
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No step data", ha="center", va="center",
                transform=ax.transAxes)

    # ── Right panel: per-epoch average loss ────────────────────────────────
    ax = axes[1]
    if epoch_losses:
        ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
                marker="o", markersize=4, linewidth=1.5, color="#27ae60")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean loss")
        ax.set_title("Per-epoch average loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No epoch data", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved loss curve → {out_path}")


def _moving_average(data: list[float], window: int) -> list[float]:
    result = []
    acc = 0.0
    for i, v in enumerate(data):
        acc += v
        if i >= window:
            acc -= data[i - window]
        result.append(acc / min(i + 1, window))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §5  Move accuracy benchmark  (before / after comparison)
# ══════════════════════════════════════════════════════════════════════════════

def query_engine(engine_path: str, fen: str, depth: int = 14) -> str:
    """
    Ask the engine for its best move at the given FEN and depth.
    Returns the best move in UCI format, or "" on failure.

    Replicates the logic from test_mate.py so we can run the evaluation
    directly in this script without requiring that file to be importable.
    """
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        def send(cmd: str) -> None:
            assert proc.stdin
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

        def read_until(token: str) -> list[str]:
            lines = []
            assert proc.stdout
            for line in proc.stdout:
                line = line.rstrip()
                lines.append(line)
                if line.startswith(token):
                    break
            return lines

        send("uci")
        read_until("uciok")
        send("ucinewgame")
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        lines = read_until("bestmove")
        proc.stdin.close()
        proc.stdout.close()
        proc.wait()

        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else ""
    except Exception as exc:
        print(f"[Bench] Engine query failed: {exc}", file=sys.stderr)

    return ""


def benchmark_accuracy(
    engine_path: str,
    rows:        list[dict],
    label:       str = "Benchmark",
    depth:       int = 14,
    max_pos:     int = 0,
) -> float:
    """
    Run the engine on up to max_pos positions and return the fraction correct.
    If max_pos == 0, all rows are used.
    """
    sample = rows if max_pos <= 0 else rows[:max_pos]
    passed = failed = 0

    print(f"\n[{label}] Evaluating {len(sample)} position(s) at depth {depth} …")

    for row in sample:
        fen      = row["fen"].strip()
        expected = row["best_move"].strip()
        actual   = query_engine(engine_path, fen, depth)
        if actual == expected:
            passed += 1
        else:
            failed += 1

    total    = passed + failed
    accuracy = passed / total if total > 0 else 0.0
    print(f"[{label}] {passed}/{total} correct  ({accuracy*100:.1f}%)")
    return accuracy


# ══════════════════════════════════════════════════════════════════════════════
# §6  Makefile.am snippet helper
# ══════════════════════════════════════════════════════════════════════════════

MAKEFILE_PATCH = textwrap.dedent("""\
    # ── Supervised fine-tuning binary (suptuna) ───────────────────────────
    # Add these lines to your src/Makefile.am inside the `if ENABLE_TUNER`
    # block, right after the tuna target:
    #
    # if ENABLE_TUNER
    # ...existing tuna target...
    #
    bin_PROGRAMS += suptuna
    suptuna_SOURCES = $(ENGINE_SOURCES) supervised_tune.c
    suptuna_CFLAGS  = $(AM_CFLAGS) -DSUPERVISED_STANDALONE
    suptuna_LDADD   = -lm $(OPENBLAS_LIBS) -lpthread
    # endif
""")


# ══════════════════════════════════════════════════════════════════════════════
# §7  Argument parsing and main
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O")
    io.add_argument("--csv",       required=True,
                    help="CSV file with 'fen' and 'best_move' columns")
    io.add_argument("--ckpt-in",   default=None,
                    help="Load initial NNUE weights from this .bin file")
    io.add_argument("--ckpt-out",  default="nnue_sup.bin",
                    help="Save trained weights here (default: nnue_sup.bin)")
    io.add_argument("--binary",    default=None,
                    help="Path to compiled suptuna binary (auto-detected if omitted)")
    io.add_argument("--build-dir", default=None,
                    help="Directory to search for / build suptuna")
    io.add_argument("--src-dir",   default="src",
                    help="Source directory for `make suptuna` (default: src)")

    # ── Training hyper-parameters ──────────────────────────────────────────
    hp = p.add_argument_group("Hyper-parameters")
    hp.add_argument("--epochs",    type=int,   default=50,
                    help="Training epochs (default: 50)")
    hp.add_argument("--batch",     type=int,   default=64,
                    help="Minibatch size (default: 64)")
    hp.add_argument("--lr",        type=float, default=1e-3,
                    help="Peak Adam learning rate (default: 1e-3)")
    hp.add_argument("--lr-min",    type=float, default=1e-5,
                    help="Cosine-schedule floor LR (default: 1e-5)")
    hp.add_argument("--n-neg",     type=int,   default=7,
                    help="Random alternative moves per example (default: 7)")
    hp.add_argument("--margin",    type=float, default=0.05,
                    help="Hinge margin in raw NNUE units (default: 0.05 ≈ 20 cp)")
    hp.add_argument("--threads",   type=int,   default=1,
                    help="Gradient computation threads (default: 1)")
    hp.add_argument("--seed",      type=int,   default=42,
                    help="PRNG seed (default: 42)")
    hp.add_argument("--log-every", type=int,   default=50,
                    help="Print a PROGRESS line every N steps (default: 50)")

    # ── Evaluation ─────────────────────────────────────────────────────────
    ev = p.add_argument_group("Evaluation")
    ev.add_argument("--engine",    default=None,
                    help="Path to the UCI engine binary for before/after accuracy test")
    ev.add_argument("--depth",     type=int,   default=14,
                    help="Search depth for accuracy benchmark (default: 14)")
    ev.add_argument("--bench-n",   type=int,   default=0,
                    help="Positions to benchmark (0 = all, default: 0)")

    # ── Misc ───────────────────────────────────────────────────────────────
    misc = p.add_argument_group("Misc")
    misc.add_argument("--plot",    action="store_true",
                      help="Save a training-loss PNG (requires matplotlib)")
    misc.add_argument("--plot-out", default="training_loss.png",
                      help="Output path for the loss plot (default: training_loss.png)")
    misc.add_argument("--print-makefile-patch", action="store_true",
                      help="Print the Makefile.am snippet to add suptuna, then exit")
    misc.add_argument("--validate-only", action="store_true",
                      help="Validate the CSV and exit without training")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Print Makefile patch and exit ──────────────────────────────────────
    if args.print_makefile_patch:
        print(MAKEFILE_PATCH)
        return

    # ── Validate dataset ───────────────────────────────────────────────────
    print(f"[Step 1/5] Validating CSV: {args.csv}", file=sys.stderr)
    rows, warnings = validate_csv(args.csv)
    print_dataset_stats(rows, warnings)

    if args.validate_only:
        print("\n--validate-only specified; stopping after validation.")
        return

    # ── Locate / build binary ─────────────────────────────────────────────
    print("\n[Step 2/5] Locating suptuna binary …", file=sys.stderr)

    binary = args.binary
    if binary is None:
        binary = find_suptuna(args.build_dir)

    if binary is None:
        print("[Build] suptuna not found; attempting to build …", file=sys.stderr)
        try:
            binary = build_suptuna(args.src_dir)
        except RuntimeError as exc:
            sys.exit(
                f"[Build] Could not build suptuna:\n  {exc}\n\n"
                "To add suptuna to the build, run:\n"
                "  python supervised_tune.py --print-makefile-patch\n"
                "Then add the printed lines to src/Makefile.am and rebuild."
            )
    else:
        print(f"[Build] Found binary: {binary}", file=sys.stderr)

    # ── Optional: pre-training accuracy benchmark ─────────────────────────
    pre_accuracy: Optional[float] = None
    if args.engine:
        print("\n[Step 3/5] Pre-training accuracy benchmark …", file=sys.stderr)
        pre_accuracy = benchmark_accuracy(
            args.engine, rows, label="Pre-training",
            depth=args.depth, max_pos=args.bench_n,
        )
    else:
        print("\n[Step 3/5] Skipping pre-training benchmark (no --engine)", file=sys.stderr)

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[Step 4/5] Starting training …", file=sys.stderr)
    step_losses, epoch_losses = run_training(
        binary    = binary,
        csv_path  = args.csv,
        ckpt_in   = args.ckpt_in,
        ckpt_out  = args.ckpt_out,
        epochs    = args.epochs,
        batch     = args.batch,
        lr        = args.lr,
        lr_min    = args.lr_min,
        n_neg     = args.n_neg,
        margin    = args.margin,
        threads   = args.threads,
        seed      = args.seed,
        log_every = args.log_every,
    )

    # ── Post-training accuracy benchmark ──────────────────────────────────
    post_accuracy: Optional[float] = None
    if args.engine:
        print("\n[Step 5/5] Post-training accuracy benchmark …", file=sys.stderr)
        post_accuracy = benchmark_accuracy(
            args.engine, rows, label="Post-training",
            depth=args.depth, max_pos=args.bench_n,
        )
        if pre_accuracy is not None and post_accuracy is not None:
            delta = post_accuracy - pre_accuracy
            sign  = "+" if delta >= 0 else ""
            print(
                f"\n[Summary] Accuracy: {pre_accuracy*100:.1f}% → "
                f"{post_accuracy*100:.1f}%  ({sign}{delta*100:.1f}pp)"
            )
    else:
        print("\n[Step 5/5] Skipping post-training benchmark (no --engine)",
              file=sys.stderr)

    # ── Loss plot ──────────────────────────────────────────────────────────
    if args.plot:
        if HAS_MATPLOTLIB:
            plot_losses(step_losses, epoch_losses,
                        out_path=args.plot_out,
                        log_every=args.log_every)
        else:
            print(
                "\n[Plot] matplotlib not installed.  Install it with:\n"
                "  pip install matplotlib",
                file=sys.stderr,
            )

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n[Done] Trained weights saved to: {args.ckpt_out}")
    if not HAS_MATPLOTLIB and args.plot:
        pass
    elif args.plot:
        print(f"[Done] Loss curve saved to:    {args.plot_out}")
    print(
        "\nLoad the weights in your engine with:\n"
        f"  nnue_load(g_nnue, \"{args.ckpt_out}\");"
    )


if __name__ == "__main__":
    main()