#!/usr/bin/env python3
"""
Run a cutechess-cli match between two engines and append the result to a
persistent JSONL history file.

Used by .github/workflows/elo_benchmark.yml to track Elo differences
between a feature branch and the default branch, push by push.

Usage
-----
    python3 scripts/elo_match.py \\
        --engine-a /tmp/engine-branch \\
        --engine-b /tmp/engine-main \\
        --name-a branch --name-b main \\
        --time-control 10+0.1 --rounds 200 \\
        --history elo_history/my-feature.jsonl \\
        --branch my-feature \\
        --commit-sha <40-char-sha> \\
        --commit-message "fix: improve king safety"

History file format
-------------------
JSONL — one JSON object per line.  Each record contains:

    timestamp                     ISO-8601 UTC
    branch                        branch name
    commit_sha                    full 40-char SHA
    commit_message_first_line     first line of the commit message
    engine_a / engine_b           display names
    time_control                  cutechess tc string
    n_games                       total games played
    wins_a / losses_a / draws     W/L/D for engine A
    elo_diff                      A - B, in Elo (positive = A stronger)
    elo_ci_lo / elo_ci_hi         95% confidence interval on elo_diff
    los                           likelihood of superiority (A > B), [0, 1]

To inspect the timeline
-----------------------
    jq -s 'sort_by(.timestamp)' elo_history/my-feature.jsonl \\
    | jq '.[] | {sha: .commit_sha[0:8],
                 msg: .commit_message_first_line,
                 elo: .elo_diff,
                 ci: [.elo_ci_lo, .elo_ci_hi],
                 los: .los}'

Why a separate script?
----------------------
The existing scripts/elo_from_cutechess.py is a *parser* — it takes
cutechess output and computes Elo.  This script is the *driver* — it
invokes cutechess, feeds the output to the parser, and records the
result.  Keeping them separate means the parser can still be used
standalone (e.g. on a results file from a manual match).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# Reuse the existing parser rather than duplicating the Elo math.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_from_cutechess import parse_summary, elo_diff, los  # noqa: E402


def run_cutechess(engine_a: str, engine_b: str,
                  name_a: str, name_b: str,
                  time_control: str, rounds: int,
                  openings: str | None = None) -> str:
    """Invoke cutechess-cli and return combined stdout+stderr.

    cutechess-cli prints the game-by-game results to stdout and the
    final "Score of A vs B: ..." summary line to stdout (or stderr,
    depending on version).  We capture both so the parser can find
    the summary regardless of where cutechess put it.
    """
    cmd = [
        "cutechess-cli",
        "-concurrency", str(os.cpu_count() or 2),
        "-games",       str(rounds),
        "-rounds",      "2",                 # each opening, both colours
        "-variant",     "standard",
        "-engine",      f"name={name_a}", f"cmd={engine_a}",
        "-engine",      f"name={name_b}", f"cmd={engine_b}",
        "-each",        "proto=uci", f"tc={time_control}",
    ]
    if openings:
        # Use a real opening book so the same startpos isn't played every
        # game.  The book should be in EPD format.  cutechess shuffles
        # the openings and pairs each with both colours.
        cmd += ["-openings", f"file={openings}", "format=epd", "order=random"]

    print(f"[elo_match] running: {' '.join(cmd)}", file=sys.stderr)
    # 4-hour hard cap — a stuck game shouldn't burn the whole CI budget.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
    return proc.stdout + "\n" + proc.stderr


def append_history(history_path: str, record: dict) -> None:
    """Append one record to the JSONL history file (creating it + parent
    directories if needed).  We never overwrite — only append — so
    concurrent runs (if any) can't clobber each other's entries."""
    p = Path(history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--engine-a", required=True,
                    help="Path to engine A (typically the branch build)")
    ap.add_argument("--engine-b", required=True,
                    help="Path to engine B (typically the main build)")
    ap.add_argument("--name-a", default="A",
                    help="Display name for engine A")
    ap.add_argument("--name-b", default="B",
                    help="Display name for engine B")
    ap.add_argument("--time-control", default="10+0.1",
                    help="cutechess time control, e.g. 10+0.1 (10s + 0.1s inc)")
    ap.add_argument("--rounds", type=int, default=200,
                    help="Number of rounds (each round = 2 games, colour-swapped)")
    ap.add_argument("--openings", default=None,
                    help="Optional .epd opening book file")
    ap.add_argument("--history", default="elo_history.jsonl",
                    help="JSONL history file to append to")
    ap.add_argument("--branch", required=True,
                    help="Branch name being tested")
    ap.add_argument("--commit-sha", required=True,
                    help="Full 40-char commit SHA")
    ap.add_argument("--commit-message", required=True,
                    help="Commit message (first line is extracted for the record)")
    args = ap.parse_args()

    output = run_cutechess(
        args.engine_a, args.engine_b,
        args.name_a, args.name_b,
        args.time_control, args.rounds,
        args.openings,
    )

    wins, losses, draws = parse_summary(output)
    n = wins + losses + draws
    if n == 0:
        print("ERROR: no games parsed from cutechess output", file=sys.stderr)
        print("--- cutechess stdout+stderr ---", file=sys.stderr)
        print(output, file=sys.stderr)
        return 2

    elo, lo, hi = elo_diff(wins, losses, draws)
    los_val = los(wins, losses)

    # First line of the commit message only — keeps the history file readable.
    first_line = (args.commit_message.splitlines()[0]
                  if args.commit_message else "")

    record = {
        "timestamp":                 datetime.datetime.now(
                                         datetime.timezone.utc
                                     ).isoformat(),
        "branch":                    args.branch,
        "commit_sha":                args.commit_sha,
        "commit_message_first_line": first_line,
        "engine_a":                  args.name_a,
        "engine_b":                  args.name_b,
        "time_control":              args.time_control,
        "n_games":                   n,
        "wins_a":                    wins,
        "losses_a":                  losses,
        "draws":                     draws,
        "elo_diff":                  round(elo, 1),
        "elo_ci_lo":                 round(lo, 1),
        "elo_ci_hi":                 round(hi, 1),
        "los":                       round(los_val, 4),
    }

    append_history(args.history, record)

    # Human-readable summary for the CI log.
    print()
    print("=" * 60)
    print(f"  Branch:   {args.branch}")
    print(f"  Commit:   {args.commit_sha[:8]}  {first_line}")
    print(f"  Games:    {n}  ({args.time_control})")
    print(f"  {args.name_a}: {wins} W / {losses} L / {draws} D")
    print(f"  Elo diff: {elo:+.1f}  (95% CI: {lo:+.1f} .. {hi:+.1f})")
    print(f"  LOS:      {los_val*100:.2f}%")
    print(f"  History:  {args.history}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
