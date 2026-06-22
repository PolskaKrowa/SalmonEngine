#!/usr/bin/env python3
"""
Compute Elo difference between two engines from cutechess-cli output.

Reads a cutechess-cli PGN-or-results summary from stdin (or a file) and
prints the Elo difference (engine A - engine B) with a 95% confidence
interval, plus the LOS (likelihood of superiority).

cutechess-cli with `-rounds N` prints a final summary line like:
    Score of A vs B: W + L - D ... (W+D/2)/total  -> Elo diff

This script parses that line, or, if the input is a sequence of game
results (one per line, "1-0", "0-1", "1/2-1/2"), computes the stats
directly.

The Elo formula used is the standard logistic:
    Elo_diff = -400 * log10( L / W )     (draws split evenly)
            = -400 * log10( (L + 0.5*D) / (W + 0.5*D) )

Confidence interval is computed via the binomial proportion standard
error on the win-rate (draws counted as half-wins).
"""

import argparse
import math
import re
import sys
from typing import Tuple


def parse_summary(text: str) -> Tuple[int, int, int]:
    """Extract (wins_A, losses_A, draws) from cutechess output."""
    # Try the summary line first: "Score of A vs B: 123 - 89 - 38"
    m = re.search(r"Score of\b.*?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Fall back to scanning game results: "1-0", "0-1", "1/2-1/2"
    wins = losses = draws = 0
    for line in text.splitlines():
        line = line.strip()
        if line == "1-0":
            wins += 1
        elif line == "0-1":
            losses += 1
        elif line in ("1/2-1/2", "½-½", "draw"):
            draws += 1
    return wins, losses, draws


def elo_diff(wins: int, losses: int, draws: int) -> Tuple[float, float, float]:
    """
    Returns (elo_diff, elo_diff_minus_2se, elo_diff_plus_2se) where the
    interval is a 95% CI based on the binomial standard error.

    Convention: positive elo_diff means engine A is stronger.
    """
    n = wins + losses + draws
    if n == 0:
        return 0.0, 0.0, 0.0

    # Use the standard "draws count as half-wins" scoring.
    score_a = wins + 0.5 * draws
    score_b = losses + 0.5 * draws

    # Elo difference via logistic formula.
    if score_a == 0:
        elo = -800.0
    elif score_b == 0:
        elo = 800.0
    else:
        elo = -400.0 * math.log10(score_b / score_a)

    # 95% CI: standard error of the win-rate.
    p = score_a / n
    se_p = math.sqrt(p * (1.0 - p) / n)
    # Convert proportion SE to Elo SE via the derivative of the logistic.
    # d(Elo)/dp at p=0.5 is 400 / (ln(10) * p * (1-p)) ≈ 173.7 at p=0.5.
    # More generally: dElo/dp = 400 / (ln(10) * p * (1-p))
    if 0.0 < p < 1.0:
        delo_dp = 400.0 / (math.log(10.0) * p * (1.0 - p))
        se_elo = delo_dp * se_p
    else:
        se_elo = 200.0  # rough fallback

    return elo, elo - 2.0 * se_elo, elo + 2.0 * se_elo


def los(wins: int, losses: int) -> float:
    """Likelihood of superiority (draws excluded, two-tailed)."""
    if wins + losses == 0:
        return 0.5
    # LOS via the regularised incomplete beta function — but for large
    # samples the normal approximation is fine and avoids a scipy dep.
    n = wins + losses
    p = wins / n
    se = math.sqrt(p * (1.0 - p) / n) if n > 0 else 1.0
    if se == 0:
        return 1.0 if wins > losses else 0.0
    z = (p - 0.5) / se * math.sqrt(n) / math.sqrt(n)  # = (p-0.5)/se
    # LOS = Phi(z * sqrt(n)) where z is the per-game z-score.
    # Equivalent: LOS = 0.5 * (1 + erf((wins - losses) / sqrt(2*(wins+losses))))
    z2 = (wins - losses) / math.sqrt(2.0 * (wins + losses))
    return 0.5 * (1.0 + math.erf(z2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", default="-",
                    help="cutechess output file (default: stdin)")
    ap.add_argument("--name-a", default="A", help="Name of engine A")
    ap.add_argument("--name-b", default="B", help="Name of engine B")
    args = ap.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input) as f:
            text = f.read()

    w, l, d = parse_summary(text)
    n = w + l + d
    if n == 0:
        print("ERROR: no games parsed from input", file=sys.stderr)
        return 2

    elo, lo, hi = elo_diff(w, l, d)
    los_val = los(w, l)

    print(f"Games:      {n}")
    print(f"  {args.name_a}: {w} wins")
    print(f"  {args.name_b}: {l} wins")
    print(f"  draws:      {d}")
    print(f"Win rate:    {(w + 0.5*d)/n*100:.1f}% for {args.name_a}")
    print(f"Elo diff:   {elo:+.1f}  (95% CI: {lo:+.1f} .. {hi:+.1f})")
    print(f"LOS:        {los_val*100:.2f}%  ({args.name_a} is stronger)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
