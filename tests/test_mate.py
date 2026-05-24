import csv
import subprocess
import sys
from pathlib import Path

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def query_engine(engine_path: str, fen: str, movetime_ms: int = 2000) -> str:
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def send(cmd: str):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(token: str) -> list[str]:
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            if line.startswith(token):
                break  # <-- stop reading immediately, don't wait for more
        return lines

    send("uci")
    read_until("uciok")
    send("ucinewgame")
    send(f"position fen {fen}")
    send(f"go movetime {movetime_ms}")
    lines = read_until("bestmove")
    proc.stdin.close()
    proc.stdout.close()
    proc.wait()

    for line in lines:
        if line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else ""
    return ""


def run(engine_path: str, positions_csv: str, movetime_ms: int) -> bool:
    rows = list(csv.DictReader(Path(positions_csv).read_text().splitlines()))
    passed = failed = 0

    for row in rows:
        fen      = row["fen"]
        expected = row["best_move"].strip()
        label    = row.get("label", fen)
        actual   = query_engine(engine_path, fen, movetime_ms)

        if actual == expected:
            print(f"{PASS}  {label}  (played {actual})")
            passed += 1
        else:
            print(f"{FAIL}  {label}  expected={expected} got={actual}")
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return failed == 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engine",    required=True)
    p.add_argument("--positions", default="tests/mate_positions.csv")
    p.add_argument("--movetime",  type=int, default=2000)
    args = p.parse_args()

    sys.exit(0 if run(args.engine, args.positions, args.movetime) else 1)