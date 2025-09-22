#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from typing import Sequence, Tuple, List, Union

# --------------------------------------------------------------------
# JOBS: tuple a 3 o 4 elementi
#   (name, adj, results)                         -> scelta=2
#   (name, adj, results, circuits)               -> scelta=1
# Mantengo i path così come li hai indicati (duplicati inclusi).
# --------------------------------------------------------------------
JOBS: List[Union[Tuple[str, str, str], Tuple[str, str, str, str]]] = [
    ("adj_20_penguinv4", "../Adj/adj_20_penguinv4", "./tket_results/20CNOT/", "../Input/20CNOT/"),
    ("adj_9_line", "../Adj/adj_9_line.txt", "./tket_results/FFQRAM/"),
    ("adj_9_cycle", "../Adj/adj_9_cycle.txt", "./tket_results/FFQRAM/"),
    ("adj_7_star_center3", "../Adj/adj_7_star_center3.txt", "./tket_results/FFQRAM/"),
    ("adj_7_line", "../Adj/adj_7_line", "./tket_results/FFQRAM/"),
    ("adj_7_cycle", "../Adj/adj_7_cycle", "./tket_results/FFQRAM/"),
]

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
def run_job(
    *,
    python_exe: str,
    program: Path,
    adj: str,
    results: str,
    circuits: str | None,
) -> int:
    results_dir = Path(results)
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [python_exe, str(program), adj, str(results_dir)]

    # scelta: 1 se c'è il path circuiti, altrimenti 2
    if circuits:
        user_input = f"1\n{circuits}\n"
    else:
        user_input = "2\n"

    print("\n" + "=" * 72)
    print(f"RUN → {adj} -> {results_dir}")
    print(f"stdin → {user_input!r}")
    print("=" * 72)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(user_input)
    proc.stdin.flush()
    proc.stdin.close()

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    rc = proc.wait()
    print(f"\n[Return code: {rc}]\n")
    return rc


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lancia job per main_tket.py con scelta=1 se presente cartella input, altrimenti scelta=2."
    )
    p.add_argument("--python-exe", default=sys.executable, help="Interprete Python da usare")
    p.add_argument("--program", default="./main_tket.py", help="Path allo script da eseguire")
    p.add_argument("--dry-run", action="store_true", help="Non eseguire, stampa soltanto")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    program = Path(args.program).resolve()
    if not program.exists():
        print(f"ERRORE: programma non trovato: {program}")
        return 2

    summary: List[Tuple[str, int, float]] = []

    for job in JOBS:
        # unpack robusto per 3 o 4 elementi
        if len(job) == 4:
            name, adj, results, circuits = job
        elif len(job) == 3:
            name, adj, results = job
            circuits = None
        else:
            print(f"ERRORE: job con arità non supportata: {job}")
            return 2

        start = datetime.now()
        if args.dry_run:
            scelta = 1 if circuits else 2
            extra = f" '1\\n{circuits}\\n'" if circuits else " '2\\n'"
            print(f"[DRY-RUN] {name}: {program} {adj} {results} <stdin:{extra}> (scelta={scelta})")
            rc = 0
        else:
            rc = run_job(
                python_exe=args.python_exe,
                program=program,
                adj=adj,
                results=results,
                circuits=circuits,
            )

        elapsed = (datetime.now() - start).total_seconds()
        summary.append((name, rc, elapsed))

    print("\n===== RIEPILOGO =====")
    for name, rc, elapsed in summary:
        print(f"{name:20s} rc={rc} durata≈{int(elapsed)}s")
    return 0 if all(rc == 0 for _, rc, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
