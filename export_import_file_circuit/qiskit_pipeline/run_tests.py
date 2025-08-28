#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esegue automaticamente `main_qiskit.py` per le 6 configurazioni indicate.

Per ogni job passa **2 argomenti**
    <matrice_adiacenza.txt> <cartella_results>

e risponde alle `input()` del programma con:
    scelta "1" (Circuiti da file) + path cartella circuiti.

Uso:
    python run_tests_scelta1.py --program ./main_qiskit.py

Opzionali:
    --python-exe python3   # interprete da usare (default: quello corrente)
    --dry-run              # mostra i comandi senza eseguirli

Log:
    Salva l'output di ciascun job in <cartella_results>/run_<nome_adj>.log
    e crea un marker <cartella_results>/__DONE__ se il job termina con rc=0.
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


JOBS = [
    ("16_square",     "../Adj/adj_16_square.txt",   "./qiskit_results/16CNOT/", "../Input/16CNOT/"),
    ("16_Rigetti",    "../Adj/adj_16_Rigetti.txt",  "./qiskit_results/16CNOT/", "../Input/16CNOT/"),
    ("16_IBMQX5",     "../Adj/adj_16_IBMQX5.txt",   "./qiskit_results/16CNOT/", "../Input/16CNOT/"),
    ("20_IBMQX20",    "../Adj/adj_20_IBMQX20.txt",  "./qiskit_results/20CNOT/", "../Input/20CNOT/"),
    ("20_penguinv4",  "../Adj/adj_20_penguinv4.txt","./qiskit_results/20CNOT/", "../Input/20CNOT/"),
    ("20_penguinv3",  "../Adj/adj_20_penguinv3.txt","./qiskit_results/20CNOT/", "../Input/20CNOT/"),
]

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------



def run_job(*, python_exe: str, program: Path, adj: str, results: str, circuits: str) -> int:
    results_dir = Path(results)
    results_dir.mkdir(parents=True, exist_ok=True)  

    cmd = [python_exe, str(program), adj, str(results_dir)]
    user_input = f"1\n{circuits}\n"  

    print("\n" + "="*72)
    print(f"RUN  → {adj} -> {results_dir}")
    print(f"stdin → {user_input!r}")
    print("="*72)

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
    p = argparse.ArgumentParser(description="Lancia 6 job con scelta=1 e cartelle input specificate")
    p.add_argument("--python-exe", default=sys.executable, help="Interprete Python da usare")
    p.add_argument("--program", default="./main_qiskit.py", help="Path allo script da eseguire")
    p.add_argument("--dry-run", action="store_true", help="Non eseguire, stampa soltanto")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    program = Path(args.program).resolve()
    if not program.exists():
        print(f"ERRORE: programma non trovato: {program}")
        return 2

    summary = []
    for name, adj, results, circuits in JOBS:
        start = datetime.now()
        if args.dry_run:
            print(f"[DRY-RUN] {name}: {program} {adj} {results}  <stdin: '1\\n{circuits}\\n'>")
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
        print(f"{name:14s}  rc={rc}  durata≈{int(elapsed)}s")

    
    return 0 if all(rc == 0 for _, rc, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
