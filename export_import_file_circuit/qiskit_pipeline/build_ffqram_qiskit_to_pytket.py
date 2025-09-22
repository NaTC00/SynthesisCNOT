import argparse, numpy as np
from pathlib import Path
import sys
from qiskit import qpy
from utils_ffqram import(
    FFQRAM,
    generate_normalized_dataset,
    reverse_circuit_qubit_order
)

from utils_quiskit import(
     to_clifford_t_if_needed
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, required=True, help="Numero totale di qubit disponibili (p.es. coupling_map.size())")
    ap.add_argument("--out_qpy", required=True, help="Percorso file QPY di output")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

     # --- sanity check ---
    if args.n_qubits is None or args.n_qubits < 1:
        print("ERRORE: --n_qubits deve essere >= 1", file=sys.stderr)
        sys.exit(2)

    num_qubits = args.n_qubits

    if num_qubits < 3 or num_qubits % 2 == 0:
            raise ValueError("I qubit totali devono essere dispari e ≥ 3 (2k+1).")
    K = (num_qubits - 1) // 2

    # 1) genera dataset normalizzato 2^k x 2^k con seed diverso
    dataset = generate_normalized_dataset(pow(2, K), pow(2, K), seed=args.seed)

    
    # 1) FF-QRAM
    qc_orig = FFQRAM(dataset)

    # 2) reverse globale
    qc_rev = reverse_circuit_qubit_order(qc_orig)

    # 3) decompose profondo
    qc_rev_dec = qc_rev.decompose(reps=10)

    qc_t = to_clifford_t_if_needed(qc_rev_dec)

    Path(args.out_qpy).parent.mkdir(parents=True, exist_ok=True)    

    with open(args.out_qpy, "wb") as f:
        qpy.dump(qc_t, f)

if __name__ == "__main__":
    main()