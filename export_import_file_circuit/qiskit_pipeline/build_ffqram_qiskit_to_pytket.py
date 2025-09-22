import argparse, numpy as np
import sys
from qiskit import qpy
from utils_ffqram import(
    FFQRAM,
    generate_normalized_dataset,
    reverse_circuit_qubit_order
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

    K = int(np.ceil(np.log2(num_qubits)))

    # 1) genera dataset normalizzato 2^k x 2^k con seed diverso
    dataset = generate_normalized_dataset(pow(2, K), pow(2, K), seed=args.seed)

    
    # 1) FF-QRAM
    qc_orig = FFQRAM(dataset)

    # 2) reverse globale
    qc_rev = reverse_circuit_qubit_order(qc_orig)

    # 3) decompose profondo
    qc_rev_dec = qc_rev.decompose(reps=10)

        

    with open(args.out_qpy, "wb") as f:
        qpy.dump(qc_rev_dec, f)

if __name__ == "__main__":
    main()