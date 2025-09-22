from locale import normalize
import os
from pathlib import Path
import subprocess
import sys
from typing import List
import numpy as np
from qiskit import qpy
from pytket.circuit import Circuit, OpType, Qubit, Op, QControlBox
from sklearn.preprocessing import normalize
from pytket.passes import DecomposeBoxes, DecomposeMultiQubitsCX

from pytket.extensions.qiskit import qiskit_to_tk

def ffqram_qubits_required(N, M):
    return np.ceil(np.log2(N)) + np.ceil(np.log2(M)) + 1

def adjust_dataset_for_hw(num_qubits):
    """
    Genera un dataset randomico N x M compatibile con la coupling map.
    Sceglie N,M come potenze di 2 massime che rispettano i vincoli.
    """
    Q_hw = num_qubits  # qubit fisici
    best = None

    # cerca la coppia (N,M) più grande possibile che sta dentro Q_hw
    for a in range(1, Q_hw):   # log2(N)
        for b in range(1, Q_hw):
            N, M = 2**a, 2**b
            Q_req = ffqram_qubits_required(N, M)
            if Q_req <= Q_hw:
                if best is None or N*M > best[0]*best[1]:
                    best = (N, M)

    if best is None:
        raise ValueError(f"Nessun dataset compatibile con {Q_hw} qubit hardware.")

    N, M = best
    print(f"[INFO] Uso dataset {N}x{M} (richiede {ffqram_qubits_required(N,M)} qubit su {Q_hw} disponibili)")
    return (N,M)

def generate_normalized_dataset(n_rows, n_cols, seed=17):

    # Imposta il seme del generatore di numeri casuali (così la matrice generata sarà sempre la stessa ogni volta che esegui il codice con lo stesso seed)
    np.random.seed(seed)

    # Genera una matrice n_rows x n_cols
    # I valori sono numeri reali casuali distribuiti uniformemente tra 0 e 1
    data = np.random.rand(n_rows, n_cols)

    # Applica normalizzazione L2 sulle righe
    data_normalized = normalize(data, norm='l2')
    return data_normalized



# applica X dove l'indice binario ha bit a 1 (come in Qiskit indexing) ---
def indexing_tk(circ: Circuit, qubits: List[Qubit], index: int) -> None:
    size = len(qubits)
    xored = index ^ ((1 << size) - 1)
    for p in range(size):
        if (xored >> p) & 1:
            circ.add_gate(OpType.X, [qubits[p]])

# C^n(RY(theta)) con n controlli e 1 target ---
"""def multi_controlled_ry(circ: Circuit, theta: float, controls: List[Qubit], target: Qubit) -> None:
   
    base = Op.create(OpType.Ry, [theta])
   
    # 2) versione controllata con n controlli
    cop = base.controlled(len(controls))

    # 3) Aggiungo la porta al circuito con i qubit reali
    circ.add_gate(cop, controls + [target])"""

def multi_controlled_ry(circ: Circuit, theta: float, controls, target):
    # 1) Creo l’operatore base: RY(theta) a 1 qubit
    ry_op = Op.create(OpType.Ry, theta)
    mc_ry = QControlBox(ry_op, n_controls=len(controls))   # control_state default: tutti |1>
    circ.add_gate(mc_ry, [*controls, target])


def FFQRAM_tk(data: np.ndarray) -> Circuit:
    """
    FF-QRAM che scrive data[i, j] nel qubit target tramite una Ry
    controllata da tutti i qubit indice, usando control_state della QControlBox.
    """
    N, M = data.shape
    n_row = int(np.ceil(np.log2(N))) if N > 1 else 1
    n_col = int(np.ceil(np.log2(M))) if M > 1 else 1

    circ = Circuit()

    # Registri (ordine dei controlli: row poi col)
    row = circ.add_q_register("row_index", n_row)   # controlli MSB..LSB della riga
    col = circ.add_q_register("col_index", n_col)   # controlli MSB..LSB della colonna
    reg = circ.add_q_register("register", 1)        # target

    # Uniform superposition sugli indici
    for q in row: circ.add_gate(OpType.H, [q])
    for q in col: circ.add_gate(OpType.H, [q])

    controls = [*row, *col]  # ordine big-endian coerente con addr=(i<<n_col)|j

    # Ciclo sugli indirizzi
    for i in range(N):
        for j in range(M):
            value = float(data[i, j])
           
            theta = 2.0 * float(np.arcsin(value))

            # Indirizzo big-endian: |row>|col|
            addr = (i << n_col) | j  # intero 0..2^(n_row+n_col)-1

            # MC-Ry controllata su quello stato
            ry_op = Op.create(OpType.Ry, theta)
            mc_ry = QControlBox(ry_op, n_controls=len(controls), control_state=addr)
            circ.add_gate(mc_ry, [*controls, reg[0]])

    
    # 1. scompone i Box (QControlBox → MC gates)
    DecomposeBoxes(set({OpType.CX, OpType.U3, OpType.H, OpType.S, OpType.Sdg, OpType.T, OpType.Tdg, OpType.X, OpType.Y, OpType.Z})).apply(circ)

    # 2. scompone i multi-controllo in CNOT

    DecomposeMultiQubitsCX().apply(circ)
    return circ


def build_tk_from_script(n_qubits: int, seed: int, qpy_path: str = "../qiskit_pipeline/ffqram_rev_dec.qpy"):
    # Percorsi assoluti
    script_path_abs = Path("../qiskit_pipeline/build_ffqram_qiskit_to_pytket.py").resolve()
    qpy_path_abs = Path(qpy_path).resolve()
    qpy_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Project root = cartella madre di qiskit_pipeline e tket_pipeline
    project_root = script_path_abs.parents[1]  # .../export_import_file_circuit

    # Ambiente: aggiungi il project root al PYTHONPATH (così lo script trova utils_ffqram)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), env.get("PYTHONPATH", "")]
    )
    # lancia lo script Qiskit
    cmd = [sys.executable,  str(script_path_abs),
           "--n_qubits", str(n_qubits),
           "--out_qpy", str(qpy_path_abs),
           "--seed", str(seed)]
    try:
        # Imposto cwd alla cartella dello script, così i relativi interni funzionano
        res = subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            cwd=str(script_path_abs.parent), env=env
        )
        if res.stdout.strip():
            print(res.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Errore nello script Qiskit (exit {e.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {script_path_abs.parent}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
            f"PYTHONPATH:\n{env.get('PYTHONPATH')}\n"
        ) from e

    # Carica il QPY e converti a PyTKET
    with open(qpy_path_abs, "rb") as f:
        qc_rev_dec = qpy.load(f)[0]
    tk_circ = qiskit_to_tk(qc_rev_dec)
    return tk_circ


