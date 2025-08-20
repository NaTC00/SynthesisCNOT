from locale import normalize
from typing import List
import numpy as np

from pytket.circuit import Circuit, OpType, Qubit, Op, QControlBox
from sklearn.preprocessing import normalize
from pytket.passes import DecomposeBoxes, DecomposeMultiQubitsCX

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

# --- FFQRAM in pytket ---
"""def FFQRAM_tk(data: np.ndarray) -> Circuit:
 
    N, M = data.shape

    n_row = int(np.ceil(np.log2(N))) if N > 1 else 1
    n_col = int(np.ceil(np.log2(M))) if M > 1 else 1

    circ = Circuit()

    # Registi quantistici con nomi equivalenti
    row = circ.add_q_register("row_index", n_row)       # -> List[Qubit]
    col = circ.add_q_register("col_index", n_col)       # -> List[Qubit]
    reg = circ.add_q_register("register", 1)            # target r

    # H su tutti gli indici
    for q in row: circ.add_gate(OpType.H, [q])
    for q in col: circ.add_gate(OpType.H, [q])

    # Loop come nel tuo codice
    for i in range(N):
        vector = data[i]

        # Seleziona riga i
        indexing_tk(circ, row, i)

        for j in range(len(vector)):
            # Seleziona colonna j
            indexing_tk(circ, col, j)

            # angolo 2*arcsin(value)
            theta = 2.0 * float(np.arcsin(vector[j]))

            # C^{n_row+n_col}( RY(theta) ) sul qubit r[0]
            controls = [*row, *col]  
            multi_controlled_ry(circ, theta, controls, reg[0])

            # Deseleziona colonna j
            indexing_tk(circ, col, j)

        # Deseleziona riga i
        indexing_tk(circ, row, i)

    return circ
"""

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
            # clamp opzionale per sicurezza numerica
            value = max(-1.0, min(1.0, value))
            theta = 2.0 * float(np.arcsin(value))

            # Indirizzo big-endian: |row>|col|
            addr = (i << n_col) | j  # intero 0..2^(n_row+n_col)-1

            # MC-Ry controllata su quello stato
            ry_op = Op.create(OpType.Ry, theta)
            mc_ry = QControlBox(ry_op, n_controls=len(controls), control_state=addr)
            circ.add_gate(mc_ry, [*controls, reg[0]])

    # 1. scompone i Box (QControlBox → MC gates)
    DecomposeBoxes().apply(circ)

    # 2. scompone i multi-controllo in CNOT
    DecomposeMultiQubitsCX().apply(circ)
    return circ