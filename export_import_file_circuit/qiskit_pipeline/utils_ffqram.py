from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
import numpy as np
from sklearn.preprocessing import  normalize
from qiskit.circuit.library import MCMTGate, RYGate
import numpy as np
from sklearn.preprocessing import normalize

def ffqram_qubits_required(N, M):
    return np.ceil(np.log2(N)) + np.ceil(np.log2(M)) + 1

def adjust_dataset_for_hw(coupling_map):
    """
    Genera un dataset randomico N x M compatibile con la coupling map.
    Sceglie N,M come potenze di 2 massime che rispettano i vincoli.
    """
    Q_hw = coupling_map.size()  # qubit fisici
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

def generate_normalized_dataset(n_rows, n_cols, seed):

    np.random.seed(int(seed))

    # Genera una matrice n_rows x n_cols
    # I valori sono numeri reali casuali distribuiti uniformemente tra 0 e 1
    data = np.random.rand(n_rows, n_cols)

    # Applica normalizzazione L2 sulle righe
    data_normalized = normalize(data, norm='l2')
    return data_normalized


def indexing(circuit, Qregister, index):
    size = Qregister.size # numero di qubit nel registro

    xored = index ^ (pow(2, size) - 1) # XOR tra index e 111..1 quindi porta a 0 i bit a 1
    j=1
    for k in (2**p for p in range(0, size)): # itera sulle potenze di 2 fino a size (le potenze di 2 contengono solo un 1)
        if xored & k >= j: # xored & k restituisce: k se il bit p in xored è 1; 0 se quel bit è 0
            circuit.x(Qregister[j-1]) #esegue il flip
        j = j+1
        
def FFQRAM(data):
    
    # data: matrice N x M

    N, M = data.shape # N: numero di righe (N pattern), M: numero di colonne (M features) 

    n_row = int(np.ceil(np.log2(max(1, N))))
    n_col = int(np.ceil(np.log2(max(1, M))))

    
    row_index = QuantumRegister(n_row, name="row_index") # Qubit di indirizzo per le righe

    col_index = QuantumRegister(n_col, name="col_index") # Qubit di indirizzo per le colonne

    r = QuantumRegister(1) # Qubit ausiliario dove andranno le ampiezze 

    qc = QuantumCircuit(row_index, *( [col_index] if col_index else [] ), r)

    # Hadamard solo se il registro esiste (size > 0)
    if n_row > 0:
        qc.h(row_index)
    if n_col > 0:
        qc.h(col_index)

    for i in range(N):
        # Flip riga se esistono qubit riga
        if n_row > 0:
            indexing(qc, row_index, i)

        vector = data[i]
        for j in range(M):
            amp = float(vector[j])
            if amp == 0.0:
                continue

            # Flip colonna se esistono qubit colonna
            if n_col > 0:
                indexing(qc, col_index, j)

            theta = 2.0 * np.arcsin(np.clip(amp, 0.0, 1.0))

            # Costruisci lista controlli solo con registri esistenti
            controls = list(row_index[:])
            if n_col > 0:
                controls += list(col_index[:])

            if len(controls) > 0:
                qc.append(MCMTGate(RYGate(theta), len(controls), 1), controls + [r[0]])
            else:
                # Nessun controllo (caso N=1 e M=1): scrivi direttamente su R
                qc.ry(theta, r[0])

            if n_col > 0:
                indexing(qc, col_index, j)

        if n_row > 0:
            indexing(qc, row_index, i)
       
    return qc

def reverse_circuit_qubit_order(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Crea un nuovo circuito con l'ordine dei qubit globalmente invertito.
    Mantiene i registri e i classical bits (se presenti).
    """
    # Ricrea lo stesso "schema" di registri
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs, name=(qc.name or "qc") + "_reversed")

    # Mappa qubit: vecchio -> nuovo (lista globale invertita)
    all_old = qc.qubits
    all_new = list(reversed(new_qc.qubits))
    qmap = {old_q: all_new[idx] for idx, old_q in enumerate(all_old)}

    # Copia tutte le istruzioni rimappando i qubit
    for inst, qargs, cargs in qc.data:
        new_qc.append(inst, [qmap[q] for q in qargs], cargs)

    return new_qc