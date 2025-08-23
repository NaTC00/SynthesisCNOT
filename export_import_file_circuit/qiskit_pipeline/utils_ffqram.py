from qiskit import QuantumCircuit, QuantumRegister
import pandas as pd
import numpy as np
from sklearn.preprocessing import  normalize
from qiskit.circuit.library import MCMTGate, RYGate, RZGate, CRYGate, XGate
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
    
    row_index = QuantumRegister(int(np.ceil(np.log2(N))), name="row_index") # Qubit di indirizzo per le righe

    col_index = QuantumRegister(int(np.ceil(np.log2(M))), name="col_index") # Qubit di indirizzo per le colonne

    r = QuantumRegister(1) # Qubit ausiliario dove andranno le ampiezze 

    qc = QuantumCircuit(row_index, col_index, r)
    
    # Metto i qubit di indirizzo in sovrapposizione
    qc.h(row_index)
    qc.h(col_index)
    
    # Ciclo su tutti i pattern
    for i in range(N):

        # vettore con le M features
        vector = data[i]

        indexing(qc, row_index, i) #bit-flip da bit classici sul pattern i
        
        for j in range(len(vector)): 
            indexing(qc, col_index, j) # FLIP (trasforma pattern j in tutti 1)
            qc.append(MCMTGate(RYGate(2*np.arcsin(vector[j])), len(row_index[:]+col_index[:]), 1), row_index[:]+col_index[:]+r[0:])
            indexing(qc, col_index, j) # FLOP (inverte il flip -> riporta al pattern j)
            #qc.barrier()

        indexing(qc, row_index, i)

        #qc.barrier()
        
    return qc