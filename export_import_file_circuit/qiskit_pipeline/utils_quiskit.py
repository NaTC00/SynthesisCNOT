from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout
from qiskit_aer import Aer, StatevectorSimulator
from qiskit.quantum_info import Statevector, state_fidelity
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from qiskit.circuit.library import MCMTGate, RYGate, RZGate, CRYGate, XGate
from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector, plot_histogram
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev

#crea il circuito dal file contente le porte   
def create_circuit_from_file(filename: str) -> QuantumCircuit:
    with open(filename, 'r') as f:
       
        lines = [line.strip() for line in f.readlines() if line.strip()]

        num_qubits = int(lines[0])
        qc = QuantumCircuit(num_qubits)

        for line in lines[1:]:
            if '[' not in line or ']' not in line:
                continue #ignora righe malformate
            
            gate_name, qubit_str = line.split('[') # splitta la stringa in corrispondenza di [
            qubit_str = qubit_str.replace(']', '')

            qubits = [int(q.strip()) - 1 for q in qubit_str.split(',')]
            gate_name = gate_name.strip().upper()

            if gate_name == 'H':
                qc.h(qubits[0])
            elif gate_name == 'X':
                qc.x(qubits[0])
            elif gate_name == 'S':
                qc.s(qubits[0])
            elif gate_name == 'S+':
                qc.sdg(qubits[0])
            elif gate_name == 'T':
                qc.t(qubits[0])
            elif gate_name == 'T+':
                qc.tdg(qubits[0])
            elif gate_name == 'Y':
                qc.y(qubits[0])
            elif gate_name == 'Z':
                qc.z(qubits[0])
            elif gate_name == 'CNOT':
                qc.cx(qubits[0], qubits[1])
            else:
                raise ValueError(f"Porta non supportata: {gate_name}")
    return qc


def create_circuit_from_simple_file(filename: str) -> QuantumCircuit:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        num_qubits = int(lines[0])
        qc = QuantumCircuit(num_qubits)

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue  # riga non valida

            gate_name = parts[0].upper()
            qubits = [int(q) for q in parts[1:]]

            if gate_name == 'H':
                qc.h(qubits[0])
            elif gate_name == 'X':
                qc.x(qubits[0])
            elif gate_name == 'Y':
                qc.y(qubits[0])
            elif gate_name == 'Z':
                qc.z(qubits[0])
            elif gate_name == 'S':
                qc.s(qubits[0])
            elif gate_name == 'S+':
                qc.sdg(qubits[0])
            elif gate_name == 'T':
                qc.t(qubits[0])
            elif gate_name == 'T+':
                qc.tdg(qubits[0])
            elif gate_name == 'CNOT':
                qc.cx(qubits[0], qubits[1])
            else:
                raise ValueError(f"Porta non supportata: {gate_name}")
    
    return qc


#partendo dal circuito crea un file contente le porte del circuito
def create_file_from_circuit(qc: QuantumCircuit, filepath: str, gatesSupported: list):
    with open(filepath, "w") as f:
        f.write(f"{qc.num_qubits}\n")
        for gate in qc.data:
            gateApplied = gate.operation.name.upper()
            qubitsApplied = gate.qubits

            if gateApplied == "TDG":
                gateApplied = "T+"
            elif gateApplied == "SDG":
                gateApplied = "S+"
            elif gateApplied == "CX":
                gateApplied = "CNOT"

            if gateApplied in gatesSupported:
                if len(qubitsApplied) == 1:
                    f.write(f"{gateApplied} {qubitsApplied[0]._index}\n")
                elif gateApplied == "CNOT":
                    f.write(f"{gateApplied} {qubitsApplied[0]._index} {qubitsApplied[1]._index}\n")


def load_adjacency_matrix_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        size = int(lines[0].strip())
        adj_matrix = []

        for line in lines[1:size+1]:
            row = list(map(int, line.strip().split()))
            adj_matrix.append(row)

    return adj_matrix


def matrix_to_coupling_map(adj_matrix):
    edges = []
    size = len(adj_matrix)
    for i in range(size):
        for j in range(size):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return CouplingMap(edges)

 # --- CONTEGGIO CNOT ---
def count_cx_gates(circuit):
    return circuit.count_ops().get('cx', 0)

def compute_fidelity_qiskit(qc_qiskit) -> QuantumCircuit:
    layout_mapping = qc_qiskit.layout.final_layout
    
    if layout_mapping is not None:
        n = qc_qiskit.num_qubits
        dict = list(layout_mapping.get_physical_bits().keys())
        swap_list = []
        for i in range(n):
            for j in range(n - 1):
                if dict[j] > dict[j + 1]:  # Swap if elements are in the wrong order
                    dict[j], dict[j + 1] = dict[j + 1], dict[j]
                    swap_list.append((dict[j], dict[j+1]))
        for i in range(len(swap_list)):
            qc_qiskit.swap(swap_list[i][0],swap_list[i][1])
    
    return qc_qiskit


def simulate_and_compare(qc_original, qc_qiskit, qc_java):
    simulator = StatevectorSimulator()
    state_original = simulator.run(qc_original).result().get_statevector()
    state_qiskit = simulator.run(qc_qiskit).result().get_statevector()
    state_java = simulator.run(qc_java).result().get_statevector()

    fidelity_qiskit = state_fidelity(state_original, state_qiskit)
    fidelity_java = state_fidelity(state_original, state_java)

    return fidelity_qiskit, fidelity_java

def get_layout(qr):
    layout_dict = {}
    for i, qubit in enumerate(qr):
        layout_dict[qubit] = i  

    layout = Layout(layout_dict)
    return layout

def get_statevector(circuit):  
    simulator = StatevectorSimulator()
    result = simulator.run(circuit.decompose(reps=10)).result()
    statevector = result.get_statevector()
    return statevector

def indexing(circuit, Qregister, index):
    size = Qregister.size
    xored = index ^ (pow(2, size) - 1)
    j=1
    for k in (2**p for p in range(0, size)):
        if xored & k >= j:
            circuit.x(Qregister[j-1])
        j = j+1
        
def FFQRAM(data):
    
    N, M = data.shape
    
    row_index = QuantumRegister(np.ceil(np.log2(N)))
    col_index = QuantumRegister(np.ceil(np.log2(M)))
    r = QuantumRegister(1)
    qc = QuantumCircuit(row_index, col_index, r)
    
    qc.h(row_index)
    qc.h(col_index)
    
    for i in range(N):
        vector = data[i]

        indexing(qc, row_index, i)
        
        for j in range(len(vector)): 
            indexing(qc, col_index, j)
            qc.append(MCMTGate(RYGate(2*np.arcsin(vector[j])), len(row_index[:]+col_index[:]), 1), row_index[:]+col_index[:]+r[0:])
            indexing(qc, col_index, j)
            #qc.barrier()

        indexing(qc, row_index, i)

        #qc.barrier()
        
    return qc 

def qasm_to_clifford_and_t(qc, basic_approx_depth=10):
    qc = transpile(qc,basis_gates=["cx","u3"]) # You should transpile first to cx and u3, so it will deal with 2Q gates
    basis = ['h', 's', 'sdg', 'tdg', 't', 'x', 'y', 'z']
    approx = generate_basic_approximations(basis, depth=basic_approx_depth)
    skd = SolovayKitaev(recursion_degree=1, basic_approximations=approx)
    new_qc = skd(qc)
    return new_qc